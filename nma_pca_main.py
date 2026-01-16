# nma_pca_main.py

import os
import numpy as np
from pathlib import Path

from traj_utils import build_protein_heavy_views
from nma_bio3d import run_aanma_r_from_traj
from pca_stream import run_incremental_pca_from_chunks
from analysis_utils import (
    compute_confusion_matrix,
    plot_confusion_matrix,
    compute_confusion_matrices_per_replica,
    aggregate_best_matches,
    plot_best_match_barplot,
    report_pca_variance_thresholds,
    plot_nma_pca_stacked_bars,
)
from collections import defaultdict
from io_utils import get_pdb_dir, collect_xtc_paths, print_header

def group_xtc_paths_by_replica(xtc_paths):
    """
    Attempt to group XTC paths by replica index by scanning path parts.
    Works for common layouts like .../validation/<rep>/prod.xtc
    or .../<rep>/prod.xtc.

    Returns: dict[int, list[Path]]
    """
    groups = defaultdict(list)

    for p in xtc_paths:
        parts = list(p.parts)

        rep = None

        # Common case: .../validation/<rep>/...
        if "validation" in parts:
            i = parts.index("validation")
            if i + 1 < len(parts):
                cand = parts[i + 1]
                if cand.isdigit():
                    rep = int(cand)

        # Fallback: any directory name that is an integer
        if rep is None:
            for part in reversed(parts):
                if part.isdigit():
                    rep = int(part)
                    break

        if rep is None:
            raise RuntimeError(f"Could not infer replica index from path: {p}")

        groups[rep].append(p)

    # Sort paths within each replica for reproducibility
    return {r: sorted(ps) for r, ps in sorted(groups.items(), key=lambda x: x[0])}

def main():
    pdb_code = input("Enter PDB code (4 characters): ").strip().lower()
    chunk_size = int(input("Chunk size for IncrementalPCA [500]: ") or 500)
    n_modes_keep = 20
    k_stack = 10

    # Output directory
    out_root = Path("/home/csongor/boxpred")
    out_dir = out_root / "results_validation" / pdb_code
    out_dir.mkdir(parents=True, exist_ok=True)

    pdb_dir = get_pdb_dir(pdb_code)
    xtc_paths = collect_xtc_paths(pdb_dir)

    # Build all necessary topology + reference traj
    (
        traj_protein_heavy_ref,
        top_xtc_protein,
        protein_heavy_idx,
        top_xtc_full   # not used yet, but saved for future options
    ) = build_protein_heavy_views(pdb_code)

    print_header("Starting NMA–PCA comparison pipeline")
    print(f"[MAIN] Protein heavy atoms: {traj_protein_heavy_ref.n_atoms}")

   # ----- 1) NMA -----
    nma_modes = run_aanma_r_from_traj(traj_protein_heavy_ref, n_modes_keep)
    print(f"[MAIN] NMA modes shape: {nma_modes.shape}")

    # ----- 2) PCA (per replica) -----
    xtc_by_rep = group_xtc_paths_by_replica(xtc_paths)
    print_header(f"Found replicas: {list(xtc_by_rep.keys())}")

    pca_modes_by_replica = []
    rep_order = []  # keep explicit mapping between list index and replica number

    for rep, rep_xtcs in xtc_by_rep.items():
        print_header(f"Running PCA for replica {rep}")
        print(f"[REPLICA {rep}] #xtc files: {len(rep_xtcs)}")
        for x in rep_xtcs:
            print(f"  - {x}")

        pca_rep, evr_rep = run_incremental_pca_from_chunks(
            xtc_paths=rep_xtcs,
            topology=top_xtc_full,     # exact match for XTC atom order
            n_components=n_modes_keep,
            chunk_size=chunk_size,
            atom_indices=protein_heavy_idx
        )

        # Variance reporting (INSIDE the replica loop)
        var = report_pca_variance_thresholds(
            explained_variance_ratio=evr_rep,
            out_dir=out_dir,
            prefix=f"{pdb_code}_rep{rep}",
            targets=(0.80, 0.90, 0.95, 0.99),
            save_plot=True,
        )

        print(f"[REPLICA {rep}] Variance thresholds:")
        for t, k, a in zip(var["targets"], var["n_pcs"], var["achieved"]):
            print(f"  {t*100:.0f}% -> {k} PCs (achieved {a*100:.2f}%)")

        if var["plot"] is not None:
            print(f"[REPLICA {rep}] Saved: {var['plot']}")

        print(f"[REPLICA {rep}] PCA modes shape: {pca_rep.shape}")

        pca_modes_by_replica.append(pca_rep)
        rep_order.append(rep)

    # ----- 3) Compare (per replica) -----
    rep_results = compute_confusion_matrices_per_replica(
        nma_modes=nma_modes,
        pca_modes_by_replica=pca_modes_by_replica,
        k_rmsip=10,
    )

    # Save + plot per replica
    for d in rep_results:
        r_idx = d["replica"]              # index in pca_modes_by_replica
        rep = rep_order[r_idx]            # actual replica number from path
        confusion = d["confusion"]
        kept_pcs = list(range(1, k_stack + 1))  # 1-based PC indices, only for this chart

        stack_path = out_dir / f"{pdb_code}_rep{rep}_nma_pca_stacked.png"
        plot_nma_pca_stacked_bars(
            confusion=confusion,
            kept_pca_idx=kept_pcs,
            nma_start=7,
            nma_end=n_modes_keep,
            title=f"{pdb_code.upper()} – Replica {rep}: NMA overlap with first {k_stack} PCs (stacked)",
            outfile=stack_path,
)

        # Save raw matrix
        npy_path = out_dir / f"{pdb_code}_rep{rep}_confusion_matrix.npy"
        np.save(npy_path, confusion)
        print(f"[MAIN] Saved confusion matrix array to: {npy_path}")

        # Plot
        out_path = out_dir / f"{pdb_code}_rep{rep}_confusion_matrix.png"
        plot_confusion_matrix(confusion, out_path, pdb_code=f"{pdb_code} rep{rep}")

        print(f"[MAIN] Replica {rep}: RMSIP = {d['rmsip']:.3f}")

        # Bar plot: best PCA match per NMA mode
        bar_path = out_dir / f"{pdb_code}_rep{rep}_bestmatch_barplot.png"
        plot_best_match_barplot(
            best_per_nma=d["best_per_nma"],
            argbest_per_nma=d["argbest_per_nma"],
            out_path=bar_path,
            title=f"{pdb_code.upper()} – Replica {rep}: Best PCA match per NMA mode",
        )

    # Aggregate summary across replicas
    agg = aggregate_best_matches(rep_results)
    print_header("Aggregate across replicas")
    print(f"[AGG] RMSIP mean ± std: {agg['rmsip_mean']:.3f} ± {agg['rmsip_std']:.3f}")

    print("[MAIN] Pipeline finished successfully.")



if __name__ == "__main__":
    main()
