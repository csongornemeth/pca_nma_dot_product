# run_nm_displacement_fit.py

from __future__ import annotations

from pathlib import Path
import argparse
import json

import mdtraj as md
import numpy as np
import pandas as pd

from io_utils import get_pdb_dir, collect_xtc_paths, print_header
from traj_utils import build_protein_heavy_views
from nma_bio3d import run_aanma_r_from_traj

from nm_fit_utils import (
    compute_displacement,
    select_top_displacing_atoms,
    modes_flat_to_atoms,
    slice_displacement_and_modes,
    fit_single_modes,
    fit_mode_combination,
)

"""
Run with the following command from the repository root:
python run_nm_displacement_fit.py \
  --pdb 7lak \
  --replica 0 \
  --target-mode max-displacement \
  --n-modes-keep 20 \
  --top-n 20 \
  --fit-start 0 \
  --fit-stop 20 \
  --mode-number-offset 7 \
  --outdir nm_displacement_fit_results
"""

def collect_replica_xtcs(
    pdb_code: str,
    replica: int,
) -> list[Path]:
    """
    Collect XTC files for one replica.
    """
    pdb_dir = get_pdb_dir(pdb_code)
    xtc_paths = collect_xtc_paths(pdb_dir)

    replica_paths = [
        p for p in xtc_paths
        if p.parent.name == str(replica)
    ]

    if not replica_paths:
        raise FileNotFoundError(
            f"No XTC files found for pdb={pdb_code}, replica={replica}"
        )

    return replica_paths


def load_target_frame_protein_heavy(
    xtc_paths: list[Path],
    top_xtc_full: md.Topology,
    protein_heavy_idx_full: np.ndarray,
    frame: int,
    chunk: int = 100,
) -> md.Trajectory:
    """
    Load one target frame from one replica and slice to protein-heavy atoms.

    Supports:
        frame = -1
            last frame across all XTC parts.

        frame >= 0
            global frame index across concatenated XTC parts.

    Uses md.iterload to avoid loading the full replica into memory.
    """
    if frame < -1:
        raise ValueError("Only frame=-1 or frame>=0 is supported.")

    target: md.Trajectory | None = None
    global_start = 0

    if frame == -1:
        for xtc in xtc_paths:
            for chunk_traj in md.iterload(
                xtc.as_posix(),
                chunk=chunk,
                top=top_xtc_full,
            ):
                target = chunk_traj[-1]

        if target is None:
            raise RuntimeError("No frames found in trajectory.")

        return target.atom_slice(protein_heavy_idx_full)

    for xtc in xtc_paths:
        for chunk_traj in md.iterload(
            xtc.as_posix(),
            chunk=chunk,
            top=top_xtc_full,
        ):
            n = chunk_traj.n_frames
            global_stop = global_start + n

            if global_start <= frame < global_stop:
                local_idx = frame - global_start
                target = chunk_traj[local_idx]
                return target.atom_slice(protein_heavy_idx_full)

            global_start = global_stop

    raise IndexError(
        f"Requested global frame {frame}, but trajectory only has "
        f"{global_start} frames."
    )


def load_max_displacement_frame_protein_heavy(
    xtc_paths: list[Path],
    top_xtc_full: md.Topology,
    protein_heavy_idx_full: np.ndarray,
    traj_protein_heavy_ref: md.Trajectory,
    chunk: int = 100,
) -> tuple[md.Trajectory, dict]:
    """
    Scan one replica and return the frame with the largest aligned RMS
    displacement from the reference.

    Each chunk is:
        1. sliced to protein-heavy atoms
        2. aligned to the protein-heavy reference
        3. compared to the reference

    Returns:
        best_frame
            One-frame md.Trajectory containing the aligned protein-heavy frame
            with maximum RMS displacement.

        best_info
            Dictionary with metadata about the selected frame.
    """
    ref_xyz = traj_protein_heavy_ref.xyz[0]

    best_frame: md.Trajectory | None = None
    best_rms_displacement = -np.inf
    best_global_frame: int | None = None
    best_xtc: Path | None = None
    best_local_frame_in_chunk: int | None = None

    global_frame = 0

    for xtc in xtc_paths:
        print(f"[INFO] Scanning XTC for max displacement: {xtc}")

        for chunk_traj in md.iterload(
            xtc.as_posix(),
            chunk=chunk,
            top=top_xtc_full,
        ):
            chunk_protein = chunk_traj.atom_slice(protein_heavy_idx_full)

            chunk_protein.superpose(
                traj_protein_heavy_ref,
                frame=0,
            )

            dx = chunk_protein.xyz - ref_xyz[None, :, :]

            rms_disp = np.sqrt(
                np.mean(
                    np.sum(dx**2, axis=2),
                    axis=1,
                )
            )

            local_best_idx = int(np.argmax(rms_disp))
            local_best_value = float(rms_disp[local_best_idx])

            if local_best_value > best_rms_displacement:
                best_rms_displacement = local_best_value
                best_global_frame = global_frame + local_best_idx
                best_xtc = xtc
                best_local_frame_in_chunk = local_best_idx
                best_frame = chunk_protein[local_best_idx]

            global_frame += chunk_traj.n_frames

    if best_frame is None:
        raise RuntimeError("No frames found while searching for maximum displacement.")

    best_info = {
        "selected_frame_mode": "max_displacement",
        "max_displacement_global_frame": int(best_global_frame),
        "max_displacement_rms_nm": float(best_rms_displacement),
        "max_displacement_xtc": str(best_xtc),
        "max_displacement_local_frame_in_chunk": int(best_local_frame_in_chunk),
        "n_frames_scanned": int(global_frame),
    }

    return best_frame, best_info


def add_atom_metadata(
    top: md.Topology,
    disp_mag: np.ndarray,
) -> pd.DataFrame:
    """
    Create a per-atom displacement table with useful metadata.
    """
    rows = []

    for atom, d in zip(top.atoms, disp_mag):
        res = atom.residue

        rows.append(
            {
                "atom_local_index": atom.index,
                "atom_name": atom.name,
                "element": atom.element.symbol if atom.element is not None else "",
                "residue_index_0based": res.index,
                "residue_number": res.resSeq,
                "residue_name": res.name,
                "chain_index": res.chain.index,
                "displacement_nm": float(d),
            }
        )

    return pd.DataFrame(rows)


def save_json_safe(path: Path, data: dict) -> None:
    """
    Save dictionary to JSON, converting NumPy values where needed.
    """
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        return obj

    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=convert)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Fit Bio3D normal-mode vectors to a trajectory displacement vector."
        )
    )

    parser.add_argument("--pdb", required=True, help="PDB code/system code.")

    parser.add_argument(
        "--replica",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--frame",
        type=int,
        default=-1,
        help=(
            "Global frame index in selected replica. "
            "Use -1 for last frame. "
            "Ignored if --target-mode max-displacement."
        ),
    )

    parser.add_argument(
        "--target-mode",
        choices=["frame", "max-displacement"],
        default="frame",
        help=(
            "How to choose the target displacement. "
            "'frame' uses --frame. "
            "'max-displacement' scans the whole selected replica and uses "
            "the frame with the largest aligned RMS displacement from the reference."
        ),
    )

    parser.add_argument(
        "--n-modes-keep",
        type=int,
        default=50,
        help="Number of internal Bio3D modes to keep after skipping 6 rigid modes.",
    )

    parser.add_argument(
        "--top-n",
        type=int,
        default=50,
        help="Number of most-displaced atoms to use for fitting.",
    )

    parser.add_argument(
        "--fit-start",
        type=int,
        default=0,
        help="First mode array index for linear-combination fit.",
    )

    parser.add_argument(
        "--fit-stop",
        type=int,
        default=20,
        help="Stop mode array index for linear-combination fit, exclusive.",
    )

    parser.add_argument(
        "--mode-number-offset",
        type=int,
        default=7,
        help=(
            "Reported Bio3D mode number offset. "
            "Use 7 because nma_bio3d.py skips the first 6 rigid-body modes."
        ),
    )

    parser.add_argument(
        "--chunk",
        type=int,
        default=100,
        help="Chunk size for md.iterload.",
    )

    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("nm_displacement_fit_results"),
        help="Base output directory.",
    )

    args = parser.parse_args()

    pdb_code = args.pdb.lower()

    if args.target_mode == "frame":
        run_label = f"replica_{args.replica}_frame_{args.frame}"
    else:
        run_label = f"replica_{args.replica}_max_displacement"

    outdir = args.outdir / pdb_code / run_label
    outdir.mkdir(parents=True, exist_ok=True)

    print_header("NM displacement fitting")
    print(f"[INFO] PDB/system: {pdb_code}")
    print(f"[INFO] Replica: {args.replica}")
    print(f"[INFO] Frame: {args.frame}")
    print(f"[INFO] Target mode: {args.target_mode}")
    print(f"[INFO] Output directory: {outdir}")

    # ------------------------------------------------------------
    # 1. Build protein-heavy reference and topology views
    # ------------------------------------------------------------
    print_header("Building protein-heavy topology views")

    (
        traj_protein_heavy_ref,
        top_xtc_protein,
        protein_heavy_idx_local,
        protein_heavy_idx_full,
        top_xtc_full,
    ) = build_protein_heavy_views(pdb_code)

    n_atoms = traj_protein_heavy_ref.n_atoms
    expected_dof = 3 * n_atoms

    print(f"[INFO] Protein-heavy atoms: {n_atoms}")
    print(f"[INFO] Expected DOF: {expected_dof}")

    # ------------------------------------------------------------
    # 2. Run NMA on exactly the same protein-heavy reference
    # ------------------------------------------------------------
    print_header("Running Bio3D NMA")

    modes_flat, eigvals = run_aanma_r_from_traj(
        traj_protein_heavy=traj_protein_heavy_ref,
        n_modes_keep=args.n_modes_keep,
    )

    print(f"[INFO] modes_flat shape: {modes_flat.shape}")
    print(f"[INFO] eigvals shape: {eigvals.shape}")

    if modes_flat.shape[1] != expected_dof:
        raise ValueError(
            f"NMA modes do not match protein-heavy atom basis. "
            f"modes_flat.shape={modes_flat.shape}, expected DOF={expected_dof}"
        )

    modes_atoms = modes_flat_to_atoms(
        modes_flat,
        n_atoms=n_atoms,
    )

    np.save(outdir / "modes_flat.npy", modes_flat)
    np.save(outdir / "eigvals.npy", eigvals)

    # ------------------------------------------------------------
    # 3. Load or select target trajectory frame
    # ------------------------------------------------------------
    print_header("Loading/selecting target trajectory frame")

    xtc_paths = collect_replica_xtcs(
        pdb_code=pdb_code,
        replica=args.replica,
    )

    print("[INFO] XTC files for selected replica:")
    for p in xtc_paths:
        print(f"  {p}")

    if args.target_mode == "frame":
        target_protein_heavy = load_target_frame_protein_heavy(
            xtc_paths=xtc_paths,
            top_xtc_full=top_xtc_full,
            protein_heavy_idx_full=protein_heavy_idx_full,
            frame=args.frame,
            chunk=args.chunk,
        )

        target_info = {
            "selected_frame_mode": "frame",
            "selected_frame": int(args.frame),
        }

    else:
        target_protein_heavy, target_info = load_max_displacement_frame_protein_heavy(
            xtc_paths=xtc_paths,
            top_xtc_full=top_xtc_full,
            protein_heavy_idx_full=protein_heavy_idx_full,
            traj_protein_heavy_ref=traj_protein_heavy_ref,
            chunk=args.chunk,
        )

    print(f"[INFO] Target frame atoms: {target_protein_heavy.n_atoms}")
    print("[INFO] Target selection:")
    for k, v in target_info.items():
        print(f"  {k}: {v}")

    save_json_safe(
        outdir / "target_selection.json",
        target_info,
    )

    if target_protein_heavy.n_atoms != n_atoms:
        raise ValueError(
            f"Target atom count mismatch. "
            f"Reference has {n_atoms}, target has {target_protein_heavy.n_atoms}"
        )

    # ------------------------------------------------------------
    # 4. Align target frame to reference
    # ------------------------------------------------------------
    print_header("Aligning target to reference")

    target_protein_heavy.superpose(
        traj_protein_heavy_ref,
        frame=0,
    )

    ref_xyz = traj_protein_heavy_ref.xyz[0]
    target_xyz = target_protein_heavy.xyz[0]

    # ------------------------------------------------------------
    # 5. Compute displacement vector
    # ------------------------------------------------------------
    print_header("Computing displacement")

    dx_atoms, dx_flat, disp_mag = compute_displacement(
        ref_xyz=ref_xyz,
        target_xyz=target_xyz,
    )

    np.save(outdir / "dx_atoms.npy", dx_atoms)
    np.save(outdir / "dx_flat.npy", dx_flat)
    np.save(outdir / "disp_mag.npy", disp_mag)

    atom_disp_df = add_atom_metadata(
        top=top_xtc_protein,
        disp_mag=disp_mag,
    )

    atom_disp_df = atom_disp_df.sort_values(
        "displacement_nm",
        ascending=False,
    )

    atom_disp_df.to_csv(
        outdir / "atom_displacement_magnitudes.csv",
        index=False,
    )

    print("[INFO] Largest displacements:")
    print(atom_disp_df.head(10).to_string(index=False))

    # ------------------------------------------------------------
    # 6. Select top-moving atoms
    # ------------------------------------------------------------
    print_header("Selecting top-moving atoms")

    top_idx = select_top_displacing_atoms(
        disp_mag=disp_mag,
        n_top=args.top_n,
    )

    selected_atoms_df = atom_disp_df[
        atom_disp_df["atom_local_index"].isin(top_idx)
    ].copy()

    selected_atoms_df = selected_atoms_df.sort_values(
        "displacement_nm",
        ascending=False,
    )

    selected_atoms_df.to_csv(
        outdir / "selected_top_displacing_atoms.csv",
        index=False,
    )

    print(f"[INFO] Selected top atoms: {len(top_idx)}")

    # ------------------------------------------------------------
    # 7. Slice displacement and modes to selected atoms
    # ------------------------------------------------------------
    print_header("Slicing displacement and modes")

    dx_sel, modes_sel = slice_displacement_and_modes(
        dx_atoms=dx_atoms,
        modes=modes_atoms,
        atom_idx=top_idx,
    )

    print(f"[INFO] dx_sel shape: {dx_sel.shape}")
    print(f"[INFO] modes_sel shape: {modes_sel.shape}")

    np.save(outdir / "dx_selected.npy", dx_sel)
    np.save(outdir / "modes_selected.npy", modes_sel)

    # ------------------------------------------------------------
    # 8. Fit individual modes
    # ------------------------------------------------------------
    print_header("Fitting individual normal modes")

    single_df = fit_single_modes(
        dx=dx_sel,
        modes=modes_sel,
        mode_number_offset=args.mode_number_offset,
    )

    single_df_sorted = single_df.sort_values(
        "abs_cosine_overlap",
        ascending=False,
    )

    single_df_sorted.to_csv(
        outdir / "single_mode_fits_ranked.csv",
        index=False,
    )

    single_df.to_csv(
        outdir / "single_mode_fits_original_order.csv",
        index=False,
    )

    print("[INFO] Best individual modes:")
    print(single_df_sorted.head(10).to_string(index=False))

    # ------------------------------------------------------------
    # 9. Fit linear combination of selected modes
    # ------------------------------------------------------------
    print_header("Fitting linear combination of modes")

    mode_ids = np.arange(args.fit_start, args.fit_stop)

    combo_coeff_df, combo_summary = fit_mode_combination(
        dx=dx_sel,
        modes=modes_sel,
        mode_ids=mode_ids,
        mode_number_offset=args.mode_number_offset,
    )

    combo_coeff_df.to_csv(
        outdir / "linear_combination_coefficients.csv",
        index=False,
    )

    np.save(
        outdir / "linear_combination_dx_fit.npy",
        combo_summary["dx_fit"],
    )

    combo_summary_public = {
        k: v
        for k, v in combo_summary.items()
        if k not in {"dx_fit", "singular_values", "residuals"}
    }

    combo_summary_public["singular_values"] = combo_summary["singular_values"]
    combo_summary_public["residuals"] = combo_summary["residuals"]

    save_json_safe(
        outdir / "linear_combination_summary.json",
        combo_summary_public,
    )

    # ------------------------------------------------------------
    # 9b. Test cumulative linear combinations of first N modes
    # ------------------------------------------------------------
    print_header("Testing cumulative mode combinations")

    cumulative_mode_counts = [2, 3, 4, 5, 10]

    cumulative_rows = []

    for n_modes in cumulative_mode_counts:
        mode_ids_n = np.arange(0, n_modes)

        coeff_df_n, summary_n = fit_mode_combination(
            dx=dx_sel,
            modes=modes_sel,
            mode_ids=mode_ids_n,
            mode_number_offset=args.mode_number_offset,
        )

        cumulative_rows.append(
            {
                "n_modes": int(n_modes),
                "mode_array_start": 0,
                "mode_array_stop_exclusive": int(n_modes),
                "first_bio3d_mode_number": int(args.mode_number_offset),
                "last_bio3d_mode_number": int(args.mode_number_offset + n_modes - 1),
                "cosine_overlap": float(summary_n["cosine_overlap"]),
                "abs_cosine_overlap": abs(float(summary_n["cosine_overlap"])),
                "rms_error_nm": float(summary_n["rms_error_nm"]),
            }
        )

        coeff_df_n.to_csv(
            outdir / f"linear_combination_first_{n_modes}_mode_coefficients.csv",
            index=False,
        )

        np.save(
            outdir / f"linear_combination_first_{n_modes}_dx_fit.npy",
            summary_n["dx_fit"],
        )

    cumulative_df = pd.DataFrame(cumulative_rows)

    cumulative_df.to_csv(
        outdir / "cumulative_first_modes_fit_summary.csv",
        index=False,
    )

    print("[INFO] Cumulative first-mode combination fits:")
    print(cumulative_df.to_string(index=False))
    
    # ------------------------------------------------------------
    # 10. Main run summary
    # ------------------------------------------------------------
    run_summary = {
        "pdb": pdb_code,
        "replica": args.replica,
        "frame": args.frame,
        "target_mode": args.target_mode,
        "n_protein_heavy_atoms": int(n_atoms),
        "expected_dof": int(expected_dof),
        "n_modes_keep": int(args.n_modes_keep),
        "top_n_atoms": int(args.top_n),
        "fit_start_array_index": int(args.fit_start),
        "fit_stop_array_index_exclusive": int(args.fit_stop),
        "mode_number_offset": int(args.mode_number_offset),
        "best_single_mode_array_index": int(single_df_sorted.iloc[0]["mode_array_index"]),
        "best_single_bio3d_mode_number": int(single_df_sorted.iloc[0]["bio3d_mode_number"]),
        "best_single_q_nm": float(single_df_sorted.iloc[0]["q_nm"]),
        "best_single_cosine_overlap": float(single_df_sorted.iloc[0]["cosine_overlap"]),
        "combination_cosine_overlap": float(combo_summary["cosine_overlap"]),
        "combination_rms_error_nm": float(combo_summary["rms_error_nm"]),
    }

    run_summary.update(target_info)

    save_json_safe(
        outdir / "run_summary.json",
        run_summary,
    )

    pd.DataFrame([run_summary]).to_csv(
        outdir / "run_summary.csv",
        index=False,
    )

    print_header("Done")
    print(f"[DONE] Results written to: {outdir}")
    print("\nRun summary:")
    print(pd.DataFrame([run_summary]).to_string(index=False))


if __name__ == "__main__":
    main()