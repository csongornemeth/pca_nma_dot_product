#!/usr/bin/env python3
from __future__ import annotations

"""
PCA atomic displacement plots (Bio3D-like per-residue PC contributions)

Supports:
  A) Manual mode:
     --eigdir results/1a7u/rep0_pca_eigs [results/1a7u/rep1_pca_eigs ...]

  B) Auto-discovery mode (matches your filesystem logic):
     --eigroot results/1a7u
       -> finds results/1a7u/rep*_pca_eigs

If one eigdir is used -> bar plots (original behaviour).
If multiple eigdirs are used -> replica-averaged mean ± (SD/SEM) line+band plots.

Inputs (per eigdir):
  - eigenvectors.npy  (K x 3*Natoms_protein_heavy)
  - eigenvalues.npy   (K,)  [optional; required only for --weighted]
"""

import argparse
from pathlib import Path
import re

import numpy as np
import matplotlib.pyplot as plt

from traj_utils import build_protein_heavy_views


EIGDIR_GLOB_DEFAULT = "rep*_pca_eigs"


def load_eigs(eigdir: Path) -> tuple[np.ndarray, np.ndarray | None]:
    v_path = eigdir / "eigenvectors.npy"
    if not v_path.exists():
        raise FileNotFoundError(f"Missing eigenvectors file: {v_path}")

    V = np.load(v_path)
    if V.ndim != 2:
        raise ValueError(f"Eigenvectors array must be 2D, got shape: {V.shape}")

    l_path = eigdir / "eigenvalues.npy"
    lambdas = np.load(l_path) if l_path.exists() else None
    return V, lambdas


def slice_eigenvectors_to_ca(V: np.ndarray, ca_idx: np.ndarray) -> np.ndarray:
    """
    Project all-atom (protein-heavy) PCA eigenvectors onto Cα coordinates only.

    V: (K, 3*Natoms) with Natoms = protein heavy atoms in the exact PCA ordering
    ca_idx: (Nres,) indices of CA atoms in that same atom ordering

    returns: V_ca (K, 3*Nres)
    """
    if V.shape[1] % 3 != 0:
        raise ValueError(f"Expected 3N features, got n_features={V.shape[1]} (not divisible by 3).")

    Natoms = V.shape[1] // 3
    ca_idx = np.asarray(ca_idx, dtype=int)

    if np.any(ca_idx < 0) or np.any(ca_idx >= Natoms):
        raise ValueError("CA indices are out of bounds for the eigenvector feature dimension.")

    cols = np.ravel(
        np.column_stack([
            3 * ca_idx,
            3 * ca_idx + 1,
            3 * ca_idx + 2,
        ])
    )
    return V[:, cols]


def pc_contribution(V: np.ndarray, k: int) -> np.ndarray:
    """
    Bio3D-like per-residue contribution for PC k (after CA projection):
      c_i = || v_k,i || where v_k,i is (x,y,z) loading for residue i
    """
    vk = V[k]
    if vk.size % 3 != 0:
        raise ValueError(f"Eigenvector length {vk.size} is not a multiple of 3 (expected 3N).")
    return np.linalg.norm(vk.reshape(-1, 3), axis=1)


def pc_contribution_weighted(V: np.ndarray, lambdas: np.ndarray, k: int) -> np.ndarray:
    """
    1-sigma amplitude scaling:
      c_i = sqrt(lambda_k) * || v_k,i ||
    """
    return np.sqrt(float(lambdas[k])) * pc_contribution(V, k)


def plot_contrib_panels(
    contribs: list[np.ndarray],
    pc_labels: list[str],
    res_start: int,
    out_png: Path,
    y_label: str,
    title: str | None = None,
) -> None:
    """Bar-panel plot (single replica)."""
    n_pcs = len(contribs)
    if n_pcs == 0:
        raise ValueError("No PCs to plot.")

    n_cols = 2
    n_rows = (n_pcs + n_cols - 1) // n_cols  # ceil

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(6 * n_cols, 3.6 * n_rows),
        squeeze=False,
    )

    for i, (contrib, label) in enumerate(zip(contribs, pc_labels)):
        row, col = divmod(i, n_cols)
        ax = axes[row][col]

        x = np.arange(res_start, res_start + len(contrib))
        ax.bar(x, contrib, width=1.0, alpha=0.8)

        ax.set_ylabel(f"{label}\n{y_label}")
        ax.set_xlim(x[0] - 1, x[-1] + 1)
        if row == n_rows - 1:
            ax.set_xlabel("Residue index (Cα)")
        else:
            ax.set_xticklabels([])

    # Remove unused axes
    for j in range(n_pcs, n_rows * n_cols):
        row, col = divmod(j, n_cols)
        fig.delaxes(axes[row][col])

    if title:
        fig.suptitle(title)

    fig.tight_layout()
    fig.savefig(out_png, dpi=1000)
    plt.close(fig)


def aggregate_replicas(contribs_stack: np.ndarray, band: str = "std") -> tuple[np.ndarray, np.ndarray]:
    """
    contribs_stack: (Nreplicas, Nres)
    band: "std" (mean ± SD) or "sem" (mean ± SEM)
    """
    contribs_stack = np.asarray(contribs_stack)
    mean = contribs_stack.mean(axis=0)

    if contribs_stack.shape[0] <= 1:
        return mean, np.zeros_like(mean)

    std = contribs_stack.std(axis=0, ddof=1)

    if band == "std":
        return mean, std
    if band == "sem":
        return mean, std / np.sqrt(contribs_stack.shape[0])

    raise ValueError(f"Unknown band type: {band}")


def plot_replica_avg(
    means: list[np.ndarray],
    spreads: list[np.ndarray],
    pc_labels: list[str],
    res_start: int,
    out_png: Path,
    y_label: str,
    title: str | None = None,
    band_label: str = "±1 SD",
) -> None:
    """Line + shaded band panels (replica-averaged)."""
    n_pcs = len(means)
    if n_pcs == 0:
        raise ValueError("No PCs to plot.")

    n_cols = 2
    n_rows = (n_pcs + n_cols - 1) // n_cols

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(6 * n_cols, 3.6 * n_rows),
        squeeze=False,
    )

    for i, (m, s, label) in enumerate(zip(means, spreads, pc_labels)):
        row, col = divmod(i, n_cols)
        ax = axes[row][col]

        x = np.arange(res_start, res_start + len(m))
        ax.plot(x, m, label="Mean", linewidth=2)
        ax.fill_between(x, m - s, m + s, alpha=0.25, label=band_label)

        ax.set_ylabel(f"{label}\n{y_label}")
        ax.set_xlim(x[0] - 1, x[-1] + 1)
        if row == n_rows - 1:
            ax.set_xlabel("Residue index (Cα)")
        else:
            ax.set_xticklabels([])

        ax.legend()

    # Remove unused axes
    for j in range(n_pcs, n_rows * n_cols):
        row, col = divmod(j, n_cols)
        fig.delaxes(axes[row][col])

    if title:
        fig.suptitle(title)

    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def parse_pcs(pcs_str: str) -> list[int]:
    """
    Parse PCs given as '1-3' or '1,2,5' into a list of 1-based ints.
    """
    pcs_str = pcs_str.strip()
    if "-" in pcs_str and "," not in pcs_str:
        a, b = pcs_str.split("-", 1)
        a_i, b_i = int(a), int(b)
        if b_i < a_i:
            raise ValueError(f"Invalid range '{pcs_str}': end < start")
        return list(range(a_i, b_i + 1))
    return [int(x) for x in pcs_str.split(",") if x.strip()]


def _rep_sort_key(p: Path) -> tuple[int, str]:
    """
    Sort paths like rep0_pca_eigs, rep1_pca_eigs numerically if possible.
    Falls back to name sort.
    """
    m = re.search(r"rep(\d+)", p.name)
    if m:
        return (int(m.group(1)), p.name)
    return (10**9, p.name)


def discover_eigdirs(eigroot: Path, pattern: str = EIGDIR_GLOB_DEFAULT) -> list[Path]:
    if not eigroot.exists():
        raise FileNotFoundError(f"--eigroot does not exist: {eigroot}")

    candidates = [p for p in eigroot.glob(pattern) if p.is_dir()]
    candidates.sort(key=_rep_sort_key)

    # keep only those that actually contain eigenvectors.npy
    eigdirs = [p for p in candidates if (p / "eigenvectors.npy").exists()]
    if not eigdirs:
        raise FileNotFoundError(
            f"No eigdirs found under {eigroot} matching '{pattern}' containing eigenvectors.npy"
        )
    return eigdirs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--eigdir",
        nargs="+",
        default=None,
        help="Manual mode: one or more eigdirs (each must contain eigenvectors.npy). "
             "Example: results/1a7u/rep0_pca_eigs results/1a7u/rep1_pca_eigs",
    )
    ap.add_argument(
        "--eigroot",
        default=None,
        help="Auto mode: directory containing replica eigdirs, e.g. results/1a7u",
    )
    ap.add_argument(
        "--eigglob",
        default=EIGDIR_GLOB_DEFAULT,
        help=f"Glob pattern under --eigroot (default: {EIGDIR_GLOB_DEFAULT})",
    )

    ap.add_argument("--pdb-code", required=True, help="PDB code (used to rebuild PCA atom ordering and find Cα atoms)")
    ap.add_argument("--pcs", default="1-3", help="PCs to plot, e.g. '1-3' or '1,2,5'")
    ap.add_argument("--weighted", action="store_true", help="Scale each PC by sqrt(eigenvalue) (1σ amplitude). Requires eigenvalues.npy.")
    ap.add_argument("--band", choices=["std", "sem"], default="std", help="Band type for replica-averaged plots (std or sem).")
    ap.add_argument("--res-start", type=int, default=1, help="Starting residue index for x-axis (default: 1)")
    ap.add_argument("--out", default="pca_pc_contrib.png", help="Output PNG filename")
    ap.add_argument("--title", default=None, help="Optional figure title")
    args = ap.parse_args()

    # Axis labels
    if args.weighted:
        y_label = "Displacement amplitude (Å, 1σ)"
    else:
        y_label = "PCA loading magnitude (a.u.)"

    # Decide eigdirs: manual beats auto if provided
    if args.eigdir is not None and len(args.eigdir) > 0:
        eigdirs = [Path(p) for p in args.eigdir]
    elif args.eigroot is not None:
        eigdirs = discover_eigdirs(Path(args.eigroot), pattern=args.eigglob)
    else:
        raise ValueError("Provide either --eigdir (manual) or --eigroot (auto).")

    # --- rebuild the *exact* protein-heavy topology ordering used by PCA ---
    _, top_xtc_protein, *_ = build_protein_heavy_views(args.pdb_code)

    ca_idx = top_xtc_protein.select("name CA")
    if ca_idx.size == 0:
        raise ValueError("No CA atoms found in protein-heavy topology. Check topology/selection logic.")

    pcs = parse_pcs(args.pcs)
    if any(p < 1 for p in pcs):
        raise ValueError("PC indices must be 1-based positive integers, e.g. 1,2,3")

    n_atoms_top = top_xtc_protein.n_atoms

    # Collect per-PC contributions across replicas
    per_pc_rep_contribs: dict[int, list[np.ndarray]] = {p: [] for p in pcs}
    labels: list[str] = [f"PC{p} (√λ scaled)" if args.weighted else f"PC{p}" for p in pcs]

    for eigdir in eigdirs:
        V_all, lambdas = load_eigs(eigdir)

        if V_all.shape[1] % 3 != 0:
            raise ValueError(f"[{eigdir}] Eigenvector feature dimension {V_all.shape[1]} is not divisible by 3.")

        n_atoms_V = V_all.shape[1] // 3
        if n_atoms_V != n_atoms_top:
            raise ValueError(
                f"[{eigdir}] Mismatch between eigenvectors and protein-heavy topology:\n"
                f"  eigenvectors imply Natoms={n_atoms_V} (from n_features/3)\n"
                f"  topology has         Natoms={n_atoms_top}\n"
                "This usually means the PCA atom selection/order differs from build_protein_heavy_views()."
            )

        if args.weighted and lambdas is None:
            raise FileNotFoundError(f"[{eigdir}] Requested --weighted but eigenvalues.npy was not found in eigdir")

        # project eigenvectors onto CA coords (residue-level)
        V = slice_eigenvectors_to_ca(V_all, ca_idx)

        K = V.shape[0]
        if max(pcs) > K:
            raise ValueError(f"[{eigdir}] Requested PC{max(pcs)} but only {K} components are available in eigenvectors.npy")

        for p in pcs:
            k = p - 1  # 0-based
            if args.weighted:
                c = pc_contribution_weighted(V, lambdas, k)  # type: ignore[arg-type]
            else:
                c = pc_contribution(V, k)
            per_pc_rep_contribs[p].append(c)

    out_png = Path(args.out)

    if len(eigdirs) == 1:
        contribs = [per_pc_rep_contribs[p][0] for p in pcs]
        plot_contrib_panels(
            contribs=contribs,
            pc_labels=labels,
            y_label=y_label,
            res_start=args.res_start,
            out_png=out_png,
            title=args.title,
        )
    else:
        means: list[np.ndarray] = []
        spreads: list[np.ndarray] = []
        for p in pcs:
            stack = np.stack(per_pc_rep_contribs[p], axis=0)  # (Nrep, Nres)
            m, s = aggregate_replicas(stack, band=args.band)
            means.append(m)
            spreads.append(s)

        band_label = "±1 SEM" if args.band == "sem" else "±1 SD"
        plot_replica_avg(
            means=means,
            spreads=spreads,
            pc_labels=labels,
            res_start=args.res_start,
            y_label=y_label,
            out_png=out_png,
            title=args.title,
            band_label=band_label,
        )

    print(f"Saved: {out_png}")
    print(f"Replicas: {len(eigdirs)} | PCs: {pcs} | band: {args.band if len(eigdirs) > 1 else 'n/a'}")
    print(f"Protein heavy atoms: {n_atoms_top} | CA atoms (residues): {len(ca_idx)}")
    print("Eigdirs used:")
    for p in eigdirs:
        print(f"  - {p}")


if __name__ == "__main__":
    main()
