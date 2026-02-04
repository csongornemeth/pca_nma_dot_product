#!/usr/bin/env python3
from __future__ import annotations

"""
PCA atomic displacement plots (Bio3D-like per-residue PC contributions)

This script keeps your PCA inputs unchanged (all-atom protein-heavy PCA),
but *projects* eigenvectors onto Cα coordinates at plot time using the exact
topology/selection from traj_utils.build_protein_heavy_views().

Inputs (in --eigdir):
  - eigenvectors.npy  (K x 3*Natoms_protein_heavy)
  - eigenvalues.npy   (K,)  [optional; required only for --weighted]

Requires:
  - a valid PDB code so we can rebuild the same protein-heavy topology ordering
    used for PCA, and then locate Cα atoms via MDTraj selection "name CA".

Outputs:
  - a PNG with bar plots of per-residue (Cα) contributions for selected PCs
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from traj_utils import build_protein_heavy_views


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

    returns:
      V_ca: (K, 3*Nres)
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
    title: str | None = None,
) -> None:
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

        ax.set_ylabel(label)
        ax.set_xlim(x[0] - 1, x[-1] + 1)
        if row == n_rows - 1:
            ax.set_xlabel("Residue index (Cα; PCA ordering)")
        else:
            ax.set_xticklabels([])

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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eigdir", required=True, help="Directory containing eigenvectors.npy (and optionally eigenvalues.npy)")
    ap.add_argument("--pdb-code", required=True, help="PDB code (used to rebuild PCA atom ordering and find Cα atoms)")
    ap.add_argument("--pcs", default="1-3", help="PCs to plot, e.g. '1-3' or '1,2,5'")
    ap.add_argument("--weighted", action="store_true", help="Scale each PC by sqrt(eigenvalue) (1σ amplitude). Requires eigenvalues.npy.")
    ap.add_argument("--res-start", type=int, default=1, help="Starting residue index for x-axis (default: 1)")
    ap.add_argument("--out", default="pca_pc_contrib.png", help="Output PNG filename")
    ap.add_argument("--title", default=None, help="Optional figure title")
    args = ap.parse_args()

    eigdir = Path(args.eigdir)
    V_all, lambdas = load_eigs(eigdir)

    # --- rebuild the *exact* protein-heavy topology ordering used by PCA ---
    _, top_xtc_protein, *_ = build_protein_heavy_views(args.pdb_code)

    ca_idx = top_xtc_protein.select("name CA")
    if ca_idx.size == 0:
        raise ValueError("No CA atoms found in protein-heavy topology. Check topology/selection logic.")

    # sanity: eigenvectors must match the protein-heavy atom count
    n_atoms_top = top_xtc_protein.n_atoms
    n_atoms_V = V_all.shape[1] // 3
    if V_all.shape[1] % 3 != 0:
        raise ValueError(f"Eigenvector feature dimension {V_all.shape[1]} is not divisible by 3.")
    if n_atoms_V != n_atoms_top:
        raise ValueError(
            "Mismatch between eigenvectors and protein-heavy topology:\n"
            f"  eigenvectors imply Natoms={n_atoms_V} (from n_features/3)\n"
            f"  topology has         Natoms={n_atoms_top}\n"
            "This usually means the PCA atom selection/order differs from build_protein_heavy_views()."
        )

    # project eigenvectors onto CA coords (residue-level)
    V = slice_eigenvectors_to_ca(V_all, ca_idx)

    pcs = parse_pcs(args.pcs)
    if any(p < 1 for p in pcs):
        raise ValueError("PC indices must be 1-based positive integers, e.g. 1,2,3")

    K = V.shape[0]
    if max(pcs) > K:
        raise ValueError(f"Requested PC{max(pcs)} but only {K} components are available in eigenvectors.npy")

    if args.weighted and lambdas is None:
        raise FileNotFoundError("Requested --weighted but eigenvalues.npy was not found in eigdir")

    contribs: list[np.ndarray] = []
    labels: list[str] = []
    for p in pcs:
        k = p - 1  # convert to 0-based
        if args.weighted:
            contribs.append(pc_contribution_weighted(V, lambdas, k))  # type: ignore[arg-type]
            labels.append(f"PC{p} (√λ scaled)")
        else:
            contribs.append(pc_contribution(V, k))
            labels.append(f"PC{p}")

    out_png = Path(args.out)
    plot_contrib_panels(contribs=contribs, pc_labels=labels, res_start=args.res_start, out_png=out_png, title=args.title)

    print(f"Saved: {out_png}")
    print(f"Eigenvectors (all-atom) shape: {V_all.shape}")
    print(f"Eigenvectors (Cα)      shape: {V.shape}")
    print(f"Protein heavy atoms: {n_atoms_top} | CA atoms (residues): {len(ca_idx)}")
    if lambdas is not None:
        print(f"Eigenvalues shape: {lambdas.shape}")


if __name__ == "__main__":
    main()
