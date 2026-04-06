from pathlib import Path
import argparse
import numpy as np
import mdtraj as md
import csv

from pca_adp_overlay import (
    discover_eigdirs,
    load_eigs,
    slice_eigenvectors_to_ca,
    parse_pcs,
    EIGDIR_GLOB_DEFAULT,
)
from io_utils import (
    get_pdb_dir,
    print_header,
    load_full_topology,
)
from traj_utils import build_protein_heavy_views


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdb", required=True, help="PDB code")
    ap.add_argument(
        "--eigroot",
        default=None,
        help="Root directory containing eigenvector result directories",
    )
    ap.add_argument(
        "--eigdir",
        nargs="+",
        default=None,
        help="One or more explicit eigenvector directories",
    )
    ap.add_argument(
        "--eigglob",
        default=EIGDIR_GLOB_DEFAULT,
        help="Glob pattern for eigenvector directories",
    )
    ap.add_argument(
        "--pcs",
        default="1-3",
        help="PCs to compute displacement for, e.g. '1-3' or '1,2,5'",
    )
    ap.add_argument(
        "--sigma",
        type=float,
        default=1.0,
        help="Number of standard deviations for displacement amplitude",
    )
    ap.add_argument(
        "--debug",
        action="store_true",
        help="Print debug information about atom-count matching",
    )
    ap.add_argument(
        "--outcsv",
        default="pca_displacement_summary.csv",
        help="Output CSV file for displacement summary",
    )
    return ap.parse_args()


def max_compute_pca_displacement_nm(V, eigvals, pc, sigma=1.0):
    """
    Compute per-atom displacement amplitude in nm for one PC.

    Parameters
    ----------
    V : ndarray, shape (n_pcs, 3N)
        PCA eigenvectors
    eigvals : ndarray, shape (n_pcs,)
        PCA eigenvalues
    pc : int
        1-based PC index
    sigma : float
        Number of standard deviations along the PC

    Returns
    -------
    dict with:
        max_disp : float
            Maximum atomic displacement in nm
        atom_index : int
            Atom with largest displacement
        all_disp_nm : ndarray, shape (N,)
            Per-atom displacement amplitudes in nm
    """
    k = pc - 1
    if k < 0 or k >= len(eigvals):
        raise IndexError(f"Requested PC{pc}, but only {len(eigvals)} PCs are available")

    n_atoms = V.shape[1] // 3
    vk = V[k].reshape(n_atoms, 3)

    disp = sigma * np.sqrt(eigvals[k]) * np.linalg.norm(vk, axis=1)
    imax = int(np.argmax(disp))

    return {
        "max_disp": float(disp[imax]),
        "atom_index": imax,
        "all_disp_nm": disp,
    }


def max_multi_pc_displacement_nm(V, eigvals, pcs, sigma=1.0):
    """
    Compute per-atom displacement amplitude in nm from multiple PCs combined.

    For each atom i:
        d_i = sigma * sqrt(sum_k eigvals[k] * ||v_{k,i}||^2)
    """
    n_atoms = V.shape[1] // 3
    disp2 = np.zeros(n_atoms, dtype=float)

    for pc in pcs:
        k = pc - 1
        if k < 0 or k >= len(eigvals):
            raise IndexError(f"Requested PC{pc}, but only {len(eigvals)} PCs are available")
        vk = V[k].reshape(n_atoms, 3)
        disp2 += eigvals[k] * np.sum(vk**2, axis=1)

    disp = sigma * np.sqrt(disp2)
    imax = int(np.argmax(disp))

    return {
        "max_disp": float(disp[imax]),
        "atom_index": imax,
        "all_disp_nm": disp,
    }


def format_atom_label(topology, atom_index):
    atom = topology.atom(atom_index)
    return f"{atom.residue.name}{atom.residue.resSeq}:{atom.name}"


def summarise_disp(disp):
    return {
        "mean": float(np.mean(disp)),
        "p95": float(np.percentile(disp, 95)),
        "p99": float(np.percentile(disp, 99)),
        "max": float(np.max(disp)),
    }


def resolve_eigdirs(args):
    if args.eigdir is not None:
        eigdirs = [Path(x) for x in args.eigdir]
    else:
        eigroot = args.eigroot or f"results/{args.pdb.lower()}"
        eigdirs = discover_eigdirs(Path(eigroot), args.eigglob)

    if not eigdirs:
        raise RuntimeError("No eigenvector directories found")

    return eigdirs


def choose_matching_topology(V, top_full, protein_heavy_top, ca_idx_local, debug=False):
    """
    Choose the topology that matches the PCA eigenvector atom count.

    Matching priority:
    1. full topology
    2. protein heavy topology
    3. C-alpha topology
    """
    n_atoms_vec = V.shape[1] // 3

    candidates = []

    # full system
    candidates.append(("full", V, top_full))

    # protein heavy
    candidates.append(("protein_heavy", V, protein_heavy_top))

    # C-alpha
    V_ca = slice_eigenvectors_to_ca(V, ca_idx_local)
    ca_top = protein_heavy_top.subset(ca_idx_local)
    candidates.append(("ca", V_ca, ca_top))

    if debug:
        print(f"[DEBUG] PCA atoms before any slicing: {n_atoms_vec}")
        print(f"[DEBUG] Full topology atoms:          {top_full.n_atoms}")
        print(f"[DEBUG] Protein-heavy atoms:         {protein_heavy_top.n_atoms}")
        print(f"[DEBUG] C-alpha atoms:               {ca_top.n_atoms}")

    for mode_name, V_use, top_use in candidates:
        n_match = V_use.shape[1] // 3
        if n_match == top_use.n_atoms:
            if debug:
                print(f"[DEBUG] Matched topology mode: {mode_name}")
            return mode_name, V_use, top_use

    raise ValueError(
        "Could not match PCA atom count to any known topology view.\n"
        f"PCA atoms: {n_atoms_vec}\n"
        f"Full topology atoms: {top_full.n_atoms}\n"
        f"Protein-heavy atoms: {protein_heavy_top.n_atoms}\n"
        f"C-alpha atoms: {ca_top.n_atoms}"
    )


def main():
    args = parse_args()

    pdb_code = args.pdb.lower()
    pdb_dir = get_pdb_dir(pdb_code)

    traj_full, top_full = load_full_topology(pdb_dir)
    print(f"[MAIN] Full topology natoms (npt.gro): {top_full.n_atoms}")

    pcs = parse_pcs(args.pcs)
    if not pcs:
        raise ValueError("No valid PCs were parsed from --pcs")

    eigdirs = resolve_eigdirs(args)

    print_header("PCA Displacement Analysis")
    print(f"PDB: {pdb_code}")
    print(f"PCs: {pcs}")
    print(f"Sigma: {args.sigma}")
    print(f"Found {len(eigdirs)} eigdir(s)\n")

    (
        protein_heavy_traj,
        protein_heavy_top,
        protein_heavy_idx_local,
        protein_heavy_idx_full,
        top_xtc_full,
    ) = build_protein_heavy_views(pdb_code)

    ca_idx_local = protein_heavy_top.select("name CA")
    protein_residue_count = protein_heavy_top.n_residues
    print(f"Protein residue count: {protein_residue_count}")

    results_rows = []

    for eigdir in eigdirs:
        print("=" * 80)
        print(f"Eigenvector directory: {eigdir}")

        V, eigvals = load_eigs(eigdir)

        if args.debug:
            print(f"[DEBUG] V.shape = {V.shape}")
            print(f"[DEBUG] eigvals.shape = {eigvals.shape}")

        mode_name, V_use, atom_top = choose_matching_topology(
            V,
            top_full=top_full,
            protein_heavy_top=protein_heavy_top,
            ca_idx_local=ca_idx_local,
            debug=args.debug,
        )

        print(f"Matched topology mode: {mode_name}")
        print(f"Atom count used: {atom_top.n_atoms}")

        print("\nSingle-PC displacements:")
        for pc in pcs:
            result = max_compute_pca_displacement_nm(
                V_use, eigvals, pc, sigma=args.sigma
            )

            atom_label = format_atom_label(atom_top, result["atom_index"])
            stats = summarise_disp(result["all_disp_nm"])

            print(
                f"  PC{pc:>3}: "
                f"max_disp = {result['max_disp']:.4f} nm ; "
                f"atom = {result['atom_index']} ({atom_label})"
            )
            print(
                f"         mean = {stats['mean']:.4f} nm ; "
                f"p95 = {stats['p95']:.4f} nm ; "
                f"p99 = {stats['p99']:.4f} nm"
            )

            results_rows.append({
                "pdb": pdb_code,
                "eigdir": str(eigdir),
                "topology_mode": mode_name,
                "pc_set": f"PC{pc}",
                "sigma": args.sigma,
                "atom_count": atom_top.n_atoms,
                "protein_residue_count": protein_residue_count,
                "max_disp_nm": result["max_disp"],
                "max_atom_index": result["atom_index"],
                "max_atom_label": atom_label,
                "mean_disp_nm": stats["mean"],
                "p95_disp_nm": stats["p95"],
                "p99_disp_nm": stats["p99"],
            })

        if len(pcs) > 1:
            result_multi = max_multi_pc_displacement_nm(
                V_use, eigvals, pcs, sigma=args.sigma
            )

            atom_label = format_atom_label(atom_top, result_multi["atom_index"])
            pcs_str = ",".join(map(str, pcs))
            stats = summarise_disp(result_multi["all_disp_nm"])

            print("\nCombined-PC displacement:")
            print(
                f"  PCs [{pcs_str}]: "
                f"max_disp = {result_multi['max_disp']:.4f} nm ; "
                f"atom = {result_multi['atom_index']} ({atom_label})"
            )
            print(
                f"         mean = {stats['mean']:.4f} nm ; "
                f"p95 = {stats['p95']:.4f} nm ; "
                f"p99 = {stats['p99']:.4f} nm"
            )

            results_rows.append({
                "pdb": pdb_code,
                "eigdir": str(eigdir),
                "topology_mode": mode_name,
                "pc_set": f"PCs[{pcs_str}]",
                "sigma": args.sigma,
                "atom_count": atom_top.n_atoms,
                "protein_residue_count": protein_residue_count,
                "max_disp_nm": result_multi["max_disp"],
                "max_atom_index": result_multi["atom_index"],
                "max_atom_label": atom_label,
                "mean_disp_nm": stats["mean"],
                "p95_disp_nm": stats["p95"],
                "p99_disp_nm": stats["p99"],
            })

        print()

    outcsv_path = Path(args.outcsv).resolve()
    outcsv_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[MAIN] Number of result rows: {len(results_rows)}")
    print(f"[MAIN] Writing CSV to: {outcsv_path}")

    if results_rows:
        fieldnames = [
            "pdb",
            "eigdir",
            "topology_mode",
            "pc_set",
            "sigma",
            "atom_count",
            "protein_residue_count",
            "max_disp_nm",
            "max_atom_index",
            "max_atom_label",
            "mean_disp_nm",
            "p95_disp_nm",
            "p99_disp_nm",
        ]

        with open(outcsv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results_rows)

        print(f"[MAIN] Wrote CSV successfully.")
    else:
        print("[MAIN] No results to write → CSV NOT created.")


if __name__ == "__main__":
    main()