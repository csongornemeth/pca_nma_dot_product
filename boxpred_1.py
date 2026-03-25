# boxpred_1.py

from __future__ import annotations

"""
Predicts a dodecahedron box size for a given protein-ligand system based on:
1) COM-distance profile of protein heavy atoms (relative to ligand COM)
2) PCA-based flexibility profile of protein residues (based on CA atoms)
The tool selects the residues that pass both filters and computes a radius for the box size
Usage example:
python boxpred_1.py \
  --pdb 2qhr \
  --eigroot results/2qhr \
  --pcs 1-5 \
  --weighted \
  --verbose
"""

from pathlib import Path
import argparse
import re
import numpy as np
import mdtraj as md

from io_utils import (
    get_pdb_dir,
    collect_xtc_paths,
    print_header,
)
from pca_adp_overlay import (
    discover_eigdirs,
    load_eigs,
    slice_eigenvectors_to_ca,
    pc_contribution,
    pc_contribution_weighted,
    parse_pcs as parse_pcs_overlay,
    EIGDIR_GLOB_DEFAULT,
)
from traj_utils import build_protein_heavy_views
from align_core import compute_alignment_core_aidxs
from com_profile import compute_com_profile


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Select flexible and COM-distance residues for dodecahedron box prediction."
    )
    ap.add_argument(
        "--pdb",
        required=True,
        help="PDB id of the structure",
    )
    ap.add_argument(
        "--eigroot",
        default=None,
        help="Directory containing rep*_pca_eigs, e.g. results/<pdbid>",
    )
    ap.add_argument(
        "--eigdir",
        nargs="+",
        default=None,
        help="Manual eigdirs",
    )
    ap.add_argument(
        "--eigglob",
        default=EIGDIR_GLOB_DEFAULT,
        help=f"Glob pattern under --eigroot (default: {EIGDIR_GLOB_DEFAULT})",
    )
    ap.add_argument(
        "--pcs",
        default="1-3",
        help="Principal components to use for selection, e.g. 1,2,5 or 1-3",
    )
    ap.add_argument(
        "--weighted",
        action="store_true",
        help="Use sqrt(eigenvalue)-weighted PCA displacement",
    )
    ap.add_argument(
        "--com_percentile",
        type=float,
        default=80.0,
        help="Percentile cutoff for COM-distance residue selection",
    )
    ap.add_argument(
        "--com_stat",
        choices=["mean", "p95", "max"],
        default="p95",
        help="Statistic to use for COM-distance profile",
    )
    ap.add_argument(
        "--pca_percentile",
        type=float,
        default=80.0,
        help="Percentile cutoff for PCA-displacement residue selection",
    )
    ap.add_argument(
        "--radius_stat",
        choices=["p95", "p99", "max"],
        default="p99",
        help="Statistic to use for final selected-atom COM radius",
    )
    ap.add_argument(
        "--padding",
        type=float,
        default=1.0,
        help="Padding distance in nm to add to the chosen radius",
    )
    ap.add_argument(
        "--core_max_frames",
        type=int,
        default=10000,
        help="Maximum number of frames to use for core alignment",
    )
    ap.add_argument(
        "--com_max_frames",
        type=int,
        default=10000,
        help="Maximum number of frames to use for COM profile calculation",
    )
    ap.add_argument(
        "--radius_max_frames",
        type=int,
        default=20000,
        help="Maximum number of frames to use for radius calculation",
    )
    ap.add_argument(
        "--chunk",
        type=int,
        default=2500,
        help="Chunk size for processing trajectories",
    )
    ap.add_argument(
        "--verbose",
        action="store_true",
        help="Print verbose output",
    )

    return ap.parse_args()

def rep_sort_key(repdir: Path) -> tuple[int, str]:
    m = re.match(r"rep(\d+)", repdir.name)
    if m:
        return (int(m.group(1)), repdir.name)
    return (10**9, repdir.name)

def group_heavy_atoms_by_residue(topology: md.Topology) -> dict[int, np.ndarray]:
    residue_to_atoms: dict[int, list[int]] = {}
    for atom in topology.atoms:
        resid = atom.residue.index
        residue_to_atoms.setdefault(resid, []).append(atom.index)
    return {
        resid: np.asarray(atom_indices, dtype=int)
        for resid, atom_indices in residue_to_atoms.items()
    }


def compute_residue_pca_score(
    eigdirs: list[Path],
    top_xtc_protein: md.Topology,
    pcs: list[int],
    weighted: bool = True,
) -> np.ndarray:
    ca_idx = np.asarray(top_xtc_protein.select("name CA"), dtype=int)
    if ca_idx.size == 0:
        raise ValueError("No CA atoms found in protein-heavy topology.")

    n_atoms_top = top_xtc_protein.n_atoms
    per_rep_scores: list[np.ndarray] = []

    for eigdir in eigdirs:
        V_all, lambdas = load_eigs(eigdir)

        if V_all.ndim != 2 or V_all.shape[1] % 3 != 0:
            raise ValueError(f"[{eigdir}] Invalid eigenvector shape: {V_all.shape}")

        n_atoms_v = V_all.shape[1] // 3
        if n_atoms_v != n_atoms_top:
            raise ValueError(
                f"[{eigdir}] Topology/eigenvector atom-count mismatch: "
                f"{n_atoms_v} vs {n_atoms_top}"
            )

        if weighted and lambdas is None:
            raise FileNotFoundError(
                f"[{eigdir}] --weighted requested but eigenvalues.npy is missing"
            )

        V_ca = slice_eigenvectors_to_ca(V_all, ca_idx)

        n_modes = V_ca.shape[0]
        if max(pcs) > n_modes:
            raise ValueError(
                f"[{eigdir}] Requested PC{max(pcs)} but only {n_modes} PCs are available"
            )

        score = np.zeros(ca_idx.size, dtype=float)
        for pc in pcs:
            k = pc - 1
            if weighted:
                score += pc_contribution_weighted(V_ca, lambdas, k)
            else:
                score += pc_contribution(V_ca, k)

        per_rep_scores.append(score)

    return np.mean(np.stack(per_rep_scores, axis=0), axis=0)

def select_res_by_com_pca(
        com_profile: np.ndarray,
        pca_score: np.ndarray,
        com_percentile: float,
        pca_percentile: float,
) -> tuple[np.ndarray, float, float]:
    
    if com_profile.shape != pca_score.shape:
        raise ValueError(f"COM profile and PCA score must have the same shape, "
                         f"got {com_profile.shape} and {pca_score.shape}")
    
    com_cut = np.percentile(com_profile, com_percentile)
    pca_cut = np.percentile(pca_score, pca_percentile)

    selected = np.where(
        (com_profile >= com_cut) & (pca_score >= pca_cut)
    )[0]

    if selected.size == 0:
        raise RuntimeError("No residues selected - lower percentiles")
    
    return selected, com_cut, pca_cut

def res_to_heavy_atom_indices(
    selected_residues: np.ndarray,
    residue_to_atoms: dict[int, np.ndarray],
) -> np.ndarray:
    
    atom_groups = []

    for resid in selected_residues:
        if int(resid) in residue_to_atoms:
            atom_groups.append(residue_to_atoms[int(resid)])

    if not atom_groups:
        raise RuntimeError("No heavy atoms found for selected residues")
    
    return np.unique(np.concatenate(atom_groups))

def compute_radius_profile(
    xtc_paths: list[Path],
    top_xtc_full: md.Topology,
    protein_heavy_idx_full: np.ndarray,
    selected_atom_indices: np.ndarray,
    core_aidxs: np.ndarray,
    radius_stat: str,
    max_frames: int,
    chunk: int,
) -> tuple[float, dict]:

    ref = None
    distances = []
    n_frames = 0

    for xtc in xtc_paths:
        for tr in md.iterload(xtc.as_posix(), top=top_xtc_full, chunk=chunk):

            trp = tr.atom_slice(protein_heavy_idx_full)

            if ref is None:
                ref = trp[0]

            trp.superpose(ref, 0, core_aidxs)

            com = md.compute_center_of_mass(trp)
            xyz = trp.xyz[:, selected_atom_indices, :]

            d = np.linalg.norm(xyz - com[:, None, :], axis=2)

            distances.append(d.reshape(-1))
            n_frames += trp.n_frames

            if n_frames >= max_frames:
                break
        if n_frames >= max_frames:
            break

    if not distances:
        raise RuntimeError("No frames processed for radius calculation")

    all_d = np.concatenate(distances)

    stats = {
        "p95": float(np.percentile(all_d, 95)),
        "p99": float(np.percentile(all_d, 99)),
        "max": float(np.max(all_d)),
    }

    return stats[radius_stat], stats

def main():
    args = parse_args()
    pdb_code = args.pdb.lower()

    print_header(f"Box predictor - {pdb_code}")

    pdb_dir = get_pdb_dir(pdb_code)
    xtc_paths = collect_xtc_paths(pdb_dir)

    #get heavy-only protein trajectory and topology views
    (
        traj_protein_heavy_ref,
        top_xtc_protein,
        protein_heavy_idx_local,
        protein_heavy_idx_full,
        top_xtc_full,
    ) = build_protein_heavy_views(pdb_code)
    
    #core alignment
    core_aidxs = compute_alignment_core_aidxs(
        xtc_paths=xtc_paths,
        topology=top_xtc_full,
        atom_indices_full=protein_heavy_idx_full,
        max_frames=args.core_max_frames,
        verbose=args.verbose,
    )

    #COM profile
    com_prof = compute_com_profile(
        xtc_paths=xtc_paths,
        top_xtc_full=top_xtc_full,
        protein_heavy_idx_full=protein_heavy_idx_full,
        top_xtc_protein=top_xtc_protein,
        stat = args.com_stat,
        max_core_frames=args.core_max_frames,
        max_com_frames=args.com_max_frames,
        chunk=args.chunk,
        verbose=args.verbose,
    )

    #PCA profile
    if args.eigdir:
        eigdirs = [Path(p) for p in args.eigdir]
    else:
        eigroot = Path(args.eigroot) if args.eigroot else Path("results") / pdb_code
        eigdirs = discover_eigdirs(eigroot, args.eigglob)
    
    pcs = parse_pcs_overlay(args.pcs)

    pca_score = compute_residue_pca_score(
        eigdirs = eigdirs,
        top_xtc_protein = top_xtc_protein,
        pcs = pcs,
        weighted = args.weighted,
    )

    #selection of atoms based on COM and PCA profiles
    select_residues, com_cut, pca_cut = select_res_by_com_pca(
        com_prof,
        pca_score,
        args.com_percentile,
        args.pca_percentile,
    )

    print(f"[BOX] Selected {select_residues.size} residues based on COM and PCA profiles")

    #atoms corresponding to selected residues
    residue_map = group_heavy_atoms_by_residue(top_xtc_protein)

    selected_atoms = res_to_heavy_atom_indices(
        selected_residues=select_residues,
        residue_to_atoms=residue_map,
    )

    print(f"[BOX] Selected {selected_atoms.size} heavy atoms in selected residues")

    #Radious calculation
    radius, stats = compute_radius_profile(
        xtc_paths,
        top_xtc_full,
        protein_heavy_idx_full,
        selected_atoms,
        core_aidxs,
        args.radius_stat,
        args.radius_max_frames,
        args.chunk,
    )

    print(f"\n[BOX] Computed radius: {radius:.3f} nm; stats: [{stats}]")
    print(f"  p95 : {stats['p95']:.3f}")
    print(f"  p99 : {stats['p99']:.3f}")
    print(f"  max : {stats['max']:.3f}")

    clearance = radius + args.padding

    print(f"\n[BOX] Recommended box radius (with {args.padding:.1f} nm padding): {clearance:.3f} nm")
    print("\n[BOX] Recommendation")
    print(f"  radius ({args.radius_stat}) : {radius:.3f} nm")
    print(f"  padding                    : {args.padding:.3f} nm")
    print(f"  recommended clearance      : {clearance:.3f} nm")

if __name__ == "__main__":    
    main()
