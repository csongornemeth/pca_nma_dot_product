from __future__ import annotations

"""
Anisotropic PCA-aligned triclinic/orthorhombic box prediction.
1) COM distance profile
2) PCA displacement profile
3) Select residues based on the above two
4) Estimate the axes of the box based on the movements
5) Measure the distances along said axes
6) Define box dimensions based on the above

Basic run commands:
        python boxpred2.py \
            --pdb 1a7q \
            --eigroot results/1a7q \
            --pcs 1-20 \
            --weighted \
            --extent_stat max \
            --padding 1.0 \
            --verbose
"""

from pathlib import Path
import argparse
import json
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
        description="Predict box dimensions from a trajectory using PCA-alignment and COM distance profiles.",
    )
    ap.add_argument(
        "--pdb",
        type=str,
        required=True,
        help="PDB code of the system (e.g. 1a7u)",
    )
    ap.add_argument(
        "--eigdir",
        nargs="+",
        default=None,
        help="Manual eigdirs",
    )
    ap.add_argument(
        "--eigroot",
        default=None,
        help="Directory for containing the eigenvector files.",
    )
    ap.add_argument(
        "--eigglob",
        default=EIGDIR_GLOB_DEFAULT,
        help=f"Glob pattern under --eigroot (default: {EIGDIR_GLOB_DEFAULT})",
    )
    ap.add_argument(
        "--pcs",
        default="1-20",
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
        default="max",
        help="Statistic to use for COM-distance profile",
    )
    ap.add_argument(
        "--pca_percentile",
        type=float,
        default=80.0,
        help="Percentile cutoff for PCA-displacement residue selection",
    )
    ap.add_argument(
        "--extent_stat",
        choices=["p95", "p99", "max"],
        default="max",
        help="Statistic to use for directional half-extents",
    )
    ap.add_argument(
        "--padding",
        type=float,
        default=0.8,
        help="Padding distance in nm to add to each half-extent",
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
        "--axes_max_frames",
        type=int,
        default=10000,
        help="Maximum number of frames to use for principal-axis estimation",
    )
    ap.add_argument(
        "--extent_max_frames",
        type=int,
        default=10000,
        help="Maximum number of frames to use for directional extent calculation",
    )
    ap.add_argument(
        "--chunk",
        type=int,
        default=2500,
        help="Chunk size for processing trajectories",
    )
    ap.add_argument(
        "--outdir",
        default=None,
        help="Output directory (default: results/<pdbid>/triclinic_box)",
    )
    ap.add_argument(
        "--save_projections",
        action="store_true",
        help="Save projected coordinates used for extent statistics",
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
        raise ValueError(
            "COM profile and PCA score must have the same shape, "
            f"got {com_profile.shape} and {pca_score.shape}"
        )

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


def _iter_selected_centered_xyz(
    xtc_paths: list[Path],
    top_xtc_full: md.Topology,
    protein_heavy_idx_full: np.ndarray,
    selected_atom_indices_local: np.ndarray,
    core_aidxs: np.ndarray,
    max_frames: int,
    chunk: int,
):
    """
    Yield arrays of shape (n_points, 3), where points are selected heavy atoms
    from aligned trajectories, centered on the protein-heavy COM.
    """
    ref = None
    n_frames = 0

    for xtc in xtc_paths:
        for tr in md.iterload(xtc.as_posix(), top=top_xtc_full, chunk=chunk):
            trp = tr.atom_slice(protein_heavy_idx_full)

            if ref is None:
                ref = trp[0]

            trp.superpose(ref, 0, core_aidxs)

            com = md.compute_center_of_mass(trp)  # (n_frames, 3)
            xyz = trp.xyz[:, selected_atom_indices_local, :]  # (n_frames, n_sel, 3)
            centered = xyz - com[:, None, :]

            yield centered.reshape(-1, 3)

            n_frames += trp.n_frames
            if n_frames >= max_frames:
                return


def compute_principal_axes(
    xtc_paths: list[Path],
    top_xtc_full: md.Topology,
    protein_heavy_idx_full: np.ndarray,
    selected_atom_indices_local: np.ndarray,
    core_aidxs: np.ndarray,
    max_frames: int,
    chunk: int,
    verbose: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Estimate principal axes from pooled selected-atom coordinates across frames.

    Returns
    -------
    axes : (3, 3)
        Rows are principal-axis unit vectors in Cartesian space.
    evals : (3,)
        Eigenvalues of the covariance matrix.
    mean_vec : (3,)
        Mean of pooled centered coordinates (usually near zero, but not assumed).
    """
    n = 0
    sum_x = np.zeros(3, dtype=float)
    sum_xx = np.zeros((3, 3), dtype=float)

    for pts in _iter_selected_centered_xyz(
        xtc_paths=xtc_paths,
        top_xtc_full=top_xtc_full,
        protein_heavy_idx_full=protein_heavy_idx_full,
        selected_atom_indices_local=selected_atom_indices_local,
        core_aidxs=core_aidxs,
        max_frames=max_frames,
        chunk=chunk,
    ):
        if pts.size == 0:
            continue
        n_chunk = pts.shape[0]
        n += n_chunk
        sum_x += pts.sum(axis=0)
        sum_xx += pts.T @ pts

    if n == 0:
        raise RuntimeError("No coordinates available for principal-axis estimation")

    mean_vec = sum_x / n
    cov = (sum_xx / n) - np.outer(mean_vec, mean_vec)

    evals, evecs = np.linalg.eigh(cov)
    order = np.argsort(evals)[::-1]
    evals = evals[order]
    evecs = evecs[:, order]

    # Use rows as axes for easier projection: proj = pts @ axes.T
    axes = evecs.T

    # Enforce right-handed basis
    if np.linalg.det(axes) < 0:
        axes[2] *= -1.0

    if verbose:
        print("[AXES] Covariance eigenvalues:", ", ".join(f"{x:.6f}" for x in evals))

    return axes, evals, mean_vec


def compute_directional_extents(
    xtc_paths: list[Path],
    top_xtc_full: md.Topology,
    protein_heavy_idx_full: np.ndarray,
    selected_atom_indices_local: np.ndarray,
    core_aidxs: np.ndarray,
    axes: np.ndarray,
    extent_stat: str,
    max_frames: int,
    chunk: int,
    save_projections: bool = False,
) -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray | None]:
    """
    Project centered coordinates onto the principal axes and compute
    directional half-extents.

    Returns
    -------
    half_extents : (3,)
        Chosen directional half-extents along principal axes.
    stats : dict[str, np.ndarray]
        p95, p99, max arrays each with shape (3,)
    projections_all : (N, 3) or None
        Optional saved projections.
    """
    proj_chunks = []

    for pts in _iter_selected_centered_xyz(
        xtc_paths=xtc_paths,
        top_xtc_full=top_xtc_full,
        protein_heavy_idx_full=protein_heavy_idx_full,
        selected_atom_indices_local=selected_atom_indices_local,
        core_aidxs=core_aidxs,
        max_frames=max_frames,
        chunk=chunk,
    ):
        if pts.size == 0:
            continue
        proj = pts @ axes.T
        proj_chunks.append(proj)

    if not proj_chunks:
        raise RuntimeError("No coordinates available for extent calculation")

    projections = np.concatenate(proj_chunks, axis=0)
    abs_proj = np.abs(projections)

    stats = {
        "p95": np.percentile(abs_proj, 95, axis=0),
        "p99": np.percentile(abs_proj, 99, axis=0),
        "max": np.max(abs_proj, axis=0),
    }
    half_extents = stats[extent_stat]

    return half_extents, stats, (projections if save_projections else None)


def main():
    args = parse_args()
    pdb_code = args.pdb.lower()

    outdir = Path(args.outdir) if args.outdir else Path("results") / pdb_code / "triclinic_box"
    outdir.mkdir(parents=True, exist_ok=True)

    print_header(f"Anisotropic triclinic box predictor - {pdb_code}")
    print(f"Output dir: {outdir}")

    pdb_dir = get_pdb_dir(pdb_code)
    xtc_paths = collect_xtc_paths(pdb_dir)

    if not xtc_paths:
        raise RuntimeError(f"No trajectories found for {pdb_code}")

    (
        traj_protein_heavy_ref,
        top_xtc_protein,
        protein_heavy_idx_local,
        protein_heavy_idx_full,
        top_xtc_full,
    ) = build_protein_heavy_views(pdb_code)

    # Core alignment atoms
    core_aidxs = compute_alignment_core_aidxs(
        xtc_paths=xtc_paths,
        topology=top_xtc_full,
        atom_indices_full=protein_heavy_idx_full,
        max_frames=args.core_max_frames,
        verbose=args.verbose,
    )

    # COM-distance profile
    com_prof = compute_com_profile(
        xtc_paths=xtc_paths,
        top_xtc_full=top_xtc_full,
        protein_heavy_idx_full=protein_heavy_idx_full,
        top_xtc_protein=top_xtc_protein,
        stat=args.com_stat,
        max_core_frames=args.core_max_frames,
        max_com_frames=args.com_max_frames,
        chunk=args.chunk,
        verbose=args.verbose,
    )

    # PCA profile
    if args.eigdir:
        eigdirs = [Path(p) for p in args.eigdir]
    else:
        eigroot = Path(args.eigroot) if args.eigroot else Path("results") / pdb_code
        eigdirs = discover_eigdirs(eigroot, args.eigglob)

    if not eigdirs:
        raise RuntimeError("No eigdirs found")

    pcs = parse_pcs_overlay(args.pcs)

    pca_score = compute_residue_pca_score(
        eigdirs=eigdirs,
        top_xtc_protein=top_xtc_protein,
        pcs=pcs,
        weighted=args.weighted,
    )

    # Select residues based on COM and PCA profiles
    selected_residues, com_cut, pca_cut = select_res_by_com_pca(
        com_prof,
        pca_score,
        args.com_percentile,
        args.pca_percentile,
    )

    print(f"[SEL] Selected {selected_residues.size} residues based on COM and PCA filters")

    residue_map = group_heavy_atoms_by_residue(top_xtc_protein)
    selected_atoms = res_to_heavy_atom_indices(
        selected_residues=selected_residues,
        residue_to_atoms=residue_map,
    )
    print(f"[SEL] Selected {selected_atoms.size} heavy atoms in selected residues")

    # Principal axes from pooled selected-atom coordinates
    axes, evals, mean_vec = compute_principal_axes(
        xtc_paths=xtc_paths,
        top_xtc_full=top_xtc_full,
        protein_heavy_idx_full=protein_heavy_idx_full,
        selected_atom_indices_local=selected_atoms,
        core_aidxs=core_aidxs,
        max_frames=args.axes_max_frames,
        chunk=args.chunk,
        verbose=args.verbose,
    )

    # Directional half-extents along the principal axes
    half_extents, extent_stats, projections = compute_directional_extents(
        xtc_paths=xtc_paths,
        top_xtc_full=top_xtc_full,
        protein_heavy_idx_full=protein_heavy_idx_full,
        selected_atom_indices_local=selected_atoms,
        core_aidxs=core_aidxs,
        axes=axes,
        extent_stat=args.extent_stat,
        max_frames=args.extent_max_frames,
        chunk=args.chunk,
        save_projections=args.save_projections,
    )

    padded_half_extents = half_extents + args.padding
    full_lengths = 2.0 * padded_half_extents

    # Box vectors in Cartesian coordinates
    # vector i = full_length_i * principal_axis_i
    box_vectors = np.vstack([
        full_lengths[0] * axes[0],
        full_lengths[1] * axes[1],
        full_lengths[2] * axes[2],
    ])

    print("\n[BOX] Directional half-extents before padding (nm)")
    print(f"  axis1 : {half_extents[0]:.3f}")
    print(f"  axis2 : {half_extents[1]:.3f}")
    print(f"  axis3 : {half_extents[2]:.3f}")

    print(f"\n[BOX] Padding added to each half-extent: {args.padding:.3f} nm")

    print("\n[BOX] Directional half-extents after padding (nm)")
    print(f"  axis1 : {padded_half_extents[0]:.3f}")
    print(f"  axis2 : {padded_half_extents[1]:.3f}")
    print(f"  axis3 : {padded_half_extents[2]:.3f}")

    print("\n[BOX] Full box lengths in principal-axis frame (nm)")
    print(f"  Lx : {full_lengths[0]:.3f}")
    print(f"  Ly : {full_lengths[1]:.3f}")
    print(f"  Lz : {full_lengths[2]:.3f}")

    print("\n[BOX] Principal axes (unit vectors)")
    for i in range(3):
        print(
            f"  axis{i+1}: "
            f"[{axes[i,0]: .6f}, {axes[i,1]: .6f}, {axes[i,2]: .6f}]"
        )

    print("\n[BOX] Triclinic box vectors in Cartesian coordinates (nm)")
    for i in range(3):
        print(
            f"  v{i+1}: "
            f"[{box_vectors[i,0]: .6f}, {box_vectors[i,1]: .6f}, {box_vectors[i,2]: .6f}]"
        )

    summary = {
        "pdb": pdb_code,
        "pcs": pcs,
        "weighted": args.weighted,
        "com_percentile": args.com_percentile,
        "pca_percentile": args.pca_percentile,
        "extent_stat": args.extent_stat,
        "padding_nm": args.padding,
        "selected_residue_count": int(selected_residues.size),
        "selected_atom_count": int(selected_atoms.size),
        "com_cutoff": float(com_cut),
        "pca_cutoff": float(pca_cut),
        "covariance_eigenvalues": evals.tolist(),
        "mean_centered_coordinate": mean_vec.tolist(),
        "half_extents_nm": {
            "raw": half_extents.tolist(),
            "padded": padded_half_extents.tolist(),
        },
        "full_lengths_nm": full_lengths.tolist(),
        "axes": axes.tolist(),
        "box_vectors_nm": box_vectors.tolist(),
        "extent_stats_nm": {
            "p95": extent_stats["p95"].tolist(),
            "p99": extent_stats["p99"].tolist(),
            "max": extent_stats["max"].tolist(),
        },
    }

    np.save(outdir / "principal_axes.npy", axes)
    np.save(outdir / "box_vectors_nm.npy", box_vectors)
    np.save(outdir / "box_lengths_nm.npy", full_lengths)
    np.save(outdir / "half_extents_nm.npy", padded_half_extents)

    if projections is not None:
        np.save(outdir / "projections_nm.npy", projections)

    with open(outdir / "triclinic_box_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n[WRITE] Saved:")
    print(f"  {outdir / 'principal_axes.npy'}")
    print(f"  {outdir / 'box_vectors_nm.npy'}")
    print(f"  {outdir / 'box_lengths_nm.npy'}")
    print(f"  {outdir / 'half_extents_nm.npy'}")
    print(f"  {outdir / 'triclinic_box_summary.json'}")
    if projections is not None:
        print(f"  {outdir / 'projections_nm.npy'}")


if __name__ == "__main__":
    main()