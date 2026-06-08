#bob_the_boxer.py
#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mdtraj as md
import numpy as np

"""
python bob_the_boxer.py \
  --target target_7lak.pdb \
  --helpers nma_helper_7lak/*.pdb \
  --reference test.pdb \
  --align-selection "backbone" \
  --fit-selection "protein and not element H" \
  --center-selection "protein" \
  --padding 0.7 \
  --outdir common_box_7lak_test \
  --save-helpers
"""

def load_single_frame(path: Path) -> md.Trajectory:
    traj = md.load(path.as_posix())
    if traj.n_frames != 1:
        traj = traj[0]
    return traj


def check_same_topology(reference: md.Trajectory, trajs: list[md.Trajectory]) -> None:
    n_atoms = reference.n_atoms
    for i, tr in enumerate(trajs):
        if tr.n_atoms != n_atoms:
            raise ValueError(
                f"Structure {i} has {tr.n_atoms} atoms, but reference has {n_atoms}. "
                "This first version assumes identical atom order/topology."
            )


def select_atoms(traj: md.Trajectory, selection: str) -> np.ndarray:
    idx = traj.topology.select(selection)
    if idx.size == 0:
        raise ValueError(f"Selection returned zero atoms: {selection}")
    return idx.astype(int)


def atom_masses(topology: md.Topology, atom_indices: np.ndarray) -> np.ndarray:
    masses = []
    for idx in atom_indices:
        atom = topology.atom(int(idx))
        if atom.element is None or atom.element.mass is None:
            masses.append(12.0)
        else:
            masses.append(float(atom.element.mass))
    return np.array(masses, dtype=float)


def compute_com(xyz: np.ndarray, atom_indices: np.ndarray, masses: np.ndarray) -> np.ndarray:
    coords = xyz[atom_indices]
    return np.sum(coords * masses[:, None], axis=0) / np.sum(masses)


def align_to_reference(
    target: md.Trajectory,
    helpers: list[md.Trajectory],
    reference: md.Trajectory,
    align_indices: np.ndarray,
) -> tuple[md.Trajectory, list[md.Trajectory]]:
    """
    Align target and helpers to reference using align_indices.
    All trajectories are copied before modification.
    """
    target_aligned = target[:]
    target_aligned.superpose(reference, frame=0, atom_indices=align_indices)

    helpers_aligned = []
    for helper in helpers:
        h = helper[:]
        h.superpose(reference, frame=0, atom_indices=align_indices)
        helpers_aligned.append(h)

    return target_aligned, helpers_aligned


def centre_by_whole_protein_com(
    trajs: list[md.Trajectory],
    protein_indices: np.ndarray,
) -> list[md.Trajectory]:
    """
    Centre each structure by its own whole-protein COM.
    """
    masses = atom_masses(trajs[0].topology, protein_indices)
    centred = []

    for tr in trajs:
        out = tr[:]
        com = compute_com(out.xyz[0], protein_indices, masses)
        out.xyz[0] -= com
        centred.append(out)

    return centred


def pca_axes(points: np.ndarray) -> np.ndarray:
    """
    Return axes as rows: axes[0], axes[1], axes[2].
    """
    center = points.mean(axis=0)
    X = points - center

    cov = X.T @ X / X.shape[0]
    eigvals, eigvecs = np.linalg.eigh(cov)

    order = np.argsort(eigvals)[::-1]
    axes = eigvecs[:, order].T

    # Ensure right-handed coordinate system
    if np.linalg.det(axes) < 0:
        axes[2] *= -1.0

    return axes


def oriented_box_from_points(
    points: np.ndarray,
    axes: np.ndarray,
    padding: float,
) -> dict:
    """
    Build an oriented box around points.

    axes are rows.
    Projection is points @ axes.T.
    """
    projected = points @ axes.T

    mins = projected.min(axis=0)
    maxs = projected.max(axis=0)

    extents = maxs - mins
    lengths = extents + 2.0 * padding

    box_center_projected = 0.5 * (mins + maxs)

    return {
        "axes": axes,
        "mins_projected": mins,
        "maxs_projected": maxs,
        "extents": extents,
        "lengths": lengths,
        "box_center_projected": box_center_projected,
    }


def transform_to_box_frame(
    traj: md.Trajectory,
    axes: np.ndarray,
    box_center_projected: np.ndarray,
    lengths: np.ndarray,
) -> md.Trajectory:
    """
    Transform coordinates into the oriented box frame.

    This makes the oriented box become the simulation box basis.
    The target is shifted so the fitted envelope centre is at the box centre.
    """
    out = traj[:]

    projected = out.xyz[0] @ axes.T

    # Move fitted envelope centre to origin, then shift to middle of box
    projected = projected - box_center_projected[None, :]
    projected = projected + 0.5 * lengths[None, :]

    out.xyz[0] = projected

    return out


def make_restricted_triclinic_vectors(lengths: np.ndarray) -> np.ndarray:
    """
    First version: after transforming coordinates into the box frame,
    the GROMACS-compatible box is rectangular.

    This is still the practical representation of the oriented ensemble box:
    the molecule was transformed into the triclinic/oriented box frame.

    Returns a 3x3 box matrix with vectors as rows:
        v1 = [Lx, 0,  0]
        v2 = [0,  Ly, 0]
        v3 = [0,  0,  Lz]
    """
    Lx, Ly, Lz = lengths
    return np.array(
        [
            [Lx, 0.0, 0.0],
            [0.0, Ly, 0.0],
            [0.0, 0.0, Lz],
        ],
        dtype=float,
    )


def set_unitcell_vectors(traj: md.Trajectory, box_vectors: np.ndarray) -> md.Trajectory:
    out = traj[:]
    out.unitcell_vectors = box_vectors[None, :, :]
    return out


def save_gro(traj: md.Trajectory, path: Path) -> None:
    traj.save_gro(path.as_posix())


def save_pdb(traj: md.Trajectory, path: Path) -> None:
    traj.save_pdb(path.as_posix())


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Fit a common box from a target structure plus NMA helper structures, "
            "then write only the target structure with the common box."
        )
    )

    parser.add_argument("--target", required=True, type=Path)
    parser.add_argument("--helpers", required=True, nargs="+", type=Path)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--align-selection", default="backbone")
    parser.add_argument("--fit-selection", default="protein and not element H")
    parser.add_argument("--center-selection", default="protein")
    parser.add_argument("--padding", type=float, default=1.2)
    parser.add_argument("--outdir", required=True, type=Path)
    parser.add_argument("--save-helpers", action="store_true")

    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)

    target = load_single_frame(args.target)
    helpers = [load_single_frame(p) for p in args.helpers]
    reference = load_single_frame(args.reference)

    check_same_topology(reference, [target] + helpers)

    align_indices = select_atoms(reference, args.align_selection)
    fit_indices = select_atoms(reference, args.fit_selection)
    center_indices = select_atoms(reference, args.center_selection)

    print(f"[info] target: {args.target}")
    print(f"[info] n_helpers: {len(helpers)}")
    print(f"[info] align selection: {args.align_selection}")
    print(f"[info] align atoms: {align_indices.size}")
    print(f"[info] fit selection: {args.fit_selection}")
    print(f"[info] fit atoms: {fit_indices.size}")
    print(f"[info] center selection: {args.center_selection}")
    print(f"[info] center atoms: {center_indices.size}")
    print(f"[info] padding: {args.padding:.3f} nm")

    target_aligned, helpers_aligned = align_to_reference(
        target=target,
        helpers=helpers,
        reference=reference,
        align_indices=align_indices,
    )

    all_aligned = [target_aligned] + helpers_aligned

    all_centred = centre_by_whole_protein_com(
        trajs=all_aligned,
        protein_indices=center_indices,
    )

    target_centred = all_centred[0]
    helpers_centred = all_centred[1:]

    # Build union point cloud from target + helpers
    point_cloud = np.concatenate(
        [tr.xyz[0, fit_indices, :] for tr in all_centred],
        axis=0,
    )

    axes = pca_axes(point_cloud)

    box_info = oriented_box_from_points(
        points=point_cloud,
        axes=axes,
        padding=args.padding,
    )

    lengths = box_info["lengths"]
    box_center_projected = box_info["box_center_projected"]

    print("[box] raw extents nm:", box_info["extents"])
    print("[box] final lengths nm:", lengths)
    print("[box] axes rows:")
    print(axes)

    # Transform only target into the fitted box frame
    target_box_frame = transform_to_box_frame(
        traj=target_centred,
        axes=axes,
        box_center_projected=box_center_projected,
        lengths=lengths,
    )

    box_vectors = make_restricted_triclinic_vectors(lengths)
    target_boxed = set_unitcell_vectors(target_box_frame, box_vectors)

    out_gro = args.outdir / "target_common_box.gro"
    out_pdb = args.outdir / "target_common_box.pdb"

    save_gro(target_boxed, out_gro)
    save_pdb(target_boxed, out_pdb)

    np.save(args.outdir / "box_axes_rows.npy", axes)
    np.save(args.outdir / "box_lengths_nm.npy", lengths)
    np.save(args.outdir / "box_vectors_nm.npy", box_vectors)
    np.save(args.outdir / "combined_fit_points_nm.npy", point_cloud)

    summary = {
        "target": str(args.target),
        "helpers": [str(p) for p in args.helpers],
        "reference": str(args.reference),
        "n_helpers": len(helpers),
        "align_selection": args.align_selection,
        "fit_selection": args.fit_selection,
        "center_selection": args.center_selection,
        "padding_nm": args.padding,
        "raw_extents_nm": box_info["extents"].tolist(),
        "box_lengths_nm": lengths.tolist(),
        "axes_rows": axes.tolist(),
        "box_center_projected_nm": box_center_projected.tolist(),
        "box_vectors_nm": box_vectors.tolist(),
        "output_gro": str(out_gro),
        "output_pdb": str(out_pdb),
        "note": (
            "Coordinates were transformed into the fitted box frame. "
            "The final .gro contains only the target structure with the common box."
        ),
    }

    with open(args.outdir / "common_box_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    if args.save_helpers:
        helper_dir = args.outdir / "helpers_box_frame"
        helper_dir.mkdir(exist_ok=True)

        for i, h in enumerate(helpers_centred):
            h_box_frame = transform_to_box_frame(
                traj=h,
                axes=axes,
                box_center_projected=box_center_projected,
                lengths=lengths,
            )
            h_boxed = set_unitcell_vectors(h_box_frame, box_vectors)
            h_boxed.save_pdb((helper_dir / f"helper_{i:03d}_box_frame.pdb").as_posix())

    print(f"[done] wrote: {out_gro}")
    print(f"[done] wrote: {out_pdb}")
    print(f"[done] summary: {args.outdir / 'common_box_summary.json'}")


if __name__ == "__main__":
    main()