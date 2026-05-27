#!/usr/bin/env python3

import argparse
from pathlib import Path

import mdtraj as md
import numpy as np

from io_utils import print_header
from traj_utils import build_protein_heavy_views
from nma_bio3d import run_aanma_r_from_traj

"""
python run_combined_nm_visualisation.py \
  --pdb 7lak \
  --n-modes-keep 20 \
  --combine-first 10 \
  --scheme equal \
  --amplitude 5.0 \
  --n-frames 60 \
  --save-individual-modes \
  --individual-first 10 \
  --outdir combined_nm_visualisation_results
  """

def normalise_vector(v: np.ndarray) -> np.ndarray:
    """Return unit-normalised vector."""
    v = np.asarray(v, dtype=float)
    norm = np.linalg.norm(v)

    if norm == 0:
        raise ValueError("Cannot normalise a zero vector.")

    return v / norm


def combine_modes_with_weights(
    modes: np.ndarray,
    eigvals: np.ndarray,
    n_modes: int,
    scheme: str = "equal",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Combine selected NMA modes into one vector.

    Parameters
    ----------
    modes
        Shape: (n_modes_available, 3N)
        These are already internal Bio3D modes, after skipping the first 6.
        modes[0] corresponds to Bio3D mode 7.
    eigvals
        Shape: (n_modes_available,)
        eigvals[i] corresponds to modes[i].
    n_modes
        Number of modes to combine from the start of the selected internal modes.
        Example: n_modes=10 combines Bio3D modes 7..16.
    scheme
        Weighting scheme:
        - "equal": all selected modes get weight 1
        - "eigenvalue": weight by eigenvalue
        - "inverse_eigenvalue": weight by 1/eigenvalue
        - "inverse_sqrt_eigenvalue": weight by 1/sqrt(eigenvalue)

    Returns
    -------
    combined
        Shape: (3N,)
    weights
        Shape: (n_modes,)
    """

    modes = np.asarray(modes, dtype=float)
    eigvals = np.asarray(eigvals, dtype=float)

    if modes.ndim != 2:
        raise ValueError(f"Expected modes to be 2D, got {modes.shape}")

    if eigvals.ndim != 1:
        raise ValueError(f"Expected eigvals to be 1D, got {eigvals.shape}")

    if modes.shape[0] != eigvals.shape[0]:
        raise ValueError(
            f"Mode/eigenvalue mismatch: {modes.shape[0]} modes, "
            f"{eigvals.shape[0]} eigenvalues"
        )

    if n_modes < 1:
        raise ValueError("n_modes must be >= 1")

    if n_modes > modes.shape[0]:
        raise ValueError(
            f"Requested n_modes={n_modes}, but only {modes.shape[0]} are available"
        )

    selected_modes = modes[:n_modes]
    selected_eigvals = eigvals[:n_modes]

    if scheme == "equal":
        weights = np.ones(n_modes, dtype=float)

    elif scheme == "eigenvalue":
        weights = selected_eigvals.copy()

    elif scheme == "inverse_eigenvalue":
        if np.any(selected_eigvals <= 0):
            raise ValueError(
                "inverse_eigenvalue requires positive eigenvalues. "
                f"Got: {selected_eigvals}"
            )
        weights = 1.0 / selected_eigvals

    elif scheme == "inverse_sqrt_eigenvalue":
        if np.any(selected_eigvals <= 0):
            raise ValueError(
                "inverse_sqrt_eigenvalue requires positive eigenvalues. "
                f"Got: {selected_eigvals}"
            )
        weights = 1.0 / np.sqrt(selected_eigvals)

    else:
        raise ValueError(
            f"Unknown scheme '{scheme}'. Use one of: "
            "equal, eigenvalue, inverse_eigenvalue, inverse_sqrt_eigenvalue"
        )

    # Normalise weights so only the relative contribution matters
    weights = weights / np.sum(np.abs(weights))

    # Same linear combination as your R script:
    # combined = sum_i weight_i * mode_i
    combined = np.einsum("k,kj->j", weights, selected_modes)

    # Normalise final direction for clean visualisation scaling
    combined = normalise_vector(combined)

    return combined, weights


def make_mode_morph_traj(
    ref_traj: md.Trajectory,
    combined_mode_flat: np.ndarray,
    amplitude: float = 2.0,
    n_frames: int = 60,
) -> md.Trajectory:
    """
    Build an oscillating trajectory along the combined mode.

    ref_traj
        MDTraj trajectory with one frame and N atoms.
    combined_mode_flat
        Shape: (3N,)
    amplitude
        Visual exaggeration factor in nm after mode normalisation.
    n_frames
        Number of frames in output morph.
    """

    if ref_traj.n_frames < 1:
        raise ValueError("Reference trajectory has no frames.")

    coords = ref_traj.xyz[0]
    n_atoms = ref_traj.n_atoms

    combined_mode_flat = np.asarray(combined_mode_flat, dtype=float)

    expected = 3 * n_atoms
    if combined_mode_flat.shape != (expected,):
        raise ValueError(
            f"Expected combined mode shape ({expected},), "
            f"got {combined_mode_flat.shape}"
        )

    combined_mode = combined_mode_flat.reshape(n_atoms, 3)
    combined_mode = normalise_vector(combined_mode.reshape(-1)).reshape(n_atoms, 3)

    phases = np.linspace(0.0, 2.0 * np.pi, n_frames, endpoint=False)

    frames = []
    for phase in phases:
        displaced = coords + amplitude * np.sin(phase) * combined_mode
        frames.append(displaced)

    xyz = np.stack(frames, axis=0)

    return md.Trajectory(
        xyz=xyz,
        topology=ref_traj.topology,
    )

def save_individual_mode_morphs(
    ref_traj: md.Trajectory,
    modes: np.ndarray,
    eigvals: np.ndarray,
    n_modes: int,
    outdir: Path,
    pdb_code: str,
    amplitude: float = 2.0,
    n_frames: int = 60,
    mode_number_offset: int = 7,
) -> None:
    """
    Save one oscillating morph trajectory per individual normal mode.

    modes
        Shape: (n_modes_available, 3N).
        modes[0] corresponds to Bio3D mode 7 if mode_number_offset=7.
    """

    individual_dir = outdir / "individual_modes"
    individual_dir.mkdir(parents=True, exist_ok=True)

    n_modes = min(n_modes, modes.shape[0])

    for i in range(n_modes):
        bio3d_mode = mode_number_offset + i

        mode_vec = normalise_vector(modes[i])

        morph = make_mode_morph_traj(
            ref_traj=ref_traj,
            combined_mode_flat=mode_vec,
            amplitude=amplitude,
            n_frames=n_frames,
        )

        ref_pdb = individual_dir / f"{pdb_code}_mode_{bio3d_mode:02d}_ref.pdb"
        xtc = individual_dir / f"{pdb_code}_mode_{bio3d_mode:02d}_morph.xtc"
        extreme_pdb = individual_dir / f"{pdb_code}_mode_{bio3d_mode:02d}_extreme.pdb"

        morph[0].save_pdb(ref_pdb.as_posix())
        morph.save_xtc(xtc.as_posix())

        write_arrow_pdb(
            ref_traj=ref_traj,
            combined_mode_flat=mode_vec,
            out_pdb=extreme_pdb,
            amplitude=amplitude,
        )

        print(
            f"[individual] Bio3D mode {bio3d_mode:2d} "
            f"eigval={eigvals[i]:.10e} -> {xtc}"
        )

def write_arrow_pdb(
    ref_traj: md.Trajectory,
    combined_mode_flat: np.ndarray,
    out_pdb: Path,
    amplitude: float = 2.0,
) -> None:
    """
    Write a single extreme displaced PDB.

    B-factor column stores per-atom displacement magnitude.
    """

    coords = ref_traj.xyz[0]
    n_atoms = ref_traj.n_atoms

    mode = combined_mode_flat.reshape(n_atoms, 3)
    mode = normalise_vector(mode.reshape(-1)).reshape(n_atoms, 3)

    displaced = coords + amplitude * mode

    out = md.Trajectory(
        xyz=displaced[None, :, :],
        topology=ref_traj.topology,
    )

    out.save_pdb(out_pdb.as_posix())


def save_weights_table(
    weights: np.ndarray,
    eigvals: np.ndarray,
    out_csv: Path,
    mode_number_offset: int = 7,
) -> None:
    """
    Save mode weights and eigenvalues.

    mode_number_offset=7 means array index 0 is reported as Bio3D mode 7.
    """

    rows = ["array_index,bio3d_mode,eigenvalue,weight"]

    for i, w in enumerate(weights):
        bio3d_mode = mode_number_offset + i
        rows.append(f"{i},{bio3d_mode},{eigvals[i]:.12e},{w:.12e}")

    out_csv.write_text("\n".join(rows) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run Bio3D NMA from the usual input structure, combine normal modes, "
            "and write a morph trajectory for visualisation."
        )
    )

    parser.add_argument(
        "--pdb",
        required=True,
        help="PDB/system code, e.g. 8bdp",
    )

    parser.add_argument(
        "--n-modes-keep",
        type=int,
        default=20,
        help="Number of internal Bio3D modes to keep after skipping first 6.",
    )

    parser.add_argument(
        "--combine-first",
        type=int,
        default=10,
        help=(
            "Number of kept internal modes to combine. "
            "Example: 10 combines Bio3D modes 7..16."
        ),
    )

    parser.add_argument(
        "--scheme",
        default="equal",
        choices=[
            "equal",
            "eigenvalue",
            "inverse_eigenvalue",
            "inverse_sqrt_eigenvalue",
        ],
        help="Mode weighting scheme.",
    )

    parser.add_argument(
        "--amplitude",
        type=float,
        default=2.0,
        help="Visual exaggeration amplitude in nm.",
    )

    parser.add_argument(
        "--n-frames",
        type=int,
        default=60,
        help="Number of frames in morph trajectory.",
    )

    parser.add_argument(
        "--outdir",
        default="combined_nm_visualisation_results",
        help="Output directory.",
    )

    parser.add_argument(
        "--save-individual-modes",
        action="store_true",
        help="Also save one morph trajectory per individual selected mode.",
    )

    parser.add_argument(
        "--individual-first",
        type=int,
        default=None,
        help=(
            "Number of individual modes to save. "
            "Default: same as --combine-first."
        ),
    )

    args = parser.parse_args()

    pdb_code = args.pdb.lower()

    outdir = Path(args.outdir) / pdb_code
    outdir.mkdir(parents=True, exist_ok=True)

    print_header("Building protein-heavy input structure")

    (
        traj_protein_heavy_ref,
        top_xtc_protein,
        protein_heavy_idx_local,
        protein_heavy_idx_full,
        top_xtc_full,
    ) = build_protein_heavy_views(pdb_code)

    print_header("Running NMA")

    modes, eigvals = run_aanma_r_from_traj(
        traj_protein_heavy=traj_protein_heavy_ref,
        n_modes_keep=args.n_modes_keep,
    )

    if args.save_individual_modes:
        print_header("Writing individual mode visualisations")

        n_individual = (
            args.individual_first
            if args.individual_first is not None
            else args.combine_first
        )

        save_individual_mode_morphs(
            ref_traj=traj_protein_heavy_ref,
            modes=modes,
            eigvals=eigvals,
            n_modes=n_individual,
            outdir=outdir,
            pdb_code=pdb_code,
            amplitude=args.amplitude,
            n_frames=args.n_frames,
            mode_number_offset=7,
        )

    print_header("Combining modes")

    combined, weights = combine_modes_with_weights(
        modes=modes,
        eigvals=eigvals,
        n_modes=args.combine_first,
        scheme=args.scheme,
    )

    print(f"[combine] Scheme: {args.scheme}")
    print(f"[combine] Combined first {args.combine_first} internal modes")
    print(
        "[combine] Bio3D modes combined: "
        f"7..{7 + args.combine_first - 1}"
    )

    for i, w in enumerate(weights):
        print(
            f"  array_index={i:2d} "
            f"bio3d_mode={7 + i:2d} "
            f"eigval={eigvals[i]:.10e} "
            f"weight={w:.10e}"
        )

    npy_out = outdir / f"{pdb_code}_combined_mode.npy"
    weights_out = outdir / f"{pdb_code}_combined_mode_weights.csv"
    morph_pdb = outdir / f"{pdb_code}_combined_mode_ref.pdb"
    morph_xtc = outdir / f"{pdb_code}_combined_mode_morph.xtc"
    extreme_pdb = outdir / f"{pdb_code}_combined_mode_extreme.pdb"

    np.save(npy_out, combined)
    save_weights_table(
        weights=weights,
        eigvals=eigvals,
        out_csv=weights_out,
        mode_number_offset=7,
    )

    print_header("Writing visualisation trajectory")

    morph = make_mode_morph_traj(
        ref_traj=traj_protein_heavy_ref,
        combined_mode_flat=combined,
        amplitude=args.amplitude,
        n_frames=args.n_frames,
    )

    # Save first frame reference and oscillating trajectory
    morph[0].save_pdb(morph_pdb.as_posix())
    morph.save_xtc(morph_xtc.as_posix())

    # Also save a single positive-extreme structure
    write_arrow_pdb(
        ref_traj=traj_protein_heavy_ref,
        combined_mode_flat=combined,
        out_pdb=extreme_pdb,
        amplitude=args.amplitude,
    )

    print_header("Done")

    print(f"Combined mode vector: {npy_out}")
    print(f"Weights table:        {weights_out}")
    print(f"Morph reference PDB:  {morph_pdb}")
    print(f"Morph trajectory XTC: {morph_xtc}")
    print(f"Extreme PDB:          {extreme_pdb}")

    print("\nOpen for visualisation with:")
    print(f"vmd {morph_pdb} {morph_xtc}")


if __name__ == "__main__":
    main()