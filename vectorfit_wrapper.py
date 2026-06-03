# vectorfit_wrapper.py

import argparse
from html import parser
from pathlib import Path

import numpy as np

from traj_utils import build_protein_heavy_views
from nma_bio3d_structure_modes import run_aanma_r_from_traj
from pc1_nma_projection import (
    project_pc_onto_nma_modes,
    normalize_vector,
    save_endpoint_structures,
)

"""
python vectorfit_wrapper.py \
  --pdb 7lak \
  --replica 5
"""

def get_pc_vector_from_eigenvectors(eigenvectors: np.ndarray, expected_dof: int, pc_index: int):
    """
    Extract one PC eigenvector from eigenvectors.npy.

    Handles both common layouts:
        (3N, n_pcs)
        (n_pcs, 3N)

    pc_index is zero-based:
        pc_index=0 means PC1
    """

    eigenvectors = np.asarray(eigenvectors, dtype=float)

    if eigenvectors.ndim != 2:
        raise ValueError(
            f"Expected eigenvectors.npy to be 2D, got shape {eigenvectors.shape}"
        )

    # Layout: columns are PCs, like Bio3D pca$au[, 1]
    if eigenvectors.shape[0] == expected_dof:
        if pc_index >= eigenvectors.shape[1]:
            raise ValueError(
                f"Requested PC index {pc_index}, but eigenvectors only has "
                f"{eigenvectors.shape[1]} PCs in columns."
            )
        pc = eigenvectors[:, pc_index]
        layout = "(3N, n_pcs), columns are PCs"

    # Layout: rows are PCs
    elif eigenvectors.shape[1] == expected_dof:
        if pc_index >= eigenvectors.shape[0]:
            raise ValueError(
                f"Requested PC index {pc_index}, but eigenvectors only has "
                f"{eigenvectors.shape[0]} PCs in rows."
            )
        pc = eigenvectors[pc_index, :]
        layout = "(n_pcs, 3N), rows are PCs"

    else:
        raise ValueError(
            "eigenvectors.npy does not match protein-heavy atom count:\n"
            f"  eigenvectors shape: {eigenvectors.shape}\n"
            f"  expected DOF:       {expected_dof}\n"
            "Expected either (3N, n_pcs) or (n_pcs, 3N)."
        )

    pc = normalize_vector(pc)

    return pc, layout


def make_endpoint_coordinates_with_amplitude(
    xyz0: np.ndarray,
    dr_dir: np.ndarray,
    amplitude: float,
):
    """
    Make endpoint coordinates from a normalized direction and scalar amplitude.

    xyz_plus  = xyz0 + amplitude * dr_dir
    xyz_minus = xyz0 - amplitude * dr_dir
    """

    xyz0 = np.asarray(xyz0, dtype=float)

    if xyz0.ndim != 2 or xyz0.shape[1] != 3:
        raise ValueError(f"Expected xyz0 shape (N, 3), got {xyz0.shape}")

    n_atoms = xyz0.shape[0]

    dr_dir = np.asarray(dr_dir, dtype=float).reshape(-1)

    if dr_dir.shape[0] != n_atoms * 3:
        raise ValueError(
            "dr_dir length does not match xyz0:\n"
            f"  dr_dir length: {dr_dir.shape[0]}\n"
            f"  expected:      {n_atoms * 3}"
        )

    dr_scaled = float(amplitude) * dr_dir
    dr_scaled_atoms = dr_scaled.reshape(n_atoms, 3)

    xyz_plus = xyz0 + dr_scaled_atoms
    xyz_minus = xyz0 - dr_scaled_atoms

    return xyz_minus, xyz_plus, dr_scaled_atoms


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run Bio3D NMA, load PCA eigenvectors/eigenvalues from "
            "results/{pdb}/rep{replica}_pca_eigs/, project PC onto selected "
            "NMA modes, and save endpoint structures."
        )
    )

    parser.add_argument("--pdb", required=True)
    parser.add_argument("--replica", type=int, required=True)

    parser.add_argument(
        "--base-dir",
        default="results",
        help="Base results directory."
    )

    parser.add_argument(
        "--pc-number",
        type=int,
        default=1,
        help="PC number to use. Use 1 for PC1, 2 for PC2, etc."
    )

    parser.add_argument(
        "--pca-dir",
        default=None,
        help="Default: results/{pdb}/rep{replica}_pca_eigs"
    )

    parser.add_argument(
        "--nma-dir",
        default=None,
        help="Default: results/{pdb}/nma"
    )

    parser.add_argument(
        "--nma-keep",
        type=int,
        default=1000,
        help="Number of internal modes to save after skipping first 6."
    )

    parser.add_argument(
        "--mode-start",
        type=int,
        default=6,
        help="Zero-based mode start in raw_modes_all.npy. Use 6 for Bio3D mode 7."
    )

    parser.add_argument(
        "--n-modes",
        type=int,
        default=1000,
        help="Number of NMA modes to use. Use 5 for Bio3D modes 7-11."
    )

    parser.add_argument(
        "--outdir",
        default=None,
        help="Default: results/{pdb}/replica_{replica}/pc{pc}_nma_projection"
    )

    parser.add_argument(
        "--skip-existing-nma",
        action="store_true",
        help="If raw_modes_all.npy already exists, do not rerun Bio3D NMA."
    )

    parser.add_argument(
        "--amplitude",
        type=float,
        default=None,
        help=(
            "Amplitude for endpoint structures in nm. "
            "If omitted, sqrt(PC eigenvalue) is used as a fallback."
        )
    )

    args = parser.parse_args()

    pdb_code = args.pdb.strip().lower()
    replica = args.replica
    if args.pc_number < 1:
        raise ValueError("--pc-number must be >= 1. Use 1 for PC1, 2 for PC2, etc.")

    pc_index = args.pc_number - 1

    base_dir = Path(args.base_dir)

    if args.pca_dir is None:
        pca_dir = base_dir / pdb_code / f"rep{replica}_pca_eigs"
    else:
        pca_dir = Path(args.pca_dir)

    if args.nma_dir is None:
        nma_dir = base_dir / pdb_code / "nma"
    else:
        nma_dir = Path(args.nma_dir)

    if args.outdir is None:
        outdir = (
            base_dir
            / pdb_code
            / f"replica_{replica}"
            / f"pc{pc_index + 1}_nma_projection"
        )
    else:
        outdir = Path(args.outdir)

    eigenvectors_path = pca_dir / "eigenvectors.npy"
    eigenvalues_path = pca_dir / "eigenvalues.npy"

    raw_modes_all_path = nma_dir / "raw_modes_all.npy"
    eigvals_all_path = nma_dir / "eigenvalues_all.npy"

    nma_dir.mkdir(parents=True, exist_ok=True)
    outdir.mkdir(parents=True, exist_ok=True)

    print("==========================================")
    print("PC projection onto NMA modes with NMA run")
    print("==========================================")
    print(f"PDB/system:        {pdb_code}")
    print(f"Replica:           {replica}")
    print(f"PC number:         {args.pc_number}")
    print(f"Internal index:    {pc_index}")
    print(f"Base dir:          {base_dir}")
    print(f"PCA dir:           {pca_dir}")
    print(f"Eigenvectors:      {eigenvectors_path}")
    print(f"Eigenvalues:       {eigenvalues_path}")
    print(f"NMA dir:           {nma_dir}")
    print(f"Output dir:        {outdir}")
    print(f"Mode start index:  {args.mode_start}")
    print(f"Number of modes:   {args.n_modes}")

    if not eigenvectors_path.exists():
        raise FileNotFoundError(f"Missing PCA eigenvectors file: {eigenvectors_path}")

    if not eigenvalues_path.exists():
        raise FileNotFoundError(f"Missing PCA eigenvalues file: {eigenvalues_path}")

    # ------------------------------------------------------------
    # 1. Build protein-heavy reference from your existing file logic
    # ------------------------------------------------------------
    print("")
    print("[Reference] Building protein-heavy reference from DB topology...")

    (
        traj_protein_heavy_ref,
        top_xtc_protein,
        protein_heavy_idx_local,
        protein_heavy_idx_full,
        top_xtc_full,
    ) = build_protein_heavy_views(pdb_code)

    xyz0 = traj_protein_heavy_ref.xyz[0]
    expected_dof = traj_protein_heavy_ref.n_atoms * 3

    print("")
    print("[Reference] Counts:")
    print(f"  protein-heavy atoms: {traj_protein_heavy_ref.n_atoms}")
    print(f"  expected DOF:        {expected_dof}")
    print(f"  xyz0 shape:          {xyz0.shape}")

    # ------------------------------------------------------------
    # 2. Run NMA
    # ------------------------------------------------------------
    if args.skip_existing_nma and raw_modes_all_path.exists():
        print("")
        print("[NMA] Skipping Bio3D NMA because raw_modes_all.npy exists.")
    else:
        print("")
        print("[NMA] Running Bio3D aanma.pdb...")

        run_aanma_r_from_traj(
            traj_protein_heavy=traj_protein_heavy_ref,
            n_modes_keep=args.nma_keep,
            save_raw_modes_dir=nma_dir,
            save_prefix=None,
        )

    if not raw_modes_all_path.exists():
        raise FileNotFoundError(
            f"NMA did not create expected file: {raw_modes_all_path}"
        )

    # ------------------------------------------------------------
    # 3. Load files
    # ------------------------------------------------------------
    print("")
    print("[Load] Loading PCA and NMA files...")

    eigenvectors = np.load(eigenvectors_path)
    eigenvalues = np.load(eigenvalues_path).reshape(-1)
    nma_modes = np.load(raw_modes_all_path)

    pc_vector, eigvec_layout = get_pc_vector_from_eigenvectors(
        eigenvectors=eigenvectors,
        expected_dof=expected_dof,
        pc_index=pc_index,
    )

    if pc_index >= eigenvalues.shape[0]:
        raise ValueError(
            f"Requested PC{pc_index + 1}, but eigenvalues.npy only has "
            f"{eigenvalues.shape[0]} values."
        )

    pc_eigenvalue = float(eigenvalues[pc_index])

    print("")
    print("[Load] Shapes:")
    print(f"  eigenvectors.npy: {eigenvectors.shape}")
    print(f"  eigenvector layout detected: {eigvec_layout}")
    print(f"  eigenvalues.npy:  {eigenvalues.shape}")
    print(f"  selected PC:      {pc_vector.shape}")
    print(f"  PC eigenvalue:    {pc_eigenvalue}")
    print(f"  NMA modes:        {nma_modes.shape}")

    if nma_modes.shape[0] != expected_dof:
        raise ValueError(
            "NMA mode matrix does not match protein-heavy reference:\n"
            f"  NMA rows:     {nma_modes.shape[0]}\n"
            f"  expected DOF: {expected_dof}"
        )

    # ------------------------------------------------------------
    # 4. Project PC onto selected NMA modes
    # ------------------------------------------------------------
    print("")
    print("[Projection] Projecting PC onto selected NMA modes...")

    result = project_pc_onto_nma_modes(
        pc1_vector=pc_vector,
        nma_modes=nma_modes,
        mode_start=args.mode_start,
        n_modes=args.n_modes,
    )

    # ------------------------------------------------------------
    # 5. Endpoint amplitude
    # ------------------------------------------------------------
    if args.amplitude is not None:
        amplitude = float(args.amplitude)
        amplitude_source = "--amplitude"
    else:
        # Fallback if you only have eigenvalues.npy, not pca$z scores.
        # This is NOT the same as half-range from pca$z[,1].
        # It is the RMS amplitude along that PC if eigenvalues are variances.
        amplitude = float(np.sqrt(max(pc_eigenvalue, 0.0)))
        amplitude_source = "sqrt(selected PCA eigenvalue)"

    print("")
    print("[Endpoints] Scaling reconstructed direction...")
    print(f"  amplitude:        {amplitude}")
    print(f"  amplitude source: {amplitude_source}")

    xyz_minus, xyz_plus, dr_scaled_atoms = make_endpoint_coordinates_with_amplitude(
        xyz0=xyz0,
        dr_dir=result["dr_dir"],
        amplitude=amplitude,
    )

    # ------------------------------------------------------------
    # 6. Save output
    # ------------------------------------------------------------
    print("")
    print("[Save] Saving output...")

    np.save(outdir / "pc_vector.npy", pc_vector)
    np.save(outdir / "pc_eigenvalue.npy", np.array([pc_eigenvalue]))
    np.save(outdir / "coef.npy", result["coef"])
    np.save(outdir / "captured.npy", np.array([result["captured"]]))
    np.save(outdir / "dr_dir.npy", result["dr_dir"])
    np.save(outdir / "dr_scaled_atoms.npy", dr_scaled_atoms)
    np.save(outdir / "xyz_minus.npy", xyz_minus)
    np.save(outdir / "xyz_plus.npy", xyz_plus)
    np.save(outdir / "bio3d_mode_numbers.npy", result["bio3d_mode_numbers"])
    np.save(outdir / "mode_indices_zero_based.npy", result["mode_indices_zero_based"])

    if eigvals_all_path.exists():
        np.save(outdir / "nma_eigenvalues_all.npy", np.load(eigvals_all_path))

    save_endpoint_structures(
        reference_traj=traj_protein_heavy_ref,
        xyz_minus=xyz_minus,
        xyz_plus=xyz_plus,
        outdir=outdir,
        prefix=f"rep{replica}_pc{pc_index + 1}_lowest{args.n_modes}nm",
    )

    summary_path = outdir / "summary.txt"

    with open(summary_path, "w") as f:
        f.write("PC projection onto NMA modes with Bio3D NMA\n")
        f.write("==========================================\n\n")
        f.write(f"PDB/system: {pdb_code}\n")
        f.write(f"Replica: {replica}\n")
        f.write(f"PC: PC{pc_index + 1}\n")
        f.write(f"PCA dir: {pca_dir}\n")
        f.write(f"Eigenvectors: {eigenvectors_path}\n")
        f.write(f"Eigenvalues: {eigenvalues_path}\n")
        f.write(f"Eigenvectors shape: {eigenvectors.shape}\n")
        f.write(f"Eigenvector layout: {eigvec_layout}\n")
        f.write(f"Selected PC eigenvalue: {pc_eigenvalue}\n")
        f.write(f"NMA dir: {nma_dir}\n")
        f.write(f"Raw NMA modes: {raw_modes_all_path}\n")
        f.write(f"NMA mode matrix shape: {nma_modes.shape}\n")
        f.write(f"Protein-heavy atoms: {traj_protein_heavy_ref.n_atoms}\n")
        f.write(f"Expected DOF: {expected_dof}\n")
        f.write(f"Mode start index zero-based: {args.mode_start}\n")
        f.write(f"Number of modes: {args.n_modes}\n")
        f.write(f"Bio3D mode numbers: {result['bio3d_mode_numbers']}\n")
        f.write(f"Mode indices zero-based: {result['mode_indices_zero_based']}\n")
        f.write(f"Coefficients: {result['coef']}\n")
        f.write(f"Captured: {result['captured']}\n")
        f.write(f"Amplitude: {amplitude}\n")
        f.write(f"Amplitude source: {amplitude_source}\n")

    print("")
    print("================================")
    print("Done")
    print("================================")
    print(f"PC:                 PC{pc_index + 1}")
    print(f"Bio3D modes:        {result['bio3d_mode_numbers']}")
    print(f"Coefficients:       {result['coef']}")
    print(f"Captured:           {result['captured']:.6f}")
    print(f"Amplitude:          {amplitude}")
    print(f"Amplitude source:   {amplitude_source}")
    print(f"Output:             {outdir}")
    print(f"Summary:            {summary_path}")


if __name__ == "__main__":
    main()