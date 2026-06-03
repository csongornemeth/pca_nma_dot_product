# run_pc_nm_fit.py

import argparse
from pathlib import Path

import numpy as np
import mdtraj as md

from nma_bio3d_structure_modes import run_aanma_r_from_traj
from pc1_nma_projection import (
    project_pc_onto_nma_modes,
    make_pc1_nma_endpoint_coordinates,
    save_endpoint_structures,
)

"""
python run_pc_nm_fit.py \
  --pdb 7lak \
  --replica 0 \
  --ref-pdb results/7lak/protein_heavy_ref.pdb \
  --pc1-vector results/7lak/rep0_pca_eigs/pc1_vector.npy \
  --pc1-scores results/7lak/rep0_pca_eigs/pc1_scores.npy
  """

def main():
    parser = argparse.ArgumentParser(
        description="Run Bio3D NMA, project PC1 onto selected NMA modes, and save endpoint structures."
    )

    parser.add_argument("--pdb", required=True)
    parser.add_argument("--replica", type=int, required=True)

    parser.add_argument(
        "--base-dir",
        default="results",
        help="Base results directory."
    )

    parser.add_argument(
        "--ref-pdb",
        required=True,
        help="Reference PDB with exactly the same atom selection/order as PCA."
    )

    parser.add_argument(
        "--pc1-vector",
        required=True,
        help="Path to PC1 eigenvector .npy file, shape (3N,)."
    )

    parser.add_argument(
        "--pc1-scores",
        required=True,
        help="Path to PC1 scores/projections .npy file, equivalent to pca$z[,1]."
    )

    parser.add_argument(
        "--nma-dir",
        default=None,
        help="Directory to save NMA files. Default: results/{pdb}/nma"
    )

    parser.add_argument(
        "--nma-keep",
        type=int,
        default=20,
        help="Number of internal non-trivial NMA modes to save after skipping first 6."
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
        default=5,
        help="Number of NMA modes to project onto. Use 5 for Bio3D modes 7-11."
    )

    parser.add_argument(
        "--outdir",
        default=None,
        help="Output directory for projection results."
    )

    parser.add_argument(
        "--skip-existing-nma",
        action="store_true",
        help="If raw_modes_all.npy already exists, do not rerun Bio3D NMA."
    )

    args = parser.parse_args()

    pdb_code = args.pdb
    replica = args.replica
    base_dir = Path(args.base_dir)

    if args.nma_dir is None:
        nma_dir = base_dir / pdb_code / "nma"
    else:
        nma_dir = Path(args.nma_dir)

    if args.outdir is None:
        outdir = base_dir / pdb_code / f"replica_{replica}" / "pc1_nma_projection"
    else:
        outdir = Path(args.outdir)

    nma_dir.mkdir(parents=True, exist_ok=True)
    outdir.mkdir(parents=True, exist_ok=True)

    raw_modes_all_path = nma_dir / "raw_modes_all.npy"
    eigvals_all_path = nma_dir / "eigenvalues_all.npy"

    print("==========================================")
    print("PC1 projection onto NMA modes with NMA run")
    print("==========================================")
    print(f"PDB/system:       {pdb_code}")
    print(f"Replica:          {replica}")
    print(f"Reference PDB:    {args.ref_pdb}")
    print(f"PC1 vector:       {args.pc1_vector}")
    print(f"PC1 scores:       {args.pc1_scores}")
    print(f"NMA directory:    {nma_dir}")
    print(f"Projection outdir:{outdir}")
    print(f"Mode start index: {args.mode_start}")
    print(f"Number of modes:  {args.n_modes}")

    print("")
    print("Loading reference structure...")
    ref = md.load(args.ref_pdb)

    if ref.n_frames != 1:
        print(f"[WARNING] Reference has {ref.n_frames} frames. Using frame 0 only.")
        ref = ref[0]

    print(f"Reference atoms: {ref.n_atoms}")
    print(f"Reference xyz shape: {ref.xyz.shape}")

    # ------------------------------------------------------------
    # 1. Run Bio3D NMA unless existing output should be reused
    # ------------------------------------------------------------
    if args.skip_existing_nma and raw_modes_all_path.exists():
        print("")
        print("[NMA] Existing raw_modes_all.npy found.")
        print("[NMA] Skipping Bio3D NMA because --skip-existing-nma was used.")
    else:
        print("")
        print("[NMA] Running Bio3D aanma.pdb through R...")

        run_aanma_r_from_traj(
            traj_protein_heavy=ref,
            n_modes_keep=args.nma_keep,
            save_raw_modes_dir=nma_dir,
            save_prefix=None,
        )

    if not raw_modes_all_path.exists():
        raise FileNotFoundError(
            f"Expected NMA mode file was not created/found:\n{raw_modes_all_path}"
        )

    # ------------------------------------------------------------
    # 2. Load NMA + PCA data
    # ------------------------------------------------------------
    print("")
    print("Loading NMA and PCA data...")

    nma_modes = np.load(raw_modes_all_path)
    pc1_vector = np.load(args.pc1_vector)
    pc1_scores = np.load(args.pc1_scores)

    xyz0 = ref.xyz[0]

    print("")
    print("Shapes:")
    print(f"  raw NMA modes: {nma_modes.shape}")
    print(f"  PC1 vector:    {pc1_vector.shape}")
    print(f"  PC1 scores:    {pc1_scores.shape}")
    print(f"  xyz0:          {xyz0.shape}")

    expected_dof = ref.n_atoms * 3

    if nma_modes.shape[0] != expected_dof:
        raise ValueError(
            "NMA mode matrix does not match reference atom count:\n"
            f"  reference atoms: {ref.n_atoms}\n"
            f"  expected DOF:    {expected_dof}\n"
            f"  NMA rows:        {nma_modes.shape[0]}"
        )

    if pc1_vector.reshape(-1).shape[0] != expected_dof:
        raise ValueError(
            "PC1 vector does not match reference atom count:\n"
            f"  reference atoms: {ref.n_atoms}\n"
            f"  expected DOF:    {expected_dof}\n"
            f"  PC1 length:      {pc1_vector.reshape(-1).shape[0]}"
        )

    # ------------------------------------------------------------
    # 3. Project PC1 onto selected NMA modes
    # ------------------------------------------------------------
    print("")
    print("[Projection] Projecting PC1 onto selected NMA modes...")

    result = project_pc_onto_nma_modes(
        pc1_vector=pc1_vector,
        nma_modes=nma_modes,
        mode_start=args.mode_start,
        n_modes=args.n_modes,
    )

    # ------------------------------------------------------------
    # 4. Make endpoint structures
    # ------------------------------------------------------------
    print("")
    print("[Endpoints] Building plus/minus endpoint structures...")

    xyz_minus, xyz_plus, A_pc, dr_scaled_atoms = make_pc1_nma_endpoint_coordinates(
        xyz0=xyz0,
        dr_dir=result["dr_dir"],
        pc1_scores=pc1_scores,
    )

    # ------------------------------------------------------------
    # 5. Save arrays and PDBs
    # ------------------------------------------------------------
    np.save(outdir / "coef.npy", result["coef"])
    np.save(outdir / "dr_dir.npy", result["dr_dir"])
    np.save(outdir / "dr_scaled_atoms.npy", dr_scaled_atoms)
    np.save(outdir / "xyz_minus.npy", xyz_minus)
    np.save(outdir / "xyz_plus.npy", xyz_plus)

    if eigvals_all_path.exists():
        eigvals = np.load(eigvals_all_path)
        np.save(outdir / "eigenvalues_all.npy", eigvals)

    summary_path = outdir / "summary.txt"

    with open(summary_path, "w") as f:
        f.write("PC1 projection onto NMA modes with Bio3D NMA\n")
        f.write("==========================================\n\n")
        f.write(f"PDB/system: {pdb_code}\n")
        f.write(f"Replica: {replica}\n")
        f.write(f"Reference PDB: {args.ref_pdb}\n")
        f.write(f"NMA directory: {nma_dir}\n")
        f.write(f"Raw NMA modes: {raw_modes_all_path}\n")
        f.write(f"PC1 vector: {args.pc1_vector}\n")
        f.write(f"PC1 scores: {args.pc1_scores}\n")
        f.write(f"Reference atoms: {ref.n_atoms}\n")
        f.write(f"NMA mode matrix shape: {nma_modes.shape}\n")
        f.write(f"PC1 vector shape: {pc1_vector.shape}\n")
        f.write(f"PC1 scores shape: {pc1_scores.shape}\n")
        f.write(f"Bio3D mode numbers: {result['bio3d_mode_numbers']}\n")
        f.write(f"Coefficients: {result['coef']}\n")
        f.write(f"Captured: {result['captured']}\n")
        f.write(f"A_pc: {A_pc}\n")

    save_endpoint_structures(
        reference_traj=ref,
        xyz_minus=xyz_minus,
        xyz_plus=xyz_plus,
        outdir=outdir,
        prefix="pc1_lowest5nm",
    )

    print("")
    print("================================")
    print("Done")
    print("================================")
    print(f"Bio3D mode numbers: {result['bio3d_mode_numbers']}")
    print(f"Coefficients:       {result['coef']}")
    print(f"Captured:           {result['captured']:.6f}")
    print(f"A_pc:               {A_pc:.6f}")
    print("")
    print(f"NMA files:          {nma_dir}")
    print(f"Projection output:  {outdir}")
    print(f"Summary:            {summary_path}")


if __name__ == "__main__":
    main()