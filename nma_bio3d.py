# nma_bio3d.py

import numpy as np
import mdtraj as md

from rpy2 import robjects as ro
from rpy2.robjects import numpy2ri
from rpy2.robjects import default_converter
from rpy2.robjects.conversion import localconverter

from io_utils import print_header


def traj_to_pdb_string(traj: md.Trajectory) -> str:
    """
    Convert an MDTraj Trajectory to a PDB string for Bio3D.

    Assumes:
      - traj contains ONLY the atoms you want to use for NMA
        (in your case: protein heavy atoms).
      - Coordinates are in nm (MDTraj default) and are converted to Å.

    No additional atom selection is done here: we trust that traj has
    already been sliced appropriately (e.g. by traj_utils.build_protein_heavy_views).
    """
    traj_sel = traj  # already sliced to protein-heavy

    # First (and only) frame coordinates in nm
    xyz = traj_sel.xyz[0]  # shape (n_atoms_sel, 3)

    lines = []
    serial = 1

    for atom, (x, y, z) in zip(traj_sel.topology.atoms, xyz):
        res = atom.residue
        resname = (res.name or "RES")[:3]
        atomname = (atom.name or "X")[:4]

        # MDTraj uses nm, PDB uses Å
        xA, yA, zA = x * 10.0, y * 10.0, z * 10.0
        resseq = res.index + 1

        line = (
            f"ATOM  {serial:5d} "
            f"{atomname:^4s}"
            f"{resname:>4s}"
            f" A{resseq:4d}    "
            f"{xA:8.3f}{yA:8.3f}{zA:8.3f}"
            f"  1.00  0.00           "
        )
        lines.append(line)
        serial += 1

    lines.append("END")
    pdb_text = "\n".join(lines) + "\n"
    print(f"[nma_bio3d] Generated PDB string with {serial-1} atoms from protein-heavy traj")
    return pdb_text


def setup_r_nma_function() -> None:
    """
    Define the run_aanma(pdb_text) function in the R global environment.

    This:
      - reads the PDB text into Bio3D,
      - selects protein atoms,
      - runs aanma.pdb with heavy-atom modes ('noh'),
      - returns the mode matrix U (3N x nmodes_total).
    """
    r_code = r"""
    suppressPackageStartupMessages({
      library(bio3d)
    })

    run_aanma <- function(pdb_text) {
      # pdb_text: single big string from Python containing PDB contents

      # 1) Write PDB text to a temporary file
      tf <- tempfile(fileext = ".pdb")
      writeLines(pdb_text, tf)

      # 2) Read with Bio3D
      pdb <- read.pdb(tf)

      # 3) Clean up temp file
      unlink(tf)

      # 4) Select protein atoms (traj already protein-heavy, this should be 1:1)
      sel <- atom.select(pdb, "protein")
      pdb_trimmed <- trim.pdb(pdb, sel)

      # 5) All-atom NMA, heavy-atom modes (noh)
      nma_aa <- aanma.pdb(pdb_trimmed, rtb = FALSE, outmodes = "noh")

      # U: columns are modes, rows are 3N coordinates
      return(nma_aa$U)
    }
    """
    ro.r(r_code)


def run_aanma_r_from_traj(
    traj_protein_heavy: md.Trajectory,
    n_modes_keep: int,
) -> np.ndarray:
    """
    Run Bio3D aanma.pdb NMA via R on a protein-heavy MDTraj trajectory.

    Parameters
    ----------
    traj_protein_heavy : md.Trajectory
        Trajectory containing ONLY protein heavy atoms, in the exact order
        used for PCA (from traj_utils.build_protein_heavy_views).
        Typically a single frame.

    n_modes_keep : int
        Number of INTERNAL modes to keep for comparison (after skipping
        the first 6 rigid-body modes).

    Returns
    -------
    modes : np.ndarray
        Array of shape (n_modes_keep, 3N), where N is the number of
        protein heavy atoms. Modes are normalised in Bio3D’s convention;
        you can apply additional normalisation in Python if desired.
    """
    print_header("Running R aanma.pdb NMA in Bio3D (from protein-heavy traj)...")

    # 1) Build PDB string from already-sliced protein-heavy trajectory
    pdb_text = traj_to_pdb_string(traj_protein_heavy)

    # 2) Ensure R-side function exists
    setup_r_nma_function()
    r_run_aanma = ro.globalenv["run_aanma"]

    # 3) Call R and convert result → NumPy
    with localconverter(default_converter + numpy2ri.converter):
        U_r = r_run_aanma(pdb_text)
        U = np.array(U_r, dtype=float)

    # U shape: (3N_atoms, n_modes_total)
    n_dof, n_modes_r = U.shape
    print(f"[nma_bio3d] Mode matrix U shape: (3N_atoms={n_dof}, n_modes={n_modes_r})")

    # 4) Skip first 6 rigid-body modes (3 translations + 3 rotations)
    start = 6
    end = min(start + n_modes_keep, n_modes_r)

    if end <= start:
        raise ValueError(
            f"[nma_bio3d] Requested {n_modes_keep} modes, but only {n_modes_r} "
            f"modes available in R (after 6 rigid-body)."
        )

    # U_selected: (3N, n_internal_modes) → transpose to (n_internal_modes, 3N)
    U_selected = U[:, start:end].T
    print(
        f"[nma_bio3d] Selected modes {start+1}..{end} "
        f"(total {end-start} modes for comparison)"
    )

    return U_selected
