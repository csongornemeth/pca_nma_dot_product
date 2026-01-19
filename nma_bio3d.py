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

    Returns a list with:
      - U      : mode matrix (3N x nmodes_total)
      - values : eigenvalues (length nmodes_total)
    """
    r_code = r"""
    suppressPackageStartupMessages({
      library(bio3d)
    })

    run_aanma <- function(pdb_text) {
      tf <- tempfile(fileext = ".pdb")
      writeLines(pdb_text, tf)

      pdb <- read.pdb(tf)
      unlink(tf)

      sel <- atom.select(pdb, "protein")
      pdb_trimmed <- trim.pdb(pdb, sel)

      return(list(
        U = nma_aa$U,
        values = nma_aa$L
      ))
    }
    """
    ro.r(r_code)

def run_aanma_r_from_traj(
    traj_protein_heavy: md.Trajectory,
    n_modes_keep: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run Bio3D aanma.pdb NMA via R on a protein-heavy MDTraj trajectory.

    Returns
    -------
    modes : np.ndarray
        Shape (n_modes_keep, 3N). INTERNAL modes only (after skipping first 6).

    eigvals : np.ndarray
        Shape (n_modes_total,). Eigenvalues in Bio3D order
        (includes the first 6 rigid-body modes).
    """
    print_header("Running R aanma.pdb NMA in Bio3D (from protein-heavy traj)...")

    pdb_text = traj_to_pdb_string(traj_protein_heavy)

    setup_r_nma_function()
    r_run_aanma = ro.globalenv["run_aanma"]

    res = r_run_aanma(pdb_text)

    with localconverter(default_converter + numpy2ri.converter):
        U_r = res.rx2("U")
        vals_r = res.rx2("values")

    U = np.array(U_r, dtype=float)
    eigvals = np.array(vals_r, dtype=float).reshape(-1)

    n_dof, n_modes_r = U.shape
    print(f"[nma_bio3d] Mode matrix U shape: (3N_atoms={n_dof}, n_modes={n_modes_r})")
    print(f"[nma_bio3d] Eigenvalues shape: {eigvals.shape}")

    # Skip first 6 rigid-body modes
    start = 6
    end = min(start + n_modes_keep, n_modes_r)

    if end <= start:
        raise ValueError(
            f"[nma_bio3d] Requested {n_modes_keep} modes, but only {n_modes_r} "
            f"modes available in R (after 6 rigid-body)."
        )

    modes = U[:, start:end].T
    print(
        f"[nma_bio3d] Selected modes {start+1}..{end} "
        f"(total {end-start} modes for comparison)"
    )

    return modes, eigvals
