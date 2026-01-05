# io_utils.py
from pathlib import Path
import mdtraj as md

def print_header(title: str):
    bar = "=" * len(title)
    print(f"\n{bar}\n{title}\n{bar}\n")


def get_pdb_dir(pdb_code: str) -> Path:
    """Return base directory for a given PDB code."""
    return Path(
        f"~/Documents/box_project/nma_pca_comparison/data/{pdb_code}"
    ).expanduser()


def collect_xtc_paths(pdb_dir: Path) -> list[Path]:
    # Collect all prod.part*.xtc from all replica directories.
    xtc_paths: list[Path] = []
    val_root = pdb_dir / "validation"

    _require_exists(val_root, "validation directory")

    for i in range(10):  # replicas 0..9
        traj_dir = val_root / str(i)
        if not traj_dir.exists():
            print(f"[WARN] Missing replica directory: {traj_dir}")
            continue

        parts = list(traj_dir.glob("prod.part*.xtc"))
        if not parts:
            print(f"[WARN] No trajectory files found in: {traj_dir}")
            continue

        def key(p: Path) -> int:
            m = _part_re.search(p.name)
            return int(m.group(1)) if m else 10**9

        xtc_paths.extend(sorted(parts, key=key))

    if not xtc_paths:
        raise RuntimeError(f"No XTC files found under {val_root}/0..9. Cannot run PCA.")

    print_header("Trajectory files discovered (all replicas)")
    for x in xtc_paths:
        print(f"    {x}")
    return xtc_paths



def load_full_topology(pdb_dir: Path) -> tuple[md.Trajectory, md.Topology]:
    """
    Load full topology that matches the XTC (with water + ions).
    Here we assume prod.gro is the full system; if not, you can swap to prod.tpr.
    """
    prod_gro = (pdb_dir / "validation" / "0" / "prod.gro").expanduser()
    if not prod_gro.exists():
        raise FileNotFoundError(f"prod.gro not found at: {prod_gro}")

    traj_full = md.load(prod_gro.as_posix())
    top_full = traj_full.topology
    print(f"[MAIN] Full topology natoms (prod.gro): {top_full.n_atoms}")
    return traj_full, top_full


def get_protein_heavy_indices(top_full: md.Topology):
    """Return indices of protein heavy atoms on the full topology."""
    protein_heavy_idx = top_full.select("protein and not element H")
    print(f"[MAIN] Protein heavy-atom count: {protein_heavy_idx.size}")
    return protein_heavy_idx
