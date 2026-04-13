from pathlib import Path
import subprocess
from io_utils import get_pdb_dir, print_header

# user setting
gmx = "/work001/software/gromacs-bekker-2025/build/bin/gmx"
pdbid = input("pdbid: ").strip().lower()

pdb_dir = get_pdb_dir(pdbid)
val_root = pdb_dir / "validation"

out_root = Path(f"results/{pdbid}")
tmp_dir = out_root / "tmp"
dummy_tpr = out_root / "dummy.tpr"

out_root.mkdir(parents=True, exist_ok=True)
tmp_dir.mkdir(parents=True, exist_ok=True)

print_header(f"Cleaning trajectories for {pdbid}")
print(f"PDB dir: {pdb_dir}")
print(f"Validation dir: {val_root}")
print(f"Temporary dir: {tmp_dir}")
print(f"Dummy TPR: {dummy_tpr}")

for replica_id in range(10):
    traj_dir = val_root / str(replica_id)

    if not traj_dir.exists():
        print(f"[WARN] Trajectory directory {traj_dir} does not exist. Skipping replica {replica_id}.")
        continue

    xtc_files = sorted(traj_dir.glob("prod.part*.xtc"))

    if not xtc_files:
        print(f"[WARN] No .xtc files found in {traj_dir}. Skipping replica {replica_id}.")
        continue

    print_header(f"Replica {replica_id}")
    tmp_xtcs = []

    for fl in xtc_files:
        partid = fl.stem
        tmpfile = tmp_dir / f"{pdbid}_{replica_id}_{partid}.xtc"
        tmp_xtcs.append(str(tmpfile))

        cmd = [
            gmx, "trjconv",
            "-s", str(dummy_tpr),
            "-f", str(fl),
            "-o", str(tmpfile),
            "-pbc", "mol",
        ]

        print(f"[RUN] {' '.join(cmd)}")
        subprocess.run(
            cmd,
            input="Protein-H\n",
            text=True,
            check=True,
        )

    out_xtc = out_root / f"cleaned_{pdbid}_{replica_id}.xtc"

    cmd_cat = [gmx, "trjcat", "-f", *tmp_xtcs, "-o", str(out_xtc)]
    print(f"[RUN] {' '.join(cmd_cat)}")
    subprocess.run(cmd_cat, check=True)

    for f in tmp_xtcs:
        Path(f).unlink(missing_ok=True)

    print(f"[OK] Replica {replica_id} processed. Output: {out_xtc}")