#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
from collections import defaultdict
from pathlib import Path

from io_utils import get_pdb_dir, collect_xtc_paths, print_header

"""
Run example:
python dummy_tpr.py --pdb 1a7u --group Protein-H 
"""


GMX_DEFAULT = "/work001/software/gromacs-bekker-2025/build/bin/gmx"
gmx = GMX_DEFAULT

def run_cmd(cmd, input_text=None):
    result = subprocess.run(
        cmd,
        input=input_text,
        text=True,
        capture_output=True,
    )
    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr)
        raise RuntimeError(f"Failed: {' '.join(cmd)}")


# -------------------------
# CLEANUP LOGIC
# -------------------------
def cleanup_tmp_dir(tmp_dir: Path, pdb_code: str, wipe: bool):
    print_header("Cleanup")

    if not tmp_dir.exists():
        return

    if wipe:
        print("[INFO] Wiping entire tmp directory")
        for f in tmp_dir.glob("*"):
            try:
                f.unlink()
            except IsADirectoryError:
                pass
        return

    # selective cleanup
    removed = 0

    for f in tmp_dir.glob("*"):
        name = f.name

        # remove intermediate xtc parts
        if name.startswith(f"{pdb_code}_") and name.endswith(".xtc"):
            print(f"[CLEAN] Removing temp: {name}")
            f.unlink()
            removed += 1

        # remove dummy tpr
        elif name == "dummy.tpr":
            print(f"[CLEAN] Removing old dummy.tpr")
            f.unlink()
            removed += 1

    print(f"[INFO] Removed {removed} leftover files")


# -------------------------
def make_dummy_tpr(gmx, build_dir, dummy_tpr, group_name):
    print_header("Creating dummy.tpr")

    cmd = [
        gmx, "convert-tpr",
        "-s", str(build_dir / "npt.tpr"),
        "-n", str(build_dir / "index.ndx"),
        "-o", str(dummy_tpr),
    ]
    run_cmd(cmd, input_text=f"{group_name}\n")


def group_xtcs_by_replica(xtc_paths):
    replica_groups = defaultdict(list)

    for xtc in xtc_paths:
        replica_id = xtc.parent.name
        replica_groups[replica_id].append(xtc)

    for r in replica_groups:
        replica_groups[r] = sorted(replica_groups[r])

    return replica_groups


def clean_replica(gmx, pdb_code, replica_id, xtc_files, dummy_tpr, group_name, tmp_dir):
    print_header(f"Replica {replica_id}")

    temp_files = []

    for i, xtc in enumerate(xtc_files):
        tmp_xtc = tmp_dir / f"{pdb_code}_{replica_id}_{i}.xtc"

        print(f"[INFO] {xtc.name} -> {tmp_xtc.name}")

        run_cmd(
            [
                gmx, "trjconv",
                "-s", str(dummy_tpr),
                "-f", str(xtc),
                "-o", str(tmp_xtc),
                "-pbc", "mol",
            ],
            input_text=f"{group_name}\n",
        )

        temp_files.append(str(tmp_xtc))

    out_xtc = tmp_dir / f"cleaned_{pdb_code}_{replica_id}.xtc"

    run_cmd(
        [gmx, "trjcat", "-f", *temp_files, "-o", str(out_xtc)]
    )

    # remove intermediates immediately
    for f in temp_files:
        try:
            os.remove(f)
        except FileNotFoundError:
            pass

    print(f"[OK] {out_xtc}")
    return out_xtc


# -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdb", required=True)
    parser.add_argument("--group", required=True, choices=["Protein-H", "solute-H"])
    parser.add_argument("--gmx", default=GMX_DEFAULT)

    # NEW FLAGS
    parser.add_argument("--wipe", action="store_true", help="Delete all files in tmp before running")

    args = parser.parse_args()

    pdb_code = args.pdb.lower()
    pdb_dir = get_pdb_dir(pdb_code)
    build_dir = pdb_dir / "build"

    out_root = Path(f"results/{pdb_code}")
    tmp_dir = out_root / "tmp"

    out_root.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # 🔥 CLEANUP FIRST
    cleanup_tmp_dir(tmp_dir, pdb_code, wipe=args.wipe)

    dummy_tpr = tmp_dir / "dummy.tpr"

    print_header("Paths")
    print(f"PDB dir: {pdb_dir}")
    print(f"TMP dir: {tmp_dir}")
    print(f"GROMACS: {args.gmx}")
    print(f"Group:   {args.group}")

    # 1. dummy tpr
    make_dummy_tpr(args.gmx, build_dir, dummy_tpr, args.group)

    # 2. xtc collection
    xtc_paths = collect_xtc_paths(pdb_dir)
    replica_groups = group_xtcs_by_replica(xtc_paths)

    # 3. process
    for replica_id in sorted(replica_groups, key=lambda x: int(x)):
        clean_replica(
            args.gmx,
            pdb_code,
            replica_id,
            replica_groups[replica_id],
            dummy_tpr,
            args.group,
            tmp_dir,
        )

    print_header("DONE")


if __name__ == "__main__":
    main()