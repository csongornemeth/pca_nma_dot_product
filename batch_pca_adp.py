#!/usr/bin/env python3
from pathlib import Path
import subprocess

RESULTS = Path("results")
SCRIPT = Path("pca_adp.py")   # adjust if needed

PCS = "1-20"

for pdb_dir in sorted(RESULTS.iterdir()):
    if not pdb_dir.is_dir():
        continue

    pdb_code = pdb_dir.name

    for eigdir in sorted(pdb_dir.glob("rep*_pca_eigs")):
        out_png = eigdir / "pc_contrib_ca.png"

        cmd = [
            "python", str(SCRIPT),
            "--eigdir", str(eigdir),
            "--pdb-code", pdb_code,
            "--pcs", PCS,
            "--out", str(out_png),
        ]

        print("Running:", " ".join(cmd))
        subprocess.run(cmd, check=True)
