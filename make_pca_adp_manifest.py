#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import re
import sys

RESULTS = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results")

# matches: results/{pdb}/{pdb}_rep{n}_ipca.json.gz  (used only to validate pdb dirs)
PDB_RE = re.compile(r"^[0-9a-zA-Z]{4}$")

out_lines = []
for pdb_dir in sorted(RESULTS.iterdir()):
    if not pdb_dir.is_dir():
        continue
    pdb = pdb_dir.name
    if not PDB_RE.match(pdb):
        continue

    for eigdir in sorted(pdb_dir.glob("rep*_pca_eigs")):
        v = eigdir / "eigenvectors.npy"
        if v.exists():
            out_lines.append(f"{pdb}\t{eigdir}")

print("\n".join(out_lines))
