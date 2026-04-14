#!/bin/bash
set -euo pipefail

ROOT="$(pwd)"

SCRIPT="${ROOT}/dummy_tpr.py"
PYTHON="/work001/csongor/env_items/envs/boxenv/bin/python"

GROUP="${GROUP:-Protein-H}"
PDB_LIST="${1:-failed_pdbs.txt}"   # default file if not provided

if [[ ! -f "$PDB_LIST" ]]; then
    echo "[ERROR] PDB list file not found: $PDB_LIST"
    exit 1
fi

echo "Running dummy_tpr.py using list: $PDB_LIST"

FAILED=()

while read -r PDB_CODE; do
    # skip empty lines or comments
    [[ -z "$PDB_CODE" || "$PDB_CODE" =~ ^# ]] && continue

    echo "[RUN] $PDB_CODE"

    if ! "$PYTHON" "$SCRIPT" \
        --pdb "$PDB_CODE" \
        --group "$GROUP"
    then
        echo "[FAIL] $PDB_CODE"
        FAILED+=("$PDB_CODE")
        continue
    fi

    echo "[OK] $PDB_CODE"
done < "$PDB_LIST"

echo
if [[ ${#FAILED[@]} -gt 0 ]]; then
    echo "[DONE] Finished with failures:"
    printf '  %s\n' "${FAILED[@]}"
    exit 1
else
    echo "[DONE] All PDBs completed successfully."
fi