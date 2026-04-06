#!/bin/bash
set -euo pipefail

ROOT="$(pwd)"
RESULTS_DIR="${ROOT}/results"

SCRIPT="${ROOT}/pca_displacement_stat_multi.py"
PYTHON="/work001/csongor/env_items/envs/boxenv/bin/python"

PCS="${PCS:-1-3}"
SIGMA="${SIGMA:-1.0}"

OUTCSV="${ROOT}/combined_disp.csv"

echo "Running PCA displacement for all PDBs..."

# remove old combined file
rm -f "$OUTCSV"

FIRST=1

for d in "$RESULTS_DIR"/*; do
    [[ -d "$d" ]] || continue

    PDB_CODE="$(basename "$d")"

    # skip if no eigdirs
    if ! ls "$d"/*_pca_eigs >/dev/null 2>&1; then
        echo "[SKIP] $PDB_CODE (no eigdirs)"
        continue
    fi

    echo "[RUN] $PDB_CODE"

    TMPCSV="${ROOT}/tmp_${PDB_CODE}.csv"

    "$PYTHON" "$SCRIPT" \
        --pdb "$PDB_CODE" \
        --eigroot "$d" \
        --pcs "$PCS" \
        --sigma "$SIGMA" \
        --outcsv "$TMPCSV"

    if [[ ! -s "$TMPCSV" ]]; then
        echo "[SKIP] $PDB_CODE (no output)"
        continue
    fi

    if [[ "$FIRST" -eq 1 ]]; then
        cat "$TMPCSV" > "$OUTCSV"
        FIRST=0
    else
        tail -n +2 "$TMPCSV" >> "$OUTCSV"
    fi

    rm -f "$TMPCSV"
done

echo "[DONE] Combined CSV: $OUTCSV"