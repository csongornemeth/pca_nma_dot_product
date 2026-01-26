python - <<'PY'
import pandas as pd

df = pd.read_csv("pdb_selection.csv")

# EDIT column name if needed
codes = df["pdb_code"].dropna().unique()

with open("pdbcodes.txt", "w") as f:
    for c in codes:
        f.write(f"{c}\n")

print(f"Wrote {len(codes)} pdb codes")
PY

