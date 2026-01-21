from __future__ import annotations

from pathlib import Path
import csv
import mdtraj as md

RAW_ROOT = Path("/work001/misc/bekker/kakC/dynamicsdb/raw")
PHASES = {"1", "2", "2b", "3", "3b", "4", "4b", "4c", "4d", "4f", "5"}
GRO_NAME = "complex.gro"


def atom_counter(top: md.Topology) -> dict:
    """Count protein size metrics from an mdtraj Topology (GRO-friendly)."""
    prot_idx = top.select("protein")
    prot_atoms = [top.atom(i) for i in prot_idx]

    protein_atoms = len(prot_idx)
    protein_heavy_atoms = sum(1 for a in prot_atoms if not a.name.upper().startswith("H"))

    n_ca = int(top.select("protein and name CA").size)
    n_residues = sum(1 for r in top.residues if r.is_protein)
    n_chains = sum(1 for c in top.chains if any(r.is_protein for r in c.residues))

    return {
        "protein_atoms": protein_atoms,
        "protein_heavy_atoms": protein_heavy_atoms,
        "n_ca": n_ca,
        "n_residues": n_residues,
        "n_chains": n_chains,
        "total_atoms": top.n_atoms,
    }


def main() -> None:
    out_csv = Path("atom_counts.csv")
    rows: list[dict] = []
    seen_pdbs: set[str] = set()

    # If a PDB appears in multiple phases, we keep the FIRST successful one encountered.
    for phase in sorted(PHASES):
        phase_dir = RAW_ROOT / phase
        if not phase_dir.exists():
            continue

        for pdb_dir in sorted(p for p in phase_dir.iterdir() if p.is_dir()):
            pdb_code = pdb_dir.name
            if pdb_code in seen_pdbs:
                continue

            gro = pdb_dir / "build" / GRO_NAME
            if not gro.exists():
                continue  # don't write errors/missing entries

            try:
                traj = md.load(str(gro))
                counts = atom_counter(traj.topology)

                rows.append(
                    {
                        "pdb_code": pdb_code,
                        "phase": phase,
                        "gro": str(gro),
                        **counts,
                    }
                )
                seen_pdbs.add(pdb_code)

            except Exception:
                # Skip failures entirely (no row written)
                continue

    # Write only successful PDB entries
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "pdb_code",
                "phase",
                "gro",
                "protein_atoms",
                "protein_heavy_atoms",
                "n_ca",
                "n_residues",
                "n_chains",
                "total_atoms",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} PDB entries to {out_csv}")


if __name__ == "__main__":
    main()
