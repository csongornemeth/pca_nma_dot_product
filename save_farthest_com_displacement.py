# save_farthest_com_displacement.py

from __future__ import annotations

from pathlib import Path
import argparse
import json
import re

import numpy as np
import mdtraj as md

from io_utils import print_header
from traj_utils import build_protein_heavy_views
from align_core import compute_alignment_core_aidxs

"""
python save_farthest_com_displacement.py \
  --pdb 7lak \
  --results-root results \
  --outdir farthest_com_displacement_results
  """

def collect_cleaned_xtc_paths(
    pdb_code: str,
    results_root: str | Path = "results",
) -> list[Path]:
    """
    Collect cleaned XTC files from:
        results/{pdb}/tmp2/cleaned_{pdb}_*.xtc
    """

    pdb_code = pdb_code.strip().lower()
    tmp2_dir = Path(results_root) / pdb_code / "tmp2"

    if not tmp2_dir.exists():
        raise FileNotFoundError(f"tmp2 directory not found: {tmp2_dir}")

    xtc_paths = list(tmp2_dir.glob(f"cleaned_{pdb_code}_*.xtc"))

    def sort_key(path: Path):
        """
        Natural-ish sorting for names like:
            cleaned_8bdp_0.xtc
            cleaned_8bdp_1.xtc
            cleaned_8bdp_10.xtc
        """
        m = re.search(rf"cleaned_{re.escape(pdb_code)}_(\d+)\.xtc$", path.name)
        if m:
            return int(m.group(1))
        return 10**9

    xtc_paths = sorted(xtc_paths, key=sort_key)

    if not xtc_paths:
        raise FileNotFoundError(
            f"No cleaned XTC files found matching: "
            f"{tmp2_dir}/cleaned_{pdb_code}_*.xtc"
        )

    print_header("Cleaned XTC files discovered")
    for i, x in enumerate(xtc_paths):
        print(f"  [{i}] {x}")

    return xtc_paths


def find_atom_farthest_from_com_over_simulation(
    xtc_paths: list[Path],
    top_xtc_full: md.Topology,
    protein_heavy_idx_full: np.ndarray,
    core_aidxs: np.ndarray | None = None,
    max_frames: int | None = None,
    chunk: int = 2500,
    align: bool = True,
):
    """
    Scan all frames and find the protein-heavy atom that becomes farthest
    from the protein-heavy COM at any point during the simulation.

    Then calculate the displacement vector of that same atom from frame 0
    to the frame where it is farthest from the COM.

    The displacement vector is:

        r_atom(best_frame) - r_atom(frame_0)

    If align=True, all frames are first aligned to frame 0 using core_aidxs.
    """

    ref = None
    reference_xyz_frame0 = None

    best_distance = -np.inf
    best_global_frame = None
    best_atom_index_local = None
    best_position = None
    best_com = None
    best_xtc = None
    best_xtc_local_frame = None

    global_frame_counter = 0

    for xtc in xtc_paths:
        xtc_local_frame_counter = 0

        for tr in md.iterload(
            xtc.as_posix(),
            top=top_xtc_full,
            chunk=chunk,
        ):
            # Slice heavy full trajectory to protein-heavy only.
            trp = tr.atom_slice(protein_heavy_idx_full)

            # Stop exactly at max_frames if requested.
            if max_frames is not None:
                remaining = max_frames - global_frame_counter
                if remaining <= 0:
                    break
                if trp.n_frames > remaining:
                    trp = trp[:remaining]

            # First frame of the whole simulation is the reference.
            if ref is None:
                ref = trp[0]

            # Align current chunk to reference frame.
            if align:
                if core_aidxs is None:
                    trp.superpose(ref)
                else:
                    trp.superpose(ref, 0, atom_indices=core_aidxs)

            # Save frame 0 coordinates after alignment.
            if reference_xyz_frame0 is None:
                reference_xyz_frame0 = trp.xyz[0].copy()

            # COM for each frame.
            com = md.compute_center_of_mass(trp)  # shape: (n_frames, 3)

            # Distances of all protein-heavy atoms from COM in all frames.
            centered = trp.xyz - com[:, None, :]
            distances = np.linalg.norm(centered, axis=2)

            # Find best atom-frame pair inside this chunk.
            flat_idx = int(np.argmax(distances))
            chunk_frame_idx, atom_idx = np.unravel_index(
                flat_idx,
                distances.shape,
            )

            distance = float(distances[chunk_frame_idx, atom_idx])

            if distance > best_distance:
                best_distance = distance
                best_global_frame = global_frame_counter + chunk_frame_idx
                best_atom_index_local = int(atom_idx)
                best_position = trp.xyz[chunk_frame_idx, atom_idx].copy()
                best_com = com[chunk_frame_idx].copy()
                best_xtc = xtc
                best_xtc_local_frame = xtc_local_frame_counter + chunk_frame_idx

            global_frame_counter += trp.n_frames
            xtc_local_frame_counter += trp.n_frames

            if max_frames is not None and global_frame_counter >= max_frames:
                break

        if max_frames is not None and global_frame_counter >= max_frames:
            break

    if best_atom_index_local is None:
        raise RuntimeError("No frames were processed.")

    reference_position = reference_xyz_frame0[best_atom_index_local]
    displacement_vector = best_position - reference_position
    displacement_norm = float(np.linalg.norm(displacement_vector))

    atom = ref.topology.atom(best_atom_index_local)

    return {
        "atom_index_local": int(best_atom_index_local),
        "atom_name": atom.name,
        "residue": str(atom.residue),
        "best_global_frame": int(best_global_frame),
        "best_xtc": str(best_xtc),
        "best_xtc_local_frame": int(best_xtc_local_frame),
        "max_distance_from_com_nm": float(best_distance),
        "reference_position_frame0_nm": reference_position,
        "best_position_nm": best_position,
        "best_com_nm": best_com,
        "displacement_vector_nm": displacement_vector,
        "displacement_norm_nm": displacement_norm,
        "n_frames_processed": int(global_frame_counter),
        "aligned": bool(align),
    }


def save_farthest_com_displacement(
    result: dict,
    outdir: Path,
    prefix: str,
):
    """
    Save displacement vector and metadata.
    """

    outdir.mkdir(parents=True, exist_ok=True)

    vector_npy = outdir / f"{prefix}_farthest_com_displacement_vector.npy"
    vector_txt = outdir / f"{prefix}_farthest_com_displacement_vector.txt"
    metadata_json = outdir / f"{prefix}_farthest_com_displacement_metadata.json"

    np.save(vector_npy, result["displacement_vector_nm"])

    np.savetxt(
        vector_txt,
        result["displacement_vector_nm"].reshape(1, 3),
        header="dx_nm dy_nm dz_nm",
        fmt="%.8f",
    )

    serialisable = {}
    for key, value in result.items():
        if isinstance(value, np.ndarray):
            serialisable[key] = value.tolist()
        else:
            serialisable[key] = value

    with open(metadata_json, "w") as f:
        json.dump(serialisable, f, indent=2)

    print_header("Saved outputs")
    print(f"[SAVED] Vector NPY:   {vector_npy}")
    print(f"[SAVED] Vector TXT:   {vector_txt}")
    print(f"[SAVED] Metadata JSON:{metadata_json}")

    return vector_npy, vector_txt, metadata_json


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--pdb",
        required=True,
        help="PDB/system code, e.g. 8bdp",
    )
    parser.add_argument(
        "--results-root",
        default="results",
        help="Root containing results/{pdb}/tmp2/cleaned_{pdb}_*.xtc",
    )
    parser.add_argument(
        "--outdir",
        default="farthest_com_displacement_results",
        help="Output directory",
    )
    parser.add_argument(
        "--chunk",
        type=int,
        default=2500,
        help="Trajectory chunk size",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Maximum number of frames to scan. Default: all frames.",
    )
    parser.add_argument(
        "--no-align",
        action="store_true",
        help="Do not align trajectory frames to frame 0 before measuring displacement.",
    )
    parser.add_argument(
        "--core-max-frames",
        type=int,
        default=10000,
        help="Maximum frames used to determine alignment core.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
    )

    args = parser.parse_args()

    pdb_code = args.pdb.strip().lower()

    print_header(f"Farthest-COM displacement vector - {pdb_code}")

    print_header("Building topology views")

    (
        traj_protein_heavy_ref,
        top_xtc_protein,
        protein_heavy_idx_local,
        protein_heavy_idx_full,
        top_xtc_full,
    ) = build_protein_heavy_views(pdb_code)

    print_header("Collecting cleaned XTC files")

    xtc_paths = collect_cleaned_xtc_paths(
        pdb_code=pdb_code,
        results_root=args.results_root,
    )

    align = not args.no_align

    if align:
        print_header("Computing alignment core")

        core_aidxs = compute_alignment_core_aidxs(
            xtc_paths=xtc_paths,
            topology=top_xtc_full,
            atom_indices_full=protein_heavy_idx_full,
            max_frames=args.core_max_frames,
            verbose=args.verbose,
        )
    else:
        core_aidxs = None
        print("[INFO] Alignment disabled.")

    print_header("Scanning trajectory for farthest atom from COM")

    result = find_atom_farthest_from_com_over_simulation(
        xtc_paths=xtc_paths,
        top_xtc_full=top_xtc_full,
        protein_heavy_idx_full=protein_heavy_idx_full,
        core_aidxs=core_aidxs,
        max_frames=args.max_frames,
        chunk=args.chunk,
        align=align,
    )

    print_header("Result")

    print(f"[RESULT] Atom local index:       {result['atom_index_local']}")
    print(f"[RESULT] Atom:                   {result['atom_name']} {result['residue']}")
    print(f"[RESULT] Best global frame:      {result['best_global_frame']}")
    print(f"[RESULT] Best XTC:               {result['best_xtc']}")
    print(f"[RESULT] Best frame in XTC:      {result['best_xtc_local_frame']}")
    print(f"[RESULT] Max COM distance:       {result['max_distance_from_com_nm']:.6f} nm")
    print(f"[RESULT] Displacement vector:    {result['displacement_vector_nm']}")
    print(f"[RESULT] Displacement norm:      {result['displacement_norm_nm']:.6f} nm")
    print(f"[RESULT] Frames processed:       {result['n_frames_processed']}")
    print(f"[RESULT] Aligned before measure: {result['aligned']}")

    outdir = Path(args.outdir) / pdb_code
    prefix = pdb_code

    save_farthest_com_displacement(
        result=result,
        outdir=outdir,
        prefix=prefix,
    )


if __name__ == "__main__":
    main()