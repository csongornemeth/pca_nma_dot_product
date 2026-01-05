# pca_stream.py
from pathlib import Path
import numpy as np
import mdtraj as md
from sklearn.decomposition import IncrementalPCA

from io_utils import print_header


def yield_pca_chunks(
    xtc_paths: list[Path],
    topology: md.Topology,
    chunk_size: int,
    atom_indices: np.ndarray,
):
    """
    Generator that yields PCA-ready chunks X_chunk from one or more trajectories.
    Each X_chunk has shape (n_frames_chunk, 3 * n_atoms_sel).

    atom_indices: indices of the protein-heavy atoms on the FULL topology
                  (must be the same as used for NMA).
    """
    print_header("Streaming trajectory chunks for IncrementalPCA")

    ref_coords = None
    ref_topology = None

    for xtc in xtc_paths:
        print(f"[CHUNK] Reading from trajectory file: {xtc}")

        for traj_chunk in md.iterload(
            xtc.as_posix(),
            top=topology,  # FULL topology, matches XTC natoms
            chunk=chunk_size,
        ):
            # Slice to the EXACT same protein-heavy atoms as NMA
            traj_sel = traj_chunk.atom_slice(atom_indices)

            if ref_coords is None:
                ref_coords = traj_sel[0].xyz.copy()  # (1, n_atoms_sel, 3)
                ref_topology = traj_sel.topology
                print(f"[CHUNK] Global reference frame set with {traj_sel.n_atoms} atoms")

            ref_traj = md.Trajectory(ref_coords, ref_topology)
            traj_sel.superpose(ref_traj)

            xyz = traj_sel.xyz  # (n_frames_chunk, n_atoms_sel, 3)
            n_frames_chunk, n_atoms_sel, _ = xyz.shape
            X_chunk = xyz.reshape(n_frames_chunk, n_atoms_sel * 3)

            print(f"[CHUNK] Yielding chunk with shape: {X_chunk.shape}")
            yield X_chunk


def run_incremental_pca_from_chunks(
    xtc_paths: list[Path],
    topology: md.Topology,
    n_components: int,
    chunk_size: int,
    atom_indices: np.ndarray,
) -> np.ndarray:

    print_header("Running IncrementalPCA from streamed chunks")

    ipca = None
    total_frames = 0
    n_features = None

    for X_chunk in yield_pca_chunks(
        xtc_paths=xtc_paths,
        topology=topology,
        chunk_size=chunk_size,
        atom_indices=atom_indices,
    ):
        n_frames_chunk, n_features_chunk = X_chunk.shape
        total_frames += n_frames_chunk

        if ipca is None:
            n_features = n_features_chunk
            n_components_eff = min(n_components, n_features)
            print(f"[IPCA] First chunk shape: {X_chunk.shape}")
            print(f"[IPCA] Using n_components = {n_components_eff}")
            ipca = IncrementalPCA(n_components=n_components_eff)

        if n_features_chunk != n_features:
            raise ValueError(
                f"Chunk feature mismatch: expected {n_features}, got {n_features_chunk}"
            )

        ipca.partial_fit(X_chunk)

    if ipca is None:
        raise RuntimeError("No data chunks were produced. Check xtc_paths and atom_indices.")

    print(f"[IPCA] Total frames processed: {total_frames}")
    print(f"[IPCA] Final components shape: {ipca.components_.shape}")
    print(
        "[IPCA] Explained variance ratio (first few):",
        ipca.explained_variance_ratio_[: min(5, ipca.n_components_)],
    )

    return ipca.components_
