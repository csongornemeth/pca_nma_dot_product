#!/usr/bin/env python3
"""
rmsf.py

RMSF over time windows + per-replica + replica-mean (SEM/SD), using MDTraj.

DB-aware:
- Uses io_utils.get_pdb_dir(pdb_code) to locate RAW_ROOT/<phase>/<pdb_code>
- Uses io_utils.collect_xtc_paths(pdb_dir) to discover validation/<rep>/prod.part*.xtc
- Groups XTC parts per replica by parsing the path (parent dir name)

Topology/atom views:
- Uses traj_utils.build_protein_heavy_views(pdb_code) to get:
    top_xtc_full (heavy protein+ligand topology matching XTC atom ordering)
    protein_heavy_idx_full (protein-heavy indices on top_xtc_full)
    top_xtc_protein (protein-heavy topology)
- Computes a stable alignment core ONCE using align_core.compute_alignment_core_aidxs()
  over sampled frames from all selected replicas (core indices are in protein-heavy LOCAL space).
- Loads each replica's concatenated XTC parts with top=top_xtc_full
- Slices to protein-heavy (or CA-only)
- Superposes frames within each time window to the first frame of that window
  using the stable core indices mapped into the current selection index space.
- Computes RMSF per-atom relative to the window-mean position:
    RMSF_i = sqrt( <|r_i(t) - <r_i>|^2>_t )
- Optionally reduces to per-residue RMSF by averaging per-atom RMSF within each residue.

Optional per-chain mode (--by-chain):
- Primary method: uses top_xtc_full.find_molecules() to detect separate protein molecules.
- Fallback (when find_molecules cannot split but you know there are 2 chains):
  infer chain boundary by a large CA–CA gap along residue order (default threshold 5.0 Å).
  This works when the topology concatenates chain A then chain B in residue order.

Outputs:
Single/all-protein outputs go to:
  outdir/
Per-chain outputs (if enabled and split succeeds) go to:
  outdir/chain1/, outdir/chain2/, ...

Per window:
- rmsf_replicas_overlay__{window}.png
- rmsf_mean_band__{window}.png
- rmsf__{window}.csv
"""

import argparse
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import mdtraj as md
import matplotlib.pyplot as plt
import numpy as np

from align_core import compute_alignment_core_aidxs
from io_utils import collect_xtc_paths, get_pdb_dir, load_full_topology
from traj_utils import build_protein_heavy_views


# ----------------------------
# Data structures
# ----------------------------

@dataclass
class Window:
    start_ns: float
    end_ns: float

    @property
    def tag(self) -> str:
        a = str(self.start_ns).replace(".", "p")
        b = str(self.end_ns).replace(".", "p")
        return f"{a}_{b}ns"


# ----------------------------
# Basic helpers
# ----------------------------

def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def parse_windows(window_args: List[str]) -> List[Window]:
    out: List[Window] = []
    for w in window_args:
        parts = w.split("-")
        if len(parts) != 2:
            raise ValueError(f"Bad window '{w}'. Use START-END in ns, e.g. 0-50")
        out.append(Window(float(parts[0]), float(parts[1])))
    return out


def frames_in_window(traj: md.Trajectory, window: Window) -> np.ndarray:
    if traj.time is None or len(traj.time) == 0:
        raise RuntimeError(
            "Trajectory has no time information (traj.time is None/empty); "
            "cannot window by ns."
        )

    t_ns = traj.time / 1000.0  # ps -> ns
    mask = (t_ns >= window.start_ns) & (t_ns < window.end_ns)
    idx = np.where(mask)[0]

    if idx.size < 2:
        raise RuntimeError(
            f"Window {window.start_ns}-{window.end_ns} ns has <2 frames (found {idx.size}). "
            f"Available t(ns): {t_ns[0]:.3f}->{t_ns[-1]:.3f}"
        )
    return idx


def rmsf_per_atom_window(traj: md.Trajectory, frame_idx: np.ndarray, align_idx: np.ndarray) -> np.ndarray:
    """
    align_idx must be in the index space of traj (after any atom_slice).
    """
    tw = traj.slice(frame_idx, copy=True)
    tw.superpose(tw, 0, atom_indices=align_idx)

    xyz = tw.xyz  # (n_frames, n_atoms, 3) in nm
    mean_xyz = xyz.mean(axis=0, keepdims=True)
    d = xyz - mean_xyz
    return np.sqrt((d * d).sum(axis=2).mean(axis=0))


def atom_to_residue_rmsf(top: md.Topology, rmsf_atom: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Per-residue RMSF by averaging per-atom RMSF within each residue.
    X-axis is forced to 1..Nres to be replica-consistent (do not use resSeq).
    """
    residues = list(top.residues)
    res_rmsf = np.zeros(len(residues), dtype=float)
    res_id = np.arange(1, len(residues) + 1, dtype=int)

    for i, r in enumerate(residues):
        aidx = [a.index for a in r.atoms]
        res_rmsf[i] = float(np.mean(rmsf_atom[np.asarray(aidx, dtype=int)])) if aidx else np.nan

    ok = ~np.isnan(res_rmsf)
    return res_id[ok], res_rmsf[ok]


def plot_overlay(x, ys: Dict[str, np.ndarray], title: str, outpath: str, xlabel="Residue", ylabel="RMSF (nm)"):
    plt.figure()
    for name, y in ys.items():
        plt.plot(x, y, label=name)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(frameon=False, fontsize=8)
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()


def plot_mean_band(x, mean, band, band_label: str, title: str, outpath: str, xlabel="Residue", ylabel="RMSF (nm)"):
    plt.figure()
    plt.plot(x, mean, label="mean")
    plt.fill_between(x, mean - band, mean + band, alpha=0.2, label=band_label)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(frameon=False, fontsize=8)
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()


def write_csv(outpath: str, x, per_rep: Dict[str, np.ndarray], mean: np.ndarray, band: np.ndarray, band_name: str):
    reps = sorted(per_rep.keys())
    header = ["x"] + reps + ["mean", band_name]
    cols = [x] + [per_rep[r] for r in reps] + [mean, band]
    arr = np.vstack(cols).T
    np.savetxt(outpath, arr, delimiter=",", header=",".join(header), comments="")


def map_values_to_slice_indices(values: np.ndarray, slice_values: np.ndarray) -> np.ndarray:
    """
    Given:
      - slice_values: the indices you used in atom_slice, in the exact order passed
      - values: a subset of those indices (same value-space)
    Return:
      - indices in sliced trajectory (0..len(slice_values)-1) corresponding to values
    """
    slice_values = np.asarray(slice_values, dtype=int)
    values = np.asarray(values, dtype=int)
    idx_map = {int(v): i for i, v in enumerate(slice_values)}
    out = [idx_map[int(v)] for v in values if int(v) in idx_map]
    return np.unique(np.asarray(out, dtype=int))


# ----------------------------
# Chain detection helpers
# ----------------------------

def detect_protein_molecules(top_xtc_full: md.Topology) -> List[np.ndarray]:
    """
    Attempt to recover separate protein molecules (often chains) from topology connectivity.
    Returns a list of FULL-topology atom index arrays, sorted by descending size.
    """
    mols = top_xtc_full.find_molecules()  # list[set[Atom]]
    protein_mols: List[np.ndarray] = []

    for m in mols:
        atoms = list(m)
        if not atoms:
            continue
        if all(a.residue.is_protein for a in atoms):
            protein_mols.append(np.array([a.index for a in atoms], dtype=int))

    protein_mols.sort(key=lambda a: a.size, reverse=True)
    return protein_mols


def infer_chain_splits_by_ca_gap(
    traj_ca: md.Trajectory,
    gap_threshold_nm: float = 0.20,  # 2.0 Å
) -> List[Tuple[int, int]]:
    """
    Infer chain splits from a CA–CA distance gap along residue order.
    traj_ca should be CA-only trajectory whose residues are in the same order as the protein.

    Returns list of residue-index ranges [(0,cut), (cut,n_res)] if a strong gap exists,
    else [(0,n_res)].
    """
    top = traj_ca.topology
    residues = list(top.residues)
    n_res = len(residues)
    if n_res < 2:
        return [(0, n_res)]

    # in CA-only traj, there should be exactly one CA atom per residue
    # but we’ll still map robustly:
    ca_atoms = []
    for r in residues:
        ca = [a.index for a in r.atoms if a.name == "CA"]
        if not ca:
            return [(0, n_res)]
        ca_atoms.append(ca[0])
    ca_atoms = np.asarray(ca_atoms, dtype=int)

    xyz = traj_ca.xyz[0, ca_atoms, :]  # (n_res, 3)
    d = np.sqrt(((xyz[1:] - xyz[:-1]) ** 2).sum(axis=1))  # (n_res-1,)

    j = int(np.argmax(d))
    dmax = float(d[j])

    if dmax < gap_threshold_nm:
        return [(0, n_res)]

    return [(0, j + 1), (j + 1, n_res)]


def residue_ranges_to_full_atom_indices(
    top_xtc_protein: md.Topology,
    protein_heavy_idx_full: np.ndarray,
    residue_ranges: List[Tuple[int, int]],
    mode: str,
) -> List[np.ndarray]:
    """
    residue_ranges are ranges in residue-index space of top_xtc_protein (protein-heavy topology).

    Returns list of FULL-topology atom-index arrays for each range, for the requested mode.
    """
    ph_residues = list(top_xtc_protein.residues)

    # local atom indices per residue (in protein-heavy local space)
    res_to_ph_atoms: List[List[int]] = []
    for r in ph_residues:
        res_to_ph_atoms.append([a.index for a in r.atoms])

    ca_local = None
    if mode == "ca":
        ca_local = np.asarray(top_xtc_protein.select("name CA"), dtype=int)

    chains_full: List[np.ndarray] = []
    for (rs, re) in residue_ranges:
        ph_atoms_local: List[int] = []
        for ri in range(rs, re):
            ph_atoms_local.extend(res_to_ph_atoms[ri])

        ph_atoms_local_arr = np.asarray(ph_atoms_local, dtype=int)

        if mode == "ca":
            ph_atoms_local_arr = np.intersect1d(ph_atoms_local_arr, ca_local)

        full_idx = protein_heavy_idx_full[ph_atoms_local_arr]
        chains_full.append(np.asarray(full_idx, dtype=int))

    return chains_full


# ----------------------------
# Window computation helper
# ----------------------------

def compute_window_outputs(
    *,
    outdir: str,
    windows: List[Window],
    mode: str,
    per: str,
    band: str,
    replicas: Dict[str, md.Trajectory],
    core_idx: np.ndarray,
    title_prefix: str,
) -> None:
    example_top = next(iter(replicas.values())).topology

    for w in windows:
        per_rep_series: Dict[str, np.ndarray] = {}
        x_axis: Optional[np.ndarray] = None

        for rep_name, traj in replicas.items():
            idx = frames_in_window(traj, w)
            rmsf_atom = rmsf_per_atom_window(traj, idx, core_idx)

            if per == "atom":
                x = np.arange(1, rmsf_atom.size + 1, dtype=int)
                y = rmsf_atom
            else:
                x, y = atom_to_residue_rmsf(example_top, rmsf_atom)

            if x_axis is None:
                x_axis = x
            else:
                if x.shape != x_axis.shape or not np.all(x == x_axis):
                    raise RuntimeError("Axis mismatch across replicas (unexpected).")

            per_rep_series[rep_name] = y

        reps_sorted = sorted(per_rep_series.keys())
        Y = np.vstack([per_rep_series[k] for k in reps_sorted])
        mean = Y.mean(axis=0)

        if band == "none":
            band_arr = np.zeros_like(mean)
            band_name = "band"
        elif band == "sd":
            band_arr = Y.std(axis=0, ddof=1) if Y.shape[0] > 1 else np.zeros_like(mean)
            band_name = "sd"
        else:
            sd = Y.std(axis=0, ddof=1) if Y.shape[0] > 1 else np.zeros_like(mean)
            band_arr = sd / np.sqrt(Y.shape[0]) if Y.shape[0] > 0 else np.zeros_like(mean)
            band_name = "sem"

        tag = w.tag
        xlab = "Atom index" if per == "atom" else "Residue"
        ylabel = "RMSF (nm)"

        overlay_png = os.path.join(outdir, f"rmsf_replicas_overlay__{tag}.png")
        plot_overlay(
            x_axis,
            per_rep_series,
            title=f"{title_prefix} RMSF ({mode}, {per}) {w.start_ns}-{w.end_ns} ns",
            outpath=overlay_png,
            xlabel=xlab,
            ylabel=ylabel,
        )

        mean_png = os.path.join(outdir, f"rmsf_mean_band__{tag}.png")
        plot_mean_band(
            x_axis,
            mean,
            band_arr,
            band_label=band_name,
            title=f"{title_prefix} RMSF mean±{band_name} ({mode}, {per}) {w.start_ns}-{w.end_ns} ns",
            outpath=mean_png,
            xlabel=xlab,
            ylabel=ylabel,
        )

        csv_path = os.path.join(outdir, f"rmsf__{tag}.csv")
        write_csv(csv_path, x_axis, per_rep_series, mean, band_arr, band_name)

        print(f"[OK] {title_prefix} window {w.start_ns}-{w.end_ns} ns -> {overlay_png}, {mean_png}, {csv_path}")


# ----------------------------
# Main
# ----------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdb-code", required=True, help="PDB code used by io_utils.get_pdb_dir()")
    ap.add_argument("--outdir", required=True, help="Output directory")
    ap.add_argument("--windows", nargs="+", required=True, help='Windows in ns: e.g. 0-50 50-100')
    ap.add_argument("--mode", choices=["protein-heavy", "ca"], default="ca",
                    help="Atoms used for RMSF: protein-heavy or CA-only")
    ap.add_argument("--per", choices=["residue", "atom"], default="residue",
                    help="Report RMSF per residue (averaged) or per atom")
    ap.add_argument("--band", choices=["sem", "sd", "none"], default="sem",
                    help="Band for replica-mean plot")
    ap.add_argument("--replicas", nargs="*", type=int, default=None,
                    help="Optional subset of replicas to use, e.g. --replicas 0 2 5")
    ap.add_argument("--core-max-frames", type=int, default=2000,
                    help="Max frames sampled (across all selected replicas) to determine alignment core")
    ap.add_argument("--by-chain", action="store_true",
                    help="If multiple chains exist, also compute RMSF per chain")
    ap.add_argument("--max-chains", type=int, default=4,
                    help="Max number of chains to output in --by-chain mode")
    ap.add_argument("--chain-gap-A", type=float, default=5.0,
                    help="Fallback CA–CA gap threshold in Å to infer chain break when find_molecules() cannot split")
    args = ap.parse_args()

    ensure_dir(args.outdir)
    windows = parse_windows(args.windows)

    # --- DB-aware discovery ---
    pdb_dir = get_pdb_dir(args.pdb_code)
    _ = load_full_topology(pdb_dir)  # sanity check / prints natoms

    xtc_paths = collect_xtc_paths(pdb_dir)  # list[Path], flattened across replicas

    replica_parts: Dict[int, List[os.PathLike]] = defaultdict(list)
    for p in xtc_paths:
        replica_parts[int(p.parent.name)].append(p)

    for rid in list(replica_parts.keys()):
        replica_parts[rid] = sorted(replica_parts[rid])

    rep_ids = sorted(replica_parts.keys())
    if args.replicas:
        wanted = set(args.replicas)
        rep_ids = [r for r in rep_ids if r in wanted]
        if not rep_ids:
            raise RuntimeError(
                f"No replicas matched --replicas {args.replicas}. Available: {sorted(replica_parts.keys())}"
            )

    print(f"[RMSF] Using replicas: {rep_ids}")
    for r in rep_ids:
        parts = replica_parts[r]
        print(f"[RMSF]  rep{r}: {len(parts)} parts (first={parts[0].name}, last={parts[-1].name})")

    # --- topology views + indices ---
    _, top_xtc_protein, _, protein_heavy_idx_full, top_xtc_full = build_protein_heavy_views(args.pdb_code)

    # --- compute stable core once (protein-heavy LOCAL indices) ---
    all_parts = []
    for rid in rep_ids:
        all_parts.extend(replica_parts[rid])

    core_aidxs_local = compute_alignment_core_aidxs(
        xtc_paths=all_parts,
        topology=top_xtc_full,
        atom_indices_full=protein_heavy_idx_full,
        max_frames=args.core_max_frames,
        verbose=True,
    )
    print(f"[RMSF] Alignment core size (protein-heavy local): {core_aidxs_local.size}")

    # --- single(all-protein) selection + core mapping ---
    if args.mode == "protein-heavy":
        atom_idx_full_single = protein_heavy_idx_full
        core_idx_single = core_aidxs_local
    else:
        ca_local = np.asarray(top_xtc_protein.select("name CA"), dtype=int)
        if ca_local.size == 0:
            raise RuntimeError("No CA atoms found in protein-heavy topology.")
        atom_idx_full_single = protein_heavy_idx_full[ca_local]

        core_ph = np.intersect1d(core_aidxs_local, ca_local)
        core_idx_single = map_values_to_slice_indices(core_ph, ca_local)

        if core_idx_single.size < 10:
            raise RuntimeError(
                f"Core too small after mapping to CA space: {core_idx_single.size}. "
                "Try --mode protein-heavy or increase --core-max-frames."
            )

    print(f"[RMSF] Core atoms used for alignment (single selection): {core_idx_single.size}")

    # --- load single replicas ---
    replicas_single: Dict[str, md.Trajectory] = {}
    for rid in rep_ids:
        parts = replica_parts[rid]
        t = md.load([pp.as_posix() for pp in parts], top=top_xtc_full)
        t = t.atom_slice(atom_idx_full_single)

        if t.time is not None and len(t.time):
            print(f"[RMSF] rep{rid} (single): frames={t.n_frames}  t(ns)={t.time[0]/1000:.3f}->{t.time[-1]/1000:.3f}")
        else:
            print(f"[RMSF] rep{rid} (single): frames={t.n_frames}  t=NONE")

        replicas_single[f"rep{rid}"] = t

    # --- compute single outputs ---
    compute_window_outputs(
        outdir=args.outdir,
        windows=windows,
        mode=args.mode,
        per=args.per,
        band=args.band,
        replicas=replicas_single,
        core_idx=core_idx_single,
        title_prefix=f"{args.pdb_code} (all protein)",
    )

    # --- per-chain outputs (optional) ---
    if not args.by_chain:
        return

    # First try connectivity-based molecules
    chain_candidates_full = detect_protein_molecules(top_xtc_full)

    # If that fails but you suspect multiple chains, fall back to CA-gap inference
    if len(chain_candidates_full) < 2:
        print("[chains] find_molecules() could not split protein; trying CA-gap inference fallback...")

        # load a small reference chunk (first part of first selected replica)
        first_part = replica_parts[rep_ids[0]][0]
        tr0_full = md.load(first_part.as_posix(), top=top_xtc_full).atom_slice(protein_heavy_idx_full)

        # make CA-only reference in protein-heavy space
        ca_local_ref = np.asarray(top_xtc_protein.select("name CA"), dtype=int)
        if ca_local_ref.size == 0:
            print("[chains] No CA atoms available for gap inference; skipping per-chain.")
            return

        tr0_ca = tr0_full.atom_slice(ca_local_ref)

        gap_nm = float(args.chain_gap_A) / 10.0  # Å -> nm
        ranges = infer_chain_splits_by_ca_gap(tr0_ca, gap_threshold_nm=gap_nm)
        print(f"[chains] inferred residue ranges (protein-heavy residue indices): {ranges}")

        if len(ranges) >= 2:
            chain_candidates_full = residue_ranges_to_full_atom_indices(
                top_xtc_protein=top_xtc_protein,
                protein_heavy_idx_full=protein_heavy_idx_full,
                residue_ranges=ranges,
                mode=args.mode,
            )
            print(f"[chains] CA-gap split succeeded: sizes={[c.size for c in chain_candidates_full]}")
        else:
            print("[chains] CA-gap inference did not find a strong split; skipping per-chain.")
            return

    if len(chain_candidates_full) < 2:
        print("[chains] Still could not obtain >=2 chains. Skipping per-chain.")
        return

    keep = min(args.max_chains, len(chain_candidates_full))
    print(f"[chains] Computing per-chain RMSF for first {keep} chains (sizes={[c.size for c in chain_candidates_full[:keep]]})")

    # For CA-mode mapping, we need CA indices in protein-heavy local and FULL.
    ca_local_global = None
    ca_full_global = None
    if args.mode == "ca":
        ca_local_global = np.asarray(top_xtc_protein.select("name CA"), dtype=int)
        ca_full_global = np.asarray(protein_heavy_idx_full[ca_local_global], dtype=int)

        # core indices in global CA-sliced space (0..Nca-1)
        core_ph = np.intersect1d(core_aidxs_local, ca_local_global)
        core_idx_global_ca = map_values_to_slice_indices(core_ph, ca_local_global)
    else:
        core_idx_global_ca = None

    for ci in range(keep):
        chain_full = np.asarray(chain_candidates_full[ci], dtype=int)
        chain_dir = os.path.join(args.outdir, f"chain{ci+1}")
        ensure_dir(chain_dir)

        if args.mode == "protein-heavy":
            # chain_full are FULL indices for protein-heavy atoms already (from molecules or from residue ranges)
            atom_idx_full_chain = np.asarray(chain_full, dtype=int)

            # Need core indices in chain-sliced space:
            # core is in protein-heavy local indices; build chain slice values in protein-heavy local indices.
            inv_full_to_ph_local = {int(a): i for i, a in enumerate(np.asarray(protein_heavy_idx_full, dtype=int))}
            chain_local_ph = np.array([inv_full_to_ph_local[int(a)] for a in atom_idx_full_chain if int(a) in inv_full_to_ph_local], dtype=int)

            core_local_chain = np.intersect1d(core_aidxs_local, chain_local_ph)
            core_idx_chain = map_values_to_slice_indices(core_local_chain, chain_local_ph)

        else:
            # CA mode:
            assert ca_full_global is not None and core_idx_global_ca is not None

            # atom_idx_full_chain are FULL indices of CA atoms for this chain
            atom_idx_full_chain = np.sort(chain_full)

            # Which entries in global CA list belong to this chain?
            chain_pos_in_global_ca = np.where(np.isin(ca_full_global, atom_idx_full_chain))[0]

            # Restrict core (in global CA slice) to this chain (still indices in global CA-slice)
            core_in_chain_global = np.intersect1d(core_idx_global_ca, chain_pos_in_global_ca)

            # Convert those to FULL indices and then map into chain slice indices
            core_full_values = ca_full_global[core_in_chain_global]
            core_idx_chain = map_values_to_slice_indices(core_full_values, atom_idx_full_chain)

        if core_idx_chain.size < 10:
            print(f"[chains] chain{ci+1}: core too small ({core_idx_chain.size}); skipping")
            continue

        print(f"[chains] chain{ci+1}: natoms(full)={atom_idx_full_chain.size} core_atoms={core_idx_chain.size}")

        # load chain replicas
        replicas_chain: Dict[str, md.Trajectory] = {}
        for rid in rep_ids:
            parts = replica_parts[rid]
            t_full = md.load([pp.as_posix() for pp in parts], top=top_xtc_full)
            t_chain = t_full.atom_slice(atom_idx_full_chain)
            replicas_chain[f"rep{rid}"] = t_chain

        compute_window_outputs(
            outdir=chain_dir,
            windows=windows,
            mode=args.mode,
            per=args.per,
            band=args.band,
            replicas=replicas_chain,
            core_idx=core_idx_chain,
            title_prefix=f"{args.pdb_code} chain{ci+1}",
        )


if __name__ == "__main__":
    main()
