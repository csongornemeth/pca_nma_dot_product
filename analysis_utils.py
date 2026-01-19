# analysis_utils.py
from pathlib import Path
import numpy as np
from io_utils import print_header

def normalize_modes(modes: np.ndarray) -> np.ndarray:
    """Row-wise normalisation of mode vectors, safe for zero vectors."""
    norms = np.linalg.norm(modes, axis=1, keepdims=True)
    norms[norms == 0] = 1  # Prevent division by zero
    return modes / norms


def compute_confusion_matrix(
    nma_modes: np.ndarray,
    pca_modes: np.ndarray,
) -> np.ndarray:
    """
    Compute absolute dot-product similarity matrix between NMA and PCA modes.

    nma_modes : shape (n_modes, 3N)
    pca_modes : shape (n_modes, 3N)

    Returns:
        similarity : shape (n_modes, n_modes)
    """
    print_header("Computing confusion matrix between NMA and PCA modes...")

    # Make sure both have same dimensionality
    D = min(nma_modes.shape[1], pca_modes.shape[1])
    if nma_modes.shape[1] != pca_modes.shape[1]:
        print(
            f"[WARNING] Dimension mismatch: NMA={nma_modes.shape[1]}, "
            f"PCA={pca_modes.shape[1]}. Using D={D}."
        )
        nma_modes = nma_modes[:, :D]
        pca_modes = pca_modes[:, :D]

    print("[CHECK] NMA modes shape:", nma_modes.shape)
    print("[CHECK] PCA modes shape:", pca_modes.shape)

    # Normalise rows
    nma_norm = normalize_modes(nma_modes)
    pca_norm = normalize_modes(pca_modes)

    # Absolute dot product similarity matrix
    similarity = np.abs(np.dot(nma_norm, pca_norm.T))

    print(f"[CONFUSION] Similarity matrix shape: {similarity.shape}")
    print(
        f"[CONFUSION] Max value: {np.max(similarity):.4f}, "
        f"Min value: {np.min(similarity):.4f}"
    )

    return similarity

def rmsip(
    modes_a: np.ndarray,
    modes_b: np.ndarray,
    k: int = 10,
) -> float:
    """
    RMSIP between the first k eigenvectors (subspace similarity).

    modes_a: (n_a, D)
    modes_b: (n_b, D)
    """
    a = normalize_modes(modes_a)
    b = normalize_modes(modes_b)

    # Match dimensionality defensively
    D = min(a.shape[1], b.shape[1])
    if a.shape[1] != b.shape[1]:
        print(
            f"[WARNING] RMSIP dimension mismatch: A={a.shape[1]}, B={b.shape[1]}. Using D={D}."
        )
        a = a[:, :D]
        b = b[:, :D]

    k_eff = min(k, a.shape[0], b.shape[0])
    A = a[:k_eff]
    B = b[:k_eff]

    IP = A @ B.T  # (k_eff, k_eff)
    return float(np.sqrt(np.sum(IP ** 2) / k_eff))


def compute_confusion_matrices_per_replica(
    nma_modes: np.ndarray,
    pca_modes_by_replica: list[np.ndarray],
    k_rmsip: int = 10,
) -> list[dict]:
    """
    Compute NMA vs PCA confusion matrix for each PCA replica separately.

    Returns a list of dicts, each containing:
      - replica: int
      - confusion: np.ndarray (n_nma, n_pca)
      - best_per_nma: np.ndarray (n_nma,)  [max similarity in each NMA row]
      - argbest_per_nma: np.ndarray (n_nma,) [index of best PCA mode per NMA mode]
      - rmsip: float
    """
    results = []
    for r, pca_modes in enumerate(pca_modes_by_replica):
        print_header(f"Replica {r}: NMA vs PCA confusion matrix")

        confusion = compute_confusion_matrix(nma_modes, pca_modes)

        best_per_nma = confusion.max(axis=1)        # (n_nma,)
        argbest_per_nma = confusion.argmax(axis=1)  # (n_nma,)

        r_rmsip = rmsip(nma_modes, pca_modes, k=k_rmsip)

        print(f"[REPLICA {r}] RMSIP (k={min(k_rmsip, nma_modes.shape[0], pca_modes.shape[0])}): {r_rmsip:.3f}")

        results.append({
            "replica": r,
            "confusion": confusion,
            "best_per_nma": best_per_nma,
            "argbest_per_nma": argbest_per_nma,
            "rmsip": r_rmsip,
        })

    return results


def aggregate_best_matches(replica_results: list[dict]) -> dict:
    """
    Aggregate best_per_nma across replicas (mean/std) + RMSIP mean/std.
    """
    best_stack = np.stack([d["best_per_nma"] for d in replica_results], axis=0)  # (n_rep, n_nma)
    rmsips = np.array([d["rmsip"] for d in replica_results], dtype=float)

    best_mean = best_stack.mean(axis=0)
    best_std = best_stack.std(axis=0, ddof=1) if best_stack.shape[0] > 1 else np.zeros_like(best_mean)

    return {
        "best_mean": best_mean,
        "best_std": best_std,
        "rmsip_mean": float(rmsips.mean()),
        "rmsip_std": float(rmsips.std(ddof=1)) if rmsips.size > 1 else 0.0,
    }


def report_pca_variance_thresholds(
    explained_variance_ratio: np.ndarray,
    out_dir: str | Path,
    prefix: str,
    targets=(0.75, 0.80, 0.90, 0.95, 0.99),
    save_plot: bool = True,
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    evr = np.asarray(explained_variance_ratio, dtype=float).ravel()
    if evr.size == 0:
        raise ValueError("Empty explained variance ratio array provided.")

    # normalise
    s = evr.sum()
    if s <= 0:
        raise ValueError("Sum of explained variance ratios is <= 0, cannot normalise.")
    evr = evr / s

    cum = np.cumsum(evr)

    targets = tuple(float(t) for t in targets)
    n_pcs = []
    achieved = []
    for t in targets:
        if not (0.0 < t <= 1.0):
            raise ValueError(f"Target {t} is out of valid range (0, 1].")
        k = int(np.searchsorted(cum, t, side="left") + 1)  # 1-based PC count
        k = min(k, evr.size)
        n_pcs.append(k)
        achieved.append(float(cum[k - 1]))

    #plot

    plot_path = None
    if save_plot:
        fig = plt.figure(figsize=(8, 5))
        plt.plot(np.arange(1, cum.size + 1), cum)

        plt.xticks(np.arange(1, cum.size + 1))  # 👈 ALL integers
        for t in targets:
            plt.axhline(t, linestyle="--")
        plt.xlabel("Number of principal components")
        plt.ylabel("Cumulative explained variance")
        plt.title("PCA cumulative explained variance")
        plt.ylim(0, 1.01)
        plt.xlim(1, cum.size)
        plt.tight_layout()

        plot_path = out_dir / f"{prefix}_pca_explained_variance.png"
        fig.savefig(plot_path, dpi=200)
        plt.close(fig)
    return {
        "targets": targets,
        "n_pcs": tuple(n_pcs),
        "achieved": tuple(achieved),
        "cumulative": cum,
        "plot": plot_path,
    }

def plot_nma_pca_stacked_bars(
    confusion: np.ndarray,
    kept_pca_idx,
    nma_start: int = 7,
    nma_end: int | None = None,
    title: str = "NMA vs PCA mode overlap",
    outfile: str | Path | None = None,
    dpi: int = 200,
) -> None:
    """
    Stacked bar chart where the stack height indicates how well the subspace
    spanned by the selected PCA modes describes each NMA mode.
    """

    confusion_m = np.asarray(confusion)
    M, P = confusion_m.shape

    kept = list(kept_pca_idx)
    if len(kept) == 0:
        raise ValueError("kept_pca_idx is empty.")

    # PCA indexing: allow 0-based or 1-based inputs
    zero_based = any(k == 0 for k in kept)
    kept_cols = kept if zero_based else [k - 1 for k in kept]

    if min(kept_cols) < 0 or max(kept_cols) >= P:
        raise ValueError(f"kept_pca_idx out of range for P={P} PCA modes.")

    # NMA range (inputs are 1-based)
    if nma_end is None:
        nma_end = M
    if nma_start < 1 or nma_end > M or nma_start > nma_end:
        raise ValueError(f"Invalid NMA range: {nma_start}..{nma_end} for M={M}")

    nma_rows = np.arange(nma_start - 1, nma_end)   # 0-based rows
    x_labels = np.arange(nma_start, nma_end + 1)   # display as 1-based

    # submatrix: selected NMAs x kept PCAs
    D_sub = confusion_m[np.ix_(nma_rows, kept_cols)]

    fig, ax = plt.subplots(figsize=(max(8, 0.35 * len(x_labels)), 6))
    bottom = np.zeros(len(x_labels), dtype=float)

    for j, col in enumerate(kept_cols):
        heights = D_sub[:, j]
        ax.bar(x_labels, heights, bottom=bottom, label=f"PC{col + 1}")
        bottom += heights

    ax.set_xlabel("NMA mode")
    ax.set_ylabel("Sum of |dot(NMA, PCA)| (stacked by PCA)")
    ax.set_title(title)
    ax.set_xticks(x_labels)
    ax.margins(x=0.01)

    if len(kept_cols) <= 10:
        ax.legend(loc="upper right", frameon=True)
    else:
        ax.legend(loc="upper left", frameon=True, ncol=2, fontsize=8)

    fig.tight_layout()

    if outfile is not None:
        outfile = Path(outfile)
        fig.savefig(outfile, dpi=dpi)
        plt.close(fig)
    
def compute_nma_subspace_capture(
    confusion: np.ndarray,
    kept_pca_idx,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the subspace coverage of each NMA mode by the selected PCA modes.

    capture[m] = Σ_p (u_m · v_p)^2  (bounded ~[0,1] if PCA modes are orthonormal)
    projection[m] = sqrt(capture[m])
    """
    confusion = np.asarray(confusion)

    kept = list(kept_pca_idx)
    if len(kept) == 0:
        raise ValueError("kept_pca_idx is empty.")

    zero_based = any(k == 0 for k in kept)
    kept_cols = kept if zero_based else [k - 1 for k in kept]

    n_nma, n_pca = confusion.shape
    if min(kept_cols) < 0 or max(kept_cols) >= n_pca:
        raise ValueError(f"kept_pca_idx out of range for n_pca={n_pca}.")

    sub = confusion[:, kept_cols]             # (n_nma, n_kept)
    capture = np.sum(sub ** 2, axis=1)        # (n_nma,)
    projection = np.sqrt(capture)

    return capture, projection


def plot_nma_pca_subspace_overlap(
    capture: np.ndarray,
    nma_start: int = 7,
    nma_end: int | None = None,
    title: str = "NMA subspace capture by selected PCA modes",
    outfile: str | Path | None = None,
    dpi: int = 200,
) -> None:
    """
    Bar plot of subspace capture per NMA mode.
    """
    capture = np.asarray(capture)

    if nma_end is None:
        nma_end = capture.size

    y = capture[nma_start - 1 : nma_end]
    x = np.arange(nma_start, nma_end + 1)

    fig, ax = plt.subplots(figsize=(max(8, 0.4 * len(x)), 4))
    ax.bar(x, y)

    y_max = max(1.0, float(np.max(y)) * 1.1)

    ax.set_xlabel("NMA mode")
    ax.set_ylabel("Subspace capture Σ(dot²)")
    ax.set_ylim(0, y_max)
    ax.set_title(title)
    ax.set_xticks(x)

    fig.tight_layout()

    if outfile is not None:
        outfile = Path(outfile)
        fig.savefig(outfile, dpi=dpi)
        plt.close(fig)
        print(f"[PLOT] NMA subspace capture plot saved to: {outfile}")
    else:
        plt.show()
