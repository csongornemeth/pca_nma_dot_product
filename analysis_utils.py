# analysis_utils.py
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

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

def plot_confusion_matrix(
    confusion_matrix: np.ndarray,
    out_path: Path,
    pdb_code: str,
) -> None:
    print_header("Plotting confusion matrix heatmap...")

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(confusion_matrix, cmap="viridis", aspect="auto")

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Absolute Dot Product Similarity")

    ax.set_title(f"Confusion Matrix: NMA vs PCA Modes for {pdb_code.upper()}")
    ax.set_xlabel("PCA Modes")
    ax.set_ylabel("NMA Modes")

    n_nma, n_pca = confusion_matrix.shape

    ax.set_xticks(np.arange(n_pca))
    ax.set_xticklabels(np.arange(1, n_pca + 1))
    ax.set_yticks(np.arange(n_nma))
    ax.set_yticklabels(np.arange(1, n_nma + 1))

    # Optional: keep them horizontal
    plt.setp(ax.get_xticklabels(), rotation=0, ha="center")

    # --- Add numeric labels in each cell ---
    for i in range(n_nma):
        for j in range(n_pca):
            val = confusion_matrix[i, j]
            # pick text colour based on brightness for readability
            text_color = "white" if val > 0.5 else "black"
            ax.text(
                j,
                i,
                f"{val:.2f}",
                ha="center",
                va="center",
                fontsize=6,
                color=text_color,
            )

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)

    print(f"[PLOT] Confusion matrix saved to: {out_path}")

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

def plot_best_match_barplot(
    best_per_nma: np.ndarray,
    argbest_per_nma: np.ndarray,
    out_path: Path,
    title: str,
) -> None:
    """
    Bar plot of best PCA match per NMA mode,
    annotated with the PCA mode index.
    """
    print_header("Plotting best-match bar plot...")

    n_modes = best_per_nma.shape[0]
    x = np.arange(1, n_modes + 1)

    fig, ax = plt.subplots(figsize=(9, 4))
    bars = ax.bar(x, best_per_nma)

    ax.set_xlabel("NMA mode index")
    ax.set_ylabel("Best |dot(NMA, PCA)|")
    ax.set_ylim(0, 1.0)
    ax.set_title(title)
    ax.set_xticks(x)

    # Annotate PCA index above each bar
    for i, bar in enumerate(bars):
        pca_idx = argbest_per_nma[i] + 1  # +1 for human-readable indexing
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"PC{pca_idx}",
            ha="center",
            va="bottom",
            fontsize=8,
            rotation=90,
        )

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)

    print(f"[PLOT] Best-match bar plot saved to: {out_path}")

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
        print(f"[PLOT] Stacked NMA–PCA bar chart saved to: {outfile}")
    else:
        plt.show()


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
    
    return {
        "targets": targets,
        "n_pcs": tuple(n_pcs),
        "achieved": tuple(achieved),
        "cumulative": cum,
        "plot": plot_path,
    }
