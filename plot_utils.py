# plot_utils.py
from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from io_utils import print_header


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

    plt.setp(ax.get_xticklabels(), rotation=0, ha="center")

    # Numeric labels in each cell
    for i in range(n_nma):
        for j in range(n_pca):
            val = confusion_matrix[i, j]
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


def plot_best_match_barplot(
    best_per_nma: np.ndarray,
    argbest_per_nma: np.ndarray,
    out_path: Path,
    title: str,
) -> None:
    """
    Bar plot of best PCA match per NMA mode, annotated with the PCA mode index.
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
