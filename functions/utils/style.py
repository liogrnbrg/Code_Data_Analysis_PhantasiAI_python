from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def pretty_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.tick_params(direction="out", width=1, labelsize=11)

    ax.grid(True, alpha=0.18)

    for spine in ax.spines.values():
        spine.set_linewidth(1)


def save_pretty_fig(fig, filename, plots_dir, dpi=300):
    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    fig.patch.set_facecolor("white")
    fig.savefig(
        plots_dir / filename,
        dpi=dpi,
        bbox_inches="tight",
        facecolor="white",
    )

    plt.show()

def get_robust_ylims(y, lower_pct=2.5, upper_pct=97.5, pad_fraction=0.10):
    y = np.asarray(y, dtype=float)
    y = y[np.isfinite(y)]

    if y.size == 0:
        return (-1, 1)

    y_low = np.percentile(y, lower_pct)
    y_high = np.percentile(y, upper_pct)

    if y_low == y_high:
        pad = max(abs(y_low) * pad_fraction, 1)
    else:
        pad = pad_fraction * (y_high - y_low)

    return (y_low - pad, y_high + pad)