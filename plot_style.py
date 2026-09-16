"""
Shared matplotlib style for all analysis scripts.
Usage:
    import plot_style
    plot_style.apply()
"""

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ── Colours ──────────────────────────────────────────────────────────────────
SPINE_COLOR   = "#1f77b4"
PANDORA_COLOR = "#ff7f0e"

PARTICLE_COLORS = {
    "e":     "#2ca02c",
    "mu":    "#1f77b4",
    "gamma": "#9467bd",
    "pi0":   "#8c564b",
    "pi":    "#ff7f0e",
    "K":     "#e377c2",
    "p":     "#17becf",
    "n":     "#d62728",
    "other": "#7f7f7f",
}

# ── Standard figure sizes ─────────────────────────────────────────────────────
FIGSIZE_1COL  = (7,  5)
FIGSIZE_2COL  = (12, 5)
FIGSIZE_3X2   = (15, 10)
FIGSIZE_4X2   = (16, 10)

# ── Apply global rcParams ─────────────────────────────────────────────────────
def apply():
    plt.rcParams.update({
        # Spines
        "axes.spines.top":   False,
        "axes.spines.right": False,
        # Grid
        "axes.grid":      True,
        "grid.alpha":     0.3,
        "grid.linewidth": 0.6,
        # Font sizes
        "font.size":        13,
        "axes.labelsize":   13,
        "axes.titlesize":   13,
        "xtick.labelsize":  11,
        "ytick.labelsize":  11,
        # Lines / markers
        "lines.linewidth": 1.8,
        "lines.markersize": 5,
        # Legend
        "legend.fontsize":   10,
        "legend.framealpha": 0.8,
        # Output quality
        "figure.dpi":  150,
        "savefig.dpi": 150,
        "figure.autolayout": False,
    })


# ── Tick formatting helpers ───────────────────────────────────────────────────
def fmt_log_ticks(ax, axis="x"):
    """Apply uniform '0.1'-style labels on a log-scale axis (no trailing zeros)."""
    a = ax.xaxis if axis == "x" else ax.yaxis
    a.set_major_formatter(mticker.StrMethodFormatter("{x:.1f}"))
    a.set_minor_formatter(mticker.NullFormatter())


def clamp_bottom(ax, bottom=0.0):
    """Ensure the y-axis lower bound is never below `bottom` (default 0)."""
    lo, hi = ax.get_ylim()
    if lo < bottom:
        ax.set_ylim(bottom=bottom, top=hi)
