"""
Shared plotting style for publication figures.

Designed around Elsevier artwork guidance:
- uniform lettering and sizing
- Arial/Helvetica-family sans serif
- approximately 7--8 pt lettering at final two-column size
- color-blind-conscious, high-contrast palette
- vector EPS plus high-resolution PNG output
"""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

# ---------------------------------------------------------------------
# Figure geometry
# ---------------------------------------------------------------------
DOUBLE_COLUMN_WIDTH = 7.20   # inches, ~183 mm
SINGLE_COLUMN_WIDTH = 3.50   # inches, ~89 mm

# ---------------------------------------------------------------------
# Consistent colors used across all manuscript figures
# ---------------------------------------------------------------------
OBS_COLOR = "#222222"
MODEL_COLOR = "#0072B2"      # Okabe-Ito blue
SEASONAL_COLOR = "#009E73"   # Okabe-Ito bluish green
CLIM_COLOR = "#7A7A7A"
MAJOR_COLOR = "#D55E00"      # vermillion
MODERATE_COLOR = "#E69F00"   # orange
MINOR_COLOR = "#56B4E9"      # sky blue
HURRICANE_SHADE = "#F2EFE6"
GRID_COLOR = "#D8D8D8"
ONE_TO_ONE_COLOR = "#555555"

FEATURE_COLORS = {
    "minimal": "#B5B5B5",
    "base": "#666666",
    "static": "#E69F00",
    "antecedent": "#D55E00",
    "forward": "#009E73",
    "forward_plus": "#56B4E9",
    "relative": "#CC79A7",
    "forward_static": "#117733",
    "antecedent_static": "#332288",
    "full": "#0072B2",
}

FEATURE_MARKERS = {
    "minimal": "o",
    "base": "s",
    "static": "^",
    "antecedent": "v",
    "forward": "D",
    "forward_plus": "P",
    "relative": "X",
    "forward_static": "d",
    "antecedent_static": "<",
    "full": "*",
}

def _pick_font():
    """Use an Elsevier-recommended sans-serif font if available."""
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in ("Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"):
        if name in available:
            return name
    return "sans-serif"

FONT_FAMILY = _pick_font()

def apply_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": [FONT_FAMILY, "Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
        "font.size": 7.5,
        "axes.titlesize": 8.0,
        "axes.labelsize": 7.5,
        "legend.fontsize": 6.8,
        "xtick.labelsize": 6.8,
        "ytick.labelsize": 6.8,
        "axes.linewidth": 0.7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.axisbelow": True,
        "axes.grid": False,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "lines.linewidth": 1.3,
        "lines.markersize": 4.0,
        "legend.frameon": False,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

def clean_axis(ax, grid_axis="y"):
    ax.grid(True, axis=grid_axis, color=GRID_COLOR, linewidth=0.45)
    ax.tick_params(pad=2)

def panel_label(ax, letter, title=None, x=0.0, y=1.03):
    """Consistent lower-case panel lettering: (a), (b), ..."""
    label = f"({letter})"
    if title:
        label += f" {title}"
    ax.text(
        x, y, label,
        transform=ax.transAxes,
        ha="left", va="bottom",
        fontsize=8.2, fontweight="bold",
        clip_on=False,
    )

def save_figure(fig, outdir, stem, png_dpi=600):
    """
    Save every figure in the two requested formats.

    EPS is the submission-quality vector file.
    PNG is retained as a high-resolution preview/drafting file.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    png = outdir / f"{stem}.png"
    eps = outdir / f"{stem}.eps"
    fig.savefig(png, dpi=png_dpi, bbox_inches="tight", facecolor="white")
    fig.savefig(eps, format="eps", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  -> {png}")
    print(f"  -> {eps}")
