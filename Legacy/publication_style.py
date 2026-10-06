"""Shared publication style. Does not alter figure layout or data representation."""
import matplotlib.pyplot as plt

OBS_COLOR = "#222222"
MODEL_COLOR = "#0072B2"       # consistent model blue
SEASONAL_COLOR = "#009E73"    # consistent seasonal baseline green
CLIM_COLOR = "#777777"
POS_COLOR = "#D55E00"
NEG_COLOR = "#0072B2"
MAJOR_COLOR = "#D55E00"
HURRICANE_SHADE = "#F3E6BC"

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
    "full": MODEL_COLOR,
}

def apply_publication_style():
    """Only typography/line styling. No subplot geometry is changed."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
        "font.size": 8.0,
        "axes.titlesize": 8.5,
        "axes.labelsize": 8.0,
        "legend.fontsize": 7.0,
        "xtick.labelsize": 7.0,
        "ytick.labelsize": 7.0,
        "axes.linewidth": 0.7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.45,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
