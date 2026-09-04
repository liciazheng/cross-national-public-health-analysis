"""
Shared chart styling for analysis.py and panel_analysis.py.

One palette, one set of chrome rules, so the two analyses read as one project.
Colours are assigned by entity and never reordered: a country keeps its hue in
every figure it appears in.
"""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

# Categorical slots 1-3. Validated as a set for all-pairs separation, including
# under colour-vision deficiency, so they are safe in scatters as well as lines.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]  # blue, orange, aqua

FOCUS = {
    "South Africa": SERIES[0],
    "Kenya": SERIES[1],
    "Botswana": SERIES[2],
}

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"

# The recessive grey used for the un-highlighted mass in an emphasis chart.
CONTEXT = "#c9c8c2"

# Status, not a series. Reserved for "this moved the wrong way" and always
# shipped with a label, so the meaning never rests on the colour alone. Kept
# distinct from the categorical slots so it cannot be mistaken for a country.
WORSE = "#d03b3b"


def style():
    plt.rcParams.update({
        "font.family": ["Segoe UI", "DejaVu Sans", "sans-serif"],
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "text.color": INK,
        "axes.labelcolor": INK_SECONDARY,
        "xtick.color": INK_MUTED,
        "ytick.color": INK_MUTED,
        "axes.titlesize": 14,
        "axes.titleweight": "bold",
        "axes.titlecolor": INK,
        "axes.titlelocation": "left",
        "axes.titlepad": 14,
        "font.size": 10,
        "figure.dpi": 140,
    })


def frame(ax, axis="y"):
    """Hairline, solid, recessive chrome. No dashes anywhere."""
    ax.grid(axis=axis, color=GRIDLINE, linewidth=0.8, solid_capstyle="butt")
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BASELINE)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(length=0)


def titles(ax, title, subtitle=None):
    """Title above subtitle, both flush left, with room for neither to collide."""
    ax.set_title(title, pad=34)
    if subtitle:
        ax.text(0, 1.015, subtitle, transform=ax.transAxes, fontsize=10,
                color=INK_SECONDARY, va="bottom")


def save(fig, path):
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path.name
