"""One consistent visual theme for every static chart (matplotlib) and the
Plotly dashboard."""
from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt

# palette: deep navy ink, one sand/gold accent (Dubai), muted greys
INK = "#1B2A41"
ACCENT = "#C8963E"      # highlight / the model
ACCENT_2 = "#2E7D7A"    # second series (e.g. Ready vs Off-Plan)
MUTED = "#A7B0BE"       # baselines / context
GRID = "#E6E9EF"
BG = "#FFFFFF"
GOOD = "#2E7D32"
BAD = "#B23A48"

SERIES = {"Off-Plan": ACCENT, "Ready": ACCENT_2}


def apply():
    mpl.rcParams.update({
        "figure.figsize": (10, 5.6),
        "figure.dpi": 110,
        "savefig.dpi": 160,
        "savefig.bbox": "tight",
        "figure.facecolor": BG,
        "axes.facecolor": BG,
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK,
        "axes.titlecolor": INK,
        "axes.titlesize": 15,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.titlepad": 26,
        "axes.labelsize": 11,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": INK,
        "ytick.color": INK,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.frameon": False,
        "font.family": "DejaVu Sans",
        "text.color": INK,
    })


def title(ax, headline: str, sub: str | None = None):
    """Headline = the takeaway; sub = what is plotted."""
    ax.set_title(headline)
    if sub:
        ax.text(0, 1.02, sub, transform=ax.transAxes, fontsize=10.5,
                color="#5B6676", va="bottom", ha="left")


def source(fig, text="Source: Dubai Land Department open data via dubairealestatedata.com (CC BY 4.0)"):
    fig.text(0.01, -0.02, text, fontsize=8, color="#7A8494", ha="left")


def save(fig, path):
    fig.savefig(path)
    plt.close(fig)


PLOTLY_LAYOUT = dict(
    font=dict(family="DejaVu Sans, Arial, sans-serif", color=INK, size=13),
    paper_bgcolor=BG, plot_bgcolor=BG,
    xaxis=dict(gridcolor=GRID, zeroline=False),
    yaxis=dict(gridcolor=GRID, zeroline=False),
    margin=dict(l=60, r=30, t=80, b=50),
    colorway=[ACCENT, ACCENT_2, INK, MUTED, BAD, GOOD],
)
