"""Shared figure style for the eegbits paper figures (matplotlib, PDF + PNG preview).
Palette: reference categorical slots (validated, colour-blind separated); controls in two greys with distinct dashes
and direct labels, so identity never rests on colour alone. Lines 2 px, markers >= 8 px with a surface ring, hairline grid."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SERIES = {"blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a", "yellow": "#eda100", "magenta": "#e87ba4", "green": "#008300", "violet": "#6250d6", "red": "#e34948"}
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#dcdbd6", "#fcfcfb"
GREY1, GREY2 = "#8a8984", "#b5b4ae"
ROLE = {  # fixed assignment, never cycled
    "embedding": SERIES["blue"], "raw": SERIES["orange"], "rrr": SERIES["aqua"], "quant": SERIES["violet"],
    "learning": SERIES["magenta"], "ceiling": SERIES["green"], "randenc": SERIES["yellow"], "trainedenc": SERIES["violet"],
    "noise": GREY1, "trial_shuffled": GREY2, "shuffled_pair": GREY1,
}
DASH = {"noise": (0, (4, 2)), "trial_shuffled": (0, (1, 2)), "shuffled_pair": (0, (6, 2, 1, 2))}
LABEL = {"eeg": "EEG", "noise": "noise input", "trial_shuffled": "trial-shuffled EEG", "shuffled_pair": "shuffled pairs",
         "embedding": "encoder embedding", "raw": "raw 250 Hz EEG", "rrr": "informed bottleneck", "quant": "quantised",
         "randenc": "random-init encoder embedding", "trainedenc": "trained NICE-GA embedding"}


def setup():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.family": "sans-serif", "font.size": 8.5, "axes.labelsize": 8.5, "axes.titlesize": 9.5, "legend.fontsize": 7.5,
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "axes.edgecolor": GRID, "axes.linewidth": 0.8,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "grid.linestyle": "-",
        "axes.spines.top": False, "axes.spines.right": False, "xtick.color": INK2, "ytick.color": INK2,
        "axes.labelcolor": INK, "text.color": INK, "legend.frameon": False, "lines.linewidth": 2.0,
        "lines.markersize": 5.5, "lines.markeredgewidth": 1.2, "lines.markeredgecolor": SURFACE,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def line(ax, x, y, role, label=None, ci=None, marker="o", **kw):
    """One series: 2 px line, ringed markers, optional CI band at 15% alpha. Controls get their dash."""
    c = ROLE[role]
    ls = DASH.get(role, "-")
    ax.plot(x, y, ls=ls, marker=marker if role not in DASH else None, color=c, label=label, zorder=3, **kw)
    if ci is not None:
        lo, hi = ci
        ax.fill_between(x, lo, hi, color=c, alpha=0.15, lw=0, zorder=2)


def end_label(ax, x, y, text, color=INK2, dx=4):
    ax.annotate(text, (x[-1], y[-1]), xytext=(dx, 0), textcoords="offset points", va="center", ha="left", fontsize=7.5, color=color)


def save(fig, stem, figdir):
    fig.savefig(f"{figdir}/{stem}.pdf", bbox_inches="tight")
    fig.savefig(f"{figdir}/{stem}.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
