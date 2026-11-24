"""Shared style for the paper/v2 figures (fig_diagnostic, fig_method).
Numbers are read only through paper/figs/results_io.py (load / load_file, gated by CITABLE); nothing is typed by hand.
Run every script through the compute node:
  srun -n1 -t 5 python -I -u <script>
"""
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("/path/to/eegaudit")
sys.path.insert(0, str(ROOT / "paper" / "figs"))
import results_io as rio  # noqa: E402

FIGS = ROOT / "paper" / "v2" / "figs"
PREV = FIGS / "preview"
TABLES = ROOT / "paper" / "v2" / "tables"

# modality hues (colour-blind-safe, fixed per modality in every figure)
HUE = {"image": "#0072B2", "text": "#D55E00", "prediction_error": "#009E73", "motor_imagery": "#C2478F"}
TINT = {"image": "#D6E8F3", "text": "#F8E0D0", "prediction_error": "#D3EEE6", "motor_imagery": "#F5DDEA"}
NAME = {"image": "Image", "text": "Text", "prediction_error": "Prediction error", "motor_imagery": "Motor imagery"}
ORDER = ["image", "text", "prediction_error", "motor_imagery"]
INK, MUTE, GRID, RAIL = "#222222", "#5F5F5F", "#DDDDDD", "#ECEFF2"
REPL = "#E69F00"   # replicate-yield diamond
# grade colours match fig_grading.pdf; every grade also has its own symbol
GRADE = {"Pass": ("supported", "#56B4E9", "✓"), "Partial": ("qualified", "#F0E442", "◐"),
         "Fail": ("violated", "#D55E00", "✗"), "undefined": ("undefined", "#E6E6E6", "–")}
FS = 7


def style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": FS, "axes.labelsize": FS, "axes.titlesize": FS + 0.5,
        "xtick.labelsize": FS, "ytick.labelsize": FS, "legend.fontsize": FS,
        "axes.linewidth": 0.6, "xtick.major.width": 0.6, "xtick.major.size": 2.0, "xtick.major.pad": 1.5,
        "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
        "pdf.fonttype": 42, "ps.fonttype": 42, "hatch.linewidth": 0.7,
        "axes.edgecolor": INK, "text.color": INK, "axes.labelcolor": INK, "xtick.color": INK,
    })
    return plt


def outline(pdf):
    """AAAI: no CID / Identity-H fonts inside graphics. pdf.fonttype 42 (set in style()) embeds DejaVu as CID TrueType, so every
    saved PDF gets its text converted to outlines with ghostscript (as src/outline_fonts.sh does for existing files)."""
    import shutil, subprocess
    gs = shutil.which("gs")
    if gs is None:
        print("WARNING: gs not found; run src/outline_fonts.sh on a compute node before using", pdf); return
    tmp = str(pdf) + ".outlined"
    subprocess.run([gs, "-q", "-o", tmp, "-sDEVICE=pdfwrite", "-dNoOutputFonts", "-dCompatibilityLevel=1.5", str(pdf)], check=True)
    shutil.move(tmp, str(pdf))


def save(fig, name):
    PREV.mkdir(exist_ok=True)
    fig.savefig(FIGS / f"{name}.pdf")
    outline(FIGS / f"{name}.pdf")
    fig.savefig(PREV / f"{name}.png", dpi=220)
    plt.close(fig)
    print("wrote", FIGS / f"{name}.pdf")


def tw(fig, s, fontsize=FS, **kw):
    """Rendered width of a string in inches."""
    t = fig.text(0, 0, s, fontsize=fontsize, **kw)
    w = t.get_window_extent(fig.canvas.get_renderer()).width / fig.dpi
    t.remove()
    return w


def check_text(fig, min_pt=7.0):
    """Layout gate: every text is >= min_pt, inside the canvas, and no two texts overlap. Prints offenders and returns their count."""
    r = fig.canvas.get_renderer()
    W, H = fig.get_size_inches() * fig.dpi
    items = []
    for ax in fig.axes:
        for t in ax.texts + (ax.get_xticklabels() if ax.axison else []):
            if t.get_text().strip() and t.get_visible():
                items.append((t, t.get_window_extent(r)))
    bad = 0
    for t, b in items:
        if t.get_fontsize() < min_pt - 1e-6:
            print("SMALL TEXT", t.get_fontsize(), repr(t.get_text())); bad += 1
        if b.x0 < -0.5 or b.y0 < -0.5 or b.x1 > W + 0.5 or b.y1 > H + 0.5:
            print("OUTSIDE CANVAS", repr(t.get_text())); bad += 1
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            a, b = items[i][1], items[j][1]
            if min(a.x1, b.x1) - max(a.x0, b.x0) > 1.0 and min(a.y1, b.y1) - max(a.y0, b.y0) > 1.0:
                print("TEXT OVERLAP", repr(items[i][0].get_text()), "|", repr(items[j][0].get_text())); bad += 1
    print("layout check:", "clean" if bad == 0 else f"{bad} problem(s)")
    return bad


# Text rows, main-text table and figure alike: which control is shown as "without EEG" (the blank).
#   "shuffled"      -> without EEG = shuffled pairs; baseline = noise-trained control   (first spec, 23 Nov; superseded)
#   "noise_trained" -> without EEG = noise-trained control; baseline = random-init decoder (3 seeds), as tables/tab_profile.tex and the prose (coordinator decision, 23 Nov)
TEXT_BLANK = "noise_trained"
