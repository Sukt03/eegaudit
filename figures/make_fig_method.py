"""fig_method.pdf: one procedure, four operationalisations (methodology by modality). Full text width (7.0 in).
A schematic: it carries no measured value. Its words follow paper/v2/tables/tab_data.tex, tab_d_modality.tex and
tab_exposure_main.tex; the only numerals are protocol constants (200 candidate images, 4 classes, dataset identifiers).
Run: srun -n1 -t 5 python -I -u make_fig_method.py"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import *  # noqa: F401,F403
from matplotlib.patches import Rectangle, FancyBboxPatch, Circle

plt = style()
# step label, symbol, research question it serves, one entry per modality (image, text, prediction error, motor imagery)
STEPS = [
    ("Data", "", "", ["THINGS-EEG2", "ZuCo (SR, NR2, NR)", "ds003458, ds004595,\nds007647", "EEGMMIDB"]),
    ("Target", "$T$", "", ["image among 200\n(NICE-GA: concept)", "sentence in the test set", "residualised\nprediction error", "cue class among 4"]),
    ("Decoders", "$q_\\theta$", "", ["UBP, NICE-GA, BrainFLORA", "BrainTranslator,\nDeWave-style", "EEGNet, raw-ERP ridge", "ridge, EEGNet,\nfoundation models"]),
    ("Scorer, unit", "$Y$", "", ["InfoNCE, bits per query", "InfoNCE, bits per record", "Gaussian log-density\ngain, bits per trial", "class log-score gain,\nbits per trial"]),
    ("Blank yield", "$B$", "", ["noise input,\nshuffled pairs", "noise-trained control,\nshuffled pairs", "circular shift,\nphase and shuffle refits", "noise input,\nphase and shuffle refits"]),
    ("Replicate yield", "$R$", "", ["repetition replicate\n(split-half, matched $k$)", "reader replicate:\nanchor check fails", "read-out replicate\n(two fitted read-outs)", "same-session retest only;\nfails as anchor"]),
    ("Nuisance", "$Z$", "", ["image category", "task identity, sentence\nlength, gaze", "eye movements,\noutcome and choice", "proxy regression\n(raw read-out only)"]),
    ("Exposure", "", "", [["L", "U", "R"], ["L", "T", "V"], ["B", "U", "L", "T"], ["V", "U", "T"]]),
]
EXPO = {"T": ("white", INK, "-"), "V": ("white", INK, "-"), "B": ("#C8C8C8", INK, "-"), "U": ("#8C8C8C", "white", "-"),
        "L": ("#262626", "white", "-"), "R": ("white", MUTE, (0, (2, 1.2)))}   # same shades as fig_exposure.pdf

W = 7.0
XS = 1.56                    # spine width
CW = (W - XS - 0.02) / 4
LH = 0.112                   # line height of 7 pt text (inches)
PAD = 0.042
HEADH, FOOT = 0.24, 0.27
heights = []
for _, _, _, ent in STEPS:
    nl = 1 if isinstance(ent[0], list) else max(e.count("\n") + 1 for e in ent)
    heights.append(nl * LH + 2 * PAD)
H = HEADH + sum(heights) + FOOT + 0.06
fig = plt.figure(figsize=(W, H))
bg = fig.add_axes([0, 0, 1, 1]); bg.set_xlim(0, W); bg.set_ylim(0, H); bg.axis("off")

ytop = H - 0.02
ybody = ytop - HEADH
ybot = ybody - sum(heights)
# modality columns: tinted card + header chip
for j, t in enumerate(ORDER):
    x = XS + j * CW
    bg.add_patch(FancyBboxPatch((x + 0.03, ybot - 0.02), CW - 0.06, ytop - ybot + 0.02, boxstyle="round,pad=0,rounding_size=0.05", fc=TINT[t], ec="none"))
    bg.add_patch(FancyBboxPatch((x + 0.03, ybody + 0.03), CW - 0.06, HEADH - 0.03, boxstyle="round,pad=0,rounding_size=0.05", fc=HUE[t], ec="none"))
    bg.text(x + CW / 2, ybody + 0.03 + (HEADH - 0.03) / 2, NAME[t], color="white", fontweight="bold", fontsize=FS + 0.5, ha="center", va="center")
bg.text(0.02, ybody + 0.03 + (HEADH - 0.03) / 2, "One procedure", fontweight="bold", fontsize=FS + 0.5, va="center")

# spine: numbered steps joined by a line, arrow at the bottom
XC = 0.12
y = ybody
centres = []
for i, ((lab, sym, rq, ent), h) in enumerate(zip(STEPS, heights)):
    yc = y - h / 2
    centres.append(yc)
    if i > 0:
        bg.plot([XS - 0.02, W - 0.02], [y, y], color="white", lw=1.0, zorder=2)
    bg.text(XC + 0.16, yc, lab, fontsize=FS, fontweight="bold", va="center")
    xr = XS - 0.06
    if rq:
        bg.text(xr, yc, rq, fontsize=FS, color=MUTE, va="center", ha="right"); xr -= tw(fig, rq) + 0.07
    if sym:
        bg.text(xr, yc, sym, fontsize=FS + 0.5, va="center", ha="right")
    for j, t in enumerate(ORDER):
        xc = XS + j * CW + CW / 2
        e = ent[j]
        if isinstance(e, list):                      # exposure letters as small boxes (shade + letter, R dashed)
            bw, gap = 0.17, 0.05
            x0 = xc - (len(e) * bw + (len(e) - 1) * gap) / 2
            for k, c in enumerate(e):
                fc, tc, ls = EXPO[c]
                bg.add_patch(Rectangle((x0 + k * (bw + gap), yc - 0.075), bw, 0.15, fc=fc, ec=INK, lw=0.6, ls=ls, zorder=3))
                bg.text(x0 + k * (bw + gap) + bw / 2, yc - 0.004, c, fontsize=FS, color=tc, ha="center", va="center", zorder=4)
        else:
            bg.text(xc, yc, e, fontsize=FS, ha="center", va="center", linespacing=1.12, zorder=3)
    y -= h
yg = ybot - 0.06 - (FOOT - 0.06) / 2
centres.append(yg)
bg.plot([XC, XC], [centres[0], centres[-1] + 0.11], color=MUTE, lw=0.8, zorder=1)
bg.annotate("", xy=(XC, centres[-1] + 0.075), xytext=(XC, centres[-1] + 0.16), arrowprops=dict(arrowstyle="-|>", color=MUTE, lw=0.8, mutation_scale=7), zorder=1)
for i, yc in enumerate(centres[:-1]):
    bg.add_patch(Circle((XC, yc), 0.075, fc="white", ec=MUTE, lw=0.7, zorder=3))
    bg.text(XC, yc - 0.004, str(i + 1), fontsize=FS, ha="center", va="center", zorder=4)
# last step: grading, shared by all four targets
bg.add_patch(Circle((XC, yg), 0.075, fc=INK, ec=INK, lw=0.7, zorder=3))
bg.text(XC, yg - 0.004, str(len(STEPS) + 1), fontsize=FS, color="white", ha="center", va="center", zorder=4)
bg.text(XC + 0.16, yg, "Grade", fontsize=FS, fontweight="bold", va="center")
bg.add_patch(FancyBboxPatch((XS + 0.03, yg - 0.105), 4 * CW - 0.06, 0.21, boxstyle="round,pad=0,rounding_size=0.05", fc="white", ec=INK, lw=0.7))
bg.text(XS + 0.14, yg, "the same ten diagnostics D0–D9, one grade per cell and target, never summed:", fontsize=FS, va="center")
x = XS + 0.14 + tw(fig, "the same ten diagnostics D0–D9, one grade per cell and target, never summed:") + 0.10
for g in ("Pass", "Partial", "Fail", "undefined"):
    lab, col, s = GRADE[g]
    bg.add_patch(Rectangle((x, yg - 0.07), 0.16, 0.14, fc=col, ec="none"))
    bg.text(x + 0.08, yg - 0.004, s, ha="center", va="center", fontsize=8)
    x += 0.16 + 0.05
nbad = check_text(fig)
save(fig, "fig_method")
sys.exit(1 if nbad else 0)
