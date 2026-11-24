"""fig_method_flow.pdf: the EEGauge methodology as a left-to-right flow (full text width, 7.0 in; height not above fig_method.pdf).
Plain boxes and labelled arrows. Solid boxes are the real run; dashed boxes are the controls that replace the EEG or check the scorer.
The same decoder is scored with the EEG and with the EEG removed; the difference is the yield. A replicate arm gives the usage and a
nuisance arm the nuisance share; all three go to the diagnostic protocol. A strip under the flow names, in words, what plays each role
for the four targets. Layout after the author's reference page (eegauge_method_map.html).
A schematic: labels only, no measured value and no typed number. Run:
  srun -n1 -t 5 python -I -u -B make_fig_method_flow.py"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import *  # noqa: F401,F403
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

plt = style()
W, H = 7.0, 2.90
fig = plt.figure(figsize=(W, H))
bg = fig.add_axes([0, 0, 1, 1]); bg.set_xlim(0, W); bg.set_ylim(0, H); bg.axis("off")
T = lambda s, **kw: tw(fig, s, **kw)
LH = 0.124                       # line height of 7 pt text (inches)
DASH = (0, (3.2, 2.0))


def box(x, yc, w, h, head, body, dashed=False, heavy=False):
    """Rounded plain box centred on yc, bold head lines and plain body lines; asserts that every line fits."""
    bg.add_patch(FancyBboxPatch((x, yc - h / 2), w, h, boxstyle="round,pad=0,rounding_size=0.06", fc="white", ec=INK,
                                lw=1.5 if heavy else 0.9, ls=DASH if dashed else "-", zorder=2))
    head = [head] if isinstance(head, str) else head
    rows = [(s, dict(fontsize=FS + 0.5, fontweight="bold")) for s in head] + [(s, dict(fontsize=FS)) for s in body]
    y = yc + (len(rows) - 1) * LH / 2
    for i, (s, kw) in enumerate(rows):
        assert T(s, **kw) <= w - 0.08, f"line too wide for its box ({T(s, **kw):.2f} > {w - 0.08:.2f}): {s!r}"
        bg.text(x + w / 2, y - i * LH - 0.003, s, ha="center", va="center", zorder=4, **kw)


def arrow(p0, p1, dashed=False):
    bg.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=7, lw=0.9, color=INK, ls=DASH if dashed else "-",
                                 shrinkA=0, shrinkB=0, zorder=3))


def label(x, y, s, ha="left"):
    bg.text(x, y, s, fontsize=FS, color=MUTE, ha=ha, va="center", zorder=4)


# ------------------------------------------------------------------ geometry (inches)
XA, WA = 0.05, 1.08              # EEG trials
XB, WB = 1.38, 1.20              # decoder, EEG removed
XC, WC = 2.83, 1.49              # scores, replicate arm, nuisance arm
XD, WD = 4.57, 1.30              # usage, yield, nuisance share
XE, WE = 6.12, 0.82              # diagnostic protocol
Y1, Y2, Y3, Y4 = 2.67, 2.08, 1.49, 0.90      # replicate row, real run, control row, nuisance and calibration row
BH = 0.40
YY = (Y2 + Y3) / 2; HY = 0.76    # yield box, between the real run and the control row

# ------------------------------------------------------------------ the real run
box(XA, Y2, WA, BH, "EEG trials", ["held-out test split"])
box(XB, Y2, WB, BH, "Decoder", ["released or retrained"])
box(XC, Y2, WC, BH, "Score with EEG", ["log score, bits"])
arrow((XA + WA, Y2), (XB, Y2)); arrow((XB + WB, Y2), (XC, Y2))

# ------------------------------------------------------------------ the control: the same decoder with the EEG removed
box(XB, Y3, WB, BH, "EEG removed", ["same decoder, no EEG"], dashed=True)
box(XC, Y3, WC, BH, "Score without EEG", ["noise, shuffle, noise-trained"], dashed=True)
xm = XA + WA / 2
bg.plot([xm, xm, XB - 0.07], [Y2 - BH / 2, Y3, Y3], color=INK, lw=0.9, ls=DASH, zorder=3, solid_capstyle="butt")
arrow((XB - 0.08, Y3), (XB, Y3))
label(xm + 0.07, Y3 + 0.115, "blank input")
arrow((XB + WB, Y3), (XC, Y3), dashed=True)

# ------------------------------------------------------------------ yield
box(XD, YY, WD, HY, "Yield", ["with − without", "chance and baseline", "on the same table"], heavy=True)
arrow((XC + WC, Y2 - 0.02), (XD, YY + 0.17)); arrow((XC + WC, Y3 + 0.02), (XD, YY - 0.17))


# ------------------------------------------------------------------ replicate arm -> usage
box(XC, Y1, WC, BH, "Replicate arm", ["split-half · two read-outs"])
box(XD, Y1, WD, BH, "Usage", ["yield ÷ replicate yield"])
arrow((XC + WC, Y1), (XD, Y1)); arrow((XD + WD / 2, YY + HY / 2), (XD + WD / 2, Y1 - BH / 2))

# ------------------------------------------------------------------ nuisance arm -> nuisance share
box(XC, Y4, WC, BH, "Nuisance arm", ["category · task · eye proxies"])
box(XD, Y4, WD, BH, "Nuisance share", ["yield lost when removed"])
arrow((XC + WC, Y4), (XD, Y4)); arrow((XD + WD / 2, YY - HY / 2), (XD + WD / 2, Y4 + BH / 2))

# ------------------------------------------------------------------ planted-signal calibration (a control: dashed)
XP, WP = XA, XB + WB - XA
box(XP, Y4, WP, BH, "Planted-signal calibration", ["a known signal the scorer must find"], dashed=True)
arrow((XB + WB - 0.25, Y4 + BH / 2), (XC + 0.22, Y3 - BH / 2), dashed=True)
label(XB + WB - 0.32, (Y4 + BH / 2 + Y3 - BH / 2) / 2, "checks the scorer", ha="right")

# ------------------------------------------------------------------ diagnostic protocol
YE0, YE1 = Y4 + 0.02, Y1 - 0.02
box(XE, (YE0 + YE1) / 2, WE, YE1 - YE0, ["Diagnostic", "protocol"], ["", "graded for", "each target,", "never summed"], heavy=True)
arrow((XD + WD, Y1), (XE, YE1 - 0.22)); arrow((XD + WD, YY), (XE, YY)); arrow((XD + WD, Y4), (XE, YE0 + 0.22))

# ------------------------------------------------------------------ the four targets, in words
STRIP = {"image": ["THINGS-EEG2 · image identity", "InfoNCE, bits per query"],
         "text": ["ZuCo · sentence identity", "InfoNCE, bits per record"],
         "prediction_error": ["feedback EEG · signed error", "density gain, bits per trial"],
         "motor_imagery": ["EEGMMIDB · cue class", "log score, bits per trial"]}
GAPS, SY0, SH = 0.07, 0.03, 0.50
sw = (W - 0.10 - 3 * GAPS) / 4
assert Y4 - BH / 2 > SY0 + SH + 0.08, "flow runs into the strip"
for j, t in enumerate(ORDER):
    x = 0.05 + j * (sw + GAPS)
    bg.add_patch(FancyBboxPatch((x, SY0), sw, SH, boxstyle="round,pad=0,rounding_size=0.05", fc=TINT[t], ec="none", zorder=1))
    bg.add_patch(Rectangle((x, SY0 + 0.04), 0.055, SH - 0.08, fc=HUE[t], ec="none", zorder=2))      # colour bar; the name carries the identity in words
    bg.text(x + 0.12, SY0 + SH - 0.105, NAME[t], fontsize=FS + 0.5, fontweight="bold", va="center", zorder=4)
    for i, s in enumerate(STRIP[t]):
        assert T(s) <= sw - 0.16, s
        bg.text(x + 0.12, SY0 + SH - 0.105 - (i + 1) * LH, s, fontsize=FS, va="center", zorder=4)

nbad = check_text(fig)
save(fig, "fig_method_flow")
sys.exit(1 if nbad else 0)
