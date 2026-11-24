"""fig_method_flow_v2.pdf: post-freeze candidate for Figure 2. Same boxes, labels and strip as fig_method_flow.pdf, laid out as a
two-row flow so that it reads at 7 pt when included at \textwidth and stays about as tall as the frozen figure is in print.
Top row: the real run (solid). Bottom row: the controls (dashed) and the nuisance arm. The decoder, the yield and the diagnostic
protocol span both rows. The four-target strip sits below. A schematic: labels only, no measured value and no typed number.
Run: srun -n1 -t 5 python -I -u -B make_fig_method_flow_v2.py"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import *  # noqa: F401,F403
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

plt = style()
LH = 0.124
DASH = (0, (3.2, 2.0))
BH, RGAP = 0.50, 0.20                    # box height (three lines), gap between the two rows
CALH, CGAP = 0.26, 0.17                  # calibration box (one line) and the gap above it
NOSTRIP = "--nostrip" in sys.argv        # write fig_method_flow_v2_nostrip.pdf: the box flow without the four-target strip
SH, SGAP = (0.0, 0.0) if NOSTRIP else (0.50, 0.10)   # strip and the gap above it
H = 0.03 + SH + SGAP + CALH + CGAP + 2 * BH + RGAP + 0.03
W = 7.0
fig = plt.figure(figsize=(W, H))
bg = fig.add_axes([0, 0, 1, 1]); bg.set_xlim(0, W); bg.set_ylim(0, H); bg.axis("off")
T = lambda s, **kw: tw(fig, s, **kw)


def box(x, y0, w, h, head, body, dashed=False, heavy=False):
    bg.add_patch(FancyBboxPatch((x, y0), w, h, boxstyle="round,pad=0,rounding_size=0.06", fc="white", ec=INK,
                                lw=1.5 if heavy else 0.9, ls=DASH if dashed else "-", zorder=2))
    head = [head] if isinstance(head, str) else head
    rows = [(s, dict(fontsize=FS + 0.5, fontweight="bold")) for s in head] + [(s, dict(fontsize=FS)) for s in body]
    yc = y0 + h / 2
    y = yc + (len(rows) - 1) * LH / 2
    for i, (s, kw) in enumerate(rows):
        assert T(s, **kw) <= w - 0.07, f"line too wide for its box ({T(s, **kw):.2f} > {w - 0.07:.2f}): {s!r}"
        bg.text(x + w / 2, y - i * LH - 0.003, s, ha="center", va="center", zorder=4, **kw)


def arrow(p0, p1, dashed=False):
    bg.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=6.5, lw=0.9, color=INK, ls=DASH if dashed else "-",
                                 shrinkA=0, shrinkB=0, zorder=3))


def label(x, y, s, ha="left"):
    bg.text(x, y, s, fontsize=FS, color=MUTE, ha=ha, va="center", zorder=4)


# rows (bottom edges) and columns
YB = 0.03 + SH + SGAP + CALH + CGAP      # bottom row: controls
YT = YB + BH + RGAP                      # top row: the real run
WID = [0.84, 0.72, 1.12, 0.88, 0.94, 0.94, 0.76]
GAPX = (W - 0.06 - sum(WID)) / 6
assert GAPX >= 0.09, f"columns do not fit ({GAPX:.3f})"
X = [0.03 + sum(WID[:i]) + i * GAPX for i in range(7)]
mid = lambda y0: y0 + BH / 2
tall = dict(y0=YB, h=2 * BH + RGAP)

# column 1: EEG trials, and the EEG removed (control)
box(X[0], YT, WID[0], BH, "EEG trials", ["held-out", "test split"])
box(X[0], YB, WID[0], BH, "EEG removed", ["same decoder,", "no EEG"], dashed=True)
arrow((X[0] + WID[0] - 0.16, YT), (X[0] + WID[0] - 0.16, YB + BH), dashed=True)
label(X[0] + WID[0] - 0.21, YB + BH + RGAP / 2, "blank input", ha="right")
# column 2: the same decoder for both rows
box(X[1], tall["y0"], WID[1], tall["h"], "Decoder", ["released or", "retrained"])
arrow((X[0] + WID[0], mid(YT)), (X[1], mid(YT))); arrow((X[0] + WID[0], mid(YB)), (X[1], mid(YB)), dashed=True)
# column 3: the two scores
box(X[2], YT, WID[2], BH, "Score with EEG", ["log score, bits"])
box(X[2], YB, WID[2], BH, "Score without EEG", ["noise, shuffle,", "noise-trained"], dashed=True)
arrow((X[1] + WID[1], mid(YT)), (X[2], mid(YT))); arrow((X[1] + WID[1], mid(YB)), (X[2], mid(YB)), dashed=True)
# column 4: yield
box(X[3], tall["y0"], WID[3], tall["h"], "Yield", ["with − without", "chance and", "baseline on the", "same table"], heavy=True)
arrow((X[2] + WID[2], mid(YT)), (X[3], mid(YT))); arrow((X[2] + WID[2], mid(YB)), (X[3], mid(YB)))
# columns 5 and 6: replicate arm -> usage, nuisance arm -> nuisance share
box(X[4], YT, WID[4], BH, "Replicate arm", ["split-half ·", "two read-outs"])
box(X[5], YT, WID[5], BH, "Usage", ["yield ÷", "replicate yield"])
box(X[4], YB, WID[4], BH, "Nuisance arm", ["category · task ·", "eye proxies"])
box(X[5], YB, WID[5], BH, "Nuisance share", ["yield lost", "when removed"])
for y0 in (YT, YB):
    arrow((X[3] + WID[3], mid(y0)), (X[4], mid(y0))); arrow((X[4] + WID[4], mid(y0)), (X[5], mid(y0))); arrow((X[5] + WID[5], mid(y0)), (X[6], mid(y0)))
# column 7: diagnostic protocol
box(X[6], tall["y0"], WID[6], tall["h"], ["Diagnostic", "protocol"], ["", "graded for", "each target,", "never", "summed"], heavy=True)
# planted-signal calibration (control), one line under the scores
YC = 0.03 + SH + SGAP
cal_h, cal_b = "Planted-signal calibration", "a known signal the scorer must find"
wc = T(cal_h, fontsize=FS + 0.5, fontweight="bold") + T(cal_b) + 0.34
xc = X[2] + WID[2] / 2 - wc / 2
bg.add_patch(FancyBboxPatch((xc, YC), wc, CALH, boxstyle="round,pad=0,rounding_size=0.06", fc="white", ec=INK, lw=0.9, ls=DASH, zorder=2))
bg.text(xc + 0.09, YC + CALH / 2 - 0.003, cal_h, fontsize=FS + 0.5, fontweight="bold", va="center", zorder=4)
bg.text(xc + wc - 0.09, YC + CALH / 2 - 0.003, cal_b, fontsize=FS, ha="right", va="center", zorder=4)
xa = X[2] + WID[2] / 2
arrow((xa, YC + CALH), (xa, YB), dashed=True)
label(xa + 0.06, YC + CALH + CGAP / 2, "checks the scorer")

# the four targets, in words
STRIP = {"image": ["THINGS-EEG2 · image identity", "InfoNCE, bits per query"],
         "text": ["ZuCo · sentence identity", "InfoNCE, bits per record"],
         "prediction_error": ["feedback EEG · signed error", "density gain, bits per trial"],
         "motor_imagery": ["EEGMMIDB · cue class", "log score, bits per trial"]}
GS = 0.07
sw = (W - 0.06 - 3 * GS) / 4
for j, t in enumerate([] if NOSTRIP else ORDER):
    x = 0.03 + j * (sw + GS)
    bg.add_patch(FancyBboxPatch((x, 0.03), sw, SH, boxstyle="round,pad=0,rounding_size=0.05", fc=TINT[t], ec="none", zorder=1))
    bg.add_patch(Rectangle((x, 0.07), 0.055, SH - 0.08, fc=HUE[t], ec="none", zorder=2))
    bg.text(x + 0.12, 0.03 + SH - 0.105, NAME[t], fontsize=FS + 0.5, fontweight="bold", va="center", zorder=4)
    for i, s in enumerate(STRIP[t]):
        assert T(s) <= sw - 0.16, s
        bg.text(x + 0.12, 0.03 + SH - 0.105 - (i + 1) * LH, s, fontsize=FS, va="center", zorder=4)

nbad = check_text(fig)
save(fig, "fig_method_flow_v2_nostrip" if NOSTRIP else "fig_method_flow_v2")
sys.exit(1 if nbad else 0)
