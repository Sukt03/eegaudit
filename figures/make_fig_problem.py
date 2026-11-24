"""fig_problem.pdf: the evaluation problem, as a concept (one column, no numbers). A decoder's reported score is drawn as one bar,
split into the parts the audit separates; three brackets carry the three questions. Segment widths are schematic, not data.
Run: srun -n1 -t 5 python -I -u -B make_fig_problem.py"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import *  # noqa: F401,F403
from matplotlib.patches import Rectangle

plt = style()
NOBR = "--nobrackets" in sys.argv       # write fig_problem_nobrackets.pdf: the bar alone, without the three question brackets
W, H = 3.32, (0.86 if NOBR else 1.58)
fig = plt.figure(figsize=(W, H))
bg = fig.add_axes([0, 0, 1, 1]); bg.set_xlim(0, W); bg.set_ylim(0, H); bg.axis("off")
LH = 0.118
YB, BH = (0.36 if NOBR else 0.50), 0.24
# parts of the reported score (schematic widths): name lines, fill, hatch, edge style
SEG = [(["EEG", "removed"], "#E3E3E3", None, 0.62), (["test-set", "calibration"], "white", "....", 0.66),
       (["label, task,", "eye structure"], "#F6D58A", "////", 0.72), (["usable", "EEG yield"], "#56B4E9", None, 0.62)]
x = 0.05; xs = []
for lines, fc, hatch, w in SEG:
    bg.add_patch(Rectangle((x, YB), w, BH, fc=fc, ec=INK, lw=0.8, hatch=hatch, zorder=2))
    for i, s in enumerate(lines):
        assert tw(fig, s) <= w + 0.06, s
        bg.text(x + w / 2, YB - 0.10 - i * LH, s, fontsize=FS, ha="center", va="center")
    xs.append((x, x + w)); x += w
XE = x
# what a replicate of the EEG reaches (outline only)
WR = W - 0.05 - XE
bg.add_patch(Rectangle((XE, YB), WR, BH, fc="none", ec=INK, lw=0.8, ls=(0, (3.2, 2.0)), zorder=2))
bg.text(XE + WR / 2, YB - 0.10, "replicate", fontsize=FS, ha="center", va="center", color=MUTE)
bg.text(XE + WR / 2, YB - 0.10 - LH, "of the EEG", fontsize=FS, ha="center", va="center", color=MUTE)


def bracket(x0, x1, y, text):
    if NOBR:
        return
    bg.plot([x0 + 0.02, x0 + 0.02, x1 - 0.02, x1 - 0.02], [y - 0.05, y, y, y - 0.05], color=INK, lw=0.8, solid_capstyle="butt")
    bg.text((x0 + x1) / 2, y + 0.085, text, fontsize=FS, fontweight="bold", ha="center", va="center")


y1 = YB + BH + 0.10
bracket(xs[1][0], xs[2][1], y1, "about the target?")
bracket(xs[3][0], W - 0.05, y1, "how much is used?")
y2 = y1 + 0.27
bracket(xs[1][0], xs[3][1], y2, "usable signal above the blank?")
bg.text(0.05, H - 0.10, "One reported score", fontsize=FS + 0.5, fontweight="bold", va="center")
bg.plot([0.05, 0.05, XE, XE], [YB + BH + 0.03, H - 0.21, H - 0.21, H - 0.21], color=MUTE, lw=0.0)
nbad = check_text(fig, min_pt=6.5)
save(fig, "fig_problem_nobrackets" if NOBR else "fig_problem")
sys.exit(1 if nbad else 0)
