"""Width curve of the motor-imagery raw-voltage read-out on the full penalty grid (one AAAI column, vector PDF, text >= 7 pt).
  fig_mi_width.pdf   yield (bits per trial) against the number of input components L: PCA (first L of 256 training-fit components) and an equal-width
                     random orthonormal projection (mean of three fixed draws); 95% participant intervals.
Every number is read through results_io (gated by CITABLE): runner_fixes_20261123/rf7c_fullgrid (review, PASS WITH CONDITIONS).
Nothing is typed by hand; the values are checked against tables/new_numbers_batch7.tex so figure and prose cannot drift apart.
Reviewer limits kept: intervals resample participants only (the three random bases are fixed); filled markers = no outer fold chose the smallest
penalty, open markers = at least one did (the yield is a lower bound there); the dotted line marks 90% of the 256-component PCA yield, and the
width reaching it is 'at most 128 on this grid', so no exact L90 is drawn; the planted control is not shown.
Run: srun -n1 -t 5 python -I -u make_fig_mi_width.py"""
import re
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import *  # noqa: F401,F403

plt = style()
S = rio.load_file("runner_fixes_20261123/rf7c_fullgrid", "summary.json")
assert S is not None, "source missing or not citable"
MI = HUE["motor_imagery"]; MIT = TINT["motor_imagery"]; RC = "#5F5F5F"; RT = "#E3E3E3"
W = [int(w) for w in S["widths"]]
PCA = [float(x) for x in S["pca"]["yield_mean"]]; PLO = [float(c[0]) for c in S["pca"]["yield_ci95"]]; PHI = [float(c[1]) for c in S["pca"]["yield_ci95"]]
RND = [float(x) for x in S["rand"]["yield_mean"]]; RLO = [float(c[0]) for c in S["rand"]["yield_ci95"]]; RHI = [float(c[1]) for c in S["rand"]["yield_ci95"]]
EP = [int(S["edges"][f"pca|eeg|{L}"]["lambda_lower_edge"]) for L in W]
ER = [max(int(S["edges"][f"rand|eeg|{L}|d{d}"]["lambda_lower_edge"]) for d in range(3)) for L in W]
REF = 0.9 * PCA[-1]
assert S["pca"]["L90"]["point"] == 128 and PCA[W.index(64)] < REF <= PCA[W.index(128)]   # the dotted line is crossed between 64 and 128

# agreement with the paper's macros (tables/new_numbers_batch7.tex)
MAC = dict(re.findall(r"\\newcommand\{\\(\w+)\}\{(.*)\}", (TABLES / "new_numbers_batch7.tex").read_text()))
WN = {8: "Eight", 16: "Sixteen", 32: "ThirtyTwo", 64: "SixtyFour", 128: "HundredTwentyEight", 256: "TwoHundredFiftySix"}
num = lambda s: float(s.replace("$", "").replace("{", "").replace("}", ""))
for i, L in enumerate(W):
    for nm, v in ((f"WrRfSevenCPca{WN[L]}", PCA[i]), (f"WrRfSevenCRand{WN[L]}", RND[i]), (f"WrRfSevenCPca{WN[L]}Lo", PLO[i]), (f"WrRfSevenCRand{WN[L]}Hi", RHI[i])):
        assert abs(num(MAC[nm]) - v) <= 0.5e-3 + 1e-9, f"macro \\{nm} = {MAC[nm]} disagrees with the source {v}"
print("values agree with tables/new_numbers_batch7.tex")

WC = 3.32
fig = plt.figure(figsize=(WC, 2.55))
ax = fig.add_axes([0.155, 0.19, 0.80, 0.78])
ax.set_xscale("log", base=2)
ax.set_xlim(W[0] / 1.18, W[-1] * 1.18); ax.set_ylim(0, 0.42)
ax.set_xticks(W); ax.set_xticklabels([str(w) for w in W]); ax.minorticks_off()
ax.set_yticks([0, 0.1, 0.2, 0.3, 0.4]); ax.set_yticklabels(["0", "0.1", "0.2", "0.3", "0.4"])
ax.grid(axis="y", color=GRID, lw=0.5); ax.set_axisbelow(True)
ax.spines["bottom"].set_visible(True); ax.tick_params(axis="y", length=0)
ax.axhline(REF, color=INK, lw=0.7, ls=(0, (1.5, 1.8)), zorder=2)
ax.fill_between(W, RLO, RHI, color=RT, lw=0, zorder=1); ax.fill_between(W, PLO, PHI, color=MIT, lw=0, alpha=0.9, zorder=1.1)
ax.plot(W, RND, color=RC, lw=1.1, zorder=3.5)
ax.plot(W, PCA, color=MI, lw=1.6, zorder=4)
for x, yv, e, col in [(W[i], RND[i], ER[i], RC) for i in range(len(W))] + [(W[i], PCA[i], EP[i], MI) for i in range(len(W))]:
    ax.plot([x], [yv], marker="o", ms=4.6, mfc=col if e == 0 else "white", mec=col, mew=1.1, zorder=5, ls="none")
ax.set_xlabel("input components $L$", labelpad=2); ax.set_ylabel("yield (bits per trial)", labelpad=2)
# direct labels
ax.plot([9.0], [0.395], marker="o", ms=4.6, mfc=MI, mec=MI); ax.text(10.4, 0.395, "PCA, first $L$ components", fontsize=FS, color=MI, fontweight="bold", va="center")
ax.plot([9.0], [0.355], marker="o", ms=4.6, mfc=RC, mec=RC); ax.text(10.4, 0.355, "random projection, mean of 3", fontsize=FS, color=RC, fontweight="bold", va="center")
ax.text(8.6, REF + 0.016, "90% of the 256-component PCA yield", fontsize=FS, color=INK, va="bottom", ha="left")
ax.text(8.6, 0.245, "filled: no fold at the\nsmallest penalty;\nopen: some folds at it\n(a lower bound)", fontsize=FS, color=MUTE, va="top", ha="left", linespacing=1.1)
nbad = check_text(fig)
save(fig, "fig_mi_width")
sys.exit(1 if nbad else 0)
