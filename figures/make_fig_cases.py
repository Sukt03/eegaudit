"""Two audit cases as small, direct-labelled graphs (vector PDF, text >= 7 pt at print size).
  fig_glim.pdf            (A) text, GLIM: the standard control against the audit reading, plus the genre-only control (one column)
  fig_mi_derivations.pdf  (B) motor imagery: the ocular proxies against 60 random derivation sets (one column)
  fig_cases.pdf           A and B side by side (full text width)
Every number is read through results_io (gated by CITABLE): ext/glim_within_task (card12 grid, sentence-cluster intervals),
runner_fixes_20261123/rf9_mi_random_derivations and rf2_mi. Nothing is typed by hand; the values agree with tables/new_numbers_batch5.tex.
Reviewer limits kept: within-task GLIM gaps with sentence intervals only; MI draws shown one by one, no percentile band.
Run: srun -n1 -t 5 python -I -u make_fig_cases.py"""
import re
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import *  # noqa: F401,F403
from matplotlib.patches import FancyBboxPatch, Rectangle

plt = style()
G = rio.load_file("ext/glim_within_task", "summary.json")
R9 = rio.load_file("runner_fixes_20261123/rf9_mi_random_derivations", "summary.json")
R2 = rio.load_file("runner_fixes_20261123/rf2_mi", "summary.json")
assert G is not None and R9 is not None and R2 is not None, "source missing or not citable"

# ------------------------------------------------------------------ data: GLIM (card12 grid)
TXT = HUE["text"]; MI = HUE["motor_imagery"]
CELL, T1, T2 = "cell:<NR>ZuCo1", "cell:<NR>ZuCo1|task1", "cell:<NR>ZuCo1|task2"


def gap(sel):
    c = G["selections"][sel]["grids"]["card12"]
    ci = c["ci95"]["sentence"]["diff:eeg_minus_noise_test_time"]
    return c["diffs"]["eeg_minus_noise_test_time"]["mean"], ci[0], ci[1]


def eeg(sel):
    return G["selections"][sel]["grids"]["card12"]["arms"]["eeg"]["yield_bits"]


GENRE = G["planted_controls"]["planted_genre_only_constant_text"]
GAPS = [("standard control", None), ("pooled over tasks", gap("pooled")), ("ZuCo1 <NR> cell", gap(CELL)),
        ("audit reading: one task at a time", None), ("task 1, sentiment", gap(T1)), ("task 2, Wikipedia", gap(T2))]
GEN = [("ZuCo1 <NR> cell", eeg(CELL), GENRE[CELL]["card12"]), ("task 1, sentiment", eeg(T1), GENRE[T1]["card12"]),
       ("task 2, Wikipedia", eeg(T2), GENRE[T2]["card12"])]

# ------------------------------------------------------------------ data: MI random derivations
assert abs(R9["proxy_kept"] - R2["proxy"]["kept"]) < 1e-12
PROXY = (R9["proxy_kept"], R9["proxy_ci95"][0], R9["proxy_ci95"][1])
FAM = [("frontal", "frontal", "o", "#E69F00"), ("nonfrontal", "non-frontal", "^", "#56B4E9"), ("any", "any non-proxy", "s", "#7A7A7A")]
DRAWS = {k: [float(v) for v in R2["families"][k]["kept_sorted"]] for k, *_ in FAM}
for k, *_ in FAM:
    assert len(DRAWS[k]) == int(R9["families"][k]["n_draws"])
N_DRAWS = sum(len(v) for v in DRAWS.values())
N_LE = sum(int(R9["families"][k]["n_draws_le_proxy"]) for k, *_ in FAM)
assert N_LE == sum(1 for v in DRAWS.values() for x in v if x <= PROXY[0])
LOWEST = min(min(v) for v in DRAWS.values())

# check against the paper's macros (tables/new_numbers_batch5.tex), so figure and prose cannot drift apart
MAC = dict(re.findall(r"\\newcommand\{\\(\w+)\}\{(.*)\}", (TABLES / "new_numbers_batch5.tex").read_text()))
num = lambda s: float(s.replace("$", "").replace("{", "").replace("}", ""))
for nm, v in (("WrGOnePoolGap", gap("pooled")[0]), ("WrGOneCellGap", gap(CELL)[0]), ("WrGOneTaskOneGap", gap(T1)[0]), ("WrGOneTaskTwoGap", gap(T2)[0]),
              ("WrGOneTaskOneGapLo", gap(T1)[1]), ("WrGOneTaskTwoGapHi", gap(T2)[2]), ("WrGOneGenreCell", GEN[0][2]), ("WrGOneCellEeg", GEN[0][1]),
              ("WrGOneGenreTaskOne", GEN[1][2]), ("WrGOneGenreTaskTwo", GEN[2][2]), ("WrGTwoProxyKept", PROXY[0]), ("WrGTwoProxyKeptLo", PROXY[1]),
              ("WrGTwoProxyKeptHi", PROXY[2]), ("WrGTwoDrawMin", LOWEST), ("WrGTwoDraws", N_DRAWS), ("WrGTwoDrawsLeProxy", N_LE)):
    d = len(MAC[nm].replace("$", "").split(".")[1]) if "." in MAC[nm] else 0
    assert abs(num(MAC[nm]) - v) <= 0.5 * 10 ** (-d) + 1e-9, f"macro \\{nm} = {MAC[nm]} disagrees with the source {v}"
print("values agree with tables/new_numbers_batch5.tex")

sg = lambda x, d=3: (f"{x:+.{d}f}").replace("-", "\u2212")
us = lambda x, d=3: (f"{x:.{d}f}").replace("-", "\u2212")
WC = 3.32                      # one AAAI column (239.4 pt)
HA1, HA2 = 1.52, 1.18          # heights of the two parts of A (inches)


def canvas(W, H):
    fig = plt.figure(figsize=(W, H))
    bg = fig.add_axes([0, 0, 1, 1]); bg.set_xlim(0, W); bg.set_ylim(0, H); bg.axis("off")
    return fig, bg


def axes(fig, x, y, w, h, xlim, n, ticks):
    W, H = fig.get_size_inches()
    ax = fig.add_axes([x / W, y / H, w / W, h / H])
    ax.set_xlim(*xlim); ax.set_ylim(-0.5, n - 0.5); ax.set_yticks([]); ax.set_xticks(ticks)
    ax.set_xticklabels([("%g" % v).replace("-", "\u2212") for v in ticks])
    ax.grid(axis="x", color=GRID, lw=0.5); ax.set_axisbelow(True); ax.patch.set_alpha(0)
    return ax


def title(bg, x, y, letter, text):
    bg.text(x, y, letter, fontsize=9, fontweight="bold", va="center")
    bg.text(x + 0.17, y, text, fontsize=FS + 0.5, fontweight="bold", va="center")


def draw_gaps(fig, bg, x0, ytop, letter):
    """A, upper part: EEG minus noise input, pooled and cell (standard control) against within task (audit reading)."""
    title(bg, x0 + 0.02, ytop - 0.10, letter, "Text (GLIM): the gap is between tasks")
    n = len(GAPS); RH = 0.16
    XL = x0 + 1.30; ax0 = XL + 0.08; aw = x0 + WC - 0.04 - ax0
    y1 = ytop - 0.24; y0 = y1 - n * RH
    ax = axes(fig, ax0, y0, aw, n * RH, (-0.12, 0.66), n, [0, 0.2, 0.4, 0.6])
    # soft band behind the audit rows
    bg.add_patch(FancyBboxPatch((x0 + 0.02, y0 + 0.005), WC - 0.04, 3 * RH - 0.01, boxstyle="round,pad=0,rounding_size=0.05", fc=TINT["text"], ec="none", zorder=0))
    ax.plot([0, 0], [-0.5, n - 0.5], color=INK, lw=0.8, ls=(0, (2.2, 1.6)), zorder=2)
    for i, (lab, v) in enumerate(GAPS):
        yy = n - 1 - i; yin = y0 + (yy + 0.5) * RH
        if v is None:                      # group header
            bg.text(x0 + 0.06, yin - 0.01, lab, fontsize=FS, color=MUTE, style="italic", va="center"); continue
        m, lo, hi = v
        audit = i > 3
        bg.text(XL, yin, lab, fontsize=FS, ha="right", va="center")
        ax.plot([lo, hi], [yy, yy], color=TXT, lw=2.2, solid_capstyle="round", zorder=3)
        ax.plot([m], [yy], marker="D" if audit else "o", ms=5.0 if audit else 5.6, mfc="white" if audit else TXT, mec=TXT if audit else "white",
                mew=1.2 if audit else 0.7, zorder=4, ls="none")
        ax.text(hi + 0.02, yy, sg(m), fontsize=FS, va="center", ha="left", zorder=5)
    bg.text(ax0 + aw / 2, y0 - 0.22, "EEG minus noise input (bits per record)", fontsize=FS, ha="center", va="center")
    bg.text(ax0 + aw * ((0.28 + 0.12) / 0.78), y0 + 1.0 * RH, "no detected gap", fontsize=FS, color=TXT, fontweight="bold", va="center", ha="left")
    return y0 - 0.30


def draw_genre(fig, bg, x0, ytop, letter):
    """A, lower part: the EEG arm against a constant text that knows only the reading task (genre-only planted control)."""
    title(bg, x0 + 0.02, ytop - 0.10, letter, "A genre-only text scores the same")
    n = len(GEN); RH = 0.16
    XL = x0 + 1.30; ax0 = XL + 0.08
    XV1 = x0 + WC - 0.78; XV2 = x0 + WC - 0.04          # right edges of the two value columns
    aw = XV1 - 0.42 - ax0
    yhd = ytop - 0.30
    bg.plot([XV1 - tw(fig, "EEG") - 0.07], [yhd + 0.005], marker="o", ms=5.0, mfc=TXT, mec="white", mew=0.6)
    bg.text(XV1, yhd, "EEG", fontsize=FS, ha="right", va="center")
    bg.plot([XV2 - tw(fig, "genre only") - 0.07], [yhd + 0.005], marker="s", ms=4.4, mfc="white", mec=INK, mew=0.9)
    bg.text(XV2, yhd, "genre only", fontsize=FS, ha="right", va="center")
    y1 = yhd - 0.10; y0 = y1 - n * RH
    ax = axes(fig, ax0, y0, aw, n * RH, (-0.08, 0.40), n, [0, 0.2, 0.4])
    ax.plot([0, 0], [-0.5, n - 0.5], color=INK, lw=0.8, ls=(0, (2.2, 1.6)), zorder=2)
    for i, (lab, e, g) in enumerate(GEN):
        yy = n - 1 - i; yin = y0 + (yy + 0.5) * RH
        bg.text(XL, yin, lab, fontsize=FS, ha="right", va="center")
        ax.plot([e, g], [yy, yy], color="#9AA5AE", lw=1.6, solid_capstyle="round", zorder=3)
        ax.plot([g], [yy], marker="s", ms=4.4, mfc="white", mec=INK, mew=0.9, zorder=4, ls="none")
        ax.plot([e], [yy], marker="o", ms=5.0, mfc=TXT, mec="white", mew=0.6, zorder=5, ls="none")
        bg.text(XV1, yin, sg(e), fontsize=FS, ha="right", va="center")
        bg.text(XV2, yin, sg(g), fontsize=FS, ha="right", va="center")
    bg.text(ax0 + aw / 2, y0 - 0.22, "yield (bits per record)", fontsize=FS, ha="center", va="center")
    return y0 - 0.30


def swarm(vals, aw, xlim, d=0.055):
    """Greedy beeswarm in inches: each point is moved off the row centre until it clears the points already placed."""
    placed = []
    for v in sorted(vals):
        x = (v - xlim[0]) / (xlim[1] - xlim[0]) * aw
        for k in range(200):
            off = (k + 1) // 2 * d / 6 * (1 if k % 2 else -1) if k else 0.0
            if all((x - px) ** 2 + (off - py) ** 2 >= d * d for px, py, _ in placed):
                break
        placed.append((x, off, v))
    return placed


def draw_mi(fig, bg, x0, ytop, height, letter):
    """B: kept share of the raw-voltage yield after regressing 60 random derivation sets, against the ocular proxies."""
    title(bg, x0 + 0.02, ytop - 0.10, letter, "Motor imagery: proxies against random sets")
    bg.text(x0 + 0.19, ytop - 0.27, f"{N_LE} of {N_DRAWS} random sets fall to the proxy value; lowest {us(LOWEST)}", fontsize=FS, color=MUTE, va="center")
    XL = x0 + 0.98; ax0 = XL + 0.06; aw = x0 + WC - 0.10 - ax0
    ybot = ytop - height + 0.36
    HEADB = 0.34                                  # band above the strips for the proxy label
    y1 = ytop - 0.38; hax = y1 - ybot
    SH = (hax - HEADB) / 3                        # strip height (inches)
    xlim = (0, 1.0)
    ax = axes(fig, ax0, ybot, aw, hax, xlim, 3, [0, 0.25, 0.5, 0.75, 1])
    ax.set_ylim(0, hax)                           # y in inches, so the swarm keeps round spacing
    # proxy value: interval band and bold line
    ax.axvspan(PROXY[1], PROXY[2], color=TINT["motor_imagery"], zorder=1, lw=0)
    ax.plot([PROXY[0], PROXY[0]], [0, hax - 0.03], color=MI, lw=2.4, solid_capstyle="round", zorder=3)
    ax.text(PROXY[2] + 0.025, hax - 0.085, "ocular proxies", fontsize=FS, fontweight="bold", color=MI, va="center", ha="left")
    ax.text(PROXY[2] + 0.025, hax - 0.215, f"{us(PROXY[0], 4)} [{us(PROXY[1])}, {us(PROXY[2])}]", fontsize=FS, va="center", ha="left")
    for i, (k, lab, mk, col) in enumerate(FAM):
        yc = hax - HEADB - (i + 0.5) * SH
        if i:
            ax.plot([0, 1], [yc + SH / 2, yc + SH / 2], color=GRID, lw=0.5, zorder=1)
        for x, off, v in swarm(DRAWS[k], aw, xlim):
            assert abs(off) < SH / 2 - 0.03, f"swarm leaves its strip: offset {off:.3f} in, strip {SH:.3f} in"
            ax.plot([v], [yc + off], marker=mk, ms=3.3, mfc=col, mec=INK, mew=0.45, zorder=4, ls="none")
        yin = ybot + yc
        bg.plot([x0 + 0.10], [yin], marker=mk, ms=4.6, mfc=col, mec=INK, mew=0.5)
        bg.text(x0 + 0.20, yin, lab, fontsize=FS, va="center")
    bg.text(ax0 + aw / 2, ybot - 0.22, "share of the raw-voltage yield kept", fontsize=FS, ha="center", va="center")


def done(fig, name):
    nbad = check_text(fig); save(fig, name); return nbad


bad = 0
# (A) one column
HA = HA1 + HA2 + 0.02
fig, bg = canvas(WC, HA)
y = draw_gaps(fig, bg, 0, HA - 0.02, "a"); draw_genre(fig, bg, 0, y - 0.02, "b")
bad += done(fig, "fig_glim")
# (B) one column
HB = 2.35
fig, bg = canvas(WC, HB)
draw_mi(fig, bg, 0, HB - 0.02, HB - 0.02, "")
bad += done(fig, "fig_mi_derivations")
# A and B side by side, full width
fig, bg = canvas(7.0, HA)
y = draw_gaps(fig, bg, 0, HA - 0.02, "a"); draw_genre(fig, bg, 0, y - 0.02, "b")
draw_mi(fig, bg, 7.0 - WC, HA - 0.02, HA - 0.02, "c")
bad += done(fig, "fig_cases")
sys.exit(1 if bad else 0)
