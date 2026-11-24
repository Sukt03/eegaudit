"""Graphs of four appendix number tables (vector PDF, one column each, text >= 7 pt at print size, no research-question labels).
  fig_blank_floors.pdf        phase-randomised and trial-shuffled real-input scores of the released UBP encoder, per seed group
  fig_mi_foundation.pdf       foundation-model read-outs on motor imagery (proxy-regressed input): pretrained against random-init
  fig_planted_calibration.pdf planted-information calibration, prediction error (ds003458)
  fig_category_erasure.pdf    category erasure through frozen image decoders, against the permuted-profile control
Every number is read through results_io (gated by CITABLE), from the same files and keys as paper/figs/make_tables.py uses for
tables/tab_controls_extra.tex and tables/tab_mi.tex. The only constants are the two protocol tolerances named in those tables
(0.02 bits for a blank, 0.20 for a calibration miss).
Run: srun -n1 -t 5 python -I -u -B make_fig_appendix.py"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import *  # noqa: F401,F403
from matplotlib.patches import FancyBboxPatch

plt = style()
WC = 3.32
TOL_BLANK, TOL_MISS = 0.02, 0.20          # protocol tolerances (tab_controls_extra.tex)
IMG, PE, MI = HUE["image"], HUE["prediction_error"], HUE["motor_imagery"]
mn = lambda s: s.replace("-", "−")


def canvas(H):
    fig = plt.figure(figsize=(WC, H))
    bg = fig.add_axes([0, 0, 1, 1]); bg.set_xlim(0, WC); bg.set_ylim(0, H); bg.axis("off")
    return fig, bg


def axes(fig, x, y, w, h, xlim, ylim, ticks):
    W, H = fig.get_size_inches()
    ax = fig.add_axes([x / W, y / H, w / W, h / H])
    ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.set_yticks([]); ax.set_xticks(ticks)
    ax.set_xticklabels([mn("%g" % v) for v in ticks])
    ax.grid(axis="x", color=GRID, lw=0.5); ax.set_axisbelow(True); ax.patch.set_alpha(0)
    return ax


def head(bg, H, text):
    bg.text(0.02, H - 0.11, text, fontsize=FS + 0.5, fontweight="bold", va="center")


def done(fig, name):
    nbad = check_text(fig); save(fig, name); return nbad


def ci_dot(ax, m, lo, hi, y, col, marker="o", hollow=False, ms=5.0):
    ax.plot([lo, hi], [y, y], color=col, lw=2.0, solid_capstyle="round", zorder=3)
    ax.plot([m], [y], marker=marker, ms=ms, mfc="white" if hollow else col, mec=col if hollow else "white", mew=1.1 if hollow else 0.6, zorder=4, ls="none")


bad = 0
# ================================================================== G1: blank floors of the released UBP encoder
BLOCKS = [("baselines/ubp_phase_control", "seeds 0–2"), ("baselines/ubp_phase_control_s3to6", "seeds 3–6"), ("baselines/ubp_phase_control_s7to9", "seeds 7–9")]
G1 = {"phase": [], "shuffle": []}
n_img = None
for rel, lab in BLOCKS:
    S = rio.load_file(rel, "summary.json"); assert S is not None, rel
    n_img = int(S["n_participants"])
    for kind in ("phase", "shuffle"):
        for k in ("80", "40"):
            b = S["bits"][f"{kind}_k{k}"]
            sv = S["bits"][f"{kind}_k{k}_seed_values"]
            vals = [sv[sd][i] for sd in list(sv.keys()) for i in range(len(sv[sd]))]
            G1[kind].append((f"{lab}, {k} rep.", b["mean"], b["ci95"][0], b["ci95"][1], max(vals, key=abs)))
RH = 0.150
H = 0.36 + 2 * (0.17 + 6 * RH) + 0.46
fig, bg = canvas(H)
head(bg, H, "Blank checks of the released image decoder")
XL = 1.20; ax0 = XL + 0.07; aw = WC - 0.08 - ax0
y = H - 0.34
# key
bg.plot([0.10], [y + 0.02], marker="o", ms=5.0, mfc=IMG, mec="white", mew=0.6); bg.text(0.18, y + 0.02, "mean, 95% interval", fontsize=FS, va="center")
kx = 0.18 + tw(fig, "mean, 95% interval") + 0.16
bg.plot([kx], [y + 0.02], marker="D", ms=4.2, mfc="white", mec=INK, mew=0.9); bg.text(kx + 0.08, y + 0.02, "largest single value", fontsize=FS, va="center")
y -= 0.10
for kind, title in (("phase", "phase-randomised input"), ("shuffle", "trial-shuffled input")):
    rows = G1[kind]; n = len(rows)
    bg.text(0.04, y - 0.085, title, fontsize=FS, color=MUTE, style="italic", va="center")
    y1 = y - 0.17; y0 = y1 - n * RH
    last = kind == "shuffle"
    ax = axes(fig, ax0, y0, aw, n * RH, (-0.026, 0.026), (-0.5, n - 0.5), [-TOL_BLANK, 0, TOL_BLANK])
    if not last:
        ax.set_xticklabels([]); ax.tick_params(length=0)
    ax.axvspan(-TOL_BLANK, TOL_BLANK, color="#ECECEC", zorder=0, lw=0)
    ax.axvline(0, color=INK, lw=0.8, ls=(0, (2.2, 1.6)), zorder=2)
    for i, (lab, m, lo, hi, mx) in enumerate(rows):
        yy = n - 1 - i
        bg.text(XL, y0 + (yy + 0.5) * RH, lab, fontsize=FS, ha="right", va="center")
        ci_dot(ax, m, lo, hi, yy, IMG)
        ax.plot([mx], [yy], marker="D", ms=4.2, mfc="white", mec=INK, mew=0.9, zorder=5, ls="none")
    y = y0 - 0.02
bg.text(ax0 + aw / 2, y - 0.20, "score without usable EEG (bits per query)", fontsize=FS, ha="center", va="center")
bad += done(fig, "fig_blank_floors")

# ================================================================== G2: foundation models on motor imagery (proxy-regressed input)
FMS = [("cbramod", "CBraMod", "disjoint"), ("reve", "REVE", "disjoint"), ("labram", "LaBraM", "transductive"), ("eegpt", "EEGPT", "transductive")]
G2 = []
# participant-level detection (A3): yield above the 95th percentile of the pooled stored blank values of the cell
A3S = rio.load_file("ext/a3_participant_detection", "summary.json"); assert A3S is not None
assert float(A3S["constants"]["percentile"]) == 0.95
A3 = {c["cell"].replace("MI_FM_eogreg_", ""): c for c in A3S["cells"] if c["cell"].startswith("MI_FM_eogreg_")}
MACB = dict(__import__("re").findall(r"\\newcommand\{\\(\w+)\}\{(.*)\}", (TABLES / "new_numbers_batch8.tex").read_text()))
for m, name, lbl in FMS:
    S = rio.load_file("ext/mi_fm_eogreg", f"{m}/summary.json"); assert S is not None, m
    f_, t_, d_ = S["rows"]["fm"]["eeg"], S["rows"]["twin"]["eeg"], S["dividend_fm_minus_twin"]
    G2.append((name, lbl, (f_["mean"], f_["ci"][0], f_["ci"][1]), (t_["mean"], t_["ci"][0], t_["ci"][1]), int(A3[m]["k"]), int(A3[m]["n"])))
RH = 0.30; n = len(G2)
H = 0.56 + n * RH + 0.38
fig, bg = canvas(H)
head(bg, H, "Pretraining adds a small yield")
XL = 1.02; ax0 = XL + 0.07; XR = WC - 0.04; aw = XR - 0.62 - ax0
yk = H - 0.33
bg.plot([0.10], [yk], marker="o", ms=5.0, mfc=MI, mec="white", mew=0.6); bg.text(0.18, yk, "pretrained", fontsize=FS, va="center")
kx = 0.18 + tw(fig, "pretrained") + 0.16
bg.plot([kx], [yk], marker="o", ms=4.6, mfc="white", mec=MI, mew=1.1); bg.text(kx + 0.08, yk, "random-init", fontsize=FS, va="center")
bg.text(XR, yk + 0.06, "detected (p95 of", fontsize=FS, color=MUTE, ha="right", va="center")
bg.text(XR, yk - 0.06, "pooled blank)", fontsize=FS, color=MUTE, ha="right", va="center")
y1 = H - 0.50; y0 = y1 - n * RH
ax = axes(fig, ax0, y0, aw, n * RH, (-0.008, 0.07), (-0.5, n - 0.5), [0, 0.02, 0.04, 0.06])
ax.axvline(0, color=INK, lw=0.8, ls=(0, (2.2, 1.6)), zorder=2)
for i, (name, lbl, f_, t_, npos, nn) in enumerate(G2):
    yy = n - 1 - i; yin = y0 + (yy + 0.5) * RH
    bg.text(XL, yin + 0.058, name, fontsize=FS, fontweight="bold", ha="right", va="center")
    bg.text(XL, yin - 0.062, lbl, fontsize=FS, color=MUTE, ha="right", va="center")
    ci_dot(ax, *t_, yy - 0.17, MI, hollow=True, ms=4.6)
    ci_dot(ax, *f_, yy + 0.17, MI)
    assert MACB[f"WrAThree{FMS[i][0].capitalize()}K"] == str(npos) and MACB[f"WrAThree{FMS[i][0].capitalize()}N"] == str(nn), "A3 count disagrees with new_numbers_batch8.tex"
    bg.text(XR, yin, f"{npos} of {nn}", fontsize=FS, ha="right", va="center")
bg.text(ax0 + aw / 2, y0 - 0.23, "yield (bits per trial)", fontsize=FS, ha="center", va="center")
bad += done(fig, "fig_mi_foundation")

# ================================================================== G3: planted-information calibration, prediction error
PCS = rio.load_file("gpu_fill_20261122/planted_calib", "summary.json"); assert PCS is not None
n_pe = int(PCS["n_participants"])
G3 = {}
for L in ("L64", "Lfull"):
    G3[L] = []
    for fp in ("0.0", "0.25", "0.5", "0.75"):
        r = PCS["rows"][f"{L}|f={fp}"]
        G3[L].append((r["f_true_measured"]["value"], r["f_true_measured"]["ci95"][0], r["f_true_measured"]["ci95"][1],
                      r["est_nuisance_share"], r["est_ci95"][0], r["est_ci95"][1], bool(r["miss_over_020"])))
H = 3.05
fig, bg = canvas(H)
head(bg, H, "Estimated nuisance share against planted fraction")
SZ = 2.22; ax0 = 0.52; ay0 = 0.42
ax = axes(fig, ax0, ay0, SZ, SZ, (-0.2, 1.0), (-0.2, 1.0), [0, 0.25, 0.5, 0.75, 1])
ax.spines["left"].set_visible(True); ax.set_yticks([0, 0.25, 0.5, 0.75, 1]); ax.set_yticklabels(["0", "0.25", "0.5", "0.75", "1"]); ax.grid(axis="y", color=GRID, lw=0.5)
ax.fill_between([-0.2, 1.0], [-0.2 - TOL_MISS, 1.0 - TOL_MISS], [-0.2 + TOL_MISS, 1.0 + TOL_MISS], color="#ECECEC", zorder=0, lw=0)
ax.plot([-0.2, 1.0], [-0.2, 1.0], color=INK, lw=0.8, ls=(0, (2.2, 1.6)), zorder=1)
MK = {"L64": "o", "Lfull": "s"}
for L in ("L64", "Lfull"):
    for (x, xl, xh, yv, yl, yh, miss) in G3[L]:
        ax.plot([xl, xh], [yv, yv], color=PE, lw=1.4, solid_capstyle="round", zorder=3)
        ax.plot([x, x], [yl, yh], color=PE, lw=1.4, solid_capstyle="round", zorder=3)
        ax.plot([x], [yv], marker=MK[L], ms=5.4, mfc="white" if miss else PE, mec=PE if miss else "white", mew=1.3 if miss else 0.6, zorder=4, ls="none")
ax.text(0.80, 0.74, "estimate = planted", fontsize=FS, color=MUTE, rotation=45, rotation_mode="anchor", ha="center", va="top")
# key (direct, right of the plot)
kx = ax0 + SZ + 0.10; ky = ay0 + SZ - 0.08
for lab, mk in (("64", "o"), ("full", "s")):
    bg.plot([kx + 0.04], [ky], marker=mk, ms=5.4, mfc=PE, mec="white", mew=0.6); bg.text(kx + 0.13, ky, lab, fontsize=FS, va="center"); ky -= 0.15
bg.text(kx - 0.02, ay0 + SZ + 0.07, "width", fontsize=FS, color=MUTE, va="center")
ky -= 0.10
bg.plot([kx + 0.04], [ky], marker="o", ms=5.4, mfc="white", mec=PE, mew=1.3); bg.text(kx + 0.13, ky, "miss", fontsize=FS, va="center")
bg.text(ax0 + SZ / 2, ay0 - 0.27, "planted fraction, as measured", fontsize=FS, ha="center", va="center")
bg.text(0.10, ay0 + SZ / 2, "estimated nuisance share", fontsize=FS, rotation=90, ha="center", va="center")
bad += done(fig, "fig_planted_calibration")

# ================================================================== G4: category erasure through frozen image decoders
KM = rio.load_file("ext/km3_decoders_v3", "summary.json"); assert KM is not None
DEC = [("nice_ga", "NICE-GA"), ("brainflora", "BrainFLORA"), ("ubp", "UBP")]
G4 = []
for sc, sclab in (("soft12", "soft12 scorer"), ("exemplar_exact", "exemplar-exact scorer")):
    G4.append((sclab, None))
    for d, name in DEC:
        q = KM["decoders"][d][sc]
        G4.append((name, ((q["kept_full"], q["kept_full_ci"][0], q["kept_full_ci"][1]), (q["kept_permprof"], q["kept_permprof_ci"][0], q["kept_permprof_ci"][1]))))
RH = 0.165; n = len(G4)
H = 0.52 + n * RH + 0.38
fig, bg = canvas(H)
head(bg, H, "Erasing category removes part of the score")
XL = 0.92; ax0 = XL + 0.07
XV1 = WC - 0.62; XV2 = WC - 0.04
aw = XV1 - 0.40 - ax0
yk = H - 0.36
bg.plot([XV1 - tw(fig, "erased") - 0.07], [yk + 0.005], marker="o", ms=5.0, mfc=IMG, mec="white", mew=0.6); bg.text(XV1, yk, "erased", fontsize=FS, ha="right", va="center")
bg.plot([XV2 - tw(fig, "control") - 0.07], [yk + 0.005], marker="s", ms=4.4, mfc="white", mec=INK, mew=0.9); bg.text(XV2, yk, "control", fontsize=FS, ha="right", va="center")
y1 = yk - 0.10; y0 = y1 - n * RH
ax = axes(fig, ax0, y0, aw, n * RH, (0, 1.08), (-0.5, n - 0.5), [0, 0.5, 1])
ax.axvline(1, color=INK, lw=0.8, ls=(0, (2.2, 1.6)), zorder=2)
for i, (lab, v) in enumerate(G4):
    yy = n - 1 - i; yin = y0 + (yy + 0.5) * RH
    if v is None:
        bg.text(0.04, yin - 0.01, lab, fontsize=FS, color=MUTE, style="italic", va="center"); continue
    (e, el, eh), (c, cl, ch) = v
    bg.text(XL, yin, lab, fontsize=FS, ha="right", va="center")
    ax.plot([e, c], [yy, yy], color="#9AA5AE", lw=1.4, solid_capstyle="round", zorder=2)
    ax.plot([cl, ch], [yy, yy], color=INK, lw=1.6, solid_capstyle="round", zorder=3)
    ax.plot([c], [yy], marker="s", ms=4.4, mfc="white", mec=INK, mew=0.9, zorder=4, ls="none")
    ci_dot(ax, e, el, eh, yy, IMG)
    bg.text(XV1, yin, mn(f"{e:.2f}"), fontsize=FS, ha="right", va="center")
    bg.text(XV2, yin, mn(f"{c:.2f}"), fontsize=FS, ha="right", va="center")
bg.text(ax0 + aw / 2, y0 - 0.23, "share of the score kept", fontsize=FS, ha="center", va="center")
bad += done(fig, "fig_category_erasure")
print(f"n: image {n_img}, prediction error {n_pe}, motor imagery {G2[0][5]}")
sys.exit(1 if bad else 0)
