"""fig_diagnostic.pdf: the Diagnostic Audit Grid (D0-D9 x four targets, symbol + colour) beside the yield-profile scores.
Full text width (7.0 in). Every number is read through results_io (gated); grades come from paper/v2/rubric/grades.json
(written by paper/v2/rubric/grade.py) and are checked against paper/v2/tables/tab_profile.tex before drawing.
Run: srun -n1 -t 5 python -I -u make_fig_diagnostic.py"""
import re
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import *  # noqa: F401,F403
from matplotlib.patches import Rectangle, FancyBboxPatch
from matplotlib.lines import Line2D

plt = style()
get = rio.get

S = rio.load("standard/summary.json")
PFD = rio.load("ext/ueyc_v2/profile.json")
EA = get(rio.load("ext/eu_bac_v2/full_final.json"), "A_ubp_budget")
NI = rio.load_file("baselines/ubp_noise_input", "summary.json")
KM = rio.load_file("ext/km3_decoders_v3", "summary.json")
T5 = rio.load("baselines/text_card_v2/text_yield_card.json")          # 5-seed text card
M3 = rio.load("ext/mi_card_v3/primary_lpgo/summary_mi.json")
REVE = rio.load_file("ext/mi_fm_eogreg", "reve/summary.json")
UIMG = rio.load_file("gpu_fill_20261123/image_untrained", "summary.json")
UPE = rio.load_file("gpu_fill_20261123/pe_untrained", "summary.json")
for nm, o in dict(S=S, PFD=PFD, EA=EA, NI=NI, KM=KM, T5=T5, M3=M3, REVE=REVE, UIMG=UIMG, UPE=UPE).items():
    assert o is not None, f"source {nm} missing or not citable"

im = S["image"]; bf = im["open_weight_models"]["brainflora_native"]
ykey = "baseline_yield" if "baseline_yield" in EA else "twin_yield"
pe = S["prediction_error"]
PF4 = PFD["profile"][4]


def ci(o, k="ci"):
    c = get(o, k)
    return None if c is None else (c[0], c[1])


def tcell(key, sk, what):
    e = T5["summary_over_seeds"][key]["beam/" + sk][what]
    return e["mean"], (e["seed_t_ci95"][0], e["seed_t_ci95"][1])


def row(label, y, c, blank=None, base=None, rep=None, usage=None, nuis="–", d=3):
    return dict(label=label, y=y, ci=c, blank=blank, base=base, rep=rep, usage=usage, nuis=nuis, d=d)


def pct(x):
    return f"{100 * x:.1f}%"


kept = lambda d: "kept %.2f" % KM["decoders"][d]["soft12"]["kept_full"]
ubp_ratio = PFD["profile"][0]["own_reference_fraction"]["recomputed_gpu"]["ratio_of_means"]
ROWS = {}
ROWS["image"] = [
    row("UBP", EA["raw"]["40"]["mean"], (EA["raw"]["40"]["lo"], EA["raw"]["40"]["hi"]), NI["summary"]["noise_input_bits"]["mean"],
        EA[ykey]["40"]["mean"], EA["raw"]["40"]["mean"] / ubp_ratio, pct(PFD["profile"][0]["own_reference_fraction"]["value"]), kept("ubp"), 2),
    row("NICE-GA", im["yield_per_query"]["mean"], ci(im["yield_per_query"]), im["yield_per_query"]["floor"],
        UIMG["nice_ga"]["k40"]["mean"], im["ceiling"]["mean"], pct(im["fraction"]["value"]), kept("nice_ga"), 2),
    row("BrainFLORA", bf["k40"]["mean"], ci(bf["k40"]), bf["k80_noise"]["mean"],
        UIMG["brainflora"]["k40"]["mean"], bf["ceiling_k40"]["mean"], pct(bf["fraction_k40"]["value"]), kept("brainflora"), 2),
]
ROWS["text"] = []
for key, lab in (("braintranslator", "BrainTranslator"), ("dewave_style", "DeWave-style")):
    y_, c_ = tcell(key, "pooled", "eeg_yield_bits")
    # same choice as tables/tab_core_main.tex (common.TEXT_BLANK): blank = shuffled pairs and baseline = noise-trained control, or
    # blank = noise-trained control and baseline = random-init decoder
    tb_, tbase_ = (("eeg_floor_bits", "noise_yield_bits") if TEXT_BLANK == "shuffled" else ("noise_yield_bits", "random_init_yield_bits"))
    ROWS["text"].append(row(lab, y_, c_, tcell(key, "pooled", tb_)[0], tcell(key, "pooled", tbase_)[0],
                            None, "not usable", "SR %+.3f" % tcell(key, "within_SR", "eeg_minus_noise")[0]))
# prediction error: the read-out reproducibility is not an EEG replicate, so no usage is shown (the two ratios stay in Table tab:profile)
NOREP = "no replicate"
ROWS["prediction_error"] = [
    row("EEGNet", pe["yield_ds003458"]["mean"], ci(pe["yield_ds003458"]), pe["yield_ds003458"]["floor"], UPE["untrained_seedmean"]["mean"],
        None, NOREP),
    row("Raw-ERP ridge", PF4["raw_yield"]["value"], ci(PF4["raw_yield"]), PF4["noise_input_floor"]["value"], None, None,
        NOREP, "kept %.2f" % PF4["specific_yield"]["eye_quiet_kept_ratio"]["value"]),
]
mi = lambda k: (M3["rows"][k]["eeg"]["mean"], ci(M3["rows"][k]["eeg"]))
# motor imagery: the loss under proxy regression is specific to the raw-voltage read-out (EEGNet, tangent space and band power keep
# their yield), so the column names the read-out and carries no ocular label
ROWS["motor_imagery"] = [
    row("Raw voltage", *mi("raw"), M3["rows"]["raw"]["noise"]["mean"], None, None, "anchor fails"),
    row("Raw, regressed", *mi("raw_eogreg"), M3["rows"]["raw_eogreg"]["noise"]["mean"], None, None, "anchor fails", "raw voltage only"),
    row("EEGNet, regressed", *mi("eegnet_eogreg"), M3["rows"]["eegnet_eogreg"]["noise"]["mean"], None, None, "anchor fails", "yield kept"),
    row("REVE, regressed", REVE["rows"]["fm"]["eeg"]["mean"], ci(REVE["rows"]["fm"]["eeg"]), REVE["rows"]["fm"]["noise"]["mean"],
        REVE["rows"]["twin"]["eeg"]["mean"], None, "anchor fails"),
]

# ------------------------------------------------------------------ grades: paper/v2/rubric/grades.json
# The grid is computed by paper/v2/rubric/grade.py from rubric.json (criteria of tables/tab_crit_po.tex) and the result files.
# The grid printed in paper/v2/tables/tab_profile.tex must agree with it (grade.py --apply writes both).
import json
GJ = json.load(open(ROOT / "paper" / "v2" / "rubric" / "grades.json"))
GKEY = {"supported": "Pass", "qualified": "Partial", "violated": "Fail", "undefined": "undefined"}
GR = {t: [GKEY[g] for g in GJ["grid"][t]] for t in ORDER}
SYM = {r"\checkmark": "Pass", r"$\circ$": "Partial", r"$\times$": "Fail", "--": "undefined"}
tab = (TABLES / "tab_profile.tex").read_text()
for t, nm in (("image", "image"), ("text", "text"), ("prediction_error", "prediction error"), ("motor_imagery", "motor imagery")):
    line = re.search(r"Audit grid, " + nm + r":\} (.*?)\} \\\\", tab).group(1)
    printed = [SYM[c] for c in re.findall(r"D\d\\,(\\checkmark|\$\\circ\$|\$\\times\$|--)", line)]
    assert len(GR[t]) == 10 and GR[t] == printed, f"grid mismatch for {t}: grades.json {GR[t]} vs tab_profile.tex {printed}"
print("grades read from rubric/grades.json; agree with tab_profile.tex")

DNAME = ["Partition integrity", "Target specification", "Blank control", "Detectability", "Replicate anchoring",
         "Input sufficiency", "Nuisance sensitivity", "Read-out adequacy", "Sampling adequacy", "Replicability"]


# ------------------------------------------------------------------ canvas (inches)
RH, TH, AXH, GAP = 0.148, 0.18, 0.15, 0.07   # row height, title line, tick-label band, gap (inches)
CHANCE_LS = (0, (2.2, 1.6))
HEAD = {"image": "Image", "text": "Text", "prediction_error": "Pred.\nerror", "motor_imagery": "Motor\nimagery"}
SHORT = {"Raw, regressed": "Raw, regr.", "EEGNet, regressed": "EEGNet, regr.", "REVE, regressed": "REVE, regr."}
PAN = {"image": ("bits per 40-rep. query", (0, 8.2), [0, 2, 4, 6, 8], "category erased"),
       "text": ("bits per record, 5 seeds", (-0.062, 0.125), [-0.05, 0, 0.05, 0.1], "within task"),
       "prediction_error": ("bits per trial", (0, 0.2), [0, 0.05, 0.1, 0.15], "eye-quiet trials"),
       "motor_imagery": ("bits per trial", (0, 0.5), [0, 0.1, 0.2, 0.3, 0.4], "proxy regression")}


def build(name, panel_b, compact=False):
    """panel_b: "bars" (fig_diagnostic.pdf), "dumbbell" (same rows as a with/without-EEG dumbbell) or None (the grid alone, one column)."""
    grid_only = panel_b is None
    dumb = panel_b == "dumbbell"
    rh, th, axh, gp = (0.138, 0.165, 0.14, 0.045) if compact else (RH, TH, AXH, GAP)   # compact: same content, 7 pt text, less height
    KEYH = 0.0 if grid_only else 0.17              # second key row under panel b
    if grid_only:
        W, PITCH = 3.32, 0.226
        XD, XNAME, X0, CW = 0.02, 0.185, 1.20, 0.53
        H = 0.57 + 0.045 + 10 * PITCH + 0.26
    else:
        W, H, PITCH = (7.0, 3.80, 0.2875) if compact else (7.0, 3.95 + KEYH, 0.296 + KEYH / 10)
        XD, XNAME, X0, CW = 0.02, 0.205, 1.23, 0.555
    fig = plt.figure(figsize=(W, H))
    bg = fig.add_axes([0, 0, 1, 1]); bg.set_xlim(0, W); bg.set_ylim(0, H); bg.axis("off")
    T = lambda s, **kw: tw(fig, s, **kw)

    def chip(x, y, w, h, fc, text, tc="white", fs=FS):
        bg.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.035", fc=fc, ec="none", zorder=2))
        bg.text(x + w / 2, y + h / 2, text, ha="center", va="center", color=tc, fontsize=fs, fontweight="bold", linespacing=0.95, zorder=3)

    # ---- (a) grid
    if grid_only:
        bg.text(0.02, H - 0.10, "Diagnostic Audit Grid", fontsize=FS + 0.5, fontweight="bold", va="center")
    else:
        bg.text(0.02, H - 0.10, "a", fontsize=9, fontweight="bold", va="center")
        bg.text(0.20, H - 0.10, "Diagnostic Audit Grid", fontsize=FS + 0.5, fontweight="bold", va="center")
    YH = H - 0.57            # bottom of header chips
    for j, t in enumerate(ORDER):
        chip(X0 + j * CW + 0.015, YH, CW - 0.03, 0.31, HUE[t], HEAD[t])
    for i in range(10):
        yc = YH - 0.045 - (i + 0.5) * PITCH
        bg.text(XD, yc, f"D{i}", fontsize=FS, fontweight="bold", va="center")
        bg.text(XNAME, yc, DNAME[i], fontsize=FS, va="center")
        for j, t in enumerate(ORDER):
            lab, col, sym = GRADE[GR[t][i]]
            bg.add_patch(Rectangle((X0 + j * CW + 0.015, yc - PITCH / 2 + 0.012), CW - 0.03, PITCH - 0.024, fc=col, ec="none"))
            bg.text(X0 + j * CW + CW / 2, yc - 0.005, sym, ha="center", va="center", fontsize=9.5, color=INK)
    # grade key (symbol + colour + word), spread over the grid width
    ky = 0.11 + KEYH / 2
    labs = [GRADE[g][0] for g in ("Pass", "Partial", "Fail", "undefined")]
    gap = (X0 + 4 * CW - 0.02 - sum(0.23 + T(l) for l in labs)) / 3
    assert gap > 0.04, "grade key does not fit"
    x = 0.02
    for g in ("Pass", "Partial", "Fail", "undefined"):
        lab, col, sym = GRADE[g]
        bg.add_patch(Rectangle((x, ky - 0.085), 0.19, 0.17, fc=col, ec="none"))
        bg.text(x + 0.095, ky - 0.005, sym, ha="center", va="center", fontsize=8.5)
        bg.text(x + 0.23, ky, lab, fontsize=FS, va="center")
        x += 0.23 + T(lab) + gap
    if grid_only:
        nbad = check_text(fig); save(fig, name); return nbad

    # ---- (b) yield profile
    XB = 3.57                       # left edge of panel b
    XL = XB + max(T(SHORT.get(r["label"], r["label"])) for t in ORDER for r in ROWS[t]) + 0.02
    XA0 = XL + 0.06
    XN = W - 0.02 - max([T(r["nuis"].replace("-", "−")) for t in ORDER for r in ROWS[t]] + [T("nuisance"), T("category erased", style="italic")])
    UW = max([T(r["usage"]) for t in ORDER for r in ROWS[t]] + [T("usage")])
    XU = XN - 0.10 - UW / 2         # usage column centre
    XA1 = XU - UW / 2 - 0.10
    if dumb:                        # room for the tinted cards
        XB += 0.05; XL += 0.05; XA0 += 0.09; XN -= 0.04; XU -= 0.04; XA1 -= 0.04
    bg.text(3.57, H - 0.10, "b", fontsize=9, fontweight="bold", va="center")
    bg.text(3.75, H - 0.10, "With EEG against without EEG, one axis per target" if dumb else "Yield profile, one axis per target",
            fontsize=FS + 0.5, fontweight="bold", va="center")
    yh = H - 0.36
    bg.text((XA0 + XA1) / 2, yh, "yield", fontsize=FS, color=MUTE, ha="center", va="center")
    bg.text(XU, yh, "usage", fontsize=FS, color=MUTE, ha="center", va="center")
    bg.text(XN, yh, "nuisance", fontsize=FS, color=MUTE, ha="left", va="center")
    if not dumb:
        bg.plot([XB, W - 0.02], [yh - 0.10, yh - 0.10], color=GRID, lw=0.6)
    ytop = yh - 0.16
    for t in ORDER:
        sub, xlim, ticks, nz = PAN[t]
        rows = ROWS[t]; n = len(rows)
        if dumb and t == "motor_imagery": ticks = [0, 0.2, 0.4]      # narrower axis in the card layout
        y1 = ytop - th; y0 = y1 - n * rh
        if dumb:                    # soft tinted card behind the block
            bg.add_patch(FancyBboxPatch((3.57, y0 - axh - 0.005), W - 0.01 - 3.57, ytop + 0.015 - (y0 - axh - 0.005),
                                        boxstyle="round,pad=0,rounding_size=0.06", fc=TINT[t], ec="none", zorder=0))
        # title line: coloured chip, unit, nuisance name
        wchip = T(NAME[t], fontweight="bold") + 0.12
        chip(XB, ytop - th + 0.025, wchip, th - 0.035, HUE[t], NAME[t])
        bg.text(XB + wchip + 0.06, ytop - th / 2 + 0.005, sub, fontsize=FS, color=MUTE, va="center")
        bg.text(XN, ytop - th / 2 + 0.005, nz, fontsize=FS, color=MUTE, va="center", style="italic")
        ax = fig.add_axes([XA0 / W, y0 / H, (XA1 - XA0) / W, (y1 - y0) / H])
        ax.set_xlim(*xlim); ax.set_ylim(-0.5, n - 0.5); ax.set_yticks([]); ax.set_xticks(ticks)
        ax.set_xticklabels([("%g" % v).replace("-", "−") for v in ticks])
        ax.grid(axis="x", color="white" if dumb else GRID, lw=0.7 if dumb else 0.5); ax.set_axisbelow(True); ax.patch.set_alpha(0)
        ax.plot([0, 0], [-0.72, n - 0.28], color=INK, lw=0.8, ls=CHANCE_LS, zorder=2.5, clip_on=False)   # chance: a decoder that knows nothing scores 0 bits
        span = xlim[1] - xlim[0]
        for i, r in enumerate(rows):
            yy = n - 1 - i
            yin = y0 + (yy + 0.5) * rh
            bg.text(XL, yin, SHORT.get(r["label"], r["label"]), fontsize=FS, ha="right", va="center")
            b = r["blank"] if r["blank"] is not None else 0.0
            lo, hi = r["ci"]
            if r["rep"] is not None:                                  # gauge rail: blank to replicate
                ax.barh(yy, r["rep"] - b, left=b, height=0.74, color="white" if dumb else RAIL, ec="none", zorder=1)
                ax.plot([b, r["rep"]], [yy, yy], color="#9AA5AE", lw=0.6, zorder=2)
                ax.plot([r["rep"]], [yy], marker="D", ms=4.6, mfc=REPL, mec=INK, mew=0.6, zorder=6, ls="none", clip_on=False)
            has_base = r["base"] is not None and r["base"] > 0.012 * span
            if dumb:
                if has_base:                                          # baseline, hatched band from chance
                    ax.barh(yy, r["base"], left=0, height=0.56, fc="white", ec=HUE[t], hatch="//////", lw=0.5, zorder=2.8)
                ax.plot([b, r["y"]], [yy, yy], color=HUE[t], lw=2.4, solid_capstyle="round", zorder=4)          # the dumbbell bar
                ax.plot([lo, hi], [yy, yy], color=INK, lw=0.9, zorder=5, solid_capstyle="butt")
                for e in (lo, hi):
                    ax.plot([e, e], [yy - 0.16, yy + 0.16], color=INK, lw=0.8, zorder=5)
                if r["blank"] is not None:                            # without EEG: hollow circle
                    ax.plot([b], [yy], marker="o", ms=4.4, mfc="white", mec=INK, mew=0.9, zorder=7, ls="none", clip_on=False)
                ax.plot([r["y"]], [yy], marker="o", ms=5.2, mfc=HUE[t], mec="white", mew=0.6, zorder=8, ls="none")  # with EEG: filled circle
            else:
                ax.barh(yy, r["y"] - b, left=b, height=0.50, color=HUE[t], ec="none", zorder=3)
                if has_base:                                          # untrained-encoder baseline, hatched
                    ax.barh(yy, r["base"], left=0, height=0.50, fc=TINT[t], ec=HUE[t], hatch="//////", lw=0.5, zorder=4)
                ax.plot([lo, hi], [yy, yy], color=INK, lw=0.9, zorder=5, solid_capstyle="butt")
                for e in (lo, hi):
                    ax.plot([e, e], [yy - 0.16, yy + 0.16], color=INK, lw=0.8, zorder=5)
                if r["blank"] is not None:                            # blank yield: black tick
                    ax.plot([b, b], [yy - 0.40, yy + 0.40], color=INK, lw=1.5, zorder=7, solid_capstyle="butt")
            s = (("%." + str(r["d"]) + "f") % r["y"]).replace("-", "−")
            ax.text(hi + 0.025 * span, yy, s, fontsize=FS, va="center", ha="left", zorder=9)   # direct label, right of the interval
            u = r["usage"]
            bg.text(XU, yin, u, fontsize=FS, ha="center", va="center", color=INK if u.endswith("%") else MUTE)
            z = r["nuis"].replace("-", "−")
            bg.text(XN, yin, z, fontsize=FS, ha="left", va="center", color=MUTE if z == "–" else INK)
        ytop = y0 - axh - gp
    assert ytop + gp > 0.22 + KEYH, "panels run into the key"

    # mark key for panel b, two rows spread over the panel width; wording as in tables/tab_core_main.tex
    def k_yield(x, ky):
        if dumb:
            bg.plot([x + 0.02, x + 0.17], [ky, ky], color=MUTE, lw=2.4, solid_capstyle="round")
            bg.plot([x + 0.17], [ky], marker="o", ms=5.2, mfc=MUTE, mec="white", mew=0.6)
        else:
            bg.add_patch(Rectangle((x, ky - 0.04), 0.20, 0.08, fc=MUTE, ec="none"))
    def k_ci(x, ky):
        bg.plot([x + 0.03, x + 0.19], [ky, ky], color=INK, lw=0.9)
        for e in (x + 0.03, x + 0.19): bg.plot([e, e], [ky - 0.03, ky + 0.03], color=INK, lw=0.8)
    def k_blank(x, ky):
        if dumb:
            bg.plot([x + 0.11], [ky], marker="o", ms=4.4, mfc="white", mec=INK, mew=0.9)
        else:
            bg.plot([x + 0.11, x + 0.11], [ky - 0.07, ky + 0.07], color=INK, lw=1.5)
    def k_chance(x, ky):
        bg.plot([x + 0.11, x + 0.11], [ky - 0.075, ky + 0.075], color=INK, lw=0.8, ls=CHANCE_LS)
    def k_base(x, ky):
        bg.add_patch(Rectangle((x, ky - 0.04), 0.20, 0.08, fc="white" if dumb else "#EEEEEE", ec=MUTE, hatch="//////", lw=0.5))
    def k_rep(x, ky):
        bg.add_patch(Rectangle((x, ky - 0.06), 0.20, 0.12, fc=RAIL, ec="none")); bg.plot([x + 0.14], [ky], marker="D", ms=4.6, mfc=REPL, mec=INK, mew=0.6)
    KEYS = [(0.11 + KEYH, [(k_yield, "with EEG (yield)"), (k_blank, "without EEG (blank)"), (k_chance, "chance")]),
            (0.11, [(k_ci, "95% interval"), (k_base, "untrained encoder"), (k_rep, "replicate yield")])]
    for ky, items in KEYS:
        gap = (W - 0.02 - 3.57 - sum(0.24 + T(l) for _, l in items)) / (len(items) - 1)
        assert gap > 0.06, "mark key does not fit"
        x = 3.57
        for draw, lab in items:
            draw(x, ky); bg.text(x + 0.24, ky, lab, fontsize=FS, va="center"); x += 0.24 + T(lab) + gap
    nbad = check_text(fig); save(fig, name); return nbad


ONLY_V2 = "--v2-only" in sys.argv
bad = 0
if not ONLY_V2:
    bad += build("fig_diagnostic", "bars")                 # the bars variant
    bad += build("fig_diagnostic_dumbbell", "dumbbell")   # the figure the frozen paper includes
    bad += build("fig_grid", None)                        # the grid alone, one column wide
bad += build("fig_diagnostic_dumbbell_v2", "dumbbell", compact=True)
sys.exit(1 if bad else 0)
