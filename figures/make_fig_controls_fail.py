"""fig_controls_fail.pdf: 'where controls fail' at one column (3.32 in), text floor 6 pt. A one-axis version of panels A-D of
fig_creative_D_control_constellation.pdf; its panel E (planted calibration) is fig_planted_calibration.pdf.
Unlike the older script, every value here is read through results_io (gated): no maximum is typed, and the text rows are the 5-seed card.
The only constant is the protocol tolerance of 0.02 bits.
Run: srun -n1 -t 5 python -I -u -B make_fig_controls_fail.py"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import *  # noqa: F401,F403

plt = style()
F6 = 6.5
TOL = 0.02
FAIL = "#D55E00"
L = rio.load; LF = rio.load_file
S = L("standard/summary.json"); T5 = L("baselines/text_card_v2/text_yield_card.json"); M3 = L("ext/mi_card_v3/primary_lpgo/summary_mi.json")
NI = LF("baselines/ubp_noise_input", "summary.json"); NS = LF("gpu_fill_20261122/ubp_noise_seeds", "summary.json")
PEF = LF("gpu_fill_20261122/pe_eegnet_floor", "summary.json"); MIF = LF("gpu_fill_20261122/mi_eegnet_floor", "summary.json")
for o in (S, T5, M3, NI, NS, PEF, MIF):
    assert o is not None, "source missing or not citable"
im = S["image"]; bf = im["open_weight_models"]["brainflora_native"]; pe = S["prediction_error"]["yield_ds003458"]


def row(label, m, ci, mx=None, kind="blank"):
    return (label, m, ci[0], ci[1], None if mx is None else abs(mx), kind)


IMG = [row("UBP noise, original draw", NI["summary"]["noise_input_bits"]["mean"], NI["summary"]["noise_input_bits"]["ci95"])]
for k in ("80", "40"):
    IMG.append(row(f"UBP noise, new seeds, {k} rep.", NS["bits"][f"n01_k{k}"]["mean"], NS["bits"][f"n01_k{k}"]["ci95"], NS["floor_max_abs_per_participant"][f"n01_k{k}"]))
for rel, lab in (("baselines/ubp_phase_control", "0–2"), ("baselines/ubp_phase_control_s3to6", "3–6"), ("baselines/ubp_phase_control_s7to9", "7–9")):
    P = LF(rel, "summary.json"); assert P is not None
    for kind, nm in (("phase", "phase"), ("shuffle", "shuffle")):
        sv = P["bits"][f"{kind}_k80_seed_values"]
        mx = max((sv[sd][i] for sd in list(sv.keys()) for i in range(len(sv[sd]))), key=abs)
        IMG.append(row(f"UBP {nm}, seeds {lab}", P["bits"][f"{kind}_k80"]["mean"], P["bits"][f"{kind}_k80"]["ci95"], mx))
IMG.append(row("NICE-GA shuffled pairs", im["yield_per_query"]["floor"], im["yield_per_query"]["floor_ci"]))
IMG.append(row("BrainFLORA noise", bf["k80_noise"]["mean"], bf["k80_noise"]["ci"]))
PE = [row("EEGNet circular shift", pe["floor"], pe["floor_ci"]),
      row("EEGNet phase surrogate", PEF["bits"]["phase_seedmean"]["mean"], PEF["bits"]["phase_seedmean"]["ci95"], PEF["phase_max_abs_per_participant_seedmean"]),
      row("EEGNet trial shuffle", PEF["bits"]["shuffle_seedmean"]["mean"], PEF["bits"]["shuffle_seedmean"]["ci95"], PEF["shuffle_max_abs_per_participant_seedmean"])]
TX = []
for key, nm in (("braintranslator", "BrainTranslator noise-trained"), ("dewave_style", "DeWave-style noise-trained")):
    e = T5["summary_over_seeds"][key]["beam/pooled"]["noise_yield_bits"]
    assert len(e["per_seed"]) == 5
    TX.append(row(nm, e["mean"], e["seed_t_ci95"]))
MI = [row("Raw voltage noise", M3["rows"]["raw"]["noise"]["mean"], M3["rows"]["raw"]["noise"]["ci"]),
      row("EEGNet phase surrogate", MIF["bits"]["phase_seedmean"]["mean"], MIF["bits"]["phase_seedmean"]["ci95"], MIF["phase_max_abs_per_participant_seedmean"]),
      row("EEGNet trial shuffle", MIF["bits"]["shuffle_seedmean"]["mean"], MIF["bits"]["shuffle_seedmean"]["ci95"], MIF["shuffle_max_abs_per_participant_seedmean"])]
for m, nm in (("cbramod", "CBraMod"), ("reve", "REVE"), ("labram", "LaBraM"), ("eegpt", "EEGPT")):
    d = LF("gpu_fill_20261122/mi_fm_phase", f"{m}/summary.json"); assert d is not None
    r = d["rows"]["fm|phase_seedmean"]
    MI.append(row(f"{nm} phase surrogate", r["mean"], r["ci95"], d["fm_floor_max_abs_per_participant_seedmean"], "fm"))
BLOCKS = [("image", "Image, bits per query", IMG), ("prediction_error", "Prediction error, bits per trial", PE),
          ("text", "Text, bits per record (5 seeds)", TX), ("motor_imagery", "Motor imagery, bits per trial", MI)]

W = 3.32; RH = 0.118
nrow = sum(len(b[2]) + 1 for b in BLOCKS)
KEY = 0.56
H = 0.24 + nrow * RH + 0.34 + KEY
fig = plt.figure(figsize=(W, H))
bg = fig.add_axes([0, 0, 1, 1]); bg.set_xlim(0, W); bg.set_ylim(0, H); bg.axis("off")
bg.text(0.02, H - 0.11, "Where blank controls fail", fontsize=FS + 0.5, fontweight="bold", va="center")
XL = 1.50; ax0 = XL + 0.06; aw = W - 0.10 - ax0
y1 = H - 0.24; y0 = y1 - nrow * RH
ax = fig.add_axes([ax0 / W, y0 / H, aw / W, (y1 - y0) / H])
ax.set_xscale("symlog", linthresh=TOL, linscale=0.8); ax.set_xlim(-0.03, 0.15); ax.set_ylim(nrow - 0.5, -0.5)
ticks = [-TOL, 0, TOL, 0.1]
ax.set_xticks(ticks); ax.set_xticklabels([("%g" % t).replace("-", "−") for t in ticks], fontsize=F6); ax.minorticks_off(); ax.set_yticks([])
ax.axvspan(-TOL, TOL, color="#ECECEC", zorder=0, lw=0); ax.axvline(0, color=INK, lw=0.8, ls=(0, (2.2, 1.6)), zorder=1)
ax.xaxis.grid(True, color=GRID, lw=0.5); ax.set_axisbelow(True); ax.patch.set_alpha(0)
i = 0
for t, title, rows in BLOCKS:
    yin = y1 - (i + 0.5) * RH
    bg.add_patch(plt.Rectangle((0.02, yin - 0.045), 0.05, 0.09, fc=HUE[t], ec="none"))
    bg.text(0.10, yin, title, fontsize=F6, fontweight="bold", va="center")
    i += 1
    for (lab, m, lo, hi, mx, kind) in rows:
        yin = y1 - (i + 0.5) * RH
        bg.text(XL, yin, lab, fontsize=F6, ha="right", va="center")
        ax.plot([lo, hi], [i, i], color=INK, lw=1.0, zorder=3)
        if kind == "fm" and lo > 0:
            ax.plot(m, i, marker="D", ms=4.2, color=FAIL, mec="white", mew=0.4, zorder=5)
        else:
            ax.plot(m, i, marker="o", ms=3.4, color=INK, mec="white", mew=0.4, zorder=5)
        if mx is not None:
            bad = mx > TOL
            ax.plot(mx, i, marker="^", ms=4.6 if bad else 3.9, color=FAIL if bad else "white", mec=FAIL if bad else MUTE, mew=0.7, zorder=5, clip_on=False)
        i += 1
bg.text(ax0 + aw / 2, y0 - 0.215, "score without usable EEG (bits)", fontsize=F6, ha="center", va="center")
ky = KEY - 0.10
for mk, col, hollow, txt in (("o", INK, False, "cohort mean, 95% interval"), ("^", MUTE, True, "largest single value, inside the tolerance (grey)"),
                             ("^", FAIL, False, "largest single value, outside the tolerance"), ("D", FAIL, False, "phase-surrogate score above zero")):
    bg.plot([0.10], [ky], marker=mk, ms=4.4, color="white" if hollow else col, mec=col if hollow else "white", mew=0.7 if hollow else 0.4)
    bg.text(0.20, ky, txt, fontsize=F6, va="center"); ky -= 0.118
nbad = check_text(fig, min_pt=6.0)
save(fig, "fig_controls_fail")
sys.exit(1 if nbad else 0)
