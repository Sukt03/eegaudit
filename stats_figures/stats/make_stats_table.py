"""Writes paper/tab_stats.tex from results/stats/paper_stats.json (every cell read from the file; no typed numbers)."""
import json, math
from pathlib import Path
S = json.load(open("/path/to/eegaudit/results/stats/paper_stats.json"))
idx = {r["name"]: r for rows in S.values() for r in rows}
def p(x):
    if x != x: return "--"
    if x >= 0.001: return f"{x:.3f}"
    e = math.floor(math.log10(x)); m = x / 10 ** e
    return f"${m:.1f}{{\\times}}10^{{{e}}}$"
def f(x, d=2): return "--" if x is None or x != x else (f"{x:+.{d}f}")
ROWS = [  # (section, label, key, unit-decimals, show TOST)
 ("Image (THINGS-EEG2, NICE-GA; unit: participant)", None, None, 0, False),
 (None, "EEG vs noise input, bits ($k{=}80$)", "image_k80_eeg_vs_noise_bits", 2, False),
 (None, "Split-half top-1: trained space vs raw voltage", "image_splithalf_trained_vs_raw_top1", 3, False),
 (None, "Target: CLIP vs pixels, bits (paired)", "image_target_clip_vs_pixels_bits", 2, False),
 (None, "Target: DINOv2 vs pixels, bits (paired)", "image_target_dinov2_vs_pixels_bits", 2, False),
 (None, "Target: DINOv2 vs CLIP, bits (paired)", "image_target_dinov2_vs_clip_bits", 2, False),
 (None, "Recording: 32 el.\\ retrained vs full", "image_recording_occpar32_retrained_vs_full", 2, True),
 (None, "Recording: 0.6\\,s retrained vs full", "image_recording_w06_retrained_vs_full", 2, True),
 (None, "Recording: 16 el.\\ retrained vs full (pilot)", "image_recording_occpar16_retrained_vs_full", 2, False),
 (None, "Recording: 0.4\\,s retrained vs full (pilot)", "image_recording_w04_retrained_vs_full", 2, False),
 (None, "Control: shuffled-label encoder vs full", "image_control_shuffle_retrained_vs_full", 2, False),
 (None, "Yield: input $L_{90}$ vs representation $L_{90}$ ($\\log_{10}$ dims)", "yield_image_L90_input_vs_representation_log10", 2, False),
 (None, "Yield: input rank 100 vs full width (InfoNCE)", "yield_image_input_L100_vs_full_InfoNCE", 2, False),
 (None, "Usable fraction ($k{=}40$): InfoNCE vs Wolpaw", "yield_image_fraction_k40_InfoNCE_vs_Wolpaw", 3, False),
 ("Sentence (ZuCo SR)", None, None, 0, False),
 (None, "Ridge: EEG vs noise, 2-way (unit: participant)", "text_ridge_participants_eeg_vs_noise_two_way", 3, False),
 (None, "BrainTranslator: EEG vs noise-trained control, greedy top-1", "text_braintranslator_greedy_top1_eeg_vs_noise", 3, False),
 (None, "Yield (InfoNCE, full width): EEG vs noise-trained control, SR", "yield_text_SR_full_eeg_vs_noise_InfoNCE", 3, False),
 ("Prediction error (ds003458; Fisher $z$ of LOPO coupling; unit: participant)", None, None, 0, False),
 (None, "EEGNet vs zero", "pe_eegnet_fisherz_vs_zero", 2, False),
 (None, "LaBraM pooled vs zero", "pe_labram_fisherz_vs_zero", 2, False),
 (None, "CBraMod: unpooled tokens vs pooled", "pe_fmtok_vs_fm_fisherz", 2, False),
 (None, "REVE tokens vs EEGNet", "pe_revetok_vs_eegnet_fisherz", 2, False),
 (None, "EEGNet vs raw ERP", "pe_eegnet_vs_raw_fisherz", 2, False),
 (None, "Filter ensemble (ours) vs EEGNet", "pe_ens_vs_eegnet_fisherz", 2, False),
 (None, "EEGNet usable fraction vs zero", "yield_pe_eegnet_fraction_vs_zero", 3, False),
 (None, "Yield, raw ERP (Gaussianised): PCA-64 vs full", "yield_pe_gauss_raw_L64_vs_full", 3, False),
 (None, "Yield, raw ERP (Gaussianised): PCA-64 vs random projection", "yield_pe_gauss_raw_pca_vs_random_L64", 3, False),
]
L = [r"\begin{table*}[t]\centering\footnotesize\setlength{\tabcolsep}{4pt}", r"\begin{tabular}{@{}lrrrrrr@{}}\toprule",
     r"Contrast & $n$ & $\Delta$ [95\% CI] & $d_z$ & $+$/$n$ & $p_{\mathrm{Holm}}$ & $p_{\mathrm{TOST}}$ \\\midrule"]
for sec, lab, key, dec, tost in ROWS:
    if sec: L.append(r"\multicolumn{7}{@{}l}{\textit{" + sec + r"}} \\"); continue
    r = idx[key]; ci = r.get("diff_ci") or []
    cis = f"{f(r['diff'], dec)} [{ci[0]:+.{dec}f}, {ci[1]:+.{dec}f}]" if ci else f"{f(r['diff'], dec)}"
    pos = f"{r['n_positive']}/{r['n']}" if "n_positive" in r else "--"
    dz = f"{r['dz']:.2f}" if r.get("dz") == r.get("dz") and r.get("dz") is not None else "--"
    pt = p(r["p_tost"]) + r" ($\pm$0.05)" if tost and "p_tost" in r else "--"
    L.append(f"\\quad {lab} & {r['n']} & {cis} & {dz} & {pos} & {p(r.get('p_holm', r['p_t']))} & {pt} \\\\")
L += [r"\bottomrule\end{tabular}",
      r"\caption{\textbf{Statistical tests.} $\Delta$: mean paired difference (first minus second) with a 10{,}000-sample bootstrap CI over the unit of analysis; $d_z$: mean over standard deviation of the differences; $+/n$: units with a positive difference; $p_{\mathrm{Holm}}$: paired $t$ test (one-sided where the floor or the direction is fixed in advance), Holm-adjusted within each modality; Fisher exact test for BrainTranslator over the "
      r"pooled test queries. $p_{\mathrm{TOST}}$: two one-sided tests for equivalence within $\pm$0.05 bits, the margin fixed before the test; $p_{\mathrm{TOST}}>0.05$ means equivalence is not shown. Generated from \texttt{results/stats/paper\_stats.json}.}\label{tab:stats}\end{table*}"]
Path("/path/to/eegaudit/paper/tab_stats.tex").write_text("\n".join(L) + "\n"); print("\n".join(L[:6]))
