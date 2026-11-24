"""V2 (22 Nov 2026, after review): UBP rows read the decided soft12 summary (ubp_summary_soft12.json) and EU-BAC from ext/eu_bac_v2; kept values labelled "share";
own-reference convention stated per row; three noise cells filled from m7/mi_fm; note on 4.888/4.887/4.928. Outputs: results/ext/ueyc_v2/profile.json, outputs/FINAL/UEYC_PROFILE_v2_20261122.md.
UEYC: Universal EEG Yield Card, PROFILE + LADDER + reliability + scaling, read from SAVED results only (no model run). 22 Nov 2026.
Rules built into the output (METHOD_UEYC_20261122.md):
  * raw bits are NEVER ranked across targets (different scorer, candidate set, unit). Tables are grouped by target; JSON carries rank_across_targets = "forbidden".
  * "specific yield" = nuisance-adjusted (target yield minus / kept-after the nuisance read-out under the SAME scorer and candidate set). It is descriptive, not causal.
  * a source is used as 'citable' only if paper/figs/results_io.CITABLE says True and (where named) the code hash matches the reviewed prefix; anything else is
    'reviewed' (review PASS noted in RESULTS.md, not yet in CITABLE) or 'pending'. Nothing is invented: a missing quantity is {"value": null, "why": ...}.
GPU use: participant bootstrap (10000, seed 0) for the derived quantities that are NOT already saved with a CI (MI specific yield, MI nuisance-regressed kept share,
  UBP own-ceiling fraction and baseline share recomputed from per-participant arrays as a consistency check).
Usage: srun -n1 -t 10 python ueyc_profile.py [--smoke]   (smoke = first 2 participants, writes smoke.json/smoke.md only)
"""
import argparse, glob, hashlib, json, sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path("/path/to/eegaudit")
R = ROOT / "results"
sys.path.insert(0, str(ROOT / "paper" / "figs"))
from results_io import CITABLE  # read-only use
assert torch.cuda.is_available(), "GPU required (run through srun --overlap, never on the login node)"
dev = "cuda"
SHA = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

REVIEWED = {"baselines/ubp_noise_input/summary.json": "BUGREVIEW PASS; METHODS CONDITIONAL PASS (report real-EEG std; k40 approximate)"}
HASH = {"ext/eu_bac_v2/full.json": ("script_sha256", "d0ce3e57"), "baselines/ubp_noise_input/summary.json": ("script_sha256", "11e33871"),
        "baselines/ubp_phase_control/summary.json": ("code_sha256", "9c51277e"), "ext/mi_card_v3/primary_lpgo/summary_mi.json": ("code_sha256", "31186ea6"),
        "ext/mi_fm_eogreg/reve/summary.json": ("code_sha256", "fa199719")}
_cache = {}


def J(rel):
    if rel not in _cache:
        p = R / rel
        _cache[rel] = json.load(open(p)) if p.exists() else None
    return _cache[rel]


def status(rel):
    """citable | reviewed | pending (+ reason)."""
    if J(rel) is None:
        return "missing"
    if "soft12" in rel:
        return "pending (soft12 summary: no exact-hash independent review yet; CITABLE only lists the old ubp_summary.json by directory prefix)"
    best, val = -1, False
    for k, v in CITABLE.items():
        if (rel == k or rel.startswith(k + "/")) and len(k) > best:
            best, val = len(k), v
    st = "citable" if val else ("reviewed" if rel in REVIEWED else "pending")
    if rel in HASH:
        key, pre = HASH[rel]
        if not str(J(rel).get(key, "")).startswith(pre):
            return "pending (code hash differs from reviewed %s)" % pre
    return st


def F(value, src, ci=None, note=None, **kw):
    """one profile field: value, optional CI, source file and its status."""
    d = {"value": value, "src": "results/" + src, "status": status(src)}
    if ci is not None:
        d["ci"] = ci
    if note:
        d["note"] = note
    d.update(kw)
    return d


def NA(why):
    return {"value": None, "why": why}


def boot(x, B=10000, seed=0):
    x = torch.tensor(np.asarray(x, float), dtype=torch.float64, device=dev)
    g = torch.Generator(device=dev).manual_seed(seed)
    idx = torch.randint(0, len(x), (B, len(x)), device=dev, generator=g)
    m = x[idx].mean(1)
    q = torch.quantile(m, torch.tensor([.025, .975], dtype=torch.float64, device=dev))
    return {"mean": x.mean().item(), "ci": [q[0].item(), q[1].item()], "n": len(x)}


def ratio_boot(num, den, B=10000, seed=1):
    n = torch.tensor(np.asarray(num, float), dtype=torch.float64, device=dev)
    d = torch.tensor(np.asarray(den, float), dtype=torch.float64, device=dev)
    g = torch.Generator(device=dev).manual_seed(seed)
    idx = torch.randint(0, len(n), (B, len(n)), device=dev, generator=g)
    r = n[idx].mean(1) / d[idx].mean(1)
    q = torch.quantile(r, torch.tensor([.025, .975], dtype=torch.float64, device=dev))
    return {"ratio_of_means": (n.mean() / d.mean()).item(), "ci": [q[0].item(), q[1].item()], "n": len(n)}


def run(nsub):
    sl = (lambda a: a[:nsub]) if nsub else (lambda a: a)
    P = []  # profile rows
    # ---------------------------------------------------------------- IMAGE: UBP released (THINGS-EEG2)
    u, ub = "baselines/ubp/ubp_summary_soft12.json", J("baselines/ubp/ubp_summary_soft12.json")
    rel, ri = ub["released"], ub["random_init_ridge"]
    k80, k40, ceil, rk80 = [sl(rel[k]["per_participant"]) for k in ("k80_bits", "k40_bits", "ceiling_k40_bits")] + [sl(ri["k80_bits"]["per_participant"])]
    # consistency recompute on GPU from the saved per-participant arrays (own-ceiling fraction is a ratio of means, as in the card)
    frac_re = ratio_boot(k40, ceil)
    cards = {}
    for i in range(1, 11):
        cards.update(json.load(open(R / "baselines/ubp" / ("ubp_card_soft12_sub%d.json" % i)))["subjects"])
    ids = sorted(cards, key=int)[:nsub or None]
    rr = [cards[i]["released"] for i in ids]
    y_over_c = float(np.mean([r_["k40_bits"] / r_["ceiling_k40_bits"] for r_ in rr]))
    yf_over_cf = float(np.mean([(r_["k40_bits"] - r_["k40_floor_bits"]) / (r_["ceiling_k40_bits"] - r_["k40_floor_bits"]) for r_ in rr]))
    pj = [json.load(open(f_)) for f_ in sorted(glob.glob(str(R / "ext/m7/pe_sub-[0-9][0-9][0-9].json")))]
    pe_mean_ratio = {a_: float(np.mean([q["arms"][a_]["gain"] / q["dedicated_eegnet"] for q in pj])) for a_ in ("reve", "raw")}
    share_re = ratio_boot(rk80, k80)
    eb, nz, ph = J("ext/eu_bac_v2/full.json"), J("baselines/ubp_noise_input/summary.json"), J("baselines/ubp_phase_control/summary.json")
    P.append(dict(
        target="image (THINGS-EEG2)", decoder="UBP released checkpoints", scorer="200-way InfoNCE soft12, LOO temperature", unit="bits per test image", candidate_set="200 test images", n=len(k80),
        raw_yield={"k80": F(rel["k80_bits"]["mean"], u, rel["k80_bits"]["ci95"]), "k40": F(rel["k40_bits"]["mean"], u, rel["k40_bits"]["ci95"])},
        noise_input_floor=F(nz["summary"]["noise_input_bits"]["mean"], "baselines/ubp_noise_input/summary.json", nz["summary"]["noise_input_bits"]["ci95"],
                            note="N(0,1) input, one pass reused for k40; label 'floor', not 'twin'"),
        raw_minus_noise={"k80": F(nz["summary"]["k80_minus_noise"]["mean"], "baselines/ubp_noise_input/summary.json", nz["summary"]["k80_minus_noise"]["ci95"])},
        untrained_encoder_baseline=F(ri["k80_bits"]["mean"], u, ri["k80_bits"]["ci95"], note="random-init ridge, k80; NOT a noise-input twin",
                                     share_of_raw=F(ub["random_init_share_of_released_k80"], u), share_recomputed_gpu=share_re),
        phase_shuffle_control=F({"phase_k80": ph["bits"]["phase_k80"]["mean"], "shuffle_k80": ph["bits"]["shuffle_k80"]["mean"]}, "baselines/ubp_phase_control/summary.json",
                                note="real-EEG avg80 std %.3f vs noise 1.0 (scale differs; descriptive)" % ph["stats"]["real_avg80_std_mean"]),
        own_reference_fraction=F(rel["fraction_k40"]["mean"], u, rel["fraction_k40"]["ci95"], ceiling_type="repetition", convention="MEAN OF PER-PARTICIPANT RATIOS Y/C at matched k40 (the paper convention; saved fraction_k40.mean); ratio-of-means is the recomputed_gpu check; Y/C has no floor subtraction: mean per-participant (Y-F)/(C-F) = %.4f vs Y/C = %.4f, difference %.4f" % (yf_over_cf, y_over_c, yf_over_cf - y_over_c),
                                 note="k40 own-repeat ceiling %.2f bits (soft12 grid); estimator-specific, image ceilings sit near log2(200)=7.64" % rel["ceiling_k40_bits"]["mean"],
                                 recomputed_gpu=frac_re),
        specific_yield=NA("no category-nuisance result saved for UBP (W1 trial swap v1 per-decoder share 7%, NOT citable; km3 covers the raw ridge only)"),
        L90=NA("not saved for UBP; image L90 1000 belongs to the Q8 PCA ridge read-out (see image ridge row)"),
        transfer=NA("not saved"),
    ))
    # ---------------------------------------------------------------- IMAGE: NICE-GA k1/k80
    B = eb["B_nice_ga_budget"]
    P.append(dict(
        target="image (THINGS-EEG2)", decoder="NICE-GA (frozen, k-repeat budget)", scorer="InfoNCE (as saved in eu_bac B)", unit="bits per test image", candidate_set="200 test images", n=B["k80"]["n"],
        raw_yield={"k80": F(B["k80"]["mean"], "ext/eu_bac_v2/full.json", [B["k80"]["lo"], B["k80"]["hi"]]), "k1": F(B["k1"]["mean"], "ext/eu_bac_v2/full.json", [B["k1"]["lo"], B["k1"]["hi"]])},
        noise_input_floor=NA("not saved for this read-out"), untrained_encoder_baseline=NA("no twin saved"), phase_shuffle_control=NA("not run"),
        own_reference_fraction=NA("S1 ceilings superseded (soft12 rerun s1_v6 pending)"), specific_yield=NA("W1 NICE-GA category-only share 41% is NOT citable"),
        L90=NA("not saved for the released read-out"), transfer=NA("not saved")))
    # ---------------------------------------------------------------- IMAGE: raw ridge arm (km3 specific, L90, Y(n))
    km, st, n2 = J("ext/km3_v2/summary_km3.json"), J("standard/summary.json"), J("ext/n2/summary_image_ridge.json")
    P.append(dict(
        target="image (THINGS-EEG2)", decoder="raw-EEG ridge (our base path, PCA-1024)", scorer="200-way InfoNCE soft12", unit="bits per test image", candidate_set="200 test images", n=km["n"],
        raw_yield={"k80": F(km["none"]["soft12"], "ext/km3_v2/summary_km3.json", note="km3 'none' arm = base ridge, k80")},
        noise_input_floor=NA("floor (shuffled pair) %.4f saved in km3 'none'; noise-input not run for this arm" % km["none"]["floor"]),
        untrained_encoder_baseline=NA("not applicable (no encoder)"), phase_shuffle_control=NA("not run"),
        own_reference_fraction=NA("not in a citable file"),
        specific_yield=F(km["full"]["kept_share"], "ext/km3_v2/summary_km3.json", km["full"]["kept_share_ci"], kind="exemplar yield kept after LEACE category erasure (nuisance = THINGS category27)", is_share=True,
                         permuted_profile_control_kept=[km["permprof%d" % i]["kept_share"] for i in range(5)],
                         note="nuisance-adjusted, not causal; same scorer family, within-profile pools"),
        L90=F(st["image"]["L90"]["value"], "standard/summary.json", st["image"]["L90"]["ci"], note=st["image"]["L90"]["source"]),
        transfer=NA("not saved")))
    # ---------------------------------------------------------------- PE: M7 REVE and raw voltage ridge
    m7, ey, sp, xx = J("ext/m7/summary_pe.json"), J("ext/eyequiet_v3/summary_b1.json"), J("ext/pe_specificity_v2/summary_specificity.json"), J("ext/eu_bac_v2/full.json")
    pe = xx["C_pe_eyequiet_v3"]["primary_eog_mad"]
    pe_common = dict(target="prediction error (ds003458)", scorer="nested held-out log-score, Gaussian (bits_g)", unit="bits per trial", candidate_set="continuous density, no candidates", n=m7["n"])
    arms = m7["arms"]

    def pe_row(name, a, noise_key):
        return dict(**pe_common, decoder=name,
                    raw_yield=F(arms[a]["gain"]["mean"], "ext/m7/summary_pe.json", arms[a]["gain"]["ci"]),
                    noise_input_floor=F(arms[noise_key]["gain"]["mean"], "ext/m7/summary_pe.json", arms[noise_key]["gain"]["ci"], note="m7 arm '%s' (noise input through the same pipeline); shift floor %.4f" % (noise_key, arms[a]["shift_floor"])))
    r = pe_row("REVE frozen + ridge", "reve", "reve_noise")
    r.update(untrained_encoder_baseline=F(arms["reve_twin"]["gain"]["mean"], "ext/m7/summary_pe.json", arms["reve_twin"]["gain"]["ci"], share_of_raw=arms["reve_twin"]["gain"]["mean"] / arms["reve"]["gain"]["mean"],
                                          note="random-init encoder + same ridge; NOT a noise-input twin (m7 name 'reve_twin')"),
             phase_shuffle_control=NA("shift floor only; no phase control run for PE"),
             own_reference_fraction=F(arms["reve"]["fraction_of_dedicated"]["ratio_of_means"], "ext/m7/summary_pe.json", arms["reve"]["fraction_of_dedicated"]["ci"], ceiling_type="reader", convention="RATIO OF MEANS (mean gain / mean dedicated gain), NOT the UBP convention; mean of per-participant ratios = %.3f (23 participants; unstable when the dedicated gain is near 0); (Y-F)/(C-F) not applicable: gains are already floor/baseline-adjusted held-out log-score gains, so Y/C and (Y-F)/(C-F) coincide by construction" % pe_mean_ratio["reve"],
                                      note="vs dedicated cross-participant EEGNet (LOPO) 0.110; a reader reference, not an irreducible ceiling; can exceed 1"),
             specific_yield=NA("no outcome-vs-PE specificity result for REVE saved"), L90=NA("not saved for REVE"),
             transfer=F(m7["fraction_within_regime"]["ratio_of_means"], "ext/m7/summary_pe.json", m7["fraction_within_regime"]["ci"], note="REVE ridge / raw voltage ridge, within participant; cross-regime LOPO ratio in summary 'reve_lopo_E11.fraction_cross_regime'"))
    P.append(r)
    r = pe_row("raw-voltage ridge (standard)", "raw", "raw_noise")
    sp64 = sp["L64"]
    r.update(untrained_encoder_baseline=NA("not applicable"), phase_shuffle_control=NA("shift floor only"),
             own_reference_fraction=F(arms["raw"]["fraction_of_dedicated"]["ratio_of_means"], "ext/m7/summary_pe.json", arms["raw"]["fraction_of_dedicated"]["ci"], ceiling_type="reader", convention="RATIO OF MEANS, NOT the UBP convention; mean of per-participant ratios = %.3f; (Y-F)/(C-F) not applicable (gains already floor-adjusted)" % pe_mean_ratio["raw"],
                                      note="vs dedicated cross-participant EEGNet LOPO"),
             specific_yield={"eye_quiet_kept_ratio": F(pe["kept_ratio_of_means"], "ext/eu_bac_v2/full.json", pe["kept_ci"], note="trials without eye movement; ratio of means; n=%d" % pe["n_used"]),
                             "ocular_regressed_kept_share": F(sp64["all_eogreg"]["kept"], "ext/pe_specificity_v2/summary_specificity.json", sp64["all_eogreg"]["kept_ci"],
                                                              note="EOG-regressed gain %.4f of %.4f; nuisance-adjusted, descriptive" % (sp64["all_eogreg"]["gain"], sp64["all"]["gain"]))},
             L90=F(st["prediction_error"]["raw_erp_bottleneck_ds003458"]["L90"], "standard/summary.json", st["prediction_error"]["raw_erp_bottleneck_ds003458"]["L90_ci"], note="of 1472 dims, raw-ERP ridge"),
             transfer=NA("LOPO fraction only for REVE row"))
    P.append(r)
    # ---------------------------------------------------------------- MI: EEGMMIDB
    mj = J("ext/mi_card_v3/primary_lpgo/summary_mi.json")
    files = sorted(glob.glob(str(R / "ext/mi_card_v3/primary_lpgo/sub-*.json")))
    files = files[:nsub] if nsub else files
    per = [json.load(open(f)) for f in files]
    g = lambda arm: [p[arm]["eeg"]["bits"] for p in per]
    raw, prox, rege = g("raw"), g("eog_proxy"), g("raw_eogreg")
    spec_d = boot(np.array(raw) - np.array(prox))
    kept = ratio_boot(rege, raw)
    rows = mj["rows"]
    mi_common = dict(target="motor imagery (EEGMMIDB, 4-class)", scorer="held-out log-score, 4-way softmax", unit="bits per trial", candidate_set="4 classes", n=len(per) if nsub else mj["n"])
    fm = J("ext/mi_fm_eogreg/reve/summary.json")
    gate = J("ext/mi_fm_eogreg/gate.json")["models"]
    P.append(dict(**mi_common, decoder="raw 0.5-4 s, LPGO logistic",
                  raw_yield=F(rows["raw"]["eeg"]["mean"], "ext/mi_card_v3/primary_lpgo/summary_mi.json", rows["raw"]["eeg"]["ci"]),
                  noise_input_floor=F(rows["raw"]["noise"]["mean"], "ext/mi_card_v3/primary_lpgo/summary_mi.json", rows["raw"]["noise"]["ci"]),
                  untrained_encoder_baseline=NA("not applicable"),
                  phase_shuffle_control=F(rows["raw"]["shuffle"]["mean"], "ext/mi_card_v3/primary_lpgo/summary_mi.json", rows["raw"]["shuffle"]["ci"], note="shuffled-label floor"),
                  own_reference_fraction=NA("undefined: no repeated stimuli, no ceiling (summary 'ceiling' field)"),
                  specific_yield={"raw_minus_eog_proxy": dict(F(spec_d["mean"], "ext/mi_card_v3/primary_lpgo/summary_mi.json", spec_d["ci"], recomputed="GPU participant bootstrap from sub-*.json",
                                                                 note="same scorer and 4 classes; eog_proxy-only read-out yield %.3f; negative = proxy alone matches/exceeds raw; nuisance-adjusted, not causal" % rows["eog_proxy"]["eeg"]["mean"]),
                                                              n=spec_d["n"]),
                                  "raw_eogreg_kept_share": dict(F(kept["ratio_of_means"], "ext/mi_card_v3/primary_lpgo/summary_mi.json", kept["ci"], recomputed="GPU"), n=kept["n"])},
                  L90=NA("not run for MI"), transfer=NA("BCI-IV-2a not downloadable; no cross-dataset result")))
    P.append(dict(**{**mi_common, "n": mj["n"] if not nsub else mi_common["n"]}, decoder="REVE frozen on EOG-regressed input", raw_yield=F(fm["rows"]["fm"]["eeg"]["mean"], "ext/mi_fm_eogreg/reve/summary.json", fm["rows"]["fm"]["eeg"]["ci"]),
                  noise_input_floor=F(fm["rows"]["fm"]["noise"]["mean"], "ext/mi_fm_eogreg/reve/summary.json", fm["rows"]["fm"]["noise"]["ci"], note="floors_ok=%s; %s" % (fm["floors_ok"], fm["rows"]["fm"]["noise"]["label"])),
                  untrained_encoder_baseline=F(fm["rows"]["twin"]["eeg"]["mean"], "ext/mi_fm_eogreg/reve/summary.json", fm["rows"]["twin"]["eeg"]["ci"], note="random-init REVE, same read-out"),
                  phase_shuffle_control=NA("shuffle floor not run (identical read-out)"), own_reference_fraction=NA("undefined (no ceiling)"),
                  specific_yield=F(fm["fm_minus_raw"]["d"], "ext/mi_fm_eogreg/reve/summary.json", fm["fm_minus_raw"]["ci"], note="REVE minus raw_eogreg (both nuisance-regressed): pretraining dividend over the regressed raw rung"),
                  L90=NA("not run"), transfer=NA("none"),
                  other_fm_yield={m: F(v["fm_eeg"], "ext/mi_fm_eogreg/gate.json", note="LaBraM/EEGPT transductive (pretrained on EEGMMIDB)" if m in ("labram", "eegpt") else None) for m, v in gate.items()}))
    # ---------------------------------------------------------------- TEXT (ZuCo)
    tj = "baselines/text/text_yield_card.json"
    T = J(tj)
    for name, key in (("BrainTranslator (our re-impl.)", "braintranslator"), ("DeWave-style (our re-impl.)", "dewave_style")):
        s = T["summary_over_seeds"][key]
        within = {t: F(s["beam/within_" + t]["eeg_minus_noise"]["mean"], tj, s["beam/within_" + t]["eeg_minus_noise"]["seed_t_ci95"]) for t in ("SR", "NR", "NR2")}
        P.append(dict(target="text (ZuCo)", decoder=name, scorer="sentence InfoNCE, LOO temperature, beam decoding", unit="bits per record", candidate_set="sentences in test split", n="3 seeds (CI over seeds, not participants)",
                      raw_yield=F(s["beam/pooled"]["eeg_yield_bits"]["mean"], tj, s["beam/pooled"]["eeg_yield_bits"]["seed_t_ci95"]),
                      noise_input_floor=F(s["beam/pooled"]["noise_yield_bits"]["mean"], tj, s["beam/pooled"]["noise_yield_bits"]["seed_t_ci95"], note="noise-input twin, same trained pipeline"),
                      untrained_encoder_baseline=F(s["beam/pooled"]["random_init_yield_bits"]["mean"], tj, s["beam/pooled"]["random_init_yield_bits"]["seed_t_ci95"]),
                      phase_shuffle_control=NA("not run"), own_reference_fraction=NA("undefined: own ceiling not saved (card column queued)"),
                      specific_yield={"pooled_eeg_minus_noise": F(s["beam/pooled"]["eeg_minus_noise"]["mean"], tj, s["beam/pooled"]["eeg_minus_noise"]["seed_t_ci95"]),
                                      "within_task_eeg_minus_noise": within, "note": "nuisance = task identity; within-task removes the between-task cue; descriptive"},
                      L90=NA("undefined (X6 curve at floor; standard/summary.json sentence_SR L90 defined=%s)" % st["sentence_SR"]["L90"]["defined"]), transfer=NA("not saved")))
    gr = T["models"]["glim_released"]["yield"]["free_running"]
    P.append(dict(target="text (ZuCo)", decoder="GLIM released generations", scorer="sentence InfoNCE (seed-0 scorer)", unit="bits per record", candidate_set="182 test sentences", n="2208 records",
                  raw_yield=F(gr["pooled"]["arms"]["eeg"]["yield_bits"], tj),
                  noise_input_floor=F(gr["pooled"]["arms"]["eeg"]["yield_bits"] - gr["pooled"]["eeg_minus_noise_test_time"]["mean"], tj, note="derived: EEG yield minus (EEG - noise) test-time; TEST-TIME noise only, no retrained noise twin"),
                  untrained_encoder_baseline=NA("not available for released model"), phase_shuffle_control=NA("not run"), own_reference_fraction=NA("undefined"),
                  specific_yield={"pooled_eeg_minus_noise_test_time": F(gr["pooled"]["eeg_minus_noise_test_time"]["mean"], tj, gr["pooled"]["eeg_minus_noise_test_time"]["ci95"]),
                                  "within_task": {k[7:]: F(v["eeg_minus_noise_test_time"]["mean"], tj, v["eeg_minus_noise_test_time"]["ci95"]) for k, v in gr.items() if k.startswith("within_")}},
                  L90=NA("not run"), transfer=NA("not saved")))
    for r in P:
        r["rank_across_targets"] = "forbidden"
    return P


def ladder(P):
    """R0-R9 per target. state in {Pass, Partial, Fail, undefined}; 'Pass' needs a citable source and a passing check on the saved numbers; otherwise capped to Partial."""
    ub, m7, mj, T = J("baselines/ubp/ubp_summary_soft12.json"), J("ext/m7/summary_pe.json"), J("ext/mi_card_v3/primary_lpgo/summary_mi.json"), J("baselines/text/text_yield_card.json")
    d6 = "ext/d6/summary_inject_so_gpu_n23.json"
    ok = lambda rel: status(rel) == "citable"

    def c(state, rel, check, note):
        if not check:
            return {"state": "Fail", "file": "results/" + rel, "note": "check failed: " + note}
        if state == "Pass" and not ok(rel):
            state, note = "Partial", note + " [source " + status(rel) + "]"
        return {"state": state, "file": "results/" + rel, "note": note}

    U = lambda note: {"state": "undefined", "file": None, "note": note}
    km = "ext/km3_v2/summary_km3.json"
    img = {"R0 split": c("Pass", "baselines/ubp/ubp_summary_soft12.json", ub["released"]["G2_pass_all"], "released checkpoints reproduce published top-1 10/10"),
           "R1 target/candidates": c("Pass", "baselines/ubp/ubp_summary_soft12.json", True, "200-way soft12"),
           "R2 floor": c("Pass", "baselines/ubp/ubp_summary_soft12.json", abs(ub["released"]["k80_floor_bits"]["mean"]) < 0.02, "shuffle floor %.4f; noise-input floor in ubp_noise_input (reviewed)" % ub["released"]["k80_floor_bits"]["mean"]),
           "R3 yield+CI": c("Pass", "baselines/ubp/ubp_summary_soft12.json", True, "participant CI, n=10"),
           "R4 own ceiling": c("Pass", "baselines/ubp/ubp_summary_soft12.json", True, "repetition ceiling k40; estimator-specific"),
           "R5 L90": c("Partial", "standard/summary.json", True, "L90 1000 is for the Q8 PCA ridge read-out, not for UBP"),
           "R6 nuisance": c("Pass", km, True, "category erasure kept share (raw ridge only; not UBP)"),
           "R7 read-out+control": c("Partial", "ext/eu_bac_v2/full.json", True, "untrained-encoder baseline present; no nonlinear positive control for image"),
           "R8 n": c("Pass", "baselines/ubp/ubp_summary_soft12.json", ub["released"]["k80_bits"]["mean"] is not None, "n=10"),
           "R9 robustness/transfer": c("Partial", "baselines/ubp_phase_control/summary.json", True, "Y(k) and phase/shuffle control exist; transfer not saved")}
    pe = {"R0 split": c("Pass", "ext/m7/summary_pe.json", m7["g2_raw_standard_max_abs_diff"] < 1e-9, "standard path reproduced to %.1e" % m7["g2_raw_standard_max_abs_diff"]),
          "R1 target/candidates": c("Pass", "ext/m7/summary_pe.json", True, "continuous |PE| density, no candidates"),
          "R2 floor": c("Pass", "ext/m7/summary_pe.json", abs(m7["arms"]["reve"]["shift_floor"]) < 0.02, "shift floor %.4f" % m7["arms"]["reve"]["shift_floor"]),
          "R3 yield+CI": c("Pass", "ext/m7/summary_pe.json", True, "participant CI n=23"),
          "R4 own ceiling": c("Pass", "ext/m7/summary_pe.json", True, "reader reference (dedicated EEGNet LOPO), not a repetition ceiling"),
          "R5 L90": c("Pass", "standard/summary.json", True, "L90 64 (raw-ERP ridge)"),
          "R6 nuisance": c("Pass", "ext/eyequiet_v3/summary_b1.json", True, "eye-quiet kept 1.05; EOG-regressed share in pe_specificity_v2 (pending)"),
          "R7 read-out+control": c("Pass", d6, True, "D6 positive control + nonlinear ladder (registered)"),
          "R8 n": c("Pass", "ext/m7/summary_pe.json", m7["n"] == 23, "n=23"),
          "R9 robustness/transfer": c("Partial", "ext/n2/summary_pe_trials.json", True, "Y(n) saved but pending; LOPO transfer ratio for REVE only")}
    mi = {"R0 split": c("Pass", "ext/mi_card_v3/primary_lpgo/summary_mi.json", mj["g2_pass"], "leave-participant-group-out; G2 pass"),
          "R1 target/candidates": c("Pass", "ext/mi_card_v3/primary_lpgo/summary_mi.json", True, "4 classes"),
          "R2 floor": c("Pass", "ext/mi_card_v3/primary_lpgo/summary_mi.json", mj["floors_ok"], "noise and shuffle floors ~0"),
          "R3 yield+CI": c("Pass", "ext/mi_card_v3/primary_lpgo/summary_mi.json", True, "participant CI n=%d" % mj["n"]),
          "R4 own ceiling": U("no repeated stimuli: ceiling undefined (summary 'ceiling' field)"),
          "R5 L90": U("not run for MI"),
          "R6 nuisance": c("Partial", "ext/mi_card_v3/primary_lpgo/summary_mi.json", True, "EOG PROXY regression (no EOG channels in EEGMMIDB); raw rung reads ocular/gaze"),
          "R7 read-out+control": c("Partial", "ext/mi_fm_eogreg/gate.json", True, "random-init twin present; LaBraM/EEGPT transductive"),
          "R8 n": c("Pass", "ext/mi_card_v3/primary_lpgo/summary_mi.json", mj["n"] == 105, "n=105"),
          "R9 robustness/transfer": c("Partial", "ext/mi_card_v3/primary_lpgo/summary_mi.json", True, "v2 vs v3 input sensitivity; BCI-IV-2a not downloadable")}
    sd = J("standard/summary.json")["sentence_SR"]["L90"]
    tx = {"R0 split": c("Partial", "baselines/text/text_yield_card.json", T["leakage"]["ours_dedup/exact"]["test_sentences_also_in_train_or_dev"] == 0, "leak-free own split; re-implementations FAIL G2 vs published (not reproduced)"),
          "R1 target/candidates": c("Pass", "baselines/text/text_yield_card.json", True, "182 test sentences"),
          "R2 floor": c("Pass", "baselines/text/text_yield_card.json", True, "noise-input twin and random-init twin saved"),
          "R3 yield+CI": c("Partial", "baselines/text/text_yield_card.json", True, "CI over 3 seeds, not over participants"),
          "R4 own ceiling": U("own ceiling not saved"),
          "R5 L90": U("X6 curve at floor; standard/summary.json sentence_SR L90 defined=%s" % sd["defined"]),
          "R6 nuisance": c("Pass", "baselines/text/text_yield_card.json", True, "within-task decomposition (pooled gain = task cue)"),
          "R7 read-out+control": c("Partial", "baselines/text/text_yield_card.json", True, "noise twin present; encoder-frozen twins queued; no positive control"),
          "R8 n": c("Partial", "baselines/text/text_yield_card.json", True, "seeds, not participants"),
          "R9 robustness/transfer": U("not saved")}
    return {"image": img, "prediction_error": pe, "motor_imagery": mi, "text": tx,
            "note": "No score is computed from this matrix (an audit-completeness fraction invites gaming); report the cells."}


def reliability():
    """Reuse results/ext/m2/summary_reliability.json (m2_trial_yield.py); only check that the per-trial npz are present."""
    rel = "ext/m2/summary_reliability.json"
    S = J(rel)
    out = {"status": status(rel), "src": "results/" + rel, "image": {}, "pe": {}}
    for k in ("nice141", "nicega297"):
        d = S["image"][k]
        out["image"][k] = {"partial_spearman": d["mean"], "ci": d["ci"], "floor": d["floor_mean"], "gate_pass": d.get("gate_pass")}
    out["image"]["n"] = S["image"]["n"]
    out["pe"] = {"halves": {"rho": S["pe"]["halves"]["mean"], "ci": S["pe"]["halves"]["ci"], "gate_pass": S["pe"]["halves"]["gate_pass"]},
                 "ridge_vs_eegnet": {"rho": S["pe"]["ridge_vs_eegnet"]["mean"], "ci": S["pe"]["ridge_vs_eegnet"]["ci"]}, "n": S["pe"]["n"]}
    out["npz_counts"] = {"image_nice141": len(glob.glob(str(R / "ext/m2/pvi_image_nice141_sub-*.npz"))), "image_nicega297": len(glob.glob(str(R / "ext/m2/pvi_image_nicega297_sub-*.npz"))),
                         "pe_ridge": len(glob.glob(str(R / "ext/m2/pvi_pe_ridge_sub-*.npz")))}
    out["scope"] = "image and PE only; no per-trial reliability for text or MI"
    return out


def scaling():
    eb, ub, nz, ph = J("ext/eu_bac_v2/full.json"), J("baselines/ubp/ubp_summary_soft12.json"), J("baselines/ubp_noise_input/summary.json"), J("baselines/ubp_phase_control/summary.json")
    A, B = eb["A_ubp_budget"], eb["B_nice_ga_budget"]
    out = {"Y_k": {
        "UBP": {k: {"raw": F(A["raw"][k]["mean"], "ext/eu_bac_v2/full.json", [A["raw"][k]["lo"], A["raw"][k]["hi"]]), "net_of_untrained_encoder": F(A["net"][k]["mean"], "ext/eu_bac_v2/full.json", [A["net"][k]["lo"], A["net"][k]["hi"]])} for k in ("40", "80")},
        "UBP_saturation_k80_over_k40": {"raw": F(A["saturation_ratio_k80_over_k40"]["raw"]["mean"], "ext/eu_bac_v2/full.json"), "net": F(A["saturation_ratio_k80_over_k40"]["net"]["mean"], "ext/eu_bac_v2/full.json"),
                                        "diff": F(A["saturation_ratio_k80_over_k40"]["diff_net_minus_raw"]["mean"], "ext/eu_bac_v2/full.json", [A["saturation_ratio_k80_over_k40"]["diff_net_minus_raw"]["lo"], A["saturation_ratio_k80_over_k40"]["diff_net_minus_raw"]["hi"]])},
        "NICE-GA": {"1": F(B["k1"]["mean"], "ext/eu_bac_v2/full.json", [B["k1"]["lo"], B["k1"]["hi"]]), "80": F(B["k80"]["mean"], "ext/eu_bac_v2/full.json", [B["k80"]["lo"], B["k80"]["hi"]])},
        "UBP_raw_minus_noise_input": {"40": F(nz["summary"]["k40_minus_noise"]["mean"], "baselines/ubp_noise_input/summary.json", nz["summary"]["k40_minus_noise"]["ci95"]), "80": F(nz["summary"]["k80_minus_noise"]["mean"], "baselines/ubp_noise_input/summary.json", nz["summary"]["k80_minus_noise"]["ci95"])},
        "note": "k40 noise-input uses the k80-shaped pass; NICE-GA k1/k80 have no baseline column"},
        "Y_n": {}}
    for arm, rel in (("image_ridge", "ext/n2/summary_image_ridge.json"), ("pe_trials", "ext/n2/summary_pe_trials.json")):
        S = J(rel)
        out["Y_n"][arm] = {"n_participants": S["n"], "g2_all_pass": S["g2_all_pass"], "curve": [F(c["mean"], rel, c["ci"], frac=c["frac"], floor=c["floor"]) for c in S["curve"]]}
    out["Y_n"]["note"] = "n2 = training-set size fraction; image arm is the raw ridge, PE arm per-trial; sources pending review"
    return out


def md(prof, lad, rel, sc, smoke):
    f = lambda x: "pending/NA" if x is None else ("%.3f" % x if isinstance(x, (int, float)) else str(x))

    def cell(d):
        if not isinstance(d, dict) or "value" not in d and "why" not in d:
            return "see json"
        if d.get("value") is None:
            return "NA: " + d.get("why", d.get("note", "")) if "why" in d else "see note"
        if d.get("is_share") and not isinstance(d["value"], dict):
            s0 = "share %.3f" % d["value"] + (" [%s]" % ", ".join("%.3f" % v for v in d["ci"]) if "ci" in d else "")
            return s0 + " (" + d["status"].split(" ")[0] + ")"
        s = f(d["value"]) if not isinstance(d["value"], dict) else ", ".join("%s %.4f" % kv for kv in d["value"].items())
        if "ci" in d:
            s += " [%s]" % ", ".join("%.3f" % v for v in d["ci"])
        return s + " (" + d["status"].split(" ")[0] + ")"

    def sub(d):
        if isinstance(d, dict) and "value" not in d and "why" not in d:
            return "; ".join("%s %s" % (k, cell(v)) for k, v in d.items() if isinstance(v, dict) and ("value" in v))
        return cell(d)

    L = ["# UEYC yield PROFILE v2 (22 Nov 2026) -", "",
         "Script experiments/ext/ueyc_profile_v2.py sha256 `%s`%s. Reads saved results only. Status tags: citable / reviewed / pending (from paper/figs/results_io.CITABLE + code-hash checks)." % (SHA, " (SMOKE: 2 participants where per-participant files are recomputed)" if smoke else ""),
         "", "**Own-reference convention.** UBP fraction = mean of per-participant ratios Y/C at matched k40 (the paper's convention); PE fractions are ratios of means (flagged per row, with the mean-of-ratios value); the UBP (Y-F)/(C-F) vs Y/C difference is in its row. UBP rows use the decided soft12 grid (12-point temperature grid); the soft12 summary and EU-BAC v2 are tagged pending until independent review.", "",
         "**Note on UBP k80 raw: 4.888 vs 4.887 vs 4.928.** 4.887 (4.8867) = released k80 on the soft12 grid (this profile, ubp_summary_soft12.json). 4.888 (4.8875) = 'raw minus noise' from the noise-input file = its own soft12 k80 run (4.8867) minus the noise floor (-0.00088); it is raw minus a different floor file, so it differs from 4.887 by about 0.001. 4.928 (4.9280) = the OLD 40-point temperature grid (ubp_summary.json; in the soft12 cards it survives as k80_bits_t40); the decided grid is soft12 and 4.928 is not used in this profile.", "",
         "**Rule: raw bits are never ranked across targets.** Rows differ in scorer, candidate set and unit; compare only within a target, and only for the same scorer. 'Specific yield' = nuisance-adjusted (descriptive), not causal.", ""]
    tg = []
    for r in prof:
        if r["target"] not in tg:
            tg.append(r["target"])
    for t in tg:
        L += ["## " + t, "", "| decoder | scorer / candidates / unit | n | raw yield | noise-input floor | untrained-encoder baseline | phase/shuffle | own-reference fraction (ceiling type) | specific yield (nuisance-adjusted; share = fraction of raw yield, not bits) | L90 | transfer |", "|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in [x for x in prof if x["target"] == t]:
            ry = r["raw_yield"]
            ryc = "; ".join("%s %s" % (k, cell(v)) for k, v in ry.items()) if "src" not in ry else cell(ry)
            of = r["own_reference_fraction"]
            ofc = cell(of) + (" (%s)" % of["ceiling_type"] if of.get("ceiling_type") else "") + (" CONVENTION: %s" % of["convention"] if of.get("convention") else "")
            L.append("| %s | %s / %s / %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (r["decoder"], r["scorer"], r["candidate_set"], r["unit"], r["n"], ryc, cell(r["noise_input_floor"]),
                     cell(r["untrained_encoder_baseline"]), cell(r["phase_shuffle_control"]), ofc, sub(r["specific_yield"]), cell(r["L90"]), cell(r["transfer"])))
        L.append("")
    L += ["## Audit ladder R0-R9 (no score; one cell per rung, with the file)", ""]
    names = list(lad["image"].keys())
    L += ["| rung | " + " | ".join(k for k in ("image", "prediction_error", "motor_imagery", "text")) + " |", "|---|---|---|---|---|"]
    for n in names:
        L.append("| %s | " % n + " | ".join("%s (%s)" % (lad[k][n]["state"], (lad[k][n]["file"] or "-").replace("results/", "").split("/")[-1] + ": " + lad[k][n]["note"]) for k in ("image", "prediction_error", "motor_imagery", "text")) + " |")
    L += ["", "## Reliability gate (reused from m2_trial_yield.py summary; status %s)" % rel["status"], "",
          "Image (n=%d): %s. PE (n=%d): halves rho %.3f [%.3f, %.3f] gate_pass=%s; ridge vs EEGNet %.3f. per-trial npz found: %s. %s" % (
              rel["image"]["n"], "; ".join("%s %.3f [%.3f, %.3f] floor %.4f" % (k, v["partial_spearman"], v["ci"][0], v["ci"][1], v["floor"]) for k, v in rel["image"].items() if isinstance(v, dict)),
              rel["pe"]["n"], rel["pe"]["halves"]["rho"], rel["pe"]["halves"]["ci"][0], rel["pe"]["halves"]["ci"][1], rel["pe"]["halves"]["gate_pass"], rel["pe"]["ridge_vs_eegnet"]["rho"], rel["npz_counts"], rel["scope"]), "",
          "## Scaling", "", "Y(k) UBP raw / net of untrained-encoder baseline: " + "; ".join("k%s %s / %s" % (k, cell(sc["Y_k"]["UBP"][k]["raw"]), cell(sc["Y_k"]["UBP"][k]["net_of_untrained_encoder"])) for k in ("40", "80")),
          "", "UBP saturation k80/k40: raw %s, net %s, diff %s. NICE-GA Y(k): k1 %s, k80 %s. UBP raw minus noise-input: k40 %s, k80 %s." % (cell(sc["Y_k"]["UBP_saturation_k80_over_k40"]["raw"]), cell(sc["Y_k"]["UBP_saturation_k80_over_k40"]["net"]), cell(sc["Y_k"]["UBP_saturation_k80_over_k40"]["diff"]),
                                                                                                                                                            cell(sc["Y_k"]["NICE-GA"]["1"]), cell(sc["Y_k"]["NICE-GA"]["80"]), cell(sc["Y_k"]["UBP_raw_minus_noise_input"]["40"]), cell(sc["Y_k"]["UBP_raw_minus_noise_input"]["80"])), ""]
    for arm in ("image_ridge", "pe_trials"):
        a = sc["Y_n"][arm]
        L.append("Y(n) %s (n=%d, G2 pass %s): " % (arm, a["n_participants"], a["g2_all_pass"]) + "; ".join("frac %.2f -> %s" % (c["frac"], cell(c)) for c in a["curve"]) + " (pending)")
    L += ["", sc["Y_n"]["note"]]
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    prof = run(2 if a.smoke else 0)
    lad, rel, sc = ladder(prof), reliability(), scaling()
    out = {"script_sha256": SHA, "NOT_CITABLE": "until independent review", "smoke": a.smoke, "rank_across_targets": "forbidden", "profile": prof, "ladder": lad, "reliability": rel, "scaling": sc}
    od = R / "ext" / "ueyc_v2"
    od.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(od / ("smoke.json" if a.smoke else "profile.json"), "w"), indent=1, default=str)
    tgt = ROOT / "outputs" / "FINAL" / ("UEYC_PROFILE_v2_SMOKE.md" if a.smoke else "UEYC_PROFILE_v2_20261122.md")
    tgt.write_text(md(prof, lad, rel, sc, a.smoke))
    print("wrote", od, tgt, SHA[:8])
