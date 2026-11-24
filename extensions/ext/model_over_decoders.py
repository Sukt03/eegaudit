"""Experiment F (locked protocol, 24 Nov 2026): the EEGauge model equation read on top of the decoders already registered.

One row per decoder with the terms of the model, measured from EXISTING result files only (no retraining, no new model run):
  Y   yield with 95% interval            B   blank yield (type stated)        Y0  untrained-encoder baseline (where it exists)
  net = Y - max(B, Y0)  (terms reported; NOT an attribution)                   R   replicate yield (typed anchor, where valid)
  U   usage (Y - B)/(R - B) (only where a valid anchor exists)                 K   nuisance-adjusted yield / share (where it exists)
and the identity checks: the participant-level (or seed-level) sum/difference relations the definitions imply hold numerically.

Every value is read through paper/figs/results_io.py (load / load_file; CITABLE gate), as paper/figs/make_v11.py and
paper/po/figs/apply_untrained.py do. A source whose flag is False (or that is not listed) is 'held back' and gets no number.
register() is never called, so the paper's number registry is not touched. Nothing under paper/ is written.

Intervals: participant bootstrap (10000 resamples, torch on the GPU, seed 0, same generator recipe as gf_image_untrained_v2.boot)
where per-participant values are stored behind the gate; otherwise the stored interval is copied and labelled 'stored'.
Text rows: the unit is the training seed (t interval, stored; recomputed here as a check).

Bug checks: each number is compared with the value printed in paper/po/tables (tab_profile.tex, v11_numbers.tex, untrained_cells.tex)
at the printed precision; mismatches are listed, never fixed.

Run (GPU node):
  srun -n1 -c3 -t 10 python -I -u \
       /path/to/eegaudit/experiments/ext/model_over_decoders.py [--smoke]
Output: results/ext/model_over_decoders/{summary.json, rows.csv, table.md}  ()
"""
import csv
import hashlib
import json
import math
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, "/path/to/eegaudit/paper/figs")
import results_io as rio
from results_io import load, load_file, get

import numpy as np
import torch

T0 = time.time()
ROOT = Path("/path/to/eegaudit")
SMOKE = "--smoke" in sys.argv
OUT = ROOT / "results" / "ext" / ("model_over_decoders_smoke" if SMOKE else "model_over_decoders")
OUT.mkdir(parents=True, exist_ok=True)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
NB = 10000
CODE_SHA = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
PO = ROOT / "paper" / "po" / "tables"

FINDINGS = []      # mismatches and provenance findings (never fixed here)
IDENT = []         # identity checks
CONTROLS = []      # self-tests of this script


def fl(x):
    return None if x is None else float(x)


def ci2(c):
    return None if c is None else [float(c[0]), float(c[1])]


def boot(v, B=NB):
    """Participant bootstrap of the mean: same recipe as gf_image_untrained_v2.boot (GPU generator, seed 0)."""
    v = torch.tensor(np.asarray(v, np.float64), device=DEV)
    g = torch.Generator(device=DEV); g.manual_seed(0)
    m = v[torch.randint(len(v), (B, len(v)), device=DEV, generator=g)].mean(1)
    return float(v.mean()), [float(m.quantile(.025)), float(m.quantile(.975))]


def boot_ratio(a, b, B=NB):
    """Bootstrap of the ratio of means sum(a)/sum(b) over participants (paired resampling, seed 0)."""
    a = torch.tensor(np.asarray(a, np.float64), device=DEV); b = torch.tensor(np.asarray(b, np.float64), device=DEV)
    g = torch.Generator(device=DEV); g.manual_seed(0)
    ix = torch.randint(len(a), (B, len(a)), device=DEV, generator=g)
    r = a[ix].mean(1) / b[ix].mean(1)
    return float(a.mean() / b.mean()), [float(r.quantile(.025)), float(r.quantile(.975))]


def term(value=None, ci=None, src="", ci_source=None, note="", status="ok", stored=None, stored_ci=None, n=None, kind=""):
    return dict(status=status, value=fl(value), ci=ci2(ci), ci_source=ci_source, stored_value=fl(stored), stored_ci=ci2(stored_ci),
                n=n, kind=kind, src=src, note=note)


def held(src, why="CITABLE flag False or source not listed in results_io.CITABLE"):
    return dict(status="held back", value=None, ci=None, src=src, note=why)


def undef(reason):
    return dict(status="undefined", value=None, ci=None, src="", note=reason)


def ident(name, eq, lhs, rhs, tol, level, note=""):
    ok = (lhs is not None and rhs is not None and abs(lhs - rhs) <= tol)
    # 'passed' uses the strict tolerance fixed before the first run; 'within_project_gate_1e-4' is reported beside it (the project's
    # saved-value gate), so a float32 round-off (about 1e-8 in the text card) is visible as such and no tolerance is changed after the fact.
    IDENT.append(dict(name=name, equation=eq, lhs=lhs, rhs=rhs, abs_diff=None if lhs is None or rhs is None else abs(lhs - rhs),
                      tol=tol, level=level, passed=bool(ok), within_project_gate_1e4=bool(lhs is not None and rhs is not None and abs(lhs - rhs) <= 1e-4), note=note))
    return ok


def finding(kind, where, detail):
    FINDINGS.append(dict(kind=kind, where=where, detail=detail))


def control(name, passed, detail):
    CONTROLS.append(dict(name=name, passed=bool(passed), detail=detail))
    print(("CONTROL PASS " if passed else "CONTROL FAIL ") + name + " :: " + detail)


# ============================================================================================ printed tables (read only)
def _macros():
    M = {}
    for fn in ("v11_numbers.tex", "untrained_cells.tex"):
        for line in (PO / fn).read_text().splitlines():
            m = re.match(r"\\newcommand\{\\([A-Za-z]+)\}\{(.*)\}\s*$", line)
            if m:
                M[m.group(1)] = (m.group(2), fn)
    return M


MAC = _macros()
NUM = re.compile(r"[-+]?\d+\.\d+")


def _clean(s):
    s = re.sub(r"\$\^\{[a-z]\}\$", "", s)       # floor-type superscripts
    s = s.replace("$-$", "-").replace("$", "").replace(r"\%", "").replace(r"{\scriptsize", "").replace("{", "").replace("}", "")
    return s


def _expand(s):
    return re.sub(r"\\([A-Za-z]+)(\{\})?", lambda m: MAC[m.group(1)][0] if m.group(1) in MAC else m.group(0), s)


def _profile_rows():
    rows = {}
    for line in (PO / "tab_profile.tex").read_text().splitlines():
        if " & " in line and line.rstrip().endswith(r"\\") and not line.startswith(r"\multicolumn") and not line.startswith("Decoder"):
            cells = line.rstrip()[:-2].split(" & ")
            if len(cells) == 6:
                rows[cells[0].strip()] = [NUM.findall(_clean(_expand(c))) for c in cells[1:]]
    return rows


PROF = _profile_rows()


def tok_equal(value, token, pct=False):
    """True when value, formatted with the token's number of decimals, equals the printed token (-0.000 == 0.000)."""
    d = len(token.split(".")[1])
    v = float(value) * (100.0 if pct else 1.0)
    return float(f"{v:.{d}f}") == float(token), f"{v:.{d}f}"


def check_profile(label, spec):
    """spec: {column 0..4: [(name, value, pct, stored_value_or_None)]}. Compares with the printed tokens of the row."""
    out = dict(row=label, n_compared=0, n_mismatch=0, mismatches=[])
    if label not in PROF:
        finding("print", f"tab_profile.tex / {label}", "row not found in the printed table"); out["missing"] = True
        return out
    cols = ("Yield", "Blank", "Untrained", "Usage", "Nuisance")
    for c in range(5):
        exp = spec.get(c, []); toks = PROF[label][c]
        if len(exp) != len(toks):
            if exp and not toks:
                out.setdefault("not_printed", []).append(cols[c] + ": " + ", ".join(e[0] for e in exp))
            elif toks and not exp:
                finding("print", f"tab_profile.tex / {label} / {cols[c]}", f"table prints {toks} but this run has no number for the cell")
            else:
                finding("print", f"tab_profile.tex / {label} / {cols[c]}", f"count differs: table prints {toks}, this run expects {[e[0] for e in exp]}")
            continue
        for (name, v, pct, sv), t in zip(exp, toks):
            out["n_compared"] += 1
            ok, s = tok_equal(v, t, pct)
            if not ok:
                oks = None
                if sv is not None:
                    oks, ss = tok_equal(sv, t, pct)
                out["n_mismatch"] += 1
                d = dict(cell=cols[c], name=name, printed=t, this_run=s, stored_matches_print=oks)
                out["mismatches"].append(d)
                finding("print" if oks is not True else "recompute", f"tab_profile.tex / {label} / {cols[c]} / {name}",
                        f"printed {t}, this run {s}" + ("" if oks is None else ("; the stored value matches the print (bootstrap recompute differs in the last digit)" if oks else "; the stored value does NOT match the print either")))
    return out


def check_macro(name, value, pct=False):
    if name not in MAC:
        finding("print", f"macro \\{name}", "macro not found in v11_numbers.tex / untrained_cells.tex"); return None
    txt, fn = MAC[name]
    toks = re.findall(r"[-+]?\d+\.\d+|[-+]?\d+", _clean(txt))
    if value is None or len(toks) != 1:
        finding("print", f"{fn} / \\{name}", f"cannot compare: printed '{txt}', value {value}"); return False
    t = toks[0]
    d = len(t.split(".")[1]) if "." in t else 0
    v = float(value) * (100.0 if pct else 1.0)
    ok = float(f"{v:.{d}f}") == float(t)
    if not ok:
        finding("print", f"{fn} / \\{name}", f"printed {t}, source value formats to {v:.{d}f}")
    return ok


# ============================================================================================ controls of this script
def self_tests():
    # gate: False-flag or unlisted sources return None
    g = [("ext/m7/summary_pe.json", load("ext/m7/summary_pe.json")), ("baselines/ubp (soft12 cards)", load_file("baselines/ubp", "ubp_card_soft12_sub1.json")),
         ("gpu_fill_20261123/mi_retest_summary.json", load("gpu_fill_20261123/mi_retest_summary.json")),
         ("ext/d2_ds007647_splithalf", load_file("ext/d2_ds007647_splithalf", "summary.json")),
         ("ext/mi_fm (unregressed)", load_file("ext/mi_fm", "reve/summary.json"))]
    control("gate_holds_back_unflagged_sources", all(x is None for _, x in g), "; ".join(f"{k}: {'None' if x is None else 'LOADED'}" for k, x in g))
    # planted positive control for the bootstrap: known mean and standard error
    gen = torch.Generator().manual_seed(123)
    v = (0.3 + torch.randn(2000, generator=gen, dtype=torch.float64)).numpy()
    m, ci = boot(v)
    hw = (ci[1] - ci[0]) / 2; ana = 1.96 * v.std(ddof=1) / math.sqrt(len(v))
    control("bootstrap_planted_width", abs(hw / ana - 1) < 0.05 and ci[0] < 0.3 < ci[1] or abs(m - 0.3) < 3 * ana and abs(hw / ana - 1) < 0.05,
            f"planted mean 0.3, n 2000: mean {m:.4f}, half-width {hw:.4f} vs analytic {ana:.4f}")
    m, ci = boot(np.full(10, 0.25))
    control("bootstrap_constant_null", ci[0] == ci[1] == 0.25, f"constant input gives a zero-width interval: {ci}")
    # noise-input null for the bootstrap: zero-mean noise, interval must contain 0
    gen = torch.Generator().manual_seed(7)
    m, ci = boot(torch.randn(200, generator=gen, dtype=torch.float64).numpy() * 0.01)
    control("bootstrap_noise_null", ci[0] < 0 < ci[1], f"zero-mean noise: mean {m:+.5f}, interval [{ci[0]:+.5f}, {ci[1]:+.5f}]")
    # mismatch detector: planted wrong value must fire, right value must pass, signed zero must pass
    a = tok_equal(0.3026, "0.302")[0]; b = tok_equal(0.3016, "0.302")[0]; c = tok_equal(-0.0003, "0.000")[0]; d = tok_equal(0.5903, "59.0", pct=True)[0]; e = tok_equal(0.5957, "59.0", pct=True)[0]
    control("mismatch_detector_planted", (not a) and b and c and d and (not e), f"planted 0.3026 vs 0.302 fires: {not a}; 0.3016 passes: {b}; -0.0003 vs 0.000 passes: {c}; pct passes: {d}; planted pct fires: {not e}")


# ============================================================================================ rows
ROWS = []


def add(**r):
    ROWS.append(r)
    return r


def net_of(Y, B, Y0, paired=None, stored=None, stored_src=""):
    """net = Y - max(B, Y0) on cohort means. Interval: bootstrap of paired Y_j - ref_j where both are stored per participant,
    else the stored interval of the same difference, else none."""
    y = Y.get("value"); b = B.get("value"); y0 = Y0.get("value")
    if y is None:
        return undef("yield missing")
    refs = [(n_, v_) for n_, v_ in (("B", b), ("Y0", y0)) if v_ is not None]
    if not refs:
        return undef("neither blank nor untrained baseline available")
    who, ref = max(refs, key=lambda t: t[1])
    t = term(y - ref, None, kind=f"Y - {who}", note=f"max(B, Y0) = {who}" + ("" if len(refs) == 2 else " (only term available)"))
    if paired is not None and who == paired[0]:
        m, ci = boot(paired[1]); t["ci"] = ci; t["ci_source"] = "bootstrap (paired participants)"; t["paired_mean"] = m
    elif stored is not None and who == "Y0":
        t["ci"] = ci2(stored[1]); t["ci_source"] = "stored"; t["stored_value"] = fl(stored[0]); t["src"] = stored_src
    else:
        t["ci_source"] = "none (terms come from different stored summaries)"
    return t


def build():
    S = load("standard/summary.json"); EB = load("ext/eu_bac_v2/full_final.json"); EA = get(EB, "A_ubp_budget")
    PFD = load("ext/ueyc_v2/profile.json"); NI = load_file("baselines/ubp_noise_input", "summary.json")
    KM = load_file("ext/km3_decoders_v3", "summary.json"); T2 = load("baselines/text_card_v2/text_yield_card.json")
    Dd = load("ext/d2_ds007647_v2/summary.json"); M3 = load("ext/mi_card_v3/primary_lpgo/summary_mi.json")
    IMG = load_file("gpu_fill_20261123/image_untrained", "summary.json"); PE = load_file("gpu_fill_20261123/pe_untrained", "summary.json")
    MI = load_file("gpu_fill_20261123/mi_untrained", "summary.json"); EQ = load("ext/eyequiet_v3/summary_b1.json")
    RT = load("gpu_fill_20261123/mi_retest_summary.json"); SH = load_file("ext/d2_ds007647_splithalf", "summary.json")
    for nm, o in (("standard/summary.json", S), ("ext/eu_bac_v2/full_final.json", EB), ("ext/ueyc_v2/profile.json", PFD), ("baselines/ubp_noise_input", NI),
                  ("ext/km3_decoders_v3", KM), ("baselines/text_card_v2/text_yield_card.json", T2), ("ext/d2_ds007647_v2/summary.json", Dd),
                  ("ext/mi_card_v3/primary_lpgo/summary_mi.json", M3), ("gpu_fill_20261123/image_untrained", IMG), ("gpu_fill_20261123/pe_untrained", PE),
                  ("gpu_fill_20261123/mi_untrained", MI), ("ext/eyequiet_v3/summary_b1.json", EQ)):
        if o is None:
            finding("gate", nm, "source came back None (held back or unreadable); its cells carry no number")
    im = get(S, "image"); bf = get(im, "open_weight_models", "brainflora_native")
    KMd = lambda d: (get(KM, "decoders", d, "soft12", "kept_full"), get(KM, "decoders", d, "soft12", "kept_full_ci"))
    PFr = lambda i, k: get(PFD, "profile", i, k)

    # ================================================================== IMAGE
    unit_i = "bits per query (mean of k repetitions), 200-way InfoNCE soft12, leave-one-query-out temperature (transductive)"
    cand_i = "200 test images"
    pp = get(NI, "per_participant"); ids = sorted(pp, key=int) if pp is not None else []
    ni_j = np.array([pp[i]["noise_input_bits"] for i in ids]) if ids else None
    for k in ("80", "40"):
        yj = np.array([pp[i][f"k{k}_bits"] for i in ids])
        ym, yci = boot(yj)
        Y = term(get(EA, "raw", k, "mean"), yci, src=f"ext/eu_bac_v2/full_final.json:A_ubp_budget.raw.{k}; per participant baselines/ubp_noise_input/summary.json:per_participant.k{k}_bits",
                 ci_source="bootstrap", stored=get(EA, "raw", k, "mean"), stored_ci=[get(EA, "raw", k, "lo"), get(EA, "raw", k, "hi")], n=len(ids))
        ident(f"UBP k{k}: G1 per-participant mean vs stored yield", "mean_j Y_j = Y(stored)", ym, fl(get(EA, "raw", k, "mean")), 1e-4, "participant")
        bm, bci = boot(ni_j)
        B = term(bm, bci, src="baselines/ubp_noise_input/summary.json:per_participant.noise_input_bits", ci_source="bootstrap", kind="noise input (one N(0,1) pass at the 80-repetition shape, reused for k40)",
                 stored=get(NI, "summary", "noise_input_bits", "mean"), stored_ci=get(NI, "summary", "noise_input_bits", "ci95"), n=len(ids))
        B["max_abs_participant"] = float(np.abs(ni_j).max()); B["n_abs_gt_0p02"] = int((np.abs(ni_j) > 0.02).sum())
        d_j = np.array([pp[i][f"k{k}_minus_noise"] for i in ids])
        ident(f"UBP k{k}: Y_j - B_j per participant", "max_j |(Y_j - B_j) - saved (Y - B)_j| = 0", float(np.abs((yj - ni_j) - d_j).max()), 0.0, 1e-9, "participant")
        ident(f"UBP k{k}: mean(Y - B) = mean Y - mean B", "mean_j (Y_j - B_j) = Ybar - Bbar (stored)", float((yj - ni_j).mean()), fl(get(NI, "summary", f"k{k}_minus_noise", "mean")), 1e-9, "participant")
        Y0 = term(get(EA, "baseline_yield", k, "mean"), [get(EA, "baseline_yield", k, "lo"), get(EA, "baseline_yield", k, "hi")], ci_source="stored",
                  src=f"ext/eu_bac_v2/full_final.json:A_ubp_budget.baseline_yield.{k}", kind="untrained encoder + fitted ridge read-out", n=fl(get(EA, "baseline_yield", k, "n")),
                  note="per-participant values sit in baselines/ubp soft12 cards (flag False: held back), so the stored interval is copied")
        net = net_of(Y, B, Y0, stored=(get(EA, "net", k, "mean"), [get(EA, "net", k, "lo"), get(EA, "net", k, "hi")]), stored_src=f"ext/eu_bac_v2/full_final.json:A_ubp_budget.net.{k}")
        ident(f"UBP k{k}: net = Y - Y0", "mean_j (Y_j - Y0_j) (stored net) = Ybar - Y0bar", fl(get(EA, "net", k, "mean")), fl(get(EA, "raw", k, "mean")) - fl(get(EA, "baseline_yield", k, "mean")), 1e-9,
              "cohort (per-participant Y0 held back)")
        kv, kc = KMd("ubp")
        K = term(kv, kc, ci_source="stored", src="ext/km3_decoders_v3/summary.json:decoders.ubp.soft12.kept_full", kind="share kept after category-subspace erasure, measured at 80 repetitions",
                 note="(Y_erased - floor)/(Y - floor), ratio; descriptive, not an attribution" + ("" if k == "80" else "; measured at k80, shown on the k40 row as in the paper table"))
        if k == "80":
            R = undef("no repetition replicate at the matched 80-repetition budget (the split-half replicate uses 40 + 40)"); U = undef("no anchor at the matched budget")
        else:
            rom = get(PFr(0, "own_reference_fraction"), "recomputed_gpu", "ratio_of_means"); orf = PFr(0, "own_reference_fraction")
            Rv = None if not rom else fl(get(EA, "raw", "40", "mean")) / fl(rom)
            R = term(Rv, None, ci_source="none", kind="repetition replicate (40 vs 40 in the decoder's own code space)", src="derived: A_ubp_budget.raw.40 / profile[0].own_reference_fraction.recomputed_gpu.ratio_of_means (as make_v11.ubp_tick)",
                     note="derived from the stored ratio of means; per-participant replicate yields sit in the held-back soft12 cards")
            U = term(get(orf, "value"), get(orf, "ci"), ci_source="stored", src="ext/ueyc_v2/profile.json:profile.0.own_reference_fraction", kind="repetition; stored = mean over participants of Y_j/R_j, blank not subtracted",
                     note="per-participant R_j held back, so (Y_j - B_j)/(R_j - B_j) cannot be formed per participant")
            if Rv is not None:
                U["cohort_with_blank"] = (Y["value"] - B["value"]) / (Rv - B["value"]); U["ratio_of_means_stored"] = fl(rom)
                ident("UBP k40: usage with and without the blank agree", "(Ybar - Bbar)/(Rbar - Bbar) ~= Ybar/Rbar (protocol.tex, eq. usage, text below it)", U["cohort_with_blank"], fl(rom), 1e-3, "cohort")
        r = add(target="image", decoder=f"UBP, {k} rep.", label="UBP, %s rep." % k, n=len(ids), unit=unit_i, candidates=cand_i, interval_unit="participant", exposure="transductive (L)",
                Y=Y, B=B, Y0=Y0, net=net, R=R, U=U, K=K, d=dict(Y=2, B=4, Y0=2, net=2, R=2, U=1, K=2))
        spec = {0: [("Y", Y["value"], False, None), ("Y lo", Y["ci"][0], False, Y["stored_ci"][0]), ("Y hi", Y["ci"][1], False, Y["stored_ci"][1])],
                1: [("B", B["value"], False, B["stored_value"])],
                2: [("Y0", Y0["value"], False, None), ("Y0 lo", Y0["ci"][0], False, None), ("Y0 hi", Y0["ci"][1], False, None)]}
        if k == "40":
            spec[3] = [("U", U["value"], True, None), ("U lo", U["ci"][0], True, None), ("U hi", U["ci"][1], True, None)]
            spec[4] = [("K", K["value"], False, None), ("K lo", K["ci"][0], False, None), ("K hi", K["ci"][1], False, None)]
        r["print_check"] = check_profile(r["label"], spec)
    if SMOKE:
        return

    # ---- NICE-GA and BrainFLORA at 40 repetitions
    subs = [load_file("gpu_fill_20261123/image_untrained", f"sub-{i:02d}.json") for i in range(1, 11)]
    assert all(s is not None for s in subs), "image_untrained per-participant files missing"
    sid = [s["participant"] for s in subs]

    def img_row(m, label, yblk, yci_stored, B, R_mean, R_pp, frac, kname, Y_pp_std):
        y0j = np.array([s["models"][m]["k40"] for s in subs]); tsj = np.array([s["trained_saved"][m]["k40_mean"] for s in subs])
        y0m, y0ci = boot(y0j)
        Y0 = term(y0m, y0ci, ci_source="bootstrap", src=f"gpu_fill_20261123/image_untrained/sub-*.json:models.{m}.k40 (mean of 5 init seeds)", kind="untrained encoder + fitted ridge read-out",
                  stored=get(IMG, m, "k40", "mean"), stored_ci=get(IMG, m, "k40", "ci95"), n=len(subs))
        ident(f"{label}: Y0 per-participant mean vs stored", "mean_j Y0_j = Y0(stored)", y0m, fl(get(IMG, m, "k40", "mean")), 1e-9, "participant")
        if Y_pp_std is not None:     # per-participant yields stored in the same file as the registered yield
            ym, yci = boot(Y_pp_std)
            Y = term(yblk, yci, ci_source="bootstrap", stored=yblk, stored_ci=yci_stored, n=len(Y_pp_std), src=f"standard/summary.json:image.open_weight_models.brainflora_native (k40, per_participant)")
            ident(f"{label}: G1 per-participant mean vs stored yield", "mean_j Y_j = Y(stored)", ym, fl(yblk), 1e-4, "participant")
            yj = np.asarray(Y_pp_std)
        else:
            Y = term(yblk, yci_stored, ci_source="stored", stored=yblk, stored_ci=yci_stored, n=fl(get(im, "yield_per_query", "n")), src="standard/summary.json:image.yield_per_query",
                     note="registered summary stores no per-participant yields; interval copied")
            yj = None
        # second stored copy of the trained k40 score (image_untrained/trained_saved) against the registered one
        g1 = ident(f"{label}: G1 second stored copy of the trained k40 yield", "mean_j trained_saved.k40_mean_j = Y(registered)", float(tsj.mean()), fl(yblk), 1e-4, "participant",
                   note="trained_saved.k40_mean is the per-participant mean over the k40 rows of the saved image_models reference")
        if not g1:
            tb = boot(tsj)
            finding("gate-1e-4", f"{label}: Y at 40 repetitions", f"two stored copies of the trained yield differ by {abs(tsj.mean() - fl(yblk)):.2e} bits (> 1e-4): standard/summary.json {fl(yblk):.6f} "
                    f"[{yci_stored[0]:.4f}, {yci_stored[1]:.4f}] vs image_untrained trained_saved mean {tsj.mean():.6f} [{tb[1][0]:.4f}, {tb[1][1]:.4f}]; same at 2 decimals for the mean; the paper prints the first")
        ident(f"{label}: stored share Y0/Y (mean of ratios)", "mean_j (Y0_j / Y_j) = stored share_of_trained_k40", float((y0j / tsj).mean()), fl(get(IMG, m, "share_of_trained_k40", "mean")), 1e-9, "participant")
        ident(f"{label}: stored share Y0/Y (ratio of means)", "Y0bar / Ybar = stored ratio_of_means_k40", float(y0j.mean() / tsj.mean()), fl(get(IMG, m, "ratio_of_means_k40")), 1e-9, "participant")
        paired = None
        if yj is not None:
            paired = ("Y0", yj - y0j)
        else:
            paired = ("Y0", tsj - y0j)
        net = net_of(Y, B, Y0, paired=paired)
        if yj is None and net.get("ci") is not None:
            net["note"] += "; interval from the paired per-participant difference of the second stored copy (trained_saved.k40_mean) minus Y0, point value from the registered yield"
        kv, kc = KMd(kname)
        K = term(kv, kc, ci_source="stored", src=f"ext/km3_decoders_v3/summary.json:decoders.{kname}.soft12.kept_full", kind="share kept after category-subspace erasure, measured at 80 repetitions",
                 note="ratio; descriptive, not an attribution; measured at k80, shown on the k40 row as in the paper table")
        R = term(R_mean[0], R_mean[1], ci_source=R_mean[2], src=R_mean[3], kind="repetition replicate (40 vs 40 in the decoder's own code space)")
        if R_pp is not None and yj is not None:
            bj = B["_pp"]
            uj = (yj - bj) / (np.asarray(R_pp) - bj)
            um, uci = boot(uj)
            U = term(um, uci, ci_source="bootstrap", kind="repetition; mean over participants of (Y_j - B_j)/(R_j - B_j)", stored=get(frac, "value"), stored_ci=get(frac, "ci"), n=len(uj),
                     src="standard/summary.json:...brainflora_native.per_participant (k40, k80_noise, ceiling_k40)", note="stored value is the ratio of means without the blank")
            U["mean_of_ratios_no_blank"] = float((yj / np.asarray(R_pp)).mean()); U["ratio_of_means_no_blank"] = float(yj.mean() / np.asarray(R_pp).mean())
            ident(f"{label}: stored usage (ratio of means)", "Ybar / Rbar = stored fraction_k40.value", U["ratio_of_means_no_blank"], fl(get(frac, "value")), 1e-9, "participant")
            ident(f"{label}: stored usage (mean of ratios)", "mean_j Y_j/R_j = stored fraction_k40.mean_of_ratios", U["mean_of_ratios_no_blank"], fl(get(frac, "mean_of_ratios")), 1e-9, "participant")
            ident(f"{label}: usage with and without the blank agree", "mean_j (Y_j - B_j)/(R_j - B_j) ~= mean_j Y_j/R_j", um, U["mean_of_ratios_no_blank"], 1e-3, "participant")
            perm = np.roll(np.asarray(R_pp), 1)
            control("alignment_positive_control_" + m, abs(float((yj / perm).mean()) - fl(get(frac, "mean_of_ratios"))) > 1e-6,
                    f"participants deliberately misaligned (R rolled by one): mean of ratios {float((yj / perm).mean()):.6f} vs stored {fl(get(frac, 'mean_of_ratios')):.6f}; the identity check fires")
        else:
            U = term(get(frac, "value"), get(frac, "ci"), ci_source="stored", src="standard/summary.json:image.fraction", kind="repetition; stored = ratio of means Ybar/Rbar, blank not subtracted")
            U["cohort_with_blank"] = (Y["value"] - B["value"]) / (R["value"] - B["value"])
            ident(f"{label}: stored usage (ratio of means)", "Ybar / Rbar = stored fraction.value", Y["value"] / R["value"], fl(get(frac, "value")), 1e-9, "cohort")
            ident(f"{label}: usage with and without the blank agree", "(Ybar - Bbar)/(Rbar - Bbar) ~= Ybar/Rbar", U["cohort_with_blank"], fl(get(frac, "value")), 2e-3, "cohort")
        B = {k_: v_ for k_, v_ in B.items() if k_ != "_pp"}
        r = add(target="image", decoder=label, label=label, n=len(subs), unit=unit_i, candidates=cand_i, interval_unit="participant", exposure="transductive (L)",
                Y=Y, B=B, Y0=Y0, net=net, R=R, U=U, K=K, d=dict(Y=2, B=4 if m == "brainflora" else 3, Y0=3, net=2, R=2, U=1, K=2))
        spec = {0: [("Y", Y["value"], False, None), ("Y lo", Y["ci"][0], False, Y["stored_ci"][0]), ("Y hi", Y["ci"][1], False, Y["stored_ci"][1])],
                1: [("B", B["value"], False, B.get("stored_value"))],
                2: [("Y0", Y0["value"], False, Y0["stored_value"]), ("Y0 lo", Y0["ci"][0], False, Y0["stored_ci"][0]), ("Y0 hi", Y0["ci"][1], False, Y0["stored_ci"][1])],
                3: [("U (stored convention)", U["stored_value"] if U["stored_value"] is not None else U["value"], True, None),
                    ("U lo (stored)", (U["stored_ci"] or U["ci"])[0], True, None), ("U hi (stored)", (U["stored_ci"] or U["ci"])[1], True, None)],
                4: [("K", K["value"], False, None), ("K lo", K["ci"][0], False, None), ("K hi", K["ci"][1], False, None)]}
        r["print_check"] = check_profile(label, spec)
        return r

    im_y = get(im, "yield_per_query"); im_ce = get(im, "ceiling"); im_fr = get(im, "fraction")
    Bn = term(get(im_y, "floor"), get(im_y, "floor_ci"), ci_source="stored", src="standard/summary.json:image.yield_per_query.floor", kind="shuffled pairs")
    ident("NICE-GA: Y - B", "Ybar - Bbar = stored minus_floor", fl(get(im_y, "mean")) - fl(get(im_y, "floor")), fl(get(im_y, "minus_floor")), 1e-9, "cohort (per-participant values not in the registered summary)")
    Bn["_pp"] = None
    img_row("nice_ga", "NICE-GA, 40 rep. (ours)", get(im_y, "mean"), ci2(get(im_y, "ci")), Bn, (get(im_ce, "mean"), get(im_ce, "ci"), "stored", "standard/summary.json:image.ceiling"), None, im_fr, "nice_ga", None)
    bpp = get(bf, "per_participant"); bpp = [dict.copy(dict(x)) for x in list(bpp)]
    assert [b["subject"] for b in bpp] == sid, "participant order differs between standard and image_untrained"
    bnj = np.array([b["k80_noise"] for b in bpp]); bm, bci = boot(bnj)
    Bb = term(bm, bci, ci_source="bootstrap", src="standard/summary.json:...brainflora_native.per_participant.k80_noise", kind="noise input (measured at 80 repetitions; no k40 noise pass stored for the trained model)",
              stored=get(bf, "k80_noise", "mean"), stored_ci=get(bf, "k80_noise", "ci"), n=len(bnj))
    Bb["max_abs_participant"] = float(np.abs(bnj).max()); Bb["n_abs_gt_0p02"] = int((np.abs(bnj) > 0.02).sum()); Bb["_pp"] = bnj
    rj = np.array([b["ceiling_k40"] for b in bpp]); rm, rci = boot(rj)
    ident("BrainFLORA: replicate per-participant mean vs stored", "mean_j R_j = R(stored)", rm, fl(get(bf, "ceiling_k40", "mean")), 1e-9, "participant")
    img_row("brainflora", "BrainFLORA, 40 rep.", get(bf, "k40", "mean"), ci2(get(bf, "k40", "ci")), Bb, (rm, rci, "bootstrap", "standard/summary.json:...brainflora_native.per_participant.ceiling_k40"),
            rj, get(bf, "fraction_k40"), "brainflora", np.array([b["k40"] for b in bpp]))

    # ================================================================== TEXT (unit = training seed)
    from scipy import stats as _st
    ss = get(T2, "summary_over_seeds")
    srok = get(S, "sentence_SR", "fraction_interpretable"); nrok = get(S, "sentence_NR2", "fraction_interpretable")
    for key, lab in (("braintranslator", "BrainTranslator, 5 seeds"), ("dewave_style", "DeWave-style, 5 seeds")):
        def tc(sk, what):
            e = get(ss, key, "beam/" + sk, what)
            ps = [float(x) for x in list(e["per_seed"])]
            return e, ps
        def tint(ps):
            n = len(ps); m = float(np.mean(ps)); h = float(_st.t.ppf(0.975, n - 1) * np.std(ps, ddof=1) / math.sqrt(n))
            return m, [m - h, m + h]
        terms = {}
        for nm, sk, what in (("Y", "pooled", "eeg_yield_bits"), ("B", "pooled", "noise_yield_bits"), ("Y0", "pooled", "random_init_yield_bits"), ("Dp", "pooled", "eeg_minus_noise"), ("Dsr", "within_SR", "eeg_minus_noise"),
                             ("Ysr", "within_SR", "eeg_yield_bits")):
            e, ps = tc(sk, what); m, ci = tint(ps)
            terms[nm] = term(get(e, "mean"), get(e, "seed_t_ci95"), ci_source="stored (t interval over training seeds)", src=f"baselines/text_card_v2/text_yield_card.json:summary_over_seeds.{key}.beam/{sk}.{what}", n=len(ps))
            terms[nm]["per_seed"] = ps
            ident(f"{lab}: {what} ({sk}) mean over seeds", "mean_s x_s = stored mean", m, fl(get(e, "mean")), 1e-9, "seed")
            ident(f"{lab}: {what} ({sk}) t interval", "recomputed t interval (lower) = stored", ci[0], fl(get(e, "seed_t_ci95", 0)), 1e-9, "seed")
        ident(f"{lab}: Y - B per seed (pooled)", "max_s |(Y_s - B_s) - saved (Y - B)_s| = 0",
              float(np.abs(np.array(terms["Y"]["per_seed"]) - np.array(terms["B"]["per_seed"]) - np.array(terms["Dp"]["per_seed"])).max()), 0.0, 1e-9, "seed")
        ident(f"{lab}: mean(Y - B) = mean Y - mean B (pooled)", "mean_s (Y_s - B_s) = Ybar - Bbar", terms["Dp"]["value"], terms["Y"]["value"] - terms["B"]["value"], 1e-9, "seed")
        terms["B"]["kind"] = "noise-trained control (whole recipe repeated on noise)"
        terms["Y0"]["kind"] = "random-init encoder"
        n_seed = int(get(ss, key, "n_seeds")); n0 = terms["Y0"]["n"]
        if n0 != n_seed:
            terms["Y0"]["note"] = f"only {n0} of {n_seed} seeds stored for the random-init arm"
            finding("denominator", f"{lab}: untrained column", f"the random-init (untrained) value on the '5 seeds' row is a mean over {n0} seeds, the yield and blank over {n_seed}; the printed row label does not say so")
        net = net_of(terms["Y"], terms["B"], terms["Y0"])
        net["ci_source"] = "none (Y0 has fewer seeds than Y; no paired interval)" if n0 != n_seed else net["ci_source"]
        K = dict(terms["Dsr"]); K["kind"] = "within-task (SR) EEG minus noise-trained control, bits (a restricted comparison, not a ratio)"
        K["pooled_minus_noise"] = dict(value=terms["Dp"]["value"], ci=terms["Dp"]["ci"]); K["within_SR_yield"] = dict(value=terms["Ysr"]["value"], ci=terms["Ysr"]["ci"])
        K["note"] = "pooled gallery mixes tasks, so the pooled yield can carry task identity; the SR contrast removes it"
        why = "reader anchor fails the anchor check: standard/summary.json sentence_SR/NR2 fraction_interpretable = %s/%s (reader split-half interval includes its blank)" % (srok, nrok)
        r = add(target="text", decoder=lab, label=lab, n=n_seed, unit="bits per record (beam decoding, InfoNCE over test sentences, pooled over tasks)",
                candidates="test sentences of all tasks pooled (within-task: SR %s, NR2 %s candidates)" % (get(S, "sentence_SR", "candidates"), get(S, "sentence_NR2", "candidates")),
                interval_unit="training seed (t interval)", exposure="re-implementation; train-only",
                Y=terms["Y"], B=terms["B"], Y0=terms["Y0"], net=net, R=undef(why), U=undef(why), K=K, d=dict(Y=3, B=3, Y0=3, net=3, R=3, U=1, K=3))
        t_ = terms
        r["print_check"] = check_profile(lab, {0: [("Y", t_["Y"]["value"], False, None), ("Y lo", t_["Y"]["ci"][0], False, None), ("Y hi", t_["Y"]["ci"][1], False, None)],
                                               1: [("B", t_["B"]["value"], False, None)], 2: [("Y0", t_["Y0"]["value"], False, None)],
                                               4: [("pooled Y-B", t_["Dp"]["value"], False, None), ("lo", t_["Dp"]["ci"][0], False, None), ("hi", t_["Dp"]["ci"][1], False, None),
                                                   ("SR Y-B", t_["Dsr"]["value"], False, None), ("lo", t_["Dsr"]["ci"][0], False, None), ("hi", t_["Dsr"]["ci"][1], False, None)]})

    # ================================================================== PREDICTION ERROR
    unit_p = "bits per feedback trial (Gaussian log-density gain over an intercept-only baseline)"
    cand_p = "continuous target, no candidate set"
    pe_y = get(S, "prediction_error", "yield_ds003458"); pe_ce = get(S, "prediction_error", "ceiling_ds003458"); pe_fr = get(S, "prediction_error", "fraction_ds003458")
    # EEGNet
    PP = load_file("gpu_fill_20261123/pe_untrained", "per_participant.json"); assert PP is not None
    nseed = int(PE["seeds"]); pids = sorted(PP["rand0"], key=int)
    y0j = np.array([[PP[f"rand{k}"][p]["gain"] for p in pids] for k in range(nseed)]).mean(0)
    y0m, y0ci = boot(y0j)
    Y = term(get(pe_y, "mean"), get(pe_y, "ci"), ci_source="stored", src="standard/summary.json:prediction_error.yield_ds003458", n=fl(get(pe_y, "n")),
             note="per-participant trained yields sit in standard/pe_ceiling (not a gated path); interval copied")
    B = term(get(pe_y, "floor"), get(pe_y, "floor_ci"), ci_source="stored", src="standard/summary.json:prediction_error.yield_ds003458.floor", kind="circular shift of the target")
    Y0 = term(y0m, y0ci, ci_source="bootstrap", src="gpu_fill_20261123/pe_untrained/per_participant.json:rand0..9 (mean over 10 init seeds)", kind="untrained EEGNet + fitted read-out",
              stored=get(PE, "untrained_seedmean", "mean"), stored_ci=get(PE, "untrained_seedmean", "ci95"), n=len(pids))
    ident("PE EEGNet: Y0 per-participant seed-mean vs stored", "mean_j Y0_j = Y0(stored)", y0m, fl(get(PE, "untrained_seedmean", "mean")), 1e-9, "participant")
    ident("PE EEGNet: G1 saved trained yield in two files", "pe_untrained.saved_trained_eegnet.mean = standard yield mean", fl(get(PE, "saved_trained_eegnet", "mean")), fl(get(pe_y, "mean")), 1e-4, "cohort")
    ident("PE EEGNet: Y - B", "Ybar - Bbar = stored minus_floor", fl(get(pe_y, "mean")) - fl(get(pe_y, "floor")), fl(get(pe_y, "minus_floor")), 1e-9, "cohort")
    ident("PE EEGNet: net = Y - Y0", "mean_j (Y_j - Y0_j) (stored) = Ybar - Y0bar", fl(get(PE, "trained_minus_untrained", "mean")), fl(get(PE, "saved_trained_eegnet", "mean")) - y0m, 1e-9, "cohort (paired difference stored; trained per-participant values not gated)")
    ident("PE EEGNet: Y0/Y", "Y0bar / Ybar = stored untrained_over_trained_ratio", y0m / fl(get(PE, "saved_trained_eegnet", "mean")), fl(get(PE, "untrained_over_trained_ratio")), 1e-9, "cohort")
    net = net_of(Y, B, Y0, stored=(get(PE, "trained_minus_untrained", "mean"), get(PE, "trained_minus_untrained", "ci95")), stored_src="gpu_fill_20261123/pe_untrained/summary.json:trained_minus_untrained")
    R = term(get(pe_ce, "mean"), get(pe_ce, "ci"), ci_source="stored", src="standard/summary.json:prediction_error.ceiling_ds003458",
             kind="READ-OUT REPRODUCIBILITY (closed form at the mean split-half reliability of two independently trained read-outs); not an EEG replicate")
    U = term(get(pe_fr, "value"), get(pe_fr, "ci"), ci_source="stored", src="standard/summary.json:prediction_error.fraction_ds003458", kind="read-out reproducibility anchor; (Ybar - Bbar)/(Rbar - Bbar)")
    ident("PE EEGNet: usage", "(Ybar - Bbar)/(Rbar - Bbar) = stored fraction", (Y["value"] - B["value"]) / (R["value"] - B["value"]), fl(get(pe_fr, "value")), 1e-9, "cohort")
    r = add(target="prediction error", decoder="EEGNet, ds003458", label="EEGNet, ds003458", n=len(pids), unit=unit_p, candidates=cand_p, interval_unit="participant", exposure="leave-one-participant-out read-out",
            Y=Y, B=B, Y0=Y0, net=net, R=R, U=U, K=undef("no nuisance control stored for this read-out"), d=dict(Y=3, B=3, Y0=3, net=3, R=3, U=1, K=3))
    r["print_check"] = check_profile("EEGNet, ds003458", {0: [("Y", Y["value"], False, None), ("Y lo", Y["ci"][0], False, None), ("Y hi", Y["ci"][1], False, None)], 1: [("B", B["value"], False, None)],
                                                         2: [("Y0", Y0["value"], False, Y0["stored_value"]), ("Y0 lo", Y0["ci"][0], False, Y0["stored_ci"][0]), ("Y0 hi", Y0["ci"][1], False, Y0["stored_ci"][1])],
                                                         3: [("U", U["value"], True, None), ("U lo", U["ci"][0], True, None), ("U hi", U["ci"][1], True, None)]})
    # raw-ERP ridge
    ry = PFr(4, "raw_yield"); nf = PFr(4, "noise_input_floor"); orf = PFr(4, "own_reference_fraction"); eq_ = get(PFr(4, "specific_yield"), "eye_quiet_kept_ratio")
    Y = term(get(ry, "value"), get(ry, "ci"), ci_source="stored", src="ext/ueyc_v2/profile.json:profile.4.raw_yield", n=fl(PFr(4, "n")), note="profile entry points to ext/m7/summary_pe.json (flag False); read through the profile as make_v11 does")
    B = term(get(nf, "value"), get(nf, "ci"), ci_source="stored", src="ext/ueyc_v2/profile.json:profile.4.noise_input_floor", kind="noise input (m7 arm raw_noise)")
    R = term(get(pe_y, "mean"), get(pe_y, "ci"), ci_source="stored", src="standard/summary.json:prediction_error.yield_ds003458",
             kind="READ-OUT REPRODUCIBILITY (a second, independently trained read-out of the same recordings: EEGNet leave-one-participant-out); not an EEG replicate")
    U = term(get(orf, "value"), get(orf, "ci"), ci_source="stored", src="ext/ueyc_v2/profile.json:profile.4.own_reference_fraction", kind="read-out anchor; stored = ratio of means Ybar/Rbar, blank not subtracted")
    U["cohort_with_blank"] = (Y["value"] - B["value"]) / (R["value"] - fl(get(pe_y, "floor")))
    ident("PE ridge: stored usage (ratio of means)", "Ybar_ridge / Ybar_EEGNet = stored own_reference_fraction", Y["value"] / R["value"], fl(get(orf, "value")), 1e-6, "cohort")
    K = term(get(eq_, "value"), get(eq_, "ci"), ci_source="stored", src="ext/ueyc_v2/profile.json:profile.4.specific_yield.eye_quiet_kept_ratio", kind="eye-quiet retention: yield on eye-quiet trials over a matched random subset (ratio of means)")
    e64 = get(EQ, "primary_eog_mad", "L64")
    ident("PE ridge: eye-quiet retention", "(Y_quiet - B)/(Y_random - B) = stored kept ratio", fl(get(e64, "quiet_minus_floor")) / fl(get(e64, "random_minus_floor")), fl(get(e64, "kept_ratio_of_means")), 1e-9, "cohort")
    ident("PE ridge: eye-quiet value in two files", "profile kept ratio = eyequiet_v3 kept ratio", fl(get(eq_, "value")), fl(get(e64, "kept_ratio_of_means")), 1e-9, "cohort")
    K["ocular_regressed_kept_share"] = fl(get(PFr(4, "specific_yield"), "ocular_regressed_kept_share", "value"))
    bw = get(S, "prediction_error", "raw_erp_bottleneck_ds003458")
    ident("PE ridge: G1 yield in two files", "profile raw_yield = standard full_width_gain", fl(get(ry, "value")), fl(get(bw, "full_width_gain")), 1e-4, "cohort")
    sci = ci2(get(bw, "full_width_gain_ci"))
    if sci is not None and (not tok_equal(sci[0], f"{fl(get(ry, 'ci', 0)):.3f}")[0] or not tok_equal(sci[1], f"{fl(get(ry, 'ci', 1)):.3f}")[0]):
        finding("two-intervals", "Raw-ERP ridge, ds003458: Y interval", f"the same mean ({fl(get(ry, 'value')):.5f}) carries two stored intervals: profile [{fl(get(ry, 'ci', 0)):.4f}, {fl(get(ry, 'ci', 1)):.4f}] (printed) "
                f"and standard/summary.json raw_erp_bottleneck [{sci[0]:.4f}, {sci[1]:.4f}]; they differ at 3 decimals")
    sf = fl(get(bw, "shift_floor"))
    finding("label", "Raw-ERP ridge, ds003458: blank", f"tab_profile marks the blank with superscript c (circular shift), but the value read (make_v11: profile[4].noise_input_floor = {B['value']:.5f}) is the noise-input floor; "
            f"the circular-shift floor is {sf:.5f}; both print as -0.003, so the number is right and the label is not")
    r = add(target="prediction error", decoder="Raw-ERP ridge, ds003458", label="Raw-ERP ridge, ds003458", n=int(PFr(4, "n")), unit=unit_p, candidates=cand_p, interval_unit="participant", exposure="within-participant, nested held-out",
            Y=Y, B=B, Y0=undef("not applicable: no encoder (linear read-out of the raw ERP)"), net=net_of(Y, B, undef("")), R=R, U=U, K=K, d=dict(Y=3, B=3, Y0=3, net=3, R=3, U=1, K=2))
    r["print_check"] = check_profile("Raw-ERP ridge, ds003458", {0: [("Y", Y["value"], False, None), ("Y lo", Y["ci"][0], False, None), ("Y hi", Y["ci"][1], False, None)], 1: [("B", B["value"], False, None)],
                                                                3: [("U", U["value"], True, None), ("U lo", U["ci"][0], True, None), ("U hi", U["ci"][1], True, None)],
                                                                4: [("K", K["value"], False, None), ("K lo", K["ci"][0], False, None), ("K hi", K["ci"][1], False, None)]})
    # ridge, ds007647
    cu = get(Dd, "cells", "all|pe_raw|L64"); cr = get(Dd, "cells", "all|pe_resid|L64")
    Y = term(get(cu, "gain", "mean"), get(cu, "gain", "ci"), ci_source="stored", src="ext/d2_ds007647_v2/summary.json:cells.all|pe_raw|L64.gain", n=fl(get(cu, "n")),
             note="per-participant files are not behind a gated directory key; interval copied")
    B = term(get(cu, "g3_noise", "mean"), get(cu, "g3_noise", "ci"), ci_source="stored", src="ext/d2_ds007647_v2/summary.json:cells.all|pe_raw|L64.g3_noise", kind="noise input",
             note="stored label: '%s' (passed = %s); circular-shift floor %.4f" % (get(cu, "g3_noise", "label"), get(cu, "g3_noise", "passed"), fl(get(cu, "g3_shift", "mean"))))
    K = term(get(cr, "gain", "mean"), get(cr, "gain", "ci"), ci_source="stored", src="ext/d2_ds007647_v2/summary.json:cells.all|pe_resid|L64.gain",
             kind="yield for the target residualised on outcome, choice and time, bits (a nuisance-adjusted yield, not a ratio)")
    K["share_of_unadjusted"] = K["value"] / Y["value"]
    hb = held("ext/d2_ds007647_splithalf/summary.json", "trial-split reliability result exists on disk but ext/d2_ds007647_splithalf is not listed in results_io.CITABLE") if SH is None else None
    if not ident("ds007647: Y - B", "Ybar - Bbar = stored gain_minus_floor", Y["value"] - B["value"], fl(get(cu, "gain_minus_floor", "mean")), 1e-9, "cohort",
                 note="stored gain_minus_floor - gain = %.5f; noise floor %.5f; shift floor %.5f" % (fl(get(cu, "gain_minus_floor", "mean")) - Y["value"], B["value"], fl(get(cu, "g3_shift", "mean")))):
        finding("identity", "Ridge, ds007647: Y - B", "stored gain_minus_floor (%.5f) minus gain (%.5f) implies a floor of %.5f, which equals neither stored floor mean (noise input %.5f, circular shift %.5f); "
                "the floor used by gain_minus_floor cannot be resolved from the gated summary (per-participant files are not behind a gated key); neither number is printed in the paper tables, which use 'gain'"
                % (fl(get(cu, "gain_minus_floor", "mean")), Y["value"], Y["value"] - fl(get(cu, "gain_minus_floor", "mean")), B["value"], fl(get(cu, "g3_shift", "mean"))))
    r = add(target="prediction error", decoder="Ridge, ds007647 (n=39)", label="Ridge, ds007647 ($n{=}39$)", n=int(get(cu, "n")), unit=unit_p + "; target NOT residualised", candidates=cand_p, interval_unit="participant",
            exposure="within-participant, in-fold PCA-64 + ridge", Y=Y, B=B, Y0=undef("not applicable: no encoder"), net=net_of(Y, B, undef("")),
            R=hb or undef("no anchor"), U=hb or undef("no anchor"), K=K, d=dict(Y=3, B=3, Y0=3, net=3, R=3, U=1, K=3))
    r["print_check"] = check_profile("Ridge, ds007647 ($n{=}39$)", {0: [("Y", Y["value"], False, None), ("Y lo", Y["ci"][0], False, None), ("Y hi", Y["ci"][1], False, None)], 1: [("B", B["value"], False, None)],
                                                                   4: [("K", K["value"], False, None), ("K lo", K["ci"][0], False, None), ("K hi", K["ci"][1], False, None)]})

    # ================================================================== MOTOR IMAGERY
    unit_m = "bits per trial (log2 p - log2 training class prior), leave-participant-group-out"
    cand_m = "4 classes"
    rt_note = "same-session retest rows exist in results/gpu_fill_20261123/mi_retest_summary.json, but that file is not listed in results_io.CITABLE (unreviewed)"
    def RU():
        return (held("gpu_fill_20261123/mi_retest_summary.json", rt_note), held("gpu_fill_20261123/mi_retest_summary.json", rt_note)) if RT is None else (undef("see retest file"), undef("see retest file"))
    def mrow(k):
        r_ = get(M3, "rows", k)
        Y = term(get(r_, "eeg", "mean"), get(r_, "eeg", "ci"), ci_source="stored", src=f"ext/mi_card_v3/primary_lpgo/summary_mi.json:rows.{k}.eeg", n=fl(get(M3, "n")),
                 note="per-participant files are not behind a gated directory key; interval copied")
        nz = get(r_, "noise")
        B = undef("no noise row") if nz is None else term(get(nz, "mean"), get(nz, "ci"), ci_source="stored", src=f"ext/mi_card_v3/primary_lpgo/summary_mi.json:rows.{k}.noise", kind="noise input",
                                                         note="participants within 0.02 bits: %s/%s" % (get(nz, "g3", "n_abs_le_0p02"), get(nz, "g3", "n")))
        return Y, B
    Yraw, Braw = mrow("raw"); Ypx, _ = mrow("eog_proxy"); Yreg, Breg = mrow("raw_eogreg")
    R_, U_ = RU()
    K = dict(Ypx); K["kind"] = "ocular proxies alone: yield of the five proxy derivations, bits (a nuisance read-out, not a ratio)"
    r = add(target="motor imagery", decoder="Raw voltage", label="Raw voltage", n=int(get(M3, "n")), unit=unit_m, candidates=cand_m, interval_unit="participant", exposure="train-only read-out; per-participant standardisation label-free",
            Y=Yraw, B=Braw, Y0=undef("not applicable: no encoder"), net=net_of(Yraw, Braw, undef("")), R=R_, U=U_, K=K, d=dict(Y=3, B=4, Y0=4, net=3, R=3, U=1, K=3))
    r["print_check"] = check_profile("Raw voltage", {0: [("Y", Yraw["value"], False, None), ("Y lo", Yraw["ci"][0], False, None), ("Y hi", Yraw["ci"][1], False, None)], 1: [("B", Braw["value"], False, None)],
                                                    4: [("K", K["value"], False, None), ("K lo", K["ci"][0], False, None), ("K hi", K["ci"][1], False, None)]})
    R_, U_ = RU()
    K = term(Yreg["value"] / Yraw["value"], None, ci_source="none (ratio of two stored cohort means)", src="rows.raw_eogreg.eeg.mean / rows.raw.eeg.mean (as make_v11.mi_ratio)", kind="share kept after regressing the ocular proxies (ratio of means)")
    r = add(target="motor imagery", decoder="Raw, proxies regressed", label="Raw, proxies regressed", n=int(get(M3, "n")), unit=unit_m, candidates=cand_m, interval_unit="participant", exposure="proxy regression uses all trials without labels (transductive, U)",
            Y=Yreg, B=Breg, Y0=undef("not applicable: no encoder"), net=net_of(Yreg, Breg, undef("")), R=R_, U=U_, K=K, d=dict(Y=3, B=4, Y0=4, net=3, R=3, U=1, K=3))
    r["print_check"] = check_profile("Raw, proxies regressed", {0: [("Y", Yreg["value"], False, None), ("Y lo", Yreg["ci"][0], False, None), ("Y hi", Yreg["ci"][1], False, None)], 1: [("B", Breg["value"], False, None)],
                                                               4: [("K", K["value"], True, None)]})
    # EEGNet, as recorded (unregressed): untrained baseline exists
    MP = load_file("gpu_fill_20261123/mi_untrained", "per_participant.json"); assert MP is not None
    ns = int(MI["seeds"]); mids = sorted(MP["rand0"], key=int)
    u0 = np.array([[MP[f"rand{k}"][p] for p in mids] for k in range(ns)]).mean(0); um, uci = boot(u0)
    Yen, Ben = mrow("eegnet")
    Y = term(get(MI, "saved_trained_eegnet", "mean"), get(MI, "saved_trained_eegnet", "ci95"), ci_source="stored", src="gpu_fill_20261123/mi_untrained/summary.json:saved_trained_eegnet", n=len(mids),
             note="trained per-participant values are not in the gated per_participant.json; interval copied")
    ident("MI EEGNet (unregressed): G1 saved trained yield in two files", "mi_untrained.saved_trained_eegnet.mean = mi_card_v3 rows.eegnet.eeg.mean", Y["value"], Yen["value"], 1e-4, "cohort")
    Y0 = term(um, uci, ci_source="bootstrap", src="gpu_fill_20261123/mi_untrained/per_participant.json:rand0..4 (mean over 5 init seeds)", kind="untrained EEGNet + fitted read-out",
              stored=get(MI, "untrained_seedmean", "mean"), stored_ci=get(MI, "untrained_seedmean", "ci95"), n=len(mids))
    ident("MI EEGNet (unregressed): Y0 per-participant seed-mean vs stored", "mean_j Y0_j = Y0(stored)", um, fl(get(MI, "untrained_seedmean", "mean")), 1e-9, "participant")
    ident("MI EEGNet (unregressed): net = Y - Y0", "mean_j (Y_j - Y0_j) (stored) = Ybar - Y0bar", fl(get(MI, "trained_minus_untrained", "mean")), Y["value"] - um, 1e-9, "cohort (paired difference stored)")
    net = net_of(Y, Ben, Y0, stored=(get(MI, "trained_minus_untrained", "mean"), get(MI, "trained_minus_untrained", "ci95")), stored_src="gpu_fill_20261123/mi_untrained/summary.json:trained_minus_untrained")
    if net.get("kind") == "Y - B":
        net["note"] += "; blank (%.4f) and untrained baseline (%.4f) are both at zero within their intervals; stored paired Y - Y0 = %.4f [%.4f, %.4f]" % (
            Ben["value"], um, fl(get(MI, "trained_minus_untrained", "mean")), fl(get(MI, "trained_minus_untrained", "ci95", 0)), fl(get(MI, "trained_minus_untrained", "ci95", 1)))
    R_, U_ = RU()
    r = add(target="motor imagery", decoder="EEGNet, as recorded (unregressed)", label="EEGNet, as recorded (unregressed)", n=len(mids), unit=unit_m, candidates=cand_m, interval_unit="participant", exposure="train-only",
            Y=Y, B=Ben, Y0=Y0, net=net, R=R_, U=U_, K=undef("no nuisance control on this input (the regressed row below is the control)"), d=dict(Y=3, B=4, Y0=4, net=3, R=3, U=1, K=3))
    r["print_check"] = check_profile("EEGNet, as recorded (unregressed)", {0: [("Y", Y["value"], False, None), ("Y lo", Y["ci"][0], False, None), ("Y hi", Y["ci"][1], False, None)], 1: [("B", Ben["value"], False, None)],
                                                                          2: [("Y0", Y0["value"], False, Y0["stored_value"]), ("Y0 lo", Y0["ci"][0], False, Y0["stored_ci"][0]), ("Y0 hi", Y0["ci"][1], False, Y0["stored_ci"][1])]})
    Yer, Ber = mrow("eegnet_eogreg")
    R_, U_ = RU()
    y0h = held("gpu_fill_20261123/na_mi_untrained_eogreg", "an untrained-EEGNet run on the regressed input exists on disk but is not listed in results_io.CITABLE") if load_file("gpu_fill_20261123/na_mi_untrained_eogreg", "summary.json") is None else undef("")
    K = term(Yer["value"] / Y["value"], None, ci_source="none (ratio of two stored cohort means)", src="rows.eegnet_eogreg.eeg.mean / rows.eegnet.eeg.mean", kind="share kept after regressing the ocular proxies (ratio of means)",
             note="not printed in the paper table; same definition as the raw-voltage row")
    r = add(target="motor imagery", decoder="EEGNet, proxies regressed", label="EEGNet, proxies regressed", n=int(get(M3, "n")), unit=unit_m, candidates=cand_m, interval_unit="participant", exposure="proxy regression transductive (U)",
            Y=Yer, B=Ber, Y0=y0h, net=net_of(Yer, Ber, y0h), R=R_, U=U_, K=K, d=dict(Y=3, B=4, Y0=4, net=3, R=3, U=1, K=3))
    r["print_check"] = check_profile("EEGNet, proxies regressed", {0: [("Y", Yer["value"], False, None), ("Y lo", Yer["ci"][0], False, None), ("Y hi", Yer["ci"][1], False, None)], 1: [("B", Ber["value"], False, None)],
                                                                  4: [("K", K["value"], True, None)]})
    # foundation models (regressed input)
    for m, nm, lab in (("reve", "REVE (disjoint), regressed", "disjoint"), ("cbramod", "CBraMod (disjoint), regressed", "disjoint"), ("labram", "LaBraM (transductive), regressed", "transductive"), ("eegpt", "EEGPT (transductive), regressed", "transductive")):
        F = load_file("ext/mi_fm_eogreg", f"{m}/summary.json"); assert F is not None
        ps = [int(p) for p in list(F["participants"])]
        fj, tj, nj = [], [], []
        for p in ps:
            o = load_file("ext/mi_fm_eogreg", f"{m}/sub-{p:03d}.json"); assert o is not None, (m, p)
            fj.append(o["fm"]["eeg"]["bits"]); tj.append(o["twin"]["eeg"]["bits"]); nj.append(o["fm"]["noise"]["bits"])
        fj, tj, nj = np.array(fj), np.array(tj), np.array(nj)
        ym, yci = boot(fj); tm, tci = boot(tj); bm, bci = boot(nj); dm, dci = boot(fj - tj)
        Y = term(ym, yci, ci_source="bootstrap", src=f"ext/mi_fm_eogreg/{m}/sub-*.json:fm.eeg.bits", stored=get(F, "rows", "fm", "eeg", "mean"), stored_ci=get(F, "rows", "fm", "eeg", "ci"), n=len(ps))
        B = term(bm, bci, ci_source="bootstrap", src=f"ext/mi_fm_eogreg/{m}/sub-*.json:fm.noise.bits", kind="noise input (unregressed noise through the same read-out)", stored=get(F, "rows", "fm", "noise", "mean"),
                 stored_ci=get(F, "rows", "fm", "noise", "ci"), n=len(ps))
        B["max_abs_participant"] = float(np.abs(nj).max()); B["n_abs_gt_0p02"] = int((np.abs(nj) > 0.02).sum())
        if B["n_abs_gt_0p02"]:
            finding("blank-tolerance", f"{nm}: noise-input blank per participant", f"{B['n_abs_gt_0p02']}/{len(nj)} participants have |B_j| > 0.02 bits (max {B['max_abs_participant']:.4f}); cohort mean {bm:+.4f}; "
                    "protocol.tex requires |B_j| <= 0.02 per participant and lists its exceptions")
        Y0 = term(tm, tci, ci_source="bootstrap", src=f"ext/mi_fm_eogreg/{m}/sub-*.json:twin.eeg.bits", kind="random-init counterpart of the same architecture, same fitted read-out",
                  stored=get(F, "rows", "twin", "eeg", "mean"), stored_ci=get(F, "rows", "twin", "eeg", "ci"), n=len(ps))
        ident(f"{nm}: G1 per-participant mean vs stored yield", "mean_j Y_j = Y(stored)", ym, fl(get(F, "rows", "fm", "eeg", "mean")), 1e-4, "participant")
        ident(f"{nm}: Y0 per-participant mean vs stored", "mean_j Y0_j = Y0(stored)", tm, fl(get(F, "rows", "twin", "eeg", "mean")), 1e-9, "participant")
        ident(f"{nm}: B per-participant mean vs stored", "mean_j B_j = B(stored)", bm, fl(get(F, "rows", "fm", "noise", "mean")), 1e-9, "participant")
        ident(f"{nm}: net = Y - Y0", "mean_j (Y_j - Y0_j) = stored dividend = Ybar - Y0bar", dm, fl(get(F, "dividend_fm_minus_twin", "d")), 1e-9, "participant")
        ident(f"{nm}: difference of means", "mean_j (Y_j - Y0_j) = Ybar - Y0bar", dm, ym - tm, 1e-12, "participant")
        net = net_of(Y, B, Y0, paired=("Y0", fj - tj)); net["stored_value"] = fl(get(F, "dividend_fm_minus_twin", "d")); net["stored_ci"] = ci2(get(F, "dividend_fm_minus_twin", "ci"))
        if net.get("kind") == "Y - B":
            net["note"] += "; paired Y - Y0 = %.4f [%.4f, %.4f]" % (dm, dci[0], dci[1])
        R_, U_ = RU()
        r = add(target="motor imagery", decoder=nm, label=nm, n=len(ps), unit=unit_m, candidates=cand_m, interval_unit="participant", exposure=("pretraining %s with EEGMMIDB" % lab) + "; frozen encoder, fitted read-out; proxy regression transductive (U)",
                Y=Y, B=B, Y0=Y0, net=net, R=R_, U=U_, K=undef("input already proxy-regressed; no further nuisance control stored for this model"), d=dict(Y=3, B=4, Y0=3, net=3, R=3, U=1, K=3))
        r["paired_Y_minus_Y0"] = dict(value=dm, ci=dci)
        r["print_check"] = check_profile(nm, {0: [("Y", Y["value"], False, Y["stored_value"]), ("Y lo", Y["ci"][0], False, Y["stored_ci"][0]), ("Y hi", Y["ci"][1], False, Y["stored_ci"][1])],
                                              1: [("B", B["value"], False, B["stored_value"])],
                                              2: [("Y0", Y0["value"], False, Y0["stored_value"]), ("Y - Y0", dm, False, net["stored_value"]), ("lo", dci[0], False, net["stored_ci"][0]), ("hi", dci[1], False, net["stored_ci"][1])]})

    # ------------------------------------------------------------------ macro files (v11_numbers.tex, untrained_cells.tex)
    mc = []
    def M_(name, v, pct=False):
        mc.append((name, check_macro(name, v, pct)))
    M_("VUbpNoise", get(NI, "summary", "noise_input_bits", "mean")); M_("VUbpNoiseLo", get(NI, "summary", "noise_input_bits", "ci95", 0)); M_("VUbpNoiseHi", get(NI, "summary", "noise_input_bits", "ci95", 1))
    M_("VUbpBaseForty", get(EA, "baseline_yield", "40", "mean")); M_("VUbpBaseFortyLo", get(EA, "baseline_yield", "40", "lo")); M_("VUbpBaseFortyHi", get(EA, "baseline_yield", "40", "hi"))
    M_("VUbpBaseEightyLo", get(EA, "baseline_yield", "80", "lo")); M_("VUbpBaseEightyHi", get(EA, "baseline_yield", "80", "hi"))
    M_("VUbpShareFortyP", get(EA, "baseline_fraction", "40", "ratio_of_means"), True)
    o = PFr(0, "own_reference_fraction"); M_("VUbpRefPct", get(o, "value"), True); M_("VUbpRefLoPct", get(o, "ci", 0), True); M_("VUbpRefHiPct", get(o, "ci", 1), True)
    M_("VNiceFracPct", get(im_fr, "value"), True); M_("VNiceFracLoPct", get(im_fr, "ci", 0), True); M_("VNiceFracHiPct", get(im_fr, "ci", 1), True)
    o = get(bf, "fraction_k40"); M_("VBfFracPct", get(o, "value"), True); M_("VBfFracLoPct", get(o, "ci", 0), True); M_("VBfFracHiPct", get(o, "ci", 1), True)
    M_("VBfForty", get(bf, "k40", "mean")); M_("VBfFortyLo", get(bf, "k40", "ci", 0)); M_("VBfFortyHi", get(bf, "k40", "ci", 1)); M_("VNiceFloor", get(im_y, "floor")); M_("VBfNoise", get(bf, "k80_noise", "mean"))
    for nm, d_ in (("Ubp", "ubp"), ("Nice", "nice_ga"), ("Bf", "brainflora")):
        kv, kc = KMd(d_); M_("VKept" + nm, kv); M_("VKept" + nm + "Lo", kc[0]); M_("VKept" + nm + "Hi", kc[1])
    for key, ab in (("braintranslator", "Bt"), ("dewave_style", "Dw")):
        for sk, sab in (("pooled", "Pool"), ("within_SR", "Sr"), ("within_NR2", "Nr")):
            for what, wab in (("eeg_yield_bits", "Eeg"), ("noise_yield_bits", "Noise"), ("random_init_yield_bits", "Rand"), ("eeg_minus_noise", "Diff")):
                if sk != "pooled" and wab in ("Noise", "Rand"):
                    continue
                e = get(ss, key, "beam/" + sk, what)
                M_(f"V{ab}{sab}{wab}B", get(e, "mean")); M_(f"V{ab}{sab}{wab}BLo", get(e, "seed_t_ci95", 0)); M_(f"V{ab}{sab}{wab}BHi", get(e, "seed_t_ci95", 1))
    M_("VPeFracPct", get(pe_fr, "value"), True); M_("VPeFracLoPct", get(pe_fr, "ci", 0), True); M_("VPeFracHiPct", get(pe_fr, "ci", 1), True)
    o = PFr(4, "own_reference_fraction"); M_("VPeRawRef", get(o, "value")); M_("VPeRawRefLo", get(o, "ci", 0)); M_("VPeRawRefHi", get(o, "ci", 1)); M_("VPeRawFloor", get(PFr(4, "noise_input_floor"), "value"))
    M_("VPeEyeQuiet", get(eq_, "value")); M_("VPeEyeQuietLo", get(eq_, "ci", 0)); M_("VPeEyeQuietHi", get(eq_, "ci", 1))
    for k, ab in (("raw", "Raw"), ("eog_proxy", "Prox"), ("raw_eogreg", "Reg"), ("eegnet_eogreg", "Eegnet")):
        e = get(M3, "rows", k, "eeg"); M_("VMi" + ab, get(e, "mean")); M_("VMi" + ab + "Lo", get(e, "ci", 0)); M_("VMi" + ab + "Hi", get(e, "ci", 1))
    M_("VMiKeptPct", fl(get(M3, "rows", "raw_eogreg", "eeg", "mean")) / fl(get(M3, "rows", "raw", "eeg", "mean")), True); M_("VMiNoiseRaw", get(M3, "rows", "raw", "noise", "mean"))
    for m, ab in (("cbramod", "Cb"), ("reve", "Reve"), ("labram", "Lab"), ("eegpt", "Eegpt")):
        F = load_file("ext/mi_fm_eogreg", f"{m}/summary.json"); dv = get(F, "dividend_fm_minus_twin")
        M_("VFm" + ab, get(F, "rows", "fm", "eeg", "mean")); M_("VFm" + ab + "Div", get(dv, "d")); M_("VFm" + ab + "DivLo", get(dv, "ci", 0)); M_("VFm" + ab + "DivHi", get(dv, "ci", 1))
    for k, key in (("nice_ga", "Nice"), ("brainflora", "Bf")):
        for kk, nm in (("k80", "KEighty"), ("k40", "KForty"), ("trained_k80", "TrEighty")):
            e = get(IMG, k, kk); M_(f"U{key}{nm}", get(e, "mean")); M_(f"U{key}{nm}Lo", get(e, "ci95", 0)); M_(f"U{key}{nm}Hi", get(e, "ci95", 1))
        M_(f"U{key}ShareEighty", get(IMG, k, "ratio_of_means_k80"), True)
    for nm, e in (("UPeEeg", get(PE, "untrained_seedmean")), ("UPeTr", get(PE, "saved_trained_eegnet")), ("UPeDiff", get(PE, "trained_minus_untrained")), ("UMiEeg", get(MI, "untrained_seedmean")), ("UMiTr", get(MI, "saved_trained_eegnet"))):
        M_(nm, get(e, "mean")); M_(nm + "Lo", get(e, "ci95", 0)); M_(nm + "Hi", get(e, "ci95", 1))
    M_("UPeShare", get(PE, "untrained_over_trained_ratio"), True); M_("UPeN", get(PE, "n_participants")); M_("UPeSeeds", get(PE, "seeds")); M_("UMiN", get(MI, "n_participants")); M_("UMiSeeds", get(MI, "seeds"))
    return dict(n=len(mc), n_ok=sum(1 for _, ok in mc if ok), failed=[n_ for n_, ok in mc if not ok])


# ============================================================================================ output
def fmt(t, d, pct=False, ci=True):
    st = t.get("status")
    if st == "held back":
        return "held back"
    if st == "undefined" or t.get("value") is None:
        return "undefined"
    s = 100.0 if pct else 1.0
    out = f"{t['value'] * s:.{d}f}" + ("%" if pct else "")
    if ci and t.get("ci"):
        out += f" [{t['ci'][0] * s:.{d}f}, {t['ci'][1] * s:.{d}f}]" + {"bootstrap": "", "stored": "*"}.get((t.get("ci_source") or "").split(" ")[0], "*")
    return out


def main():
    print(f"device {DEV}; smoke {SMOKE}; code sha256 {CODE_SHA}")
    assert DEV == "cuda", "numerics must run on the GPU"
    self_tests()
    macro = build()
    cols = ["target", "decoder", "n", "interval_unit", "unit", "candidates", "exposure"]
    with open(OUT / "rows.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        hdr = list(cols)
        for t in ("Y", "B", "Y0", "net", "R", "U", "K"):
            hdr += [f"{t}_status", f"{t}_value", f"{t}_lo", f"{t}_hi", f"{t}_ci_source", f"{t}_kind", f"{t}_src", f"{t}_note"]
        hdr += ["print_compared", "print_mismatch"]
        w.writerow(hdr)
        for r in ROWS:
            row = [r[c] for c in cols]
            for t in ("Y", "B", "Y0", "net", "R", "U", "K"):
                x = r[t]; ci = x.get("ci") or [None, None]
                row += [x.get("status"), x.get("value"), ci[0], ci[1], x.get("ci_source"), x.get("kind"), x.get("src"), x.get("note")]
            row += [r["print_check"]["n_compared"], r["print_check"]["n_mismatch"]]
            w.writerow(row)
    lines = ["| Decoder (n; interval unit) | Y [95%] | B (type) | Y0 [95%] | net = Y - max(B, Y0) | R (anchor type) | usage | nuisance-adjusted (K) | print check |", "|---|---|---|---|---|---|---|---|---|"]
    tgt = None
    for r in ROWS:
        if r["target"] != tgt:
            tgt = r["target"]; lines.append(f"| **{tgt}**: {r['unit']}; candidates: {r['candidates']} | | | | | | | | |")
        d = r["d"]
        bk = r["B"].get("kind", "") or ""
        bk = bk.split(" (")[0].split(":")[0]
        ustat = r["U"]
        ucell = fmt(ustat, d["U"], pct=True)
        if ustat.get("status") == "ok" and ustat.get("stored_value") is not None and ustat.get("ci_source") == "bootstrap":
            ucell += f" (stored, no blank: {ustat['stored_value'] * 100:.1f}%)"
        kk = r["K"]; kpct = False
        kcell = fmt(kk, d["K"])
        pc = r["print_check"]
        lines.append("| " + " | ".join([f"{r['decoder']} ({r['n']}; {'seed' if 'seed' in r['interval_unit'] else 'participant'})", fmt(r["Y"], d["Y"]), (fmt(r["B"], d["B"], ci=False) + (f" ({bk})" if r["B"].get("value") is not None else "")),
                                        fmt(r["Y0"], d["Y0"]), fmt(r["net"], d["net"]) + (f" ({r['net'].get('kind')})" if r["net"].get("value") is not None else ""),
                                        fmt(r["R"], d["R"], ci=False), ucell, kcell,
                                        ("row missing" if pc.get("missing") else f"{pc['n_compared'] - pc['n_mismatch']}/{pc['n_compared']} equal")]) + " |")
    (OUT / "table.md").write_text("\n".join(lines) + "\n\nIntervals without a mark: participant bootstrap recomputed here (10000, GPU, seed 0). `*`: stored interval copied (no per-participant values behind the gate; text: t interval over seeds).\n")
    n_id = len(IDENT); n_ok = sum(1 for i in IDENT if i["passed"])
    summ = dict(NOT_CITABLE="until independent blind review", experiment="Experiment F: model equation terms over the registered decoders (existing files only)", smoke=SMOKE,
                code="experiments/ext/model_over_decoders.py", code_sha256=CODE_SHA, bootstrap=f"participant, {NB} resamples, GPU torch, seed 0 (where per-participant values are gated); otherwise stored interval copied",
                model_terms="Y yield; B blank; Y0 untrained-encoder baseline; net = Y - max(B, Y0) (terms, not an attribution); R replicate yield (typed); U = (Y - B)/(R - B); K nuisance-adjusted yield or share",
                identity_equations=["mean_j (Y_j - B_j) = Ybar - Bbar", "mean_j (Y_j - Y0_j) = Ybar - Y0bar (net / dividend)", "U = (Y - B)/(R - B) against the stored usage, and its agreement with Y/R when the blank is near zero",
                                    "score identity (protocol.tex eq. identity) Y = I(X;T) + KL(p||q0) - E KL(p(T|X)||q_theta): NOT checkable from stored files (needs the true conditional); only its linear consequences above are checked"],
                rank_across_targets="forbidden", rows=ROWS, identity_checks=IDENT, identity_passed=f"{n_ok}/{n_id}",
                identity_within_project_gate_1e4=f"{sum(1 for i in IDENT if i['within_project_gate_1e4'])}/{n_id}", controls=CONTROLS, findings=FINDINGS, macro_check=macro,
                runtime_s=round(time.time() - T0, 2), device=DEV)
    (OUT / "summary.json").write_text(json.dumps(summ, indent=1))
    print("\n".join(lines))
    print(f"\nidentity checks passed {n_ok}/{n_id} at the strict tolerance; {sum(1 for i in IDENT if i['within_project_gate_1e4'])}/{n_id} within the project 1e-4 gate")
    for i in IDENT:
        if not i["passed"]:
            print("  IDENTITY FAIL:", i["name"], "|", i["equation"], "| lhs", i["lhs"], "rhs", i["rhs"], "diff", i["abs_diff"], "tol", i["tol"], "|", i["note"])
    print("macro check:", macro)
    print(f"findings ({len(FINDINGS)}):")
    for f_ in FINDINGS:
        print("  -", f_["kind"], "|", f_["where"], "|", f_["detail"])
    for r in ROWS:
        if r["print_check"].get("not_printed"):
            print("  not printed in tab_profile:", r["decoder"], "|", "; ".join(r["print_check"]["not_printed"]))
    print("controls:", sum(c["passed"] for c in CONTROLS), "/", len(CONTROLS))
    print("wrote", OUT, f"in {time.time() - T0:.1f}s")


if __name__ == "__main__":
    main()
