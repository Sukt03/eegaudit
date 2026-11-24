"""Shared reader for the paper's figure and table scripts (make_*.py in this folder).

Every number that a figure or table shows is read here from a results/ JSON file. Nothing is typed by hand.
A source is shown only when (a) its file exists and (b) it is marked citable in CITABLE below. Citable means
G1 PASS + bug-review PASS for the code hash that wrote it (METHODOLOGY_LOCK section 11; REVIEWER overnight table
in outputs/EXTENSIONS_LOOP_20261121.md), or a number already registered in the paper. Pending sources return
None, and the scripts print a dash or a placeholder. To switch a pending source on: set its flag to True after
the review passes, then re-run the scripts (see paper/PENDING_FLOATS.md).
"""
import json
from pathlib import Path

ROOT = Path("/path/to/eegaudit")
R = ROOT / "results"
FIGS = ROOT / "paper" / "figs"
import os
TABLES = Path(os.environ.get("PAPER_TABLES_OUT", ROOT / "paper" / "tables"))

CITABLE = {
    "standard/summary.json": True,                       # registered Std* macros
    "baselines/ubp/ubp_summary.json": True,              # UBP G2 + fraction of own ceiling (user-approved C2 wording)
    "baselines/text_card_v2/text_yield_card.json": True,  # 5-seed card: review PASS WITH CONDITIONS; author chose 5 seeds 23 Nov
    "baselines/text/text_yield_card.json": True,         # text noise-twin card, within task (C1)
    "ext/eyequiet_v3/summary_b1.json": True,             # eye-quiet v3, PE (kept 1.05)
    "ext/n1/summary_pe.json": True,                      # registered ExtNonePe* macros (ocular-proxy upper bound)
    "ext/x1_alignment_v3/summary_x1.json": True,         # X1 v3, appendix
    "ext/d2_ds007647_v2/summary.json": True,             # ds007647 v2, appendix diagnostic
    "ext/d6/summary_ladder_so_gpu_n23.json": True,       # registered ExtDsix* macros (Gaussianised bound)
    "ext/d6/summary_inject_so_gpu_n23.json": True,       # registered ExtDsixInj* macros
    "ext/m7/summary_pe.json": False,                      # M7 PE: REVE minus random-init twin (audit PASS, RESULTS sec. 10 / M7 row; G1 PASS m7_latent.py)
    "baselines/ubp": False,                               # UBP soft12 per-participant cards (reference sensitivity vs log2(200); reproduced in the 22 Nov denominator audit)
    "ext/eu_bac/full.json": False,                       # superseded by eu_bac_v2 (soft12), 22 Nov
    "ext/ueyc_v2/profile.json": True,
    "ext/pe_width": True, "ext/bci2a_retest": True, "ext/text_d5_width": True,   # review PASS WITH CONDITIONS (bci2a after flag-1023 fix)
    "gpu_fill_20261123/ubp_heldout": True,               # held-out-concept UBP retrain: design leak-free; cite the 1.0% MEAN inflation with 9/10 unchanged and one participant -10% (review: FAIL as worded, PASS as mean statement)
    "gpu_fill_20261123/mi_fm_meanspec": True,            # FM mean-spectrum surrogate: PASS WITH CONDITIONS (one CBraMod participant 0.0213)
    "gpu_fill_20261123/planted_v2_rho0.25": True,        # planted sweep: PASS WITH CONDITIONS (falsifier fires at rho 0.5 and 0.9)
    "gpu_fill_20261123/planted_v2_rho0.5": True,
    "gpu_fill_20261123/planted_v2_rho0.9": True,
    "gpu_fill_20261123/zuco_plant": True,                # ZuCo plant: PASS WITH CONDITIONS (one seed; recovers < 0.005 bits from a strong planted signal)
    "gpu_fill_20261122/text_e2t_v2": True,
    "gpu_fill_20261123/image_untrained": True,
    "gpu_fill_20261123/pe_untrained": True,              # PE EEGNet untrained encoder: PASS WITH CONDITIONS
    "gpu_fill_20261123/mi_untrained": True,              # MI EEGNet untrained encoder: PASS WITH CONDITIONS
    "ext/eu_bac_v2/full_final.json": True,
    # 23 Nov, v2 numbers wiring (paper/v2/figs/apply_v2_numbers.py); each flag follows a blind review in outputs/FINAL
    "gpu_fill_20261123/mi_retest_summary.json": True,
    "gpu_fill_20261123/na_km3_k40_full": True,           # review item 2: PASS WITH CONDITIONS (soft12 scorer; do not cite the stale g2 field, k40 not k80)
    "runner_fixes_20261123/rf3_pe": True,                # review item 2: PASS WITH CONDITIONS (within-participant PCA-64 ridge read-out; cite raw gain, not gain-minus-floor)
    "ext/mi_card_v3/primary_lpgo": True,                 # per-participant files of the reviewed mi_card_v3 run (review: mi_card_v3.py PASS; summary_mi.json already True)
    "runner_fixes_20261123/rf7_mi_width": True,          # review PASS WITH CONDITIONS (code 37afde58): width curve of the ocular-dominated raw read-out; L90 = 128 of 256 PCA components on an octave grid; one decoder; penalty at the lower grid edge. rf7b_lowlambda is NOT reviewed and has no flag
    "runner_fixes_20261123/rf7c_fullgrid": True,         # 24 Nov: review PASS WITH CONDITIONS (script sha b051b415): MI width curve, 6 widths, PCA and 3 fixed random draws, ridge penalty grid 1e-10..1e4 (23 points), intervals cover participants only. Cite: L90 is AT MOST 128 on this grid (never "= 128", no probability); PCA 8-128 and random 8-128 yields are lower bounds (smallest-penalty folds); no PCA advantage at 8/16/32; new-grid L=256 (0.3006) is NOT the stored mi_card_v3 raw row (0.3016, old grid only); never mix old-grid and new-grid numbers; planted control = pipeline regression test only (thresholds set after the old-grid plant numbers); reference is the 256-component PCA read-out, not the 4480-dim input; one edit to the script after the PREREG stamp (G4 assert -> flag, no effect). Supersedes rf7 / rf7b numbers (old-grid random yields and the +0.175 gap at 128 must not be quoted)
    "runner_fixes_20261123/rf8_bci2a": True,           # review PASS WITH CONDITIONS (code 2b5ad295): BCI-IV-2a participants 2-9, n = 8, t intervals; blanks within 0.02 at cohort level only (participant 2 breaches); directions share the two sessions; supersedes ext/bci2a_retest numbers
    # 24 Nov, batch 5 (outputs/FINAL/review, both items PASS WITH CONDITIONS); wired by paper/v2/figs/apply_v2_batch5.py
    "ext/glim_within_task": True,
    "runner_fixes_20261123/rf9_mi_random_derivations": True,  # random-derivation kept shares: PASS WITH CONDITIONS. summary.json has NO code_sha256 (provenance by recomputation only; the rf2 hash applies to the numbers); cite min to max and draw counts, never p05/p95 as a 90% band; blanks only for draw 0 per family
    "runner_fixes_20261123/rf2_mi": True,                # rf9 reads its arms.json and summary.json; code_sha256 c705c04b... equals rf2_mi_matched_rank.py; PASS WITH CONDITIONS
    # 24 Nov, batch 6: text-probe run, review item 4 PASS WITH CONDITIONS; wired by paper/v2/figs/apply_v2_batch6.py
    "runner_fixes_20261123/rf4_text": True,              # word-level ridge probe on 840-d record-mean word features: gate to the saved probe 7.8e-6. ONLY the original no-intercept ("|orig") cells are informative; the centred ("|c") cells collapse to the training mean (penalty at grid max) and are not cited. Pooled shuffled floor -0.05 is outside +-0.02: report, never subtract. Planted-all (amp 1) is recovered within task; the subspace plants (m = 13, 52) and every L90 are NOT validated (m = 13 not recovered in the pooled cell): do not cite L90. Probe power says nothing about the free-running decoders. No pooled "task identity" claim (no task-label control)
    # 24 Nov, batch 8: outputs/FINAL/review; wired by paper/v2/figs/apply_v2_batch8.py
    "ext/a3_participant_detection": True,               # A3 PASS WITH CONDITIONS: participant detected when its yield exceeds the 95th percentile of the POOLED STORED blank values of the cell (all arms, all participants); exact one-sided binomial against 5%, claim at p < 0.01. Conditions: (1) EEGPT is detected under the rule only, its p is about 0.02 against its own phase arm's exceedance rate: report as the weakest detection with that fact beside it; (2) the FM phase-scrambled arms breach +-0.02 in 22-56 of 105 participants: say so wherever they serve as a blank; (3) ds007647 ridge is 'not separated from its blank at the participant level' (6 of 39, p = 0.012), never 'absent' (the read-out does not recover an injected signal). Never: a 95% tolerance, a permutation null, Allefeld's prevalence test, a binomial calibrated for the phase arms, a clean blank for the FM detections
    "ext/nice_temp_fixed": True,                         # A4 PASS WITH CONDITIONS: NICE-GA k=80 Delta_L = 12.4% [10.8, 13.6] at tau 0.07, k=40 12.2% [10.6, 13.3]; NOT SMALL. Conditions: (1) an UPPER BOUND on what test-query calibration adds (training temperature not fitted for 80-rep averages or concept centres; 0.035 / 0.0525 lose 1.2% / 3.0%), say so; (2) fixed-tau floors are -0.8 to -1.6 bits (one-sided gate), never 'within 0.02 in absolute value'; (3) soft12 picks one grid point (0.0433) for all queries. Never: a rescue of E2 or a train-only temperature; 'leakage'; a D0 or 'transductive' change; a floor-subtracted fixed-tau yield; one comparison sentence with the UBP 1.0%; any BrainFLORA Delta
    "ext/a5_pe_carryover": True,                         # A5 PASS as a correctly executed FAILED check (ridge read-out, 0.094 bits, NOT EEGNet): (i) 21/23 pass, (ii) lag ratios fail at -1, +1, +2, (iii) kept share 0.145 [0.098, 0.220] < 0.5; controls 20/23 slow and 0/23 event pass. Only the pre-registered fallback: 'a slow-state route is not excluded (ridge read-out)'. Never: event-specific; a slow state demonstrated or 'explains' the yield; 85% carry-over or artefact; the block null alone excludes slow state; lag +1 makes lag 0 spurious; holds for EEGNet or the 0.110 headline; part (a) numbers as products of this run
    # 24 Nov, batch 8 (continued): outputs/FINAL/review, both PASS WITH CONDITIONS; wired by paper/v2/figs/apply_v2_batch8.py
    "ext/b1_sandwich_bracket": True,                    # B1 PASS on execution, PASS WITH CONDITIONS as a claim: neural-only yield inside [regression yield, raw yield + 0.02] in 9/9 planted cells at L64 and full width, but the ordering is expected by construction in the generator; the informative quantity is the width (narrower than the yield only at rho 0.25; 1.3-5.4x at rho 0.5, 3-21x at rho 0.9); the ocular-only yield is also inside in 5/9 cells; the nine cells rest on 23 participants, 12 reused across rho with the same neural plant. APPENDIX ONLY. Never: 'identifies/estimates/bounds tightly the neural share', coverage as evidence for the method, 'as wide as the yield only at rho 0.9', per-participant coverage, nine independent replicates; the main text does not mention a bracket
    "ext/b2_mi_cue_visual": True,
    # 24 Nov, batch 9: outputs/FINAL/review; wired by paper/v2/figs/apply_v2_batch9.py
    # pending (scripts read these paths; flip to True once reviewed)
    "ext/s1_v6/summary.json": False,                     # S1 v6 (soft12 grid)
    "ext/w1_v3/summary_w1.json": False,                  # W1 v3
    "ext/km3_v2/summary_km3.json": False,                # KM3 v2 (n = 10, G2 required)
    "ext/mi_card/primary_lpgo/summary_mi.json": False,
    "ext/mi_card_v3/primary_lpgo/summary_mi.json": True,  # MI card v3 (EOG-proxy-regressed rungs)
    "gpu_fill_20261122/pe_eegnet_floor": True,          # PE EEGNet refit floors (review PASS)
    "gpu_fill_20261122/mi_eegnet_floor": True,          # MI EEGNet refit floors (review PASS)
    "gpu_fill_20261122/ubp_noise_seeds": True,
    "baselines/ubp_phase_control": True,                # UBP phase/shuffle seeds 0-2 (earlier phase review PASS)
    "baselines/ubp_phase_control_s3to6": True,          # seeds 3-6
    "baselines/ubp_phase_control_s7to9": True,          # seeds 7-9
    "baselines/ubp_noise_input": True,                  # UBP noise-input floor, original seed (review PASS)
    "gpu_fill_20261122/mi_fm_phase": True,              # MI FM phase-surrogate scores: numbers reproduce; NOT a noise floor (exploratory, interpretation caveats)
    "gpu_fill_20261122/planted_calib": True,            # planted-information calibration: numbers reproduce; estimand mismatch (exploratory, interpretation caveats)
    "ext/km3_decoders_v3": True,
    "ext/mi_fm_eogreg": True,                           # MI FM rows on the regressed input (<model>/summary.json)
    "ext/mi_fm": False,                                  # MI FM rows on the unregressed input (<model>/summary.json)
    "baselines/dtha/dtha_summary.json": False,           # DTHA re-implementation card (per-member files -> summary)
    "baselines/brainmagick/bm_summary.json": False,      # brainmagick-style speech card
    "baselines/text/g2_paper_protocol/g2_summary.json": False,  # text G2 under the papers' own protocol
    "baselines/image_topk_g2.json": False,               # image top-k G2 (published top-k reproduced)
    "ext/i3_planted_ocular/summary_i3.json": False,      # I-3 planted ocular signal (n = 23; method review pending)
    "ext/eyequiet_v3_ds004595/summary_b1.json": False,   # eye-quiet v3, ds004595 (not yet reviewed)
    "ext/pe_specificity_v2/summary_specificity.json": False,  # event-based eye regressors (not yet reviewed)
}


# Every numeric leaf a script reads through load() is recorded here as (results-relative file, dotted key), so that
# register() can put it in the paper's number registry (ledger.py) and Gate A traces each printed number to its key.
READ = []


class _D(dict):
    def __init__(self, d, rel, path):
        super().__init__(d); self._rel, self._path = rel, path

    def __getitem__(self, k):
        return _wrap(dict.__getitem__(self, k), self._rel, self._path + [str(k)])

    def get(self, k, default=None):
        return self[k] if k in self else default


class _L(list):
    def __init__(self, v, rel, path):
        super().__init__(v); self._rel, self._path = rel, path

    def __getitem__(self, i):
        return _wrap(list.__getitem__(self, i), self._rel, self._path + [str(i)])


def _wrap(v, rel, path):
    if isinstance(v, dict):
        return _D(v, rel, path)
    if isinstance(v, list):
        return _L(v, rel, path)
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        READ.append((rel, ".".join(path)))
    return v


def load(rel):
    """Return the parsed JSON for results/<rel>, or None if it is missing, unreadable or not citable."""
    if not CITABLE.get(rel, False):
        return None
    p = R / rel
    try:
        return _wrap(json.loads(p.read_text()), rel, [])
    except Exception:
        return None


def load_raw(rel, sub):
    """Plain (untracked) JSON of one file in a gated results directory; for counts and maxima over per-participant files."""
    if not CITABLE.get(rel, False):
        return None
    try:
        return json.loads((R / rel / sub).read_text())
    except Exception:
        return None


def load_file(rel, sub):
    """Like load(), for one file inside a gated results directory (rel = the directory key in CITABLE)."""
    if not CITABLE.get(rel, False):
        return None
    try:
        return _wrap(json.loads((R / rel / sub).read_text()), f"{rel}/{sub}", [])
    except Exception:
        return None


EXP = ROOT / "experiments" / "paper_floats"
LEDGER = Path("/path/to/tools")


def _name(i):
    s = ""
    i += 26 * 26  # at least three letters
    while i:
        i, r = divmod(i, 26); s = chr(65 + r) + s
    return "Fl" + s


def register():
    """Register every value read in this run with ledger.py (experiments/paper_floats) and export paper_numbers.json.
    The registry is rebuilt from scratch each run, so it always matches the floats the scripts emit."""
    import subprocess, sys
    EXP.mkdir(parents=True, exist_ok=True)
    if not (EXP / "attempts.tsv").exists():
        subprocess.run([sys.executable, str(LEDGER), "init", str(EXP), "--question",
                        "numbers shown in the paper's generated floats (paper/figs/make_*.py)", "--metric", "none",
                        "--direction", "max"], check=True, capture_output=True)
    # same record format as `ledger.py number` (its own read_value and sha), written in one pass; then `ledger.py export`
    import importlib.util
    from datetime import datetime
    spec = importlib.util.spec_from_file_location("ledger", LEDGER); L = importlib.util.module_from_spec(spec); spec.loader.exec_module(L)
    seen = sorted(set(READ)); reg, shas = {}, {}
    for i, (rel, key) in enumerate(seen):
        p = (R / rel).resolve()
        shas.setdefault(p, L.sha(p))
        reg[_name(i)] = {"file": str(p), "key": key, "where": None, "scale": 1.0, "fmt": ".6g", "raw": L.read_value(p, key, None),
                         "sha256": shas[p], "registered": datetime.now().isoformat(timespec="seconds")}
    (EXP / "numbers_registry.json").write_text(json.dumps(reg, indent=2))
    subprocess.run([sys.executable, str(LEDGER), "export", str(EXP)], check=True, capture_output=True)
    print(f"registered {len(seen)} values in {EXP}/paper_numbers.json")


def get(obj, *keys):
    """Nested lookup that returns None instead of raising."""
    for k in keys:
        try:
            obj = obj[k]
        except Exception:
            return None
    return obj


def f(x, d=2, plus=False, pct=False):
    """Format a number for LaTeX; None -> en dash placeholder."""
    if x is None:
        return "--"
    try:
        x = float(x)
    except Exception:
        return "--"
    if pct:
        x = 100 * x
    s = f"{x:+.{d}f}" if plus else f"{x:.{d}f}"
    if s.startswith("-"):
        s = "$" + s + "$"  # a true minus sign that keeps the sign attached to the number (Gate A reads "-1.08")
    return s


def fci(ci, d=2, pct=False):
    if ci is None or len(ci) != 2 or None in ci:
        return ""
    return r"{\scriptsize[" + f(ci[0], d, pct=pct) + ", " + f(ci[1], d, pct=pct) + "]}"


def style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7, "xtick.labelsize": 6.5,
                         "ytick.labelsize": 6.5, "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "font.family": "serif",
                         "font.serif": ["Times New Roman", "Nimbus Roman", "Times", "DejaVu Serif"]})
    return plt


# Okabe-Ito colours
BLUE, VERM, GREEN, GREY, ORANGE, SKY, PURPLE = "#0072B2", "#D55E00", "#009E73", "#7f7f7f", "#E69F00", "#56B4E9", "#CC79A7"
