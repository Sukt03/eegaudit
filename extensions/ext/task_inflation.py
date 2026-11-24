"""N-A. TASK-IDENTITY INFLATION INDEX (text; rung R1/R3; twin-referenced). FIXER-2, 21 Nov 2026.
Spec: EXTENSIONS_LOOP_20261121 "N-A" + bug pre-review (length-only task reference; reuse the card's MODELS tag map).
Reuse: the HEAD of experiments/baselines/text_yield_card.py (everything before the scoring loop: tag map MODELS / REPLICATE, leakage audit,
  MiniLM emb, infonce, score_set, cboot, load_jsonl, norm_text) is executed UNCHANGED from its source file (the s13b_targets pattern; the
  card is a script, not an importable module). Its code_sha256 is printed beside ours.
Per model x seed x mode (greedy, beam) with an eeg arm and a noise arm (trained N(0,1) twin; Jo et al. 2024):
  (a) Y_pool  per-record yield on the pooled test candidate set (card 'pooled');
  (b) Y_within per-record yield on the record's own task set (card 'within_SR/NR/NR2'); record-weighted mean over the three tasks;
  (c) INFLATION = Y_pool - Y_within (per record; mean);  INFLATION(eeg) - INFLATION(noise): paired per record, sentence-cluster bootstrap
      (card cboot); the arms that exist (random_init, lm_prior_ablation, shuffled) are printed too.
  G2: mean Y_pool / Y_within of every arm reproduce the card's score_set means (computed in the same run) to 1e-9.
(d) TASK-ONLY reference (model-free, the same test records): a ridge from the EEG input features the models saw (the record's 840-d
  word-band features averaged over its words, cache_full.pkl) to the one-hot task (SR / NR / NR2), alpha and a softmax temperature chosen
  by 5-fold inner CV on TRAINING records (v2: folds = 5 contiguous blocks of sentence index, as the test split; criterion = cross-entropy); held-out p(task | EEG) -> identification
  score of a decoder that knows only the task: log2 [N_pool p(T_true) / |T_true|] (|T| = unique test candidates of task T), in bits per
  record; participant bootstrap (10,000). LENGTH-ONLY task reference (bug pre-review): the same pipeline on [n_words, log n_words];
  EEG + length = both. Floor = the label-free base-rate decoder (p(T) = training task frequency; 'prior_bits'); every reference is
  also reported minus that floor. G3: the reference refit on permuted training task labels (3 permutations) is within 0.02 of the floor.
  Ceiling for orientation: log2(N_pool / |T|) when p(T_true) = 1 (record mean printed).
Falsifier (pre-stated): pooled - within CI includes 0 for EVERY model AND the task-only reference sits at its floor -> "no task inflation".
GLIM released generations (eeg vs test-time noise; card rows): inflation printed (no trained twin, no cache features -> no task-only row).
Output: results/baselines/text/task_inflation/task_inflation[_smoke].json (code_sha256).
Usage (--smoke = random 3000 train / 600 test records for (d); generation rows are always complete): python -I -u experiments/ext/task_inflation.py [--smoke] [--models braintranslator,...] --out DIR
"""
import argparse, hashlib, json, math, os, pickle, sys, time
from pathlib import Path
import numpy as np, torch

VERSION = "v2 (contiguous sentence-block inner CV for the task-only reference)"
CODE_SHA256 = hashlib.sha256(open(__file__, "rb").read()).hexdigest()
ROOT = Path("/path/to/eegaudit"); CARD = ROOT / "experiments/baselines/text_yield_card.py"
ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--smoke", action="store_true"); ap.add_argument("--models", default="")
A = ap.parse_args(); out = Path(A.out); out.mkdir(parents=True, exist_ok=True); t0 = time.time()
src = CARD.read_text(); head = src[:src.index("\nout = {'note'")]
ns = {"__file__": str(CARD), "__name__": "text_yield_card_head"}
_argv = sys.argv; sys.argv = [str(CARD), "--out", str(out)]; exec(compile(head, str(CARD), "exec"), ns); sys.argv = _argv
dev = ns["dev"]; infonce, emb, cboot, score_set, load_jsonl, norm_text = (ns[k] for k in ("infonce", "emb", "cboot", "score_set", "load_jsonl", "norm_text"))
TASKS = ("SR", "NR", "NR2")


def per_record(rows_by_arm, mode, sel_task=None):
    """= score_set's yield computation (same record filter, candidate set, emb, infonce), returning per-record values per arm."""
    arms = list(rows_by_arm); base = rows_by_arm[arms[0]]
    keep = [i for i, r in enumerate(base) if sel_task is None or r["task"] == sel_task]
    if len(keep) < 10: return None
    tg = [base[i]["target"] for i in keep]; U = list(dict.fromkeys(tg)); pos = {u: j for j, u in enumerate(U)}
    true = torch.tensor([pos[t] for t in tg], device=dev); cl = true.clone(); EU = emb(U)
    ys = {arm: infonce(emb([rows_by_arm[arm][i][mode] or "." for i in keep]) @ EU.T, true, cl) for arm in arms}
    return dict(keep=keep, ys=ys, N=len(U))


def inflation(rows, mode):
    P = per_record(rows, mode); n = len(rows["eeg"]); arms = list(rows); W = {arm: torch.full((n,), float("nan"), device=dev) for arm in arms}; g2 = {}
    card_pool = score_set(rows, mode); g2["pool"] = max(abs(float(P["ys"][a].mean()) - card_pool["arms"][a]["yield_bits"]) for a in arms)
    for t in TASKS:
        Q = per_record(rows, mode, sel_task=t)
        if Q is None: continue
        for a in arms: W[a][torch.tensor(Q["keep"], device=dev)] = Q["ys"][a]
        cw = score_set(rows, mode, sel_task=t); g2["within_" + t] = max(abs(float(Q["ys"][a].mean()) - cw["arms"][a]["yield_bits"]) for a in arms)
    ok = torch.isfinite(W["eeg"]); assert ok.all(), "record without a within-task set"
    tg = [r["target"] for r in rows["eeg"]]; U = list(dict.fromkeys(tg))
    pos = {u: j for j, u in enumerate(U)}; cl = torch.tensor([pos[t] for t in tg], device=dev)
    R = dict(n_records=n, N_pool=P["N"], g2_max_abs_diff=max(g2.values()), g2=g2, arms={})
    infl = {a: P["ys"][a] - W[a] for a in arms}
    for a in arms:
        R["arms"][a] = dict(Y_pool=float(P["ys"][a].mean()), Y_within=float(W[a].mean()), inflation=float(infl[a].mean()), ci95=cboot({"i": infl[a]}, cl)["i"])
    for a in arms:
        if a == "eeg": continue
        d = infl["eeg"] - infl[a]; R[f"inflation_eeg_minus_{a}"] = dict(mean=float(d.mean()), ci95=cboot({"d": d}, cl)["d"])
    R["per_record_keys"] = [(r["subject"], r["g"]) for r in rows["eeg"]]
    return R


# ---------------- (d) task-only / length-only references (model-free)
def task_reference():
    D = pickle.load(open(ns["a"].cache, "rb")); recs = [r for r in D["recs"] if r["split"] in ("train", "test")]
    if A.smoke:                                                           # random subsets (all three tasks present)
        rg = np.random.default_rng(0); trr = [r for r in recs if r["split"] == "train"]; ter = [r for r in recs if r["split"] == "test"]
        recs = [trr[i] for i in rg.choice(len(trr), 3000, replace=False)] + [ter[i] for i in rg.choice(len(ter), 600, replace=False)]
    tr = [r for r in recs if r["split"] == "train"]; te = [r for r in recs if r["split"] == "test"]; del D
    ti = {t: k for k, t in enumerate(TASKS)}
    feat_eeg = lambda rs: np.stack([r["words"].mean(0) for r in rs]).astype(np.float64)
    feat_len = lambda rs: np.array([[len(r["words"]), math.log(len(r["words"]))] for r in rs], np.float64)
    texts = ns["texts"]; teU = {}
    for r in te: teU.setdefault(texts[r["g"]], r["task"])
    Np = len(teU); nT = {t: sum(1 for v in teU.values() if v == t) for t in TASKS}
    ytr = np.array([ti[r["task"]] for r in tr]); yte = np.array([ti[r["task"]] for r in te]); gtr = np.array([r["g"] for r in tr]); part = [r["subject"].split(":")[1] for r in te]
    LAMS = torch.logspace(-3, 4, 15, dtype=torch.float64, device=dev); TS = torch.logspace(-2, 1, 25, dtype=torch.float64, device=dev)

    def fit_predict(Xtr, ylab, Xte):
        Xtr = torch.tensor(Xtr, device=dev); Xte = torch.tensor(Xte, device=dev); Y = torch.nn.functional.one_hot(torch.tensor(ylab, device=dev), 3).double()
        def ridge(Xa, Ya, Xb, lam):
            mu, sd = Xa.mean(0), Xa.std(0) + 1e-9; Za = (Xa - mu) / sd; Zb = (Xb - mu) / sd; ym = Ya.mean(0)
            Wt = torch.linalg.solve(Za.T @ Za + lam * len(Za) * torch.eye(Za.shape[1], dtype=torch.float64, device=dev), Za.T @ (Ya - ym)); return Zb @ Wt + ym
        ug = np.unique(gtr); fold = np.searchsorted(ug, gtr) * 5 // len(ug); CE = torch.zeros(len(LAMS), len(TS), dtype=torch.float64, device=dev)   # v2: 5 CONTIGUOUS sentence blocks (the test split is a sentence block; v1 g % 5 interleaving was over-confident on test)
        for f in range(5):
            a_, b_ = torch.tensor(fold != f, device=dev), torch.tensor(fold == f, device=dev)
            for i, lam in enumerate(LAMS):
                S = ridge(Xtr[a_], Y[a_], Xtr[b_], lam); lp = torch.log_softmax(S[None] / TS[:, None, None], -1)
                CE[i] += -lp[:, torch.arange(int(b_.sum()), device=dev), torch.tensor(ylab, device=dev)[b_]].sum(1)
        i, j = np.unravel_index(int(CE.argmin()), CE.shape)
        S = ridge(Xtr, Y, Xte, LAMS[i]); p = torch.softmax(S / TS[j], -1)
        return p, dict(lam=float(LAMS[i]), T=float(TS[j]), lam_at_edge=bool(i in (0, len(LAMS) - 1)), T_at_edge=bool(j in (0, len(TS) - 1)))

    def bits(p):
        pt = p[torch.arange(len(yte), device=dev), torch.tensor(yte, device=dev)]; sz = torch.tensor([nT[TASKS[k]] for k in yte], dtype=torch.float64, device=dev)
        return torch.log2(Np * pt / sz)

    pids = sorted(set(part)); pix = torch.tensor([pids.index(x) for x in part], device=dev)
    g = torch.Generator(device=dev); g.manual_seed(20261121); IX = torch.randint(0, len(pids), (10000, len(pids)), generator=g, device=dev)
    def pboot(v):                                                         # participant bootstrap of the record mean
        s = torch.zeros(len(pids), dtype=torch.float64, device=dev).index_add_(0, pix, v); c = torch.bincount(pix, minlength=len(pids)).double()
        m = s[IX].sum(1) / c[IX].sum(1); return [float(torch.quantile(m, .025)), float(torch.quantile(m, .975))]
    R = dict(n_train=len(tr), n_test=len(te), N_pool=Np, task_set_sizes=nT, participants=pids, unit="bits per record (identification among the pooled test candidates)",
             ceiling_task_known=float(np.mean([math.log2(Np / nT[TASKS[k]]) for k in yte])), refs={}, per_record_keys=[(r["subject"], r["g"]) for r in te])
    prior = np.bincount(ytr, minlength=3) / len(ytr); pb = bits(torch.tensor(prior, device=dev)[None].expand(len(yte), 3))   # label-free base-rate decoder = the floor
    R["prior_bits"] = float(pb.mean()); R["prior_task_freq_train"] = prior.tolist()
    per = {}
    for name, Xtr, Xte in (("eeg", feat_eeg(tr), feat_eeg(te)), ("length", feat_len(tr), feat_len(te)), ("eeg_plus_length", np.c_[feat_eeg(tr), feat_len(tr)], np.c_[feat_eeg(te), feat_len(te)])):
        p, sel = fit_predict(Xtr, ytr, Xte); b = bits(p); per[name] = b
        R["refs"][name] = dict(bits=float(b.mean()), ci95_participant=pboot(b), minus_prior=float((b - pb).mean()), minus_prior_ci95_participant=pboot(b - pb), task_acc=float((p.argmax(1).cpu().numpy() == yte).mean()), selection=sel)
        if name == "eeg":
            fl = []
            for k in range(3):
                pp, _ = fit_predict(Xtr, np.random.default_rng(700 + k).permutation(ytr), Xte); fl.append(float(bits(pp).mean()))
            R["refs"][name]["g3_permuted_label_floor"] = fl; R["refs"][name]["g3_pass"] = bool(max(abs(x - R["prior_bits"]) for x in fl) <= 0.02)   # vs the base-rate floor
    d = per["eeg"] - per["length"]; R["eeg_minus_length"] = dict(mean=float(d.mean()), ci95_participant=pboot(d))
    d = per["eeg_plus_length"] - per["length"]; R["eeg_plus_length_minus_length"] = dict(mean=float(d.mean()), ci95_participant=pboot(d))
    R["_per_record"] = {k: v.cpu().numpy().tolist() for k, v in per.items()}
    return R


res = dict(version=VERSION, code_sha256=CODE_SHA256, card_code_sha256=ns["CODE_SHA256"], smoke=A.smoke, note=__doc__, models={})
TR = task_reference(); print("TASK REF prior", round(TR["prior_bits"], 4), {k: (round(v["bits"], 4), round(v["minus_prior"], 4), v["minus_prior_ci95_participant"], round(v["task_acc"], 3)) for k, v in TR["refs"].items()}, "g3", TR["refs"]["eeg"].get("g3_permuted_label_floor"), f"({time.time()-t0:.0f}s)", flush=True)
tr_map = {tuple(k): i for i, k in enumerate(TR["per_record_keys"])}
sel_models = [m for m in ns["MODELS"] if not A.models or m in A.models.split(",")]
for model in sel_models:
    for seed, arms in ns["MODELS"][model].items():
        if (model, f"seed{seed}") in ns["REPLICATE"]: continue
        rows = {k: load_jsonl(p + "_generations.jsonl") for k, p in arms.items() if os.path.exists(p + "_generations.jsonl")}
        if "eeg" not in rows or "noise" not in rows: continue
        order = [r["g"] for r in rows["eeg"]]
        if any([r["g"] for r in rows[k]] != order for k in rows): print("SKIP order mismatch", model, seed, flush=True); continue
        M = {}
        for mode in ("greedy", "beam"):
            R = inflation(rows, mode); keys = R.pop("per_record_keys"); ix = [tr_map.get(tuple(k)) for k in keys]
            if all(i is not None for i in ix):
                R["task_only_eeg_bits_same_records"] = float(np.mean([TR["_per_record"]["eeg"][i] for i in ix])); R["task_only_length_bits_same_records"] = float(np.mean([TR["_per_record"]["length"][i] for i in ix]))
            else:
                R["task_only_note"] = f"{sum(i is None for i in ix)} records not in the cache test split"
            M[mode] = R
        res["models"].setdefault(model, {})[f"seed{seed}"] = M; b = M["beam"]
        print(model, f"seed{seed}", "beam infl eeg/noise", round(b["arms"]["eeg"]["inflation"], 4), round(b["arms"]["noise"]["inflation"], 4), "eeg-noise", b["inflation_eeg_minus_noise"], "G2", f"{b['g2_max_abs_diff']:.1e}", f"({time.time()-t0:.0f}s)", flush=True)
# seed summary per model: mean over seeds of inflation(eeg) - inflation(noise), per mode
res["summary"] = {}
for model, by in res["models"].items():
    S = {}
    for mode in ("greedy", "beam"):
        v = [x[mode]["inflation_eeg_minus_noise"]["mean"] for x in by.values()]
        S[mode] = dict(n_seeds=len(v), per_seed=v, mean=float(np.mean(v)), per_seed_cluster_ci95=[x[mode]["inflation_eeg_minus_noise"]["ci95"] for x in by.values()],
                       inflation_eeg=float(np.mean([x[mode]["arms"]["eeg"]["inflation"] for x in by.values()])), inflation_noise=float(np.mean([x[mode]["arms"]["noise"]["inflation"] for x in by.values()])),
                       all_seed_cis_include_0=bool(all(c[0] <= 0 <= c[1] for c in (x[mode]["inflation_eeg_minus_noise"]["ci95"] for x in by.values()))))
    res["summary"][model] = S
# GLIM released (eeg vs test-time noise), card rows
import csv
from collections import defaultdict
g_rows = {}
for arm, fn in (("eeg", "wandb_export_gen_samples_glim.csv"), ("noise", "wandb_export_gen_samples_noise_input_test.csv")):
    rr = list(csv.DictReader(open(os.path.join(ns["GLIM"], fn), encoding="utf-8"))); occ = defaultdict(int); keyed = {}
    for r in rr:
        k = (r["Task"], r["Dataset"], r["Subject"], r["Input Text"]); occ[k] += 1; keyed[k + (occ[k],)] = {"target": r["Input Text"], "gen": r["Generated Text"], "task": r["Task"] + r["Dataset"], "subject": r["Subject"], "g": 0}
    g_rows[arm] = keyed
common = sorted(set(g_rows["eeg"]) & set(g_rows["noise"])); grows = {arm: [dict(g_rows[arm][k], g=i) for i, k in enumerate(common)] for arm in g_rows}
TASKS_G = sorted({r["task"] for r in grows["eeg"]})
_T = TASKS; TASKS = tuple(TASKS_G); G = inflation(grows, "gen"); G.pop("per_record_keys"); TASKS = _T
res["glim_released"] = dict(G, label="released generations; noise = test-time noise input (no trained twin); tasks " + ",".join(TASKS_G))
TR.pop("_per_record"); TR.pop("per_record_keys"); res["task_reference"] = TR
fals = all(all(x["beam"]["inflation_eeg_minus_noise"]["ci95"][0] <= 0 <= x["beam"]["inflation_eeg_minus_noise"]["ci95"][1] for x in by.values()) for by in res["models"].values())
res["falsifier"] = dict(rule="pooled - within (EEG - noise) CI includes 0 for EVERY model AND the task-only reference at floor -> no task inflation (beam, per seed)",
                        all_models_eeg_minus_noise_ci_include_0=bool(fals), task_only_eeg_minus_prior_ci=TR["refs"]["eeg"]["minus_prior_ci95_participant"],
                        triggered=bool(fals and TR["refs"]["eeg"]["minus_prior_ci95_participant"][0] <= 0))
res["g2_all_pass"] = bool(all(x[m]["g2_max_abs_diff"] < 1e-9 for by in res["models"].values() for x in by.values() for m in ("greedy", "beam")) and G["g2_max_abs_diff"] < 1e-9)
res["runtime_s"] = round(time.time() - t0, 1)
fn = out / ("task_inflation_smoke.json" if A.smoke else "task_inflation.json"); fn.write_text(json.dumps(res, indent=1))
print("SUMMARY", json.dumps(res["summary"], indent=0)[:3000], "\nGLIM", {k: round(v["inflation"], 4) for k, v in G["arms"].items()}, "\nfalsifier", res["falsifier"], "g2_all_pass", res["g2_all_pass"], "\nwrote", fn, f"{time.time()-t0:.0f}s", flush=True)
