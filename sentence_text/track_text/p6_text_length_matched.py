"""P6 text (ideator 13:35, approved): sentence identification restricted to LENGTH-MATCHED candidates (EXPERIMENTS agent, 2026-11-20).
Read-out = the best standard rung of the within-task linear probe, ridge_mean_wordsall (pilot_retrieval.py logic re-run here: Wang & Ji GD word
features averaged over all fixated words, 840-d, z-scored on train; ridge -> MiniLM sentence embedding trained on the task's TRAIN sentences pooled
over participants; alpha by dev top-1 over {1e1..1e5}). Queries = each participant's own reading of each TEST sentence (SR N = 40).
Candidate pool per query = the test sentences whose fixated-word count (this participant's reading; mean over readers if the participant did not
read it) is within +-1 of the query's; the true sentence is always in its pool. Scores per query: top-1, 2-way (ties 1/2), pool size; per
participant: mean top-1, mean chance 1/N_q, 2-way, Wolpaw at the participant's median pool size. Conditions: eeg; noise twin (ridge trained and
queried on N(0,1) features of the same shape); gaze-only (ridge on sentence-level gaze/length scalars: #words, #fixated words, total / mean / sd
nFixations, fraction fixated); shuffled pairs. Unmatched (all-40) scores for reference.
Pass rule (pre-registered): length-matched EEG - twin 2-way, participant bootstrap CI includes 0 -> "no sentence information beyond length"
(pass for the null); excludes 0 -> report the signal. Output: results/loop/p6_nuisance/p6_text_<task>.json"""
import json, os, pickle, sys, time
from pathlib import Path
import numpy as np
from sklearn.linear_model import Ridge
sys.path.insert(0, str(Path(__file__).parent))
from pilot_retrieval import TASKS, load_task, feat, embed_sentences, wolpaw_bits
DATA = "/path/to/eegaudit/data_raw/zuco/eeg2text_pickles"; MINILM = "/path/to/eegaudit/data_raw/zuco/minilm"
OUT = Path("/path/to/eegaudit/results/loop/p6_nuisance"); OUT.mkdir(parents=True, exist_ok=True)


def boot(v):
    rng = np.random.default_rng(0); v = np.asarray(v, float); b = [rng.choice(v, len(v)).mean() for _ in range(5000)]; return [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]


def scores(pred, E, true_pos, pools):
    P = pred / (np.linalg.norm(pred, axis=1, keepdims=True) + 1e-12); S = P @ E.T; t1, tw, ch, ns = [], [], [], []
    for q, (t, pool) in enumerate(zip(true_pos, pools)):
        others = [c for c in pool if c != t]
        if not others: continue
        st = S[q, t]; so = S[q, others]; beat = (so > st).sum(); tie = (so == st).sum()
        t1.append(1.0 / (1 + tie) if beat == 0 else 0.0); tw.append(1 - (beat + 0.5 * tie) / len(others)); ch.append(1.0 / (len(others) + 1)); ns.append(len(others) + 1)
    if not t1: return None
    Nmed = int(np.median(ns)); top1 = float(np.mean(t1))
    return dict(top1=top1, two_way=float(np.mean(tw)), chance_top1=float(np.mean(ch)), top1_minus_chance=top1 - float(np.mean(ch)), median_pool=Nmed, n_queries=len(t1), bits=float(wolpaw_bits(top1, Nmed)) if Nmed >= 2 else 0.0)


for task in sys.argv[1:] or ["SR"]:
    t0 = time.time(); rng = np.random.default_rng(0)
    texts, split, recs, subjects = load_task(os.path.join(DATA, TASKS[task]), task)
    raw = pickle.load(open(os.path.join(DATA, TASKS[task]), "rb"))
    tr = [i for i in split if split[i] == "train"]; dv = [i for i in split if split[i] == "dev"]; te = sorted(i for i in split if split[i] == "test")
    E_all, enc = embed_sentences([texts[i] for i in tr + dv + te], MINILM); E_all = E_all / np.linalg.norm(E_all, axis=1, keepdims=True)
    Etr = {i: E_all[j] for j, i in enumerate(tr)}; Edv = E_all[len(tr):len(tr) + len(dv)]; Ete = E_all[len(tr) + len(dv):]
    dpos = {i: j for j, i in enumerate(dv)}; tpos = {i: j for j, i in enumerate(te)}

    def gaze(sj, i):
        s = raw[sj][i]; nf = np.array([float(w.get("nFixations") or 0) for w in s["word"]]); nw = len(s["content"].split())
        return np.array([nw, len(recs[(sj, i)][0]), nf.sum(), nf[nf > 0].mean() if (nf > 0).any() else 0, nf.std(), len(recs[(sj, i)][0]) / max(nw, 1)])
    feats = {"eeg": lambda sj, i: feat(*recs[(sj, i)], "all", "mean"), "gaze": gaze}
    keys_tr = [(sj, i) for (sj, i) in recs if i in Etr]; keys_dv = [(sj, i) for (sj, i) in recs if i in dpos]
    nfix = {k: len(v[0]) for k, v in recs.items()}
    mean_len = {i: np.mean([nfix[(sj, i)] for sj in subjects if (sj, i) in recs]) for i in te}
    models = {}
    for name, f in feats.items():
        Xtr = np.stack([f(*k) for k in keys_tr]); Ytr = np.stack([Etr[k[1]] for k in keys_tr]); mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
        Xdv = (np.stack([f(*k) for k in keys_dv]) - mu) / sd; ydv = np.array([dpos[k[1]] for k in keys_dv]); best = None
        for a in (1e1, 1e2, 1e3, 1e4, 1e5):
            m = Ridge(alpha=a).fit((Xtr - mu) / sd, Ytr); r = scores(m.predict(Xdv), Edv, ydv, [list(range(len(dv)))] * len(ydv))
            if best is None or r["top1"] > best[0]: best = (r["top1"], a, m)
        models[name] = (best[2], mu, sd, best[1])
        if name == "eeg": models["noise"] = (Ridge(alpha=best[1]).fit(rng.standard_normal(Xtr.shape), Ytr), None, None, best[1])
    per = {}
    for sj in subjects:
        q = [i for i in te if (sj, i) in recs]
        if len(q) < 5: continue
        L = {i: (nfix[(sj, i)] if (sj, i) in recs else mean_len[i]) for i in te}
        pools_m = [[tpos[c] for c in te if abs(L[c] - L[i]) <= 1] for i in q]; pools_all = [list(range(len(te)))] * len(q); tp = [tpos[i] for i in q]
        out = {}
        for cond in ("eeg", "noise", "gaze", "shuffled_pair"):
            if cond == "noise":
                pred = models["noise"][0].predict(rng.standard_normal((len(q), len(models["eeg"][1]))))
            else:
                m, mu, sd, _ = models["gaze" if cond == "gaze" else "eeg"]; f = feats["gaze" if cond == "gaze" else "eeg"]
                pred = m.predict((np.stack([f(sj, i) for i in q]) - mu) / sd)
            tpc = list(np.array(tp)[rng.permutation(len(tp))]) if cond == "shuffled_pair" else tp
            pm = pools_m if cond != "shuffled_pair" else [[tpos[c] for c in te if abs(L[c] - L[te[t]]) <= 1] for t in tpc]
            out[cond] = dict(matched=scores(pred, Ete, tpc, pm), all=scores(pred, Ete, tpc, pools_all))
        per[sj] = out
    summ = {}
    for pool in ("matched", "all"):
        s = {c: {k: float(np.mean([per[p][c][pool][k] for p in per])) for k in ("top1", "two_way", "chance_top1", "top1_minus_chance", "bits", "median_pool")} for c in ("eeg", "noise", "gaze", "shuffled_pair")}
        for other in ("noise", "gaze"):
            d2 = [per[p]["eeg"][pool]["two_way"] - per[p][other][pool]["two_way"] for p in per]; d1 = [per[p]["eeg"][pool]["top1_minus_chance"] - per[p][other][pool]["top1_minus_chance"] for p in per]
            s[f"eeg_minus_{other}_two_way"] = dict(mean=float(np.mean(d2)), ci95=boot(d2), n_pos=int(np.sum(np.array(d2) > 0)))
            s[f"eeg_minus_{other}_top1_above_chance"] = dict(mean=float(np.mean(d1)), ci95=boot(d1))
        summ[pool] = s
    lo, hi = summ["matched"]["eeg_minus_noise_two_way"]["ci95"]
    res = dict(task=task, n_participants=len(per), n_test=len(te), alpha_eeg=models["eeg"][3], alpha_gaze=models["gaze"][3], sentence_encoder=enc, note=__doc__,
               verdict="null holds (no sentence information beyond length)" if lo <= 0 <= hi else "signal beyond length", per_participant=per, summary=summ, runtime_s=round(time.time() - t0, 1))
    (OUT / f"p6_text_{task}.json").write_text(json.dumps(res, indent=1))
    print(task, res["verdict"], json.dumps(summ, indent=1), f"({res['runtime_s']}s)", flush=True)
