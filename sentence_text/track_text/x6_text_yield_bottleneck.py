"""X6: ZuCo text yield vs input bottleneck width (protocol step b; EXPERIMENTS agent, 2026-11-20).
Read-out = ridge_mean_wordsall (Wang & Ji 840-d GD word features averaged over fixated words -> MiniLM sentence embedding; trained on the task's
train sentences pooled over participants; z-scored on train; alpha by dev top-1 over {1e1..1e5} at every width). Bottleneck: PCA basis fitted on the
TRAINING features only (z-scored), L in WIDTHS; random-input bottleneck = random orthonormal L-dim projection (seed 0). Queries = each participant's
own reading of each test sentence; candidates = all test sentences (N) and the length-matched pool (+-1 fixated word, as P6).
Floors at every L: noise twin (ridge trained and queried on N(0,1) of the same L-dim shape), shuffled pairs.
Output: results/loop/x6_text_bottleneck/x6_<task>.json   Usage: python x6_text_yield_bottleneck.py SR [NR2]"""
import json, os, sys, time
from pathlib import Path
import numpy as np
from sklearn.linear_model import Ridge
sys.path.insert(0, str(Path(__file__).parent))
from pilot_retrieval import TASKS, load_task, feat, embed_sentences, wolpaw_bits
DATA = "/path/to/eegaudit/data_raw/zuco/eeg2text_pickles"; MINILM = "/path/to/eegaudit/data_raw/zuco/minilm"
OUT = Path("/path/to/eegaudit/results/loop/x6_text_bottleneck"); OUT.mkdir(parents=True, exist_ok=True)
WIDTHS = [1, 2, 5, 10, 20, 50, 100, 200, 400, 840]


def boot(v):
    rng = np.random.default_rng(0); v = np.asarray(v, float); b = [rng.choice(v, len(v)).mean() for _ in range(5000)]; return [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]


TAUS = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]


def nce(pred, E, true, tau):
    """InfoNCE bound in bits over the full candidate set: log2 N + mean log2 softmax(cos/tau)[true]."""
    P = pred / (np.linalg.norm(pred, axis=1, keepdims=True) + 1e-12); S = (P @ E.T) / tau; S = S - S.max(1, keepdims=True)
    lp = S - np.log(np.exp(S).sum(1, keepdims=True)); return float(np.log2(E.shape[0]) + np.mean(lp[np.arange(len(true)), true]) / np.log(2))


def ident(pred, E, true, pools):
    P = pred / (np.linalg.norm(pred, axis=1, keepdims=True) + 1e-12); S = P @ E.T; t1, tw = [], []
    for q, (t, pool) in enumerate(zip(true, pools)):
        o = [c for c in pool if c != t]
        if not o: continue
        st, so = S[q, t], S[q, o]; beat, tie = (so > st).sum(), (so == st).sum()
        t1.append(1.0 / (1 + tie) if beat == 0 else 0.0); tw.append(1 - (beat + 0.5 * tie) / len(o))
    return float(np.mean(t1)), float(np.mean(tw))


for task in sys.argv[1:] or ["SR"]:
    t0 = time.time(); rng = np.random.default_rng(0)
    texts, split, recs, subjects = load_task(os.path.join(DATA, TASKS[task]), task)
    tr = [i for i in split if split[i] == "train"]; dv = [i for i in split if split[i] == "dev"]; te = sorted(i for i in split if split[i] == "test")
    E_all, enc = embed_sentences([texts[i] for i in tr + dv + te], MINILM); E_all = E_all / np.linalg.norm(E_all, axis=1, keepdims=True)
    Etr = {i: E_all[j] for j, i in enumerate(tr)}; Edv = E_all[len(tr):len(tr) + len(dv)]; Ete = E_all[len(tr) + len(dv):]
    dpos = {i: j for j, i in enumerate(dv)}; tpos = {i: j for j, i in enumerate(te)}; N = len(te)
    f = lambda k: feat(*recs[k], "all", "mean")
    ktr = [k for k in recs if k[1] in Etr]; kdv = [k for k in recs if k[1] in dpos]
    Xtr = np.stack([f(k) for k in ktr]); Ytr = np.stack([Etr[k[1]] for k in ktr]); mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8; Ztr = (Xtr - mu) / sd
    Zdv = (np.stack([f(k) for k in kdv]) - mu) / sd; ydv = np.array([dpos[k[1]] for k in kdv])
    _, _, Vt = np.linalg.svd(Ztr, full_matrices=False); Q, _ = np.linalg.qr(np.random.default_rng(0).standard_normal((Ztr.shape[1], Ztr.shape[1])))
    nfix = {k: len(v[0]) for k, v in recs.items()}; mean_len = {i: np.mean([nfix[(sj, i)] for sj in subjects if (sj, i) in recs]) for i in te}
    queries = {}
    for sj in subjects:
        q = [i for i in te if (sj, i) in recs]
        if len(q) < 5: continue
        L = {i: (nfix[(sj, i)] if (sj, i) in recs else mean_len[i]) for i in te}
        queries[sj] = (q, (np.stack([f((sj, i)) for i in q]) - mu) / sd, [tpos[i] for i in q], [[tpos[c] for c in te if abs(L[c] - L[i]) <= 1] for i in q])
    curve = []
    for basis_name, B in (("pca", Vt.T), ("random", Q)):
        for Lw in WIDTHS:
            P = B[:, :Lw]; Ptr, Pdv = Ztr @ P, Zdv @ P; best = None
            for a in (1e1, 1e2, 1e3, 1e4, 1e5):
                m = Ridge(alpha=a).fit(Ptr, Ytr); t1, _ = ident(m.predict(Pdv), Edv, ydv, [list(range(len(dv)))] * len(ydv))
                if best is None or t1 > best[0]: best = (t1, a, m)
            alpha, m = best[1], best[2]; mn = Ridge(alpha=alpha).fit(rng.standard_normal(Ptr.shape), Ytr)
            pdv = m.predict(Pdv); tau = max(TAUS, key=lambda t: nce(pdv, Edv, ydv, t))   # temperature chosen on dev, never on test
            per = {}
            for sj, (q, Zq, tp, pools) in queries.items():
                pe = m.predict(Zq @ P); pn = mn.predict(rng.standard_normal((len(q), Lw))); allp = [list(range(N))] * len(q)
                r = {}
                for cond, pred, tt in (("eeg", pe, tp), ("noise", pn, tp), ("shuffled_pair", pe, list(np.array(tp)[rng.permutation(len(tp))]))):
                    t1, tw = ident(pred, Ete, tt, allp); mt1, mtw = ident(pred, Ete, tt, pools if cond != "shuffled_pair" else [[c for c in range(N)]] * len(q))
                    r[cond] = dict(top1=t1, two_way=tw, wolpaw=float(wolpaw_bits(t1, N)), nce=nce(pred, Ete, np.array(tt), tau), matched_top1=mt1, matched_two_way=mtw)
                per[sj] = r
            d = [per[s]["eeg"]["two_way"] - per[s]["noise"]["two_way"] for s in per]; dm = [per[s]["eeg"]["matched_two_way"] - per[s]["noise"]["matched_two_way"] for s in per]
            row = dict(basis=basis_name, L=Lw, alpha=alpha, tau=tau, dev_top1=best[0], per_participant=per,
                       mean={c: {k: float(np.mean([per[s][c][k] for s in per])) for k in ("top1", "two_way", "wolpaw", "nce", "matched_two_way")} for c in ("eeg", "noise", "shuffled_pair")},
                       eeg_minus_noise_two_way=dict(mean=float(np.mean(d)), ci95=boot(d), n_pos=int(np.sum(np.array(d) > 0))),
                       eeg_minus_noise_matched_two_way=dict(mean=float(np.mean(dm)), ci95=boot(dm)))
            curve.append(row)
            print(f"{task} {basis_name} L={Lw} alpha={alpha:g} eeg 2way {row['mean']['eeg']['two_way']:.3f} noise {row['mean']['noise']['two_way']:.3f} pair {row['mean']['shuffled_pair']['two_way']:.3f} diff {row['eeg_minus_noise_two_way']['mean']:+.3f} {row['eeg_minus_noise_two_way']['ci95']} wolpaw {row['mean']['eeg']['wolpaw']:.4f} nce {row['mean']['eeg']['nce']:.4f}/{row['mean']['noise']['nce']:.4f}", flush=True)
    pca = [r for r in curve if r["basis"] == "pca"]; bonf = 1 - 0.05 / len(WIDTHS)
    sig = []
    for r in pca:
        d = [r["per_participant"][s]["eeg"]["two_way"] - r["per_participant"][s]["noise"]["two_way"] for s in r["per_participant"]]
        b = np.random.default_rng(1); bs = [b.choice(d, len(d)).mean() for _ in range(20000)]
        if np.percentile(bs, 100 * (1 - bonf) / 2) > 0: sig.append(r["L"])
    res = dict(task=task, n_participants=len(queries), N=N, widths=WIDTHS, sentence_encoder=enc, note=__doc__, curve=curve, widths_with_bonferroni_signal=sig,
               verdict="text yield above the twin at widths " + str(sig) if sig else "curve at its floor at every width", runtime_s=round(time.time() - t0, 1))
    YD = Path("/path/to/eegaudit/results/yield/text_bottleneck_zuco"); YD.mkdir(parents=True, exist_ok=True)
    for sj in queries:   # protocol-schema files (one per participant) for experiments/stats/yield_stats.py
        rows = []
        for r in curve:
            for cond in ("eeg", "noise", "shuffled_pair"):
                c = r["per_participant"][sj][cond]; nm = cond if r["basis"] == "pca" else ("random" if cond == "eeg" else None)
                if nm: rows.append(dict(L="full" if r["L"] == 840 else r["L"], condition=nm, bits_F=c["wolpaw"], bits_NCE=c["nce"], two_way=c["two_way"], top1=c["top1"]))
        json.dump(dict(dataset=f"zuco_{task}", decoder="ridge_mean_wordsall", participant=sj, bottleneck="input_pca_train_fit (random = random orthonormal projection)",
                       target="MiniLM sentence embedding, N = %d test sentences" % N, full_dim=840, rows=rows), open(YD / f"{task}_{sj}.json", "w"), indent=1)
    (OUT / f"x6_{task}.json").write_text(json.dumps(res, indent=1)); print(task, res["verdict"], f"({res['runtime_s']}s)")
