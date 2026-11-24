"""P2 (ideator, approved 13:25): lexical word-level targets with a gaze-only control, ZuCo (EXPERIMENTS agent, 2026-11-20).
(Solver: closed-form ridge via eigendecomposition of the training Gram matrix; alpha chosen on an inner sentence-group split of the training fold;
binary target = ridge on 0/1 labels with class-balanced sample weights, threshold = midpoint of the two class means of the training predictions.)
Recording fixed: Wang & Ji (AAAI'22) 840-d GD word features (8 bands x 105 ch) per fixated word, per participant. Decoder fixed: ridge (alpha by
RidgeCV inside the training fold only). Targets varied:
  length  : word length in characters (punctuation stripped), normal-scores transformed; bound = Gaussianised channel -1/2 log2(1 - rho_g^2),
            rho_g = Pearson(held-out prediction, normal-scored target) over all held-out words of the participant
  function: closed-class (function) vs content word, balanced ridge classifier; bound = Fano at N = 2 from held-out BALANCED accuracy, 1 - H2(bacc)
Within-participant 5-fold CV over sentences (GroupKFold: words of a sentence stay together). Conditions: eeg; noise (N(0,1) features of the same
shape, same CV); shuffled (training targets permuted within the training fold); gaze (only gaze/position scalars available in the pickles:
nFixations, log(1+nFix), word position, relative position, sentence length in words; fixation durations are NOT stored in the pickles);
eeg+gaze (both feature sets) for the increment over gaze.
Pass rule (pre-registered by the ideator): a target is carried by EEG if bits_eeg - max(bits_noise, bits_gaze) > 0 with the 95 % participant
bootstrap CI excluding 0. Output: results/loop/p2_lexical/<task>.json   Usage: python p2_lexical_targets.py SR [NR2]"""
import json, os, pickle, re, sys, time
from pathlib import Path
import numpy as np
from scipy import stats
from sklearn.model_selection import GroupKFold
sys.path.insert(0, str(Path(__file__).parent))
from pilot_retrieval import TASKS, word_vec
DATA = "/path/to/eegaudit/data_raw/zuco/eeg2text_pickles"; OUT = Path("/path/to/eegaudit/results/loop/p2_lexical"); OUT.mkdir(parents=True, exist_ok=True)
FUNCTION = set("""a an the this that these those my your his her its our their whose which who whom what i me you he him she it we us they them
myself yourself himself herself itself ourselves themselves and or but nor so yet if then than because although though while whereas unless until
since as of in on at by for with about against between into through during before after above below to from up down out off over under again
further once here there when where why how all any both each few more most other some such no not only own same too very can will just should
would could may might must shall do does did doing done be is am are was were been being have has had having one 's s t don isn wasn""".split())
ALPHAS = np.logspace(-1, 5, 13)


def h2(p): p = min(max(p, 1e-12), 1 - 1e-12); return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))


def fano2(bacc): return float(1 - h2(bacc)) if bacc > 0.5 else 0.0


def gbits(r): return float(-0.5 * np.log2(1 - r ** 2)) if r > 0 else 0.0


def nscore(x): return stats.norm.ppf((stats.rankdata(x) - 0.5) / len(x))


def zfit(Xtr, Xte): mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8; return (Xtr - mu) / sd, (Xte - mu) / sd


def ridge_path(X, y, w, alphas):
    """weighted ridge with intercept, all alphas at once; returns list of (coef, intercept)."""
    sw = w / w.sum(); mx = sw @ X; my = sw @ y; Xc = (X - mx) * np.sqrt(w)[:, None]; yc = (y - my) * np.sqrt(w)
    ev, V = np.linalg.eigh(Xc.T @ Xc); b = V.T @ (Xc.T @ yc)
    return [((V * (1.0 / (ev + a * len(y)))) @ b, my - mx @ ((V * (1.0 / (ev + a * len(y)))) @ b)) for a in alphas]


def fit_predict(Xtr, ytr, gtr, Xte, kind):
    w = np.ones(len(ytr)) if kind == "length" else np.where(ytr == 1, 0.5 / max(ytr.mean(), 1e-6), 0.5 / max(1 - ytr.mean(), 1e-6))
    inner = np.isin(gtr, np.unique(gtr)[::4]); A = [1e-4, 1e-3, 1e-2, 1e-1, 1, 10]
    path = ridge_path(Xtr[~inner], ytr[~inner], w[~inner], A); best, ba = -np.inf, A[0]
    for a, (c, b0) in zip(A, path):
        p = Xtr[inner] @ c + b0
        sc = np.corrcoef(p, ytr[inner])[0, 1] if kind == "length" else np.mean([np.mean(((p > 0.5) == (ytr[inner] == 1))[ytr[inner] == k]) for k in (0, 1)])
        if np.isfinite(sc) and sc > best: best, ba = sc, a
    c, b0 = ridge_path(Xtr, ytr, w, [ba])[0]; ptr = Xtr @ c + b0; pte = Xte @ c + b0
    if kind == "length": return pte
    thr = 0.5 * (ptr[ytr == 1].mean() + ptr[ytr == 0].mean()); return (pte > thr).astype(int)


def run_cv(X, y, groups, kind, rng, shuffle=False):
    pred = np.zeros(len(y))
    for tr, te in GroupKFold(5).split(X, y, groups):
        Xtr, Xte = zfit(X[tr], X[te]); ytr = y[tr].copy()
        if shuffle: ytr = ytr[rng.permutation(len(ytr))]
        pred[te] = fit_predict(Xtr, ytr, groups[tr], Xte, kind)
    if kind == "length":
        r = float(np.corrcoef(pred, y)[0, 1]); return dict(rho_g=r, bits=gbits(r))
    bacc = float(np.mean([np.mean(pred[y == c] == c) for c in (0, 1)])); return dict(balanced_acc=bacc, bits=fano2(bacc))


def boot(v, seed=0):
    rng = np.random.default_rng(seed); v = np.asarray(v, float); b = [rng.choice(v, len(v)).mean() for _ in range(5000)]; return [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]


for task in sys.argv[1:] or ["SR"]:
    t0 = time.time(); d = pickle.load(open(os.path.join(DATA, TASKS[task]), "rb")); per = {}
    for sj in sorted(d):
        X, G, L, F, Z = [], [], [], [], []
        for si, s in enumerate(d[sj]):
            if s is None: continue
            ws = s["word"]; n = len(ws)
            for wi, w in enumerate(ws):
                v = word_vec(w)
                tok = re.sub(r"[^\w']", "", w["content"]).lower()
                if v is None or not tok: continue
                nf = float(w.get("nFixations") or 0)
                X.append(v); G.append(si); L.append(len(re.sub(r"[^\w]", "", tok))); F.append(int(tok in FUNCTION)); Z.append([nf, np.log1p(nf), wi, wi / max(n - 1, 1), n])
        X, G, L, F, Z = np.array(X), np.array(G), np.array(L, float), np.array(F), np.array(Z, float)
        rng = np.random.default_rng(0); Lg = nscore(L); noise = rng.standard_normal(X.shape)
        res = dict(n_words=len(L), n_sentences=int(len(set(G))), function_rate=float(F.mean()))
        for kind, y in (("length", Lg), ("function", F)):
            res[kind] = dict(eeg=run_cv(X, y, G, kind, rng), noise=run_cv(noise, y, G, kind, rng), shuffled=run_cv(X, y, G, kind, rng, shuffle=True),
                             gaze=run_cv(Z, y, G, kind, rng), eeg_gaze=run_cv(np.c_[X, Z], y, G, kind, rng))
        per[sj] = res
        print(sj, res["n_words"], {k: {c: round(res[k][c]["bits"], 4) for c in res[k]} for k in ("length", "function")}, flush=True)
    summary = {}
    for kind in ("length", "function"):
        s = {c: dict(mean=float(np.mean([per[p][kind][c]["bits"] for p in per])), ci95=boot([per[p][kind][c]["bits"] for p in per])) for c in ("eeg", "noise", "shuffled", "gaze", "eeg_gaze")}
        margin = [per[p][kind]["eeg"]["bits"] - max(per[p][kind]["noise"]["bits"], per[p][kind]["gaze"]["bits"]) for p in per]
        incr = [per[p][kind]["eeg_gaze"]["bits"] - per[p][kind]["gaze"]["bits"] for p in per]
        s["eeg_minus_max_noise_gaze"] = dict(mean=float(np.mean(margin)), ci95=boot(margin), n_pos=int(np.sum(np.array(margin) > 0)))
        s["eeg_gaze_minus_gaze"] = dict(mean=float(np.mean(incr)), ci95=boot(incr), n_pos=int(np.sum(np.array(incr) > 0)))
        s["passes"] = bool(s["eeg_minus_max_noise_gaze"]["ci95"][0] > 0)
        key = "rho_g" if kind == "length" else "balanced_acc"
        s[f"eeg_{key}_mean"] = float(np.mean([per[p][kind]["eeg"][key] for p in per])); s[f"gaze_{key}_mean"] = float(np.mean([per[p][kind]["gaze"][key] for p in per]))
        summary[kind] = s
    out = dict(task=task, n_participants=len(per), note=__doc__, per_participant=per, summary=summary, runtime_s=round(time.time() - t0, 1))
    (OUT / f"{task}.json").write_text(json.dumps(out, indent=1)); print(json.dumps(summary, indent=1)); print(f"({out['runtime_s']}s)")
