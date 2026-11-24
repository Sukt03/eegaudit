"""Data-side floor for EEG-to-text (ideation card B3): can ANOTHER participant's EEG for the same sentence identify it?
Leave-one-subject-out on ZuCo word-level EEG features (Wang & Ji pickles): for each task, every sentence read by every subject.
Two read-outs, both trained on the other subjects only, tested on the held-out subject's sentences among ALL the task's sentences:
  ridge     : EEG feature (concat of first 8 word vectors, 6720-d) -> MiniLM sentence embedding, cosine identification
  template  : no language model: candidate = mean EEG feature of the other subjects for that sentence, correlation identification
Controls: noise input of the same shape through the same read-out; shuffled pairs. Bits = Wolpaw at N = number of sentences.
Output: results/track_text/cross_subject/<task>.json   Usage: python cross_subject_identification.py SR [NR NR2 ...]"""
import json, os, sys, time
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).parent))
from pilot_retrieval import TASKS, load_task, feat, embed_sentences, retrieval_scores, wolpaw_bits
DATA = "/path/to/eegaudit/data_raw/zuco/eeg2text_pickles"; MINILM = "/path/to/eegaudit/data_raw/zuco/minilm"
OUT = Path("/path/to/eegaudit/results/track_text/cross_subject"); OUT.mkdir(parents=True, exist_ok=True)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ridge_fit(X, Y, alpha):
    Xt = torch.from_numpy(X).float().to(dev); Yt = torch.from_numpy(Y).float().to(dev)
    mu, sd = Xt.mean(0, keepdim=True), Xt.std(0, keepdim=True) + 1e-6; Z = (Xt - mu) / sd; ym = Yt.mean(0, keepdim=True)
    W = torch.linalg.solve(Z.T @ Z + alpha * Z.shape[1] * torch.eye(Z.shape[1], device=dev), Z.T @ (Yt - ym))
    return lambda Xn: (((torch.from_numpy(Xn).float().to(dev) - mu) / sd) @ W + ym).cpu().numpy()


def ident(pred, cand, idx):
    top1, two = retrieval_scores(pred, cand, idx); N = cand.shape[0]
    return dict(top1=top1, two_way=two, bits=max(0.0, wolpaw_bits(top1, N)))


for task in sys.argv[1:] or ["SR"]:
    t0 = time.time(); rng = np.random.default_rng(0)
    texts, split, recs, subjects = load_task(os.path.join(DATA, TASKS[task]), task)
    sents = sorted({i for (_, i) in recs}); pos = {i: j for j, i in enumerate(sents)}
    E, enc = embed_sentences([texts[i] for i in sents], MINILM); E = E / np.linalg.norm(E, axis=1, keepdims=True)
    F = {(sj, i): feat(wv, sv, 8, "concat") for (sj, i), (wv, sv) in recs.items()}
    rows = []
    for held in subjects:
        tr = [(sj, i) for (sj, i) in F if sj != held]; te = [(sj, i) for (sj, i) in F if sj == held]
        if len(te) < 10: continue
        Xtr = np.stack([F[k] for k in tr]); Ytr = np.stack([E[pos[i]] for (_, i) in tr]); Xte = np.stack([F[k] for k in te]); idx = np.array([pos[i] for (_, i) in te])
        # ridge EEG -> sentence embedding, alpha picked on a held-out subject among the training subjects (never on the test subject)
        val = tr[0][0]; vtr = [k for k in tr if k[0] != val]; vte = [k for k in tr if k[0] == val]
        best, ba = -1, None
        for a in (1e-2, 1e-1, 1e0, 1e1, 1e2):
            f = ridge_fit(np.stack([F[k] for k in vtr]), np.stack([E[pos[i]] for (_, i) in vtr]), a)
            s = retrieval_scores(f(np.stack([F[k] for k in vte])), E, np.array([pos[i] for (_, i) in vte]))[1]
            if s > best: best, ba = s, a
        f = ridge_fit(Xtr, Ytr, ba); P = f(Xte)
        rows.append(dict(held_out=held, readout="ridge", condition="eeg", alpha=ba, n_test=len(te), **ident(P, E, idx)))
        rows.append(dict(held_out=held, readout="ridge", condition="shuffled_pair", n_test=len(te), **ident(P, E, rng.permutation(len(E))[idx] if False else idx[rng.permutation(len(idx))])))
        fn = ridge_fit(rng.standard_normal(Xtr.shape).astype(np.float32) * Xtr.std() + Xtr.mean(), Ytr, ba)
        rows.append(dict(held_out=held, readout="ridge", condition="noise", n_test=len(te), **ident(fn(rng.standard_normal(Xte.shape).astype(np.float32) * Xtr.std() + Xtr.mean()), E, idx)))
        # template: mean EEG of the other subjects per sentence
        T = np.zeros((len(sents), Xtr.shape[1])); cnt = np.zeros(len(sents))
        for k, x in zip(tr, Xtr): T[pos[k[1]]] += x; cnt[pos[k[1]]] += 1
        ok = cnt > 0; T[ok] /= cnt[ok, None]; T = (T - T.mean(0)) / (T.std(0) + 1e-6); Xz = (Xte - Xte.mean(0)) / (Xte.std(0) + 1e-6)
        rows.append(dict(held_out=held, readout="template", condition="eeg", n_test=len(te), **ident(Xz, T, idx)))
        rows.append(dict(held_out=held, readout="template", condition="shuffled_pair", n_test=len(te), **ident(Xz, T, idx[rng.permutation(len(idx))])))
        rows.append(dict(held_out=held, readout="template", condition="noise", n_test=len(te), **ident(rng.standard_normal(Xz.shape), T, idx)))
    out = dict(task=task, n_sentences=len(sents), n_subjects=len(subjects), sentence_encoder=enc, rows=rows, runtime_s=round(time.time() - t0, 1))
    (OUT / f"{task}.json").write_text(json.dumps(out, indent=1))
    s = lambda r_, c: np.mean([r["top1"] for r in rows if r["readout"] == r_ and r["condition"] == c])
    print(f"{task}: N={len(sents)} sentences, {len(subjects)} subjects | ridge top-1 eeg {100*s('ridge','eeg'):.1f}% noise {100*s('ridge','noise'):.1f}% pair {100*s('ridge','shuffled_pair'):.1f}% | template top-1 eeg {100*s('template','eeg'):.1f}% noise {100*s('template','noise'):.1f}% pair {100*s('template','shuffled_pair'):.1f}% | chance {100/len(sents):.2f}%  ({out['runtime_s']}s)", flush=True)
