"""Cross-participant split-half ceiling for ZuCo (EXPERIMENTS agent, 2026-11-20). ZuCo has no within-participant repeats, but each sentence is
read by many participants. Per random split of the participants into two halves (20 splits, seeds 0..19): for every TEST sentence of the task
(Wang & Ji split as in pilot_retrieval.load_task), average the sentence feature over the participants of half A and of half B who read it;
z-score each half's (sentence x feature) matrix over sentences; identify each A-row among the B-rows (and B among A) by Pearson correlation.
Feature (primary) = mean over fixated words of the Wang & Ji GD word vectors (840-d, 8 bands x 105 ch); secondary = the pickle's sentence-level
vector. No training, no language model, no labels beyond sentence identity. Scores: top-1, 2-way (ties count 1/2), Wolpaw bits at N = #test sentences.
Controls: (1) shuffled pairs (B rows permuted); (2) length-only: each sentence represented by its word count (and fixated-word count), identification
by |difference|, exact tie handling = the yield of a decoder that knows nothing but length; (3) EEG with sentence length regressed out of every
feature column (log word count and log fixated-word count, within each half, over sentences). Participant curve: m in {1,2,4,6} participants per half.
Usage: python cross_participant_ceiling.py SR [NR ...]   Output: results/track_text/cross_participant_ceiling/<task>.json"""
import json, os, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from pilot_retrieval import TASKS, load_task, wolpaw_bits
DATA = "/path/to/eegaudit/data_raw/zuco/eeg2text_pickles"
OUT = Path("/path/to/eegaudit/results/track_text/cross_participant_ceiling"); OUT.mkdir(parents=True, exist_ok=True)
NSPLIT = 20


def zcols(M): return (M - M.mean(0)) / (M.std(0) + 1e-8)


def ident_from_scores(S):
    """S[q, c] similarity, true candidate of query q is c = q. Ties count 1/2 in 2-way; top-1 = 1/(1+#ties) if nothing beats the truth."""
    t = np.diag(S)[:, None]; N = S.shape[0]; off = ~np.eye(N, dtype=bool)
    beat = ((S > t) & off).sum(1); tie = ((S == t) & off).sum(1)
    top1 = np.where(beat == 0, 1.0 / (1 + tie), 0.0).mean(); two = (1 - (beat + 0.5 * tie) / (N - 1)).mean()
    return float(top1), float(two)


def corr_scores(A, B):
    A = A - A.mean(1, keepdims=True); B = B - B.mean(1, keepdims=True)
    A /= np.linalg.norm(A, axis=1, keepdims=True) + 1e-12; B /= np.linalg.norm(B, axis=1, keepdims=True) + 1e-12
    return A @ B.T


def score(A, B, N):
    t1a, twa = ident_from_scores(corr_scores(A, B)); t1b, twb = ident_from_scores(corr_scores(B, A)); t1, tw = (t1a + t1b) / 2, (twa + twb) / 2
    return dict(top1=t1, two_way=tw, bits=max(0.0, wolpaw_bits(t1, N)))


def resid(M, L):
    X = np.c_[np.ones(len(L)), L]; beta, *_ = np.linalg.lstsq(X, M, rcond=None); return M - X @ beta


def summ(rows, key):
    v = np.array([r[key] for r in rows]); return dict(mean=float(v.mean()), sd=float(v.std(ddof=1)) if len(v) > 1 else 0.0, p2_5=float(np.percentile(v, 2.5)), p97_5=float(np.percentile(v, 97.5)), n=len(v))


for task in sys.argv[1:] or ["SR"]:
    t0 = time.time()
    texts, split, recs, subjects = load_task(os.path.join(DATA, TASKS[task]), task)
    test = sorted(i for i in split if split[i] == "test")
    readers = {i: [sj for sj in subjects if (sj, i) in recs] for i in test}
    feats = {"wordmean": {k: v[0].mean(0) for k, v in recs.items() if k[1] in readers}, "sentlevel": {k: np.asarray(v[1], float).ravel() for k, v in recs.items() if k[1] in readers}}
    nfix = {k: v[0].shape[0] for k, v in recs.items() if k[1] in readers}
    nwords = {i: len(texts[i].split()) for i in test}
    res = dict(task=task, n_test_sentences=len(test), n_subjects=len(subjects), readers_per_sentence=dict(min=int(min(map(len, readers.values()))), max=int(max(map(len, readers.values()))), mean=float(np.mean(list(map(len, readers.values()))))),
               n_splits=NSPLIT, note=__doc__, curves={}, controls={})
    # length-only controls (no EEG): word count of the text; fixated-word count averaged over all readers
    Lw = np.array([nwords[i] for i in test], float); Lf = np.array([np.mean([nfix[(sj, i)] for sj in readers[i]]) for i in test])
    N = len(test)
    for name, L in (("length_words", Lw), ("length_fixated_words_allreaders", Lf)):
        t1, tw = ident_from_scores(-np.abs(L[:, None] - L[None, :])); res["controls"][name] = dict(top1=t1, two_way=tw, bits=max(0.0, wolpaw_bits(t1, N)))
    for m in (1, 2, 4, 6, "half"):
        for fname, F in feats.items():
            for cond in ("eeg", "shuffled_pair", "eeg_length_residualized", "length_fixated_split"):
                rows = []
                for sp in range(NSPLIT):
                    rng = np.random.default_rng(sp); perm = list(rng.permutation(subjects)); h = len(perm) // 2; HA, HB = perm[:h], perm[h:]
                    if m != "half": HA, HB = HA[:m], HB[:m]
                    ok = [i for i in test if any((sj, i) in recs for sj in HA) and any((sj, i) in recs for sj in HB)]
                    if len(ok) < 10: continue
                    A = np.stack([np.mean([F[(sj, i)] for sj in HA if (sj, i) in recs], 0) for i in ok]); B = np.stack([np.mean([F[(sj, i)] for sj in HB if (sj, i) in recs], 0) for i in ok])
                    LA = np.array([np.mean([nfix[(sj, i)] for sj in HA if (sj, i) in recs]) for i in ok]); LB = np.array([np.mean([nfix[(sj, i)] for sj in HB if (sj, i) in recs]) for i in ok])
                    Lwo = np.array([nwords[i] for i in ok], float)
                    if cond == "length_fixated_split":   # length known only from each half's own fixation counts (the cue an EEG average could carry)
                        t1, tw = ident_from_scores(-np.abs(LA[:, None] - LB[None, :])); r = dict(top1=t1, two_way=tw, bits=max(0.0, wolpaw_bits(t1, len(ok))))
                    else:
                        Az, Bz = zcols(A), zcols(B)
                        if cond == "eeg_length_residualized":
                            Az = zcols(resid(Az, np.c_[np.log(Lwo), np.log(LA)])); Bz = zcols(resid(Bz, np.c_[np.log(Lwo), np.log(LB)]))
                        if cond == "shuffled_pair":
                            Bz = Bz[np.random.default_rng(1000 + sp).permutation(len(ok))]
                        r = score(Az, Bz, len(ok))
                    r.update(split=sp, n_sentences=len(ok), n_a=len(HA), n_b=len(HB)); rows.append(r)
                if not rows: continue
                if cond == "length_fixated_split" and fname != "wordmean": continue
                res["curves"][f"m{m}_{fname}_{cond}"] = dict(per_split=rows, top1=summ(rows, "top1"), two_way=summ(rows, "two_way"), bits=summ(rows, "bits"), n_sentences=int(np.median([r["n_sentences"] for r in rows])))
    res["runtime_s"] = round(time.time() - t0, 1)
    (OUT / f"{task}.json").write_text(json.dumps(res, indent=1))
    print(f"{task}: N={N} test sentences, {len(subjects)} subjects, readers/sentence {res['readers_per_sentence']}")
    for k, v in res["controls"].items(): print(f"  control {k}: top1 {v['top1']:.3f} 2way {v['two_way']:.3f} bits {v['bits']:.3f}")
    for k, v in res["curves"].items(): print(f"  {k}: N={v['n_sentences']} top1 {v['top1']['mean']:.3f} 2way {v['two_way']['mean']:.3f} [{v['two_way']['p2_5']:.3f},{v['two_way']['p97_5']:.3f}] bits {v['bits']['mean']:.3f}")
    print(f"  ({res['runtime_s']}s)", flush=True)
