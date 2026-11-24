"""Standard test, sentence cross-reader ceiling under the common log-score (ZuCo SR 40 / NR2 35 test candidates, 2026-11-20).
Same half-splits as experiments/track_text/cross_participant_ceiling.py (20 random reader splits, seeds 0..19; feature = mean over fixated
words of the Wang & Ji 840-d GD vectors; z-scored per half; Pearson similarity). Score = InfoNCE soft_bits (experiments/eegbits/run.py:
log2 N minus leave-one-query-out softmax cross-entropy, 12-temperature grid), averaged over both directions (A->B, B->A).
m = 1 reader per half (single-reading ceiling) and m = half (all readers split in two). Controls: shuffled pairs; length-only
(each half's mean fixated-word count; and text word count). Yield = per-participant X6 ridge InfoNCE at full width 840
(outputs/AGENT_LOOP_20261120/results/common_log_yield.json), floor = noise twin and shuffled pairs. Fraction = mean yield / mean ceiling;
CI by independent bootstrap of participants (yield) and splits (ceiling), 10000 resamples, seed 20261120.
Output: results/standard/text_ceiling/<task>.json"""
import json, os, sys, time
from pathlib import Path
import numpy as np
ROOT = Path('/path/to/eegaudit')
sys.path.insert(0, str(ROOT / 'experiments/track_text')); sys.path.insert(0, str(ROOT / 'experiments/eegbits'))
from pilot_retrieval import TASKS, load_task
from run import soft_bits
DATA = '/path/to/eegaudit/data_raw/zuco/eeg2text_pickles'
OUT = ROOT / 'results/standard/text_ceiling'; OUT.mkdir(parents=True, exist_ok=True); NSPLIT = 20


def zcols(M): return (M - M.mean(0)) / (M.std(0) + 1e-8)


def corr_scores(A, B):
    A = A - A.mean(1, keepdims=True); B = B - B.mean(1, keepdims=True)
    A = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12); B = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-12); return A @ B.T


def ident_top1(S):   # copied from cross_participant_ceiling.ident_from_scores (top-1 part), used only for the reproduction check
    t = np.diag(S)[:, None]; off = ~np.eye(S.shape[0], dtype=bool); beat = ((S > t) & off).sum(1); tie = ((S == t) & off).sum(1)
    return float(np.where(beat == 0, 1.0 / (1 + tie), 0.0).mean())


def nce2(S): return 0.5 * (soft_bits(S) + soft_bits(S.T))


def main():
    common = json.loads((ROOT / 'outputs/AGENT_LOOP_20261120/results/common_log_yield.json').read_text())
    rng = np.random.default_rng(20261120)
    for task in sys.argv[1:]:
        t0 = time.time(); texts, split, recs, subjects = load_task(os.path.join(DATA, TASKS[task]), task)
        test = sorted(i for i in split if split[i] == 'test'); N = len(test)
        F = {k: v[0].mean(0) for k, v in recs.items() if k[1] in test}; nfix = {k: v[0].shape[0] for k, v in recs.items() if k[1] in test}
        Lw = np.array([len(texts[i].split()) for i in test], float)
        res = dict(task=task, N=N, n_subjects=len(subjects), note=__doc__, ceiling={}, controls=dict(length_words=float(nce2(-np.abs(Lw[:, None] - Lw[None, :])))))
        for m in (1, 'half'):
            rows = []
            for sp in range(NSPLIT):
                perm = list(np.random.default_rng(sp).permutation(subjects)); h = len(perm) // 2; HA, HB = perm[:h], perm[h:]
                if m != 'half': HA, HB = HA[:m], HB[:m]
                ok = [i for i in test if any((sj, i) in recs for sj in HA) and any((sj, i) in recs for sj in HB)]
                if len(ok) < 10: continue
                A = np.stack([np.mean([F[(sj, i)] for sj in HA if (sj, i) in recs], 0) for i in ok]); B = np.stack([np.mean([F[(sj, i)] for sj in HB if (sj, i) in recs], 0) for i in ok])
                LA = np.array([np.mean([nfix[(sj, i)] for sj in HA if (sj, i) in recs]) for i in ok]); LB = np.array([np.mean([nfix[(sj, i)] for sj in HB if (sj, i) in recs]) for i in ok])
                S = corr_scores(zcols(A), zcols(B)); Bs = zcols(B)[np.random.default_rng(1000 + sp).permutation(len(ok))]
                rows.append(dict(split=sp, n_sentences=len(ok), eeg=float(nce2(S)), shuffled_pair=float(nce2(corr_scores(zcols(A), Bs))),
                                 length_fixated_split=float(nce2(-np.abs(LA[:, None] - LB[None, :]))),
                                 top1=0.5 * (ident_top1(S) + ident_top1(S.T)), log2N_used=float(np.log2(len(ok)))))
            res['ceiling'][f'm{m}'] = dict(per_split=rows, **{k: float(np.mean([r[k] for r in rows])) for k in ('eeg', 'shuffled_pair', 'length_fixated_split')},
                                         eeg_ci_splits=np.quantile([r['eeg'] for r in rows], [.025, .975]).tolist(), n_splits=len(rows), n_sentences_median=int(np.median([r['n_sentences'] for r in rows])))
        old = json.loads((ROOT / f'results/track_text/cross_participant_ceiling/{task}.json').read_text())['curves']
        res['checks'] = {m: dict(top1_here=float(np.mean([r['top1'] for r in res['ceiling'][m]['per_split']])), top1_existing=old[f'{m}_wordmean_eeg']['top1']['mean']) for m in ('m1', 'mhalf')}
        for v in res['checks'].values(): v['abs_diff'] = abs(v['top1_here'] - v['top1_existing'])
        yrows = common['sentence']['rows'][task]; y = np.array([r['eeg'] for r in yrows.values()]); nz = np.array([r['noise'] for r in yrows.values()]); sh = np.array([r['shuffled_pair'] for r in yrows.values()])
        ix = rng.integers(0, len(y), size=(10000, len(y))); d = y - nz
        res['yield'] = dict(n=len(y), mean=float(y.mean()), ci=np.quantile(y[ix].mean(1), [.025, .975]).tolist(), n_positive=int((y > 0).sum()),
                            noise_floor=float(nz.mean()), shuffled_floor=float(sh.mean()), minus_noise=float(d.mean()), minus_noise_ci=np.quantile(d[ix].mean(1), [.025, .975]).tolist(),
                            n_above_noise=int((d > 0).sum()), per_participant={k: v for k, v in yrows.items()})
        for m in ('m1', 'mhalf'):
            c = np.array([r['eeg'] for r in res['ceiling'][m]['per_split']]); jc = rng.integers(0, len(c), size=(10000, len(c)))
            fr = y[ix].mean(1) / c[jc].mean(1); res[f'fraction_{m}'] = dict(value=float(y.mean() / c.mean()), ci=np.quantile(fr, [.025, .975]).tolist(),
                                                                          ceiling_minus_length=float(c.mean() - res['ceiling'][m]['length_fixated_split']))
        res['runtime_s'] = round(time.time() - t0, 1); (OUT / f'{task}.json').write_text(json.dumps(res, indent=1))
        print(task, N, {m: {k: round(v, 4) for k, v in res['ceiling'][m].items() if isinstance(v, float)} for m in res['ceiling']}, 'len_words', round(res['controls']['length_words'], 4),
              'yield', round(y.mean(), 4), 'frac', {m: round(res[f'fraction_{m}']['value'], 5) for m in ('m1', 'mhalf')}, res['runtime_s'], res['checks'], flush=True)


if __name__ == '__main__':
    main()
