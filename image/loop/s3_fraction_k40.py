"""S3 (property 7): image yield as a fraction of the recording's own ceiling at MATCHED k. NICE-GA seed 141, 10 participants: yield at k = 40
(mean of the two 40-repetition halves, each scored against the CLIP class centres) under I_F and I_NCE; ceiling = split-half identification of
the two halves in the trained space (same k = 40 per half) under I_F and I_NCE; floor = shuffled pairs; fraction = (yield - floor)/(ceiling - floor)
per participant, participant bootstrap CI. Output results/loop/s3_fraction/s3_fraction_k40.json."""
import sys, json, numpy as np, torch
sys.path.insert(0, '/path/to/eegaudit/experiments/eegbits'); from run import soft_bits, wolpaw_bits; from run_acq_controls import load_encoder
dev = torch.device('cuda'); rng = np.random.default_rng(0); norm = lambda A: A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)
def sc(E, C):
    S = norm(E) @ norm(C).T; n = len(S); d = S[np.arange(n), np.arange(n)]; t1 = float(((S > d[:, None]).sum(1) == 0).mean()); return dict(I_F=wolpaw_bits(t1, n), I_NCE=soft_bits(S), top1=t1)
rows = []
for i in range(1, 11):
    s = f'sub-{i:02d}'; embed, Y, tag, X = load_encoder('nice_ga', s, 141, dev); X = np.asarray(X); Y = np.asarray(Y)
    E1, E2 = embed(X[:, :40].mean(1)), embed(X[:, 40:].mean(1)); y1, y2 = sc(E1, Y), sc(E2, Y); ceil = sc(E1, E2); perm = rng.permutation(200); fl = sc(E1, Y[perm]); flc = sc(E1, E2[perm])
    r = dict(subject=s, yield_k40={k: (y1[k] + y2[k]) / 2 for k in ('I_F', 'I_NCE')}, ceiling_k40=ceil, floor=fl, floor_ceiling=flc)
    r['fraction'] = {k: (r['yield_k40'][k] - fl[k]) / (ceil[k] - flc[k]) for k in ('I_F', 'I_NCE')}; rows.append(r)
    print(s, {k: round(v, 3) for k, v in r['yield_k40'].items()}, 'ceil', {k: round(ceil[k], 3) for k in ('I_F', 'I_NCE')}, 'frac', {k: round(v, 4) for k, v in r['fraction'].items()}, flush=True)
out = dict(rows=rows)
for k in ('I_F', 'I_NCE'):
    v = np.array([r['fraction'][k] for r in rows]); bs = [v[rng.integers(0, 10, 10)].mean() for _ in range(4000)]
    out[f'fraction_{k}'] = dict(mean=float(v.mean()), median=float(np.median(v)), ci95=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], yield_mean=float(np.mean([r['yield_k40'][k] for r in rows])), ceiling_mean=float(np.mean([r['ceiling_k40'][k] for r in rows])))
json.dump(out, open('/path/to/eegaudit/results/loop/s3_fraction/s3_fraction_k40.json', 'w'), indent=1); print({k: out[k] for k in ('fraction_I_F', 'fraction_I_NCE')})
