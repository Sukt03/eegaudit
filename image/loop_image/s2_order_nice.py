"""S2: average-then-encode vs encode-then-average for the standard NICE encoder (seed 141) at k = 8 and k = 80 (10 draws at k=8),
plus k = 1 for reference. Logic copied from experiments/method_pilots/r2_estimator_axes.py (which ran NICE-GA only); that file is not
modified. Scores: run.metrics (Wolpaw bits, soft bits, top-1, 2-way) and the soft bound of shuffled pairs as the floor.
Output: results/loop/image/s2_order_nice_<subject>_seed141.json.  Usage: python s2_order_nice.py sub-04"""
import sys, time
import numpy as np, torch
sys.path.insert(0, "/path/to/eegaudit/experiments/loop/image")
from common_image import metrics, load_encoder, save, OUT

subj = sys.argv[1]; seed = 141; p = OUT / f"s2_order_nice_{subj}_seed{seed}.json"
if p.exists():
    print("SKIP", p); sys.exit(0)
t0 = time.time(); dev = torch.device("cuda")
embed, Y, tag, X = load_encoder("nice", subj, seed, dev)
rng = np.random.default_rng(seed % 1000 + 11); rows = []


def add(E, **kw):
    r = dict(**kw, **metrics(E, Y)); r["soft_bits_shuffled_pair"] = metrics(E, Y[rng.permutation(200)])["soft_bits"]; rows.append(r); return r


for k, draws in ((1, 10), (8, 10), (80, 1)):
    for d in range(draws):
        idx = np.arange(80) if k == 80 else rng.choice(80, k, replace=False)
        add(embed(X[:, idx].mean(1)), order="avg_then_encode", k=k, draw=d)
        if k > 1:
            add(np.mean([embed(np.ascontiguousarray(X[:, i])) for i in idx], 0), order="encode_then_avg", k=k, draw=d)
    print(f"{subj} k={k} done ({time.time()-t0:.0f}s)", flush=True)
noise = metrics(embed(rng.standard_normal(X.shape[0:1] + X.shape[2:]).astype(np.float32)), Y)
out = dict(subject=subj, seed=seed, encoder="nice (standard, seed 141)", rows=rows, noise_k80=noise, runtime_s=round(time.time() - t0, 1))
save(p.name, out)
for k in (1, 8, 80):
    a = [r for r in rows if r["k"] == k and r["order"] == "avg_then_encode"]; b = [r for r in rows if r["k"] == k and r["order"] == "encode_then_avg"] or a
    print(f"k={k:2d} avg->enc: W {np.mean([r['bits'] for r in a]):.3f} soft {np.mean([r['soft_bits'] for r in a]):.3f} | enc->avg: W {np.mean([r['bits'] for r in b]):.3f} soft {np.mean([r['soft_bits'] for r in b]):.3f}")
