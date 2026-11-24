"""Round 2: do the price axes (repetitions k*, window for 90 %, 32-electrode share) change when yield is defined by the
InfoNCE soft bound or by chance-adjusted 2-way instead of Wolpaw top-1 bits?  And (audit D, question e) does
"encode then average embeddings" change the repetition price relative to "average in input space then encode"?
Frozen NICE-GA, one subject x seed per call. Fill for masked samples/channels = leave-one-out grand mean (the paper's).
Usage: python r2_estimator_axes.py sub-04 141 [--smoke]"""
import sys, time
import numpy as np, torch
sys.path.insert(0, "/path/to/eegaudit/experiments/method_pilots")
from common import load_encoder, sim, score, nn_index, save
sys.path.insert(0, "/path/to/eegaudit/experiments/eegbits")
from run_acq_nice import OCC_PAR, RAW

subj, seed = sys.argv[1], int(sys.argv[2]); SMOKE = "--smoke" in sys.argv
KS = [1, 2, 4, 8, 16, 32, 80]; DRAWS = 2 if SMOKE else 10; SOA = 0.2
t0 = time.time(); dev = torch.device("cuda")
embed, Y, tag, X = load_encoder("nice_ga", subj, seed, dev)
nn = nn_index(Y)
import json
chn = json.load(open("/path/to/eegaudit/results/eegbits/raw_sub-01.json"))["raw_meta"]["ch_names"]   # the 63-ch THINGS-EEG2 montage, cached by run.py
occ = [i for i, c in enumerate(chn) if c in OCC_PAR][:32]
assert len(occ) == 32, len(occ)
rng = np.random.default_rng(seed % 1000 + 11)
pool = X.mean(1); pool_loo = (pool.sum(0, keepdims=True) - pool) / 199

def q(x, chans=None, w=None):
    x = x.copy(); m = np.zeros((63, 250), bool)
    if chans is not None:
        cm = np.ones(63, bool); cm[chans] = False; m[cm, :] = True
    if w is not None:
        m[:, int(round(w * 250)):] = True
    x[:, m] = pool_loo[:, m]; return x

rows = []
def add(**kw):
    S = kw.pop("S"); r = dict(**kw, **score(S, nn)); r["soft_bits_sp"] = score(S[rng.permutation(200)])["soft_bits"]; rows.append(r); return r

# (1) repetitions, average-then-encode (paper) and encode-then-average (audit D question e)
for k in KS:
    for d in range(1 if k == 80 else DRAWS):
        idx = np.arange(80) if k == 80 else rng.choice(80, k, replace=False)
        add(axis="repetitions", order="avg_then_encode", k=k, draw=d, S=sim(embed(X[:, idx].mean(1)), Y))
        if k > 1:
            E = np.mean([embed(np.ascontiguousarray(X[:, i])) for i in idx], 0)
            add(axis="repetitions", order="encode_then_avg", k=k, draw=d, S=sim(E, Y))
print(f"{subj} s{seed} reps done ({time.time()-t0:.0f}s)", flush=True)
# (2) window and electrodes at k = 80 and k = 1 (DRAWS draws), mean_loo fill
full = X.mean(1)
for name, chans in (("all63", None), ("occpar32", occ)):
    for w in (0.2, 0.4, 0.6, 1.0):
        add(axis="acquisition", channels=name, window=w, k=80, draw=0, S=sim(embed(q(full, chans, w)), Y))
        for d in range(DRAWS if not SMOKE else 1):
            x1 = X[:, rng.choice(80, 1, replace=False)].mean(1)
            add(axis="acquisition", channels=name, window=w, k=1, draw=d, S=sim(embed(q(x1, chans, w)), Y))
out = dict(subject=subj, seed=seed, encoder="nice_ga", fill="mean_loo", soa_s=SOA, draws=DRAWS, rows=rows, runtime_s=round(time.time() - t0, 1))
save(f"r2_estimator_axes_{subj}_seed{seed}{'_smoke' if SMOKE else ''}.json", out)
def m(ax, **kw):
    v = [r for r in rows if r["axis"] == ax and all(r.get(a) == b for a, b in kw.items())]
    return {e: float(np.mean([r[e] for r in v])) for e in ("top1", "two_way", "bits", "soft_bits", "nn_two_way")}
print("k / avg-then-encode bits, soft | encode-then-avg bits, soft")
for k in KS:
    a = m("repetitions", order="avg_then_encode", k=k); b = m("repetitions", order="encode_then_avg", k=k) if k > 1 else a
    print(f"k={k:2d}: {a['bits']:.3f} {a['soft_bits']:.3f} 2way {a['two_way']:.3f} | {b['bits']:.3f} {b['soft_bits']:.3f}  bits/s {a['bits']/(k*SOA):.3f} soft/s {a['soft_bits']/(k*SOA):.3f}")
for name in ("all63", "occpar32"):
    for w in (0.2, 0.4, 0.6, 1.0):
        a = m("acquisition", channels=name, window=w, k=80)
        print(f"{name} w={w}: bits {a['bits']:.3f} soft {a['soft_bits']:.3f} 2way {a['two_way']:.3f} nn {a['nn_two_way']:.3f}")
