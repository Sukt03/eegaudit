"""Round 3: is the masking fill information-free?  The paper fills masked samples/channels with the leave-one-out grand
mean of the other 199 TEST images (changes with the query; estimated on the test collection).  Compare, through frozen
NICE-GA (seed 141), on the window and 32-electrode axes:
  zero          : zeros (audit D: out of distribution, under-reports)
  mean_loo      : the paper's fill
  mean_half     : fixed fill = grand mean of test images 0..99; scored on queries 100..199 only (independent of every
                  scored query; mean_loo is re-scored on the same 100 queries for a paired comparison)
  mean_othersub : fixed fill = grand mean of ANOTHER subject's test ERPs (independent of this subject's data entirely)
  mean_10rep    : grand mean over the 200 images using only 10 random repetitions (a cheap calibration-block estimate)
Usage: python r3_fill.py sub-04 sub-05 [--smoke]   (second argument = the other subject for mean_othersub)"""
import sys, time, json
import numpy as np, torch
sys.path.insert(0, "/path/to/eegaudit/experiments/method_pilots")
from common import load_encoder, load_test, sim, soft_bits, wolpaw_bits, nn_index, save
sys.path.insert(0, "/path/to/eegaudit/experiments/eegbits")
from run_acq_nice import OCC_PAR

subj, other = sys.argv[1], sys.argv[2]; SMOKE = "--smoke" in sys.argv; DRAWS = 1 if SMOKE else 5
t0 = time.time(); dev = torch.device("cuda")
embed, Y, tag, X = load_encoder("nice_ga", subj, 141, dev)
nn = nn_index(Y)
chn = json.load(open("/path/to/eegaudit/results/eegbits/raw_sub-01.json"))["raw_meta"]["ch_names"]
occ = [i for i, c in enumerate(chn) if c in OCC_PAR][:32]
rng = np.random.default_rng(3)
pool = X.mean(1)
fills = dict(zero=np.zeros((200, 63, 250), np.float32),
             mean_loo=((pool.sum(0, keepdims=True) - pool) / 199).astype(np.float32),
             mean_half=np.broadcast_to(pool[:100].mean(0, keepdims=True), (200, 63, 250)).astype(np.float32),
             mean_othersub=np.broadcast_to(load_test(other).mean(1).mean(0, keepdims=True), (200, 63, 250)).astype(np.float32),
             mean_10rep=np.broadcast_to(X[:, rng.choice(80, 10, replace=False)].mean((0, 1))[None], (200, 63, 250)).astype(np.float32))

def score_rows(S, idx):
    """Metrics for query rows idx of a (200, 200) similarity matrix, candidates = all 200."""
    Sr = S[idx]; diag = S[idx, idx]
    rank = (Sr > diag[:, None]).sum(1) + 1; top1 = float((rank == 1).mean())
    # soft bound on the subset rows with LOO temperature: reuse soft_bits on a square matrix built so the true item is on the diagonal
    order = np.concatenate([idx, np.setdiff1d(np.arange(200), idx)])
    Ssq = S[np.ix_(idx, order)]                      # (n, 200) with true item of row j at column j
    n = len(idx); LOG2 = np.log(2); taus = np.logspace(-2.5, 0, 40); rows = np.arange(n)
    CE = []
    for t in taus:
        Z = Ssq / t; Z = Z - Z.max(1, keepdims=True); lp = Z - np.log(np.exp(Z).sum(1, keepdims=True)); CE.append(-lp[rows, rows] / LOG2)
    CE = np.stack(CE); loo = (CE.sum(1, keepdims=True) - CE) / (n - 1); ti = loo.argmin(0)
    return dict(top1=top1, two_way=float(((Sr < diag[:, None]).sum(1) / 199).mean()), bits=wolpaw_bits(top1, 200),
                soft_bits=float(np.log2(200) - CE[ti, rows].mean()), nn_two_way=float((diag > S[idx, nn[idx]]).mean()))

def q(x, fill, chans=None, w=None):
    x = x.copy(); m = np.zeros((63, 250), bool)
    if chans is not None:
        cm = np.ones(63, bool); cm[chans] = False; m[cm, :] = True
    if w is not None:
        m[:, int(round(w * 250)):] = True
    x[:, m] = fill[:, m]; return x

ALL, HALF = np.arange(200), np.arange(100, 200)
rows = []
conds = [("all63", None, w) for w in (0.2, 0.4, 0.6)] + [("occpar32", occ, 1.0), ("occpar32", occ, 0.4)]
if SMOKE: conds = conds[1:2] + conds[3:4]
S_full = sim(embed(pool), Y)
for name, qidx in (("all200", ALL), ("q100-199", HALF)):
    rows.append(dict(fill="none", channels="all63", window=1.0, k=80, draw=0, queries=name, **score_rows(S_full, qidx)))
x1s = [X[:, rng.choice(80, 1, replace=False)].mean(1) for _ in range(DRAWS)]
for d, x1 in enumerate(x1s):
    S1 = sim(embed(x1), Y)
    for name, qidx in (("all200", ALL), ("q100-199", HALF)):
        rows.append(dict(fill="none", channels="all63", window=1.0, k=1, draw=d, queries=name, **score_rows(S1, qidx)))
for fname, fill in fills.items():
    for cname, chans, w in conds:
        S = sim(embed(q(pool, fill, chans, w)), Y)
        for name, qidx in (("all200", ALL), ("q100-199", HALF)):
            rows.append(dict(fill=fname, channels=cname, window=w, k=80, draw=0, queries=name, **score_rows(S, qidx)))
        for d, x1 in enumerate(x1s):
            S1 = sim(embed(q(x1, fill, chans, w)), Y)
            for name, qidx in (("all200", ALL), ("q100-199", HALF)):
                rows.append(dict(fill=fname, channels=cname, window=w, k=1, draw=d, queries=name, **score_rows(S1, qidx)))
    r = [x for x in rows if x["fill"] == fname and x["k"] == 80 and x["queries"] == "all200"]
    print(f"{fname:14s} k=80 all200: " + " | ".join(f"{x['channels']} w{x['window']}: {x['bits']:.2f}b {x['soft_bits']:.2f}s {x['two_way']:.3f}" for x in r), f"({time.time()-t0:.0f}s)", flush=True)
save(f"r3_fill_{subj}_other-{other}{'_smoke' if SMOKE else ''}.json", dict(subject=subj, other_subject=other, encoder="nice_ga seed141", draws=DRAWS, rows=rows, runtime_s=round(time.time() - t0, 1)))
