"""S3: input-side PCA bottleneck through the frozen ATM-S encoder (sub-08 checkpoint). Logic copied from
experiments/method_pilots/r1_input_bottleneck.py (NICE-GA), unchanged except the encoder: standardise the 63 x 250 test EEG, PCA basis
from the 16,000 single test trials of the same subject, reconstruct each 80-rep (and 1-rep) query at rank L, encode, score 200-way.
Controls at every L: shuffled pairs and N(0,1) noise through the same bottleneck. Scores: run.metrics (Wolpaw + soft bits).
Output: results/loop/image/s3_atm_input_bottleneck_sub-08.json.  Usage: python s3_atm_input_bottleneck.py [sub-08]"""
import sys, time
import numpy as np, torch
sys.path.insert(0, "/path/to/eegaudit/experiments/loop/image")
from common_image import metrics, load_encoder, save, OUT

subj = sys.argv[1] if len(sys.argv) > 1 else "sub-08"; p = OUT / f"s3_atm_input_bottleneck_{subj}.json"
if p.exists():
    print("SKIP", p); sys.exit(0)
LS = [25, 50, 100, 250, 500, 1000]
t0 = time.time(); dev = torch.device("cuda")
embed, Y, tag, X = load_encoder("atm", subj, 0, dev); X = np.asarray(X); D = 63 * 250
Zb = torch.from_numpy(np.ascontiguousarray(X).reshape(-1, D)).to(dev)
mu, sd = Zb.mean(0), Zb.std(0) + 1e-6; Zb = (Zb - mu) / sd
C = (Zb.T @ Zb) / Zb.shape[0]; del Zb
w, V = torch.linalg.eigh(C.double()); w, V = torch.flip(w, [0]), torch.flip(V, [1]).float(); del C; torch.cuda.empty_cache()
ev = (w / w.sum()).cpu().numpy()
print(f"{subj} ATM-S: PCA done, var explained " + " ".join(f"L={L}:{ev[:L].sum():.3f}" for L in LS), f"({time.time()-t0:.0f}s)", flush=True)


def recon(x, L):
    Z = (torch.from_numpy(np.ascontiguousarray(x).reshape(len(x), D)).to(dev) - mu) / sd
    P = V[:, :L]; R = (Z @ P) @ P.T
    return ((R * sd + mu).reshape(len(x), 63, 250)).cpu().numpy()


rng = np.random.default_rng(0); full = X.mean(1)
rows = [dict(L="full", k=80, condition="eeg", draw=0, **metrics(embed(full), Y))]
rows.append(dict(L="full", k=80, condition="shuffled_pair", draw=0, **metrics(embed(full), Y[rng.permutation(200)])))
print("full:", {k: round(v, 3) for k, v in rows[0].items() if isinstance(v, float)}, flush=True)
k1 = [X[:, rng.choice(80, 1, replace=False)].mean(1) for _ in range(3)]
for d, x1 in enumerate(k1):
    rows.append(dict(L="full", k=1, condition="eeg", draw=d, **metrics(embed(x1), Y)))
for L in LS:
    e = embed(recon(full, L))
    rows.append(dict(L=L, k=80, condition="eeg", draw=0, var_explained=float(ev[:L].sum()), **metrics(e, Y)))
    rows.append(dict(L=L, k=80, condition="shuffled_pair", draw=0, **metrics(e, Y[rng.permutation(200)])))
    noise = rng.standard_normal(full.shape).astype(np.float32) * float(sd.mean().cpu()) + float(mu.mean().cpu())
    rows.append(dict(L=L, k=80, condition="noise", draw=0, **metrics(embed(recon(noise, L)), Y)))
    for d, x1 in enumerate(k1):
        rows.append(dict(L=L, k=1, condition="eeg", draw=d, **metrics(embed(recon(x1, L)), Y)))
    r = [q for q in rows if q["L"] == L and q["k"] == 80 and q["condition"] == "eeg"][0]
    print(f"L={L:5d} var={ev[:L].sum():.3f} k=80: top1 {r['top1']:.3f} bits {r['bits']:.2f} soft {r['soft_bits']:.2f} | k=1 bits {np.mean([q['bits'] for q in rows[-3:]]):.3f} soft {np.mean([q['soft_bits'] for q in rows[-3:]]):.3f} ({time.time()-t0:.0f}s)", flush=True)
save(p.name, dict(subject=subj, encoder="atm-s sub-08 checkpoint (frozen)", basis="single test trials of the same subject", Ls=LS,
                  var_explained_by_L={str(L): float(ev[:L].sum()) for L in LS}, rows=rows, runtime_s=round(time.time() - t0, 1)))
