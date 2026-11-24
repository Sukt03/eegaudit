"""Round 1: is the linear bottleneck a measurement of the EEG when it is placed on the ENCODER INPUT?
Standardise the raw test EEG (63 x 250 = 15,750 features), PCA basis fitted on single trials (own subject, or another
subject's trials for a leakage-free basis), reconstruct each 80-rep (and 1-rep) test query at rank L, feed it to the
frozen NICE-GA encoder, score its native 200-way retrieval. Reference: the paper's embedding-side probe saturates at
L = 25-50 (91 / 95 % of full-width 2-way yield).
Usage: python r1_input_bottleneck.py sub-04 [basis_subject | avg] [--smoke]"""
import sys, time
import numpy as np, torch
sys.path.insert(0, "/path/to/eegaudit/experiments/method_pilots")
from common import load_encoder, load_test, sim, score, nn_index, save

subj = sys.argv[1]; basis_subj = sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].startswith("--") else subj
SMOKE = "--smoke" in sys.argv
LS = [1, 2, 5, 10, 25, 50, 100, 250, 500, 1000, 2000, 4000] if not SMOKE else [5, 50, 500]
t0 = time.time(); dev = torch.device("cuda")
embed, Y, tag, X = load_encoder("nice_ga", subj, 141, dev)
nn = nn_index(Y)
D = 63 * 250
# PCA basis from single trials of the basis subject (unsupervised; no labels, no candidates involved)
if basis_subj == "avg":      # basis from the 80-rep means of the 200 test queries (signal-dominated, rank <= 199)
    Zb = torch.from_numpy(np.ascontiguousarray(X.mean(1)).reshape(-1, D)).to(dev); LS = [L for L in LS if L < 200] + [199]
else:
    Xb = load_test(basis_subj) if basis_subj != subj else X
    Zb = torch.from_numpy(np.ascontiguousarray(Xb).reshape(-1, D)).to(dev)
mu, sd = Zb.mean(0), Zb.std(0) + 1e-6
Zb = (Zb - mu) / sd
C = (Zb.T @ Zb) / Zb.shape[0]; del Zb
w, V = torch.linalg.eigh(C.double()); w, V = torch.flip(w, [0]), torch.flip(V, [1]).float(); del C
ev = (w / w.sum()).cpu().numpy()
print(f"{subj} basis={basis_subj}: PCA done, var explained at L=25/50/250/1000: " + " ".join(f"{ev[:L].sum():.3f}" for L in (25, 50, 250, 1000)), f"({time.time()-t0:.0f}s)", flush=True)

def recon(x, L):
    """x: (200, 63, 250) -> rank-L reconstruction in the standardised basis, back in volts."""
    Z = (torch.from_numpy(np.ascontiguousarray(x).reshape(len(x), D)).to(dev) - mu) / sd
    P = V[:, :L]
    R = (Z @ P) @ P.T
    return ((R * sd + mu).reshape(len(x), 63, 250)).cpu().numpy()

rng = np.random.default_rng(0)
full = X.mean(1)
rows = [dict(L="full", k=80, condition="eeg", draw=0, **score(sim(embed(full), Y), nn))]
print("full:", {k: round(v, 3) for k, v in rows[0].items() if isinstance(v, float)}, flush=True)
k1 = [X[:, rng.choice(80, 1, replace=False)].mean(1) for _ in range(3)]
for d, x1 in enumerate(k1):
    rows.append(dict(L="full", k=1, condition="eeg", draw=d, **score(sim(embed(x1), Y), nn)))
for L in LS:
    e = embed(recon(full, L)); S = sim(e, Y)
    rows.append(dict(L=L, k=80, condition="eeg", draw=0, var_explained=float(ev[:L].sum()), **score(S, nn)))
    rows.append(dict(L=L, k=80, condition="shuffled_pair", draw=0, **score(S[rng.permutation(200)], nn)))
    noise = rng.standard_normal(full.shape).astype(np.float32) * float(sd.mean().cpu()) + float(mu.mean().cpu())
    rows.append(dict(L=L, k=80, condition="noise", draw=0, **score(sim(embed(recon(noise, L)), Y), nn)))
    for d, x1 in enumerate(k1):
        rows.append(dict(L=L, k=1, condition="eeg", draw=d, **score(sim(embed(recon(x1, L)), Y), nn)))
    r = rows[-3 - len(k1)]
    print(f"L={L:5d} var={ev[:L].sum():.3f} k=80: top1 {r['top1']:.3f} 2way {r['two_way']:.3f} bits {r['bits']:.2f} soft {r['soft_bits']:.2f} nn2way {r['nn_two_way']:.3f} | k=1 bits {np.mean([q['bits'] for q in rows[-len(k1):]]):.3f} ({time.time()-t0:.0f}s)", flush=True)
save(f"r1_input_bottleneck_{subj}_basis-{basis_subj}{'_smoke' if SMOKE else ''}.json",
     dict(subject=subj, basis_subject=basis_subj, encoder="nice_ga seed141", Ls=LS, var_explained_by_L={str(L): float(ev[:L].sum()) for L in LS}, rows=rows, runtime_s=round(time.time() - t0, 1)))
