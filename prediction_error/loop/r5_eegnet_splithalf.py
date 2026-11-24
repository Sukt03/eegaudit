"""R5 part B: decoder-determination ceiling for EEGNet LOPO. For each held-out participant h, two EEGNets are trained exactly as
deep_decoders.train_fold (same train/val participants, same early stopping, same seed rule) but each on one of two interleaved halves
of the pooled training trials; r12 = corr of their predictions on h's trials -> Spearman-Brown rel_dec -> rho_max_decoder = sqrt(rel_dec)
(the same bound r4_learning_ceiling.py measures for the ens read-out). Also r_half (each half-decoder vs h's labels) and r_full (the
cached full LOPO prediction, results/features/eegnet) for the Spearman-Brown check. Must run from the repo root (deep_decoders paths).
Usage: python -I experiments/loop/r5_eegnet_splithalf.py <lo> <hi> [--epochs N] [--smoke]   (participants lo..hi-1, 0-based)"""
import sys, json, time, argparse
from pathlib import Path
import numpy as np, torch
ROOT = Path("/path/to/eegaudit"); sys.path.insert(0, str(ROOT))
import deep_decoders as DD
ap = argparse.ArgumentParser(); ap.add_argument("lo", type=int); ap.add_argument("hi", type=int); ap.add_argument("--epochs", type=int, default=30)
ap.add_argument("--smoke", action="store_true"); a = ap.parse_args()
t0 = time.time(); dev = "cuda" if torch.cuda.is_available() else "cpu"
OUT = ROOT / "results/loop/r5_eegnet_ceiling"; OUT.mkdir(parents=True, exist_ok=True)
data, n_ch = DD.load_all(); print(f"loaded {len(data)} participants, {n_ch} channels ({time.time()-t0:.0f}s)", flush=True)

def train_half(h, half, seed, epochs):
    torch.manual_seed(seed); np.random.seed(seed)
    rest = [k for k in range(len(data)) if k != h]; val = rest[(h * 7) % len(rest)]; tr = [k for k in rest if k != val]
    Xtr = np.concatenate([data[k][0] for k in tr]); ytr = np.concatenate([data[k][1] for k in tr])
    Xtr, ytr = Xtr[half::2], ytr[half::2]                                   # interleaved half of the pooled training trials
    fwd, net, lr, _ = DD.make("eegnet", n_ch); net.to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-2)
    Xv = torch.tensor(data[val][0], device=dev); yv = torch.tensor(data[val][1], device=dev)
    best, best_state, bs = -1e9, None, 128
    for ep in range(epochs):
        net.train(); perm = np.random.permutation(len(Xtr))
        for b in range(0, len(perm), bs):
            idx = perm[b:b + bs]; xb = torch.tensor(Xtr[idx], device=dev); yb = torch.tensor(ytr[idx], device=dev)
            loss = ((fwd(xb) - yb) ** 2).mean(); opt.zero_grad(); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            pv = torch.cat([fwd(Xv[b:b + 256]) for b in range(0, len(Xv), 256)]); r = float(torch.corrcoef(torch.stack([pv, yv]))[0, 1])
        if r > best: best = r; best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
    net.load_state_dict(best_state); net.eval(); Xh = torch.tensor(data[h][0], device=dev)
    with torch.no_grad():
        ph = torch.cat([fwd(Xh[b:b + 256]) for b in range(0, len(Xh), 256)]).cpu().numpy()
    return ph, best
corr = lambda x, y: float(np.corrcoef(x, y)[0, 1])
rows = []; tag = f"{a.lo}-{a.hi}" + ("_smoke" if a.smoke else "")
for h in range(a.lo, a.hi):
    s = h + 1; y = data[h][1]
    P = [train_half(h, half, 0 + s, a.epochs)[0] for half in (0, 1)]       # seed rule of deep_decoders.main (--seed 0)
    r12 = corr(P[0], P[1]); rel = 2 * r12 / (1 + r12); r_half = float(np.mean([corr(p, y) for p in P]))
    full = np.load(ROOT / f"results/features/eegnet/sub-{s:03d}.npz"); r_full = corr(full["F"][:, 0], y)
    assert len(full["F"]) == len(y)
    rows.append(dict(subject=s, n=len(y), r_full=r_full, r_half_data=r_half, r12_half_decoders=r12, rel_decoder_sb=rel, rho_max_decoder=float(np.sqrt(max(rel, 0))), epochs=a.epochs))
    print(f"sub {s:2d}: r_full {r_full:.3f} r_half {r_half:.3f} r12 {r12:.3f} rel_dec {rel:.3f} rho_max_dec {np.sqrt(max(rel,0)):.3f} ({time.time()-t0:.0f}s)", flush=True)
    (OUT / f"r5_eegnet_splithalf_{tag}.json").write_text(json.dumps(dict(decoder="eegnet", epochs=a.epochs, rows=rows, runtime_s=round(time.time() - t0, 1)), indent=1))
