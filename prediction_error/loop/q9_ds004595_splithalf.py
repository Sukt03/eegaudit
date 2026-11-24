"""Q9: EEGNet decoder-determination ceiling on ds004595 (as R5 part B / S16 on ds003458). For each held-out participant h, two EEGNets are
trained exactly as deep_decoders_ds004595.train_fold (same train/val participants, same early stopping on the validation participant, seed = the
participant label) but each on one interleaved half of the pooled training trials; r12 = corr of their predictions on h's trials ->
Spearman-Brown rel = 2 r12 / (1 + r12) -> rho_max = sqrt(rel); ceiling_i = -1/2 log2(1 - rho_max^2). r_full from the cached full LOPO
prediction (results/features_ds004595/eegnet). Run from the repo root. Usage: python -I experiments/loop/q9_ds004595_splithalf.py <lo> <hi> [--epochs 10]"""
import sys, json, time, argparse
from pathlib import Path
import numpy as np, torch
ROOT = Path("/path/to/eegaudit"); sys.path.insert(0, str(ROOT))
import deep_decoders_ds004595 as D2
ap = argparse.ArgumentParser(); ap.add_argument("lo", type=int); ap.add_argument("hi", type=int); ap.add_argument("--epochs", type=int, default=10); a = ap.parse_args()
t0 = time.time(); dev = "cuda"; OUT = ROOT / "results/loop/q9_ds004595_ceiling"
data, n_ch = D2.load_all(); subs = list(D2.SUBS); print(f"loaded {len(data)} participants, {n_ch} channels ({time.time()-t0:.0f}s)", flush=True)
orig = {s: data[s] for s in data}


def train_half(h, half, epochs):
    rest = [s for s in subs if s != h]; val = rest[(subs.index(h) * 7) % len(rest)]
    halved = {s: ((orig[s][0][half::2], orig[s][1][half::2], orig[s][2][half::2]) if (s != h and s != val) else orig[s]) for s in subs}
    return D2.train_fold(halved, h, n_ch, dev, seed=h, epochs=epochs)[0]   # validation participant and held-out participant keep all trials


corr = lambda x, y: float(np.corrcoef(x, y)[0, 1]); rows = []
for h in subs[a.lo:a.hi]:
    y = orig[h][1]; P = [train_half(h, half, a.epochs) for half in (0, 1)]
    r12 = corr(P[0], P[1]); rel = 2 * r12 / (1 + r12); rho = float(np.sqrt(max(rel, 0)))
    full = np.load(ROOT / f"results/features_ds004595/eegnet/sub-{h:03d}.npz"); assert len(full["F"]) == len(y); r_full = corr(full["F"][:, 0], y)
    rows.append(dict(subject=int(h), n=len(y), r_full=r_full, r_half_data=float(np.mean([corr(p, y) for p in P])), r12_half_decoders=r12, rel_decoder_sb=rel, rho_max_decoder=rho,
                     ceiling_bits=float(-0.5 * np.log2(1 - min(rho, 0.999) ** 2)), epochs=a.epochs))
    print(f"sub {h:3d}: r_full {r_full:.3f} r12 {r12:.3f} rho_max {rho:.3f} ceiling {rows[-1]['ceiling_bits']:.3f} ({time.time()-t0:.0f}s)", flush=True)
    (OUT / f"q9_splithalf_{a.lo}-{a.hi}.json").write_text(json.dumps(dict(dataset="ds004595", decoder="eegnet_lopo", rows=rows, runtime_s=round(time.time() - t0, 1)), indent=1))
print("done", time.time() - t0)
