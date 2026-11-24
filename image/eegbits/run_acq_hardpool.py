"""Hard-negative candidate pools through the frozen NICE / NICE-GA / ATM-S encoders (audit C): does the decoder tell an image from
its nearest CLIP neighbours, or only from unrelated images?  For each test image the pool is the target plus its N-1 nearest
candidates by CLIP cosine (N = 2, 10, 50) versus a random pool of the same size; top-1 inside the pool, and shuffled-pair floor
(scores against a permuted assignment inside the same pools).  k = 80 (all repetitions) and k = 1 (5 draws).
Output: results/eegbits/acq_hardpool_<variant>_<subject>_seed<seed>.json
Usage : python run_acq_hardpool.py nice sub-01 141 | nice_ga sub-01 1024 | atm sub-08 0
"""
import json, sys, time
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).parent))
import run_acq_controls as C
OUT = Path("/path/to/eegaudit/results/eegbits")


def pools(Y, rng):
    Yn = Y / np.linalg.norm(Y, axis=1, keepdims=True); S = Yn @ Yn.T; np.fill_diagonal(S, -np.inf)
    order = np.argsort(-S, axis=1)                               # nearest first
    P = {}
    for N in (2, 10, 50):
        P[("hard", N)] = np.stack([np.r_[i, order[i, :N - 1]] for i in range(200)])
        P[("random", N)] = np.stack([np.r_[i, rng.choice([j for j in range(200) if j != i], N - 1, replace=False)] for i in range(200)])
    return P


def score(E, Y, pool, perm=None):
    En = E / np.linalg.norm(E, axis=1, keepdims=True); Yn = Y / np.linalg.norm(Y, axis=1, keepdims=True)
    S = En @ Yn.T                                                # (200 queries, 200 candidates)
    if perm is not None:
        S = S[perm]                                              # EEG of another image scored in this image's pool (shuffled pair)
    sub = np.take_along_axis(S, pool, axis=1)                    # column 0 is the target
    top1 = float((sub.argmax(1) == 0).mean())
    two_way = float((sub[:, :1] > sub[:, 1:]).mean())
    N = pool.shape[1]; P = min(max(top1, 1e-9), 1 - 1e-9)
    bits = float(np.log2(N) + P * np.log2(P) + (1 - P) * np.log2((1 - P) / (N - 1))) if N > 1 else 0.0
    return dict(N=int(N), top1=top1, two_way=two_way, bits=max(bits, 0.0))


def main(variant, subject, seed):
    t0 = time.time(); dev = torch.device("cuda"); rng = np.random.default_rng(int(seed) % 1000 + 11)
    embed, Y, tag, X = C.load_encoder(variant, subject, seed, dev)
    P = pools(Y, rng); rows = []
    for k in (80, 1):
        for draw in range(1 if k == 80 else 5):
            idx = np.arange(80) if k == 80 else rng.choice(80, k, replace=False)
            E = embed(X[:, idx].mean(1)); perm = rng.permutation(200)
            for (kind, N), pool in P.items():
                rows.append(dict(axis="hardpool", pool=kind, k=k, draw=draw, condition="eeg", **score(E, Y, pool)))
                rows.append(dict(axis="hardpool", pool=kind, k=k, draw=draw, condition="shuffled_pair", **score(E, Y, pool, perm)))
    out = dict(variant=tag, subject=subject, seed=int(seed), rows=rows, runtime_s=round(time.time() - t0, 1))
    (OUT / f"acq_hardpool_{tag}_{subject}_seed{seed}.json").write_text(json.dumps(out, indent=1))
    s = {f"{p}{N}": round(100 * np.mean([r["top1"] for r in rows if r["pool"] == p and r["N"] == N and r["k"] == 80 and r["condition"] == "eeg"]), 1) for p in ("hard", "random") for N in (2, 10, 50)}
    f = {f"{p}{N}": round(100 * np.mean([r["top1"] for r in rows if r["pool"] == p and r["N"] == N and r["k"] == 80 and r["condition"] == "shuffled_pair"]), 1) for p in ("hard",) for N in (2,)}
    print(f"{tag} {subject} seed{seed} top-1 in pool at k=80: {s} | shuffled-pair floor hard2 {f}  ({out['runtime_s']}s)", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "0")
