"""I-1 DECODER-FREE EXEMPLAR CEILING (IDEATOR overnight item; THINGS-EEG2, 10 participants). FIXER, 21 Nov 2026.
EEG-space ceiling: per test image, the k-repetition mean vs the mean of the remaining 80 - k repetitions (S1 construction, the same 20
split seeds 4000 + 97k + j), raw 63 x 250 test epochs z-scored per feature with the across-image statistics of the full 80-repetition
means (label-free), cosine similarity, scored with the S1 v5 estimators (s1_averaging imported unchanged): 200-way InfoNCE (LOO tau) and
the within-category N-way exemplar score (LOO N-way tau, matched queries), k in {1, 4, 16, 40}.
Comparison: the decoder-space ceilings of results/ext/s1_v4 (ceiling200 / ceiling_exemplar, NICE-GA and BrainFLORA) at the same k;
ratio decoder / EEG per participant and as a ratio of participant means with a participant-bootstrap CI (GPU).
Output results/ext/i1_eeg_ceiling/{sub-XX.json, summary.json} with code_sha256."""
import argparse, hashlib, json, sys, time
from pathlib import Path
import numpy as np, torch
ROOT = Path("/path/to/eegaudit"); sys.path.insert(0, str(ROOT / "experiments/ext"))
import s1_averaging as S1
CODE_SHA256 = hashlib.sha256(open(__file__, "rb").read()).hexdigest(); KI = [1, 4, 16, 40]; REF = ROOT / "results/ext/s1_v4"   # v4, not v5: the v5 ceiling200 degenerates (LOO tau down to 1e-6 on saturated rows)


def subject(s, out):
    t0 = time.time(); sub = f"sub-{s:02d}"; X = np.load(ROOT / f"data_raw/things_eeg2_250hz/test_cache/{sub}_test.npy", mmap_mode="r")
    Xg = torch.as_tensor(np.ascontiguousarray(X), dtype=torch.float32, device=S1.DEV).reshape(200, 80, -1)          # (200, 80, 15750)
    M = Xg.mean(1); mu, sd = M.mean(0), M.std(0) + 1e-6
    te = S1.categories(); N, pool_of, classes, cov = S1.pools_from(te); res = dict(participant=sub, code_sha256=CODE_SHA256, rows={})
    for k in KI:
        sp = [(lambda p: (p[:k], p[k:]))(np.random.default_rng(4000 + 97 * k + j).permutation(80)) for j in range(S1.NSUB)]
        Sc = torch.stack([S1.nrm(((Xg[:, a].mean(1) - mu) / sd).double()) @ S1.nrm(((Xg[:, b].mean(1) - mu) / sd).double()).T for a, b in sp])
        y, _, tc = S1.info200(Sc); ce = S1.nway(Sc, None, pool_of, N, 50, 70 + k)
        row = dict(eeg_ceiling200=float(y.mean()), eeg_ceiling_exemplar=float(ce["matched"].mean()), tau200=float(tc.median()), tau_exemplar=ce["tau"])
        if (REF / f"{sub}.json").exists():
            R = json.loads((REF / f"{sub}.json").read_text())["rows"]
            for m in ("nice_ga", "brainflora"):
                row[f"{m}_ceiling200"] = R[f"{m}_k{k}"]["ceiling200"]; row[f"{m}_ceiling_exemplar"] = R[f"{m}_k{k}"]["ceiling_exemplar"]
        res["rows"][f"k{k}"] = row; print(sub, k, {a: round(b, 4) for a, b in row.items()}, f"{time.time() - t0:.0f}s", flush=True)
    del Xg; torch.cuda.empty_cache(); (out / f"{sub}.json").write_text(json.dumps(res, indent=1))


def summarize(out):
    P = [json.loads(f.read_text()) for f in sorted(out.glob("sub-*.json"))]; n = len(P)
    g = torch.Generator(device=S1.DEV); g.manual_seed(20261121); ix = torch.randint(0, n, (10000, n), generator=g, device=S1.DEV)
    T = lambda v: torch.as_tensor(np.asarray(v, float), dtype=torch.float64, device=S1.DEV); q = lambda b: torch.quantile(b, torch.tensor([.025, .975], dtype=torch.float64, device=S1.DEV)).tolist()
    res = dict(code_sha256=CODE_SHA256, n=n, participants=[p["participant"] for p in P], rows={})
    for k in KI:
        r = {}
        for c in ("ceiling200", "ceiling_exemplar"):
            E = T([p["rows"][f"k{k}"][f"eeg_{c}"] for p in P]); r[f"eeg_{c}"] = float(E.mean())
            for m in ("nice_ga", "brainflora"):
                if all(f"{m}_{c}" in p["rows"][f"k{k}"] for p in P):
                    D = T([p["rows"][f"k{k}"][f"{m}_{c}"] for p in P]); r[f"{m}_{c}"] = float(D.mean())
                    r[f"{m}_{c}_over_eeg"] = dict(ratio_of_means=float(D.mean() / E.mean()), ci=q(D[ix].mean(1) / E[ix].mean(1)), per_participant=(D / E).tolist())
        res["rows"][f"k{k}"] = r
    (out / "summary.json").write_text(json.dumps(res, indent=1)); print(json.dumps(res, indent=1)[:3000])


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--subjects", default="1,2,3,4,5,6,7,8,9,10"); ap.add_argument("--out", default=str(ROOT / "results/ext/i1_eeg_ceiling")); ap.add_argument("--summarize", action="store_true")
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    if a.summarize: summarize(out)
    else:
        for s in [int(x) for x in a.subjects.split(",")]:
            if not (out / f"sub-{s:02d}.json").exists(): subject(s, out)
