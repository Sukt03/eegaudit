"""KM3 LOCK §4 noise-input floor (FIXER, 21 Nov 2026). km3_category_erasure.py is imported unchanged.
Per participant: the KM3 'none' read-out (training PCA-1024 + ridge, run.py rule, D1.pca_batch / fit_batch) is refit exactly as KM3 and
applied to (a) the real k80 test EEG (G2: soft12 and exemplar_exact reproduce results/ext/km3_v2/km3_sub-XX.json 'none' within 1e-6 when
that file exists) and (b) N(0,1) test input of the same shape scaled to the training data's per-feature mean / sd (seed 2000 + s).
Scored with KM3's own exemplar_bits (exact-profile pools) and D1.gscore soft12. Output results/ext/km3_noise/{km3n_sub-XX.json, summary.json}."""
import argparse, hashlib, json, sys, time
from pathlib import Path
import numpy as np, torch
ROOT = Path("/path/to/eegaudit"); sys.path.insert(0, str(ROOT / "experiments/ext"))
import km3_category_erasure as K
CODE_SHA256 = hashlib.sha256(open(__file__, "rb").read()).hexdigest(); DEV = "cuda"


def subject(s, out, Ctr, Cte):
    t0 = time.time(); sub = f"sub-{s:02d}"; D1, R = K.D1, K.R
    Ytr, Yte = R.load_torch(R.CLIP / "ViT-H-14_features_train.pt"), R.load_torch(R.CLIP / "ViT-H-14_features_test.pt")
    Yg = torch.as_tensor(Ytr, dtype=torch.float64, device=DEV); Yte_g = torch.as_tensor(Yte, dtype=torch.float64, device=DEV)
    Xtr, Xk80, _, _, ch = D1.load_gpu(sub); n = len(Xtr); Ct = torch.as_tensor(Cte, dtype=torch.float64, device=DEV)
    rng = np.random.default_rng(0); perms = {L: rng.permutation(n) for L in K.LS}
    F = Xtr.reshape(n, -1); mu, sd = F.mean(0), F.std(0)
    g = torch.Generator(device=DEV); g.manual_seed(2000 + s); Xn = torch.randn((200, F.shape[1]), generator=g, device=DEV) * sd + mu
    Ztr, (Zte, Zn) = D1.pca_batch(F[None], [Xk80.reshape(200, -1)[None], Xn[None]], 1024)
    fits = D1.fit_batch(Ztr, [Zte, Zn], Yg, K.LS, perms); res = dict(participant=sub, code_sha256=CODE_SHA256, conditions={})
    for i, nm in enumerate(("none_real", "none_noise_input")):
        P = fits[1024][1][i]; S = D1.gcos(P, Yte_g); sc = D1.gscore(S)
        res["conditions"][nm] = dict(soft12=float(sc["soft12"][0]), exemplar_exact=float(K.exemplar_bits(S, Ct, exact=True)[0]), exemplar=float(K.exemplar_bits(S, Ct)[0]))
    ref = ROOT / f"results/ext/km3_v2/km3_{sub}.json"
    if ref.exists():
        o = json.loads(ref.read_text())["conditions"]["none"]; r = res["conditions"]["none_real"]
        res["g2"] = dict(soft12=abs(o["soft12"] - r["soft12"]), exemplar_exact=abs(o["exemplar_exact"] - r["exemplar_exact"])); res["g2"]["pass"] = bool(max(res["g2"].values()) < 1e-6)
    res["runtime_s"] = round(time.time() - t0, 1); (out / f"km3n_{sub}.json").write_text(json.dumps(res, indent=1))
    print(sub, {k: {a: round(b, 4) for a, b in v.items()} for k, v in res["conditions"].items()}, res.get("g2"), f"{res['runtime_s']}s", flush=True)
    del Xtr, Xk80, Ztr, Zte, Zn, fits; torch.cuda.empty_cache()


def summarize(out):
    P = [json.loads(f.read_text()) for f in sorted(out.glob("km3n_sub-*.json"))]
    res = dict(code_sha256=CODE_SHA256, n=len(P), g2_all_pass=all(p.get("g2", {}).get("pass", False) for p in P),
               noise_input={q: float(np.mean([p["conditions"]["none_noise_input"][q] for p in P])) for q in ("soft12", "exemplar_exact", "exemplar")},
               real={q: float(np.mean([p["conditions"]["none_real"][q] for p in P])) for q in ("soft12", "exemplar_exact", "exemplar")})
    res["noise_within_0.02"] = bool(all(abs(v) <= 0.02 for v in res["noise_input"].values()))
    (out / "summary.json").write_text(json.dumps(res, indent=1)); print(json.dumps(res, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--subjects", default="1,2,3,4,5,6,7,8,9,10"); ap.add_argument("--out", default=str(ROOT / "results/ext/km3_noise")); ap.add_argument("--summarize", action="store_true")
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    if a.summarize: summarize(out)
    else:
        Ctr, Cte, _, _, _ = K.categories()
        for s in [int(x) for x in a.subjects.split(",")]:
            if not (out / f"km3n_sub-{s:02d}.json").exists(): subject(s, out, Ctr, Cte)
