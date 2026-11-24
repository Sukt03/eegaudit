"""Round 4 (question c): can the learning-target ceiling be set by a measured reliability instead of the convention
rho = 0.9?  Two measured bounds on the single-trial coupling between a read-out and the prediction error (PE):
 (A) target reliability: the PE is a model quantity computed at the participant's parameter estimate; under posterior
     uncertainty about the parameters (Laplace approximation at the RWD MAP, prior included) the residualised PE series
     changes.  rel_target = corr between the residualised PE series under two independent posterior draws (mean over
     draw pairs).  Any read-out's correlation with the MAP-based PE is attenuated by sqrt(rel_target): rho_max_A.
 (B) decoder determination: within each CV fold, fit the PCA-64 ridge read-out on two interleaved halves of the
     training trials; corr(P1, P2) on held-out trials -> Spearman-Brown -> rel_dec; a read-out cannot correlate with
     anything above sqrt(rel_dec): rho_max_B.
Then re-read the exact joint curve (results/ceiling_RWD_v2) at each ceiling and recompute the ensemble's efficiency.
Usage: python r4_learning_ceiling.py [--smoke]"""
import sys, json, time
from pathlib import Path
import numpy as np, torch
ROOT = Path("/path/to/eegaudit"); sys.path.insert(0, str(ROOT))
from belief_bottleneck import cv_predict, folds, zs, LAM
from cogmodels import models as M
SMOKE = "--smoke" in sys.argv; t0 = time.time()
ARGS = [a for a in sys.argv[1:] if not a.startswith("--")]; LO, HI = (int(ARGS[0]), int(ARGS[1])) if len(ARGS) == 2 else (0, 23)   # participant index range [LO, HI)
Q = np.load(ROOT / "results/full_36318/behaviour_quantities_flow.npz")
model = M.EXTRA_MODELS["RWD"]
Z = torch.tensor(np.load(ROOT / "results/map_RWD.npy")).float()
probs = M.reward_schedule(480); nS = Z.shape[0]
C = json.load(open(ROOT / "results/ceiling_RWD_v2/results.json")); W = json.load(open(ROOT / "results/worth_ceiling_RWD_v2/results.json"))
keys = sorted(C["joint"], key=float); rhos = [float(k) for k in keys]; cs = [C["joint"][k]["exact"]["logit_alpha"]["contraction"] for k in keys]
bits = lambda c: -0.5 * np.log2(1 - c)
def incr(rho):   # EEG increment over the same-batch rho = 0 baseline, bits (linear interpolation; extrapolated beyond 0.9 by the Fisher form)
    if rho <= 0.9:
        return bits(np.interp(rho, rhos, cs)) - bits(cs[0])
    s = lambda r: r * r / (1 - r * r); A, B = 8.15, 9.38          # audit B fit: 1/(1-c) = A + B s
    return 0.5 * np.log2(A + B * s(rho)) - 0.5 * np.log2(A)

def pe_series(i, z):
    with torch.no_grad():
        return M.run(model, z[None], probs, torch.tensor(Q["choice"][i])[None], torch.tensor(Q["reward"][i])[None], torch.tensor(Q["missing"][i])[None], record=True)["q"]["pe"][0].numpy()
def resid(v, cols):
    Cm = np.c_[np.ones(len(v)), np.column_stack(cols)]; b, *_ = np.linalg.lstsq(Cm, v, rcond=None); return v - Cm @ b
def qtil(i, tri, z):
    c, r = Q["choice"][i][tri], Q["reward"][i][tri]
    cols = [r] + [(c == k).astype(float) for k in range(3)] + [(c == k).astype(float) * r for k in range(3)] + [(tri / 480.0) ** p for p in (1, 2, 3)]
    v = resid(pe_series(i, z)[tri].astype(float), cols); return (v - v.mean()) / (v.std() + 1e-9)
def corr(a, b):
    a = a - a.mean(); b = b - b.mean(); return float(a @ b / np.sqrt((a @ a) * (b @ b) + 1e-18))

def laplace_cov(i):
    ch, rw, ms = torch.tensor(Q["choice"][i])[None], torch.tensor(Q["reward"][i])[None], torch.tensor(Q["missing"][i])[None]
    mu, sd = torch.tensor(model.prior_mean), torch.tensor(model.prior_sd)
    def nlp(z):
        ll = M.run(model, z[None], probs, ch, rw, ms)["loglik"][0]
        return -ll + 0.5 * (((z - mu) / sd) ** 2).sum()
    def grad(z):
        z = z.clone().requires_grad_(True); g, = torch.autograd.grad(nlp(z), z); return g
    z0 = Z[i].clone(); eps = 1e-3; H = torch.zeros(4, 4)           # Hessian by central differences of the gradient (8 passes)
    for j in range(4):
        e = torch.zeros(4); e[j] = eps
        H[:, j] = (grad(z0 + e) - grad(z0 - e)) / (2 * eps)
    H = 0.5 * (H + H.T)
    return torch.linalg.inv(H + 1e-6 * torch.eye(4)).numpy()

def split_half_decoder(F, y, colw):
    """Two read-outs of every held-out trial from decoders fitted on interleaved halves of the training trials."""
    n = len(y); f = folds(n); P = np.zeros((n, 2))
    for b in range(10):
        tr, te = np.where(f != b)[0], f == b
        for h in range(2):
            trh = tr[h::2]
            A, B = zs(F[trh], F[te]); A, B = A * colw, B * colw
            U, S, Vt = np.linalg.svd(A - A.mean(0), full_matrices=False); Vk = Vt[:64].T
            Atr, Ate = A @ Vk, B @ Vk
            Wt = np.linalg.solve(Atr.T @ Atr + LAM * len(Atr) * np.eye(64), Atr.T @ (y[trh] - y[trh].mean()))
            P[te, h] = Ate @ Wt
    return P

rows = []; NDRAW = 2 if SMOKE else 4
for i in (range(3) if SMOKE else range(LO, HI)):
    fz = np.load(ROOT / "results/features/ens" / f"sub-{i + 1:03d}.npz"); F, colw, tri = fz["F"].astype(np.float64), fz["colw"].astype(np.float64), fz["tri"]
    y = qtil(i, tri, Z[i])
    # (A) target reliability under parameter uncertainty
    cov = laplace_cov(i); rng = np.random.default_rng(100 + i)
    draws = rng.multivariate_normal(Z[i].numpy(), cov, size=NDRAW)
    ys = [qtil(i, tri, torch.tensor(d).float()) for d in draws]
    rel_pairs = [corr(ys[a], ys[b]) for a in range(NDRAW) for b in range(a + 1, NDRAW)]
    rel_map = [corr(y, yy) for yy in ys]
    # (B) decoder determination
    P = cv_predict(F, y[:, None], "pca", 64, colw=colw)[:, 0]; r_full = corr(P, y)
    Ph = split_half_decoder(F, y, colw); r12 = corr(Ph[:, 0], Ph[:, 1]); rel_dec = 2 * r12 / (1 + r12)
    r_half = float(np.mean([corr(Ph[:, h], y) for h in range(2)]))
    rows.append(dict(subject=i + 1, n=len(tri), r_full=r_full, r_half_data=r_half, r12_half_decoders=r12, rel_decoder_sb=rel_dec,
                     rho_max_decoder=float(np.sqrt(max(rel_dec, 0))), post_sd=np.sqrt(np.diag(cov)).round(3).tolist(),
                     rel_target_pairs=float(np.mean(rel_pairs)), corr_map_vs_draw=float(np.mean(rel_map)), rho_max_target=float(np.sqrt(max(np.mean(rel_pairs), 0)))))
    (ROOT / "results/method_pilots" / f"r4_partial_{LO}-{HI}.json").write_text(json.dumps(rows))
    print(f"sub {i+1:2d}: r={r_full:.3f} half-data r={r_half:.3f} r12={r12:.3f} rel_dec={rel_dec:.3f} | PE rel across posterior draws={np.mean(rel_pairs):.3f} (MAP vs draw {np.mean(rel_map):.3f}) post sd {np.sqrt(np.diag(cov)).round(2)} ({time.time()-t0:.0f}s)", flush=True)

R = {k: np.array([r[k] for r in rows]) for k in ("r_full", "r_half_data", "r12_half_decoders", "rel_decoder_sb", "rho_max_decoder", "rel_target_pairs", "rho_max_target")}
rho_eff = W["decoders"]["ens"]["rho_eff"]
summary = {k: dict(mean=float(v.mean()), sd=float(v.std(ddof=1)) if len(v) > 1 else 0.0, min=float(v.min()), max=float(v.max())) for k, v in R.items()}
eff = {}
for name, rc in (("convention_0.9", 0.9), ("rho_max_target_mean", float(R["rho_max_target"].mean())), ("rho_max_decoder_mean", float(R["rho_max_decoder"].mean())),
                 ("rho_max_target_min", float(R["rho_max_target"].min())), ("rho_max_product", float((R["rho_max_target"] * R["rho_max_decoder"]).mean()))):
    eff[name] = dict(ceiling_rho=rc, ceiling_increment_bits=incr(rc), ens_increment_bits=incr(rho_eff), efficiency_pct=100 * incr(rho_eff) / incr(rc))
out = dict(decoder="ens", rho_eff_ens=rho_eff, n_subjects=len(rows), n_posterior_draws=NDRAW, rows=rows, summary=summary, efficiency=eff, runtime_s=round(time.time() - t0, 1))
Path(ROOT / "results/method_pilots").mkdir(exist_ok=True)
(ROOT / "results/method_pilots" / f"r4_learning_ceiling{'_smoke' if SMOKE else f'_{LO}-{HI}'}.json").write_text(json.dumps(out, indent=1))
print(json.dumps(summary, indent=0)); print(json.dumps(eff, indent=0))
