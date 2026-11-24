"""Group-level statistics for single-trial decoding, following the EEG/MVPA literature.

Per participant i and decoder: cross-validated decoding r_i of the read-out quantity (prediction error of RW with decay
at the participant's MAP, residualized on outcome, chosen arm, interaction, cubic time) from feedback EEG, and the
exact session-permutation null: the same EEG paired with each of the other participants' quantities on the same trials
(residualized on their own histories), giving r_ij for all j != i.
1. Prevalence inference (Allefeld, Goergen & Haynes 2016, NeuroImage 141:378-392): permutation-based minimum
   statistic; global-null p and the largest prevalence gamma0 rejected at alpha = .05 (majority null gamma0 = 0.5).
2. Random-effects meta-analysis of null-corrected Fisher z (DerSimonian-Laird): d_i = atanh(r_i) - mean_j atanh(r_ij),
   var_i = var_j atanh(r_ij); pooled mean, 95% CI, tau^2, prediction interval; back-transformed coupling.
3. Paired decoder comparisons: the same random-effects model on per-participant differences d_i(A) - d_i(B).
Usage: python group_stats.py --decoders raw ens fm fmtok labram eegpt interior eegnet cbramod_ft --out results/group_stats
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy import stats

from belief_bottleneck import LAM, cv_predict, folds

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def cv_predict_gpu(F, Y, L=64, colw=None):
    """Same computation as belief_bottleneck.cv_predict(mode='pca') in float64 torch on the GPU."""
    F = torch.tensor(F, dtype=torch.float64, device=DEV); Y = torch.tensor(Y, dtype=torch.float64, device=DEV)
    cw = None if colw is None else torch.tensor(colw, dtype=torch.float64, device=DEV)
    f = torch.tensor(folds(len(F)), device=DEV); P = torch.zeros_like(Y)
    for b in range(10):
        tr, te = f != b, f == b
        mu, sd = F[tr].mean(0), F[tr].std(0, unbiased=False) + 1e-6
        A, B = (F[tr] - mu) / sd, (F[te] - mu) / sd
        if cw is not None:
            A, B = A * cw, B * cw
        _, _, Vt = torch.linalg.svd(A - A.mean(0), full_matrices=False)
        Vk = Vt[:min(L, Vt.shape[0])].T
        Atr, Ate = A @ Vk, B @ Vk
        k = Vk.shape[1]
        W = torch.linalg.solve(Atr.T @ Atr + LAM * len(Atr) * torch.eye(k, dtype=torch.float64, device=DEV), Atr.T @ (Y[tr] - Y[tr].mean(0)))
        P[te] = Ate @ W
    return P.cpu().numpy()
from cogmodels import models as M

QUANT = "results/full_36318/behaviour_quantities_flow.npz"


def resid(v, cols):
    C = np.c_[np.ones(len(v)), np.column_stack(cols)]
    b, *_ = np.linalg.lstsq(C, v, rcond=None)
    return v - C @ b


def corr(a, b):
    a = a - a.mean(); b = b - b.mean()
    d = np.sqrt((a @ a) * (b @ b))
    return float(a @ b / d) if d > 0 else 0.0


def dersimonian_laird(d, v):
    w = 1 / v
    mu_f = np.sum(w * d) / np.sum(w)
    Qs = np.sum(w * (d - mu_f) ** 2); k = len(d)
    tau2 = max(0.0, (Qs - (k - 1)) / (np.sum(w) - np.sum(w ** 2) / np.sum(w)))
    ws = 1 / (v + tau2)
    mu = np.sum(ws * d) / np.sum(ws); se = np.sqrt(1 / np.sum(ws))
    tcrit = stats.t.ppf(0.975, k - 2) if k > 2 else 1.96
    return {"mu": float(mu), "ci": [float(mu - 1.96 * se), float(mu + 1.96 * se)], "se": float(se), "tau2": float(tau2),
            "pi": [float(mu - tcrit * np.sqrt(tau2 + se ** 2)), float(mu + tcrit * np.sqrt(tau2 + se ** 2))],
            "z": float(mu / se), "p": float(2 * stats.norm.sf(abs(mu / se))), "Q": float(Qs), "I2": float(max(0, (Qs - (k - 1)) / Qs)) if Qs > 0 else 0.0}


def prevalence(real, nulls, n_perm=2000000, alpha=0.05, seed=0):
    """real (N,), nulls list of arrays (N x M_i). Minimum-statistic permutation test (Allefeld et al. 2016)."""
    rng = np.random.default_rng(seed)
    N = len(real)
    m_real = float(np.min(real))
    # each participant's first-level null includes the observed labelling (Phipson & Smyth 2010)
    full = [np.r_[nl, rv] for nl, rv in zip(nulls, real)]
    mins = np.full(n_perm, np.inf)
    for nl in full:
        mins = np.minimum(mins, nl[rng.integers(len(nl), size=n_perm)])
    p_global = (1 + np.sum(mins >= m_real)) / (n_perm + 1)
    # prevalence bound: largest gamma0 such that p_global^(1/N)-based p for H0 gamma <= gamma0 is < alpha
    p_uncorr = p_global
    gammas = np.linspace(0, 1, 1001)
    p_prev = np.array([((1 - g) * p_uncorr ** (1 / N) + g) ** N for g in gammas])
    ok = gammas[p_prev <= alpha]
    gamma0 = float(ok.max()) if len(ok) else 0.0
    return {"min_r": m_real, "p_global": float(p_global), "gamma0": gamma0,
            "p_majority": float(((1 - 0.5) * p_uncorr ** (1 / N) + 0.5) ** N)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--decoders", nargs="+", required=True)
    ap.add_argument("--model", default="RWD")
    ap.add_argument("--out", default="results/group_stats")
    a = ap.parse_args()
    Q = np.load(QUANT)
    model = {**M.MODELS, **M.EXTRA_MODELS}[a.model]
    Z = torch.tensor(np.load(f"results/map_{a.model}.npy")).float()
    probs = M.reward_schedule(480)
    nS = Z.shape[0]
    full_q = []
    for i in range(nS):
        with torch.no_grad():
            full_q.append(M.run(model, Z[i][None], probs, torch.tensor(Q["choice"][i])[None], torch.tensor(Q["reward"][i])[None],
                                torch.tensor(Q["missing"][i])[None], record=True)["q"]["pe"][0].numpy())

    def qtil(j, tri):
        c, r = Q["choice"][j][tri], Q["reward"][j][tri]
        cols = [r] + [(c == k).astype(float) for k in range(3)] + [(c == k).astype(float) * r for k in range(3)] + \
               [(tri / 480.0) ** p for p in (1, 2, 3)]
        v = resid(full_q[j][tri].astype(float), cols)
        return (v - v.mean()) / (v.std() + 1e-9)

    out = {"decoders": {}, "pairs": {}}
    D = {}
    for dec in a.decoders:
        R, NL, d, v, ZN = [], [], [], [], []
        for i in range(nS):
            fz = np.load(Path(f"results/features/{dec}") / f"sub-{i + 1:03d}.npz")
            F, colw, tri = fz["F"].astype(np.float64), fz["colw"].astype(np.float64), fz["tri"]
            Y = np.column_stack([qtil(j, tri) for j in range(nS)])           # own quantity + all others' (exact null)
            P = cv_predict_gpu(F, Y, 64, colw=colw)                       # per-target ridge: batching is exact
            rr = [corr(P[:, j], Y[:, j]) for j in range(nS)]
            r = rr[i]
            nl = np.array([rr[j] for j in range(nS) if j != i])
            R.append(r); NL.append(nl)
            zn = np.arctanh(np.clip(nl, -0.999, 0.999))
            d.append(float(np.arctanh(np.clip(r, -0.999, 0.999)) - zn.mean()))
            v.append(float(zn.var(ddof=1) * (1 + 1 / len(zn))))     # var(z_real) + var(mean of nulls)
            ZN.append(zn)
        d, v = np.array(d), np.array(v)
        meta = dersimonian_laird(d, v)
        prev = prevalence(np.array(R) - np.array([n.mean() for n in NL]), [n - n.mean() for n in NL])
        indiv = int(np.sum(np.array(R) > np.array([np.quantile(n, 0.95) for n in NL])))
        D[dec] = (d, v, ZN)
        out["decoders"][dec] = {"r": R, "d": d.tolist(), "v": v.tolist(), "null_z": [z.tolist() for z in ZN], "meta": meta, "prevalence": prev,
                                "n_individually_above_95pct_null": indiv,
                                "rho_eff_pooled": float(np.tanh(meta["mu"])), "rho_eff_ci": [float(np.tanh(x)) for x in meta["ci"]]}
        print(dec, "pooled z %.3f [%.3f, %.3f] p=%.1e tau2=%.4f | prevalence gamma0=%.2f p_global=%.1e | indiv %d/23" % (
            meta["mu"], *meta["ci"], meta["p"], meta["tau2"], prev["gamma0"], prev["p_global"], indiv), flush=True)
    names = list(D)
    for x in range(len(names)):
        for y in range(x + 1, len(names)):
            A, B = names[x], names[y]
            dd = D[A][0] - D[B][0]                                   # same participants, trials and null pairings:
            vv = np.array([np.var(za - zb, ddof=1) * (1 + 1 / len(za)) for za, zb in zip(D[A][2], D[B][2])])  # paired null variance
            out["pairs"][f"{A}-{B}"] = dersimonian_laird(dd, vv)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    Path(a.out, "results.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
