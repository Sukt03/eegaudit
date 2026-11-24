"""Shared pieces for the LEARNER smokes (L2-L4): same data, targets, read-out and statistics as group_stats.py.
Coupling = cross-validated PCA-64 ridge read-out of the target from the cached features, exact session-permutation null
(same EEG paired with every other participant's target on the same trials), null-corrected Fisher z, DerSimonian-Laird."""
import json, sys
from pathlib import Path
import numpy as np, torch
ROOT = Path("/path/to/eegaudit"); sys.path.insert(0, str(ROOT))
from belief_bottleneck import cv_predict
from group_stats import dersimonian_laird, prevalence, corr, resid
from cogmodels import models as M

Q = np.load(ROOT / "results/full_36318/behaviour_quantities_flow.npz")
MODEL = M.EXTRA_MODELS["RWD"]; Z = torch.tensor(np.load(ROOT / "results/map_RWD.npy")).float(); PROBS = M.reward_schedule(480); NS = Z.shape[0]
_PE = {}
def pe_full(i):
    if i not in _PE:
        with torch.no_grad():
            _PE[i] = M.run(MODEL, Z[i][None], PROBS, torch.tensor(Q["choice"][i])[None], torch.tensor(Q["reward"][i])[None],
                           torch.tensor(Q["missing"][i])[None], record=True)["q"]["pe"][0].numpy().astype(float)
    return _PE[i]
def nuisance(j, tri):
    c, r = Q["choice"][j][tri], Q["reward"][j][tri]
    return [r] + [(c == k).astype(float) for k in range(3)] + [(c == k).astype(float) * r for k in range(3)] + [(tri / 480.0) ** p for p in (1, 2, 3)]
def z(v): return (v - v.mean()) / (v.std() + 1e-9)
def target(name, j, tri):
    """Target series of participant j on trials tri. 'pe_resid' is the paper's read-out quantity (group_stats.qtil)."""
    pe, r = pe_full(j)[tri], Q["reward"][j][tri].astype(float)
    prev = np.where(tri > 0, Q["reward"][j][np.maximum(tri - 1, 0)].astype(float), 0.0)
    v = {"pe_resid": lambda: resid(pe, nuisance(j, tri)), "pe_signed": lambda: pe, "pe_abs": lambda: np.abs(pe),
         "pe_abs_resid": lambda: resid(np.abs(pe), nuisance(j, tri)), "valence": lambda: r, "prev_reward": lambda: prev}[name]()
    return z(v)
def load_feat(dec, s, cols=None):
    fz = np.load(ROOT / f"results/features/{dec}/sub-{s:03d}.npz")
    F, colw, tri = fz["F"].astype(np.float64), fz["colw"].astype(np.float64), fz["tri"]
    if cols is not None: F, colw = F[:, cols], colw[cols]
    return F, colw, tri
def coupling(dec, tname, cols=None, L=64):
    """group_stats.py statistics for one (features, target) pair over all participants."""
    R, NL, d, v, ZN = [], [], [], [], []
    for i in range(NS):
        F, colw, tri = load_feat(dec, i + 1, cols)
        Y = np.column_stack([target(tname, j, tri) for j in range(NS)])
        P = cv_predict(F, Y, "pca", L, colw=colw)
        rr = [corr(P[:, j], Y[:, j]) for j in range(NS)]
        r = rr[i]; nl = np.array([rr[j] for j in range(NS) if j != i])
        zn = np.arctanh(np.clip(nl, -0.999, 0.999))
        R.append(r); NL.append(nl); ZN.append(zn)
        d.append(float(np.arctanh(np.clip(r, -0.999, 0.999)) - zn.mean())); v.append(float(zn.var(ddof=1) * (1 + 1 / len(zn))))
    d, v = np.array(d), np.array(v); meta = dersimonian_laird(d, v)
    prev = prevalence(np.array(R) - np.array([n.mean() for n in NL]), [n - n.mean() for n in NL])
    indiv = int(np.sum(np.array(R) > np.array([np.quantile(n, 0.95) for n in NL])))
    return {"r": R, "r_mean": float(np.mean(R)), "null_r_mean": float(np.mean([n.mean() for n in NL])), "d": d.tolist(), "v": v.tolist(), "meta": meta,
            "rho_eff_pooled": float(np.tanh(meta["mu"])), "rho_eff_ci": [float(np.tanh(x)) for x in meta["ci"]], "prevalence": prev,
            "n_individually_above_95pct_null": indiv, "_ZN": ZN}
def paired(A, B):
    dd = np.array(A["d"]) - np.array(B["d"])
    vv = np.array([np.var(za - zb, ddof=1) * (1 + 1 / len(za)) for za, zb in zip(A["_ZN"], B["_ZN"])])
    return dersimonian_laird(dd, vv)
# exact ceiling curve (logit_alpha contraction vs rho) and the EEG increment in bits, as in r4_learning_ceiling.py / r5
def make_incr(curve_json):
    C = json.load(open(curve_json)); keys = sorted(C["joint"], key=float)
    rhos = [float(k) for k in keys]; cs = [C["joint"][k]["exact"]["logit_alpha"]["contraction"] for k in keys]
    bits = lambda c: -0.5 * np.log2(1 - c)
    def incr(rho):
        if rho <= 0.9: return float(bits(np.interp(rho, rhos, cs)) - bits(cs[0]))
        s = lambda r: r * r / (1 - r * r); A, B = 8.15, 9.38
        return float(0.5 * np.log2(A + B * s(rho)) - 0.5 * np.log2(A))
    return incr, max(rhos)
def strip(c): return {k: v for k, v in c.items() if not k.startswith("_")}
