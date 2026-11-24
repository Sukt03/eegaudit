"""Protocol audit E: are the two yield currencies (Wolpaw bits for an N-way image target, posterior-entropy reduction for
a continuous learning-rate target) the same quantity, and does the cross-target asymmetry follow from target entropy alone?

Both currencies are read as I(target; recording) = H(target) - E[H(target | recording)]:
  discrete   : log2 N - H(Y | Yhat) >= log2 N - h2(Pe) - Pe log2(N-1) = Wolpaw I(P,N)   (Fano; equality iff errors uniform)
  continuous : H(theta) - E[H(theta | data)] >= 0.5 log2(var_prior / E[var_post])      (Gaussian = max entropy; Jensen)
Simulation (the reviewer's falsifier). Two synthetic covariates with MATCHED per-trial mutual information m:
  A. categorical: Y_t ~ U{1..N}, X_t = d' e_{Y_t} + eps in R^N (fresh target every trial); I(Y;X) exact by Monte Carlo,
     top-1 P, Wolpaw(P,N), soft-posterior bits; also a clustered-confusion variant where errors are not uniform.
  B. continuous (RW story): theta = logit(alpha) ~ N(0, 1.5^2), 3-arm restless bandit, 480 trials, softmax choices,
     per-trial read-out z_t = rho * pe~_t(theta) + sqrt(1-rho^2) eps_t (pe residualised on outcome, z-scored), exact 1-D
     grid posterior; information from behaviour, from EEG given behaviour, and from EEG alone, exact entropy and the
     Gaussian-variance estimator the paper uses.
Then: trials asymmetry at matched m against the closed form that uses only the entropy structure, and normalised
efficiency (bits / available bits) across coupling, N and trial count.
Writes results/protocol_audit/E_comparability.json. Read-only elsewhere.
"""
import json, time, sys
from pathlib import Path
import numpy as np

OUT = Path("/path/to/eegaudit/results/protocol_audit"); OUT.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(0)
SMOKE = "--smoke" in sys.argv
N_SUB = 40 if SMOKE else 200           # synthetic learners
T_MAX = 480
GRID = np.linspace(-6, 6, 121 if SMOKE else 241)   # theta = logit(alpha)
PRIOR_SD = 1.5                                      # paper's RWD prior on logit_alpha
lp_prior = -0.5 * (GRID / PRIOR_SD) ** 2; lp_prior -= np.log(np.sum(np.exp(lp_prior)))
H_PRIOR_GRID = -np.sum(np.exp(lp_prior) * lp_prior) / np.log(2)
BETA, DECAY = 5.0, 1 / (1 + np.exp(2.0))            # fixed inverse temperature and decay (paper's prior means)
RHO_CEIL = 0.9                                      # the paper's ceiling coupling
RHOS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.9, 0.97]
TS = [60, 120, 240, 480]


def h2(p):
    p = np.clip(p, 1e-12, 1 - 1e-12); return float(-(p * np.log2(p) + (1 - p) * np.log2(1 - p)))


def wolpaw(P, N):
    P = min(max(P, 1e-9), 1 - 1e-9)
    return float(np.log2(N) + P * np.log2(P) + (1 - P) * np.log2((1 - P) / (N - 1)))


# ----------------------------------------------------------------------------- A. categorical channel
def categorical(dp, N, n_mc=4000 if SMOKE else 20000, clustered=False):
    """X = dp * e_Y + eps (+ cluster bump if clustered). Returns exact I(Y;X), top-1 P, Wolpaw, I(Y;Yhat) plug-in."""
    y = rng.integers(0, N, n_mc)
    X = rng.standard_normal((n_mc, N)); X[np.arange(n_mc), y] += dp
    if clustered:   # candidates in groups of 10 share half the signal: errors concentrate inside the group
        g = y // 10
        for k in range(N // 10):
            X[g == k, k * 10:(k + 1) * 10] += 0.5 * dp
    # exact posterior over candidates: p(y|x) proportional to exp(dp * x_y) (clustered: likelihood known too)
    if clustered:
        ll = dp * X.copy()
        G = N // 10
        grp = ll.reshape(n_mc, G, 10).sum(2, keepdims=True) * 0.5
        ll = (ll.reshape(n_mc, G, 10) + grp).reshape(n_mc, N)
        # exact: log p(x|y) = dp*x_y + 0.5*dp*sum_{j in grp(y)} x_j  (up to const) -> handled above
    else:
        ll = dp * X
    ll -= ll.max(1, keepdims=True); post = np.exp(ll); post /= post.sum(1, keepdims=True)
    Hpost = -np.sum(post * np.log2(np.clip(post, 1e-300, None)), 1).mean()
    I_exact = float(np.log2(N) - Hpost)
    yhat = post.argmax(1); P = float((yhat == y).mean())
    C = np.zeros((N, N)); np.add.at(C, (y, yhat), 1); pj = C / n_mc
    py, pyh = pj.sum(1, keepdims=True), pj.sum(0, keepdims=True)
    nz = pj > 0
    I_hard = float((pj[nz] * np.log2(pj[nz] / (py @ pyh)[nz])).sum())
    I_hard_mm = I_hard + (nz.sum() - N - (pyh > 0).sum() + 1) / (2 * n_mc * np.log(2))   # Miller-Madow
    return dict(dp=dp, N=N, I_exact=I_exact, top1=P, wolpaw=wolpaw(P, N), I_hard_plugin=I_hard, I_hard_mm=float(I_hard_mm))


def dp_for_bits(m, N):
    lo, hi = 0.0, 12.0
    for _ in range(22):
        mid = 0.5 * (lo + hi)
        if categorical(mid, N, n_mc=2000 if SMOKE else 6000)["I_exact"] < m: lo = mid
        else: hi = mid
    return 0.5 * (lo + hi)


# ----------------------------------------------------------------------------- B. RW story, exact 1-D posterior
def sim_task(T):
    p = np.empty((T, 3)); p[0] = rng.uniform(0.2, 0.8, 3)
    for t in range(1, T):
        p[t] = np.clip(p[t - 1] + rng.normal(0, 0.03, 3), 0.05, 0.95)
    return p


def run_learner(theta, p):
    T = len(p); a = 1 / (1 + np.exp(-theta)); Q = np.full(3, 0.5); ch = np.empty(T, int); rw = np.empty(T)
    for t in range(T):
        logits = BETA * Q; pr = np.exp(logits - logits.max()); pr /= pr.sum()
        c = rng.choice(3, p=pr); r = float(rng.random() < p[t, c])
        ch[t], rw[t] = c, r
        pe = r - Q[c]; Q[c] += a * pe
        oth = np.arange(3) != c; Q[oth] += DECAY * (0.5 - Q[oth])
    return ch, rw


def replay(ch, rw):
    """For every grid theta: choice log-likelihood per trial (T,G) and residualised, z-scored PE series (T,G)."""
    T = len(ch); G = len(GRID); a = 1 / (1 + np.exp(-GRID))
    Q = np.full((G, 3), 0.5); ll = np.empty((T, G)); pe = np.empty((T, G))
    for t in range(T):
        logits = BETA * Q; logits -= logits.max(1, keepdims=True)
        lse = np.log(np.exp(logits).sum(1)); ll[t] = logits[:, ch[t]] - lse
        e = rw[t] - Q[:, ch[t]]; pe[t] = e
        Q[:, ch[t]] += a * e
        oth = np.ones(3, bool); oth[ch[t]] = False
        Q[:, oth] += DECAY * (0.5 - Q[:, oth])
    # residualise on outcome within learner (as the paper does), z-score within learner
    for col in (0.0, 1.0):
        msk = rw == col
        if msk.sum() > 1: pe[msk] -= pe[msk].mean(0, keepdims=True)
    pe = (pe - pe.mean(0, keepdims=True)) / (pe.std(0, keepdims=True) + 1e-9)
    return ll, pe


def post_stats(logpost):
    lp = logpost - logpost.max(); p = np.exp(lp); p /= p.sum()
    mean = (p * GRID).sum(); var = (p * (GRID - mean) ** 2).sum()
    H = -(p * np.log2(np.clip(p, 1e-300, None))).sum()
    return var, H


def continuous_bank():
    """Simulate learners once; return per-learner choice-ll (T,G), pe~ (T,G), true theta index on grid."""
    bank = []
    for s in range(N_SUB):
        theta = rng.normal(0, PRIOR_SD); p = sim_task(T_MAX); ch, rw = run_learner(theta, p)
        ll, pe = replay(ch, rw); j = int(np.argmin(np.abs(GRID - theta)))
        bank.append((ll, pe, j))
    return bank


def continuous_info(bank, rho, T, n_eeg_draws=1):
    """Exact (grid) information about theta from behaviour, behaviour+EEG, EEG alone, on T trials; plus the
    Gaussian-variance estimator the paper uses (0.5 log2 var_prior / mean var_post) and mean per-trial I(theta; z_t)."""
    r2 = 1 - rho ** 2
    H_b, H_j, H_e, V_b, V_j, V_e = [], [], [], [], [], []
    for ll, pe, j in bank:
        lb = lp_prior + ll[:T].sum(0)
        for _ in range(n_eeg_draws):
            z = rho * pe[:T, j] + np.sqrt(r2) * rng.standard_normal(T)
            le = (-0.5 * (z[:, None] - rho * pe[:T]) ** 2 / r2).sum(0) if rho > 0 else np.zeros(len(GRID))
            vb, hb = post_stats(lb); vj, hj = post_stats(lb + le); ve, he = post_stats(lp_prior + le)
            H_b.append(hb); H_j.append(hj); H_e.append(he); V_b.append(vb); V_j.append(vj); V_e.append(ve)
    H_b, H_j, H_e = map(np.mean, (H_b, H_j, H_e)); V_b, V_j, V_e = map(np.mean, (V_b, V_j, V_e))
    vp = PRIOR_SD ** 2
    return dict(rho=rho, T=T,
                exact=dict(behaviour=H_PRIOR_GRID - H_b, eeg_given_behaviour=H_b - H_j, eeg_alone=H_PRIOR_GRID - H_e),
                gauss=dict(behaviour=0.5 * np.log2(vp / V_b), eeg_given_behaviour=0.5 * np.log2(V_b / V_j),
                           eeg_alone=0.5 * np.log2(vp / V_e)))


def per_trial_mi(bank, rho, n_z=200 if SMOKE else 400):
    """mean_t I(theta; z_t): single-trial read-out information about theta under the prior (grid + MC over z)."""
    r2 = 1 - rho ** 2; prior = np.exp(lp_prior); vals = []
    for ll, pe, j in bank[: max(20, N_SUB // 5)]:
        ts = rng.choice(T_MAX, 24, replace=False)
        for t in ts:
            th = rng.choice(len(GRID), n_z, p=prior)
            z = rho * pe[t, th] + np.sqrt(r2) * rng.standard_normal(n_z)
            lpz = -0.5 * (z[:, None] - rho * pe[t][None, :]) ** 2 / r2          # (n_z, G)
            lpz_true = lpz[np.arange(n_z), th]
            lmarg = np.log(np.sum(np.exp(lpz) * prior[None, :], 1))
            vals.append((lpz_true - lmarg).mean() / np.log(2))
    return float(np.mean(vals))


def rho_for_bits(bank, m, rho_max=0.995):
    lo, hi = 0.0, rho_max
    if per_trial_mi(bank, rho_max) < m: return None
    for _ in range(16):
        mid = 0.5 * (lo + hi)
        if per_trial_mi(bank, mid) < m: lo = mid
        else: hi = mid
    return 0.5 * (lo + hi)


def main():
    t0 = time.time(); R = {"smoke": SMOKE, "n_sub": N_SUB, "grid": len(GRID), "prior_sd": PRIOR_SD, "beta": BETA, "decay": DECAY}
    # --- A1. Wolpaw vs exact MI on the symmetric channel and the clustered channel
    R["categorical_estimators"] = []
    for N in [2, 10, 50, 200, 1000]:
        for dp in [0.5, 1.0, 2.0, 3.0, 4.0]:
            if N == 1000 and SMOKE: continue
            R["categorical_estimators"].append(categorical(dp, N))
    R["categorical_clustered"] = [categorical(dp, 200, clustered=True) for dp in [1.0, 2.0, 3.0]]
    print("A1 done", round(time.time() - t0), "s", flush=True)

    # --- B1. RW bank and the information curves
    bank = continuous_bank(); print("bank done", round(time.time() - t0), "s", flush=True)
    R["continuous_curves"] = [continuous_info(bank, rho, T) for rho in [0.0] + RHOS for T in TS]
    R["continuous_per_trial_mi"] = {str(rho): per_trial_mi(bank, rho) for rho in RHOS}
    print("B1 done", round(time.time() - t0), "s", flush=True)

    # --- C. matched per-trial MI
    m_max = R["continuous_per_trial_mi"][str(RHOS[-1])]
    ms = [m for m in [0.002, 0.005, 0.01, 0.02, 0.05, 0.1] if m < m_max]
    R["matched"] = []
    for m in ms:
        rho = rho_for_bits(bank, m)
        row = dict(m=m, rho=rho, categorical={}, continuous={})
        for N in [10, 200, 1000]:
            dp = dp_for_bits(m, N); c = categorical(dp, N)
            row["categorical"][str(N)] = dict(dp=dp, I_exact=c["I_exact"], top1=c["top1"], wolpaw=c["wolpaw"],
                                              eff_exact=c["I_exact"] / np.log2(N), eff_wolpaw=c["wolpaw"] / np.log2(N),
                                              bits_480=480 * c["I_exact"])
        for T in TS:
            ci = continuous_info(bank, rho, T); cc = continuous_info(bank, RHO_CEIL, T); c0 = continuous_info(bank, 0.0, T)
            snr = 2 ** (2 * m) - 1
            row["continuous"][str(T)] = dict(
                eeg_alone_bits=ci["exact"]["eeg_alone"], eeg_given_beh_bits=ci["exact"]["eeg_given_behaviour"],
                behaviour_bits=c0["exact"]["behaviour"],
                ceiling_eeg_alone=cc["exact"]["eeg_alone"], ceiling_eeg_given_beh=cc["exact"]["eeg_given_behaviour"],
                eff_alone=ci["exact"]["eeg_alone"] / cc["exact"]["eeg_alone"],
                eff_given_beh=ci["exact"]["eeg_given_behaviour"] / cc["exact"]["eeg_given_behaviour"],
                closed_form_linear_gauss=0.5 * np.log2(1 + T * snr),           # entropy-structure prediction
                upper_bound_T_m=T * m)
        R["matched"].append(row); print("matched m", m, "rho", rho, round(time.time() - t0), "s", flush=True)

    # --- D. closed-form regime map for the continuous efficiency: eff = log2(1+T s)/log2(1+T s_c)
    R["closed_form_regimes"] = []
    for sc in [0.01, 0.1, 1.0]:
        for frac in [0.05, 0.2, 0.5]:
            s = frac * sc
            R["closed_form_regimes"].append(dict(snr_ceiling=sc, snr_over_ceiling=frac,
                                                 eff_by_T={str(T): float(np.log2(1 + T * s) / np.log2(1 + T * sc)) for T in [10, 100, 480, 5000, 50000]}))
    R["runtime_s"] = time.time() - t0
    (OUT / ("E_comparability_smoke.json" if SMOKE else "E_comparability.json")).write_text(json.dumps(R, indent=1, default=float))
    print(json.dumps({k: R[k] for k in ["continuous_per_trial_mi", "categorical_clustered"]}, indent=1, default=float))
    for row in R["matched"]:
        print(f"m={row['m']:.3f} rho={row['rho']}", {N: round(v['eff_exact'], 4) for N, v in row["categorical"].items()},
              {T: (round(v['eeg_alone_bits'], 3), round(v['eff_alone'], 3), round(v['eff_given_beh'], 3), round(v['closed_form_linear_gauss'], 3)) for T, v in row["continuous"].items()})
    print("done", round(R["runtime_s"]), "s")


if __name__ == "__main__":
    main()
