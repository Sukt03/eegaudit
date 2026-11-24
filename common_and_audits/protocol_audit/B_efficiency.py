"""Protocol audit B: how robust is the headline "normalised efficiency ~18% for both targets"?

Visual target  : ATM-S native retrieval (cosine, 200-way, 80-rep test average) on THINGS-EEG2, 10 subjects.
                 efficiency = Wolpaw bits(top-1, N) / log2 N.
Learning target: feedback-EEG increment in posterior bits about the learning rate (RWD, 480 trials) / ceiling increment,
                 bits = -0.5 log2(1 - c), c = contraction, read off the exact joint curve c(rho) at the decoder's rho_eff.

(1) denominator: N in {25, 50, 100, 200} random candidate subsets (20 draws); Wolpaw and an InfoNCE lower bound.
(2) learning ceiling: rho=0.9 (as in the draft) vs Gaussian-form extrapolation to rho = 0.95, 0.99, 1; baseline joint
    rho=0 vs separate behaviour-only block; per decoder; exact vs flow posterior.
(3) sampling: bootstrap over subjects (and over test images) for the visual number; overlap with the learning CI.
Writes results/protocol_audit/B_efficiency.json; the report is written by hand from the printout.
"""
import json
from pathlib import Path
import numpy as np
import torch

P = Path("/path/to/eegaudit")
EMB = Path("/path/to/atm/data/hf_probe/emb_eeg")
CLIP = Path("/path/to/atm/data/hf_probe/ViT-H-14_features_test.pt")
SUBJECTS = [f"sub-{i:02d}" for i in range(1, 11)]
NS = [25, 50, 100, 200]
N_DRAWS = 20
rng = np.random.default_rng(0)


def wolpaw_bits(Pacc, N):
    Pacc = min(max(Pacc, 1e-9), 1 - 1e-9)
    return float(np.log2(N) + Pacc * np.log2(Pacc) + (1 - Pacc) * np.log2((1 - Pacc) / (N - 1)))


def sim(pred, tgt):
    p = pred / np.linalg.norm(pred, axis=1, keepdims=True)
    t = tgt / np.linalg.norm(tgt, axis=1, keepdims=True)
    return p @ t.T


def top1_from_S(S):
    d = np.diag(S)
    return float(((S > d[:, None]).sum(1) == 0).mean())


def infonce_bits(S):
    """I >= log2 N - L_NCE; tau chosen on the same matrix (optimistic, but a bound for every tau)."""
    N = S.shape[0]
    best = -1.0
    for tau in [0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.2, 0.5, 1.0]:
        Z = S / tau
        Z = Z - Z.max(1, keepdims=True)
        logp = np.diag(Z) - np.log(np.exp(Z).sum(1))
        best = max(best, float(np.log2(N) + logp.mean() / np.log(2)))
    return best


def load(subject):
    x = torch.load(EMB / f"ATM_S_eeg_features_{subject}_test.pt", map_location="cpu", weights_only=False).float().numpy()
    y = torch.load(CLIP, map_location="cpu", weights_only=False)["img_features"].float().numpy()
    return x, y


# ------------------------------------------------------------------ (1) visual target vs denominator
visual = {}
S_by_sub = {}
for s in SUBJECTS:
    x, y = load(s)
    S = sim(x, y)
    S_by_sub[s] = S
    visual[s] = {}
    for N in NS:
        draws = [np.arange(200)] if N == 200 else [np.sort(rng.choice(200, N, replace=False)) for _ in range(N_DRAWS)]
        eff_w, eff_n, acc = [], [], []
        for idx in draws:
            Ss = S[np.ix_(idx, idx)]
            a = top1_from_S(Ss)
            acc.append(a)
            eff_w.append(wolpaw_bits(a, N) / np.log2(N))
            eff_n.append(infonce_bits(Ss) / np.log2(N))
        visual[s][N] = dict(top1=float(np.mean(acc)), bits_wolpaw=float(np.mean(eff_w) * np.log2(N)),
                            eff_wolpaw=float(np.mean(eff_w)), eff_wolpaw_sd_draws=float(np.std(eff_w)),
                            eff_infonce=float(np.mean(eff_n)))

print("\n(1) VISUAL: efficiency vs candidate-set size N (mean over 10 subjects; sd across subjects)")
print(f"{'N':>4} {'log2N':>6} {'top1':>6} {'bits':>6} {'effW':>7} {'sd_subj':>8} {'range':>14} {'effNCE':>7}")
vis_summary = {}
for N in NS:
    e = np.array([visual[s][N]["eff_wolpaw"] for s in SUBJECTS])
    en = np.array([visual[s][N]["eff_infonce"] for s in SUBJECTS])
    t1 = np.mean([visual[s][N]["top1"] for s in SUBJECTS])
    b = np.mean([visual[s][N]["bits_wolpaw"] for s in SUBJECTS])
    vis_summary[N] = dict(top1=float(t1), bits=float(b), eff_wolpaw_mean=float(e.mean()), eff_wolpaw_sd=float(e.std(ddof=1)),
                          eff_wolpaw_min=float(e.min()), eff_wolpaw_max=float(e.max()), eff_infonce_mean=float(en.mean()))
    print(f"{N:>4} {np.log2(N):6.2f} {t1:6.3f} {b:6.3f} {e.mean():7.3f} {e.std(ddof=1):8.3f} [{e.min():.3f},{e.max():.3f}] {en.mean():7.3f}")

# ------------------------------------------------------------------ (3) visual sampling: bootstraps
eff200 = np.array([visual[s][200]["eff_wolpaw"] for s in SUBJECTS])
B = 10000
boot_subj = np.array([rng.choice(eff200, 10, replace=True).mean() for _ in range(B)])
ci_subj = np.percentile(boot_subj, [2.5, 97.5])
# per-subject bootstrap over the 200 test images (resample query rows; the candidate set stays the full 200)
img_boot = {}
for s in SUBJECTS:
    S = S_by_sub[s]
    d = np.diag(S)
    correct = ((S > d[:, None]).sum(1) == 0).astype(float)
    bs = [wolpaw_bits(rng.choice(correct, 200, replace=True).mean(), 200) / np.log2(200) for _ in range(2000)]
    img_boot[s] = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
# two-level: resample subjects, then images within subject
two_level = []
corrects = {s: ((S_by_sub[s] > np.diag(S_by_sub[s])[:, None]).sum(1) == 0).astype(float) for s in SUBJECTS}
for _ in range(B):
    subs = rng.choice(SUBJECTS, 10, replace=True)
    two_level.append(np.mean([wolpaw_bits(rng.choice(corrects[s], 200, replace=True).mean(), 200) / np.log2(200) for s in subs]))
two_level = np.array(two_level)
ci_two = np.percentile(two_level, [2.5, 97.5])
print("\n(3) VISUAL sampling: mean eff = %.3f; bootstrap over subjects 95%% CI [%.3f, %.3f]; subjects+images [%.3f, %.3f]"
      % (eff200.mean(), *ci_subj, *ci_two))
print("    per-subject eff (N=200):", {s: round(float(v), 3) for s, v in zip(SUBJECTS, eff200)})
print("    per-subject image-bootstrap CI:", {s: [round(a, 3) for a in img_boot[s]] for s in SUBJECTS})

# ------------------------------------------------------------------ (2) learning target: ceiling / baseline / decoder
C = json.load(open(P / "results/ceiling_RWD_v2/results.json"))
W = json.load(open(P / "results/worth_ceiling_RWD_v2/results.json"))
rhos = np.array(sorted(float(r) for r in C["joint"]))
PAR = "logit_alpha"


def bits(c):
    c = min(max(float(c), 0.0), 1 - 1e-12)
    return -0.5 * np.log2(1 - c)


def curve(kind="exact"):
    return np.array([C["joint"][("0.0" if r == 0 else f"{r:g}")][kind][PAR]["contraction"] for r in rhos])


def s_of(rho):
    return rho ** 2 / (1 - rho ** 2)


def fit_gauss(c):
    """Gaussian/linear-information form: 1/(1-c) = A + B*s(rho). Least squares on 1/(1-c) vs s."""
    yv = 1 / (1 - c)
    X = np.c_[np.ones_like(rhos), s_of(rhos)]
    coef, *_ = np.linalg.lstsq(X, yv, rcond=None)
    pred = X @ coef
    return coef, pred


c_exact, c_flow = curve("exact"), curve("flow")
coef, pred = fit_gauss(c_exact)
c_fit = 1 - 1 / pred
print("\n(2) LEARNING: exact joint curve for the learning rate, and the Gaussian-form fit 1/(1-c) = A + B s, s = rho^2/(1-rho^2)")
print("    rho   :", rhos)
print("    c exact:", np.round(c_exact, 4))
print("    c fit  :", np.round(c_fit, 4), "  A=%.2f B=%.2f" % tuple(coef))
print("    bits exact:", np.round([bits(c) for c in c_exact], 3))
print("    bits fit  :", np.round([bits(c) for c in c_fit], 3))


def c_at(rho, c=c_exact, mode="interp"):
    if mode == "interp":
        return float(np.interp(rho, rhos, c))
    A, Bc = fit_gauss(c)[0]
    return 1 - 1 / (A + Bc * s_of(rho))


base_joint0 = c_exact[0]
base_beh = C["behaviour"]["480"]["exact"][PAR]["contraction"]
print("    baseline contraction: joint rho=0 = %.4f (%.3f bits); separate behaviour block = %.4f (%.3f bits); difference %.3f bits"
      % (base_joint0, bits(base_joint0), base_beh, bits(base_beh), bits(base_beh) - bits(base_joint0)))

learn = {}
ceil_rhos = [0.9, 0.95, 0.99, 0.999]
print("\n    ceiling increment (bits over joint rho=0) at ceiling rho: ",
      {r: round(bits(c_at(r, mode="interp" if r <= 0.9 else "fit")) - bits(base_joint0), 3) for r in ceil_rhos}, " rho=1 -> infinite under the Gaussian form")
print(f"\n    {'decoder':10s} {'rho_eff':>7} {'inc bits':>8} | eff%% vs ceiling rho: " + " ".join(f"{r:>7}" for r in ceil_rhos) + " | beh-base eff%  | flow eff%")
for dec, d in W["decoders"].items():
    rho = d["rho_eff"]
    lo, hi = d["rho_ci"]
    inc = bits(c_at(rho)) - bits(base_joint0)
    row = dict(rho_eff=rho, rho_ci=[lo, hi], inc_bits=inc, eff={})
    effs = []
    for r in ceil_rhos:
        cc = c_at(r, mode="interp" if r <= 0.9 else "fit")
        ceil_inc = bits(cc) - bits(base_joint0)
        row["eff"][str(r)] = dict(point=100 * inc / ceil_inc,
                                  ci=[100 * (bits(c_at(lo)) - bits(base_joint0)) / ceil_inc, 100 * (bits(c_at(hi)) - bits(base_joint0)) / ceil_inc])
        effs.append(100 * inc / ceil_inc)
    # behaviour-block baseline: increment and ceiling both measured from the separate behaviour-only block
    inc_b = bits(c_at(rho)) - bits(base_beh)
    ceil_b = bits(c_at(0.9)) - bits(base_beh)
    row["eff_behaviour_baseline"] = 100 * inc_b / ceil_b
    # flow posterior (the amortised one), same recipe, ceiling 0.9
    inc_f = bits(c_at(rho, c_flow)) - bits(c_flow[0])
    ceil_f = bits(c_at(0.9, c_flow)) - bits(c_flow[0])
    row["eff_flow"] = 100 * inc_f / ceil_f
    # Gaussian-fit interpolation at rho_eff instead of linear interpolation
    row["eff_fit_interp"] = 100 * (bits(c_at(rho, mode="fit")) - bits(c_fit[0])) / (bits(c_at(0.9, mode="fit")) - bits(c_fit[0]))
    learn[dec] = row
    print(f"    {dec:10s} {rho:7.3f} {inc:8.3f} | " + " ".join(f"{e:7.1f}" for e in effs) + f" | {row['eff_behaviour_baseline']:13.1f} | {row['eff_flow']:8.1f}  (fit-interp {row['eff_fit_interp']:.1f})")

# analytic view: in the Gaussian form, eff(rho_eff; rho_c) = log(1 + s_eff B/A) / log(1 + s_c B/A)
A, Bc = coef
k = Bc / A
rho_e = W["decoders"]["ens"]["rho_eff"]
print("\n    Gaussian-form closed form: eff = ln(1 + k s_eff)/ln(1 + k s_c), k = B/A = %.3f, s_eff(ens) = %.3f" % (k, s_of(rho_e)))
for r in [0.7, 0.8, 0.9, 0.95, 0.99]:
    print(f"      ceiling rho={r}: eff = {100*np.log(1+k*s_of(rho_e))/np.log(1+k*s_of(r)):.1f}%")


# local Gaussian-form extrapolation anchored on the upper part of the curve (rho >= 0.5), where the fit is good
m = rhos >= 0.5
Xl = np.c_[np.ones(m.sum()), s_of(rhos[m])]
cl, *_ = np.linalg.lstsq(Xl, 1 / (1 - c_exact[m]), rcond=None)
print("    local fit (rho>=0.5): A=%.2f B=%.2f; predicted c at 0.5/0.7/0.9 = %s vs exact %s" % (cl[0], cl[1], np.round(1 - 1 / (Xl @ cl), 4), np.round(c_exact[m], 4)))
inc_ens = bits(c_at(rho_e)) - bits(base_joint0)
for r in [0.95, 0.99, 0.999]:
    cc = 1 - 1 / (cl[0] + cl[1] * s_of(r))
    print(f"      ens eff vs ceiling rho={r} (local extrapolation): {100 * inc_ens / (bits(cc) - bits(base_joint0)):.1f}%  (ceiling increment {bits(cc) - bits(base_joint0):.2f} bits)")

# ------------------------------------------------------------------ overlap of the two numbers
ens = learn["ens"]
ens_ci = ens["eff"]["0.9"]["ci"]
print("\n(3b) overlap: visual %.1f%% [%.1f, %.1f] (subject bootstrap) vs learning (ens, ceiling 0.9) %.1f%% [%.1f, %.1f] (coupling CI)"
      % (100 * eff200.mean(), 100 * ci_subj[0], 100 * ci_subj[1], ens["eff"]["0.9"]["point"], *ens_ci))
# fraction of bootstrap visual means that land inside the learning CI, and the difference distribution assuming the
# learning estimate ~ normal with sd from its CI
sd_l = (ens_ci[1] - ens_ci[0]) / (2 * 1.96) / 100
diff = boot_subj - (ens["eff"]["0.9"]["point"] / 100 + rng.standard_normal(B) * sd_l)
print("     difference visual - learning: mean %.3f, 95%% CI [%.3f, %.3f]  (a 12%% vs 25%% gap = 0.13)" % (diff.mean(), *np.percentile(diff, [2.5, 97.5])))

out = dict(visual_per_subject={s: {str(N): v for N, v in visual[s].items()} for s in SUBJECTS},
           visual_summary={str(N): v for N, v in vis_summary.items()},
           visual_eff200_mean=float(eff200.mean()), visual_ci_subjects=[float(a) for a in ci_subj], visual_ci_two_level=[float(a) for a in ci_two],
           visual_img_boot_ci=img_boot,
           learning=dict(rhos=rhos.tolist(), c_exact=c_exact.tolist(), c_flow=c_flow.tolist(), gauss_fit_A=float(A), gauss_fit_B=float(Bc),
                         c_fit=c_fit.tolist(), base_joint0=float(base_joint0), base_behaviour_block=float(base_beh), decoders=learn),
           diff_visual_minus_learning_ci=[float(a) for a in np.percentile(diff, [2.5, 97.5])])
Path(P / "results/protocol_audit").mkdir(parents=True, exist_ok=True)
(P / "results/protocol_audit/B_efficiency.json").write_text(json.dumps(out, indent=1))
print("\nwrote", P / "results/protocol_audit/B_efficiency.json")
