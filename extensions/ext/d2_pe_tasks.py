"""D2.0 + D2.1 PE YIELD PER LEARNING TASK (B2/B3; rungs R3/R9). EXECUTOR-3, 21 Nov 2026.
Design = EXPERIMENT_TABLES_FINAL.md D2 + "## D2 REVIEW v1" binding changes (EXTENSIONS_LOOP_20261121.md). Six on-disk RL datasets:
  ds003458 (23, restless 3-arm), ds004595 (50, novelty 3-arm), ds007647 (39, doors, learnable/unlearnable blocks), ds004147 (12, cues; PILOT),
  ds003822 (25, PST affective RL1), ds003753 (24, PST affective RL2).
Target (per dataset, its own locked behavioural model at the participant's MAP; existing code paths reused):
  ds003458 / ds004595 = s13b_gauss_bits.targets_ds003458 / targets_ds004595 (RWD; residualised on outcome, chosen arm, arm x outcome, cubic time)
  ds007647 / ds004147 = cogmodels/doors.py, model = fit_<ds>.json best_model (RWD), map_<ds>_<model>.npy; residualised as
                        group_stats_twochoice.qtil_from (outcome, choice, choice x outcome, condition, condition x outcome, cubic time)
  ds003822 / ds003753 = cogmodels/pst.py, model by summed BIC over the cached maps (as group_stats_pst), behaviour from the cached
                        results/behaviour_<ds>.npz; residualised as group_stats_pst --resid full (outcome = positive/negative affective
                        feedback image, i.e. the valence category; chosen option, chosen x outcome, condition dummies, cubic time)
  Residualisation uses behavioural covariates only (no EEG), on the analysed trial subset, then z-scored (as X7). The log-score uses
  normal scores of nothing: the locked PE estimator below works on the z-scored residualised PE, exactly as pe_bottleneck_logscore.
Read-out (METHOD CHANGE 12): PRIMARY = in-fold PCA-64 + ridge (X7 pipeline: in-fold z-score, PCA on training rows, ridge LAM * n,
  10 contiguous folds); full width and the L curve (1..256, full) are secondary; matched-size sensitivity = first 150 analysed trials
  per participant (w0600, contiguous), the basis of between-dataset contrasts (unpaired bootstrap vs ds003458, Holm). Data: the
  cleaned feedback epochs results/eeg_measures[_<ds>]/sub-XXX_epochs.npz (50 Hz, -0.2..0.8 s), common channels of the dataset, trials
  = ~bad (and finite RewP for the two-choice / PST sets, as their group_stats scripts). Two windows: 'w0600' = 0-600 ms (PRIMARY,
  D2 REVIEW 2) and 'x7' = 150-600 ms (the X7 window; G2 anchor).
Score: locked PE log-score = held-out log2 Gaussian density gain over intercept-only, calibration nested on inner 9-fold read-outs of
  the training rows (pe_bottleneck_logscore.gain_rows; same algebra as m3_yield_map.pe_gain). Floors: target circularly shifted by
  n/4, n/2, 3n/4 (mean) and N(0,1) input (seed 1000 + participant); floor = max. Ceiling (D2 REVIEW 5, common for all datasets): ridge
  split-half = two ridges on interleaved halves of each training fold, r12 over all held-out rows, Spearman-Brown rel, -1/2 log2(1 - rel).
ds007647: learnable (cond 2) and unlearnable (cond 1) trial subsets scored separately (folds contiguous within condition; target
  residualised within the subset), paired L - U (D2 REVIEW 3; P2a validity check concerns trials-worth, not yield).
D2.0 comparability table (written before any fit): n with EEG, channels, sfreq, epoch window, trials (mean, range), ICA components removed,
  behaviour model, behaviour T; PST alignment check (epoch rewp length == behaviour T, trial indices in range, behaviour valid on the
  analysed trials) -> a dataset failing it is EXCLUDED on that technical ground (D2 PROPOSAL v2).
G2: ds003458 'x7' gain == results/standard/pe_bottleneck/sub-XXX.json ('eeg','pca','full') gain (|diff| < 1e-9) when the trial set is equal.
G3 (METHOD CHANGE 12): per dataset and floor type, |mean| <= 0.01, bootstrap CI includes 0, sign test p >= 0.05; participants with
  |floor| > 0.06 listed; else the dataset is labelled "floor not at chance". Ceiling with r12 <= 0 = "no usable ceiling" (never clipped).
  ds004147 = "pilot (n = 12)", descriptive, outside the Holm family (the other 5 datasets).
CPU exception (named, METHOD G1 checklist): scalar p-value lookups (t, binomial) via scipy on already-aggregated numbers; loading/IO.
Usage: python d2_pe_tasks.py --table --out DIR | --ds ds003458[,..] [--subjects 1,2] --out DIR | --summarize --out DIR
"""
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np
import torch

ROOT = Path("/path/to/eegaudit"); os.chdir(ROOT)
for p in (ROOT, ROOT / "experiments/loop", ROOT / "experiments/standard", ROOT / "experiments/agent_loop_20261120"):
    sys.path.insert(0, str(p))
from belief_bottleneck import LAM, folds
from group_stats import resid

DEV = "cuda"
DSETS = ["ds003458", "ds004595", "ds007647", "ds004147", "ds003822", "ds003753"]
WINS = {"w0600": (0.0, 0.60), "x7": (0.15, 0.60)}
LN2 = np.log(2)
MATCHED_T = 150
PILOT = {"ds004147"}


def gain_rows(y_tr, x_tr, y_te, x_te):
    """= pe_bottleneck_logscore.gain_rows (per-test-row log2 density gain, calibration on (x_tr, y_tr) only)."""
    D = np.column_stack([np.ones(len(x_tr)), x_tr]); c = np.linalg.lstsq(D, y_tr, rcond=None)[0]
    pt, pe = D @ c, c[0] + c[1] * x_te; rv = np.mean((y_tr - pt) ** 2); m = y_tr.mean(); vv = np.mean((y_tr - m) ** 2)
    return (np.log(vv / rv) + (y_te - m) ** 2 / vv - (y_te - pe) ** 2 / rv) / (2 * LN2)


def gain_t(y_tr, x_tr, y_te, x_te):
    """Torch/GPU batched gain_rows: leading dims batch (widths x columns); OLS calibration y = a + b x fit on the training rows only
    (closed form = np.linalg.lstsq with intercept), residual variance, intercept-only variance (ddof 0); per-test-row log2 density gain."""
    xm, ym = x_tr.mean(-1, keepdim=True), y_tr.mean(-1, keepdim=True)
    b = ((x_tr - xm) * (y_tr - ym)).mean(-1, keepdim=True) / ((x_tr - xm) ** 2).mean(-1, keepdim=True); a = ym - b * xm
    rv = ((y_tr - a - b * x_tr) ** 2).mean(-1, keepdim=True); vv = ((y_tr - ym) ** 2).mean(-1, keepdim=True)
    return (torch.log(vv / rv) + (y_te - ym) ** 2 / vv - (y_te - a - b * x_te) ** 2 / rv) / (2 * LN2)


def gnoise(shape, seed):
    g = torch.Generator(device=DEV); g.manual_seed(int(seed)); return torch.randn(tuple(shape), generator=g, device=DEV, dtype=torch.float64)


LS = [1, 2, 4, 8, 16, 32, 64, 128, 256, "full"]
PRIMARY_L = 64


def _ridge_L(F, Y, tr, te, Ls):
    """In-fold z-score, one SVD of the training rows, ridge (LAM * n) at every width in Ls (= pe_bottleneck_logscore.ridge_preds 'pca')."""
    A, B = F[tr], F[te]; mu, sd = A.mean(0), A.std(0, unbiased=False) + 1e-6; A = (A - mu) / sd; B = (B - mu) / sd
    Yc = Y[tr] - Y[tr].mean(0); out = {}
    Vt = torch.linalg.svd(A - A.mean(0), full_matrices=False)[2] if any(L != "full" for L in Ls) else None
    for L in Ls:
        if L == "full": Atr, Ate = A, B
        else: Vk = Vt[:min(L, Vt.shape[0])].T; Atr, Ate = A @ Vk, B @ Vk
        k = Atr.shape[1]
        out[L] = Ate @ torch.linalg.solve(Atr.T @ Atr + LAM * len(Atr) * torch.eye(k, dtype=torch.float64, device=DEV), Atr.T @ Yc)
    return out


def ridge_batch(F, Y, M, Ls):
    """ALL fits at once (rule 11): F (n, D), Y (n, J) float64 GPU; M (K, n) bool training masks. Per fit k: z-score on its training rows,
    SVD of the centred training block (rows outside the mask are zero), ridge (LAM * n_train_k) on the top-L scores; L = 'full' uses every
    component (= the primal full-width ridge, whose solution lies in the training row space). Returns (K, nL, n, J) predictions for ALL rows
    (callers pick each fit's held-out rows). Same algebra as pe_bottleneck_logscore.ridge_preds, batched over fits."""
    m = M.double(); cnt = m.sum(1); n, J = Y.shape
    if F.dim() == 3:                                                           # per-fit inputs (K, n, D), e.g. in-fold nuisance-regressed
        mu = (m[:, :, None] * F).sum(1) / cnt[:, None]; Bf = F - mu[:, None]
    else:
        mu = (m @ F) / cnt[:, None]; Bf = F[None] - mu[:, None]
    sd = ((Bf ** 2 * m[:, :, None]).sum(1) / cnt[:, None]).sqrt() + 1e-6; Bf = Bf / sd[:, None]; A = Bf * m[:, :, None]
    Ac = (A - (A.sum(1) / cnt[:, None])[:, None] * m[:, :, None])
    if Ac.shape[1] <= Ac.shape[2]:                                             # n <= D: right singular vectors from the n x n Gram (batched eigh,
        e, U = torch.linalg.eigh(Ac @ Ac.transpose(1, 2))                       # far faster than a batched SVD); zero-variance directions dropped
        e, U = e.flip(-1), U.flip(-1); Sv = e.clamp_min(0).sqrt(); ok = Sv > 1e-9 * Sv[:, :1]
        Vh = (U.transpose(1, 2) @ Ac) / torch.where(ok, Sv, torch.ones_like(Sv))[:, :, None] * ok[:, :, None]
    else:
        Vh = torch.linalg.svd(Ac, full_matrices=False)[2]
    del Ac                                                                     # Vh (K, r, D), r = min(n, D), rows sorted by singular value
    P = Bf @ Vh.transpose(1, 2); Ptr = P * m[:, :, None]; del Bf, A           # scores of all rows / training rows
    Yc = (Y[None] - ((m @ Y) / cnt[:, None])[:, None]) * m[:, :, None]
    out = []
    for L in Ls:
        k = Vh.shape[1] if L == "full" else min(L, Vh.shape[1])
        Z = Ptr[..., :k]; G = Z.transpose(1, 2) @ Z + LAM * cnt[:, None, None] * torch.eye(k, dtype=torch.float64, device=DEV)
        out.append(P[..., :k] @ torch.linalg.solve(G, Z.transpose(1, 2) @ Yc))
    return torch.stack(out, 1)


def nested_masks(n, embargo=0):
    """Outer 10 contiguous folds + 9 inner contiguous folds of each outer training set: (100, n) training masks, test index lists.
    embargo > 0 (METHOD G1 FIX c1): training rows within `embargo` rows of a held-out block are removed from that fit's training set
    (outer and inner); embargo = 0 is the original rule (identical masks)."""
    fo = folds(n); M, tests, trains = [], [], []
    def emb_mask(base, te):
        mk = base.copy()
        if embargo and len(te):
            lo, hi = te.min(), te.max(); mk[max(0, lo - embargo):lo] = False; mk[hi + 1:hi + 1 + embargo] = False
        return mk
    for b in range(10):
        te = np.where(fo == b)[0]; mk = emb_mask(fo != b, te); M.append(mk); tests.append(te); trains.append(np.where(mk)[0])
    for b in range(10):
        tr = trains[b]; fi = folds(len(tr), 9)
        for c in range(9):
            te = tr[fi == c]; base = np.zeros(n, bool); base[tr[fi != c]] = True; M.append(emb_mask(base, te)); tests.append(te)
    return torch.as_tensor(np.array(M), device=DEV), tests, trains


def pe_gain(F, Ys, Ls=("full",), transform=None, embargo=0):
    """Nested held-out density gain per row and column of Ys at every width; all 100 fits of one block in ONE batched GPU call;
    calibration on inner 9-fold read-outs of the training rows (gain_t, batched over widths x columns).
    transform(F, M) -> (K, n, D) optional per-fit input map fit on each fit's training rows only (e.g. ocular regression)."""
    F = torch.as_tensor(F, dtype=torch.float64, device=DEV); Y = torch.as_tensor(Ys, dtype=torch.float64, device=DEV); n = len(Y)
    M, tests, trains = nested_masks(n, embargo); Pall = ridge_batch(F if transform is None else transform(F, M), Y, M, Ls)    # (100, nL, n, J)
    sc = torch.full((len(Ls), n, Y.shape[1]), float("nan"), dtype=torch.float64, device=DEV)
    for b in range(10):
        tr = trains[b]; te = torch.as_tensor(tests[b], device=DEV); trg = torch.as_tensor(tr, device=DEV)
        inner = torch.zeros((len(Ls), len(tr), Y.shape[1]), dtype=torch.float64, device=DEV); pos = {int(r): i for i, r in enumerate(tr)}
        for c in range(9):
            k = 10 + 9 * b + c; idx = torch.as_tensor(tests[k], device=DEV)
            inner[:, torch.as_tensor([pos[int(r)] for r in tests[k]], device=DEV)] = Pall[k][:, idx]
        g = gain_t(Y[trg].T[None], inner.transpose(1, 2), Y[te].T[None], Pall[b][:, te].transpose(1, 2))
        sc[:, te] = g.transpose(1, 2)
    assert torch.isfinite(sc).all()
    sc = sc.cpu().numpy(); out = {L: sc[i] for i, L in enumerate(Ls)}
    return out if len(Ls) > 1 else out[Ls[0]]


def split_half_ceiling(F, y, L=PRIMARY_L):
    """Read-out determinacy: two read-outs (same width L) on interleaved halves of each training fold (20 fits, one batched call);
    r12 over all held-out rows; Spearman-Brown. METHOD CHANGE 12 (3): rel <= 0 -> 'no usable ceiling' (bits None), never clipped."""
    F = torch.as_tensor(F, dtype=torch.float64, device=DEV); Y = torch.as_tensor(y[:, None], dtype=torch.float64, device=DEV)
    n = len(y); fo = folds(n); M = []
    for b in range(10):
        tr = np.where(fo != b)[0]
        for h in range(2):
            mk = np.zeros(n, bool); mk[tr[h::2]] = True; M.append(mk)
    Pall = ridge_batch(F, Y, torch.as_tensor(np.array(M), device=DEV), [L])[:, 0, :, 0]     # (20, n)
    P = torch.zeros((n, 2), dtype=torch.float64, device=DEV)
    for b in range(10):
        te = torch.as_tensor(np.where(fo == b)[0], device=DEV)
        for h in range(2): P[te, h] = Pall[2 * b + h, te]
    r = float(torch.corrcoef(P.T)[0, 1]); rel = 2 * r / (1 + r)
    usable = bool(r > 0)
    return dict(L=L, r12=r, rel=float(rel), usable=usable, bits=float(-0.5 * np.log2(1 - min(rel, 0.999))) if usable else None,
                label=None if usable else "no usable ceiling")


def s13b_targets():
    src = open(ROOT / "experiments/loop/s13b_gauss_bits.py").read(); src = src[:src.index("DATASETS = {")]
    ns = {}; exec(compile(src, "s13b_head", "exec"), ns); return ns


def dataset(ds):
    """-> dict(qtil(label, tri) -> z-scored residualised PE, subs = participant labels with EEG, epoch dir, model, T, cond(label) or None)."""
    if ds in ("ds003458", "ds004595"):
        ns = s13b_targets(); qtil, subs, _, _ = ns["targets_" + ds](); E = ROOT / ("results/eeg_measures" if ds == "ds003458" else "results/eeg_measures_ds004595")
        subs = [s for s in subs if (E / f"sub-{s:03d}_epochs.npz").exists()]
        return dict(qtil=qtil, subs=subs, E=E, model="RWD", T=480 if ds == "ds003458" else None, cond=None, rewp_filter=False)
    if ds in ("ds007647", "ds004147"):
        from cogmodels import doors as Dm
        B = Dm.LOADERS[ds](f"data_raw/{ds}"); mname = json.loads((ROOT / f"results/fit_{ds}.json").read_text())["best_model"]
        z = torch.tensor(np.load(ROOT / f"results/map_{ds}_{mname}.npy")).float(); assert z.shape[0] == len(B["sub"])
        with torch.no_grad():
            PE = Dm.run(Dm.MODELS[mname], z, *Dm.tensors(B), record=True)["pe"].numpy()
        row = {int(s): k for k, s in enumerate(B["sub"])}; T = B["choice"].shape[1]
        def qtil(lab, tri):
            j = row[lab]; c, r, k = B["choice"][j][tri], B["reward"][j][tri], B["cond"][j][tri]
            cols = [r, c, c * r] + ([k == 2, (k == 2) * r] if len(np.unique(k)) > 1 else []) + [(tri / float(T)) ** p for p in (1, 2, 3)]
            v = resid(PE[j][tri].astype(float), [np.asarray(x, float) for x in cols]); return (v - v.mean()) / (v.std() + 1e-9)
        E = ROOT / f"results/eeg_measures_{ds}"; subs = [int(s) for s in B["sub"] if (E / f"sub-{int(s):03d}_epochs.npz").exists()]
        return dict(qtil=qtil, subs=subs, E=E, model=mname, T=T, cond=lambda lab: B["cond"][row[lab]], valid=lambda lab: B["valid"][row[lab]], rewp_filter=True)
    from cogmodels import pst as P
    B = dict(np.load(ROOT / f"results/behaviour_{ds}.npz")); cond, ch, rw, va = (torch.tensor(B[k]) for k in ("cond", "choice", "reward", "valid"))
    fits = {}
    for nm, m in P.MODELS.items():
        z = torch.tensor(np.load(ROOT / f"results/map_{ds}_{nm}.npy")).float(); ll = P.run(m, z, cond, ch, rw.float(), va)["loglik"]
        fits[nm] = (z, float((-2 * ll + len(m.param_names) * np.log(B["cond"].shape[1])).sum()))
    mname = min(fits, key=lambda k: fits[k][1])
    with torch.no_grad():
        PE = P.run(P.MODELS[mname], fits[mname][0], cond, ch, rw.float(), va, record=True)["q"]["pe"].numpy()
    row = {int(s): k for k, s in enumerate(B["sub"])}; T = B["cond"].shape[1]
    def qtil(lab, tri):
        j = row[lab]; c, r, k = B["choice"][j][tri], B["reward"][j][tri], B["cond"][j][tri]
        cols = [r, c, c * r] + [(k == x).astype(float) for x in range(1, 4)] + [(tri / float(T)) ** p for p in (1, 2, 3)]
        v = resid(PE[j][tri].astype(float), cols); return (v - v.mean()) / (v.std() + 1e-9)
    E = ROOT / f"results/eeg_measures_{ds}"; subs = [int(s) for s in B["sub"] if (E / f"sub-{int(s):03d}_epochs.npz").exists()]
    return dict(qtil=qtil, subs=subs, E=E, model=mname, bic={k: v[1] for k, v in fits.items()}, T=T, cond=None,
                valid=lambda lab: B["valid"][row[lab]], rewp_filter=True)


def common_channels(info):
    common = None
    for s in info["subs"]:
        c = [str(x) for x in np.load(info["E"] / f"sub-{s:03d}_epochs.npz")["ch"]]; common = c if common is None else [x for x in common if x in c]
    return common


def epochs(info, s, common):
    Ep = np.load(info["E"] / f"sub-{s:03d}_epochs.npz"); keep = ~Ep["bad"]; chl = [str(x) for x in Ep["ch"]]
    X = Ep["X"][keep][:, [chl.index(c) for c in common]].astype(np.float64); tri = Ep["trial"][keep]; times = Ep["times"]
    if info["rewp_filter"]:
        ok = np.isfinite(Ep["rewp"][tri]); X, tri = X[ok], tri[ok]
    return X, tri, times, Ep


# ------------------------------------------------------------------ D2.0
def table(out):
    rows = {}
    for ds in DSETS:
        info = dataset(ds); common = common_channels(info); ntr, ica, al = [], [], []
        for s in info["subs"]:
            X, tri, times, Ep = epochs(info, s, common); ntr.append(len(tri)); ica.append(int(Ep["n_ica_removed"]) if "n_ica_removed" in Ep.files else None)
            if info.get("valid") is not None:
                T = info["T"]; v = np.asarray(info["valid"](s)); rl = len(Ep["rewp"]) if "rewp" in Ep.files else None
                al.append(dict(sub=s, rewp_len=rl, T=T, len_ok=(rl == T) if rl is not None else None, tri_in_range=bool(tri.max() < T), valid_frac=float(v[tri].mean())))
        sf = float(1 / np.diff(times).mean())
        rows[ds] = dict(n=len(info["subs"]), subs=info["subs"], n_channels=len(common), channels=common, sfreq=round(sf, 2), epoch_s=[float(times[0]), float(times[-1])],
                        trials_mean=float(np.mean(ntr)), trials_min=int(min(ntr)), trials_max=int(max(ntr)), ica_removed_mean=float(np.mean([x for x in ica if x is not None])) if any(x is not None for x in ica) else None,
                        model=info["model"], bic=info.get("bic"), behaviour_T=info["T"], pilot=len(info["subs"]) < 15,
                        alignment=dict(all_ok=bool(all(a["tri_in_range"] and a["len_ok"] is not False and a["valid_frac"] == 1.0 for a in al)), per=al) if al else "n/a (X7 pipeline)")
        print(ds, {k: v for k, v in rows[ds].items() if k not in ("channels", "alignment", "subs")}, "alignment_ok", rows[ds]["alignment"] if isinstance(rows[ds]["alignment"], str) else rows[ds]["alignment"]["all_ok"], flush=True)
    (out / "d2_table.json").write_text(json.dumps(dict(note="D2.0 comparability table (EXECUTOR-3), written before any fit", datasets=rows), indent=1))


# ------------------------------------------------------------------ D2.1
def run_ds(ds, out, subjects):
    t0 = time.time(); info = dataset(ds); common = common_channels(info)
    subs = [s for s in info["subs"] if not subjects or s in subjects]
    for s in subs:
        dest = out / f"{ds}_sub-{s:03d}.json"
        if dest.exists(): continue
        X, tri, times, _ = epochs(info, s, common); res = dict(dataset=ds, participant=f"sub-{s:03d}", target=f"{ds} residualised PE ({info['model']}), z-scored",
                                                         unit="bits per feedback trial", split="10 contiguous blocks, within participant", seed=0,
                                                         bottleneck="none (full width)", N=None, n_trials=int(len(tri)), n_channels=len(common), windows={})
        subsets = {"all": np.ones(len(tri), bool)}
        if ds == "ds007647":
            k = np.asarray(info["cond"](s))[tri]; subsets.update(learnable=k == 2, unlearnable=k == 1)
        def block(F, y):
            n = len(y); Ys = np.column_stack([y] + [np.roll(y, round(n * q / 4)) for q in (1, 2, 3)])
            sc = pe_gain(F, Ys, LS); nz = pe_gain(gnoise(F.shape, 1000 + s), Ys[:, :1], LS)
            curve = {str(L): dict(gain=float(sc[L][:, 0].mean()), shifted=[float(v) for v in sc[L][:, 1:].mean(0)], shift_mean=float(sc[L][:, 1:].mean()),
                                  noise=float(nz[L][:, 0].mean())) for L in LS}
            ce = split_half_ceiling(F, y); pr = curve[str(PRIMARY_L)]
            return dict(n=int(n), dim=int(F.shape[1]), read_out=f"in-fold PCA-{PRIMARY_L} + ridge (primary)", gain=pr["gain"], shift_mean=pr["shift_mean"],
                        shifted=pr["shifted"], noise=pr["noise"], floor=max(pr["shift_mean"], pr["noise"]), ceiling=ce,
                        fraction=pr["gain"] / ce["bits"] if ce["usable"] and ce["bits"] > 0 else None, curve=curve)
        for wname, (lo, hi) in WINS.items():
            w = (times >= lo - 1e-9) & (times <= hi + 1e-9); Fw = X[:, :, w].reshape(len(X), -1)
            for sname, m in subsets.items():
                res["windows"].setdefault(wname, {})[sname] = block(Fw[m], info["qtil"](s, tri[m]).astype(np.float64))
            if wname == "w0600" and len(tri) >= MATCHED_T:          # METHOD CHANGE 12 (2): matched size, first 150 analysed trials (contiguous)
                res["windows"]["w0600"]["matched150"] = block(Fw[:MATCHED_T], info["qtil"](s, tri[:MATCHED_T]).astype(np.float64))
        if ds == "ds003458":
            old = json.loads((ROOT / f"results/standard/pe_bottleneck/sub-{s:03d}.json").read_text())
            nw = res["windows"]["x7"]["all"]; g2 = {}
            for L in ("full", 64):
                o = next(r for r in old["rows"] if r["input"] == "eeg" and r["mode"] == "pca" and r["L"] == L); new_ = nw["curve"][str(L)]["gain"]
                g2[str(L)] = dict(old_gain=o["gain"], new_gain=new_, abs_diff=abs(o["gain"] - new_))
            res["g2"] = dict(file=f"results/standard/pe_bottleneck/sub-{s:03d}.json", widths=g2, old_n=old["n_trials"], new_n=nw["n"],
                             abs_diff=max(v["abs_diff"] for v in g2.values()), passed=bool(old["n_trials"] == nw["n"] and max(v["abs_diff"] for v in g2.values()) < 1e-9))
        fl = [max(abs(v["noise"]), abs(v["shift_mean"])) for wv in res["windows"].values() for v in wv.values()]
        res["g3"] = dict(max_abs_floor=float(max(fl)), note="per participant, informational; G3 is decided at dataset level (METHOD CHANGE 12)"); res["runtime_s"] = round(time.time() - t0, 1)
        dest.write_text(json.dumps(res, indent=1))
        a = res["windows"]["w0600"]["all"]; b = res["windows"]["x7"]["all"]
        print(ds, f"sub-{s:03d}", f"w0600 pca64 {a['gain']:.4f} full {a['curve']['full']['gain']:.4f} (floor {a['floor']:.4f}, ceil {a['ceiling']['bits']})", f"x7 pca64 {b['gain']:.4f}",
              ("G2 " + str(res["g2"]["passed"]) + f" {res['g2']['abs_diff']:.1e}") if "g2" in res else "", f"{time.time() - t0:.0f}s", flush=True)


def boot_ci(v, B=10000, seed=20261121):
    """Participant bootstrap of the mean on the GPU: one [B, n] index tensor."""
    v = torch.as_tensor(np.asarray(v, float), dtype=torch.float64, device=DEV); g = torch.Generator(device=DEV); g.manual_seed(seed)
    m = v[torch.randint(0, len(v), (B, len(v)), generator=g, device=DEV)].mean(1)
    return torch.quantile(m, torch.tensor([.025, .975], dtype=torch.float64, device=DEV)).cpu().tolist()


def g3_test(v):
    """METHOD CHANGE 12 (1): |mean| <= 0.01, bootstrap CI includes 0, sign test p >= 0.05; else 'floor not at chance'."""
    from scipy import stats
    v = np.asarray(v, float); ci = boot_ci(v); k = int((v > 0).sum()); p = float(stats.binomtest(k, len(v), 0.5).pvalue)
    ok = bool(abs(v.mean()) <= 0.01 and ci[0] <= 0 <= ci[1] and p >= 0.05)
    return dict(mean=float(v.mean()), ci=ci, n_pos=k, sign_p=p, passed=ok, label=None if ok else "floor not at chance")


def summarize(out):
    from scipy import stats
    res = {}; tests = []
    for ds in DSETS:
        per = [json.loads(f.read_text()) for f in sorted(out.glob(f"{ds}_sub-*.json"))]
        if not per: continue
        R_ = res[ds] = dict(n=len(per), pilot=ds in PILOT, label="pilot (n = 12), descriptive only, outside the Holm family" if ds in PILOT else None)
        for wname in per[0]["windows"]:
            for sname in per[0]["windows"][wname]:
                P_ = [p for p in per if sname in p["windows"][wname]]; get = lambda f: np.array([f(p["windows"][wname][sname]) for p in P_])
                g, fl = get(lambda r: r["gain"]), get(lambda r: r["floor"]); d = g - fl
                ce = [p["windows"][wname][sname]["ceiling"] for p in P_]; cu = np.array([c["bits"] for c in ce if c["usable"]])
                row = dict(n=len(P_), read_out=f"PCA-{PRIMARY_L} ridge", gain=float(g.mean()), gain_ci=boot_ci(g), floor=float(fl.mean()), gain_minus_floor=float(d.mean()),
                           gain_minus_floor_ci=boot_ci(d), n_above_floor=int((d > 0).sum()), p_one_sided=float(stats.ttest_1samp(d, 0, alternative="greater").pvalue),
                           ceiling_usable_n=int(len(cu)), ceiling_no_usable_n=int(len(ce) - len(cu)), ceiling_mean_usable=float(cu.mean()) if len(cu) else None,
                           rel_mean=float(np.mean([c["rel"] for c in ce])), fraction_ratio_of_means=float(g.mean() / cu.mean()) if len(cu) and cu.mean() > 0 else None,
                           g3_shift=g3_test(get(lambda r: r["shift_mean"])), g3_noise=g3_test(get(lambda r: r["noise"])),
                           floor_outliers=[p["participant"] for p in P_ if max(abs(p["windows"][wname][sname]["shift_mean"]), abs(p["windows"][wname][sname]["noise"])) > 0.06],
                           curve={str(L): float(np.mean([p["windows"][wname][sname]["curve"][str(L)]["gain"] for p in P_])) for L in LS})
                R_[f"{wname}|{sname}"] = row
                if wname == "w0600" and sname == "all" and ds not in PILOT:
                    tests.append((ds, row))
        if ds == "ds007647":
            for wname in WINS:
                L_ = np.array([p["windows"][wname]["learnable"]["gain"] for p in per]); U = np.array([p["windows"][wname]["unlearnable"]["gain"] for p in per])
                Uf = np.array([p["windows"][wname]["unlearnable"]["floor"] for p in per])
                R_[f"{wname}|learnable_minus_unlearnable"] = dict(d=float((L_ - U).mean()), ci=boot_ci(L_ - U), n_pos=int((L_ > U).sum()))
                R_[f"{wname}|D2.2_unlearnable_vs_floor"] = dict(d=float((U - Uf).mean()), ci=boot_ci(U - Uf), passed=bool(boot_ci(U - Uf)[0] <= 0))
        if ds == "ds003458":
            R_["g2_all_pass"] = bool(all(p["g2"]["passed"] for p in per)); R_["g2_max_abs_diff"] = float(max(p["g2"]["abs_diff"] for p in per))
    ps = [r["p_one_sided"] for _, r in tests]; o = np.argsort(ps); m = 0.0
    for rank, i in enumerate(o):
        m = max(m, min(1.0, (len(ps) - rank) * ps[i])); tests[i][1]["p_holm_yield_above_floor"] = m
    # between-dataset contrasts at matched size (150 trials, PCA-64, w0600), unpaired bootstrap vs ds003458, Holm over the non-pilot datasets
    ref = [json.loads(f.read_text())["windows"]["w0600"].get("matched150", {}).get("gain") for f in sorted(out.glob("ds003458_sub-*.json"))]
    ref = np.array([x for x in ref if x is not None]); con = []
    for ds in DSETS:
        if ds in ("ds003458",) or ds in PILOT or ds not in res: continue
        v = np.array([x for x in (json.loads(f.read_text())["windows"]["w0600"].get("matched150", {}).get("gain") for f in sorted(out.glob(f"{ds}_sub-*.json"))) if x is not None])
        if not len(v) or not len(ref): continue
        g = torch.Generator(device=DEV); g.manual_seed(7); vt, rt = (torch.as_tensor(x, dtype=torch.float64, device=DEV) for x in (v, ref))
        bd = vt[torch.randint(0, len(vt), (10000, len(vt)), generator=g, device=DEV)].mean(1) - rt[torch.randint(0, len(rt), (10000, len(rt)), generator=g, device=DEV)].mean(1)
        con.append(dict(dataset=ds, minus="ds003458", d=float(v.mean() - ref.mean()), ci=torch.quantile(bd, torch.tensor([.025, .975], dtype=torch.float64, device=DEV)).cpu().tolist(),
                        p_two_sided=float(min(1.0, 2 * min(float((bd <= 0).double().mean()), float((bd >= 0).double().mean())))), n=[int(len(v)), int(len(ref))]))
    ps = [c["p_two_sided"] for c in con]; o = np.argsort(ps); m = 0.0
    for rank, i in enumerate(o):
        m = max(m, min(1.0, (len(ps) - rank) * ps[i])); con[i]["p_holm"] = m
    (out / "summary_d2.json").write_text(json.dumps(dict(experiment="D2.1 PE yield per task (B3) + D2.2 doors control", method_change="METHOD CHANGE 12",
                                                         primary="w0600 | all | in-fold PCA-64 ridge", holm_family=[d for d, _ in tests],
                                                         matched150_contrasts=con, datasets=res), indent=1))
    print(json.dumps({ds: {k: (v if not isinstance(v, dict) else {kk: vv for kk, vv in v.items() if kk in ("n", "gain", "gain_ci", "floor", "n_above_floor", "p_holm_yield_above_floor", "fraction_ratio_of_means", "d", "ci", "passed")})
                           for k, v in r.items() if k.startswith("w0600|") or not isinstance(v, dict)} for ds, r in res.items()}, indent=1)[:6000])
    print(json.dumps(con, indent=1))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--ds", default=""); ap.add_argument("--subjects", default=""); ap.add_argument("--out", required=True)
    ap.add_argument("--table", action="store_true"); ap.add_argument("--summarize", action="store_true")
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    if a.table: return table(out)
    if a.summarize: return summarize(out)
    subj = [int(x) for x in a.subjects.split(",")] if a.subjects else []
    for ds in (a.ds.split(",") if a.ds else DSETS):
        run_ds(ds, out, subj)


if __name__ == "__main__":
    main()
