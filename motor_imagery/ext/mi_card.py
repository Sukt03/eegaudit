"""MI card (Track B, user priority 21 Nov): motor imagery on EEGMMIDB (PhysioNet, Schalk 2004 / Goldberger 2000) as the
strong-signal positive-control modality and the real-data D6 check. Version 2 = METHOD G1 round 17:20 binding changes 1-5.

Task: 4 classes = left fist / right fist imagery (runs 4, 8, 12: T1 / T2) and both fists / both feet imagery (runs 6, 10, 14: T1 / T2).
Window: 0.5-4.0 s after the cue for EVERY read-out (0-0.5 s carries the lateralised cue response; METHOD diagnostic).
Designs:
  primary_lpgo     leave-participant-GROUP-out: 10 groups = index mod 10 over the sorted QC-passing list. Outer: train on the other 9
                   groups, test every trial of the held-out group's participants. Inner: validation group (g+1) mod 10 inside the outer
                   training set; (lambda, T) / (step, T) chosen jointly on its CE; then refit on all 9 groups. Features: per-participant
                   column z-scoring on all of that participant's trials (label-free, transductive; class counts asserted balanced), then
                   PCA-256 fitted on the outer-training rows (label-free), then exact-Newton multinomial logistic. EEGNet: one member per
                   (group, sel/fin, eeg/noise) = 40, batch 64, 4000 steps, checkpoint every 100.
  secondary_within leave one run PAIR out within participant (pairs (4,6), (8,10), (12,14)); inner = the two training pairs, each once as
                   validation; kernel-form logistic on per-fit z-scored features; labelled "within participant, 56 training trials; weak".
                   EEGNet in this design only with --within_eegnet (600 steps).
Score: held-out log2 p(true class) - log2 prior(train class frequencies, +1 smoothed), bits per trial; temperature on the grid
2**linspace(-3, 6, 37), chosen jointly with lambda (17-point grid 10**linspace(-4, 4, 17)) or step. Per participant = mean over its test
trials; group mean, participant bootstrap CI (10 000), paired sign-flip p (10 000) with Holm over bp/ts/eegnet - raw.
Read-outs: raw (1-40 Hz voltage, 20 Hz), raw_nofp, bp (log band power 4-8/8-13/13-20/20-30/30-40 Hz), bp_nofp, ts (log-Euclidean tangent
space of 8-30 Hz covariances, 1 % shrinkage), eegnet (EEGNet-8,2, 4-40 Hz). nofp = Fp1/Fpz/Fp2/AF7/AF3/AFz/AF4/AF8/F7/F8/FT7/FT8 removed
(no EOG in EEGMMIDB). Floors: noise input (Gaussian 1-40 Hz, per-participant per-channel SD: label-free, trial-invariant) through the
same pipeline (eegnet = trained noise twin); shuffled training labels (within participant) for logistic rungs. No ceiling (no repeats).
QC: dropped if sfreq != 160 Hz, a run is missing, a run has < 8 task trials, or class counts differ by > 2 (listed).
G2: Newton vs sklearn (|dp|), and ts-logistic left-vs-right accuracy (classes 0/1, argmax over their 2 logits) >= 0.60.
D6 falsifier (primary design): triggered iff the CI lower bounds of bp - raw AND eegnet - raw are both <= 0.
Usage: python mi_card.py --out DIR [--subjects 1,2,3] [--designs primary_lpgo,secondary_within] [--check_sklearn] [--within_eegnet]
"""
import argparse, hashlib, json, math, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path("/path/to/eegaudit"); DATA = ROOT / "data_raw/third_party/eegmmidb"
DEV = "cuda"; SF = 160; PRE = 160                       # epochs -1..5 s (960 samples); onset at sample 160
W0, W1 = PRE + 80, PRE + 640                             # analysis window 0.5-4.0 s (560 samples)
RUNS = [4, 6, 8, 10, 12, 14]; PAIR = {4: 0, 6: 0, 8: 1, 10: 1, 12: 2, 14: 2}
BANDS = [(4, 8), (8, 13), (13, 20), (20, 30), (30, 40)]
FP = {"FP1", "FPZ", "FP2", "AF7", "AF3", "AFZ", "AF4", "AF8", "F7", "F8", "FT7", "FT8"}
LAMS = 10.0 ** np.linspace(-4, 4, 17); TGRID = torch.tensor(2.0 ** np.linspace(-3, 6, 37), dtype=torch.float64)
FEATS = ["raw", "raw_nofp", "bp", "bp_nofp", "ts"]
torch.backends.cudnn.benchmark = True
t0 = time.time()
log = lambda *m: print(f"[{time.time() - t0:7.1f}s]", *m, flush=True)


# ---------------------------------------------------------------- data
def load_sub(s, cache):
    f = cache / f"sub-{s:03d}.npz"
    if f.exists():
        z = np.load(f); return dict(X=z["X"], y=z["y"], pair=z["pair"], run=z["run"], ch=list(z["ch"])), None
    import mne
    X, y, pr, rn, ch = [], [], [], [], None
    for r in RUNS:
        p = DATA / f"S{s:03d}/S{s:03d}R{r:02d}.edf"
        if not p.exists(): return None, f"missing run {r}"
        raw = mne.io.read_raw_edf(p, preload=True, verbose="ERROR")
        if raw.info["sfreq"] != SF: return None, f"sfreq {raw.info['sfreq']}"
        if len(raw.ch_names) != 64: return None, f"{len(raw.ch_names)} channels"
        ch = ch or [c.strip(".").upper() for c in raw.ch_names]
        x = raw.get_data() * 1e6; ev, ed = mne.events_from_annotations(raw, verbose="ERROR"); inv = {v: k for k, v in ed.items()}
        k = 0
        for on, _, code in ev:
            lab = inv[code]
            if lab not in ("T1", "T2") or on - PRE < 0 or on + 800 > x.shape[1]: continue
            X.append(x[:, on - PRE:on + 800]); y.append((0 if lab == "T1" else 1) + (0 if r in (4, 8, 12) else 2)); pr.append(PAIR[r]); rn.append(r); k += 1
        if k < 8: return None, f"run {r}: {k} task trials"
    d = dict(X=np.stack(X).astype(np.float16), y=np.array(y, np.int64), pair=np.array(pr, np.int64), run=np.array(rn, np.int64), ch=np.array(ch))
    np.savez(f, **d); d["ch"] = ch; return d, None


def bandpass(X, lo, hi, chunk=1500):
    """Zero-phase FFT band-pass with 2 Hz cosine edges on the -1..5 s epoch (edges are cropped away by every caller)."""
    n = X.shape[-1]; fr = torch.fft.rfftfreq(n, 1 / SF, device=X.device)
    m = torch.ones_like(fr)
    lo_e = (fr < lo) & (fr > lo - 2); m[fr <= lo - 2] = 0; m[lo_e] = 0.5 - 0.5 * torch.cos(math.pi * (fr[lo_e] - (lo - 2)) / 2)
    hi_e = (fr > hi) & (fr < hi + 2); m[fr >= hi + 2] = 0; m[hi_e] = 0.5 + 0.5 * torch.cos(math.pi * (fr[hi_e] - hi) / 2)
    out = torch.empty_like(X)
    for i in range(0, len(X), chunk):
        x = X[i:i + chunk]; x = x - x.mean(-1, keepdim=True)
        out[i:i + chunk] = torch.fft.irfft(torch.fft.rfft(x, dim=-1) * m, n=n, dim=-1)
    return out


def features(X, keep_nofp):
    """X [N, 64, 960] float32 GPU -> dict of feature matrices [N, F] (float64) + EEGNet input [N, 64, 560] (all 0.5-4.0 s)."""
    out = {}; N = len(X)
    r = bandpass(X, 1, 40)[:, :, W0:W1].reshape(N, 64, 70, 8).mean(-1)
    out["raw"] = r.reshape(N, -1).double(); out["raw_nofp"] = r[:, keep_nofp].reshape(N, -1).double(); del r
    bp = torch.stack([bandpass(X, lo, hi)[:, :, W0:W1].pow(2).mean(-1).log() for lo, hi in BANDS], -1)   # [N, 64, 5]
    out["bp"] = bp.reshape(N, -1).double(); out["bp_nofp"] = bp[:, keep_nofp].reshape(N, -1).double()
    s = bandpass(X, 8, 30)[:, :, W0:W1].double(); C = s @ s.transpose(1, 2) / s.shape[-1]; del s
    C = C + 0.01 * C.diagonal(dim1=1, dim2=2).mean(-1)[:, None, None] * torch.eye(64, dtype=C.dtype, device=C.device)
    w, V = torch.linalg.eigh(C); L = (V * w.log()[:, None, :]) @ V.transpose(1, 2)
    iu = torch.triu_indices(64, 64, device=X.device); sc = torch.where(iu[0] == iu[1], 1.0, math.sqrt(2.0)).double()
    out["ts"] = L[:, iu[0], iu[1]] * sc
    return out, bandpass(X, 4, 40)[:, :, W0:W1].float()


# ---------------------------------------------------------------- batched multinomial logistic (exact Newton)
def newton_logistic(Phi, Y, mask, lam, iters=30):
    """Phi [M, n, d] (float64, padded rows zero), Y [M, n] int, mask [M, n], lam [M] -> W [M, d+1, 4] (last row = intercept).
    Minimises mean_i CE_i + lam/2 ||w||^2 + 1e-6/2 ||b||^2 per member; Newton steps with per-member backtracking."""
    M, n, d = Phi.shape; K = 4; P1 = torch.cat([Phi, torch.ones(M, n, 1, dtype=Phi.dtype, device=Phi.device)], -1); D = d + 1
    Yo = F.one_hot(Y, K).double() * mask[..., None]; nm = mask.sum(1).clamp(min=1)
    reg = torch.cat([lam[:, None].expand(M, d), torch.full((M, 1), 1e-6, dtype=Phi.dtype, device=Phi.device)], 1)   # [M, D]
    W = torch.zeros(M, D, K, dtype=Phi.dtype, device=Phi.device)

    def obj(W):
        Z = P1 @ W; ce = -(Yo * torch.log_softmax(Z, -1)).sum(-1).sum(1) / nm
        return ce + 0.5 * (reg[..., None] * W ** 2).sum((1, 2))

    for it in range(iters):
        Z = P1 @ W; Pr = torch.softmax(Z, -1) * mask[..., None]
        G = (P1.transpose(1, 2) @ (Pr - Yo)) / nm[:, None, None] + reg[..., None] * W                     # [M, D, K]
        S = torch.diag_embed(Pr) - Pr[..., :, None] * Pr[..., None, :]                                      # [M, n, K, K]
        SP = (S[..., None] * P1[:, :, None, None, :]).reshape(M, n, K * K * D)                             # [M, n, K*K*D]
        H = (P1.transpose(1, 2) @ SP).reshape(M, D, K, K, D).permute(0, 1, 2, 4, 3) / nm[:, None, None, None, None]   # [M, D(a), K(k), D(b), K(l)]
        del SP
        H = H.reshape(M, D * K, D * K) + torch.diag_embed((reg[..., None].expand(M, D, K)).reshape(M, D * K))
        step = torch.linalg.solve(H, G.reshape(M, D * K, 1)).reshape(M, D, K)
        f0 = obj(W); t = torch.ones(M, dtype=W.dtype, device=W.device); done = torch.zeros(M, dtype=torch.bool, device=W.device)
        for _ in range(12):
            Wn = W - t[:, None, None] * step; ok = (obj(Wn) <= f0 + 1e-12) & ~done
            W = torch.where(ok[:, None, None], Wn, W); done |= ok; t = torch.where(done, t, t / 2)
        gmax = float(G.abs().amax())
        if gmax < 1e-9: break
    return W, gmax


def kernel_fits(Xf, fits, ylab, lams, chunk=600):
    """Within-participant design. fits: list of (train_idx, eval_idx). Returns eval logits per fit: [len(lams), n_eval, 4]."""
    nt = max(len(f[0]) for f in fits); ne = max(len(f[1]) for f in fits); Fd = Xf.shape[1]
    out = [None] * len(fits); gm = 0.0; step = max(1, chunk // len(lams))
    for c0 in range(0, len(fits), step):
        fc = fits[c0:c0 + step]; B = len(fc)
        Phi = torch.zeros(B, nt, nt, dtype=torch.float64, device=DEV); Pev = torch.zeros(B, ne, nt, dtype=torch.float64, device=DEV)
        Y = torch.zeros(B, nt, dtype=torch.long, device=DEV); Mk = torch.zeros(B, nt, dtype=torch.float64, device=DEV)
        for j, f in enumerate(fc):
            tr, ev = f[0], f[1]
            A, E_ = Xf[tr], Xf[ev]; mu, sd = A.mean(0), A.std(0) + 1e-8
            A = (A - mu) / sd / math.sqrt(Fd); E_ = (E_ - mu) / sd / math.sqrt(Fd)
            Kt = A @ A.T; w, U = torch.linalg.eigh(Kt); keep = w > 1e-10 * w.max()
            ph = U[:, keep] * w[keep].sqrt(); pe = (E_ @ A.T) @ U[:, keep] / w[keep].sqrt()
            Phi[j, :len(tr), :ph.shape[1]] = ph; Pev[j, :len(ev), :pe.shape[1]] = pe
            Y[j, :len(tr)] = ylab[j + c0]; Mk[j, :len(tr)] = 1
        L = len(lams); lam = torch.tensor(lams, dtype=torch.float64, device=DEV).repeat(B)
        W, g = newton_logistic(Phi.repeat_interleave(L, 0), Y.repeat_interleave(L, 0), Mk.repeat_interleave(L, 0), lam); gm = max(gm, g)
        Pe1 = torch.cat([Pev, torch.ones(B, ne, 1, dtype=Pev.dtype, device=DEV)], -1).repeat_interleave(L, 0)
        Z = (Pe1 @ W).reshape(B, L, ne, 4)
        for j, f in enumerate(fc): out[c0 + j] = Z[j, :, :len(f[1])]
    return out, gm


def primal_fits(Xf, fits, ylab, lams, pca_rows, ncomp=256, lchunk=4):
    """LPGO design. fits: list of (train_idx, eval_idx, pca_key); pca_rows[pca_key] = rows the label-free PCA is fitted on.
    Returns eval logits per fit [len(lams), n_eval, 4], max |grad|, explained variance per pca_key."""
    out = [None] * len(fits); gm = 0.0; basis, ev_share = {}, {}
    for key, rows in pca_rows.items():
        A = Xf[rows]; mu = A.mean(0); A = A - mu; Cv = A.T @ A / len(A); w, V = torch.linalg.eigh(Cv)
        k = min(ncomp, Cv.shape[0]); basis[key] = (mu, V[:, -k:].flip(1) / math.sqrt(Xf.shape[1])); ev_share[key] = float(w[-k:].sum() / w.sum())
    for i, (tr, ev, key) in enumerate(fits):
        mu, V = basis[key]; Ptr = (Xf[tr] - mu) @ V; Pev = (Xf[ev] - mu) @ V
        Pe1 = torch.cat([Pev, torch.ones(len(ev), 1, dtype=Pev.dtype, device=DEV)], -1); Zs = []
        for l0 in range(0, len(lams), lchunk):
            lc = lams[l0:l0 + lchunk]; L = len(lc)
            W, g = newton_logistic(Ptr[None].expand(L, -1, -1).contiguous(), ylab[i][None].expand(L, -1).contiguous(),
                                   torch.ones(L, len(tr), dtype=torch.float64, device=DEV), torch.tensor(lc, dtype=torch.float64, device=DEV))
            gm = max(gm, g); Zs.append(torch.einsum("nd,ldk->lnk", Pe1, W))
        out[i] = torch.cat(Zs)
    return out, gm, ev_share


# ---------------------------------------------------------------- batched EEGNet (one independent member per unit x role x twin)
class MEEGNet(nn.Module):
    def __init__(s, M, C=64, T=560, F1=8, D=2, F2=16, K1=80, p=0.25):
        super().__init__(); s.M, s.p = M, p
        s.c1 = nn.Conv2d(M, M * F1, (1, K1), groups=M, padding=(0, K1 // 2), bias=False); s.bn1 = nn.BatchNorm2d(M * F1)
        s.c2 = nn.Conv2d(M * F1, M * F1 * D, (C, 1), groups=M * F1, bias=False); s.bn2 = nn.BatchNorm2d(M * F1 * D)
        s.c3 = nn.Conv2d(M * F1 * D, M * F1 * D, (1, 16), groups=M * F1 * D, padding=(0, 8), bias=False)
        s.c4 = nn.Conv2d(M * F1 * D, M * F2, 1, groups=M, bias=False); s.bn3 = nn.BatchNorm2d(M * F2)
        nf = F2 * (((T + 1) // 4 + 1) // 8)
        s.W = nn.Parameter(torch.randn(M, nf, 4) / math.sqrt(nf)); s.b = nn.Parameter(torch.zeros(M, 4))

    def forward(s, x):                                       # x [B, M, C, T] -> [B, M, 4]
        z = F.elu(s.bn2(s.c2(s.bn1(s.c1(x))))); z = F.dropout(F.avg_pool2d(z, (1, 4)), s.p, s.training)
        z = F.elu(s.bn3(s.c4(s.c3(z)))); z = F.dropout(F.avg_pool2d(z, (1, 8)), s.p, s.training)
        return torch.einsum("bmi,mic->bmc", z.reshape(z.shape[0], s.M, -1), s.W) + s.b


def eegnet_members(Xin, yall, members, steps, every, batch, mchunk, seed):
    """members: list of (train_idx, eval_idx) global rows of Xin. Returns eval logits per member [n_ckpt, n_eval, 4]."""
    out = [None] * len(members)
    for c0 in range(0, len(members), mchunk):
        mc = members[c0:c0 + mchunk]; M = len(mc); torch.manual_seed(seed + c0)
        lt = max(len(a) for a, _ in mc); le = max(len(b) for _, b in mc)
        trp = torch.zeros(M, lt, dtype=torch.long, device=DEV); tl = torch.tensor([len(a) for a, _ in mc], device=DEV)
        evp = torch.zeros(M, le, dtype=torch.long, device=DEV)
        for j, (a, b) in enumerate(mc): trp[j, :len(a)] = a; evp[j, :len(b)] = b; evp[j, len(b):] = b[0]
        net = MEEGNet(M, T=Xin.shape[-1]).to(DEV); opt = torch.optim.Adam(net.parameters(), lr=1e-3); ck = []; ts = time.time()
        for st in range(1, steps + 1):
            net.train(); pos = (torch.rand(M, batch, device=DEV) * tl[:, None]).long(); idx = trp.gather(1, pos)   # [M, batch]
            x = Xin[idx].transpose(0, 1).contiguous(); yy = yall[idx].T                                          # [batch, M, C, T]
            loss = F.cross_entropy(net(x).reshape(-1, 4), yy.reshape(-1), reduction="none").reshape(batch, M).mean(0).sum()
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            if st % every == 0:
                net.eval(); zz = []
                with torch.no_grad():
                    for e0 in range(0, le, 128): zz.append(net(Xin[evp[:, e0:e0 + 128]].transpose(0, 1).contiguous()).transpose(0, 1).float())
                ck.append(torch.cat(zz, 1))                                                                    # [M, le, 4]
            if st == every: log(f"  eegnet chunk {c0 // mchunk + 1}: {(time.time() - ts) / every:.3f} s/step")
        Z = torch.stack(ck, 1)
        for j, (_, b) in enumerate(mc): out[c0 + j] = Z[j, :, :len(b)]
        log(f"eegnet chunk {c0 // mchunk + 1}/{math.ceil(len(members) / mchunk)} M={M} mem {torch.cuda.max_memory_allocated() / 2**30:.1f} GB")
        del net, opt; torch.cuda.empty_cache()
    return out


# ---------------------------------------------------------------- scoring helpers
def ce_grid(Z, y):                                           # CE (nats) at every T of TGRID -> [nT]
    T = TGRID.to(Z.device); lp = torch.log_softmax(Z.double()[None] / T[:, None, None], -1)
    return -lp.gather(2, y[None, :, None].expand(len(T), -1, 1))[..., 0].mean(1)


def select(Zinner, yinner):
    """Zinner: [n_opt, n, 4] (options = lambda or checkpoint). Joint (option, T) minimising inner CE -> (option index, T, edge flags)."""
    ce = torch.stack([ce_grid(Zinner[o], yinner) for o in range(len(Zinner))])   # [n_opt, nT]
    i = int(ce.argmin()); o, t = divmod(i, ce.shape[1])
    return o, float(TGRID[t]), dict(opt_edge=o in (0, len(Zinner) - 1), T_edge=t in (0, len(TGRID) - 1))


def bits(Z, y, T, prior):                                    # per-trial log2 p(y) - log2 prior(y)
    return (torch.log_softmax(Z.double() / T, -1).gather(1, y[:, None])[:, 0] - prior[y].log()) / math.log(2)


def prior_of(ytr):
    c = torch.bincount(ytr, minlength=4).double() + 1; return c / c.sum()


def boot(v, n=10000, seed=0):
    v = torch.as_tensor(np.asarray(v, np.float64), device=DEV); g = torch.Generator(device=DEV); g.manual_seed(seed)
    m = v[torch.randint(0, len(v), (n, len(v)), generator=g, device=DEV)].mean(1)
    return [float(torch.quantile(m, 0.025)), float(torch.quantile(m, 0.975))]


def signflip_p(d, n=10000, seed=1):                          # one-sided p for mean(d) > 0
    d = torch.as_tensor(np.asarray(d, np.float64), device=DEV); g = torch.Generator(device=DEV); g.manual_seed(seed)
    s = torch.randint(0, 2, (n, len(d)), generator=g, device=DEV).double() * 2 - 1
    return float(((s * d).mean(1) >= d.mean()).double().mean())


def holm(ps):
    o = np.argsort(ps); m = len(ps); adj = np.empty(m); run = 0.0
    for r, i in enumerate(o): run = max(run, (m - r) * ps[i]); adj[i] = min(1.0, run)
    return adj.tolist()


# ---------------------------------------------------------------- one design
def run_design(name, units, fits, Fdict, y, ylab_by_cond, sid, nP, engine, eeg_members, eeg_units, Xin, yin, eeg_cfg):
    """units: list of dict(inner=[fit idx], final=fit idx) over the logistic fits; eeg_units: same over EEGNet members.
    Returns per-participant dict rows[feat][cond] = (bits, acc, n) + selection diagnostics."""
    N = len(y); R = {}; diag = dict(edges={}, grad={}, pca_ev={})
    for cond, FS in Fdict.items():
        ylab = ylab_by_cond[cond]
        for fname, Xf in FS.items():
            Z, gm, *extra = engine(Xf, fits, ylab); diag["grad"][f"{cond}|{fname}"] = gm
            if extra: diag["pca_ev"][f"{cond}|{fname}"] = extra[0]
            bt = torch.full((N,), float("nan"), dtype=torch.float64, device=DEV); ok = torch.zeros(N, dtype=torch.float64, device=DEV)
            lr = torch.full((N,), float("nan"), dtype=torch.float64, device=DEV); ne = [0, 0]
            for u in units:
                Zi = torch.cat([Z[i] for i in u["inner"]], 1); yi = torch.cat([y[fits[i][1]] for i in u["inner"]])
                l, T, e = select(Zi, yi); ne[0] += e["opt_edge"]; ne[1] += e["T_edge"]
                f = u["final"]; ev = fits[f][1]; Zf = Z[f][l]
                bt[ev] = bits(Zf, y[ev], T, prior_of(ylab[f])); ok[ev] = (Zf.argmax(-1) == y[ev]).double()
                m = y[ev] < 2; lr[ev[m]] = (Zf[m][:, :2].argmax(-1) == y[ev][m]).double()
            diag["edges"][f"{cond}|{fname}"] = dict(lambda_edge=ne[0], T_edge=ne[1], n_units=len(units))
            assert not torch.isnan(bt).any(), f"{name} {cond} {fname}: untested trials"
            for k in range(nP):
                r = sid == k
                R.setdefault(k, {}).setdefault(fname, {})[cond] = dict(bits=float(bt[r].mean()), acc=float(ok[r].mean()), lr_acc=float(lr[r].nanmean()), n_test=int(r.sum()))
            log(f"{name} logistic {cond} {fname}: max|grad| {gm:.1e}; mean bits {float(bt.mean()):+.4f} acc {float(ok.mean()):.3f}")
    if eeg_members:
        Zn = eegnet_members(Xin, yin, eeg_members, **eeg_cfg)
        for tw in ("eeg", "noise"):
            bt = torch.full((N,), float("nan"), dtype=torch.float64, device=DEV); ok = torch.zeros(N, dtype=torch.float64, device=DEV); ne = [0, 0]
            for u in [u for u in eeg_units if u["twin"] == tw]:
                isel, ifin = u["inner"][0], u["final"]
                c, T, e = select(Zn[isel], yin[eeg_members[isel][1]]); ne[0] += e["opt_edge"]; ne[1] += e["T_edge"]
                ev = eeg_members[ifin][1]; evy = ev % N; Zf = Zn[ifin][c]
                bt[evy] = bits(Zf, yin[ev], T, prior_of(yin[eeg_members[ifin][0]])); ok[evy] = (Zf.argmax(-1) == yin[ev]).double()
            diag["edges"][f"{tw}|eegnet"] = dict(step_edge=ne[0], T_edge=ne[1])
            assert not torch.isnan(bt).any(), f"{name} eegnet {tw}: untested trials"
            for k in range(nP):
                r = sid == k; R[k].setdefault("eegnet", {})[tw] = dict(bits=float(bt[r].mean()), acc=float(ok[r].mean()), n_test=int(r.sum()))
            log(f"{name} eegnet {tw}: mean bits {float(bt.mean()):+.4f} acc {float(ok.mean()):.3f}")
    return R, diag


def summarize(name, R, P, diag, label, extra):
    rows = [r for r in FEATS + ["eegnet"] if r in R[0]]
    S = dict(experiment=f"MI card v2 (EEGMMIDB, 4-class imagery) - {name}", label=label, n=len(P), participants=P,
             window="0.5-4.0 s after the cue (all read-outs)", unit="bits per trial (log2 p - log2 train prior)",
             ceiling="none (no repeated stimuli; reported against floors and twins only)",
             labels=dict(raw_nofp="Fp/AF + F7/F8/FT7/FT8 removed (no EOG in EEGMMIDB)", bp_nofp="Fp/AF + F7/F8/FT7/FT8 removed (no EOG in EEGMMIDB)",
                         eegnet_noise="trained noise twin", noise="noise input (per-participant per-channel SD)", shuffle="labels permuted within participant"),
             code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), diag=diag, **extra)
    for r in rows:
        for cond in R[0][r]:
            v = [R[k][r][cond]["bits"] for k in range(len(P))]; ac = [R[k][r][cond]["acc"] for k in range(len(P))]
            row = dict(mean=float(np.mean(v)), ci=boot(v), acc=float(np.mean(ac)))
            if cond != "eeg": row["g3"] = dict(n_abs_le_0p02=int(sum(abs(x) <= 0.02 for x in v)), n=len(v))
            S.setdefault("rows", {}).setdefault(r, {})[cond] = row
    S["g2_lr_ts_acc"] = float(np.mean([R[k]["ts"]["eeg"]["lr_acc"] for k in range(len(P))])); S["g2_pass"] = S["g2_lr_ts_acc"] >= 0.60
    ps = []
    for r in ("bp", "ts", "eegnet"):
        if r not in R[0]: continue
        d = [R[k][r]["eeg"]["bits"] - R[k]["raw"]["eeg"]["bits"] for k in range(len(P))]; p = signflip_p(d); ps.append(p)
        S.setdefault("d6_minus_raw", {})[r] = dict(d=float(np.mean(d)), ci=boot(d), n_pos=int(sum(x > 0 for x in d)), p_signflip=p)
    for r, ph in zip([r for r in ("bp", "ts", "eegnet") if r in R[0]], holm(ps)): S["d6_minus_raw"][r]["p_holm"] = ph
    d = [R[k]["bp_nofp"]["eeg"]["bits"] - R[k]["raw_nofp"]["eeg"]["bits"] for k in range(len(P))]
    S["d6_minus_raw_nofp"] = dict(bp_nofp=dict(d=float(np.mean(d)), ci=boot(d), n_pos=int(sum(x > 0 for x in d)), p_signflip=signflip_p(d)))
    d = [R[k]["bp"]["eeg"]["bits"] - R[k]["bp_nofp"]["eeg"]["bits"] for k in range(len(P))]; S["bp_minus_bp_nofp"] = dict(d=float(np.mean(d)), ci=boot(d))
    if "eegnet" in R[0]:
        S["d6_falsifier_triggered"] = bool(S["d6_minus_raw"]["bp"]["ci"][0] <= 0 and S["d6_minus_raw"]["eegnet"]["ci"][0] <= 0)
    return S


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--subjects", default="")
    ap.add_argument("--designs", default="primary_lpgo,secondary_within"); ap.add_argument("--check_sklearn", action="store_true")
    ap.add_argument("--within_eegnet", action="store_true"); ap.add_argument("--lpgo_steps", type=int, default=4000)
    ap.add_argument("--within_steps", type=int, default=600); ap.add_argument("--mchunk", type=int, default=20)
    a = ap.parse_args(); out = Path(a.out); cache = ROOT / "results/ext/mi_card_cache"; cache.mkdir(parents=True, exist_ok=True)
    subs = [int(x) for x in a.subjects.split(",")] if a.subjects else list(range(1, 110))
    D, excl = {}, {}
    for s in subs:
        d, why = load_sub(s, cache)
        if d is None: excl[s] = why; log(f"sub-{s:03d} excluded: {why}"); continue
        c = np.bincount(d["y"], minlength=4)
        if c.max() - c.min() > 2:                             # QC (METHOD rule): per-participant standardisation would be label-informative
            excl[s] = f"class counts {c.tolist()} differ by > 2"; log(f"sub-{s:03d} excluded: {excl[s]}"); continue
        D[s] = d
    P = sorted(D); nP = len(P); log(f"{nP} participants, excluded {excl}")
    ch = D[P[0]]["ch"]; keep_nofp = torch.tensor([c not in FP for c in ch], device=DEV)
    assert all(D[s]["ch"] == ch for s in P), "channel order differs"; assert int((~keep_nofp).sum()) == len(FP), "ocular channel names not all found"
    X = torch.tensor(np.concatenate([D[s]["X"] for s in P]), dtype=torch.float32, device=DEV)
    y = torch.tensor(np.concatenate([D[s]["y"] for s in P]), device=DEV); pair = torch.tensor(np.concatenate([D[s]["pair"] for s in P]), device=DEV)
    sid = torch.tensor(np.concatenate([[k] * len(D[s]["y"]) for k, s in enumerate(P)]), device=DEV); N = len(y)
    g = torch.Generator(device=DEV); g.manual_seed(20261121)
    Xb = bandpass(X, 1, 40); Nz = bandpass(torch.randn(X.shape, generator=g, device=DEV), 1, 40)
    for k in range(nP):
        r = sid == k; Nz[r] = Nz[r] / Nz[r].std((0, 2), keepdim=True) * Xb[r].std((0, 2), keepdim=True)
    del Xb
    FE, XinE = features(X, keep_nofp); FN, XinN = features(Nz, keep_nofp); del X, Nz; torch.cuda.empty_cache()
    log("features", {k: tuple(v.shape) for k, v in FE.items()})
    Xin = torch.cat([XinE, XinN]); del XinE, XinN
    for k in range(nP):                                       # per-participant scalar scale (label-free)
        r = torch.where(sid == k)[0]; sc = Xin[r].std(); Xin[r] /= sc; Xin[r + N] /= sc
    yin = torch.cat([y, y])
    gperm = torch.Generator(device=DEV); gperm.manual_seed(7)
    ysh = y.clone()                                           # labels permuted within each participant (shuffle floor)
    for k in range(nP):
        r = torch.where(sid == k)[0]; ysh[r] = y[r[torch.randperm(len(r), generator=gperm, device=DEV)]]
    designs = a.designs.split(",")

    if "primary_lpgo" in designs:
        G = min(10, nP); grp = torch.tensor([k % G for k in range(nP)], device=DEV)[sid]
        for k in range(nP):
            c = torch.bincount(y[sid == k], minlength=4); assert int(c.max() - c.min()) <= 2, f"sub-{P[k]:03d} class counts {c.tolist()}"
        def zpp(FS):                                         # per-participant column z-scoring (label-free)
            O = {}
            for fn, Xf in FS.items():
                Z = torch.empty_like(Xf)
                for k in range(nP):
                    r = sid == k; Z[r] = (Xf[r] - Xf[r].mean(0)) / (Xf[r].std(0) + 1e-8)
                O[fn] = Z
            return O
        FEz, FNz = zpp(FE), zpp(FN)
        fits, units, pca_rows = [], [], {}
        for gg in range(G):
            v = (gg + 1) % G; outer = torch.where(grp != gg)[0]; pca_rows[gg] = outer
            tr_in = torch.where((grp != gg) & (grp != v))[0]; ev_in = torch.where(grp == v)[0]; ev = torch.where(grp == gg)[0]
            fits.append((tr_in, ev_in, gg)); fits.append((outer, ev, gg)); units.append(dict(inner=[len(fits) - 2], final=len(fits) - 1))
        ylab = {c: [yy[f[0]] for f in fits] for c, yy in (("eeg", y), ("noise", y), ("shuffle", ysh))}
        mem, eu = [], []
        for gg in range(G):
            v = (gg + 1) % G; tr_in = torch.where((grp != gg) & (grp != v))[0]; ev_in = torch.where(grp == v)[0]
            outer = torch.where(grp != gg)[0]; ev = torch.where(grp == gg)[0]
            for tw, off in (("eeg", 0), ("noise", N)):
                mem.append((tr_in + off, ev_in + off)); mem.append((outer + off, ev + off)); eu.append(dict(inner=[len(mem) - 2], final=len(mem) - 1, twin=tw))
        R, diag = run_design("primary_lpgo", units, fits, {"eeg": FEz, "noise": FNz, "shuffle": FEz}, y, ylab, sid, nP,
                             lambda Xf, fi, yl: primal_fits(Xf, fi, yl, LAMS, pca_rows), mem, eu, Xin, yin,
                             dict(steps=a.lpgo_steps, every=100, batch=64, mchunk=a.mchunk, seed=11))
        o = out / "primary_lpgo"; o.mkdir(parents=True, exist_ok=True)
        for k, s in enumerate(P): (o / f"sub-{s:03d}.json").write_text(json.dumps(dict(participant=s, group=k % G, **R[k]), indent=1))
        S = summarize("primary_lpgo", R, P, diag, "PRIMARY: leave-participant-group-out (10 groups), per-participant standardisation (label-free, transductive), PCA-256",
                      dict(groups=G, steps=a.lpgo_steps, runtime_s=round(time.time() - t0), excluded={str(k): v for k, v in excl.items()}))
        (o / "summary_mi.json").write_text(json.dumps(S, indent=1)); log("PRIMARY", json.dumps({k: S[k] for k in ("rows", "d6_minus_raw", "g2_lr_ts_acc", "g2_pass")}, indent=1))

    if "secondary_within" in designs:
        fits, units = [], []
        for k in range(nP):
            for oo in range(3):
                pa, pb = [q for q in range(3) if q != oo]
                ia, ib, io = [torch.where((sid == k) & (pair == q))[0] for q in (pa, pb, oo)]
                fits += [(ia, ib), (ib, ia), (torch.cat([ia, ib]), io)]; units.append(dict(inner=[len(fits) - 3, len(fits) - 2], final=len(fits) - 1))
        ylab = {}
        for c in ("eeg", "noise", "shuffle"):
            ylab[c] = [y[f[0]] if c != "shuffle" else y[f[0]][torch.randperm(len(f[0]), generator=gperm, device=DEV)] for f in fits]
        mem, eu = [], []
        if a.within_eegnet:
            for k in range(nP):
                for oo in range(3):
                    pa, pb = [q for q in range(3) if q != oo]
                    ia, ib, io = [torch.where((sid == k) & (pair == q))[0] for q in (pa, pb, oo)]
                    for tw, off in (("eeg", 0), ("noise", N)):
                        mem.append((ia + off, ib + off)); mem.append((torch.cat([ia, ib]) + off, io + off)); eu.append(dict(inner=[len(mem) - 2], final=len(mem) - 1, twin=tw))
        g2 = None
        if a.check_sklearn:                                   # G2: Newton solution vs sklearn on 3 fits of bp (CPU, validation only)
            from sklearn.linear_model import LogisticRegression
            Z, _ = kernel_fits(FE["bp"], fits[:3], ylab["eeg"][:3], LAMS[8:9]); g2 = []
            for j in range(3):
                tr, ev = fits[j]; A, E_ = FE["bp"][tr], FE["bp"][ev]; mu, sd = A.mean(0), A.std(0) + 1e-8; Fd = A.shape[1]
                A = ((A - mu) / sd / math.sqrt(Fd)).cpu().numpy(); E_ = ((E_ - mu) / sd / math.sqrt(Fd)).cpu().numpy()
                m = LogisticRegression(C=1 / (LAMS[8] * len(tr)), tol=1e-12, max_iter=100000).fit(A, ylab["eeg"][j].cpu().numpy())
                g2.append(float(np.abs(m.predict_proba(E_) - torch.softmax(Z[j][0], -1).cpu().numpy()).max()))
            log("G2 sklearn max |p diff| per fit:", g2)
        R, diag = run_design("secondary_within", units, fits, {"eeg": FE, "noise": FN, "shuffle": FE}, y, ylab, sid, nP,
                             lambda Xf, fi, yl: kernel_fits(Xf, fi, yl, LAMS), mem, eu, Xin, yin,
                             dict(steps=a.within_steps, every=20, batch=32, mchunk=72, seed=13))
        o = out / "secondary_within"; o.mkdir(parents=True, exist_ok=True)
        for k, s in enumerate(P): (o / f"sub-{s:03d}.json").write_text(json.dumps(dict(participant=s, **R[k]), indent=1))
        S = summarize("secondary_within", R, P, diag, "SECONDARY: within participant, 56 training trials; weak",
                      dict(g2_sklearn=g2, runtime_s=round(time.time() - t0), excluded={str(k): v for k, v in excl.items()}))
        (o / "summary_mi.json").write_text(json.dumps(S, indent=1)); log("SECONDARY", json.dumps({k: S[k] for k in ("rows", "d6_minus_raw", "g2_lr_ts_acc", "g2_pass")}, indent=1))


if __name__ == "__main__":
    main()
