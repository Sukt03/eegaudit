"""MI card v3 (REVIEWER overnight spec, loop file "## REVIEWER overnight (G1 + bug review)" A, + addendum 18:30): mi_card.py v2
(sha256 6921c84b...) with the ocular-control arms. The v2 rows (raw, raw_nofp, bp, bp_nofp, ts, eegnet) are recomputed unchanged
(same code path, groups, seeds) and printed with the v2 D6 reading. Only the primary LPGO design is run (secondary_within skipped in v3).

v2 description (unchanged): 4 classes = left / right fist imagery (runs 4, 8, 12) and both fists / both feet imagery (runs 6, 10, 14).
Window 0.5-4.0 s after the cue. primary_lpgo: 10 groups = index mod 10 over the sorted QC-passing list; inner validation group
(g+1) mod 10; (lambda, T) / (step, T) chosen jointly on its CE, refit on all 9 groups. Per-participant column z-scoring (label-free,
transductive), PCA-256 on the outer-training rows, exact-Newton multinomial logistic. EEGNet: 40 members, batch 64, 4000 steps,
checkpoint every 100. Bits = log2 p(true) - log2 train prior. Floors: noise input (Gaussian 1-40 Hz, per-participant per-channel SD)
through the same pipeline (eegnet = trained noise twin) and within-participant shuffled labels (logistic rows).

v3 arms (all on the same LPGO path, same groups and seeds):
  A  eog_proxy   R0 = bandpass(X, 1, 40); P = [F7-F8, FT7-FT8, T7-T8, T9-T10, mean(FP1, FPZ, FP2)] -> 0.5-4.0 s -> 20 Hz block means
                 (350 features). Label: ocular proxy (no EOG channels in EEGMMIDB; the derivations also carry frontal/temporal EEG).
  B  *_eogreg    per participant, float64 OLS of each of the 64 channels of R0 on the 5 proxy series + intercept over ALL samples of ALL
                 that participant's trials (label-free; transductive like the z-scoring); R = residual; features() on R (raw, raw_nofp,
                 bp, bp_nofp, ts, and EEGNet on bandpass(R, 4, 40), same members / steps / seeds). The noise input is regressed on its
                 OWN proxies. Contrasts bp/ts/eegnet_eogreg - raw_eogreg: paired CI, sign-flip p, Holm over 3.
  C  time course raw and bp in 0.5-1.5 / 1.5-2.5 / 2.5-4.0 s and 1.5-4.0 s (descriptive); paired bp_t15_40 - raw_t15_40.
  D  raw_hp4     bandpass(X, 4, 40)[0.5-4.0 s] in 20 Hz block means (EEGNet's band; descriptive).
  E  raw_bp      concat(raw_z / sqrt(4480), bp_z / sqrt(320)) (per-participant z-scored blocks), PCA-256; raw_bp - raw paired.
  Lambda / step edge counts split into lower and upper edge.
Summary: v2 D6 reading as pre-stated (with the ocular diagnosis label) next to the REVIEWER's proposed amended D6 primary (B, with E as
the second contrast), reported as "proposed, post-result, awaiting user decision". code_sha256 in every summary.
Usage: python mi_card_v3.py --out DIR [--subjects 1,2,3] [--lpgo_steps 4000] [--mchunk 20]
       python mi_card_v3.py --out DIR --resummarize        (rebuilds summary_mi.json + gate.json from the per-participant files)
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


# ---------------------------------------------------------------- v3: ocular proxies, EOG regression, time course, 4-40 Hz raw
PROXY = [("F7", "F8"), ("FT7", "FT8"), ("T7", "T8"), ("T9", "T10")]; FPM = ["FP1", "FPZ", "FP2"]
TWIN = {"t05_15": (W0, W0 + 160), "t15_25": (W0 + 160, W0 + 320), "t25_40": (W0 + 320, W1), "t15_40": (W0 + 160, W1)}


def proxies(R0, ch):
    """R0 [N, 64, T] (1-40 Hz) -> [N, 5, T]: F7-F8, FT7-FT8, T7-T8, T9-T10, mean(FP1, FPZ, FP2)."""
    ix = {c: i for i, c in enumerate(ch)}
    return torch.stack([R0[:, ix[a]] - R0[:, ix[b]] for a, b in PROXY] + [R0[:, [ix[c] for c in FPM]].mean(1)], 1)


def eog_regress(R0, sid, nP, ch):
    """Per participant: float64 OLS of every channel of R0 on its 5 proxy series + intercept over all samples of all that participant's
    trials (label-free). Returns the residual (float32, same device) and the per-participant variance share removed."""
    out = torch.empty_like(R0); T = R0.shape[-1]; share = []
    for k in range(nP):
        r = torch.where(sid == k)[0]; x = R0[r].to(DEV).double(); Pk = proxies(x, ch)
        A = torch.cat([Pk.permute(0, 2, 1).reshape(-1, 5), torch.ones(len(r) * T, 1, dtype=torch.float64, device=DEV)], 1)
        Y = x.permute(0, 2, 1).reshape(-1, 64); B = torch.linalg.lstsq(A, Y).solution; E = Y - A @ B
        share.append(float(1 - E.pow(2).sum() / (Y - Y.mean(0)).pow(2).sum()))
        out[r] = E.reshape(len(r), T, 64).permute(0, 2, 1).float().to(out.device); del x, Pk, A, Y, E
    return out, share


def features_v3(X, ch):
    """Arms A, C, D from X [N, 64, 960] (GPU): eog_proxy, raw_/bp_ in the TWIN sub-windows, raw_hp4. float64 [N, F]."""
    out = {}; N = len(X); R0 = bandpass(X, 1, 40)
    out["eog_proxy"] = proxies(R0, ch)[:, :, W0:W1].reshape(N, 5, 70, 8).mean(-1).reshape(N, -1).double()
    for nm, (a, b) in TWIN.items(): out[f"raw_{nm}"] = R0[:, :, a:b].reshape(N, 64, -1, 8).mean(-1).reshape(N, -1).double()
    del R0; bpw = {nm: [] for nm in TWIN}
    for lo, hi in BANDS:
        Bd = bandpass(X, lo, hi)
        for nm, (a, b) in TWIN.items(): bpw[nm].append(Bd[:, :, a:b].pow(2).mean(-1).log())
        del Bd
    for nm in TWIN: out[f"bp_{nm}"] = torch.stack(bpw[nm], -1).reshape(N, -1).double()
    out["raw_hp4"] = bandpass(X, 4, 40)[:, :, W0:W1].reshape(N, 64, 70, 8).mean(-1).reshape(N, -1).double()
    return out


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
    return o, float(TGRID[t]), dict(opt_edge=o in (0, len(Zinner) - 1), opt_lo=o == 0, opt_hi=o == len(Zinner) - 1, T_edge=t in (0, len(TGRID) - 1))


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
def run_design(name, units, fits, Fdict, y, ylab_by_cond, sid, nP, engine, eeg_members, eeg_units, Xin, yin, eeg_cfg, eeg_name="eegnet"):
    """units: list of dict(inner=[fit idx], final=fit idx) over the logistic fits; eeg_units: same over EEGNet members.
    Returns per-participant dict rows[feat][cond] = (bits, acc, n) + selection diagnostics."""
    N = len(y); R = {}; diag = dict(edges={}, grad={}, pca_ev={})
    for cond, FS in Fdict.items():
        ylab = ylab_by_cond[cond]
        for fname, Xf in FS.items():
            Z, gm, *extra = engine(Xf, fits, ylab); diag["grad"][f"{cond}|{fname}"] = gm
            if extra: diag["pca_ev"][f"{cond}|{fname}"] = extra[0]
            bt = torch.full((N,), float("nan"), dtype=torch.float64, device=DEV); ok = torch.zeros(N, dtype=torch.float64, device=DEV)
            lr = torch.full((N,), float("nan"), dtype=torch.float64, device=DEV); ne = [0, 0, 0, 0]
            for u in units:
                Zi = torch.cat([Z[i] for i in u["inner"]], 1); yi = torch.cat([y[fits[i][1]] for i in u["inner"]])
                l, T, e = select(Zi, yi); ne[0] += e["opt_edge"]; ne[1] += e["T_edge"]; ne[2] += e["opt_lo"]; ne[3] += e["opt_hi"]
                f = u["final"]; ev = fits[f][1]; Zf = Z[f][l]
                bt[ev] = bits(Zf, y[ev], T, prior_of(ylab[f])); ok[ev] = (Zf.argmax(-1) == y[ev]).double()
                m = y[ev] < 2; lr[ev[m]] = (Zf[m][:, :2].argmax(-1) == y[ev][m]).double()
            diag["edges"][f"{cond}|{fname}"] = dict(lambda_edge=ne[0], lambda_lower_edge=ne[2], lambda_upper_edge=ne[3], T_edge=ne[1], n_units=len(units))
            assert not torch.isnan(bt).any(), f"{name} {cond} {fname}: untested trials"
            for k in range(nP):
                r = sid == k
                R.setdefault(k, {}).setdefault(fname, {})[cond] = dict(bits=float(bt[r].mean()), acc=float(ok[r].mean()), lr_acc=float(lr[r].nanmean()), n_test=int(r.sum()))
            log(f"{name} logistic {cond} {fname}: max|grad| {gm:.1e}; mean bits {float(bt.mean()):+.4f} acc {float(ok.mean()):.3f}")
    if eeg_members:
        Zn = eegnet_members(Xin, yin, eeg_members, **eeg_cfg)
        for tw in ("eeg", "noise"):
            bt = torch.full((N,), float("nan"), dtype=torch.float64, device=DEV); ok = torch.zeros(N, dtype=torch.float64, device=DEV); ne = [0, 0, 0, 0]
            for u in [u for u in eeg_units if u["twin"] == tw]:
                isel, ifin = u["inner"][0], u["final"]
                c, T, e = select(Zn[isel], yin[eeg_members[isel][1]]); ne[0] += e["opt_edge"]; ne[1] += e["T_edge"]; ne[2] += e["opt_lo"]; ne[3] += e["opt_hi"]
                ev = eeg_members[ifin][1]; evy = ev % N; Zf = Zn[ifin][c]
                bt[evy] = bits(Zf, yin[ev], T, prior_of(yin[eeg_members[ifin][0]])); ok[evy] = (Zf.argmax(-1) == yin[ev]).double()
            diag["edges"][f"{tw}|{eeg_name}"] = dict(step_edge=ne[0], step_lower_edge=ne[2], step_upper_edge=ne[3], T_edge=ne[1])
            assert not torch.isnan(bt).any(), f"{name} {eeg_name} {tw}: untested trials"
            for k in range(nP):
                r = sid == k; R.setdefault(k, {}).setdefault(eeg_name, {})[tw] = dict(bits=float(bt[r].mean()), acc=float(ok[r].mean()), n_test=int(r.sum()))
            log(f"{name} {eeg_name} {tw}: mean bits {float(bt.mean()):+.4f} acc {float(ok.mean()):.3f}")
    return R, diag


V2_ROWS = ["raw", "raw_nofp", "bp", "bp_nofp", "ts", "eegnet"]
EOG_ROWS = [f"{r}_eogreg" for r in V2_ROWS]
TC_ROWS = [f"{b}_{w}" for b in ("raw", "bp") for w in TWIN]
ROWS = V2_ROWS + ["eog_proxy"] + EOG_ROWS + TC_ROWS + ["raw_hp4", "raw_bp"]
LABELS = dict(
    raw_nofp="Fp/AF + F7/F8/FT7/FT8 removed (no EOG in EEGMMIDB)", bp_nofp="Fp/AF + F7/F8/FT7/FT8 removed (no EOG in EEGMMIDB)",
    eog_proxy="ocular proxy (no EOG channels in EEGMMIDB; the derivations also carry frontal/temporal EEG): F7-F8, FT7-FT8, T7-T8, T9-T10, "
              "mean(FP1, FPZ, FP2), 1-40 Hz, 0.5-4.0 s, 20 Hz block means",
    eogreg="per-participant float64 OLS of all 64 channels (1-40 Hz, whole -1..5 s epoch, all trials) on the 5 proxies + intercept; residual "
           "fed to the v2 pipeline (label-free, transductive); noise input regressed on its own proxies",
    time_course="descriptive: raw (1-40 Hz, 20 Hz block means) and bp (log band power) in the named sub-window after the cue (t05_15 = 0.5-1.5 s)",
    raw_hp4="descriptive: raw at 4-40 Hz (EEGNet's band), 0.5-4.0 s, 20 Hz block means",
    raw_bp="raw + bp: per-participant z-scored blocks scaled by 1/sqrt(4480) and 1/sqrt(320), concatenated, PCA-256",
    eegnet_noise="trained noise twin", noise="noise input (per-participant per-channel SD)", shuffle="labels permuted within participant")


def paired_rows(R, P, a, b):
    d = [R[k][a]["eeg"]["bits"] - R[k][b]["eeg"]["bits"] for k in range(len(P))]
    return dict(d=float(np.mean(d)), ci=boot(d), n_pos=int(sum(x > 0 for x in d)), n=len(d), p_signflip=signflip_p(d))


def holm_family(R, P, pairs):
    out = {f"{a}_minus_{b}": paired_rows(R, P, a, b) for a, b in pairs}
    for key, ph in zip(out, holm([v["p_signflip"] for v in out.values()])): out[key]["p_holm"] = ph
    return out


def write_summary(o, v2_dir):
    """Rebuild summary_mi.json and gate.json from <o>/meta.json + per-participant files (the summary path)."""
    meta = json.loads((o / "meta.json").read_text()); P = meta["participants"]
    R = [json.loads((o / f"sub-{s:03d}.json").read_text()) for s in P]
    S = dict(meta, experiment="MI card v3 (EEGMMIDB, 4-class imagery) - primary_lpgo, with the REVIEWER ocular-control arms A-E",
             code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), labels=LABELS, rows={})
    for r in [r for r in ROWS if r in R[0]]:
        for cond in R[0][r]:
            v = [R[k][r][cond]["bits"] for k in range(len(P))]
            row = dict(mean=float(np.mean(v)), ci=boot(v), acc=float(np.mean([R[k][r][cond]["acc"] for k in range(len(P))])))
            if "lr_acc" in R[0][r][cond]: row["lr_acc"] = float(np.nanmean([R[k][r][cond]["lr_acc"] for k in range(len(P))]))
            if cond != "eeg": row["g3"] = dict(n_abs_le_0p02=int(sum(abs(x) <= 0.02 for x in v)), n=len(v))
            S["rows"].setdefault(r, {})[cond] = row
    S["noise_floors"] = {r: S["rows"][r]["noise"]["mean"] for r in S["rows"] if "noise" in S["rows"][r]}
    S["floors_ok"] = all(abs(v) <= 0.02 for v in S["noise_floors"].values())
    S["shuffle_floors"] = {r: S["rows"][r]["shuffle"]["mean"] for r in S["rows"] if "shuffle" in S["rows"][r]}
    S["g2_lr_ts_acc"] = float(np.mean([R[k]["ts"]["eeg"]["lr_acc"] for k in range(len(P))])); S["g2_pass"] = S["g2_lr_ts_acc"] >= 0.60
    # v2 D6 reading, as pre-stated (rows recomputed with the v2 code path)
    v2 = holm_family(R, P, [("bp", "raw"), ("ts", "raw"), ("eegnet", "raw")])
    eog = S["rows"]["eog_proxy"]["eeg"]["mean"]
    S["v2_d6_reading"] = dict(
        rule="pre-stated v2 falsifier: triggered iff the CI lower bounds of bp - raw AND eegnet - raw (0.5-4.0 s) are both <= 0",
        triggered=bool(v2["bp_minus_raw"]["ci"][0] <= 0 and v2["eegnet_minus_raw"]["ci"][0] <= 0), contrasts_holm3=v2,
        bp_nofp_minus_raw_nofp=paired_rows(R, P, "bp_nofp", "raw_nofp"), bp_minus_bp_nofp=paired_rows(R, P, "bp", "bp_nofp"),
        label=f"raw rung dominated by an ocular component: the 5 EOG proxies alone carry {eog:.3f} bits (eog_proxy row); "
              f"eegnet - raw compares different bands (EEGNet input 4-40 Hz; raw at 4-40 Hz = {S['rows']['raw_hp4']['eeg']['mean']:.3f} bits)")
    # arms B, C, E
    S["eogreg_contrasts_holm3"] = holm_family(R, P, [("bp_eogreg", "raw_eogreg"), ("ts_eogreg", "raw_eogreg"), ("eegnet_eogreg", "raw_eogreg")])
    S["time_course_bp_minus_raw_t15_40"] = dict(paired_rows(R, P, "bp_t15_40", "raw_t15_40"), label="descriptive, 1.5-4.0 s")
    S["increment_raw_bp_minus_raw"] = paired_rows(R, P, "raw_bp", "raw")
    B = S["eogreg_contrasts_holm3"]
    S["amended_d6_proposed"] = dict(
        status="proposed, post-result, awaiting user decision",
        source="REVIEWER overnight 21 Nov, loop file section A: B as the amended MI D6 primary, E as the second contrast",
        label="amended after the 0.5-4 s rows showed an ocular raw rung",
        primary_B=B, second_contrast_E=S["increment_raw_bp_minus_raw"],
        v2_rule_applied_to_B="triggered iff the CI lower bounds of bp_eogreg - raw_eogreg AND eegnet_eogreg - raw_eogreg are both <= 0 "
                             "(the v2 rule transposed to the regressed rungs; not adopted)",
        v2_rule_applied_to_B_triggered=bool(B["bp_eogreg_minus_raw_eogreg"]["ci"][0] <= 0 and B["eegnet_eogreg_minus_raw_eogreg"]["ci"][0] <= 0))
    rs = meta.get("eog_regression_variance_share", {})
    S["eog_regression_variance_share_mean"] = {c: float(np.mean(v)) for c, v in rs.items()}
    v2s = Path(v2_dir) / "summary_mi.json"
    if v2s.exists():
        V = json.loads(v2s.read_text())
        if V.get("participants") == P:
            S["g2_vs_v2_run"] = dict(source=str(v2s), v2_code_sha256=V.get("code_sha256"),
                                     diff={r: S["rows"][r]["eeg"]["mean"] - V["rows"][r]["eeg"]["mean"] for r in V["rows"] if r in S["rows"]})
    (o / "summary_mi.json").write_text(json.dumps(S, indent=1))
    G = dict(pass_floors=S["floors_ok"], noise_floors=S["noise_floors"], code_sha256=S["code_sha256"], n=len(P),
             smoke_pass_floors=all(abs(v) <= 0.03 for v in S["noise_floors"].values()),
             criteria="pass_floors (full-run gate): every row |noise group mean| <= 0.02 bits. smoke_pass_floors (6-participant smoke "
                      "only): every row |noise mean| <= 0.03, because at n = 6 (5 training participants per fold) the UNCHANGED v2 code "
                      "gives bp / bp_nofp noise floors -0.022 / -0.029, identical to the G1-passed v2 smoke (mi_smoke3.log); at n = 105 "
                      "the v2 floors are <= 0.001. The chain also requires rc = 0.")
    (o / "gate.json").write_text(json.dumps(G, indent=1))
    return S


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--subjects", default="")
    ap.add_argument("--lpgo_steps", type=int, default=4000); ap.add_argument("--mchunk", type=int, default=20)
    ap.add_argument("--lchunk", type=int, default=4, help="lambdas per batched Newton call (speed/memory only; 17 = all, for small smokes)")
    ap.add_argument("--v2_dir", default=str(ROOT / "results/ext/mi_card/primary_lpgo")); ap.add_argument("--resummarize", action="store_true")
    a = ap.parse_args(); out = Path(a.out); o = out / "primary_lpgo"
    if a.resummarize:
        S = write_summary(o, a.v2_dir); log("SUMMARY", json.dumps({k: S[k] for k in ("rows", "v2_d6_reading", "amended_d6_proposed", "floors_ok", "g2_pass")}, indent=1)); return
    cache = ROOT / "results/ext/mi_card_cache"; cache.mkdir(parents=True, exist_ok=True)
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
    assert all(c in ch for pr in PROXY for c in pr) and all(c in ch for c in FPM), "proxy channel names not all found"
    X = torch.tensor(np.concatenate([D[s]["X"] for s in P]), dtype=torch.float32, device=DEV)
    y = torch.tensor(np.concatenate([D[s]["y"] for s in P]), device=DEV)
    sid = torch.tensor(np.concatenate([[k] * len(D[s]["y"]) for k, s in enumerate(P)]), device=DEV); N = len(y); sidc = sid.cpu()
    g = torch.Generator(device=DEV); g.manual_seed(20261121)
    Xb = bandpass(X, 1, 40); Nz = bandpass(torch.randn(X.shape, generator=g, device=DEV), 1, 40)      # = v2 noise draw
    for k in range(nP):
        r = sid == k; Nz[r] = Nz[r] / Nz[r].std((0, 2), keepdim=True) * Xb[r].std((0, 2), keepdim=True)
    del Xb; Nz = Nz.cpu(); torch.cuda.empty_cache(); share = {}

    def build(Z, tag):                                        # Z: GPU input -> CPU feature dict, EEGNet inputs (plain, regressed) on CPU
        Fd, Xi = features(Z, keep_nofp); Fd = {k: v.cpu() for k, v in Fd.items()}; Xi = Xi.cpu()
        Fd.update({k: v.cpu() for k, v in features_v3(Z, ch).items()})
        Rr, share[tag] = eog_regress(bandpass(Z, 1, 40), sid, nP, ch); torch.cuda.empty_cache()
        Fr, Xr = features(Rr, keep_nofp); del Rr
        Fd.update({f"{k}_eogreg": v.cpu() for k, v in Fr.items()}); del Fr; torch.cuda.empty_cache()
        log(f"features {tag}: eog regression removes {np.mean(share[tag]):.3f} of the variance (mean over participants)")
        return Fd, Xi, Xr.cpu()
    FE, XinE, XrE = build(X, "eeg"); del X; torch.cuda.empty_cache()
    FN, XinN, XrN = build(Nz.to(DEV), "noise"); del Nz; torch.cuda.empty_cache()
    log("features", {k: tuple(v.shape) for k, v in FE.items()})

    def zpp(FS):                                              # per-participant column z-scoring (label-free), in place, on CPU
        for fn, Xf in FS.items():
            for k in range(nP):
                r = sidc == k; Xf[r] = (Xf[r] - Xf[r].mean(0)) / (Xf[r].std(0) + 1e-8)
        FS["raw_bp"] = torch.cat([FS["raw"] / math.sqrt(FS["raw"].shape[1]), FS["bp"] / math.sqrt(FS["bp"].shape[1])], 1)   # arm E
        return FS
    FEz, FNz = zpp(FE), zpp(FN)
    order = [r for r in ROWS if r in FEz]; FEz = {r: FEz[r] for r in order}; FNz = {r: FNz[r] for r in order}

    def eeg_in(A, B_):                                        # per-participant scalar scale (label-free), as v2
        Xin = torch.cat([A, B_])
        for k in range(nP):
            r = torch.where(sidc == k)[0]; sc = Xin[r].std(); Xin[r] /= sc; Xin[r + N] /= sc
        return Xin
    yin = torch.cat([y, y])
    gperm = torch.Generator(device=DEV); gperm.manual_seed(7)
    ysh = y.clone()                                           # labels permuted within each participant (shuffle floor), as v2
    for k in range(nP):
        r = torch.where(sid == k)[0]; ysh[r] = y[r[torch.randperm(len(r), generator=gperm, device=DEV)]]
    G = min(10, nP); grp = torch.tensor([k % G for k in range(nP)], device=DEV)[sid]
    for k in range(nP):
        c = torch.bincount(y[sid == k], minlength=4); assert int(c.max() - c.min()) <= 2, f"sub-{P[k]:03d} class counts {c.tolist()}"
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
    eng = lambda Xf, fi, yl: primal_fits(Xf.to(DEV), fi, yl, LAMS, pca_rows, lchunk=a.lchunk)
    ecfg = dict(steps=a.lpgo_steps, every=100, batch=64, mchunk=a.mchunk, seed=11)
    Xin = eeg_in(XinE, XinN).to(DEV); del XinE, XinN
    R, diag = run_design("primary_lpgo", units, fits, {"eeg": FEz, "noise": FNz, "shuffle": FEz}, y, ylab, sid, nP, eng, mem, eu, Xin, yin, ecfg)
    del Xin, FEz, FNz; torch.cuda.empty_cache()
    Xin = eeg_in(XrE, XrN).to(DEV); del XrE, XrN
    R2, diag2 = run_design("primary_lpgo", [], fits, {}, y, ylab, sid, nP, eng, mem, eu, Xin, yin, ecfg, eeg_name="eegnet_eogreg")
    del Xin; torch.cuda.empty_cache()
    for k in range(nP): R[k].update(R2[k])
    diag["edges"].update(diag2["edges"])
    o.mkdir(parents=True, exist_ok=True)
    for k, s in enumerate(P): (o / f"sub-{s:03d}.json").write_text(json.dumps(dict(participant=s, group=k % G, **R[k]), indent=1))
    meta = dict(label="PRIMARY: leave-participant-group-out (10 groups), per-participant standardisation (label-free, transductive), PCA-256",
                n=nP, participants=P, groups=G, steps=a.lpgo_steps, mchunk=a.mchunk, window="0.5-4.0 s after the cue (v2 rows and arms A, B, D, E)",
                unit="bits per trial (log2 p - log2 train prior)", ceiling="none (no repeated stimuli; reported against floors and twins only)",
                design_note="secondary_within design not run in v3", eog_regression_variance_share=share, diag=diag,
                runtime_s=round(time.time() - t0), excluded={str(k): v for k, v in excl.items()})
    (o / "meta.json").write_text(json.dumps(meta, indent=1))
    S = write_summary(o, a.v2_dir)
    log("PRIMARY", json.dumps({k: S[k] for k in ("rows", "v2_d6_reading", "amended_d6_proposed", "eogreg_contrasts_holm3", "floors_ok", "g2_pass")}, indent=1))


if __name__ == "__main__":
    main()
