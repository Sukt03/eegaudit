"""FRONTIER SMOKES (EXECUTOR-5, 21 Nov 2026; user ask 05:05 "write smoke scripts for these methods and run them on the GPU").
Nine information/geometry measures on the M3 grid (experiments/ext/m3_yield_map.py, regions.py), ONE participant per modality:
  image = THINGS-EEG2 sub-01 (target CLIP ViT-H/14; top PCs where a scalar / low-dimensional target is needed),
  pe    = ds003458 sub-001 (target residualised PE; normal scores for the scalar-target methods; M3's raw target for method 4).
Cells = 8 named regions x 5 bands x windows (image 10 x 100 ms from onset; PE 6 x 100 ms after feedback), M3 "combined" features
(power + 5 voltage bins per channel, 36 per cell), built with M3's own functions (band_power, band_voltage, window_feats); the
read-out/score estimators are M3's (image_batch, soft_bits_gpu, pe_gain_batch) or are checked against them (G2).
All numerical work is torch on CUDA, batched over cells x nulls x SNRs x projections x members. CPU = file I/O + bookkeeping.
Nulls: image = target permutation within the train / test split; PE = circular shifts of the target (round(n k / 24), k = 2..22).
Methods:
 1 matrix-based Renyi-alpha MI (alpha 1.01, 2), normalised RBF Gram matrices, median-heuristic width x {0.5, 1, 2}.
 2 conditional I-MMSE: I(Y;Z) = 1/2 int [mmse(Y|Y_snr) - mmse(Y|Y_snr,Z)] dsnr (Guo-Shamai-Verdu 2005; Kong et al. ICLR 2023).
   NOTE: the pasted formula I = 1/2 int MMSE(Y|Z,snr) dsnr is NOT the I-MMSE identity (it is the conditional entropy-type term only);
   the correct difference form is implemented. First term analytic 1/(1+snr) for the unit-variance normal-scores target (an
   empirical held-out first term is reported beside it as a bias check); second term = held-out ridge denoiser on [Y_snr, Z_cell].
 3 sliced-Wasserstein dependence: SW2^2 between (Z_cell, Y) and (Z_cell, Y_pi0), 256 random projections.
 4 causal cell patching for a linear read-out on all cells concatenated (M3 scorers); donors same vs different category proxy.
 5 Gaussian PID (MMI redundancy) for all 28 pairs of the 8 top M3 cells, Gaussian MI from held-out ridge predictions.
 6 Koopman / DMD per region (PCA-20 of the 5-band region features per window, linear operator over consecutive windows);
   M3 yield of slow-mode vs fast-mode amplitudes.
 7 pullback Riemannian volume 1/2 logdet(J J^T + eps I) of a member-batched tanh MLP read-out (torch.func vmap + jacrev);
   Spearman with per-trial yield; a linear read-out gives a constant volume (sanity check).
 8 discrete Hodge decomposition of the antisymmetric lagged cross-covariance flow on the region x time grid; null = time reversal.
 9 persistent homology beta0 (MST edge lengths) of class-mean cell latents; beta1 only if ripser/gudhi is installed (it is not).
Units: methods 1-3, 5 in nats; methods 4, 6, 7 yields in bits as M3 (image: bits per query; PE: bits per trial).
Usage: python frontier_smokes.py --out DIR [--modality pe|image|both]
"""
import argparse, json, subprocess, sys, threading, time, traceback
from pathlib import Path
import numpy as np
import torch

ROOT = Path("/path/to/eegaudit")
for p in (ROOT, ROOT / "experiments/ext", ROOT / "experiments/eegbits", ROOT / "experiments/agent_loop_20261120", ROOT / "experiments/standard"):
    sys.path.insert(0, str(p))
import m3_yield_map as M3
import regions as RG
from belief_bottleneck import LAM, folds

DEV = "cuda"; F64 = torch.float64
torch.backends.cuda.matmul.allow_tf32 = True


# ------------------------------------------------------------------ bookkeeping
class GpuMon:
    """nvidia-smi device utilisation sampled every second (the device is shared with other steps: this is whole-GPU use)."""
    def __init__(s):
        s.samples, s.stop = [], False; threading.Thread(target=s.run, daemon=True).start()
    def run(s):
        while not s.stop:
            try:
                o = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
                                   capture_output=True, text=True, timeout=5).stdout.strip().splitlines()
                o = next(l for l in o if "N/A" not in l); u, m = (float(x) for x in o.split(",")); s.samples.append((time.time(), u, m))
            except Exception:
                pass
            time.sleep(1.0)
    def window(s, t0, t1):
        v = [(u, m) for t, u, m in s.samples if t0 <= t <= t1]
        if not v: return dict(util_mean=None, util_max=None, n=0)
        u = [a for a, _ in v]; return dict(util_mean=round(float(np.mean(u)), 1), util_max=float(max(u)), mem_used_max_mib=float(max(m for _, m in v)), n=len(v))


def spearman(a, b):
    a, b = torch.as_tensor(a, dtype=F64, device=DEV), torch.as_tensor(b, dtype=F64, device=DEV)
    ra, rb = a.argsort().argsort().double(), b.argsort().argsort().double(); ra, rb = ra - ra.mean(), rb - rb.mean()
    return float((ra * rb).sum() / (ra.norm() * rb.norm()))


def zp(real, null):
    null = torch.as_tensor(null, dtype=F64).flatten().cpu(); r = float(real); m, sd = float(null.mean()), float(null.std())
    return dict(real=r, null_mean=m, null_sd=sd, z=(r - m) / sd if sd > 0 else None, p=float((1 + (null >= r).sum()) / (1 + len(null))), n_null=len(null))


def map_summary(real, null, D, m3=True):
    """real (C,), null (K, C) per-cell values -> max-cell (FWER max-statistic) summary, per-cell perm p, Spearman with the M3 map."""
    real, null = real.double(), null.double(); i = int(real.argmax()); nmax = null.max(1).values
    pc = (1 + (null >= real[None]).sum(0)).double() / (1 + null.shape[0])
    out = dict(max_cell=D["cells"][i], **{f"max_{k}": v for k, v in zp(real.max(), nmax).items()},
               median_cell_real=float(real.median()), null_mean_all_cells=float(null.mean()), n_cells_p05=int((pc < 0.05).sum()), n_cells=len(real))
    if m3: out["spearman_with_m3_yield"] = spearman(real, D["m3"])
    return out


def zstd(X, rows=None):
    """z-score along dim -2 using the stats of `rows` (ddof 0)."""
    A = X if rows is None else X[..., rows, :]
    return (X - A.mean(-2, keepdim=True)) / (A.std(-2, unbiased=False, keepdim=True) + 1e-6)


def nscore(v):
    r = v.argsort().argsort().double(); z = torch.special.ndtri((r + 0.5) / len(v)); return (z - z.mean()) / z.std(unbiased=False)


def ridge_ho(F, Y, splits, lam=LAM):
    """Held-out ridge predictions for B members: F (B, N, k), Y (N, J) -> (B, N, J); in-split z-scoring (ddof 0), penalty lam * n_train,
    centred target (same algebra as M3.pe_gain_batch's pred, plus the training mean); dual form when n_train < k."""
    F = F.to(F64); Y = Y.to(F64); B, N, k = F.shape; out = torch.full((B, N, Y.shape[1]), float("nan"), dtype=F64, device=DEV)
    for tr, te in splits:
        A, Bt = F[:, tr], F[:, te]; mu, sd = A.mean(1, keepdim=True), A.std(1, unbiased=False, keepdim=True) + 1e-6
        A, Bt = (A - mu) / sd, (Bt - mu) / sd; ym = Y[tr].mean(0); Yc = (Y[tr] - ym).expand(B, -1, -1); l = lam * len(tr)
        if k <= len(tr): W = torch.linalg.solve(A.mT @ A + l * torch.eye(k, dtype=F64, device=DEV), A.mT @ Yc)
        else: W = A.mT @ torch.linalg.solve(A @ A.mT + l * torch.eye(len(tr), dtype=F64, device=DEV), Yc)
        out[:, te] = Bt @ W + ym
    return out


def gauss_mi(P, Y, rows):
    """Gaussian MI (nats) from held-out predictions: -1/2 ln(1 - r+^2), r+ = max(corr, 0). P (B, N, J), Y (N, J) -> (B, J)."""
    p, y = P[:, rows], Y[rows].to(F64); p = p - p.mean(1, keepdim=True); y = y - y.mean(0)
    r = ((p * y).sum(1) / (p.norm(dim=1) * y.norm(dim=0) + 1e-300)).clamp(0, 0.999999); return -0.5 * torch.log1p(-r ** 2)


def soft_per_query(S):
    """Per-query version of M3.soft_bits_gpu (same LOO temperature rule over run._TAUS): S (B, n, n) -> (B, n); mean == soft_bits_gpu."""
    import run as R
    T = torch.tensor(R._TAUS, dtype=F64, device=DEV); n = S.shape[-1]; Z = S[:, None].double() / T[None, :, None, None]
    CE = -torch.diagonal(torch.log_softmax(Z, -1), dim1=-2, dim2=-1) / np.log(2); loo = (CE.sum(-1, keepdim=True) - CE) / (n - 1)
    return np.log2(n) - torch.gather(CE, 1, loo.argmin(1, keepdim=True))[:, 0]


def cosS(P, Yn):
    P = P / P.norm(dim=-1, keepdim=True); return P @ Yn.T


# ------------------------------------------------------------------ data (M3 feature functions; one load per modality)
def gather_cells(P, V, sets, nw):
    """P (N, C, 5, W) log-power, V (N, C, 5, W, 5) -> (cells, N, 36) combined features (M3 cell_feats order: 6 power | 30 voltage)."""
    grid = [(r, chs, b, w) for r, chs in sets.items() for b in range(5) for w in range(nw)]
    CH = torch.as_tensor([g[1] for g in grid], device=DEV); BI = torch.as_tensor([g[2] for g in grid], device=DEV); WI = torch.as_tensor([g[3] for g in grid], device=DEV)
    p = P[:, CH, BI[:, None], WI[:, None]].permute(1, 0, 2); v = V[:, CH, BI[:, None], WI[:, None], :].permute(1, 0, 2, 3).flatten(2)
    return torch.cat([p, v], 2).contiguous(), grid


def region_signal(V, sets):
    """Broadband (sum of the 5 band-passed voltages) region-mean signal in 20-ms bins: (N, 8, W, 5) for the Hodge flow."""
    return torch.stack([V[:, chs].sum(2).mean(1) for chs in sets.values()], 1)


def m3_lookup(path, cells, tmap):
    j = json.loads(Path(path).read_text()); lut = {(c["region"], c["band"], c["t_ms"], c["kind"]): c["yield_"] for c in j["cells"]}
    return torch.tensor([lut[(c[0], c[1], tmap(c[2]), "combined")] for c in cells], dtype=F64, device=DEV), j


def null_index(n, kind, K, tr=None, te=None, seed=300):
    """(K, n) index rows: y_null = y[idx]. 'shift': torch.roll by round(n k / 24), k = 2..K+1; 'perm': permutation within tr and within te."""
    ar = torch.arange(n, device=DEV)
    if kind == "shift": return torch.stack([(ar - round(n * k / 24)) % n for k in range(2, K + 2)])
    out = []
    for k in range(K):
        g = torch.Generator(device=DEV); g.manual_seed(seed + k); idx = ar.clone()
        for part in (tr, te): idx[part] = part[torch.randperm(len(part), generator=g, device=DEV)]
        out.append(idx)
    return torch.stack(out)


def load_pe(s=1):
    from i2_complementarity import targets
    t0 = time.time(); evaluate, _, fpath, _ = targets(); sub, fz, tri, n, Ys, X, ch = M3.pe_load(s, evaluate, fpath)
    sf, T = 200.0, 400; Mw, nfft, Wf = M3.morlet_bank(sf, T); nfv = int(2 ** np.ceil(np.log2(3 * T))); G = M3.bandpass_gains(sf, nfv); Ps, Vs = [], []
    for i in range(0, n, 64):
        xg = torch.tensor(X[i:i + 64], device=DEV)
        P, V = M3.window_feats(M3.band_power(xg, sf, Mw, nfft, Wf), M3.band_voltage(xg, sf, G, nfv), 160, 20, 8)
        Ps.append(torch.log10(P + 1e-20)); Vs.append(V)
    P, V = torch.cat(Ps)[..., 2:], torch.cat(Vs)[:, :, :, 2:]          # windows 0..600 ms after feedback (M3 w = 2..7)
    sets = {k: RG.indices(ch, v) for k, v in RG.REGIONS.items()}; Z, grid = gather_cells(P, V, sets, 6)
    cells = [(r, M3.BNAMES[b], 100 * w) for r, _, b, w in grid]
    m3, j = m3_lookup(ROOT / f"results/ext/m3/pe_{sub}.json", cells, lambda t: t)
    fo = folds(n); splits = [(torch.as_tensor(np.where(fo != b)[0], device=DEV), torch.as_tensor(np.where(fo == b)[0], device=DEV)) for b in range(10)]
    y_raw = torch.as_tensor(Ys[:, 0], dtype=F64, device=DEV); ys = nscore(y_raw)
    nidx = null_index(n, "shift", 21)
    D = dict(mod="pe", sub=sub, n=n, Z=Z, cells=cells, sets=sets, ch=ch, m3=m3, m3_json=j, splits=splits, eval_rows=torch.arange(n, device=DEV),
             fit_rows=torch.arange(n, device=DEV), y_raw=y_raw, ys=ys, nidx=nidx, Ykern=ys[:, None], Ylow=ys[:, None], W=6,
             sig=region_signal(V, sets), labels_src=ys)
    print(sub, "PE loaded + features", tuple(Z.shape), f"{time.time()-t0:.0f}s", flush=True); return D


def image_feats(X, reps, sf, k):
    """M3.image_subject's inner feats() (not importable), same primitives and chunking: power = single-repetition power averaged over
    repetitions; voltage on the repetition mean; 10 windows of 25 samples from onset."""
    Mw, nfft, Wf, G, nfv = k; Ps, Vs = [], []; step = max(1, 800 // reps)
    for i in range(0, len(X), step):
        xb = torch.tensor(np.ascontiguousarray(X[i:i + step]), device=DEV); nI = xb.shape[0]
        pw = M3.band_power(xb.reshape(nI * reps, *xb.shape[2:]), sf, Mw, nfft, Wf); pw = pw.reshape(nI, reps, *pw.shape[1:]).mean(1)
        P, V = M3.window_feats(pw, M3.band_voltage(xb.mean(1), sf, G, nfv), 0, 25, 10); Ps.append(torch.log10(P + 1e-20)); Vs.append(V)
    return torch.cat(Ps), torch.cat(Vs)


def load_image(s=1):
    import run as R
    t0 = time.time(); sub, Ytr, Yte, Xtr4, Xte80, ch = M3.image_load(s, R); sf, T = 250.0, 250
    Mw, nfft, Wf = M3.morlet_bank(sf, T); nfv = int(2 ** np.ceil(np.log2(3 * T))); G = M3.bandpass_gains(sf, nfv); kk = (Mw, nfft, Wf, G, nfv)
    sets = {k: RG.indices(ch, v) for k, v in RG.REGIONS.items()}
    Ptr, Vtr = image_feats(Xtr4, 4, sf, kk); Ztr, grid = gather_cells(Ptr, Vtr, sets, 10); sig = region_signal(Vtr, sets); del Ptr, Vtr
    Pte, Vte = image_feats(Xte80, 80, sf, kk); Zte, _ = gather_cells(Pte, Vte, sets, 10); del Pte, Vte; del Xtr4
    ntr = Ztr.shape[1]; Z = torch.cat([Ztr, Zte], 1); del Ztr, Zte
    cells = [(r, M3.BNAMES[b], 100 * w) for r, _, b, w in grid]; m3, j = m3_lookup(ROOT / f"results/ext/m3/image_{sub}.json", cells, lambda t: t)
    tr, te = torch.arange(ntr, device=DEV), torch.arange(ntr, ntr + 200, device=DEV)
    Yall = torch.as_tensor(np.concatenate([Ytr, Yte]), dtype=F64, device=DEV); Yc = Yall - Yall[tr].mean(0)
    ev, U = torch.linalg.eigh(Yc[tr].T @ Yc[tr] / ntr); U = U[:, ev.argsort(descending=True)[:16]]; pcs = Yc @ U
    pc1 = torch.empty(len(Yall), dtype=F64, device=DEV); pc1[tr], pc1[te] = nscore(pcs[tr, 0]), nscore(pcs[te, 0])
    rng = torch.Generator(device=DEV); rng.manual_seed(0); sub_rows = tr[torch.randperm(ntr, generator=rng, device=DEV)]
    D = dict(mod="image", sub=sub, n=len(Yall), ntr=ntr, Z=Z, cells=cells, sets=sets, ch=ch, m3=m3, m3_json=j, splits=[(tr, te)], eval_rows=te,
             fit_rows=tr, ys=pc1, nidx=null_index(len(Yall), "perm", 21, tr, te), Ykern=Yall / Yall.norm(dim=1, keepdim=True),
             Ylow=zstd(pcs, tr), sub_rows=sub_rows, W=10, sig=sig, Ytr=Yall[tr], Yte=Yall[te], Xte80=Xte80, kk=kk, sf=sf, pcs=pcs)
    print(sub, "image loaded + features", tuple(Z.shape), f"{time.time()-t0:.0f}s", flush=True); return D


def rows_small(D, n_img):
    return D["sub_rows"][:n_img] if D["mod"] == "image" else torch.arange(D["n"], device=DEV)


def small_nulls(D, rows, K, seed=500):
    """Nulls on a row subset: PE circular shifts (rows = all trials), image random permutations of the subset."""
    n = len(rows)
    if D["mod"] == "pe": return null_index(n, "shift", K)
    g = torch.Generator(device=DEV); g.manual_seed(seed); return torch.stack([torch.randperm(n, generator=g, device=DEV) for _ in range(K)])


# ------------------------------------------------------------------ 1. matrix-based Renyi-alpha MI
def gram_norm(X, f):
    D2 = torch.cdist(X, X) ** 2; n = X.shape[-2]; iu = torch.triu_indices(n, n, 1, device=DEV)
    med = D2[..., iu[0], iu[1]].sqrt().median(-1).values; return torch.exp(-D2 / (2 * (f * med)[..., None, None] ** 2)) / n


def renyi(A, a):
    if a == 2: return -torch.log((A ** 2).sum((-1, -2)))
    lam = torch.linalg.eigvalsh(A).clamp(min=0); return torch.log((lam ** a).sum(-1)) / (1 - a)


def m1_renyi(D, n_img=512, K=21, cc=8):
    rows = rows_small(D, n_img); n = len(rows); Z = zstd(D["Z"][:, rows].double()).float(); Y = D["Ykern"][rows].float()
    idx = torch.cat([torch.arange(n, device=DEV)[None], small_nulls(D, rows, K)]); C = Z.shape[0]; out = {}
    for f in (0.5, 1.0, 2.0):
        B = gram_norm(Y, f); Bq = B[idx[:, :, None], idx[:, None, :]]; I = {a: torch.zeros(C, K + 1, dtype=F64, device=DEV) for a in (1.01, 2)}
        for c0 in range(0, C, cc):
            A = gram_norm(Z[c0:c0 + cc], f); J = A[:, None] * Bq[None]; J = J / torch.diagonal(J, dim1=-2, dim2=-1).sum(-1)[..., None, None]
            for a in (1.01, 2): I[a][c0:c0 + cc] = (renyi(A, a)[:, None] + renyi(B, a) - renyi(J, a)).double()
        for a in (1.01, 2): out[f"alpha{a}_width{f}"] = map_summary(I[a][:, 0], I[a][:, 1:].T, D)
    prim = out["alpha1.01_width1.0"]; return dict(n_rows=n, n_null=K, unit="nats", detail=out, headline=dict(stat="max-cell I_alpha=1.01 (median width)", **{k: prim[f"max_{k}"] for k in ("real", "null_mean", "z", "p")}, spearman_m3=prim["spearman_with_m3_yield"]))


# ------------------------------------------------------------------ 2. conditional I-MMSE
def m2_immse(D, K=21, nd=4, cc=16, lam0=1.0):
    snr = torch.logspace(-3, 4, 29, dtype=F64, device=DEV); ls = torch.log(snr); rs = snr.sqrt()
    trap = lambda g: ((g[..., 1:] + g[..., :-1]) / 2 * (ls[1:] - ls[:-1])).sum(-1)
    rho = 0.3; chk = 0.5 * trap(snr * (1 / (1 + snr) - 1 / (1 / (1 - rho ** 2) + snr))); exact = -0.5 * np.log(1 - rho ** 2)
    Yq = D["ys"][torch.cat([torch.arange(D["n"], device=DEV)[None], D["nidx"][:K]])].T     # (N, Q)
    g = torch.Generator(device=DEV); g.manual_seed(7); E = torch.randn(D["n"], nd, generator=g, device=DEV, dtype=F64)
    C = D["Z"].shape[0]; Q = K + 1; S = len(snr); sse2 = torch.zeros(C, Q, S, dtype=F64, device=DEV); sse1 = torch.zeros(Q, S, dtype=F64, device=DEV); cnt = 0
    for tr, te in D["splits"]:
        u, ut = Yq[tr], Yq[te]; e, et = E[tr], E[te]; um, em = u.mean(0), e.mean(0); u, ut, e, et = u - um, ut - um, e - em, et - em
        uu, ue, ee = (u * u).sum(0), u.T @ e, (e * e).sum(0)                                                  # (Q), (Q, nd), (nd)
        g00 = snr[None, :, None] * uu[:, None, None] + 2 * rs[None, :, None] * ue[:, None, :] + ee[None, None, :]   # (Q, S, nd)
        r0 = rs[None, :, None] * uu[:, None, None] + ue[:, None, :]                                          # sum_i Ysnr_i u_i (Q, S, nd)
        ysn_te = rs[None, :, None, None] * ut.T[:, None, None, :] + et.T[None, None, :, :]                   # (Q, S, nd, nte)
        w1 = r0 / g00; sse1 += ((ut.T[:, None, None, :] - w1[..., None] * ysn_te) ** 2).sum(-1).mean(-1)       # Ysnr-only denoiser
        for c0 in range(0, C, cc):
            Zc = D["Z"][c0:c0 + cc].double(); mu, sd = Zc[:, tr].mean(1, keepdim=True), Zc[:, tr].std(1, unbiased=False, keepdim=True) + 1e-6
            Zt, Ze = (Zc[:, tr] - mu) / sd, (Zc[:, te] - mu) / sd; c = Zt.shape[0]; k = Zt.shape[2]
            Zu, Zee = Zt.mT @ u, Zt.mT @ e                                                                     # (c,k,Q), (c,k,nd)
            gz0 = rs[None, None, :, None, None] * Zu.permute(0, 2, 1)[:, :, None, None, :] + Zee.permute(0, 2, 1)[:, None, None, :, :]   # (c,Q,S,nd,k)
            M = torch.zeros(c, Q, S, nd, k + 1, k + 1, dtype=F64, device=DEV)
            M[..., 0, 0] = g00[None]; M[..., 0, 1:] = gz0; M[..., 1:, 0] = gz0
            M[..., 1:, 1:] = (Zt.mT @ Zt + lam0 * len(tr) * torch.eye(k, dtype=F64, device=DEV))[:, None, None, None]
            rhs = torch.cat([r0[None].expand(c, -1, -1, -1)[..., None], Zu.permute(0, 2, 1)[:, :, None, None, :].expand(-1, -1, S, nd, -1)], -1)
            w = torch.linalg.solve(M, rhs[..., None])[..., 0]                                                  # (c,Q,S,nd,k+1)
            pred = w[..., :1] * ysn_te[None] + torch.einsum("cik,cqsdk->cqsdi", Ze, w[..., 1:])
            sse2[c0:c0 + cc] += ((ut.T[None, :, None, None, :] - pred) ** 2).sum(-1).mean(-1)
        cnt += len(te)
    m2, m1 = sse2 / cnt, sse1 / cnt
    I_an = 0.5 * trap(snr * (1 / (1 + snr) - m2)); I_emp = 0.5 * trap(snr * (m1[None] - m2))                 # (C, Q)
    out = dict(analytic_first_term=map_summary(I_an[:, 0], I_an[:, 1:].T, D), empirical_first_term=map_summary(I_emp[:, 0], I_emp[:, 1:].T, D),
               integration_check=dict(rho=rho, trapezoid=float(chk), exact=exact), snr_grid="logspace(-3, 4, 29), trapezoid in log snr",
               denoiser=f"held-out ridge on [Y_snr, z-scored Z_cell], penalty {lam0} x n_train on the Z block, {nd} noise draws",
               formula_note="pasted formula I = 1/2 int MMSE(Y|Z,snr) dsnr is not the I-MMSE identity; implemented I(Y;Z) = 1/2 int [mmse(Y|Y_snr) - mmse(Y|Y_snr,Z)] dsnr",
               linear_note="with a linear denoiser and Gaussian target this equals -1/2 ln(1 - rho^2) of the ridge read-out (Gaussian MI)")
    a = out["analytic_first_term"]; return dict(unit="nats", detail=out, headline=dict(stat="max-cell I (analytic first term)", **{k: a[f"max_{k}"] for k in ("real", "null_mean", "z", "p")}, spearman_m3=a["spearman_with_m3_yield"]))


# ------------------------------------------------------------------ 3. sliced-Wasserstein dependence
def m3_sw(D, n_img=2000, K=21, P=256, cc=16):
    rows = rows_small(D, n_img); n = len(rows); Z = zstd(D["Z"][:, rows].float()); Y = zstd(D["Ylow"][rows].float())
    dz, dy = Z.shape[2], Y.shape[1]; Y = Y * np.sqrt(dz / dy)                      # equal total variance for the two blocks
    g = torch.Generator(device=DEV); g.manual_seed(11); th = torch.randn(P, dz + dy, generator=g, device=DEV); th = th / th.norm(dim=1, keepdim=True)
    pi0 = torch.randperm(n, generator=g, device=DEV); idx = torch.cat([torch.arange(n, device=DEV)[None], small_nulls(D, rows, K, 600)])
    pY = Y @ th[:, dz:].T; pYq = pY[idx]; pY0 = pY[pi0]; C = Z.shape[0]; sw = torch.zeros(C, K + 1, dtype=F64, device=DEV)
    for c0 in range(0, C, cc):
        pZ = Z[c0:c0 + cc] @ th[:, :dz].T                                           # (c, n, P)
        prod = (pZ + pY0[None]).sort(1).values; joint = (pZ[:, None] + pYq[None]).sort(2).values
        sw[c0:c0 + cc] = ((joint - prod[:, None]) ** 2).mean((-1, -2)).double()
    s = map_summary(sw[:, 0], sw[:, 1:].T, D)
    return dict(n_rows=n, projections=P, unit="squared SW2 (z-scored units)", detail=s, headline=dict(stat="max-cell SW2^2(joint, product)", **{k: s[f"max_{k}"] for k in ("real", "null_mean", "z", "p")}, spearman_m3=s["spearman_with_m3_yield"]))


# ------------------------------------------------------------------ 4. causal cell patching (linear read-out on all cells)
def image_allcell_ridge(Xfn, N, k, Ytr, Yq, chunk=2048):
    """M3.image_batch rule for one wide member: train-stat z-score (ddof 0); alpha = first max dev-MRR over run.ALPHAS on the rng(0)
    90/10 split; ridge on all train rows. Chunked float64 Gram (k ~ 1e4) and Cholesky solves. Xfn(rows) -> (r, k) raw features.
    Yq: list of (N, J) training targets (index 0 = real; others reuse its alpha). Returns alpha, W (k, J*Q), ym (J*Q), mu, sd."""
    import run as R
    s1 = torch.zeros(k, dtype=F64, device=DEV); s2 = torch.zeros(k, dtype=F64, device=DEV); allr = torch.arange(N, device=DEV)
    for i in range(0, N, chunk): x = Xfn(allr[i:i + chunk]).double(); s1 += x.sum(0); s2 += (x * x).sum(0)
    mu = s1 / N; sd = (s2 / N - mu ** 2).clamp(min=0).sqrt() + 1e-6; Afn = lambda r: (Xfn(r).double() - mu) / sd
    idx = np.random.default_rng(0).permutation(N); cut = int(0.9 * N); tr, va = torch.as_tensor(idx[:cut], device=DEV), torch.as_tensor(idx[cut:], device=DEV)
    G = torch.zeros(k, k, dtype=F64, device=DEV); Sy = torch.zeros(k, Ytr.shape[1], dtype=F64, device=DEV); sa = torch.zeros(k, dtype=F64, device=DEV)
    for i in range(0, len(tr), chunk): a = Afn(tr[i:i + chunk]); G += a.T @ a; Sy += a.T @ Ytr[tr[i:i + chunk]]; sa += a.sum(0)
    ym = Ytr[tr].mean(0); Cm = Sy - sa[:, None] * ym[None]; Av = Afn(va); T = Ytr[va]; T = T / T.norm(dim=1, keepdim=True); best, alpha = -1.0, None
    for a in R.ALPHAS:
        L = torch.linalg.cholesky(G + a * torch.eye(k, dtype=F64, device=DEV)); pv = Av @ torch.cholesky_solve(Cm, L) + ym
        pv = pv / pv.norm(dim=1, keepdim=True); Sm = pv @ T.T; m = float((1.0 / ((Sm > Sm.diagonal()[:, None]).sum(1) + 1)).mean())
        if m > best: best, alpha = m, a
        del L
    for i in range(0, len(va), chunk): a = Afn(va[i:i + chunk]); G += a.T @ a
    Ycat = torch.cat(Yq, 1); Sy = torch.zeros(k, Ycat.shape[1], dtype=F64, device=DEV); sa.zero_()
    for i in range(0, N, chunk): a = Afn(allr[i:i + chunk]); Sy += a.T @ Ycat[i:i + chunk]; sa += a.sum(0)
    ymq = Ycat.mean(0); W = torch.cholesky_solve(Sy - sa[:, None] * ymq[None], torch.linalg.cholesky(G + alpha * torch.eye(k, dtype=F64, device=DEV)))
    return alpha, W, ymq, mu, sd


def m4_patch(D, Kimg=3, cc=50):
    C = D["Z"].shape[0]; k = 36 * C; g = torch.Generator(device=DEV); g.manual_seed(21)
    if D["mod"] == "image":
        import run as R
        ntr = D["ntr"]; Ytr = D["Ytr"]; Yn = D["Yte"] / D["Yte"].norm(dim=1, keepdim=True); tr = D["fit_rows"]
        # G2a: feature reconstruction = M3 json cells (M3.image_batch on the first 40 cells); G2b: this ridge == M3.image_batch on cell 0
        r40 = M3.image_batch(D["Z"][:40, :ntr], D["Z"][:40, ntr:], Ytr, Yn, R, 1); y40 = torch.tensor([r["yield_"] for r in r40], dtype=F64)
        a0, W0, ym0, mu0, sd0 = image_allcell_ridge(lambda r: D["Z"][0, r], ntr, 36, Ytr, [Ytr])
        b0 = float(M3.soft_bits_gpu(cosS((D["Z"][0, ntr:].double() - mu0) / sd0 @ W0 + ym0, Yn)[None])[0])
        perm = [Ytr] + [Ytr[torch.randperm(ntr, generator=g, device=DEV)] for _ in range(Kimg)]
        Xfn = lambda r: D["Z"][:, r].permute(1, 0, 2).reshape(len(r), k)
        alpha, W, ymq, mu, sd = image_allcell_ridge(Xfn, ntr, k, Ytr, perm); Q = Kimg + 1; J = Ytr.shape[1]
        Ate = (Xfn(torch.arange(ntr, ntr + 200, device=DEV)).double() - mu) / sd; Pb = Ate @ W + ymq                      # (200, J*Q)
        base = torch.stack([M3.soft_bits_gpu(cosS(Pb[:, q * J:(q + 1) * J], Yn)[None])[0] for q in range(Q)])
        Sy = cosS(D["Yte"], Yn); Sy.fill_diagonal_(-2); same = Sy.argmax(1)                                                    # CLIP-nearest other test item
        R_ = torch.rand(200, 200, generator=g, device=DEV); R_[Sy > Sy.median(1, keepdim=True).values] = -1; R_.fill_diagonal_(-1); diff = R_.argmax(1)
        drops = {}
        for name, don in (("same", same), ("diff", diff)):
            dd = torch.zeros(C, Q, dtype=F64, device=DEV)
            for c0 in range(0, C, cc):
                cs = range(c0, min(C, c0 + cc)); dA = torch.stack([Ate[don, 36 * c:36 * c + 36] - Ate[:, 36 * c:36 * c + 36] for c in cs])
                dP = dA @ torch.stack([W[36 * c:36 * c + 36] for c in cs])                                                       # (c, 200, J*Q)
                for q in range(Q): dd[c0:c0 + len(cs), q] = base[q] - M3.soft_bits_gpu(cosS(Pb[None, :, q * J:(q + 1) * J] + dP[:, :, q * J:(q + 1) * J], Yn))
            drops[name] = dd
        m3j = D["m3_json"]
        g2 = dict(cells40_vs_m3_json=dict(corr=float(np.corrcoef(y40.numpy(), D["m3"][:40].cpu().numpy())[0, 1]), max_abs_diff=float((y40 - D["m3"][:40].cpu()).abs().max())),
                  cell0_this_ridge_vs_m3_image_batch=dict(this=b0, m3=float(y40[0]), abs_diff=abs(b0 - float(y40[0])), alpha_this=a0, alpha_m3=r40[0]["alpha"]),
                  allcell_no_patch=float(base[0]), m3_all_feature_yield_pca1024_raw=m3j["all_feature_yield"], m3_broadband_cell=m3j["broadband_cell"]["yield_"],
                  abs_diff_vs_m3_all_feature=abs(float(base[0]) - m3j["all_feature_yield"]), alpha_allcell=alpha,
                  note="M3 has no concatenated all-cell yield; the all-cell baseline is compared with M3's all-feature (PCA-1024 raw) and broadband yields as context")
        null_base = base[1:]
    else:
        n = D["n"]; Q = 22; idx = torch.cat([torch.arange(n, device=DEV)[None], D["nidx"]]); Yq = D["y_raw"][idx].T            # (n, Q): y + 21 shifts
        F = D["Z"].permute(1, 0, 2).reshape(n, k).double(); fo = folds(n); sc = torch.zeros(n, Q, dtype=F64, device=DEV); drops = {"same": torch.zeros(C, Q, dtype=F64, device=DEV), "diff": torch.zeros(C, Q, dtype=F64, device=DEV)}
        def fit(tr, te):
            A = F[tr]; mu, sd = A.mean(0), A.std(0, unbiased=False) + 1e-6; A, B = (A - mu) / sd, (F[te] - mu) / sd
            W = A.T @ torch.linalg.solve(A @ A.T + LAM * len(tr) * torch.eye(len(tr), dtype=F64, device=DEV), Yq[tr] - Yq[tr].mean(0)); return B, W
        sgn = torch.sign(D["y_raw"])
        for b in range(10):
            tr = np.where(fo != b)[0]; te = np.where(fo == b)[0]; B, W = fit(tr, te); P = B @ W; fi = folds(len(tr), 9); inner = torch.zeros(len(tr), Q, dtype=F64, device=DEV)
            for c in range(9): Bi, Wi = fit(tr[fi != c], tr[fi == c]); inner[torch.as_tensor(fi == c, device=DEV)] = Bi @ Wi
            yt, ye = Yq[tr], Yq[te]; xm, ym = inner.mean(0), yt.mean(0); slope = ((inner - xm) * (yt - ym)).sum(0) / ((inner - xm) ** 2).sum(0); icpt = ym - slope * xm
            rv = ((yt - icpt - slope * inner) ** 2).mean(0); vv = ((yt - ym) ** 2).mean(0)
            gain = lambda Pp: (torch.log(vv / rv) + (ye - ym) ** 2 / vv - (ye - icpt - slope * Pp) ** 2 / rv) / (2 * np.log(2))
            g0 = gain(P); sc[te] = g0; st = sgn[te]
            for name, m in (("same", st[:, None] == st[None]), ("diff", st[:, None] != st[None])):
                R_ = torch.rand(len(te), len(te), generator=g, device=DEV); R_[~m] = -1; R_.fill_diagonal_(-1); don = R_.argmax(1)
                dB = (B[don] - B).view(len(te), C, 36).permute(1, 0, 2); dP = dB @ W.view(C, 36, Q)                         # (C, nte, Q)
                drops[name] += (g0[None] - gain(P[None] + dP)).sum(1)
        drops = {kk: v / n for kk, v in drops.items()}; base = sc.mean(0)
        E21 = M3.pe_gain_batch(F[None], Yq, LAM)[0]                                                                          # M3's own batched estimator, same targets
        cells = M3.pe_gain_batch(D["Z"].double(), Yq[:, :1], LAM)[:, :, 0].mean(1)
        m3j = D["m3_json"]
        g2 = dict(allcell_no_patch_vs_m3_pe_gain_batch=dict(this=float(base[0]), m3_estimator=float(E21[:, 0].mean()), max_abs_diff_per_trial_all_targets=float(np.abs(E21 - sc.cpu().numpy()).max())),
                  cells_vs_m3_json=dict(corr=float(np.corrcoef(cells, D["m3"].cpu().numpy())[0, 1]), max_abs_diff=float(np.abs(cells - D["m3"].cpu().numpy()).max())),
                  m3_all_feature_yield_x7=m3j["all_feature_yield"], m3_broadband_cell=m3j["broadband_cell"]["yield_"], abs_diff_vs_m3_all_feature=abs(float(base[0]) - m3j["all_feature_yield"]),
                  m3_shift_floor_this=float(base[1:].mean()), note="M3 has no concatenated all-cell yield; X7 / broadband are context")
        null_base = base[1:]
    dr = drops["diff"]; s = map_summary(dr[:, 0], dr[:, 1:].T, D)
    out = dict(g2=g2, base_yield=float(base[0]), null_base_yield_mean=float(null_base.mean()), diff_donor=s, same_donor=map_summary(drops["same"][:, 0], drops["same"][:, 1:].T, D),
               diff_minus_same_mean=float((dr[:, 0] - drops["same"][:, 0]).mean()), sum_of_cell_drops_diff=float(dr[:, 0].sum()),
               donors="image: same = CLIP-nearest other test item (200 test items are 200 concepts, no same-category trials), diff = random item below the median similarity; PE: same / different sign of the residualised PE within the test fold")
    return dict(unit="bits (M3 yield)", detail=out, headline=dict(stat="max-cell yield drop (different-class donor)", **{k: s[f"max_{k}"] for k in ("real", "null_mean", "z", "p")}, spearman_m3=s["spearman_with_m3_yield"]))


# ------------------------------------------------------------------ 5. Gaussian PID (MMI) on the top-8 M3 cells
def m5_pid(D, K=21):
    top = D["m3"].argsort(descending=True)[:8].tolist(); pairs = [(i, j) for a, i in enumerate(top) for j in top[a + 1:]]
    Z = D["Z"]; z0 = torch.zeros_like(Z[0])
    F = torch.stack([torch.cat([Z[i], z0], 1) for i in top] + [torch.cat([Z[i], Z[j]], 1) for i, j in pairs])                     # (36, N, 72)
    Yq = D["ys"][torch.cat([torch.arange(D["n"], device=DEV)[None], D["nidx"][:K]])].T
    I = gauss_mi(ridge_ho(F, Yq, D["splits"]), Yq, D["eval_rows"]); Is = {c: I[a] for a, c in enumerate(top)}
    I1 = torch.stack([Is[i] for i, _ in pairs]); I2 = torch.stack([Is[j] for _, j in pairs]); I12 = I[8:]
    Rd = torch.minimum(I1, I2); syn = I12 - torch.maximum(I1, I2); U = (I1 - Rd) + (I2 - Rd)
    res = {nm: zp(v[:, 0].mean(), v[:, 1:].mean(0)) for nm, v in (("redundancy", Rd), ("unique_sum", U), ("synergy", syn), ("I12", I12))}
    b = int(syn[:, 0].argmax())
    out = dict(top8=[D["cells"][i] for i in top], pid_means=res, best_synergy_pair=dict(pair=[D["cells"][pairs[b][0]], D["cells"][pairs[b][1]]], synergy=float(syn[b, 0]),
               I1=float(I1[b, 0]), I2=float(I2[b, 0]), I12=float(I12[b, 0]), null_max_synergy_mean=float(syn[:, 1:].max(0).values.mean())),
               n_pairs_synergy_gt0=int((syn[:, 0] > 0).sum()), redundancy_fraction=float(Rd[:, 0].sum() / I12[:, 0].sum()),
               rule="MMI: R = min(I1, I2); U_i = I_i - R; S = I12 - max(I1, I2); Gaussian MI from held-out ridge (penalty LAM x n_train)")
    r = res["synergy"]; return dict(unit="nats", detail=out, headline=dict(stat="mean synergy over 28 pairs", **{k: r[k] for k in ("real", "null_mean", "z", "p")}))


# ------------------------------------------------------------------ 6. Koopman / DMD per region
def m6_koopman(D, r=20):
    import run as R
    W = D["W"]; N = D["n"]; X = D["Z"].view(8, 5, W, N, 36).permute(0, 3, 2, 1, 4).reshape(8, N, W, 180).float()
    fr = D["fit_rows"]; Xf = X[:, fr].reshape(8, -1, 180); mu, sd = Xf.mean(1, keepdim=True), Xf.std(1, unbiased=False, keepdim=True) + 1e-6
    Xs = (X - mu[:, None]) / sd[:, None]; Xf = Xs[:, fr].reshape(8, -1, 180); ev, U = torch.linalg.eigh((Xf.mT @ Xf).double() / Xf.shape[1]); U = U.flip(-1)[..., :r].float()
    Y = Xs @ U[:, None]                                                                                    # (8, N, W, r)
    Yf = Y[:, fr]; X0 = Yf[:, :, :-1].reshape(8, -1, r); X1 = Yf[:, :, 1:].reshape(8, -1, r)
    A = torch.linalg.solve((X0.mT @ X0).double(), (X0.mT @ X1).double())                                                     # y_{t+1} = y_t A
    lam, V = torch.linalg.eig(A.mT); order = lam.abs().argsort(-1, descending=True); lam = torch.gather(lam, 1, order); V = torch.gather(V, 2, order[:, None].expand(-1, r, -1))
    amp = Y.to(torch.complex64) @ torch.linalg.inv(V).mT[:, None].to(torch.complex64)                                               # mode amplitudes (8, N, W, r)
    h = r // 2; feat = lambda s: torch.cat([amp[..., s].real, amp[..., s].imag], -1).reshape(8, N, -1)
    Fb = torch.cat([feat(slice(0, h)), feat(slice(h, r))])                                               # (16, N, 2 h W): slow 8 regions, then fast
    if D["mod"] == "image":
        ntr = D["ntr"]; Yn = D["Yte"] / D["Yte"].norm(dim=1, keepdim=True); rr = M3.image_batch(Fb[:, :ntr].float(), Fb[:, ntr:].float(), D["Ytr"], Yn, R, 1)
        yv = torch.tensor([x["yield_"] for x in rr], dtype=F64); fl = torch.tensor([x["shuffled"] for x in rr], dtype=F64)
    else:
        Ys = D["y_raw"][torch.stack([torch.arange(N, device=DEV)] + [(torch.arange(N, device=DEV) - round(N * kq / 4)) % N for kq in (1, 2, 3)])].T
        E = M3.pe_gain_batch(Fb, Ys, LAM); yv = torch.tensor(E[:, :, 0].mean(1)); fl = torch.tensor(E[:, :, 1:].mean((1, 2)))
    regs = list(D["sets"]); d = yv[:8] - yv[8:]; dn = fl[:8] - fl[8:]
    per = {rg: dict(abs_lambda=[round(float(v), 3) for v in lam[i].abs()], slow_yield=float(yv[i]), fast_yield=float(yv[8 + i]), slow_null=float(fl[i]), fast_null=float(fl[8 + i])) for i, rg in enumerate(regs)}
    out = dict(per_region=per, mean_slow_minus_fast=float(d.mean()), null_slow_minus_fast=float(dn.mean()), n_regions_slow_gt_fast=int((d > 0).sum()),
               max_abs_lambda=float(lam.abs().max()), mean_slow_yield=float(yv[:8].mean()), mean_fast_yield=float(yv[8:].mean()), null_kind="image: shuffled test pairs (M3 floor); PE: circular shifts n/4, n/2, 3n/4",
               fit="z-score + PCA-20 per region (5 bands x 36 features per window), operator fit on training rows pooled over consecutive windows (target-free)")
    t = float(d.mean() / (d.std() / np.sqrt(8))) if d.std() > 0 else None
    return dict(unit="bits (M3 yield)", detail=out, headline=dict(stat="mean over regions: slow-mode yield - fast-mode yield", real=float(d.mean()), null_mean=float(dn.mean()), z=t, p=None, z_kind="paired t over 8 regions"))


# ------------------------------------------------------------------ 7. pullback Riemannian volume of an MLP read-out
def m7_pullback(D, hid=64, epochs=200, Kn=6):
    from torch.func import vmap, jacrev
    C = D["Z"].shape[0]; k = 36 * C; N = D["n"]; tr0 = D["fit_rows"]
    X = D["Z"].permute(1, 0, 2).reshape(N, k); mu, sd = X[tr0].mean(0), X[tr0].std(0, unbiased=False) + 1e-6; X = ((X - mu) / sd).float()
    g = torch.Generator(device=DEV); g.manual_seed(31)
    if D["mod"] == "image":
        Q = 4; idx = torch.cat([torch.arange(N, device=DEV)[None], D["nidx"][:Q - 1]]); Yt = D["Ylow"][idx].float()                      # (Q, N, 16)
        masks = torch.zeros(Q, N, dtype=torch.bool, device=DEV); masks[:, tr0] = True; own = [D["eval_rows"]] * Q
    else:
        nf = 10; Q = 1 + Kn; idx = torch.cat([torch.arange(N, device=DEV)[None], D["nidx"][::21 // Kn][:Kn]]); fo = torch.as_tensor(folds(N), device=DEV)
        Yt = D["ys"][idx].float()[:, None, :, None].expand(-1, nf, -1, -1).reshape(Q * nf, N, 1)
        masks = (fo[None] != torch.arange(nf, device=DEV)[:, None]).repeat(Q, 1); own = [torch.where(~masks[m])[0] for m in range(Q * nf)]
    Mb, out = Yt.shape[0], Yt.shape[2]
    W1 = (torch.randn(Mb, k, hid, generator=g, device=DEV) / np.sqrt(k)).requires_grad_(); b1 = torch.zeros(Mb, 1, hid, device=DEV, requires_grad=True)
    W2 = (torch.randn(Mb, hid, out, generator=g, device=DEV) / np.sqrt(hid)).requires_grad_(); b2 = torch.zeros(Mb, 1, out, device=DEV, requires_grad=True)
    opt = torch.optim.AdamW([W1, b1, W2, b2], lr=1e-3, weight_decay=1e-2); mf = masks.float(); cnt = mf.sum(1)
    fwd = lambda: torch.tanh((X @ W1.permute(1, 0, 2).reshape(k, Mb * hid)).view(N, Mb, hid).permute(1, 0, 2) + b1) @ W2 + b2
    for ep in range(epochs):
        opt.zero_grad(); loss = ((((fwd() - Yt) ** 2).mean(-1) * mf).sum(1) / cnt).sum(); loss.backward(); opt.step()
    with torch.no_grad(): P = fwd()
    f = lambda x, w1, c1, w2, c2: torch.tanh(x @ w1 + c1[0]) @ w2 + c2[0]
    flin = lambda x, w1, c1, w2, c2: (x @ w1 + c1[0]) @ w2 + c2[0]
    def vol(fn, m, rows):
        J = vmap(jacrev(fn), in_dims=(0, None, None, None, None))(X[rows], W1[m].detach(), b1[m].detach(), W2[m].detach(), b2[m].detach())   # (r, out, k)
        JJ = J @ J.mT; eps = 1e-6 * torch.diagonal(JJ, dim1=-2, dim2=-1).mean(); return 0.5 * torch.logdet((JJ + eps * torch.eye(out, device=DEV)).double())
    vols = [vol(f, m, own[m]) for m in range(Mb)]; lin = vol(flin, 0, own[0])
    if D["mod"] == "image":
        Yc = D["Ylow"][D["eval_rows"]].float(); Yc = Yc / Yc.norm(dim=1, keepdim=True)
        yq = soft_per_query(torch.stack([cosS(P[q, D["eval_rows"]], Yc) for q in range(Q)]))
        assert abs(float(yq[0].mean()) - float(M3.soft_bits_gpu(cosS(P[0, D["eval_rows"]], Yc)[None])[0])) < 1e-9
        rhos = torch.tensor([spearman(vols[q], yq[q]) for q in range(Q)]); ymean = yq.mean(1)
    else:
        rhos, ymean = [], []
        for q in range(Q):
            p = torch.zeros(N, dtype=F64, device=DEV); v = torch.zeros(N, dtype=F64, device=DEV); y = D["ys"][idx[q]]
            for b in range(10): m = q * 10 + b; p[own[m]] = P[m, own[m], 0].double(); v[own[m]] = vols[m]
            Dm = torch.stack([torch.ones_like(p), p], 1); c = torch.linalg.lstsq(Dm, y[:, None]).solution[:, 0]; rv = ((y - Dm @ c) ** 2).mean(); vv = ((y - y.mean()) ** 2).mean()
            gi = (torch.log(vv / rv) + (y - y.mean()) ** 2 / vv - (y - Dm @ c) ** 2 / rv) / (2 * np.log(2)); rhos.append(spearman(v, gi)); ymean.append(float(gi.mean()))
        rhos = torch.tensor(rhos); ymean = torch.tensor(ymean)
    r = zp(rhos[0], rhos[1:])
    out = dict(spearman_volume_vs_trial_yield=r, mlp_yield_real=float(ymean[0]), mlp_yield_null_mean=float(ymean[1:].mean()),
               volume_sd_mlp=float(torch.cat([v.flatten() for v in vols[:1 if D['mod'] == 'image' else 10]]).std()), linear_readout_volume_sd=float(lin.std()), linear_readout_volume_mean=float(lin.mean()),
               members=Mb, epochs=epochs, hidden=hid, output_dim=out, yield_kind="image: per-query InfoNCE (M3 soft-bits rule) on 16 CLIP PCs; PE: per-trial Gaussian log-score gain, calibration on pooled held-out predictions")
    return dict(unit="Spearman rho", detail=out, headline=dict(stat="Spearman(volume, per-trial yield)", **{k: r[k] for k in ("real", "null_mean", "z", "p")}))


# ------------------------------------------------------------------ 8. discrete Hodge decomposition on the region x time grid
def m8_hodge(D, k_nn=3, nb=200):
    keep, xyz = M3.montage_xyz(D["ch"]); pos = {c: xyz[i] for i, c in enumerate(keep)}; regs = list(D["sets"])
    cen = torch.tensor(np.array([np.mean([pos[c] for c in D["sets"][r]], 0) for r in regs]), device=DEV); dist = torch.cdist(cen, cen); dist.fill_diagonal_(1e9)
    nn = dist.argsort(1)[:, :k_nn].cpu().numpy(); sp = sorted({(min(i, j), max(i, j)) for i in range(8) for j in nn[i]})
    W = D["W"]; node = lambda r, t: r * W + t; edges = [(node(i, t), node(j, t)) for t in range(W) for i, j in sp] + [(node(r, t), node(r, t + 1)) for r in range(8) for t in range(W - 1)]
    tri = [(i, j, l) for i in range(8) for j in range(i + 1, 8) for l in range(j + 1, 8) if (i, j) in sp and (j, l) in sp and (i, l) in sp]
    faces = [[node(i, t), node(j, t), node(l, t)] for t in range(W) for i, j, l in tri] + [[node(i, t), node(j, t), node(j, t + 1), node(i, t + 1)] for t in range(W - 1) for i, j in sp]
    E = len(edges); lut = {}
    for e, (u, v) in enumerate(edges): lut[(u, v)] = (e, 1.0); lut[(v, u)] = (e, -1.0)
    B1 = torch.zeros(8 * W, E, dtype=F64, device=DEV); B2 = torch.zeros(E, len(faces), dtype=F64, device=DEV)
    for e, (u, v) in enumerate(edges): B1[u, e] = -1; B1[v, e] = 1
    for fi, cyc in enumerate(faces):
        for a, b in zip(cyc, cyc[1:] + cyc[:1]): e, s = lut[(a, b)]; B2[e, fi] = s
    assert float((B1 @ B2).abs().max()) == 0.0
    Pg = B1.T @ torch.linalg.pinv(B1.T); Pc = B2 @ torch.linalg.pinv(B2)
    eu, ev = torch.tensor([u for u, _ in edges], device=DEV), torch.tensor([v for _, v in edges], device=DEV)
    sig = D["sig"][D["fit_rows"]].double(); N = sig.shape[0]; S = sig.reshape(N, -1)
    def flows(M):        # M (B, 8*W*5) trial means -> (B, E) antisymmetric lag-1 cross-covariance flow
        M = M.view(-1, 8 * W, 5); M = M - M.mean(-1, keepdim=True); a, b = M[:, eu], M[:, ev]
        return ((a[..., :-1] * b[..., 1:]).sum(-1) - (b[..., :-1] * a[..., 1:]).sum(-1)) / 4
    def frac(f):
        gr, cu = f @ Pg, f @ Pc; tot = (f ** 2).sum(-1); h = f - gr - cu
        return torch.stack([(gr ** 2).sum(-1) / tot, (cu ** 2).sum(-1) / tot, (h ** 2).sum(-1) / tot], -1)
    gi = torch.Generator(device=DEV); gi.manual_seed(41); ix = torch.randint(N, (nb, N), generator=gi, device=DEV)
    cnts = torch.zeros(nb, N, dtype=F64, device=DEV).scatter_add_(1, ix, torch.ones(nb, N, dtype=F64, device=DEV))
    Srev = sig.flip(2).flip(3).reshape(N, -1); res = {}
    for nm, SS in (("real", S), ("time_reversed", Srev)):
        fr0 = frac(flows(SS.mean(0)[None]))[0]; fb = frac(flows(cnts @ SS / N)); res[nm] = dict(grad=float(fr0[0]), curl=float(fr0[1]), harmonic=float(fr0[2]), boot_sd=[float(x) for x in fb.std(0)])
    rk1, rk2 = int(torch.linalg.matrix_rank(B1)), int(torch.linalg.matrix_rank(B2)); ref = dict(grad=rk1 / E, curl=rk2 / E, harmonic=(E - rk1 - rk2) / E)
    fr_rand = frac(torch.randn(2000, E, generator=gi, device=DEV, dtype=F64)); res["random_iid_flow"] = dict(grad=float(fr_rand[:, 0].mean()), curl=float(fr_rand[:, 1].mean()), harmonic=float(fr_rand[:, 2].mean()), sd=[float(x) for x in fr_rand.std(0)])
    res["time_reversed"]["degenerate"] = bool(abs(res["real"]["grad"] - res["time_reversed"]["grad"]) < 1e-9)
    zr = zp(res["real"]["grad"], fr_rand[:, 0])
    out = dict(fractions=res, random_flow_reference=ref, n_nodes=8 * W, n_edges=E, n_faces=len(faces), spatial_edges=[(regs[i], regs[j]) for i, j in sp],
               note="trial-averaged region signals (target-free); faces = spatial triangles + region-pair x adjacent-window squares; time reversal maps lead -> lag")
    out["note"] += "; time reversal is an exact automorphism of this grid (t -> W-1-t) that negates the flow, so its energy fractions equal the real ones (degenerate null); the reported null is 2,000 iid Gaussian edge flows"
    return dict(unit="energy fraction", detail=out, headline=dict(stat="gradient energy fraction", **{k: zr[k] for k in ("real", "null_mean", "z", "p")}, z_kind="vs iid random flows", time_reversed=res["time_reversed"]["grad"]))


# ------------------------------------------------------------------ 9. persistent homology beta0 (MST) of class-mean cell latents
def mst_lengths(Dm):
    B, m, _ = Dm.shape; ar = torch.arange(B, device=DEV); intree = torch.zeros(B, m, dtype=torch.bool, device=DEV); intree[:, 0] = True
    best = Dm[:, 0].clone(); out = torch.empty(B, m - 1, dtype=Dm.dtype, device=DEV)
    for i in range(m - 1):
        bb = best.masked_fill(intree, float("inf")); v = bb.argmin(1); out[:, i] = bb[ar, v]; intree[ar, v] = True; best = torch.minimum(best, Dm[ar, v])
    return out


def m9_ph(D, K=12, cc=40):
    if D["mod"] == "image":
        Xs = D["Xte80"].reshape(-1, 1, *D["Xte80"].shape[2:]); P, V = image_feats(Xs, 1, D["sf"], D["kk"]); Zs, _ = gather_cells(P, V, D["sets"], 10); del P, V
        m, reps = 200, D["Xte80"].shape[1]; chk = float((Zs.view(Zs.shape[0], m, reps, 36)[..., 6:].mean(2) - D["Z"][:, D["ntr"]:, 6:]).abs().max())   # voltage columns (power: mean of log != log of mean)
        g = torch.Generator(device=DEV); g.manual_seed(51); perms = [torch.arange(m * reps, device=DEV)] + [torch.randperm(m * reps, generator=g, device=DEV) for _ in range(K)]
        means = lambda Zc: torch.stack([Zc[:, p].view(Zc.shape[0], m, reps, 36).mean(2) for p in perms], 1)          # (c, Q, m, 36)
        Zsrc = Zs
    else:
        m = 20; n = D["n"]; idx = torch.cat([torch.arange(n, device=DEV)[None], D["nidx"][::21 // K][:K]]); lab = (D["ys"][idx].argsort(1).argsort(1) * m) // n
        OH = torch.nn.functional.one_hot(lab, m).double().transpose(1, 2); OH = OH / OH.sum(-1, keepdim=True)           # (Q, m, n)
        means = lambda Zc: torch.einsum("qmn,cnk->cqmk", OH, Zc.double()); Zsrc = D["Z"]; chk = None
    C = Zsrc.shape[0]; Lt = torch.zeros(C, K + 1, dtype=F64, device=DEV); Ln = torch.zeros(C, K + 1, dtype=F64, device=DEV)
    for c0 in range(0, C, cc):
        Zc = zstd(Zsrc[c0:c0 + cc].double()); Mc = means(Zc); c = Mc.shape[0]; Dm = torch.cdist(Mc.reshape(-1, m, 36), Mc.reshape(-1, m, 36))
        L = mst_lengths(Dm).sum(1).view(c, K + 1); mp = Dm.sum((1, 2)).view(c, K + 1) / (m * (m - 1)); Lt[c0:c0 + c] = L; Ln[c0:c0 + c] = L / ((m - 1) * mp)
    s, sn = map_summary(Lt[:, 0], Lt[:, 1:].T, D), map_summary(Ln[:, 0], Ln[:, 1:].T, D)
    try:
        import ripser; b1 = "ripser installed but beta1 not run (CPU-only library; GPU-only rule)"
    except ImportError:
        b1 = "skipped: neither ripser nor gudhi is installed (no pip install)"
    out = dict(total_mst_length=s, normalised_mst_length=sn, n_points=m, beta1=b1, class_mean_vs_feature_max_abs_diff=chk,
               classes="image: 200 test stimuli, single-trial cell features, null = stimulus labels permuted over the 16,000 trials; PE: 20 quantile bins of the normal-scores PE, null = shifted target")
    return dict(unit="MST length (z-scored units)", detail=out, headline=dict(stat="max-cell total MST length", **{k: s[f"max_{k}"] for k in ("real", "null_mean", "z", "p")}, spearman_m3=s["spearman_with_m3_yield"]))


METHODS = [("1_renyi_mi", m1_renyi), ("2_cond_immse", m2_immse), ("3_sliced_wasserstein", m3_sw), ("4_cell_patching", m4_patch), ("5_gaussian_pid", m5_pid),
           ("6_koopman_dmd", m6_koopman), ("7_pullback_volume", m7_pullback), ("8_hodge", m8_hodge), ("9_ph_beta0", m9_ph)]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--modality", default="both", choices=["pe", "image", "both"])
    ap.add_argument("--methods", default="", help="comma list of method numbers (default all)")
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True); mon = GpuMon(); T0 = time.time()
    pick = {int(x) for x in a.methods.split(",")} if a.methods else set(range(1, 10))
    for mod in (["pe", "image"] if a.modality == "both" else [a.modality]):
        t0 = time.time(); D = load_pe() if mod == "pe" else load_image(); res = dict(modality=mod, participant=D["sub"], load_s=round(time.time() - t0, 1), load_gpu=mon.window(t0, time.time()), methods={})
        for name, fn in METHODS:
            if int(name[0]) not in pick: continue
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); t1 = time.time()
            try: r = fn(D)
            except Exception as e: traceback.print_exc(); r = dict(error=repr(e))
            torch.cuda.synchronize(); t2 = time.time()
            r.update(runtime_s=round(t2 - t1, 1), gpu=mon.window(t1, t2), peak_mem_gb=round(torch.cuda.max_memory_allocated() / 1e9, 2)); res["methods"][name] = r
            print(f"{D['sub']} {name}: {json.dumps(r.get('headline', r.get('error')))} | {r['runtime_s']}s GPU {r['gpu']} peak {r['peak_mem_gb']} GB", flush=True)
            if name.startswith("4"): print(" G2", json.dumps(r.get("detail", {}).get("g2")), flush=True)
            torch.cuda.empty_cache()
        (out / f"frontier_{mod}_{D['sub']}.json").write_text(json.dumps(res, indent=1, default=str)); del D; torch.cuda.empty_cache()
    mon.stop = True; print("total", f"{time.time()-T0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
