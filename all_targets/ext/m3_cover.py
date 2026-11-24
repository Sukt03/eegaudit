"""A4 greedy cell cover + B2 leave-one-cell-out on the M3 grid (rung R5). EXECUTOR-2, 21 Nov 2026.
Spec: outputs/IDEAS_user_diagnostics_20261121-0435.md (A4, B2); design + amendment in outputs/EXTENSIONS_LOOP_20261121.md
("G1 REQUEST A4 + B2", "G1 REQUEST A4 amendment"). GPU only (PLAN §10): every fit is a batched torch solve.

Ground set V = the M3 named combined cells (power + 5 voltage bins per channel, 6 channels: 36 features), non-reference:
  image 8 regions x 5 bands x 10 windows (0-1000 ms) = 400; PE 8 x 5 x 6 (0-600 ms after feedback) = 240. Features = m3_yield_map's.
f(S) = the locked score of ONE ridge on the concatenated features of the cells in S:
  image: train-stat standardisation of every feature (as the M3 cell scorer), primal Gram trick (G = Z'Z, C = Z'(Y - ym) on all training
         rows and on the rng(0) 90 % dev-fit rows; any S is a sub-block), alpha re-selected per subset by dev MRR over run.ALPHAS, InfoNCE
         soft bound over the 200 test candidates (k = 80).
  PE:    dual (kernel) form of m3_yield_map.pe_gain (in-fold z-scoring, ridge LAM * n_train, inner 9-fold calibration, Gaussian density
         gain), K_S = sum of per-cell kernels, candidates batched as members.
A4: greedy forward selection on the mean f(S) of a SELECTION half of the participants (odd / even IDs), up to --max_cells; every step also
    records f(S_t) for all participants. Greedy updates are exact and incremental: image = block-inverse ridge update at the participant's
    all-cell dev alpha (fixed during the greedy; reported f(S_t) uses the same alpha), PE = per-fold-step dual Woodbury rank-36 updates. Reported for the held-out half: f(S_t) / f(V) and the sizes |S| at which the selection half
    reaches 90 % and 95 % of its mean f(V); both directions. Greedy is a heuristic (f need not be submodular).
B2: Delta_c = f(V) - f(V \\ c) per participant and cell at the all-cell alpha (image: exact block downdate of the all-cell ridge;
    PE: K_V - K_c). Group: mean, participant bootstrap CI (GPU index matrix), max-t sign-flip FWER over cells.
G2 (printed, asserted): image f({c}) = the M3 map cell (results/ext/m3/image_sub-XX.json) within 1e-4; image downdate = brute-force refit
    for 2 cells within 1e-6; PE f({c}) dual = pe_gain_batch primal within 1e-8; PE f(V) dual = primal within 1e-8.
Floors: f(V) with shuffled pairs (image) / circular-shift targets (PE), and noise input (image) printed per participant.
Contrarian fixes (outputs/cards/contrarian_diagnostics_20261121.md): --frac 0.5 / 1.0 training-size arms (k90 at two N); B2 unique
  contributions printed beside the single-cell map; fractions only where the f(V) - floor CI clears 0; framing = cells this read-out needs
  at this N (weak submodularity: Das & Kempe 2011; Elenberg et al. 2018).
Usage: python m3_cover.py --target image|pe --out results/ext/m3_cover [--subjects 1,2] [--max_cells 60] [--frac 0.5]
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import torch

ROOT = Path("/path/to/eegaudit")
for p in (ROOT, ROOT / "experiments/ext", ROOT / "experiments/eegbits", ROOT / "experiments/agent_loop_20261120", ROOT / "experiments/standard"):
    sys.path.insert(0, str(p))
import m3_yield_map as M
import regions as RG
DEV = "cuda"; LN2 = np.log(2)


# ------------------------------------------------------------------ image
def image_state(s, R, frac=1.0):
    """GPU state of one participant: standardised features of all 400 named cells (fp32), Grams for the dev split and all training rows."""
    sub, Ytr, Yte, Xtr4, Xte80, ch = M.image_load(s, R); sf, T = 250.0, 250
    Mw, nfft, Wf = M.morlet_bank(sf, T); nfv = int(2 ** np.ceil(np.log2(3 * T))); Gb = M.bandpass_gains(sf, nfv)
    def feats(X, reps):
        Ps, Vs = [], []
        for i in range(0, len(X), max(1, 800 // reps)):
            xb = torch.tensor(np.ascontiguousarray(X[i:i + max(1, 800 // reps)]), device=DEV); nI = xb.shape[0]
            pw = M.band_power(xb.reshape(nI * reps, *xb.shape[2:]), sf, Mw, nfft, Wf).reshape(nI, reps, 63, 5, T).mean(1)
            P, V = M.window_feats(pw, M.band_voltage(xb.mean(1), sf, Gb, nfv), 0, 25, 10); Ps.append(torch.log10(P + 1e-20)); Vs.append(V)
        return torch.cat(Ps), torch.cat(Vs)
    Ptr, Vtr = feats(Xtr4, 4); Pte, Vte = feats(Xte80, 80); del Xtr4, Xte80
    grid = [(r, RG.indices(ch, RG.REGIONS[r]), b, w) for r in RG.REGIONS for b in range(5) for w in range(10)]
    CH = torch.as_tensor([g[1] for g in grid], device=DEV); BI = torch.as_tensor([g[2] for g in grid], device=DEV)[:, None]; WI = torch.as_tensor([g[3] for g in grid], device=DEV)[:, None]
    def cells(P, V):   # (N, 400 cells, 36) in the m3 combined order [6 power | 6 x 5 voltage]
        return torch.cat([P[:, CH, BI, WI], V[:, CH, BI, WI, :].flatten(2)], 2)
    Ztr, Zte = cells(Ptr, Vtr).double(), cells(Pte, Vte).double(); del Ptr, Vtr, Pte, Vte
    if frac < 1:   # training-size arm: a fixed random subset of the training rows (seed 1); standardisation and Grams on the subset only
        keep = np.sort(np.random.default_rng(1).choice(len(Ztr), round(frac * len(Ztr)), replace=False)); Ztr = Ztr[torch.as_tensor(keep, device=DEV)]; Ytr = Ytr[keep]
    mu, sd = Ztr.mean(0, keepdim=True), Ztr.std(0, unbiased=False, keepdim=True) + 1e-6; Ztr = ((Ztr - mu) / sd).flatten(1); Zte = ((Zte - mu) / sd).flatten(1)
    Y = torch.tensor(Ytr, dtype=torch.float64, device=DEV); N = len(Y)
    idx = np.random.default_rng(0).permutation(N); cut = int(0.9 * N); tr, va = torch.as_tensor(idx[:cut], device=DEV), torch.as_tensor(idx[cut:], device=DEV)
    ym_tr, ym = Y[tr].mean(0), Y.mean(0)
    st = dict(sub=sub, grid=[(g[0], M.BNAMES[g[2]], 100 * g[3]) for g in grid],
              G_tr=(Ztr[tr].T @ Ztr[tr]).float(), C_tr=(Ztr[tr].T @ (Y[tr] - ym_tr)).float(), ym_tr=ym_tr, Z_va=Ztr[va].float(),
              T_va=(Y[va] / Y[va].norm(dim=1, keepdim=True)), G=(Ztr.T @ Ztr).float(), C=(Ztr.T @ (Y - ym)).float(), ym=ym, Z_te=Zte.float(),
              Yn=torch.as_tensor(Yte / np.linalg.norm(Yte, axis=1, keepdims=True), dtype=torch.float64, device=DEV))
    del Ztr, Zte; torch.cuda.empty_cache()
    # noise-input floor for f(V) (one draw, fixed generator), computed now and not kept
    import run as R
    g = torch.Generator(device=DEV); g.manual_seed(100); Nz = torch.randn((N, st["G"].shape[0]), generator=g, device=DEV, dtype=torch.float64)
    nst = dict(G_tr=(Nz[tr].T @ Nz[tr]).float(), C_tr=(Nz[tr].T @ (Y[tr] - ym_tr)).float(), Z_va=Nz[va].float(), G=(Nz.T @ Nz).float(), C=(Nz.T @ (Y - ym)).float())
    del Nz; nst["Z_te"] = torch.randn((200, st["G"].shape[0]), generator=g, device=DEV, dtype=torch.float64).float()
    st["floor_noise"] = float(image_eval(st, [list(range(len(grid)))], R, src=nst)[0]); del nst; torch.cuda.empty_cache()
    return st


def feat_idx(sets, k=36):
    """sets: list of equal-size cell-index lists -> (B, k * |S|) feature indices."""
    c = torch.as_tensor(sets, device=DEV); return (c[:, :, None] * k + torch.arange(k, device=DEV)).flatten(1)


def image_eval(st, sets, R, src=None, mem=1.2e9, alphas_out=False):
    """f(S) for a batch of equal-size cell sets (one participant): dev-MRR alpha, final ridge, test InfoNCE (soft_bits_gpu)."""
    src = src or st; F = feat_idx(sets); B, k = F.shape; step = max(1, int(mem // (k * k * 8 + k * 1024 * 8 * 2))); ys, als = [], []
    I = torch.eye(k, dtype=torch.float64, device=DEV)
    for i in range(0, B, step):
        f = F[i:i + step]; nb = len(f)
        Gt = src["G_tr"][f[:, :, None], f[:, None, :]].double(); Ct = src["C_tr"][f].double(); Zv = src["Z_va"][:, f].permute(1, 0, 2).double()
        best = torch.full((nb,), -1.0, dtype=torch.float64, device=DEV); al = torch.zeros(nb, dtype=torch.float64, device=DEV)
        for a in R.ALPHAS:
            pv = Zv @ torch.linalg.solve(Gt + a * I, Ct) + st["ym_tr"]; pv = pv / pv.norm(dim=2, keepdim=True); S = pv @ st["T_va"].T
            d = torch.diagonal(S, dim1=1, dim2=2); m = (1.0 / ((S > d[..., None]).sum(2) + 1)).mean(1)
            upd = m > best; best = torch.where(upd, m, best); al = torch.where(upd, torch.full_like(al, float(a)), al)
        del Gt, Ct, Zv
        G = src["G"][f[:, :, None], f[:, None, :]].double(); C = src["C"][f].double(); Zt = src["Z_te"][:, f].permute(1, 0, 2).double()
        p = Zt @ torch.linalg.solve(G + al[:, None, None] * I, C) + st["ym"]; p = p / p.norm(dim=2, keepdim=True)
        ys.append(M.soft_bits_gpu(p @ st["Yn"].T)); als.append(al); del G, C, Zt, p
    y = torch.cat(ys)
    return (y, torch.cat(als)) if alphas_out else y


def image_loco(st, R):
    """B2 for one participant: all-cell ridge at its dev alpha, exact block downdate for every cell (batched)."""
    nC = len(st["grid"]); fV, aV = image_eval(st, [list(range(nC))], R, alphas_out=True); a = float(aV[0]); K = nC * 36
    G = st["G"].double() + a * torch.eye(K, dtype=torch.float64, device=DEV); Mi = torch.linalg.inv(G); del G
    beta = Mi @ st["C"].double(); Zt = st["Z_te"].double(); base = Zt @ beta + st["ym"]; ZM = Zt @ Mi
    F = feat_idx([[c] for c in range(nC)])                                                     # (400, 36)
    Mcc = Mi[F[:, :, None], F[:, None, :]]; bc = beta[F]; Ztc = Zt[:, F].permute(1, 0, 2)          # (400,36,36), (400,36,1024), (400,200,36)
    corr = (ZM[:, F].permute(1, 0, 2) - Ztc @ Mcc) @ torch.linalg.solve(Mcc, bc)
    p = base[None] - Ztc @ bc - corr; p = p / p.norm(dim=2, keepdim=True); f_minus = M.soft_bits_gpu(p @ st["Yn"].T)
    # G2: brute-force refit without cells 0 and 1 at the same alpha
    chk = []
    for c in (0, 1):
        keep = torch.as_tensor([j for j in range(K) if j // 36 != c], device=DEV)
        Gk = st["G"].double()[keep][:, keep] + a * torch.eye(len(keep), dtype=torch.float64, device=DEV)
        q = Zt[:, keep] @ torch.linalg.solve(Gk, st["C"].double()[keep]) + st["ym"]; q = q / q.norm(dim=1, keepdim=True)
        chk.append(abs(float(M.soft_bits_gpu((q @ st["Yn"].T)[None])[0]) - float(f_minus[c])))
    assert max(chk) < 1e-6, chk
    pf = base / base.norm(dim=1, keepdim=True); fullv = float(M.soft_bits_gpu((pf @ st["Yn"].T)[None])[0])
    shuf = float(np.mean([float(M.soft_bits_gpu((pf @ st["Yn"][torch.as_tensor(np.random.default_rng(200 + j).permutation(200), device=DEV)].T)[None])[0]) for j in range(3)]))
    return dict(fV=float(fV[0]), fV_alpha=a, fV_at_alpha=fullv, f_minus=f_minus.cpu().numpy().tolist(), downdate_check=max(chk), floor_shuffled=shuf, floor_noise=st["floor_noise"])


# ------------------------------------------------------------------ PE (dual form)
def pe_state(s, evaluate, fpath, frac=1.0):
    from belief_bottleneck import LAM, folds
    sub, fz, tri, n, Ys, X, ch = M.pe_load(s, evaluate, fpath)
    sf, T = 200.0, 400; Mw, nfft, Wf = M.morlet_bank(sf, T); nfv = int(2 ** np.ceil(np.log2(3 * T))); Gb = M.bandpass_gains(sf, nfv); Ps, Vs = [], []
    for i in range(0, n, 64):
        xg = torch.tensor(X[i:i + 64], device=DEV); P, V = M.window_feats(M.band_power(xg, sf, Mw, nfft, Wf), M.band_voltage(xg, sf, Gb, nfv), 200, 20, 6)
        Ps.append(torch.log10(P + 1e-20)); Vs.append(V)
    P, V = torch.cat(Ps), torch.cat(Vs)
    grid = [(r, RG.indices(ch, RG.REGIONS[r]), b, w) for r in RG.REGIONS for b in range(5) for w in range(6)]
    CH = torch.as_tensor([g[1] for g in grid], device=DEV); BI = torch.as_tensor([g[2] for g in grid], device=DEV)[:, None]; WI = torch.as_tensor([g[3] for g in grid], device=DEV)[:, None]
    F = torch.cat([P[:, CH, BI, WI], V[:, CH, BI, WI, :].flatten(2)], 2).double()            # (n, 240, 36)
    fo = folds(n); steps = []
    for b in range(10):
        tr = np.where(fo != b)[0]; te = np.where(fo == b)[0]
        if frac < 1: tr = np.sort(np.random.default_rng(100 + b).choice(tr, round(frac * len(tr)), replace=False))   # training-size arm
        steps.append((b, -1, tr, te)); fi = folds(len(tr), 9)
        for c in range(9): steps.append((b, c, tr[fi != c], tr[fi == c]))
    st = dict(sub=sub, grid=[(g[0], M.BNAMES[g[2]], 100 * g[3]) for g in grid], F=F, Y=torch.as_tensor(Ys, dtype=torch.float64, device=DEV), LAM=LAM, steps=[], n=n, fo=fo)
    for b, c, tr, te in steps:
        st["steps"].append(dict(b=b, c=c, tr=tr, te=te))
    return st


def step_Z(st, stp, cells=None):
    """Standardised features of (train | test) rows of one fold step, training-row statistics (as pe_gain): (m, |cells|, 36)."""
    F = st["F"] if cells is None else st["F"][:, cells]; tr = torch.as_tensor(stp["tr"], device=DEV)
    A = F[tr]; mu, sd = A.mean(0, keepdim=True), A.std(0, unbiased=False, keepdim=True) + 1e-6
    return (F[torch.as_tensor(np.concatenate([stp["tr"], stp["te"]]), device=DEV)] - mu) / sd


def pe_eval(st, sets, col=slice(0, 1), loco=False):
    """f(S) for a batch of cell sets: returns (B, J) mean per-trial gains for target columns col. loco=True: sets ignored, members =
    V minus each cell (K_V - K_c)."""
    LAM = st["LAM"]; Y = st["Y"][:, col]; J = Y.shape[1]; n = st["n"]; nC = len(st["grid"]); B = nC if loco else len(sets)
    outer = {}; inner = {}
    for stp in st["steps"]:
        nt = len(stp["tr"])
        if loco:
            Z = step_Z(st, stp); Zf = Z.flatten(1); Kc = torch.einsum("ick,jck->cij", Z, Z); K = (Zf @ Zf.T)[None] - Kc; del Kc
        else:
            K = torch.stack([(lambda Zs: Zs @ Zs.T)(step_Z(st, stp, s_).flatten(1)) for s_ in sets])
        Ytr = Y[torch.as_tensor(stp["tr"], device=DEV)]; Yc = Ytr - Ytr.mean(0)
        alpha = torch.linalg.solve(K[:, :nt, :nt] + LAM * nt * torch.eye(nt, dtype=torch.float64, device=DEV), Yc.expand(B, -1, -1))
        P = K[:, nt:, :nt] @ alpha                                                                                   # (B, n_te, J)
        if stp["c"] < 0: outer[stp["b"]] = (stp, P)
        else: inner.setdefault(stp["b"], []).append((stp, P))
    sc = torch.zeros((B, n, J), dtype=torch.float64, device=DEV)
    for b, (stp, P) in outer.items():
        tr = stp["tr"]; pos = {t: i for i, t in enumerate(tr)}; xin = torch.zeros((B, len(tr), J), dtype=torch.float64, device=DEV)
        for s2, Pi in inner[b]: xin[:, [pos[t] for t in s2["te"]]] = Pi
        yt, ye = Y[torch.as_tensor(tr, device=DEV)][None], Y[torch.as_tensor(stp["te"], device=DEV)][None]; xm, ym = xin.mean(1, keepdim=True), yt.mean(1, keepdim=True)
        slope = ((xin - xm) * (yt - ym)).sum(1, keepdim=True) / ((xin - xm) ** 2).sum(1, keepdim=True); icpt = ym - slope * xm
        rv = ((yt - icpt - slope * xin) ** 2).mean(1, keepdim=True); vv = ((yt - ym) ** 2).mean(1, keepdim=True)
        sc[:, torch.as_tensor(stp["te"], device=DEV)] = (torch.log(vv / rv) + (ye - ym) ** 2 / vv - (ye - icpt - slope * P) ** 2 / rv) / (2 * LN2)
    return sc.mean(1)


def pe_checks(st, s):
    """G2: dual f({c}) = pe_gain_batch primal for 3 cells; dual f(V) = primal (pe_gain_batch on all 8640 features)."""
    Fb = st["F"].cpu().numpy(); Ys = st["Y"].cpu().numpy()
    d1 = pe_eval(st, [[0], [1], [2]], col=slice(0, 4)).cpu().numpy()
    p1 = M.pe_gain_batch(np.transpose(Fb[:, :3], (1, 0, 2)), Ys, st["LAM"]).mean(1)
    dV = pe_eval(st, [list(range(len(st["grid"])))], col=slice(0, 4)).cpu().numpy()[0]
    pV = M.pe_gain_batch(Fb.reshape(len(Fb), -1)[None], Ys, st["LAM"]).mean(1)[0]
    chk = dict(single=float(np.abs(d1 - p1).max()), all=float(np.abs(dV - pV).max()))
    assert max(chk.values()) < 1e-8, chk
    return chk, dV


# ------------------------------------------------------------------ incremental greedy (exact; block-inverse / Woodbury updates)
def image_fixed_dev(st, S, a):
    """Direct ridge on the dev system (G_tr, C_tr; first 200 dev items) at a fixed alpha: G2 twin of ImageInc.value."""
    f = feat_idx([S])[0]; G = st["G_tr"][f][:, f].double() + a * torch.eye(len(f), dtype=torch.float64, device=DEV)
    p = st["Z_va"][:200][:, f].double() @ torch.linalg.solve(G, st["C_tr"][f].double()) + st["ym_tr"]; p = p / p.norm(dim=1, keepdim=True)
    return float(M.soft_bits_gpu((p @ st["T_va"][:200].T)[None])[0])


def image_fixed(st, S, a):
    """Direct ridge on the cells S at a fixed alpha, all training rows, test InfoNCE (the reported f(S_t))."""
    f = feat_idx([S])[0]; G = st["G"][f][:, f].double() + a * torch.eye(len(f), dtype=torch.float64, device=DEV)
    p = st["Z_te"][:, f].double() @ torch.linalg.solve(G, st["C"][f].double()) + st["ym"]; p = p / p.norm(dim=1, keepdim=True)
    return float(M.soft_bits_gpu((p @ st["Yn"].T)[None])[0])


class ImageInc:
    """DEV-criterion greedy state (binding G1 condition): ridge fit on the 90 % dev-fit training rows (G_tr, C_tr) at a fixed alpha (the
    participant's all-cell dev alpha), scored by 200-way InfoNCE among the first 200 dev items (their own CLIP features); exact block-inverse
    update. Test rows are never used here; test f(S_t) is computed separately (image_fixed) for the selected sequence."""
    def __init__(self, st, alpha):
        self.st, self.a, self.S = st, alpha, []; self.Ainv = None; self.W = None; self.ZW = None
        self.G, self.C, self.ym = st["G_tr"], st["C_tr"], st["ym_tr"]; self.Zq = st["Z_va"][:200]; self.Tq = st["T_va"][:200]
    def scores(self, cand):
        Fc = feat_idx([[c] for c in cand]); I = torch.eye(36, dtype=torch.float64, device=DEV); out = []
        for i in range(0, len(cand), 100):
            f = Fc[i:i + 100]; d = self.G[f[:, :, None], f[:, None, :]].double() + self.a * I; Cc = self.C[f].double(); Zqc = self.Zq[:, f].permute(1, 0, 2).double()
            if self.S:
                Sf = feat_idx([self.S])[0]; b = self.G[Sf][:, f].permute(1, 0, 2).double()
                u = self.Ainv @ b; s_ = d - b.transpose(1, 2) @ u; Wc = torch.linalg.solve(s_, Cc - b.transpose(1, 2) @ self.W)
                p = self.ZW[None] - (self.Zq[:, Sf].double() @ u) @ Wc + Zqc @ Wc
            else:
                Wc = torch.linalg.solve(d, Cc); p = Zqc @ Wc
            p = p + self.ym; p = p / p.norm(dim=2, keepdim=True); out.append(M.soft_bits_gpu(p @ self.Tq.T))
        return torch.cat(out)
    def add(self, c):
        f = feat_idx([[c]])[0]; I = torch.eye(36, dtype=torch.float64, device=DEV); d = self.G[f][:, f].double() + self.a * I; Cc = self.C[f].double()
        if not self.S:
            self.Ainv = torch.linalg.inv(d); self.W = self.Ainv @ Cc
        else:
            Sf = feat_idx([self.S])[0]; b = self.G[Sf][:, f].double(); u = self.Ainv @ b; si = torch.linalg.inv(d - b.T @ u)
            Wc = si @ (Cc - b.T @ self.W); self.W = torch.cat([self.W - u @ Wc, Wc])
            self.Ainv = torch.cat([torch.cat([self.Ainv + u @ si @ u.T, -u @ si], 1), torch.cat([-si @ u.T, si], 1)])
        self.S = self.S + [c]; self.ZW = self.Zq[:, feat_idx([self.S])[0]].double() @ self.W
    def value(self):   # dev value of the current S
        p = self.ZW + self.ym; p = p / p.norm(dim=1, keepdim=True); return float(M.soft_bits_gpu((p @ self.Tq.T)[None])[0])
    def test_value(self):
        return image_fixed(self.st, self.S, self.a)


class PEInc:
    """Dual ridge per fold step with (K_S + lam I)^-1 kept; candidates by Woodbury (rank-36 update), same nested calibration as pe_eval."""
    def __init__(self, st):
        self.st, self.S = st, []; self.Ainv = [None] * len(st["steps"])
    def _Z(self, stp, cells):
        return step_Z(self.st, stp, cells)                                                                       # (m, |cells|, 36)
    def scores(self, cand):
        """DEV criterion (binding G1 condition): inner 9-fold CV predictions inside every outer training fold, calibrated in-sample on those
        training rows (2 parameters), Gaussian density gain averaged over the training rows and the 10 outer folds. Test rows unused."""
        st = self.st; B = len(cand); y = st["Y"][:, :1]; dev = torch.zeros(B, dtype=torch.float64, device=DEV)
        inner_preds = {}
        for si, stp in enumerate(st["steps"]):
            if stp["c"] < 0: continue
            nt = len(stp["tr"]); lam = st["LAM"] * nt; Ytr = y[torch.as_tensor(stp["tr"], device=DEV)]; Yc = Ytr - Ytr.mean(0)
            Zc = self._Z(stp, cand).permute(1, 0, 2)
            if self.S:
                ZS = self._Z(stp, self.S).flatten(1); Ai = self.Ainv[si]; KSte = ZS[nt:] @ ZS[:nt].T; a0 = Ai @ Yc
            else:
                Ai = torch.eye(nt, dtype=torch.float64, device=DEV) / lam; KSte = torch.zeros((len(stp["te"]), nt), dtype=torch.float64, device=DEV); a0 = Yc / lam
            U = Zc[:, :nt]; AU = Ai @ U; inner_ = torch.eye(36, dtype=torch.float64, device=DEV) + U.transpose(1, 2) @ AU
            alpha = a0[None] - AU @ torch.linalg.solve(inner_, U.transpose(1, 2) @ a0[None])
            inner_preds.setdefault(stp["b"], []).append((stp, (KSte[None] + Zc[:, nt:] @ U.transpose(1, 2)) @ alpha))
        for b, lst in inner_preds.items():
            tr = next(s_ for s_ in st["steps"] if s_["b"] == b and s_["c"] < 0)["tr"]; pos = {t: i for i, t in enumerate(tr)}
            xin = torch.zeros((B, len(tr), 1), dtype=torch.float64, device=DEV)
            for s2, Pi in lst: xin[:, [pos[t] for t in s2["te"]]] = Pi
            yt = y[torch.as_tensor(tr, device=DEV)][None]; xm, ym = xin.mean(1, keepdim=True), yt.mean(1, keepdim=True)
            slope = ((xin - xm) * (yt - ym)).sum(1, keepdim=True) / ((xin - xm) ** 2).sum(1, keepdim=True); icpt = ym - slope * xm
            rv = ((yt - icpt - slope * xin) ** 2).mean(1, keepdim=True); vv = ((yt - ym) ** 2).mean(1, keepdim=True)
            dev += ((torch.log(vv / rv) + (yt - ym) ** 2 / vv - (yt - icpt - slope * xin) ** 2 / rv) / (2 * LN2)).mean(1)[:, 0] / len(inner_preds)
        return dev
    def _gain(self, preds, B):
        st = self.st; Y = st["Y"][:, :1]; n = st["n"]; sc = torch.zeros((B, n, 1), dtype=torch.float64, device=DEV)
        for b in range(10):
            stp, P = preds[(b, -1)]; tr = stp["tr"]; pos = {t: i for i, t in enumerate(tr)}; xin = torch.zeros((B, len(tr), 1), dtype=torch.float64, device=DEV)
            for c in range(9):
                s2, Pi = preds[(b, c)]; xin[:, [pos[t] for t in s2["te"]]] = Pi
            yt, ye = Y[torch.as_tensor(tr, device=DEV)][None], Y[torch.as_tensor(stp["te"], device=DEV)][None]; xm, ym = xin.mean(1, keepdim=True), yt.mean(1, keepdim=True)
            slope = ((xin - xm) * (yt - ym)).sum(1, keepdim=True) / ((xin - xm) ** 2).sum(1, keepdim=True); icpt = ym - slope * xm
            rv = ((yt - icpt - slope * xin) ** 2).mean(1, keepdim=True); vv = ((yt - ym) ** 2).mean(1, keepdim=True)
            sc[:, torch.as_tensor(stp["te"], device=DEV)] = (torch.log(vv / rv) + (ye - ym) ** 2 / vv - (ye - icpt - slope * P) ** 2 / rv) / (2 * LN2)
        return sc.mean(1)[:, 0]
    def add(self, c):
        self.S = self.S + [c]; st = self.st
        for si, stp in enumerate(st["steps"]):
            nt = len(stp["tr"]); Z = self._Z(stp, self.S).flatten(1)[:nt]
            self.Ainv[si] = torch.linalg.inv(Z @ Z.T + st["LAM"] * nt * torch.eye(nt, dtype=torch.float64, device=DEV))
    def test_value(self):
        return float(pe_eval(self.st, [self.S])[0, 0])


# ------------------------------------------------------------------ driver
def greedy(incs, sel, max_cells, nC):
    """Forward greedy on the mean DEV score over participants in `sel` (incremental exact updates). Every step records each participant's
    dev score of the chosen set and its TEST f(S_t) (reported only, never used for selection)."""
    curve = []
    for t in range(max_cells):
        S = incs[0].S; cand = [c for c in range(nC) if c not in S]
        sc = torch.stack([incs[p].scores(cand) for p in sel]); score = sc.mean(0); j = int(score.argmax()); best = cand[j]   # dev scores: selection half only
        for inc in incs: inc.add(best)
        curve.append(dict(cell=best, dev_sel=sc[:, j].cpu().numpy().tolist(), f=[inc.test_value() for inc in incs]))
        print("  step", t + 1, "cell", best, "sel dev-mean", round(float(score.max()), 4), "test-mean", round(float(np.mean(curve[-1]["f"])), 4), flush=True)
    return incs[0].S, curve


def boot_ci(v, B=10000, seed=20261121):
    vt = torch.as_tensor(np.asarray(v, dtype=float), device=DEV); g = torch.Generator(device=DEV); g.manual_seed(seed)
    ix = torch.randint(0, len(vt), (B, len(vt)), generator=g, device=DEV)
    return torch.quantile(vt[ix].mean(1), torch.tensor([.025, .975], dtype=vt.dtype, device=DEV)).tolist()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--target", required=True, choices=["image", "pe"]); ap.add_argument("--out", required=True)
    ap.add_argument("--subjects", default=""); ap.add_argument("--max_cells", type=int, default=60)
    ap.add_argument("--frac", type=float, default=1.0, help="training-size arm (contrarian fix 1): fraction of training rows")
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True); t0 = time.time()
    if a.target == "image":
        import run as R
        subs = [int(x) for x in a.subjects.split(",")] if a.subjects else list(range(1, 11))
        states, loco = [], []
        for s in subs:
            st = image_state(s, R, a.frac)
            m3 = out.parent / "m3" / f"image_{st['sub']}.json"
            if m3.exists() and a.frac == 1.0:   # G2: single-cell f = the M3 map cell
                ref = {(c["region"], c["band"], c["t_ms"]): c["yield_"] for c in json.loads(m3.read_text())["cells"] if c["kind"] == "combined"}
                f1 = image_eval(st, [[0], [1], [2]], R).cpu().numpy(); d = max(abs(f1[i] - ref[tuple(st["grid"][i])]) for i in range(3))
                assert d < 1e-4, d; print(st["sub"], "G2 single-cell vs M3 map", d, flush=True)
            L = image_loco(st, R); loco.append(L); states.append(st)
            print(st["sub"], "fV", round(L["fV"], 4), "downdate check", L["downdate_check"], f"{time.time()-t0:.0f}s", flush=True)
        mk = lambda i: ImageInc(states[i], loco[i]["fV_alpha"])
    else:
        from i2_complementarity import targets
        evaluate, allsubs, fpath, _ = targets(); subs = [int(x) for x in a.subjects.split(",")] if a.subjects else allsubs
        states, loco = [], []
        for s in subs:
            st = pe_state(s, evaluate, fpath, a.frac); chk, dV = pe_checks(st, s); nC = len(st["grid"])
            fm = pe_eval(st, None, loco=True).cpu().numpy()[:, 0]
            loco.append(dict(fV=float(dV[0]), floor_shift=float(dV[1:].mean()), f_minus=fm.tolist(), g2=chk)); states.append(st)
            print(st["sub"], "fV", round(float(dV[0]), 4), "G2", chk, f"{time.time()-t0:.0f}s", flush=True)
        mk = lambda i: PEInc(states[i])
    nC = len(states[0]["grid"]); fV = np.array([L["fV"] for L in loco])
    # B2 group statistics
    D = fV[:, None] - np.array([L["f_minus"] for L in loco]); n = len(D)
    tt = lambda X: X.mean(0) / (X.std(0, ddof=1) / np.sqrt(len(X)) + 1e-12); tobs = tt(D)
    flips = (np.array([[1 if (i >> j) & 1 == 0 else -1 for j in range(n)] for i in range(2 ** n)]) if n <= 12
             else np.vstack([np.ones(n), np.random.default_rng(20261121).choice([-1, 1], size=(9999, n))]))
    Dg = torch.as_tensor(D, device=DEV); Fl = torch.as_tensor(flips, dtype=torch.float64, device=DEV)
    X = Fl[:, :, None] * Dg[None]; tmax = (X.mean(1) / (X.std(1) / np.sqrt(n) + 1e-12)).max(1).values.cpu().numpy()
    b2 = [dict(region=g[0], band=g[1], t_ms=g[2], delta_mean=float(D[:, i].mean()), delta_ci=boot_ci(D[:, i]), p_maxstat=float((tmax >= tobs[i]).mean()))
          for i, g in enumerate(states[0]["grid"])]
    # A4 greedy, 2-fold cross-participant
    ids = [int(st["sub"].split("-")[1]) for st in states]; halves = {"odd": [i for i, x in enumerate(ids) if x % 2], "even": [i for i, x in enumerate(ids) if x % 2 == 0]}
    a4 = {}
    for name, sel in halves.items():
        other = [i for i in range(len(states)) if i not in sel]
        print("greedy on", name, flush=True); incs = [mk(i) for i in range(len(states))]
        if name == "odd":   # G2: incremental objects reproduce the direct evaluation for a 2-cell set
            for inc in incs[:2]: inc.add(0); inc.add(1)
            if a.target == "image":   # dev twin and test twin of the incremental object
                direct = [image_fixed_dev(states[i], [0, 1], loco[i]["fV_alpha"]) for i in range(2)]; inc_v = [incs[i].value() for i in range(2)]
            else:
                direct = [float(pe_eval(states[i], [[0, 1]])[0, 0]) for i in range(2)]; inc_v = [incs[i].test_value() for i in range(2)]
            print("G2 incremental vs direct", inc_v, direct, flush=True); assert max(abs(x - y) for x, y in zip(inc_v, direct)) < 1e-6
            if a.target == "pe":      # PE dev criterion: Woodbury candidate score for {0} + 1 equals a fresh PEInc built with {0, 1} scored on nothing new
                p0 = PEInc(states[0]); p0.add(0); d1 = float(p0.scores([1])[0]); p1 = PEInc(states[0]); p1.add(1); d2 = float(p1.scores([0])[0])
                print("G2 PE dev Woodbury {0}+1 vs {1}+0", d1, d2, flush=True); assert abs(d1 - d2) < 1e-8
            incs = [mk(i) for i in range(len(states))]
        S, curve = greedy(incs, sel, a.max_cells, nC)
        frac = lambda idx, t: float(np.mean([curve[t]["f"][i] for i in idx]) / np.mean(fV[idx]))
        t90 = next((t + 1 for t in range(len(curve)) if frac(sel, t) >= 0.90), None); t95 = next((t + 1 for t in range(len(curve)) if frac(sel, t) >= 0.95), None)
        heldout = lambda t: None if t is None else dict(frac_mean=frac(other, t - 1), per_participant=[curve[t - 1]["f"][i] / fV[i] for i in other])
        a4[name] = dict(selection=[states[i]["sub"] for i in sel], heldout=[states[i]["sub"] for i in other], cells=[states[0]["grid"][c] for c in S],
                        curve=[dict(t=t + 1, cell=states[0]["grid"][c["cell"]], sel_frac=frac(sel, t), heldout_frac=frac(other, t)) for t, c in enumerate(curve)],
                        size90=t90, size95=t95, heldout_at_size90=heldout(t90), heldout_at_size95=heldout(t95), f_by_participant=[c["f"] for c in curve])
    fl = np.array([max(L.get("floor_shuffled", -1e9), L.get("floor_noise", -1e9)) if a.target == "image" else L["floor_shift"] for L in loco])
    ex_ci = boot_ci(fV - fl); gate = bool(ex_ci[0] > 0)          # contrarian fix 3: cover fractions only where f(V) clears its floor
    single = {}
    for st_ in states:
        f_ = out.parent / "m3" / f"{a.target}_{st_['sub']}.json"
        if f_.exists():
            single[st_["sub"]] = {(c["region"], c["band"], c["t_ms"]): c["yield_"] for c in json.loads(f_.read_text())["cells"] if c["kind"] == "combined"}
    if single:   # B2 unique (leave-one-out) vs total (single-cell map) per cell
        for i, r in enumerate(b2):
            r["single_cell_mean"] = float(np.mean([v[(r["region"], r["band"], r["t_ms"])] for v in single.values()]))
    res = dict(target=a.target, frac=a.frac, participants=[st["sub"] for st in states], n_cells=nC, fV=dict(mean=float(fV.mean()), ci=boot_ci(fV), per_participant=fV.tolist()),
               floor_gate=dict(fV_minus_floor_ci=ex_ci, passes=gate, note="k90/k95 and fractions are interpreted only if passes"),
               floors={k: float(np.mean([L[k] for L in loco])) for k in loco[0] if k.startswith("floor")}, loco_participants=loco, b2=b2,
               b2_n_sig=int(sum(r["p_maxstat"] < 0.05 for r in b2)), a4=a4, runtime_s=round(time.time() - t0, 1), note=__doc__)
    (out / f"{a.target}_cover_frac{a.frac:g}.json").write_text(json.dumps(res, indent=1)); print("done", res["runtime_s"], flush=True)


if __name__ == "__main__":
    main()
