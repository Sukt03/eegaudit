"""D6 (EXTENSIONS_LOOP_20261121 / EXPERIMENT_TABLES_FINAL section 3): is the low raw-EEG prediction-error yield caused by the linear read-out?
ds003458, regime W (within participant), same rows, folds, features, targets and estimator as X7 (experiments/loop/q1_pe_yield_bottleneck.py):
  features : results/features/raw (64 ch x 23 samples at 50 Hz, 150-600 ms = 1472 columns, column = ch * 23 + t), in-fold z-scoring * colw
  folds    : belief_bottleneck.folds (10 contiguous blocks); targets: RAW residualised PE of all 23 participants on the same trials (s13b design)
  width    : L = 64 -> in-fold PCA-64 (fit on the training fold only); L = full -> all 1472 columns.  EEGNet receives the rank-L in-fold PCA
             reconstruction reshaped to (64 ch, 23 samples) so that every decoder sees the same information at the same width.
  estimator: Gaussianised null-corrected bound exactly as q1/s13b: r_g = corr(pred_j, nscore(target_j)); d = atanh r_g,own - mean_j atanh r_g,j;
             bits_g = -1/2 log2(1 - tanh(max(d,0))^2).
Decoders (multi-output: one fit predicts all 23 target columns, so own and null columns go through the identical model):
  ridge  : closed form, lambda = belief_bottleneck.LAM (the paper's estimator; no tuning; G2 reproduces X7)
  krr    : kernel ridge, RBF kernel; 20 configs = lambda {1e-2, 1e-1, 1, 10} x gamma {1/4, 1/2, 1, 2, 4} / n_features
  mlp    : 1 hidden layer, ELU; 12 configs = hidden {64, 256} x weight decay {1e-4, 1e-3, 1e-2} x dropout {0, 0.5}; AdamW lr 1e-3
  eegnet : compact EEGNet (Lawhern 2018 layout: temporal conv k=7 (140 ms) F1=8 -> depthwise spatial conv D=2 -> BN/ELU/avgpool 2 -> separable
           conv F2=16 k=5 -> BN/ELU/avgpool 2 -> linear), sized for 23 samples at 50 Hz; 8 configs = dropout {.25, .5} x wd {1e-3, 1e-2} x lr {1e-3, 3e-3}
  MLP/EEGNet: standardised targets, MSE, max 60 epochs, early stopping (patience 8) on an inner dev block (last 10 % of the training fold).
Hyper-parameters (selection = leave-one-participant-out, G1 fix 1): pass 1 scores every config on every participant (fold 0's training set, inner
dev = its last 1/9, own-target correlation); participant s then uses the config with the highest MEAN dev score over the OTHER participants, so no
label of s enters s's selection. The chosen config is reused for every fold and every control condition (inject mode: per SNR).
MLP/EEGNet early stopping (G1 fix 2): on the OWN target column's dev MSE (training stays multi-output so the null columns pass through the identical
network and stopping epoch). --single_check: own-target and 5 seeded null columns fitted as single-output models (sub-001..003, full width) vs the
multi-output result, to bound the multi-output handicap.
Conditions: eeg; noise (N(0,1) input of the same shape, rng 1000+s as X7); shift (targets circularly shifted by n//2 trials against the EEG, train and
test); tshuffle (training targets permuted within the training fold; test targets intact).
D6.0 positive control (--inject SNRs): adds to every epoch a random-phase 10 Hz burst with a fixed random spatial pattern whose amplitude is
Phi(z_t), z = normal scores of the participant's own PE (monotone in signed PE; zero conditional mean of voltage given PE because the phase is random,
so invisible to any linear read-out); amplitude = SNR x the channel's sd. Oracle = the known-template quadrature power detector (no training), scored by the same
estimator; recovered share = decoder bits_g / oracle bits_g.
Output: results/ext/d6/<mode>_sub-XXX.json per participant, summary_<mode>.json (mode = ladder | inject); --smoke writes *_smoke files."""
import sys, json, time, argparse, itertools, numpy as np, torch, torch.nn as nn
from pathlib import Path
from scipy import stats
ROOT = Path('/path/to/eegaudit'); sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'experiments/loop'))
from belief_bottleneck import LAM, folds
from group_stats import DEV, dersimonian_laird, corr
src = open(ROOT / "experiments/loop/s13b_gauss_bits.py").read(); src = src[:src.index("DATASETS = {")]
ns = {}; exec(compile(src, "s13b_head", "exec"), ns); targets_ds003458 = ns["targets_ds003458"]; nscore = ns["nscore"]
ap = argparse.ArgumentParser()
ap.add_argument("--subs", default="all"); ap.add_argument("--dec", default="ridge,krr,mlp,eegnet"); ap.add_argument("--L", default="64,full")
ap.add_argument("--conds", default="eeg,noise,shift,tshuffle"); ap.add_argument("--inject", default=""); ap.add_argument("--max_epochs", type=int, default=60)
ap.add_argument("--smoke", action="store_true"); ap.add_argument("--single_check", action="store_true"); ap.add_argument("--single_output", action="store_true"); ap.add_argument("--dev_file", default=""); ap.add_argument("--pass1_only", action="store_true"); ap.add_argument("--summarize_only", action="store_true"); ap.add_argument("--tag", default=""); ap.add_argument("--ens", action="store_true")
a = ap.parse_args()
OUT = ROOT / "results/ext/d6"; OUT.mkdir(parents=True, exist_ok=True)
MODE = "inject" if a.inject else ("single" if a.single_check else "ladder"); SUF = ("_smoke" if a.smoke else "") + ("_so" if a.single_output else "") + (f"_{a.tag}" if a.tag else "")
gb = lambda d: float(-0.5 * np.log2(1 - min(np.tanh(max(d, 0.0)), 0.999) ** 2))
NCH, NT, SF = 64, 23, 50.0
GRID = dict(ridge=[dict()],
            krr=[dict(lam=l, g=g) for l in (1e-2, 1e-1, 1.0, 10.0) for g in (0.25, 0.5, 1.0, 2.0, 4.0)],
            mlp=[dict(h=h, wd=w, dr=d) for h in (64, 256) for w in (1e-4, 1e-3, 1e-2) for d in (0.0, 0.5)],
            eegnet=[dict(dr=d, wd=w, lr=l) for d in (0.25, 0.5) for w in (1e-3, 1e-2) for l in (1e-3, 3e-3)])


def stats_for(P, Yg, me):
    rg = np.array([corr(P[:, j], Yg[:, j]) for j in range(Yg.shape[1])]); nl = np.delete(rg, me); zn = np.arctanh(np.clip(nl, -0.999, 0.999))
    d = float(np.arctanh(np.clip(rg[me], -0.999, 0.999)) - zn.mean()); v = float(zn.var(ddof=1) * (1 + 1 / len(zn)))
    return dict(r_g=float(rg[me]), d=d, v=v, p95=float(np.quantile(nl, 0.95)), bits_g=gb(d))


class EEGNetS(nn.Module):
    def __init__(self, n_out, dr, F1=8, D=2, F2=16):
        super().__init__()
        self.b1 = nn.Sequential(nn.Conv2d(1, F1, (1, 7), padding=(0, 3), bias=False), nn.BatchNorm2d(F1),
                                nn.Conv2d(F1, F1 * D, (NCH, 1), groups=F1, bias=False), nn.BatchNorm2d(F1 * D), nn.ELU(), nn.AvgPool2d((1, 2)), nn.Dropout(dr))
        self.b2 = nn.Sequential(nn.Conv2d(F1 * D, F1 * D, (1, 5), padding=(0, 2), groups=F1 * D, bias=False), nn.Conv2d(F1 * D, F2, 1, bias=False),
                                nn.BatchNorm2d(F2), nn.ELU(), nn.AvgPool2d((1, 2)), nn.Dropout(dr))
        self.fc = nn.Linear(F2 * (NT // 4), n_out)

    def forward(self, x):
        return self.fc(self.b2(self.b1(x.view(-1, 1, NCH, NT))).flatten(1))


def fit_nn(kind, cfg, Xtr, Ytr, Xte, seed, me):
    """Standardised multi-output regression; early stopping on the last 10 % (contiguous) of the training fold."""
    torch.manual_seed(seed); np.random.seed(seed)
    mu, sd = Ytr.mean(0), Ytr.std(0) + 1e-8; Ys = (Ytr - mu) / sd
    nd = max(1, len(Xtr) // 10); Xa, Ya, Xd, Yd = Xtr[:-nd], Ys[:-nd], Xtr[-nd:], Ys[-nd:]
    t = lambda z: torch.tensor(z, dtype=torch.float32, device=DEV)
    Xa, Ya, Xd, Yd, Xe = t(Xa), t(Ya), t(Xd), t(Yd), t(Xte)
    if kind == "mlp":
        net = nn.Sequential(nn.Linear(Xa.shape[1], cfg["h"]), nn.ELU(), nn.Dropout(cfg["dr"]), nn.Linear(cfg["h"], Ya.shape[1])); lr = 1e-3
    else:
        net = EEGNetS(Ya.shape[1], cfg["dr"]); lr = cfg["lr"]
    net.to(DEV); opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=cfg["wd"])
    best, state, bad = 1e9, None, 0
    for ep in range(a.max_epochs):
        net.train(); perm = torch.randperm(len(Xa), device=DEV)
        for b in range(0, len(perm), 64):
            i = perm[b:b + 64]
            if len(i) < 2: continue
            opt.zero_grad(); loss = ((net(Xa[i]) - Ya[i]) ** 2).mean(); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad(): dl = float(((net(Xd)[:, me] - Yd[:, me]) ** 2).mean())   # own column only
        if dl < best - 1e-5: best, bad, state = dl, 0, {k: v.detach().clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= 8: break
    net.load_state_dict(state); net.eval()
    with torch.no_grad(): P = net(Xe).cpu().numpy()
    return P * sd + mu


class EnsMLP(nn.Module):
    """M independent copies of the single-output MLP, run as one batched GPU model (each copy starts from the same seeded init as fit_nn)."""
    def __init__(self, base, M):
        super().__init__(); l1, l2 = base[0], base[3]; self.dr = base[2].p
        self.w1 = nn.Parameter(l1.weight.detach().T.unsqueeze(0).repeat(M, 1, 1).clone()); self.b1 = nn.Parameter(l1.bias.detach().unsqueeze(0).repeat(M, 1).clone())
        self.w2 = nn.Parameter(l2.weight.detach().T.unsqueeze(0).repeat(M, 1, 1).clone()); self.b2 = nn.Parameter(l2.bias.detach().unsqueeze(0).repeat(M, 1).clone())

    def forward(self, x):                                   # x [B, i] -> [B, M]
        h = nn.functional.elu(torch.einsum("bi,mih->mbh", x, self.w1) + self.b1[:, None]); h = nn.functional.dropout(h, self.dr, self.training) if getattr(self, 'pd', None) is None else (h * (torch.rand_like(h) >= self.pd[:, None, None]).float() / (1 - self.pd[:, None, None]) if self.training and bool((self.pd > 0).any()) else h)
        return (torch.einsum("mbh,mho->mbo", h, self.w2) + self.b2[:, None])[..., 0].T


class EnsEEGNet(nn.Module):
    """M independent copies of EEGNetS(1) as grouped convolutions (groups never mix copies; BatchNorm/Dropout are per channel/element)."""
    def __init__(self, base, M, F1=8, D=2, F2=16):
        super().__init__(); self.M = M; c1, bn1, c2, bn2, _, _, dr = base.b1; c3, c4, bn3 = base.b2[0], base.b2[1], base.b2[2]
        def conv(src, cin, cout, k, g, pad=(0, 0)):
            c = nn.Conv2d(cin, cout, k, padding=pad, groups=g, bias=False); c.weight.data.copy_(src.weight.detach().repeat(M, 1, 1, 1)); return c
        def bn(src):
            b = nn.BatchNorm2d(src.num_features * M); b.weight.data.copy_(src.weight.detach().repeat(M)); b.bias.data.copy_(src.bias.detach().repeat(M)); return b
        self.b1 = nn.Sequential(conv(c1, M, M * F1, (1, 7), M, (0, 3)), bn(bn1), conv(c2, M * F1, M * F1 * D, (NCH, 1), M * F1), bn(bn2), nn.ELU(), nn.AvgPool2d((1, 2)), nn.Dropout(dr.p))
        self.b2 = nn.Sequential(conv(c3, M * F1 * D, M * F1 * D, (1, 5), M * F1 * D, (0, 2)), conv(c4, M * F1 * D, M * F2, 1, M), bn(bn3), nn.ELU(), nn.AvgPool2d((1, 2)), nn.Dropout(base.b2[5].p))
        self.w = nn.Parameter(base.fc.weight.detach().T.unsqueeze(0).repeat(M, 1, 1).clone()); self.b = nn.Parameter(base.fc.bias.detach().unsqueeze(0).repeat(M, 1).clone())

    def forward(self, x):                                   # x [B, NCH*NT] -> [B, M]
        z = self.b2(self.b1(x.view(-1, 1, NCH, NT).expand(-1, self.M, -1, -1))); z = z.reshape(z.shape[0], self.M, -1)
        return (torch.einsum("bmi,mio->bmo", z, self.w) + self.b[None])[..., 0]


def fit_nn_ens(kind, cfg, Xtr, Ytr, Xte, seed):
    """All target columns at once: identical to calling fit_nn on each column separately (same seeded init, same mini-batches, per-column
    early stopping on the column's own dev loss), except that dropout masks are drawn independently per column. Per-column loss terms are
    summed, so each column's gradient (and its AdamW state) equals the separate fit's. The prediction kept per column is the one at that
    column's best dev epoch before its patience ran out (= load_state_dict(best) in fit_nn)."""
    torch.manual_seed(seed); np.random.seed(seed)
    mu, sd = Ytr.mean(0), Ytr.std(0) + 1e-8; Ys = (Ytr - mu) / sd; M = Ys.shape[1]
    nd = max(1, len(Xtr) // 10); Xa, Ya, Xd, Yd = Xtr[:-nd], Ys[:-nd], Xtr[-nd:], Ys[-nd:]
    t = lambda z: torch.tensor(z, dtype=torch.float32, device=DEV)
    Xa, Ya, Xd, Yd, Xe = t(Xa), t(Ya), t(Xd), t(Yd), t(Xte)
    if kind == "mlp":
        base = nn.Sequential(nn.Linear(Xa.shape[1], cfg["h"]), nn.ELU(), nn.Dropout(cfg["dr"]), nn.Linear(cfg["h"], 1)); lr = 1e-3; net = EnsMLP(base, M)
    else:
        base = EEGNetS(1, cfg["dr"]); lr = cfg["lr"]; net = EnsEEGNet(base, M)
    net.to(DEV); opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=cfg["wd"])
    best = torch.full((M,), 1e9, device=DEV); bad = torch.zeros(M, dtype=torch.long, device=DEV); live = torch.ones(M, dtype=torch.bool, device=DEV)
    Pbest = torch.zeros(len(Xe), M, device=DEV)
    for ep in range(a.max_epochs):
        net.train(); perm = torch.randperm(len(Xa), device=DEV)
        for b in range(0, len(perm), 64):
            i = perm[b:b + 64]
            if len(i) < 2: continue
            opt.zero_grad(); loss = ((net(Xa[i]) - Ya[i]) ** 2).mean(0).sum(); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            dl = ((net(Xd) - Yd) ** 2).mean(0); imp = live & (dl < best - 1e-5)
            if imp.any(): Pe = net(Xe); Pbest[:, imp] = Pe[:, imp]
            best = torch.where(imp, dl, best); bad = torch.where(imp, torch.zeros_like(bad), bad + live.long()); live = live & (bad < 8)
        if not live.any(): break
    return Pbest.cpu().numpy() * sd + mu


class PDrop(nn.Module):
    """Dropout with a per-channel rate (channel axis 1); used to batch configs that differ only in dropout."""
    def __init__(self, p):
        super().__init__(); self.register_buffer("p", p); self.on = bool((p > 0).any())

    def forward(self, x):
        if not self.training or not self.on: return x
        q = self.p.view(1, -1, *([1] * (x.dim() - 2))); return x * (torch.rand_like(x) >= q).float() / (1 - q)


def fit_nn_cfgs(kind, cfgs, Xtr, ytr, Xte, seed):
    """Pass-1 selection on the GPU: several configs of ONE architecture fitted to ONE target column as one batched model. Equals fit_nn run
    once per cfg (same seeded init, same mini-batches, AdamW with each cfg's own lr and weight decay, per-cfg early stopping on the dev
    loss, prediction at the best dev epoch), except that dropout masks are drawn per cfg. Returns predictions [len(Xte), len(cfgs)]."""
    import math
    torch.manual_seed(seed); np.random.seed(seed)
    mu, sd = ytr.mean(0), ytr.std(0) + 1e-8; ys = (ytr - mu) / sd; M = len(cfgs)
    nd = max(1, len(Xtr) // 10); Xa, Ya, Xd, Yd = Xtr[:-nd], ys[:-nd], Xtr[-nd:], ys[-nd:]
    t = lambda z: torch.tensor(z, dtype=torch.float32, device=DEV)
    Xa, Ya, Xd, Yd, Xe = t(Xa), t(Ya), t(Xd), t(Yd), t(Xte)
    pv = torch.tensor([c["dr"] for c in cfgs], dtype=torch.float32)
    if kind == "mlp":
        assert len({c["h"] for c in cfgs}) == 1
        base = nn.Sequential(nn.Linear(Xa.shape[1], cfgs[0]["h"]), nn.ELU(), nn.Dropout(0.0), nn.Linear(cfgs[0]["h"], 1)); net = EnsMLP(base, M)
        net.pd = pv.to(DEV); lrs = [1e-3] * M
    else:
        base = EEGNetS(1, 0.0); net = EnsEEGNet(base, M); net.b1[6] = PDrop(pv.repeat_interleave(16)); net.b2[5] = PDrop(pv.repeat_interleave(16)); lrs = [c["lr"] for c in cfgs]
    net.to(DEV); params = list(net.parameters()); mom = [(torch.zeros_like(q), torch.zeros_like(q)) for q in params]
    LR, WD = torch.tensor(lrs, device=DEV), torch.tensor([c["wd"] for c in cfgs], device=DEV)
    vec = lambda v, q: v.repeat_interleave(q.shape[0] // M).view(-1, *([1] * (q.dim() - 1)))
    lrq, wdq = [vec(LR, q) for q in params], [vec(WD, q) for q in params]; step = 0
    best = torch.full((M,), 1e9, device=DEV); bad = torch.zeros(M, dtype=torch.long, device=DEV); live = torch.ones(M, dtype=torch.bool, device=DEV)
    Pbest = torch.zeros(len(Xe), M, device=DEV)
    for ep in range(a.max_epochs):
        net.train(); perm = torch.randperm(len(Xa), device=DEV)
        for b in range(0, len(perm), 64):
            i = perm[b:b + 64]
            if len(i) < 2: continue
            for q in params: q.grad = None
            loss = ((net(Xa[i]) - Ya[i]) ** 2).mean(0).sum(); loss.backward(); step += 1
            with torch.no_grad():                                  # AdamW (torch defaults: betas .9/.999, eps 1e-8, decoupled decay), per-cfg lr/wd
                b1, b2 = 1 - 0.9 ** step, 1 - 0.999 ** step
                for q, (m, v), lr, wd in zip(params, mom, lrq, wdq):
                    q.mul_(1 - lr * wd); m.lerp_(q.grad, 0.1); v.mul_(0.999).addcmul_(q.grad, q.grad, value=0.001)
                    q.sub_(lr / b1 * m / (v.sqrt() / math.sqrt(b2) + 1e-8))
        net.eval()
        with torch.no_grad():
            dl = ((net(Xd) - Yd) ** 2).mean(0); imp = live & (dl < best - 1e-5)
            if imp.any(): Pe = net(Xe); Pbest[:, imp] = Pe[:, imp]
            best = torch.where(imp, dl, best); bad = torch.where(imp, torch.zeros_like(bad), bad + live.long()); live = live & (bad < 8)
        if not live.any(): break
    return Pbest.cpu().numpy() * sd + mu


def fit_predict(dec, cfg, Atr, Ytr, Ate, seed, me):
    if dec == "ridge":
        A = torch.tensor(Atr, dtype=torch.float64, device=DEV); B = torch.tensor(Ate, dtype=torch.float64, device=DEV); Y = torch.tensor(Ytr, dtype=torch.float64, device=DEV)
        W = torch.linalg.solve(A.T @ A + LAM * len(A) * torch.eye(A.shape[1], dtype=torch.float64, device=DEV), A.T @ (Y - Y.mean(0)))
        return (B @ W).cpu().numpy()
    if dec == "krr":
        A = torch.tensor(Atr, dtype=torch.float64, device=DEV); B = torch.tensor(Ate, dtype=torch.float64, device=DEV); Y = torch.tensor(Ytr, dtype=torch.float64, device=DEV)
        g = cfg["g"] / A.shape[1]; K = torch.exp(-g * torch.cdist(A, A) ** 2); Kt = torch.exp(-g * torch.cdist(B, A) ** 2)
        al = torch.linalg.solve(K + cfg["lam"] * len(A) * torch.eye(len(A), dtype=torch.float64, device=DEV), Y - Y.mean(0))
        return (Kt @ al).cpu().numpy()
    return fit_nn(dec, cfg, Atr, Ytr, Ate, seed, me)


def design(F, colw, tr, te, L, dec):
    """In-fold z-scoring * colw, then the width bottleneck; EEGNet gets the rank-L reconstruction in channel x time layout."""
    mu, sd = F[tr].mean(0), F[tr].std(0) + 1e-6; A, B = (F[tr] - mu) / sd * colw, (F[te] - mu) / sd * colw
    if L == "full": return A, B
    m = A.mean(0); _, _, Vt = np.linalg.svd(A - m, full_matrices=False); Vk = Vt[:min(L, Vt.shape[0])].T
    if dec == "eegnet": return (A - m) @ Vk @ Vk.T + m, (B - m) @ Vk @ Vk.T + m
    return A @ Vk, B @ Vk


def cv(F, colw, Y, L, dec, cfg, seed, me, tshuffle=False):
    f = folds(len(F)); P = np.zeros_like(Y); rng = np.random.default_rng(seed)
    for b in range(10):
        tr, te = np.where(f != b)[0], np.where(f == b)[0]
        Atr, Ate = design(F, colw, tr, te, L, dec); Ytr = Y[tr][rng.permutation(len(tr))] if tshuffle else Y[tr]
        if a.single_output and a.ens and dec in ("mlp", "eegnet"): P[te] = fit_nn_ens(dec, cfg, Atr, Ytr, Ate, seed + b)   # same as below, batched on the GPU
        elif a.single_output and dec in ("mlp", "eegnet"):   # one network per target column (own and every null), each early-stopped on itself
            P[te] = np.column_stack([fit_predict(dec, cfg, Atr, Ytr[:, [j]], Ate, seed + b, 0)[:, 0] for j in range(Y.shape[1])])
        else: P[te] = fit_predict(dec, cfg, Atr, Ytr, Ate, seed + b, me)
    return P


def select(F, colw, Y, L, dec, me, seed):
    """Config choice on fold 0's training set only: inner dev = its last 1/9 (contiguous); criterion = own-target correlation."""
    if len(GRID[dec]) == 1: return GRID[dec][0], []
    f = folds(len(F)); tr = np.where(f != 0)[0]; nd = len(tr) // 9; itr, idv = tr[:-nd], tr[-nd:]
    scores = []
    if a.single_output and a.ens and dec in ("mlp", "eegnet"):     # batched on the GPU: one model per architecture, all its configs at once
        Atr, Adv = design(F, colw, itr, idv, L, dec); scores = [None] * len(GRID[dec]); grp = {}
        for i, c in enumerate(GRID[dec]): grp.setdefault(c.get("h"), []).append(i)
        for idx in grp.values():
            P = fit_nn_cfgs(dec, [GRID[dec][i] for i in idx], Atr, Y[itr][:, [me]], Adv, seed)
            for k, i in enumerate(idx): scores[i] = corr(P[:, k], Y[idv, me])
        return GRID[dec][int(np.nanargmax(scores))], [float(x) for x in scores]
    for cfg in GRID[dec]:
        Atr, Adv = design(F, colw, itr, idv, L, dec)
        if a.single_output and dec in ("mlp", "eegnet"): scores.append(corr(fit_predict(dec, cfg, Atr, Y[itr][:, [me]], Adv, seed, 0)[:, 0], Y[idv, me]))
        else: P = fit_predict(dec, cfg, Atr, Y[itr], Adv, seed, me); scores.append(corr(P[:, me], Y[idv, me]))
    return GRID[dec][int(np.nanargmax(scores))], [float(x) for x in scores]


def inject(F, z, snr, seed):
    """Random-phase 10 Hz burst, fixed random spatial pattern, amplitude Phi(z_t) * snr * channel sd. Returns F', template (w, sin, cos)."""
    rng = np.random.default_rng(seed); n = len(F); X = F.reshape(n, NCH, NT).copy()
    w = rng.standard_normal(NCH); w /= np.linalg.norm(w) / np.sqrt(NCH); tt = np.arange(NT) / SF
    ph = rng.uniform(0, 2 * np.pi, n); amp = stats.norm.cdf(z)
    sd = X.std(axis=(0, 2))
    sig = amp[:, None, None] * (snr * sd * w)[None, :, None] * np.sin(2 * np.pi * 10 * tt[None, None, :] + ph[:, None, None])
    return (X + sig).reshape(n, -1), w, np.sin(2 * np.pi * 10 * tt), np.cos(2 * np.pi * 10 * tt)


def oracle(F, w, s, c):
    X = F.reshape(len(F), NCH, NT); x = np.einsum("nct,c->nt", X, w); return (x @ s) ** 2 + (x @ c) ** 2


t0 = time.time(); qtil, subs, fpath, _ = targets_ds003458()
if a.subs == "all": use = subs
else:
    lo, hi = (int(x) for x in a.subs.split("-")) if "-" in a.subs else (int(a.subs), int(a.subs)); use = [s for s in subs if lo <= s <= hi]
DECS = a.dec.split(","); LS = [x if x == "full" else int(x) for x in a.L.split(",")]; CONDS = a.conds.split(",")
SNRS = [float(x) for x in a.inject.split(",")] if a.inject else []
SCHEMA = dict(output_mode="single-output MLP/EEGNet (one network per target column)" if a.single_output else "multi-output (one network, 23 columns)", dataset="ds003458", target="ds003458 residualised PE, normal scores", seed=0, bottleneck="input", split="10 contiguous blocks, within participant",
              selection="leave-one-participant-out: config = argmax of the mean fold-0 inner-dev own-target correlation over the OTHER participants", N=None, regime="W")


def load(s):
    fz = np.load(fpath("raw", s)); F0, colw, tri = fz["F"].astype(np.float64), fz["colw"].astype(np.float64), fz["tri"]
    assert F0.shape[1] == NCH * NT
    Y = np.column_stack([qtil(j, tri) for j in subs]); Yg = np.column_stack([nscore(Y[:, j]) for j in range(Y.shape[1])])
    return F0, colw, tri, Y, Yg, subs.index(s)


def lopo(devsc, s, dec):
    """Config for participant s = argmax of the mean dev score over the other participants (G1 fix 1)."""
    if len(GRID[dec]) == 1: return GRID[dec][0]
    others = [devsc[t] for t in devsc if t != s]; assert others, "LOPO selection needs >= 2 participants"
    return GRID[dec][int(np.nanargmax(np.nanmean(np.array(others), 0)))]


# pass 1: dev scores of every config for every participant (inject mode: on the injected data at each SNR)
if a.summarize_only:                                   # aggregate per-participant JSONs written by participant chunks
    per = {s: json.load(open(OUT / f"{MODE}_sub-{s:03d}{SUF}.json")) for s in use if (OUT / f"{MODE}_sub-{s:03d}{SUF}.json").exists()}
    print(f"summarize_only: {len(per)} participant files", flush=True)
else:
    dev = {}
    # pass 1 may be computed once for ALL participants and shared by participant chunks (--dev_file): LOPO needs every participant's dev scores
    if a.dev_file and not a.pass1_only:                  # comma-separated pass-1 files (e.g. one per participant chunk) are merged
        for fdev in a.dev_file.split(","):
            raw = json.load(open(fdev)); assert raw["mode"] == MODE and raw["single_output"] == a.single_output and raw["decs"] == DECS and raw["widths"] == [str(L) for L in LS]
            for k, d in raw["dev"].items(): dev.setdefault(tuple(x if i < 2 else float(x) for i, x in enumerate(k.split("|"))), {}).update({int(t): v for t, v in d.items()})
        print(f"pass 1 loaded from {a.dev_file}: {len(next(iter(dev.values())))} participants", flush=True)
    else:
        for s in use:
            F0, colw, tri, Y, Yg, me = load(s)
            for dec, L in itertools.product(DECS, LS):
                if MODE in ("ladder", "single"): dev.setdefault((dec, str(L)), {})[s] = select(F0, colw, Y, L, dec, me, 0)[1]
                else:
                    for snr in SNRS: dev.setdefault((dec, str(L), snr), {})[s] = select(inject(F0, Yg[:, me], snr, 7 + s)[0], colw, Y, L, dec, me, 0)[1]
        if a.dev_file:
            json.dump(dict(mode=MODE, single_output=a.single_output, decs=DECS, widths=[str(L) for L in LS], participants=use,
                           dev={"|".join(str(x) for x in k): {str(t): v for t, v in d.items()} for k, d in dev.items()}), open(a.dev_file, "w"))
        if a.pass1_only: print("PASS1_ONLY saved", a.dev_file, f"({time.time()-t0:.0f}s)", flush=True); sys.exit(0)
    print(f"pass 1 (selection scores) done ({time.time()-t0:.0f}s)", flush=True)
    per = {}
    for s in use:
        F0, colw, tri, Y, Yg, me = load(s)
        res = dict(SCHEMA, participant=f"sub-{s:03d}", n_epochs=int(len(tri)), mode=MODE, rows=[])
        if MODE == "ladder":
            for dec, L in itertools.product(DECS, LS):
                cfg = lopo(dev[(dec, str(L))], s, dec)
                for cond in CONDS:
                    F = np.random.default_rng(1000 + s).standard_normal(F0.shape) if cond == "noise" else F0
                    Yc, Ygc = (np.roll(Y, len(Y) // 2, 0), np.roll(Yg, len(Yg) // 2, 0)) if cond == "shift" else (Y, Yg)
                    st = stats_for(cv(F, colw, Yc, L, dec, cfg, 0, me, tshuffle=(cond == "tshuffle")), Ygc, me)
                    res["rows"].append(dict(decoder=dec, L=L, condition=cond, config=cfg, **st))
                    print(f"sub-{s:03d} {dec:6s} L={L} {cond:8s} bits_g {st['bits_g']:.4f} d {st['d']:+.3f} ({time.time()-t0:.0f}s)", flush=True)
                res["rows"][-1]["dev_scores_own"] = dev[(dec, str(L))][s]
        elif MODE == "inject":
            z = Yg[:, me]; cfg_top = {}
            for snr in SNRS:
                F, w, sn, cs = inject(F0, z, snr, 7 + s)
                o = stats_for(np.column_stack([oracle(F, w, sn, cs)] * Y.shape[1]), Yg, me)
                res["rows"].append(dict(decoder="oracle", L="full", condition="inject", snr=snr, **o))
                for dec in DECS:
                    for L in LS:
                        cfg = lopo(dev[(dec, str(L), snr)], s, dec); cfg_top[(dec, str(L))] = cfg; st = stats_for(cv(F, colw, Y, L, dec, cfg, 0, me), Yg, me)
                        b0 = stats_for(cv(F0, colw, Y, L, dec, cfg, 0, me), Yg, me)["bits_g"]       # same decoder + config without injection
                        res["rows"].append(dict(decoder=dec, L=L, condition="inject", snr=snr, config=cfg, share_of_oracle=(st["bits_g"] / o["bits_g"] if o["bits_g"] > 0 else None),
                                                bits_g_no_inject=b0, share_increment_of_oracle=((st["bits_g"] - b0) / o["bits_g"] if o["bits_g"] > 0 else None), **st))
                        print(f"sub-{s:03d} snr {snr} {dec:6s} L={L} bits_g {st['bits_g']:.4f} oracle {o['bits_g']:.4f} ({time.time()-t0:.0f}s)", flush=True)
            Fn = np.random.default_rng(1000 + s).standard_normal(F0.shape)
            for dec in DECS:                                   # noise row with the config selected at the top SNR (G1 fix 5)
                cfg = cfg_top[(dec, str(LS[-1]))]; st = stats_for(cv(Fn, colw, Y, LS[-1], dec, cfg, 0, me), Yg, me)
                res["rows"].append(dict(decoder=dec, L=LS[-1], condition="noise", snr=None, config=cfg, config_source=f"selected at snr {SNRS[-1]}", **st))
        else:                                                  # single-output check (G1 fix 2): own + 5 seeded null columns, each its own model
            nulls = sorted(np.random.default_rng(100 + s).choice([j for j in range(Y.shape[1]) if j != me], 5, replace=False).tolist())
            def d5(rg_own, rg_null): zn = np.arctanh(np.clip(rg_null, -0.999, 0.999)); return float(np.arctanh(np.clip(rg_own, -0.999, 0.999)) - zn.mean())
            for dec, L in itertools.product(DECS, LS):
                cfg = lopo(dev[(dec, str(L))], s, dec)
                Pm = cv(F0, colw, Y, L, dec, cfg, 0, me); rm = [corr(Pm[:, j], Yg[:, j]) for j in [me] + nulls]
                rs = [corr(cv(F0, colw, Y[:, [j]], L, dec, cfg, 0, 0)[:, 0], Yg[:, j]) for j in [me] + nulls]
                dm, ds_ = d5(rm[0], rm[1:]), d5(rs[0], rs[1:])
                res["rows"].append(dict(decoder=dec, L=L, condition="single_check", config=cfg, null_columns=nulls, multi_bits_g5=gb(dm), single_bits_g5=gb(ds_),
                                        multi_r_own=rm[0], single_r_own=rs[0], gain_single_minus_multi=gb(ds_) - gb(dm), bits_g=gb(ds_), r_g=rs[0], p95=float("nan")))
                print(f"sub-{s:03d} {dec:6s} L={L} single-check multi {gb(dm):.4f} single {gb(ds_):.4f} ({time.time()-t0:.0f}s)", flush=True)
        res["runtime_s"] = round(time.time() - t0, 1); per[s] = res
        json.dump(res, open(OUT / f"{MODE}_sub-{s:03d}{SUF}.json", "w"), indent=1, default=str)


# summary: per decoder x L x condition (x snr) mean bits over participants with bootstrap CI; ladder: paired (decoder - ridge) at each L
rng = np.random.default_rng(0); boot = lambda v: [float(np.percentile(m, q)) for m in [[np.mean(rng.choice(v, len(v))) for _ in range(4000)]] for q in (2.5, 97.5)]
keys = sorted({(r["decoder"], str(r["L"]), r["condition"], str(r.get("snr"))) for s in per for r in per[s]["rows"]})
summ = {}
for k in keys:
    v = np.array([r["bits_g"] for s in per for r in per[s]["rows"] if (r["decoder"], str(r["L"]), r["condition"], str(r.get("snr"))) == k])
    blk = dict(n=len(v), bits_g_mean_pp=float(v.mean()), bits_g_mean_pp_ci=boot(v) if len(v) > 1 else None)
    if k[2] == "eeg" or k[2] == "inject":
        blk["n_above_own_p95"] = int(sum(r["r_g"] > r["p95"] for s in per for r in per[s]["rows"] if (r["decoder"], str(r["L"]), r["condition"], str(r.get("snr"))) == k))
    if MODE == "ladder" and k[2] == "eeg" and k[0] != "ridge":
        ref = {s: r["bits_g"] for s in per for r in per[s]["rows"] if (r["decoder"], str(r["L"]), r["condition"]) == ("ridge", k[1], "eeg")}
        dd = np.array([r["bits_g"] - ref[s] for s in per for r in per[s]["rows"] if (r["decoder"], str(r["L"]), r["condition"], str(r.get("snr"))) == k and s in ref])
        if len(dd):
            blk["minus_ridge_mean"] = float(dd.mean()); blk["minus_ridge_ci"] = boot(dd) if len(dd) > 1 else None
            blk["n_positive"] = f"{int((dd > 0).sum())}/{len(dd)}"
            # TOST (G1 fix 4): margin = 0.20 x ridge mean bits at the same L; 90 % bootstrap CI of the paired mean difference; bootstrap p = max of the two one-sided p
            mg = 0.20 * float(np.mean(list(ref.values()))); bm = np.array([np.mean(rng.choice(dd, len(dd))) for _ in range(4000)])
            blk["tost_margin"] = mg; blk["minus_ridge_ci90"] = [float(np.percentile(bm, 5)), float(np.percentile(bm, 95))]
            blk["tost_p"] = float(max((bm <= -mg).mean(), (bm >= mg).mean())); blk["superiority_p"] = float((bm <= 0).mean())
    if k[2] == "inject" and k[0] != "oracle":
        sh = [r["share_of_oracle"] for s in per for r in per[s]["rows"] if (r["decoder"], str(r["L"]), r["condition"], str(r.get("snr"))) == k and r.get("share_of_oracle") is not None]
        blk["share_of_oracle_mean"] = float(np.mean(sh)) if sh else None
        si = [r["share_increment_of_oracle"] for s in per for r in per[s]["rows"] if (r["decoder"], str(r["L"]), r["condition"], str(r.get("snr"))) == k and r.get("share_increment_of_oracle") is not None]
        blk["share_increment_of_oracle_mean"] = float(np.mean(si)) if si else None
    if k[2] == "single_check":
        g = [r["gain_single_minus_multi"] for s in per for r in per[s]["rows"] if (r["decoder"], str(r["L"]), r["condition"]) == k[:3]]
        blk["gain_single_minus_multi_mean"] = float(np.mean(g)); blk["single_output_to_phase2"] = bool(np.mean(g) > 0.01)
    summ["|".join(k)] = blk
for L in LS:                                               # Holm across the nonlinear decoders within each L (TOST p and superiority p)
    ks = [k for k in summ if k.endswith(f"|{L}|eeg|None") and "tost_p" in summ[k]]
    for pk in ("tost_p", "superiority_p"):
        order = sorted(ks, key=lambda k: summ[k][pk]); m = len(order); run = 0.0
        for i, k in enumerate(order): run = max(run, min(1.0, (m - i) * summ[k][pk])); summ[k][pk + "_holm"] = run
json.dump(dict(SCHEMA, note=__doc__, mode=MODE, n=len(per), participants=[f"sub-{s:03d}" for s in per], decoders=DECS, widths=[str(L) for L in LS], summary=summ,
               runtime_s=round(time.time() - t0, 1)), open(OUT / f"summary_{MODE}{SUF}.json", "w"), indent=1)
for k, b in summ.items(): print("SUMMARY", k, round(b["bits_g_mean_pp"], 4), {x: (round(y, 4) if isinstance(y, float) else y) for x, y in b.items() if x in ("minus_ridge_mean", "share_of_oracle_mean", "n_above_own_p95", "tost_p_holm", "n_positive", "gain_single_minus_multi_mean", "share_increment_of_oracle_mean")})
print("D6_DONE", MODE, len(per), round(time.time() - t0, 1))
