"""M6 (= A5; rung R5, appendix cross-check): linear variational information bottleneck (VIB) vs the bottleneck references.
Spec: EXTENSIONS_LOOP_20261121 "## M6 PROPOSAL", "## M6 REVIEW" (METHOD 04:50), "## T3 RULINGS" (FiLM variant dropped); coordinator 05:30:
lower priority than M7/A1, kept as an appendix tuning/cross-check.
Encoder   : linear-Gaussian, z = W x + b + s * eps, s = exp(log_s) per unit (input-independent), K = 256 units; prior N(0, I).
Head      : linear. Image: z -> CLIP ViT-H/14 (1024), Gaussian distortion 0.5 * ||y - yhat||^2 on CLIP centred and scaled by ONE global sd
            (keeps the ridge geometry); test score = InfoNCE log-score (torch port of run.soft_bits) of the mean-encoder prediction over the
            200 official test images (k = 80 and k = 40 repetition means). PE: z -> scalar, distortion 0.5 * (y - yhat)^2 on the standardized
            residualised PE; test score = held-out Gaussian log-density gain over intercept-only (calibration a + b * pred and residual
            variance fit on the member's DEV rows; reference mean/variance on its training rows).
Loss      : distortion + beta * KL(q(z|x) || N(0, I)) (nats per row). Rate = mean KL in bits per row on DEV rows at the best epoch;
            per-unit KL gives the effective dimension = #units with KL > 0.1 bit (defined in advance, METHOD M6 REVIEW item 2).
Betas     : BETAS = 12 log-spaced values in [1e-4, 1e1] fixed before outcomes, plus a deterministic beta = 0 member (s -> 0, no noise) for G2.
Members   : ONE batched model per participant (member axis via einsum; Adam state is elementwise so one AdamW = per-member AdamW; losses summed
            over members so gradients are independent). Image members = {eeg, noise input, shuffled training targets} x 12 betas + det (37).
            PE members = 10 contiguous folds (belief_bottleneck.folds) x {eeg, noise, circular shift n/2} x 12 betas + 10 det (370).
Training  : AdamW lr 1e-3, wd 0, batch 256 (image) / 64 (PE), max 150 epochs, per-member early stopping (patience 10) on the member's own dev
            objective (mean-encoder distortion + beta * KL); the test predictions, dev rate and per-unit KL of the best dev epoch are kept.
            Dev = the Q7/M7 held-out tenth of the training images (image); the last 10 % of each training fold (PE, as d6 fit_nn_ens).
Inputs    : image = raw voltage 63 x 250 (repetition-mean, standardized on training rows), 15,750 dims; PE = results/features/raw (X7 1472
            features, in-fold z-scoring * colw). Noise input = N(0, 1) of the same shape (standardized space).
Reference : ridge on the SAME training rows (image: primal, alpha by MRR on dev from run.ALPHAS; PE: LAM * n), scored by the same estimator.
            G2 = the det member's test predictions correlate >= 0.98 with ridge, and its yield lies within [0.9 x ridge, ridge + 0.02].
Report    : yield vs rate per beta; R90 = smallest dev rate whose yield >= 90 % of the max eeg yield; effective dims at the R90 beta (image);
            beside them P5 R90 (retained bits), representation-side L90 (RRR) and the input-side PCA L90 (context only).
Output    : results/ext/m6/<target>_<sub>.json; --summarize image|pe.
"""
import sys, json, time, math, argparse
from pathlib import Path
import numpy as np, torch
ROOT = Path('/path/to/eegaudit'); sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'experiments/eegbits')); sys.path.insert(0, str(ROOT / 'experiments/ext'))
DEV = torch.device('cuda'); OUT = ROOT / 'results/ext/m6'; OUT.mkdir(parents=True, exist_ok=True)
BETAS = [float(b) for b in np.logspace(-4, 1, 12)]; K = 256; LN2 = math.log(2)
ap = argparse.ArgumentParser(); ap.add_argument('--target', default='image'); ap.add_argument('--subs', default='1'); ap.add_argument('--smoke', action='store_true')
ap.add_argument('--max_epochs', type=int, default=150); ap.add_argument('--summarize', default=''); ap.add_argument('--betas', default='')
a = ap.parse_args()
if a.betas: BETAS = [float(b) for b in a.betas.split(',')]
sys.argv = sys.argv[:1]                                                    # m7_latent parses its own args at import
from m7_latent import metrics_t, soft_bits_t, boot_ci, gain_rows_t


class VIB(torch.nn.Module):
    def __init__(self, M, D, O, seed):
        super().__init__(); g = torch.Generator().manual_seed(seed)
        self.W1 = torch.nn.Parameter(torch.randn(M, D, K, generator=g) / math.sqrt(D)); self.b1 = torch.nn.Parameter(torch.zeros(M, K))
        self.ls = torch.nn.Parameter(torch.full((M, K), -2.0)); self.W2 = torch.nn.Parameter(torch.randn(M, K, O, generator=g) / math.sqrt(K))
        self.b2 = torch.nn.Parameter(torch.zeros(M, O))

    def forward(self, X, noisy, sample):
        """X [M, B, D] (or [B, D] shared) -> yhat [M, B, O], per-unit KL [M, B, K] (nats)."""
        mu = (torch.einsum('bd,mdk->mbk', X, self.W1) if X.dim() == 2 else torch.einsum('mbd,mdk->mbk', X, self.W1)) + self.b1[:, None]
        s2 = torch.exp(2 * self.ls)[:, None]; kl = 0.5 * (mu ** 2 + s2 - 1 - 2 * self.ls[:, None]) * noisy[:, None, None]
        z = mu + (torch.randn_like(mu) * torch.exp(self.ls)[:, None] * noisy[:, None, None] if sample else 0)
        return torch.einsum('mbk,mko->mbo', z, self.W2) + self.b2[:, None], kl


def fit(Xsrc, src, Yfn, beta, noisy, train_mask, dev_mask, test_fn, O, bs, seed):
    """Generic batched trainer. Xsrc: list of [n, D] (or [F, n, D]) sources; src[m] picks the source (and fold) of member m via Xget.
    Yfn(rows) -> [M, len(rows), O] member targets; masks [M, n] bool. test_fn(net) -> dict of [M, ...] test outputs. Returns best snapshots."""
    torch.manual_seed(seed); M = len(beta); n = train_mask.shape[1]
    net = VIB(M, Xsrc(None, torch.arange(1, device=DEV)).shape[-1], O, seed).to(DEV); opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=0.0)
    B = torch.tensor(beta, device=DEV); NZ = torch.tensor(noisy, device=DEV, dtype=torch.float32)
    best = torch.full((M,), float('inf'), device=DEV); bad = torch.zeros(M, dtype=torch.long, device=DEV); live = torch.ones(M, dtype=torch.bool, device=DEV)
    snap = None; best_ep = torch.zeros(M, dtype=torch.long, device=DEV); tw = train_mask.float(); dw = dev_mask.float()
    for ep in range(a.max_epochs):
        net.train(); perm = torch.randperm(n, device=DEV)
        for b in range(0, n, bs):
            i = perm[b:b + bs]; w = tw[:, i]
            if w.sum() < 2: continue
            yh, kl = net(Xsrc(None, i), NZ, True); loss_r = 0.5 * ((yh - Yfn(i)) ** 2).sum(-1) + B[:, None] * kl.sum(-1)
            loss = ((loss_r * w).sum(1) / w.sum(1).clamp_min(1)).sum(); opt.zero_grad(); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            di = torch.nonzero(dev_mask.any(0)).squeeze(1); obj = torch.zeros(M, device=DEV); klu = torch.zeros(M, K, device=DEV)
            for c in range(0, len(di), 2 * bs):
                i = di[c:c + 2 * bs]; yh, kl = net(Xsrc(None, i), NZ, False); w = dw[:, i]
                obj += ((0.5 * ((yh - Yfn(i)) ** 2).sum(-1) + B[:, None] * kl.sum(-1)) * w).sum(1); klu += (kl * w[..., None]).sum(1)
            obj /= dw.sum(1); klu /= dw.sum(1)[:, None]; imp = live & (obj < best - 1e-6)
            if imp.any():
                cur = dict(test_fn(net), dev_klu=klu / LN2)
                if snap is None: snap = {k: v.clone() for k, v in cur.items()}
                else:
                    for k_, v in cur.items(): snap[k_][imp] = v[imp]
            best_ep = torch.where(imp, torch.full_like(best_ep, ep), best_ep); best = torch.where(imp, obj, best)
            bad = torch.where(imp, torch.zeros_like(bad), bad + live.long()); live = live & (bad < 10)
        if not live.any(): break
    snap['best_epoch'] = best_ep; snap['epochs_run'] = ep + 1; snap['dev_index'] = di
    return snap


# ---------------- image ----------------
def run_image(sub):
    from run import CLIP, load_torch, load_raw, ALPHAS
    t0 = time.time(); sid = f'sub-{sub:02d}'
    Xtr4, Xte80, meta = load_raw(sid); X = torch.from_numpy(Xtr4.mean(1).reshape(len(Xtr4), -1)).to(DEV); del Xtr4
    T80 = torch.from_numpy(Xte80.mean(1).reshape(200, -1)).to(DEV); T40 = torch.from_numpy(Xte80[:, :40].mean(1).reshape(200, -1)).to(DEV); del Xte80
    mu, sd = X.mean(0), X.std(0, unbiased=False) + 1e-6; X = (X - mu) / sd; T80 = (T80 - mu) / sd; T40 = (T40 - mu) / sd
    Ytr = torch.from_numpy(load_torch(CLIP / 'ViT-H-14_features_train.pt')).to(DEV); Yte = torch.from_numpy(load_torch(CLIP / 'ViT-H-14_features_test.pt')).to(DEV)
    Ym = Ytr.mean(0); gsd = float((Ytr - Ym).pow(2).sum(1).mean().sqrt() / math.sqrt(Ytr.shape[1])); Ys = (Ytr - Ym) / gsd
    n = len(X); idx = torch.randperm(n, generator=torch.Generator().manual_seed(0)).to(DEV); cut = int(0.9 * n); tr, va = idx[:cut], idx[cut:]
    Xn = torch.randn(X.shape, generator=torch.Generator().manual_seed(1000 + sub)).to(DEV); Xn_te = torch.randn(T80.shape, generator=torch.Generator().manual_seed(2000 + sub)).to(DEV)
    perm_y = torch.randperm(n, generator=torch.Generator().manual_seed(3000 + sub)).to(DEV)
    # ridge reference on the same training rows (primal, eigendecomposition once, float64)
    A = X[tr].double(); C = A.T @ A; w, V = torch.linalg.eigh(C); del C; AtY = A.T @ Ys[tr].double(); VtAtY = V.T @ AtY; del A, AtY
    def ridge_pred(al, Z): return (Z.double() @ V) @ (VtAtY / (w + al)[:, None])
    al = max(ALPHAS, key=lambda q: metrics_t(ridge_pred(q, X[va]).float(), Ys[va])['mrr'])
    R80, R40 = ridge_pred(al, T80).float(), ridge_pred(al, T40).float(); del V
    ridge = dict(alpha=al, k80=metrics_t(R80 * gsd + Ym, Yte), k40=metrics_t(R40 * gsd + Ym, Yte)); torch.cuda.empty_cache()
    print(f"{sid} ridge alpha {al:g}: k80 {ridge['k80']['soft_bits']:.4f} k40 {ridge['k40']['soft_bits']:.4f} ({time.time()-t0:.0f}s)", flush=True)
    conds = ['eeg'] * 12 + ['noise'] * 12 + ['tshuffle'] * 12 + ['eeg']; beta = BETAS * 3 + [0.0]; noisy = [1.0] * 36 + [0.0]; M = len(beta)
    is_noise = torch.tensor([c == 'noise' for c in conds], device=DEV); is_tsh = torch.tensor([c == 'tshuffle' for c in conds], device=DEV)
    def Xsrc(_, i):
        return torch.where(is_noise[:, None, None], Xn[i][None], X[i][None])
    def Yfn(i): return torch.where(is_tsh[:, None, None], Ys[perm_y[i]][None], Ys[i][None])
    trm = torch.zeros(M, n, dtype=torch.bool, device=DEV); trm[:, tr] = True; dvm = torch.zeros(M, n, dtype=torch.bool, device=DEV); dvm[:, va] = True
    def test_fn(net):
        p80, _ = net(T80, torch.zeros(M, device=DEV), False); p40, _ = net(T40, torch.zeros(M, device=DEV), False); pn, _ = net(Xn_te, torch.zeros(M, device=DEV), False)
        p80 = torch.where(is_noise[:, None, None], pn, p80); return dict(p80=p80, p40=p40)
    snap = fit(Xsrc, None, Yfn, beta, noisy, trm, dvm, test_fn, Ys.shape[1], 256, sub)
    rows = []
    for m in range(M):
        r = dict(cond=conds[m], beta=beta[m], deterministic=noisy[m] == 0, best_epoch=int(snap['best_epoch'][m]), rate_bits=float(snap['dev_klu'][m].sum()),
                 eff_dims=int((snap['dev_klu'][m] > 0.1).sum()), k80=metrics_t(snap['p80'][m] * gsd + Ym, Yte), k40=metrics_t(snap['p40'][m] * gsd + Ym, Yte))
        if conds[m] == 'eeg': r['shuffled_pair_k80'] = metrics_t(snap['p80'][m] * gsd + Ym, Yte[torch.randperm(200, generator=torch.Generator().manual_seed(sub)).to(DEV)])['soft_bits']
        rows.append(r)
    det = M - 1; g2 = dict(corr_det_vs_ridge_k80=float(torch.corrcoef(torch.stack([snap['p80'][det].flatten(), R80.flatten()]))[0, 1]),
                           det_k80=rows[det]['k80']['soft_bits'], ridge_k80=ridge['k80']['soft_bits'])
    g2['pass'] = g2['corr_det_vs_ridge_k80'] >= 0.98 and 0.9 * g2['ridge_k80'] <= g2['det_k80'] <= g2['ridge_k80'] + 0.02
    res = dict(target='image', subject=sid, n_train=int(len(tr)), n_dev=int(len(va)), K=K, betas=BETAS, ridge=ridge, g2=g2, rows=rows, epochs_run=snap['epochs_run'],
               runtime_s=round(time.time() - t0, 1))
    (OUT / f"image_{sid}{'_smoke' if a.smoke else ''}.json").write_text(json.dumps(res, indent=1))
    for r in rows: print(f"  {r['cond']:8s} beta {r['beta']:.2e} rate {r['rate_bits']:9.2f} bits effdim {r['eff_dims']:3d} k80 {r['k80']['soft_bits']:.4f} ep {r['best_epoch']}", flush=True)
    print(sid, 'G2', g2, round(time.time() - t0), 's', flush=True)


# ---------------- PE ----------------
def run_pe(subs_arg):
    from belief_bottleneck import LAM, folds
    sys.path.insert(0, str(ROOT / 'experiments/loop'))
    src = open(ROOT / 'experiments/loop/s13b_gauss_bits.py').read(); src = src[:src.index('DATASETS = {')]; ns = {}; exec(compile(src, 's13b_head', 'exec'), ns)
    qtil, subs, fpath, _ = ns['targets_ds003458']()
    run_subs = subs[:1] if a.smoke else ([int(x) for x in subs_arg.split(',')] if subs_arg not in ('all', '') else subs)
    for s in run_subs:
        dest = OUT / f"pe_sub-{s:03d}{'_smoke' if a.smoke else ''}.json"
        if dest.exists() and not a.smoke: continue
        t0 = time.time(); fz = np.load(fpath('raw', s)); tri = fz['tri']; n = len(tri); fo = folds(n)
        F = torch.tensor(fz['F'], dtype=torch.float32, device=DEV); cw = torch.tensor(fz['colw'], dtype=torch.float32, device=DEV)
        Fn = torch.tensor(np.random.default_rng(1000 + s).standard_normal(tuple(F.shape)), dtype=torch.float32, device=DEV)
        y = torch.tensor(qtil(s, tri), dtype=torch.float32, device=DEV); ysh = torch.roll(y, n // 2)
        Xall = torch.zeros(2, 10, n, F.shape[1], device=DEV); trm0 = torch.zeros(10, n, dtype=torch.bool, device=DEV); dvm0 = trm0.clone(); tem0 = trm0.clone()
        ystd = torch.zeros(2, 10, n, device=DEV); ref = {}
        for b in range(10):
            tr = np.where(fo != b)[0]; nd = max(1, len(tr) // 10); fit_r, dev_r = tr[:-nd], tr[-nd:]
            trm0[b, fit_r] = True; dvm0[b, dev_r] = True; tem0[b, np.where(fo == b)[0]] = True
            for si, S in enumerate((F, Fn)):
                mu, sd = S[tr].mean(0), S[tr].std(0, unbiased=False) + 1e-6; Xall[si, b] = (S - mu) / sd * cw
            for yi, yy in enumerate((y, ysh)):
                m_, s_ = yy[fit_r].mean(), yy[fit_r].std() + 1e-8; ystd[yi, b] = (yy - m_) / s_
        conds = []; beta = []; noisy = []; fold = []
        for c in ('eeg', 'noise', 'shift'):
            for b in range(10):
                for be in BETAS: conds.append(c); beta.append(be); noisy.append(1.0); fold.append(b)
        for b in range(10): conds.append('eeg'); beta.append(0.0); noisy.append(0.0); fold.append(b)
        M = len(conds); fold_t = torch.tensor(fold, device=DEV); srcx = torch.tensor([c == 'noise' for c in conds], device=DEV).long(); srcy = torch.tensor([c == 'shift' for c in conds], device=DEV).long()
        Xsrc = lambda _, i: Xall[srcx[:, None], fold_t[:, None], i[None, :]]
        Yfn = lambda i: ystd[srcy[:, None], fold_t[:, None], i[None, :]][..., None]
        trm, dvm = trm0[fold_t], dvm0[fold_t]
        def test_fn(net):
            yh, _ = net(Xall[srcx, fold_t], torch.zeros(M, device=DEV), False); return dict(pall=yh[..., 0])
        snap = fit(Xsrc, None, Yfn, beta, noisy, trm, dvm, test_fn, 1, 64, s)
        # ridge reference on the same rows (LAM * n_fit), same calibration
        Pr = torch.zeros(2, 10, n, device=DEV)
        for si in range(2):
            for b in range(10):
                A = Xall[si, b][trm0[b]].double(); Yc = ystd[0, b][trm0[b]].double(); Yc = Yc - Yc.mean()
                W = torch.linalg.solve(A.T @ A + LAM * len(A) * torch.eye(A.shape[1], dtype=torch.float64, device=DEV), A.T @ Yc); Pr[si, b] = (Xall[si, b].double() @ W).float()
        def score(P, yy, b):
            """P [n] predictions of a fold-b model; calibration on dev rows, reference on training rows (fit + dev); returns test-row gains."""
            d, t, trall = dvm0[b], tem0[b], trm0[b] | dvm0[b]
            Dm = torch.stack([torch.ones(int(d.sum()), device=DEV), P[d]], 1).double(); c = torch.linalg.lstsq(Dm, yy[d].double()[:, None]).solution[:, 0]
            rv = ((yy[d].double() - Dm @ c) ** 2).mean(); m_ = yy[trall].double().mean(); vv = ((yy[trall].double() - m_) ** 2).mean()
            pe = c[0] + c[1] * P[t].double(); return (torch.log(vv / rv) + (yy[t].double() - m_) ** 2 / vv - (yy[t].double() - pe) ** 2 / rv) / (2 * LN2)
        rows = []
        for c_ in ('eeg', 'noise', 'shift'):
            yy = ysh if c_ == 'shift' else y
            for be in BETAS + ([0.0] if c_ == 'eeg' else []):
                ms = [m for m in range(M) if conds[m] == c_ and beta[m] == be]; G = torch.cat([score(snap['pall'][m], yy, fold[m]) for m in ms])
                rows.append(dict(cond=c_, beta=be, deterministic=be == 0.0, gain=float(G.mean()), rate_bits=float(np.mean([float(snap['dev_klu'][m].sum()) for m in ms])),
                                 eff_dims=float(np.mean([int((snap['dev_klu'][m] > 0.1).sum()) for m in ms])), best_epoch=float(np.mean([int(snap['best_epoch'][m]) for m in ms]))))
        rg = {k: float(torch.cat([score(Pr[si, b], yy, b) for b in range(10)]).mean()) for k, si, yy in (('eeg', 0, y), ('noise', 1, y), ('shift', 0, ysh))}
        dets = [m for m in range(M) if noisy[m] == 0]; Pd = torch.cat([snap['pall'][m][tem0[fold[m]]] for m in dets]); Pq = torch.cat([Pr[0, b][tem0[b]] for b in range(10)])
        g2 = dict(corr_det_vs_ridge=float(torch.corrcoef(torch.stack([Pd, Pq]))[0, 1]), det_gain=rows[-1]['gain'] if rows[-1]['deterministic'] else next(r['gain'] for r in rows if r['deterministic']), ridge_gain=rg['eeg'])
        g2['pass'] = g2['corr_det_vs_ridge'] >= 0.98 and 0.9 * g2['ridge_gain'] <= g2['det_gain'] <= g2['ridge_gain'] + 0.02
        res = dict(target='pe', subject=f'sub-{s:03d}', n_trials=n, K=K, betas=BETAS, ridge_devcal=rg, g2=g2, rows=rows, epochs_run=snap['epochs_run'], runtime_s=round(time.time() - t0, 1))
        dest.write_text(json.dumps(res, indent=1))
        for r in rows: print(f"  {r['cond']:6s} beta {r['beta']:.2e} rate {r['rate_bits']:8.3f} bits gain {r['gain']:.4f} ep {r['best_epoch']:.0f}", flush=True)
        print(f'sub-{s:03d} ridge {rg} G2 {g2} {time.time()-t0:.0f}s', flush=True)


def r90(rows, key):
    e = sorted([r for r in rows if r['cond'] == 'eeg' and not r['deterministic']], key=lambda r: r['rate_bits'])
    v = [key(r) for r in e]; mx = max(v); hit = next(r for r, x in zip(e, v) if x >= 0.9 * mx); return hit, mx


def summarize(target):
    R = [json.loads(f.read_text()) for f in sorted(OUT.glob(f'{target}_sub-*.json')) if 'smoke' not in f.name]
    key = (lambda r: r['k80']['soft_bits']) if target == 'image' else (lambda r: r['gain'])
    out = dict(target=target, n=len(R), betas=BETAS, per_participant={}, note=__doc__)
    for r in R:
        h, mx = r90(r['rows'], key); out['per_participant'][r['subject']] = dict(R90_bits=h['rate_bits'], beta_R90=h['beta'], eff_dims_R90=h.get('eff_dims'), max_yield=mx,
                                                                               ridge=r['ridge']['k80']['soft_bits'] if target == 'image' else r['ridge_devcal']['eeg'], g2=r['g2'])
    pp = out['per_participant']; v = lambda k: [p[k] for p in pp.values()]
    out['R90_bits'] = dict(median=float(np.median(v('R90_bits'))), mean=float(np.mean(v('R90_bits'))), ci=boot_ci(v('R90_bits')))
    out['vib_max_minus_ridge'] = dict(mean=float(np.mean(np.array(v('max_yield')) - np.array(v('ridge')))), ci=boot_ci(np.array(v('max_yield')) - np.array(v('ridge'))))
    out['g2_all_pass'] = all(p['g2']['pass'] for p in pp.values())
    curve = {}
    for c in ('eeg', 'noise', 'tshuffle', 'shift'):
        for be in BETAS:
            rr = [next((x for x in r['rows'] if x['cond'] == c and x['beta'] == be), None) for r in R]
            if rr[0] is None: continue
            curve[f'{c}_{be:.2e}'] = dict(rate=float(np.mean([x['rate_bits'] for x in rr])), yield_=float(np.mean([key(x) for x in rr])), yield_ci=boot_ci([key(x) for x in rr]))
    out['curve'] = curve
    if target == 'image':
        ys = json.loads((ROOT / 'results/stats/yield_stats.json').read_text())['image']; rep = ys['representation_nicega_Wolpaw']['L90_per_participant']
        p5 = {q['subject']: q['R90_bits'] for q in json.loads((ROOT / 'results/loop/p5/p5_rate_curve.json').read_text())['image']['per_participant']}
        ratio = [rep[s] / max(p['eff_dims_R90'], 1) for s, p in pp.items()]; lr = np.log(ratio)
        out['eff_dims_R90'] = dict(median=float(np.median(v('eff_dims_R90'))), ci=boot_ci(v('eff_dims_R90')))
        out['RRR_L90_over_VIB_effdims'] = dict(geo_mean=float(np.exp(lr.mean())), ci=[float(np.exp(x)) for x in boot_ci(lr)],
                                               disagree=bool(np.exp(boot_ci(lr)[0]) > 2 or np.exp(-boot_ci(lr)[1]) > 2))
        out['references'] = dict(P5_R90_bits_median=float(np.median([p5[s] for s in pp])), RRR_L90_median=float(np.median([rep[s] for s in pp])), PCA_L90_input='924 [589, 1038] (context only)')
    (OUT / f'summary_{target}.json').write_text(json.dumps(out, indent=1)); print(json.dumps({k: v for k, v in out.items() if k not in ('note', 'curve')}, indent=1)[:5000])


if __name__ == '__main__':
    if a.summarize: summarize(a.summarize)
    elif a.target == 'image':
        for s in [int(x) for x in a.subs.split(',')]:
            if (OUT / f"image_sub-{s:02d}.json").exists() and not a.smoke: continue
            run_image(s)
    else: run_pe(a.subs)
