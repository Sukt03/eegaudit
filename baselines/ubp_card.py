"""B1 UBP card (Wu et al. CVPR 2025, Apache-2.0; METHOD G1 PASS 08:45). IMPLEMENTER, 21 Nov 2026.
Stage G2: released intra-subject checkpoints (hf Haitao999/ubp_exp, last.ckpt, weights_only=True) on UBP's released test data
  (Preprocessed_data_250Hz_whiten, 17 posterior channels, samples 0:250, 80 repetitions averaged), RN50 fovea-blur 'medium' image features;
  top-1 vs the released test_results.json (target within 0.5 pt).
Stage YIELD (CHANGE 13): InfoNCE log-score (log2 200 - LOO-temperature cross-entropy over cosines, 40-point grid) of EEG embeddings vs the
  200 image embeddings in UBP's own RN50 space, at k = 80 and k = 40 (sessions 1-2 and 3-4 each, averaged); own split-half ceiling =
  the same estimator between encoder(sessions 1-2) and encoder(sessions 3-4) (both directions averaged); fraction at matched k = 40;
  floor = shuffled pairs.
Stage RANDOM-INIT twin: EEGProjectLayer re-initialised (torch default init, seed 0), frozen; ridge probe (GPU, penalty on a held-out tenth
  of training images) from its 1024-d output to the RN50 'medium' training features; then the same yield/ceiling.
Stage RETRAIN (batched faithful re-implementation of UBP main.py, no Lightning): per member = (participant, arm, seed), arm in {eeg, noise};
  EEGProjectLayer, ClipLoss on unnormalised EEG z vs normalised image z with softplus(logit_scale); AdamW lr 1e-4 wd 1e-4; bs 1024;
  <= 50 epochs; EarlyStopping on epoch train loss (min_delta 0.001, patience 5); test on the last weights; uncertainty-aware blur choice
  (gamma 0.3, alpha 0.05) with UBP's quirk kept (its validation loader is the test loader, and validation_step updates the blur state of
  training indices 0..199; one sanity validation batch before training). Noise arm: fresh N(0,1) input of identical shape at train and
  test (Jo et al. 2024). AdamW is elementwise, so one optimizer over member-stacked weights equals per-member optimizers; stopped
  members are restored after each step.
Output: results/baselines/ubp/ubp_card.json (+ per-member rows). Run: python -I -u ubp_card.py [--subs 1] [--seeds 0,1,2] [--skip_retrain]"""
import hashlib as _hl
CODE_SHA256 = _hl.sha256(open(__file__, 'rb').read()).hexdigest()  # every result JSON carries the exact code hash
import argparse, json, math, os, sys, time
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, '/path/to/eegaudit/experiments/baselines')
from safe_load import safe_load

ap = argparse.ArgumentParser()
ap.add_argument('--subs', default='1,2,3,4,5,6,7,8,9,10'); ap.add_argument('--seeds', default='0,1,2')
ap.add_argument('--epochs', type=int, default=50); ap.add_argument('--skip_retrain', action='store_true'); ap.add_argument('--retrain_only', action='store_true')
ap.add_argument('--out', default='/path/to/eegaudit/results/baselines/ubp')
ap.add_argument('--tag', default='')
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
dev = torch.device('cuda'); t0 = time.time()
D = '/path/to/eegaudit/data_raw/third_party/ubp'
CH = ['Fp1', 'Fp2', 'AF7', 'AF3', 'AFz', 'AF4', 'AF8', 'F7', 'F5', 'F3', 'F1', 'F2', 'F4', 'F6', 'F8', 'FT9', 'FT7', 'FC5', 'FC3', 'FC1',
      'FCz', 'FC2', 'FC4', 'FC6', 'FT8', 'FT10', 'T7', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'T8', 'TP9', 'TP7', 'CP5', 'CP3', 'CP1',
      'CPz', 'CP2', 'CP4', 'CP6', 'TP8', 'TP10', 'P7', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO3', 'POz', 'PO4', 'PO8',
      'O1', 'Oz', 'O2']
SEL = [CH.index(c) for c in ['P7', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO3', 'POz', 'PO4', 'PO8', 'O1', 'Oz', 'O2']]
T0, T1 = 0, 250
TAUS = torch.logspace(-2.5, 0, int(os.environ.get('CARD_NTAU', '12')), device=dev)  # FIXER L1 (LOCK §3): 12-point LOO grid PRIMARY; CARD_NTAU=40 = sensitivity
TAUS40 = torch.logspace(-2.5, 0, 40, device=dev)  # labelled sensitivity field (*_t40), computed in the same run
LOG2 = math.log(2)
subs = [int(s) for s in a.subs.split(',')]; seeds = [int(s) for s in a.seeds.split(',')]
log = open(os.path.join(a.out, f'ubp_card{a.tag}.log'), 'a')
def P(*s):
    m = ' '.join(str(x) for x in s); print(m, flush=True); log.write(m + '\n'); log.flush()


def soft_bits(S, taus=None):
    """S (n, n) query x candidate cosines, true on the diagonal; LOO-temperature InfoNCE log-score, bits (per-row vector)."""
    taus = TAUS if taus is None else taus
    n = S.shape[0]; r = torch.arange(n, device=dev)
    Z = S[None] / taus[:, None, None]; lp = Z - torch.logsumexp(Z, -1, keepdim=True)
    CE = -lp[:, r, r] / LOG2
    loo = (CE.sum(1, keepdim=True) - CE) / (n - 1)
    return math.log2(n) - CE[loo.argmin(0), r]


def card(zq, zc, name):
    """zq queries (n, d), zc candidates (n, d); returns top-1, yield bits, shuffled-pair floor."""
    S = F.normalize(zq.float(), dim=1) @ F.normalize(zc.float(), dim=1).T
    top1 = float((S.argmax(1) == torch.arange(len(S), device=dev)).float().mean())
    y = soft_bits(S); g = torch.Generator(device=dev); g.manual_seed(0)
    fl = soft_bits(S[:, torch.randperm(len(S), device=dev, generator=g)][:, :])  # candidates permuted: shuffled pairs
    return {f'{name}_top1': top1, f'{name}_bits': float(y.mean()), f'{name}_floor_bits': float(fl.mean()), f'{name}_bits_t40': float(soft_bits(S, TAUS40).mean())}


# ---------------- image features (UBP release, RN50 fovea blur), keyed by image path
FT = safe_load(f'{D}/things-eeg/Image_feature/FoveaBlur/eeg_intra-subject_ubp_EEGProjectLayer_RN50_test.pt')['img_features']


class Enc(torch.nn.Module):  # UBP base/eeg_backbone.EEGProjectLayer (c_num 17, timesteps [0,250], z_dim 1024, drop_proj 0.3)
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Sequential(torch.nn.Linear(17 * 250, 1024),
                                         Res(torch.nn.Sequential(torch.nn.GELU(), torch.nn.Linear(1024, 1024), torch.nn.Dropout(0.3))),
                                         torch.nn.LayerNorm(1024))
        self.logit_scale = torch.nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
    def forward(self, x):
        return self.model(x.reshape(x.shape[0], -1))


class Res(torch.nn.Module):
    def __init__(self, f):
        super().__init__(); self.f = f
    def forward(self, x):
        return x + self.f(x)


def load_test(s):
    t = safe_load(f'{D}/things-eeg/Preprocessed_data_250Hz_whiten/sub-{s:02d}/test.pt')
    E = torch.from_numpy(np.ascontiguousarray(np.asarray(t['eeg'])[:, :, SEL]))         # as UBP: select channels, average in the file dtype
    ses = np.asarray(t['session']); imgs = list(np.asarray(t['img'])[:, 0])
    assert imgs == sorted(imgs)
    hA = torch.from_numpy(np.isin(ses[0], [0, 1])); hB = ~hA
    assert all((np.isin(ses[i], [0, 1]) == hA.numpy()).all() for i in range(len(ses)))
    Y = torch.stack([FT['medium'][p] for p in imgs]).float()
    av = lambda Z: Z.mean(1).float()[..., T0:T1].to(dev)
    return av(E), av(E[:, hA]), av(E[:, hB]), Y.to(dev)


SAVE_Z = {}  # row -> fp16 test embeddings, so any estimator can be recomputed without retraining


def full_card(fn, Xs, name, key=None):
    """fn: callable EEG (n,17,250) -> z (n,1024) in UBP's RN50 space (or probe output). Xs = (k80, A, B, Y)."""
    X80, XA, XB, Y = Xs
    with torch.no_grad():
        z80, zA, zB = fn(X80), fn(XA), fn(XB)
    r = card(z80, Y, 'k80'); rA = card(zA, Y, 'kA40'); rB = card(zB, Y, 'kB40')
    r['k40_bits'] = (rA['kA40_bits'] + rB['kB40_bits']) / 2; r['k40_top1'] = (rA['kA40_top1'] + rB['kB40_top1']) / 2
    r['k40_floor_bits'] = (rA['kA40_floor_bits'] + rB['kB40_floor_bits']) / 2
    cAB = card(zA, zB, 'ceilAB'); cBA = card(zB, zA, 'ceilBA')
    r['ceiling_k40_bits'] = (cAB['ceilAB_bits'] + cBA['ceilBA_bits']) / 2; r['ceiling_k40_top1'] = (cAB['ceilAB_top1'] + cBA['ceilBA_top1']) / 2
    r['fraction_k40'] = r['k40_bits'] / r['ceiling_k40_bits'] if r['ceiling_k40_bits'] > 0 else None
    r['k40_bits_t40'] = (rA['kA40_bits_t40'] + rB['kB40_bits_t40']) / 2                          # sensitivity: 40-point grid
    r['ceiling_k40_bits_t40'] = (cAB['ceilAB_bits_t40'] + cBA['ceilBA_bits_t40']) / 2
    r['fraction_k40_t40'] = r['k40_bits_t40'] / r['ceiling_k40_bits_t40'] if r['ceiling_k40_bits_t40'] > 0 else None
    gn = torch.Generator(device=dev); gn.manual_seed(7)
    with torch.no_grad():
        zn = fn(torch.randn(X80.shape, device=dev, generator=gn))   # LOCK §4 noise-input floor: N(0,1) test input through the same model
    rn = card(zn, Y, 'noise_input'); r['noise_input_floor_bits'] = rn['noise_input_bits']; r['noise_input_floor_top1'] = rn['noise_input_top1']
    if key is not None:
        SAVE_Z[key] = {'k80': z80.half().cpu(), 'A': zA.half().cpu(), 'B': zB.half().cpu(), 'noise_input': zn.half().cpu()}
    return r


def load_train(s):
    t = safe_load(f'{D}/things-eeg/Preprocessed_data_250Hz_whiten/sub-{s:02d}/train.pt')
    X = torch.from_numpy(np.ascontiguousarray(np.asarray(t['eeg'])[:, :, SEL])).mean(1)[..., T0:T1].half()  # train_avg: (16540, 17, 250)
    imgs = list(np.asarray(t['img'])[:, 0]); del t
    return X, imgs


out = {'note': __doc__, 'tau_grid_points': len(TAUS), 'subjects': {}, 'retrain': {}}
fn_out = os.path.join(a.out, f'ubp_card{a.tag}.json')
FTR = safe_load(f'{D}/things-eeg/Image_feature/FoveaBlur/eeg_intra-subject_ubp_EEGProjectLayer_RN50_train.pt')['img_features']
TRAIN, TEST, IMGS = {}, {}, None
for s in subs:
    TEST[s] = load_test(s)
    if a.retrain_only:
        TRAIN[s], imgs = load_train(s); IMGS = IMGS or imgs; assert imgs == IMGS; continue
    # ---- G2 + yield of the released checkpoint
    ck = safe_load(f'{D}/ckpt/sub-{s:02d}/last.ckpt')
    enc = Enc().to(dev); sd = {k[len('brain.'):]: v for k, v in ck['state_dict'].items() if k.startswith('brain.')}
    enc.load_state_dict(sd); enc.eval()
    pub = json.load(open(f'{D}/ckpt/sub-{s:02d}/test_results.json'))[0]
    r = full_card(enc, TEST[s], 'released', key=f'released/sub-{s:02d}')
    r['released_test_top1_json'] = pub['test_top1_acc']; r['G2_abs_diff_pt'] = abs(r['k80_top1'] - pub['test_top1_acc']) * 100
    r['G2_pass'] = r['G2_abs_diff_pt'] <= 0.5; r['ckpt_epoch'] = ck.get('epoch')
    out['subjects'][s] = {'released': r}
    P(f'sub-{s:02d} released: top1 {r["k80_top1"]:.3f} (json {pub["test_top1_acc"]:.3f}) G2 {r["G2_pass"]} | k80 {r["k80_bits"]:.3f} b, k40 {r["k40_bits"]:.3f} b, '
      f'ceiling k40 {r["ceiling_k40_bits"]:.3f} b, fraction {r["fraction_k40"]}, floor {r["k80_floor_bits"]:.3f} ({time.time()-t0:.0f}s)')
    # ---- random-init twin + ridge probe (GPU)
    Xtr, imgs = load_train(s); IMGS = IMGS or imgs; assert imgs == IMGS
    TRAIN[s] = Xtr
    torch.manual_seed(0); rnd = Enc().to(dev).eval()
    with torch.no_grad():
        H = torch.cat([rnd(Xtr[i:i + 4096].to(dev).float()) for i in range(0, len(Xtr), 4096)])
    Ytr = torch.stack([FTR['medium'][p] for p in imgs]).float().to(dev)
    perm = torch.randperm(len(H), generator=torch.Generator().manual_seed(0)).to(dev); nv = len(H) // 10
    va, tr = perm[:nv], perm[nv:]
    mu = H[tr].mean(0); Hc = H - mu; ym = Ytr[tr].mean(0)
    U, Sv, Vh = torch.linalg.svd(Hc[tr].double(), full_matrices=False)
    UtY = U.T @ (Ytr[tr] - ym).double(); best = (-1e9, None)
    for lam in torch.logspace(-2, 6, 25, dtype=torch.float64):
        W = Vh.T @ ((Sv / (Sv ** 2 + lam))[:, None] * UtY)
        pv = Hc[va].double() @ W + ym.double(); sc = float(F.cosine_similarity(pv, Ytr[va].double(), dim=1).mean())
        if sc > best[0]: best = (sc, float(lam))
    U, Sv, Vh = torch.linalg.svd(Hc.double(), full_matrices=False)
    W = (Vh.T @ ((Sv / (Sv ** 2 + best[1]))[:, None] * (U.T @ (Ytr - Ytr.mean(0)).double()))).float(); b0 = Ytr.mean(0)
    probe = lambda X: (rnd(X) - mu) @ W + b0
    rr = full_card(probe, TEST[s], 'random_init', key=f'random_init_ridge/sub-{s:02d}'); rr['ridge_lambda'] = best[1]; rr['ridge_dev_cos'] = best[0]
    # ceiling of the random-init space itself (encoder output, no probe)
    rr_raw = full_card(rnd, TEST[s], 'random_init_raw'); rr['ceiling_k40_bits_encoder_space'] = rr_raw['ceiling_k40_bits']
    out['subjects'][s]['random_init_ridge'] = rr
    P(f'sub-{s:02d} random-init+ridge: top1 {rr["k80_top1"]:.3f}, k80 {rr["k80_bits"]:.3f} b, k40 {rr["k40_bits"]:.3f} b, lam {best[1]:.3g} ({time.time()-t0:.0f}s)')
    del H, Hc, U, Vh; torch.cuda.empty_cache()
    out['code_sha256'] = CODE_SHA256; json.dump(out, open(fn_out, 'w'), indent=1)

if a.skip_retrain:
    torch.save(SAVE_Z, os.path.join(a.out, f'ubp_test_embeddings{a.tag}.pt')); P('done (no retrain)', time.time() - t0); sys.exit(0)

# ================= batched faithful retrain: members = subjects x arms x seeds
# arms: eeg / noise = CLEAN card arms (no test data touches any training choice; METHOD binding condition 1);
# eeg_quirk = UBP's val-is-test blur-state update kept, for the G2 reproduction of main.py ONLY (never a card row)
members = [(s, arm, sd) for s in subs for arm in ('eeg', 'noise', 'eeg_quirk') for sd in seeds]
M = len(members); N = len(IMGS); BS = 1024; nb = math.ceil(N / BS)
sidx = {s: i for i, s in enumerate(subs)}
DATA = torch.stack([TRAIN[s] for s in subs]).to(dev)                     # (S, N, 17, 250) fp16
TAGF = torch.stack([torch.stack([FTR[t][p] for p in IMGS]) for t in ('low', 'medium', 'high')]).float().to(dev)  # (3, N, 1024)
TAGF = F.normalize(TAGF, dim=-1)
msub = torch.tensor([sidx[s] for s, _, _ in members], device=dev); is_noise = torch.tensor([arm == 'noise' for _, arm, _ in members], device=dev); is_quirk = torch.tensor([arm == 'eeg_quirk' for _, arm, _ in members], device=dev)
P(f'retrain: {M} members, data {tuple(DATA.shape)} ({time.time()-t0:.0f}s)')
# init: each member = a fresh Enc() under torch.manual_seed(seed) (UBP: seed_everything(seed) before model init)
def init_member(seed):
    torch.manual_seed(seed); e = Enc()
    l1, l2, ln = e.model[0], e.model[1].f[1], e.model[2]
    return [l1.weight.T.contiguous(), l1.bias, l2.weight.T.contiguous(), l2.bias, ln.weight, ln.bias, e.logit_scale.reshape(1)]
ps = [torch.nn.Parameter(torch.stack(z).to(dev)) for z in zip(*[init_member(sd) for _, _, sd in members])]
W1, B1, W2, B2, G, Bt, LS = ps
opt = torch.optim.AdamW(ps, lr=1e-4, weight_decay=1e-4)


def fwd(x, train):  # x (M, b, 4250)
    h = torch.baddbmm(B1[:, None], x, W1)
    r = torch.baddbmm(B2[:, None], F.gelu(h), W2)
    if train:
        r = F.dropout(r, 0.3, True)
    return F.layer_norm(h + r, (1024,), None, None) * G[:, None] + Bt[:, None]


def clip_loss(z, y):  # z (M,b,d) unnormalised EEG, y (M,b,d) normalised image; returns per-member loss and diag of logits_per_image
    sc = F.softplus(LS.reshape(-1))[:, None, None]
    L = sc * z @ y.transpose(1, 2); lab = torch.arange(z.shape[1], device=dev).expand(z.shape[0], -1)
    l1 = F.cross_entropy(L.transpose(1, 2), lab, reduction='none').mean(1); l2 = F.cross_entropy(L, lab, reduction='none').mean(1)
    return (l1 + l2) / 2, torch.diagonal(L, dim1=1, dim2=2)


sim = torch.ones(M, N, device=dev); mlab = torch.ones(M, N, dtype=torch.long, device=dev); Z2 = 1.959963984540054


def ua_update(idx, diag):  # idx (M,b) indices into the train arrays; diag (M,b)
    bsim = 0.3 * diag + 0.7 * sim.gather(1, idx)
    mu_, sd_ = bsim.mean(1, keepdim=True), bsim.std(1, keepdim=True)
    lab = torch.ones_like(idx); lab[diag > mu_ + Z2 * sd_] = 0; lab[diag < mu_ - Z2 * sd_] = 2
    sim.scatter_(1, idx, bsim); mlab.scatter_(1, idx, lab)


TESTX = torch.stack([TEST[s][0] for s in subs]); TESTY = torch.stack([TEST[s][3] for s in subs])  # (S,200,17,250), (S,200,1024)


def test_input(gen):
    x = TESTX[msub].reshape(M, 200, -1)
    return torch.where(is_noise[:, None, None], torch.randn(x.shape, device=dev, generator=gen), x)


def val_quirk(gen):  # UBP: validation_step (eval mode) on the test loader updates sim/match_label at train indices 0..199 (eeg_quirk members only)
    with torch.no_grad():
        z = fwd(test_input(gen), False); _, dg = clip_loss(z, TESTY[msub])
        s0, l0 = sim.clone(), mlab.clone()
        ua_update(torch.arange(200, device=dev).expand(M, -1), dg)
        sim[~is_quirk] = s0[~is_quirk]; mlab[~is_quirk] = l0[~is_quirk]


gen = torch.Generator(device=dev); gen.manual_seed(1234)
active = torch.ones(M, dtype=torch.bool, device=dev); best_loss = torch.full((M,), float('inf'), device=dev)
wait = torch.zeros(M, dtype=torch.long, device=dev); stop_ep = torch.full((M,), a.epochs, device=dev)
val_quirk(gen)  # sanity validation batch
perm_gen = torch.Generator(device=dev); perm_gen.manual_seed(0)
for ep in range(a.epochs):
    lab_snap = mlab.clone()                                              # dataloader workers see the labels at epoch start
    perms = torch.argsort(torch.rand(M, N, device=dev, generator=perm_gen), dim=1)
    ep_loss = torch.zeros(M, device=dev); nseen = 0
    for b in range(nb):
        idx = perms[:, b * BS:(b + 1) * BS]; bsz = idx.shape[1]
        x = DATA[msub[:, None], idx].reshape(M, bsz, -1).float()
        x = torch.where(is_noise[:, None, None], torch.randn(x.shape, device=dev, generator=gen), x)
        y = TAGF[lab_snap.gather(1, idx), idx]                           # per-sample blur level (0 low, 1 medium, 2 high)
        z = fwd(x, True); loss_m, dg = clip_loss(z, y)
        snap = [p.detach().clone() for p in ps] if not bool(active.all()) else None
        opt.zero_grad(); loss_m.sum().backward(); opt.step()
        if snap is not None:
            with torch.no_grad():
                for p, s0 in zip(ps, snap):
                    p[~active] = s0[~active]
        with torch.no_grad():
            ua_update(idx, dg.detach())
        ep_loss += loss_m.detach() * bsz; nseen += bsz
    ep_loss /= nseen
    val_quirk(gen)
    with torch.no_grad():
        imp = ep_loss < best_loss - 0.001
        best_loss = torch.where(imp & active, ep_loss, best_loss); wait = torch.where(imp, torch.zeros_like(wait), wait + 1)
        newly = active & (wait >= 5); stop_ep[newly] = ep + 1; active &= ~newly
    if ep % 5 == 0 or ep == a.epochs - 1:
        P(f'epoch {ep+1}: loss eeg {ep_loss[~is_noise].mean():.4f} noise {ep_loss[is_noise].mean():.4f} active {int(active.sum())}/{M} ({time.time()-t0:.0f}s)')
    if not bool(active.any()):
        break

# ---- evaluate every member with the card (last weights)
with torch.no_grad():
    for m, (s, arm, sd) in enumerate(members):
        def fn(X, m=m, arm=arm):
            x = X.reshape(len(X), -1)
            if arm == 'noise':
                x = torch.randn(x.shape, device=dev, generator=gen)
            h = x @ W1[m] + B1[m]; r = F.gelu(h) @ W2[m] + B2[m]
            return F.layer_norm(h + r, (1024,), G[m], Bt[m])
        r = full_card(fn, TEST[s], 'retrain', key=f'retrain/sub-{s:02d}/{arm}/seed{sd}'); r['stop_epoch'] = int(stop_ep[m]); r['final_train_loss'] = float(best_loss[m])
        out['retrain'][f'sub-{s:02d}/{arm}/seed{sd}'] = r
        P(f'sub-{s:02d} {arm} seed{sd}: top1 {r["k80_top1"]:.3f} k80 {r["k80_bits"]:.3f} b k40 {r["k40_bits"]:.3f} b ceiling {r["ceiling_k40_bits"]:.3f} b stop {r["stop_epoch"]}')
out['runtime_s'] = time.time() - t0
out['code_sha256'] = CODE_SHA256; json.dump(out, open(fn_out, 'w'), indent=1)
torch.save(SAVE_Z, os.path.join(a.out, f'ubp_test_embeddings{a.tag}.pt'))
try:  # Track A: save the member weights in fp16
    torch.save({'members': members, 'state': {n_: p_.detach().half().cpu() for n_, p_ in zip(['W1', 'B1', 'W2', 'B2', 'G', 'Bt', 'LS'], ps)}}, os.path.join(a.out, f'ubp_weights{a.tag}.pt'))
except Exception as e:
    P('weight save failed', repr(e))
P('wrote', fn_out, f'{time.time()-t0:.0f}s')
