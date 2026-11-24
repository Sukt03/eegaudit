"""DTHA-style dual-teacher EEG-to-image model — OUR RE-IMPLEMENTATION of Wu et al., arXiv 2611.12798 (no code or weights released).
IMPLEMENTER, 21 Nov 2026. One batched GPU model: members = participants x arms {eeg, noise, random_init} for one seed.
Encoder (AMSPBE, Fig. 3): tri-region time attention (posterior 17 / central 20 / frontal 26 channels, curves of length 250, init 1) x
  per-channel attention (init 1) -> depthwise temporal conv (63 groups, k 7) -> ELU -> adaptive average pool to 128 -> positivity-
  constrained spatial filter bank (|w|, 1 -> 16, kernel (63,1)) -> BatchNorm -> MLP 2048 -> 1024 (Linear + residual GELU-Linear-Dropout 0.5
  + LayerNorm, as UBP/NICE projectors; the paper says only "MLP") -> z_b (unnormalised).
Teachers: CLIP RN50 at 4 levels (UBP release: sharp + fovea low/medium/high; the paper uses 12 Gaussian blur levels) and DINOv2 ViT-L/14
  CLS after blocks 0..14; per-dimension softmax level weights (Eq. 9-10) + MLP (Linear-GELU-Linear, 1024) -> z_clip, z_dino.
Loss (Eq. 1-7): S = z_b . normalise(z_v)^T with tau = 1; SCE on S_clip and S_dino plus 0.25 x SCE on S_fused = softmax(eta) weighted sum.
Optimiser: AdamW lr 1e-3, wd 1e-4, 10-epoch linear warm-up then cosine, 250 epochs, bs 1024; 5 % of each participant's training images
  held out (seeded) as validation; checkpoint = best validation fused top-1 (their protocol), evaluated every epoch.
Arms: eeg; noise = fresh N(0,1) input of identical shape at train, validation and test (Jo et al. 2024); random_init = encoder frozen at
  its seeded initialisation (BatchNorm statistics still tracked), teacher heads and fusion trained identically.
Card per member (CHANGE 13): fused top-1 at k80 (G2 vs the paper's per-subject table), branch top-1s; InfoNCE log-score (LOO temperature)
  over the fused cosine score at k80 and k40 (sessions 1-2, 3-4); own split-half ceiling = the same estimator between z_b(sessions 1-2) and
  z_b(sessions 3-4); fraction at k40; shuffled-pair floor.
Run: python -I -u dtha_train.py --subs 1 --seed 2025 [--epochs 250] --tag smoke"""
import hashlib as _hl
CODE_SHA256 = _hl.sha256(open(__file__, 'rb').read()).hexdigest()  # every result JSON carries the exact code hash
import argparse, json, math, os, sys, time
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, '/path/to/eegaudit/experiments/baselines')
from safe_load import safe_load

ap = argparse.ArgumentParser()
ap.add_argument('--subs', default='1,2,3,4,5,6,7,8,9,10'); ap.add_argument('--seed', type=int, default=2025)
ap.add_argument('--epochs', type=int, default=250); ap.add_argument('--arms', default='eeg,noise,random_init')
ap.add_argument('--tag', default=''); ap.add_argument('--bs', type=int, default=1024)
ap.add_argument('--out', default='/path/to/eegaudit/results/baselines/dtha')
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True); dev = torch.device('cuda'); t0 = time.time()
U = '/path/to/eegaudit/data_raw/third_party/ubp/things-eeg/Preprocessed_data_250Hz_whiten'
CH = ['Fp1', 'Fp2', 'AF7', 'AF3', 'AFz', 'AF4', 'AF8', 'F7', 'F5', 'F3', 'F1', 'F2', 'F4', 'F6', 'F8', 'FT9', 'FT7', 'FC5', 'FC3', 'FC1',
      'FCz', 'FC2', 'FC4', 'FC6', 'FT8', 'FT10', 'T7', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'T8', 'TP9', 'TP7', 'CP5', 'CP3', 'CP1',
      'CPz', 'CP2', 'CP4', 'CP6', 'TP8', 'TP10', 'P7', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO3', 'POz', 'PO4', 'PO8',
      'O1', 'Oz', 'O2']
REG = torch.tensor([0 if c[0] in 'PO' else (1 if (c[0] in 'CT' and not c.startswith('FT')) else 2) for c in CH], device=dev)  # 0 post, 1 central, 2 frontal
assert [int((REG == r).sum()) for r in range(3)] == [17, 20, 26]
TAUS = torch.logspace(-2.5, 0, int(os.environ.get('CARD_NTAU', '12')), device=dev)  # FIXER L1 (LOCK §3): 12-point LOO grid PRIMARY; CARD_NTAU=40 = sensitivity
TAUS40 = torch.logspace(-2.5, 0, 40, device=dev)  # labelled sensitivity field (*_t40)
LOG2 = math.log(2)
subs = [int(s) for s in a.subs.split(',')]; arms = a.arms.split(',')
members = [(s, arm) for s in subs for arm in arms]; M = len(members)
logf = open(os.path.join(a.out, f'dtha{a.tag}_seed{a.seed}.log'), 'a')
def P(*x):
    m = ' '.join(str(v) for v in x); print(m, flush=True); logf.write(m + '\n'); logf.flush()

# ---------------- teacher features and data
TF = safe_load('/path/to/eegaudit/results/baselines/dtha/teacher_features.pt')  # QUEUE-GUARD: plain weights_only load crashed on the numpy scalars in the image-name lists; safe_load allow-lists numpy globals only
key_tr = {k: i for i, k in enumerate(TF['train_imgs'])}; key_te = {k: i for i, k in enumerate(TF['test_imgs'])}
DATA, TEST = [], []
for s in subs:
    t = safe_load(f'{U}/sub-{s:02d}/train.pt'); imgs = list(np.asarray(t['img'])[:, 0])
    DATA.append(torch.from_numpy(np.asarray(t['eeg'])).mean(1).half()); del t
    order_tr = torch.tensor([key_tr[k] for k in imgs])
    t = safe_load(f'{U}/sub-{s:02d}/test.pt'); E = torch.from_numpy(np.asarray(t['eeg'])); ses = np.asarray(t['session'])
    hA = torch.from_numpy(np.isin(ses[0], [0, 1])); timg = list(np.asarray(t['img'])[:, 0])
    TEST.append((E.mean(1).float(), E[:, hA].mean(1).float(), E[:, ~hA].mean(1).float())); del t, E
    P(f'loaded sub-{s:02d} ({time.time()-t0:.0f}s)')
order_te = torch.tensor([key_te[k] for k in timg])
CLIP_TR = TF['clip_train'][order_tr].float().to(dev); DINO_TR = TF['dino_train'][order_tr].float().to(dev)    # (N, 4|15, 1024), same image order for all participants
CLIP_TE = TF['clip_test'][order_te].float().to(dev); DINO_TE = TF['dino_test'][order_te].float().to(dev)
DATA = torch.stack(DATA).to(dev); N = DATA.shape[1]
g = torch.Generator().manual_seed(a.seed); perm = torch.randperm(N, generator=g)
nval = int(round(0.05 * N)); VAL, TRN = perm[:nval].to(dev), perm[nval:].to(dev)
msub = torch.tensor([subs.index(s) for s, _ in members], device=dev)
is_noise = torch.tensor([arm == 'noise' for _, arm in members], device=dev); is_rand = torch.tensor([arm == 'random_init' for _, arm in members], device=dev)
P(f'{M} members; train {len(TRN)} val {nval} images; data {tuple(DATA.shape)} ({time.time()-t0:.0f}s)')

# ---------------- member-stacked parameters (initialisation seeded per member, identical across arms of a participant)
def lin(i, o, gen):
    b = 1 / math.sqrt(i); return (torch.rand(i, o, generator=gen) * 2 - 1) * b, (torch.rand(o, generator=gen) * 2 - 1) * b
def init_member(sidx):
    gen = torch.Generator().manual_seed(a.seed * 100 + sidx)
    p = {'tatt': torch.ones(3, 250), 'catt': torch.ones(63),
         'dw': (torch.rand(63, 7, generator=gen) * 2 - 1) / math.sqrt(7), 'dwb': (torch.rand(63, generator=gen) * 2 - 1) / math.sqrt(7),
         'sf': (torch.rand(16, 63, generator=gen) * 2 - 1) / math.sqrt(63), 'sfb': (torch.rand(16, generator=gen) * 2 - 1) / math.sqrt(63),
         'bng': torch.ones(16), 'bnb': torch.zeros(16)}
    p['p1'], p['p1b'] = lin(2048, 1024, gen); p['p2'], p['p2b'] = lin(1024, 1024, gen); p['lng'], p['lnb'] = torch.ones(1024), torch.zeros(1024)
    for h in ('c', 'd'):
        p[f'{h}1'], p[f'{h}1b'] = lin(1024, 1024, gen); p[f'{h}2'], p[f'{h}2b'] = lin(1024, 1024, gen)
    p['eta_c'] = torch.zeros(4, 1024); p['eta_d'] = torch.zeros(15, 1024); p['fuse'] = torch.zeros(2)
    return p
inits = [init_member(subs.index(s)) for s, _ in members]
PR = {k: torch.nn.Parameter(torch.stack([d[k] for d in inits]).to(dev)) for k in inits[0]}
ENC_KEYS = ['tatt', 'catt', 'dw', 'dwb', 'sf', 'sfb', 'bng', 'bnb', 'p1', 'p1b', 'p2', 'p2b', 'lng', 'lnb']
ENC0 = {k: PR[k].detach().clone() for k in ENC_KEYS}
run_mean = torch.zeros(M, 16, device=dev); run_var = torch.ones(M, 16, device=dev)
opt = torch.optim.AdamW(list(PR.values()), lr=1e-3, weight_decay=1e-4)
steps_ep = math.ceil(len(TRN) / a.bs); total = a.epochs * steps_ep; warm = 10 * steps_ep
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda st: (st + 1) / warm if st < warm else 0.5 * (1 + math.cos(math.pi * (st - warm) / max(1, total - warm))))


def encode(x, train):
    """x (M, B, 63, 250) -> z_b (M, B, 1024)."""
    Mm, B = x.shape[:2]
    x = x * PR['tatt'][:, None][:, :, REG] * PR['catt'][:, None, :, None]                  # TSAM
    h = F.conv1d(x.permute(1, 0, 2, 3).reshape(B, Mm * 63, 250), PR['dw'].reshape(Mm * 63, 1, 7), PR['dwb'].reshape(-1), padding=3, groups=Mm * 63)
    h = F.adaptive_avg_pool1d(F.elu(h), 128).reshape(B, Mm, 63, 128).permute(1, 0, 2, 3)  # (M, B, 63, 128)
    h = torch.einsum('mbct,mkc->mbkt', h, PR['sf'].abs()) + PR['sfb'][:, None, :, None]   # positivity-constrained filter bank
    if train:
        mu = h.mean((1, 3)); var = h.var((1, 3), unbiased=False)
        with torch.no_grad():
            run_mean.mul_(0.9).add_(0.1 * mu); run_var.mul_(0.9).add_(0.1 * h.var((1, 3), unbiased=True))
    else:
        mu, var = run_mean, run_var
    h = (h - mu[:, None, :, None]) / torch.sqrt(var[:, None, :, None] + 1e-5) * PR['bng'][:, None, :, None] + PR['bnb'][:, None, :, None]
    h = h.reshape(Mm, B, 2048)
    h1 = torch.baddbmm(PR['p1b'][:, None], h, PR['p1'])
    r = torch.baddbmm(PR['p2b'][:, None], F.gelu(h1), PR['p2'])
    if train:
        r = F.dropout(r, 0.5, True)
    return F.layer_norm(h1 + r, (1024,), None, None) * PR['lng'][:, None] + PR['lnb'][:, None]


def head(feat, w, k):
    """feat (M, n, L, 1024) or (n, L, 1024) shared; w = softmax over L per dim; MLP."""
    wt = torch.softmax(PR[f'eta_{k[0]}'], dim=1)                                          # (M, L, 1024)
    h = (feat * wt[:, None]).sum(2) if feat.dim() == 4 else torch.einsum('nld,mld->mnd', feat, wt)
    h = torch.baddbmm(PR[f'{k}1b'][:, None], h, PR[f'{k}1'])
    return torch.baddbmm(PR[f'{k}2b'][:, None], F.gelu(h), PR[f'{k}2'])


def sce(S):
    lab = torch.arange(S.shape[1], device=dev).expand(S.shape[0], -1)
    return (F.cross_entropy(S.transpose(1, 2), lab, reduction='none').mean(1) + F.cross_entropy(S, lab, reduction='none').mean(1)) / 2


def sims(zb, zc, zd, cos=False):
    if cos:
        zb = F.normalize(zb, dim=-1)
    Sc = zb @ F.normalize(zc, dim=-1).transpose(1, 2); Sd = zb @ F.normalize(zd, dim=-1).transpose(1, 2)
    w = torch.softmax(PR['fuse'], 1)
    return Sc, Sd, w[:, 0, None, None] * Sc + w[:, 1, None, None] * Sd


def noise_like(x, gen):
    return torch.where(is_noise[:, None, None, None], torch.randn(x.shape, device=dev, generator=gen), x)


gen = torch.Generator(device=dev); gen.manual_seed(a.seed); pgen = torch.Generator(device=dev); pgen.manual_seed(a.seed + 1)
best = torch.full((M,), -1.0, device=dev); best_ep = torch.zeros(M, dtype=torch.long, device=dev)
BEST = {k: v.detach().clone() for k, v in PR.items()}; BEST_RM, BEST_RV = run_mean.clone(), run_var.clone()


@torch.no_grad()
def val_top1():
    x = noise_like(DATA[msub[:, None], VAL[None]].float(), gen)
    zb = torch.cat([encode(x[:, i:i + 512], False) for i in range(0, nval, 512)], 1)
    zc = head(CLIP_TR[VAL], None, 'c'); zd = head(DINO_TR[VAL], None, 'd')
    _, _, Sf = sims(zb, zc, zd)
    return (Sf.argmax(2) == torch.arange(nval, device=dev)).float().mean(1)


for ep in range(a.epochs):
    order = torch.argsort(torch.rand(M, len(TRN), device=dev, generator=pgen), 1); run = torch.zeros(M, device=dev)
    for b in range(steps_ep):
        idx = TRN[order[:, b * a.bs:(b + 1) * a.bs]]                                   # (M, B) image indices
        x = noise_like(DATA[msub[:, None], idx].float(), gen)
        zb = encode(x, True)
        zc = head(CLIP_TR[idx], None, 'c'); zd = head(DINO_TR[idx], None, 'd')
        Sc, Sd, Sf = sims(zb, zc, zd)
        loss = sce(Sc) + sce(Sd) + 0.25 * sce(Sf)
        opt.zero_grad(); loss.sum().backward(); opt.step(); sched.step(); run += loss.detach()
        with torch.no_grad():
            for k in ENC_KEYS:                                                           # random-init twin: encoder frozen at init
                PR[k][is_rand] = ENC0[k][is_rand]
    v = val_top1(); imp = v > best
    with torch.no_grad():
        best = torch.where(imp, v, best); best_ep[imp] = ep + 1
        for k in PR:
            BEST[k][imp] = PR[k][imp].detach()
        BEST_RM[imp] = run_mean[imp]; BEST_RV[imp] = run_var[imp]
    if ep % 10 == 0 or ep == a.epochs - 1:
        msg = ' '.join(f'{arm}:{float(v[[i for i, (_, x) in enumerate(members) if x == arm]].mean()):.3f}' for arm in arms)
        P(f'epoch {ep+1}/{a.epochs} loss {float(run.mean()/steps_ep):.3f} val fused top1 {msg} ({time.time()-t0:.0f}s)')

# ---------------- card on the selected checkpoints
with torch.no_grad():
    for k in PR:
        PR[k].copy_(BEST[k])
    run_mean.copy_(BEST_RM); run_var.copy_(BEST_RV)
try:
    torch.save({'members': members, 'best_ep': best_ep.cpu(), 'PR': {k: v.detach().cpu() for k, v in PR.items()}, 'run_mean': run_mean.detach().cpu(), 'run_var': run_var.detach().cpu()}, f"{a.out}/{a.tag or 'run'}_best_members.pt")  # coordinator: keep weights
except Exception as e:
    print(f'checkpoint save FAILED ({e}); run continues', flush=True)


def soft_bits(S, taus=None):
    taus = TAUS if taus is None else taus
    n = S.shape[0]; r = torch.arange(n, device=dev)
    Z = S[None] / taus[:, None, None]; lp = Z - torch.logsumexp(Z, -1, keepdim=True); CE = -lp[:, r, r] / LOG2
    loo = (CE.sum(1, keepdim=True) - CE) / (n - 1)
    return math.log2(n) - CE[loo.argmin(0), r]


res = {'note': __doc__, 'cfg': vars(a), 'members': {}}
with torch.no_grad():
    zc = head(CLIP_TE, None, 'c'); zd = head(DINO_TE, None, 'd')
    Z = {}
    for j, name in enumerate(('k80', 'A', 'B')):
        X = torch.stack([TEST[subs.index(s)][j] for s, _ in members]).to(dev)
        Z[name] = encode(noise_like(X, gen), False)
    Z['noise_input'] = encode(torch.randn((M, 200, 63, 250), device=dev, generator=gen), False)   # LOCK §4 noise-input floor (every arm)
    fl_perm = torch.randperm(200, device=dev, generator=gen)
    for name in ('k80', 'A', 'B', 'noise_input'):
        Sc, Sd, Sf = sims(Z[name], zc, zd, cos=True); Z[name + '_S'] = (Sc, Sd, Sf)
    for m, (s, arm) in enumerate(members):
        r = {'best_val_fused_top1': float(best[m]), 'best_epoch': int(best_ep[m]), 'fusion_w_clip': float(torch.softmax(PR['fuse'][m], 0)[0])}
        lab = torch.arange(200, device=dev)
        Sc, Sd, Sf = (S[m] for S in Z['k80_S'])
        r.update(top1_fused_k80=float((Sf.argmax(1) == lab).float().mean()), top1_clip_k80=float((Sc.argmax(1) == lab).float().mean()),
                 top1_dino_k80=float((Sd.argmax(1) == lab).float().mean()), top5_fused_k80=float((Sf.topk(5, 1).indices == lab[:, None]).any(1).float().mean()))
        r['k80_bits'] = float(soft_bits(Sf).mean()); r['k80_floor_bits'] = float(soft_bits(Sf[:, fl_perm]).mean())
        r['k80_bits_clip_branch'] = float(soft_bits(Sc).mean()); r['k80_bits_dino_branch'] = float(soft_bits(Sd).mean())
        r['k40_bits'] = float((soft_bits(Z['A_S'][2][m]).mean() + soft_bits(Z['B_S'][2][m]).mean()) / 2)
        r['k40_top1'] = float(((Z['A_S'][2][m].argmax(1) == lab).float().mean() + (Z['B_S'][2][m].argmax(1) == lab).float().mean()) / 2)
        C = F.normalize(Z['A'][m], dim=1) @ F.normalize(Z['B'][m], dim=1).T
        r['ceiling_k40_bits'] = float((soft_bits(C).mean() + soft_bits(C.T).mean()) / 2)
        r['ceiling_k40_bits_t40'] = float((soft_bits(C, TAUS40).mean() + soft_bits(C.T, TAUS40).mean()) / 2)      # sensitivity
        r['k80_bits_t40'] = float(soft_bits(Sf, TAUS40).mean())
        r['k40_bits_t40'] = float((soft_bits(Z['A_S'][2][m], TAUS40).mean() + soft_bits(Z['B_S'][2][m], TAUS40).mean()) / 2)
        r['noise_input_floor_bits'] = float(soft_bits(Z['noise_input_S'][2][m]).mean())
        r['ceiling_k40_top1'] = float(((C.argmax(1) == lab).float().mean() + (C.T.argmax(1) == lab).float().mean()) / 2)
        r['fraction_k40'] = r['k40_bits'] / r['ceiling_k40_bits'] if r['ceiling_k40_bits'] > 0 else None
        res['members'][f'sub-{s:02d}/{arm}'] = r
        P(f'sub-{s:02d} {arm}: top1 fused {r["top1_fused_k80"]:.3f} (clip {r["top1_clip_k80"]:.3f}, dino {r["top1_dino_k80"]:.3f}) | k80 {r["k80_bits"]:.3f} b '
          f'floor {r["k80_floor_bits"]:.3f} | k40 {r["k40_bits"]:.3f} b ceiling {r["ceiling_k40_bits"]:.3f} b frac {r["fraction_k40"]} | best ep {r["best_epoch"]}')
res['runtime_s'] = time.time() - t0
torch.save({k: (v.half().cpu() if torch.is_tensor(v) else None) for k, v in Z.items() if torch.is_tensor(v)}, os.path.join(a.out, f'dtha{a.tag}_seed{a.seed}_test_embeddings.pt'))
try:  # Track A: save the selected member weights in fp16
    torch.save({'members': members, 'state': {k: v.detach().half().cpu() for k, v in PR.items()}, 'bn_mean': run_mean.cpu(), 'bn_var': run_var.cpu()},
               os.path.join(a.out, f'dtha{a.tag}_seed{a.seed}_weights.pt'))
except Exception as e:
    P('weight save failed', repr(e))
fn = os.path.join(a.out, f'dtha{a.tag}_seed{a.seed}.json'); res['code_sha256'] = CODE_SHA256; json.dump(res, open(fn, 'w'), indent=1); P('wrote', fn, f'{time.time()-t0:.0f}s')
