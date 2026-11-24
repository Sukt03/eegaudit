"""brainmagick-style segment retrieval — OUR RE-IMPLEMENTATION of Défossez et al. 2023 (NMI), Brennan2019 EEG. IMPLEMENTER, 21 Nov 2026.
Step 2: train + card. Paper recipe kept: 3 s windows from 0.5 s before each word onset; brain input shifted +150 ms; per-window baseline
(first 0.5 s); splits 70/20/10 by SENTENCE (seeded), windows whose speech spans a sentence of another split are dropped; test candidates =
the unique test windows (one per test word); brain module = spatial attention (270 out, Fourier K = 32, spatial dropout radius 0.2) ->
1x1 conv -> per-subject 270x270 layer -> 5 blocks [2 residual dilated k3 convs (320 ch, BN, GELU; dilations 2^(2k mod 5), 2^(2k+1 mod 5))
+ k3 conv to 640 + GLU] -> 1x1 640 + GELU -> 1x1 1024; CLIP loss with score <Z, Y_j / ||Y_j||> (in-batch negatives); Adam 3e-4, batch 256,
epoch = 1,200 updates, early stop on validation loss (patience 10), best-validation checkpoint.
Change (compute): max_epochs cap (default 40). Arms: eeg | noise (N(0,1) input of identical shape, train and test; Jo et al. 2024) |
random_init (brain module frozen at its seeded init except the two final 1x1 convs = the head; LOCK §4).
Card: G2 = segment-level top-10 / top-1 over all test candidates (paper Table 2: 25.7 +- 2.9 / 5.2 +- 0.8 %); yield = soft12 InfoNCE
log-score (LOCK §3; temperature chosen leave-one-SEGMENT-out over the cosine score) in bits per 3-s sample, + `_t40` sensitivity;
floors: shuffled pairs and the noise-input floor (N(0,1) test input through the trained model); per-participant yields, participant
bootstrap CI95 (GPU). No usable ceiling (single presentation of each segment). Saves fp16 weights and the test score matrices."""
import hashlib as _hl
CODE_SHA256 = _hl.sha256(open(__file__, 'rb').read()).hexdigest()  # every result JSON carries the exact code hash
import argparse, json, math, os, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

ap = argparse.ArgumentParser()
ap.add_argument('--arm', choices=['eeg', 'noise', 'random_init'], default='eeg'); ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--max_epochs', type=int, default=40); ap.add_argument('--updates', type=int, default=1200); ap.add_argument('--bs', type=int, default=256)
ap.add_argument('--subs', type=int, default=0, help='0 = all; n = first n participants (smoke)'); ap.add_argument('--tag', default='')
ap.add_argument('--out', default='/path/to/eegaudit/results/baselines/brainmagick')
a = ap.parse_args()
torch.manual_seed(a.seed); np.random.seed(a.seed); dev = torch.device('cuda'); t0 = time.time()
TAG = f'bm_{a.arm}_s{a.seed}{a.tag}'; logf = open(os.path.join(a.out, TAG + '.log'), 'w')
def P(*x):
    m = ' '.join(str(v) for v in x); print(m, flush=True); logf.write(m + '\n'); logf.flush()

PR = torch.load(os.path.join(a.out, 'prep.pt'), map_location='cpu', weights_only=True)
SR = PR['sr']; L = 3 * SR; SHIFT = int(round(0.15 * SR)); PRE = int(round(0.5 * SR))
subs = sorted(PR['subjects'])[: a.subs or None]; S = len(subs)
EEG = [PR['subjects'][s]['eeg'].float().to(dev) for s in subs]; C = EEG[0].shape[0]
W2V = {k: v.float().to(dev) for k, v in PR['w2v'].items()}
pos = PR['pos2d'].to(dev)

# ---------------- splits by sentence (seeded), windows, exclusion of cross-split windows
words0 = PR['subjects'][subs[0]]['words']
sents = sorted({w['sentence_id'] for s in subs for w in PR['subjects'][s]['words']})
rng = np.random.default_rng(1000 + a.seed); perm = rng.permutation(sents); n = len(perm)
split_of = {sid: ('train' if i < 0.7 * n else 'valid' if i < 0.9 * n else 'test') for i, sid in enumerate(perm)}
story = sorted({(w['word_index'], w['wav'], w['audio_onset'], w['sentence_id'], w['word']) for s in subs for w in PR['subjects'][s]['words']})
by_wav = {}
for wi, wav, ao, sid, wd in story:
    by_wav.setdefault(wav, []).append((ao, sid, wi))
def window_split(wav, ao):
    """split of the speech window [ao - 0.5, ao + 2.5] in this wav, or None if it spans >1 split or leaves the wav."""
    st, en = ao - 0.5, ao + 2.5
    if st < 0 or int(round(en * SR)) > W2V[wav].shape[1]:
        return None
    sp = {split_of[sid] for o, sid, _ in by_wav[wav] if st <= o < en}
    return sp.pop() if len(sp) == 1 else None
seg_of_word = {}; segs = []                                   # unique speech segment per word_index
for wi, wav, ao, sid, wd in story:
    sp = window_split(wav, ao)
    if sp is not None:
        seg_of_word[wi] = len(segs); segs.append({'wav': wav, 'a0': int(round((ao - 0.5) * SR)), 'split': sp, 'word': wd, 'word_index': wi})
samples = {'train': [], 'valid': [], 'test': []}                # (subject idx, eeg start sample, segment idx)
for si, s in enumerate(subs):
    T = EEG[si].shape[1]
    for w in PR['subjects'][s]['words']:
        g = seg_of_word.get(w['word_index'])
        if g is None:
            continue
        e0 = int(round((w['eeg_onset'] - 0.5) * SR)) + SHIFT
        if e0 < 0 or e0 + L > T:
            continue
        samples[segs[g]['split']].append((si, e0, g))
SAMP = {k: torch.tensor(v, device=dev) for k, v in samples.items()}
test_segs = sorted({g for _, _, g in samples['test']}); tpos = {g: j for j, g in enumerate(test_segs)}
P(f'{TAG}: {S} participants, segments train/valid/test {sum(x["split"]=="train" for x in segs)}/{sum(x["split"]=="valid" for x in segs)}/{len(test_segs)}; '
  f'samples {len(samples["train"])}/{len(samples["valid"])}/{len(samples["test"])} ({time.time()-t0:.0f}s)')
SEGY = torch.stack([W2V[x['wav']][:, x['a0']:x['a0'] + L] for x in segs])            # (n_seg, 1024, L)
EEGP = [F.pad(e, (0, L)) for e in EEG]


def batch_x(idx):
    si, e0 = idx[:, 0], idx[:, 1]
    x = torch.stack([EEGP[int(i)][:, int(t):int(t) + L] for i, t in zip(si.tolist(), e0.tolist())])
    x = x - x[:, :, :PRE].mean(-1, keepdim=True)                 # baseline: first 0.5 s of the window
    if a.arm == 'noise':
        x = torch.randn_like(x)
    return x, si


# ---------------- brain module
class SpatialAttention(nn.Module):
    def __init__(self, pos, d=270, K=32, drop=0.2):
        super().__init__()
        self.z = nn.Parameter(torch.randn(d, K, K, 2) / (K * K)); self.drop = drop
        k = torch.arange(1, K + 1, device=pos.device).float()
        arg = 2 * math.pi * (pos[:, 0, None, None] * k[None, :, None] + pos[:, 1, None, None] * k[None, None, :])   # (C, K, K)
        self.register_buffer('cos', torch.cos(arg)); self.register_buffer('sin', torch.sin(arg)); self.register_buffer('pos', pos)
    def forward(self, x):
        A = torch.einsum('dklr,cklr->dc', self.z, torch.stack([self.cos, self.sin], -1))       # (d, C)
        if self.training and self.drop > 0:
            c0 = torch.rand(2, device=x.device) * 0.8 + 0.1
            A = A.masked_fill(((self.pos - c0).norm(dim=1) < self.drop)[None], float('-inf'))
        return torch.einsum('dc,bct->bdt', torch.softmax(A, -1), x)


class Brain(nn.Module):
    def __init__(self, pos, S, d1=270, d2=320, F_=1024):
        super().__init__()
        self.sa = SpatialAttention(pos, d1); self.conv1 = nn.Conv1d(d1, d1, 1)
        self.subj = nn.Parameter(torch.eye(d1)[None].repeat(S, 1, 1))
        self.blocks = nn.ModuleList()
        for k in range(5):
            cin = d1 if k == 0 else d2
            self.blocks.append(nn.ModuleList([nn.Conv1d(cin, d2, 3, padding=2 ** ((2 * k) % 5), dilation=2 ** ((2 * k) % 5)), nn.BatchNorm1d(d2),
                                              nn.Conv1d(d2, d2, 3, padding=2 ** ((2 * k + 1) % 5), dilation=2 ** ((2 * k + 1) % 5)), nn.BatchNorm1d(d2),
                                              nn.Conv1d(d2, 2 * d2, 3, padding=1)]))
        self.head = nn.Sequential(nn.Conv1d(d2, 2 * d2, 1), nn.GELU(), nn.Conv1d(2 * d2, F_, 1))
    def forward(self, x, si):
        h = self.conv1(self.sa(x)); h = torch.einsum('bdt,bde->bet', h, self.subj[si])
        for k, (c1, b1, c2, b2, c3) in enumerate(self.blocks):
            h = F.gelu(b1(c1(h))) if k == 0 else h + F.gelu(b1(c1(h)))
            h = h + F.gelu(b2(c2(h)))
            h = F.glu(c3(h), dim=1)
        return self.head(h)


model = Brain(pos, S).to(dev)
if a.arm == 'random_init':
    for n_, p_ in model.named_parameters():
        if not n_.startswith('head.'):
            p_.requires_grad_(False)
params = [p for p in model.parameters() if p.requires_grad]
opt = torch.optim.Adam(params, lr=3e-4)
P(f'params {sum(p.numel() for p in model.parameters())/1e6:.2f}M, trainable {sum(p.numel() for p in params)/1e6:.2f}M')


def clip_scores(Z, Y):
    return torch.einsum('bct,oct->bo', Z, Y / (Y.flatten(1).norm(dim=1)[:, None, None] + 1e-8))


@torch.no_grad()
def eval_loss(split, nmax=4096):
    model.eval(); idx = SAMP[split]; idx = idx[torch.randperm(len(idx), device=dev, generator=torch.Generator(device=dev).manual_seed(0))[:nmax]]
    tot = 0; n = 0
    for i in range(0, len(idx), a.bs):
        b = idx[i:i + a.bs]; x, si = batch_x(b); Z = model(x, si); Y = SEGY[b[:, 2]]
        sc = clip_scores(Z, Y); tot += F.cross_entropy(sc, torch.arange(len(b), device=dev), reduction='sum').item(); n += len(b)
    model.train(); return tot / n


best = (float('inf'), None, -1); wait = 0
for ep in range(a.max_epochs):
    model.train(); run = 0
    for u in range(a.updates):
        b = SAMP['train'][torch.randint(0, len(SAMP['train']), (a.bs,), device=dev)]
        x, si = batch_x(b); Z = model(x, si); sc = clip_scores(Z, SEGY[b[:, 2]])
        loss = F.cross_entropy(sc, torch.arange(len(b), device=dev)); opt.zero_grad(); loss.backward(); opt.step(); run += loss.item()
    vl = eval_loss('valid')
    if vl < best[0]:
        best = (vl, {k: v.detach().clone() for k, v in model.state_dict().items()}, ep + 1); wait = 0
    else:
        wait += 1
    P(f'epoch {ep+1}: train {run/a.updates:.4f} valid {vl:.4f} best {best[2]} ({time.time()-t0:.0f}s)')
    if wait >= 10:
        break
model.load_state_dict(best[1]); model.eval()

# ---------------- card
TAUS = torch.logspace(-2.5, 0, 12, device=dev); TAUS40 = torch.logspace(-2.5, 0, 40, device=dev); LOG2 = math.log(2)
def infonce(Sm, true, cl, taus):
    Z_ = Sm[None] / taus[:, None, None]; lp = Z_ - torch.logsumexp(Z_, -1, keepdim=True)
    CE = -lp[:, torch.arange(len(true), device=dev), true] / LOG2
    ncl = int(cl.max()) + 1; csum = torch.zeros(len(taus), ncl, device=dev).index_add_(1, cl, CE); ccnt = torch.bincount(cl, minlength=ncl).float()
    loo = (CE.sum(1, keepdim=True) - csum) / (len(true) - ccnt).clamp(min=1)
    return math.log2(Sm.shape[1]) - CE[loo.argmin(0)[cl], torch.arange(len(true), device=dev)]
YT = SEGY[test_segs]; YTn = F.normalize(YT.flatten(1), dim=1)
idx = SAMP['test']; Ssc, Snoise = [], []
with torch.no_grad():
    for i in range(0, len(idx), a.bs):
        b = idx[i:i + a.bs]; x, si = batch_x(b)
        Ssc.append(F.normalize(model(x, si).flatten(1), dim=1) @ YTn.T)
        g = torch.Generator(device=dev); g.manual_seed(7 + i)
        Snoise.append(F.normalize(model(torch.randn(x.shape, device=dev, generator=g), si).flatten(1), dim=1) @ YTn.T)   # LOCK §4 noise-input floor
Ssc = torch.cat(Ssc); Snoise = torch.cat(Snoise)
true = torch.tensor([tpos[int(g)] for g in idx[:, 2].tolist()], device=dev); cl = true.clone(); subj = idx[:, 0]
rank = (Ssc > Ssc[torch.arange(len(true), device=dev), true][:, None]).sum(1)
y = infonce(Ssc, true, cl, TAUS); y40 = infonce(Ssc, true, cl, TAUS40)
gp = torch.Generator(device=dev); gp.manual_seed(0); yf = infonce(Ssc[torch.randperm(len(true), device=dev, generator=gp)], true, cl, TAUS)
yn = infonce(Snoise, true, cl, TAUS)
per = {}
for si_ in range(S):
    m = subj == si_
    if m.any():
        per[subs[si_]] = {'n': int(m.sum()), 'top1': float((rank[m] == 0).float().mean()), 'top10': float((rank[m] < 10).float().mean()),
                          'yield_bits': float(y[m].mean()), 'floor_shuffled_bits': float(yf[m].mean()), 'noise_input_floor_bits': float(yn[m].mean())}
def pboot(v):
    v = torch.tensor(v, device=dev, dtype=torch.float64); i = torch.randint(0, len(v), (10000, len(v)), device=dev, generator=torch.Generator(device=dev).manual_seed(0))
    m = v[i].mean(1); return [float(torch.quantile(m, 0.025)), float(torch.quantile(m, 0.975))]
res = {'note': __doc__, 'cfg': vars(a), 'tag': TAG, 'best_epoch': best[2], 'best_valid_loss': best[0], 'n_candidates': len(test_segs),
       'n_test_samples': len(true), 'chance_top10': min(1.0, 10 / len(test_segs)),
       'top1': float((rank == 0).float().mean()), 'top10': float((rank < 10).float().mean()),
       'published_top10_pct': 25.7, 'published_top1_pct': 5.2,
       'yield_bits': float(y.mean()), 'yield_bits_t40': float(y40.mean()), 'floor_shuffled_bits': float(yf.mean()), 'noise_input_floor_bits': float(yn.mean()),
       'per_participant': per}
for k in ('top1', 'top10', 'yield_bits', 'noise_input_floor_bits'):
    res[f'{k}_participant_mean'] = float(np.mean([v[k] for v in per.values()])); res[f'{k}_participant_ci95'] = pboot([v[k] for v in per.values()])
res['runtime_s'] = time.time() - t0
res['code_sha256'] = CODE_SHA256; json.dump(res, open(os.path.join(a.out, TAG + '.json'), 'w'), indent=1)
torch.save({'scores': Ssc.half().cpu(), 'scores_noise_input': Snoise.half().cpu(), 'true': true.cpu(), 'subject': subj.cpu()}, os.path.join(a.out, TAG + '_test_scores.pt'))
try:
    torch.save({k: v.half().cpu() for k, v in model.state_dict().items()}, os.path.join(a.out, TAG + '_weights_fp16.pt'))
except Exception as e:
    P('weight save failed', repr(e))
P(f'{TAG}: top10 {res["top10"]:.3f} top1 {res["top1"]:.3f} (published 0.257 / 0.052; chance top10 {res["chance_top10"]:.3f}) | yield {res["yield_bits"]:.4f} b '
  f'[{res["yield_bits_participant_ci95"][0]:.4f}, {res["yield_bits_participant_ci95"][1]:.4f}] t40 {res["yield_bits_t40"]:.4f} | floors shuffled {res["floor_shuffled_bits"]:.4f} '
  f'noise-input {res["noise_input_floor_bits"]:.4f} | {len(test_segs)} candidates ({time.time()-t0:.0f}s)')
