"""GPU-FILL v2 MI UNTRAINED-ENCODER baseline (, 23 Nov 2026; new script, header/prep copied from gf_mi_eegnet_floor.py).
Encoder = mi_card_v3.MEEGNet at its random initialisation (never trained, BN default stats, eval mode); frozen features = the 272-d flattened pre-classifier map, per-participant z-scored; read-out = the SAME LPGO 10-group logistic PCA-256 read-out, lambda/T chosen on the inner group (mi_card_v3.primal_fits/run_design) as the G1 raw read-out. S random inits (--seeds), plus a row-shuffle floor of init 0 (rows permuted within participant, labels fixed). Gate G1 identical to the floors script.
ORIGINAL DOCSTRING FOLLOWS (floors script):
The saved mi_card_v3 EEGNet read-outs cannot take perturbed input, so the EEGNet arm is REFIT per fold with the mi_card_v3 recipe, imported unchanged
(mi_card_v3.eegnet_members / select / bits / prior_of; 4000 steps, checkpoint every 100, batch 64, Adam 1e-3, (step, T) chosen on the inner validation group g+1,
refit on the other 9 groups, LPGO 10 groups, same participants/QC, same EEGNet input = 4-40 Hz, 0.5-4.0 s, per-participant scalar scale).
Members: 10 groups x {inner, final} = 20 per condition; TWO conditions share one batched model (40 members, mchunk 40) so the GPU is filled.
Conditions (input epoch 64 ch x 960 samples replaced BEFORE band-pass, training and test rows alike):
  real      unperturbed (refit; sanity against saved eegnet bits, not an identity gate: member composition and GPU kernels differ)
  phaseK    per trial, per channel INDEPENDENT phase randomisation of the full epoch (amplitude kept), K in 0,1
  shuffleK  trial order permuted within participant (labels fixed), K in 0,1
Gates: G1 (logistic raw read-out on the unperturbed input reproduces saved mi_card_v3 raw bits to 1e-4: pipeline/participant/group identity; needs all 105);
G2 floors: per-participant seed-mean |bits| <= 0.02 (cohort mean reported with a participant bootstrap).
Usage: --out DIR [--subjects 1,2] [--conds real,phase0,...] [--steps 4000] [--mchunk 40]"""
import hashlib as _hl
CODE_SHA256 = _hl.sha256(open(__file__, 'rb').read()).hexdigest()
import argparse, json, math, sys, time
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, '/path/to/eegaudit/experiments/ext')
import mi_card_v3 as M
ROOT = M.ROOT; DEV = 'cuda'; t0 = time.time(); log = M.log
ap = argparse.ArgumentParser(); ap.add_argument('--out', required=True); ap.add_argument('--subjects', default='')
ap.add_argument('--seeds', type=int, default=5)
a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
cache = ROOT / 'results/ext/mi_card_cache'; subs = [int(x) for x in a.subjects.split(',')] if a.subjects else list(range(1, 110))
D = {}
for s in subs:
    d, why = M.load_sub(s, cache)
    if d is None: continue
    c = np.bincount(d['y'], minlength=4)
    if c.max() - c.min() > 2: continue
    D[s] = d
P = sorted(D); nP = len(P); log(nP, 'participants')
X = torch.tensor(np.concatenate([D[s]['X'] for s in P]), dtype=torch.float32, device=DEV)
y = torch.tensor(np.concatenate([D[s]['y'] for s in P]), device=DEV)
sid = torch.tensor(np.concatenate([[k] * len(D[s]['y']) for k, s in enumerate(P)]), device=DEV); N = len(y)


def phase_rand(X, g, chunk=1000):
    o = torch.empty_like(X); n = X.shape[-1]
    for i in range(0, len(X), chunk):
        Fq = torch.fft.rfft(X[i:i + chunk], dim=-1); ph = torch.rand(Fq.shape, device=DEV, generator=g) * 2 * math.pi
        ph[..., 0] = 0
        if n % 2 == 0: ph[..., -1] = 0
        o[i:i + chunk] = torch.fft.irfft(Fq.abs() * torch.exp(1j * ph), n=n, dim=-1)
    return o


def shuffle_rows(X, g):
    o = X.clone()
    for k in range(nP):
        r = torch.where(sid == k)[0]; o[r] = X[r[torch.randperm(len(r), device=DEV, generator=g)]]
    return o


def cond_input(c):
    g = torch.Generator(device=DEV)
    if c == 'real': Z = X
    elif c.startswith('phase'): g.manual_seed(1000 + int(c[5:])); Z = phase_rand(X, g)
    else: g.manual_seed(2000 + int(c[7:])); Z = shuffle_rows(X, g)
    Xi = M.bandpass(Z, 4, 40)[:, :, M.W0:M.W1].float()                     # EEGNet input, as mi_card_v3.features
    if Z is not X: del Z
    for k in range(nP):                                                    # per-participant scalar scale (label-free), as mi_card_v3.eeg_in
        r = sid == k; Xi[r] /= Xi[r].std()
    return Xi


G = min(10, nP); grp = torch.tensor([k % G for k in range(nP)], device=DEV)[sid]
mem0, units0 = [], []
for gg in range(G):
    v = (gg + 1) % G; tr_in = torch.where((grp != gg) & (grp != v))[0]; ev_in = torch.where(grp == v)[0]
    outer = torch.where(grp != gg)[0]; ev = torch.where(grp == gg)[0]
    mem0.append((tr_in, ev_in)); mem0.append((outer, ev)); units0.append(dict(inner=len(mem0) - 2, final=len(mem0) - 1))

# ---- G1: logistic raw read-out on unperturbed input reproduces the saved mi_card_v3 raw bits (pipeline identity)
def raw_feat(Z):
    r = M.bandpass(Z, 1, 40)[:, :, M.W0:M.W1].reshape(len(Z), 64, 70, 8).mean(-1).reshape(len(Z), -1).double().cpu()
    for k in range(nP):
        rr = (sid.cpu() == k); r[rr] = (r[rr] - r[rr].mean(0)) / (r[rr].std(0) + 1e-8)
    return r
fits, units, pca_rows = [], [], {}
for gg in range(G):
    v = (gg + 1) % G; outer = torch.where(grp != gg)[0]; pca_rows[gg] = outer
    fits.append((torch.where((grp != gg) & (grp != v))[0], torch.where(grp == v)[0], gg)); fits.append((outer, torch.where(grp == gg)[0], gg))
    units.append(dict(inner=[len(fits) - 2], final=len(fits) - 1))
gate = {'code_sha256': CODE_SHA256, 'n_participants': nP}
if nP == 105:
    ylab = {'eeg': [y[f[0]] for f in fits]}
    Rg, _ = M.run_design('g1', units, fits, {'eeg': {'raw': raw_feat(X)}}, y, ylab, sid, nP, lambda Xf, fi, yl: M.primal_fits(Xf.to(DEV), fi, yl, M.LAMS, pca_rows), [], [], None, None, None)
    d = [abs(Rg[k]['raw']['eeg']['bits'] - json.load(open(ROOT / f'results/ext/mi_card_v3/primary_lpgo/sub-{s:03d}.json'))['raw']['eeg']['bits']) for k, s in enumerate(P)]
    gate['G1_max_abs_diff'] = max(d); gate['G1_pass'] = bool(max(d) < 1e-4)
else:
    gate['G1_pass'] = None; gate['G1_note'] = 'subset run: groups depend on the participant list; identity only meaningful with all 105'
log('gate', gate); (out / 'gate.json').write_text(json.dumps(gate, indent=1))
if gate['G1_pass'] is False: raise SystemExit('G1 FAIL')

import torch.nn.functional as Fnn
ylab = {'eeg': [y[f[0]] for f in fits]}
S = a.seeds; Xr = cond_input('real'); torch.manual_seed(4242)
enc = M.MEEGNet(S, T=Xr.shape[-1]).to(DEV).eval()      # S independent random inits, never trained
feats = []
with torch.no_grad():
    for e0 in range(0, N, 256):
        x = Xr[e0:e0 + 256][:, None].expand(-1, S, -1, -1).contiguous()
        z = Fnn.elu(enc.bn2(enc.c2(enc.bn1(enc.c1(x))))); z = Fnn.avg_pool2d(z, (1, 4)); z = Fnn.elu(enc.bn3(enc.c4(enc.c3(z)))); z = Fnn.avg_pool2d(z, (1, 8))
        feats.append(z.reshape(len(x), S, -1).double().cpu())
Fe = torch.cat(feats); del Xr, enc; torch.cuda.empty_cache(); log('frozen feature map', tuple(Fe.shape))
def zs(Fk):
    r = Fk.clone()
    for k in range(nP):
        rr = (sid.cpu() == k); r[rr] = (r[rr] - r[rr].mean(0)) / (r[rr].std(0) + 1e-8)
    return torch.nan_to_num(r)
Fd = {f'rand{k}': zs(Fe[:, k]) for k in range(S)}
g = torch.Generator().manual_seed(77); pm = torch.arange(N)
for k in range(nP):
    r = torch.where(sid.cpu() == k)[0]; pm[r] = r[torch.randperm(len(r), generator=g)]
Fd['shuffle_rand0'] = Fd['rand0'][pm]
Rr, diag = M.run_design('untrained', units, fits, {'eeg': Fd}, y, ylab, sid, nP, lambda Xf, fi, yl: M.primal_fits(Xf.to(DEV), fi, yl, M.LAMS, pca_rows), [], [], None, None, None)
saved = {s: json.load(open(ROOT / f'results/ext/mi_card_v3/primary_lpgo/sub-{s:03d}.json')) for s in P if (ROOT / f'results/ext/mi_card_v3/primary_lpgo/sub-{s:03d}.json').exists()}
def boot(v, B=10000):
    v = torch.tensor(np.asarray(v, np.float64), device=DEV); gg = torch.Generator(device=DEV); gg.manual_seed(0)
    m = v[torch.randint(len(v), (B, len(v)), device=DEV, generator=gg)].mean(1)
    return {'mean': float(v.mean()), 'ci95': [float(m.quantile(0.025)), float(m.quantile(0.975))]}
bits_ = {c: [Rr[k][c]['eeg']['bits'] for k in range(nP)] for c in Fd}
seedmean = np.mean([bits_[f'rand{k}'] for k in range(S)], 0)
Sm = {'NOT_CITABLE': 'until independent review', 'code_sha256': CODE_SHA256, 'n_participants': nP, 'seeds': S, 'gate': gate, 'bootstrap': 'participant, 10000 resamples, GPU torch, seed 0',
      'unit': 'bits per trial (log2 p - log2 train prior)', 'bits': {c: boot(v) for c, v in bits_.items()}, 'untrained_seedmean': boot(seedmean),
      'per_seed_cohort_means': [float(np.mean(bits_[f'rand{k}'])) for k in range(S)], 'shuffle_floor_max_abs_participant': float(np.max(np.abs(bits_['shuffle_rand0']))),
      'shuffle_floor_ok_le_0.02': bool(np.max(np.abs(bits_['shuffle_rand0'])) <= 0.02), 'diag_edges': diag['edges']}
if saved:
    sv = np.array([saved[s]['eegnet']['eeg']['bits'] for s in P]); Sm['saved_trained_eegnet'] = boot(sv); Sm['trained_minus_untrained'] = boot(sv - seedmean)
    Sm['untrained_over_trained_ratio'] = float(seedmean.mean() / sv.mean())
(out / 'summary.json').write_text(json.dumps(Sm, indent=1)); (out / 'per_participant.json').write_text(json.dumps({c: dict(zip(map(str, P), v)) for c, v in bits_.items()}))
log(json.dumps({k: v for k, v in Sm.items() if k not in ('gate', 'diag_edges')}, indent=1)[:3000])
