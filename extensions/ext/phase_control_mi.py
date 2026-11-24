"""MI phase/trial-shuffle input controls for the raw (1-40 Hz block-mean -> per-participant z -> PCA-256 -> logistic) read-out (, 22 Nov 2026). Reuses mi_card_v3 functions unchanged (imported, not edited): same participants/QC, 10 participant groups, inner validation
selection, raw features. The control replaces the input epoch (64 ch x 960 samples) before the pipeline:
  phase   per-trial, per-channel INDEPENDENT phase randomisation of the full epoch (rfft, amplitude kept, uniform phase, DC/Nyquist real), seeds 0,1,2
  shuffle trial order permuted within participant (labels fixed), seeds 0,1,2
This refits the closed-form PCA+logistic read-out on the control input (same as the noise-floor arm of mi_card); EEGNet and foundation-model arms are NOT run
(EEGNet needs training; FM arms would need a forward pass plus refit and were not run). G1: unmodified input reproduces saved results/ext/mi_card_v3 raw eeg bits to 1e-4
per participant (only meaningful with all 105 participants, because the groups depend on the participant list).
Usage: --out DIR [--subjects 1,2] [--conds real,phase0,...] [--gate_only]"""
import hashlib as _hl
CODE_SHA256 = _hl.sha256(open(__file__, 'rb').read()).hexdigest()
import argparse, json, math, sys, time
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, '/path/to/eegaudit/experiments/ext')
import mi_card_v3 as M
ROOT = M.ROOT; DEV = 'cuda'; t0 = time.time()
log = lambda *m: print(f'[{time.time()-t0:7.1f}s]', *m, flush=True)
ap = argparse.ArgumentParser(); ap.add_argument('--out', required=True); ap.add_argument('--subjects', default='')
ap.add_argument('--conds', default='real,phase0,phase1,phase2,shuffle0,shuffle1,shuffle2'); ap.add_argument('--gate_only', action='store_true')
ap.add_argument('--lchunk', type=int, default=4)
a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
conds = ['real'] if a.gate_only else a.conds.split(',')
cache = ROOT / 'results/ext/mi_card_cache'
subs = [int(x) for x in a.subjects.split(',')] if a.subjects else list(range(1, 110))
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
sid = torch.tensor(np.concatenate([[k] * len(D[s]['y']) for k, s in enumerate(P)]), device=DEV); N = len(y); sidc = sid.cpu()

def phase_rand(X, g, chunk=1000):
    o = torch.empty_like(X); n = X.shape[-1]
    for i in range(0, len(X), chunk):
        Fq = torch.fft.rfft(X[i:i+chunk], dim=-1); ph = torch.rand(Fq.shape, device=DEV, generator=g) * 2 * math.pi
        ph[..., 0] = 0
        if n % 2 == 0: ph[..., -1] = 0
        o[i:i+chunk] = torch.fft.irfft(Fq.abs() * torch.exp(1j * ph), n=n, dim=-1)
    return o

def shuffle_rows(X, g):
    o = X.clone()
    for k in range(nP):
        r = torch.where(sid == k)[0]; o[r] = X[r[torch.randperm(len(r), device=DEV, generator=g)]]
    return o

def raw_feat(Z):
    r = M.bandpass(Z, 1, 40)[:, :, M.W0:M.W1].reshape(len(Z), 64, 70, 8).mean(-1).reshape(len(Z), -1).double().cpu()
    for k in range(nP):
        rr = sidc == k; r[rr] = (r[rr] - r[rr].mean(0)) / (r[rr].std(0) + 1e-8)
    return r
Fd = {}; stats = {}
for c in conds:
    g = torch.Generator(device=DEV)
    if c == 'real': Z = X
    elif c.startswith('phase'):
        g.manual_seed(1000 + int(c[5:])); Z = phase_rand(X, g)
        stats[c] = {'power_ratio': float(Z.pow(2).mean() / X.pow(2).mean())}
    else:
        g.manual_seed(2000 + int(c[7:])); Z = shuffle_rows(X, g)
    Fd[c] = {'raw': raw_feat(Z)}
    if Z is not X: del Z
    torch.cuda.empty_cache()
del X; torch.cuda.empty_cache()
G = min(10, nP); grp = torch.tensor([k % G for k in range(nP)], device=DEV)[sid]
fits, units, pca_rows = [], [], {}
for gg in range(G):
    v = (gg + 1) % G; outer = torch.where(grp != gg)[0]; pca_rows[gg] = outer
    tr_in = torch.where((grp != gg) & (grp != v))[0]; ev_in = torch.where(grp == v)[0]; ev = torch.where(grp == gg)[0]
    fits.append((tr_in, ev_in, gg)); fits.append((outer, ev, gg)); units.append(dict(inner=[len(fits) - 2], final=len(fits) - 1))
ylab = {c: [y[f[0]] for f in fits] for c in conds}
eng = lambda Xf, fi, yl: M.primal_fits(Xf.to(DEV), fi, yl, M.LAMS, pca_rows, lchunk=a.lchunk)
R, diag = M.run_design('pc_mi', units, fits, Fd, y, ylab, sid, nP, eng, [], [], None, None, None)
res = {}
for k, s in enumerate(P):
    r = {'participant': s, 'code_sha256': CODE_SHA256, 'NOT_CITABLE': 'until independent review', 'bits': {c: R[k]['raw'][c]['bits'] for c in conds}}
    sv = ROOT / f'results/ext/mi_card_v3/primary_lpgo/sub-{s:03d}.json'
    if sv.exists():
        r['saved_raw_eeg_bits'] = json.load(open(sv))['raw']['eeg']['bits']; r['G1_abs_diff'] = abs(r['bits']['real'] - r['saved_raw_eeg_bits']) if 'real' in conds else None
    res[s] = r; (out / f'sub{s:03d}.json').write_text(json.dumps(r, indent=1))
def boot(v, B=10000):
    v = torch.tensor(v, device=DEV, dtype=torch.float64); g = torch.Generator(device=DEV); g.manual_seed(0)
    m = v[torch.randint(len(v), (B, len(v)), device=DEV, generator=g)].mean(1)
    return {'mean': float(v.mean()), 'ci95': [float(m.quantile(0.025)), float(m.quantile(0.975))]}
S = {'NOT_CITABLE': 'until independent review', 'code_sha256': CODE_SHA256, 'n_participants': nP, 'bootstrap': 'participant, 10000 resamples, GPU torch, seed 0',
     'conds': conds, 'readout': 'raw (1-40 Hz, 0.5-4.0 s, 20 Hz block means) -> per-participant z -> PCA-256 -> logistic; LPGO 10 groups', 'stats': stats,
     'g1_comparable': nP == 105}
d = [r['G1_abs_diff'] for r in res.values() if r.get('G1_abs_diff') is not None]
S['G1_max_abs_diff'] = max(d) if d else None; S['G1_pass'] = bool(d and max(d) < 1e-4 and nP == 105)
for c in conds: S[f'bits_{c}'] = boot([res[s]['bits'][c] for s in P])
for fam in ('phase', 'shuffle'):
    cs = [c for c in conds if c.startswith(fam)]
    if cs:
        m = [float(np.mean([res[s]['bits'][c] for c in cs])) for s in P]; S[f'bits_{fam}_seedmean'] = boot(m)
        S[f'{fam}_max_abs_per_participant_seedmean'] = float(np.max(np.abs(m))); S[f'{fam}_floor_ok_le_0.02'] = bool(np.max(np.abs(m)) <= 0.02)
S['per_participant_file_sha256'] = {f'sub{s:03d}': _hl.sha256((out / f'sub{s:03d}.json').read_bytes()).hexdigest() for s in P}
(out / 'summary.json').write_text(json.dumps(S, indent=1)); log({k: v for k, v in S.items() if k.startswith(('bits', 'G1', 'phase', 'shuffle'))})
