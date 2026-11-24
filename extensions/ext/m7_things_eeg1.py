"""M7/A1 second image dataset (METHOD T3 ruling (3)): THINGS-EEG1 (OpenNeuro ds003825, CC0), participants sub-02..sub-11.
Metadata gate (EXECUTOR-4, 21 Nov): sub-02 onward show 22,248 images once and 200 TEST images x 12 repeats (events.tsv isteststim /
teststimnumber); sub-01 has no repeats (excluded). Caveats for every row: 10 Hz RSVP (next image 100 ms later), one session, 12 repeats.
Preprocessing (GPU, torch): BrainVision float32, 63 channels, 1000 Hz (vhdr); events.tsv onset = sample index (checked against the vmrk
E 1 markers). Zero-phase FFT band-pass 0.1-40 Hz on the continuous recording, decimation by 4 -> 250 Hz, epochs 0-1.0 s after onset
(250 samples, the THINGS-EEG2 layout), baseline = mean of -100..0 ms. No MVNN (THINGS-EEG2's released data are whitened; stated).
Targets   : CLIP ViT-H/14 features of the THINGS-EEG2 image set (bbeeg-atm/data/hf_probe, ATM order = sorted concept folders x sorted files).
            Training trials = single-shown EEG1 images that are THINGS-EEG2 TRAINING images; test = the 178 THINGS-EEG2 TEST images that EEG1
            also shows once (single trial, k = 1), 178-way InfoNCE log-score (torch port of run.soft_bits). No dedicated read-out exists on EEG1,
            so no fraction; yields are raw bits per single trial and are never compared with THINGS-EEG2 in raw bits.
Arms      : raw voltage, five-band log power, REVE pretrained, REVE random-init twin (m7_latent.py functions, identical probe; layer chosen on
            the dev tenth of the training trials, candidates {6, 11, 16, 22}; layer 22 also scored).
Ceiling   : split-half identification on the 200 repeated test images (mean of repeats 1-6 vs 7-12, standardized with training-trial
            statistics, cosine, 200-way InfoNCE) for raw, REVE (chosen layer) and twin; floor = shuffled halves. (No CLIP for these images.)
Output    : results/ext/m7/things1_sub-XX.json ; --summarize things1
"""
import sys, json, time, re, argparse
from pathlib import Path
import numpy as np, torch
ROOT = Path('/path/to/eegaudit'); sys.path.insert(0, str(ROOT / 'experiments/ext')); sys.path.insert(0, str(ROOT / 'experiments/eegbits'))
ap = argparse.ArgumentParser(); ap.add_argument('--subs', default='2'); ap.add_argument('--summarize', action='store_true'); ap.add_argument('--smoke', action='store_true'); a = ap.parse_args()
sys.argv = sys.argv[:1]
from m7_latent import DEV, OUT, LAYERS, metrics_t, ridge_image, split_half, bandpower, resample200, load_reve, reve_layers, boot_ci, signflip_p, holm
D1 = ROOT / 'data_raw/third_party/things_eeg1'; IMG = ROOT / 'data_raw/things_eeg2_images/images_set'


def clip_maps():
    from run import CLIP, load_torch
    tr = [f.name for d in sorted((IMG / 'training_images').iterdir()) for f in sorted(d.glob('*.jpg'))]
    te = [f.name for d in sorted((IMG / 'test_images').iterdir()) for f in sorted(d.glob('*.jpg'))]
    Ytr, Yte = load_torch(CLIP / 'ViT-H-14_features_train.pt'), load_torch(CLIP / 'ViT-H-14_features_test.pt'); assert len(tr) == len(Ytr) and len(te) == len(Yte)
    return {n: Ytr[i] for i, n in enumerate(tr)}, {n: Yte[i] for i, n in enumerate(te)}


def load_sub(s):
    vh = (D1 / f'sub-{s:02d}/eeg/sub-{s:02d}_task-rsvp_eeg.vhdr').read_text(errors='ignore')
    nch = int(re.search(r'NumberOfChannels=(\d+)', vh).group(1)); assert int(re.search(r'SamplingInterval=(\d+)', vh).group(1)) == 1000
    names = [re.search(rf'^Ch{k}=([^,]+),', vh, re.M).group(1) for k in range(1, nch + 1)]
    raw = np.fromfile(D1 / f'sub-{s:02d}/eeg/sub-{s:02d}_task-rsvp_eeg.eeg', dtype=np.float32).reshape(-1, nch).T
    ev = (D1 / f'sub-{s:02d}/eeg/sub-{s:02d}_task-rsvp_events.tsv').read_text().splitlines(); h = ev[0].split('\t'); rows = [r.split('\t') for r in ev[1:]]
    c = {k: h.index(k) for k in ('onset', 'isteststim', 'teststimnumber', 'stimname')}
    mk = [int(l.split(',')[2]) for l in (D1 / f'sub-{s:02d}/eeg/sub-{s:02d}_task-rsvp_eeg.vmrk').read_text().splitlines() if l.startswith('Mk') and ',E  1,' in l]
    on = np.array([int(float(r[c['onset']])) for r in rows]); assert np.isin(on, mk).mean() > 0.99, 'onset is not the E 1 marker sample'
    return names, raw, on, np.array([int(r[c['isteststim']]) for r in rows]), np.array([int(r[c['teststimnumber']]) for r in rows]), [r[c['stimname']] for r in rows]


def preprocess(raw, on):
    X = torch.from_numpy(raw).to(DEV); T = X.shape[1]; F = torch.fft.rfft(X, dim=1); f = torch.fft.rfftfreq(T, 1 / 1000).to(DEV)
    F[:, (f < 0.1) | (f > 40)] = 0; X = torch.fft.irfft(F, n=T, dim=1); del F
    X = X[:, ::4]; o = torch.tensor(on // 4, device=DEV); assert (o - 25 >= 0).all() and (o + 250 <= X.shape[1]).all()
    E = X[:, o[:, None] + torch.arange(250, device=DEV)].permute(1, 0, 2); B = X[:, o[:, None] + torch.arange(-25, 0, device=DEV)].mean(2).T
    return (E - B[:, :, None]).contiguous()                                 # (n_events, ch, 250)


def run(s, mtr, mte):
    from m7_latent import wait_gpu; wait_gpu(10.0)
    t0 = time.time(); g = torch.Generator().manual_seed(s); names, raw, on, ist, tsn, stim = load_sub(s); E = preprocess(raw, on); del raw
    itr = [i for i, n in enumerate(stim) if ist[i] == 0 and n in mtr]; ite = [i for i, n in enumerate(stim) if ist[i] == 0 and n in mte]
    Ytr = torch.from_numpy(np.stack([mtr[stim[i]] for i in itr])).to(DEV).float(); Yte = torch.from_numpy(np.stack([mte[stim[i]] for i in ite])).to(DEV).float()
    rep = {k: [i for i in range(len(stim)) if ist[i] == 1 and tsn[i] == k] for k in sorted(set(tsn[ist == 1]))}; assert len(rep) == 200 and all(len(v) == 12 for v in rep.values())
    H1 = E[torch.tensor([v[:6] for v in rep.values()], device=DEV)].mean(1); H2 = E[torch.tensor([v[6:] for v in rep.values()], device=DEV)].mean(1)
    Xtr, Xte = E[torch.tensor(itr, device=DEV)], E[torch.tensor(ite, device=DEV)]; del E
    if a.smoke: Xtr, Ytr = Xtr[:4000], Ytr[:4000]
    res = dict(dataset='THINGS-EEG1 ds003825', subject=f'sub-{s:02d}', n_train=len(Xtr), n_test=len(Xte), N_candidates=len(Xte), k=1, smoke=a.smoke, channels=len(names), arms={},
               caveats='10 Hz RSVP; single session; 12 test repeats; no MVNN; test = THINGS-EEG2 test images shown once in EEG1',
               input_check='0-1.0 s at 250 Hz -> Fourier resample to 200 Hz = 1 REVE patch (200 samples, overlap 20)')
    def score(name, Ftr, Fte, extra=None):
        al, dev, (p,) = ridge_image(Ftr, Ytr, [Fte]); r = dict(alpha=al, dev_soft_bits=dev, k1=metrics_t(p, Yte), shuffled_pair=metrics_t(p, Yte[torch.randperm(len(Yte), generator=g).to(DEV)]), **(extra or {}))
        res['arms'][name] = r; print(f"sub-{s:02d} {name}: k1 {r['k1']['soft_bits']:.4f} 2-way {r['k1']['two_way']:.3f} pair {r['shuffled_pair']['soft_bits']:.4f} {extra.get('split_half', '') if extra else ''} ({time.time()-t0:.0f}s)", flush=True)
    R = Xtr.reshape(len(Xtr), -1); fm, fs = R.mean(0), R.std(0, unbiased=False) + 1e-6
    score('raw', R, Xte.reshape(len(Xte), -1), split_half(H1.reshape(200, -1), H2.reshape(200, -1), fm, fs, g)); del R
    score('bandpower', bandpower(Xtr, 250), bandpower(Xte, 250))
    Z = resample200(Xtr); zmu, zsd = Z.mean((0, 2), keepdim=True), Z.std((0, 2), unbiased=False, keepdim=True) + 1e-6; std = lambda q: (resample200(q) - zmu) / zsd; del Z
    for name, pre in (('reve', True), ('reve_twin', False)):
        m, pos = load_reve(pre, names); Ftr = reve_layers(m, pos, std(Xtr)); Fte = reve_layers(m, pos, std(Xte)); F1 = reve_layers(m, pos, std(H1)); F2 = reve_layers(m, pos, std(H2))
        per = {l: dict(zip(('alpha', 'dev_soft_bits'), ridge_image(Ftr[l], Ytr, [])[:2])) for l in LAYERS}; lc = max(LAYERS, key=lambda l: per[l]['dev_soft_bits'])
        for nm, l in ((name, lc), (name + '_L22', 22)):
            f = Ftr[l].float(); fm_, fs_ = f.mean(0), f.std(0, unbiased=False) + 1e-6; del f
            score(nm, Ftr[l], Fte[l], dict(layer=l, dev_by_layer=per, **split_half(F1[l], F2[l], fm_, fs_, g)))
        del m, Ftr, Fte, F1, F2; torch.cuda.empty_cache()
    res['runtime_s'] = round(time.time() - t0, 1); (OUT / f"things1_sub-{s:02d}{'_smoke' if a.smoke else ''}.json").write_text(json.dumps(res, indent=1)); print('wrote', s, res['runtime_s'], flush=True)


def summarize():
    R = [json.loads(f.read_text()) for f in sorted(OUT.glob('things1_sub-*.json')) if 'smoke' not in f.name]; out = dict(dataset='THINGS-EEG1', n=len(R), arms={}, note=__doc__); fam = []
    for arm in [x for x in R[0]['arms'] if all(x in r['arms'] for r in R)]:
        v = [r['arms'][arm]['k1']['soft_bits'] for r in R]; fl = [r['arms'][arm]['shuffled_pair']['soft_bits'] for r in R]
        d = dict(k1=dict(mean=float(np.mean(v)), ci=boot_ci(v)), floor=float(np.mean(fl)), n_above_floor=int(sum(x > y for x, y in zip(v, fl))), p=signflip_p(np.array(v) - np.array(fl)))
        if 'split_half' in R[0]['arms'][arm]:
            sh = [r['arms'][arm]['split_half'] for r in R]; d['split_half'] = dict(mean=float(np.mean(sh)), ci=boot_ci(sh)); d['split_half_floor'] = float(np.mean([r['arms'][arm]['split_half_shuffled'] for r in R]))
        if 'layer' in R[0]['arms'][arm]: d['layers'] = [r['arms'][arm]['layer'] for r in R]
        out['arms'][arm] = d; fam.append((arm + '>floor', d['p']))
    for y in ('reve_twin', 'bandpower', 'raw'):
        dv = [r['arms']['reve']['k1']['soft_bits'] - r['arms'][y]['k1']['soft_bits'] for r in R]; out[f'reve_minus_{y}'] = dict(mean=float(np.mean(dv)), ci=boot_ci(dv), p=signflip_p(dv)); fam.append((f'reve>{y}', out[f'reve_minus_{y}']['p']))
    out['holm'] = {k: dict(p=p, p_holm=q) for (k, p), q in zip(fam, holm([p for _, p in fam]))}
    (OUT / 'summary_things1.json').write_text(json.dumps(out, indent=1)); print(json.dumps({k: v for k, v in out.items() if k != 'note'}, indent=1)[:4000])


if __name__ == '__main__':
    if a.summarize: summarize()
    else:
        mtr, mte = clip_maps()
        for s in [int(x) for x in a.subs.split(',')]:
            if (OUT / f'things1_sub-{s:02d}.json').exists() and not a.smoke: continue
            run(s, mtr, mte)
