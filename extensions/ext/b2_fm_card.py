"""B2 FM card (C3) with N4 pretraining dividend (rung R7). Extension of m7_latent.py under its G1 conditions (METHOD "B2 FM card: PASS";
NOVEL SOLUTIONS ROUND 2, N4): every frozen EEG foundation model is scored with the IDENTICAL probe as its random-initialised twin, the
layer is chosen on dev only, and the pretraining dividend = FM yield - twin yield (participant bootstrap, Holm over FMs within target).
FMs       : CBraMod (external/CBraMod, torch.load weights_only=True), LaBraM and EEGPT (braindecode classes + local safetensors, the fm_zoo.load
            recipe: time-position tables cut to our patch count, channels by name). Twin = the same class and config, random init (seed 0),
            no checkpoint. (REVE is the M7 arm; its numbers are read from results/ext/m7 for the card.)
Features  : the token outputs of candidate blocks (forward hooks on the model's block list; four evenly spaced depths incl. the last),
            flattened (unpooled), the registered fm_probe convention. EEGPT: if a block output is larger than 40,000 dims, only its summary
            tokens are kept (stated per output).
Input     : one 1-s epoch per trial, per-channel z-scoring with training statistics (image) / session statistics (PE), Fourier-resampled to
            the model's rate (CBraMod, LaBraM 200 Hz; EEGPT 256 Hz). Image = THINGS-EEG2 0-1.0 s (as Q7/M7); PE = results/fm_epochs samples
            180-380 = -0.1..0.9 s around feedback (the SAME window as the M7 REVE arm and raw_win).
Read-outs : as m7_latent.py (image: dual ridge -> CLIP ViT-H, alpha by MRR on the Q7 dev tenth, InfoNCE log-score k80/k40; PE: within-
            participant ridge with the STANDARD_TEST nested calibration; layer per outer fold by inner-CV corr). Floors: shuffled pairs (image),
            three circular shifts (PE), and N(0,1) noise input through the same model/layer/probe (image: 4,000 noise training rows).
Card      : per FM x target, yield, twin yield, dividend (with CI, Holm over FMs), band power / raw from the M7 files on the same rows.
Output    : results/ext/b2/<target>_<sub>.json, summary_<target>.json.
"""
import sys, json, time, math, argparse, re
from pathlib import Path
import numpy as np, torch
ROOT = Path('/path/to/eegaudit'); sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'experiments/ext')); sys.path.insert(0, str(ROOT / 'experiments/eegbits'))
ap = argparse.ArgumentParser(); ap.add_argument('--target', default='image'); ap.add_argument('--subs', default='1'); ap.add_argument('--fms', default='cbramod,labram,eegpt')
ap.add_argument('--smoke', action='store_true'); ap.add_argument('--summarize', default=''); ap.add_argument('--min_free_gb', type=float, default=10.0)
a = ap.parse_args(); sys.argv = sys.argv[:1]
from m7_latent import DEV, metrics_t, ridge_image, nested_pe, boot_ci, ratio_ci, signflip_p, holm, wait_gpu
OUT = ROOT / 'results/ext/b2'; OUT.mkdir(parents=True, exist_ok=True); SF = dict(cbramod=200, labram=200, eegpt=256); MAXD = 40000


def resample_t(X, num):
    """scipy.signal.resample (Fourier) for real input along the last axis, on the GPU."""
    N = X.shape[-1]
    if num == N: return X.float()
    F = torch.fft.rfft(X.float(), dim=-1); Y = torch.zeros(X.shape[:-1] + (num // 2 + 1,), dtype=F.dtype, device=X.device); m = min(N, num)
    Y[..., :m // 2 + 1] = F[..., :m // 2 + 1]
    if m % 2 == 0:
        if num < N: Y[..., m // 2] *= 2
        elif num > N: Y[..., m // 2] *= 0.5
    return torch.fft.irfft(Y, n=num, dim=-1) * (num / N)


def load_fm(name, pretrained, chans):
    """-> embed(x (B, C, T) GPU float) -> {layer: (B, D)}, layer ids, model channel list used, note."""
    torch.manual_seed(0); sf = SF[name]
    if name == 'cbramod':
        sys.path.insert(0, str(ROOT / 'external/CBraMod')); from models.cbramod import CBraMod
        m = CBraMod()
        if pretrained: m.load_state_dict(torch.load(ROOT / 'external/CBraMod/pretrained_weights/pretrained_weights.pth', map_location='cpu', weights_only=True))
        m.proj_out = torch.nn.Identity(); m = m.to(DEV).eval(); blocks = m.encoder.layers; call = lambda x: m(x.reshape(x.shape[0], x.shape[1], -1, 200))
        use = list(chans)
    else:
        import braindecode.models as bm
        from safetensors.torch import load_file
        path = ROOT / f'external/{name}'; cfg = json.load(open(path / 'config.json'))
        kw = {k: v for k, v in cfg.items() if k in ('chs_info', 'n_chans', 'n_outputs') and v is not None}; kw.update(n_times=sf, sfreq=sf)
        if name == 'eegpt': kw.update(chan_proj_type='none')
        m = getattr(bm, 'Labram' if name == 'labram' else 'EEGPT')(**kw)
        if pretrained:
            sd = load_file(str(path / 'model.safetensors')); own = m.state_dict()
            for k, v in list(sd.items()):
                if k in own and own[k].shape != v.shape:
                    if v.dim() == own[k].dim() and all(o <= w for o, w in zip(own[k].shape, v.shape)) and k != 'chans_id': sd[k] = v[tuple(slice(0, o) for o in own[k].shape)]
                    else: sd.pop(k)
            m.load_state_dict(sd, strict=False)
        m = m.to(DEV).eval()
        lists = [mod for _, mod in m.named_modules() if isinstance(mod, torch.nn.ModuleList)]; blocks = max(lists, key=len)
        if name == 'labram':
            use = list(chans); call = lambda x: m(x, return_patch_tokens=True, ch_names=[c.upper() for c in use])
        else:
            use = [c['ch_name'].upper() for c in cfg['chs_info']]; call = lambda x: m(x, return_features=True)
    n = len(blocks); L = sorted({max(1, round(n * q)) for q in (0.25, 0.5, 0.75, 1.0)}); store = {}
    for k in L:
        blocks[k - 1].register_forward_hook(lambda mod, i, o, k=k: store.__setitem__(k, o[0] if isinstance(o, tuple) else o))
    @torch.no_grad()
    def embed(x):
        store.clear(); call(x); out = {}
        for k in L:
            h = store[k]
            if h.dim() == 4: h = h[:, :, -1] if name == 'cbramod' else h       # CBraMod (B, C, patches, d): last (only) patch
            f = h.reshape(len(x), -1)
            if f.shape[1] > MAXD and name == 'eegpt' and h.dim() == 3: f = h[:, -4:].reshape(len(x), -1)   # summary tokens only
            out[k] = f
        return out
    return embed, L, use, sf


def to_model_layout(X, chans, use):
    """(n, C, T) in our channel order -> the model's layout by name (unmatched model channels = 0)."""
    if [c.upper() for c in chans] == [c.upper() for c in use]: return X
    up = [c.upper() for c in chans]; Z = torch.zeros(X.shape[0], len(use), X.shape[2], device=X.device)
    for k, nm in enumerate(use):
        if nm.upper() in up: Z[:, k] = X[:, up.index(nm.upper())]
    return Z


def layers_of(embed, L, X, bs=128, cpu=False):
    out = {l: [] for l in L}
    for b in range(0, len(X), bs):
        o = embed(X[b:b + bs].to(DEV).float())
        for l in L: out[l].append(o[l].half().cpu() if cpu else o[l].half())
    return {l: torch.cat(v) for l, v in out.items()}


def run_image(sub, fms):
    from run import CLIP, load_torch, load_raw
    wait_gpu(a.min_free_gb); t0 = time.time(); sid = f'sub-{sub:02d}'; g = torch.Generator().manual_seed(sub)
    Xtr4, Xte80, meta = load_raw(sid); chans = meta['ch_names']; Xtr = torch.from_numpy(Xtr4.mean(1)).to(DEV); del Xtr4
    Xte = torch.from_numpy(Xte80); del Xte80
    h1 = sum(Xte[:, r_].to(DEV) for r_ in range(40)) / 40; h2 = sum(Xte[:, r_].to(DEV) for r_ in range(40, 80)) / 40; k80, k40 = (h1 + h2) / 2, h1; del Xte
    Ytr = torch.from_numpy(load_torch(CLIP / 'ViT-H-14_features_train.pt')).to(DEV); Yte = torch.from_numpy(load_torch(CLIP / 'ViT-H-14_features_test.pt')).to(DEV)
    if a.smoke: sel = torch.randperm(len(Xtr), generator=g)[:4000].to(DEV); Xtr, Ytr = Xtr[sel], Ytr[sel]
    sd_ch = Xtr.std((0, 2), unbiased=False, keepdim=True); mu_ch = Xtr.mean((0, 2), keepdim=True)
    ntr = torch.randn((4000,) + Xtr.shape[1:], generator=torch.Generator().manual_seed(1000 + sub)).to(DEV) * sd_ch + mu_ch
    nte = torch.randn((200,) + Xtr.shape[1:], generator=torch.Generator().manual_seed(2000 + sub)).to(DEV) * sd_ch + mu_ch
    res = dict(target='image', subject=sid, n_train=int(len(Xtr)), smoke=a.smoke, arms={}); perm = torch.randperm(200, generator=g).to(DEV)
    for fm in fms:
        sf = SF[fm]; Z = resample_t(Xtr, sf); zmu, zsd = Z.mean((0, 2), keepdim=True), Z.std((0, 2), unbiased=False, keepdim=True) + 1e-6; del Z
        for pre in (True, False):
            name = fm if pre else fm + '_twin'; embed, L, use, _ = load_fm(fm, pre, chans)
            prep = lambda X: to_model_layout((resample_t(X, sf) - zmu) / zsd, chans, use)
            Ftr = {l: [] for l in L}
            for b in range(0, len(Xtr), 2000):
                o = layers_of(embed, L, prep(Xtr[b:b + 2000]), cpu=True)
                for l in L: Ftr[l].append(o[l])
            Ftr = {l: torch.cat(v) for l, v in Ftr.items()}; F80 = layers_of(embed, L, prep(k80)); F40 = layers_of(embed, L, prep(k40))
            per = {l: dict(zip(('alpha', 'dev_soft_bits'), ridge_image(Ftr[l], Ytr, [])[:2])) for l in L}; lc = max(L, key=lambda l: per[l]['dev_soft_bits'])
            Nt = layers_of(embed, L, prep(ntr)); Ne = layers_of(embed, L, prep(nte))
            for nm, l in ((name, lc), (name + '_last', L[-1])):
                al, dev, (p80, p40) = ridge_image(Ftr[l], Ytr, [F80[l], F40[l]]); _, _, (pn,) = ridge_image(Nt[l], Ytr[:4000] if not a.smoke else Ytr[:len(Nt[l])], [Ne[l]])
                res['arms'][nm] = dict(layer=l, layers=L, dim=int(Ftr[l].shape[1]), alpha=al, dev_soft_bits=dev, dev_by_layer=per, k80=metrics_t(p80, Yte), k40=metrics_t(p40, Yte),
                                       shuffled_pair_k80=metrics_t(p80, Yte[perm]), noise_k80=metrics_t(pn, Yte), channels_used=len(use))
                r = res['arms'][nm]; print(f"{sid} {nm} L{l}/{L}: k80 {r['k80']['soft_bits']:.4f} k40 {r['k40']['soft_bits']:.4f} dev {dev:.4f} pair {r['shuffled_pair_k80']['soft_bits']:.4f} "
                                          f"noise {r['noise_k80']['soft_bits']:.4f} D {r['dim']} ({time.time()-t0:.0f}s)", flush=True)
            del Ftr, F80, F40, Nt, Ne, embed; torch.cuda.empty_cache()
    res['runtime_s'] = round(time.time() - t0, 1); res['gpu_peak_GB'] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
    (OUT / f"image_{sid}{'_smoke' if a.smoke else ''}.json").write_text(json.dumps(res, indent=1)); print('wrote', sid, res['runtime_s'], flush=True)


def run_pe(fms):
    from belief_bottleneck import LAM, folds
    sys.path.insert(0, str(ROOT / 'experiments/loop'))
    src = open(ROOT / 'experiments/loop/s13b_gauss_bits.py').read(); src = src[:src.index('DATASETS = {')]; ns = {}; exec(compile(src, 's13b_head', 'exec'), ns)
    qtil, subs, fpath, _ = ns['targets_ds003458'](); t0 = time.time(); wait_gpu(a.min_free_gb)
    run_subs = subs[:2] if a.smoke else ([int(x) for x in a.subs.split(',')] if a.subs not in ('all', '1') else subs)
    models = {}
    for s in run_subs:
        dest = OUT / f"pe_sub-{s:03d}{'_smoke' if a.smoke else ''}.json"
        if dest.exists() and not a.smoke: continue
        tri = np.load(fpath('fm', s))['tri']; E = np.load(ROOT / f'results/fm_epochs/sub-{s:03d}.npz'); pidx = {t: k for k, t in enumerate(E['trial'])}
        chans = [str(c) for c in E['ch']]
        X = torch.from_numpy(E['X'][[pidx[t] for t in tri]].reshape(len(tri), len(chans), -1).astype(np.float32)).to(DEV)
        X = (X - X.mean((0, 2), keepdim=True)) / (X.std((0, 2), unbiased=False, keepdim=True) + 1e-6); X = X[:, :, 180:380]
        Xn = torch.randn(X.shape, generator=torch.Generator().manual_seed(3000 + s)).to(DEV)
        y = torch.tensor(qtil(s, tri), dtype=torch.float64, device=DEV); n = len(y); fo = folds(n)
        Ys = torch.stack([y] + [torch.roll(y, round(n * k / 4)) for k in (1, 2, 3)], 1)
        r = dict(subject=f'sub-{s:03d}', n_trials=n, window='-0.1..0.9 s post-feedback (fm_epochs samples 180-380), as M7 REVE / raw_win', arms={})
        for fm in fms:
            for pre in (True, False):
                name = fm if pre else fm + '_twin'
                if (fm, pre, tuple(chans)) not in models: models = {(fm, pre, tuple(chans)): load_fm(fm, pre, chans)}
                embed, L, use, sf = models[(fm, pre, tuple(chans))]
                for inp, XX in (('', X), ('_noise', Xn)):
                    Fl = layers_of(embed, L, to_model_layout(resample_t(XX, sf), chans, use), bs=64)
                    outs = {l: nested_pe(Fl[l].double(), torch.ones(Fl[l].shape[1], dtype=torch.float64, device=DEV), Ys, fo, LAM) for l in L}
                    Gs = torch.full((n, 4), float('nan'), dtype=torch.float64, device=DEV); chosen = {}
                    for b in range(10):
                        lb = max(L, key=lambda l: outs[l][1][b]); chosen[b] = lb; mb = torch.tensor(fo == b, device=DEV); Gs[mb] = outs[lb][2][mb]
                    r['arms'][name + inp] = dict(gain=float(Gs[:, 0].mean()), shifted=[float(v) for v in Gs[:, 1:].mean(0)], layers=chosen, layer_ids=L,
                                                 by_layer={l: float(outs[l][2][:, 0].mean()) for l in L}, dim={l: int(Fl[l].shape[1]) for l in L}, channels_used=len(use))
                    print(f"sub-{s:03d} {name+inp}: gain {r['arms'][name+inp]['gain']:.4f} shift {np.mean(r['arms'][name+inp]['shifted']):.4f} ({time.time()-t0:.0f}s)", flush=True)
                    del Fl, outs; torch.cuda.empty_cache()
        r['runtime_s'] = round(time.time() - t0, 1); dest.write_text(json.dumps(r, indent=1))


def summarize(target):
    R = [json.loads(f.read_text()) for f in sorted(OUT.glob(f'{target}_sub-*.json')) if 'smoke' not in f.name]
    key = (lambda d: d['k80']['soft_bits']) if target == 'image' else (lambda d: d['gain'])
    m7 = {r['subject']: json.loads((ROOT / f"results/ext/m7/{target}_{r['subject']}.json").read_text()) for r in R if (ROOT / f"results/ext/m7/{target}_{r['subject']}.json").exists()}
    out = dict(target=target, n=len(R), participants=[r['subject'] for r in R], arms={}, dividend={}, note=__doc__)
    for arm in [x for x in R[0]['arms'] if all(x in r['arms'] for r in R)]:
        v = [key(r['arms'][arm]) for r in R]; out['arms'][arm] = dict(mean=float(np.mean(v)), ci=boot_ci(v), layers=[r['arms'][arm].get('layer', r['arms'][arm].get('layers')) for r in R])
    fam = []
    fms = sorted({x for x in R[0]['arms'] if not x.endswith(('_twin', '_last', '_noise', '_twin_last', '_twin_noise'))})
    for fm in fms:
        d = [key(r['arms'][fm]) - key(r['arms'][fm + '_twin']) for r in R]
        out['dividend'][fm] = dict(label='N4 pretraining dividend = FM - random-init twin (identical probe, dev-chosen layers)', mean=float(np.mean(d)), ci=boot_ci(d), p=signflip_p(d), n_pos=int(sum(np.array(d) > 0)))
        fam.append((fm, out['dividend'][fm]['p']))
    if m7 and len(m7) == len(R):                                           # REVE dividend and reference arms from M7 on the same participants
        for ref in (['raw', 'raw_win', 'bandpower'] if target == 'pe' else ['raw', 'bandpower']):
            if all(ref in m7[r['subject']]['arms'] for r in R):
                out['arms']['M7_' + ref] = dict(mean=float(np.mean([key(m7[r['subject']]['arms'][ref]) for r in R])))
        d = [key(m7[r['subject']]['arms']['reve']) - key(m7[r['subject']]['arms']['reve_twin']) for r in R]
        out['dividend']['reve'] = dict(mean=float(np.mean(d)), ci=boot_ci(d), p=signflip_p(d), n_pos=int(sum(np.array(d) > 0)), source='results/ext/m7'); fam.append(('reve', out['dividend']['reve']['p']))
    adj = holm([p for _, p in fam]); out['holm_over_fms'] = {k: dict(p=p, p_holm=q) for (k, p), q in zip(fam, adj)}
    (OUT / f'summary_{target}.json').write_text(json.dumps(out, indent=1)); print(json.dumps({k: v for k, v in out.items() if k != 'note'}, indent=1)[:5000])


if __name__ == '__main__':
    if a.summarize: summarize(a.summarize)
    elif a.target == 'image':
        for s in [int(x) for x in a.subs.split(',')]:
            if (OUT / f'image_sub-{s:02d}.json').exists() and not a.smoke: continue
            run_image(s, a.fms.split(','))
    else: run_pe(a.fms.split(','))
