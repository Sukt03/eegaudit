"""Idea I: TOKEN-GRID YIELD MAP (rung R5 x R7; EXTENSIONS_LOOP frontier batch "Idea I", METHOD; owner EXECUTOR-4).
Question: where does a frozen EEG foundation model's token grid keep or lose the target information, compared with raw voltage in the
SAME region x time cell, and is the pooled-embedding deficit spatially localised?
Model     : CBraMod frozen (primary; final layer = the registered fm_probe convention, tokens = channel x 1-s patch x 200), REVE frozen
            (second; final layer, channel x patch x 512). Tokens are NOT re-trained; the read-out is the locked one per target.
Cells     : the eight fixed 6-channel M3 regions (experiments/ext/regions.py REGIONS) x the model's 1-s time patches. Image (THINGS-EEG2,
            0-1 s): one patch -> 8 region cells. PE (ds003458 fm_epochs, -1..1 s at 200 Hz): CBraMod's two non-overlapping patches
            (-1..0 s, 0..1 s) -> 8 x 2 cells; REVE uses its post-feedback patch only (8 cells), as M7.
            Token cell = the tokens of the region's channels in that patch (flattened). Raw cell = raw voltage of the SAME channels over the
            SAME samples (image: 250 Hz; PE: 20-ms bins). Plus 'all' (every token / every sample) and 'pooled' (channel-mean token).
Read-outs : image = Q7 dual ridge -> CLIP ViT-H, alpha by MRR on the dev tenth, InfoNCE log-score k80 (m7_latent.ridge_image);
            PE = within-participant ridge with the STANDARD_TEST nested calibration (m7_latent.nested_pe), Gaussian log-density gain.
Statistics: per cell, token - raw difference per participant; FWER over cells by the max-|t| sign-flip null (exact 2^10 for image;
            100,000 random flips for PE; GPU). Adoption (pre-stated): G2 = the all-token yield reproduces the registered unpooled-token yield
            (image: fm_probe_cbramod 2-way per participant; PE: M7/B2 full-token arm), and >= 1 difference cell with FWER p < 0.05 on BOTH
            targets; otherwise the figure goes to the appendix as "no localised deficit".
Floors    : shuffled pairs per cell (image); three circular shifts (PE).
Output    : results/ext/idea1/<target>_<sub>.json, summary_<target>.json.
"""
import sys, json, time, argparse
from pathlib import Path
import numpy as np, torch
ROOT = Path('/path/to/eegaudit'); sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'experiments/ext')); sys.path.insert(0, str(ROOT / 'experiments/eegbits'))
ap = argparse.ArgumentParser(); ap.add_argument('--target', default='image'); ap.add_argument('--subs', default='1'); ap.add_argument('--models', default='cbramod,reve')
ap.add_argument('--smoke', action='store_true'); ap.add_argument('--summarize', default=''); ap.add_argument('--min_free_gb', type=float, default=10.0)
a = ap.parse_args(); sys.argv = sys.argv[:1]
from m7_latent import DEV, metrics_t, ridge_image, nested_pe, boot_ci, wait_gpu, resample200, load_reve
from regions import REGIONS
OUT = ROOT / 'results/ext/idea1'; OUT.mkdir(parents=True, exist_ok=True)


def idx(chans, names):
    up = [c.upper() for c in chans]; return [up.index(n.upper()) for n in names]


def cbramod_tokens(X, pretrained=True):
    """X (n, C, T multiple of 200) at 200 Hz, standardized -> tokens (n, C, patches, 200) on CPU fp16 (final layer = registered convention)."""
    sys.path.insert(0, str(ROOT / 'external/CBraMod')); from models.cbramod import CBraMod
    m = CBraMod(); m.load_state_dict(torch.load(ROOT / 'external/CBraMod/pretrained_weights/pretrained_weights.pth', map_location='cpu', weights_only=True)); m.proj_out = torch.nn.Identity(); m = m.to(DEV).eval()
    out = []
    with torch.no_grad():
        for b in range(0, len(X), 256):
            xb = X[b:b + 256].to(DEV).float(); out.append(m(xb.reshape(xb.shape[0], xb.shape[1], -1, 200)).half().cpu())
    return torch.cat(out)


def reve_tokens(X, chans):
    m, pos = load_reve(True, chans); out = []
    with torch.no_grad():
        for b in range(0, len(X), 128):
            xb = X[b:b + 128].to(DEV).float(); t = m(xb, pos.expand(len(xb), -1, -1)); out.append(t.half().cpu())   # (B, C, patches, 512)
    return torch.cat(out)


def run_image(sub, models):
    from run import CLIP, load_torch, load_raw
    wait_gpu(a.min_free_gb); t0 = time.time(); sid = f'sub-{sub:02d}'; g = torch.Generator().manual_seed(sub)
    Xtr4, Xte80, meta = load_raw(sid); chans = meta['ch_names']; Xtr = torch.from_numpy(Xtr4.mean(1)); del Xtr4
    Xte = torch.from_numpy(Xte80); del Xte80; k80 = sum(Xte[:, r_].to(DEV) for r_ in range(80)).cpu() / 80; del Xte
    Ytr = torch.from_numpy(load_torch(CLIP / 'ViT-H-14_features_train.pt')).to(DEV); Yte = torch.from_numpy(load_torch(CLIP / 'ViT-H-14_features_test.pt')).to(DEV)
    if a.smoke: sel = torch.randperm(len(Xtr), generator=g)[:4000]; Xtr, Ytr = Xtr[sel], Ytr[sel.to(DEV)]
    perm = torch.randperm(200, generator=g).to(DEV); res = dict(target='image', subject=sid, n_train=len(Xtr), smoke=a.smoke, cells={})
    def cell(name, Ftr, Fte):
        al, dev, (p,) = ridge_image(Ftr, Ytr, [Fte]); r = dict(dim=int(Ftr.shape[1]), alpha=al, k80=metrics_t(p, Yte), pair=metrics_t(p, Yte[perm])['soft_bits'])
        res['cells'][name] = r; print(f"{sid} {name}: {r['k80']['soft_bits']:.4f} (2-way {r['k80']['two_way']:.3f}, pair {r['pair']:.4f}, D {r['dim']}) {time.time()-t0:.0f}s", flush=True)
    cell('raw|all', Xtr.reshape(len(Xtr), -1), k80.reshape(200, -1))
    for rg, nm in REGIONS.items():
        ii = idx(chans, nm); cell(f'raw|{rg}', Xtr[:, ii].reshape(len(Xtr), -1), k80[:, ii].reshape(200, -1))
    Z = resample200(Xtr.to(DEV)); mu, sd = Z.mean((0, 2), keepdim=True), Z.std((0, 2), unbiased=False, keepdim=True) + 1e-6; Z = ((Z - mu) / sd).cpu()
    Zt = ((resample200(k80.to(DEV)) - mu) / sd).cpu()
    for mdl in models:
        T = cbramod_tokens(Z) if mdl == 'cbramod' else reve_tokens(Z, chans); Tt = cbramod_tokens(Zt) if mdl == 'cbramod' else reve_tokens(Zt, chans)
        cell(f'{mdl}|all', T.reshape(len(T), -1), Tt.reshape(200, -1)); cell(f'{mdl}|pooled', T.float().mean(1).reshape(len(T), -1), Tt.float().mean(1).reshape(200, -1))
        for rg, nm in REGIONS.items():
            ii = idx(chans, nm); cell(f'{mdl}|{rg}', T[:, ii].reshape(len(T), -1), Tt[:, ii].reshape(200, -1))
        del T, Tt; torch.cuda.empty_cache()
    reg = ROOT / f'results/eegbits/fm_probe_cbramod_{sid}.json'
    if reg.exists(): res['g2_registered_cbramod_two_way_k80'] = json.loads(reg.read_text())['rows'][0]['two_way']
    res['runtime_s'] = round(time.time() - t0, 1); (OUT / f"image_{sid}{'_smoke' if a.smoke else ''}.json").write_text(json.dumps(res, indent=1)); print('wrote', sid, flush=True)


def run_pe(models):
    from belief_bottleneck import LAM, folds
    sys.path.insert(0, str(ROOT / 'experiments/loop'))
    src = open(ROOT / 'experiments/loop/s13b_gauss_bits.py').read(); src = src[:src.index('DATASETS = {')]; ns = {}; exec(compile(src, 's13b_head', 'exec'), ns)
    qtil, subs, fpath, _ = ns['targets_ds003458'](); wait_gpu(a.min_free_gb); t0 = time.time()
    for s in (subs[:2] if a.smoke else subs):
        dest = OUT / f"pe_sub-{s:03d}{'_smoke' if a.smoke else ''}.json"
        if dest.exists() and not a.smoke: continue
        tri = np.load(fpath('fm', s))['tri']; E = np.load(ROOT / f'results/fm_epochs/sub-{s:03d}.npz'); pidx = {t: k for k, t in enumerate(E['trial'])}; chans = [str(c) for c in E['ch']]
        X = torch.from_numpy(E['X'][[pidx[t] for t in tri]].reshape(len(tri), len(chans), -1).astype(np.float32)).to(DEV)
        X = (X - X.mean((0, 2), keepdim=True)) / (X.std((0, 2), unbiased=False, keepdim=True) + 1e-6)
        y = torch.tensor(qtil(s, tri), dtype=torch.float64, device=DEV); n = len(y); fo = folds(n); Ys = torch.stack([y] + [torch.roll(y, round(n * k / 4)) for k in (1, 2, 3)], 1)
        r = dict(subject=f'sub-{s:03d}', n_trials=n, cells={}); one = lambda F: torch.ones(F.shape[1], dtype=torch.float64, device=DEV)
        def cell(name, F):
            F = F.to(DEV).double(); _, _, G = nested_pe(F, one(F), Ys, fo, LAM); r['cells'][name] = dict(dim=int(F.shape[1]), gain=float(G[:, 0].mean()), shifted=[float(v) for v in G[:, 1:].mean(0)])
        bins = X.reshape(n, len(chans), 100, 4).mean(-1)                    # 20-ms bins, -1..1 s: patch 0 = bins 0-49, patch 1 = bins 50-99
        cell('raw|all', bins.reshape(n, -1))
        for p_, sl in (('pre', slice(0, 50)), ('post', slice(50, 100))):
            for rg, nm in REGIONS.items(): cell(f'raw|{rg}|{p_}', bins[:, idx(chans, nm), sl].reshape(n, -1))
        for mdl in models:
            if mdl == 'cbramod':
                T = cbramod_tokens(X)                                          # (n, C, 2, 200)
                cell('cbramod|all', T.reshape(n, -1)); cell('cbramod|pooled', T.float().mean(1).reshape(n, -1))
                for pi, p_ in enumerate(('pre', 'post')):
                    for rg, nm in REGIONS.items(): cell(f'cbramod|{rg}|{p_}', T[:, idx(chans, nm), pi].reshape(n, -1))
            else:
                from transformers import AutoModel
                positions = AutoModel.from_pretrained(str(ROOT / 'external/reve/positions'), trust_remote_code=True, local_files_only=True, use_safetensors=True)
                keep = [c for c in chans if c in positions.mapping]; Xk = X[:, idx(chans, keep)]; T = reve_tokens(Xk, keep)[:, :, -1]   # post-feedback patch
                cell('reve|all', T.reshape(n, -1)); cell('reve|pooled', T.float().mean(1))
                for rg, nm in REGIONS.items():
                    if all(c in keep for c in nm): cell(f'reve|{rg}|post', T[:, idx(keep, nm)].reshape(n, -1))
        r['runtime_s'] = round(time.time() - t0, 1); dest.write_text(json.dumps(r, indent=1))
        print(f"sub-{s:03d}", {k: round(v['gain'], 3) for k, v in r['cells'].items() if k.endswith(('all', 'pooled')) or '|post' in k}, f'{time.time()-t0:.0f}s', flush=True)


def maxt(D, B=100000, seed=20261121):
    """D (participants x cells) differences -> per-cell FWER p by the max-|t| sign-flip null (exact when n <= 16)."""
    D = torch.as_tensor(D, dtype=torch.float64, device=DEV); n = D.shape[0]
    S = (((torch.arange(2 ** n, device=DEV)[:, None] >> torch.arange(n, device=DEV)) & 1).double() * 2 - 1) if n <= 16 else \
        (torch.randint(0, 2, (B, n), device=DEV, generator=torch.Generator(device=DEV).manual_seed(seed)).double() * 2 - 1)
    t = lambda M: M.mean(-2) / (M.std(-2) / n ** 0.5 + 1e-12)
    t0 = t(D); null = torch.cat([t(S[b:b + 4096, :, None] * D[None]).abs().max(1).values for b in range(0, len(S), 4096)])
    return t0.tolist(), [float(((null >= abs(x)).sum() + 1) / (len(null) + 1)) for x in t0.tolist()]


def summarize(target):
    R = [json.loads(f.read_text()) for f in sorted(OUT.glob(f'{target}_sub-*.json')) if 'smoke' not in f.name]
    key = (lambda c: c['k80']['soft_bits']) if target == 'image' else (lambda c: c['gain'])
    cells = [c for c in R[0]['cells'] if all(c in r['cells'] for r in R)]; out = dict(target=target, n=len(R), cells={}, diff={}, note=__doc__)
    for c in cells:
        v = [key(r['cells'][c]) for r in R]; out['cells'][c] = dict(mean=float(np.mean(v)), ci=boot_ci(v))
    for mdl in ('cbramod', 'reve'):
        pairs = [(c, 'raw|' + c.split('|', 1)[1]) for c in cells if c.startswith(mdl + '|') and ('raw|' + c.split('|', 1)[1]) in cells and not c.endswith(('|all', '|pooled'))]
        if not pairs: continue
        D = np.array([[key(r['cells'][x]) - key(r['cells'][y]) for x, y in pairs] for r in R]); t, p = maxt(D)
        out['diff'][mdl] = {x: dict(token_minus_raw=float(D[:, i].mean()), ci=boot_ci(D[:, i]), t=t[i], p_fwer=p[i]) for i, (x, _) in enumerate(pairs)}
        out['diff'][mdl]['any_cell_p_fwer_lt_0.05'] = bool(min(p) < 0.05)
    (OUT / f'summary_{target}.json').write_text(json.dumps(out, indent=1)); print(json.dumps({k: v for k, v in out.items() if k != 'note'}, indent=1)[:5000])


if __name__ == '__main__':
    if a.summarize: summarize(a.summarize)
    elif a.target == 'image':
        for s in [int(x) for x in a.subs.split(',')]:
            if (OUT / f'image_sub-{s:02d}.json').exists() and not a.smoke: continue
            run_image(s, a.models.split(','))
    else: run_pe(a.models.split(','))
