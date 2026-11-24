"""RUNNER FIX 2 (round_20261123_auto; METHODS B2, IDEATION B1 falsifier): MI matched-rank regression control on EEGMMIDB.
Question: the proxy-regressed raw read-out keeps 10.0% of the raw yield (0.030 of 0.302 bits, mi_card_v3). Is that loss specific to the five 'ocular proxy'
derivations, or does regressing ANY five derivations of the same form (4 bipolar differences + 1 three-channel mean, 11 distinct channels, rank 5) remove as much?
Pipeline = mi_card_v3 raw read-out, imported unchanged (bandpass 1-40 Hz, per-participant float64 OLS of all 64 channels on the 5 series + intercept over all samples
of all the participant's trials, residual -> features 'raw' (0.5-4.0 s, 20 Hz block means, 4480-d) -> per-participant z-score -> PCA-256 -> exact-Newton multinomial
logistic, leave-participant-group-out (10 groups), (lambda, T) on the inner group; bits = log2 p(true) - log2 train prior).
Arms: raw (no regression) | proxy (the v3 derivations; GATE: raw, raw_eogreg eeg/noise/shuffle == saved mi_card_v3 means to 1e-4) |
  frontal_dXX: 20 draws from the 12 frontal channels that are NOT proxy channels (AF7 AF3 AFZ AF4 AF8 F5 F3 F1 FZ F2 F4 F6) |
  nonfrontal_dXX: 20 draws from channels that are neither proxy nor Fp/AF/F (central, parietal, occipital, temporal-parietal) |
  any_dXX: 20 draws from all 53 non-proxy channels.
Kept share = mean bits(regressed) / mean bits(raw) (ratio of participant means; participant-bootstrap CI). Each arm also stores the share of 1-40 Hz variance removed.
Nulls: noise input regressed on its OWN derivations (proxy and draw 0 of each family), shuffled labels (proxy and draw 0).
Planted positive control (known location): noise input + a class-specific 3 Hz pattern on C3 C4 CZ CP3 CP4 (amp x per-channel SD); regressing the proxies (which do not
contain those channels) must keep the planted yield, regressing derivations built from those five channels must remove most of it.
Usage: --out DIR [--subjects 1,2,..] [--draws 20] [--families frontal,nonfrontal,any]"""
import argparse, hashlib, json, math, sys, time
from pathlib import Path
import numpy as np, torch
ROOT = Path('/path/to/eegaudit'); sys.path.insert(0, str(ROOT / 'experiments/ext'))
import mi_card_v3 as M
DEV = 'cuda'; CODE_SHA256 = hashlib.sha256(open(__file__, 'rb').read()).hexdigest(); t0 = time.time(); log = lambda *m: print(f'[{time.time()-t0:7.1f}s]', *m, flush=True)
FRONT = ['AF7', 'AF3', 'AFZ', 'AF4', 'AF8', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6']; PLANT_CH = ['C3', 'C4', 'CZ', 'CP3', 'CP4']

def dmat(ch, pairs, mean3):                               # [64, 5] derivation weights: 4 differences + 1 mean of 3
    ix = {c: i for i, c in enumerate(ch)}; D = torch.zeros(64, 5, dtype=torch.float64, device=DEV)
    for j, (p, q) in enumerate(pairs): D[ix[p], j] = 1; D[ix[q], j] = -1
    for c in mean3: D[ix[c], 4] = 1 / 3
    return D

def regress(R0, sid, nP, D):                              # = mi_card_v3.eog_regress with the derivations given as a weight matrix
    out = torch.empty_like(R0); T = R0.shape[-1]; share = []
    for k in range(nP):
        r = torch.where(sid == k)[0]; x = R0[r].double(); Pk = torch.einsum('nct,cj->njt', x, D)
        A = torch.cat([Pk.permute(0, 2, 1).reshape(-1, 5), torch.ones(len(r) * T, 1, dtype=torch.float64, device=DEV)], 1)
        Y = x.permute(0, 2, 1).reshape(-1, 64); B = torch.linalg.lstsq(A, Y).solution; E = Y - A @ B
        share.append(float(1 - E.pow(2).sum() / (Y - Y.mean(0)).pow(2).sum())); out[r] = E.reshape(len(r), T, 64).permute(0, 2, 1).float(); del x, Pk, A, Y, E
    return out, float(np.mean(share))

def raw_feat(Z, sid, nP):                                 # = features()['raw'] + zpp (per-participant column z-score)
    N = len(Z); Xf = M.bandpass(Z, 1, 40)[:, :, M.W0:M.W1].reshape(N, 64, 70, 8).mean(-1).reshape(N, -1).double()
    for k in range(nP):
        r = sid == k; Xf[r] = (Xf[r] - Xf[r].mean(0)) / (Xf[r].std(0) + 1e-8)
    return Xf

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out', required=True); ap.add_argument('--subjects', default=''); ap.add_argument('--draws', type=int, default=20)
    ap.add_argument('--families', default='frontal,nonfrontal,any'); ap.add_argument('--lchunk', type=int, default=4); a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    cache = ROOT / 'results/ext/mi_card_cache'; D_, excl = {}, {}
    for s in ([int(x) for x in a.subjects.split(',')] if a.subjects else range(1, 110)):
        d, why = M.load_sub(s, cache)
        if d is None: excl[s] = why; continue
        c = np.bincount(d['y'], minlength=4)
        if c.max() - c.min() > 2: excl[s] = 'class counts'; continue
        D_[s] = d
    P = sorted(D_); nP = len(P); ch = D_[P[0]]['ch']; assert all(D_[s]['ch'] == ch for s in P); log(f'{nP} participants, excluded {len(excl)}')
    X = torch.tensor(np.concatenate([D_[s]['X'] for s in P]), dtype=torch.float32, device=DEV); y = torch.tensor(np.concatenate([D_[s]['y'] for s in P]), device=DEV)
    sid = torch.tensor(np.concatenate([[k] * len(D_[s]['y']) for k, s in enumerate(P)]), device=DEV); N = len(y); del D_
    g = torch.Generator(device=DEV); g.manual_seed(20261121); R0 = M.bandpass(X, 1, 40); Nz = M.bandpass(torch.randn(X.shape, generator=g, device=DEV), 1, 40)   # = v3 noise draw
    for k in range(nP):
        r = sid == k; Nz[r] = Nz[r] / Nz[r].std((0, 2), keepdim=True) * R0[r].std((0, 2), keepdim=True)
    Nz0 = M.bandpass(Nz, 1, 40); torch.cuda.empty_cache()            # v3: eog_regress(bandpass(Z, 1, 40)) with Z = Nz (already band-passed once)
    gperm = torch.Generator(device=DEV); gperm.manual_seed(7); ysh = y.clone()
    for k in range(nP):
        r = torch.where(sid == k)[0]; ysh[r] = y[r[torch.randperm(len(r), generator=gperm, device=DEV)]]
    G = min(10, nP); grp = torch.tensor([k % G for k in range(nP)], device=DEV)[sid]; fits, units, pca_rows = [], [], {}
    for gg in range(G):
        v = (gg + 1) % G; outer = torch.where(grp != gg)[0]; pca_rows[gg] = outer
        fits.append((torch.where((grp != gg) & (grp != v))[0], torch.where(grp == v)[0], gg)); fits.append((outer, torch.where(grp == gg)[0], gg)); units.append(dict(inner=[len(fits) - 2], final=len(fits) - 1))
    yl = {c: [yy[f[0]] for f in fits] for c, yy in (('eeg', y), ('shuffle', ysh))}
    eng = lambda Xf, fi, ylab: M.primal_fits(Xf, fi, ylab, M.LAMS, pca_rows, lchunk=a.lchunk)
    def score(Xf, cond='eeg'):                             # per-participant bits through the unchanged v3 LPGO path
        R, _ = M.run_design('rf2', units, fits, {cond: {'x': Xf}}, y, {cond: yl['shuffle' if cond == 'shuffle' else 'eeg']}, sid, nP, eng, [], [], None, None, None)
        return [R[k]['x'][cond]['bits'] for k in range(nP)]
    res = dict(code_sha256=CODE_SHA256, participants=P, n=nP, arms={}); dest = out / 'arms.json'
    if dest.exists(): res = json.loads(dest.read_text()); assert res['participants'] == P
    def run(name, src, Dm, cond='eeg', meta=None):
        if name in res['arms']: return
        if Dm is None: Z, sh = src, 0.0
        else: Z, sh = regress(src, sid, nP, Dm)
        b = score(raw_feat(Z, sid, nP), cond); res['arms'][name] = dict(bits=b, mean=float(np.mean(b)), var_share_removed=sh, **(meta or {})); del Z; torch.cuda.empty_cache()
        res['runtime_s'] = round(time.time() - t0); dest.write_text(json.dumps(res)); log(name, f'mean bits {np.mean(b):+.4f} var removed {sh:.3f}')
    Dp = dmat(ch, M.PROXY, M.FPM); pch = {c for p in M.PROXY for c in p} | set(M.FPM); assert len(pch) == 11
    assert torch.allclose(torch.einsum('nct,cj->njt', R0[:4].double(), Dp), M.proxies(R0[:4].double(), ch)), 'proxy matrix != mi_card_v3.proxies'
    # v3 path: unregressed rows = features(X) (ONE band-pass inside raw_feat); regressed rows = features(regress(bandpass(X))) (band-pass, regress, band-pass)
    run('raw', X, None); run('raw|shuffle', X, None, 'shuffle'); run('raw|noise', Nz, None); del X, Nz; torch.cuda.empty_cache()
    run('proxy', R0, Dp); run('proxy|noise', Nz0, Dp); run('proxy|shuffle', R0, Dp, 'shuffle')
    # gate vs saved mi_card_v3 (only on the full cohort)
    sv = json.loads((ROOT / 'results/ext/mi_card_v3/primary_lpgo/summary_mi.json').read_text())
    if sv['participants'] == P:
        gate = {}
        for nm, (row, cond) in {'raw': ('raw', 'eeg'), 'proxy': ('raw_eogreg', 'eeg'), 'raw|noise': ('raw', 'noise'), 'proxy|noise': ('raw_eogreg', 'noise'), 'raw|shuffle': ('raw', 'shuffle'), 'proxy|shuffle': ('raw_eogreg', 'shuffle')}.items():
            gate[nm] = dict(this=res['arms'][nm]['mean'], saved=sv['rows'][row][cond]['mean'], abs_diff=abs(res['arms'][nm]['mean'] - sv['rows'][row][cond]['mean']))
        res['gate'] = gate; res['gate_pass_1e-4'] = bool(all(v['abs_diff'] < 1e-4 for v in gate.values())); log('GATE', json.dumps(gate)); assert res['gate_pass_1e-4'], 'GATE FAIL vs mi_card_v3'
    else: log('gate skipped (subset run)')
    nonfront = [c for c in ch if c not in pch and c not in M.FP and c not in FRONT]; anyp = [c for c in ch if c not in pch]
    assert all(c in ch for c in FRONT + PLANT_CH) and not (set(FRONT) & pch); pools = dict(frontal=FRONT, nonfrontal=nonfront, any=anyp); res['pools'] = pools
    for fam in a.families.split(','):
        rng = np.random.default_rng(dict(frontal=101, nonfrontal=202, any=303)[fam])
        for d in range(a.draws):
            c = list(rng.choice(pools[fam], 11, replace=False)); pairs = [(c[0], c[1]), (c[2], c[3]), (c[4], c[5]), (c[6], c[7])]; m3 = c[8:]; Dm = dmat(ch, pairs, m3); assert int(torch.linalg.matrix_rank(Dm)) == 5
            run(f'{fam}_d{d:02d}', R0, Dm, meta=dict(pairs=pairs, mean3=m3))
            if d == 0: run(f'{fam}_d00|noise', Nz0, Dm); run(f'{fam}_d00|shuffle', R0, Dm, 'shuffle')
    # planted positive control: class pattern on C3 C4 CZ CP3 CP4 in the noise input
    ix = [ch.index(c) for c in PLANT_CH]; gp = torch.Generator(device=DEV); gp.manual_seed(5); pat = torch.randint(0, 2, (4, 5), generator=gp, device=DEV).float() * 2 - 1
    tt = torch.arange(Nz0.shape[-1], device=DEV) / M.SF; wave = torch.sin(2 * math.pi * 3 * tt); Zp = Nz0.clone(); sdc = Nz0[:, ix].std((0, 2))
    Zp[:, ix] += 0.5 * sdc[None, :, None] * pat[y][:, :, None] * wave[None, None]
    Dc = dmat(ch, [('C3', 'C4'), ('CP3', 'CP4'), ('C3', 'CP4'), ('CZ', 'CP3')], ['C3', 'CZ', 'C4'])
    run('planted', Zp, None); run('planted_proxyreg', Zp, Dp); run('planted_centralreg', Zp, Dc); del Zp
    # ---------------- summary
    A = res['arms']; T = lambda v: torch.tensor(np.asarray(v, np.float64), device=DEV); gb = torch.Generator(device=DEV); gb.manual_seed(0); IX = torch.randint(0, nP, (10000, nP), generator=gb, device=DEV)
    raw = T(A['raw']['bits'])
    def share(nm, base=raw):
        v = T(A[nm]['bits']); b = v[IX].mean(1) / base[IX].mean(1); return dict(kept=float(v.mean() / base.mean()), ci95=[float(b.quantile(.025)), float(b.quantile(.975))], bits=float(v.mean()), var_share_removed=A[nm]['var_share_removed'])
    S = dict(NOT_CITABLE='until independent review', code_sha256=CODE_SHA256, n=nP, gate=res.get('gate'), gate_pass=res.get('gate_pass_1e-4'), raw_bits=float(raw.mean()), proxy=share('proxy'), families={},
             nulls={k: A[k]['mean'] for k in A if '|' in k}, definition='kept = mean bits(regressed) / mean bits(raw), ratio of participant means; CI = participant bootstrap 10000 (GPU, seed 0)')
    S['nulls_ok_abs_le_0.02'] = bool(all(abs(v) <= 0.02 for v in S['nulls'].values()))
    for fam in a.families.split(','):
        dr = [share(k) for k in sorted(A) if k.startswith(fam + '_d') and '|' not in k]; ks = np.array([d['kept'] for d in dr]); vs = np.array([d['var_share_removed'] for d in dr])
        dboot = torch.stack([T(A[k]['bits'])[IX].mean(1) for k in sorted(A) if k.startswith(fam + '_d') and '|' not in k]).mean(0) / raw[IX].mean(1)   # mean-over-draws kept share, participant bootstrap
        pb = T(A['proxy']['bits'])[IX].mean(1) / raw[IX].mean(1)
        S['families'][fam] = dict(n_draws=len(ks), kept_mean=float(ks.mean()), kept_sd=float(ks.std(ddof=1)) if len(ks) > 1 else None, kept_min=float(ks.min()), kept_max=float(ks.max()), kept_sorted=sorted(ks.tolist()),
                                  kept_mean_participant_ci95=[float(dboot.quantile(.025)), float(dboot.quantile(.975))], var_removed_mean=float(vs.mean()), var_removed_range=[float(vs.min()), float(vs.max())],
                                  n_draws_kept_le_proxy=int((ks <= S['proxy']['kept']).sum()), empirical_p_proxy_below_draws=float((1 + (ks <= S['proxy']['kept']).sum()) / (1 + len(ks))),
                                  proxy_minus_family_mean=dict(d=float(S['proxy']['kept'] - ks.mean()), ci95=[float((pb - dboot).quantile(.025)), float((pb - dboot).quantile(.975))]),
                                  corr_kept_vs_var_removed=float(np.corrcoef(ks, vs)[0, 1]) if len(ks) > 2 else None)
    pl = T(A['planted']['bits']); S['planted_control'] = dict(planted_bits=float(pl.mean()), kept_after_proxy_regression=share('planted_proxyreg', pl), kept_after_central_regression=share('planted_centralreg', pl),
                                                               expect='proxy regression keeps ~1 (proxy channels do not carry the plant); central regression removes most')
    S['runtime_s'] = round(time.time() - t0); (out / 'summary.json').write_text(json.dumps(S, indent=1)); log(json.dumps(S, indent=1)[:7000])

if __name__ == '__main__': main()
