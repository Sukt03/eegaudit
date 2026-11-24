"""GPU-FILL v2 MI foundation-model MEAN-SPECTRUM surrogate and within-participant LABEL-PERMUTATION floors (, 23 Nov 2026; new script derived from gf_mi_fm_phase.py).
Why: the phase surrogate keeps each trial's own amplitude spectrum, so trial-wise band power (the cue-dependent mu/beta power the floors review flagged) survives; here every trial of a participant gets that participant's MEAN per-channel amplitude spectrum with fresh random phase (conds meanspec0, meanspec1: no trial-wise power, no phase, no label information in the input), and permlab0/1 keep the real input but permute the labels within participant (the read-out floor). Same input / 105 participants / 10 groups / read-out as below.
ORIGINAL DOCSTRING (phase floor):
Frozen CBraMod / REVE (label: disjoint) and LaBraM / EEGPT (label: TRANSDUCTIVE, pretrained on EEGMMIDB), each with its random-init twin, on the EOG-proxy-regressed
MI card input of mi_fm_eogreg.py (imported unchanged: load_model, extract, score, SPEC, ELIG, the same 105 participants / 10 groups / read-out), with the input epoch
replaced by a per-trial, per-channel INDEPENDENT phase randomisation (amplitude spectrum kept, hence still 1-40 Hz band-limited; DC/Nyquist real) of the regressed
1-40 Hz epoch BEFORE resampling and cropping; REVE per-participant z-stats are recomputed from the perturbed input. Conditions: phase0, phase1 (fm and twin arms).
G1: the UNPERTURBED regressed input, fm arm, reproduces the saved results/ext/mi_fm_eogreg fm|eeg bits per participant to 1e-4 (all 105; skipped for subset runs).
G2: per-participant seed-mean |bits| <= 0.02 for every (model, arm). Output: <out>/<model>/{sub-XXX.json, summary.json}, <out>/gate_<models>.json.
Usage: --out DIR --models cbramod,reve [--subjects 1,2] [--seeds 0,1] [--no_g1]"""
import hashlib as _hl
CODE_SHA256 = _hl.sha256(open(__file__, 'rb').read()).hexdigest()
import argparse, json, math, sys, time
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, '/path/to/eegaudit/experiments/ext')
import mi_fm_eogreg as FM
from mi_fm_eogreg import ROOT, DEV, log, load_model, extract, score, SPEC, ELIG
from mi_card_v3 import load_sub, bandpass, eog_regress, W0, W1


def phase_rand(X, g, chunk=1000):                          # X CPU [N, 64, 960] -> CPU
    o = torch.empty_like(X); n = X.shape[-1]
    for i in range(0, len(X), chunk):
        x = X[i:i + chunk].to(DEV); Fq = torch.fft.rfft(x, dim=-1); ph = torch.rand(Fq.shape, device=DEV, generator=g) * 2 * math.pi
        ph[..., 0] = 0
        if n % 2 == 0: ph[..., -1] = 0
        o[i:i + chunk] = torch.fft.irfft(Fq.abs() * torch.exp(1j * ph), n=n, dim=-1).cpu()
    return o


def meanspec_rand(X, sidc, g, chunk=500):                  # X CPU [N, 64, 960]; per participant and channel MEAN amplitude spectrum, independent random phase per trial and channel
    o = torch.empty_like(X); n = X.shape[-1]
    for k in torch.unique(sidc).tolist():
        rows = torch.where(sidc == k)[0]; A = torch.zeros(X.shape[1], n // 2 + 1, device=DEV, dtype=torch.float64)
        for i in range(0, len(rows), chunk): A += torch.fft.rfft(X[rows[i:i + chunk]].to(DEV), dim=-1).abs().double().sum(0)
        A = (A / len(rows)).float()
        for i in range(0, len(rows), chunk):
            r = rows[i:i + chunk]; ph = torch.rand((len(r),) + tuple(A.shape), device=DEV, generator=g) * 2 * math.pi; ph[..., 0] = 0
            if n % 2 == 0: ph[..., -1] = 0
            o[r] = torch.fft.irfft(A[None] * torch.exp(1j * ph), n=n, dim=-1).cpu()
    return o


def boot(v, B=10000):
    v = torch.tensor(np.asarray(v, np.float64), device=DEV); g = torch.Generator(device=DEV); g.manual_seed(0)
    m = v[torch.randint(len(v), (B, len(v)), device=DEV, generator=g)].mean(1)
    return {'mean': float(v.mean()), 'ci95': [float(m.quantile(0.025)), float(m.quantile(0.975))]}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out', required=True); ap.add_argument('--models', default='cbramod,reve,labram,eegpt'); ap.add_argument('--subjects', default='')
    ap.add_argument('--seeds', default='0,1'); ap.add_argument('--no_g1', action='store_true'); ap.add_argument('--raw_dir', default=str(ROOT / 'results/ext/mi_fm_eogreg'))
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True); models = a.models.split(','); seeds = [int(x) for x in a.seeds.split(',')]
    cache = ROOT / 'results/ext/mi_card_cache'; subs = [int(x) for x in a.subjects.split(',')] if a.subjects else list(range(1, 110)); D = {}
    for s in subs:
        d, why = load_sub(s, cache)
        if d is None: continue
        c = np.bincount(d['y'], minlength=4)
        if c.max() - c.min() > 2: continue
        D[s] = d
    P = sorted(D); nP = len(P); ch = D[P[0]]['ch']; log(nP, 'participants')
    X = torch.tensor(np.concatenate([D[s]['X'] for s in P]), dtype=torch.float32)
    y = torch.tensor(np.concatenate([D[s]['y'] for s in P]), device=DEV)
    sid = torch.tensor(np.concatenate([[k] * len(D[s]['y']) for k, s in enumerate(P)]), device=DEV); N = len(y); sidc = sid.cpu()
    Xb = torch.empty_like(X)
    for i in range(0, N, 1500): Xb[i:i + 1500] = bandpass(X[i:i + 1500].to(DEV), 1, 40).cpu()
    del X
    Xb, share = eog_regress(Xb, sidc, nP, ch); torch.cuda.empty_cache()          # same spec-B residual as mi_fm_eogreg
    inputs = {'real': Xb}
    for sd in seeds:
        g = torch.Generator(device=DEV); g.manual_seed(1000 + sd); inputs[f'meanspec{sd}'] = meanspec_rand(Xb, sidc, g)
    pw = {c: float(inputs[c].pow(2).mean() / Xb.pow(2).mean()) for c in inputs if c != 'real'}; log('power ratios', pw)
    G = min(10, nP); grp = torch.tensor([k % G for k in range(nP)], device=DEV)[sid]
    fits, units, pca_rows = [], [], {}
    for gg in range(G):
        v = (gg + 1) % G; outer = torch.where(grp != gg)[0]; pca_rows[gg] = outer
        fits.append((torch.where((grp != gg) & (grp != v))[0], torch.where(grp == v)[0], gg)); fits.append((outer, torch.where(grp == gg)[0], gg))
        units.append(dict(inner=[len(fits) - 2], final=len(fits) - 1))
    ypm = {}
    for sd in seeds:
        gp = torch.Generator(); gp.manual_seed(5000 + sd); yy = y.clone(); sc_ = sidc
        for k in range(nP):
            r = torch.where(sc_ == k)[0]; yy[r] = y[r[torch.randperm(len(r), generator=gp)]]
        ypm[sd] = yy
    zst = {}
    for nm, XX in inputs.items():
        mu = torch.stack([XX[sidc == k][:, :, W0:W1].mean((0, 2)) for k in range(nP)]).to(DEV)
        sd_ = torch.stack([XX[sidc == k][:, :, W0:W1].std((0, 2)) for k in range(nP)]).to(DEV) + 1e-6; zst[nm] = (mu, sd_)
    gates = {}
    for mname in models:
        mdir = out / mname; mdir.mkdir(parents=True, exist_ok=True); per = [dict(participant=s) for s in P]; info = {}; diag = {}
        for arm, pre in (('fm', True), ('twin', False)):
            for cond, XX in inputs.items():
                if cond == 'real' and (a.no_g1 or nP != 105) and not seeds: continue
                F, inf = extract(mname, pre, XX, sid, zst[cond], ch); info[f'{arm}|{cond}'] = inf
                runs = [(cond, y)] + ([(f'permlab{sd}', ypm[sd]) for sd in seeds] if cond == 'real' else [])
                for cn, yy in runs:
                    if cn == 'real' and arm == 'twin' and (a.no_g1 or nP != 105): continue
                    pp, dg = score(F, yy, sid, nP, fits, units, pca_rows); diag[f'{arm}|{cn}'] = dg
                    for k in range(nP): per[k].setdefault(arm, {})[cn] = pp[k]
                    log(f'{mname} {arm} {cn}: mean bits {np.mean([q["bits"] for q in pp]):+.4f} acc {np.mean([q["acc"] for q in pp]):.3f}')
                del F; torch.cuda.empty_cache()
        for k, s in enumerate(P): (mdir / f'sub-{s:03d}.json').write_text(json.dumps(per[k], indent=1))
        S = {'NOT_CITABLE': 'until independent review', 'code_sha256': CODE_SHA256, 'model': mname, 'label': ELIG[mname][0], 'n_participants': nP, 'seeds': seeds,
             'input': 'EOG-proxy-regressed 1-40 Hz epoch (mi_card_v3 spec B); meanspec = participant-mean per-channel amplitude spectrum with random phase per trial; permlab = real input, labels permuted within participant', 'power_ratio': pw,
             'bootstrap': 'participant, 10000 resamples, GPU torch, seed 0', 'unit': 'bits per trial', 'mi_fm_eogreg_sha256': _hl.sha256((ROOT / 'experiments/ext/mi_fm_eogreg.py').read_bytes()).hexdigest(), 'rows': {}}
        for arm in ('fm', 'twin'):
            for fam in ('meanspec', 'permlab'):
                cs = [f'{fam}{sd}' for sd in seeds]
                for c in cs: S['rows'][f'{arm}|{c}'] = boot([per[k][arm][c]['bits'] for k in range(nP)])
                m = [float(np.mean([per[k][arm][c]['bits'] for c in cs])) for k in range(nP)]; S['rows'][f'{arm}|{fam}_seedmean'] = boot(m)
                S[f'{arm}_{fam}_max_abs_per_participant_seedmean'] = float(np.max(np.abs(m))); S[f'{arm}_{fam}_floor_ok_le_0.02'] = bool(np.max(np.abs(m)) <= 0.02); S[f'{arm}_{fam}_n_participants_above_0.02'] = int((np.abs(m) > 0.02).sum())
        if 'real' in per[0].get('fm', {}):
            rd = Path(a.raw_dir) / mname; sv = [json.loads((rd / f'sub-{s:03d}.json').read_text())['fm']['eeg']['bits'] for s in P]
            d = [abs(per[k]['fm']['real']['bits'] - sv[k]) for k in range(nP)]; S['G1_max_abs_diff'] = max(d); S['G1_pass'] = bool(max(d) < 1e-4)
            S['rows']['fm|real_refit'] = boot([per[k]['fm']['real']['bits'] for k in range(nP)]); S['rows']['fm|saved_mi_fm_eogreg'] = boot(sv)
        else: S['G1_pass'] = None
        (mdir / 'summary.json').write_text(json.dumps(S, indent=1)); gates[mname] = {k: S.get(k) for k in ('label', 'G1_pass', 'G1_max_abs_diff', 'fm_meanspec_floor_ok_le_0.02', 'twin_meanspec_floor_ok_le_0.02', 'fm_permlab_floor_ok_le_0.02', 'twin_permlab_floor_ok_le_0.02')}
        log(mname, json.dumps(S['rows'], indent=1)[:1800], gates[mname])
    (out / f'gate_{"_".join(models)}.json').write_text(json.dumps(dict(code_sha256=CODE_SHA256, models=gates), indent=1))


if __name__ == '__main__':
    main()
