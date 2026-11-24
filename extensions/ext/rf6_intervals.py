"""RUNNER FIX 6 (round_20261123_auto; METHODS B7): BCa and t intervals beside the percentile bootstrap for the n <= 11 rows, and the paired
EOG-regressed minus raw contrast on BCI-IV-2a. Aggregation only (no model is refitted); torch on the GPU.
Rows: image n = 10 (UBP k80 / k40 yield, untrained-encoder ridge and its share, k40 usage; NICE-GA / BrainFLORA untrained and share),
      BCI-IV-2a n = 8 (every arm x direction; paired eeg_eogreg - eeg per direction and for the direction mean; eog_only),
      text seeds (text_card_v2 summary_over_seeds: every model, greedy/pooled and within-task; eeg yield, eeg - noise, eeg - random-init).
Intervals: percentile bootstrap (10000 resamples of the unit, GPU), BCa (bias z0 from the bootstrap, acceleration from the jackknife), Student t (n - 1 d.f.).
For ratios the statistic is the ratio of means (resampling participants jointly) AND the mean of per-participant ratios (as the paper's share).
GATES: (a) NICE-GA untrained k80 percentile CI == image_untrained/summary.json (same generator call; 1e-4); (b) text t intervals == saved seed_t_ci95 (1e-4);
(c) BCI-IV-2a arm means == saved summary means (1e-4; its saved CIs come from a shared CPU numpy stream and are compared within Monte Carlo error only).
Implementation check: coverage simulation (4000 simulated samples x 2000 resamples) for n = 5, 8, 10 under normal and log-normal data."""
import argparse, glob, hashlib, json, math
from pathlib import Path
import numpy as np, torch
ROOT = Path('/path/to/eegaudit'); DEV = 'cuda'; CODE_SHA256 = hashlib.sha256(open(__file__, 'rb').read()).hexdigest()
from scipy.stats import t as tdist                                               # scalar quantile lookup only
ND = torch.distributions.Normal(torch.tensor(0., dtype=torch.float64, device=DEV), torch.tensor(1., dtype=torch.float64, device=DEV)); ZQ = torch.tensor([-1.959963984540054, 1.959963984540054], dtype=torch.float64, device=DEV)

def iv(X, f=None, B=10000, seed=0):
    """X [n] or [n, p] unit-level values; f maps [..., n, p] -> [...] (default: mean of column 0). Percentile, BCa, t (t only for the plain mean)."""
    X = torch.as_tensor(np.asarray(X, np.float64), device=DEV); X = X[:, None] if X.dim() == 1 else X; n = len(X); plain = f is None; f = f or (lambda Z: Z[..., 0].mean(-1))
    g = torch.Generator(device=DEV); g.manual_seed(seed); ix = torch.randint(n, (B, n), device=DEV, generator=g); th = f(X); bs = f(X[ix]); bs = bs[torch.isfinite(bs)]
    o = dict(est=float(th), n=n, percentile95=[float(bs.quantile(.025)), float(bs.quantile(.975))])
    z0 = ND.icdf(((bs < th).double().mean() + 0.5 * (bs == th).double().mean()).clamp(1e-9, 1 - 1e-9)); keep = ~torch.eye(n, dtype=torch.bool, device=DEV); jk = f(X[None].expand(n, -1, -1)[keep].reshape(n, n - 1, -1)); d = jk.mean() - jk
    acc = (d ** 3).sum() / (6 * (d ** 2).sum() ** 1.5) if float((d ** 2).sum()) > 0 else torch.tensor(0., dtype=torch.float64, device=DEV)
    al = ND.cdf(z0 + (z0 + ZQ) / (1 - acc * (z0 + ZQ))); o['bca95'] = [float(bs.quantile(al[0])), float(bs.quantile(al[1]))]; o['bca_z0'] = float(z0); o['bca_acc'] = float(acc)
    if plain: h = float(tdist.ppf(.975, n - 1)) * float(X[:, 0].std(unbiased=True)) / math.sqrt(n); o['t95'] = [o['est'] - h, o['est'] + h]
    return o
rom = lambda Z: Z[..., 0].mean(-1) / Z[..., 1].mean(-1)                           # ratio of means

def coverage():
    out = {}; g = torch.Generator(device=DEV); g.manual_seed(123); S, Bs = 4000, 2000
    for n in (5, 8, 10):
        for dist in ('normal', 'lognormal'):
            z = torch.randn(S, n, generator=g, device=DEV, dtype=torch.float64); X = z if dist == 'normal' else torch.exp(z); mu = 0.0 if dist == 'normal' else math.exp(0.5)
            ix = torch.randint(n, (Bs, n), device=DEV, generator=g); bs = X[:, ix].mean(-1); th = X.mean(1)                    # [S, Bs]
            lo, hi = bs.quantile(.025, dim=1), bs.quantile(.975, dim=1); cp = ((lo <= mu) & (mu <= hi)).double().mean()
            z0 = ND.icdf((bs < th[:, None]).double().mean(1).clamp(1e-9, 1 - 1e-9)); jk = (X.sum(1, keepdim=True) - X) / (n - 1); d = jk.mean(1, keepdim=True) - jk; acc = (d ** 3).sum(1) / (6 * (d ** 2).sum(1) ** 1.5)
            al = ND.cdf(z0[:, None] + (z0[:, None] + ZQ) / (1 - acc[:, None] * (z0[:, None] + ZQ))); srt = bs.sort(1).values; pos = (al * (Bs - 1)).round().long().clamp(0, Bs - 1); bl, bh = srt.gather(1, pos[:, :1])[:, 0], srt.gather(1, pos[:, 1:])[:, 0]
            cb = ((bl <= mu) & (mu <= bh)).double().mean(); h = float(tdist.ppf(.975, n - 1)) * X.std(1, unbiased=True) / math.sqrt(n); ct = ((th - h <= mu) & (mu <= th + h)).double().mean()
            out[f'n{n}|{dist}'] = dict(percentile=float(cp), bca=float(cb), t=float(ct))
    return out

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out', required=True); a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    S = dict(NOT_CITABLE='until independent review', code_sha256=CODE_SHA256, method=__doc__, coverage_simulation=coverage(), image={}, bci2a={}, text={}, gates={})
    c = S['coverage_simulation']; S['gates']['coverage_t_normal_within_0.94_0.96'] = bool(all(0.94 <= c[f'n{n}|normal']['t'] <= 0.96 for n in (5, 8, 10)))
    # ---------------- image, n = 10
    U = [json.loads((ROOT / f'results/baselines/ubp/ubp_card_soft12_sub{s}.json').read_text())['subjects'][str(s)] for s in range(1, 11)]
    k80 = [u['released']['k80_bits'] for u in U]; k40 = [u['released']['k40_bits'] for u in U]; un = [u['random_init_ridge']['k80_bits'] for u in U]; fr = [u['released']['fraction_k40'] for u in U]; ce = [u['released']['ceiling_k40_bits'] for u in U]
    I = S['image']; I['ubp_k80_yield'] = iv(k80); I['ubp_k40_yield'] = iv(k40); I['ubp_untrained_ridge_k80'] = iv(un); I['ubp_untrained_share_k80_mean_of_ratios'] = iv(np.array(un) / np.array(k80))
    I['ubp_untrained_share_k80_ratio_of_means'] = iv(np.c_[un, k80], rom); I['ubp_usage_k40_mean_of_ratios'] = iv(fr); I['ubp_usage_k40_ratio_of_means'] = iv(np.c_[k40, ce], rom)
    V = [json.loads(f.read_text()) for f in sorted((ROOT / 'results/gpu_fill_20261123/image_untrained').glob('sub-*.json'))]
    for m in ('nice_ga', 'brainflora'):
        u = [v['models'][m]['k80'] for v in V]; t = [v['trained_saved'][m]['k80'] for v in V]; I[f'{m}_untrained_ridge_k80'] = iv(u); I[f'{m}_trained_k80'] = iv(t); I[f'{m}_untrained_share_k80_mean_of_ratios'] = iv(np.array(u) / np.array(t)); I[f'{m}_untrained_share_k80_ratio_of_means'] = iv(np.c_[u, t], rom)
    sv = json.loads((ROOT / 'results/gpu_fill_20261123/image_untrained/summary.json').read_text())['nice_ga']['k80']['ci95']; d = max(abs(x - y) for x, y in zip(sv, I['nice_ga_untrained_ridge_k80']['percentile95']))
    S['gates']['image_percentile_vs_saved'] = dict(saved=sv, this=I['nice_ga_untrained_ridge_k80']['percentile95'], max_abs_diff=d, passed=bool(d < 1e-4))
    # ---------------- BCI-IV-2a, n = 8
    Bc = [json.loads(open(f).read()) for f in sorted(glob.glob(str(ROOT / 'results/ext/bci2a_retest/A*.json')))]; svb = json.loads((ROOT / 'results/ext/bci2a_retest/summary.json').read_text()); gd = []
    S['bci2a']['participants'] = [b['sub'] for b in Bc]
    for arm in ('eeg', 'eeg_eogreg', 'eog_only', 'noise', 'shuffled'):
        for dr in ('T_to_E', 'E_to_T'):
            v = [b[dr][arm] for b in Bc]; S['bci2a'][f'{arm}|{dr}'] = iv(v); gd.append(abs(np.mean(v) - svb[arm][dr]['mean'])); S['bci2a'][f'{arm}|{dr}']['saved_percentile95'] = svb[arm][dr]['ci95']
        S['bci2a'][f'{arm}|direction_mean'] = iv([(b['T_to_E'][arm] + b['E_to_T'][arm]) / 2 for b in Bc])
    for x, y in (('eeg_eogreg', 'eeg'), ('eeg', 'noise'), ('eeg_eogreg', 'noise'), ('eog_only', 'noise'), ('eeg', 'eog_only')):
        for dr in ('T_to_E', 'E_to_T'):
            dd = [b[dr][x] - b[dr][y] for b in Bc]; S['bci2a'][f'paired_{x}_minus_{y}|{dr}'] = dict(iv(dd), n_pos=int(sum(v > 0 for v in dd)), per_participant=dd)
        dd = [(b['T_to_E'][x] + b['E_to_T'][x] - b['T_to_E'][y] - b['E_to_T'][y]) / 2 for b in Bc]; S['bci2a'][f'paired_{x}_minus_{y}|direction_mean'] = dict(iv(dd), n_pos=int(sum(v > 0 for v in dd)), per_participant=dd)
    S['gates']['bci2a_means_vs_saved'] = dict(max_abs_diff=float(max(gd)), passed=bool(max(gd) < 1e-4))
    # ---------------- text seeds
    T = json.loads((ROOT / 'results/baselines/text_card_v2/text_yield_card.json').read_text())['summary_over_seeds']; gt = []
    for m, d in T.items():
        for cell, q in d.items():
            if not (isinstance(q, dict) and cell.startswith(('greedy/', 'beam/')) and 'bleu' not in cell): continue
            for key in ('eeg_yield_bits', 'eeg_minus_noise', 'eeg_minus_random_init', 'noise_yield_bits', 'random_init_yield_bits'):
                e = q.get(key)
                if not (isinstance(e, dict) and e.get('per_seed') and len(e['per_seed']) >= 3): continue
                r = iv(e['per_seed']); r['saved_seed_t_ci95'] = e.get('seed_t_ci95'); S['text'][f'{m}|{cell}|{key}'] = r
                if e.get('seed_t_ci95'): gt.append(max(abs(x - y) for x, y in zip(r['t95'], e['seed_t_ci95'])))
    S['gates']['text_t_vs_saved'] = dict(n_cells=len(gt), max_abs_diff=float(max(gt)) if gt else None, passed=bool(gt and max(gt) < 1e-4))
    (out / 'summary.json').write_text(json.dumps(S, indent=1))
    f3 = lambda x: [round(v, 3) for v in x]
    print('coverage', json.dumps(S['coverage_simulation'])); print('gates', json.dumps(S['gates']))
    for sec in ('image', 'bci2a', 'text'):
        for k, r in S[sec].items():
            if isinstance(r, dict) and 'est' in r: print(sec, k, 'n', r['n'], 'est', round(r['est'], 4), 'pct', f3(r['percentile95']), 'bca', f3(r['bca95']), 't', f3(r['t95']) if 't95' in r else None, 'npos', r.get('n_pos'))

if __name__ == '__main__': main()
