"""P6 (PE): nuisance-partitioned information. Partial correlation of the EEGNet and REVE LOPO read-outs (z-scored) with normal-scores PE given
nuisances Z = (outcome valence = reward sign, |PE| (normal-scores of |residualised PE|), previous-trial PE): residualise both read-out and target on
[1, Z] within participant, rho_partial = corr of residuals; bits_partial = -1/2 log2(1 - rho_partial^2); unconditioned bits_g for comparison;
floor = the same partial with the read-out circularly shifted (>= 40 trials, 20 shifts, 95th percentile of the shift distribution per participant);
also the single-nuisance versions (valence only; |PE| only; previous PE only) and the share explained = 1 - bits_partial / bits_g. Output results/loop/p6/p6_pe_partial.json."""
import sys, json, numpy as np
from pathlib import Path
from scipy import stats
ROOT = Path('/path/to/eegaudit'); sys.path.insert(0, str(ROOT)); rng = np.random.default_rng(0)
import deep_decoders as DD
data, _ = DD.load_all(); Q = np.load(ROOT / DD.QUANT)
nscore = lambda y: stats.norm.ppf((stats.rankdata(y) - 0.5) / len(y)); b = lambda r: float(-0.5 * np.log2(1 - min(abs(r), 0.999) ** 2)); z = lambda v: (v - v.mean()) / (v.std() + 1e-12)
def resid(v, Z): A = np.c_[np.ones(len(v)), Z]; w, *_ = np.linalg.lstsq(A, v, rcond=None); return v - A @ w
out = {}
for dec in ['eegnet', 'reve_lopo']:
    rows = []
    for i, (X, yt, tri) in enumerate(data):
        y = z(np.load(ROOT / f'results/features/{dec}/sub-{i + 1:03d}.npz')['F'][:, 0]); t = nscore(yt)
        val = (Q['reward'][i][tri] > 0).astype(float); apef = nscore(np.abs(yt)); prev = np.r_[0.0, yt[:-1]]  # previous EEG-trial PE (trial order within the EEG set)
        nuis = {'valence': val[:, None], 'abs_pe': apef[:, None], 'prev_pe': prev[:, None], 'all': np.c_[val, apef, prev]}
        r = dict(subject=i + 1, bits_g=b(np.corrcoef(y, t)[0, 1]), rho_g=float(np.corrcoef(y, t)[0, 1]))
        for k, Z in nuis.items():
            ry, rt = resid(y, Z), resid(t, Z); rp = float(np.corrcoef(ry, rt)[0, 1]); n = len(y)
            fl = [abs(np.corrcoef(resid(np.roll(y, s), Z), rt)[0, 1]) for s in rng.integers(40, n - 40, 20)]
            r[k] = dict(rho_partial=rp, bits_partial=b(rp), floor_p95_bits=b(np.percentile(fl, 95)), above_floor=bool(b(rp) > b(np.percentile(fl, 95))), share_explained=float(1 - b(rp) / max(r['bits_g'], 1e-9)))
        rows.append(r)
    S = {k: dict(bits_partial_mean=float(np.mean([r[k]['bits_partial'] for r in rows])), bits_partial_ci95=[float(np.percentile([np.mean([rows[j][k]['bits_partial'] for j in rng.integers(0, 23, 23)]) for _ in range(4000)], q)) for q in (2.5, 97.5)],
                 n_above_floor=int(sum(r[k]['above_floor'] for r in rows)), share_explained_mean=float(np.mean([r[k]['share_explained'] for r in rows])), share_explained_median=float(np.median([r[k]['share_explained'] for r in rows]))) for k in ('valence', 'abs_pe', 'prev_pe', 'all')}
    out[dec] = dict(rows=rows, bits_g_mean=float(np.mean([r['bits_g'] for r in rows])), summary=S)
    print(dec, 'bits_g', round(out[dec]['bits_g_mean'], 4), {k: (round(v['bits_partial_mean'], 4), v['n_above_floor'], round(v['share_explained_mean'], 3)) for k, v in S.items()})
json.dump(out, open(ROOT / 'results/loop/p6/p6_pe_partial.json', 'w'), indent=1)
