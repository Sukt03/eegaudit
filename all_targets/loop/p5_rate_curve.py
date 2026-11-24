"""P5: information-rate curve, R90 and used fraction u = I(R90)/R90.
Image (embedding side): results/eegbits/budget_embedding_sub-XX.json rows part='quant', condition='eeg_quant' (L dims x b bits per dim = budget R bits;
yield = Wolpaw bits at k = 80; the files carry no soft bits) and the float32 full-L reference; per participant R90 = smallest R reaching 90 % of the
participant's maximum yield over the grid (linear interpolation on the monotone envelope), u = I(R90)/R90.
PE: EEGNet and REVE LOPO read-outs quantised per participant to b in {1,2,3,4,6,8} bits (uniform over the participant's read-out range), rho_g vs
normal-scores(PE) recomputed (uncorrected, as P3), bits_g; R = b bits per trial; R90 and u as above. Output results/loop/p5/p5_rate_curve.json."""
import sys, json, glob, numpy as np
from pathlib import Path
from scipy import stats
ROOT = Path('/path/to/eegaudit'); sys.path.insert(0, str(ROOT)); rng = np.random.default_rng(0)
def r90(R, I):
    R, I = np.asarray(R, float), np.asarray(I, float); o = np.argsort(R); R, I = R[o], I[o]; env = np.maximum.accumulate(I); tgt = 0.9 * env.max()
    j = int(np.argmax(env >= tgt)); 
    if j == 0: return float(R[0]), float(env[0])
    x0, x1, y0, y1 = R[j - 1], R[j], env[j - 1], env[j]; r = x0 + (tgt - y0) / (y1 - y0) * (x1 - x0) if y1 > y0 else x1; return float(r), float(tgt)
out = {'image': {}, 'pe': {}}
rows_img = []
for f in sorted(glob.glob(str(ROOT / 'results/eegbits/budget_embedding_sub-*.json'))):
    d = json.load(open(f)); q = [r for r in d['rows'] if r.get('part') == 'quant' and r.get('condition') == 'eeg_quant' and r.get('k') == 80]
    if not q: continue
    R = [r['budget_bits'] for r in q]; I = [r['bits'] for r in q]; rr, i90 = r90(R, I)
    full = [r for r in d['rows'] if r.get('part') == 'quant' and r.get('condition') == 'eeg_float32' and r.get('k') == 80]; imax = max(I)
    rows_img.append(dict(subject=d['subject'], R90_bits=rr, I_at_R90=i90, u=i90 / rr, I_max_quant=imax, I_full_float32_maxL=max([r['bits'] for r in full]) if full else None, grid={str(r['budget_bits']): r['bits'] for r in q}))
    print(f"image {d['subject']}: R90 {rr:.0f} bits -> I {i90:.3f}, u {i90/rr:.4f} (max quant {imax:.3f})")
A = {k: np.array([r[k] for r in rows_img]) for k in ('R90_bits', 'u', 'I_at_R90')}
out['image'] = dict(n=len(rows_img), per_participant=rows_img, R90_median=float(np.median(A['R90_bits'])), R90_iqr=[float(np.percentile(A['R90_bits'], 25)), float(np.percentile(A['R90_bits'], 75))], u_mean=float(A['u'].mean()), u_median=float(np.median(A['u'])), unit='Wolpaw bits at k = 80 per 200-way query vs quantised embedding budget in bits')
import deep_decoders as DD
data, _ = DD.load_all(); nscore = lambda y: stats.norm.ppf((stats.rankdata(y) - 0.5) / len(y)); b_ = lambda r: float(-0.5 * np.log2(1 - min(abs(r), 0.999) ** 2))
for dec in ['eegnet', 'reve_lopo']:
    rows = []
    for i, (X, yt, tri) in enumerate(data):
        y = np.load(ROOT / f'results/features/{dec}/sub-{i + 1:03d}.npz')['F'][:, 0]; xg = nscore(yt); full = b_(np.corrcoef(y, xg)[0, 1]); grid = {}
        for bb in (1, 2, 3, 4, 6, 8):
            edges = np.linspace(y.min(), y.max(), 2 ** bb + 1); q = np.clip(np.digitize(y, edges[1:-1]), 0, 2 ** bb - 1).astype(float)
            grid[bb] = b_(np.corrcoef(q, xg)[0, 1]) if q.std() > 0 else 0.0
        rr, i90 = r90(list(grid), list(grid.values())); rows.append(dict(subject=i + 1, bits_g_full=full, grid={str(k): v for k, v in grid.items()}, R90_bits=rr, I_at_R90=i90, u=i90 / rr))
    A = {k: np.array([r[k] for r in rows]) for k in ('R90_bits', 'u', 'bits_g_full')}; G = {bb: float(np.mean([r['grid'][str(bb)] for r in rows])) for bb in (1, 2, 3, 4, 6, 8)}
    out['pe'][dec] = dict(n=len(rows), per_participant=rows, mean_bits_g_by_b=G, mean_bits_g_full=float(A['bits_g_full'].mean()), R90_median=float(np.median(A['R90_bits'])), R90_mean=float(A['R90_bits'].mean()), u_mean=float(A['u'].mean()), u_median=float(np.median(A['u'])), unit='Gaussianised bits per epoch (uncorrected) vs read-out quantisation b bits per trial')
    print(dec, 'mean bits_g by b', {k: round(v, 4) for k, v in G.items()}, 'full', round(float(A['bits_g_full'].mean()), 4), 'R90 median', float(np.median(A['R90_bits'])), 'u mean', round(float(A['u'].mean()), 4))
json.dump(out, open(ROOT / 'results/loop/p5/p5_rate_curve.json', 'w'), indent=1)
print('image R90 median', out['image']['R90_median'], 'IQR', out['image']['R90_iqr'], 'u mean', round(out['image']['u_mean'], 5))
