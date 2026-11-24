"""Exploratory A1: fixed-target temporal embargo, three-participant smoke pilot.

Run only inside an existing authorized A100 node. No source/result registry writes.
Historical whole-participant target residualization and ranking are retained.
"""
import argparse
import ast
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy import stats

ROOT = Path('/path/to/eegaudit')
sys.path.insert(0, str(ROOT))
from belief_bottleneck import LAM, folds
from group_stats import corr, resid, QUANT
from cogmodels import models as M


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def make_targets():
    # Extract only the previously inspected target function, never script main.
    path = ROOT / 'experiments/loop/s13b_gauss_bits.py'
    tree = ast.parse(path.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'targets_ds003458')
    ns = dict(ROOT=ROOT, QUANT=QUANT, np=np, torch=torch, M=M, resid=resid)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), ns)
    return ns['targets_ds003458']()


def partitions(tri, subject, gap):
    out = {}
    fold = folds(len(tri))
    for b in range(10):
        te = np.flatnonzero(fold == b)
        tr = np.flatnonzero(fold != b)
        far = np.min(np.abs(tri[tr, None] - tri[te][None, :]), axis=1) > gap
        embargo = tr[far]
        rng = np.random.default_rng(20261120 + 1000 * subject + b)
        matched = np.sort(rng.choice(tr, size=len(embargo), replace=False))
        assert len(embargo) == len(matched)
        assert len(np.intersect1d(embargo, te)) == 0
        assert np.all(np.min(np.abs(tri[embargo, None] - tri[te][None, :]), axis=1) > gap)
        out[b] = dict(test=te, gap0=tr, embargo=embargo, matched=matched)
    return out


def predict(F, Y, colw, parts, arm):
    ft = torch.as_tensor(F, dtype=torch.float64, device='cuda')
    yt = torch.as_tensor(Y, dtype=torch.float64, device='cuda')
    cw = torch.as_tensor(colw, dtype=torch.float64, device='cuda')
    p = torch.zeros_like(yt)
    diagnostics = {}
    for b, part in parts.items():
        tr = torch.as_tensor(part[arm], device='cuda')
        te = torch.as_tensor(part['test'], device='cuda')
        mu, sd = ft[tr].mean(0), ft[tr].std(0, unbiased=False) + 1e-6
        a, z = (ft[tr] - mu) / sd * cw, (ft[te] - mu) / sd * cw
        _, singular, vt = torch.linalg.svd(a - a.mean(0), full_matrices=False)
        rank = int((singular > singular[0] * max(a.shape) * torch.finfo(a.dtype).eps).sum().item())
        assert rank >= 64, f'PCA rank mismatch in {arm} fold {b}: {rank}'
        vk = vt[:min(64, len(vt))].T
        diagnostics[str(b)] = dict(rank=rank, effective_width=int(vk.shape[1]))
        a, z = a @ vk, z @ vk
        w = torch.linalg.solve(a.T @ a + LAM * len(a) * torch.eye(a.shape[1], dtype=a.dtype, device='cuda'), a.T @ (yt[tr] - yt[tr].mean(0)))
        p[te] = z @ w
    result = p.cpu().numpy()
    assert np.isfinite(result).all(), 'nonfinite predictions'
    return result, diagnostics


def summarize(p, yg, me):
    rg = np.array([corr(p[:, j], yg[:, j]) for j in range(yg.shape[1])])
    null = np.delete(rg, me)
    assert np.isfinite(rg).all(), 'nonfinite correlations'
    d = float(np.arctanh(np.clip(rg[me], -.999, .999)) - np.arctanh(np.clip(null, -.999, .999)).mean())
    return dict(r_g=float(rg[me]), d=d, null_r_g=null.tolist(), p95=float(np.quantile(null, .95)), above_p95=bool(rg[me] > np.quantile(null, .95)), yield_value=float(-.5 * np.log2(1 - np.tanh(max(d, 0)) ** 2)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--subs', default='1,2,3')
    ap.add_argument('--gap', type=int, default=10)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    assert torch.cuda.is_available(), 'A100 execution required; no CPU fallback'
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    torch.set_num_threads(4)
    qtil, allsubs, fpath, _ = make_targets()
    source_paths = [ROOT / QUANT, ROOT / 'results/map_RWD.npy', ROOT / 'belief_bottleneck.py', ROOT / 'group_stats.py', ROOT / 'experiments/loop/s13b_gauss_bits.py', Path(__file__)]
    base_hashes = {str(p): digest(p) for p in source_paths}
    for s in [int(x) for x in args.subs.split(',')]:
        data = np.load(fpath('raw', s), allow_pickle=False)
        f, cw, tri = data['F'].astype(float), data['colw'].astype(float), data['tri']
        assert np.isfinite(f).all() and np.isfinite(cw).all()
        assert np.all(np.diff(tri) > 0), 'original trial IDs must be strictly increasing'
        y = np.column_stack([qtil(j, tri) for j in allsubs])
        yg = np.column_stack([stats.norm.ppf((stats.rankdata(y[:, j]) - .5) / len(y)) for j in range(y.shape[1])])
        parts = partitions(tri, s, args.gap)
        fn = np.random.default_rng(20261120 + s).standard_normal(f.shape)
        rows = {}
        for arm in ['gap0', 'embargo', 'matched']:
            rows[arm] = {}
            for condition, features in [('eeg', f), ('noise', fn)]:
                pred, diagnostics = predict(features, y, cw, parts, arm)
                rows[arm][condition] = summarize(pred, yg, allsubs.index(s))
                rows[arm][condition]['pca_folds'] = diagnostics
                np.save(out / f'sub-{s:03d}_{arm}_{condition}_predictions.npy', pred)
                print(s, arm, condition, rows[arm][condition], flush=True)
        reference = ROOT / f'results/yield/pe_bottleneck_ds003458/raw_sub-{s:03d}.json'
        prior = json.loads(reference.read_text())
        prior_row = next(row for row in prior['rows'] if row['L'] == 64 and row['condition'] == 'eeg')
        baseline_error = abs(rows['gap0']['eeg']['r_g'] - prior_row['r_g'])
        assert np.array_equal(np.sort(np.concatenate([p['test'] for p in parts.values()])), np.arange(len(tri)))
        record = dict(subject=s, pilot=True, n_participants_planned=3, selection='first three participant IDs, exploratory; historical results already exposed', gap=args.gap, estimator='signed null-corrected Fisher d primary; nonnegative Gaussianised yield secondary', target_limitation='whole-participant nuisance residualization and normal-score ranks retained; no leakage elimination claim', rows=rows, folds={str(b): {k: tri[v].tolist() for k,v in p.items()} for b,p in parts.items()}, fold_counts={str(b):{k:len(v) for k,v in p.items()} for b,p in parts.items()}, training_removal_seed_rule='20261120 + 1000*subject + fold', noise_seed=20261120+s, pca_rank_gate='every fold/arm/features rank >=64; stop on mismatch', baseline_reference=str(reference), baseline_reference_sha256=digest(reference), baseline_r_g_error=baseline_error, baseline_d_error=abs(rows['gap0']['eeg']['d']-prior_row['d']), baseline_tolerance_r_g=1e-5, baseline_reproduced=bool(baseline_error <= 1e-5), hashes={**base_hashes, str(fpath('raw', s)):digest(fpath('raw', s))}, versions=dict(numpy=np.__version__, torch=torch.__version__, cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0)), runtime_s=time.monotonic()-t0)
        (out / f'sub-{s:03d}.json').write_text(json.dumps(record, indent=2))
        if baseline_error > 1e-5 or any(rows[a]['noise']['yield_value'] > .02 for a in rows):
            raise RuntimeError('Technical gate failed: stop, do not proceed to remaining participants')


if __name__ == '__main__':
    main()
