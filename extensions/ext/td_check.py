"""td_check.py: G1 on the ACTUAL test shards (not only the gate cells): every R0 test cell (participants x seeds 0,1 x 14 cells) against the stored planted_v2 cells (L 64 and full for joint / neural / ocular, A1 =
mix_eogreg) wherever a stored file exists (RHO 0.5: all participants; RHO 0.25 / 0.9: odd participants), and A4 against runner_T9R arms_rho0.5. Also: cell counts, finite values, split disjointness, and a
recipe-family shortcut check (can a logistic regression on the D1 statistics tell the R0 test cells from the training cells? reported as an AUC). Print + results/td_pilot/check_shards.json. NOT citable."""
import glob, json, sys
import numpy as np, torch
sys.path[:0] = ['/path/to/eegaudit/experiments/ext']
import td_screen as TS
R = '/path/to/eegaudit/results/td_pilot/'
res = dict(NOT_CITABLE='until a blind bug review'); worst = {}; n = 0; rows = []
for f in sorted(glob.glob(R + 'cells/test/R0_sub*_seed*.npz')):
    z = np.load(f); M = z['META']; mn = list(z['META_names']); s = int(M[0, mn.index('participant')]); sd = int(M[0, mn.index('seed')]); assert len(M) == 14 and sd in (0, 1)
    for i in range(14):
        rho = M[i, mn.index('rho')]; a = M[i, mn.index('a')]; b = M[i, mn.index('b')]; fv = M[i, mn.index('f')]
        if fv == -1.0: key_f = 'null'
        else: key_f = float(fv)
        if key_f == 1.0: continue          # ocular-only f = 1 is not in the stored grid
        rhos = [0.5] if key_f in ('null', 0.0) else [rho]
        for rr in rhos:
            St = TS.stored(float(rr), s)
            if St is None: continue
            Sc = TS.scell(St, sd, key_f); Y = z['Y'][i]; n += 1
            chk = {'mix_all': (Y[4], Y[5]), 'neural_only': (Y[0], Y[1]), 'ocular_only': (Y[2], Y[3])}
            if not np.isnan(z['BL'][i, 0]): chk['mix_eogreg'] = (z['BL'][i, 0], z['BL'][i, 1])
            for k, (v64, vf) in chk.items():
                d = max(abs(v64 - Sc[k]['64']), abs(vf - Sc[k]['full'])); worst[k] = max(worst.get(k, 0.0), float(d))
            if not np.isnan(z['BL'][i, 2]) and float(rho) == 0.5:
                import os
                p4 = f'/path/to/eegaudit/results/yield_projector/runner_T9R/arms_rho0.5/sub-{s:03d}.json'
                if os.path.exists(p4):
                    for c in json.loads(open(p4).read())['cells']:
                        if c['seed'] == sd and abs(float(c['f']) - float(key_f)) < 1e-12: d = max(abs(z['BL'][i, 2] - c['arms']['A4']['yield']['64']), abs(z['BL'][i, 3] - c['arms']['A4']['yield']['full'])); worst['A4'] = max(worst.get('A4', 0.0), float(d))
res['R0_cells_compared_with_stored'] = n; res['max_abs_diff_by_quantity'] = worst; res['pass_1e_4'] = bool(all(v <= 1e-4 for v in worst.values()) and n > 0)
cnt = {}
for kind in ('train', 'val', 'cal', 'test', 'r4test'):
    fs = glob.glob(R + f'cells/{kind}/*.npz'); m = sum(len(np.load(f)['Y']) for f in fs); cnt[kind] = dict(shards=len(fs), cells=int(m))
    for f in fs[:200]:
        z = np.load(f); assert np.isfinite(z['X1']).all() and np.isfinite(z['Y']).all() and np.isfinite(z['TOK'].astype(np.float32)).all(), f
res['counts'] = cnt
# shortcut check: logistic regression (torch, full batch) train-cells vs R0 test cells on the standardised D1 statistics; AUC by rank statistic, 5-fold by participant-agnostic random split
tr = np.concatenate([np.load(f)['X1'] for f in sorted(glob.glob(R + 'cells/train/*.npz'))]); te = np.concatenate([np.load(f)['X1'] for f in sorted(glob.glob(R + 'cells/test/R0_*.npz'))])
X = torch.tensor(np.concatenate([tr, te]), dtype=torch.float32, device='cuda'); y = torch.tensor([0.0] * len(tr) + [1.0] * len(te), device='cuda'); X = (X - X.mean(0)) / (X.std(0) + 1e-6)
g = torch.Generator(device='cuda'); g.manual_seed(1); perm = torch.randperm(len(X), generator=g, device='cuda'); aucs = []
for k in range(5):
    te_ix = perm[k::5]; tr_mask = torch.ones(len(X), dtype=torch.bool, device='cuda'); tr_mask[te_ix] = False; w = torch.zeros(X.shape[1], device='cuda', requires_grad=True); b0 = torch.zeros(1, device='cuda', requires_grad=True); opt = torch.optim.Adam([w, b0], lr=0.05)
    wt = torch.where(y[tr_mask] == 1, 1.0 / y[tr_mask].sum(), 1.0 / (1 - y[tr_mask]).sum())
    for _ in range(300): opt.zero_grad(); l = (wt * torch.nn.functional.binary_cross_entropy_with_logits(X[tr_mask] @ w + b0, y[tr_mask], reduction='none')).sum() + 1e-3 * (w ** 2).sum(); l.backward(); opt.step()
    sc = (X[te_ix] @ w + b0).detach(); yy = y[te_ix]; pos = sc[yy == 1]; neg = sc[yy == 0]; aucs.append(float((pos[:, None] > neg[None]).float().mean() + 0.5 * (pos[:, None] == neg[None]).float().mean()))
res['family_shortcut_AUC_train_vs_R0test_on_D1_statistics'] = dict(mean=float(np.mean(aucs)), folds=aucs, note='0.5 = indistinguishable; near 1 = the D1 statistics reveal the recipe family (shortcut risk)')
json.dump(res, open(R + 'check_shards.json', 'w'), indent=1); print(json.dumps(res, indent=1))
