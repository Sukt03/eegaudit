"""NA-CELL fill (, 23 Nov 2026), NEW script; imports fit_all / MEEG / g0_equivalence / boot from gf_pe_eegnet_floor.py UNCHANGED.
Question (Nuisance column, PE EEGNet ds003458): does the EEGNet yield survive when it is TRAINED and SCORED only on eye-quiet trials?
Quiet trial = b1_eyequiet primary rule (vertical-EOG peak-to-peak <= participant median + 3 MAD, cached results/ext/eyequiet_v3/ptp2_sub-XXX.npz).
Conditions (each ONE batched leave-one-participant-out EEGNet, all 23 members, recipe of gf_pe_eegnet_floor / deep_decoders):
  quiet        rows = quiet trials of every participant (training pool, validation and held-out alike)
  randK        size-matched uniform random subset of ALL trials per participant (same n as quiet, time order kept), K = 0,1,2 (b1's stratified comparator is NOT used here)
  quiet_shuf   quiet rows with the input permuted within participant (labels fixed) = floor
Score: density_gain on the subset (target = locked PE of the subset trials, 10 contiguous folds over the subset rows), as b1 and the floor script.
kept = mean over participants of quiet gain / mean over participants of the mean random gain (participants with >= 100 quiet trials), participant bootstrap 10000.
Gates: G0 (batched == braindecode) and G1 (saved-prediction gain reproduces log_yield_replication to 1e-10) as in the floor script; floor |quiet_shuf| per participant <= 0.02.
Usage: --out DIR [--conds quiet,quiet_shuf | rand0,rand1,rand2] [--epochs 30] [--smoke]"""
import hashlib as _hl
CODE_SHA256 = _hl.sha256(open(__file__, 'rb').read()).hexdigest()
import argparse, json, sys, time
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, '/path/to/eegaudit/experiments/ext')
import gf_pe_eegnet_floor as G
from gf_pe_eegnet_floor import DD, density_gain, folds, targets, ROOT, DEV, log, boot

ap = argparse.ArgumentParser(); ap.add_argument('--out', required=True); ap.add_argument('--conds', default='quiet,quiet_shuf,rand0,rand1,rand2')
ap.add_argument('--epochs', type=int, default=30); ap.add_argument('--smoke', action='store_true'); ap.add_argument('--seed', type=int, default=0)
a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
data, n_ch = DD.load_all(); allsub = list(range(1, 24))
if a.smoke: allsub = [1, 2, 3]; data = data[:3]
nP = len(data); evaluate, ids, _, _ = targets()
gate = {'code_sha256': CODE_SHA256, 'G0_batched_vs_braindecode_max_abs_diff': G.g0_equivalence(n_ch)}; gate['G0_pass'] = bool(max(gate['G0_batched_vs_braindecode_max_abs_diff'].values()) < 1e-5)
rep = {r['subject']: r for r in json.loads((ROOT / 'outputs/AGENT_LOOP_20261120/results/log_yield_replication.json').read_text())['datasets']['ds003458']}
gd = []
for s in allsub:
    z = np.load(ROOT / f'results/features/eegnet/sub-{s:03d}.npz'); tri = z['tri']; assert (tri == data[s - 1][2]).all(); yv = evaluate(s, tri)
    gd.append(abs(density_gain(yv, z['F'].astype(float), folds(len(yv))) - rep[s]['gain']))
gate['G1_saved_gain_max_abs_diff'] = max(gd); gate['G1_pass'] = bool(max(gd) < 1e-10)
# quiet masks (b1 primary rule)
quiet = {}; nq = {}
for s in allsub:
    z = np.load(ROOT / f'results/ext/eyequiet_v3/ptp2_sub-{s:03d}.npz'); me = dict(zip(z['tri'].tolist(), z['eog'].tolist())); tri = data[s - 1][2]
    has = np.array([t in me for t in tri]); E = np.array([me.get(t, np.inf) for t in tri]); med = float(np.median(E[has])); mad = float(np.median(np.abs(E[has] - med)))
    quiet[s] = has & (E <= med + 3 * mad); nq[s] = int(quiet[s].sum())
gate['n_quiet'] = nq; gate['min_quiet'] = min(nq.values()); log('G0/G1', gate['G0_pass'], gate['G1_pass'], 'quiet counts', nq)
(out / 'gate.json').write_text(json.dumps(gate, indent=1))
if not (gate['G0_pass'] and gate['G1_pass']): raise SystemExit('GATE FAIL')
X0 = torch.tensor(np.concatenate([d[0] for d in data]).reshape(sum(len(d[1]) for d in data), n_ch, 400), device=DEV)
y0 = torch.tensor(np.concatenate([d[1] for d in data]), device=DEV); off = np.cumsum([0] + [len(d[1]) for d in data])


def rows_for(c):
    """per participant: indices (into that participant's trials) of the rows used in condition c"""
    if c.startswith('quiet'): return {s: np.where(quiet[s])[0] for s in allsub}
    k = int(c[4:]); return {s: np.sort(np.random.default_rng(1000 * k + s).choice(len(data[s - 1][1]), nq[s], replace=False)) for s in allsub}


for c in a.conds.split(','):
    f = out / f'cond_{c}.json'
    if f.exists(): continue
    R = rows_for(c); gi = torch.tensor(np.concatenate([off[s - 1] + R[s] for s in allsub]), device=DEV)
    sid = torch.tensor(np.concatenate([[s - 1] * len(R[s]) for s in allsub]), device=DEV); X = X0[gi]; y = y0[gi]
    if c == 'quiet_shuf':
        g = torch.Generator(device=DEV); g.manual_seed(3000 + a.seed); X = X.clone(); X0s = X.clone()
        for k in range(nP):
            r = torch.where(sid == k)[0]; X[r] = X0s[r[torch.randperm(len(r), device=DEV, generator=g)]]
    log(f'cond {c}: rows {len(y)}'); ts = time.time()
    preds, bv, be = G.fit_all(X, y, sid, nP, a.epochs, a.seed + 1000 * (1 + len(list(out.glob('cond_*.json')))))
    rows = {}
    for s in allsub:
        tri = data[s - 1][2][R[s]]; yv = evaluate(s, tri); p = preds[s - 1].astype(float); assert len(p) == len(yv)
        rows[str(s)] = dict(gain=density_gain(yv, p[:, None], folds(len(yv))), n=len(yv), best_val_r=float(bv[s - 1]), best_epoch=int(be[s - 1]))
    f.write_text(json.dumps(dict(rows=rows, seconds=time.time() - ts), indent=1)); log(f'cond {c}: mean gain {np.mean([r["gain"] for r in rows.values()]):+.4f} ({time.time() - ts:.0f}s)')
    del X, y; torch.cuda.empty_cache()
# summary (whatever conditions exist)
C = {p.stem[5:]: json.loads(p.read_text()) for p in out.glob('cond_*.json')}
S = {'NOT_CITABLE': 'until independent review', 'code_sha256': CODE_SHA256, 'gate': gate, 'unit': 'bits per feedback trial', 'bits': {}}
use = [s for s in allsub if nq[s] >= 100]; S['n_used'] = len(use)
for c, r in C.items(): S['bits'][c] = boot([r['rows'][str(s)]['gain'] for s in allsub])
if 'quiet_shuf' in C: S['floor_max_abs_participant'] = float(max(abs(C['quiet_shuf']['rows'][str(s)]['gain']) for s in allsub)); S['floor_ok_le_0.02'] = S['floor_max_abs_participant'] <= 0.02
rc = [c for c in C if c.startswith('rand')]
if 'quiet' in C and rc:
    q = torch.tensor([C['quiet']['rows'][str(s)]['gain'] for s in use], device=DEV, dtype=torch.float64)
    r = torch.tensor([np.mean([C[c]['rows'][str(s)]['gain'] for c in rc]) for s in use], device=DEV, dtype=torch.float64)
    g = torch.Generator(device=DEV); g.manual_seed(0); ix = torch.randint(len(use), (10000, len(use)), device=DEV, generator=g)
    k = q[ix].mean(1) / r[ix].mean(1); S['kept_ratio_of_means'] = {'value': float(q.mean() / r.mean()), 'ci95': [float(k.quantile(.025)), float(k.quantile(.975))], 'n_rand': len(rc), 'quiet_mean': float(q.mean()), 'rand_mean': float(r.mean())}
(out / 'summary.json').write_text(json.dumps(S, indent=1)); log(json.dumps({k: v for k, v in S.items() if k != 'gate'}, indent=1)[:2500])
