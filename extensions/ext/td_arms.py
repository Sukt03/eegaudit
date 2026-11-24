"""td_arms.py: ATTEMPT B (exploratory, declared after the pre-registered attempt A had been read; NOT part of the pre-registered verdict).
Question: does D1 improve when the arm yields A1 (in-fold ocular regression) and A4 (target-adjusted regression), both functions of the mixed cell, y and the measured eye regressors, are ADDED as inputs
(the ideator's optional ablation, T0 screen S2)? This script regenerates every cell of the stored shards deterministically (same rng streams, same recipe parameters), scores A1 and A4 on it (A4 by the
per-column rule) and stores them in results/td_pilot/cells_arms/<kind>/ with the SAME file names and row order. Alignment gate: the META rows must equal the stored shard's META rows, and the
joint yield recomputed on every 4th cell must equal the stored joint yield (1e-8); on the six F1 cells the new A1 / A4 must equal the stored BL (1e-8).
Usage: td_arms.py --kinds train | val,cal,test,r4test [--limit N]"""
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np, torch
sys.path[:0] = ['/path/to/eegaudit/experiments/ext']
import td_cells as TC
from td_cells import *      # noqa

ARMS = OUT / 'cells_arms'


def save(path, BL, extra):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True); tmp = path.with_name(f'tmp{os.getpid()}_{path.name}')
    with open(tmp, 'wb') as fh: np.savez(fh, BL=np.array(BL, np.float64), **extra)
    os.replace(tmp, path)


def arms_of(Fm, y, Om):
    a1 = yld(readout(Fm, y, PE.eogreg(Om))); a4 = a4_yield(Fm, y, Om); return [a1['64'], a1['full'], a4['64'], a4['full']]


def do_train(kind, s, chunk, nper):
    name = f'{kind}_sub{s:03d}_c{chunk:02d}.npz'; path = ARMS / kind / name
    if path.exists(): return
    ref = np.load(TC.CELLS / kind / name); P = pb(s); fc = 100 if kind == 'train' else 101; rt = np.random.default_rng([fc, s, chunk, 0]); BL = []; t1 = time.time(); worst = 0.0
    for i in range(nper):
        th = draw_theta_train(rt); rd = np.random.default_rng([fc, s, chunk, 1 + i]); D = draw_data(P, th, rd)
        mr = np.array(meta_row(th, s, chunk, i, th['a'], th['b'], th['rho'], th['kappa'], float('nan'), False), float)
        assert np.allclose(mr, ref['META'][i], equal_nan=True), ('META mismatch', name, i)
        Fn, Fo, Fm, Om = assemble(D, th['a'], th['b'], th['rho'], th['kappa']); BL.append(arms_of(Fm, D['y'], Om))
        if i % 4 == 0: j = yld(readout(Fm, D['y']))['64']; worst = max(worst, abs(j - ref['Y'][i, 4])); assert worst < 1e-8, ('joint yield mismatch', name, i, worst)
    save(path, BL, dict(joint_check_max=np.array(worst))); log(f'arms {kind} sub {s} chunk {chunk}: {nper} cells, {time.time() - t1:.0f}s, joint check {worst:.1e}')


def do_family(kind, fam, s, seed):
    name = f'{fam}_sub{s:03d}_seed{seed}.npz'; path = ARMS / kind / name
    if path.exists(): return
    ref = np.load(TC.CELLS / kind / name); P = pb(s); th = fam_theta(fam); t1 = time.time(); BL = []; worst = 0.0; worst_bl = 0.0
    rng = np.random.default_rng(910000 + 100 * s + seed) if fam == 'R0' else np.random.default_rng([200 + FAMCODE[fam], s, seed]); D = draw_data(P, th, rng)
    for ci, (nm, rho, a, b, f) in enumerate(grid_cells()):
        r = 0.5 if rho is None else rho; thc = dict(th); thc['rho'] = r; thc['a'] = a; thc['b'] = b; thc['f'] = f if f == f else -1.0
        has = (kind == 'test') and rho in (0.25, 0.5) and nm in ('f0.25', 'f0.5', 'f0.75')
        mr = np.array(meta_row(thc, s, seed, ci, a, b, r, KAPPA0, thc['f'], has), float); assert np.allclose(mr, ref['META'][ci], equal_nan=True), ('META mismatch', name, ci)
        Fn, Fo, Fm, Om = assemble(D, a, b, r, KAPPA0); bl = arms_of(Fm, D['y'], Om); BL.append(bl)
        if ci % 4 == 0: j = yld(readout(Fm, D['y']))['64']; worst = max(worst, abs(j - ref['Y'][ci, 4])); assert worst < 1e-8, ('joint yield mismatch', name, ci, worst)
        if has: worst_bl = max(worst_bl, float(np.abs(np.array(bl) - ref['BL'][ci]).max())); assert worst_bl < 1e-8, ('A1/A4 differ from the stored baselines', name, ci, worst_bl)
    save(path, BL, dict(joint_check_max=np.array(worst), bl_check_max=np.array(worst_bl))); log(f'arms {kind} {fam} sub {s} seed {seed}: 14 cells, {time.time() - t1:.0f}s, joint check {worst:.1e}, stored-BL check {worst_bl:.1e}')


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--kinds', required=True); ap.add_argument('--nper', type=int, default=25); ap.add_argument('--train-chunks', type=int, default=5); ap.add_argument('--val-chunks', type=int, default=4); a = ap.parse_args()
    ledger('start:arms:' + a.kinds)
    try:
        for kind in a.kinds.split(','):
            if kind == 'val':
                for s in VAL:
                    for c in range(a.val_chunks): do_train('val', s, c, a.nper)
            elif kind == 'cal':
                for s in VAL:
                    for sd in (200, 201): do_family('cal', 'R4', s, sd)
            elif kind == 'train':
                for c in range(a.train_chunks):
                    for s in TRAIN: do_train('train', s, c, a.nper)
            elif kind == 'test':
                for fam, seeds in (('R0', (0, 1)), ('R2', (100, 101)), ('R3', (100, 101))):
                    for sd in seeds:
                        for s in TEST: do_family('test', fam, s, sd)
            elif kind == 'r4test':
                for s in TEST: do_family('r4test', 'R4', s, 202)
    finally: ledger('end:arms:' + a.kinds)


if __name__ == '__main__':
    main()
