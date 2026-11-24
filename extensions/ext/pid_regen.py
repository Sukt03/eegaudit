"""Regenerate the planted_v2 data of gf_planted_calib_v2.py exactly (same rng stream) and expose the true generator pieces (, 23 Nov 2026).
Generator: F' = F + a*y*u + b*sig*v, sig = RHO*y + sqrt(1-RHO^2)*e; a = a0*sqrt(1-f), b = b0*sqrt(f) (u, v = random patterns in PCs 20-51 of the participant's z-scored features)."""
import os, sys, numpy as np
ROOT = '/path/to/eegaudit'; os.chdir(ROOT); sys.path[:0] = [ROOT + '/experiments/ext', ROOT]
import d2_pe_tasks as D2, pe_eogreg as PE
KPC = 32; PC0 = 20; A0 = 0.14; B0 = 0.45; LS = (64, 'full')

def load(s, rho, sd):
    fz = np.load(f'{ROOT}/results/features/raw/sub-{s:03d}.npz'); tri0 = fz['tri']; F0 = fz['F'].astype(np.float64)
    O, Ev, keep = PE.ocular_v2(s, tri0); F = F0[keep]; n = len(F); O = O.astype(np.float64)
    Fsd = F.std(0) + 1e-9; rng = np.random.default_rng(910000 + 100 * s + sd); y = rng.standard_normal(n)
    Fz = (F - F.mean(0)) / Fsd; _, Sv, Vt = np.linalg.svd(Fz, full_matrices=False); skd = Sv[PC0:PC0 + KPC] / np.sqrt(n)
    pat = lambda: ((rng.standard_normal(KPC) * skd) @ Vt[PC0:PC0 + KPC]) * Fsd
    u = pat(); sg = rng.choice([-1.0, 1.0], O.shape[1]); v = pat(); sig = rho * y + np.sqrt(1 - rho ** 2) * rng.standard_normal(n)
    # PC basis of the z-scored features (top 64) for the oracle; Sigma = diag(lam)
    lam = (Sv[:64] ** 2) / n; Pb = Vt[:64]
    return dict(F=F, y=y, u=u, v=v, sig=sig, n=n, lam=lam, uz=Pb @ (u / Fsd), vz=Pb @ (v / Fsd))

def amp(f): return (0.0, 0.0) if f == 'null' else (A0 * np.sqrt(1 - f), B0 * np.sqrt(f))

def yield_of(F, y):
    Ys = np.column_stack([y] + [np.roll(y, round(len(y) * q / 4)) for q in (1, 2, 3)])
    sc = D2.pe_gain(F, Ys, LS)
    return {str(L): float(sc[L][:, 0].mean() - sc[L][:, 1:].mean()) for L in LS}
