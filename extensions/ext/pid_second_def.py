"""Step 2 of IDEATOR_MIXED_ATTRIBUTION (, 23 Nov 2026). GPU torch.
Second redundancy definitions on the planted_v2 cells (regenerated exactly, gate vs stored yields). dit/BROJA package is NOT installed (offline), so BROJA-2PID is
implemented here in torch (mirror descent + iterative proportional fitting on the discretised held-out read-outs; gated on AND/XOR/COPY/UNQ gates).
Predictors: held-out outer-fold ridge predictions (L=64, d2_pe_tasks.ridge_batch) of y from Fn (neural only), Fo (ocular only), Fm (both). Per participant (mean over 2 seeds of atoms).
Arms: MMI_yield (stored yields), Gauss_MMI (Barrett 2015 MMI on the Gaussian MI from the covariance of y and the predictions), Disc_MMI and BROJA (4 quantile bins of y and of each read-out).
Flag 'redundancy not identified' when |R_BROJA - R_MMI| (discrete, same units) > 0.03 bit."""
import json, sys, time, numpy as np, torch
sys.path.insert(0, '/path/to/eegaudit/experiments/ext')
import pid_regen as G
D2 = G.D2; DEV = 'cuda'; t0 = time.time(); LN2 = float(np.log(2)); NB = 4
log = lambda *m: print(f'[{time.time()-t0:7.1f}s]', *m, flush=True)

def _lg(x): return torch.where(x > 0, torch.log(x.clamp_min(1e-300)), torch.zeros_like(x))
def mi_yx(p):  # p (B,Y,X): I(Y;X) in bits
    py = p.sum(2, keepdim=True); px = p.sum(1, keepdim=True); return (p * (_lg(p) - _lg(py) - _lg(px))).sum((1, 2)) / LN2
def cmi(q):  # q (B,Y,X1,X2): I(Y;X1|X2)
    qx2 = q.sum((1, 2), keepdim=True); qyx2 = q.sum(2, keepdim=True); qx1x2 = q.sum(1, keepdim=True)
    return (q * (_lg(q) + _lg(qx2) - _lg(qyx2) - _lg(qx1x2))).sum((1, 2, 3)) / LN2

def broja(p, iters=4000, eta=0.5, ipf=30):
    """p (B,Y,X1,X2) joint. Returns dict R,U1,U2,S, and objective gap (final vs 3/4 of the iterations). UI_1 = min_q I_q(Y;X1|X2), q has the (Y,X1),(Y,X2) marginals of p."""
    p1 = p.sum(3); p2 = p.sum(2); py = p.sum((2, 3))
    out = {}
    for tag in ('1', '2'):
        pt = p if tag == '1' else p.transpose(2, 3)      # source 2 unique: swap roles of X1, X2
        pa = pt.sum(3); pb = pt.sum(2); q = (pa[:, :, :, None] * pb[:, :, None, :]) / py.clamp_min(1e-300)[:, :, None, None]
        mid = None
        for it in range(iters):
            qx2 = q.sum((1, 2), keepdim=True); qyx2 = q.sum(2, keepdim=True); qx1x2 = q.sum(1, keepdim=True)
            g = _lg(q) + _lg(qx2) - _lg(qyx2) - _lg(qx1x2)
            q = q * torch.exp(-eta * g)
            for _ in range(ipf):
                q = q * (pa / q.sum(3).clamp_min(1e-300))[:, :, :, None]
                q = q * (pb / q.sum(2).clamp_min(1e-300))[:, :, None, :]
            if it == int(0.75 * iters): mid = cmi(q)
        out['U' + tag] = cmi(q); out['gap' + tag] = (mid - out['U' + tag]).abs()
    I1 = mi_yx(p1); I2 = mi_yx(p2); Ij = mi_yx(p.reshape(p.shape[0], p.shape[1], -1))
    R = I1 - out['U1']; S = Ij - I1 - out['U2']
    return dict(R=R, U1=out['U1'], U2=out['U2'], S=S, I1=I1, I2=I2, Ij=Ij, gap=torch.maximum(out['gap1'], out['gap2']))

def mmi(I1, I2, Ij):
    R = torch.minimum(I1, I2); return dict(R=R, U1=I1 - R, U2=I2 - R, S=Ij - I1 - I2 + R)

def gates():
    def mk(f):
        p = torch.zeros(1, 2, 2, 2, dtype=torch.float64, device=DEV)
        for y, a, b in f: p[0, y, a, b] += 1
        return p / p.sum()
    T = [(a, b) for a in (0, 1) for b in (0, 1)]
    cases = {'AND': (mk([(a & b, a, b) for a, b in T]), dict(R=0.3113, U1=0, U2=0, S=0.5)), 'XOR': (mk([(a ^ b, a, b) for a, b in T]), dict(R=0, U1=0, U2=0, S=1.0)),
             'COPY': (mk([(0, 0, 0), (1, 1, 1)]), dict(R=1.0, U1=0, U2=0, S=0)), 'UNQ': (mk([(a, a, b) for a, b in T]), dict(R=0, U1=1.0, U2=0, S=0))}
    res = {}; ok = True
    for k, (p, ex) in cases.items():
        r = broja(p, iters=1500); got = {n: float(r[n][0]) for n in ('R', 'U1', 'U2', 'S')}; res[k] = got
        ok &= all(abs(got[n] - ex[n]) < 0.01 for n in ex)
    return ok, res

def gauss_mi(C, tgt, srcs):  # C (B,k,k) covariance; bits
    s = list(srcs); Cxx = C[:, s][:, :, s]; Cxy = C[:, s, tgt]
    expl = (Cxy[:, None, :] @ torch.linalg.solve(Cxx, Cxy[:, :, None]))[:, 0, 0]; return 0.5 * torch.log2(C[:, tgt, tgt] / (C[:, tgt, tgt] - expl))

def qbin(x, k=NB):  # x (n,) -> quantile bin ids 0..k-1
    r = torch.argsort(torch.argsort(x)); return (r * k // len(x)).long()

def heldout_pred(F, y, M, tests):
    P = D2.ridge_batch(torch.as_tensor(F, dtype=torch.float64, device=DEV), torch.as_tensor(y[:, None], dtype=torch.float64, device=DEV), M[:10], [64])
    out = torch.zeros(len(y), dtype=torch.float64, device=DEV)
    for b in range(10): te = torch.as_tensor(tests[b], device=DEV); out[te] = P[b, 0, te, 0]
    return out

def run_rho(rho, subs, seeds, ROOTD, smoke=False):
    res = {}; S0 = json.load(open(f'{G.ROOT}/results/gpu_fill_20261123/planted_v2_rho{rho}/summary.json'))
    fs = [0.0, 0.25, 0.5, 0.75]; per = {f: [] for f in fs}; gate_err = 0.0
    import os
    for s in subs:
        if not os.path.exists(f'{G.ROOT}/results/gpu_fill_20261123/planted_v2_rho{rho}/sub-{s:03d}.json'): continue   # only participants present in that rho run
        st = json.load(open(f'{G.ROOT}/results/gpu_fill_20261123/planted_v2_rho{rho}/sub-{s:03d}.json'))['cells']
        seed_rows = {f: [] for f in fs}
        for sd in seeds:
            d = G.load(s, float(rho), sd); y = d['y']; n = d['n']; M, tests, _ = D2.nested_masks(n); yt = torch.as_tensor(y, device=DEV)
            for f in fs:
                a, b = G.amp(f); Fn = d['F'] + a * np.outer(y, d['u']); Fo = d['F'] + b * np.outer(d['sig'], d['v']); Fm = Fn + b * np.outer(d['sig'], d['v'])
                if f == 0.5 and s == subs[0] and sd == seeds[0]:   # reproduction gate vs stored yields (same pipeline)
                    st_c = [c for c in st if c['seed'] == sd and c['f'] == f][0]
                    for nm, FF, key in (('N', Fn, 'neural_only'), ('O', Fo, 'ocular_only'), ('NO', Fm, 'mix_all')):
                        gate_err = max(gate_err, abs(G.yield_of(FF, y)['64'] - st_c[key]['64']))
                st_c = [c for c in st if c['seed'] == sd and c['f'] == f][0]
                pN, pO, pNO = [heldout_pred(FF, y, M, tests) for FF in (Fn, Fo, Fm)]
                X = torch.stack([yt, pN, pO, pNO], 1); Xc = X - X.mean(0); C = (Xc.T @ Xc / n)[None]
                gI = {k: gauss_mi(C, 0, sr)[0] for k, sr in (('N', [1]), ('O', [2]), ('NO', [3]), ('J', [1, 2]))}
                gm = mmi(gI['N'][None], gI['O'][None], gI['J'][None])
                # discrete joint over (y, bin(pN), bin(pO)); floor = y circularly shifted by n/2
                def joint(yy):
                    yb = qbin(yy); a1 = qbin(pN); a2 = qbin(pO); p = torch.zeros(NB, NB, NB, dtype=torch.float64, device=DEV)
                    p.index_put_((yb, a1, a2), torch.ones(n, dtype=torch.float64, device=DEV), accumulate=True); return p / n
                pj = torch.stack([joint(yt), joint(torch.roll(yt, n // 2))]); br = broja(pj)
                dm = mmi(br['I1'], br['I2'], br['Ij'])
                row = dict(yield_MMI=dict(zip(('R', 'U_N', 'U_O', 'S'), [float(x) for x in (lambda yn, yo, yno: (min(yn, yo), yn - min(yn, yo), yo - min(yn, yo), yno - yn - yo + min(yn, yo)))(st_c['neural_only']['64'], st_c['ocular_only']['64'], st_c['mix_all']['64'])])),
                           gauss_MMI={k: float(v[0]) for k, v in zip(('R', 'U_N', 'U_O', 'S'), (gm['R'], gm['U1'], gm['U2'], gm['S']))}, gauss_I=dict(N=float(gI['N']), O=float(gI['O']), J=float(gI['J']), NO=float(gI['NO'])),
                           disc_MMI={k: float(v[0]) for k, v in zip(('R', 'U_N', 'U_O', 'S'), (dm['R'], dm['U1'], dm['U2'], dm['S']))}, BROJA={k: float(br[k][0]) for k in ('R', 'U1', 'U2', 'S')},
                           BROJA_floor={k: float(br[k][1]) for k in ('R', 'U1', 'U2', 'S')}, disc_MMI_floor={k: float(v[1]) for k, v in zip(('R', 'U_N', 'U_O', 'S'), (dm['R'], dm['U1'], dm['U2'], dm['S']))},
                           broja_conv_gap=float(br['gap'].max()), disc_I=dict(N=float(br['I1'][0]), O=float(br['I2'][0]), J=float(br['Ij'][0])))
                seed_rows[f].append(row)
        for f in fs:
            r0 = seed_rows[f][0]; per[f].append({k: ({kk: float(np.mean([r[k][kk] for r in seed_rows[f]])) for kk in r0[k]} if isinstance(r0[k], dict) else float(np.max([r[k] for r in seed_rows[f]]))) for k in r0})
        log(f'rho{rho} sub{s} done')
    return per, gate_err

def boot(x, B=10000):
    x = torch.as_tensor(x, dtype=torch.float64, device=DEV); g = torch.Generator(device=DEV); g.manual_seed(0)
    m = x[torch.randint(len(x), (B, len(x)), generator=g, device=DEV)].mean(1); return [float(m.quantile(.025)), float(m.quantile(.975))]

def main():
    out = sys.argv[1]; smoke = len(sys.argv) > 2 and sys.argv[2] == 'smoke'; subs = [1, 2] if smoke else (list(range(int(__import__("os").environ["PID_SUBS"].split("-")[0]), int(__import__("os").environ["PID_SUBS"].split("-")[1]) + 1)) if "PID_SUBS" in __import__("os").environ else list(range(1, 24))); rhos = ['0.9'] if smoke else (sys.argv[3].split(',') if len(sys.argv) > 3 else ['0.25', '0.5', '0.9'])
    ok, gres = gates(); log('BROJA gates', ok, gres)
    if not ok: raise SystemExit('BROJA gate FAIL')
    R = dict(NOT_CITABLE='until independent review', broja_gates=gres, broja_gate_pass=ok, definitions=dict(MMI_yield='min(Y_N,Y_O) on stored yields (bits, log-score gain)', gauss_MMI='Barrett 2015 MMI on Gaussian MI from covariance of held-out predictions (y, yhat_N, yhat_O)',
             disc_MMI='MMI on plug-in MI of 4-quantile-bin discretisation', BROJA='BROJA-2PID (Bertschinger 2014) torch implementation on the same discretisation', package='dit/BROJA_2PID not installed offline; own implementation, gated'), cells={})
    for rho in rhos:
        per, ge = run_rho(rho, subs, [0, 1], out, smoke); R['cells'][rho] = {'repro_gate_max_abs_yield_diff': ge}
        for f, rows in per.items():
            cell = {}
            for arm in ('yield_MMI', 'gauss_MMI', 'disc_MMI', 'BROJA', 'BROJA_floor', 'disc_MMI_floor'):
                kk = ('R', 'U1', 'U2', 'S') if arm.startswith('BROJA') else ('R', 'U_N', 'U_O', 'S') if arm != 'gauss_MMI' else ('R', 'U_N', 'U_O', 'S')
                cell[arm] = {k: dict(mean=float(np.mean([r[arm][k] for r in rows])), ci95=boot([r[arm][k] for r in rows])) for k in kk}
            gap = [r['BROJA']['R'] - r['disc_MMI']['R'] for r in rows]; cell['R_gap_BROJA_minus_discMMI'] = dict(mean=float(np.mean(gap)), ci95=boot(gap))
            Rs = {a: cell[a]['R']['mean'] for a in ('yield_MMI', 'gauss_MMI', 'disc_MMI', 'BROJA')}
            cell['R_range_all_arms'] = [min(Rs.values()), max(Rs.values())]; cell['R_width_all_arms'] = max(Rs.values()) - min(Rs.values())
            cell['redundancy_not_identified_def_gap'] = bool(abs(cell['R_gap_BROJA_minus_discMMI']['mean']) > 0.03)
            cell['redundancy_not_identified_all_arms'] = bool(cell['R_width_all_arms'] > 0.03); cell['max_broja_conv_gap'] = float(max(r['broja_conv_gap'] for r in rows))
            R['cells'][rho][str(f)] = cell
            print(rho, f, {a: round(v, 3) for a, v in Rs.items()}, 'def_gap', round(cell['R_gap_BROJA_minus_discMMI']['mean'], 3), 'NOT-IDENT' if cell['redundancy_not_identified_def_gap'] else 'ok', flush=True)
    json.dump(R, open(f'{out}/pid_second_def{"_smoke" if smoke else ""}{"_" + "_".join(rhos) if not smoke else ""}{"_subs" + __import__("os").environ["PID_SUBS"] if "PID_SUBS" in __import__("os").environ else ""}.json', 'w'), indent=1); log('done')

if __name__ == '__main__': main()
