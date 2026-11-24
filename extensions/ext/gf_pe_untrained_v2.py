"""GPU-FILL v2 PE UNTRAINED-ENCODER baseline, ds003458 (, 23 Nov 2026; new script, imports the helpers of gf_pe_eegnet_floor.py unchanged).
Encoder = braindecode EEGNet (n_chans=common, n_times=400) at its library random initialisation, never trained, eval mode with default BatchNorm statistics, frozen; S independent
inits as one batched MEEG (G0-checked equal to braindecode). Features = 192-d pre-classifier map, then the SAME leave-one-participant-out protocol as the trained refit:
validation participant (h*7)%22 of the rest, ridge read-out (alpha grid 1e-2..1e6) chosen by held-out correlation on the validation participant, features standardised on the training rows,
same target (common_log_yield.targets), scored with density_gain on the same 10 contiguous folds. Floor: init 0 features with rows permuted within participant (labels fixed).
Gates: G0 (batched == braindecode), G1 (saved-prediction gain reproduces log_yield_replication.json to 1e-10, target to 1e-4) as in the floors script; floor <= 0.02 (cohort mean).
Usage: --out DIR --seeds S [--subs 1,2,..]"""
import hashlib as _hl
CODE_SHA256 = _hl.sha256(open(__file__, 'rb').read()).hexdigest()
import argparse, json, sys, time
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, '/path/to/eegaudit/experiments/ext')
import gf_pe_eegnet_floor as G
from gf_pe_eegnet_floor import DEV, log, ROOT, DD, MEEG, EEGNet, density_gain, folds, targets, boot

def features(net, X, S, chunk=256):
    out = []
    with torch.no_grad():
        for e0 in range(0, len(X), chunk):
            x = X[e0:e0 + chunk][:, None].expand(-1, S, -1, -1).contiguous()
            w = torch.renorm(net.sp, p=2, dim=0, maxnorm=1.0)
            z = F.conv2d(net.bn1(net.c1(x)), w, groups=net.M * 8)
            z = F.avg_pool2d(F.elu(net.bn2(z)), (1, 4)); z = F.avg_pool2d(F.elu(net.bn3(net.c4(net.c3(z)))), (1, 8))
            out.append(z.reshape(len(x), S, -1).double())
    return torch.cat(out)                                     # [N, S, 192]

AL = 10.0 ** torch.arange(-2, 7, dtype=torch.float64)
def ridge_lopo(Fk, y, sid, nP):
    """Fk [N, d] float64 GPU. Returns held-out predictions per participant (np) and chosen alpha."""
    preds, als = [], []
    for h in range(nP):
        rest = [k for k in range(nP) if k != h]; v = rest[(h * 7) % len(rest)]
        tr = torch.cat([torch.where(sid == k)[0] for k in rest if k != v]); va = torch.where(sid == v)[0]; te = torch.where(sid == h)[0]
        mu, sd = Fk[tr].mean(0), Fk[tr].std(0) + 1e-8; A = (Fk[tr] - mu) / sd; yt = y[tr].double(); ym = yt.mean()
        w, U = torch.linalg.eigh(A.T @ A); b = U.T @ (A.T @ (yt - ym)); Av = (Fk[va] - mu) / sd; At = (Fk[te] - mu) / sd
        best, ba = -9, None
        for al in AL.to(DEV):
            beta = U @ (b / (w + al)); pv = Av @ beta
            r = float(torch.nan_to_num(torch.corrcoef(torch.stack([pv, y[va].double()]))[0, 1], nan=-1.0)) if pv.std() > 0 else -1.0
            if r > best: best, ba, bb = r, float(al), beta
        preds.append((At @ bb + ym).cpu().numpy()); als.append(ba)
    return preds, als

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out', required=True); ap.add_argument('--seeds', type=int, default=10)
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    data, n_ch = DD.load_all(); allsub = list(range(1, 24)); evaluate, ids, _, _ = targets()
    gate = {'G0_batched_vs_braindecode_max_abs_diff': G.g0_equivalence(n_ch)}; gate['G0_pass'] = bool(max(gate['G0_batched_vs_braindecode_max_abs_diff'].values()) < 1e-5)
    rep = {r['subject']: r for r in json.loads((ROOT / 'outputs/AGENT_LOOP_20261120/results/log_yield_replication.json').read_text())['datasets']['ds003458']}
    gd, ty = [], []
    for s in allsub:
        z = np.load(ROOT / f'results/features/eegnet/sub-{s:03d}.npz'); tri = z['tri']; yv = evaluate(s, tri)
        ty.append(float(np.abs(yv - data[s - 1][1]).max())); assert (tri == data[s - 1][2]).all()
        gd.append(abs(density_gain(yv, z['F'].astype(float), folds(len(yv))) - rep[s]['gain']))
    gate.update(G1_saved_gain_max_abs_diff=max(gd), G1_target_max_abs_diff=max(ty), G1_pass=bool(max(gd) < 1e-10 and max(ty) < 1e-4)); log('gate', gate)
    (out / 'gate.json').write_text(json.dumps(dict(code_sha256=CODE_SHA256, **gate), indent=1))
    if not (gate['G0_pass'] and gate['G1_pass']): raise SystemExit('GATE FAIL')
    X0 = torch.tensor(np.concatenate([d[0] for d in data]).reshape(sum(len(d[1]) for d in data), n_ch, 400), device=DEV)
    y = torch.tensor(np.concatenate([d[1] for d in data]), device=DEV); nP = len(data)
    sid = torch.tensor(np.concatenate([[k] * len(d[1]) for k, d in enumerate(data)]), device=DEV)
    S = a.seeds; torch.manual_seed(4343); nets = [EEGNet(n_chans=n_ch, n_outputs=1, n_times=400) for _ in range(S)]
    net = MEEG(nets, n_ch).to(DEV).eval(); Fe = features(net, X0, S); log('features', tuple(Fe.shape)); del net, X0; torch.cuda.empty_cache()
    gen = torch.Generator(device=DEV).manual_seed(55); pm = torch.arange(len(y), device=DEV)
    for k in range(nP):
        r = torch.where(sid == k)[0]; pm[r] = r[torch.randperm(len(r), device=DEV, generator=gen)]
    conds = {f'rand{k}': Fe[:, k] for k in range(S)}; conds['shuffle_rand0'] = Fe[:, 0][pm]; res = {}
    for c, Fk in conds.items():
        pr, als = ridge_lopo(Fk, y, sid, nP); rows = {}
        for s in allsub:
            yv = evaluate(s, data[s - 1][2]); rows[str(s)] = dict(gain=density_gain(yv, pr[s - 1].astype(float)[:, None], folds(len(yv))), alpha=als[s - 1])
        res[c] = rows; log(c, 'mean gain', float(np.mean([r['gain'] for r in rows.values()])))
    g = {c: [res[c][str(s)]['gain'] for s in allsub] for c in res}; seedmean = np.mean([g[f'rand{k}'] for k in range(S)], 0); sv = np.array([rep[s]['gain'] for s in allsub])
    Sm = dict(NOT_CITABLE='until independent review', code_sha256=CODE_SHA256, n_participants=23, seeds=S, gate=gate, bootstrap='participant, 10000 resamples, GPU torch, seed 0',
              unit='bits per feedback trial (density_gain, 10 contiguous folds)', bits={c: boot(v) for c, v in g.items()}, untrained_seedmean=boot(seedmean),
              per_seed_cohort_means=[float(np.mean(g[f'rand{k}'])) for k in range(S)], saved_trained_eegnet=boot(sv), trained_minus_untrained=boot(sv - seedmean),
              shuffle_floor_cohort_ok=bool(abs(np.mean(g['shuffle_rand0'])) <= 0.02), shuffle_floor_max_abs_participant=float(np.max(np.abs(g['shuffle_rand0']))),
              untrained_over_trained_ratio=float(seedmean.mean() / sv.mean()))
    (out / 'per_participant.json').write_text(json.dumps(res)); (out / 'summary.json').write_text(json.dumps(Sm, indent=1))
    log(json.dumps({k: v for k, v in Sm.items() if k != 'gate'}, indent=1)[:2500])

if __name__ == '__main__': main()
