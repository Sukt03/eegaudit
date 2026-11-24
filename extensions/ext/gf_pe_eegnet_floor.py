"""GPU-FILL PE EEGNet input-perturbation floors, ds003458 (, 22 Nov 2026).
The saved PE EEGNet read-outs (results/features/eegnet, deep_decoders.py) cannot take perturbed input, so the leave-one-participant-out EEGNet is REFIT
with the recipe of deep_decoders.train_fold: braindecode EEGNet (n_chans=common channels, n_times=400, 1 output), AdamW lr 1e-3 wd 1e-2, MSE, batch 128,
30 epochs, early stopping on the correlation on one validation participant ((h*7) % 22 of the rest), seed rule torch.manual_seed(seed + s), same
folds, same target (locked PE: results of common_log_yield/targets()), scored with common_log_yield.density_gain on the same 10 contiguous folds.
All 23 folds of one input condition are ONE batched GPU model (grouped convs, per-member BatchNorm / AdamW state / early stopping; members are initialised by
copying braindecode EEGNet instances, so the initialisation law is the library's). Deviations (documented): members draw 128 rows per step from their
own epoch permutation and every member runs ceil(max n_train / 128) steps per epoch (members with fewer rows cycle slightly more than one pass);
the held-out prediction is kept at the best validation epoch (same as loading the best state).
Conditions (input = the 64-channel 200 Hz epoch, 2 x 200 samples, applied to training, validation AND held-out trials alike):
  real      unperturbed input (refit; sanity against the saved read-out, not an identity gate: the batched model and RNG differ)
  phaseK    per trial, per channel INDEPENDENT phase randomisation of the 400-sample epoch (amplitude spectrum kept), seed K in 0,1,2
  shuffleK  trial order permuted within participant (labels fixed), seed K in 0,1,2
Gates: G0 (equivalence of the batched network to braindecode, eval and train mode, max abs diff < 1e-4); G1 (density_gain on the SAVED predictions reproduces
log_yield_replication.json gains to 1e-10 and the saved target equals the recomputed target to 1e-4); G2 floors: cohort-mean phase / shuffle gain |.| <= 0.02.
Usage: python gf_pe_eegnet_floor.py --out DIR [--subs 1,2,..] [--conds real,phase0,...] [--epochs 30] [--gate_only]"""
import hashlib as _hl
CODE_SHA256 = _hl.sha256(open(__file__, 'rb').read()).hexdigest()
import argparse, json, math, os, sys, time
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
ROOT = Path('/path/to/eegaudit'); os.chdir(ROOT)
sys.path[:0] = [str(ROOT), str(ROOT / 'external/CBraMod'), str(ROOT / 'experiments/agent_loop_20261120')]
import deep_decoders as DD
from common_log_yield import density_gain, folds, targets
from braindecode.models import EEGNet
DEV = 'cuda'; t0 = time.time()
torch.backends.cudnn.allow_tf32 = False; torch.backends.cuda.matmul.allow_tf32 = False   # G0 root cause: TF32 convs gave 2e-4 (eval) / 1e-2 (train) differences
log = lambda *m: print(f'[{time.time()-t0:7.1f}s]', *m, flush=True)


class MEEG(nn.Module):
    """M independent braindecode EEGNets (C channels, T samples, 1 output) as grouped convolutions. Built from a list of braindecode EEGNet instances."""
    def __init__(s, nets, C, T=400, p=0.25):
        super().__init__(); M = len(nets); s.M, s.p = M, p
        s.c1 = nn.Conv2d(M, M * 8, (1, 64), groups=M, padding=(0, 32), bias=False); s.bn1 = nn.BatchNorm2d(M * 8, momentum=0.01, eps=1e-3)
        s.sp = nn.Parameter(torch.cat([n.conv_spatial.parametrizations.weight.original.detach().clone() for n in nets]))   # [M*16,1,C,1]
        s.bn2 = nn.BatchNorm2d(M * 16, momentum=0.01, eps=1e-3)
        s.c3 = nn.Conv2d(M * 16, M * 16, (1, 16), groups=M * 16, padding=(0, 8), bias=False)
        s.c4 = nn.Conv2d(M * 16, M * 16, 1, groups=M, bias=False); s.bn3 = nn.BatchNorm2d(M * 16, momentum=0.01, eps=1e-3)
        s.fc = nn.Conv2d(M * 16, M, (1, 12), groups=M)
        with torch.no_grad():
            s.c1.weight.copy_(torch.cat([n.conv_temporal.weight for n in nets])); s.c3.weight.copy_(torch.cat([n.conv_separable_depth.weight for n in nets]))
            s.c4.weight.copy_(torch.cat([n.conv_separable_point.weight for n in nets]))
            s.fc.weight.copy_(torch.cat([n.final_layer.conv_classifier.weight for n in nets])); s.fc.bias.copy_(torch.cat([n.final_layer.conv_classifier.bias for n in nets]))
            for bn, nm in ((s.bn1, 'bnorm_temporal'), (s.bn2, 'bnorm_1'), (s.bn3, 'bnorm_2')):
                bn.weight.copy_(torch.cat([getattr(n, nm).weight for n in nets])); bn.bias.copy_(torch.cat([getattr(n, nm).bias for n in nets]))
                bn.running_mean.copy_(torch.cat([getattr(n, nm).running_mean for n in nets])); bn.running_var.copy_(torch.cat([getattr(n, nm).running_var for n in nets]))   # braindecode runs a dummy forward at construction (G0 debugging)

    def forward(s, x):                                   # x [B, M, C, T] -> [B, M]
        w = torch.renorm(s.sp, p=2, dim=0, maxnorm=1.0)                      # braindecode max-norm constraint on the spatial filters (per filter)
        z = F.conv2d(s.bn1(s.c1(x)), w, groups=s.M * 8)
        z = F.dropout(F.avg_pool2d(F.elu(s.bn2(z)), (1, 4)), s.p, s.training)
        z = F.dropout(F.avg_pool2d(F.elu(s.bn3(s.c4(s.c3(z)))), (1, 8)), s.p, s.training)
        return s.fc(z).reshape(x.shape[0], s.M)


def g0_equivalence(C):
    """Batched net == braindecode EEGNet per member (eval mode, and train mode with dropout 0 so BatchNorm batch statistics are compared)."""
    torch.manual_seed(0); nets = [EEGNet(n_chans=C, n_outputs=1, n_times=400, drop_prob=0.0).to(DEV) for _ in range(3)]
    m = MEEG(nets, C, p=0.0).to(DEV); x = torch.randn(16, 3, C, 400, device=DEV); out = {}
    for mode in ('eval', 'train'):
        for n in nets: getattr(n, mode)()
        getattr(m, mode)()
        with torch.no_grad():
            yb = m(x); ref = torch.stack([n(x[:, j]).reshape(-1) for j, n in enumerate(nets)], 1)
        out[mode] = float((yb - ref).abs().max() / ref.abs().max())   # relative: eval-mode outputs are large (unnormalised activations), float32 summation order gives ~1e-6
    return out


def phase_rand(X, g, chunk=2000):                         # X [n, C, 400]; independent uniform phase per trial and channel, DC/Nyquist real
    o = torch.empty_like(X); n = X.shape[-1]
    for i in range(0, len(X), chunk):
        Fq = torch.fft.rfft(X[i:i + chunk], dim=-1); ph = torch.rand(Fq.shape, device=DEV, generator=g) * 2 * math.pi
        ph[..., 0] = 0
        if n % 2 == 0: ph[..., -1] = 0
        o[i:i + chunk] = torch.fft.irfft(Fq.abs() * torch.exp(1j * ph), n=n, dim=-1)
    return o


def fit_all(X, y, sid, nP, epochs, seed0, log_every=5):
    """X [N, C, 400] GPU float32 (condition input), y [N] float32 GPU, sid [N] long GPU. One member per held-out participant. Returns held-out predictions
    (np arrays in participant row order) and best validation r per member."""
    C = X.shape[1]; torch.manual_seed(seed0); members = []
    for h in range(nP):
        rest = [k for k in range(nP) if k != h]; val = rest[(h * 7) % len(rest)]
        members.append((torch.cat([torch.where(sid == k)[0] for k in rest if k != val]), torch.where(sid == val)[0], torch.where(sid == h)[0]))
    M = nP; ntr = torch.tensor([len(a) for a, _, _ in members], device=DEV); nmax = int(ntr.max())
    tr = torch.zeros(M, nmax, dtype=torch.long, device=DEV); ve = torch.zeros(M, max(len(b) for _, b, _ in members), dtype=torch.long, device=DEV)
    te = torch.zeros(M, max(len(c) for _, _, c in members), dtype=torch.long, device=DEV)
    for j, (a, b, c) in enumerate(members):
        tr[j, :len(a)] = a; ve[j, :len(b)] = b; ve[j, len(b):] = b[0]; te[j, :len(c)] = c; te[j, len(c):] = c[0]
    nets = [EEGNet(n_chans=C, n_outputs=1, n_times=400) for _ in range(M)]
    net = MEEG(nets, C).to(DEV); opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-2)
    spe = math.ceil(nmax / 128); bs = 128
    best = torch.full((M,), -1e9, device=DEV); pred = torch.zeros(M, te.shape[1], device=DEV); best_ep = torch.zeros(M, dtype=torch.long, device=DEV)
    nv = torch.tensor([len(b) for _, b, _ in members], device=DEV)

    def predict(idx):
        out = []
        for e0 in range(0, idx.shape[1], 256):
            out.append(net(X[idx[:, e0:e0 + 256]].transpose(0, 1).contiguous()).T)        # [M, chunk]
        return torch.cat(out, 1)
    for ep in range(epochs):
        net.train(); key = torch.rand(M, nmax, device=DEV); key[torch.arange(nmax, device=DEV)[None] >= ntr[:, None]] = 2.0
        perm = tr.gather(1, key.argsort(1)); ts = time.time()
        for st in range(spe):
            pos = (st * bs + torch.arange(bs, device=DEV))[None] % ntr[:, None]; idx = perm.gather(1, pos)                  # [M, bs]
            x = X[idx].transpose(0, 1).contiguous(); yy = y[idx].T                                                       # [bs, M, C, T], [bs, M]
            loss = ((net(x) - yy) ** 2).mean(0).sum()
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            pv = predict(ve); pt = predict(te)
            for j in range(M):
                a = pv[j, :nv[j]]; b = y[ve[j, :nv[j]]]
                r = torch.corrcoef(torch.stack([a, b]))[0, 1] if a.std() > 0 else torch.tensor(-1.0, device=DEV)
                r = torch.nan_to_num(r, nan=-1.0)
                if r > best[j]: best[j] = r; pred[j] = pt[j]; best_ep[j] = ep
        if ep == 0: log(f'  epoch time {time.time()-ts:.1f}s, steps/epoch {spe}, mem {torch.cuda.max_memory_allocated()/2**30:.1f} GB')
        if (ep + 1) % log_every == 0: log(f'  epoch {ep+1}/{epochs} mean best val r {float(best.mean()):.3f}')
    out = [pred[j, :len(members[j][2])].cpu().numpy() for j in range(M)]
    del net, opt; torch.cuda.empty_cache()
    return out, best.cpu().numpy(), best_ep.cpu().numpy()


def boot(v, B=10000):
    v = torch.tensor(np.asarray(v, np.float64), device=DEV); g = torch.Generator(device=DEV); g.manual_seed(0)
    m = v[torch.randint(len(v), (B, len(v)), device=DEV, generator=g)].mean(1)
    return {'mean': float(v.mean()), 'ci95': [float(m.quantile(0.025)), float(m.quantile(0.975))]}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out', required=True); ap.add_argument('--subs', default=','.join(str(i) for i in range(1, 24)))
    ap.add_argument('--conds', default='real,phase0,phase1,phase2,shuffle0,shuffle1,shuffle2'); ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--gate_only', action='store_true'); ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True); subs = [int(x) for x in a.subs.split(',')]
    data, n_ch = DD.load_all(); allsub = list(range(1, 24))
    log('data loaded', len(data), 'participants,', n_ch, 'channels')
    gate = {'G0_batched_vs_braindecode_max_abs_diff': g0_equivalence(n_ch)}; log('G0', gate)
    evaluate, ids, _, _ = targets()
    rep = {r['subject']: r for r in json.loads((ROOT / 'outputs/AGENT_LOOP_20261120/results/log_yield_replication.json').read_text())['datasets']['ds003458']}
    gd, ty = [], []
    for s in allsub:
        z = np.load(ROOT / f'results/features/eegnet/sub-{s:03d}.npz'); tri = z['tri']; yv = evaluate(s, tri)
        ty.append(float(np.abs(yv - data[s - 1][1]).max())); assert (tri == data[s - 1][2]).all()
        gd.append(abs(density_gain(yv, z['F'].astype(float), folds(len(yv))) - rep[s]['gain']))
    gate['G1_saved_gain_max_abs_diff'] = max(gd); gate['G1_target_max_abs_diff'] = max(ty)
    gate['G1_pass'] = bool(max(gd) < 1e-10 and max(ty) < 1e-4); gate['G0_pass'] = bool(max(gate['G0_batched_vs_braindecode_max_abs_diff'].values()) < 1e-5)
    log('G1', gate)
    (out / 'gate.json').write_text(json.dumps(dict(code_sha256=CODE_SHA256, **gate), indent=1))
    if not (gate['G0_pass'] and gate['G1_pass']): raise SystemExit('GATE FAIL (G0/G1): see gate.json')
    if a.gate_only: return
    # rows of the pooled array in participant order; condition inputs are built on the GPU from the same rows
    X0 = torch.tensor(np.concatenate([d[0] for d in data]).reshape(sum(len(d[1]) for d in data), n_ch, 400), device=DEV)
    y = torch.tensor(np.concatenate([d[1] for d in data]), device=DEV)
    sid = torch.tensor(np.concatenate([[k] * len(d[1]) for k, d in enumerate(data)]), device=DEV); nP = len(data)
    # (all 23 participants are always in the training pool; --subs only selects which held-out members are reported, kept at all 23 for the full run)
    res = {}
    for c in a.conds.split(','):
        f = out / f'cond_{c}.json'
        if f.exists(): res[c] = json.loads(f.read_text()); continue
        g = torch.Generator(device=DEV)
        if c == 'real': X = X0
        elif c.startswith('phase'):
            g.manual_seed(1000 + int(c[5:])); X = phase_rand(X0, g); pw = float(X.pow(2).mean() / X0.pow(2).mean())
        else:
            g.manual_seed(2000 + int(c[7:])); X = X0.clone()
            for k in range(nP):
                r = torch.where(sid == k)[0]; X[r] = X0[r[torch.randperm(len(r), device=DEV, generator=g)]]
        log(f'cond {c}: fit 23 folds as one batched model'); ts = time.time()
        preds, bv, be = fit_all(X, y, sid, nP, a.epochs, a.seed + 1000 * (0 if c == 'real' else 1 + len(res)))
        if X is not X0: del X
        torch.cuda.empty_cache(); rows = {}
        for s in allsub:
            yv = evaluate(s, data[s - 1][2]); p = preds[s - 1].astype(float)
            rows[str(s)] = dict(gain=density_gain(yv, p[:, None], folds(len(yv))), r_heldout=float(np.corrcoef(p, data[s - 1][1])[0, 1]) if p.std() > 0 else 0.0,
                                best_val_r=float(bv[s - 1]), best_epoch=int(be[s - 1]), saved_gain=rep[s]['gain'] if c == 'real' else None)
        res[c] = dict(rows=rows, seconds=time.time() - ts, **({'power_ratio': pw} if c.startswith('phase') else {}))
        f.write_text(json.dumps(res[c], indent=1)); log(f'cond {c}: mean gain {np.mean([r["gain"] for r in rows.values()]):+.4f} ({time.time()-ts:.0f}s)')
    S = {'NOT_CITABLE': 'until independent review', 'code_sha256': CODE_SHA256, 'bootstrap': 'participant, 10000 resamples, GPU torch, seed 0', 'n_participants': 23,
         'epochs': a.epochs, 'gate': gate, 'unit': 'bits per feedback trial (density_gain, 10 contiguous folds)', 'bits': {}}
    for c, r in res.items(): S['bits'][c] = boot([r['rows'][str(s)]['gain'] for s in allsub])
    for fam in ('phase', 'shuffle'):
        cs = [c for c in res if c.startswith(fam)]
        if cs:
            m = [float(np.mean([res[c]['rows'][str(s)]['gain'] for c in cs])) for s in allsub]; S['bits'][f'{fam}_seedmean'] = boot(m)
            S[f'{fam}_max_abs_per_participant_seedmean'] = float(np.max(np.abs(m))); S[f'{fam}_floor_cohort_ok_le_0.02'] = bool(abs(np.mean(m)) <= 0.02)
            S[f'{fam}_floor_all_participants_le_0.02'] = bool(np.max(np.abs(m)) <= 0.02)
    if 'real' in res:
        rg = np.array([res['real']['rows'][str(s)]['gain'] for s in allsub]); sg = np.array([rep[s]['gain'] for s in allsub])
        S['real_refit_vs_saved'] = dict(refit_mean=float(rg.mean()), saved_mean=float(sg.mean()), mean_diff=float((rg - sg).mean()), participant_corr=float(np.corrcoef(rg, sg)[0, 1]),
                                         refit_minus_saved=boot(rg - sg))
        for c in res:
            if c != 'real': S['bits'][f'{c}_minus_real'] = boot([res[c]['rows'][str(s)]['gain'] - res['real']['rows'][str(s)]['gain'] for s in allsub])
    (out / 'summary.json').write_text(json.dumps(S, indent=1)); log(json.dumps({k: v for k, v in S.items() if k != 'gate'}, indent=1)[:3000])


if __name__ == '__main__':
    main()
