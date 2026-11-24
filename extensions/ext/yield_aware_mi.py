"""yield_aware_mi.py - RUNNER code for PROTOCOL_yield_aware_loss.md (card A revised).
ONE script; imports mi_card_v3 (M) and rf2_mi_matched_rank (RF) unchanged. Hash-pinned by results/yield_aware/PREREG.json.
Question: does a yield-aware loss (CE + alpha*KL(pi||softmax f(blank)) + mu*proxy-topography penalty) make a 1-40 Hz CNN+MLP better on raw
input X than the same net trained on the proxy-regressed input R (plain-R)?
Stages (--stage): smoke | stamp | run | eval60 | summary.
  python yield_aware_mi.py --stage smoke
  python yield_aware_mi.py --stage stamp
  python yield_aware_mi.py --stage run --items 1,2,3,4 [--out DIR --subjects LIST --steps N]
  python yield_aware_mi.py --stage eval60 ;  python yield_aware_mi.py --stage summary
All protocol constants are fixed in this file (section numbers refer to the protocol)."""
import argparse, datetime, hashlib, json, math, sys, time
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

ROOT = Path('/path/to/eegaudit'); sys.path.insert(0, str(ROOT / 'experiments/ext'))
import mi_card_v3 as M
import rf2_mi_matched_rank as RF

DEV = 'cuda'; W0, W1 = M.W0, M.W1
RES = ROOT / 'results/yield_aware'; PROTO = ROOT / 'outputs/FINAL/PROTOCOL_yield_aware_loss.md'; SCRIPT = Path(__file__).resolve()
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
t0 = time.time(); log = lambda *m: print(f'[{time.time() - t0:7.1f}s]', *m, flush=True)
IST = datetime.timezone(datetime.timedelta(hours=5, minutes=30)); now_ist = lambda: datetime.datetime.now(IST)
STEPS, EVERY, BATCH, NBLANK, AMP, CAP_S, RESERVE_S = 2000, 100, 64, 32, 0.03, 10800, 600

# ---------------------------------------------------------------- section 4: items
# arm: (alpha, mu, Q kind). inp: X | R | NZ | HP4 | P0 | P9. lab: true | sh. rt: forward R-test rows (X-trained, true labels). rh: held-out R_h rows. snap: save snapshots.
ARMS = dict(plain=(0, 0, 'p5'), full=(1, 10, 'p5'), blank=(1, 0, 'p5'), pen=(0, 10, 'p5'), full3=(1, 10, 'p3'), permB=(1, 10, 'perm'),
            randset=(1, 10, 'rand'), full100=(1, 100, 'p5'))
TYPE = dict(plain='plain', pen='penalty', blank='blank', full='full', full3='full', permB='full', randset='full', full100='full')


def _it(arm, inp='X', seed=11, lab='true', rt=False, rh=False, snap=False, name=''):
    return dict(arm=arm, inp=inp, seed=seed, lab=lab, rt=rt, rh=rh, snap=snap, name=name)


ITEMS = {1: _it('plain', rt=True, rh=True, snap=True, name='plain-X s11'), 2: _it('plain', 'R', name='plain-R s11'),
         3: _it('plain', 'NZ', name='plain noise twin'), 4: _it('plain', lab='sh', name='plain shuffled-label X'),
         5: _it('plain', 'HP4', rt=True, name='plain-HP4-X'),
         6: _it('full', rt=True, snap=True, name='full-X s11'), 7: _it('full', seed=12, rt=True, snap=True, name='full-X s12'),
         8: _it('full', seed=13, rt=True, snap=True, name='full-X s13'),
         9: _it('plain', seed=12, rt=True, snap=True, name='plain-X s12'), 10: _it('plain', seed=13, rt=True, snap=True, name='plain-X s13'),
         11: _it('plain', 'R', seed=12, name='plain-R s12'), 12: _it('plain', 'R', seed=13, name='plain-R s13'),
         13: _it('full', lab='sh', name='full shuffled-label X'), 14: _it('pen', 'NZ', name='penalty-only noise twin'),
         15: _it('plain', 'P0', name='planted rho0 plain'), 16: _it('full', 'P0', name='planted rho0 full'),
         17: _it('full3', rt=True, rh=True, name='full-3 X (single seed)'), 18: _it('permB', rt=True, name='permuted-B full X (single seed)'),
         19: _it('randset', rt=True, name='randset full X (single seed)'), 20: _it('blank', rt=True, name='blank-only X (single seed)'),
         21: _it('pen', rt=True, name='penalty-only X (single seed)'), 22: _it('plain', 'P9', name='planted rho0.9 plain'),
         23: _it('full', 'P9', name='planted rho0.9 full'), 24: _it('full', 'R', name='full-R (single seed)'),
         25: _it('full100', rt=True, name='full mu=100 dose (single seed)')}
ORDER_RUN = list(range(1, 26))


def item_type(it): return TYPE[ITEMS[it]['arm']]


# ---------------------------------------------------------------- data / inputs
def load_cohort(spec):
    cache = ROOT / 'results/ext/mi_card_cache'; first = None; subs = list(range(1, 110))
    if spec.startswith('first:'): first = int(spec[6:])
    elif spec: subs = [int(x) for x in spec.split(',')]
    D, excl = {}, {}
    for s in subs:
        d, why = M.load_sub(s, cache)
        if d is None: excl[s] = why; continue
        c = np.bincount(d['y'], minlength=4)
        if c.max() - c.min() > 2: excl[s] = 'class counts'; continue
        D[s] = d
        if first and len(D) == first: break
    return D, excl


def loadings(Xb, sid, nP, D):
    """Per-participant float64 OLS loadings [nP, 64, k] of all 64 channels of Xb on the k derivation series (columns of D [64,k]) + intercept
    (= rows 0..k-1 of the lstsq solution of mi_card_v3.eog_regress, transposed)."""
    k = D.shape[1]; out = torch.empty(nP, 64, k, dtype=torch.float64, device=DEV); T = Xb.shape[-1]
    for i in range(nP):
        r = torch.where(sid == i)[0]; x = Xb[r].double(); Pk = torch.einsum('nct,cj->njt', x, D)
        A = torch.cat([Pk.permute(0, 2, 1).reshape(-1, k), torch.ones(len(r) * T, 1, dtype=torch.float64, device=DEV)], 1)
        Y = x.permute(0, 2, 1).reshape(-1, 64); B = torch.linalg.lstsq(A, Y).solution; out[i] = B[:k].T
    return out


def regress_k(R0, sid, nP, D):
    """= mi_card_v3.eog_regress with the derivations given as a weight matrix D [64,k] (k series), residual float32 + mean variance share removed."""
    k = D.shape[1]; out = torch.empty_like(R0); T = R0.shape[-1]; share = []
    for i in range(nP):
        r = torch.where(sid == i)[0]; x = R0[r].double(); Pk = torch.einsum('nct,cj->njt', x, D)
        A = torch.cat([Pk.permute(0, 2, 1).reshape(-1, k), torch.ones(len(r) * T, 1, dtype=torch.float64, device=DEV)], 1)
        Y = x.permute(0, 2, 1).reshape(-1, 64); B = torch.linalg.lstsq(A, Y).solution; E = Y - A @ B
        share.append(float(1 - E.pow(2).sum() / (Y - Y.mean(0)).pow(2).sum())); out[r] = E.reshape(len(r), T, 64).permute(0, 2, 1).float(); del x, Pk, A, Y, E
    return out, float(np.mean(share))


class Env:
    def __init__(s, spec):
        D, s.excl = load_cohort(spec); s.P = sorted(D); s.nP = len(s.P); s.ch = D[s.P[0]]['ch']; assert all(D[p]['ch'] == s.ch for p in s.P)
        s.X = torch.tensor(np.concatenate([D[p]['X'] for p in s.P]), dtype=torch.float32)               # CPU
        s.y = torch.tensor(np.concatenate([D[p]['y'] for p in s.P]), device=DEV)
        s.sid = torch.tensor(np.concatenate([[k] * len(D[p]['y']) for k, p in enumerate(s.P)]), device=DEV); s.N = len(s.y); del D
        s.G = min(10, s.nP); s.pgrp = torch.arange(s.nP, device=DEV) % s.G; s.grp = s.pgrp[s.sid]
        for k in range(s.nP):
            c = torch.bincount(s.y[s.sid == k], minlength=4); assert int(c.max() - c.min()) <= 2
        log(f'{s.nP} participants, {s.N} trials, excluded {len(s.excl)}')
        s.Xb = M.bandpass(s.X.to(DEV), 1, 40); s.sk = torch.empty(s.nP, device=DEV)
        Wn = s.Xb[:, :, W0:W1].contiguous()
        for k in range(s.nP): s.sk[k] = Wn[s.sid == k].std()
        del Wn
        gperm = torch.Generator(device=DEV); gperm.manual_seed(7); s.ysh = s.y.clone()                  # v3 ysh
        for k in range(s.nP):
            r = torch.where(s.sid == k)[0]; s.ysh[r] = s.y[r[torch.randperm(len(r), generator=gperm, device=DEV)]]
        s.Dp = RF.dmat(s.ch, M.PROXY, M.FPM); s.cache = {}; s.Nz0 = None; s._B = {}; s.share = {}
        x4 = M.proxies(s.Xb[:4].double(), s.ch); assert torch.allclose(torch.einsum('nct,cj->njt', s.Xb[:4].double(), s.Dp), x4), 'proxy matrix mismatch'
        s.perm = np.random.default_rng(4242).permutation(64)

    def B(s, kind):                                                   # per-participant loadings used for Q (label-free, real X only)
        if kind not in s._B:
            if kind == 'p5': D = s.Dp
            elif kind == 'p3': D = s.Dp[:, [0, 1, 4]]                  # F7-F8, FT7-FT8, mean(FP1,FPZ,FP2); built without T7-T8, T9-T10
            elif kind == 'rand': D = s.randset_D()
            else: raise KeyError(kind)
            s._B[kind] = loadings(s.Xb, s.sid, s.nP, D)
        return s._B[kind]

    def randset_D(s):                                                 # rf2 'any' family, draw 0 (rng seed 303)
        pch = {c for p in M.PROXY for c in p} | set(M.FPM); anyp = [c for c in s.ch if c not in pch]
        c = list(np.random.default_rng(303).choice(anyp, 11, replace=False)); pairs = [(c[0], c[1]), (c[2], c[3]), (c[4], c[5]), (c[6], c[7])]
        s.randset_pairs = dict(pairs=[[str(a), str(b)] for a, b in pairs], mean3=[str(x) for x in c[8:]])
        return RF.dmat(s.ch, pairs, c[8:])

    def Q(s, kind, meta):
        """[M,64,r] orthonormal basis per member: QR of the mean over that member's TRAINING participants of the per-participant loadings."""
        Mm = len(meta); mask = torch.zeros(Mm, s.nP, dtype=torch.float64, device=DEV)
        for j, m in enumerate(meta):
            mask[j] = ((s.pgrp != m['gg']) & (s.pgrp != m['v'])).double() if m['role'] == 'inner' else (s.pgrp != m['gg']).double()
        Bk = s.B('p5' if kind == 'perm' else kind); Bm = (mask[:, :, None, None] * Bk[None]).sum(1) / mask.sum(1)[:, None, None]
        if kind == 'perm': Bm = Bm[:, s.perm]
        return torch.linalg.qr(Bm).Q.float()

    def nz0(s):
        if s.Nz0 is None:
            g = torch.Generator(device=DEV); g.manual_seed(20261121)                                    # = v3 noise draw
            Nz = M.bandpass(torch.randn(s.X.shape, generator=g, device=DEV), 1, 40)
            for k in range(s.nP):
                r = s.sid == k; Nz[r] = Nz[r] / Nz[r].std((0, 2), keepdim=True) * s.Xb[r].std((0, 2), keepdim=True)
            s.Nz0 = M.bandpass(Nz, 1, 40).cpu(); del Nz; torch.cuda.empty_cache()
        return s.Nz0

    def win(s, Z, sc=None):                                           # window + per-participant scalar (own std if sc is None)
        W = Z[:, :, W0:W1].contiguous(); own = torch.empty(s.nP, device=DEV)
        for k in range(s.nP):
            r = s.sid == k; own[k] = W[r].std()
            W[r] /= (own[k] if sc is None else sc[k])
        return W, own

    def pat(s, rho):
        gp = torch.Generator(device=DEV); gp.manual_seed(5); u5 = torch.randint(0, 2, (4, 5), generator=gp, device=DEV).double() * 2 - 1
        U = torch.zeros(4, 64, dtype=torch.float64, device=DEV); U[:, [s.ch.index(c) for c in RF.PLANT_CH]] = u5
        b = torch.linalg.svd(s.B('p5').mean(0), full_matrices=False)[0][:, 0]; b = b / b.norm()
        Uh = U - (U @ b)[:, None] * b[None]; Uh = Uh / Uh.norm(dim=1, keepdim=True) * math.sqrt(5)
        sg = torch.tensor([1., 1., -1., -1.], dtype=torch.float64, device=DEV)
        return math.sqrt(1 - rho ** 2) * Uh + rho * sg[:, None] * math.sqrt(5) * b[None]

    def get(s, key):
        if key in s.cache: return s.cache[key]
        if key == 'XIN': s.cache[key], _ = s.win(s.Xb, s.sk)
        elif key in ('RIN', 'RTEST'):
            R, s.share['R'] = M.eog_regress(s.Xb, s.sid, s.nP, s.ch); s.cache['RIN'], _ = s.win(R, None); s.cache['RTEST'], _ = s.win(R, s.sk); del R
        elif key == 'RHTEST':
            R, s.share['Rh'] = regress_k(s.Xb, s.sid, s.nP, s.Dp[:, [2, 3]]); s.cache[key], _ = s.win(R, s.sk); del R
        elif key in ('HP4', 'RHP4'):
            s.cache['HP4'], sc = s.win(M.bandpass(s.X.to(DEV), 4, 40), None)
            R, _ = M.eog_regress(s.Xb, s.sid, s.nP, s.ch); s.cache['RHP4'], _ = s.win(M.bandpass(R, 4, 40), sc); del R
        elif key == 'NZ': s.cache[key], _ = s.win(s.nz0().to(DEV), s.sk)
        elif key in ('P0', 'P9'):
            rho = 0.0 if key == 'P0' else 0.9; Nz = s.nz0().to(DEV); sdc = Nz.std((0, 2)); pat = s.pat(rho).float()
            wave = torch.sin(2 * math.pi * 3 * torch.arange(Nz.shape[-1], device=DEV) / M.SF)
            Z = Nz + AMP * sdc[None, :, None] * pat[s.y][:, :, None] * wave[None, None]; s.cache[key], _ = s.win(Z, s.sk); del Z, Nz
        else: raise KeyError(key)
        return s.cache[key]

    def need(s, keys):
        for k in [k for k in s.cache if k not in keys]: del s.cache[k]
        torch.cuda.empty_cache()
        for k in keys: s.get(k)


def make_members(env, groups):
    mem, meta = [], []
    for gg in groups:
        v = (gg + 1) % env.G; g = env.grp
        mem.append((torch.where((g != gg) & (g != v))[0], torch.where(g == v)[0])); meta.append(dict(gg=gg, v=v, role='inner'))
        mem.append((torch.where(g != gg)[0], torch.where(g == gg)[0])); meta.append(dict(gg=gg, v=v, role='final'))
    return mem, meta


# ---------------------------------------------------------------- net, blank, penalty
class YNet(M.MEEGNet):
    """MEEGNet trunk (unchanged __init__ => identical init draws). head 'linear' = v3 forward exactly; head 'mlp' = Linear(nf,64)-ELU-Dropout(.25)-Linear(64,4)."""
    def __init__(s, Mm, T, head):
        super().__init__(Mm, T=T); s.head = head
        if head == 'mlp':
            nf = s.W.shape[1]; del s.W, s.b; b1, b2 = 1 / math.sqrt(nf), 1 / math.sqrt(64)
            s.W1 = nn.Parameter((torch.rand(Mm, nf, 64) * 2 - 1) * b1); s.b1 = nn.Parameter((torch.rand(Mm, 64) * 2 - 1) * b1)
            s.W2 = nn.Parameter((torch.rand(Mm, 64, 4) * 2 - 1) * b2); s.b2 = nn.Parameter((torch.rand(Mm, 4) * 2 - 1) * b2)

    def forward(s, x):
        if s.head == 'linear': return super().forward(x)
        z = F.elu(s.bn2(s.c2(s.bn1(s.c1(x))))); z = F.dropout(F.avg_pool2d(z, (1, 4)), s.p, s.training)
        z = F.elu(s.bn3(s.c4(s.c3(z)))); z = F.dropout(F.avg_pool2d(z, (1, 8)), s.p, s.training)
        z = z.reshape(z.shape[0], s.M, -1); h = F.elu(torch.einsum('bmi,mij->bmj', z, s.W1) + s.b1); h = F.dropout(h, 0.25, s.training)
        return torch.einsum('bmj,mjc->bmc', h, s.W2) + s.b2


def phase_blank(x, gen):
    """x [B,M,C,T] -> phase-randomised copy of the SAME trial: magnitudes kept, ONE random phase offset per (row, member, frequency) COMMON to all 64 channels
    (added to every channel's phase, so the cross-spectral matrix and the cross-channel covariance are kept exactly), DC and Nyquist kept."""
    T = x.shape[-1]; assert T % 2 == 0; Xf = torch.fft.rfft(x, dim=-1)
    ph = torch.rand(x.shape[0], x.shape[1], 1, Xf.shape[-1], generator=gen, device=x.device) * (2 * math.pi)
    rot = torch.polar(torch.ones_like(ph), ph); rot[..., 0] = 1; rot[..., -1] = 1
    return torch.fft.irfft(Xf * rot, n=T, dim=-1)


def pen_ratio(net, Q):
    """[M] mean over the 16 first-layer spatial filters of ||Q'w||^2 / (||w||^2 + 1e-12)."""
    w = net.c2.weight.reshape(net.M, -1, 64); pr = torch.einsum('mcr,mfc->mfr', Q, w)
    return (pr.pow(2).sum(-1) / (w.pow(2).sum(-1) + 1e-12)).mean(-1)


@torch.no_grad()
def fwd_rows(net, inp, evp, gen=None):
    zz = []
    for e0 in range(0, evp.shape[1], 128):
        xx = inp[evp[:, e0:e0 + 128]].transpose(0, 1).contiguous()
        if gen is not None: xx = phase_blank(xx, gen)
        zz.append(net(xx).transpose(0, 1).float())
    return torch.cat(zz, 1)


def train_chunk(Xin, ytr, members, steps, every, batch, seed, c0, cfg, evalsets, Qpen=None, Qlog=None, snapshot=False, blank_eval=True,
                log_every=50, pad_eval=None):
    """Train one chunk of members (batched). cfg: alpha, mu, head. Returns per-member lists of logits [n_ckpt, n_eval, 4] per eval set, penalty ratios,
    loss curves, timing, snapshots. With alpha = mu = 0 and head 'linear' this is mi_card_v3.eegnet_members bit for bit (no extra global-RNG draw)."""
    Mm = len(members); alpha, mu = cfg['alpha'], cfg['mu']; torch.manual_seed(seed + c0)
    lt = max(len(a) for a, _ in members); le = max(len(b) for _, b in members)
    trp = torch.zeros(Mm, lt, dtype=torch.long, device=DEV); tl = torch.tensor([len(a) for a, _ in members], device=DEV)
    evp = torch.zeros(Mm, le, dtype=torch.long, device=DEV)
    for j, (a, b) in enumerate(members): trp[j, :len(a)] = a; evp[j, :len(b)] = b; evp[j, len(b):] = b[0]
    if pad_eval: evp = evp[:, torch.arange(pad_eval, device=DEV) % le]; le = pad_eval
    net = YNet(Mm, Xin.shape[-1], cfg['head']).to(DEV); opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    pri = torch.stack([M.prior_of(ytr[a]) for a, _ in members]).float(); lpri = pri.log()
    gen_b = torch.Generator(device=DEV); gen_b.manual_seed(9001 + cfg['seed']); gen_e = torch.Generator(device=DEV)
    Qlog = Qlog or {}; r0 = {k: pen_ratio(net, q).detach().cpu() for k, q in Qlog.items()}
    ck = {nm: [] for nm in list(evalsets) + (['blank'] if blank_eval else [])}; snaps = []; curves = []; bad = torch.zeros((), dtype=torch.bool, device=DEV)
    acc = torch.zeros(3, device=DEV); torch.cuda.synchronize(); ts = time.time(); t_eval = 0.0
    for st in range(1, steps + 1):
        net.train(); pos = (torch.rand(Mm, batch, device=DEV) * tl[:, None]).long(); idx = trp.gather(1, pos)
        x = Xin[idx].transpose(0, 1).contiguous(); yy = ytr[idx].T
        if alpha > 0:
            with torch.no_grad(): xb = phase_blank(x[:NBLANK], gen_b)
            out = net(torch.cat([x, xb], 0)); real = out[:batch]
        else: out = real = net(x)
        ce = F.cross_entropy(real.reshape(-1, 4), yy.reshape(-1), reduction='none').reshape(batch, Mm).mean(0).sum(); loss = ce
        if alpha > 0:
            lp = F.log_softmax(out[batch:], -1); kl = (pri[None] * (lpri[None] - lp)).sum(-1).mean(0).sum(); loss = loss + alpha * kl
        if mu > 0:
            pr = pen_ratio(net, Qpen); loss = loss + mu * pr.sum()
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        with torch.no_grad():
            bad |= ~torch.isfinite(loss.detach()); acc[0] += ce.detach()
            if alpha > 0: acc[1] += kl.detach()
            if mu > 0: acc[2] += pr.detach().sum()
        if st % log_every == 0:
            a = (acc / (log_every * Mm)).tolist(); acc.zero_(); curves.append(dict(step=st, ce=a[0], kl=a[1], pen=a[2]))
        if st % every == 0:
            torch.cuda.synchronize(); te = time.time(); net.eval()
            with torch.no_grad():
                for nm, inp in evalsets.items(): ck[nm].append(fwd_rows(net, inp, evp))
                if blank_eval: gen_e.manual_seed(424242 + c0); ck['blank'].append(fwd_rows(net, Xin, evp, gen_e))
            if snapshot: snaps.append({k: v.detach().cpu().clone() for k, v in net.state_dict().items()})
            torch.cuda.synchronize(); t_eval += time.time() - te
        if st == every: log(f'  chunk c0={c0} M={Mm}: {(time.time() - ts) / every:.3f} s/step (incl. first ckpt eval) loss {float(loss.detach()):.4f}')
    torch.cuda.synchronize(); tt = time.time() - ts
    rT = {k: pen_ratio(net, q).detach().cpu() for k, q in Qlog.items()}; wlast = net.c2.weight.detach().reshape(Mm, -1, 64).cpu()
    Zs = {nm: torch.stack(v, 1) for nm, v in ck.items()}                                          # [M, n_ckpt, le, 4]
    Zl = {nm: [(Zs[nm][j, :, :len(members[j][1])] if not pad_eval else Zs[nm][j]) for j in range(Mm)] for nm in Zs}
    res = dict(Z=Zl, ratio0=r0, ratioT=rT, curves=curves, finite=not bool(bad), wlast=wlast, snaps=snaps,
               timing=dict(total_s=tt, eval_s=t_eval, steps=steps, s_per_step=tt / steps, mem_gb=torch.cuda.max_memory_allocated() / 2 ** 30),
               rng_cpu=torch.get_rng_state().clone(), rng_cuda=torch.cuda.get_rng_state().clone())
    del net, opt; torch.cuda.empty_cache(); return res


# ---------------------------------------------------------------- scoring one item
def agg(v, ev, env):
    s = env.sid[ev]; num = torch.zeros(env.nP, dtype=torch.float64, device=DEV).index_add_(0, s, v.double())
    cnt = torch.zeros(env.nP, dtype=torch.float64, device=DEV).index_add_(0, s, torch.ones_like(v, dtype=torch.float64)); return num / cnt.clamp(min=1), cnt > 0


def score_item(env, mem, meta, Zl, ytr):
    """(ckpt, T) per member pair from the INNER group of that arm's own input (mi_card_v3.select); final member scored on its eval group."""
    names = ['bits', 'bits1', 'acc']; sets = [k for k in Zl if k not in ('x', 'blank')]
    for k in sets: names += [f'{k}_bits', f'{k}_bits1', f'{k}_acc']
    if 'rt' in Zl: names += ['rt_bitsTsel']
    if 'blank' in Zl: names += ['blank_bits']
    out = {k: torch.full((env.nP,), float('nan'), dtype=torch.float64, device=DEV) for k in names}; pairs = []
    for p in range(len(mem) // 2):
        ji, jf = 2 * p, 2 * p + 1; ev_i, ev_f = mem[ji][1], mem[jf][1]; yi, yf = env.y[ev_i], env.y[ev_f]
        c, T, e = M.select(Zl['x'][ji], yi); prior = M.prior_of(ytr[mem[jf][0]])
        def put(prefix, Zf, T_, y_=yf):
            b = M.bits(Zf, y_, T_, prior); b1 = M.bits(Zf, y_, 1.0, prior); a = (Zf.argmax(-1) == y_).double()
            for nm, v in ((f'{prefix}bits', b), (f'{prefix}bits1', b1), (f'{prefix}acc', a)):
                m, ok = agg(v, ev_f, env); out[nm][ok] = m[ok]
        put('', Zl['x'][jf][c], T)
        for k in sets: put(f'{k}_', Zl[k][jf][c], T)
        if 'rt' in Zl:
            Tr = float(M.TGRID[int(M.ce_grid(Zl['rt'][ji][c], yi).argmin())]); b = M.bits(Zl['rt'][jf][c], yf, Tr, prior); m, ok = agg(b, ev_f, env)
            out['rt_bitsTsel'][ok] = m[ok]
        if 'blank' in Zl:
            b = M.bits(Zl['blank'][jf][c], yf, T, prior); m, ok = agg(b, ev_f, env); out['blank_bits'][ok] = m[ok]
        pairs.append(dict(gg=meta[jf]['gg'], ckpt=int(c), step=(int(c) + 1) * EVERY, T=T, ckpt_lt_last=bool(c < len(Zl['x'][ji]) - 1), **{k: bool(v) for k, v in e.items()}))
    return {k: v.cpu().tolist() for k, v in out.items()}, pairs


def eval_pen_own(env, meta, wlast):
    """Diagnostic (not used in training): proxy-energy of the final members' last-step filters in the EVAL participants' OWN loadings B_k."""
    res = []
    for j, m in enumerate(meta):
        if m['role'] != 'final': continue
        ks = torch.where(env.pgrp == m['gg'])[0]; Qk = torch.linalg.qr(env.B('p5')[ks]).Q.float(); w = wlast[j].to(DEV)
        pr = torch.einsum('kcr,fc->kfr', Qk, w); res.append(float((pr.pow(2).sum(-1) / (w.pow(2).sum(-1)[None] + 1e-12)).mean()))
    return res


def run_item(env, it, out, steps, groups, hashes, mchunk=20):
    spec = ITEMS[it]; alpha, mu, qk = ARMS[spec['arm']]; ti = time.time()
    keys = {'X': ['XIN'], 'R': ['RIN'], 'NZ': ['NZ'], 'HP4': ['HP4'], 'P0': ['P0'], 'P9': ['P9']}[spec['inp']]
    if spec['inp'] == 'X' and spec['rt']: keys += ['RTEST', 'RIN']
    if spec['inp'] == 'X' and spec['rh']: keys += ['RHTEST']
    if spec['inp'] == 'HP4': keys += ['RHP4']
    env.need(keys); Xin = env.get(keys[0])
    ev = {'x': Xin}
    if spec['inp'] == 'X' and spec['rt']: ev.update(rt=env.get('RTEST'), ro=env.get('RIN'))
    if spec['inp'] == 'X' and spec['rh']: ev['rh'] = env.get('RHTEST')
    if spec['inp'] == 'HP4': ev['rt'] = env.get('RHP4')
    ytr = env.ysh if spec['lab'] == 'sh' else env.y
    mem, meta = make_members(env, groups); Qpen = env.Q(qk, meta) if mu > 0 else None
    Qlog = {'proxy': env.Q('p5', meta)}
    if qk != 'p5': Qlog['own'] = Qpen if Qpen is not None else env.Q(qk, meta)
    cfg = dict(alpha=alpha, mu=mu, head='mlp', seed=spec['seed']); Zl, curves, snaps_all = None, None, []; r0 = {}; rT = {}; wl = []; fin = True; tim = []
    subs = [(c0, min(c0 + mchunk, len(mem))) for c0 in range(0, len(mem), mchunk)]
    try: res_l = [one(Xin, ytr, mem, meta, a, b, steps, cfg, ev, Qpen, Qlog, spec, qk) for a, b in subs]
    except torch.cuda.OutOfMemoryError:
        log('OOM at mchunk 20 -> mchunk 10 (two sub-chunks, seeds s + c0); logged'); torch.cuda.empty_cache(); mchunk = 10
        subs = [(c0, min(c0 + mchunk, len(mem))) for c0 in range(0, len(mem), mchunk)]
        res_l = [one(Xin, ytr, mem, meta, a, b, steps, cfg, ev, Qpen, Qlog, spec, qk) for a, b in subs]
    Zl = {nm: sum([r['Z'][nm] for r in res_l], []) for nm in res_l[0]['Z']}
    rr0 = {k: torch.cat([r['ratio0'][k] for r in res_l]) for k in Qlog}; rrT = {k: torch.cat([r['ratioT'][k] for r in res_l]) for k in Qlog}
    wlast = torch.cat([r['wlast'] for r in res_l]); finite = all(r['finite'] for r in res_l)
    per, pairs = score_item(env, mem, meta, Zl, ytr)
    isfin = [j for j, m in enumerate(meta) if m['role'] == 'final']
    rec = dict(item=it, spec=spec, name=spec['name'], steps=steps, groups=groups, M=len(mem), mchunk=mchunk, seed=spec['seed'], alpha=alpha, mu=mu, qkind=qk,
               participants=env.P, per=per, pairs=pairs, finite=finite, penalty_ratio_step0={k: v.tolist() for k, v in rr0.items()},
               penalty_ratio_last={k: v.tolist() for k, v in rrT.items()}, final_members=isfin, eval_participants_own_Bk_proxy_energy=eval_pen_own(env, meta, wlast),
               curves=[dict(step=c['step'], ce=float(np.mean([r['curves'][i]['ce'] for r in res_l])), kl=float(np.mean([r['curves'][i]['kl'] for r in res_l])),
                            pen=float(np.mean([r['curves'][i]['pen'] for r in res_l]))) for i, c in enumerate(res_l[0]['curves'])],
               timing=dict(s_per_step=[r['timing']['s_per_step'] for r in res_l], eval_s=[r['timing']['eval_s'] for r in res_l], item_wall_s=time.time() - ti,
                           mem_gb=max(r['timing']['mem_gb'] for r in res_l)), hashes=hashes, shrink_note='transductive, label-free scaling s_k',
               var_share_removed=dict(env.share), randset=getattr(env, 'randset_pairs', None))
    out.mkdir(parents=True, exist_ok=True); (out / 'items').mkdir(exist_ok=True); (out / 'logits').mkdir(exist_ok=True)
    torch.save({k: [z.cpu() for z in v] for k, v in Zl.items()}, out / 'logits' / f'item_{it:02d}.pt')
    if spec['snap']:
        (out / 'snap').mkdir(exist_ok=True)
        torch.save(dict(chunks=[dict(c0=a, M=b - a, states=r['snaps']) for (a, b), r in zip(subs, res_l)], groups=groups), out / 'snap' / f'item_{it:02d}.pt')
    (out / 'items' / f'item_{it:02d}.json').write_text(json.dumps(rec))
    P = np.array(per['bits']); log(f'ITEM {it} {spec["name"]}: Y_X {np.nanmean(P):+.4f} acc {np.nanmean(per["acc"]):.3f} T1 {np.nanmean(per["bits1"]):+.4f}'
                                   + (f' | Rtest {np.nanmean(per["rt_bits"]):+.4f}' if 'rt_bits' in per else '') + f' | finite {finite} | {time.time() - ti:.0f}s')
    del Zl, res_l; torch.cuda.empty_cache(); return rec


def one(Xin, ytr, mem, meta, a, b, steps, cfg, ev, Qpen, Qlog, spec, qk):
    sl = slice(a, b); Qp = Qpen[sl] if Qpen is not None else None; Ql = {k: q[sl] for k, q in Qlog.items()}
    return train_chunk(Xin, ytr, mem[sl], steps, EVERY, BATCH, spec['seed'], a, cfg, ev, Qp, Ql, snapshot=spec['snap'], blank_eval=True)


# ---------------------------------------------------------------- statistics
class Stats:
    def __init__(s, nv):
        g = torch.Generator(device=DEV); g.manual_seed(0); s.nv = nv; s.IX = torch.randint(0, nv, (10000, nv), generator=g, device=DEV)

    T = staticmethod(lambda v: torch.as_tensor(np.asarray(v, np.float64), device=DEV))
    q = staticmethod(lambda d: [float(torch.quantile(d, 0.025)), float(torch.quantile(d, 0.975))])

    def dist(s, v): return s.T(v)[s.IX].mean(1)
    def mean(s, v): return dict(mean=float(np.mean(v)), ci=s.q(s.dist(v)))
    def ratio(s, a, b): d = s.dist(a) / s.dist(b); return dict(point=float(np.mean(a) / np.mean(b)), ci=s.q(d))
    def d_ratio(s, a, b, c, dd): d = s.dist(a) / s.dist(b) - s.dist(c) / s.dist(dd); return dict(point=float(np.mean(a) / np.mean(b) - np.mean(c) / np.mean(dd)), ci=s.q(d))


def load_items(out):
    its = {}
    for f in sorted((out / 'items').glob('item_*.json')): d = json.loads(f.read_text()); its[d['item']] = d
    return its


def ledger_cum():
    f = RES / 'ledger.csv'
    if not f.exists(): return 0.0
    L = [l for l in f.read_text().strip().split('\n') if l and not l.startswith('item,')]
    return float(L[-1].split(',')[-1]) if L else 0.0


def g0_decide(its, st):
    """G0 headroom on seed 11 (items 1, 2): K_inv, K_aud upper CI <= 0.85 and Y_X - Y_Rtest >= 0.05 with lower CI > 0."""
    a = lambda i, k: np.array(its[i]['per'][k], float)
    x, rt, r = a(1, 'bits'), a(1, 'rt_bits'), a(2, 'bits'); ok = np.isfinite(x) & np.isfinite(rt) & np.isfinite(r); x, rt, r = x[ok], rt[ok], r[ok]
    K = st.ratio(rt, x); Ka = st.ratio(r, x); dif = st.mean(x - rt)
    ok_ = dict(K_inv_upper_le_0p85=K['ci'][1] <= 0.85, K_aud_upper_le_0p85=Ka['ci'][1] <= 0.85, diff_ge_0p05=dif['mean'] >= 0.05, diff_lower_gt_0=dif['ci'][0] > 0)
    return dict(seed11=True, Y_X=st.mean(x), Y_Rtest=st.mean(rt), Y_R=st.mean(r), K_inv=K, K_aud=Ka, Y_X_minus_Y_Rtest=dif, checks=ok_, pass_=all(ok_.values()),
                verdict='G0 pass' if all(ok_.values()) else 'DEAD-no-headroom')


# ---------------------------------------------------------------- stages: run
def budget_ok(spec_item, proj, step_elapsed, cum0, final=False):
    need = proj[item_type(spec_item)] * STEPS * 1.1
    ok = cum0 + step_elapsed + need <= CAP_S - RESERVE_S
    n = now_ist(); end = n + datetime.timedelta(seconds=need)
    clock = n.hour < 12 and end.replace(tzinfo=IST) <= n.replace(hour=14, minute=0, second=0, microsecond=0)
    return ok, clock, need


def stage_run(a):
    out = Path(a.out); smoke = out.name == '_smoke'; out.mkdir(parents=True, exist_ok=True)
    hashes = dict(protocol_sha256=sha(PROTO), script_sha256=sha(SCRIPT))
    if not smoke:
        pr = json.loads((RES / 'PREREG.json').read_text()); assert pr['script_sha256'] == hashes['script_sha256'], 'script changed after PREREG stamp'
        assert pr['protocol_sha256'] == hashes['protocol_sha256']; hashes['prereg_sha256'] = sha(RES / 'PREREG.json'); proj = pr['smoke_timing']['s_per_step']
    else: proj = None
    env = Env(a.subjects); groups = [int(g) for g in a.groups.split(',')] if a.groups else list(range(env.G))
    if not smoke and env.nP == 105:
        sv = json.loads((ROOT / 'results/ext/mi_card_v3/primary_lpgo/summary_mi.json').read_text()); assert sv['participants'] == env.P, 'cohort != saved v3 cohort'
        log('GATE cohort == saved mi_card_v3 participants (105): pass')
    log('transductive, label-free scaling s_k (std of the participant windowed rows, own test trials included)')
    items = [int(x) for x in a.items.split(',')]; ts = time.time(); cum0 = ledger_cum() if not smoke else 0.0; steps = a.steps
    for it in items:
        if (out / 'items' / f'item_{it:02d}.json').exists() and not a.force: log(f'item {it} exists, skipped'); continue
        if proj is not None:
            ok, clock, need = budget_ok(it, proj, time.time() - ts, cum0, False)
            if not ok: log(f'STOP_CAP before item {it}: cum {cum0 + time.time() - ts:.0f}s + projected {need:.0f}s > {CAP_S - RESERVE_S}'); (out / 'STOP_CAP').write_text(str(it)); sys.exit(9)
            if not clock: log(f'STOP_CLOCK before item {it} ({now_ist()})'); (out / 'STOP_CLOCK').write_text(str(it)); sys.exit(10)
        run_item(env, it, out, steps, groups, hashes)
        if it == 4 and (out / 'items' / 'item_01.json').exists() and (out / 'items' / 'item_02.json').exists():
            its = load_items(out); ok = np.isfinite(np.array(its[1]['per']['bits'])); g0 = g0_decide(its, Stats(int(ok.sum()))); g0['hashes'] = hashes
            (out / 'G0.json').write_text(json.dumps(g0, indent=1)); log('G0', json.dumps(g0))
            if not g0['pass_'] and not smoke: log('G0 FAIL: DEAD-no-headroom, stopping the run'); sys.exit(7)
    log(f'run stage done, {time.time() - ts:.0f}s')


# ---------------------------------------------------------------- eval-only controls (section 5)
def make_sets(env, per_family):
    pch = {c for p in M.PROXY for c in p} | set(M.FPM); ch = env.ch
    pools = dict(frontal=RF.FRONT, nonfrontal=[c for c in ch if c not in pch and c not in M.FP and c not in RF.FRONT], any=[c for c in ch if c not in pch])
    sets = [dict(name='proxy', fam='proxy', draw=0, D=env.Dp)]
    for fam, seed in (('frontal', 101), ('nonfrontal', 202), ('any', 303)):
        rng = np.random.default_rng(seed)
        for d in range(20):
            c = list(rng.choice(pools[fam], 11, replace=False)); pairs = [(c[0], c[1]), (c[2], c[3]), (c[4], c[5]), (c[6], c[7])]; m3 = c[8:]
            Dm = RF.dmat(ch, pairs, m3); assert int(torch.linalg.matrix_rank(Dm)) == 5
            if d < per_family: sets.append(dict(name=f'{fam}_d{d:02d}', fam=fam, draw=d, D=Dm, pairs=[[str(p), str(q)] for p, q in pairs], mean3=[str(x) for x in m3]))
    return sets, pools


def slice_state(sd, Mfull, js):
    """state_dict of a batched M-member net -> state_dict of the sub-net holding only members js (every tensor is member-major along dim 0)."""
    out = {}
    for k, v in sd.items():
        if v.dim() == 0: out[k] = v
        else: kd = v.shape[0] // Mfull; out[k] = v.reshape(Mfull, kd, *v.shape[1:])[js].reshape(len(js) * kd, *v.shape[1:])
    return out


def stage_eval60(a, env=None):
    out = Path(a.out); smoke = out.name == '_smoke'; hashes = dict(protocol_sha256=sha(PROTO), script_sha256=sha(SCRIPT))
    if not smoke: pr = json.loads((RES / 'PREREG.json').read_text()); assert pr['script_sha256'] == hashes['script_sha256']; hashes['prereg_sha256'] = sha(RES / 'PREREG.json')
    if not smoke:                                                     # budget guard (protocol 6): projected cost from the smoke timing, x1.1
        tm = pr['smoke_timing']; est = (61 * (tm['t_regress_20_participants_s'] * 105 / 20 + 6 * (0.5 * tm['t_fwd_M20_1000rows_s'] + 0.05)) + 90) * 1.1
        if ledger_cum() + est > CAP_S - RESERVE_S: log(f'STOP_CAP before eval60: cum {ledger_cum():.0f}s + projected {est:.0f}s > {CAP_S - RESERVE_S}'); sys.exit(9)
    env = env or Env(a.subjects); its = load_items(out); ids = [1, 9, 10, 6, 7, 8]; assert all(i in its for i in ids), f'missing items {set(ids) - set(its)}'
    sets, pools = make_sets(env, a.per_family); ts = time.time(); log(f'{len(sets) - 1} derivation sets + proxy')
    nets = {}
    for i in ids:                                                     # snapshots + final-member selection per item
        sn = torch.load(out / 'snap' / f'item_{i:02d}.pt'); groups = sn['groups']; mem, meta = make_members(env, groups); ch_ = []
        for cdef in sn['chunks']:
            a0, a1 = cdef['c0'], cdef['c0'] + cdef['M']; mm = mem[a0:a1]; le = max(len(b) for _, b in mm); evp = torch.zeros(len(mm), le, dtype=torch.long, device=DEV)
            for j, (_, b) in enumerate(mm): evp[j, :len(b)] = b; evp[j, len(b):] = b[0]
            ch_.append(dict(a0=a0, M=cdef['M'], evp=evp, states=cdef['states']))
        # sub-nets holding ONLY the final members, grouped by their selected checkpoint (inner members are not needed); built once, reused for every derivation set
        subs = []
        for ck in ch_:
            want = {}
            for j in range(ck['M']):
                if meta[ck['a0'] + j]['role'] == 'final': want.setdefault(its[i]['pairs'][(ck['a0'] + j) // 2]['ckpt'], []).append(j)
            for c, js in want.items():
                net = YNet(len(js), 560, 'mlp').to(DEV).eval(); net.load_state_dict({k: v.to(DEV) for k, v in slice_state(ck['states'][c], ck['M'], js).items()})
                subs.append(dict(net=net, evp=ck['evp'][js], members=[ck['a0'] + j for j in js]))
        nets[i] = dict(subs=subs, mem=mem, meta=meta, pairs=its[i]['pairs'], ytr=env.y, seed=its[i]['seed'])
    inputs = {}; res = dict(sets=[], items={str(i): {} for i in ids}, hashes=hashes, checks={})
    X_in = env.get('XIN')
    for sdef in [dict(name='X', fam='X', draw=0, D=None)] + sets:
        if sdef['name'] == 'X': Rin, share = X_in, 0.0
        elif sdef['name'] == 'proxy': Rin, share = env.get('RTEST'), env.share.get('R', float('nan'))
        else:
            R, share = RF.regress(env.Xb, env.sid, env.nP, sdef['D']); Rin, _ = env.win(R, env.sk); del R
        for i in ids:
            nd = nets[i]; bits = torch.full((env.nP,), float('nan'), dtype=torch.float64, device=DEV)
            for sb in nd['subs']:
                Z = fwd_rows(sb['net'], Rin, sb['evp'])
                for jj, g in enumerate(sb['members']):
                    pair = nd['pairs'][g // 2]; ev = nd['mem'][g][1]; Zf = Z[jj, :len(ev)]
                    prior = M.prior_of(nd['ytr'][nd['mem'][g][0]]); b = M.bits(Zf, env.y[ev], pair['T'], prior); m, ok = agg(b, ev, env); bits[ok] = m[ok]
            res['items'][str(i)][sdef['name']] = bits.cpu().tolist()
        res['sets'].append({k: v for k, v in sdef.items() if k != 'D'} | dict(var_share_removed=share))
        log(f'set {sdef["name"]} done ({time.time() - ts:.0f}s)')
        if sdef['name'] not in ('X', 'proxy'): del Rin
    # consistency checks of the snapshot path against the main-run scores (1e-4 bit on the cohort mean, max per-participant 1e-3)
    chk = {}
    for i in ids:
        for nm, key in (('X', 'bits'), ('proxy', 'rt_bits')):
            u = np.array(res['items'][str(i)][nm], float); v = np.array(its[i]['per'][key], float); ok = np.isfinite(u) & np.isfinite(v)
            chk[f'item{i}_{nm}'] = dict(mean_abs_diff=float(abs(u[ok].mean() - v[ok].mean())), max_abs_diff=float(np.abs(u[ok] - v[ok]).max()))
    res['checks'] = chk; res['snapshot_path_gate_pass'] = bool(all(c['mean_abs_diff'] < 1e-4 and c['max_abs_diff'] < 1e-3 for c in chk.values()))
    res['runtime_s'] = time.time() - ts; res['per_family'] = a.per_family; (out / 'eval60.json').write_text(json.dumps(res)); log('eval60 checks', json.dumps(chk), res['snapshot_path_gate_pass'])
    return res


# ---------------------------------------------------------------- summary (sections 8-10)
def stage_summary(a):
    out = Path(a.out); smoke = out.name == '_smoke'; its = load_items(out); S = dict(NOT_CITABLE='until blind review', out=str(out)); st_ = None
    hashes = dict(protocol_sha256=sha(PROTO), script_sha256=sha(SCRIPT)); S['hashes'] = hashes
    if not smoke: S['prereg'] = json.loads((RES / 'PREREG.json').read_text()); S['hashes']['prereg_sha256'] = sha(RES / 'PREREG.json')
    S['items_present'] = sorted(its)
    if 1 not in its: S['verdict'] = 'NOT TESTABLE WITHIN THE CAP'; (out / 'summary.json').write_text(json.dumps(S, indent=1)); return S
    A = lambda i, k: np.array(its[i]['per'][k], float) if i in its and k in its[i]['per'] else None
    valid = np.isfinite(A(1, 'bits')); nv = int(valid.sum()); st = Stats(nv); S['n_participants_scored'] = nv; S['participants'] = [p for p, v in zip(its[1]['participants'], valid) if v]
    V = lambda v: None if v is None else v[valid]
    def sa(ids, k):                                                    # seed-averaged per participant
        got = [V(A(i, k)) for i in ids if i in its and V(A(i, k)) is not None]; return (np.mean(got, 0), len(got)) if got else (None, 0)
    ids_p, ids_f, ids_r = [1, 9, 10], [6, 7, 8], [2, 11, 12]
    # ---- all arms table
    arms = {}
    for i, d in its.items():
        r = {}
        for k in d['per']:
            v = V(A(i, k))
            if v is not None and np.isfinite(v).all(): r[k] = st.mean(v)
        r['name'] = d['name']; r['seed'] = d['seed']; r['single_seed_label'] = 'single seed' if i not in (1, 2, 6, 7, 8, 9, 10, 11, 12) else ''
        r['penalty_ratio_step0_proxyQ_mean'] = float(np.mean(d['penalty_ratio_step0']['proxy'])); r['penalty_ratio_last_proxyQ_mean'] = float(np.mean(d['penalty_ratio_last']['proxy']))
        if 'own' in d['penalty_ratio_last']: r['penalty_ratio_last_ownQ_mean'] = float(np.mean(d['penalty_ratio_last']['own']))
        r['eval_participants_own_Bk_proxy_energy_mean'] = float(np.mean(d['eval_participants_own_Bk_proxy_energy']))
        r['edge_flags'] = dict(n_pairs=len(d['pairs']), ckpt_lo=sum(p['opt_lo'] for p in d['pairs']), ckpt_hi=sum(p['opt_hi'] for p in d['pairs']), T_edge=sum(p['T_edge'] for p in d['pairs']))
        r['loss_curve'] = d['curves'][::4]; r['finite'] = d['finite']; r['s_per_step'] = d['timing']['s_per_step']; r['item_wall_s'] = d['timing']['item_wall_s']; arms[str(i)] = r
    S['arms'] = arms
    # ---- Gate S (item 1)
    if 1 in its:
        inner_ok = None
        S['gate_S_item1_info'] = 'pairs with inner joint minimum before the last checkpoint: %d of %d' % (sum(p['ckpt_lt_last'] for p in its[1]['pairs']), len(its[1]['pairs']))
        S['gate_S'] = dict(pass_=sum(p['ckpt_lt_last'] for p in its[1]['pairs']) >= (8 if len(its[1]['pairs']) == 10 else math.ceil(0.8 * len(its[1]['pairs']))), n_ok=sum(p['ckpt_lt_last'] for p in its[1]['pairs']), n=len(its[1]['pairs']),
                           label_if_fail='undertrained at 2000 steps')
    # ---- G0 table (seed 11 decision + 3-seed report), HP4, context
    if 2 in its: S['G0_seed11'] = g0_decide(its, st)
    px, npx = sa(ids_p, 'bits'); prt, _ = sa(ids_p, 'rt_bits'); pr_, npr = sa(ids_r, 'bits'); fx, nfx = sa(ids_f, 'bits'); frt, _ = sa(ids_f, 'rt_bits')
    S['seeds_available'] = dict(plain_X=npx, plain_R=npr, full_X=nfx)
    if prt is not None and pr_ is not None:
        S['G0_table_seedavg'] = dict(plain=dict(Y_X=st.mean(px), Y_Rtest=st.mean(prt), Y_R=st.mean(pr_), K_inv=st.ratio(prt, px), K_aud=st.ratio(pr_, px), dY=st.mean(px - prt)),
                                     context=dict(ridge_Y_X=0.30, ridge_Y_Rtest=0.03, eegnet_4_40_Y_X=0.104, eegnet_4_40_Y_Rtest=0.098))
        if 5 in its:
            h, hr = V(A(5, 'bits')), V(A(5, 'rt_bits')); S['G0_table_seedavg']['HP4'] = dict(Y_X=st.mean(h), Y_Rtest_descriptive=st.mean(hr), K_inv=st.ratio(hr, h), single_seed=True,
                                                                                               note='R_test = bandpass(R,4,40), HP4 own scalar (descriptive extension)')
    # ---- seed noise null, ADD-VALUE
    if 9 in its:
        d = V(A(1, 'bits')) - V(A(9, 'bits')); S['seed_noise_null_plain_s11_minus_s12'] = dict(mean=float(d.mean()), se=float(d.std(ddof=1) / math.sqrt(nv)), ci=st.q(st.dist(d)))
    mdd = None
    if fx is not None and pr_ is not None and nfx == 3 and npr == 3:
        D = fx - pr_; se = float(D.std(ddof=1) / math.sqrt(nv)); mdd = 2.8 * se; Dci = st.mean(D)
        f1, p1 = sa(ids_f, 'bits1'), sa(ids_r, 'bits1'); D1 = f1[0] - p1[0]
        per_seed = {str(i): float((V(A(i, 'bits')) - pr_).mean()) for i in ids_f}
        S['ADD_VALUE'] = dict(D=Dci, SE=se, MDD_2p8SE=mdd, D_per_seed_vs_plainR_mean=per_seed, D_positive_each_seed=all(v > 0 for v in per_seed.values()), D_T1=st.mean(D1),
                              Y_X_full=st.mean(fx), Y_R_plainR=st.mean(pr_))
        dl = st.mean(fx - px) if px is not None else None
        S['secondary'] = dict(PASS_L=dict(dY_X=dl, flag=bool(dl['ci'][0] >= -0.02)) if dl else None)
        if frt is not None and prt is not None:
            dr = st.mean(frt - prt); Kg = st.d_ratio(frt, fx, prt, px); S['secondary']['PASS_R'] = dict(dY_Rtest=dr, flag=bool(dr['ci'][0] >= -0.02)); S['secondary']['K_inv_gain'] = Kg
            S['secondary']['K_inv_full'] = st.ratio(frt, fx); S['secondary']['K_inv_plain'] = st.ratio(prt, px)
        if 24 in its: S['secondary']['K_aud_full_item24_single_seed'] = st.ratio(V(A(24, 'bits')), V(A(6, 'bits')))
        if 5 in its: S['secondary']['Y_X_full_minus_HP4'] = st.mean(fx - V(A(5, 'bits'))); S['secondary']['not_beyond_band_limiting'] = bool(S['secondary']['Y_X_full_minus_HP4']['ci'][0] <= 0)
    # ---- floors G3
    fl = {}
    for i in (3, 4, 13, 14):
        if i in its: m = float(np.mean(V(A(i, 'bits')))); fl[str(i)] = dict(name=its[i]['name'], mean=m, ok=abs(m) <= 0.02, ci=st.q(st.dist(V(A(i, 'bits')))))
    S['G3_floors'] = dict(items=fl, pass_=bool(fl) and all(v['ok'] for v in fl.values()), complete=all(str(i) in fl for i in (3, 4, 13, 14)))
    # ---- planted G6
    pl = {}
    if 15 in its and 16 in its:
        p, f = V(A(15, 'bits')), V(A(16, 'bits')); rr = st.ratio(f, p)
        pl['rho0'] = dict(Y_plain=st.mean(p), Y_full=st.mean(f), retention=rr, informative=bool(0.1 <= p.mean() <= 1.5), pass_=bool(rr['point'] >= 0.8 and rr['ci'][0] >= 0.7))
    if 22 in its and 23 in its:
        p, f = V(A(22, 'bits')), V(A(23, 'bits')); rr = st.ratio(f, p)
        pl['rho0.9'] = dict(Y_plain=st.mean(p), Y_full=st.mean(f), retention=rr, informative=bool(0.1 <= p.mean() <= 1.5), manipulation_fail=bool(p.mean() <= 1.2 and rr['point'] >= 0.8))
    S['G6_planted'] = pl
    # ---- manipulation
    if 6 in its and 1 in its:
        rf, rp = float(np.mean(its[6]['penalty_ratio_last']['proxy'])), float(np.mean(its[1]['penalty_ratio_last']['proxy']))
        S['manipulation'] = dict(ratio_full_last=rf, ratio_plain_last=rp, ratio_over_plain=rf / rp, pass_=bool(rf <= 0.3 * rp), rule='full <= 0.3 x plain at step 2000 (item 6 vs item 1, all 20 members)')
    # ---- specificity A (eval-only) and held-out
    e60 = out / 'eval60.json'; spA = None
    if e60.exists():
        E = json.loads(e60.read_text()); S['eval60_checks'] = E['checks']; S['eval60_snapshot_path_gate_pass'] = E['snapshot_path_gate_pass']; rows = []
        names = [s_['name'] for s_ in E['sets'] if s_['name'] not in ('X', 'proxy')]
        PX, FX = px, fx                                                # main-run X bits (the eval-path 'X' set is only a consistency check)
        gain_proxy = float(frt.mean() / fx.mean() - prt.mean() / px.mean()) if (frt is not None) else None
        gains = []
        for s_ in E['sets']:
            if s_['name'] in ('X', 'proxy'): continue
            pr_s = np.mean([np.array(E['items'][str(i)][s_['name']], float)[valid] for i in ids_p], 0); fr_s = np.mean([np.array(E['items'][str(i)][s_['name']], float)[valid] for i in ids_f], 0)
            Kp, Kf = pr_s.mean() / PX.mean(), fr_s.mean() / FX.mean(); gains.append(Kf - Kp)
            rows.append(dict(set=s_['name'], family=s_['fam'], var_share_removed=s_['var_share_removed'], Y_Rs_plain=float(pr_s.mean()), Y_Rs_full=float(fr_s.mean()), K_inv_plain=float(Kp), K_inv_full=float(Kf), gain=float(Kf - Kp)))
        if gain_proxy is not None and gains:
            n_ge = int(sum(g >= gain_proxy for g in gains)); pval = (1 + n_ge) / (1 + len(gains)); spA = dict(gain_proxy=gain_proxy, n_sets=len(gains), n_ge_proxy=n_ge, p=pval,
                                                                                                          pass_=bool(pval <= 0.05 and len(gains) == 60), complete=len(gains) == 60, table=rows)
        S['specificity_A'] = spA
    # ---- specificity B
    spB = None
    if 'ADD_VALUE' in S and 18 in its and 19 in its:
        Dp_ = S['ADD_VALUE']['D']['mean']; spB = {}
        for i in (18, 19): dc = V(A(i, 'bits')) - pr_; spB[str(i)] = dict(D_control=st.mean(dc), ratio_to_D=float(dc.mean() / Dp_) if Dp_ > 0 else None, ok=bool(Dp_ > 0 and dc.mean() < 0.5 * Dp_), single_seed=True)
        spB['pass_'] = bool(Dp_ > 0 and all(spB[str(i)]['ok'] for i in (18, 19)))
    S['specificity_B'] = spB
    ho = None
    if 17 in its and 1 in its and 'rh_bits' in its[17]['per'] and 'rh_bits' in its[1]['per']:
        x1, h1, x17, h17 = V(A(1, 'bits')), V(A(1, 'rh_bits')), V(A(17, 'bits')), V(A(17, 'rh_bits')); Kp_ = st.ratio(h1, x1); dk = st.d_ratio(h17, x17, h1, x1)
        ho = dict(K_inv_plain_Rh=Kp_, K_inv_full3_Rh=st.ratio(h17, x17), dK_inv=dk, valid=bool(Kp_['ci'][1] <= 0.85), pass_=bool(dk['point'] > 0.05 and dk['ci'][0] > 0), single_seed=True,
                  note='validity uses the upper CI of K_inv(plain; R_h) <= 0.85 (stricter reading)')
    S['held_out'] = ho
    # ---- verdict
    S['verdict'], S['verdict_reasons'], S['flags'] = verdict(S, its, mdd)
    sm = (out / 'summary.json'); sm.write_text(json.dumps(S, indent=1))
    if not smoke:
        pp = out / 'per_participant'; pp.mkdir(exist_ok=True)
        for k, p in enumerate(its[1]['participants']):
            (pp / f'sub-{p:03d}.json').write_text(json.dumps(dict(participant=p, arms={str(i): {kk: its[i]['per'][kk][k] for kk in its[i]['per']} for i in its})))
    log('VERDICT', S['verdict'], S['verdict_reasons']); return S


def verdict(S, its, mdd):
    why = []; flags = []; have = lambda r: all(i in its for i in r)
    g0 = S.get('G0_seed11', {})
    if g0 and not g0.get('pass_', False): return 'DEAD-no-headroom', [json.dumps(g0.get('checks'))], flags
    if not have(range(1, 17)): return 'NOT TESTABLE WITHIN THE CAP', ['items 1-16 not all completed: ' + str(sorted(set(range(1, 17)) - set(its)))], flags
    g0 = S.get('G0_seed11', {})
    if not g0.get('pass_', False): return 'DEAD-no-headroom', [json.dumps(g0.get('checks'))], flags
    # gates
    gate_fail = []
    if S['G3_floors']['complete'] and not S['G3_floors']['pass_']: gate_fail.append('G3 floors')
    if not S['G3_floors']['complete']: gate_fail.append('G3 floors incomplete')
    pl = S.get('G6_planted', {}); unin = [k for k, v in pl.items() if not v['informative']]
    if 'rho0' in pl and pl['rho0']['informative'] and not pl['rho0']['pass_']: gate_fail.append('G6 rho=0 retention')
    gcode = json.loads((RES / 'PREREG.json').read_text())['smoke_gates']['G_code']['pass'] if (RES / 'PREREG.json').exists() else None
    if gcode is not True: gate_fail.append('G-code not recorded as passed')
    manip = S.get('manipulation', {}).get('pass_', False)
    if 'rho0.9' in pl and pl['rho0.9'].get('manipulation_fail'): manip = False; why.append('rho=0.9 manipulation fail')
    if not S['gate_S']['pass_']: flags.append('undertrained at 2000 steps')
    if 'ADD_VALUE' not in S: return 'INCONCLUSIVE', ['seed-averaged arms incomplete'], flags
    Dci = S['ADD_VALUE']['D']['ci']; flags.append(f'MDD(2.8 SE)={mdd:.4f}')
    hard = [g for g in gate_fail if g != 'G3 floors incomplete']
    if hard: return 'DEAD (gate failed: ' + '; '.join(hard) + ')', why, flags
    if unin: return 'INCONCLUSIVE', [f'planted cell uninformative: {unin}'] + why, flags
    if gate_fail or not manip: return 'INCONCLUSIVE', ['gates incomplete or manipulation not tested: ' + str(gate_fail) + f' manip={manip}'] + why, flags
    if Dci[1] < 0.02: return 'PREPROCESSING IS ENOUGH', [f'D upper CI {Dci[1]:.4f} < +0.02'], flags
    if Dci[0] > 0.02:
        if not have(range(17, 20)) or 'specificity_A' not in S or S['specificity_A'] is None or not S['specificity_A']['complete'] or S['held_out'] is None:
            return 'INCONCLUSIVE', ['lower CI of D > 0.02 but specificity/held-out controls missing'], flags
        if not S['held_out']['valid']: return 'INCONCLUSIVE', ['K headroom missing on R_h'], flags
        bad = []
        if not S['specificity_A']['pass_']: bad.append('specificity A')
        if not S['specificity_B']['pass_']: bad.append('specificity B')
        if not S['held_out']['pass_']: bad.append('held-out transfer')
        if bad: return 'DEAD (' + ', '.join(bad) + ' failed; no value claim)', [f'D lower CI {Dci[0]:.4f}'], flags
        if not S['ADD_VALUE']['D_positive_each_seed']: return 'INCONCLUSIVE', ['D not > 0 in each of the 3 seeds'], flags
        d1 = S['ADD_VALUE']['D_T1']
        if d1['mean'] <= 0: return 'INCONCLUSIVE', ['T=1 bits difference has the opposite sign'], flags
        if d1['ci'][0] <= 0: flags.append('calibration-sensitive')
        if S['secondary'].get('not_beyond_band_limiting'): flags.append('not beyond band-limiting')
        flags.append('PASS-L ' + str(S['secondary']['PASS_L']['flag'])); flags.append('PASS-R ' + str(S['secondary'].get('PASS_R', {}).get('flag')))
        if not S['eval60_snapshot_path_gate_pass']: return 'INCONCLUSIVE', ['snapshot path check failed'], flags
        if not have(range(1, 20)): return 'INCONCLUSIVE', ['items 1-19 incomplete'], flags
        return 'ADD-VALUE PASS (beats plain-R on raw input)', [f'D lower CI {Dci[0]:.4f} > 0.02'], flags
    return 'INCONCLUSIVE', [f'D CI {Dci} straddles +0.02'], flags


# ---------------------------------------------------------------- stamp
def stage_stamp(a):
    sm = RES / '_smoke' / 'smoke_report.json'; assert sm.exists(), 'no smoke report'; rep = json.loads(sm.read_text()); assert rep['all_pass'], 'smoke did not pass'
    assert not list((RES / 'items').glob('*.json')) if (RES / 'items').exists() else True, 'scored outputs already exist'
    tim = dict(rep['timing'])
    if a.timing_from:
        tf = json.loads(Path(a.timing_from).read_text()); tim.update(s_per_step_contended_attempt=tim['s_per_step'], detail_contended_attempt=tim['detail'], s_per_step=tf['s_per_step'], detail=tf['detail'],
                                                              s_per_step_source=a.timing_from, s_per_step_source_note='s_per_step of the smoke attempt measured on an exclusive GPU; the final smoke attempt overlapped with other steps on the compute node')
    pr = dict(protocol_sha256=sha(PROTO), script_sha256=sha(SCRIPT), time_ist=str(now_ist()), smoke_timing=tim, smoke_gates=rep['gates'], items_planned=ORDER_RUN,
              smoke_report=str(sm), note='stamped after a passing smoke, before any scored output')
    (RES / 'PREREG.json').write_text(json.dumps(pr, indent=1)); h = sha(RES / 'PREREG.json'); (RES / 'PREREG.sha256').write_text(f'{h}  PREREG.json\n'); log('PREREG', h)


# ---------------------------------------------------------------- smoke
def smoke_blocks(a):
    out = RES / '_smoke'; out.mkdir(parents=True, exist_ok=True); rep = dict(gates={}, hashes=dict(protocol_sha256=sha(PROTO), script_sha256=sha(SCRIPT)), time_ist=str(now_ist()))
    ts = time.time(); env = Env('first:20'); rep['setup_s'] = time.time() - ts; log(f'smoke cohort {env.nP}')
    groups = [0, 1]; mem, meta = make_members(env, groups)
    # ---- G-code: new function (head linear, alpha = mu = 0) vs mi_card_v3.eegnet_members, HP4 input, seed 11, 2 groups, 200 steps
    Xh = env.get('HP4'); y = env.y
    # attempt 1: mi_card_v3.eegnet_members is NOT bit-reproducible against itself under its cudnn.benchmark=True (non-deterministic conv kernels).
    # The gate therefore runs BOTH functions under cudnn.deterministic=True, benchmark=False (flags only; no code of either function changed); the
    # default-backend difference is reported as information.
    def gcode(tag):
        ref = M.eegnet_members(Xh, y, mem, steps=200, every=100, batch=64, mchunk=20, seed=11); rs_ref_c, rs_ref_g = torch.get_rng_state().clone(), torch.cuda.get_rng_state().clone()
        new = train_chunk(Xh, y, mem, 200, 100, 64, 11, 0, dict(alpha=0, mu=0, head='linear', seed=11), {'x': Xh}, blank_eval=False)
        eq = all(torch.equal(ref[j].cpu(), new['Z']['x'][j].cpu()) for j in range(len(mem))); md = max(float((ref[j] - new['Z']['x'][j]).abs().max()) for j in range(len(mem)))
        rng_eq = bool(torch.equal(rs_ref_c, new['rng_cpu']) and torch.equal(rs_ref_g, new['rng_cuda']))
        ref2 = M.eegnet_members(Xh, y, mem, steps=200, every=100, batch=64, mchunk=20, seed=11)
        selfeq = all(torch.equal(ref[j].cpu(), ref2[j].cpu()) for j in range(len(mem))); md_ref = max(float((ref[j] - ref2[j]).abs().max()) for j in range(len(mem)))
        return dict(pass_=bool(eq and md == 0.0), **{'pass': bool(eq and md == 0.0)}, max_abs_diff=md, global_rng_state_equal=rng_eq, v3_rerun_identical=selfeq,
                    v3_vs_v3_max_abs_diff=md_ref, steps=200, n_members=len(mem), backend=tag)
    bm, dt = torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic
    default_info = gcode('default (benchmark=True), information only') if not a.quick else dict(skipped='quick smoke; attempt 1 recorded max abs diff 2.5e-3 and v3 rerun not identical'); log('G-code default backend', default_info)
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    rep['gates']['G_code'] = gcode('cudnn.deterministic=True, benchmark=False'); torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic = bm, dt
    rep['G_code_default_backend_info'] = default_info
    log('G-code', rep['gates']['G_code']); env.cache.pop('HP4', None); env.cache.pop('RHP4', None)
    # ---- all items through the real code path (300 steps, groups 0 and 1)
    hashes = rep['hashes']; recs = {}
    for it in ([1, 6, 7, 8, 9, 10, 17] if a.quick else ORDER_RUN): recs[it] = run_item(env, it, out, 300, groups, hashes)
    rep['gates']['finite_loss'] = dict(pass_=all(r['finite'] for r in recs.values()), **{'pass': all(r['finite'] for r in recs.values())}, per_item={str(i): r['finite'] for i, r in recs.items()})
    fr0, frT = float(np.mean(recs[6]['penalty_ratio_step0']['proxy'])), float(np.mean(recs[6]['penalty_ratio_last']['proxy']))
    rep['gates']['penalty_fall'] = dict(**{'pass': bool(frT <= 0.5 * fr0)}, full_step0=fr0, full_last=frT, plain_step0=float(np.mean(recs[1]['penalty_ratio_step0']['proxy'])),
                                        plain_last=float(np.mean(recs[1]['penalty_ratio_last']['proxy'])), ratio_last_over_step0=frT / fr0)
    rep['smoke_items'] = {str(i): dict(name=r['name'], Y_X=float(np.nanmean(r['per']['bits'])), Y_X_T1=float(np.nanmean(r['per']['bits1'])), acc=float(np.nanmean(r['per']['acc'])),
                                       Y_Rtest=float(np.nanmean(r['per']['rt_bits'])) if 'rt_bits' in r['per'] else None, blank_bits=float(np.nanmean(r['per']['blank_bits'])),
                                       blank_kl_last=r['curves'][-1]['kl'], ce_first=r['curves'][0]['ce'], ce_last=r['curves'][-1]['ce'], pen_last=r['curves'][-1]['pen'],
                                       ratio0=float(np.mean(r['penalty_ratio_step0']['proxy'])), ratioT=float(np.mean(r['penalty_ratio_last']['proxy'])),
                                       own_Bk_energy=float(np.mean(r['eval_participants_own_Bk_proxy_energy'])), s_per_step=r['timing']['s_per_step']) for i, r in recs.items()}
    # ---- timing at M = 20, 200 steps, 4 arm types (cycled members, different inits; eval rows padded to 1000 like a full group)
    Xin = env.get('XIN'); env.need(['XIN', 'RTEST', 'RIN']); ev = dict(x=Xin, rt=env.get('RTEST'), ro=env.get('RIN')); m20 = [mem[j % 4] for j in range(20)]; mt = [meta[j % 4] for j in range(20)]
    Qp = env.Q('p5', mt); tim = {}
    for nm, arm in (('plain', 'plain'), ('penalty', 'pen'), ('blank', 'blank'), ('full', 'full')):
        al, mu, _ = ARMS[arm]; r = train_chunk(Xin, y, m20, 200, 100, 64, 11, 0, dict(alpha=al, mu=mu, head='mlp', seed=11), ev, Qp if mu > 0 else None, {'proxy': Qp},
                                              snapshot=True, blank_eval=True, pad_eval=1000)
        tim[nm] = dict(s_per_step=r['timing']['s_per_step'], eval_s_per_200=r['timing']['eval_s'], mem_gb=r['timing']['mem_gb'], finite=r['finite']); log('timing', nm, tim[nm])
    sps = {k: v['s_per_step'] for k, v in tim.items()}
    # forward cost for eval60: one M=20 forward of 1000 rows
    net = YNet(20, 560, 'mlp').to(DEV).eval(); evp = torch.arange(1000, device=DEV)[None].expand(20, -1).contiguous(); torch.cuda.synchronize(); tf = time.time()
    for _ in range(3): fwd_rows(net, Xin, evp)
    torch.cuda.synchronize(); t_fwd = (time.time() - tf) / 3; del net
    torch.cuda.synchronize(); tr_ = time.time(); Rr_, _ = RF.regress(env.Xb, env.sid, env.nP, env.Dp); Rw_, _ = env.win(Rr_, env.sk); torch.cuda.synchronize(); t_reg = time.time() - tr_; del Rr_, Rw_
    rep['timing'] = dict(s_per_step=sps, detail=tim, t_fwd_M20_1000rows_s=t_fwd, t_regress_20_participants_s=t_reg, setup_s_smoke20=rep['setup_s'])
    return env, out, rep, recs, sps, t_fwd, t_reg


def stage_smoke(a):
    env, out, rep, recs, sps, t_fwd, t_reg = smoke_blocks(a); ts = time.time()
    # ---- eval-only code paths (R-test is in the items; held-out rh in items 1 & 17; 60-set path on 1 draw per family)
    a2 = argparse.Namespace(out=str(out), subjects='first:20', per_family=1); t_e = time.time(); E = stage_eval60(a2, env); t_e = time.time() - t_e
    rep['gates']['code_paths'] = dict(**{'pass': bool(E['snapshot_path_gate_pass'])}, rtest_in_items=bool('rt_bits' in recs[1]['per']), heldout_in_items=bool('rh_bits' in recs[17]['per']),
                                      eval60_ran=True, eval60_sets=len(E['sets']), snapshot_path_checks=E['checks'])
    # ---- summary code path on the smoke items (not interpreted)
    a3 = argparse.Namespace(out=str(out)); S = stage_summary(a3); rep['gates']['summary_code_path'] = {'pass': 'verdict' in S}
    # ---- projection (protocol 7): t_c = s/step x 2000; items that fit in 10800 - 600 - smoke cost - estimated eval-only cost
    tc = {k: v * STEPS for k, v in sps.items()}; per_item = {i: tc[item_type(i)] * 1.1 for i in ORDER_RUN}
    sm_cost = rep['setup_s'] + (time.time() - t0); ev60_est = 61 * (t_reg * 105 / 20 + 6 * (0.5 * t_fwd + 0.05)) + 60; avail = CAP_S - RESERVE_S - sm_cost - ev60_est - 6 * 150
    cum, fit = 0.0, 0
    for i in ORDER_RUN:
        if cum + per_item[i] <= avail: cum += per_item[i]; fit = i
        else: break
    rep['projection'] = dict(t_c_s=tc, per_item_s_with_1p1=per_item, smoke_cost_s_est=sm_cost, eval60_est_s=ev60_est, step_setups_reserve_s=6 * 150, available_s=avail, items_fit_in_order=fit,
                             cum_to_fit_s=cum, sum_items_1_16_s=sum(per_item[i] for i in range(1, 17)), sum_items_1_19_s=sum(per_item[i] for i in range(1, 20)), sum_all_25_s=sum(per_item.values()))
    g = rep['gates']; rep['all_pass'] = bool(all(v['pass'] for v in g.values())); rep['smoke_wall_s'] = time.time() - t0
    (out / 'timing.json').write_text(json.dumps(rep['timing'], indent=1)); (out / 'smoke_report.json').write_text(json.dumps(rep, indent=1, default=str))
    log('SMOKE', 'PASS' if rep['all_pass'] else 'FAIL', json.dumps({k: v['pass'] for k, v in g.items()})); log('projection', json.dumps(rep['projection']))
    sys.exit(0 if rep['all_pass'] else 5)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--stage', required=True); ap.add_argument('--out', default=str(RES)); ap.add_argument('--items', default='1,2,3,4')
    ap.add_argument('--seed', type=int, default=11); ap.add_argument('--subjects', default=''); ap.add_argument('--steps', type=int, default=STEPS)
    ap.add_argument('--quick', action='store_true'); ap.add_argument('--timing_from', default=''); ap.add_argument('--groups', default=''); ap.add_argument('--force', action='store_true'); ap.add_argument('--per_family', type=int, default=20)
    a = ap.parse_args(); RES.mkdir(parents=True, exist_ok=True)
    {'smoke': stage_smoke, 'stamp': stage_stamp, 'run': stage_run, 'eval60': stage_eval60, 'summary': stage_summary}[a.stage](a)


if __name__ == '__main__': main()
