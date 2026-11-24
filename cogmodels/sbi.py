"""Amortized simulation-based inference: GRU summary network + conditional flow matching."""
import time

import torch
import torch.nn as nn


class Summary(nn.Module):
    def __init__(self, d_in, h=128, out=128):
        super().__init__()
        self.gru = nn.GRU(d_in, h, num_layers=2, batch_first=True, bidirectional=True)
        self.head = nn.Sequential(nn.Linear(4 * h, out), nn.SiLU(), nn.Linear(out, out))

    def forward(self, x):
        o, _ = self.gru(x)
        return self.head(torch.cat([o.mean(1), o.max(1).values], -1))


class FlowPosterior(nn.Module):
    """q(theta | x) as a continuous normalizing flow trained by conditional flow matching."""

    def __init__(self, d_in, D, h=256, sdim=128):
        super().__init__()
        self.D = D
        self.summ = Summary(d_in, out=sdim)
        self.vel = nn.Sequential(
            nn.Linear(D + 1 + sdim, h), nn.SiLU(), nn.Linear(h, h), nn.SiLU(),
            nn.Linear(h, h), nn.SiLU(), nn.Linear(h, D))

    def loss(self, theta, x):
        e = self.summ(x)
        th0 = torch.randn_like(theta)
        t = torch.rand(theta.shape[0], 1, device=theta.device)
        tht = (1 - t) * th0 + t * theta
        v = self.vel(torch.cat([tht, t, e], -1))
        return ((v - (theta - th0)) ** 2).mean()

    @torch.no_grad()
    def sample(self, x, n=500, steps=64, chunk=64):
        outs = []
        for i in range(0, x.shape[0], chunk):
            e = self.summ(x[i:i + chunk]).repeat_interleave(n, 0)
            th = torch.randn(e.shape[0], self.D, device=e.device)
            for k in range(steps):
                t = torch.full((e.shape[0], 1), k / steps, device=e.device)
                th = th + self.vel(torch.cat([th, t, e], -1)) / steps
            outs.append(th.view(-1, n, self.D))
        return torch.cat(outs)


class Classifier(nn.Module):
    def __init__(self, d_in, K):
        super().__init__()
        self.summ = Summary(d_in)
        self.out = nn.Linear(128, K)

    def forward(self, x):
        return self.out(self.summ(x))


def train(net, batch_loss, n_bank, steps, bs=256, lr=1e-3, log=print, tag=""):
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.05)
    net.train()
    t0, run = time.time(), 0.0
    for i in range(steps):
        idx = torch.randint(0, n_bank, (bs,))
        loss = batch_loss(idx)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sched.step()
        run = 0.98 * run + 0.02 * loss.item() if i else loss.item()
        if (i + 1) % max(1, steps // 10) == 0:
            log(f"  [{tag}] step {i+1}/{steps} loss {run:.4f} ({time.time()-t0:.0f}s)")
    net.eval()
    return time.time() - t0


def recovery_metrics(z_true, samples, names, prior_sd=None):
    """Per parameter: Pearson r (true vs posterior mean), RMSE, central-interval coverage at 50/90/95%,
    posterior contraction 1 - Var_post/Var_prior, and SBC rank uniformity (KS distance of normalized ranks)."""
    mean = samples.mean(1)
    res = {}
    S = samples.shape[1]
    for j, nm in enumerate(names):
        a, b = z_true[:, j], mean[:, j]
        r = torch.corrcoef(torch.stack([a, b]))[0, 1].item()
        d = {"r": round(r, 3), "rmse": round((a - b).pow(2).mean().sqrt().item(), 3)}
        for lev in (0.5, 0.9, 0.95):
            lo = samples[:, :, j].quantile((1 - lev) / 2, dim=1)
            hi = samples[:, :, j].quantile(1 - (1 - lev) / 2, dim=1)
            d[f"cov{int(lev*100)}"] = round(((a >= lo) & (a <= hi)).float().mean().item(), 3)
        if prior_sd is not None:
            d["contraction"] = round(1 - (samples[:, :, j].var(1).mean() / float(prior_sd[j]) ** 2).item(), 3)
        u = ((samples[:, :, j] < a[:, None]).float().sum(1) / S).sort().values.cpu()
        ecdf = torch.arange(1, len(u) + 1) / len(u)
        d["sbc_ks"] = round((ecdf - u).abs().max().item(), 3)
        res[nm] = d
    return res


class LateFusionPosterior(FlowPosterior):
    """Flow posterior with separate summary networks for behaviour (first d_beh features) and neural read-outs
    (remaining features), concatenated before the velocity field (late fusion; Schmitt et al. 2023)."""

    def __init__(self, d_beh, d_neu, D, h=256, sdim=128):
        super().__init__(d_beh + d_neu, D, h=h, sdim=sdim)
        self.d_beh = d_beh
        self.summ = _Fused(Summary(d_beh, out=sdim // 2), Summary(d_neu + 1, out=sdim // 2), d_beh)


class _Fused(nn.Module):
    def __init__(self, sb, sn, d_beh):
        super().__init__()
        self.sb, self.sn, self.d_beh = sb, sn, d_beh

    def forward(self, x):
        xb = x[..., :self.d_beh]
        xn = torch.cat([x[..., self.d_beh:], x[..., 4:5]], -1)   # neural read-outs + missing flag
        return torch.cat([self.sb(xb), self.sn(xn)], -1)
