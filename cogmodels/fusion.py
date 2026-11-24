"""Fusion summary networks for joint behaviour + neural read-out posteriors (encoder ablation).

All take per-trial input x (B, T, F) laid out as [behaviour 5 | read-out y | rho | optional interaction features] and
return a summary vector (B, out). The flow head is cogmodels.sbi.FlowPosterior; only .summ is replaced.
  early     one bidirectional GRU over the concatenated per-trial features (fusion before pooling)
  late      separate GRUs for behaviour and read-out, each mean/max pooled, then concatenated (Schmitt et al. 2023)
  inter     early fusion with explicit trial-aligned interaction features y*outcome, y*arm, y*arm*outcome
  late_inter late fusion whose read-out branch receives the interaction features (behaviour enters the neural branch
            only through per-trial products)
  xattn     read-out tokens cross-attend to GRU-contextualized behaviour tokens (no pooling before fusion)
  yonly     read-out stream alone (no behaviour): estimator of I(theta; y)
"""
import math

import torch
import torch.nn as nn

from cogmodels.sbi import Summary


def interaction_features(X, y):
    """X (B,T,5) [onehot3, signed reward, missing]; y (B,T,1) -> (B,T,7): y*r, y*onehot(3), y*onehot*r(3)."""
    oh, r = X[..., :3], X[..., 3:4]
    return torch.cat([y * r, y * oh, y * oh * r], -1)


class LateSummary(nn.Module):
    def __init__(self, d_beh, d_neu, out=128):
        super().__init__()
        self.d_beh = d_beh
        self.sb, self.sn = Summary(d_beh, out=out // 2), Summary(d_neu + 1, out=out // 2)

    def forward(self, x):
        xb = x[..., :self.d_beh]
        xn = torch.cat([x[..., self.d_beh:], x[..., 4:5]], -1)      # read-out stream + missing flag
        return torch.cat([self.sb(xb), self.sn(xn)], -1)


def _pos(T, d, device):
    p = torch.arange(T, device=device, dtype=torch.float32)[:, None]
    i = torch.arange(0, d, 2, device=device, dtype=torch.float32)[None]
    ang = p / (10000 ** (i / d))
    pe = torch.zeros(T, d, device=device)
    pe[:, 0::2], pe[:, 1::2] = torch.sin(ang), torch.cos(ang)
    return pe


class XAttnSummary(nn.Module):
    """Read-out tokens (y_t, rho) query behaviour tokens contextualized by a bidirectional GRU."""

    def __init__(self, d=64, heads=4, out=128):
        super().__init__()
        self.gru = nn.GRU(5, d // 2, num_layers=1, batch_first=True, bidirectional=True)
        self.tn = nn.Linear(2, d)
        self.att = nn.MultiheadAttention(d, heads, batch_first=True)
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, 2 * d), nn.SiLU(), nn.Linear(2 * d, d))
        self.sb = Summary(5, out=out // 2)
        self.head = nn.Sequential(nn.Linear(2 * d, out // 2), nn.SiLU(), nn.Linear(out // 2, out // 2))
        self.d = d

    def forward(self, x):
        B, T, _ = x.shape
        xb, yr = x[..., :5], x[..., 5:7]
        valid = (x[..., 4] < 0.5) & (yr[..., 1] > -1)                 # observed trials
        pe = _pos(T, self.d, x.device)[None]
        kb, _ = self.gru(xb)
        kb = kb + pe
        q = self.tn(yr) + pe
        h, _ = self.att(q, kb, kb, key_padding_mask=~valid, need_weights=False)
        h = self.ln1(q + h)
        h = self.ln2(h + self.ff(h))
        m = valid.float()[..., None]
        mean = (h * m).sum(1) / m.sum(1).clamp(min=1)
        mx = (h - 1e4 * (1 - m)).max(1).values
        return torch.cat([self.sb(xb), self.head(torch.cat([mean, mx], -1))], -1)


def build(arch, D):
    """Returns (FlowPosterior with the arch's summary, input builder fn(X, y, rho) -> x)."""
    from cogmodels.sbi import FlowPosterior
    if arch == "early":
        net = FlowPosterior(7, D)
        fx = lambda X, y, r: torch.cat([X, y, r], -1)
    elif arch == "late":
        net = FlowPosterior(7, D); net.summ = LateSummary(5, 2)
        fx = lambda X, y, r: torch.cat([X, y, r], -1)
    elif arch == "inter":
        net = FlowPosterior(14, D)
        fx = lambda X, y, r: torch.cat([X, y, r, interaction_features(X, y)], -1)
    elif arch == "late_inter":
        net = FlowPosterior(14, D); net.summ = LateSummary(5, 9)
        fx = lambda X, y, r: torch.cat([X, y, r, interaction_features(X, y)], -1)
    elif arch == "xattn":
        net = FlowPosterior(7, D); net.summ = XAttnSummary()
        fx = lambda X, y, r: torch.cat([X, y, r], -1)
    elif arch == "yonly":
        net = FlowPosterior(3, D)
        fx = lambda X, y, r: torch.cat([y, r, torch.zeros_like(y)], -1)
    else:
        raise ValueError(arch)
    return net, fx
