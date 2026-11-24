"""Tabular belief-based cognitive models for the ds003458 three-armed bandit.

All models are batched in torch so the same code simulates (GPU, many agents at once)
and evaluates the exact log-likelihood of observed choices (used for MAP baselines).

Models (unconstrained parameter vector z, priors are Gaussian in z):
  RW     Rescorla-Wagner + softmax                         z = [logit alpha, log beta, kappa]
  HGF    3-level binary HGF per arm (Mathys et al. 2011/14) z = [omega2, omega3, log beta, kappa]
  BAMDP  Bayes-adaptive Beta belief with forgetting (POMDP  z = [logit lambda, log phi, log beta, kappa]
         belief over drifting reward probs) + one-step
         knowledge-gradient value of information
All models add a choice-stickiness bonus kappa * 1[c = c_{t-1}] to the softmax logits.
"""
import math

import torch
import torch.nn.functional as F

N_ARMS = 3
OMEGA = 2 * 0.33 * math.pi * 0.025  # from Cavanagh's BEH_BANDIT.m


def reward_schedule(T=480, m=0.6, a=0.4, device="cpu"):
    """Reward probabilities (3, T) for bandits X, Y, Z exactly as in BEH_BANDIT.m (V1)."""
    t = torch.arange(1, T + 81, dtype=torch.float32, device=device)
    base = m + a * torch.cos(OMEGA * t)
    return torch.stack([base[0:T], base[40:40 + T], base[80:80 + T]]).clamp(0, 1)


def _sel(x, c):
    return (x * F.one_hot(c, N_ARMS)).sum(1)


class RW:
    name = "RW"
    param_names = ["logit_alpha", "log_beta", "kappa"]
    prior_mean = [0.0, 2.0, 0.3]
    prior_sd = [1.5, 0.6, 0.8]

    def params(self, z):
        return {"alpha": torch.sigmoid(z[:, 0]), "beta": torch.exp(z[:, 1])}

    def init(self, B, dev):
        return {"Q": torch.full((B, N_ARMS), 0.5, device=dev)}

    def pre(self, s, p):
        return s

    def logits(self, s, p):
        return p["beta"][:, None] * s["Q"]

    def update(self, s, p, c, r, valid):
        oh = F.one_hot(c, N_ARMS).float() * valid[:, None]
        pe = r - _sel(s["Q"], c)
        Q = s["Q"] + oh * p["alpha"][:, None] * pe[:, None]
        return {"Q": Q}, {"pe": pe * valid}


class HGF:
    name = "HGF"
    param_names = ["omega2", "omega3", "log_beta", "kappa"]
    prior_mean = [-3.0, -6.0, 2.0, 0.3]
    prior_sd = [1.5, 1.5, 0.6, 0.8]

    def params(self, z):
        return {"om2": z[:, 0], "om3": z[:, 1], "beta": torch.exp(z[:, 2])}

    def init(self, B, dev):
        o = torch.ones((B, N_ARMS), device=dev)
        return {"mu2": 0 * o, "sa2": o.clone(), "mu3": o.clone(), "sa3": o.clone()}

    def pre(self, s, p):
        s = dict(s)
        lv = s["mu3"] + p["om2"][:, None]
        s["v2"] = torch.exp(6.0 - F.softplus(6.0 - lv))          # smooth cap on volatility, keeps gradients
        s["sa2h"] = s["sa2"] + s["v2"]
        s["sa3h"] = (s["sa3"] + torch.exp(p["om3"])[:, None]).clamp(max=1e3)
        return s

    def logits(self, s, p):
        return p["beta"][:, None] * torch.sigmoid(s["mu2"])

    def update(self, s, p, c, r, valid):
        oh = F.one_hot(c, N_ARMS).float() * valid[:, None]
        m1 = torch.sigmoid(s["mu2"])
        d1 = r[:, None] - m1
        pi2h = 1.0 / s["sa2h"]
        pi2 = pi2h + m1 * (1 - m1)
        mu2n = s["mu2"] + d1 / pi2
        sa2n = 1.0 / pi2
        w2 = s["v2"] * pi2h
        r2 = (s["v2"] - s["sa2"]) * pi2h
        d2 = (sa2n + (mu2n - s["mu2"]) ** 2) * pi2h - 1
        pi3 = (1.0 / s["sa3h"] + 0.5 * w2 * (w2 + r2 * d2)).clamp(min=1e-3)
        mu3n = s["mu3"] + 0.5 * w2 * d2 / pi3
        new = {
            "mu2": oh * mu2n + (1 - oh) * s["mu2"],
            "sa2": oh * sa2n + (1 - oh) * s["sa2h"],
            "mu3": (oh * mu3n + (1 - oh) * s["mu3"]).clamp(-8, 8),
            "sa3": oh * (1.0 / pi3) + (1 - oh) * s["sa3h"],
        }
        q = {"pe": _sel(d1, c) * valid, "eps2": _sel(d1 / pi2, c) * valid,
             "sa2": _sel(s["sa2h"], c), "mu3": _sel(s["mu3"], c)}
        return new, q


class BAMDP:
    name = "BAMDP"
    param_names = ["logit_lambda", "log_phi", "log_beta", "kappa"]
    prior_mean = [2.0, 1.5, 2.0, 0.3]
    prior_sd = [1.0, 1.0, 0.6, 0.8]

    def params(self, z):
        return {"lam": torch.sigmoid(z[:, 0]), "phi": torch.exp(z[:, 1]), "beta": torch.exp(z[:, 2])}

    def init(self, B, dev):
        o = torch.ones((B, N_ARMS), device=dev)
        return {"a": o.clone(), "b": o.clone()}

    def pre(self, s, p):
        lam = p["lam"][:, None]
        return {"a": lam * s["a"] + (1 - lam), "b": lam * s["b"] + (1 - lam)}

    @staticmethod
    def kg(a, b):
        n = a + b
        m = a / n
        up, dn = (a + 1) / (n + 1), a / (n + 1)
        top = m.topk(2, dim=1).values
        best, second = top[:, :1], top[:, 1:2]
        bo = torch.where(m >= best, second.expand_as(m), best.expand_as(m))
        kg = m * torch.maximum(up, bo) + (1 - m) * torch.maximum(dn, bo) - torch.maximum(m, bo)
        return m, kg

    def logits(self, s, p):
        m, kg = self.kg(s["a"], s["b"])
        return p["beta"][:, None] * (m + p["phi"][:, None] * kg)

    def update(self, s, p, c, r, valid):
        m, kg = self.kg(s["a"], s["b"])
        n = s["a"] + s["b"]
        oh = F.one_hot(c, N_ARMS).float() * valid[:, None]
        new = {"a": s["a"] + oh * r[:, None], "b": s["b"] + oh * (1 - r[:, None])}
        q = {"pe": (r - _sel(m, c)) * valid, "sd": _sel(torch.sqrt(m * (1 - m) / (n + 1)), c),
             "kg": _sel(kg, c)}
        return new, q


MODELS = {"RW": RW(), "HGF": HGF(), "BAMDP": BAMDP()}


def prior(model, n, dev):
    mu = torch.tensor(model.prior_mean, device=dev)
    sd = torch.tensor(model.prior_sd, device=dev)
    return mu + sd * torch.randn(n, len(mu), device=dev)


def run(model, z, probs, choices=None, rewards=None, missing=None, p_miss=0.02, record=False, count=None):
    """Simulate (choices is None) or evaluate log-likelihood of observed data.
    choices: (B,T) long in {0,1,2} (any value where missing), rewards (B,T) float, missing (B,T) bool."""
    B, T, dev = z.shape[0], probs.shape[1], z.device
    p = model.params(z)
    s = model.init(B, dev)
    simulate = choices is None
    if simulate:
        missing = torch.rand(B, T, device=dev) < p_miss
        choices = torch.zeros(B, T, dtype=torch.long, device=dev)
        rewards = torch.zeros(B, T, device=dev)
    ll = torch.zeros(B, device=dev)
    rec = {}
    kappa = z[:, -1:]                      # choice stickiness (all models): bonus for repeating last choice
    prev = torch.zeros(B, N_ARMS, device=dev)
    for t in range(T):
        s = model.pre(s, p)
        lg = model.logits(s, p) + kappa * prev
        valid = (~missing[:, t]).float()
        if simulate:
            c = torch.multinomial(F.softmax(lg, -1), 1).squeeze(1)
            r = (torch.rand(B, device=dev) < probs[c, t]).float()
            c = torch.where(valid.bool(), c, torch.zeros_like(c))
            r = r * valid
            choices[:, t], rewards[:, t] = c, r
        else:
            c, r = choices[:, t].clamp(min=0), rewards[:, t]
        ll = ll + valid * (1.0 if count is None else count[:, t].float()) * _sel(F.log_softmax(lg, -1), c)
        # model log-probability, before this choice, of repeating the previous choice (0 on the first trial)
        lp_rep = _sel(F.log_softmax(lg, -1), prev.argmax(1)) * (prev.sum(1) > 0).float()
        prev = F.one_hot(c, N_ARMS).float() * valid[:, None] + prev * (1 - valid[:, None])
        s, q = model.update(s, p, c, r, valid)
        if record:
            q = dict(q)
            q["logp_rep"] = lp_rep
            lsm = F.log_softmax(lg, -1)
            q["entropy"] = -(lsm.exp() * lsm).sum(-1)              # choice difficulty (RT read-out latent)
            for k, v in q.items():
                rec.setdefault(k, []).append(v)
    out = {"choices": choices, "rewards": rewards, "missing": missing, "loglik": ll}
    if record:
        out["q"] = {k: torch.stack(v, 1) for k, v in rec.items()}
    return out


def features(choices, rewards, missing):
    """Per-trial network input (B,T,5): one-hot choice, signed reward, missing flag."""
    v = (~missing).float()[..., None]
    oh = F.one_hot(choices.clamp(min=0), N_ARMS).float() * v
    return torch.cat([oh, (2 * rewards[..., None] - 1) * v, 1 - v], -1)


# approximation, not a solved BAMDP. Both names refer to the same class; results files keep the "BAMDP" key.
KGBA = BAMDP


class RWDecay(RW):
    """Rescorla-Wagner with decay of unchosen values toward 0.5 at rate d (forgetting control for KG-BA)."""
    name = "RWD"
    param_names = ["logit_alpha", "logit_d", "log_beta", "kappa"]
    prior_mean = [0.0, -2.0, 2.0, 0.3]
    prior_sd = [1.5, 1.5, 0.6, 0.8]

    def params(self, z):
        return {"alpha": torch.sigmoid(z[:, 0]), "d": torch.sigmoid(z[:, 1]), "beta": torch.exp(z[:, 2])}

    def update(self, s, p, c, r, valid):
        oh = F.one_hot(c, N_ARMS).float() * valid[:, None]
        pe = r - _sel(s["Q"], c)
        Q = s["Q"] + oh * p["alpha"][:, None] * pe[:, None]
        Q = Q + (1 - oh) * valid[:, None] * p["d"][:, None] * (0.5 - Q)
        return {"Q": Q}, {"pe": pe * valid}


class BAMDPNoBonus(BAMDP):
    """KG-BA with phi = 0: Beta beliefs with forgetting, no value-of-information bonus."""
    name = "BAMDP0"
    param_names = ["logit_lambda", "log_beta", "kappa"]
    prior_mean = [2.0, 2.0, 0.3]
    prior_sd = [1.0, 0.6, 0.8]

    def params(self, z):
        return {"lam": torch.sigmoid(z[:, 0]), "phi": torch.zeros_like(z[:, 0]), "beta": torch.exp(z[:, 1])}


EXTRA_MODELS = {"RWD": RWDecay(), "BAMDP0": BAMDPNoBonus()}
