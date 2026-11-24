"""Synthetic interactive-POMDP (level-1 theory of mind) agent in a repeated hide-and-seek game.

The agent (seeker) wins when it matches the opponent's action. The opponent (hider) is a
level-0 Rescorla-Wagner learner with an unknown learning rate drawn from a grid. The agent
keeps a belief over the opponent's type (grid) and simulates the opponent's Q-values under
each hypothesis (tabular I-POMDP belief). Belief updates are noisy (sigma_b), so the marginal
likelihood of the agent's choices integrates over latent belief paths and is intractable:
this is the case amortized simulation-based inference is meant for.

z = [logit w (weight on ToM prediction vs own RW), log beta, log sigma_b, logit alpha_self]
"""
import torch
import torch.nn.functional as F

GRID = torch.linspace(0.05, 0.95, 8)
BETA_OPP = 6.0


class IPOMDP:
    name = "IPOMDP"
    param_names = ["logit_w", "log_beta", "log_sigma_b", "logit_alpha_self"]
    prior_mean = [0.0, 1.5, -1.0, 0.0]
    prior_sd = [1.5, 0.6, 1.0, 1.5]


def simulate(z, T=150):
    B, dev = z.shape[0], z.device
    w, beta = torch.sigmoid(z[:, 0])[:, None], torch.exp(z[:, 1])[:, None]
    sb, als = torch.exp(z[:, 2])[:, None], torch.sigmoid(z[:, 3])[:, None]
    g = GRID.to(dev)
    alpha_o = g[torch.randint(0, len(g), (B,), device=dev)][:, None]
    Qo = torch.full((B, 2), 0.5, device=dev)
    Qg = torch.full((B, len(g), 2), 0.5, device=dev)
    logw = torch.zeros(B, len(g), device=dev)
    Qs = torch.full((B, 2), 0.5, device=dev)
    feats = torch.zeros(B, T, 5, device=dev)
    for t in range(T):
        o = torch.multinomial(F.softmax(BETA_OPP * Qo, -1), 1).squeeze(1)
        pg = F.softmax(BETA_OPP * Qg, -1)                      # (B,G,2)
        ppred = (F.softmax(logw, -1)[..., None] * pg).sum(1)   # (B,2)
        V = w * ppred + (1 - w) * Qs
        a = torch.multinomial(F.softmax(beta * V, -1), 1).squeeze(1)
        win = (a == o).float()[:, None]
        oho, oha = F.one_hot(o, 2).float(), F.one_hot(a, 2).float()
        Qo = Qo + oho * alpha_o * ((1 - win) - Qo)
        lik = pg.gather(2, o[:, None, None].expand(B, len(g), 1)).squeeze(2)
        logw = logw + torch.log(lik + 1e-6) + sb * torch.randn(B, len(g), device=dev)
        logw = logw - logw.logsumexp(1, keepdim=True)
        Qg = Qg + oho[:, None, :] * g[None, :, None] * ((1 - win)[:, :, None] - Qg)
        Qs = Qs + oha * als * (win - Qs)
        feats[:, t] = torch.cat([oha, oho, 2 * win - 1], -1)
    return feats
