"""Two-process learner for the 3-armed restless bandit: slow incremental RL plus a fast, decaying memory.

Inspired by RL + working-memory accounts (Collins & Frank 2018) and memory-rich reward learning (Eckstein et al. 2026).
Per trial, with chosen arm c and outcome r in {0,1}:
  RL values  Q_c += alpha_RL (r - Q_c);  unchosen Q_k += d (0.5 - Q_k)            (as RW with decay, RWD)
  memory     M_c  = r (one-shot, learning rate 1); unchosen M_k += phi (0.5 - M_k) (decays toward 0.5 per trial)
  choice     softmax(beta * (w M + (1 - w) Q) + kappa * [k == previous choice])
Recorded quantities: 'pe' = RL prediction error r - Q_c (the RL component; name kept so the existing pipeline uses it),
'pe_mix' = r - (w M_c + (1 - w) Q_c) (the prediction error of the mixed choice value), 'pe_mem' = r - M_c.
Parameters (unconstrained): logit_alpha (alpha_RL), logit_d, logit_w, logit_phi, log_beta, kappa (stickiness, last,
as required by cogmodels.models.run).
"""
import torch
import torch.nn.functional as F

from cogmodels.models import N_ARMS, _sel


class TwoProcess:
    name = "TP"
    param_names = ["logit_alpha", "logit_d", "logit_w", "logit_phi", "log_beta", "kappa"]
    prior_mean = [0.0, -2.0, 0.0, -1.0, 2.0, 0.3]
    prior_sd = [1.5, 1.5, 1.5, 1.5, 0.6, 0.8]

    def params(self, z):
        return {"alpha": torch.sigmoid(z[:, 0]), "d": torch.sigmoid(z[:, 1]), "w": torch.sigmoid(z[:, 2]),
                "phi": torch.sigmoid(z[:, 3]), "beta": torch.exp(z[:, 4])}

    def init(self, B, dev):
        return {"Q": torch.full((B, N_ARMS), 0.5, device=dev), "M": torch.full((B, N_ARMS), 0.5, device=dev)}

    def pre(self, s, p):
        return s

    def logits(self, s, p):
        V = p["w"][:, None] * s["M"] + (1 - p["w"][:, None]) * s["Q"]
        return p["beta"][:, None] * V

    def update(self, s, p, c, r, valid):
        oh = F.one_hot(c, N_ARMS).float() * valid[:, None]
        q_c, m_c = _sel(s["Q"], c), _sel(s["M"], c)
        pe = r - q_c
        pe_mix = r - (p["w"] * m_c + (1 - p["w"]) * q_c)
        pe_mem = r - m_c
        Q = s["Q"] + oh * p["alpha"][:, None] * pe[:, None]
        Q = Q + (1 - oh) * valid[:, None] * p["d"][:, None] * (0.5 - Q)
        M = s["M"] * (1 - oh) + oh * r[:, None]
        M = M + (1 - oh) * valid[:, None] * p["phi"][:, None] * (0.5 - M)
        return {"Q": Q, "M": M}, {"pe": pe * valid, "pe_mix": pe_mix * valid, "pe_mem": pe_mem * valid}
