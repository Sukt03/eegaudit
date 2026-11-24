"""RW with decay of unchosen values plus a choice kernel (decaying choice-history trace), batched torch.
Values Q as in RWDecay; choice kernel CK_k <- CK_k + eta * (1[c=k] - CK_k) on valid trials; logits = beta*Q + w_ck*CK
(+ the one-trial stickiness kappa that models.run adds for every model). Records the RL prediction error 'pe'.
Parameters (z): logit_alpha, logit_d, log_beta, logit_eta, w_ck, kappa."""
import torch
import torch.nn.functional as F
from cogmodels.models import N_ARMS, _sel


class RWDChoiceKernel:
    name = "RWDCK"
    param_names = ["logit_alpha", "logit_d", "log_beta", "logit_eta", "w_ck", "kappa"]
    prior_mean = [0.0, -2.0, 2.0, -1.0, 0.0, 0.3]
    prior_sd = [1.5, 1.5, 0.6, 1.5, 1.0, 0.8]

    def params(self, z):
        return {"alpha": torch.sigmoid(z[:, 0]), "d": torch.sigmoid(z[:, 1]), "beta": torch.exp(z[:, 2]),
                "eta": torch.sigmoid(z[:, 3]), "wck": z[:, 4]}

    def init(self, B, dev):
        return {"Q": torch.full((B, N_ARMS), 0.5, device=dev), "CK": torch.zeros((B, N_ARMS), device=dev)}

    def pre(self, s, p):
        return s

    def logits(self, s, p):
        return p["beta"][:, None] * s["Q"] + p["wck"][:, None] * s["CK"]

    def update(self, s, p, c, r, valid):
        oh = F.one_hot(c, N_ARMS).float() * valid[:, None]
        pe = r - _sel(s["Q"], c)
        Q = s["Q"] + oh * p["alpha"][:, None] * pe[:, None]
        Q = Q + (1 - oh) * valid[:, None] * p["d"][:, None] * (0.5 - Q)
        CK = s["CK"] + valid[:, None] * p["eta"][:, None] * (F.one_hot(c, N_ARMS).float() - s["CK"])
        return {"Q": Q, "CK": CK}, {"pe": pe * valid}
