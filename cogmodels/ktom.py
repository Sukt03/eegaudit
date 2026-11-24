"""Recursive theory of mind (k-ToM; Devaine, Hollard & Daunizeau 2014) for repeated 2-action games.

Correctness-first: the belief dynamics are the reference implementation (tomsup.ktom_functions, a port of the
original VBA toolbox code). This module adds what fitting needs:
  * `KToM.loglik(self_choices, op_choices)`  exact log-likelihood of a player's choices given level k and parameters
  * `simulate(level, params, opponent, n_rounds)`  a game against a scripted or k-ToM opponent
  * `Opponent` helpers for the bots used in human experiments (random-bias, 0/1/2-ToM)
Parameters (tomsup conventions): volatility (log), b_temp (log temperature), optional bias.
Games: matching pennies / hide-and-seek ("penny_competitive" for the hider vs "penny_cooperative" is not used).
"""
import copy

import numpy as np
from tomsup import ktom_functions as kf
from tomsup.payoffmatrix import PayoffMatrix

GAME = "penny_competitive"


class KToM:
    def __init__(self, level, volatility=-2.0, b_temp=-1.0, bias=0.0, agent=0, game=GAME):
        self.level = int(level)
        self.params = {"volatility": float(volatility), "b_temp": float(b_temp)}
        if bias is not None:
            self.params["bias"] = float(bias)
        self.agent = agent
        self.pm = PayoffMatrix(name=game)
        self.reset()

    def reset(self):
        self.state = kf.init_k_tom(self.params, self.level)

    def step(self, prev_self, prev_op):
        """Update on the previous round (None on the first round), return P(choose 1) for this round."""
        if prev_self is not None:
            self.state = kf.learning_function(self.state, self.params, prev_self, prev_op, self.level,
                                              self.agent, self.pm)
        lp, _ = kf.decision_function(self.state, self.params, self.agent, self.level, self.pm)
        return float(np.clip(1 / (1 + np.exp(-lp)), 1e-3, 1 - 1e-3))

    def loglik(self, self_choices, op_choices, return_p=False):
        """Exact log-likelihood of observed choices (arrays of 0/1, NaN/-1 = missed round, no update)."""
        self.reset()
        ll, ps, ps_, po_ = 0.0, [], None, None
        for s, o in zip(self_choices, op_choices):
            p = self.step(ps_, po_)
            ps.append(p)
            if s in (0, 1):
                ll += np.log(p if s == 1 else 1 - p)
                ps_, po_ = int(s), int(o)
            else:
                ps_, po_ = None, None
        return (ll, np.array(ps)) if return_p else ll


class RandomBias:
    """Bot that chooses 1 with a fixed probability."""

    def __init__(self, p=0.65, rng=None):
        self.p, self.rng = p, rng or np.random.default_rng()

    def reset(self):
        pass

    def choose(self, prev_self, prev_op):
        return int(self.rng.random() < self.p)


class KToMBot:
    """k-ToM opponent (plays from the other side of the payoff matrix)."""

    def __init__(self, level, volatility=-2.0, b_temp=-1.0, rng=None):
        self.agent = KToM(level, volatility, b_temp, agent=1)
        self.rng = rng or np.random.default_rng()

    def reset(self):
        self.agent.reset()

    def choose(self, prev_self, prev_op):
        p = self.agent.step(prev_self, prev_op)
        return int(self.rng.random() < p)


def simulate(level, volatility, b_temp, opponent, n_rounds=60, rng=None, bias=0.0):
    """Simulate a k-ToM player (agent 0) against `opponent` (agent 1). Returns (self_choices, op_choices, p_self)."""
    rng = rng or np.random.default_rng()
    me = KToM(level, volatility, b_temp, bias=bias, agent=0)
    opponent.reset()
    s_prev = o_prev = None
    S, O, P = [], [], []
    for _ in range(n_rounds):
        p = me.step(s_prev, o_prev)
        s = int(rng.random() < p)
        o = opponent.choose(o_prev, s_prev)          # opponent's own previous choice first
        S.append(s); O.append(o); P.append(p)
        s_prev, o_prev = s, o
    return np.array(S), np.array(O), np.array(P)


def payoff(s, o, agent=0, game=GAME):
    pm = PayoffMatrix(name=game)
    return np.array([pm.payoff(int(a), int(b), agent) for a, b in zip(s, o)])
