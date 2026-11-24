"""Apply the locked log-density contrast to two EEG cohorts and time-shift controls."""
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from common_log_yield import atomic_json, density_gain, folds, targets
from cogmodels import models as M
import deep_decoders_ds004595 as replication


def replication_targets():
    behavior = replication.B
    duration = int(behavior["T"])
    parameters = torch.tensor(np.load(ROOT / "results/ds004595/map_RWD.npy")).float()
    with torch.no_grad():
        history = replication.run_reset(M.EXTRA_MODELS["RWD"], parameters, torch.full((3, duration), 0.5),
                                        *(torch.tensor(behavior[k]) for k in ("resets","choice","reward","missing")),
                                        record=True)
        pe = history["q"]["pe"].numpy()
    ids = [int(s) for s in behavior["sub"]]
    def evaluate(subject, trials):
        i = ids.index(subject)
        rewards = behavior["reward"][i,trials]
        actions = np.eye(3)[behavior["choice"][i,trials]]
        time = trials / duration
        nuisance = np.column_stack((np.ones(len(trials)), rewards, actions, actions*rewards[:,None], time, time**2, time**3))
        y = replication.resid(pe[i,trials].astype(float), nuisance)
        return (y-y.mean())/(y.std()+1e-9)
    return evaluate, sorted(replication.SUBS)


def main():
    primary, ids, _, _ = targets()
    secondary, other_ids = replication_targets()
    datasets = {}
    for name, folder, evaluate, subjects in [
        ("ds003458","features",primary,ids),
        ("ds004595","features_ds004595",secondary,other_ids)
    ]:
        rows = []
        for s in subjects:
            with np.load(ROOT/f"results/{folder}/eegnet/sub-{s:03d}.npz") as cache:
                x, trial = cache["F"].astype(float), cache["tri"]
            assert x.shape == (len(trial),1) and (np.diff(trial)>0).all()
            y, fold = evaluate(s,trial), folds(len(trial))
            score = density_gain(y,x,fold)
            rolled = [density_gain(y,np.roll(x,round(len(y)*k/4),axis=0),fold) for k in (1,2,3)]
            rows.append(dict(subject=int(s),n=len(y),gain=score,shifted=rolled,margin=score-max(rolled)))
        datasets[name] = rows
    old=json.loads((ROOT/"outputs/AGENT_LOOP_20261120/results/common_log_yield.json").read_text())
    expected={r["subject"]:r["gain"] for r in old["behavioral"]["rows"]["eegnet"]}
    assert all(abs(r["gain"]-expected[r["subject"]])<1e-10 for r in datasets["ds003458"])
    assert [len(v) for v in datasets.values()]==[23,50]
    atomic_json(ROOT/"outputs/AGENT_LOOP_20261120/results/log_yield_replication.json",
                dict(step=os.getenv("SLURM_STEP_ID"),allocation=os.getenv("SLURM_JOB_ID"),datasets=datasets,
                     caveat="Time shifts are descriptive controls; whole-participant historical target fits and within-participant calibration."))
if __name__=="__main__":
    main()
