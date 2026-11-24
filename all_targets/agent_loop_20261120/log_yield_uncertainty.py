"""Participant bootstrap of the exploratory score and its paired controls."""
import json
from pathlib import Path

import numpy as np


def main():
    base = Path(__file__).resolve().parents[2]/"outputs/AGENT_LOOP_20261120/results"
    def read(file):
        return json.loads((base/file).read_text())
    common = read("common_log_yield.json")
    replication = read("log_yield_replication.json")
    summaries = {}
    rng = np.random.default_rng(20261120)

    def summarize(name, scores, floors):
        x, c = np.asarray(scores), np.asarray(floors)
        assert x.shape == c.shape and x.ndim == 1
        delta = x-c
        ix = rng.integers(0,len(x),size=(10000,len(x)))
        summaries[name] = dict(n=len(x),mean_eeg=float(x.mean()), mean_control=float(c.mean()),
                               paired_gain=float(delta.mean()), ci95=np.quantile(delta[ix].mean(1),[.025,.975]).tolist(),
                               positive=int(np.sum(delta>0)))
    images = list(common["image"]["rows"].values())
    summarize("image k40 paired shuffle",[r["gain"] for r in images],[r["shuffled"] for r in images])
    for task, block in common["sentence"]["rows"].items():
        rows = list(block.values())
        summarize(task+" paired noise",[r["eeg"] for r in rows],[r["noise"] for r in rows])
    for task, rows in replication["datasets"].items():
        summarize(task+" EEGNet vs average of 3 circular shifts",[r["gain"] for r in rows],[np.mean(r["shifted"]) for r in rows])
    target = base/"log_yield_uncertainty.json"
    temp = target.with_suffix(".tmp")
    temp.write_text(json.dumps(summaries,indent=2,allow_nan=False)+"\n")
    temp.replace(target)
    print(json.dumps(summaries))
if __name__=="__main__":
    main()
