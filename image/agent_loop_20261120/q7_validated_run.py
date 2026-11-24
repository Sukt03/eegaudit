"""Run a remaining REVE subject into a staging directory; publish validated JSON only."""
import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import q7_reve_probe as probe

BASE = ROOT / "outputs/AGENT_LOOP_20261120"
OUT = BASE / "results/q7"
CODE_HASH = "dd09646bde8a12e5c135b9f35921d8d79b679cc3a64f258cd7731fc51faf8a43"
EXPECTED = {("eeg",1,i) for i in range(5)} | {(c,80,0) for c in ("eeg","shuffled_pair","noise","trial_shuffled")}


def validate(path, subject):
    d = json.loads(path.read_text())
    if (d["subject"],d["model"],d["n_train"],d["slice"],d["seed"],d["channels_matched"]) != (subject,"reve",16540,0,0,63):
        raise ValueError("Result identity or training dimensions differ")
    rows=d["rows"]
    if len(rows)!=9 or {(r["condition"],r["k"],r["draw"]) for r in rows}!=EXPECTED:
        raise ValueError("Incomplete test conditions")
    for row in rows:
        for field in ("top1","top5","two_way","mrr","cosine","bits","soft_bits"):
            if not math.isfinite(row[field]):
                raise ValueError(f"Nonfinite {field}")
    if d["hashes"][str(Path(probe.__file__).relative_to(ROOT))] != CODE_HASH:
        raise ValueError("Unrecognized numerical implementation")


def event(subject,status,**details):
    record=dict(subject=subject,event=status,allocation=os.getenv("SLURM_JOB_ID"),step=os.getenv("SLURM_STEP_ID"),
                purpose="Q7 staged validated REVE recovery",unix_time=time.time(),**details)
    with (BASE/"JOBS.jsonl").open("a") as f:
        f.write(json.dumps(record)+"\n")


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("subjects",nargs="+",choices=["sub-04","sub-09","sub-10"])
    subjects=ap.parse_args().subjects
    if hashlib.sha256(Path(probe.__file__).read_bytes()).hexdigest()!=CODE_HASH:
        raise RuntimeError("Numerical script changed")
    for subject in subjects:
        final=OUT/f"fm_probe_reve_{subject}.json"
        with (OUT/f".{subject}.recovery.lock").open("a") as lock:
            fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
            if final.exists():
                validate(final,subject)
                event(subject,"validated_existing")
                continue
            event(subject,"started")
            try:
                with tempfile.TemporaryDirectory(dir=OUT,prefix=f".{subject}_") as tmp:
                    probe.OUT=Path(tmp)
                    sys.argv=[probe.__file__,"--subject",subject,"--model","reve"]
                    probe.main()
                    staged=Path(tmp)/final.name
                    validate(staged,subject)
                    os.link(staged,final)
                event(subject,"validated_completed",output_path=str(final))
            except Exception as error:
                event(subject,"failed",error=repr(error))
                raise
if __name__=="__main__":
    main()
