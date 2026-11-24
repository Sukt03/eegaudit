"""Regression checks for staged recovery and the new score implementations."""
import copy
import json
from pathlib import Path
import sys
import tempfile

import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from q7_validated_run import OUT,validate
from resolution_yield import decompose,log_probs,loo
from common_log_yield import density_gain

subject="sub-01"
path=OUT/f"fm_probe_reve_{subject}.json"
validate(path,subject)
valid=json.loads(path.read_text())
with tempfile.TemporaryDirectory() as folder:
    tmp=Path(folder)/path.name
    for defect in ("duplicate condition","missing row","wrong participant","nonfinite","wrong checksum","truncated"):
        d=copy.deepcopy(valid)
        if defect=="duplicate condition":
            d["rows"][-1]=d["rows"][0]
        elif defect=="missing row":
            d["rows"].pop()
        elif defect=="wrong participant":
            d["subject"]="sub-99"
        elif defect=="nonfinite":
            d["rows"][0]["cosine"]=float("nan")
        elif defect=="wrong checksum":
            d["hashes"][next(iter(d["hashes"]))]="wrong"
        tmp.write_text("{" if defect=="truncated" else json.dumps(d))
        try:
            validate(tmp,subject)
        except (ValueError,KeyError):
            continue
        raise AssertionError(f"Validator accepted {defect}")
rng=np.random.default_rng(20261120)
S=rng.normal(size=(12,12))
ids=np.arange(len(S))
groups=ids%5
lp=loo(S)
baseline=decompose(lp,ids,groups)
perm=rng.permutation(ids)
new_truth=np.argsort(perm)
renamed=decompose(lp[:,perm],new_truth,groups[perm])
for metric in ("coarse","within","total"):
    assert abs(renamed[metric]-baseline[metric])<1e-12
equal=decompose(log_probs(np.zeros_like(S),1),ids,groups)
assert max(abs(equal[k]) for k in ("coarse","within","total"))<1e-12
assert np.max(np.abs(loo(S+rng.normal(size=(len(ids),1)))-lp))<1e-10
x=rng.normal(size=(300,1))
y=x[:,0]+rng.normal(size=len(x))
fold=np.repeat(np.arange(10),30)
gain=density_gain(y,x,fold)
assert gain>0 and abs(gain-density_gain(100*y,x,fold))<1e-11
assert abs(density_gain(y,np.ones_like(x),fold))<1e-11
print("PASS: 6 malformed artifacts rejected, valid artifact accepted, categorical and density invariance checks passed.")
