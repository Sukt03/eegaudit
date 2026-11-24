"""Post-outcome diagnostic only, no change to frozen I1 scorer or verdict."""
import hashlib,json,os,sys
from pathlib import Path
import numpy as np
from scipy.special import logsumexp
ROOT=Path('/path/to/eegaudit');O=ROOT/'outputs/AGENT_LOOP_20261120/results/i1'
sys.path.insert(0,str(ROOT/'experiments/eegbits'));from run import soft_bits,_TAUS
z=np.load(O/'i1_sub-01_embeddings.npz');a,b=z['half_a'],z['half_b'];order=z['axis_order'];perms=z['permutations'];rows=[]
norm=lambda v:v/(np.linalg.norm(v,axis=1,keepdims=True)+1e-12)
for cond in (1,10,100):
 v=np.geomspace(1.,float(cond),a.shape[1])[order] if cond!=1 else np.ones(a.shape[1]);S=norm(a*v)@norm(b*v).T;n=len(S)
 ce=logsumexp(S,axis=1)-S.mean(axis=1)
 exact_fixedtau_random_floor=float(np.log2(n)-ce.mean()/np.log(2))
 floor=float(np.mean([soft_bits(S[:,p]) for p in perms]))
 # Expected performance for truly uniform prediction is zero by algebra; not an amended estimator run.
 rows.append(dict(condition_number=cond,actual_frozen_LOO_shuffle_mean=floor,exact_uniform_target_fixed_tau1_floor=exact_fixedtau_random_floor,loo_minus_exact_fixedtau=floor-exact_fixedtau_random_floor,jensen_gap_bits=-exact_fixedtau_random_floor,tau_min=float(_TAUS.min()),tau_max=float(_TAUS.max()),hypothetical_uniform_predictor_yield_algebra=0.0))
r=dict(kind='post_outcome_floor_cause_diagnostic_not_confirmatory',step=os.getenv('SLURM_STEP_ID'),allocation=os.getenv('SLURM_JOB_ID'),formula='E_permutation CE_i(tau=1)=logsumexp(S_i)-mean_jS_ij; log2N-minus-CE is nonpositive by Jensen',rows=rows,verdict='Frozen finite-temperature softmax cannot become exactly uniform; a negative shuffled floor is expected for variable score rows. Original absolute0.02 gate remains failed. No temperature/grid/decision rule changed.',script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
p=O/'floor_diagnostic.json';t=p.with_suffix('.json.tmp');t.write_text(json.dumps(r,indent=2,allow_nan=False));t.replace(p);print(json.dumps(r),flush=True)
