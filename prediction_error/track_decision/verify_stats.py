"""NOTE (code review 2026-11-20): results/real_twin_eegnet_{a,b,c} are SIMULATED TWINS (real_joint_ext.py --part twin, 4 per donor),
not real participants; this script therefore gives the expected V1-V3 under a correctly specified read-out. The paper's real-data
Verify numbers come from results/real_verify_eegnet/results.json (23 participants) via the ledger (Ver* macros), not from here.

Twin (simulated) V1-V3 statistics, from results/real_twin_eegnet_{a,b,c}/results.json
V1: held-out posterior-predictive log-likelihood, joint minus behaviour, per trial.  V2: split-half reliability (Pearson r across
participants of odd- vs even-block posterior means), behaviour vs joint, for the learning rate and inverse temperature.
V3: ratio of joint to behavioural posterior variance of the learning rate (logit alpha), observed median vs the exact prediction
(1-c(rho))/(1-c(0)) at the pooled EEGNet coupling from results/group_stats/results.json and the ceiling curve."""
import glob, json, sys
from pathlib import Path
import numpy as np
from scipy.stats import wilcoxon, pearsonr
ROOT = Path("/path/to/eegaudit"); sys.path.insert(0, str(ROOT))
from worth_ceiling import load_curve
subs = {}
for f in sorted(glob.glob(str(ROOT / "results/real_twin_eegnet_[abc]/results.json"))):
    subs.update(json.load(open(f))["subjects"])
A, B = 0, 2   # parameter order: logit alpha, logit d, log beta, kappa
ratio = np.array([s["A_joint"]["var"][A] / s["A_beh"]["var"][A] for s in subs.values()])
dl = np.array([(s["A_joint"]["lpred"] - s["A_beh"]["lpred"]) / s["n_test"] for s in subs.values()])
def split_half(cond, j):
    o = np.array([s[f"B_odd_{cond}"]["mean"][j] for s in subs.values()]); e = np.array([s[f"B_even_{cond}"]["mean"][j] for s in subs.values()])
    return float(pearsonr(o, e)[0])
G = json.load(open(ROOT / "results/group_stats/results.json"))["decoders"]["eegnet"]
rho = G["rho_eff_pooled"]; rhos, cur, *_ = load_curve(ROOT / "results/ceiling_RWD_v2/results.json")
c = cur["logit_alpha"]; pred = float(np.interp(rho, rhos, (1 - c) / (1 - c[0])))
out = dict(n_twins=len(subs), source="simulated twins (real_joint_ext.py --part twin)", decoder="eegnet (LOPO) read-out, correctly specified twin",
           v3_ratio_median=float(np.median(ratio)), v3_ratio_iqr=[float(np.percentile(ratio, 25)), float(np.percentile(ratio, 75))],
           v3_wilcoxon_p=float(wilcoxon(np.log(ratio)).pvalue), v3_predicted_ratio_at_pooled_rho=pred, v3_pooled_rho=rho,
           v1_delta_lpred_per_trial_mean=float(dl.mean()), v1_delta_lpred_per_trial_median=float(np.median(dl)), v1_wilcoxon_p=float(wilcoxon(dl).pvalue),
           v2_split_half_alpha_beh=split_half("beh", A), v2_split_half_alpha_joint=split_half("joint", A),
           v2_split_half_beta_beh=split_half("beh", B), v2_split_half_beta_joint=split_half("joint", B))
Path(ROOT / "results/track_decision/verify_stats_TWINS.json").write_text(json.dumps(out, indent=1)); print(json.dumps(out, indent=1))
