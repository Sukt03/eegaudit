"""I2 fixed within-participant calibration of two cached LOPO readouts.

CPU linear algebra inside an authorized Slurm allocation; no new EEG model fit.
Both base models exclude the evaluated participant. Meta-fits use only that
participant's training trial blocks, independently for each of 23 null targets.
"""
import argparse,ast,datetime,hashlib,json,os,sys,tempfile,time
from pathlib import Path
import numpy as np
import torch
from scipy import stats

ROOT=Path('/path/to/eegaudit')
sys.path.insert(0,str(ROOT))
from group_stats import corr,resid,QUANT
from belief_bottleneck import folds
from cogmodels import models as M

ARMS=['eegnet','reve','joint','eegnet_duplicate','reve_duplicate','eegnet_noise','reve_noise','noise_only']

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def atomic_json(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile('w',dir=path.parent,delete=False) as f:
        json.dump(obj,f,indent=2,allow_nan=False);tmp=f.name
    os.replace(tmp,path)

def targets():
    path=ROOT/'experiments/loop/s13b_gauss_bits.py'
    tree=ast.parse(path.read_text())
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='targets_ds003458')
    ns=dict(ROOT=ROOT,QUANT=QUANT,np=np,torch=torch,M=M,resid=resid)
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),ns)
    return ns['targets_ds003458']()

def design(z,arm):
    if arm=='eegnet':return z[:,0:1]
    if arm=='reve':return z[:,1:2]
    if arm=='joint':return z[:,:2]
    if arm=='eegnet_duplicate':return np.repeat(z[:,0:1],2,axis=1)/np.sqrt(2.)
    if arm=='reve_duplicate':return np.repeat(z[:,1:2],2,axis=1)/np.sqrt(2.)
    if arm=='eegnet_noise':return z[:,[0,2]]
    if arm=='reve_noise':return z[:,[1,2]]
    if arm=='noise_only':return z[:,2:3]
    raise ValueError(arm)

def calibration(x,y,fold,penalty):
    pred={a:np.full_like(y,np.nan) for a in ARMS}
    coverage=np.zeros(len(x),dtype=int);partitions=[]
    for b in range(10):
        tr=np.flatnonzero(fold!=b);te=np.flatnonzero(fold==b)
        assert len(tr)>0 and len(te)>0
        mu=x[tr].mean(0);sd=x[tr].std(0)+1e-6
        za,zb=(x[tr]-mu)/sd,(x[te]-mu)/sd
        # Duplicate scaling occurs AFTER fitting/applying original scalar scaler.
        centered=y[tr]-y[tr].mean(0)
        for arm in ARMS:
            a,z=design(za,arm),design(zb,arm)
            w=np.linalg.solve(a.T@a+penalty*len(a)*np.eye(a.shape[1]),a.T@centered)
            pred[arm][te]=z@w
        coverage[te]+=1
        partitions.append(dict(fold=b,train_rows=tr.tolist(),test_rows=te.tolist(),scaler_mean=mu.tolist(),scaler_sd=sd.tolist()))
    assert np.all(coverage==1)
    assert all(np.isfinite(p).all() for p in pred.values())
    return pred,partitions

def measure(p,yg,me):
    rg=np.array([corr(p[:,j],yg[:,j]) for j in range(yg.shape[1])])
    assert np.isfinite(rg).all()
    nl=np.delete(rg,me);d=float(np.arctanh(np.clip(rg[me],-.999,.999))-np.arctanh(np.clip(nl,-.999,.999)).mean())
    value=float(-.5*np.log2(1-np.tanh(max(d,0))**2))
    raw=float(-.5*np.log2(1-min(abs(rg[me]),.999)**2))
    return dict(r_g=float(rg[me]),null_r_g=nl.tolist(),d=d,yield_value=value,raw_yield_value=raw,p95=float(np.quantile(nl,.95)),above_own_p95=bool(rg[me]>np.quantile(nl,.95)))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);ap.add_argument('--expected-sha',required=True);a=ap.parse_args()
    start=time.monotonic();source_sha=sha(__file__)
    assert source_sha==a.expected_sha,'immutable source hash differs'
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='','I2 must be an explicitly CPU-only allocation step'
    torch.set_num_threads(1)
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    config=dict(experiment='I2',dataset='ds003458',participants=[1,2,3],arms=ARMS,penalty=10.,folds=10,seed_rule='20261120+participant',target='historicalglobalresidualizedRWDPE; historicalnormalranks',calibration='withinparticipant10contiguoustrialfolds; no crossparticipantstacking',device='CPU-only, zeroGPUstep',source_sha256=source_sha)
    qtil,subs,fpath,_=targets();allresults=[]
    common_paths=[ROOT/'deep_decoders.py',ROOT/'lopo_reve_readout.py',ROOT/'experiments/loop/s13b_gauss_bits.py',ROOT/'group_stats.py',ROOT/'belief_bottleneck.py',ROOT/QUANT,ROOT/'results/map_RWD.npy']
    common_hashes={str(p):sha(p) for p in common_paths}
    for s in config['participants']:
        path=out/f'sub-{s:03d}.json'
        caches=[fpath(dec,s) for dec in ['eegnet','reve_lopo']]
        source_hashes={**common_hashes,**{str(p):sha(p) for p in caches}}
        if path.exists():
            old=json.loads(path.read_text())
            assert old['config']==config and old['hashes']==source_hashes and old['subject']==s and old['technical_pass'],'existing result validation failed; no skip'
            assert sorted(old['arms'])==sorted(ARMS)
            allresults.append(old);continue
        e,r=[np.load(p,allow_pickle=False) for p in caches]
        assert np.array_equal(e['tri'],r['tri']),'base trial IDs differ; do not intersect silently'
        tri=e['tri'];assert np.all(np.diff(tri)>0)
        assert e['F'].shape==r['F'].shape==(len(tri),1)
        ef,rf=e['F'][:,0].astype(float),r['F'][:,0].astype(float)
        noise=np.random.default_rng(20261120+s).standard_normal(len(tri))
        x=np.column_stack([ef,rf,noise]);assert np.isfinite(x).all()
        y=np.column_stack([qtil(j,tri) for j in subs])
        yg=np.column_stack([stats.norm.ppf((stats.rankdata(y[:,j])-.5)/len(y)) for j in range(y.shape[1])])
        pred,parts=calibration(x,y,folds(len(tri)),10.)
        values={arm:measure(p,yg,subs.index(s)) for arm,p in pred.items()}
        duplicate={base:max(abs(corr(pred[base][:,j],pred[base+'_duplicate'][:,j])-1) for j in range(len(subs))) for base in ['eegnet','reve']}
        duplicate_maxabs={base:float(np.max(np.abs(pred[base]-pred[base+'_duplicate']))) for base in ['eegnet','reve']}
        perm=np.random.default_rng(20261120+s).permutation(len(tri))
        shuffled={arm:measure(p,yg[perm],subs.index(s)) for arm,p in pred.items()}
        technical=max(duplicate.values())<=1e-6 and values['noise_only']['yield_value']<=.02 and all(z['yield_value']<=.02 for z in shuffled.values())
        record=dict(config=config,subject=s,n=len(tri),n_targets=len(subs),original_trial_ids=tri.tolist(),folds=parts,arms=values,shuffled_pair=shuffled,duplicate_correlation_error=duplicate,duplicate_max_abs_prediction_error=duplicate_maxabs,technical_pass=bool(technical),hashes=source_hashes,limitations='baseLOPOprovenanceverifiedinsource; newmeta-readoutuseswithinparticipanttraininglabels; wholeparticipanttargetmodel/residualizer/ranksremainhistoricalexposed; session-null not independenttargetproof; no distinctneuralsourceclaim',runtime_s=time.monotonic()-start)
        with tempfile.NamedTemporaryFile(dir=out,delete=False) as f:
            np.savez_compressed(f,**pred);temp=f.name
        os.replace(temp,out/f'sub-{s:03d}_predictions.npz')
        atomic_json(path,record);allresults.append(record)
        print(s,{k:round(v['d'],6) for k,v in values.items()},'technical',technical,flush=True)
        if not technical:break
    gains=[dict(subject=r['subject'],joint_minus_eegnet=r['arms']['joint']['d']-r['arms']['eegnet']['d'],joint_minus_reve=r['arms']['joint']['d']-r['arms']['reve']['d'],joint_minus_eegnet_noise=r['arms']['joint']['d']-r['arms']['eegnet_noise']['d'],joint_minus_reve_noise=r['arms']['joint']['d']-r['arms']['reve_noise']['d']) for r in allresults]
    if len(allresults)!=3 or not all(r['technical_pass'] for r in allresults):verdict='technical_indeterminate'
    elif all(g['joint_minus_eegnet']>.02 and g['joint_minus_reve']>.02 and g['joint_minus_eegnet_noise']>0 and g['joint_minus_reve_noise']>0 for g in gains):verdict='complementarity_worth_full_cohort_followup'
    elif all(g['joint_minus_eegnet']<=.02 and g['joint_minus_reve']<=.02 for g in gains):verdict='no_diagnostic_gain_at_pilot_threshold'
    else:verdict='mixed_indeterminate'
    summary=dict(config=config,n=len(allresults),verdict=verdict,gains=gains,runtime_s=time.monotonic()-start,claim_status='exploratory3participantpilot; no automatic23extension',completed=datetime.datetime.now().astimezone().isoformat())
    atomic_json(out/'summary.json',summary);print(json.dumps(summary),flush=True)
    if verdict=='technical_indeterminate':raise SystemExit(2)

if __name__=='__main__':main()
