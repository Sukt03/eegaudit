"""I1 exploratory coordinate sensitivity; frozen NICE-GA target predictions invariant.
All outputs GPT-owned. No training, downloads, or mutation of original artifacts.
"""
import argparse, hashlib, json, sys, time
from pathlib import Path
import numpy as np
import torch
ROOT=Path('/path/to/eegaudit')
sys.path.insert(0,str(ROOT/'experiments/eegbits'))
from run import soft_bits
import run_acq_nice as N

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def norm(x):
    x=np.asarray(x,np.float64)
    return x/(np.linalg.norm(x,axis=1,keepdims=True)+1e-12)
def sim(a,b): return norm(a)@norm(b).T

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--subs',default='sub-01');ap.add_argument('--out',required=True)
    a=ap.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4);dev=torch.device('cuda')
    refpath=ROOT/'results/loop/image_p1/p1_rank_bound.json'
    startup_hashes={str(p):sha(p) for p in (Path(__file__),refpath,ROOT/'experiments/eegbits/run.py',ROOT/'experiments/eegbits/run_acq_nice.py')}
    old=json.loads(refpath.read_text())['per_participant']
    for subject in a.subs.split(','):
        t0=time.monotonic();dst=out/f'i1_{subject}.json'
        if dst.exists():raise RuntimeError(f'Refuse overwrite {dst}')
        cache=ROOT/f'data_raw/things_eeg2_250hz/test_cache/{subject}_test.npy'
        assert cache.exists(),'Existing original cache required; do not mutate source data'
        X=np.load(cache,mmap_mode='r');assert X.shape==(200,80,63,250)
        Enc,Proj=N.load_classes('nice_ga');enc,proj=Enc().to(dev).eval(),Proj().to(dev).eval()
        tag=f'nice_ga_dgx_v2_{subject.replace("-","")}_seed141'
        ep=N.NICE/'model'/f'{tag}_Enc_eeg_cls.pth';pp=N.NICE/'model'/f'{tag}_Proj_eeg_cls.pth'
        input_hashes={str(ep):sha(ep),str(pp):sha(pp)}
        enc.load_state_dict(torch.load(ep,map_location=dev,weights_only=True),strict=True)
        proj.load_state_dict(torch.load(pp,map_location=dev,weights_only=True),strict=True)
        Y=np.squeeze(np.load(N.NICE/'dnn_feature/center_clip.npy',allow_pickle=True)).astype(np.float64)
        @torch.no_grad()
        def embed(x):return proj(enc(torch.from_numpy(np.ascontiguousarray(x)).unsqueeze(1).to(dev))).cpu().numpy().astype(np.float64)
        ea=embed(np.asarray(X[:,:40].mean(1)));eb=embed(np.asarray(X[:,40:].mean(1)))
        ef=embed(np.asarray(X.mean(1)))
        rng=np.random.default_rng(20261120)
        # Match original controls' N(0,1) query inputs; not a claim to simulate 40 averaged independent trials.
        na=embed(rng.standard_normal((200,63,250)).astype(np.float32))
        nb=embed(rng.standard_normal((200,63,250)).astype(np.float32))
        nf=embed(rng.standard_normal((200,63,250)).astype(np.float32))
        d=ef.shape[1];q,_=np.linalg.qr(rng.standard_normal((d,d)))
        order=rng.permutation(d);scales={1:np.ones(d)}
        for cond in (10,100):
            v=np.geomspace(1.,float(cond),d);scales[cond]=v[order]
        perms=[rng.permutation(200) for _ in range(100)]
        st=sim(ef,Y);target=float(soft_bits(st));original=float(soft_bits(sim(ea,eb)))
        target_noise=float(soft_bits(sim(nf,Y)))
        target_pair=float(np.mean([soft_bits(st[:,p]) for p in perms]))
        base_target=float(old['nice_ga'][subject]['k80']['I_NCE'])
        base_ceiling=float(old['splithalf_trained'][subject]['I_NCE'])
        assert abs(target-base_target)<1e-5,('historical target reproduction',target,base_target)
        assert abs(original-base_ceiling)<1e-5,('historical ceiling reproduction',original,base_ceiling)
        rotation_s=sim(ea@q,eb@q);rotation_error=float(np.max(np.abs(rotation_s-sim(ea,eb))))
        assert rotation_error<1e-8,rotation_error
        rows=[]
        for cond,v in scales.items():
            inv=sim((ef*v)/v,Y);err=float(np.max(np.abs(inv-st)));assert err<1e-8,err
            sr=sim(ea*v,eb*v);ceiling=float(soft_bits(sr))
            floor_pair=float(np.mean([soft_bits(sr[:,p]) for p in perms]))
            floor_noise=float(soft_bits(sim(na*v,nb*v)))
            denom=ceiling-floor_pair
            valid=bool(denom>.02 and abs(floor_pair)<=.02 and abs(floor_noise)<=.02)
            rows.append(dict(condition_number=cond,ceiling=ceiling,reference_shuffled_pair_mean=floor_pair,reference_noise=floor_noise,denominator=denom,denominator_valid_descriptive=valid,target_yield=target,target_matrix_inverse_error=err,fraction=(target-target_pair)/denom if valid else None,denominator_relative_change=(denom/(rows[0]['denominator'])-1) if rows else 0.))
        finite=all(np.isfinite(v) for r in rows for v in r.values() if isinstance(v,(int,float)) and not isinstance(v,bool)) and all(np.isfinite(v) for v in (target_noise,target_pair,target,original,rotation_error))
        assert finite,'Nonfinite metric'
        assert all(sha(p)==h for p,h in startup_hashes.items()),'Source changed during run'
        assert all(sha(p)==h for p,h in input_hashes.items()),'Checkpoint changed during run'
        controls=bool(abs(target_noise)<=.02 and abs(target_pair)<=.02 and all(abs(r['reference_shuffled_pair_mean'])<=.02 and abs(r['reference_noise'])<=.02 for r in rows))
        np.savez_compressed(out/f'i1_{subject}_embeddings.npz',half_a=ea,half_b=eb,full=ef,candidates=Y,noise_a=na,noise_b=nb,noise_full=nf,orthogonal=q,axis_order=order,permutations=np.stack(perms))
        result=dict(subject=subject,decoder='NICE-GA ours seed141',exploratory=True,n_participants_planned=3,N=200,reference='40-vs40 independent repetition-half code identification; metric-dependent proxy',numerator='80-repetition EEG-to-image CLIP-centre identification',scorer='experiments/eegbits/run.py soft_bits12taus LOO',seed=20261120,transformation='diagonal positive invertible; target scorer first applies exact inverse',target_noise=target_noise,target_pair=target_pair,rotation_matrix_error=rotation_error,rotation_ceiling=float(soft_bits(rotation_s)),baseline_target_error=target-base_target,baseline_ceiling_error=original-base_ceiling,all_finite=finite,control_gate_pass=controls,rows=rows,runtime_s=time.monotonic()-t0,device=torch.cuda.get_device_name(0),source_hashes={**startup_hashes,**input_hashes},scope='Fixed original predictions, changed code-self-identification reference read-out; not a contradiction of V-information and not a neural source claim')
        temp=dst.with_suffix('.json.tmp');temp.write_text(json.dumps(result,indent=2,allow_nan=False));temp.replace(dst);print(json.dumps({k:result[k] for k in ('subject','control_gate_pass','rotation_matrix_error','baseline_target_error','baseline_ceiling_error','rows','runtime_s')}),flush=True)
        del X,enc,proj;torch.cuda.empty_cache()
if __name__=='__main__':main()
