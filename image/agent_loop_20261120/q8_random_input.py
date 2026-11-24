"""Complete legacy Q8 without altering earlier scripts or results.

Reuse frozen own-basis PCA curves, add nested seeded random orthonormal
projection of real inputs and matched Gaussian input through that projection.
All scores use the identical frozen NICE-GA native candidate matrix and LOO
InfoNCE implementation used by r1. Inputs/basis remain historically exposed.
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path('/path/to/eegaudit')
sys.path.insert(0, str(ROOT/'experiments/method_pilots'))
from common import load_encoder, nn_index, score, sim


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--subs',default='sub-01')
    ap.add_argument('--out',required=True)
    ap.add_argument('--smoke',action='store_true')
    a=ap.parse_args()
    out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4)
    dev=torch.device('cuda')
    for subj in a.subs.split(','):
        t0=time.monotonic()
        tag=f'q8_{subj}'+('_smoke' if a.smoke else '')
        if (out/f'{tag}.json').exists():
            print('SKIP existing GPT result',tag,flush=True); continue
        embed,y,_,x=load_encoder('nice_ga',subj,141,dev)
        priorpath=ROOT/f'results/method_pilots/r1_input_bottleneck_{subj}_basis-{subj}.json'
        prior=json.loads(priorpath.read_text())
        widths=[25,100,500] if a.smoke else [25,50,100,250,500,1000,2000]
        d=63*250
        # Same own-single-trial mean and unbiased std as historical r1.
        z=torch.from_numpy(np.asarray(x).reshape(-1,d).copy()).to(dev)
        mu,sd=z.mean(0),z.std(0)+1e-6
        del z
        gen=torch.Generator(device=dev).manual_seed(0)
        basis=torch.linalg.qr(torch.randn(d,max(widths),generator=gen,device=dev),mode='reduced').Q
        rng=np.random.default_rng(0)
        full=np.asarray(x.mean(1))
        k1=[np.asarray(x[:,rng.choice(80,1,replace=False)].mean(1)) for _ in range(3)]
        nn=nn_index(y)
        rows=[]
        fullscore=score(sim(embed(full),y),nn)
        reference=next(r for r in prior['rows'] if r['L']=='full' and r['k']==80 and r['condition']=='eeg')
        assert abs(fullscore['soft_bits']-reference['soft_bits'])<1e-5,'Full-width reproduction gate failed'
        def recon(inp,L):
            v=(torch.from_numpy(np.ascontiguousarray(inp).reshape(len(inp),d)).to(dev)-mu)/sd
            p=basis[:,:L]
            return (((v@p)@p.T)*sd+mu).reshape(len(inp),63,250).cpu().numpy()
        for L in widths:
            s=sim(embed(recon(full,L)),y)
            rows.append(dict(L=L,k=80,condition='random_projection_eeg',draw=0,**score(s,nn)))
            rows.append(dict(L=L,k=80,condition='random_projection_shuffled_pair',draw=0,**score(s[rng.permutation(200)],nn)))
            noise=rng.standard_normal(full.shape).astype(np.float32)*float(sd.mean().cpu())+float(mu.mean().cpu())
            rows.append(dict(L=L,k=80,condition='random_projection_gaussian_input',draw=0,**score(sim(embed(recon(noise,L)),y),nn)))
            for draw,inp in enumerate(k1):
                rows.append(dict(L=L,k=1,condition='random_projection_eeg',draw=draw,**score(sim(embed(recon(inp,L)),y),nn)))
            print(subj,L,rows[-6]['soft_bits'],rows[-4]['soft_bits'],time.monotonic()-t0,flush=True)
        assert all(np.isfinite(r['soft_bits']) for r in rows)
        res=dict(subject=subj,pilot=a.smoke,decoder='frozen NICE-GA (ours),seed141',N=200,widths=widths,seed=0,projection='one nested QR-orthonormal Gaussian basis per participant; real EEG and Gaussian floor separately labelled',basis_standardization='same exposed own single trials as historical r1, no new tuning',pca_source=str(priorpath),pca_source_sha256=hashlib.sha256(priorpath.read_bytes()).hexdigest(),pca_rows=[r for r in prior['rows'] if r['L'] in widths],full_width=fullscore,full_width_soft_error=fullscore['soft_bits']-reference['soft_bits'],rows=rows,runtime_s=time.monotonic()-t0,device=torch.cuda.get_device_name(0),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        (out/f'{tag}.json').write_text(json.dumps(res,indent=2))
        del x,basis,mu,sd
        torch.cuda.empty_cache()


if __name__=='__main__': main()
