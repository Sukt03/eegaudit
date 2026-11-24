"""Parent-authorized raw image curves; price inference remains blocked by I1 floor.
Frozen U1 participants/draws/scorer; no training, no checkpoint selection.
"""
import argparse,hashlib,json,os,random,sys,time
from pathlib import Path
import numpy as np,torch
ROOT=Path('/path/to/eegaudit');O=ROOT/'outputs/AGENT_LOOP_20261120';SRC=O/'models/brainflora/source'
sys.path.insert(0,str(SRC));sys.path.insert(0,str(ROOT/'experiments/eegbits'))
from run import soft_bits,wolpaw_bits,_TAUS
import run_acq_nice as N

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def norm(v):
 v=np.asarray(v,np.float64);return v/(np.linalg.norm(v,axis=1,keepdims=True)+1e-12)
def atom(p,v):
 t=p.with_suffix('.json.tmp');t.write_text(json.dumps(v,indent=2,allow_nan=False));t.replace(p)
def loadfeatures(p):
 v=torch.load(p,map_location='cpu',weights_only=True)
 if isinstance(v,dict):v=v['img_features']
 assert torch.is_tensor(v);return v.float().numpy()
def score(e,y,perms):
 S=norm(e)@norm(y).T;assert np.isfinite(S).all()
 r=(S>np.diag(S)[:,None]).sum(1)+1;top=float((r==1).mean())
 Z=S[None,:,:]/_TAUS[:,None,None];Z=Z-Z.max(2,keepdims=True)
 lp=Z-np.log(np.exp(Z).sum(2,keepdims=True));ce=-lp/np.log(2);rr=np.arange(len(S));values=[]
 for perm in perms:
  c=ce[:,rr,perm];chosen=((c.sum(1,keepdims=True)-c)/(len(S)-1)).argmin(0);values.append(float(np.log2(len(S))-c[chosen,rr].mean()))
 for perm,v in zip(perms[:2],values[:2]):assert abs(v-soft_bits(S[:,perm]))<1e-10,'Vectorizedshuffle scorer mismatch'
 from scipy.special import logsumexp
 independentCE=np.stack([(logsumexp(S/tau,axis=1)-np.diag(S)/tau)/np.log(2) for tau in _TAUS]);ii=((independentCE.sum(1,keepdims=True)-independentCE)/(len(S)-1)).argmin(0);independent=float(np.log2(len(S))-independentCE[ii,rr].mean())
 assert abs(independent-soft_bits(S))<1e-10,'Independent scipy scorer mismatch'
 return dict(InfoNCE=float(soft_bits(S)),Wolpaw=float(wolpaw_bits(top,200)),top1=top,shuffled_InfoNCE_mean=float(np.mean(values)))
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--subs',default='sub-01');ap.add_argument('--audit-only',action='store_true');a=ap.parse_args()
 out=O/'results/u1_native_v2';out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(4);torch.manual_seed(20261120);random.seed(20261120)
 from model.medformer_encoders import eeg_encoder
 weight=O/'models/brainflora/eeg_01-06_01-46_150.pth';candidate=O/'models/brainflora/ViT-H-14_features_test.pt';local=Path('/path/to/atm/data/hf_probe/ViT-H-14_features_test.pt')
 Yb=loadfeatures(candidate);Ylocal=loadfeatures(local);assert Yb.shape==Ylocal.shape==(200,1024)
 C=norm(Yb)@norm(Ylocal).T;match=C.argmax(1);identity=bool(np.array_equal(match,np.arange(200)))
 audit=dict(native_candidate=str(candidate),candidate_sha256=sha(candidate),comparison_local_sha256=sha(local),shape=list(Yb.shape),nearest_neighbor_identity_all200=identity,n_diagonal_nearest=int((match==np.arange(200)).sum()),n_unique_nearest=int(len(np.unique(match))),min_matched_cosine=float(np.diag(C).min()),mean_matched_cosine=float(np.diag(C).mean()),nearest_index=match.tolist(),crosscheck_gate_pass=bool(identity and float(np.diag(C).min())>.99),scope='Official released native retrieval features; rowidentity crosschecked against existing200THINGS-EEG2 test candidates',source_commit='2b94307c4412c38698da9b2e870492b431042c95')
 atom(out/'candidate_audit.json',audit)
 if a.audit_only:
  print(json.dumps(audit),flush=True);return
 assert audit['nearest_neighbor_identity_all200'] and audit['n_unique_nearest']==200,'Native candidate identity audit failed'
 audit['analysis_amendment']='Nativeofficialfeatureanalysisv2; originalnumerical>.99equivalencefailed andpreserved; noEEGscoresviewedbeforeamendment'
 atom(out/'candidate_audit.json',audit)
 bf=eeg_encoder();bf.load_state_dict(torch.load(weight,map_location='cpu',weights_only=True),strict=True);bf.eval().cuda()
 script_sha=sha(__file__);weight_sha=sha(weight)
 for sub in a.subs.split(','):
  start=time.monotonic();dest=out/f'{sub}.json';assert not dest.exists(),'Refuse overwrite result'
  x=np.load(ROOT/f'data_raw/things_eeg2_250hz/test_cache/{sub}_test.npy',mmap_mode='r');assert x.shape==(200,80,63,250)
  Enc,Proj=N.load_classes('nice_ga');enc,proj=Enc().cuda().eval(),Proj().cuda().eval();tag=f'nice_ga_dgx_v2_{sub.replace("-","")}_seed141'
  enc.load_state_dict(torch.load(N.NICE/'model'/f'{tag}_Enc_eeg_cls.pth',weights_only=True,map_location='cuda'),strict=True);proj.load_state_dict(torch.load(N.NICE/'model'/f'{tag}_Proj_eeg_cls.pth',weights_only=True,map_location='cuda'),strict=True)
  Yn=np.squeeze(np.load(N.NICE/'dnn_feature/center_clip.npy',allow_pickle=True)).astype(np.float64)
  @torch.inference_mode()
  def emb(inp,which):
   chunks=[]
   for j in range(0,len(inp),32):
    t=torch.from_numpy(np.ascontiguousarray(inp[j:j+32])).cuda()
    z=bf(t) if which=='brainflora' else proj(enc(t.unsqueeze(1)))
    chunks.append(z.cpu().numpy())
   z=np.concatenate(chunks);assert np.isfinite(z).all();return z
  rng=np.random.default_rng(20261120+int(sub[-2:]));draws=[rng.permutation(80) for _ in range(5)];perms=[rng.permutation(200) for _ in range(100)];rows=[];arrays={};references={}
  for which,y in [('brainflora',Yb),('nice_ga',Yn)]:
   ha=emb(np.asarray(x[:,:40].mean(1)),which);hb=emb(np.asarray(x[:,40:].mean(1)),which);arrays[f'{which}_half_a']=ha;arrays[f'{which}_half_b']=hb
   references[which]=score(ha,hb,perms)
   for k in (1,2,4,8,16,32,40,80):
    for draw in range(1 if k==80 else 5):
     ids=np.arange(80) if k==80 else draws[draw][:k];inp=np.asarray(x[:,ids].mean(1));e=emb(inp,which)
     noise=np.random.default_rng(20261120+1000*int(sub[-2:])+100*k+draw).standard_normal(inp.shape).astype(np.float32)/np.sqrt(k)
     en=emb(noise.astype(np.float32),which);arrays[f'{which}_k{k}_d{draw}']=e;arrays[f'{which}_noise_k{k}_d{draw}']=en
     rows.append(dict(model=which,k=k,draw=draw,repetition_ids=ids.tolist(),eeg=score(e,y,perms),noise=score(en,y,perms)))
    print(sub,which,'k',k,'elapsed',time.monotonic()-start,flush=True)
  arrays['draw_permutations']=np.stack(draws);arrays['target_permutations']=np.stack(perms);arrays['brainflora_candidates']=Yb;arrays['nice_candidates']=Yn
  np.savez_compressed(out/f'{sub}_embeddings.npz',**arrays)
  r=dict(subject=sub,pilot=True,N=200,scorer='run.soft_bits12tauLOO fixed',resource=dict(channels=63,window_seconds=1,sampling_hz=250,stimulus_soa=.2),candidate_audit=audit,native_target_spaces_differ=True,price_status='blocked_originalI1referencefloor; nativefeatureanalysisv2 rawcurvesonly no normalizedfractions',exposure='Historicaltestoutcomesknown; BrainFLORAcheckpointselectionauditunresolved',rows=rows,references=references,source_hashes={str(Path(__file__)):script_sha,str(weight):weight_sha},runtime_s=time.monotonic()-start,step=os.getenv('SLURM_STEP_ID'),allocation=os.getenv('SLURM_JOB_ID'),noise='GaussiantrialnoiseN0,1 analyticallyaveragedk hencequerysd1/sqrtk; no rawacquisitionnoiseassumption',scope='Nativejointdecoder+targetrepresentations; descriptive rawretrieval, not price inference')
  assert sha(__file__)==script_sha and sha(weight)==weight_sha
  atom(dest,r);del x,enc,proj;torch.cuda.empty_cache();print('COMPLETE',sub,r['runtime_s'],flush=True)
if __name__=='__main__':main()
