"""Exploratory frozen BIOT PE pilot; first3 IDs, matched post-feedback window."""
import argparse,ast,hashlib,json,os,sys,tempfile,time,datetime
from pathlib import Path
import numpy as np,torch
from scipy import stats
from scipy.signal import resample
from safetensors.torch import load_file
from braindecode.models import BIOT

ROOT=Path('/path/to/eegaudit');BASE=ROOT/'outputs/AGENT_LOOP_20261120'
sys.path.insert(0,str(ROOT))
from group_stats import corr,resid,QUANT
from belief_bottleneck import folds
from cogmodels import models as M
PAIRS=['FP1-F7','F7-T7','T7-P7','P7-O1','FP2-F8','F8-T8','T8-P8','P8-O2','FP1-F3','F3-C3','C3-P3','P3-O1','FP2-F4','F4-C4','C4-P4','P4-O2']
ARMS=['biot_pretrained','biot_random_init','gaussian_input','raw_matched_bipolar']

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def atomic(p,r):
 p.parent.mkdir(parents=True,exist_ok=True)
 with tempfile.NamedTemporaryFile('w',dir=p.parent,delete=False) as f:json.dump(r,f,indent=2,allow_nan=False);t=f.name
 os.replace(t,p)

def targets():
 p=ROOT/'experiments/loop/s13b_gauss_bits.py';tree=ast.parse(p.read_text())
 n=next(z for z in tree.body if isinstance(z,ast.FunctionDef) and z.name=='targets_ds003458')
 ns=dict(ROOT=ROOT,QUANT=QUANT,np=np,torch=torch,M=M,resid=resid)
 exec(compile(ast.Module(body=[n],type_ignores=[]),str(p),'exec'),ns)
 return ns['targets_ds003458']()

def model(pretrained):
 torch.manual_seed(20261120)
 m=BIOT(n_chans=18,n_times=200,sfreq=200,n_outputs=2)
 if pretrained:
  s=load_file(str(BASE/'models/biot/model.safetensors'))
  assert torch.equal(s['encoder.index'],torch.arange(18)) and torch.equal(m.encoder.index,torch.arange(18))
  s.pop('encoder.index');m.load_state_dict(s,strict=True)
 return m.cuda().eval()

def prepare(x,names):
 up={s.upper():i for i,s in enumerate(names)}
 assert all(n in up for pair in PAIRS for n in pair.split('-'))
 z=torch.as_tensor(x,device='cuda')
 z=torch.stack([z[:,up[a]]-z[:,up[c]] for a,c in [p.split('-') for p in PAIRS]],1)
 q=torch.quantile(z.abs(),.95,dim=-1,keepdim=True)
 z=z/(q+1e-6)
 return z.cpu().numpy(),dict(quantiles95=q[:,:,0].cpu().numpy().tolist(),low_quantile_guard_count=int((q<1e-6).sum()),guard='q+1e-6fixedbeforeoutcomes')

def extract(m,x):
 out=[]
 with torch.inference_mode():
  for b in range(0,len(x),64):
   z=torch.as_tensor(x[b:b+64],device='cuda')
   out.append(m.encoder(z).float().cpu().numpy())
 f=np.concatenate(out);assert f.shape==(len(x),256) and np.isfinite(f).all()
 return f

def cv(f,y):
 f=torch.tensor(f,dtype=torch.float64,device='cuda');y=torch.tensor(y,dtype=torch.float64,device='cuda')
 fo=folds(len(f));p=torch.zeros_like(y);counts=np.zeros(len(f),int);info=[]
 for b in range(10):
  tr0=np.flatnonzero(fo!=b);te0=np.flatnonzero(fo==b)
  tr=torch.as_tensor(tr0,device='cuda');te=torch.as_tensor(te0,device='cuda')
  mu,sd=f[tr].mean(0),f[tr].std(0,unbiased=False)+1e-6
  a,z=(f[tr]-mu)/sd,(f[te]-mu)/sd
  _,s,v=torch.linalg.svd(a-a.mean(0),full_matrices=False)
  rank=int((s>s[0]*max(a.shape)*torch.finfo(a.dtype).eps).sum());assert rank>=64
  v=v[:64].T;a,z=a@v,z@v
  w=torch.linalg.solve(a.T@a+10.*len(a)*torch.eye(64,dtype=a.dtype,device='cuda'),a.T@(y[tr]-y[tr].mean(0)))
  p[te]=z@w;counts[te0]+=1
  info.append(dict(fold=b,train_rows=tr0.tolist(),test_rows=te0.tolist(),pca_rank=rank,width=64))
 assert np.all(counts==1)
 p=p.cpu().numpy();assert np.isfinite(p).all()
 return p,info

def measure(p,g,me):
 r=np.array([corr(p[:,j],g[:,j]) for j in range(g.shape[1])]);assert np.isfinite(r).all()
 nl=np.delete(r,me);d=float(np.arctanh(np.clip(r[me],-.999,.999))-np.arctanh(np.clip(nl,-.999,.999)).mean())
 return dict(r_g=float(r[me]),d=d,null_r_g=nl.tolist(),p95=float(np.quantile(nl,.95)),above_p95=bool(r[me]>np.quantile(nl,.95)),yield_value=float(-.5*np.log2(1-np.tanh(max(d,0))**2)))

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);ap.add_argument('--expected-sha',required=True);a=ap.parse_args()
 script_hash=sha(__file__);assert script_hash==a.expected_sha
 assert torch.cuda.is_available();torch.set_num_threads(4)
 out=Path(a.out);out.mkdir(parents=True,exist_ok=True);t0=time.monotonic()
 weights=BASE/'models/biot/model.safetensors'
 config=dict(experiment='BIOT_PE_pilot_v2',subjects=[1,2,3],dataset='ds003458',window='observedpostfeedback0..1s200samples200Hz',channels=PAIRS,missing_pretrained_channels='SHHSmastoidslots16/17notprovided;first16indicespreserved',normalization='per-sample/channel95thabsoluteafterexactbipolarderivation',feature='BIOTpooled256;PCA64trainfold',ridge_penalty=10,folds=10,seed=20261120,arms=ARMS,script_sha256=script_hash,weights_sha256=sha(weights),target='historicalglobalRWDresidualizedPE;globalnormalranks',split='withinparticipant10contiguousfolds;readoutours',raw_baseline='identicalnormalized16bipolar200Hzpost1sinputflattened3200;PCA64ridge10;notold150..600msERP')
 qtil,subs,fpath,_=targets();pre,random=model(True),model(False);records=[]
 for s in [1,2,3]:
  rawpath=ROOT/f'results/fm_epochs/sub-{s:03d}.npz';tripath=fpath('raw',s)
  sources={str(p):sha(p) for p in [rawpath,tripath,ROOT/'experiments/loop/s13b_gauss_bits.py',ROOT/QUANT,ROOT/'results/map_RWD.npy']}
  dst=out/f'sub-{s:03d}.json'
  if dst.exists():
   r=json.loads(dst.read_text());assert r['config']==config and r['hashes']==sources and r['technical_pass'];records.append(r);continue
  e=np.load(rawpath,allow_pickle=False);tri=np.load(tripath,allow_pickle=False)['tri'];names=[str(c) for c in e['ch']]
  ids=e['trial'];assert len(np.unique(ids))==len(ids) and np.all(np.diff(tri)>0)
  row={int(t):i for i,t in enumerate(ids)};assert all(int(t) in row for t in tri)
  x=e['X'][[row[int(t)] for t in tri]].reshape(len(tri),len(names),400)[:,:,-200:].astype(np.float32)
  mu,sd=x.mean((0,2),keepdims=True),x.std((0,2),keepdims=True)
  xn=(np.random.default_rng(20261120+s).standard_normal(x.shape)*sd+mu).astype(np.float32)
  prepared,scales=prepare(x,names);prepared_noise,noise_scales=prepare(xn,names)
  features=dict(biot_pretrained=extract(pre,prepared),biot_random_init=extract(random,prepared),gaussian_input=extract(pre,prepared_noise),raw_matched_bipolar=prepared.reshape(len(x),-1))
  y=np.column_stack([qtil(j,tri) for j in subs]);g=np.column_stack([stats.norm.ppf((stats.rankdata(y[:,j])-.5)/len(y)) for j in range(y.shape[1])])
  perm=np.random.default_rng(20261120+s).permutation(len(tri));values={};floors={};foldmeta={};pred={}
  for arm,f in features.items():
   pred[arm],foldmeta[arm]=cv(f,y);values[arm]=measure(pred[arm],g,subs.index(s));floors[arm]=measure(pred[arm],g[perm],subs.index(s))
  technical=values['gaussian_input']['yield_value']<=.02 and all(z['yield_value']<=.02 for z in floors.values())
  r=dict(config=config,subject=s,n=len(tri),trial_ids=tri.tolist(),channel_names=names,normalization_scales=dict(eeg=scales,gaussian=noise_scales),arms=values,shuffled_pair=floors,folds=foldmeta,hashes=sources,technical_pass=bool(technical),device=torch.cuda.get_device_name(0),software=dict(torch=torch.__version__,numpy=np.__version__,cuda=torch.version.cuda),runtime_s=time.monotonic()-t0,limitation='existingeventpreprocessingtransfer,first16/18channels;globaltargetresidualization/ranks;randominitisbaselineNOTtargetfreefloor;noLOPOorstrongpopulationclaim')
  with tempfile.NamedTemporaryFile(dir=out,delete=False) as f:np.savez_compressed(f,**pred);temp=f.name
  os.replace(temp,out/f'sub-{s:03d}_predictions.npz');atomic(dst,r);records.append(r)
  print(s,{k:round(v['d'],5) for k,v in values.items()},'technical',technical,flush=True)
  if not technical:break
 summary=dict(config=config,n=len(records),technical_pass=len(records)==3 and all(r['technical_pass'] for r in records),participants=[dict(subject=r['subject'],arms=r['arms']) for r in records],claim='exploratorytechnical3participantpilot;noautomaticfull23',runtime_s=time.monotonic()-t0,completed=datetime.datetime.now().astimezone().isoformat())
 atomic(out/'summary.json',summary)
 if not summary['technical_pass']:raise SystemExit(2)

if __name__=='__main__':main()
