"""Actual-data CUDA adapter smoke; no scientific yield claim or fitted head."""
import hashlib,json,os,tempfile,sys,time
from pathlib import Path
import numpy as np,torch,mne
from scipy.signal import resample
from safetensors.torch import load_file
from braindecode.models import BIOT,LUNA
ROOT=Path('/path/to/eegaudit');OUT=ROOT/'outputs/AGENT_LOOP_20261120'

def atomic(path,data):
 path.parent.mkdir(parents=True,exist_ok=True)
 with tempfile.NamedTemporaryFile(mode='w',dir=path.parent,delete=False) as f:json.dump(data,f,indent=2);name=f.name
 os.replace(name,path)

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

class AuthorNormMlp(torch.nn.Module):
 """Restore declared timm.Mlp LayerNorm graph, preserving every parameter.

 Verified primary author LUNA.py specifies norm_layer=LayerNorm at these two
 sites. Primary timm1.0.15 Mlp applies norm AFTER fc1/act/drop1 BEFORE fc2.
 This changes only the instantiated GPT model, never the installed package.
 """
 def __init__(self,old):
  super().__init__()
  self.fc1=old.fc1;self.act=old.act
  self.drop1=torch.nn.Dropout(old.drop.p)
  self.norm=torch.nn.LayerNorm(old.fc1.out_features)
  self.fc2=old.fc2;self.drop2=torch.nn.Dropout(old.drop.p)
 def forward(self,x):return self.drop2(self.fc2(self.norm(self.drop1(self.act(self.fc1(x))))))

def models(names,which):
 out={}
 bpath=OUT/'models/biot/model.safetensors'
 b=BIOT(n_chans=18,n_times=200,sfreq=200,n_outputs=2)
 state=load_file(str(bpath))
 # Newer Braindecode makes this channel-ID buffer nonpersistent. Verify
 # identity before omitting that buffer; every learned parameter stays strict.
 assert torch.equal(state['encoder.index'],torch.arange(18,dtype=torch.long))
 assert torch.equal(b.encoder.index,torch.arange(18,dtype=torch.long))
 state.pop('encoder.index')
 b.load_state_dict(state,strict=True);b.eval().cuda()
 pairs=['FP1-F7','F7-T7','T7-P7','P7-O1','FP2-F8','F8-T8','T8-P8','P8-O2','FP1-F3','F3-C3','C3-P3','P3-O1','FP2-F4','F4-C4','C4-P4','P4-O2']
 up={n.upper():i for i,n in enumerate(names)}
 assert all(z in up for pair in pairs for z in pair.split('-')),'missing BIOT physical derivation'
 def bp(x):
  z=torch.stack([x[:,up[a]]-x[:,up[c]] for a,c in [p.split('-') for p in pairs]],1)
  return z/(torch.quantile(z.abs(),.95,dim=-1,keepdim=True)+1e-6)
 out['biot']=(lambda x:b.encoder(bp(x)),bpath,dict(channels=pairs,channel_semantics='first16 exact pretrainedTCPbipolar slots; SHHS2absent; no padding',normalization='within-sample/channel95thabsolutepercentile',Hz=200,seconds=1))
 if which=='biot':return {'biot':out['biot']}
 del b
 lpath=OUT/'models/luna/LUNA_base.safetensors'
 l=LUNA(n_chans=len(names),n_times=280,sfreq=256,n_outputs=0,embed_dim=64,num_queries=4,depth=8)
 l.channel_location_embedder=AuthorNormMlp(l.channel_location_embedder)
 l.cross_attn.ffn=AuthorNormMlp(l.cross_attn.ffn)
 l.load_state_dict(load_file(str(lpath)),strict=True);l.eval().cuda()
 locs={k.upper():v for k,v in mne.channels.make_standard_montage('standard_1005').get_positions()['ch_pos'].items()}
 assert all(n.upper() in locs for n in names),'missing LUNA physical position'
 pos=torch.tensor(np.array([locs[n.upper()] for n in names]),dtype=torch.float32,device='cuda')
 assert torch.isfinite(pos).all()
 def lf(x):
  x=torch.tensor(resample(x.cpu().numpy(),256,axis=-1).copy(),device='cuda')
  x=(x-x.mean(-1,keepdim=True))/(x.std(-1,keepdim=True,unbiased=False)+1e-6)
  x=torch.nn.functional.pad(x,(0,24))
  z,_=l.prepare_tokens(x,pos.expand(len(x),-1,-1))
  z,_=l.cross_attn(z);z=z.reshape(len(x),-1,l.num_queries*l.embed_dim)
  for block in l.blocks:z=block(z)
  return l.norm(z).flatten(1)
 out['luna']=(lf,lpath,dict(channels=names,positions='MNEstandard1005 actual names; no synthetic fallback',Hz=256,observed_seconds=1,pad_samples=24,normalization='within-sample/channelzscore; pad0afterzscore',pretraining_rate_validation='authorBioFoundationdocs256Hz; existing filtered event epochs remain preprocessing transfer; explicit padding adapter, technical smoke only'))
 return {'luna':out['luna']}

def main():
 torch.set_num_threads(4);torch.manual_seed(20261120)
 source_sha=sha(__file__);t0=time.monotonic()
 e=np.load(ROOT/'results/fm_epochs/sub-001.npz',allow_pickle=False)
 names=[str(c) for c in e['ch']]
 raw=e['X'][:8].reshape(8,len(names),-1)[:,:,-200:].astype(np.float32)
 x=torch.as_tensor(raw,device='cuda')
 result=dict(kind='technical_actual_data_adapter_smoke',source_sha256=source_sha,device=torch.cuda.get_device_name(0),numpy=np.__version__,torch=torch.__version__,data='ds003458 sub001 first8 post-feedback0..1s at200Hz; no scientific outcome viewed',models={})
 for name in ['biot','luna']:
  try:
   fn,weights,info=models(names,name)[name]
   with torch.inference_mode():z=fn(x);zn=fn(torch.randn_like(x))
   assert z.shape[0]==8 and z.ndim==2 and torch.isfinite(z).all() and torch.isfinite(zn).all()
   result['models'][name]=dict(status='shape_pass_not_scientific',feature_shape=list(z.shape),weight_sha256=sha(weights),input=info)
  except Exception as e:
   result['models'][name]=dict(status='blocked',error=repr(e))
  print(name,result['models'][name],flush=True)
 result['runtime_s']=time.monotonic()-t0
 atomic(OUT/'results/openweight/adapter_smoke_v5.json',result)

if __name__=='__main__':main()
