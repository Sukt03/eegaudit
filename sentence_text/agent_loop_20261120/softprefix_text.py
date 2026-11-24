"""Ours: fixed-length EEG soft prefix with frozen open-weight BART-base.

Published LM weights are NOT a pretrained EEG decoder. The EEG adapter is
trained here on the locked Wang-Ji split and saved/reloaded before generation.
"""
import argparse,copy,datetime,hashlib,json,math,os,pickle,re,sys,tempfile,time
from pathlib import Path
import numpy as np,torch
from torch import nn
from safetensors.torch import save_file,load_file
from transformers import BartForConditionalGeneration,BartTokenizer,AutoTokenizer,AutoModel

ROOT=Path('/path/to/eegaudit')
CACHE=ROOT/'results/track_text/brain_translator/cache_full.pkl'
BART=ROOT/'data_raw/bart_base';MINILM=ROOT/'data_raw/zuco/minilm'
ARMS=['eeg','noise','train_shuffled']
TAUS=np.logspace(-2.5,0,12)

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def atomic_json(p,r):
 p.parent.mkdir(parents=True,exist_ok=True)
 with tempfile.NamedTemporaryFile('w',dir=p.parent,delete=False) as f:json.dump(r,f,indent=2,allow_nan=False);t=f.name
 os.replace(t,p)
def atomic_weights(p,state):
 fd,t=tempfile.mkstemp(dir=p.parent);os.close(fd)
 save_file({k:v.detach().cpu().contiguous() for k,v in state.items()},t);os.replace(t,p)
def backbone_hash(m):
 h=hashlib.sha256()
 for n,p in m.named_parameters():h.update(n.encode());h.update(p.detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()
def clean(s):return ' '.join(s.lower().split())

class Prefix(nn.Module):
 def __init__(self,d):
  super().__init__();self.net=nn.Sequential(nn.Linear(840,256),nn.GELU(),nn.Linear(256,8*d));self.norm=nn.LayerNorm(d)
  self.offset=nn.Parameter(torch.randn(8,d)*.02);self.d=d
 def forward(self,x):return .1*self.norm(self.net(x).reshape(len(x),8,self.d)+self.offset)

def lm_embeddings(texts,tok,m):
 out=[]
 with torch.inference_mode():
  for b in range(0,len(texts),64):
   z=tok(texts[b:b+64],padding=True,truncation=True,max_length=128,return_tensors='pt').to('cuda')
   h=m(**z).last_hidden_state;mask=z['attention_mask'].unsqueeze(-1)
   v=(h*mask).sum(1)/mask.sum(1);v=torch.nn.functional.normalize(v,dim=1);out.append(v.cpu().numpy())
 return np.concatenate(out)

def similarity_metrics(s,true,tau):
 diag=s[np.arange(len(s)),true];wrong=np.ones(s.shape,bool);wrong[np.arange(len(s)),true]=False
 tw=((s<diag[:,None])&wrong).sum(1)+.5*((s==diag[:,None])&wrong).sum(1)
 z=s/tau;z=z-z.max(1,keepdims=True);lp=z-np.log(np.exp(z).sum(1,keepdims=True))
 return dict(top1=float((s.argmax(1)==true).mean()),two_way=float((tw/(s.shape[1]-1)).mean()),nce=float(np.log2(s.shape[1])+lp[np.arange(len(s)),true].mean()/np.log(2)),N=s.shape[1],n_queries=len(s),tau=float(tau))

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--task',choices=['SR','NR2'],required=True);ap.add_argument('--out',required=True);ap.add_argument('--expected-sha',required=True);ap.add_argument('--smoke',action='store_true');ap.add_argument('--epochs',type=int,default=5)
 a=ap.parse_args();source_hash=sha(__file__);assert source_hash==a.expected_sha
 assert torch.cuda.is_available();torch.set_num_threads(4)
 out=Path(a.out);out.mkdir(parents=True,exist_ok=True);t0=time.monotonic()
 with CACHE.open('rb') as f:data=pickle.load(f)  # existing trusted project cache, never a downloaded checkpoint
 texts=data['texts'];records=[r for r in data['recs'] if r['task']==a.task]
 groups={s:[i for i,r in enumerate(records) if r['split']==s] for s in ['train','dev','test']}
 full_groups=copy.deepcopy(groups)
 textsets={s:{clean(texts[records[i]['g']]) for i in ids} for s,ids in groups.items()}
 assert not(textsets['train']&textsets['dev'] or textsets['train']&textsets['test'] or textsets['dev']&textsets['test']),'sentence overlap across splits'
 candidates={s:sorted({records[i]['g'] for i in ids}) for s,ids in groups.items()}
 assert len(candidates['test'])==({'SR':40,'NR2':35}[a.task]),'locked candidate set differs'
 if a.smoke:groups={s:ids[:n] for (s,ids),n in zip(groups.items(),[128,32,32])}
 x=np.stack([r['words'].mean(0) for r in records]).astype(np.float32);mu=x[groups['train']].mean(0);sd=x[groups['train']].std(0)+1e-6;x=(x-mu)/sd
 noise=np.random.default_rng(20261120).standard_normal(x.shape).astype(np.float32)
 config=dict(model='ours_fixed8_EEG_softprefix_frozen_BARTbase',task=a.task,smoke=a.smoke,epochs=1 if a.smoke else a.epochs,seed=20261120,prefix_tokens=8,hidden=256,lr=.001,batch_size=32,max_tokens=64,arms=ARMS,noise='fixedindependentrowfeaturesacrossepochsseed20261120',shuffle='fixedtraininputpermutationseed20261120;devtestunaltered',source_sha256=source_hash,cache_sha256=sha(CACHE),bart_weights_sha256=sha(BART/'model.safetensors'),bart_config_sha256=sha(BART/'config.json'),minilm_weights_sha256=sha(MINILM/'model.safetensors'),split='lockedWangJisentenceindex;taskseparate;normalizedtextdisjoint',input='sentence-mean840GD,no wordmask/length/texttoencoder',decode='free-runninggreedyinputs_embeds-only8allones,no golddecoderprefix',candidate_ids=candidates,selected_record_indices=groups)
 meta_path=out/'config.json'
 if meta_path.exists():assert json.loads(meta_path.read_text())==config,'existing config mismatch, no silent resume'
 atomic_json(meta_path,config)
 with tempfile.NamedTemporaryFile(dir=out,delete=False) as f:np.savez(f,mean=mu,std=sd);tmp=f.name
 os.replace(tmp,out/'input_normalization.npz')
 atomic_json(out/'split_records.json',dict(all_splits={s:[dict(index=i,g=int(records[i]['g']),subject=records[i]['subject']) for i in ids] for s,ids in full_groups.items()},normalized_text_overlap=0))
 tok=BartTokenizer.from_pretrained(str(BART),local_files_only=True)
 # MiniLM is required; no fallback encoder or downloaded model.
 mtok=AutoTokenizer.from_pretrained(str(MINILM),local_files_only=True)
 mm=AutoModel.from_pretrained(str(MINILM),local_files_only=True,use_safetensors=True).cuda().eval()
 candidate_embeddings={s:lm_embeddings([texts[g] for g in candidates[s]],mtok,mm) for s in ['dev','test']}
 for arm in ARMS:
  dest=out/arm;dest.mkdir(exist_ok=True)
  done=dest/'completed.json'
  if done.exists():
   old=json.loads(done.read_text());assert old['config_sha256']==sha(meta_path) and old['adapter_sha256']==sha(dest/'adapter.safetensors') and (dest/'generations.jsonl').exists(),'completed arm invalid';print('SKIP validated',arm,flush=True);continue
  torch.manual_seed(20261120)
  bart=BartForConditionalGeneration.from_pretrained(str(BART),local_files_only=True,use_safetensors=True).cuda().eval()
  for p in bart.parameters():p.requires_grad_(False)
  bart.config.use_cache=False
  adapter=Prefix(bart.config.d_model).cuda();initial_hash=hashlib.sha256(b''.join(p.detach().cpu().numpy().tobytes() for p in adapter.parameters())).hexdigest()
  bh=backbone_hash(bart);opt=torch.optim.AdamW(adapter.parameters(),lr=.001,weight_decay=.01)
  trids=groups['train'];permuted=np.random.default_rng(20261120).permutation(trids);source_map={int(i):int(j) for i,j in zip(trids,permuted)}
  def inputs(ids,mode=None):
   if mode=='noise' or arm=='noise':return torch.tensor(noise[ids],device='cuda')
   use=[source_map[int(i)] for i in ids] if arm=='train_shuffled' and mode=='training' else ids
   return torch.tensor(x[use],device='cuda')
  def labels(ids):
   z=tok([texts[records[i]['g']] for i in ids],padding=True,truncation=True,max_length=64,return_tensors='pt').to('cuda');l=z['input_ids'];l[z['attention_mask']==0]=-100;return l
  def loss(ids,train=False):
   h=adapter(inputs(ids,'training' if train else None));mask=torch.ones((len(ids),8),dtype=torch.long,device='cuda')
   return bart(inputs_embeds=h,attention_mask=mask,labels=labels(ids)).loss
  best=float('inf');best_epoch=-1;start_ep=0;updated=False;gradient_nonzero=False;history=[]
  progress=dest/'progress.pt';progress_json=dest/'progress.json'
  if progress.exists():
   pm=json.loads(progress_json.read_text());assert pm['config_sha256']==sha(meta_path) and pm['checkpoint_sha256']==sha(progress)
   state=torch.load(progress,map_location='cuda',weights_only=True);adapter.load_state_dict(state['adapter']);opt.load_state_dict(state['optimizer']);start_ep=state['epoch'];best=state['best'];best_epoch=state['best_epoch'];updated=state['updated'];gradient_nonzero=state['gradient_nonzero'];history=pm['history']
  for epoch in range(start_ep,config['epochs']):
   adapter.train();order=np.random.default_rng(20261120+epoch).permutation(trids);ls=[]
   for b in range(0,len(order),32):
    ids=order[b:b+32].tolist();opt.zero_grad();v=loss(ids,True);assert torch.isfinite(v);v.backward()
    g=sum(float(p.grad.square().sum()) for p in adapter.parameters() if p.grad is not None);gradient_nonzero|=g>0;assert g>0,'no gradient through frozen backbone'
    before=next(adapter.parameters()).detach().clone();nn.utils.clip_grad_norm_(adapter.parameters(),1.);opt.step();updated|=bool(torch.any(before!=next(adapter.parameters())).item());ls.append(float(v))
   adapter.eval();vals=[]
   with torch.no_grad():
    for b in range(0,len(groups['dev']),64):vals.append(float(loss(groups['dev'][b:b+64])))
   dl=float(np.mean(vals));history.append(dict(epoch=epoch+1,train_loss=float(np.mean(ls)),dev_loss=dl,elapsed_s=time.monotonic()-t0))
   if dl<best:best=dl;best_epoch=epoch+1;atomic_weights(dest/'adapter.safetensors',adapter.state_dict())
   fd,tmp=tempfile.mkstemp(dir=dest);os.close(fd);torch.save(dict(adapter=adapter.state_dict(),optimizer=opt.state_dict(),epoch=epoch+1,best=best,best_epoch=best_epoch,updated=updated,gradient_nonzero=gradient_nonzero),tmp);os.replace(tmp,progress)
   atomic_json(progress_json,dict(config_sha256=sha(meta_path),checkpoint_sha256=sha(progress),history=history))
   print(a.task,arm,'epoch',history[-1],flush=True)
  # Mandatory reloading of the saved EEG adapter before any held-out generation.
  adapter=Prefix(bart.config.d_model).cuda();adapter.load_state_dict(load_file(str(dest/'adapter.safetensors'),device='cuda'),strict=True);adapter.eval()
  assert gradient_nonzero and updated and backbone_hash(bart)==bh
  @torch.inference_mode()
  def generate(ids,mode=None):
   result=[]
   for b in range(0,len(ids),32):
    batch=ids[b:b+32];h=adapter(inputs(batch,mode));mask=torch.ones((len(batch),8),dtype=torch.long,device='cuda')
    generated=bart.generate(inputs_embeds=h,attention_mask=mask,max_new_tokens=64,num_beams=1,do_sample=False,use_cache=True)
    result.extend(tok.batch_decode(generated,skip_special_tokens=True))
   return result
  dev_text=generate(groups['dev']);dev_e=lm_embeddings([s or '.' for s in dev_text],mtok,mm);dev_true=np.array([candidates['dev'].index(records[i]['g']) for i in groups['dev']]);ds=dev_e@candidate_embeddings['dev'].T
  tau=min(TAUS,key=lambda t:-similarity_metrics(ds,dev_true,t)['nce'])
  modes=['native','inference_noise','inference_shuffled'] if arm=='eeg' else ['native'];rows=[];scores={}
  for mode in modes:
   ids=groups['test'];mode_ids=ids if mode!='inference_shuffled' else np.random.default_rng(20261120).permutation(ids).tolist()
   hypotheses=generate(mode_ids,'noise' if mode=='inference_noise' else None)
   qe=lm_embeddings([s or '.' for s in hypotheses],mtok,mm);sim=qe@candidate_embeddings['test'].T;truth=np.array([candidates['test'].index(records[i]['g']) for i in ids]);scores[mode]=similarity_metrics(sim,truth,tau)
   sh=np.random.default_rng(20261120).permutation(truth);scores[mode]['shuffled_pair']=similarity_metrics(sim,sh,tau);scores[mode]['unique_generations']=len(set(hypotheses))
   # Length-only cue is a diagnostic reference and is not supplied to the generator.
   lengths=np.array([len(texts[g].split()) for g in candidates['test']]);qlengths=np.array([len(texts[records[i]['g']].split()) for i in ids]);scores[mode]['length_only_two_way']=similarity_metrics(-abs(qlengths[:,None]-lengths[None,:]),truth,1.)['two_way']
   for i,hyp in zip(ids,hypotheses):rows.append(dict(record_index=i,g=int(records[i]['g']),subject=records[i]['subject'],mode=mode,target=texts[records[i]['g']],generated=hyp,original_word_count=int(len(records[i]['words']))))
  tmp=dest/'generations.jsonl.tmp'
  with tmp.open('w') as f:
   for r in rows:f.write(json.dumps(r)+'\n')
  os.replace(tmp,dest/'generations.jsonl')
  atomic_json(done,dict(config_sha256=sha(meta_path),adapter_sha256=sha(dest/'adapter.safetensors'),backbone_parameter_hash_start=bh,backbone_parameter_hash_end=backbone_hash(bart),adapter_initial_hash=initial_hash,adapter_updated=updated,nonzero_adapter_gradient=gradient_nonzero,best_epoch=best_epoch,best_dev_loss=best,history=history,scores=scores,n_train=len(trids),n_dev=len(groups['dev']),n_test=len(groups['test']),candidate_count=len(candidates['test']),claim='oursEEGadaptertrainedhere;genericopenweightBARTbackbone;actualfreegenerationafterreload;noindependentpublishedEEGcheckpoint',runtime_s=time.monotonic()-t0))
  print(a.task,arm,'DONE',scores,flush=True);del bart,adapter,opt;torch.cuda.empty_cache()
 atomic_json(out/'completed.json',dict(config_sha256=sha(meta_path),arms=ARMS,complete=all((out/arm/'completed.json').exists() for arm in ARMS),runtime_s=time.monotonic()-t0))

if __name__=='__main__':main()
