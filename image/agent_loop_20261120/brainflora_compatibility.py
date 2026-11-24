"""Strict official-checkpoint/model actual-data shape smoke; no dataset loader/network."""
import hashlib,json,os,random,sys,time
from pathlib import Path
import numpy as np
import torch
ROOT=Path('/path/to/eegaudit');O=ROOT/'outputs/AGENT_LOOP_20261120'
SOURCE=O/'models/brainflora/source'
sys.path.insert(0,str(SOURCE))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 t0=time.monotonic();torch.set_num_threads(4);torch.manual_seed(20261120);random.seed(20261120)
 weight=O/'models/brainflora/eeg_01-06_01-46_150.pth'
 files=[Path(__file__),weight,SOURCE/'model/medformer_encoders.py',SOURCE/'model/Medformer.py',SOURCE/'layers/Embed.py',SOURCE/'layers/Augmentation.py',SOURCE/'layers/SelfAttention_Family.py',SOURCE/'layers/Medformer_EncDec.py',SOURCE/'utils/losses.py',SOURCE/'utils/masking.py']
 hashes={str(p):sha(p) for p in files}
 result=dict(kind='actual_data_strict_checkpoint_compatibility_only',step=os.getenv('SLURM_STEP_ID'),allocation=os.getenv('SLURM_JOB_ID'),checkpoint_sha256=hashes[str(weight)],source_hashes=hashes)
 try:
  from model.medformer_encoders import eeg_encoder
  sd=torch.load(weight,map_location='cpu',weights_only=True)
  assert isinstance(sd,dict) and all(torch.is_tensor(v) for v in sd.values()),'Expected tensor-only state dict'
  model=eeg_encoder();model.load_state_dict(sd,strict=True);model.eval().cuda()
  x=np.load(ROOT/'data_raw/things_eeg2_250hz/test_cache/sub-01_test.npy',mmap_mode='r')
  avg=np.asarray(x[:8].mean(1)); inp=torch.from_numpy(avg.copy()).cuda()
  with torch.inference_mode():
   z=model(inp);again=model(inp);noise=model(torch.randn_like(inp))
  assert z.shape==(8,1024) and torch.isfinite(z).all() and torch.isfinite(noise).all()
  assert torch.allclose(z,again,atol=1e-6,rtol=1e-6),'eval repeated pass nondeterministic'
  result.update(status='technical_pass_not_yield',input_shape=list(inp.shape),output_shape=list(z.shape),noise_shape=list(noise.shape),repeat_max_error=float((z-again).abs().max()),device=torch.cuda.get_device_name(0),input_description='THINGS-EEG2 sub01 first8 official200testimages averaged80repetitions; existing63channel250Hz1second cache; originalscale; no extra standardization',target_feature_space='official retrieval loader defaultViT-H-14; CLIP-L caption path excluded',remaining_gates=['exact native candidate feature identity/order','checkpoint training/test exposure audit','reference floor calibration before price inference'])
  np.savez_compressed(O/'results/i1/brainflora_smoke_embeddings.npz',eeg=z.cpu().numpy(),noise=noise.cpu().numpy())
 except Exception as e:
  result.update(status='technical_blocked',error=repr(e))
 assert all(sha(p)==h for p,h in hashes.items()),'Source changed during smoke'
 result['runtime_s']=time.monotonic()-t0
 p=O/'results/i1/brainflora_compatibility.json';tmp=p.with_suffix('.json.tmp');tmp.write_text(json.dumps(result,indent=2,allow_nan=False));tmp.replace(p)
 print(json.dumps(result),flush=True)
if __name__=='__main__':main()
