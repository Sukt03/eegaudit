"""Bounded authorized release retrieval; does not execute downloaded code."""
import datetime,hashlib,json,os,tempfile
from pathlib import Path
import requests
ROOT=Path('/path/to/eegaudit/outputs/AGENT_LOOP_20261120')
OUT=ROOT/'models';OUT.mkdir(parents=True,exist_ok=True)
specs={
 'biot':('https://huggingface.co/braindecode/biot-pretrained-six-datasets-18chs/resolve/main/', [('config.json',200000),('README.md',200000),('model.safetensors',20000000)]),
 'luna':('https://huggingface.co/PulpBio/LUNA/resolve/main/', [('README.md',300000),('config.json',200000),('LUNA_base.safetensors',35000000)]),
 'brainflora':('https://huggingface.co/datasets/LidongYang/BrainFLORA/resolve/main/', [('README.md',200000),('checkpoints/eeg_01-06_01-46_150.pth',90000000)])}
records=[]
for model,(base,files) in specs.items():
 folder=OUT/model;folder.mkdir(exist_ok=True)
 for filename,cap in files:
  path=folder/Path(filename).name;url=base+filename
  event=dict(model=model,url=url,path=str(path),cap=cap)
  try:
   with requests.get(url,stream=True,timeout=(15,45)) as r:
    r.raise_for_status();n=0;h=hashlib.sha256()
    with tempfile.NamedTemporaryFile(dir=folder,delete=False) as f:
     temp=Path(f.name)
     for chunk in r.iter_content(1024*1024):
      n+=len(chunk)
      if n>cap:raise RuntimeError('download size cap exceeded')
      h.update(chunk);f.write(chunk)
    os.replace(temp,path)
    event.update(status='retrieved',bytes=n,sha256=h.hexdigest(),resolved_url=r.url,etag=r.headers.get('etag'),commit=r.headers.get('x-repo-commit'))
  except Exception as e:
   event.update(status='unavailable',error=str(e))
  records.append(event);print(json.dumps(event),flush=True)
target=OUT/'download_manifest.json'
tmp=target.with_suffix('.tmp');tmp.write_text(json.dumps(dict(timestamp=datetime.datetime.now().astimezone().isoformat(),records=records),indent=2));os.replace(tmp,target)
