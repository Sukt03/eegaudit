"""Retrieve bounded primary implementation metadata; never execute fetched code."""
import hashlib,json
from pathlib import Path
import requests
OUT=Path('/path/to/eegaudit/outputs/AGENT_LOOP_20261120/models/luna/author_source');OUT.mkdir(exist_ok=True)
r=requests.get('https://api.github.com/repos/pulp-bio/BioFoundation/git/trees/main?recursive=1',timeout=30);r.raise_for_status();assert len(r.content)<3000000
tree=r.json();(OUT/'tree.json').write_text(json.dumps(tree,indent=2));sha=tree['sha']
paths=[x['path'] for x in tree['tree'] if 'luna' in x['path'].lower() and x['path'].endswith(('.py','.yaml','.md'))]
print('commit',sha,'candidates',paths,flush=True)
for path in paths[:12]:
 u=f'https://raw.githubusercontent.com/pulp-bio/BioFoundation/{sha}/{path}'
 r=requests.get(u,timeout=30);r.raise_for_status();assert len(r.content)<1000000
 name=path.replace('/','__');(OUT/name).write_bytes(r.content)
 print(path,len(r.content),hashlib.sha256(r.content).hexdigest(),flush=True)
