import json, sys
from huggingface_hub import HfApi
api = HfApi()
for rid in ["pyinglie/EEG-To-Text", "Jash-176/EEG-to-Text-Checkpoints"]:
    print("=====", rid)
    try:
        info = api.model_info(rid, files_metadata=True)
    except Exception as e:
        print("model_info failed:", repr(e))
        try:
            info = api.dataset_info(rid, files_metadata=True); print("(dataset repo)")
        except Exception as e2:
            print("dataset_info failed:", repr(e2)); continue
    print("sha", info.sha, "lastModified", info.last_modified, "tags", getattr(info,'tags',None))
    print("card", json.dumps(getattr(info,'card_data',None) and info.card_data.to_dict(), default=str)[:1500])
    tot=0
    for s in info.siblings:
        sz = s.size or 0; tot += sz
        print(f"{sz/1e6:10.1f} MB  {s.rfilename}  lfs={'y' if s.lfs else 'n'}")
    print(f"TOTAL {tot/1e9:.3f} GB  nfiles={len(info.siblings)}")
