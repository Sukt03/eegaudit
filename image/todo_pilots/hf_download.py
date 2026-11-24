import sys
from huggingface_hub import hf_hub_download
D="/path/to/eegaudit/data_raw/eeg2text_ckpts"
jobs=[("pyinglie/EEG-To-Text","EEG-To-Text/train_decoding.py"),("pyinglie/EEG-To-Text","EEG-To-Text/environment.yml"),("pyinglie/EEG-To-Text","EEG-To-Text/scripts/train_decoding.sh"),
      ("Jash-176/EEG-to-Text-Checkpoints","finetune_final/best.pth")]
for rid,f in jobs:
    p=hf_hub_download(rid,f,local_dir=f"{D}/{rid.split('/')[0]}",cache_dir=f"{D}/.cache"); print("OK",p,flush=True)
