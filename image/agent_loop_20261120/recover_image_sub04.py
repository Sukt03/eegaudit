"""Restore the missing official training artifact only after its recorded SHA matches."""
import hashlib
import os
from pathlib import Path
from urllib.request import Request, urlopen

ROOT=Path(__file__).resolve().parents[2]
DATA=ROOT/"data_raw/things_eeg2_250hz"
REL=Path("Preprocessed_data_250Hz/sub-04/preprocessed_eeg_training.npy")
REV="dbe34bb2407164f70883c661c76664f0d596e522"
DIGEST="a658b1a82504f6f426a53a4a3ac28601b9988ec109b676d85b211a57a13d3e27"
assert (DATA/".cache/huggingface/download"/REL.with_suffix(".npy.metadata")).read_text().splitlines()[:2]==[REV,DIGEST]
stage=ROOT/"outputs/AGENT_LOOP_20261120/models/things_recovery"
print("allocation",os.getenv("SLURM_JOB_ID"),"step",os.getenv("SLURM_STEP_ID"),"revision",REV,flush=True)
if not (DATA/REL).exists():
    stage.mkdir(parents=True,exist_ok=True)
    downloaded=stage/"sub04_training.stream.part"
    offset=downloaded.stat().st_size if downloaded.exists() else 0
    url=f"https://huggingface.co/datasets/LidongYang/EEG_Image_decode/resolve/{REV}/{REL}"
    headers={"Range":f"bytes={offset}-"} if offset else {}
    with urlopen(Request(url,headers=headers),timeout=150) as response:
        if offset:
            assert response.status==206 and response.headers.get("Content-Range","").startswith(f"bytes {offset}-")
        with downloaded.open("ab" if offset else "xb") as out:
            while chunk:=response.read(2**20):
                out.write(chunk)
    with open(downloaded,"rb") as f:
        assert hashlib.file_digest(f,"sha256").hexdigest()==DIGEST
    os.link(downloaded, DATA/REL)
    Path(downloaded).unlink()
print(DATA/REL, (DATA/REL).stat().st_size, flush=True)
