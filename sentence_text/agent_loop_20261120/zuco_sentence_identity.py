"""Check the actual texts attached to sentence indices across readers and splits."""
import argparse
import hashlib
import json
import os
import pickle
import unicodedata
from collections import defaultdict
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
FILES={"SR":"task1-SR-dataset.pickle","NR2":"task2-NR-2.0-dataset.pickle"}


def key(text):
    return " ".join(unicodedata.normalize("NFKC",text).casefold().split())


def main():
    p=argparse.ArgumentParser()
    p.add_argument("task",choices=FILES)
    task=p.parse_args().task
    path=ROOT/"data_raw/zuco/eeg2text_pickles"/FILES[task]
    with path.open("rb") as f:
        data=pickle.load(f)
    by_index=defaultdict(set)
    for records in data.values():
        for i,record in enumerate(records):
            if record is not None:
                by_index[i].add(key(record["content"]))
    ids=sorted(by_index)
    cut1,cut2=int(.8*len(ids)),int(.9*len(ids))
    split={i:s for s,section in (("train",ids[:cut1]),("dev",ids[cut1:cut2]),("test",ids[cut2:])) for i in section}
    occurrences=defaultdict(set)
    for i,strings in by_index.items():
        for string in strings:
            occurrences[string].add(split[i])
    overlaps={pair:sum(set(pair.split("/"))<=v for v in occurrences.values()) for pair in ("train/dev","train/test","dev/test")}
    mismatches={i:len(v) for i,v in by_index.items() if len(v)>1}
    result=dict(task=task,readers=len(data),n_indices=len(ids),split_sizes=[cut1,cut2-cut1,len(ids)-cut2],
                normalization="Unicode NFKC, casefold, collapse whitespace",overlap_distinct_text_counts=overlaps,
                mixed_text_indices=mismatches,
                train_test_overlap_digest=[hashlib.sha256(s.encode()).hexdigest() for s,v in occurrences.items() if {"train","test"}<=v],
                allocation=os.getenv("SLURM_JOB_ID"),step=os.getenv("SLURM_STEP_ID"))
    out=ROOT/f"outputs/AGENT_LOOP_20261120/results/sentence_identity_{task}.json"
    tmp=out.with_suffix(".tmp")
    tmp.write_text(json.dumps(result,indent=2)+"\n")
    tmp.replace(out)
    print(task,len(mismatches),overlaps,flush=True)
if __name__=="__main__":
    main()
