#!/usr/bin/env bash
# Download Alice / Brennan & Hale 2019 EEG (NEMAR nm000180, CC-BY-4.0, BIDS, 45 participants, ~3.4 GB) into data_raw/third_party/alice_nm000180.
# Git metadata from github.com/nemarDatasets/nm000180; annexed content from the public NEMAR S3 remote (publicurl in the git-annex remote.log).
# Each annexed file is checked against the byte size encoded in its SHA256E key. Skips files already complete.
set -uo pipefail
OUT=/path/to/eegaudit/data_raw/third_party/alice_nm000180; cd "$OUT"
RAW=https://raw.githubusercontent.com/nemarDatasets/nm000180/main; S3=https://nemar.s3.us-east-2.amazonaws.com/nm000180/objects
curl -s "https://api.github.com/repos/nemarDatasets/nm000180/git/trees/main?recursive=1" > .tree.json
python3 -c "
import json; t=json.load(open('.tree.json'))['tree']
for x in t:
    if x['mode'] in ('100644','120000') and not x['path'].startswith('.github'): print(x['mode'], x['path'])" > .files.txt
echo "files: $(wc -l < .files.txt)"
get() { mode=$1; p=$2; mkdir -p "$(dirname "$p")"
  c=$(curl -s -f --retry 3 "$RAW/$p" | head -c 400 | tr -d '\0')
  if [[ "$c" == *annex/objects/* ]]; then k=$(basename "$(echo "$c" | head -1)"); sz=${k#*-s}; sz=${sz%%--*}
    [ -f "$p" ] && [ "$(stat -c %s "$p")" = "$sz" ] && return 0
    curl -s -f --retry 3 -o "$p.part" "$S3/$k" && [ "$(stat -c %s "$p.part")" = "$sz" ] && mv "$p.part" "$p" || echo "FAILED $p"
  else [ -s "$p" ] || curl -s -f --retry 3 -o "$p" "$RAW/$p" || echo "FAILED $p"; fi; }
export -f get; export RAW S3
cat .files.txt | xargs -P 8 -n 2 bash -c 'get "$0" "$1"'
echo "ALICE_DONE $(date +%H:%M) $(du -sh . | cut -f1)"
