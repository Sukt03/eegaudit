#!/bin/bash
P=python
cd /path/to/eegaudit/experiments/track_text
for T in SR NR2 NR; do
  echo "=== within-task $T $(date)"; $P -I -u pilot_retrieval.py --tasks $T --seeds 0,1,2 --out /path/to/eegaudit/results/track_text/within_$T 2>&1 | grep -v -i deprecat
done
echo "=== pooled SR,NR,NR2 $(date)"; $P -I -u pilot_retrieval.py --tasks SR,NR,NR2 --seeds 0,1,2 --out /path/to/eegaudit/results/track_text/pooled_SR_NR_NR2 2>&1 | grep -v -i deprecat
echo "ALL DONE $(date)"
