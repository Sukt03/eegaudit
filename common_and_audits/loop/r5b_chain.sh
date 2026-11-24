#!/bin/bash
cd /path/to/eegaudit
while pgrep -f contrastive_retrieval.py >/dev/null; do sleep 30; done
srun --overlap -N1 -n1 --cpus-per-task=4 --mem=14G -t 60 python -I experiments/loop/r5_eegnet_splithalf.py 0 9 --epochs 10 > experiments/loop/logs/r5b_0-9.log 2>&1
echo R5B_DONE >> experiments/loop/logs/r5b_0-9.log
