#!/bin/bash
cd /path/to/eegaudit
for i in $(seq 1 30); do [ -f results/loop/r6_fm_trialshuffle/fm_probe_labram_sub-10.json ] && break; [ "$(date +%H%M)" -ge 1350 ] && break; sleep 20; done
pkill -f r6_chain.sh; sleep 2; echo "R6 chain stopped $(date +%H:%M)"
srun --overlap -N1 -n1 --cpus-per-task=4 --mem=28G -t 40 python -I experiments/loop/p1_rank_bound.py > experiments/loop/logs/p1_full.log 2>&1; echo "P1 done $(date +%H:%M)"
for m in eeg noise; do srun --overlap -N1 -n1 --cpus-per-task=4 --mem=20G -t 20 python -I experiments/loop/contrastive_retrieval.py --cache /path/to/eegaudit/results/track_text/brain_translator/cache_full.pkl --out_dir /path/to/eegaudit/results/loop/p4/text --tag z_${m}_seed0 --input $m --seed 0 --epochs 30 --emb_cache /path/to/eegaudit/results/loop/r4_contrastive/minilm_full.npz --save_z > experiments/loop/logs/p4_text_$m.log 2>&1; done; echo "P4 text runs done $(date +%H:%M)"
