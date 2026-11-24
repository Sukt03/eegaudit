#!/bin/bash
cd /path/to/eegaudit/experiments/eegbits
# (wait loop removed: pgrep -f matched the controlling shells; job 0 had already finished when this was relaunched at 12:52)
run() { srun --overlap -N1 -n1 --cpus-per-task=4 --mem=28G -t 30 python -u /path/to/eegaudit/experiments/loop/fm_probe_r6.py --model $1 --subject $2 2>&1 | grep -v -i "warn" | tail -2; }
for s in 01 02 03 05 06 07 08 09 10; do echo "START cbramod sub-$s $(date +%H:%M)"; run cbramod sub-$s; done
for s in 02 03 05 06 07 08 09 10; do echo "START labram sub-$s $(date +%H:%M)"; run labram sub-$s; done
for s in 01 02 03 05; do echo "START eegpt sub-$s $(date +%H:%M)"; run eegpt sub-$s; done
echo R6_DONE
