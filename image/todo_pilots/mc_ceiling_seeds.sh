#!/bin/bash
# Re-runs exp_exact_joint.py (unchanged) with new --seed values (new simulated participants + new IS draws), writing to NEW dirs.
# rho grid split in two calls per seed so each srun stays under the 20-min cap.
cd /path/to/eegaudit
PY=python
for seed in 8 9; do
  for grid in 0.0,0.9 0.3,0.4; do
    tag=$(echo $grid | tr ',' '_')
    out=results/todo_pilots/ceiling_RWD_v2_seed${seed}_rho${tag}
    [ -f $out/results.json ] && continue
    echo "[$(date +%H:%M:%S)] START seed $seed rhos $grid"
    RHOS=$grid srun --overlap -N1 -n1 --cpus-per-task=2 --mem=16G -t 20 -J mc_ceiling_s${seed} \
      $PY -u exp_exact_joint.py --model RWD --nets results/rate_RWD_early/e1_nets.pt --out $out --n_test 200 --m_is 20000 --chunk 40000 --seed $seed \
      || echo "FAILED seed $seed rhos $grid"
    echo "[$(date +%H:%M:%S)] END seed $seed rhos $grid"
  done
done
echo "MC_DONE $(date +%H:%M:%S)"
