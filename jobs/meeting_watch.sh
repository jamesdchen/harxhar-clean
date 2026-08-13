#!/bin/bash
# meeting_watch: every 30 min, check harvest progress; when tree_expert_16/18/19
# all complete (100 chunks), re-run the scorer and dump the tree verdict macros.
cd /u/scratch/j/jamesdc1/harxhar-clean
LOG=logs/meeting_watch.log
PY=/u/home/j/jamesdc1/.conda/envs/hpc-pi/bin/python
echo "$(date) watcher start" >> $LOG
while true; do
    n16=$(ls results/unification/tree_expert_16/chunk_*.npz 2>/dev/null | wc -l)
    n18=$(ls results/unification/tree_expert_18/chunk_*.npz 2>/dev/null | wc -l)
    n19=$(ls results/unification/tree_expert_19/chunk_*.npz 2>/dev/null | wc -l)
    echo "$(date) tree experts: 16=$n16 18=$n18 19=$n19" >> $LOG
    if [ "$n16" -ge 100 ] && [ "$n18" -ge 100 ] && [ "$n19" -ge 100 ]; then
        $PY experiments/score_unification.py --roots results/unification \
            --out /u/scratch/j/jamesdc1/tmp_opt/wave_scores_final.csv \
            --tex-dir /u/scratch/j/jamesdc1/tmp_opt/tex_final \
            > /u/scratch/j/jamesdc1/tmp_opt/scorer_final.log 2>&1
        echo "$(date) TREES COMPLETE — final scorer pass done" >> $LOG
        grep -E "TreeTuned|TreeHedge|IncrTree" /u/scratch/j/jamesdc1/tmp_opt/tex_final/campaign_numbers.tex >> $LOG 2>/dev/null
        grep "tree_tuned\|tree_hedge" /u/scratch/j/jamesdc1/tmp_opt/scorer_final.log | grep -v NOTE | head -6 >> $LOG
        exit 0
    fi
    sleep 1800
done
