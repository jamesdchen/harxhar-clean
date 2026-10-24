#!/bin/bash
#$ -cwd
#$ -j y
#$ -l h_rt=1:00:00,h_data=8G
# The Hoffman2 gates' check, after every gate array (submit_h2.sh): an MD5SUMS file in each tag's g1/pieces and
# g2/pieces (finished parquet + json), then experiments/trees_train3way.py accept -- the check CARC runs on the
# delivered gates (gate 1 <= 1e-14 relative against the templates, gate 2 bit for bit, one CPU model a tag) ->
# $B/h2/accept/pieces_h2/accepted_cpus.json -- and h2/code/gate_summary.py -> $B/h2/gate_summary.{json,txt},
# passed_cpus.txt, walk_resources.txt (the walk waves' arch and h_rt; submit_wave.sh).
set -uo pipefail
. /u/local/Modules/default/init/modules.sh
module load apptainer/1.2.2
B=/u/scratch/j/jamesdc1/t3t
echo "t3t gate summary job ${JOB_ID:-} on $(hostname) $(date)"
for d in $B/gate/h2/*/g1/pieces $B/gate/h2/*/g2/pieces; do
  (cd $d && ls | grep -E '[.](parquet|json)$' | grep -v '[.]ckpt[.]json$' | sort | xargs -r md5sum > MD5SUMS)
  echo "$d: $(wc -l < $d/MD5SUMS) files"
done
apptainer exec -B /u/scratch $B/deb.sif bash -c '
B=/u/scratch/j/jamesdc1/t3t
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 LD_LIBRARY_PATH=$B/lib OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd $B/code
export PYTHONPATH=$B/code:$B/code/experiments
$B/venv_gb/bin/python experiments/trees_train3way.py accept --h2-gates $B/gate/h2 --out $B/h2/accept
$B/venv_gb/bin/python $B/jobs/gate_summary.py --h2-gates $B/gate/h2 --carc-gates $B/carc/gate \
  --accepted $B/h2/accept/pieces_h2/accepted_cpus.json --tags $B/h2/tags.txt --design-root $B/carc/designs \
  --taskfile $B/runs/tasks/h2_w3rf_full.txt --out $B/h2
'
echo "== passed CPU models"; cat $B/h2/passed_cpus.txt 2>/dev/null
echo "== walk resources"; cat $B/h2/walk_resources.txt 2>/dev/null
echo "gate summary done $(date)"
