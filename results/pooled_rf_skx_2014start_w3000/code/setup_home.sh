#!/bin/bash
# /home1 copies for the rf_skx walks (CARC login node; some compute nodes have a failing /scratch1 mount):
#   1. the md5 check of the env copy /home1/jc_905/envs/harxhar-r13 against /scratch1/jc_905/envs/harxhar-r13 (every
#      shared library and executable)
#   2. the venv $HB/venv_skx on that copy (--system-site-packages) + scikit-learn-intelex 2026.2.0 / daal 2026.2.0 /
#      tbb 2023.2.0 / tcmlib 1.6.0 (the benchmark's versions), its shared libraries md5-compared with the benchmark's
#      venv /scratch1/jc_905/trees_final/venv_skx
#   3. the code: git archive of <commit> (tracked tree without results/, plus this run's code and rule) -> $HB/code_<c>
#   4. then sbatch stage_home.sbatch (designs and benchmark forecasts copied to /home1, md5-verified)
# usage: bash setup_home.sh <commit>
set -euo pipefail
C=$1
HB=/home1/jc_905/trees_final
SB=/scratch1/jc_905/trees_final
E1=/home1/jc_905/envs/harxhar-r13
E0=/scratch1/jc_905/envs/harxhar-r13
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
mkdir -p $HB/logs $HB/gate_skx $HB/runs_skx $HB/designs $HB/bench
echo "== 1. env copy md5"
(cd $E0 && find . -type f \( -name "*.so*" -o -path "./bin/*" \) | sort | xargs md5sum) > $HB/env_md5_scratch1.txt
(cd $E1 && find . -type f \( -name "*.so*" -o -path "./bin/*" \) | sort | xargs md5sum) > $HB/env_md5_home1.txt
if cmp -s $HB/env_md5_scratch1.txt $HB/env_md5_home1.txt; then
  echo "env copy: $(wc -l < $HB/env_md5_home1.txt) shared libraries / executables md5-equal"
else
  echo "env copy differs:"; diff $HB/env_md5_scratch1.txt $HB/env_md5_home1.txt | head -20
  diff <(awk '{print $2}' $HB/env_md5_scratch1.txt) <(awk '{print $2}' $HB/env_md5_home1.txt) | head; exit 3
fi
echo "== 2. venv_skx"
if [ ! -d $HB/venv_skx ]; then $E1/bin/python -m venv --system-site-packages $HB/venv_skx; fi
$HB/venv_skx/bin/python -m pip install -q scikit-learn-intelex==2026.2.0 daal==2026.2.0 tbb==2023.2.0 tcmlib==1.6.0 2>&1 | grep -v notice || true
$HB/venv_skx/bin/python -m pip freeze --local 2>/dev/null
L1=$($HB/venv_skx/bin/python -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")
L0=$($SB/venv_skx/bin/python -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")
(cd $L0 && find . -type f -name "*.so*" | sort | xargs md5sum) > $HB/venv_skx_md5_scratch1.txt
(cd $L1 && find . -type f -name "*.so*" | sort | xargs md5sum) > $HB/venv_skx_md5_home1.txt
cmp $HB/venv_skx_md5_scratch1.txt $HB/venv_skx_md5_home1.txt && echo "venv_skx: $(wc -l < $HB/venv_skx_md5_home1.txt) shared libraries md5-equal to the benchmark's venv"
$HB/venv_skx/bin/python -c "import sys, sklearnex, sklearn, numpy, lightgbm; print(sys.executable, sys.prefix, 'sklearn', sklearn.__version__, 'numpy', numpy.__version__, 'lightgbm', lightgbm.__version__); print(sklearn.__file__, numpy.__file__)"
echo "== 3. code"
git -C $SB/repo fetch -q origin insurance-selling-2026-09-19-e34-trees-final
S=$(git -C $SB/repo rev-parse --short=7 "$C")
D=$HB/code_$S
if [ ! -d $D ]; then
  mkdir -p $D
  P=$(git -C $SB/repo ls-tree --name-only "$C" | grep -v '^results$')
  git -C $SB/repo archive "$C" $P results/pooled_rf_skx_2014start_w3000 results/pooled_trees_final_2014start_w3000/code \
    results/rf_engine_bench_2026-10-23/code | tar -x -C $D
  echo "$(git -C $SB/repo rev-parse "$C")" > $D/COMMIT
fi
echo "code $D at $(cat $D/COMMIT), $(du -sh $D | cut -f1)"
echo "== 4. stage"
sbatch --parsable -J tf_stage_home -o $HB/logs/%x.%j.out $D/results/pooled_rf_skx_2014start_w3000/code/stage_home.sbatch
