#!/bin/bash
# Pull the unification campaign's npz trees from BOTH clusters (run locally,
# Git Bash). CARC -> results/unification_carc, Hoffman2 -> results/unification_h2.
# Incremental: tar streams everything; rerunning overwrites in place.
set -e
SSH=C:/Windows/System32/OpenSSH/ssh.exe
cd "$(dirname "$0")/.."
mkdir -p results/unification_carc results/unification_h2
echo "== CARC =="
$SSH usc-discovery \
  "cd /scratch1/jc_905/harxhar-clean/results/unification && tar cf - ." \
  | tar xf - -C results/unification_carc
echo "CARC files: $(find results/unification_carc -name '*.npz' | wc -l)"
echo "== Hoffman2 =="
$SSH hoffman2 \
  "cd /u/scratch/j/jamesdc1/harxhar-clean/results/unification && tar cf - ." \
  | tar xf - -C results/unification_h2
echo "H2 files: $(find results/unification_h2 -name '*.npz' | wc -l)"
