#!/bin/bash
# Pull the unification campaign's npz tree from CARC into
# results/unification_carc/ (run locally, Git Bash). Incremental: tar
# streams everything; rerunning overwrites in place.
set -e
SSH=C:/Windows/System32/OpenSSH/ssh.exe
cd "$(dirname "$0")/.."
mkdir -p results/unification_carc
$SSH usc-discovery \
  "cd /scratch1/jc_905/harxhar-clean/results/unification && tar cf - ." \
  | tar xf - -C results/unification_carc
echo "CARC files: $(find results/unification_carc -name '*.npz' | wc -l)"
