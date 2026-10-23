"""Provenance of the cached UNOBSERVED_NEUTRAL flag-on w1000 design (unobs/d_on) before reusing it for check (d).

1. sha256 of every cache file (recorded);
2. its design.json == the committed results/unobserved_neutral_2026-10-22/design_on.json (bytes);
3. the executor source that built it (unobs/executor_preformat.py, saved before ruff format) has the same AST as the
   std worktree's src/backtest/executor.py, and src/ has no other change since 7b01cd7 that the design reads (git);
4. against run B's design (UNOBSERVED_NEUTRAL off): the values that differ are the committed verification's count
   (151,662 in 114 columns), all set to 0; y, b_oos, dates byte-identical.
"""

from __future__ import annotations

import ast
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

SP = Path(r"C:\Users\james\AppData\Local\Temp\claude\C--Users-james-CC-Allowed-harxhar-0dte-professor\bd4224e0-12aa-4614-b6a6-3fe50535ade1\scratchpad")
WT = Path(r"C:\Users\james\AppData\Local\Temp\std_wt")
ON = SP / "unobs" / "d_on" / "_work"
B = Path(r"C:\Users\james\AppData\Local\Temp\ls_wt\results\pooled_linear_rerun_2026-10-22_13_2012_w1000_latestart_nofomcrel\_work")


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 22), b""):
            h.update(blk)
    return h.hexdigest()


rep: dict = {"sha256": {}}
for s in ("baseline", "all_features"):
    for f in ("X.npy", "y.npy", "b_oos.npy", "dates_oos.csv", "design.json"):
        rep["sha256"][f"design_{s}/{f}"] = sha(ON / f"design_{s}" / f)
rep["design_json_equals_committed_design_on"] = (
    (ON / "design_all_features" / "design.json").read_bytes()
    == (WT / "results" / "unobserved_neutral_2026-10-22" / "design_on.json").read_bytes()
)
a1 = ast.dump(ast.parse((SP / "unobs" / "executor_preformat.py").read_text(encoding="utf-8")))
a2 = ast.dump(ast.parse((WT / "src" / "backtest" / "executor.py").read_text(encoding="utf-8")))
rep["executor_ast_equal_build_vs_head"] = a1 == a2
for s in ("baseline", "all_features"):
    r: dict = {}
    for f in ("y.npy", "b_oos.npy", "dates_oos.csv"):
        r[f + "_identical_to_B"] = (ON / f"design_{s}" / f).read_bytes() == (B / f"design_{s}" / f).read_bytes()
    Xo = np.load(ON / f"design_{s}" / "X.npy", mmap_mode="r")
    Xb = np.load(B / f"design_{s}" / "X.npy", mmap_mode="r")
    assert Xo.shape == Xb.shape, (Xo.shape, Xb.shape)
    n_diff, cols, nonzero_new = 0, 0, 0
    for j in range(Xo.shape[1]):
        a, b = np.asarray(Xo[:, j]), np.asarray(Xb[:, j])
        d = ~((a == b) | (np.isnan(a) & np.isnan(b)))
        if d.any():
            n_diff += int(d.sum())
            cols += 1
            nonzero_new += int((a[d] != 0.0).sum())
    r |= dict(X_values_changed=n_diff, X_columns_changed=cols, X_changed_values_not_zero=nonzero_new,
              shape=list(Xo.shape), f_order=bool(Xo.flags.f_contiguous))
    rep[s] = r
print(json.dumps(rep, indent=1))
(SP / "std" / "d_on_provenance.json").write_text(json.dumps(rep, indent=1))
sys.exit(0)
