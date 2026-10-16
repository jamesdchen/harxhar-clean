"""Cross-cluster audit of the moments_vix Optuna run ON THE NEW BUCKET ITSELF.

The same 20 moments_vix tuning points (OOS rows 0-19, 50 TPE trials each) computed on both
clusters: CARC's canary chunk 0 (epyc-7513; a fleet chunk) and Hoffman2's audit chunk 950
(cluster/submit_momvix_optuna_h2.sh, the qualified class), relayed to this machine by
`bash cluster/relay_momvix_optuna.sh audit`.  Compared key by key bit for bit (the keys of the
gate, experiments/momvix_optuna_gate.py); one row per (model, key) in <root>/audit/audit.csv.

Usage: python experiments/momvix_optuna_audit.py [--root results/linear_subsection_trees_optuna_mask_momvix]
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "experiments") not in sys.path:
    sys.path.insert(0, str(ROOT / "experiments"))
from momvix_optuna_gate import S1_KEYS, max_rel, same  # noqa: E402

R = "results/linear_subsection_trees_optuna_mask_momvix"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=R)
    a = ap.parse_args()
    root = ROOT / a.root
    rows, ok = [], True
    for m in ("lgbm", "xgb", "rf"):
        arm = Path("stage1") / "moments_vix" / "bar1600" / m / "tw2000" / "chunks"
        inner = Path("causal_tune_trees") / m / "moments_vix" / "trials_bar1600.npz"
        ph = root / "audit" / "h2" / R / "h2audit" / arm / "c950" / inner
        pc = root / "audit" / "carc" / R / arm / "c0" / inner
        if not (ph.is_file() and pc.is_file()):
            rows.append(
                {
                    "model": m,
                    "key": "-",
                    "bit_identical": False,
                    "note": f"missing h2 {not ph.is_file()} carc {not pc.is_file()}",
                }
            )
            ok = False
            continue
        h, c = dict(np.load(ph)), dict(np.load(pc))
        hosts = []
        for p in (ph, pc):
            ht = p.parents[3] / "host.txt"
            hosts.append(ht.read_text().split("\n")[1] if ht.is_file() else "")
        for k in S1_KEYS:
            good = same(h[k], c[k])
            ok = ok and good
            rows.append(
                {
                    "model": m,
                    "key": k,
                    "bit_identical": good,
                    "max_rel_diff": max_rel(h[k], c[k]),
                    "h2_cpu": hosts[0],
                    "carc_cpu": hosts[1],
                    "points": len(c["row"]),
                }
            )
        print(
            f"{m}: {sum(r['bit_identical'] for r in rows if r['model'] == m)} of {len(S1_KEYS)} keys bit-identical "
            f"({len(c['row'])} points; h2 {hosts[0]} | carc {hosts[1]})"
        )
    out = root / "audit" / "audit.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(
            fh,
            fieldnames=sorted(
                {k for r in rows for k in r},
                key=lambda k: (k not in ("model", "key"), k),
            ),
        )
        w.writeheader()
        w.writerows(rows)
    print(f"audit {'BIT-IDENTICAL' if ok else 'DIFFERS'}; wrote {out}")


if __name__ == "__main__":
    main()
