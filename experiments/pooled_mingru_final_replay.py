"""Read-only replay of the tuning with a fixed extension trigger (2026-10-23, owner via coordinator), on the existing
tune outputs of one window (runs + salvaged copies, collect_jobs / tune_table); no GPU job.

Selection unchanged: the plain argmin of the 5-seed-mean validation QLIKE over every configuration of the current
grid (sorted name order, ties: the first).  Trigger: after a round, for each axis whose argmin value sits at an
extensible end of the current grid (not a natural bound, not a ladder cap), the axis is extended by the next ladder
value ONLY if the best configuration at that edge value beats the best configuration at the adjacent interior value
of the axis (both over the current grid) by more than k day-block bootstrap SE of their paired QLIKE difference
(fold 3's validation days, moving blocks of 10 days, 2,000 resamples, seed 20261023 -- the noise analysis'
bootstrap); all qualifying axes at once (full factorial).  No qualifying axis: stop, the argmin is frozen.  A round
needing configurations that have not run (completely, 5 seeds) stops the replay and lists them.  k = 1 and k = 2.

    PMF_WINDOW_DAYS=3000 python pooled_mingru_final_replay.py <root> <out dir>
"""

from __future__ import annotations

import itertools
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT, OUT = Path(sys.argv[1]), Path(sys.argv[2])
sys.path.insert(0, str(ROOT / "code"))
import pooled_mingru_final_2014_w3000 as m  # noqa: E402
import pooled_mingru_final_tune_analysis as an  # noqa: E402

B, BLOCK, SEED = 2000, 10, 20261023
N_REFITS = 169_850


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    data, runs = ROOT / "data", ROOT / "runs"
    rows = m.load_rows(data / "pmf_rows.csv")
    v0, v1 = m.FOLD["val"]
    day = rows["date"].iloc[v0:v1].dt.normalize().to_numpy()
    _, dcode = np.unique(day, return_inverse=True)
    nd = int(dcode.max()) + 1
    rng = np.random.default_rng(SEED)
    starts = rng.integers(0, nd - BLOCK + 1, size=(B, -(-nd // BLOCK)))
    boot_days = (starts[:, :, None] + np.arange(BLOCK)[None, None, :]).reshape(B, -1)[
        :, :nd
    ]
    report: dict = {}
    for s in m.SETS:
        f_by_id, _, y = m.collect_jobs(data, runs, s)
        t = m.tune_table(data, runs, s).set_index("cfg")
        yv = y[v0:v1]
        ok = yv > 0
        cnt = np.bincount(dcode[ok], minlength=nd).astype(float)
        ec = an.epoch_cost(s)
        dsum_cache: dict[str, np.ndarray] = {}

        def dsum(c: str) -> np.ndarray:
            if c not in dsum_cache:
                F = np.vstack([f_by_id[f"ftune_{s}_f3_{c}_s{sd}"] for sd in m.SEEDS])
                q = (yv[ok] / F.mean(axis=0)[ok]) ** 2
                dsum_cache[c] = np.bincount(
                    dcode[ok], weights=q - np.log(q) - 1.0, minlength=nd
                )
            return dsum_cache[c]

        def boot_se(a: str, b: str) -> float:
            d = dsum(a) - dsum(b)
            bs = d[boot_days].sum(axis=1) / cnt[boot_days].sum(axis=1)
            return float(bs.std(ddof=1))

        def cost(c: str) -> dict:
            mt = re.match(r"^L(\d+)_H(\d+)_", c)
            assert mt is not None
            L, H = int(mt.group(1)), int(mt.group(2))
            eps = [int(t.loc[c, f"best_epoch_s{sd}"]) for sd in m.SEEDS]
            per = ec.get((H, L), float("nan")) * float(np.mean(eps))
            return dict(
                epochs=eps, p100_s_per_refit=per, walk_p100_h=per * N_REFITS / 3600
            )

        fin = t[np.isfinite(t["val_qlike"])]
        all_cfgs = sorted(fin.index)
        res_k = {}
        for k in (1, 2):
            grid = {ax: [m.fmt(ax, v) for v in vv] for ax, vv in m.GRID0.items()}
            rounds = []
            while True:
                names = [
                    m.cfg_name(dict(zip(m.AXES, tt)))
                    for tt in itertools.product(*(grid[a] for a in m.AXES))
                ]
                missing = [c for c in names if c not in t.index]
                if missing:
                    by_hl = pd.Series(
                        [c.split("_lr")[0] for c in missing]
                    ).value_counts()
                    rounds.append(
                        dict(
                            stop="configurations not run",
                            n_missing=len(missing),
                            missing_by_LH=by_hl.to_dict(),
                        )
                    )
                    break
                cur = fin.loc[[c for c in sorted(names) if c in fin.index]]
                pick = str(cur["val_qlike"].idxmin())
                pc = m.parse_cfg(pick)
                tests, ext = [], {}
                for ax in m.AXES:
                    vals = sorted(grid[ax], key=float)
                    lad = m.ladder_str(ax)
                    v = pc[ax]
                    i = lad.index(v)
                    end = None
                    if v == vals[0] and m.NATURAL_LOW.get(ax) != v and i > 0:
                        end, nxt, adj = "low", lad[i - 1], vals[1]
                    elif v == vals[-1] and i < len(lad) - 1:
                        end, nxt, adj = "high", lad[i + 1], vals[-2]
                    if end is None:
                        continue
                    cv = cur.index.map(lambda c, ax=ax: m.parse_cfg(c)[ax])
                    best_edge = str(cur[cv == v]["val_qlike"].idxmin())
                    best_adj = str(cur[cv == adj]["val_qlike"].idxmin())
                    d = float(
                        cur.loc[best_adj, "val_qlike"] - cur.loc[best_edge, "val_qlike"]
                    )
                    se = boot_se(best_adj, best_edge)
                    yes = d > k * se
                    tests.append(
                        dict(
                            axis=ax,
                            end=end,
                            edge=v,
                            adjacent=adj,
                            best_edge=best_edge,
                            q_edge=float(cur.loc[best_edge, "val_qlike"]),
                            best_adjacent=best_adj,
                            q_adjacent=float(cur.loc[best_adj, "val_qlike"]),
                            diff=d,
                            se=se,
                            extend=bool(yes),
                            add=nxt if yes else None,
                        )
                    )
                    if yes:
                        ext[ax] = nxt
                rounds.append(
                    dict(
                        n_configs=len(names),
                        pick=pick,
                        q_pick=float(cur.loc[pick, "val_qlike"]),
                        tests=tests,
                    )
                )
                if not ext:
                    break
                for ax, nv in ext.items():
                    grid[ax] = sorted(set(grid[ax]) | {nv}, key=float)
            last = [r for r in rounds if "pick" in r][-1]
            pick = last["pick"]
            # rank: configurations (all run) within 1 / 2 SE of the final pick
            ds = dsum(pick)
            within = {1: 0, 2: 0}
            for c in all_cfgs:
                if c == pick:
                    continue
                d = float(fin.loc[c, "val_qlike"] - fin.loc[pick, "val_qlike"])
                if d < 0:
                    continue
                se = float(
                    (
                        (dsum(c) - ds)[boot_days].sum(axis=1)
                        / cnt[boot_days].sum(axis=1)
                    ).std(ddof=1)
                )
                for kk in (1, 2):
                    within[kk] += d <= kk * se
            n_better = int((fin["val_qlike"] < fin.loc[pick, "val_qlike"]).sum())
            res_k[k] = dict(
                rounds=rounds,
                final_pick=pick,
                final_q=float(fin.loc[pick, "val_qlike"]),
                complete=("stop" not in rounds[-1]),
                n_all_configs_better=n_better,
                n_within_1se=within[1],
                n_within_2se=within[2],
                **cost(pick),
            )
            print(s, "k", k, json.dumps(res_k[k], default=str)[:1500], flush=True)
        report[s] = res_k
    (OUT / "replay.json").write_text(json.dumps(report, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
