"""Plain-forecast scores of the lean_vol_vix 16:00 trees beside baseline and all_features.

User requests (2026-10-06): lean_vol_vix = the HAR + calendar backbone + sumret, sumabsret, sumret4,
sumautocov, sumvolume, vix (src/data/loading.py).  moments_vix and moments_vol_vix are dropped (user,
2026-10-06: "comment out the non lean results everywhere"; their tables are parked), so the comparisons
are lean_vol_vix vs baseline (HAR + calendar) and vs all_features.

The tables (rungs T1, T10, RS1, RS10 and the QLIKE-rule RS1q / RS10q; LightGBM, XGBoost, random forest;
experiments/trees_momvolvix_stack_1600.py) are scored with the MASTER TABLE's own plain scorer, imported
from experiments/master_table_close.py (not edited): target of results/linear_subsection_dedup,
load_one (16:00 rows, plain f^2 B, its gates) and score_one, on every trade day of the deck (866 days,
2020-01-03 .. 2024-04-30), no recalibration.  Levels use the master table's reference; the paired
comparisons call score_one with the comparison forecast as its reference, so its *_vs_ref columns are:
QLIKE % difference, Diebold-Mariano statistic and p-value of the per-day QLIKE difference (a - b;
negative = a more accurate), Sharpe mid / crossed differences, and the Newey-West t of the per-day mid
P&L difference of the sign(s) trade (q = sign(plain - iv_var); straddle = nearest out-of-the-money call +
nearest out-of-the-money put, same-day expiry, one position).
GATES: the baseline / all_features levels equal results/close_master_table/master_table.csv to REPRO_TOL;
load_one's gates on every table.

Writes (--out, default results/trees_momvolvix_1600/): levels_plain.csv (one row per rung x model x input
set), pairs_plain.csv (one row per rung x model x pair), score_gates.csv, SUMMARY.md (from the CSVs).

Run:  python experiments/trees_momvolvix_score_1600.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "notebooks", ROOT / "experiments"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import master_table_close as mtc  # noqa: E402

ARM_ROOT = "linear_subsection_dedup"  # the plain master table's --arm-root
MASTER = ROOT / "results" / "close_master_table" / "master_table.csv"
LEAN, BASE, AF = "lean_vol_vix", "baseline", "all_features"
# MVV = "moments_vol_vix"  # dropped (user, 2026-10-06): its tables are parked, not scored here
BUCKETS = (BASE, AF, LEAN)
PAIRS = ((LEAN, BASE), (LEAN, AF))  # (a, b): a - b
MODELS = ("lgbm", "xgb", "rf")
TREE_LONG = {"lgbm": "LightGBM", "xgb": "XGBoost", "rf": "random forest"}
REPRO_TOL = 1e-12
LEVELS = ("qlike_raw", "Sharpe_mid", "Sharpe_crossed", "pct_buy")
PAIRED = (
    "qlike_pct_vs_ref",
    "dm_vs_ref",
    "dm_p_vs_ref",
    "dSharpe_mid_vs_ref",
    "dSharpe_crossed_vs_ref",
    "t_hac_mid_vs_ref",
    "same_position_as_ref",
)
RUNGS = {
    "T1": "subtree_daily_{b}_{m}",
    "T10": "subtree_{m}_{b}",
    "RS1": "subtree_tuned_daily_{b}_{m}",
    "RS10": "subtree_tuned_{b}_{m}",
    "RS1q": "subtree_tunedq_daily_{b}_{m}",
    "RS10q": "subtree_tunedq_{b}_{m}",
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--tables", default=str(mtc.SPXW), help="where the new sets' tables are"
    )
    ap.add_argument("--out", default=str(ROOT / "results" / "trees_momvolvix_1600"))
    a = ap.parse_args()
    tables, out = Path(a.tables), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    mtc.set_arm_root(ARM_ROOT)
    deck = pd.read_parquet(mtc.base.DECK).sort_index()
    deck.index = pd.DatetimeIndex(pd.to_datetime(deck.index)).as_unit("ns")
    tgt = mtc.load_target()
    mtc._init(tgt, deck.index)
    paper = mtc._paper_tags()
    ref_path = next(mtc.SPXW / n for n, t in paper.items() if t == mtc.REFERENCE)
    ref = mtc.load_one(mtc.describe_table(ref_path, paper))["frame"]
    days = deck.index
    master = pd.read_csv(MASTER).set_index("key")
    gates: list[dict] = []
    levels: list[dict] = []
    pairs: list[dict] = []

    def load(path: Path, key: str) -> tuple[Any, Any]:
        spec = mtc.Spec(key, key, "per-bar tree", "table", str(path))
        res = mtc.load_one(spec)
        gates.extend(g | {"forecast": key} for g in res["gates"])
        if res["error"] or res["frame"] is None:
            gates.append(
                {
                    "gate": f"load ({res['error']})",
                    "forecast": key,
                    "ok": False,
                    "value": float("nan"),
                }
            )
            return spec, None
        f = res["frame"]
        miss = days.difference(f.index[f["plain"].notna()].normalize())
        if len(miss):
            gates.append(
                {
                    "gate": "covers every trade day",
                    "forecast": key,
                    "ok": False,
                    "value": float(len(miss)),
                }
            )
            return spec, None
        return spec, f

    for rung, pat in RUNGS.items():
        for m in MODELS:
            frames = {}
            for b in BUCKETS:
                key = pat.format(b=b, m=m)
                p = (tables if b == LEAN else mtc.SPXW) / f"yhat_{key}.parquet"
                if not p.is_file():
                    gates.append(
                        {
                            "gate": "table on disk",
                            "forecast": key,
                            "ok": False,
                            "value": float("nan"),
                        }
                    )
                    continue
                spec, f = load(p, key)
                if f is None:
                    continue
                frames[b] = (spec, f)
                lv = mtc.score_one((spec, f, ref, days, deck, False))
                levels.append(
                    {"rung": rung, "model": TREE_LONG[m], "bucket": b, "key": key}
                    | {c: float(lv[c]) for c in LEVELS}
                )
                if b in (BASE, AF):
                    dev = max(
                        abs(float(lv[c]) - float(master.loc[key, c])) for c in LEVELS
                    )
                    gates.append(
                        {
                            "gate": "reproduction of master_table.csv levels (max abs diff)",
                            "forecast": key,
                            "ok": dev <= REPRO_TOL,
                            "value": dev,
                        }
                    )
            for x, y in PAIRS:
                if x in frames and y in frames:
                    spec, f = frames[x]
                    r = mtc.score_one((spec, f, frames[y][1], days, deck, False))
                    pairs.append(
                        {"rung": rung, "model": TREE_LONG[m], "a": x, "b": y}
                        | {c: float(r[c]) for c in PAIRED}
                    )
    lv_df, pr_df, gt = pd.DataFrame(levels), pd.DataFrame(pairs), pd.DataFrame(gates)
    lv_df.to_csv(out / "levels_plain.csv", index=False)
    pr_df.to_csv(out / "pairs_plain.csv", index=False)
    gt.to_csv(out / "score_gates.csv", index=False)
    n_bad = int((~gt["ok"].astype(bool)).sum())
    write_summary(
        out, lv_df, pr_df, gt, len(days), str(days.min().date()), str(days.max().date())
    )
    print(
        lv_df.pivot_table(
            index=["rung", "model"], columns="bucket", values="qlike_raw", sort=False
        )
        .round(4)
        .to_string()
    )
    print(f"gates: {len(gt) - n_bad} ok, {n_bad} failed")
    return 1 if n_bad else 0


def write_summary(
    out: Path,
    lv: pd.DataFrame,
    pr: pd.DataFrame,
    gt: pd.DataFrame,
    n: int,
    d0: str,
    d1: str,
) -> None:
    def f(x: Any, k: int) -> str:
        return "" if pd.isna(x) else f"{float(x):.{k}f}"

    def L(r: str, m: str, b: str, c: str) -> Any:
        s = lv[(lv.rung == r) & (lv.model == m) & (lv.bucket == b)]
        return s[c].iloc[0] if len(s) else float("nan")

    def P(r: str, m: str, a: str, b: str, c: str) -> Any:
        s = pr[(pr.rung == r) & (pr.model == m) & (pr.a == a) & (pr.b == b)]
        return s[c].iloc[0] if len(s) else float("nan")

    lines = [
        "# lean_vol_vix 16:00 trees: plain-forecast scores beside baseline and all_features",
        "",
        "Written by `experiments/trees_momvolvix_score_1600.py` from `levels_plain.csv`, `pairs_plain.csv` and `score_gates.csv` in this folder.",
        "",
        "`lean_vol_vix` = HAR + calendar + `sumret`, "
        "`sumabsret`, `sumret4`, `sumautocov`, `sumvolume`, `vix`. The master table's plain scorer (forecast f^2 B, no recalibration; QLIKE "
        "against the per-bar 16:00 target, lower is better; the last-30-min trade q = sign(plain - iv_var) on the straddle -- nearest "
        "out-of-the-money call + nearest out-of-the-money put, same-day expiry, one position -- annualized Sharpe at mid / crossed fills), "
        f"{n} trade days {d0} .. {d1}. DM = Diebold-Mariano statistic of the per-day QLIKE difference a - b (negative = a more accurate), "
        "p in brackets; t HAC = Newey-West t of the per-day mid P&L difference a - b. Point estimates only, as the master table.",
        "",
        "| rung | model | QLIKE base / af / lean | Sharpe mid base / af / lean | Sharpe crossed base / af / lean "
        "| DM lean-base (p) | DM lean-af (p) | t HAC mid lean-base / lean-af |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for rung in RUNGS:
        for m in TREE_LONG.values():
            q = " / ".join(f(L(rung, m, b, "qlike_raw"), 4) for b in BUCKETS)
            sm = " / ".join(f(L(rung, m, b, "Sharpe_mid"), 2) for b in BUCKETS)
            sx = " / ".join(f(L(rung, m, b, "Sharpe_crossed"), 2) for b in BUCKETS)
            dms = [
                f"{f(P(rung, m, a, b, 'dm_vs_ref'), 2)} ({f(P(rung, m, a, b, 'dm_p_vs_ref'), 3)})"
                for a, b in PAIRS
            ]
            th = " / ".join(
                f(P(rung, m, a, b, "t_hac_mid_vs_ref"), 2) for a, b in PAIRS
            )
            lines.append(
                f"| {rung} | {m} | {q} | {sm} | {sx} | {dms[0]} | {dms[1]} | {th} |"
            )
    rep = gt[gt["gate"].astype(str).str.startswith("reproduction")]
    lines += [
        "",
        "Sharpe differences and QLIKE % differences of each pair are in `pairs_plain.csv`.",
        "",
        "## Gates",
        "",
        f"- Reproduction of the stored baseline / all_features levels: {int(rep['ok'].astype(bool).sum())} of {len(rep)} within {REPRO_TOL:g}.",
        f"- Every gate (incl. load_one's on each table): {int(gt['ok'].astype(bool).sum())} of {len(gt)} pass.",
    ]
    (out / "SUMMARY.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8", newline="\n"
    )


if __name__ == "__main__":
    sys.exit(main())
