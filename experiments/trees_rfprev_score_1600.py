"""Stack and score per-bar 16:00 random-forest runs against reference forests (2026-10-07; PREVIEW, nothing
goes to results/spxw_pnl/, the paper, the GUI or the master table).

Stack (--stack ROOT:TOKEN): the campaign's untuned builder (experiments/build_subsection_tree_yhat.py, unedited,
its gates on) on ROOT/{t1,t10} for the random forest and baseline / all_features / lean_vol_vix, written to
OUT/stack/<rung>/ and copied as OUT/yhat_subtree_daily_<bucket>_<TOKEN>.parquet (T1) and
OUT/yhat_subtree_<TOKEN>_<bucket>.parquet (T10).

Score: every table of OUT with TOKEN (the new forest) against the comparison forests (--compare
NAME=DIR:TOKEN, default rfdev in results/spxw_pnl/*_rfdev and the default forest = the canonical rf tables),
with the master table's plain scorer imported from experiments/master_table_close.py (load_one: 16:00 rows, plain
f^2 B and its gates; score_one: QLIKE and the sign(s) trade q = sign(plain - iv_var) on the straddle -- nearest
out-of-the-money call + nearest out-of-the-money put, same-day expiry, one position) on the 866 trade days
2020-01-03 .. 2024-04-30, no recalibration.  Per forecast: QLIKE; Sharpe and Sortino (the GUI's rows._sortino)
at mid and crossed fills; the threshold score ES_1 = experiments/close_murphy_splits_headline.elementary(
plain / iv_var, target / iv_var, c = 1) (the elementary score of the buy / sell decision, IV-scaled; lower is
better); the robust-clip Sharpe (rows._score_sizes: q = clip(s / median|s| of the earlier days, -1, 1),
s = plain - iv_var, scored on the 614 days after the 252-day warm-up) at mid and crossed fills.  Per pair (new -
comparison): QLIKE % difference, Diebold-Mariano statistic and p of the daily QLIKE difference (score_one with
the comparison as reference), Newey-West t of the daily mid and crossed P&L differences, the threshold-score
difference and the Newey-West t of its daily difference, days with a different position.
Gates: the default forest's levels equal results/close_master_table/master_table.csv (1e-12); load_one's gates.

Run:  python experiments/trees_rfprev_score_1600.py --stack results/linear_subsection_trees_mask_rfprev163:rfp163
          --token rfp163 --out results/trees_rfdev_preview163
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for _p in (
    ROOT,
    ROOT / "notebooks",
    ROOT / "experiments",
    ROOT / "writeup" / "master_table_gui",
):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import atm_straddle_lib as asl  # noqa: E402
import build_subsection_tree_yhat as bst  # noqa: E402
import master_table_close as mtc  # noqa: E402
import rows as gui_rows  # noqa: E402  (writeup/master_table_gui/rows.py: Sortino, robust clip)
from close_murphy_splits_headline import elementary  # noqa: E402

ARM_ROOT = "linear_subsection_dedup"
MASTER = ROOT / "results" / "close_master_table" / "master_table.csv"
BUCKETS = ("baseline", "all_features", "lean_vol_vix")
RUNGS = ("T1", "T10")
REPRO_TOL = 1e-12


# forecast families (2026-10-07, rf400): rung label -> (cadence, table-name template).  A comparison of another family
# is matched on the cadence (daily = refit every row, ten = every 10th row): fixed T1 <-> tuned RS1 <-> Optuna paths.
OPTUNA_PATHS = (
    "tp1",
    "tp5",
    "tp25",
    "tp250",
    "tp1_k10",
    "tp1_k25",
    "tp25_k10",
    "tp25_k25",
)
FAMILIES: dict[str, dict[str, tuple[str, str]]] = {
    "fixed": {
        "T1": ("daily", "yhat_subtree_daily_{b}_{tok}.parquet"),
        "T10": ("ten", "yhat_subtree_{tok}_{b}.parquet"),
    },
    "nomask": {
        "T1": ("daily", "yhat_subtree_nomask_daily_{b}_{tok}.parquet"),
        "T10": ("ten", "yhat_subtree_nomask_{tok}_{b}.parquet"),
    },
    "tuned": {
        "RS1": ("daily", "yhat_subtree_tuned_daily_{b}_{tok}.parquet"),
        "RS10": ("ten", "yhat_subtree_tuned_{b}_{tok}.parquet"),
    },
    "tunedq": {
        "RS1q": ("daily", "yhat_subtree_tunedq_daily_{b}_{tok}.parquet"),
        "RS10q": ("ten", "yhat_subtree_tunedq_{b}_{tok}.parquet"),
    },
    **{
        f"optuna_{pth}": {
            f"Optuna {pth}": (
                "daily",
                f"yhat_subtree_optuna_{pth}_{{b}}_{{tok}}.parquet",
            )
        }
        for pth in OPTUNA_PATHS
    },
    "optunaq_tp25": {
        "Optuna q tp25": ("daily", "yhat_subtree_optunaq_tp25_{b}_{tok}.parquet")
    },
}


def table_name(rung: str, bucket: str, token: str, family: str = "fixed") -> str:
    """rung: a rung label of `family`, or a cadence (daily / ten)."""
    fam = FAMILIES[family]
    if rung in fam:
        return fam[rung][1].format(b=bucket, tok=token)
    for cad, tpl in fam.values():
        if cad == rung:
            return tpl.format(b=bucket, tok=token)
    raise KeyError(f"{family} has no rung / cadence {rung}")


def has_cadence(family: str, cadence: str) -> bool:
    return any(cad == cadence for cad, _ in FAMILIES[family].values())


def stack(root: Path, token: str, out: Path) -> list[str]:
    bst.BARS = ("bar1600",)  # type: ignore[assignment]
    bst.MODELS = ("rf",)  # type: ignore[assignment]
    bst.BUCKETS = BUCKETS  # type: ignore[assignment]
    problems: list[str] = []
    for rung in ("t1", "t10"):
        st = out / "stack" / rung
        st.mkdir(parents=True, exist_ok=True)
        saved = sys.argv
        try:
            sys.argv = [bst.__file__, "--root", str(root / rung), "--out", str(st)]
            bst.main()
        except SystemExit as e:
            if e.code not in (None, 0):
                problems.append(f"{rung}: {e.code}")
        finally:
            sys.argv = saved
        for b in BUCKETS:
            src = st / f"yhat_subtree_rf_{b}.parquet"
            if not src.is_file():
                problems.append(f"missing {src}")
                continue
            dst = out / table_name(rung.upper(), b, token)
            shutil.copyfile(src, dst)
            print(f"  {src.name} ({rung}) -> {dst.relative_to(ROOT)}")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--stack", default="", help="ROOT:TOKEN -- build the new forest's tables first"
    )
    ap.add_argument(
        "--token", required=True, help="key token of the new forest's tables in --out"
    )
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--label",
        default="",
        help="name of the new forest in the tables (default: the token)",
    )
    ap.add_argument(
        "--compare",
        action="append",
        default=[],
        help="NAME=DIR:TOKEN[@FAMILY] (repeatable; FAMILY defaults to --family)",
    )
    ap.add_argument("--family", default="fixed", choices=sorted(FAMILIES))
    ap.add_argument(
        "--score-dir",
        default="",
        help="where levels / pairs / gates / SUMMARY go (default: --out)",
    )
    a = ap.parse_args()
    out = (ROOT / a.out) if not Path(a.out).is_absolute() else Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    gates: list[dict] = []
    if a.stack:
        r, tok = a.stack.rsplit(":", 1)
        probs = stack(ROOT / r, tok, out)
        gates += [
            {"gate": "stack", "forecast": p, "ok": False, "value": np.nan}
            for p in probs
        ]
    compare = [x.split("=", 1) for x in a.compare] or [
        ["rfdev", "results/spxw_pnl:rfdev"],
        ["default RF", "results/spxw_pnl:rf"],
    ]
    new_name = a.label or a.token
    sdir = (ROOT / a.score_dir) if a.score_dir else out
    sdir.mkdir(parents=True, exist_ok=True)
    mtc.set_arm_root(ARM_ROOT)
    deck = pd.read_parquet(mtc.base.DECK).sort_index()
    deck.index = pd.DatetimeIndex(pd.to_datetime(deck.index)).as_unit("ns")
    tgt = mtc.load_target()
    mtc._init(tgt, deck.index)
    paper = mtc._paper_tags()
    ref = mtc.load_one(
        mtc.describe_table(
            next(mtc.SPXW / n for n, t in paper.items() if t == mtc.REFERENCE), paper
        )
    )["frame"]
    days = deck.index
    dk = deck.loc[days]
    iv = dk["iv_var"].to_numpy(float)
    R = dk["R"].to_numpy(float)
    ex = dk["exit"].to_numpy(float)
    bid = (dk["bid_c"] + dk["bid_p"]).to_numpy(float)
    ask = (dk["ask_c"] + dk["ask_p"]).to_numpy(float)
    master = pd.read_csv(MASTER).set_index("key")
    levels: list[dict] = []
    pairs: list[dict] = []

    def load(path: Path) -> tuple[Any, Any]:
        key = path.stem.removeprefix("yhat_")
        spec = mtc.Spec(key, key, "per-bar tree", "table", str(path))
        res = mtc.load_one(spec)
        gates.extend(g | {"forecast": key} for g in res["gates"])
        if res["error"]:
            gates.append(
                {
                    "gate": f"load ({res['error']})",
                    "forecast": key,
                    "ok": False,
                    "value": np.nan,
                }
            )
            return spec, None
        f = res["frame"]
        if len(days.difference(f.index[f["plain"].notna()].normalize())):
            gates.append(
                {
                    "gate": "covers every trade day",
                    "forecast": key,
                    "ok": False,
                    "value": np.nan,
                }
            )
            return spec, None
        return spec, f

    def daily(f: pd.DataFrame) -> dict:
        g = f.set_index(f.index.normalize())
        p = g["plain"].reindex(days).to_numpy(float)
        y = g["true_raw"].reindex(days).to_numpy(float)
        t = mtc.trade_days(g.loc[days, "plain"], dk)
        sz = gui_rows._score_sizes(p - iv, R, ex, bid, ask)
        return {
            "q": t["q"].to_numpy(),
            "mid": t["mid"].to_numpy(),
            "crossed": t["crossed"].to_numpy(),
            "es1": elementary(p / iv, y / iv, 1.0),
            "clip": sz["clip"],
            "sign614": sz["sign"],
        }

    spxw = (ROOT / "results" / "spxw_pnl").resolve()
    for b in BUCKETS:
        for rung, (cad, _tpl) in FAMILIES[a.family].items():
            cells: dict[str, tuple] = {}
            for name, loc in [(new_name, f"{a.out}:{a.token}@{a.family}"), *compare]:
                d, tok = loc.rsplit(":", 1)
                tok, fam = tok.split("@", 1) if "@" in tok else (tok, a.family)
                if not has_cadence(fam, cad):
                    continue
                p = ROOT / d / table_name(cad, b, tok, fam)
                if not p.is_file():
                    gates.append(
                        {
                            "gate": "table on disk",
                            "forecast": str(p.relative_to(ROOT)),
                            "ok": False,
                            "value": np.nan,
                        }
                    )
                    continue
                spec, f = load(p)
                if f is None:
                    continue
                lv = mtc.score_one((spec, f, ref, days, deck, False))
                dd = daily(f)
                cells[name] = (spec, f, dd)
                levels.append(
                    {
                        "bucket": b,
                        "rung": rung,
                        "forecast": name,
                        "key": spec.key,
                        "qlike": float(lv["qlike_raw"]),
                        "sharpe_mid": float(lv["Sharpe_mid"]),
                        "sharpe_crossed": float(lv["Sharpe_crossed"]),
                        "sortino_mid": gui_rows._sortino(dd["mid"]),
                        "sortino_crossed": gui_rows._sortino(dd["crossed"]),
                        "es1": float(dd["es1"].mean()),
                        "clip_sharpe_mid": dd["clip"]["sharpe_mid"],
                        "clip_sharpe_crossed": dd["clip"]["sharpe_crossed"],
                        "sign614_sharpe_mid": dd["sign614"]["sharpe_mid"],
                        "sized_days": dd["clip"]["days"],
                        "pct_buy": float(lv["pct_buy"]),
                    }
                )
                if (
                    tok in ("rf", "lgbm", "xgb")
                    and (ROOT / d).resolve() == spxw
                    and spec.key in master.index
                ):
                    dev = max(
                        abs(float(lv[c]) - float(master.loc[spec.key, c]))
                        for c in (
                            "qlike_raw",
                            "Sharpe_mid",
                            "Sharpe_crossed",
                            "pct_buy",
                        )
                    )
                    gates.append(
                        {
                            "gate": "reproduction of master_table.csv (max abs diff)",
                            "forecast": spec.key,
                            "ok": dev <= REPRO_TOL,
                            "value": dev,
                        }
                    )
            if new_name not in cells:
                continue
            spec, f, dn = cells[new_name]
            for name, _loc in compare:
                if name not in cells:
                    continue
                _s, fc, dc = cells[name]
                r = mtc.score_one((spec, f, fc, days, deck, False))
                tm, _ = asl.newey_west_t(dn["mid"] - dc["mid"])
                tx, _ = asl.newey_west_t(dn["crossed"] - dc["crossed"])
                te, _ = asl.newey_west_t(dn["es1"] - dc["es1"])
                pairs.append(
                    {
                        "bucket": b,
                        "rung": rung,
                        "new": new_name,
                        "vs": name,
                        "qlike_pct": float(r["qlike_pct_vs_ref"]),
                        "dm": float(r["dm_vs_ref"]),
                        "dm_p": float(r["dm_p_vs_ref"]),
                        "dsharpe_mid": float(r["dSharpe_mid_vs_ref"]),
                        "dsharpe_crossed": float(r["dSharpe_crossed_vs_ref"]),
                        "hac_t_mid": float(tm),
                        "hac_t_crossed": float(tx),
                        "des1": float((dn["es1"] - dc["es1"]).mean()),
                        "hac_t_es1": float(te),
                        "days_other_position": int((dn["q"] != dc["q"]).sum()),
                    }
                )
    lv_df, pr_df, gt = pd.DataFrame(levels), pd.DataFrame(pairs), pd.DataFrame(gates)
    lv_df.to_csv(sdir / "levels.csv", index=False)
    pr_df.to_csv(sdir / "pairs.csv", index=False)
    gt.to_csv(sdir / "gates.csv", index=False)
    n_bad = int((~gt["ok"].astype(bool)).sum()) if len(gt) else 0
    write_summary(
        sdir,
        lv_df,
        pr_df,
        gt,
        new_name,
        len(days),
        str(days.min().date()),
        str(days.max().date()),
    )
    if len(lv_df):
        print(
            lv_df.pivot_table(
                index=["bucket", "rung"], columns="forecast", values="qlike", sort=False
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
    new: str,
    n: int,
    d0: str,
    d1: str,
) -> None:
    def f(x: Any, k: int) -> str:
        return "" if x is None or pd.isna(x) else f"{float(x):.{k}f}"

    lines = [
        f"# Per-bar 16:00 random forest `{new}` against the reference forests (PREVIEW)",
        "",
        "Written by `experiments/trees_rfprev_score_1600.py` from `levels.csv`, `pairs.csv`, `gates.csv` in this folder. Nothing here enters "
        "results/spxw_pnl/, the paper, the GUI or the master table.",
        "",
        f"The master table's plain scorer (f^2 B, no recalibration), {n} trade days {d0} .. {d1}. ES_1 = the threshold (elementary) score of the "
        "buy / sell decision at c = 1 on the IV-scaled forecast and target (lower is better); clip = the robust-clip sized trade on the 614 days "
        "after the 252-day warm-up. Pairs: new minus comparison (DM negative = new more accurate; HAC t positive = more P&L; dES_1 negative = "
        "better decisions).",
        "",
        "| set | refit | forecast | QLIKE | Sharpe mid / crossed | Sortino mid / crossed | ES_1 | clip Sharpe mid / crossed |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for _, r in lv.iterrows():
        lines.append(
            f"| {r['bucket']} | {r['rung']} | {r['forecast']} | {f(r['qlike'], 4)} | {f(r['sharpe_mid'], 2)} / {f(r['sharpe_crossed'], 2)} | "
            f"{f(r['sortino_mid'], 2)} / {f(r['sortino_crossed'], 2)} | {f(r['es1'], 5)} | {f(r['clip_sharpe_mid'], 2)} / {f(r['clip_sharpe_crossed'], 2)} |"
        )
    lines += [
        "",
        "| set | refit | vs | QLIKE % | DM (p) | HAC t mid / crossed | dES_1 (HAC t) | days other position |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for _, r in pr.iterrows():
        lines.append(
            f"| {r['bucket']} | {r['rung']} | {r['vs']} | {f(r['qlike_pct'], 2)} | {f(r['dm'], 2)} ({f(r['dm_p'], 3)}) | "
            f"{f(r['hac_t_mid'], 2)} / {f(r['hac_t_crossed'], 2)} | {f(r['des1'], 5)} ({f(r['hac_t_es1'], 2)}) | {int(r['days_other_position'])} |"
        )
    lines += [
        "",
        "## Gates",
        "",
        f"- {int(gt['ok'].astype(bool).sum()) if len(gt) else 0} of {len(gt)} pass.",
    ]
    (out / "SUMMARY.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8", newline="\n"
    )


if __name__ == "__main__":
    sys.exit(main())
