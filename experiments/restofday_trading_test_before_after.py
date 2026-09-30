"""E2 trading test before / after the de-duplicated rest-of-day tables (checklist E2 re-run).

The direct rest-of-day trading test (experiments/restofday_trading_test.py, agent W8) first ran on
the rest-of-day and per-bar tables fitted WITH the twelve HAR x open / close session-edge columns.
Checklist I1 (per-bar arms, yhat_sub_*) and I6 (the direct model's 117 arms, yhat_restofday_*)
rebuilt both on the de-duplicated design; the return ledger (the straddle returns) is unchanged.

  BEFORE  results/linear_subsection_restofday/trading_test/ (committed, dacbeb0) and its re-run on
          the pre-dedup snapshot results/spxw_pnl/pre_dedup_2026-09-29/ (same code, --tex none) in
          <new>/pre_dedup_rerun/, which supplies the per-day series; GATE: the re-run's CSVs equal
          the committed ones (so the per-day series are the committed run's)
  AFTER   <new>/ = results/linear_subsection_restofday/trading_test_dedup/ (the tables now on disk)

Per estimator x bucket x variant x fill x scope (13 entry times + 3 pools), on the same days:
Sharpe of the direct and of the current (one-step) signal before / after, direct - current before /
after, and the CHANGE of each (after - before) with a paired interval: the same circular block
bootstrap as the test (block ceil(n^(1/3)) days, 2000 draws, seed 0, 95 % percentile), one set of
resampled days for the before and after series.  Also the number of days whose daily return moved.

Outputs (<new>/): before_after_pooled.csv, before_after_by_entry_time.csv, before_after_gates.txt,
BEFORE_AFTER.md (written from those CSVs).

Run:  python experiments/restofday_trading_test_before_after.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
import restofday_trading_test as rtt  # noqa: E402

BASE = ROOT / "results" / "linear_subsection_restofday"
OLD_COMMITTED = BASE / "trading_test"
NEW = BASE / "trading_test_dedup"
KEY = ["estimator", "bucket", "variant", "fill", "scope"]
REPRO_GATE = 1e-12  # the re-run must reproduce the committed CSVs to float precision
HEAD_EST, HEAD_VAR, HEAD_POOL = "ridge", "plain", "10:00-15:30"


def _sharpe_draws(x: np.ndarray, idx: np.ndarray) -> np.ndarray:
    d = x[idx]
    return d.mean(axis=1) / d.std(axis=1, ddof=1) * np.sqrt(rtt.asl.PERIODS_PER_YEAR)


def _ci(draws: np.ndarray) -> tuple[float, float]:
    lo, hi = np.percentile(draws, rtt.ALPHA_PCT)
    return float(lo), float(hi)


def _excl(lo: float, hi: float) -> bool:
    return lo > 0 or hi < 0


def repro_gate(rerun: Path, gates: list[str]) -> None:
    """The pre-dedup re-run reproduces the committed CSVs (numbers to REPRO_GATE, text exactly)."""
    for name in ("trading_test_pooled.csv", "trading_test_by_entry_time.csv"):
        a = pd.read_csv(OLD_COMMITTED / name)
        b = pd.read_csv(rerun / name)
        assert list(a.columns) == list(b.columns) and len(a) == len(b), name
        num = a.select_dtypes("number").columns
        worst = float(
            np.nanmax(np.abs(a[num].to_numpy(float) - b[num].to_numpy(float)))
        )
        txt = [c for c in a.columns if c not in num]
        same_txt = bool((a[txt].astype(str) == b[txt].astype(str)).all().all())
        nan_same = bool((a[num].isna() == b[num].isna()).all().all())
        ok = worst <= REPRO_GATE and same_txt and nan_same
        gates.append(
            f"GATE reproduction {name}: re-run on results/spxw_pnl/pre_dedup_2026-09-29 vs the "
            f"committed CSV, {len(a)} rows, max |diff| {worst:.1e}, text columns equal {same_txt} "
            f"-> {'OK' if ok else 'FAIL'}"
        )
        assert ok, gates[-1]


def consistency_gate(
    daily: pd.DataFrame, root: Path, tag: str, gates: list[str]
) -> None:
    """The per-day series give back each CSV's Sharpe (they are the series behind those rows)."""
    worst, n = 0.0, 0
    for name, scope_col in (
        ("trading_test_pooled.csv", "pool"),
        ("trading_test_by_entry_time.csv", "entry"),
    ):
        t = pd.read_csv(root / name).rename(columns={scope_col: "scope"})
        g = daily.groupby(KEY, sort=False)
        s = g.agg(
            Sharpe_direct=("direct", rtt.sharpe),
            Sharpe_current=("current", rtt.sharpe),
            n_days=("date", "size"),
        ).reset_index()
        m = t.merge(s, on=KEY, suffixes=("", "_d"), validate="1:1")
        assert len(m) == len(t), (tag, name)
        assert (m["n_days"] == m["n_days_d"]).all(), (tag, name)
        for c in ("Sharpe_direct", "Sharpe_current"):
            worst = max(worst, float(np.nanmax(np.abs(m[c] - m[f"{c}_d"]))))
        n += len(m)
    ok = worst <= REPRO_GATE
    gates.append(
        f"GATE per-day series ({tag}): the series give back all {n} rows' Sharpe (max |diff| "
        f"{worst:.1e}) and day counts -> {'OK' if ok else 'FAIL'}"
    )
    assert ok, gates[-1]


def compare(old: pd.DataFrame, new: pd.DataFrame, gates: list[str]) -> pd.DataFrame:
    rows = []
    og = dict(tuple(old.groupby(KEY, sort=False)))
    ng = dict(tuple(new.groupby(KEY, sort=False)))
    assert set(og) == set(ng), "the two runs have different rows"
    idx_cache: dict[int, np.ndarray] = {}
    n_days_differ = 0
    for k, o in og.items():
        n_ = ng[k]
        o = o.sort_values("date")
        n_ = n_.sort_values("date")
        same_days = bool(np.array_equal(o["date"].to_numpy(), n_["date"].to_numpy()))
        if not same_days:  # pair on the common days (counted in the gates)
            n_days_differ += 1
            common = np.intersect1d(o["date"].to_numpy(), n_["date"].to_numpy())
            o = o[o["date"].isin(common)]
            n_ = n_[n_["date"].isin(common)]
        do, co = o["direct"].to_numpy(float), o["current"].to_numpy(float)
        dn, cn = n_["direct"].to_numpy(float), n_["current"].to_numpy(float)
        n = len(do)
        if n not in idx_cache:
            idx_cache[n] = rtt.boot_idx(n)
        idx = idx_cache[n]
        sdo, sco, sdn, scn = (_sharpe_draws(x, idx) for x in (do, co, dn, cn))
        r = dict(zip(KEY, k))
        S = {
            nm: rtt.sharpe(x)
            for nm, x in (("do", do), ("co", co), ("dn", dn), ("cn", cn))
        }
        r.update(
            n_days=n,
            same_days=same_days,
            Sharpe_direct_before=S["do"],
            Sharpe_direct_after=S["dn"],
            Sharpe_current_before=S["co"],
            Sharpe_current_after=S["cn"],
            dSharpe_before=S["do"] - S["co"],
            dSharpe_after=S["dn"] - S["cn"],
        )
        for nm, hat, dr in (
            ("change_direct", S["dn"] - S["do"], sdn - sdo),
            ("change_current", S["cn"] - S["co"], scn - sco),
            (
                "change_dSharpe",
                (S["dn"] - S["cn"]) - (S["do"] - S["co"]),
                (sdn - scn) - (sdo - sco),
            ),
        ):
            lo, hi = _ci(dr)
            r[nm], r[f"{nm}_lo"], r[f"{nm}_hi"] = hat, lo, hi
        for nm, a, b in (("do", do, dn), ("co", co, cn)):
            side = "direct" if nm == "do" else "current"
            r[f"days_moved_{side}"] = int((a != b).sum())
        rows.append(r)
    gates.append(
        f"same days before / after: {len(og) - n_days_differ} of {len(og)} rows"
        + (f" ({n_days_differ} paired on their common days)" if n_days_differ else "")
    )
    return pd.DataFrame(rows)


def attach_intervals(
    t: pd.DataFrame, root_before: Path, root_after: Path
) -> pd.DataFrame:
    """direct - current intervals and readings of each run, from its own CSVs."""
    cols = ["dSharpe_pct_lo", "dSharpe_pct_hi", "reading"]
    for tag, root in (("before", root_before), ("after", root_after)):
        parts = []
        for name, sc in (
            ("trading_test_pooled.csv", "pool"),
            ("trading_test_by_entry_time.csv", "entry"),
        ):
            x = pd.read_csv(root / name).rename(columns={sc: "scope"})
            parts.append(x[KEY + cols])
        x = pd.concat(parts).rename(columns={c: f"{c}_{tag}" for c in cols})
        t = t.merge(x, on=KEY, how="left", validate="1:1")
    return t


def _f(v: float) -> str:
    return f"{v:+.2f}"


def write_md(
    pool: pd.DataFrame, clk: pd.DataFrame, gates: list[str], out: Path
) -> None:
    L = [
        "# E2 trading test: before / after the de-duplicated rest-of-day tables",
        "",
        "*Generated by `experiments/restofday_trading_test_before_after.py` from `before_after_pooled.csv` and "
        "`before_after_by_entry_time.csv`; every number below is read from them.*",
        "",
        "Before = the committed run (`results/linear_subsection_restofday/trading_test/`, tables fitted with "
        "the twelve session-edge columns; re-run on the pre-dedup snapshot for its per-day series). After = "
        "this folder (`yhat_restofday_*` rebuilt by I6, `yhat_sub_*` by I1). Same ledger, days, bars, fills "
        "and hedge; only the forecasts differ. Change = after - before, paired 95 % interval (circular block "
        "bootstrap, the test's own draws).",
        "",
        f"## Headline: per-bar {HEAD_EST}, {HEAD_VAR} back-transform, pooled {HEAD_POOL}",
        "",
        "| bucket | fill | direct - current before [interval] | after [interval] | change of direct - current "
        "[interval] | direct Sharpe before -> after | current Sharpe before -> after |",
        "|---|---|---|---|---|---|---|",
    ]
    h = pool[
        (pool["estimator"] == HEAD_EST)
        & (pool["variant"] == HEAD_VAR)
        & (pool["scope"] == HEAD_POOL)
    ]
    for b in rtt.BUCKETS:
        for fill in rtt.FILLS:
            r = h[(h["bucket"] == b) & (h["fill"] == fill)].iloc[0]
            L.append(
                f"| {rtt.BUCKET_LABEL[b]} | {fill} | {_f(r.dSharpe_before)} [{_f(r.dSharpe_pct_lo_before)}, "
                f"{_f(r.dSharpe_pct_hi_before)}] | {_f(r.dSharpe_after)} [{_f(r.dSharpe_pct_lo_after)}, "
                f"{_f(r.dSharpe_pct_hi_after)}] | {_f(r.change_dSharpe)} [{_f(r.change_dSharpe_lo)}, "
                f"{_f(r.change_dSharpe_hi)}] | {r.Sharpe_direct_before:.2f} -> {r.Sharpe_direct_after:.2f} | "
                f"{r.Sharpe_current_before:.2f} -> {r.Sharpe_current_after:.2f} |"
            )
    L += ["", "## Counts", ""]
    for nm, t in (("pooled (3 pools)", pool), ("by entry time (13)", clk)):
        n = len(t)
        ex_b = int(
            sum(
                _excl(a, b)
                for a, b in zip(t["dSharpe_pct_lo_before"], t["dSharpe_pct_hi_before"])
            )
        )
        ex_a = int(
            sum(
                _excl(a, b)
                for a, b in zip(t["dSharpe_pct_lo_after"], t["dSharpe_pct_hi_after"])
            )
        )
        ahead_b = int((t["dSharpe_before"] > 0).sum())
        ahead_a = int((t["dSharpe_after"] > 0).sum())
        chx = int(
            sum(
                _excl(a, b)
                for a, b in zip(t["change_dSharpe_lo"], t["change_dSharpe_hi"])
            )
        )
        chd = int(
            sum(
                _excl(a, b)
                for a, b in zip(t["change_direct_lo"], t["change_direct_hi"])
            )
        )
        chc = int(
            sum(
                _excl(a, b)
                for a, b in zip(t["change_current_lo"], t["change_current_hi"])
            )
        )
        L.append(
            f"- {nm}, {n} rows (estimator x bucket x variant x fill x scope): direct ahead of current "
            f"{ahead_b} -> {ahead_a}; direct - current interval excludes zero {ex_b} -> {ex_a}; the change's "
            f"interval excludes zero for direct - current {chx}, the direct Sharpe {chd}, the current Sharpe "
            f"{chc}; median |change| of direct - current {t['change_dSharpe'].abs().median():.3f} (max "
            f"{t['change_dSharpe'].abs().max():.3f}); rows with any day moved: direct "
            f"{int((t['days_moved_direct'] > 0).sum())}, current {int((t['days_moved_current'] > 0).sum())}."
        )
    L += [
        "",
        "## Gates",
        "",
        *[f"- {g}" for g in gates],
        "",
        "## Files",
        "",
        "- `before_after_pooled.csv`, `before_after_by_entry_time.csv`: per estimator x bucket x variant x fill x "
        "scope: Sharpe of direct / current before and after, direct - current before and after (with each run's "
        "interval and reading), the change of each with its paired interval, days whose daily return moved.",
        "- `before_after_gates.txt`; the before run's per-day series: `pre_dedup_rerun/` (not committed).",
    ]
    (out / "BEFORE_AFTER.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"wrote {out / 'BEFORE_AFTER.md'}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--new", default=str(NEW))
    ap.add_argument("--rerun", default=None, help="default <new>/pre_dedup_rerun")
    a = ap.parse_args()
    new = Path(a.new)
    new = new if new.is_absolute() else ROOT / new
    rerun = Path(a.rerun) if a.rerun else new / "pre_dedup_rerun"
    rerun = rerun if rerun.is_absolute() else ROOT / rerun
    gates: list[str] = []
    repro_gate(rerun, gates)
    old_d = pd.read_parquet(rerun / "trading_test_daily.parquet")
    new_d = pd.read_parquet(new / "trading_test_daily.parquet")
    consistency_gate(old_d, OLD_COMMITTED, "before", gates)
    consistency_gate(new_d, new, "after", gates)
    t = compare(old_d, new_d, gates)
    t = attach_intervals(t, OLD_COMMITTED, new)
    t["scope_kind"] = np.where(t["scope"].isin(list(rtt.POOLS)), "pool", "entry")
    order = {c: i for i, c in enumerate(list(rtt.POOLS) + rtt.ENTRY_CLOCKS)}
    t = t.sort_values(
        ["scope_kind", "estimator", "bucket", "variant", "fill", "scope"],
        key=lambda s: s.map(order) if s.name == "scope" else s,
    )
    pool = t[t["scope_kind"] == "pool"].drop(columns="scope_kind")
    clk = t[t["scope_kind"] == "entry"].drop(columns="scope_kind")
    pool.to_csv(new / "before_after_pooled.csv", index=False, float_format="%.6g")
    clk.to_csv(new / "before_after_by_entry_time.csv", index=False, float_format="%.6g")
    (new / "before_after_gates.txt").write_text(
        "\n".join(gates) + "\n", encoding="utf-8"
    )
    for g in gates:
        print(g)
    write_md(pool, clk, gates, new)


if __name__ == "__main__":
    main()
