"""Shift (dating) test of the HEADLINE forecast, plain 16:00 forecast (no recalibration).

2026-10-06 (user decision): the paper has NO headline model and no live-feasible forecasts;
D.16 is re-based on the paper's MODEL SET (results/close_master_table/paper_model_set.csv):
main() shifts every model-set forecast whose table carries the 10:00 .. 16:00 session rows
(the paper's own 48-bar forecasts; the pre-tuned pooled poolpt_* rows exist only at
16:00), places each shifted row on the
traded bar's scale as yhat_k^2 * B_16:00 (plain, no smear) and reports per k the median
Sharpe and the number of forecasts that fall below their own k = 0 Sharpe; the per-bar
tables (16:00 row only) are listed as not shifted.  Outputs: shift_test_modelset.csv,
shift_test_by_k.csv, shift_test_not_shifted.csv, shift_test_gates.csv,
shift_test_modelset.png, SUMMARY.md, writeup/generated/appendix_close_shift_test.tex.
The headline run is kept as main_headline_parked() (not called); its description follows.

2026-10-06: recalibration removed, user decision.  Every forecast is its PLAIN
back-transform f^2 * B (f the adjusted-scale forecast yhat, B the diurnal baseline of
the traded 16:00 bar), never the research scorer's (f^2 + s) * B; the trade is
q = sign(f^2 B - iv_var).  A shifted row is placed on the traded bar's scale as
RV_hat(k) = yhat_k^2 * B_16:00 (at k = 0 the plain headline forecast itself).  The
smear-recovery gate is parked; the committed-output gate is a reported-only
comparison (the numbers change by design: before_after_committed.csv, SUMMARY.md).
No intervals (the repo bootstrap is disabled, commit b761b28): the paired
differences are point estimates; their _lo / _hi columns are dropped.
The text below is the pre-2026-10-06 description, kept for the record.

The 15:30 sign(s) trade reads the per-bar ridge's row labelled 16:00 (naive ET,
bar-END labelled: the forecast of the 15:30-16:00 bar, issued at 15:30).  This
test re-scores the trade with the forecast ROW shifted by k half-hour bars and
everything else unchanged -- the notebook's section 11 ("the look-ahead cliff",
notebooks/_write_0dte_nb.py), which was run on the paper's eight forecasts under
the deck's session-bar recalibration, re-run here for the headline under the
16:00-bar recalibration of the master table (experiments/master_table_close.py).

  k < 0   the same session's earlier row (15:30, 15:00, ..., 10:00 for k = -1 ..
          -12): a stale forecast, still free of look-ahead.
  k = 0   the trade (must reproduce the master table's row exactly -- gated).
  k > 0   the headline is a per-bar model with session rows only (10:00-16:00),
          so "a row issued after the close" is the NEXT session's row: 10:00,
          10:30, ..., 12:30 for k = +1 .. +6.  Its lag ladder starts at the
          traded bar, so it has seen the traded bar's realized variance.
  star    the traded bar's realized variance itself in place of the forecast
          (2026-10-15, data audit #10, main(): the ACTUAL rv_raw, "realized (unclipped)"; the winsorized
          target true_raw, "realized (target)", is still reported beside it).

Placing: read raw, another row forecasts ANOTHER bar, whose diurnal level differs,
so every shifted row is placed on the traded bar's own scale exactly as the
notebook does:  RV_hat(k) = (yhat_k^2 + s_16:00) * B_16:00, with s_16:00 the
research scorer's causal smear of the 16:00 clock (the same s the trade uses) and
B_16:00 the traded bar's profile.  At k = 0 this is pred_clock itself.  The only
thing that changes with k is what the model had seen.

Scorer, trade, days, intervals: the master table's (its functions imported --
read_table_1600, load_target, trade_days, paired_sharpe, sharpe; the smear from
score_linear_subsection_causal.causal_forecasts): the 866 deck days, mid and
crossed fills, circular day-block bootstrap (block 21, 2000 draws) paired against
k = 0.  Every shift is scored on the deck days for which EVERY shifted row exists
(the last deck day has no next session; the 10:00-11:00 rows are missing on two
sessions); k = 0 is also scored on all 866 days as the gate.

Committed-output gate (I4c, 2026-09-30): every row of this test is the headline's (or
always short), and the 16:00 campaign did not change the headline's forecast or its
positions, so shift_test_headline.csv must equal the committed copy
(git show PREV_REV:results/close_shift_test/shift_test_headline.csv; PREV_REV = HEAD
unless the environment variable CLOSE_PREV_REV names another revision) cell by cell, to
|new - old| <= 1e-9 max(1, |old|).  A failure stops the script after the CSV is written
and before the figure, the tex and SUMMARY.md are.

Outputs (results/close_shift_test/): shift_test_headline.csv, shift_test_gates.csv,
shift_test_headline.png, SUMMARY.md (written from the CSV); the appendix table and
number macros writeup/generated/appendix_close_shift_test.tex.
"""

from __future__ import annotations

import io
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "notebooks", ROOT / "experiments"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import master_table_close as mtc  # noqa: E402
import score_linear_subsection as base  # noqa: E402

# 2026-10-06: recalibration removed, user decision (the causal smear is no longer used)
# import score_linear_subsection_causal as slc

OUT = ROOT / "results" / "close_shift_test"
TEX = ROOT / "writeup" / "generated" / "appendix_close_shift_test.tex"
# 2026-10-15 (number macros): macro-only file for the rehauled paper (paper_rehaul.tex loads it in its preamble; TEX
# above mixes tables and macros, so it is not loaded there). Written by main() and by --numbers-only (from the
# stored shift_test_by_k.csv, no re-scoring).
NUMBERS_TEX = ROOT / "writeup" / "generated" / "close_shift_test_numbers.tex"
# 07_economics.tex typed (2026-10-15) "SR swings by about +-1 across j, so differences of this size are sampling
# noise": write_numbers() stops unless the swing's half-range and the noise band both round to this value
NUMBERS_TYPED_SWING = 1
HEADLINE = mtc.HEADLINE  # sub_ridge_live_feasible
TABLE = ROOT / "results" / "spxw_pnl" / "yhat_sub_ridge_live_feasible.parquet"
MASTER = mtc.OUT / "master_table.csv"
GATE_REL = 1e-9
SESSION_BARS = [f"{h:02d}:{m:02d}" for h in range(10, 17) for m in (0, 30)][:13]
SAME_SHIFTS = list(range(-12, 1))  # -12 = the 10:00 row ... 0 = the 16:00 row
NEXT_SHIFTS = list(range(1, 7))  # +1 .. +6 = the next session's 10:00 .. 12:30 rows
ALL_SHIFTS = SAME_SHIFTS + NEXT_SHIFTS
# the committed outputs the rows are gated against (the headline is unchanged by the campaign)
PREV_REV = os.environ.get("CLOSE_PREV_REV", "HEAD")
# 2026-10-06: recalibration removed, user decision -- the committed copy is the
# recalibrated run, so it is compared for the record only (no tolerance asserted)
# COMMITTED_TOL = 1e-9  # |new - old| <= COMMITTED_TOL * max(1, |old|), cell by cell
PLAIN_HEADLINE_SHARPE_MID = "1.79"  # the GUI's independent recomputation, 866 days
# 2026-10-06: no intervals (bootstrap disabled, commit b761b28) -- the interval macros the
# appendix text still names are written as visible ?? stubs so the paper compiles
PARKED_MACROS = (
    "shiftMinusOneLo",
    "shiftMinusOneHi",
    "shiftBestStaleLo",
    "shiftBestStaleHi",
    "shiftNStaleIntervalAbove",
    "shiftNStaleIntervalBelow",
    "shiftPlusOneLo",
    "shiftPlusOneHi",
    "shiftNNextIntervalAbove",
)


def committed_csv(path: Path, **kw) -> tuple[pd.DataFrame, str]:
    """The committed copy of an output CSV (git show PREV_REV:<path>) and the revision's short sha."""
    rel = path.relative_to(ROOT).as_posix()
    sha = subprocess.check_output(
        ["git", "rev-parse", "--short", PREV_REV], cwd=ROOT, text=True
    ).strip()
    raw = subprocess.check_output(["git", "show", f"{PREV_REV}:{rel}"], cwd=ROOT)
    return pd.read_csv(io.BytesIO(raw), **kw), sha


def committed_diff(new: pd.DataFrame, old: pd.DataFrame) -> float:
    """Largest |new - old| / max(1, |old|) over the committed frame's cells (NaN = NaN);
    inf when a committed row or column is missing or a text cell differs."""
    if not (old.index.isin(new.index).all() and old.columns.isin(new.columns).all()):
        return float("inf")
    n = new.loc[old.index, old.columns]
    worst = 0.0
    for c in old.columns:
        a, b = n[c], old[c]
        if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
            a_, b_ = a.to_numpy(float), b.to_numpy(float)
            same_nan = np.isnan(a_) & np.isnan(b_)
            d = np.where(same_nan, 0.0, np.abs(a_ - b_) / np.maximum(1.0, np.abs(b_)))
            if np.isnan(d).any():
                return float("inf")
            worst = max(worst, float(d.max(initial=0.0)))
        elif not (a.astype(str) == b.astype(str)).all():
            return float("inf")
    return worst


def d_sharpe(a: np.ndarray, b: np.ndarray) -> float:
    """Sharpe of a minus Sharpe of b on the same days: the point value of the master
    table's paired_sharpe (2026-10-06: its bootstrap interval is not used)."""
    return mtc.sharpe(a) - mtc.sharpe(b)


def bar_of(k: int) -> tuple[str, str]:
    """(session, clock) of shift k: same session for k <= 0, the next session for k > 0."""
    if k <= 0:
        return "same", SESSION_BARS[12 + k]
    return "next", SESSION_BARS[k - 1]


def load_panel() -> pd.DataFrame:
    d = pd.read_parquet(TABLE)
    et = pd.to_datetime(d["t"]).dt.tz_convert("America/New_York").dt.tz_localize(None)
    p = pd.DataFrame(
        {
            "pred_adj": d["yhat"].to_numpy(float),
            "baseline": d["baseline"].to_numpy(float),
        },
        index=pd.DatetimeIndex(et.to_numpy()).as_unit("ns"),
    )
    p = p[np.isfinite(p["pred_adj"])].sort_index()
    p["day"] = p.index.normalize()
    p["hhmm"] = p.index.strftime("%H:%M")
    assert not p.index.duplicated().any()
    return p


# --------------------------------------------------------------------------- the paper's model set
# 2026-10-06 (user decision): the paper has NO headline model and no live-feasible forecasts;
# this diagnostic is re-based on the paper's MODEL SET.  The exact list is the macros agent's
# results/close_master_table/paper_model_set.csv (column key).  Without that file the script
# stops, unless MODELSET_PROVISIONAL=1 (development only): then the set is read off the
# master table's families by the rule below.
MODEL_SET_CSV = mtc.OUT / "paper_model_set.csv"
SET_BUCKETS = ("baseline", "all_features", "lean_vol_vix")
# checked before the shorter names that they contain
BUCKET_NAMES = (
    "lean_vol_vix",
    "moments_vol_vix",
    "moments_vix",
    "all_features",
    "baseline",
    "live_feasible",
)
GROUPS = ("paper", "pooled linear", "per-bar linear", "per-bar trees")
GROUP_COLOURS = {
    "paper": "#1b1b1b",
    "pooled linear": "#2a78d6",
    "per-bar linear": "#1baf7a",
    "per-bar trees": "#eb6834",
}
# families the user excluded from the model set (asserted absent)
EXCLUDED_FAMILY_WORDS = ("LSTM", "VIX-only", "implied-vol", "chain-period", "check")
EXCLUDED_BUCKETS = ("live_feasible", "moments_vix", "moments_vol_vix")


def bucket_of(key: str) -> str:
    """The input set named in a master-table key ('' for the paper's own forecasts)."""
    return next((b for b in BUCKET_NAMES if b in key), "")


def group_of(family: str) -> str:
    """The model-set group of a master-table family."""
    if family == "paper":
        return "paper"
    # 2026-10-06: the pre-tuned pooled rows (poolpt_*) may carry their own family name
    if family == "pooled twin" or family.startswith("pooled"):
        return "pooled linear"
    if family == "per-bar linear":
        return "per-bar linear"
    if family.startswith("per-bar tree"):
        return "per-bar trees"
    return "other"


def model_set(mt: pd.DataFrame) -> tuple[list[str], str]:
    """The model set's keys, in master-table order, and where they came from."""
    if MODEL_SET_CSV.exists():
        keys = pd.read_csv(MODEL_SET_CSV)["key"].astype(str).tolist()
        src = MODEL_SET_CSV.relative_to(ROOT).as_posix()
    elif os.environ.get("MODELSET_PROVISIONAL") == "1":
        a = mt[mt["table"] == "A"]
        grp = a["family"].map(group_of)
        bk = pd.Series([bucket_of(k) for k in a.index], index=a.index)
        keep = (grp == "paper") | (grp.isin(GROUPS) & bk.isin(SET_BUCKETS))
        keys = a.index[keep].tolist()
        if any(
            k.startswith("subpt_") for k in keys
        ):  # the pre-tuned per-bar linear tables
            keys = [k for k in keys if not k.startswith("sub_")]
        src = "PROVISIONAL rule on the master table's families (MODELSET_PROVISIONAL=1)"
    else:
        raise SystemExit(
            f"{MODEL_SET_CSV} is missing (MODELSET_PROVISIONAL=1 for a dry run)"
        )
    assert len(keys) == len(set(keys)), "duplicated keys in the model set"
    missing = [k for k in keys if k not in mt.index]
    assert not missing, f"model-set keys not in the master table: {missing}"
    fam = mt.loc[keys, "family"].astype(str)
    bad = [
        k
        for k in keys
        if any(w in fam[k] for w in EXCLUDED_FAMILY_WORDS)
        or bucket_of(k) in EXCLUDED_BUCKETS
        or group_of(fam[k]) == "other"
    ]
    assert not bad, f"excluded forecasts in the model set: {bad}"
    order = {k: i for i, k in enumerate(mt.index)}
    return sorted(keys, key=order.__getitem__), src


# 2026-10-06: the headline macros, parked (no headline model; user decision).  main() writes
# each one that it does not redefine as a visible ?? stub so the paper still compiles.
HEADLINE_MACROS_PARKED = (
    "shiftNDays",
    "shiftBarZeroMid",
    "shiftBarZeroCrossed",
    "shiftBarZeroAllMid",
    "shiftBarZeroAllCrossed",
    "shiftMinusOneMid",
    "shiftMinusOneDelta",
    "shiftBestStaleRow",
    "shiftBestStaleMid",
    "shiftBestStaleDelta",
    "shiftNStaleAbove",
    "shiftNStaleBelow",
    "shiftPlusOneMid",
    "shiftPlusOneCrossed",
    "shiftPlusOneDelta",
    "shiftNextMinMid",
    "shiftNextMaxMid",
    "shiftNNextAbove",
    "shiftStarMid",
    "shiftStarCrossed",
    "shiftAlwaysShortMid",
    "shiftMinusOneAgree",
    "shiftPlusOneAgree",
) + PARKED_MACROS


def main_headline_parked() -> None:
    """2026-10-06: PARKED (no headline model; user decision) -- the headline run, kept for
    the record and never called; main() below runs the paper's model set."""
    OUT.mkdir(parents=True, exist_ok=True)
    gates: list[dict] = []

    def gate(name: str, value: float, bound: float, n: int) -> None:
        gates.append(
            {"gate": name, "value": value, "bound": bound, "n": n, "ok": value <= bound}
        )

    # ---- the 16:00 frame, the master table's recipe (read_table_1600 + causal_forecasts)
    tgt = mtc.load_target()
    x = mtc.read_table_1600(TABLE, ["yhat", "baseline"]).rename(
        columns={"yhat": "pred_adj"}
    )
    x = x[np.isfinite(x["pred_adj"])]
    j = x.join(tgt, how="inner")
    gate(
        "baseline equals the target's B",
        float((j["baseline"] / j["B"] - 1).abs().max()),
        GATE_REL,
        len(j),
    )
    r = pd.DataFrame(
        {
            "true_adj": j["true_adj"],
            "pred_adj": j["pred_adj"],
            "true_raw": j["true_raw"],
            "baseline": j["B"],
        },
        index=j.index,
    )
    # 2026-10-06: recalibration removed, user decision -- the trade's forecast is the plain
    # back-transform f^2 * B; the research scorer's (f^2 + s) * B and the recovery of its
    # smear s16 are parked:
    # r["e2"] = (r["true_adj"] - r["pred_adj"]) ** 2
    # r["hhmm"] = r.index.strftime("%H:%M")
    # r["day"] = r.index.normalize()
    # cf = slc.causal_forecasts(r)
    # f16["s16"] = f16["pred_clock"] / f16["B"] - f16["pred_adj"] ** 2
    # s_direct = r["e2"].rolling(slc.SMEAR_W, min_periods=slc.SMEAR_MIN).mean().shift(1)
    # gate("recovered s16 equals the causal smear", ...)
    f16 = pd.DataFrame(
        {
            "pred_adj": r["pred_adj"],
            "B": r["baseline"],
            "plain": r["pred_adj"] ** 2 * r["baseline"],
            "true_raw": r["true_raw"],
            "rv_unclipped": j["rv_unclipped"],
        },
        index=r.index,
    )
    f16 = f16.set_index(f16.index.normalize())

    # ---- the deck and its days
    deck = pd.read_parquet(base.DECK).sort_index()
    deck.index = pd.DatetimeIndex(pd.to_datetime(deck.index)).as_unit("ns")
    days = deck.index
    assert len(days) == 866, len(days)

    # ---- the k = 0 gate against the master table (both on the plain forecast)
    t0 = mtc.trade_days(f16["plain"], deck)
    gate(
        f"k = 0 Sharpe mid (866 days) = {PLAIN_HEADLINE_SHARPE_MID} to two decimals "
        "(the GUI's recomputation)",
        float(f"{mtc.sharpe(t0['mid'].to_numpy()):.2f}" != PLAIN_HEADLINE_SHARPE_MID),
        0,
        len(t0),
    )
    mt = pd.read_csv(MASTER).set_index("key").loc[HEADLINE]
    gate(
        "k = 0 days = the master table's", abs(len(t0) - int(mt["n_days"])), 0, len(t0)
    )
    gate(
        "k = 0 Sharpe mid = the master table's",
        abs(mtc.sharpe(t0["mid"].to_numpy()) / float(mt["Sharpe_mid"]) - 1),
        GATE_REL,
        len(t0),
    )
    gate(
        "k = 0 Sharpe crossed = the master table's",
        abs(mtc.sharpe(t0["crossed"].to_numpy()) / float(mt["Sharpe_crossed"]) - 1),
        GATE_REL,
        len(t0),
    )

    # ---- the shifted rows, placed on the traded bar's scale
    pan = load_panel()
    sessions = pd.DatetimeIndex(sorted(pan["day"].unique()))
    nxt = pd.Series(
        sessions[1:], index=sessions[:-1]
    )  # session -> the next session in the panel
    by_bar = {
        b: pan[pan["hhmm"] == b].set_index("day")["pred_adj"] for b in SESSION_BARS
    }
    F: dict[str, pd.Series] = {}
    for k in ALL_SHIFTS:
        which, bar = bar_of(k)
        key_days = days if which == "same" else nxt.reindex(days)
        yk = by_bar[bar].reindex(pd.DatetimeIndex(key_days.to_numpy()))
        yk.index = days
        # 2026-10-06: recalibration removed, user decision (was (yk**2 + s16) * B)
        # F[f"bar{k:+d}"] = (yk**2 + f16["s16"].reindex(days)) * f16["B"].reindex(days)
        F[f"bar{k:+d}"] = yk**2 * f16["B"].reindex(days)
    gate(
        "bar+0 placed equals the plain forecast f^2 B",
        float((F["bar+0"] / f16["plain"].reindex(days) - 1).abs().max()),
        GATE_REL,
        len(days),
    )
    F["realized (target)"] = f16["true_raw"].reindex(days)
    F["realized (unclipped)"] = f16["rv_unclipped"].reindex(days)
    FF = pd.DataFrame(F)
    have = FF.notna().all(axis=1)
    common = days[have.to_numpy()]
    n_common = len(common)

    # ---- score every shift on the common days, paired against k = 0
    t0c = mtc.trade_days(F["bar+0"].loc[common], deck)
    rows: list[dict] = []
    q0 = t0c["q"].to_numpy()
    for name, series in F.items():
        t = mtc.trade_days(series.loc[common], deck)
        assert t.index.equals(t0c.index)
        # 2026-10-06: no intervals (bootstrap disabled, commit b761b28) -- the paired
        # difference is the point estimate sharpe(a) - sharpe(b) (paired_sharpe's first value)
        # d_mid = mtc.paired_sharpe(t["mid"].to_numpy(), t0c["mid"].to_numpy())
        # d_cr = mtc.paired_sharpe(t["crossed"].to_numpy(), t0c["crossed"].to_numpy())
        d_mid = (d_sharpe(t["mid"].to_numpy(), t0c["mid"].to_numpy()),)
        d_cr = (d_sharpe(t["crossed"].to_numpy(), t0c["crossed"].to_numpy()),)
        kk: float = int(name[3:]) if name.startswith("bar") else np.nan
        which, bar = (
            bar_of(int(kk)) if name.startswith("bar") else ("traded bar", "16:00")
        )
        rows.append(
            {
                "row": name,
                "k": kk,
                "session": which,
                "clock": bar,
                "n_days": len(t),
                "Sharpe_mid": mtc.sharpe(t["mid"].to_numpy()),
                "Sharpe_crossed": mtc.sharpe(t["crossed"].to_numpy()),
                "mean_mid": float(t["mid"].mean()),
                "pct_buy": float(100 * (t["q"] > 0).mean()),
                "agree_with_bar0_pct": float(100 * (t["q"].to_numpy() == q0).mean()),
                "dSharpe_mid_vs_bar0": d_mid[0],
                # 2026-10-06: no intervals (bootstrap disabled, commit b761b28)
                # "dSharpe_mid_lo": d_mid[1],
                # "dSharpe_mid_hi": d_mid[2],
                "dSharpe_crossed_vs_bar0": d_cr[0],
                # "dSharpe_crossed_lo": d_cr[1],
                # "dSharpe_crossed_hi": d_cr[2],
            }
        )
    # always short on the same days (q = -1: the crossed return is 1 - exit/bid)
    dk = deck.loc[common]
    r_mid = -dk["R"].to_numpy(float)
    r_cr = 1.0 - dk["exit"].to_numpy(float) / (dk["bid_c"] + dk["bid_p"]).to_numpy(
        float
    )
    # 2026-10-06: point estimates only (was mtc.paired_sharpe)
    d_mid = (d_sharpe(r_mid, t0c["mid"].to_numpy()),)
    d_cr = (d_sharpe(r_cr, t0c["crossed"].to_numpy()),)
    rows.append(
        {
            "row": "always short",
            "k": np.nan,
            "session": "",
            "clock": "",
            "n_days": n_common,
            "Sharpe_mid": mtc.sharpe(r_mid),
            "Sharpe_crossed": mtc.sharpe(r_cr),
            "mean_mid": float(r_mid.mean()),
            "pct_buy": 0.0,
            "agree_with_bar0_pct": float(100 * (q0 < 0).mean()),
            "dSharpe_mid_vs_bar0": d_mid[0],
            # 2026-10-06: no intervals (bootstrap disabled, commit b761b28)
            # "dSharpe_mid_lo": d_mid[1],
            # "dSharpe_mid_hi": d_mid[2],
            "dSharpe_crossed_vs_bar0": d_cr[0],
            # "dSharpe_crossed_lo": d_cr[1],
            # "dSharpe_crossed_hi": d_cr[2],
        }
    )
    tab = pd.DataFrame(rows).set_index("row")
    tab["bar0_all_days_Sharpe_mid"] = mtc.sharpe(t0["mid"].to_numpy())
    tab["bar0_all_days_Sharpe_crossed"] = mtc.sharpe(t0["crossed"].to_numpy())
    tab["bar0_all_days_n"] = len(t0)
    # the committed copy (the recalibrated run), read BEFORE this run's CSV overwrites the file
    old, sha = committed_csv(OUT / "shift_test_headline.csv", index_col=0)
    tab.to_csv(OUT / "shift_test_headline.csv")
    new = pd.read_csv(OUT / "shift_test_headline.csv", index_col=0)
    # 2026-10-06: recalibration removed, user decision -- the committed copy is the
    # recalibrated run, so "= the committed copy" is no longer asserted; it is compared
    # for the record (before_after_committed.csv, SUMMARY.md)
    # gate(
    #     f"shift_test_headline.csv = the committed copy ({PREV_REV} = {sha}), every cell, "
    #     "max |new - old| / max(1, |old|)",
    #     committed_diff(new, old),
    #     COMMITTED_TOL,
    #     int(old.size),
    # )
    cmp_cols = [
        "n_days",
        "Sharpe_mid",
        "Sharpe_crossed",
        "pct_buy",
        "agree_with_bar0_pct",
        "dSharpe_mid_vs_bar0",
        "dSharpe_crossed_vs_bar0",
    ]
    ba = pd.DataFrame(index=new.index)
    for c in cmp_cols:
        ba[f"{c}_before"] = old[c].reindex(new.index)
        ba[f"{c}_after"] = new[c]
    ba["max_rel_diff_vs_committed"] = [
        committed_diff(new.loc[[k], cmp_cols], old.loc[[k], cmp_cols])
        if k in old.index
        else np.nan
        for k in new.index
    ]
    ba.to_csv(OUT / "before_after_committed.csv")
    pd.DataFrame(gates).to_csv(OUT / "shift_test_gates.csv", index=False)
    assert all(g["ok"] for g in gates), [g for g in gates if not g["ok"]]

    # ---- figure
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    ks = ALL_SHIFTS
    for fill, mk in (("mid", "o"), ("crossed", "s")):
        ax.plot(
            ks,
            [tab.loc[f"bar{k:+d}", f"Sharpe_{fill}"] for k in ks],
            marker=mk,
            ms=3.5,
            lw=1.2,
            label=f"sign(s), {fill} fill",
        )
    x_star = ks[-1] + 2
    for fill, mk in (("mid", "*"), ("crossed", "*")):
        ax.plot(
            [x_star],
            [tab.loc["realized (target)", f"Sharpe_{fill}"]],
            marker=mk,
            ms=11,
            ls="none",
            color="k" if fill == "mid" else "0.5",
        )
    ax.axhline(tab.loc["always short", "Sharpe_mid"], color="0.4", lw=0.8, ls="--")
    ax.text(
        ks[0],
        tab.loc["always short", "Sharpe_mid"],
        " always short (mid)",
        fontsize=8,
        color="0.4",
        va="bottom",
    )
    ax.axvline(0, color="0.8", lw=0.8)
    ax.axvline(0.5, color="0.85", lw=0.8, ls=":")
    ax.text(
        0.6,
        ax.get_ylim()[1] * 0.98,
        "next session's rows",
        fontsize=8,
        color="0.4",
        va="top",
    )
    ax.set_xticks(ks + [x_star])
    ax.set_xticklabels([str(k) for k in ks] + ["realized\nvariance"], fontsize=8)
    ax.set_xlabel(
        "forecast row read, in half-hour bars from the traded row\n"
        "(k = 0: the trade; k < 0: the same session, stale; k > 0: the next session)"
    )
    ax.set_ylabel("annualized Sharpe")
    ax.set_title(
        # 2026-10-06: recalibration removed, user decision (was "16:00-bar recalibration")
        "Headline per-bar ridge (live-feasible), plain forecast (no recalibration):\n"
        "sign(s) Sharpe as the forecast row is shifted",
        fontsize=9,
    )
    ax.legend(fontsize=8, loc="upper left")
    fig.savefig(OUT / "shift_test_headline.png", dpi=130, bbox_inches="tight")

    # ---- tex: the table and the number macros
    def fmt(v: float) -> str:
        return f"{v:+.2f}" if abs(v) < 100 else f"{v:.0f}"

    # 2026-10-06: recalibration removed (user decision) and no intervals (bootstrap
    # disabled, commit b761b28): the caption says "plain forecast", the two Delta columns
    # carry the point difference only (was "$\Delta$ mid [95\,\%]" with "[lo, hi]")
    L = [
        "% AUTO-GENERATED by experiments/close_shift_test_headline.py from results/close_shift_test/shift_test_headline.csv -- do not edit.",
        "% 2026-10-06: plain 16:00 forecast f^2 B (recalibration removed, user decision); point estimates only (bootstrap disabled, b761b28).",
        r"\begingroup\scriptsize\setlength{\tabcolsep}{2pt}",
        r"\begin{longtable}{llrrrrrr}",
        r"\caption{The shift (dating) test of the headline forecast, read as its plain forecast $\hat y^2B$ (no recalibration): the 15:30 $\mathrm{sign}(s)$ rule "
        r"with the forecast row shifted by $k$ half-hour bars, every row placed on the traded bar's scale ($\hat y_k^2B_{16{:}00}$), on the "
        f"{n_common} deck days for which every shifted row exists. $k<0$: the same session's earlier row; $k=0$: the trade; "
        r"$k>0$: the next session's row (the headline has session rows only), whose lags contain the traded bar. "
        r"$\Delta$ is the Sharpe difference against $k=0$ (point estimate).}\label{tab:app_shift_test_headline}\\",
        r"\toprule",
        r"row & row read & Sharpe mid & Sharpe crossed & buy \% & agree \% & $\Delta$ mid & $\Delta$ crossed \\",
        r"\midrule",
        r"\endfirsthead",
        r"\caption[]{(continued)}\\",
        r"\toprule",
        r"row & row read & Sharpe mid & Sharpe crossed & buy \% & agree \% & $\Delta$ mid & $\Delta$ crossed \\",
        r"\midrule",
        r"\endhead",
        r"\bottomrule",
        r"\endlastfoot",
    ]
    for name, rr in tab.iterrows():
        lab = name.replace("bar", "$k=$") if str(name).startswith("bar") else name
        read = (
            f"{rr['clock']}, {rr['session']}"
            if rr["session"] in ("same", "next")
            else (f"{rr['clock']} ({rr['session']})" if rr["clock"] else "")
        )
        L.append(
            f"{lab} & {read} & {rr['Sharpe_mid']:.2f} & {rr['Sharpe_crossed']:.2f} & "
            f"{rr['pct_buy']:.0f} & {rr['agree_with_bar0_pct']:.0f} & "
            # 2026-10-06: no intervals -- was f"... [{fmt(rr['dSharpe_mid_lo'])}, {fmt(rr['dSharpe_mid_hi'])}]"
            f"{fmt(rr['dSharpe_mid_vs_bar0'])} & "
            f"{fmt(rr['dSharpe_crossed_vs_bar0'])} \\\\"
        )
    L += [r"\end{longtable}", r"\endgroup", ""]
    b0, b1, bm1, star = (
        tab.loc["bar+0"],
        tab.loc["bar+1"],
        tab.loc["bar-1"],
        tab.loc["realized (target)"],
    )
    stale = tab.loc[[f"bar{k:+d}" for k in SAME_SHIFTS[:-1]]]
    nxt_rows = tab.loc[[f"bar{k:+d}" for k in NEXT_SHIFTS]]
    best_stale = stale["Sharpe_mid"].idxmax()
    macros = {
        "shiftNDays": str(n_common),
        "shiftBarZeroMid": f"{b0['Sharpe_mid']:.2f}",
        "shiftBarZeroCrossed": f"{b0['Sharpe_crossed']:.2f}",
        "shiftBarZeroAllMid": f"{tab['bar0_all_days_Sharpe_mid'].iloc[0]:.2f}",
        "shiftBarZeroAllCrossed": f"{tab['bar0_all_days_Sharpe_crossed'].iloc[0]:.2f}",
        "shiftMinusOneMid": f"{bm1['Sharpe_mid']:.2f}",
        "shiftMinusOneDelta": f"{bm1['dSharpe_mid_vs_bar0']:+.2f}",
        # 2026-10-06: no intervals (bootstrap disabled, commit b761b28) -- the interval
        # macros are parked (see PARKED_MACROS below)
        # "shiftMinusOneLo": f"{bm1['dSharpe_mid_lo']:+.2f}",
        # "shiftMinusOneHi": f"{bm1['dSharpe_mid_hi']:+.2f}",
        "shiftBestStaleRow": best_stale.replace("bar", "").replace("+", "$+$"),
        "shiftBestStaleMid": f"{stale.loc[best_stale, 'Sharpe_mid']:.2f}",
        "shiftBestStaleDelta": f"{stale.loc[best_stale, 'dSharpe_mid_vs_bar0']:+.2f}",
        # "shiftBestStaleLo": f"{stale.loc[best_stale, 'dSharpe_mid_lo']:+.2f}",
        # "shiftBestStaleHi": f"{stale.loc[best_stale, 'dSharpe_mid_hi']:+.2f}",
        "shiftNStaleAbove": str(int((stale["dSharpe_mid_vs_bar0"] > 0).sum())),
        # "shiftNStaleIntervalAbove": str(int((stale["dSharpe_mid_lo"] > 0).sum())),
        # "shiftNStaleIntervalBelow": str(int((stale["dSharpe_mid_hi"] < 0).sum())),
        "shiftPlusOneMid": f"{b1['Sharpe_mid']:.2f}",
        "shiftPlusOneCrossed": f"{b1['Sharpe_crossed']:.2f}",
        "shiftPlusOneDelta": f"{b1['dSharpe_mid_vs_bar0']:+.2f}",
        # "shiftPlusOneLo": f"{b1['dSharpe_mid_lo']:+.2f}",
        # "shiftPlusOneHi": f"{b1['dSharpe_mid_hi']:+.2f}",
        "shiftNextMinMid": f"{nxt_rows['Sharpe_mid'].min():.2f}",
        "shiftNextMaxMid": f"{nxt_rows['Sharpe_mid'].max():.2f}",
        # "shiftNNextIntervalAbove": str(int((nxt_rows["dSharpe_mid_lo"] > 0).sum())),
        "shiftStarMid": f"{star['Sharpe_mid']:.2f}",
        "shiftStarCrossed": f"{star['Sharpe_crossed']:.2f}",
        "shiftAlwaysShortMid": f"{tab.loc['always short', 'Sharpe_mid']:.2f}",
        "shiftMinusOneAgree": f"{bm1['agree_with_bar0_pct']:.0f}",
        "shiftPlusOneAgree": f"{b1['agree_with_bar0_pct']:.0f}",
        # 2026-10-06: point-estimate counts that replace the interval counts
        "shiftNStaleBelow": str(int((stale["dSharpe_mid_vs_bar0"] < 0).sum())),
        "shiftNNextAbove": str(int((nxt_rows["dSharpe_mid_vs_bar0"] > 0).sum())),
    }
    L += [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in macros.items()]
    L += [
        "% 2026-10-06: interval macros removed (bootstrap disabled, commit b761b28; point "
        "estimates only). Each is a visible ?? stub until sections/appendix_close_shift_test.tex "
        "stops using it.",
    ]
    L += [f"\\newcommand{{\\{k}}}{{\\textbf{{??}}}}" for k in PARKED_MACROS]
    TEX.parent.mkdir(parents=True, exist_ok=True)
    TEX.write_text("\n".join(L) + "\n", encoding="utf-8", newline="\n")

    # ---- SUMMARY.md from the CSV
    # 2026-10-06: recalibration removed (user decision) and no intervals (bootstrap disabled,
    # commit b761b28): the text below says "plain forecast", every Δ is a point estimate, the
    # interval read-offs are parked and the committed (recalibrated) run is set beside this one
    S = [
        "# Shift (dating) test of the headline forecast, plain 16:00 forecast (no recalibration)",
        "",
        "Written by `experiments/close_shift_test_headline.py` from `shift_test_headline.csv`, `before_after_committed.csv` and `shift_test_gates.csv` "
        "in this folder. Headline: the per-bar ridge on the live-feasible inputs (`results/spxw_pnl/yhat_sub_ridge_live_feasible.parquet`), read as its "
        "plain back-transform f²·B (f the adjusted-scale forecast, B the diurnal baseline of the 16:00 bar; no recalibration, 2026-10-06 user decision), "
        "the deck's 15:30 sign(s) straddle trade q = sign(f²·B − implied variance) on `results/atm_straddle_0dte_1530/daily_blk2.parquet`, mid and crossed "
        "fills, the master table's trade and Sharpe functions (`experiments/master_table_close.py`). Point estimates only (the bootstrap is disabled, commit b761b28).",
        "",
        "## What is shifted",
        "- The forecast row read by the trade: k = 0 is the row labelled 16:00 (issued at 15:30); k < 0 reads the same session's earlier rows "
        "(k = −1 is the 15:30 row … k = −12 the 10:00 row); k > 0 reads the NEXT session's rows (k = +1 the 10:00 row … +6 the 12:30 row) — the headline is a "
        "per-bar model with session rows only, and the next session's first row is the first row whose lag ladder contains the traded bar.",
        "- Every shifted row is placed on the traded bar's own scale, ŷ_k²·B_16:00, with the profile B the trade uses; at k = 0 this is "
        "the trade's forecast itself (gated). The star is the traded bar's realized variance (the scorer's target) in place of the forecast.",
        f"- Days: the {n_common} deck days for which every shifted row exists (of 866; the last deck day has no next session, and the 10:00–11:00 rows are absent on "
        f"two sessions). k = 0 on all 866 days reproduces the master table: Sharpe {tab['bar0_all_days_Sharpe_mid'].iloc[0]:.4f} mid / "
        f"{tab['bar0_all_days_Sharpe_crossed'].iloc[0]:.4f} crossed (gated to 1e-9).",
        "",
        "## Results (annualized Sharpe; Δ = Sharpe minus k = 0's, point estimate)",
        "",
        "| row | session | clock | Sharpe mid | Sharpe crossed | buy % | agree with k=0 % | Δ mid | Δ crossed |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for name, rr in tab.iterrows():
        S.append(
            f"| {name} | {rr['session']} | {rr['clock']} | {rr['Sharpe_mid']:.2f} | {rr['Sharpe_crossed']:.2f} | {rr['pct_buy']:.0f} | "
            f"{rr['agree_with_bar0_pct']:.0f} | {rr['dSharpe_mid_vs_bar0']:+.2f} | {rr['dSharpe_crossed_vs_bar0']:+.2f} |"
        )
    S += [
        "",
        "## Read-offs (from the table above)",
        f"- One bar stale (k = −1, the 15:30 row): Sharpe {bm1['Sharpe_mid']:.2f} mid, Δ {bm1['dSharpe_mid_vs_bar0']:+.2f}; "
        f"its position agrees with the trade's on {bm1['agree_with_bar0_pct']:.0f} % of days.",
        f"- Stale rows (k = −12 … −1): {int((stale['dSharpe_mid_vs_bar0'] > 0).sum())} of 12 sit above k = 0 and "
        f"{int((stale['dSharpe_mid_vs_bar0'] < 0).sum())} below. "
        f"Best stale row: {best_stale} at {stale.loc[best_stale, 'Sharpe_mid']:.2f} (Δ {stale.loc[best_stale, 'dSharpe_mid_vs_bar0']:+.2f}).",
        f"- First row after the close (k = +1, the next session's 10:00 row): Sharpe {b1['Sharpe_mid']:.2f} mid / {b1['Sharpe_crossed']:.2f} crossed, "
        f"Δ {b1['dSharpe_mid_vs_bar0']:+.2f}; agreement with the trade {b1['agree_with_bar0_pct']:.0f} %. "
        f"Next-session rows k = +1 … +6: Sharpe mid {nxt_rows['Sharpe_mid'].min():.2f} to {nxt_rows['Sharpe_mid'].max():.2f}; "
        f"{int((nxt_rows['dSharpe_mid_vs_bar0'] > 0).sum())} of 6 above k = 0.",
        f"- Realized variance in place of the forecast: {star['Sharpe_mid']:.2f} mid / {star['Sharpe_crossed']:.2f} crossed "
        f"(unclipped realized: {tab.loc['realized (unclipped)', 'Sharpe_mid']:.2f} / {tab.loc['realized (unclipped)', 'Sharpe_crossed']:.2f}). "
        f"Always short on the same days: {tab.loc['always short', 'Sharpe_mid']:.2f} / {tab.loc['always short', 'Sharpe_crossed']:.2f}.",
        "",
        f"## Before / after: this run against the committed run ({PREV_REV} = {sha}, the 16:00-bar recalibration (f² + s)·B; reported, not asserted)",
        "",
        "| row | Sharpe mid before → after | Sharpe crossed before → after | buy % before → after | Δ mid before → after |",
        "|---|---|---|---|---|",
    ]
    for name, rr in ba.iterrows():
        S.append(
            f"| {name} | {rr['Sharpe_mid_before']:.2f} → {rr['Sharpe_mid_after']:.2f} | {rr['Sharpe_crossed_before']:.2f} → "
            f"{rr['Sharpe_crossed_after']:.2f} | {rr['pct_buy_before']:.0f} → {rr['pct_buy_after']:.0f} | "
            f"{rr['dSharpe_mid_vs_bar0_before']:+.2f} → {rr['dSharpe_mid_vs_bar0_after']:+.2f} |"
        )
    S += [
        "",
        f"- k = 0 on all 866 days: Sharpe mid {float(old['bar0_all_days_Sharpe_mid'].iloc[0]):.4f} → {tab['bar0_all_days_Sharpe_mid'].iloc[0]:.4f}, "
        f"crossed {float(old['bar0_all_days_Sharpe_crossed'].iloc[0]):.4f} → {tab['bar0_all_days_Sharpe_crossed'].iloc[0]:.4f}.",
        "",
        "## Context, not comparable",
        "- The notebook's own shift test (`results/atm_straddle_0dte_1530/forecast_shift_cliff.csv`, section 11) runs the paper's eight forecasts under the deck's "
        "session-bar recalibration and has after-hours rows for k > 0; its numbers are on the other scorer and are not set beside these.",
        "",
        "## Gates",
    ]
    S += [
        f"- {g['gate']}: {g['value']:.3g} ≤ {g['bound']:.3g} on n = {g['n']} — {'PASS' if g['ok'] else 'FAIL'}"
        for g in gates
    ]
    (OUT / "SUMMARY.md").write_text("\n".join(S) + "\n", encoding="utf-8")
    print(
        tab[
            [
                "session",
                "clock",
                "n_days",
                "Sharpe_mid",
                "Sharpe_crossed",
                "pct_buy",
                "agree_with_bar0_pct",
                "dSharpe_mid_vs_bar0",
                # 2026-10-06: no intervals
                # "dSharpe_mid_lo",
                # "dSharpe_mid_hi",
            ]
        ]
        .round(3)
        .to_string()
    )
    print("gates:", sum(g["ok"] for g in gates), "of", len(gates), "pass")
    print("wrote", OUT, "and", TEX)


def load_panel_path(path: Path) -> pd.DataFrame:
    """Every finite session-bar row of a forecast table (yhat, baseline), naive-ET index."""
    d = pd.read_parquet(path, columns=["t", "yhat", "baseline"])
    t = pd.to_datetime(d["t"])
    et = (
        t.dt.tz_convert("America/New_York").dt.tz_localize(None)
        if t.dt.tz is not None
        else t
    )
    p = pd.DataFrame(
        {
            "pred_adj": d["yhat"].to_numpy(float),
            "baseline": d["baseline"].to_numpy(float),
        },
        index=pd.DatetimeIndex(et.to_numpy()).as_unit("ns"),
    )
    p = p[np.isfinite(p["pred_adj"])].sort_index()
    p["hhmm"] = p.index.strftime("%H:%M")
    p = p[p["hhmm"].isin(SESSION_BARS)]
    p["day"] = p.index.normalize()
    assert not p.index.duplicated().any(), path
    return p


def write_numbers() -> None:
    """2026-10-15 (number macros): NUMBERS_TEX, macro definitions only, from the stored shift_test_by_k.csv
    (python experiments/close_shift_test_headline.py --numbers-only).  The statistics behind the text's "SR swings
    by about +-1 across j" (median over the shifted forecasts, Sharpe ratio at mid-quotes):

      \\shiftSwingMedLo, \\shiftSwingMedHi   smallest and largest median SR over every offset j, j = 0 included;
      \\shiftSwingHalf                       half that range (the swing, +-);
      \\shiftSwingDeltaLo, \\shiftSwingDeltaHi smallest and largest median change of SR against j = 0, j != 0;
      \\shiftNoiseBand                       1.96 sqrt(252 / n): the half-width of the 95 % interval of an annualized
                                            Sharpe ratio estimated on the n common days, iid returns, at SR = 0
                                            (Lo 2002), the sampling-noise scale the text compares the swing to."""
    tab = pd.read_csv(OUT / "shift_test_by_k.csv", index_col=0)
    sh = tab[tab["n_forecasts"] > 0]
    assert 0 in set(sh["k"].astype(int)), "no j = 0 row"
    n_days = sh["n_days"].astype(int).unique()
    assert len(n_days) == 1, n_days
    n = int(n_days[0])
    lo, hi = float(sh["Sharpe_mid_median"].min()), float(sh["Sharpe_mid_median"].max())
    off = sh[sh["k"].astype(int) != 0]["dSharpe_mid_median"]
    half = (hi - lo) / 2
    band = 1.96 * float(mtc.ANN) / np.sqrt(n)
    assert abs(float(mtc.ANN) - np.sqrt(252.0)) < 1e-12, (
        "Sharpe ratios are annualized with sqrt(252)"
    )
    assert round(half) == NUMBERS_TYPED_SWING and round(band) == NUMBERS_TYPED_SWING, (
        f"half-range {half:.2f}, noise band {band:.2f}: the text's 'about +-{NUMBERS_TYPED_SWING}' no longer holds; "
        "update writeup/rehaul/07_economics.tex and NUMBERS_TYPED_SWING together"
    )

    def sgn(v: float, plus: bool = False) -> str:
        s = f"{v:+.2f}" if plus else f"{v:.2f}"
        return s.replace("-", r"\ensuremath{-}")

    macros = {
        "shiftSwingMedLo": (sgn(lo), "smallest median SR (mid) over the offsets j"),
        "shiftSwingMedHi": (sgn(hi), "largest median SR (mid) over the offsets j"),
        "shiftSwingHalf": (sgn(half), "half the range: the swing, +-"),
        "shiftSwingDeltaLo": (
            sgn(float(off.min()), True),
            "smallest median change of SR against j = 0",
        ),
        "shiftSwingDeltaHi": (
            sgn(float(off.max()), True),
            "largest median change of SR against j = 0",
        ),
        "shiftNoiseBand": (
            sgn(band),
            f"1.96 sqrt(252/{n}): 95 % half-width of an annualized SR, iid, SR = 0",
        ),
    }
    L = [
        "% AUTO-GENERATED by experiments/close_shift_test_headline.py (write_numbers) from",
        "% results/close_shift_test/shift_test_by_k.csv -- do not edit. Macro definitions only (no tables); loaded once",
        "% by the preamble of writeup/paper_rehaul.tex. Shift test: the sign strategy with each forecast read j half",
        f"% hours from the closing interval, {int(sh['n_forecasts'].max())} forecasts, {n} common trading days, "
        "medians across forecasts, SR at mid-quotes.",
    ]
    L += [rf"\newcommand{{\{k}}}{{{v}}}  % {w}" for k, (v, w) in macros.items()]
    NUMBERS_TEX.parent.mkdir(parents=True, exist_ok=True)
    NUMBERS_TEX.write_text("\n".join(L) + "\n", encoding="utf-8", newline="\n")
    print("wrote", NUMBERS_TEX)


def main() -> None:
    """The shift test on every forecast of the paper's model set that has session rows.

    2026-10-06 (user decision): no headline; the shift test of the model set.  A forecast is
    shifted when its table carries the 10:00 .. 16:00 session rows on the deck days (the
    paper's own 48-bar forecasts); a per-bar table (one regression for
    the 16:00 bar: the per-bar linear and tree forecasts) has only the 16:00 row, so there
    is no other row to read and it is listed as not shifted -- nothing new is fitted.  Each
    shifted row is placed on the traded bar's scale as yhat_k^2 * B_16:00 (no smear, no
    recalibration).  Every forecast is scored on the same days (the deck days on which
    every shifted forecast has every shifted row); per k the table reports the median
    Sharpe across the shifted forecasts and how many fall below their own k = 0 Sharpe.
    """
    OUT.mkdir(parents=True, exist_ok=True)
    gates: list[dict] = []

    def gate(name: str, value: float, bound: float, n: int) -> None:
        gates.append(
            {
                "gate": name,
                "value": float(value),
                "bound": bound,
                "n": n,
                "ok": bool(value <= bound),
            }
        )

    mt = pd.read_csv(MASTER).set_index("key")
    keys, set_src = model_set(mt)
    specs = {s.key: s for s in mtc.discover()[0]}
    D = pd.read_parquet(
        mtc.OUT / "master_table_daily.parquet", columns=["key", "day", "plain", "q"]
    )
    D["day"] = pd.to_datetime(D["day"])
    deck = pd.read_parquet(base.DECK).sort_index()
    deck.index = pd.DatetimeIndex(pd.to_datetime(deck.index)).as_unit("ns")
    days = deck.index
    assert len(days) == 866, len(days)
    tgt = mtc.load_target()
    B16 = tgt["B"].copy()
    B16.index = B16.index.normalize()
    B16 = B16.reindex(days)
    assert B16.notna().all()

    # ---- which forecasts can be shifted: their tables must carry every session row
    panels: dict[str, pd.DataFrame] = {}
    not_shifted: list[dict] = []
    for k in keys:
        s = specs.get(k)
        fam = str(mt.loc[k, "family"])
        if s is None or s.source != "table":
            not_shifted.append(
                {
                    "key": k,
                    "label": mt.loc[k, "label"],
                    "group": group_of(fam),
                    "reason": "an arm results file of the 16:00 bar alone (per-bar regression): no other row",
                }
            )
            continue
        p = load_panel_path(Path(s.path))
        on = p[p["day"].isin(days)]
        have = set(on["hhmm"])
        if have == {"16:00"}:
            # 2026-10-06: the pre-tuned pooled forecasts are exported at the 16:00 bar only
            why = (
                "pre-tuned pooled forecast exported at the 16:00 bar only: no other row"
                if group_of(fam) == "pooled linear"
                else "per-bar table: one regression for the 16:00 bar, only the 16:00 row"
            )
            not_shifted.append(
                {
                    "key": k,
                    "label": mt.loc[k, "label"],
                    "group": group_of(fam),
                    "reason": why,
                }
            )
            continue
        assert have == set(SESSION_BARS), (k, sorted(set(SESSION_BARS) - have))
        panels[k] = p
    ns = pd.DataFrame(not_shifted)
    ns.to_csv(OUT / "shift_test_not_shifted.csv", index=False)
    shifted = list(panels)
    assert shifted, "no forecast of the model set has session rows"

    # ---- the shifted rows of every forecast, on the traded bar's scale yhat_k^2 * B_16:00
    F: dict[str, dict[str, pd.Series]] = {}
    for k, p in panels.items():
        sessions = pd.DatetimeIndex(sorted(p["day"].unique()))
        nxt = pd.Series(sessions[1:], index=sessions[:-1])
        by_bar = {b: p[p["hhmm"] == b].set_index("day") for b in SESSION_BARS}
        b16 = by_bar["16:00"]["baseline"].reindex(days)
        gate(
            f"16:00 baseline equals the target's B ({k})",
            float((b16 / B16 - 1).abs().max()),
            GATE_REL,
            len(days),
        )
        Fk: dict[str, pd.Series] = {}
        for sh in ALL_SHIFTS:
            which, bar = bar_of(sh)
            key_days = days if which == "same" else nxt.reindex(days)
            yk = by_bar[bar]["pred_adj"].reindex(pd.DatetimeIndex(key_days.to_numpy()))
            yk.index = days
            Fk[f"bar{sh:+d}"] = yk**2 * B16
        dk_plain = D[D["key"] == k].set_index("day")["plain"].reindex(days)
        gate(
            f"bar+0 placed equals the master table's plain forecast ({k})",
            float((Fk["bar+0"] / dk_plain - 1).abs().max()),
            GATE_REL,
            len(days),
        )
        F[k] = Fk
    have = np.ones(len(days), bool)
    for Fk in F.values():
        have &= pd.DataFrame(Fk).notna().all(axis=1).to_numpy()
    common = days[have]
    n_common = len(common)

    # ---- every forecast at every shift, on the common days
    rows: list[dict] = []
    for k in shifted:
        t0 = mtc.trade_days(F[k]["bar+0"], deck)  # all 866 days: the k = 0 gate
        gate(
            f"k = 0 Sharpe mid on 866 days = the master table's ({k})",
            abs(mtc.sharpe(t0["mid"].to_numpy()) / float(mt.loc[k, "Sharpe_mid"]) - 1),
            GATE_REL,
            len(t0),
        )
        t0c = mtc.trade_days(F[k]["bar+0"].loc[common], deck)
        q0 = t0c["q"].to_numpy()
        for name, series in F[k].items():
            t = mtc.trade_days(series.loc[common], deck)
            assert t.index.equals(t0c.index)
            sh = int(name[3:])
            which, bar = bar_of(sh)
            rows.append(
                {
                    "key": k,
                    "label": mt.loc[k, "label"],
                    "group": group_of(str(mt.loc[k, "family"])),
                    "k": sh,
                    "session": which,
                    "clock": bar,
                    "n_days": len(t),
                    "Sharpe_mid": mtc.sharpe(t["mid"].to_numpy()),
                    "Sharpe_crossed": mtc.sharpe(t["crossed"].to_numpy()),
                    "pct_buy": float(100 * (t["q"] > 0).mean()),
                    "agree_with_bar0_pct": float(
                        100 * (t["q"].to_numpy() == q0).mean()
                    ),
                    "dSharpe_mid_vs_bar0": d_sharpe(
                        t["mid"].to_numpy(), t0c["mid"].to_numpy()
                    ),
                    "dSharpe_crossed_vs_bar0": d_sharpe(
                        t["crossed"].to_numpy(), t0c["crossed"].to_numpy()
                    ),
                    "Sharpe_mid_all_866_at_k0": mtc.sharpe(t0["mid"].to_numpy()),
                }
            )
    long = pd.DataFrame(rows)
    long.to_csv(OUT / "shift_test_modelset.csv", index=False)

    # ---- per k across the shifted forecasts
    by_k: list[dict] = []
    for sh in ALL_SHIFTS:
        x = long[long["k"] == sh]
        which, bar = bar_of(sh)
        by_k.append(
            {
                "row": f"k={sh:+d}",
                "k": sh,
                "session": which,
                "clock": bar,
                "n_forecasts": len(x),
                "n_days": n_common,
                "Sharpe_mid_median": float(x["Sharpe_mid"].median()),
                "Sharpe_mid_min": float(x["Sharpe_mid"].min()),
                "Sharpe_mid_max": float(x["Sharpe_mid"].max()),
                "Sharpe_crossed_median": float(x["Sharpe_crossed"].median()),
                "n_mid_below_k0": int((x["dSharpe_mid_vs_bar0"] < 0).sum()),
                "n_crossed_below_k0": int((x["dSharpe_crossed_vs_bar0"] < 0).sum()),
                "dSharpe_mid_median": float(x["dSharpe_mid_vs_bar0"].median()),
                "agree_median_pct": float(x["agree_with_bar0_pct"].median()),
                "pct_buy_median": float(x["pct_buy"].median()),
            }
        )
    # the rows that read no forecast, on the same days
    dk = deck.loc[common]
    tg = tgt.copy()
    tg.index = tg.index.normalize()
    r_mid = -dk["R"].to_numpy(float)
    r_cr = 1.0 - dk["exit"].to_numpy(float) / (dk["bid_c"] + dk["bid_p"]).to_numpy(
        float
    )
    for name, series in (
        ("realized (target)", tg["true_raw"].reindex(days)),
        ("realized (unclipped)", tg["rv_unclipped"].reindex(days)),
    ):
        t = mtc.trade_days(series.loc[common], deck)
        by_k.append(
            {
                "row": name,
                "session": "traded bar",
                "clock": "16:00",
                "n_forecasts": 0,
                "n_days": len(t),
                "Sharpe_mid_median": mtc.sharpe(t["mid"].to_numpy()),
                "Sharpe_crossed_median": mtc.sharpe(t["crossed"].to_numpy()),
                "pct_buy_median": float(100 * (t["q"] > 0).mean()),
            }
        )
    by_k.append(
        {
            "row": "always short",
            "session": "",
            "clock": "",
            "n_forecasts": 0,
            "n_days": n_common,
            "Sharpe_mid_median": mtc.sharpe(r_mid),
            "Sharpe_crossed_median": mtc.sharpe(r_cr),
            "pct_buy_median": 0.0,
        }
    )
    tab = pd.DataFrame(by_k).set_index("row")
    tab.to_csv(OUT / "shift_test_by_k.csv")
    pd.DataFrame(gates).to_csv(OUT / "shift_test_gates.csv", index=False)
    assert all(g["ok"] for g in gates), [g for g in gates if not g["ok"]]

    # ---- figure: every shifted forecast (thin), the median (thick), realized variance (star)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    ks = ALL_SHIFTS
    x_star = ks[-1] + 2
    for ax, fill in zip(axes, ("mid", "crossed")):
        for k in shifted:
            x = long[long["key"] == k].set_index("k").loc[ks]
            ax.plot(
                ks,
                x[f"Sharpe_{fill}"],
                lw=0.6,
                alpha=0.45,
                color=GROUP_COLOURS[group_of(str(mt.loc[k, "family"]))],
            )
        med = long.groupby("k")[f"Sharpe_{fill}"].median().loc[ks]
        ax.plot(ks, med.to_numpy(), lw=2.2, color="k", marker="o", ms=3, label="median")
        # ax.plot([x_star], [tab.loc["realized (target)", f"Sharpe_{fill}_median"]],  # 2026-10-15 (user): parked -- scored against the trimmed target; scores now use actual RV (data audit #10)
        ax.plot(
            [x_star],
            [tab.loc["realized (unclipped)", f"Sharpe_{fill}_median"]],
            marker="*",
            ms=11,
            ls="none",
            color="k",
        )
        ax.axhline(
            tab.loc["always short", f"Sharpe_{fill}_median"],
            color="0.4",
            lw=0.8,
            ls="--",
        )
        ax.axvline(0, color="0.8", lw=0.8)
        ax.axvline(0.5, color="0.85", lw=0.8, ls=":")
        ax.set_xticks(ks + [x_star])
        ax.set_xticklabels([str(k) for k in ks] + ["RV"], fontsize=7)
        ax.set_xlabel(
            "forecast row read, in half-hour bars from the traded row\n"
            "(k < 0: the same session, stale; k > 0: the next session)",
            fontsize=8,
        )
        ax.set_title(
            f"{fill} fill: sign(s) Sharpe of the {len(shifted)} shifted forecasts "
            f"(plain f²B), {n_common} days",
            fontsize=9,
        )
    axes[0].set_ylabel("annualized Sharpe")
    handles = [
        plt.Line2D([], [], color=GROUP_COLOURS[g], lw=1.2, label=g)
        for g in GROUPS
        if any(group_of(str(mt.loc[k, "family"])) == g for k in shifted)
    ]
    handles += [
        plt.Line2D([], [], color="k", lw=2.2, label="median"),
        plt.Line2D([], [], color="0.4", lw=0.8, ls="--", label="always short"),
    ]
    axes[0].legend(handles=handles, fontsize=7.5, loc="lower left")
    fig.tight_layout()
    fig.savefig(OUT / "shift_test_modelset.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    # ---- tex: the per-k table and the number macros (no headline macros)
    def fmt(v: float) -> str:
        return f"{v:+.2f}"

    n_sh = len(shifted)
    L = [
        "% AUTO-GENERATED by experiments/close_shift_test_headline.py (main: the paper's model set) from",
        "% results/close_shift_test/shift_test_by_k.csv -- do not edit.",
        f"% 2026-10-06: the paper's model set ({set_src}); no headline; plain forecast f^2 B; point estimates only.",
        r"\begingroup\scriptsize\setlength{\tabcolsep}{3pt}",
        r"\begin{longtable}{llrrrrrr}",
        r"\caption{The shift (dating) test across the paper's model set: the 15:30 $\mathrm{sign}(s)$ rule "
        r"with each forecast's row shifted by $k$ half-hour bars and placed on the traded bar's scale, "
        rf"$\hat y_k^2B_{{16{{:}}00}}$ (the plain forecast, no recalibration), for the {n_sh} forecasts whose tables "
        rf"carry the session rows, on the {n_common} deck days on which every shifted row exists. $k<0$: the "
        r"same session's earlier row; $k=0$: the trade; $k>0$: the next session's row. Per $k$: the median "
        r"Sharpe ratio across the shifted forecasts (and its range), the number of forecasts whose Sharpe ratio "
        r"falls below its own $k=0$ value, the median change and the median share of days on which the "
        r"position agrees with $k=0$ (columns: median mid, range mid, median crossed, below $k=0$, median $\Delta$ mid, agree).}\label{tab:app_shift_test_headline}\\",
        r"\toprule",
        rf"row & read & mid & range mid & crossed & below $k{{=}}0$ (of {n_sh}) & $\Delta$ mid & agree \% \\",
        r"\midrule",
        r"\endfirsthead",
        r"\caption[]{(continued)}\\",
        r"\toprule",
        rf"row & read & mid & range mid & crossed & below $k{{=}}0$ (of {n_sh}) & $\Delta$ mid & agree \% \\",
        r"\midrule",
        r"\endhead",
        r"\bottomrule",
        r"\endlastfoot",
    ]
    for name, rr in tab.iterrows():
        if rr["n_forecasts"] > 0:
            lab = f"$k={int(rr['k']):+d}$"
            L.append(
                f"{lab} & {rr['clock']}, {rr['session']} & {rr['Sharpe_mid_median']:.2f} & "
                f"{rr['Sharpe_mid_min']:.2f} to {rr['Sharpe_mid_max']:.2f} & {rr['Sharpe_crossed_median']:.2f} & "
                f"{int(rr['n_mid_below_k0'])} & {fmt(rr['dSharpe_mid_median'])} & {rr['agree_median_pct']:.0f} \\\\"
            )
        else:
            L.append(
                f"{name} & {rr['clock']} & {rr['Sharpe_mid_median']:.2f} & & {rr['Sharpe_crossed_median']:.2f} & & & \\\\"
            )
    L += [r"\end{longtable}", r"\endgroup", ""]
    stale = tab[(tab["k"] < 0) & (tab["n_forecasts"] > 0)]
    nxt_rows = tab[(tab["k"] > 0) & (tab["n_forecasts"] > 0)]
    z, m1, p1 = tab.loc["k=+0"], tab.loc["k=-1"], tab.loc["k=+1"]
    best_stale = stale["Sharpe_mid_median"].idxmax()
    per_key_stale = long[long["k"] < 0].groupby("key")["dSharpe_mid_vs_bar0"]
    per_key_next = long[long["k"] > 0].groupby("key")["dSharpe_mid_vs_bar0"]
    zero_all = long[long["k"] == 0].set_index("key")["Sharpe_mid_all_866_at_k0"]
    ns_grp = ns["group"].value_counts() if len(ns) else pd.Series(dtype=int)
    sh_grp = pd.Series(
        [group_of(str(mt.loc[k, "family"])) for k in shifted]
    ).value_counts()
    macros = {
        "shiftNSet": str(len(keys)),
        "shiftNShifted": str(n_sh),
        "shiftNShiftedPaper": str(int(sh_grp.get("paper", 0))),
        "shiftNShiftedPooled": str(int(sh_grp.get("pooled linear", 0))),
        "shiftNNotShifted": str(len(ns)),
        "shiftNNotShiftedPerbarLinear": str(int(ns_grp.get("per-bar linear", 0))),
        "shiftNNotShiftedPooled": str(int(ns_grp.get("pooled linear", 0))),
        "shiftNNotShiftedTrees": str(int(ns_grp.get("per-bar trees", 0))),
        "shiftNDays": str(n_common),
        "shiftZeroMedMid": f"{z['Sharpe_mid_median']:.2f}",
        "shiftZeroMinMid": f"{z['Sharpe_mid_min']:.2f}",
        "shiftZeroMaxMid": f"{z['Sharpe_mid_max']:.2f}",
        "shiftZeroMedCrossed": f"{z['Sharpe_crossed_median']:.2f}",
        "shiftZeroAllMedMid": f"{zero_all.median():.2f}",
        "shiftMinusOneMedMid": f"{m1['Sharpe_mid_median']:.2f}",
        "shiftMinusOneNBelow": str(int(m1["n_mid_below_k0"])),
        "shiftMinusOneMedDelta": fmt(m1["dSharpe_mid_median"]),
        "shiftMinusOneMedAgree": f"{m1['agree_median_pct']:.0f}",
        "shiftStaleNBelowMin": str(int(stale["n_mid_below_k0"].min())),
        "shiftStaleNBelowMax": str(int(stale["n_mid_below_k0"].max())),
        "shiftStaleNAllBelow": str(int((per_key_stale.max() < 0).sum())),
        "shiftBestStaleK": f"{int(stale.loc[best_stale, 'k']):+d}".replace("+", "$+$"),
        "shiftBestStaleMedMid": f"{stale.loc[best_stale, 'Sharpe_mid_median']:.2f}",
        "shiftPlusOneMedMid": f"{p1['Sharpe_mid_median']:.2f}",
        "shiftPlusOneMedCrossed": f"{p1['Sharpe_crossed_median']:.2f}",
        "shiftPlusOneNBelow": str(int(p1["n_mid_below_k0"])),
        "shiftPlusOneMedAgree": f"{p1['agree_median_pct']:.0f}",
        "shiftNextMedMin": f"{nxt_rows['Sharpe_mid_median'].min():.2f}",
        "shiftNextMedMax": f"{nxt_rows['Sharpe_mid_median'].max():.2f}",
        "shiftNextNBelowMin": str(int(nxt_rows["n_mid_below_k0"].min())),
        "shiftNextNBelowMax": str(int(nxt_rows["n_mid_below_k0"].max())),
        "shiftNextNAllBelow": str(int((per_key_next.max() < 0).sum())),
        # "shiftStarMid": f"{tab.loc['realized (target)', 'Sharpe_mid_median']:.2f}",  # 2026-10-15 (user): parked -- scored against the trimmed target; scores now use actual RV (data audit #10)
        # "shiftStarCrossed": f"{tab.loc['realized (target)', 'Sharpe_crossed_median']:.2f}",
        "shiftStarMid": f"{tab.loc['realized (unclipped)', 'Sharpe_mid_median']:.2f}",
        "shiftStarCrossed": f"{tab.loc['realized (unclipped)', 'Sharpe_crossed_median']:.2f}",
        "shiftAlwaysShortMid": f"{tab.loc['always short', 'Sharpe_mid_median']:.2f}",
        "shiftAlwaysShortCrossed": f"{tab.loc['always short', 'Sharpe_crossed_median']:.2f}",
    }
    L += [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in macros.items()]
    stubs = [m for m in HEADLINE_MACROS_PARKED if m not in macros]
    L += [
        "% 2026-10-06: the headline macros are parked (no headline model; user decision). Each is a",
        "% visible ?? stub until sections/appendix_close_shift_test.tex stops using it.",
    ]
    L += [f"\\newcommand{{\\{m}}}{{\\textbf{{??}}}}" for m in stubs]
    TEX.parent.mkdir(parents=True, exist_ok=True)
    TEX.write_text("\n".join(L) + "\n", encoding="utf-8", newline="\n")

    # ---- SUMMARY.md from the CSVs
    S = [
        "# Shift (dating) test across the paper's model set (plain forecast f²·B, no recalibration)",
        "",
        "Written by `experiments/close_shift_test_headline.py` (`main`, 2026-10-06: no headline model, user decision) from "
        "`shift_test_modelset.csv` (every forecast × k), `shift_test_by_k.csv`, `shift_test_not_shifted.csv` and `shift_test_gates.csv`. "
        f"Model set: {len(keys)} forecasts ({set_src}). Each shifted row is placed on the traded bar's scale as ŷ_k²·B_16:00 (no smear, "
        "no recalibration); the deck's 15:30 sign(s) trade, mid and crossed fills; point estimates only.",
        "",
        f"- Shifted: {n_sh} forecasts whose tables carry the 10:00–16:00 session rows "
        f"({', '.join(f'{g} {int(n)}' for g, n in sh_grp.items())}).",
        f"- Not shifted: {len(ns)} forecasts "
        + (
            f"({', '.join(f'{g} {int(n)}' for g, n in ns_grp.items())})"
            if len(ns)
            else ""
        )
        + " — their tables hold only the 16:00 row (per-bar regressions of the 16:00 bar, and the pre-tuned pooled "
        "forecasts, exported at the 16:00 bar only), so there is no earlier or next-session row to read "
        "(`shift_test_not_shifted.csv`); nothing is fitted for this test.",
        f"- Days: the {n_common} deck days on which every shifted forecast has every shifted row. k = 0 on all 866 days "
        "reproduces the master table's Sharpe for every shifted forecast (gated to 1e-9).",
        "",
        "| row | read | median Sharpe mid | range mid | median crossed | below k=0 | median Δ mid | median agree % |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name, rr in tab.iterrows():
        if rr["n_forecasts"] > 0:
            S.append(
                f"| {name} | {rr['clock']}, {rr['session']} | {rr['Sharpe_mid_median']:.2f} | {rr['Sharpe_mid_min']:.2f} to "
                f"{rr['Sharpe_mid_max']:.2f} | {rr['Sharpe_crossed_median']:.2f} | {int(rr['n_mid_below_k0'])} of {n_sh} | "
                f"{rr['dSharpe_mid_median']:+.2f} | {rr['agree_median_pct']:.0f} |"
            )
        else:
            S.append(
                f"| {name} | {rr['clock']} | {rr['Sharpe_mid_median']:.2f} | | {rr['Sharpe_crossed_median']:.2f} | | | |"
            )
    S += ["", "## Gates"]
    S += [
        f"- {g['gate']}: {g['value']:.3g} ≤ {g['bound']:.3g} on n = {g['n']} — {'PASS' if g['ok'] else 'FAIL'}"
        for g in gates
    ]
    (OUT / "SUMMARY.md").write_text("\n".join(S) + "\n", encoding="utf-8", newline="\n")
    print(tab.round(3).to_string())
    print("not shifted:", len(ns), dict(ns_grp))
    print("gates:", sum(g["ok"] for g in gates), "of", len(gates), "pass")
    print("wrote", OUT, "and", TEX)
    write_numbers()  # 2026-10-15 (number macros): the macro-only file, from the CSV just written


if __name__ == "__main__":
    # 2026-10-15 (number macros): --numbers-only writes NUMBERS_TEX from the stored CSV (no re-scoring)
    if "--numbers-only" in sys.argv[1:]:
        write_numbers()
    else:
        main()
