"""Shift (dating) test of the HEADLINE forecast under the research scorer.

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
          (the scorer's target true_raw; the unclipped rv_raw is also reported).

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
import score_linear_subsection_causal as slc  # noqa: E402

OUT = ROOT / "results" / "close_shift_test"
TEX = ROOT / "writeup" / "generated" / "appendix_close_shift_test.tex"
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
COMMITTED_TOL = 1e-9  # |new - old| <= COMMITTED_TOL * max(1, |old|), cell by cell


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


def main() -> None:
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
    r["e2"] = (r["true_adj"] - r["pred_adj"]) ** 2
    r["hhmm"] = r.index.strftime("%H:%M")
    r["day"] = r.index.normalize()
    cf = slc.causal_forecasts(r)
    f16 = pd.DataFrame(
        {
            "pred_adj": cf["pred_adj"],
            "B": cf["baseline"],
            "pred_clock": cf["pred_clock"],
            "true_raw": cf["true_raw"],
            "rv_unclipped": j["rv_unclipped"],
        },
        index=cf.index,
    )
    # the smear of the 16:00 clock, recovered from the recalibrated forecast
    f16["s16"] = f16["pred_clock"] / f16["B"] - f16["pred_adj"] ** 2
    s_direct = r["e2"].rolling(slc.SMEAR_W, min_periods=slc.SMEAR_MIN).mean().shift(1)
    ok = f16["s16"].notna()
    gate(
        "recovered s16 equals the causal smear",
        float(
            (f16.loc[ok, "s16"] - s_direct[ok]).abs().max() / s_direct[ok].abs().max()
        ),
        GATE_REL,
        int(ok.sum()),
    )
    f16 = f16.set_index(f16.index.normalize())

    # ---- the deck and its days
    deck = pd.read_parquet(base.DECK).sort_index()
    deck.index = pd.DatetimeIndex(pd.to_datetime(deck.index)).as_unit("ns")
    days = deck.index
    assert len(days) == 866, len(days)

    # ---- the k = 0 gate against the master table
    t0 = mtc.trade_days(f16["pred_clock"], deck)
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
        F[f"bar{k:+d}"] = (yk**2 + f16["s16"].reindex(days)) * f16["B"].reindex(days)
    gate(
        "bar+0 placed equals pred_clock",
        float((F["bar+0"] / f16["pred_clock"].reindex(days) - 1).abs().max()),
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
        d_mid = mtc.paired_sharpe(t["mid"].to_numpy(), t0c["mid"].to_numpy())
        d_cr = mtc.paired_sharpe(t["crossed"].to_numpy(), t0c["crossed"].to_numpy())
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
                "dSharpe_mid_lo": d_mid[1],
                "dSharpe_mid_hi": d_mid[2],
                "dSharpe_crossed_vs_bar0": d_cr[0],
                "dSharpe_crossed_lo": d_cr[1],
                "dSharpe_crossed_hi": d_cr[2],
            }
        )
    # always short on the same days (q = -1: the crossed return is 1 - exit/bid)
    dk = deck.loc[common]
    r_mid = -dk["R"].to_numpy(float)
    r_cr = 1.0 - dk["exit"].to_numpy(float) / (dk["bid_c"] + dk["bid_p"]).to_numpy(
        float
    )
    d_mid = mtc.paired_sharpe(r_mid, t0c["mid"].to_numpy())
    d_cr = mtc.paired_sharpe(r_cr, t0c["crossed"].to_numpy())
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
            "dSharpe_mid_lo": d_mid[1],
            "dSharpe_mid_hi": d_mid[2],
            "dSharpe_crossed_vs_bar0": d_cr[0],
            "dSharpe_crossed_lo": d_cr[1],
            "dSharpe_crossed_hi": d_cr[2],
        }
    )
    tab = pd.DataFrame(rows).set_index("row")
    tab["bar0_all_days_Sharpe_mid"] = mtc.sharpe(t0["mid"].to_numpy())
    tab["bar0_all_days_Sharpe_crossed"] = mtc.sharpe(t0["crossed"].to_numpy())
    tab["bar0_all_days_n"] = len(t0)
    tab.to_csv(OUT / "shift_test_headline.csv")
    # the committed copy: every row is the headline's (or always short), unchanged by the campaign
    old, sha = committed_csv(OUT / "shift_test_headline.csv", index_col=0)
    new = pd.read_csv(OUT / "shift_test_headline.csv", index_col=0)
    gate(
        f"shift_test_headline.csv = the committed copy ({PREV_REV} = {sha}), every cell, "
        "max |new - old| / max(1, |old|)",
        committed_diff(new, old),
        COMMITTED_TOL,
        int(old.size),
    )
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
        "Headline per-bar ridge (live-feasible), 16:00-bar recalibration:\n"
        "sign(s) Sharpe as the forecast row is shifted",
        fontsize=9,
    )
    ax.legend(fontsize=8, loc="upper left")
    fig.savefig(OUT / "shift_test_headline.png", dpi=130, bbox_inches="tight")

    # ---- tex: the table and the number macros
    def fmt(v: float) -> str:
        return f"{v:+.2f}" if abs(v) < 100 else f"{v:.0f}"

    L = [
        "% AUTO-GENERATED by experiments/close_shift_test_headline.py from results/close_shift_test/shift_test_headline.csv -- do not edit.",
        # eight columns with two interval strings: \scriptsize at 1pt padding fits the text width
        r"\begingroup\scriptsize\setlength{\tabcolsep}{1pt}",
        r"\begin{longtable}{llrrrrll}",
        r"\caption{The shift (dating) test of the headline forecast under the 16:00-bar recalibration: the 15:30 $\mathrm{sign}(s)$ rule "
        r"with the forecast row shifted by $k$ half-hour bars, every row placed on the traded bar's scale, on the "
        f"{n_common} deck days for which every shifted row exists. $k<0$: the same session's earlier row; $k=0$: the trade; "
        r"$k>0$: the next session's row (the headline has session rows only), whose lags contain the traded bar. "
        r"$\Delta$ is against $k=0$ with the paired day-block 95\,\% interval.}\label{tab:app_shift_test_headline}\\",
        r"\toprule",
        r"row & row read & Sharpe mid & Sharpe crossed & buy \% & agree \% & $\Delta$ mid [95\,\%] & $\Delta$ crossed [95\,\%] \\",
        r"\midrule",
        r"\endfirsthead",
        r"\caption[]{(continued)}\\",
        r"\toprule",
        r"row & row read & Sharpe mid & Sharpe crossed & buy \% & agree \% & $\Delta$ mid [95\,\%] & $\Delta$ crossed [95\,\%] \\",
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
            f"{fmt(rr['dSharpe_mid_vs_bar0'])} [{fmt(rr['dSharpe_mid_lo'])}, {fmt(rr['dSharpe_mid_hi'])}] & "
            f"{fmt(rr['dSharpe_crossed_vs_bar0'])} [{fmt(rr['dSharpe_crossed_lo'])}, {fmt(rr['dSharpe_crossed_hi'])}] \\\\"
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
        "shiftMinusOneLo": f"{bm1['dSharpe_mid_lo']:+.2f}",
        "shiftMinusOneHi": f"{bm1['dSharpe_mid_hi']:+.2f}",
        "shiftBestStaleRow": best_stale.replace("bar", "").replace("+", "$+$"),
        "shiftBestStaleMid": f"{stale.loc[best_stale, 'Sharpe_mid']:.2f}",
        "shiftBestStaleDelta": f"{stale.loc[best_stale, 'dSharpe_mid_vs_bar0']:+.2f}",
        "shiftBestStaleLo": f"{stale.loc[best_stale, 'dSharpe_mid_lo']:+.2f}",
        "shiftBestStaleHi": f"{stale.loc[best_stale, 'dSharpe_mid_hi']:+.2f}",
        "shiftNStaleAbove": str(int((stale["dSharpe_mid_vs_bar0"] > 0).sum())),
        "shiftNStaleIntervalAbove": str(int((stale["dSharpe_mid_lo"] > 0).sum())),
        "shiftNStaleIntervalBelow": str(int((stale["dSharpe_mid_hi"] < 0).sum())),
        "shiftPlusOneMid": f"{b1['Sharpe_mid']:.2f}",
        "shiftPlusOneCrossed": f"{b1['Sharpe_crossed']:.2f}",
        "shiftPlusOneDelta": f"{b1['dSharpe_mid_vs_bar0']:+.2f}",
        "shiftPlusOneLo": f"{b1['dSharpe_mid_lo']:+.2f}",
        "shiftPlusOneHi": f"{b1['dSharpe_mid_hi']:+.2f}",
        "shiftNextMinMid": f"{nxt_rows['Sharpe_mid'].min():.2f}",
        "shiftNextMaxMid": f"{nxt_rows['Sharpe_mid'].max():.2f}",
        "shiftNNextIntervalAbove": str(int((nxt_rows["dSharpe_mid_lo"] > 0).sum())),
        "shiftStarMid": f"{star['Sharpe_mid']:.2f}",
        "shiftStarCrossed": f"{star['Sharpe_crossed']:.2f}",
        "shiftAlwaysShortMid": f"{tab.loc['always short', 'Sharpe_mid']:.2f}",
        "shiftMinusOneAgree": f"{bm1['agree_with_bar0_pct']:.0f}",
        "shiftPlusOneAgree": f"{b1['agree_with_bar0_pct']:.0f}",
    }
    L += [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in macros.items()]
    TEX.parent.mkdir(parents=True, exist_ok=True)
    TEX.write_text("\n".join(L) + "\n", encoding="utf-8")

    # ---- SUMMARY.md from the CSV
    S = [
        "# Shift (dating) test of the headline forecast under the 16:00-bar recalibration",
        "",
        "Written by `experiments/close_shift_test_headline.py` from `shift_test_headline.csv` and `shift_test_gates.csv` in this folder. "
        "Headline: the per-bar ridge on the live-feasible inputs (`results/spxw_pnl/yhat_sub_ridge_live_feasible.parquet`), scored exactly as the master table "
        "(`experiments/master_table_close.py`: pred_clock = (f² + s)·B with the causal 250-session smear of the 16:00 clock, the deck's 15:30 sign(s) straddle trade on "
        "`results/atm_straddle_0dte_1530/daily_blk2.parquet`, mid and crossed fills, paired circular day-block bootstrap, block 21, 2000 draws).",
        "",
        "## What is shifted",
        "- The forecast row read by the trade: k = 0 is the row labelled 16:00 (issued at 15:30); k < 0 reads the same session's earlier rows "
        "(k = −1 is the 15:30 row … k = −12 the 10:00 row); k > 0 reads the NEXT session's rows (k = +1 the 10:00 row … +6 the 12:30 row) — the headline is a "
        "per-bar model with session rows only, and the next session's first row is the first row whose lag ladder contains the traded bar.",
        "- Every shifted row is placed on the traded bar's own scale, (ŷ_k² + s_16:00)·B_16:00, with the same smear s and profile B the trade uses; at k = 0 this is "
        "the trade's forecast itself (gated). The star is the traded bar's realized variance (the scorer's target) in place of the forecast.",
        f"- Days: the {n_common} deck days for which every shifted row exists (of 866; the last deck day has no next session, and the 10:00–11:00 rows are absent on "
        f"two sessions). k = 0 on all 866 days reproduces the master table: Sharpe {tab['bar0_all_days_Sharpe_mid'].iloc[0]:.4f} mid / "
        f"{tab['bar0_all_days_Sharpe_crossed'].iloc[0]:.4f} crossed (gated to 1e-9).",
        "",
        "## Results (annualized Sharpe; Δ vs k = 0 with the paired 95 % interval)",
        "",
        "| row | session | clock | Sharpe mid | Sharpe crossed | buy % | agree with k=0 % | Δ mid [95 %] | Δ crossed [95 %] |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for name, rr in tab.iterrows():
        S.append(
            f"| {name} | {rr['session']} | {rr['clock']} | {rr['Sharpe_mid']:.2f} | {rr['Sharpe_crossed']:.2f} | {rr['pct_buy']:.0f} | "
            f"{rr['agree_with_bar0_pct']:.0f} | {rr['dSharpe_mid_vs_bar0']:+.2f} [{rr['dSharpe_mid_lo']:+.2f}, {rr['dSharpe_mid_hi']:+.2f}] | "
            f"{rr['dSharpe_crossed_vs_bar0']:+.2f} [{rr['dSharpe_crossed_lo']:+.2f}, {rr['dSharpe_crossed_hi']:+.2f}] |"
        )
    S += [
        "",
        "## Read-offs (from the table above)",
        f"- One bar stale (k = −1, the 15:30 row): Sharpe {bm1['Sharpe_mid']:.2f} mid, Δ {bm1['dSharpe_mid_vs_bar0']:+.2f} "
        f"[{bm1['dSharpe_mid_lo']:+.2f}, {bm1['dSharpe_mid_hi']:+.2f}]; its position agrees with the trade's on {bm1['agree_with_bar0_pct']:.0f} % of days.",
        f"- Stale rows (k = −12 … −1): {int((stale['dSharpe_mid_vs_bar0'] > 0).sum())} of 12 sit above k = 0 in point estimate; "
        f"{int((stale['dSharpe_mid_lo'] > 0).sum())} with an interval above zero, {int((stale['dSharpe_mid_hi'] < 0).sum())} with an interval below zero. "
        f"Best stale row: {best_stale} at {stale.loc[best_stale, 'Sharpe_mid']:.2f} (Δ {stale.loc[best_stale, 'dSharpe_mid_vs_bar0']:+.2f} "
        f"[{stale.loc[best_stale, 'dSharpe_mid_lo']:+.2f}, {stale.loc[best_stale, 'dSharpe_mid_hi']:+.2f}]).",
        f"- First row after the close (k = +1, the next session's 10:00 row): Sharpe {b1['Sharpe_mid']:.2f} mid / {b1['Sharpe_crossed']:.2f} crossed, "
        f"Δ {b1['dSharpe_mid_vs_bar0']:+.2f} [{b1['dSharpe_mid_lo']:+.2f}, {b1['dSharpe_mid_hi']:+.2f}]; agreement with the trade {b1['agree_with_bar0_pct']:.0f} %. "
        f"Next-session rows k = +1 … +6: Sharpe mid {nxt_rows['Sharpe_mid'].min():.2f} to {nxt_rows['Sharpe_mid'].max():.2f}; "
        f"{int((nxt_rows['dSharpe_mid_lo'] > 0).sum())} of 6 intervals above zero.",
        f"- Realized variance in place of the forecast: {star['Sharpe_mid']:.2f} mid / {star['Sharpe_crossed']:.2f} crossed "
        f"(unclipped realized: {tab.loc['realized (unclipped)', 'Sharpe_mid']:.2f} / {tab.loc['realized (unclipped)', 'Sharpe_crossed']:.2f}). "
        f"Always short on the same days: {tab.loc['always short', 'Sharpe_mid']:.2f} / {tab.loc['always short', 'Sharpe_crossed']:.2f}.",
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
                "dSharpe_mid_lo",
                "dSharpe_mid_hi",
            ]
        ]
        .round(3)
        .to_string()
    )
    print("gates:", sum(g["ok"] for g in gates), "of", len(gates), "pass")
    print("wrote", OUT, "and", TEX)


if __name__ == "__main__":
    main()
