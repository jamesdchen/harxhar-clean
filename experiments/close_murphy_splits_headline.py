"""Murphy / threshold analysis and sample splits of the HEADLINE forecast, research scorer.

Two of the diagnostics parked from Section 5.4 (writeup/sections/appendix_running_parked.tex,
computed there for the paper's eight forecasts under the deck's session-bar recalibration),
re-run for the headline under the 16:00-bar recalibration of the master table
(experiments/master_table_close.py).  Nothing is re-implemented that the master table
already computes: the per-day frame results/close_master_table/master_table_daily.parquet
(the recalibrated forecast pred_clock, the scorer's target, the unclipped realized
variance, the 15:30 implied variance, the position q and the mid / crossed returns of
every table-A forecast on the 866 deck days) is the input, and the headline's Sharpe
and QLIKE are gated against master_table.csv.

PART 1 -- the decision-aligned scores (writeup/intraday_proposals/59_trade_aligned_loss.py,
mirrored):
  QLIKE      y/f - log(y/f) - 1, y the scorer's target, f = pred_clock.
  ES (c)     the elementary score of the mean at the threshold c x slice, slice = the
             15:30 implied variance: |y/slice - c| 1{f and y on opposite sides of c slice}.
             QLIKE = integral ES_c c^-2 dc (gated day by day); the trade reads c = 1.
  EL         |R| 1{q != sign(R)}: mean(q R) = mean|R| - 2 mean(EL) (gated).
  hit_var    share of days f and y fall on the same side of the slice; hit_pay: share of
             days q has the sign of the settlement return R.
  tail       long on the 10 / 20 / 50 best long-straddle days.
  Murphy     mean ES_c on the grid c = 2^(k/8), k = -16..16, one line per forecast.
  oracles    a rule told y (q = sign(y - slice)) and a rule told sign(R).
  ranks      Spearman of minus each loss with the crossed Sharpe over the comparison set
             and over every table-A forecast of the master table's rank set (table A
             without the rest-of-day check rows, exact duplicates and always short: 220
             after the 16:00 campaign), with the day-block bootstrap interval.  The
             "overall" ranks of a forecast are over the same rank set (competition ranks,
             ties share the lowest); the two oracle rules are never ranked.

PART 2 -- sample splits of the headline's sign(s) return (writeup/fill_close_option_numbers.py
and the notebook's sign split, mirrored), beside the reference (blk2 under THIS scorer)
and always short:
  COVID      drop the deck days through 2020-03-31 (n, Sharpe, t = sqrt(n) mean / sd).
  era        before / from 2022-05-16 (SPXW listings daily).
  pins       R = -1 exactly: count, share, mean return on / off pins, pin share of P&L.
  sign       mean R when s > 0 vs s <= 0, the difference with an HC0 t (statsmodels).
  Sharpes carry the day-block bootstrap 95 % interval of the master table (block 21,
  2000 draws, seed [0, n]); the difference between two disjoint periods is not paired.

Comparison set (I4c, 2026-09-30): the eight forecasts of the first run plus one canonical
row per family the 16:00 campaign added -- the untuned LightGBM refit every session (T1),
the LightGBM tuned by Optuna at every session, best of 50 trials (tp1), and the
intraday-sequence LSTM -- each on the live-feasible inputs and MSE-selected; its tree and
LSTM tables are the campaign's masked runs (the per-bar LSTM refit every session).

Committed-output gate (I4c): the headline's forecast and positions are unchanged by the
campaign and the reference, always short and the oracles are untouched, so splits.csv,
sign_split.csv, the headline's, the reference's and the oracles' rows of
scores_by_forecast.csv (every column but the ranks) and the headline's and the reference's
rows of murphy_grid.csv must equal their committed copies (git show PREV_REV:<path>;
PREV_REV = HEAD unless the environment variable CLOSE_PREV_REV names another revision),
cell by cell, to |new - old| <= 1e-9 max(1, |old|).  Every other forecast present in both
is compared for the record in before_after_committed.csv (not asserted).

Outputs: results/close_murphy_splits/*.csv, murphy_headline.png, SUMMARY.md (from the CSVs),
writeup/generated/appendix_close_murphy_splits.tex (tables + number macros).
"""

from __future__ import annotations

import io
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "notebooks", ROOT / "experiments"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import master_table_close as mtc  # noqa: E402
import score_linear_subsection as base  # noqa: E402

OUT = ROOT / "results" / "close_murphy_splits"
TEX = ROOT / "writeup" / "generated" / "appendix_close_murphy_splits.tex"
DAILY = mtc.OUT / "master_table_daily.parquet"
MASTER = mtc.OUT / "master_table.csv"
HEADLINE = mtc.HEADLINE
REFERENCE = mtc.REFERENCE
# the comparison set: the headline, the reference, the incumbent OLS, the headline's
# lasso twin, the all-features ridge, an untuned and a tuned tree, the LSTM (the first
# eight), and one canonical row per family the 16:00 campaign added (live-feasible,
# MSE-selected): the untuned LightGBM refit every session (T1), the LightGBM tuned by
# Optuna at every session (tp1, best of 50 trials), the intraday-sequence LSTM
COMPARE = [
    HEADLINE,
    REFERENCE,
    "a0",
    "sub_lasso_live_feasible",
    "sub_ridge_all_features",
    "subtree_lgbm_live_feasible",
    "subtree_tuned_all_features_lgbm",
    "lstm_live_feasible",
    "subtree_daily_live_feasible_lgbm",
    "subtree_optuna_tp1_live_feasible_lgbm",
    "lstm_intraday_live_feasible",
]
CHECK_FAMILY = "direct rest-of-day at 15:30 (check)"  # the master table's check rows
ORACLES = ["oracle: told RV (sign(RV - slice))", "oracle: told the payoff's sign"]
RANK_COLS = [
    "QLIKE_rank_all",
    "ES_rank_all",
    "Sharpe_crossed_rank_all",
    "QLIKE_rank_set",
    "ES_rank_set",
]
# the committed outputs the headline rows are gated against (the headline is unchanged by the campaign)
PREV_REV = os.environ.get("CLOSE_PREV_REV", "HEAD")
COMMITTED_TOL = 1e-9  # |new - old| <= COMMITTED_TOL * max(1, |old|), cell by cell
MURPHY_C = np.array([2.0 ** (k / 8.0) for k in range(-16, 17)])  # 1/4 .. 4, c = 1 on it
TOP_K = (10, 20, 50)
COVID_END = pd.Timestamp("2020-03-31")
ERA_CUT = pd.Timestamp("2022-05-16")
G3_GRID = 4001
GATE_REL = 1e-9
GATE_ABS = 1e-10


def rank_set(mt: pd.DataFrame) -> set[str]:
    """The master table's rank set: table A, no check rows, no exact duplicates, no always short."""
    dup = mt["duplicate_of"].fillna("").astype(str)
    keep = (
        (mt["table"] == "A")
        & (mt["family"] != CHECK_FAMILY)
        & (dup == "")
        & (mt.index != mtc.ALWAYS_SHORT)
    )
    return set(mt.index[keep])


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


def qlike(y: np.ndarray, f: np.ndarray) -> np.ndarray:
    r = y / f
    return r - np.log(r) - 1.0


def elementary(f_over_s: np.ndarray, y_over_s: np.ndarray, c: float) -> np.ndarray:
    lo = np.minimum(f_over_s, y_over_s)
    hi = np.maximum(f_over_s, y_over_s)
    return np.abs(y_over_s - c) * ((lo <= c) & (c < hi))


def stats(x: np.ndarray) -> tuple[int, float, float]:
    """(n, t = sqrt(n) mean / sd, annualized Sharpe) -- fill_close_option_numbers.stats."""
    x = np.asarray(x, float)
    sd = x.std(ddof=1)
    if not sd > 0:  # every return identical (always short on pin days): no t, no Sharpe
        return len(x), np.nan, np.nan
    return len(x), float(np.sqrt(len(x)) * x.mean() / sd), mtc.sharpe(x)


def fm(v: float, spec: str) -> str:
    """A number for a table cell, '--' when it does not exist."""
    return "--" if not np.isfinite(v) else format(v, spec)


def sharpe_ci(x: np.ndarray) -> tuple[float, float]:
    if not x.std(ddof=1) > 0:
        return np.nan, np.nan
    idx = mtc.boot_idx(len(x))
    lo, hi = np.percentile(mtc.boot_sharpe(x, idx), [2.5, 97.5])
    return float(lo), float(hi)


def spearman_ci(loss: np.ndarray, pnl_rows: np.ndarray) -> tuple[float, float, float]:
    """Spearman of minus the mean loss with the crossed Sharpe across forecasts (rows), with
    the day-block bootstrap over the columns (days); loss and pnl_rows are (d, n)."""
    from scipy.stats import spearmanr

    idx = mtc.boot_idx(pnl_rows.shape[1])
    point = float(
        spearmanr(
            -loss.mean(axis=1),
            mtc.boot_sharpe.__wrapped__(pnl_rows)
            if hasattr(mtc.boot_sharpe, "__wrapped__")
            else _sharpe_rows(pnl_rows),
        )[0]
    )
    draws = []
    for b in range(idx.shape[0]):
        cols = idx[b]
        draws.append(
            spearmanr(-loss[:, cols].mean(axis=1), _sharpe_rows(pnl_rows[:, cols]))[0]
        )
    lo, hi = np.percentile(np.array(draws, float), [2.5, 97.5])
    return point, float(lo), float(hi)


def _sharpe_rows(x: np.ndarray) -> np.ndarray:
    a = np.atleast_2d(x)
    return a.mean(axis=1) / a.std(axis=1, ddof=1) * mtc.ANN


def main() -> None:
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

    if not DAILY.exists():
        raise SystemExit(
            f"{DAILY} is missing: run experiments/master_table_close.py first"
        )
    D = pd.read_parquet(DAILY)
    D["day"] = pd.to_datetime(D["day"])
    mt = pd.read_csv(MASTER).set_index("key")
    QLIKE_COL = [
        c
        for c in mt.columns
        if c.lower().startswith("qlike")
        and "recal" in c.lower()
        and "unclip" not in c.lower()
        and "floor" not in c.lower()
    ][0]
    deck = pd.read_parquet(base.DECK).sort_index()
    deck.index = pd.DatetimeIndex(pd.to_datetime(deck.index)).as_unit("ns")
    days = deck.index
    assert len(days) == 866

    # every table-A forecast on all 866 days, as (d, n) arrays in the deck's day order
    keys = [
        k
        for k in D["key"].unique()
        if (D["key"] == k).sum() == len(days) and k in mt.index
    ]
    frames = {k: D[D["key"] == k].set_index("day").reindex(days) for k in keys}
    assert all(f["pred_clock"].notna().all() for f in frames.values())
    in_set = rank_set(mt)
    assert in_set <= set(keys), sorted(in_set - set(keys))
    assert set(COMPARE) <= in_set, sorted(set(COMPARE) - in_set)
    iset = [i for i, k in enumerate(keys) if k in in_set]
    F = np.vstack([frames[k]["pred_clock"].to_numpy(float) for k in keys])
    y = frames[HEADLINE]["target"].to_numpy(float)
    yu = frames[HEADLINE]["rv_unclipped"].to_numpy(float)
    s = frames[HEADLINE]["iv_var"].to_numpy(float)
    R = deck["R"].to_numpy(float)
    for k in keys:  # the shared target, slice and returns
        gate(
            f"target shared ({k})",
            np.abs(frames[k]["target"].to_numpy(float) / y - 1).max(),
            GATE_REL,
            len(days),
        )
    gate(
        "slice = the deck's iv_var",
        np.abs(s / deck["iv_var"].to_numpy(float) - 1).max(),
        GATE_REL,
        len(days),
    )
    ex = deck["exit"].to_numpy(float)
    ask = (deck["ask_c"] + deck["ask_p"]).to_numpy(float)
    bid = (deck["bid_c"] + deck["bid_p"]).to_numpy(float)
    cl, cs = ex / ask - 1.0, ex / bid - 1.0

    def pnl(q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        a = np.atleast_2d(q)
        return a * R, a * np.where(a > 0, cl, cs)

    Q = np.where(F > s, 1.0, -1.0)
    MID, CR = pnl(Q)
    for i, k in enumerate(keys):  # the master table's own positions and returns
        gate(
            f"q = master table ({k})",
            np.abs(Q[i] - frames[k]["q"].to_numpy(float)).max(),
            0,
            len(days),
        )
        gate(
            f"ret_mid = master table ({k})",
            np.abs(MID[i] - frames[k]["ret_mid"].to_numpy(float)).max(),
            GATE_ABS,
            len(days),
        )
        gate(
            f"ret_crossed = master table ({k})",
            np.abs(CR[i] - frames[k]["ret_crossed"].to_numpy(float)).max(),
            GATE_ABS,
            len(days),
        )
    ih = keys.index(HEADLINE)
    gate(
        "headline Sharpe mid = master_table.csv",
        abs(mtc.sharpe(MID[ih]) / float(mt.loc[HEADLINE, "Sharpe_mid"]) - 1),
        GATE_REL,
        len(days),
    )
    gate(
        "headline Sharpe crossed = master_table.csv",
        abs(mtc.sharpe(CR[ih]) / float(mt.loc[HEADLINE, "Sharpe_crossed"]) - 1),
        GATE_REL,
        len(days),
    )

    # ---------------------------------------------------------------- part 1: the scores
    QL = qlike(y[None, :], F)
    gate(
        "headline QLIKE (recal) = master_table.csv",
        abs(QL[ih].mean() / float(mt.loc[HEADLINE, QLIKE_COL]) - 1),
        GATE_REL,
        len(days),
    )
    ES1 = elementary(F / s, (y / s)[None, :], 1.0)
    sgnR = np.where(R > 0, 1.0, -1.0)
    EL = np.abs(R)[None, :] * (Q != sgnR)
    gate(
        "mean(qR) = mean|R| - 2 mean(EL), every forecast",
        np.abs(MID.mean(axis=1) - (np.abs(R).mean() - 2 * EL.mean(axis=1))).max(),
        1e-12,
        len(keys),
    )
    # G3: QLIKE equals the c^-2 mixture of elementary scores, day by day (headline)
    fs, ys = F[ih] / s, y / s
    lo, hi = np.minimum(fs, ys), np.maximum(fs, ys)
    u = np.linspace(0.0, 1.0, G3_GRID)
    c = lo[:, None] * (hi / lo)[:, None] ** u[None, :]
    integral = np.trapezoid(np.abs(ys[:, None] - c) / c**2, c, axis=1)
    gate(
        "QLIKE = integral ES_c c^-2 dc, day by day (headline)",
        np.abs(integral - QL[ih]).max(),
        1e-6,
        len(days),
    )
    hit_var = (F > s) == (y > s)[None, :]
    hit_pay = Q == sgnR[None, :]
    order = np.argsort(R)[::-1]
    rows = []
    for i, k in enumerate(keys):
        rows.append(
            {
                "key": k,
                "label": mt.loc[k, "label"],
                "in_comparison_set": k in COMPARE,
                "in_rank_set": k in in_set,
                "QLIKE": QL[i].mean(),
                "ES_at_threshold": ES1[i].mean(),
                "ES_at_threshold_unclipped": elementary(F[i] / s, yu / s, 1.0).mean(),
                "EL": EL[i].mean(),
                "hit_var_pct": 100 * hit_var[i].mean(),
                "hit_pay_pct": 100 * hit_pay[i].mean(),
                **{
                    f"long_on_top{kk}": int((Q[i, order[:kk]] > 0).sum())
                    for kk in TOP_K
                },
                "pct_buy": 100 * (Q[i] > 0).mean(),
                "Sharpe_mid": mtc.sharpe(MID[i]),
                "Sharpe_crossed": mtc.sharpe(CR[i]),
            }
        )
    # the oracles
    q_var = np.where(y > s, 1.0, -1.0)
    q_pay = sgnR.copy()
    for name, q in (
        ("oracle: told RV (sign(RV - slice))", q_var),
        ("oracle: told the payoff's sign", q_pay),
    ):
        m, cr = pnl(q)
        rows.append(
            {
                "key": name,
                "label": name,
                "in_comparison_set": True,
                "in_rank_set": False,
                "QLIKE": np.nan,
                "ES_at_threshold": np.nan,
                "ES_at_threshold_unclipped": np.nan,
                "EL": (np.abs(R) * (q != sgnR)).mean(),
                "hit_var_pct": 100 * ((q > 0) == (y > s)).mean(),
                "hit_pay_pct": 100 * (q == sgnR).mean(),
                **{f"long_on_top{kk}": int((q[order[:kk]] > 0).sum()) for kk in TOP_K},
                "pct_buy": 100 * (q > 0).mean(),
                "Sharpe_mid": mtc.sharpe(m[0]),
                "Sharpe_crossed": mtc.sharpe(cr[0]),
            }
        )
    scores = pd.DataFrame(rows).set_index("key")
    # "overall" ranks over the rank set only (the oracles, check rows and duplicates are not ranked)
    rs = scores[scores["in_rank_set"]]
    for col, v in (
        ("QLIKE_rank_all", rs["QLIKE"]),
        ("ES_rank_all", rs["ES_at_threshold"]),
        ("Sharpe_crossed_rank_all", -rs["Sharpe_crossed"]),
    ):
        scores[col] = np.nan
        scores.loc[rs.index, col] = v.rank(method="min")
    sub = scores.loc[COMPARE]
    scores["QLIKE_rank_set"] = np.nan
    scores.loc[COMPARE, "QLIKE_rank_set"] = sub["QLIKE"].rank(method="min")
    scores["ES_rank_set"] = np.nan
    scores.loc[COMPARE, "ES_rank_set"] = sub["ES_at_threshold"].rank(method="min")
    scores.to_csv(OUT / "scores_by_forecast.csv")

    # the Murphy diagram: mean ES_c, one row per forecast in the comparison set, and the headline/reference on the unclipped y
    ic = [keys.index(k) for k in COMPARE]
    grid = pd.DataFrame(
        {
            f"c={c_:.4f}": [elementary(F[i] / s, y / s, c_).mean() for i in ic]
            for c_ in MURPHY_C
        },
        index=[mt.loc[k, "label"] for k in COMPARE],
    )
    grid.index.name = "forecast"
    grid.to_csv(OUT / "murphy_grid.csv")
    # which forecast leads at each threshold, and the headline's rank
    lead = grid.idxmin(axis=0)
    hl = mt.loc[HEADLINE, "label"]
    rank_h = grid.rank(axis=0).loc[hl]
    murphy_lead = pd.DataFrame(
        {
            "c": MURPHY_C,
            "leader": lead.to_numpy(),
            "headline_rank_of_set": rank_h.to_numpy(),
            "headline_score": grid.loc[hl].to_numpy(),
            "best_score": grid.min(axis=0).to_numpy(),
        }
    )
    murphy_lead.to_csv(OUT / "murphy_leader_by_threshold.csv", index=False)

    # rank agreement of minus each loss with the crossed Sharpe: the comparison set and every table-A forecast
    rk_rows = []
    for setname, idxs in (
        ("comparison set", ic),
        ("all table-A forecasts", iset),
    ):
        for lname, L in (
            ("QLIKE", QL),
            ("ES at the threshold", ES1),
            ("EL (affine in the mid P&L)", EL),
        ):
            pt, lo_, hi_ = spearman_ci(L[idxs], CR[idxs])
            rk_rows.append(
                {
                    "set": setname,
                    "n_forecasts": len(idxs),
                    "loss": lname,
                    "spearman_minus_loss_vs_crossed_sharpe": pt,
                    "lo": lo_,
                    "hi": hi_,
                }
            )
    ranks = pd.DataFrame(rk_rows)
    ranks.to_csv(OUT / "rank_vs_sharpe.csv", index=False)

    # ---------------------------------------------------------------- part 2: sample splits
    ir = keys.index(REFERENCE)
    series = {
        "headline sign(s)": MID[ih],
        "reference sign(s) (this scorer)": MID[ir],
        "always short": -R,
    }
    series_cr = {
        "headline sign(s)": CR[ih],
        "reference sign(s) (this scorer)": CR[ir],
        "always short": cs * -1.0 * -1.0 * 0 + (1.0 - ex / bid),
    }
    covid = days <= COVID_END
    pre = days < ERA_CUT
    pin = R <= -1 + 1e-12
    gate(
        "always short collects +1 on pin days",
        abs((-R)[pin].mean() - 1.0),
        1e-9,
        int(pin.sum()),
    )
    sp_rows = []
    for name, x in series.items():
        xc = series_cr[name]
        for split, mask in (
            ("all days", np.ones(len(days), bool)),
            ("through 2020-03-31 (the COVID quarter)", covid),
            ("after 2020-03-31", ~covid),
            ("before 2022-05-16", pre),
            ("from 2022-05-16 (daily SPXW listings)", ~pre),
            ("settlement pins (R = -1)", pin),
            ("off pins", ~pin),
        ):
            n, t, sh = stats(x[mask])
            lo_, hi_ = sharpe_ci(x[mask]) if mask.sum() >= 30 else (np.nan, np.nan)
            nc, tc, shc = stats(xc[mask])
            sp_rows.append(
                {
                    "series": name,
                    "split": split,
                    "n": n,
                    "mean_mid": float(x[mask].mean()),
                    "t_mid": t,
                    "Sharpe_mid": sh,
                    "Sharpe_mid_lo": lo_,
                    "Sharpe_mid_hi": hi_,
                    "share_of_total_pnl_mid_pct": 100 * x[mask].sum() / x.sum(),
                    "mean_crossed": float(xc[mask].mean()),
                    "t_crossed": tc,
                    "Sharpe_crossed": shc,
                }
            )
    splits = pd.DataFrame(sp_rows)
    splits.to_csv(OUT / "splits.csv", index=False)
    # the sign split of the settlement return by the headline's and the reference's signal
    sg_rows = []
    for name, i in (("headline", ih), ("reference (this scorer)", ir)):
        ind = (Q[i] > 0).astype(float)
        fit = sm.OLS(R, sm.add_constant(ind)).fit(cov_type="HC0")
        sg_rows.append(
            {
                "forecast": name,
                "mean_R_when_s_le_0": float(R[ind == 0].mean()),
                "mean_R_when_s_gt_0": float(R[ind == 1].mean()),
                "diff": float(fit.params[1]),
                "t_HC0": float(fit.tvalues[1]),
                "n_sell": int((ind == 0).sum()),
                "n_buy": int((ind == 1).sum()),
            }
        )
    signsplit = pd.DataFrame(sg_rows)
    signsplit.to_csv(OUT / "sign_split.csv", index=False)

    # ---- the committed copies: the headline, the reference, always short and the oracles are unchanged
    hl, rl = mt.loc[HEADLINE, "label"], mt.loc[REFERENCE, "label"]
    for fname, idx, keep in (
        ("splits.csv", ["series", "split"], None),
        ("sign_split.csv", ["forecast"], None),
        ("scores_by_forecast.csv", ["key"], [HEADLINE, REFERENCE, *ORACLES]),
        ("murphy_grid.csv", ["forecast"], [hl, rl]),
    ):
        old, sha = committed_csv(OUT / fname)
        new = pd.read_csv(OUT / fname)
        old, new = old.set_index(idx), new.set_index(idx)
        if keep is not None:
            old = old.loc[
                old.index.isin(keep), [c for c in old.columns if c not in RANK_COLS]
            ]
        rows_ = "every row" if keep is None else f"{len(old)} rows"
        if fname == "scores_by_forecast.csv":
            rows_ += ", ranks left out"
        gate(
            f"{fname} = the committed copy ({PREV_REV} = {sha}), {rows_}, "
            "max |new - old| / max(1, |old|)",
            committed_diff(new, old),
            COMMITTED_TOL,
            int(old.size),
        )
    # every other forecast present in both runs, compared for the record (not asserted)
    old_sc, prev_sha = committed_csv(OUT / "scores_by_forecast.csv", index_col=0)
    num = [
        c
        for c in old_sc.columns
        if c not in RANK_COLS and pd.api.types.is_numeric_dtype(old_sc[c])
    ]
    ba_rows = []
    for k in scores.index:
        if k not in old_sc.index:
            ba_rows.append({"key": k, "label": scores.loc[k, "label"], "status": "new"})
            continue
        o, n_ = old_sc.loc[k, num].astype(float), scores.loc[k, num].astype(float)
        rel = (n_ - o).abs() / np.maximum(1.0, o.abs())
        ba_rows.append(
            {
                "key": k,
                "label": scores.loc[k, "label"],
                "status": "in both",
                "gated": k in (HEADLINE, REFERENCE, *ORACLES),
                "in_comparison_set_before": bool(old_sc.loc[k, "in_comparison_set"]),
                "in_comparison_set_after": bool(scores.loc[k, "in_comparison_set"]),
                **{f"{c}_before": o[c] for c in ("QLIKE", "ES_at_threshold", "EL")},
                **{f"{c}_after": n_[c] for c in ("QLIKE", "ES_at_threshold", "EL")},
                "Sharpe_crossed_before": o["Sharpe_crossed"],
                "Sharpe_crossed_after": n_["Sharpe_crossed"],
                "max_rel_diff": float(rel.fillna(0.0).max()),
            }
        )
    for k in old_sc.index.difference(scores.index):
        ba_rows.append({"key": k, "label": old_sc.loc[k, "label"], "status": "dropped"})
    before_after = pd.DataFrame(ba_rows)
    before_after.to_csv(OUT / "before_after_committed.csv", index=False)
    old_rk, _ = committed_csv(OUT / "rank_vs_sharpe.csv")
    pd.DataFrame(gates).to_csv(OUT / "gates.csv", index=False)
    assert all(g["ok"] for g in gates), [g for g in gates if not g["ok"]]

    # ---------------------------------------------------------------- figure
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # eleven lines: the headline black and thick, the other ten one tab10 colour each;
    # line style by model class (linear solid, tree dashed, LSTM dash-dot); legend below
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.6))
    colours = iter(f"C{i}" for i in range(10))  # the tab10 colours
    for k, (lab, row) in zip(COMPARE, grid.iterrows()):
        head = k == HEADLINE
        style = {
            "color": "k" if head else next(colours),
            "lw": 2.4 if head else 1.1,
            "ls": "-"
            if (k.startswith("sub_") or k in (REFERENCE, "a0"))
            else ("--" if k.startswith("subtree_") else "-."),
            "zorder": 3 if head else 2,
        }
        axes[0].plot(MURPHY_C, row.to_numpy(), label=lab, **style)
        axes[1].plot(
            MURPHY_C, row.to_numpy() - grid.mean(axis=0).to_numpy(), label=lab, **style
        )
    for ax in axes:
        ax.set_xscale("log", base=2)
        ax.axvline(1.0, color="0.5", ls=":", lw=0.8)
        ax.set_xlabel("threshold c, in units of the 15:30 implied variance")
    axes[0].set_ylabel("mean elementary score ES_c (lower is better)")
    axes[1].set_ylabel("ES_c minus the mean of the set at c")
    axes[0].set_title(
        "(a) Murphy diagram, 866 deck days, 16:00-bar recalibration", fontsize=9
    )
    axes[1].set_title("(b) each line minus the mean of the set", fontsize=9)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=3,
        fontsize=7.5,
        frameon=False,
        bbox_to_anchor=(0.5, 0.0),
    )
    fig.tight_layout(rect=(0.0, 0.17, 1.0, 1.0))
    fig.savefig(OUT / "murphy_headline.png", dpi=130, bbox_inches="tight")

    # ---------------------------------------------------------------- tex: tables + macros
    def t3(v: float) -> str:
        return f"{v:.3f}"

    L = [
        "% AUTO-GENERATED by experiments/close_murphy_splits_headline.py from results/close_murphy_splits/*.csv -- do not edit."
    ]
    L += [
        r"\begingroup\scriptsize\setlength{\tabcolsep}{2pt}",
        r"\begin{longtable}{p{0.27\textwidth}rrrrrrr}",
        r"\caption{The comparison set scored two ways on the 866 deck days under the 16:00-bar recalibration: QLIKE against the scorer's target, "
        r"the elementary score at the trade's threshold (the 15:30 implied variance), the economic loss $|R|\,\mathbf 1\{q\neq\mathrm{sign}(R)\}$, "
        r"the share of days the forecast falls on the same side of the threshold as the realized variance (same side as $RV$) and the share on which the "
        r"position has the sign of the settlement return (same side as $R$), and the $\mathrm{sign}(s)$ "
        r"Sharpe at both fills. The last two rows are rules told the realized variance, and the payoff's sign, in advance.}\label{tab:app_murphy_headline}\\",
        r"\toprule",
        r"forecast & QLIKE & ES($c{=}1$) & EL & side of $RV$ \% & side of $R$ \% & Sharpe mid & Sharpe crossed \\",
        r"\midrule",
        r"\endfirsthead",
        r"\caption[]{(continued)}\\",
        r"\toprule",
        r"forecast & QLIKE & ES($c{=}1$) & EL & side of $RV$ \% & side of $R$ \% & Sharpe mid & Sharpe crossed \\",
        r"\midrule",
        r"\endhead",
        r"\bottomrule",
        r"\endlastfoot",
    ]
    for k in COMPARE + [
        "oracle: told RV (sign(RV - slice))",
        "oracle: told the payoff's sign",
    ]:
        r_ = scores.loc[k]
        lab = str(r_["label"]).replace("_", r"\_").replace("&", r"\&")
        if k == HEADLINE:
            lab = r"\textbf{" + lab + "}"
        ql_ = "--" if np.isnan(r_["QLIKE"]) else t3(r_["QLIKE"])
        es_ = "--" if np.isnan(r_["ES_at_threshold"]) else t3(r_["ES_at_threshold"])
        L.append(
            f"\\raggedright {lab} & {ql_} & {es_} & {t3(r_['EL'])} & {r_['hit_var_pct']:.1f} & {r_['hit_pay_pct']:.1f} & "
            f"{r_['Sharpe_mid']:.2f} & {r_['Sharpe_crossed']:.2f} \\\\"
        )
    L += [r"\end{longtable}", r"\endgroup", ""]
    # splits table
    L += [
        r"\begingroup\scriptsize\setlength{\tabcolsep}{1pt}",
        r"\begin{longtable}{llrrrlr}",
        r"\caption{Sample splits of the headline's $\mathrm{sign}(s)$ return (midpoint fill) under the 16:00-bar recalibration, beside the paper's "
        r"block-diagonal ridge scored the same way and always short. $t=\sqrt{n}\,\bar x/s_x$; the Sharpe ratio is annualized with its day-block "
        r"bootstrap 95\,\% interval (circular blocks of 21 days, 2000 draws); the P\&L share is the split's sum of daily returns over the full-sample sum.}"
        r"\label{tab:app_splits_headline}\\",
        r"\toprule",
        r"series & split & $n$ & mean & $t$ & Sharpe [95\,\%] & P\&L share \% \\",
        r"\midrule",
        r"\endfirsthead",
        r"\caption[]{(continued)}\\",
        r"\toprule",
        r"series & split & $n$ & mean & $t$ & Sharpe [95\,\%] & P\&L share \% \\",
        r"\midrule",
        r"\endhead",
        r"\bottomrule",
        r"\endlastfoot",
    ]
    for _, r_ in splits.iterrows():
        ci = (
            f"[{r_['Sharpe_mid_lo']:+.2f}, {r_['Sharpe_mid_hi']:+.2f}]"
            if np.isfinite(r_["Sharpe_mid_lo"])
            else ""
        )
        L.append(
            f"{r_['series']} & {r_['split']} & {int(r_['n'])} & {r_['mean_mid']:+.3f} & {fm(r_['t_mid'], '+.2f')} & {fm(r_['Sharpe_mid'], '.2f')} {ci} & "
            f"{r_['share_of_total_pnl_mid_pct']:.0f} \\\\"
        )
    L += [r"\end{longtable}", r"\endgroup", ""]

    def sp(series_name: str, split_name: str) -> pd.Series:
        return splits[
            (splits["series"] == series_name) & (splits["split"] == split_name)
        ].iloc[0]

    h = scores.loc[HEADLINE]
    ov, op = (
        scores.loc["oracle: told RV (sign(RV - slice))"],
        scores.loc["oracle: told the payoff's sign"],
    )
    rk = ranks.set_index(["set", "loss"])
    lead_at_1 = murphy_lead.loc[np.isclose(murphy_lead["c"], 1.0)].iloc[0]
    n_lead_h = int((murphy_lead["leader"] == hl).sum())
    lin_labels = {
        mt.loc[k, "label"]
        for k in COMPARE
        if k.startswith("sub_") or k in ("blk2", "a0")
    }
    n_lead_lin = int(murphy_lead["leader"].isin(lin_labels).sum())
    sg = signsplit.set_index("forecast").loc["headline"]
    ha, hc, hb, hd = (
        sp("headline sign(s)", "all days"),
        sp("headline sign(s)", "after 2020-03-31"),
        sp("headline sign(s)", "before 2022-05-16"),
        sp("headline sign(s)", "from 2022-05-16 (daily SPXW listings)"),
    )
    hp, ho = (
        sp("headline sign(s)", "settlement pins (R = -1)"),
        sp("headline sign(s)", "off pins"),
    )
    ap_, ao = (
        sp("always short", "settlement pins (R = -1)"),
        sp("always short", "off pins"),
    )
    rb, rd = (
        sp("reference sign(s) (this scorer)", "before 2022-05-16"),
        sp("reference sign(s) (this scorer)", "from 2022-05-16 (daily SPXW listings)"),
    )
    macros = {
        "hmurHQLIKE": t3(h["QLIKE"]),
        "hmurHES": t3(h["ES_at_threshold"]),
        "hmurHQLIKERankSet": f"{int(h['QLIKE_rank_set'])}",
        "hmurHESRankSet": f"{int(h['ES_rank_set'])}",
        "hmurHQLIKERankAll": f"{int(h['QLIKE_rank_all'])}",
        "hmurHESRankAll": f"{int(h['ES_rank_all'])}",
        "hmurHSharpeRankAll": f"{int(h['Sharpe_crossed_rank_all'])}",
        "hmurNAll": f"{len(iset)}",
        "hmurNSet": f"{len(COMPARE)}",
        "hmurHHitVar": f"{h['hit_var_pct']:.0f}",
        "hmurHHitPay": f"{h['hit_pay_pct']:.0f}",
        "hmurHTop": f"{int(h['long_on_top20'])}",
        "hmurOracleAgree": f"{ov['hit_pay_pct']:.0f}",
        "hmurOracleMiss": f"{100 - ov['hit_pay_pct']:.0f}",
        "hmurOracleShMid": f"{ov['Sharpe_mid']:.2f}",
        "hmurOracleShX": f"{ov['Sharpe_crossed']:.2f}",
        "hmurPayoffOracleShX": f"{op['Sharpe_crossed']:.2f}",
        "hmurLeaderAtOne": str(lead_at_1["leader"]).replace("_", r"\_"),
        "hmurHRankAtOne": f"{int(lead_at_1['headline_rank_of_set'])}",
        "hmurNGrid": f"{len(MURPHY_C)}",
        "hmurNLeadH": f"{n_lead_h}",
        "hmurNLeadLin": f"{n_lead_lin}",
        "hmurGridMin": f"{MURPHY_C.min():.2f}",
        "hmurGridMax": f"{MURPHY_C.max():.0f}",
        "hmurRhoQSet": f"{rk.loc[('comparison set', 'QLIKE'), 'spearman_minus_loss_vs_crossed_sharpe']:+.2f}",
        "hmurRhoQSetLo": f"{rk.loc[('comparison set', 'QLIKE'), 'lo']:+.2f}",
        "hmurRhoQSetHi": f"{rk.loc[('comparison set', 'QLIKE'), 'hi']:+.2f}",
        "hmurRhoESSet": f"{rk.loc[('comparison set', 'ES at the threshold'), 'spearman_minus_loss_vs_crossed_sharpe']:+.2f}",
        "hmurRhoESSetLo": f"{rk.loc[('comparison set', 'ES at the threshold'), 'lo']:+.2f}",
        "hmurRhoESSetHi": f"{rk.loc[('comparison set', 'ES at the threshold'), 'hi']:+.2f}",
        "hmurRhoQAll": f"{rk.loc[('all table-A forecasts', 'QLIKE'), 'spearman_minus_loss_vs_crossed_sharpe']:+.2f}",
        "hmurRhoQAllLo": f"{rk.loc[('all table-A forecasts', 'QLIKE'), 'lo']:+.2f}",
        "hmurRhoQAllHi": f"{rk.loc[('all table-A forecasts', 'QLIKE'), 'hi']:+.2f}",
        "hmurRhoESAll": f"{rk.loc[('all table-A forecasts', 'ES at the threshold'), 'spearman_minus_loss_vs_crossed_sharpe']:+.2f}",
        "hmurRhoESAllLo": f"{rk.loc[('all table-A forecasts', 'ES at the threshold'), 'lo']:+.2f}",
        "hmurRhoESAllHi": f"{rk.loc[('all table-A forecasts', 'ES at the threshold'), 'hi']:+.2f}",
        "hsplHAllSh": f"{ha['Sharpe_mid']:.2f}",
        "hsplHAllT": f"{ha['t_mid']:.2f}",
        "hsplNCovid": f"{int(sp('headline sign(s)', 'through 2020-03-31 (the COVID quarter)')['n'])}",
        "hsplHPostCovidN": f"{int(hc['n'])}",
        "hsplHPostCovidSh": f"{hc['Sharpe_mid']:.2f}",
        "hsplHPostCovidT": f"{hc['t_mid']:.2f}",
        "hsplHPostCovidLo": f"{hc['Sharpe_mid_lo']:.2f}",
        "hsplHPostCovidHi": f"{hc['Sharpe_mid_hi']:.2f}",
        "hsplHPreN": f"{int(hb['n'])}",
        "hsplHPreSh": f"{hb['Sharpe_mid']:.2f}",
        "hsplHPreT": f"{hb['t_mid']:.2f}",
        "hsplHPreLo": f"{hb['Sharpe_mid_lo']:.2f}",
        "hsplHPreHi": f"{hb['Sharpe_mid_hi']:.2f}",
        "hsplHPostN": f"{int(hd['n'])}",
        "hsplHPostSh": f"{hd['Sharpe_mid']:.2f}",
        "hsplHPostT": f"{hd['t_mid']:.2f}",
        "hsplHPostLo": f"{hd['Sharpe_mid_lo']:.2f}",
        "hsplHPostHi": f"{hd['Sharpe_mid_hi']:.2f}",
        "hsplRPreSh": f"{rb['Sharpe_mid']:.2f}",
        "hsplRPostSh": f"{rd['Sharpe_mid']:.2f}",
        "hsplNPin": f"{int(hp['n'])}",
        "hsplPinPct": f"{100 * pin.mean():.1f}",
        "hsplHPinMean": f"{hp['mean_mid']:+.2f}",
        "hsplHOffPinMean": f"{ho['mean_mid']:+.2f}",
        "hsplHPinShare": f"{hp['share_of_total_pnl_mid_pct']:.0f}",
        "hsplASOffPinMean": f"{ao['mean_mid']:+.2f}",
        "hsplASPinMean": f"{ap_['mean_mid']:+.2f}",
        "hsplSignBuyMean": f"{sg['mean_R_when_s_gt_0']:+.2f}",
        "hsplSignSellMean": f"{sg['mean_R_when_s_le_0']:+.2f}",
        "hsplSignDiff": f"{sg['diff']:.2f}",
        "hsplSignT": f"{sg['t_HC0']:.2f}",
        "hsplSignNBuy": f"{int(sg['n_buy'])}",
        "hsplSignNSell": f"{int(sg['n_sell'])}",
    }
    L += [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in macros.items()]
    TEX.parent.mkdir(parents=True, exist_ok=True)
    TEX.write_text("\n".join(L) + "\n", encoding="utf-8")

    # ---------------------------------------------------------------- SUMMARY.md from the CSVs
    S = [
        "# Murphy / threshold analysis and sample splits of the headline forecast (16:00-bar recalibration)",
        "",
        "Written by `experiments/close_murphy_splits_headline.py` from the CSVs in this folder (`scores_by_forecast.csv`, `murphy_grid.csv`, "
        "`murphy_leader_by_threshold.csv`, `rank_vs_sharpe.csv`, `splits.csv`, `sign_split.csv`, `gates.csv`). Input: the master table's per-day frame "
        "(`results/close_master_table/master_table_daily.parquet`, regenerated by `experiments/master_table_close.py`); the headline's Sharpe, QLIKE, positions "
        "and returns are gated against `master_table.csv`. Scorer: pred_clock = (f² + s)·B with the causal 250-session smear of the 16:00 clock; the deck's 15:30 sign(s) "
        "straddle trade; 866 days 2020-01-03 .. 2024-04-30. The parked versions of these diagnostics (Appendix D.15) are on the deck's session-bar recalibration and are not set beside these.",
        "",
        f"## Part 1 — decision-aligned scores (comparison set of {len(COMPARE)}; all {len(keys)} table-A forecasts in `scores_by_forecast.csv`, "
        f"ranked over the master table's rank set of {len(iset)}: check rows, exact duplicates and always short left out, `in_rank_set`)",
        "",
        "| forecast | QLIKE | ES(c=1) | EL | same side as RV % | same side as R % | long on top 10/20/50 | buy % | Sharpe mid | Sharpe crossed |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for k in COMPARE + [
        "oracle: told RV (sign(RV - slice))",
        "oracle: told the payoff's sign",
    ]:
        r_ = scores.loc[k]
        S.append(
            f"| {r_['label']} | {r_['QLIKE']:.4f} | {r_['ES_at_threshold']:.4f} | {r_['EL']:.4f} | {r_['hit_var_pct']:.1f} | {r_['hit_pay_pct']:.1f} | "
            f"{int(r_['long_on_top10'])}/{int(r_['long_on_top20'])}/{int(r_['long_on_top50'])} | {r_['pct_buy']:.0f} | {r_['Sharpe_mid']:.2f} | {r_['Sharpe_crossed']:.2f} |"
        )
    S += [
        "",
        f"- Headline ranks: QLIKE {int(h['QLIKE_rank_set'])} of {len(COMPARE)} in the set ({int(h['QLIKE_rank_all'])} of {len(iset)} overall); ES at the threshold "
        f"{int(h['ES_rank_set'])} of {len(COMPARE)} ({int(h['ES_rank_all'])} overall); crossed Sharpe {int(h['Sharpe_crossed_rank_all'])} of {len(iset)} overall.",
        f"- Murphy diagram (`murphy_grid.csv`, c = 2^(k/8) from {MURPHY_C.min():.2f} to {MURPHY_C.max():.0f}): the headline has the lowest score of the set at {n_lead_h} of {len(MURPHY_C)} thresholds; "
        f"a linear forecast leads at {n_lead_lin}; at c = 1 the leader is {lead_at_1['leader']} and the headline ranks {int(lead_at_1['headline_rank_of_set'])} of {len(COMPARE)}.",
        f"- Oracles: a rule told RV takes the payoff's side on {ov['hit_pay_pct']:.1f} % of days (Sharpe {ov['Sharpe_mid']:.2f} mid / {ov['Sharpe_crossed']:.2f} crossed); "
        f"a rule told the payoff's sign: {op['Sharpe_crossed']:.2f} crossed. The headline takes the payoff's side on {h['hit_pay_pct']:.1f} % and the side of RV − slice on {h['hit_var_pct']:.1f} %.",
        "- Rank agreement of minus each loss with the crossed Sharpe (Spearman, day-block bootstrap 95 %):",
    ]
    for _, r_ in ranks.iterrows():
        S.append(
            f"  - {r_['set']} (n = {int(r_['n_forecasts'])}), {r_['loss']}: {r_['spearman_minus_loss_vs_crossed_sharpe']:+.2f} [{r_['lo']:+.2f}, {r_['hi']:+.2f}]"
        )
    S += [
        "",
        f"## Before / after: this run against the committed outputs ({PREV_REV} = {prev_sha})",
        "",
        f"- Comparison set: {len(old_sc[old_sc['in_comparison_set'].astype(bool) & ~old_sc.index.isin(ORACLES)])} -> {len(COMPARE)} forecasts; "
        "added: "
        + "; ".join(
            str(mt.loc[k, "label"])
            for k in COMPARE
            if k not in old_sc.index or not bool(old_sc.loc[k, "in_comparison_set"])
        )
        + ".",
        f"- Forecasts in `scores_by_forecast.csv`: {int((~old_sc.index.isin(ORACLES)).sum())} -> {len(keys)} "
        f"({int((before_after['status'] == 'new').sum())} new, {int((before_after['status'] == 'dropped').sum())} dropped); "
        f"the 'overall' ranks and rank correlations are now over the rank set of {len(iset)} (the first run ranked every table-A row, "
        "the check rows included, and the two oracle rules in the crossed-Sharpe rank).",
        "- Asserted unchanged (headline, reference, oracles; `gates.csv`): max relative difference "
        + f"{before_after.loc[before_after['gated'].eq(True), 'max_rel_diff'].max():.2g}.",
        "- Comparison-set members whose forecasts changed (not asserted; `before_after_committed.csv`):",
    ]
    ba = before_after[before_after["status"] == "in both"].set_index("key")
    for k in COMPARE:
        if k in ba.index and not bool(ba.loc[k, "gated"]):
            b_ = ba.loc[k]
            S.append(
                f"  - {b_['label']}: QLIKE {b_['QLIKE_before']:.4f} -> {b_['QLIKE_after']:.4f}, ES(c=1) {b_['ES_at_threshold_before']:.4f} -> "
                f"{b_['ES_at_threshold_after']:.4f}, Sharpe crossed {b_['Sharpe_crossed_before']:.2f} -> {b_['Sharpe_crossed_after']:.2f} "
                f"(max relative difference {b_['max_rel_diff']:.2g})"
            )
    S += [
        "- Rank agreement, before -> after (Spearman [95 %]):",
    ]
    ork = old_rk.set_index(["set", "loss"])
    for _, r_ in ranks.iterrows():
        o_ = ork.loc[(r_["set"], r_["loss"])]
        S.append(
            f"  - {r_['set']}, {r_['loss']}: n {int(o_['n_forecasts'])} -> {int(r_['n_forecasts'])}; "
            f"{o_['spearman_minus_loss_vs_crossed_sharpe']:+.2f} [{o_['lo']:+.2f}, {o_['hi']:+.2f}] -> "
            f"{r_['spearman_minus_loss_vs_crossed_sharpe']:+.2f} [{r_['lo']:+.2f}, {r_['hi']:+.2f}]"
        )
    S += [
        "",
        "## Part 2 — sample splits (midpoint fill; `splits.csv` also carries the crossed fill)",
        "",
        "| series | split | n | mean | t | Sharpe [95 %] | P&L share % |",
        "|---|---|---|---|---|---|---|",
    ]
    for _, r_ in splits.iterrows():
        ci = (
            f"[{r_['Sharpe_mid_lo']:+.2f}, {r_['Sharpe_mid_hi']:+.2f}]"
            if np.isfinite(r_["Sharpe_mid_lo"])
            else ""
        )
        S.append(
            f"| {r_['series']} | {r_['split']} | {int(r_['n'])} | {r_['mean_mid']:+.3f} | {fm(r_['t_mid'], '+.2f')} | {fm(r_['Sharpe_mid'], '.2f')} {ci} | {r_['share_of_total_pnl_mid_pct']:.0f} |"
        )
    S += [
        "",
        "Sign split of the settlement return R by the signal (HC0 t of the difference):",
        "",
        "| forecast | mean R, s ≤ 0 | mean R, s > 0 | difference | t | n sell / n buy |",
        "|---|---|---|---|---|---|",
    ]
    for _, r_ in signsplit.iterrows():
        S.append(
            f"| {r_['forecast']} | {r_['mean_R_when_s_le_0']:+.3f} | {r_['mean_R_when_s_gt_0']:+.3f} | {r_['diff']:.3f} | {r_['t_HC0']:.2f} | {int(r_['n_sell'])} / {int(r_['n_buy'])} |"
        )
    S += ["", "## Gates"]
    S += [
        f"- {g['gate']}: {g['value']:.3g} ≤ {g['bound']:.3g} on n = {g['n']} — {'PASS' if g['ok'] else 'FAIL'}"
        for g in gates
        if not g["gate"].startswith(("target shared", "q = master", "ret_"))
    ]
    n_shared = sum(
        1
        for g in gates
        if g["gate"].startswith(("target shared", "q = master", "ret_"))
    )
    S.append(
        f"- per-forecast identity gates vs the master table (target shared, q, ret_mid, ret_crossed): {n_shared} checked, all pass"
    )
    (OUT / "SUMMARY.md").write_text("\n".join(S) + "\n", encoding="utf-8")
    print(
        scores.loc[
            COMPARE,
            [
                "QLIKE",
                "ES_at_threshold",
                "EL",
                "hit_var_pct",
                "hit_pay_pct",
                "long_on_top20",
                "Sharpe_mid",
                "Sharpe_crossed",
            ],
        ]
        .round(4)
        .to_string()
    )
    print(ranks.round(3).to_string(index=False))
    print(
        splits[splits["series"] == "headline sign(s)"][
            [
                "split",
                "n",
                "mean_mid",
                "t_mid",
                "Sharpe_mid",
                "Sharpe_mid_lo",
                "Sharpe_mid_hi",
                "share_of_total_pnl_mid_pct",
            ]
        ]
        .round(3)
        .to_string(index=False)
    )
    print(signsplit.round(3).to_string(index=False))
    print("gates:", sum(g["ok"] for g in gates), "of", len(gates), "pass")
    print("wrote", OUT, "and", TEX)


if __name__ == "__main__":
    main()
