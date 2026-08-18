"""Time-till-close variance: is SUM-of-per-bar (A) the right construction?

At every 30-min bar t of a 0DTE session the trade needs RV(t, 16:00) --
the variance still to come. Two ways to produce it:

  A. SUM the per-bar one-step model forecasts over the remaining bars.
     This is what the repo does everywhere (pb_rem = reverse cumsum of
     pb). Note it is *not* an F_t-measurable object: bar k's forecast is
     made at bar k with F_{t+kD}, so A quietly carries information the
     trader does not have at t. It is the incumbent, and it is a
     generous incumbent.
  B. DIRECT: at t, regress log RV(t,T) on causal features available at
     t -- the model's own remaining sum, the market's model-free strip
     C_t (an implied time-till-close variance), the variance already
     realized today, and time-of-day.

The research spec proposes the information set
RV(t+kD, t+(k+1)D | F_t) = f(X_t, k, time-of-day, IV(t,T)); this script
tests the two pieces of that claim that the data on disk can decide:
does DIRECT beat SUM, and does the market strip add over the model?

CLOCK DEFECT (found while building this, and the reason for two panels)
----------------------------------------------------------------------
results/spxw_pnl/everybar_mtm_trades.parquet joins the option strip to
the model columns with merge_asof on `t` treating BOTH as UTC. The
strip stamps really are UTC; the yhat_*.parquet stamps are naive ET
wall clock (proved two ways: the RV profile peaks at stamp 10:00 and
collapses after stamp 16:30 -- the 09:30 open bar and the post-close
bar, i.e. stamps label the bar END in ET -- and that profile does not
move across DST, which a genuine UTC stamp would). The join therefore
attaches each option snapshot to an RV bar 4h (EDT) / 5h (EST) later:
the file's "15:30 ET" per-bar variance is really an overnight bar. The
tell is in the file itself -- its rv_rem/C ratio falls 28x across the
session (0.169 -> 0.006) instead of tracking remaining time.

So two panels are scored:
  fixed -- rebuilt here from yhat_a0/yhat_blk2 keyed on ET wall clock,
           end-labelled bars: entry s gets remaining = stamps > s up to
           16:00, realized-so-far = stamps <= s from the 09:30 open.
  asis  -- the existing everybar columns with the repo's inclusive
           reverse-cumsum idiom (spxw_atm_straddle_bars.py). Control
           only. Its rv_sofar (inclusive cumsum, per spec) also
           contains the current bar, so it is the one series here that
           is not strictly causal; the fixed panel's is.

A_blk2/A_a0 and every B variant built on pb_rem inherit A's peeking,
because pb_rem is itself a sum of not-yet-made forecasts. The B_ft_*
variants replace it with pb_next, the one-step forecast standing at t,
and are therefore strictly F_t-measurable (log C and log rv_sofar
already are). They are the honest challengers.

Causality everywhere: expanding fits on strictly earlier days, min 63
days, applied to the next day (identical to _expanding_ols_forecast in
spxw_options_expression.py when a day contributes one row); every
exp() forecast then gets the same expanding QLIKE-optimal
multiplicative rescale, computed in the same grouping as its fit
(per-hour models per hour, pooled models pooled). Scored by QLIKE of
remaining variance; DM t on paired daily loss differences vs A_blk2
(negative t = the challenger is better).
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "spxw_pnl")
BURN = 63
DAILY_0DTE = pd.Timestamp("2022-05-16")
GRID = [
    "10:00",
    "10:30",
    "11:00",
    "11:30",
    "12:00",
    "12:30",
    "13:00",
    "13:30",
    "14:00",
    "14:30",
    "15:00",
    "15:30",
    "16:00",
]
ENTRY = GRID[:-1]
FLOOR = 1e-18
# each model paired with the model nested inside it (same fit, added
# regressors dropped) so the increment gets its own DM test
NESTED = {
    "B_joint": "B_direct_blk2",
    "B_joint_realized": "B_joint",
    "B_ft_joint": "B_ft_blk2",
    "P_joint": "P_direct_blk2",
    "P_joint_realized": "P_joint",
    "P_ft_joint": "P_ft_blk2",
}


def _ql(f: np.ndarray, y: np.ndarray) -> np.ndarray:
    f = np.maximum(f, FLOOR)
    y = np.maximum(y, FLOOR)
    return y / f - np.log(y / f) - 1.0


def _dm(d: np.ndarray) -> float:
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    if d.size < 3 or float(d.std(ddof=1)) == 0.0:
        return float("nan")
    return float(d.mean() / (d.std(ddof=1) / np.sqrt(d.size)))


def _blocks(day_code: np.ndarray) -> list[tuple[int, int]]:
    """[start, stop) row ranges of each day; day_code must be nondecreasing."""
    cut = np.flatnonzero(np.diff(day_code)) + 1
    edge = np.concatenate([[0], cut, [day_code.size]])
    return [(int(a), int(b)) for a, b in zip(edge[:-1], edge[1:])]


def _expanding_ols(y: np.ndarray, X: np.ndarray, day_code: np.ndarray) -> np.ndarray:
    """Day-blocked expanding OLS forecast: rows of day d are predicted from a
    fit on all rows of days < d, once BURN days are in hand. With one row per
    day this is exactly spxw_options_expression._expanding_ols_forecast."""
    n = X.shape[0]
    Z = np.column_stack([np.ones(n), X])
    k = Z.shape[1]
    out = np.full(n, np.nan)
    gram = np.zeros((k, k))
    rhs = np.zeros(k)
    seen = 0
    for a, b in _blocks(day_code):
        if seen >= BURN:
            beta, *_ = np.linalg.lstsq(gram, rhs, rcond=None)
            out[a:b] = Z[a:b] @ beta
        gram += Z[a:b].T @ Z[a:b]
        rhs += Z[a:b].T @ y[a:b]
        seen += 1
    return out


def _expanding_coef_mean(
    y: np.ndarray, X: np.ndarray, day_code: np.ndarray
) -> np.ndarray:
    """Mean over the causal path of the expanding coefficient vector."""
    n = X.shape[0]
    Z = np.column_stack([np.ones(n), X])
    k = Z.shape[1]
    gram = np.zeros((k, k))
    rhs = np.zeros(k)
    seen = 0
    acc: list[np.ndarray] = []
    for a, b in _blocks(day_code):
        if seen >= BURN:
            beta, *_ = np.linalg.lstsq(gram, rhs, rcond=None)
            acc.append(beta)
        gram += Z[a:b].T @ Z[a:b]
        rhs += Z[a:b].T @ y[a:b]
        seen += 1
    if not acc:
        return np.full(k, np.nan)
    return np.asarray(acc, float).mean(0)


def _causal_scale(y: np.ndarray, f: np.ndarray, day_code: np.ndarray) -> np.ndarray:
    """QLIKE-optimal multiplicative back-transform, expanding over strictly
    earlier days (min BURN). One row per day == expanding().mean().shift(1)."""
    r = y / np.maximum(f, FLOOR)
    out = np.full(y.size, np.nan)
    tot = 0.0
    cnt = 0
    seen = 0
    for a, b in _blocks(day_code):
        if seen >= BURN and cnt > 0:
            out[a:b] = (tot / cnt) * f[a:b]
        seg = r[a:b]
        ok = np.isfinite(seg)
        tot += float(seg[ok].sum())
        cnt += int(ok.sum())
        seen += 1
    return out


def _panel_fixed(eb: pd.DataFrame) -> pd.DataFrame:
    """Rebuild the time-till-close panel with the ET wall-clock alignment."""
    a0 = pd.read_parquet(os.path.join(OUT, "yhat_a0.parquet"))
    b2 = pd.read_parquet(os.path.join(OUT, "yhat_blk2.parquet"))
    yh = a0.merge(b2, on="t", suffixes=("_a", "_b"))
    yh["t"] = pd.to_datetime(yh["t"], utc=True).dt.tz_localize(None)
    yh["pa"] = yh["yhat_a"].to_numpy(float) ** 2 * yh["baseline_a"].to_numpy(float)
    yh["pb"] = yh["yhat_b"].to_numpy(float) ** 2 * yh["baseline_b"].to_numpy(float)
    yh["rv"] = yh["rv_raw_a"].to_numpy(float)
    yh["day"] = yh["t"].dt.normalize()
    yh["hm"] = yh["t"].dt.strftime("%H:%M")
    yh = yh[yh["hm"].isin(GRID) & yh["day"].isin(eb["day"].unique())]

    wide: dict[str, pd.DataFrame] = {}
    for col in ("rv", "pa", "pb"):
        w = yh.pivot_table(index="day", columns="hm", values=col, aggfunc="first")
        wide[col] = w.reindex(columns=GRID)
    keep = wide["rv"].notna().all(1) & wide["pb"].notna().all(1)
    keep &= wide["pa"].notna().all(1)
    for col in wide:
        wide[col] = wide[col].loc[keep]

    cw = eb.pivot_table(index="day", columns="hm", values="C", aggfunc="first")
    cw = cw.reindex(columns=ENTRY)
    cw = cw.loc[cw.notna().all(1)]
    days = wide["rv"].index.intersection(cw.index)
    cw = cw.loc[days]
    for col in wide:
        wide[col] = wide[col].loc[days]

    rows: list[pd.DataFrame] = []
    rvm = wide["rv"].to_numpy(float)
    sofar = np.cumsum(rvm, axis=1)
    rem = {c: (v.to_numpy(float)[:, ::-1].cumsum(1)[:, ::-1]) for c, v in wide.items()}
    for i, h in enumerate(ENTRY):
        rows.append(
            pd.DataFrame(
                {
                    "day": days,
                    "hm": h,
                    "rv_rem": rem["rv"][:, i + 1],
                    "pa_rem": rem["pa"][:, i + 1],
                    "pb_rem": rem["pb"][:, i + 1],
                    "pb_next": wide["pb"].to_numpy(float)[:, i + 1],
                    "rv_sofar": sofar[:, i],
                    "C": cw[h].to_numpy(float),
                }
            )
        )
    out = pd.concat(rows, ignore_index=True)
    return out.sort_values(["day", "hm"]).reset_index(drop=True)


def _panel_asis(eb: pd.DataFrame) -> pd.DataFrame:
    """The repo's existing construction: inclusive reverse cumsum of the
    everybar columns (spxw_atm_straddle_bars.py idiom), 12-bar days only."""
    d = eb.sort_values(["day", "hm"]).copy()
    d = d[d.groupby("day")["hm"].transform("size") == len(ENTRY)]
    g = d.groupby("day", sort=False)
    for src, dst in (("rv", "rv_rem"), ("pa", "pa_rem"), ("pb", "pb_rem")):
        d[dst] = g[src].transform(lambda s: s.iloc[::-1].cumsum().iloc[::-1])
    d["rv_sofar"] = g["rv"].cumsum()  # inclusive, per spec (see docstring)
    d["pb_next"] = d["pb"]  # the row's own per-bar forecast
    cols = ["day", "hm", "rv_rem", "pa_rem", "pb_rem", "pb_next", "rv_sofar", "C"]
    return d[cols].reset_index(drop=True)


def _fit_panel(p: pd.DataFrame) -> pd.DataFrame:
    """Attach every candidate time-till-close forecast to the panel."""
    p = p.copy()
    ok = np.ones(len(p), bool)
    for c in ("rv_rem", "pa_rem", "pb_rem", "pb_next", "rv_sofar", "C"):
        v = p[c].to_numpy(float)
        ok &= np.isfinite(v) & (v > 0)
    p = p.loc[ok].sort_values(["day", "hm"]).reset_index(drop=True)

    p["ly"] = np.log(p["rv_rem"].to_numpy(float))
    p["lb"] = np.log(p["pb_rem"].to_numpy(float))
    p["lc"] = np.log(p["C"].to_numpy(float))
    p["ls"] = np.log(p["rv_sofar"].to_numpy(float))
    p["ln"] = np.log(p["pb_next"].to_numpy(float))

    # ---- per-hour models -------------------------------------------------
    specs = {
        "B_direct_blk2": ["lb"],
        "B_direct_iv": ["lc"],
        "B_joint": ["lb", "lc"],
        "B_joint_realized": ["lb", "lc", "ls"],
        # strictly F_t-measurable: pb_rem sums forecasts the trader cannot
        # see yet, pb_next is the one-step forecast standing at t.
        "B_ft_blk2": ["ln"],
        "B_ft_joint": ["ln", "lc", "ls"],
    }
    for name in ["A_blk2", "A_a0", *specs]:
        p[name] = np.nan
    for h, idx in p.groupby("hm").groups.items():
        s = p.loc[idx].sort_values("day")
        pos = s.index.to_numpy()
        dc = np.arange(len(s))  # one row per day
        y = s["ly"].to_numpy(float)
        rv = s["rv_rem"].to_numpy(float)
        p.loc[pos, "A_blk2"] = _causal_scale(rv, s["pb_rem"].to_numpy(float), dc)
        p.loc[pos, "A_a0"] = _causal_scale(rv, s["pa_rem"].to_numpy(float), dc)
        for name, cols in specs.items():
            X = s[cols].to_numpy(float)
            f = np.exp(_expanding_ols(y, X, dc))
            p.loc[pos, name] = _causal_scale(rv, f, dc)

    # ---- pooled models: one fit on all bars, 11 hour dummies -------------
    dummies = np.column_stack(
        [(p["hm"].to_numpy() == h).astype(float) for h in ENTRY[1:]]
    )
    day_code = pd.factorize(p["day"], sort=True)[0]
    if np.any(np.diff(day_code) < 0):
        raise SystemExit("panel rows must be in day order")
    y = p["ly"].to_numpy(float)
    rv = p["rv_rem"].to_numpy(float)
    for name, cols in specs.items():
        X = np.column_stack([p[cols].to_numpy(float), dummies])
        f = np.exp(_expanding_ols(y, X, day_code))
        p["P" + name[1:]] = _causal_scale(rv, f, day_code)
    return p


def _coef_table(p: pd.DataFrame, panel: str) -> pd.DataFrame:
    """Descriptive coefficients of the joint models, per hour and pooled."""
    names = {
        "lb": "log_pb_rem",
        "lc": "log_C",
        "ls": "log_rv_sofar",
        "ln": "log_pb_next",
    }
    rows: list[dict[str, object]] = []
    specs = {
        "B_joint": ["lb", "lc"],
        "B_joint_realized": ["lb", "lc", "ls"],
        "B_ft_joint": ["ln", "lc", "ls"],
    }
    for era, msk in _eras(p).items():
        s0 = p.loc[msk]
        for model, cols in specs.items():
            for h in [*ENTRY, "pooled"]:
                s = s0 if h == "pooled" else s0.loc[s0["hm"] == h]
                if len(s) < BURN:
                    continue
                y = s["ly"].to_numpy(float)
                X = s[cols].to_numpy(float)
                Z = np.column_stack([np.ones(len(s)), X])
                beta, *_ = np.linalg.lstsq(Z, y, rcond=None)
                dc = (
                    pd.factorize(s["day"], sort=True)[0]
                    if h == "pooled"
                    else np.arange(len(s))
                )
                mb = _expanding_coef_mean(y, X, dc)
                r: dict[str, object] = {
                    "panel": panel,
                    "era": era,
                    "hour": h,
                    "model": model,
                    "n": int(len(s)),
                    "full_const": float(beta[0]),
                    "mean_causal_const": float(mb[0]),
                }
                for j, c in enumerate(cols):
                    r[f"full_{names[c]}"] = float(beta[j + 1])
                    r[f"mean_causal_{names[c]}"] = float(mb[j + 1])
                rows.append(r)
    return pd.DataFrame(rows)


def _eras(p: pd.DataFrame) -> dict[str, np.ndarray]:
    d = pd.to_datetime(p["day"]).to_numpy()
    return {"all": np.ones(len(p), bool), "daily_0dte": d >= DAILY_0DTE.to_numpy()}


def _score(p: pd.DataFrame, panel: str, cands: list[str]) -> tuple[pd.DataFrame, ...]:
    common = np.ones(len(p), bool)
    for c in cands:
        common &= np.isfinite(p[c].to_numpy(float))
    p = p.loc[common].copy()
    rv = p["rv_rem"].to_numpy(float)
    loss = {c: _ql(p[c].to_numpy(float), rv) for c in cands}

    by_hour: list[dict[str, object]] = []
    pooled: list[dict[str, object]] = []
    for era, msk in _eras(p).items():
        s = p.loc[msk]
        for h in ENTRY:
            m = (s["hm"] == h).to_numpy()
            sub = {c: loss[c][msk][m] for c in cands}
            for c in cands:
                twin = "B" + c[1:] if c.startswith("P_") else None
                by_hour.append(
                    {
                        "panel": panel,
                        "era": era,
                        "hour": h,
                        "forecast": c,
                        "n_days": int(m.sum()),
                        "qlike": float(sub[c].mean()),
                        "dm_t_vs_A_blk2": _dm(sub[c] - sub["A_blk2"]),
                        "nested_in": NESTED.get(c, ""),
                        "dm_t_vs_nested": (
                            _dm(sub[c] - sub[NESTED[c]])
                            if c in NESTED
                            else float("nan")
                        ),
                        "dm_t_pooled_vs_perhour": (
                            _dm(sub[c] - sub[twin]) if twin in sub else float("nan")
                        ),
                    }
                )
        # pooled: average each day's loss across its bars, then DM over days
        lf = pd.DataFrame({c: loss[c][msk] for c in cands})
        lf["day"] = s["day"].to_numpy()
        daily = lf.groupby("day").mean()
        for c in cands:
            twin = "B" + c[1:] if c.startswith("P_") else None
            pooled.append(
                {
                    "panel": panel,
                    "era": era,
                    "forecast": c,
                    "n_bars": int(msk.sum()),
                    "n_days": int(len(daily)),
                    "qlike": float(loss[c][msk].mean()),
                    "dm_t_vs_A_blk2": _dm(
                        daily[c].to_numpy() - daily["A_blk2"].to_numpy()
                    ),
                    "nested_in": NESTED.get(c, ""),
                    "dm_t_vs_nested": (
                        _dm(daily[c].to_numpy() - daily[NESTED[c]].to_numpy())
                        if c in NESTED
                        else float("nan")
                    ),
                    "dm_t_pooled_vs_perhour": (
                        _dm(daily[c].to_numpy() - daily[twin].to_numpy())
                        if twin in daily.columns
                        else float("nan")
                    ),
                }
            )
    return pd.DataFrame(by_hour), pd.DataFrame(pooled)


def _alignment_diag(eb: pd.DataFrame, fixed: pd.DataFrame) -> pd.DataFrame:
    """Evidence for the clock defect: remaining-variance / strip ratio by bar."""
    rows: list[dict[str, object]] = []
    asis = _panel_asis(eb)
    for panel, p in (("asis", asis), ("fixed", fixed)):
        for h in ENTRY:
            s = p.loc[p["hm"] == h]
            r = (s["rv_rem"] / s["C"]).to_numpy(float)
            rows.append(
                {
                    "panel": panel,
                    "hour": h,
                    "n": int(len(s)),
                    "med_rv_rem": float(s["rv_rem"].median()),
                    "med_C": float(s["C"].median()),
                    "med_ratio": float(np.nanmedian(r)),
                    "corr_log": float(
                        np.corrcoef(
                            np.log(s["rv_rem"].to_numpy(float)),
                            np.log(s["C"].to_numpy(float)),
                        )[0, 1]
                    ),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    eb = pd.read_parquet(os.path.join(OUT, "everybar_mtm_trades.parquet"))
    eb["day"] = eb["et"].dt.normalize().dt.tz_localize(None)
    eb["hm"] = eb["et"].dt.strftime("%H:%M")
    eb = eb[eb["hm"].isin(ENTRY)].sort_values(["day", "hm"]).reset_index(drop=True)

    panels = {"fixed": _panel_fixed(eb), "asis": _panel_asis(eb)}
    diag = _alignment_diag(eb, panels["fixed"])
    diag.to_csv(os.path.join(OUT, "ttc_structure_alignment.csv"), index=False)
    print("clock check -- remaining variance vs the strip, by entry bar", flush=True)
    print(
        diag.pivot(index="hour", columns="panel", values="med_ratio").to_string(),
        flush=True,
    )

    cands = [
        "A_blk2",
        "A_a0",
        "B_direct_blk2",
        "B_direct_iv",
        "B_joint",
        "B_joint_realized",
        "B_ft_blk2",
        "B_ft_joint",
        "P_direct_blk2",
        "P_direct_iv",
        "P_joint",
        "P_joint_realized",
        "P_ft_blk2",
        "P_ft_joint",
    ]
    hour_t: list[pd.DataFrame] = []
    pool_t: list[pd.DataFrame] = []
    coef_t: list[pd.DataFrame] = []
    for panel, raw in panels.items():
        p = _fit_panel(raw)
        print(f"\npanel={panel}: bars={len(p)} days={p['day'].nunique()}", flush=True)
        h, q = _score(p, panel, cands)
        hour_t.append(h)
        pool_t.append(q)
        coef_t.append(_coef_table(p, panel))

    hour = pd.concat(hour_t, ignore_index=True)
    pool = pd.concat(pool_t, ignore_index=True)
    coef = pd.concat(coef_t, ignore_index=True)
    hour.to_csv(os.path.join(OUT, "ttc_structure_byhour.csv"), index=False)
    pool.to_csv(os.path.join(OUT, "ttc_structure_pooled.csv"), index=False)
    coef.to_csv(os.path.join(OUT, "ttc_structure_coefs.csv"), index=False)

    for era in ("all", "daily_0dte"):
        s = hour[(hour["panel"] == "fixed") & (hour["era"] == era)]
        print(f"\n=== per entry hour, panel=fixed, era={era} (QLIKE) ===", flush=True)
        print(
            s.pivot(index="hour", columns="forecast", values="qlike")[cands].to_string(
                float_format=lambda v: f"{v:.4f}"
            ),
            flush=True,
        )
        print(f"--- DM t vs A_blk2 (negative = better) era={era} ---", flush=True)
        print(
            s.pivot(index="hour", columns="forecast", values="dm_t_vs_A_blk2")[
                cands[1:]
            ]
            .to_string(float_format=lambda v: f"{v:+.2f}")
            .replace("nan", "  ."),
            flush=True,
        )
        print(
            f"--- DM t vs the nested model (the added-regressor increment) "
            f"era={era} ---",
            flush=True,
        )
        print(
            s[s["forecast"].isin(NESTED)]
            .pivot(index="hour", columns="forecast", values="dm_t_vs_nested")
            .to_string(float_format=lambda v: f"{v:+.2f}"),
            flush=True,
        )
        print(f"--- DM t pooled vs its per-hour twin era={era} ---", flush=True)
        print(
            s[s["forecast"].str.startswith("P_")]
            .pivot(index="hour", columns="forecast", values="dm_t_pooled_vs_perhour")
            .to_string(float_format=lambda v: f"{v:+.2f}"),
            flush=True,
        )

    print("\n=== pooled over hours ===", flush=True)
    print(pool.to_string(index=False, float_format=lambda v: f"{v:.4f}"), flush=True)

    print("\n=== joint-model coefficients, panel=fixed ===", flush=True)
    c = coef[(coef["panel"] == "fixed")]
    print(c.to_string(index=False, float_format=lambda v: f"{v:+.3f}"), flush=True)
    print("\nwrote ttc_structure_{alignment,byhour,pooled,coefs}.csv", flush=True)


if __name__ == "__main__":
    main()
