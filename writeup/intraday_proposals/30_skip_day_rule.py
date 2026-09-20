"""30 - which days should the seller sit out?

The book is the one proposals 26/27 and the standalone PDF score: every
session sells the nearest-OTM SPX 0DTE straddle at 11:00 ET, delta-hedges
on the vendor spot every 30 minutes, and buys the straddle back at 15:30
(the exit book) - with hold-to-cash-settlement carried alongside for
reference.  Nothing here is ever long the straddle: a rule either sells
the straddle that day or sits out, so a sat-out day contributes exactly
zero and the honest comparison against always-sell is the mean per
CALENDAR day, not per traded day.

Four candidate skip rules, all decided at 11:00 from information the
11:00 stamp already carries:

  1  forecast skip     sit out when the window-matched signal is positive,
                       s_matched(11:00) = rv_hat - IV_hr^2 h_rem w_slice > 0
                       (section 5b of notebooks/_write_0dte_intraday_T_nb.py),
                       one rule per forecast in asl.MODEL_ORDER;
  2  implied-level     sit out when the 11:00 implied variance slice sits in
                       its lowest expanding LAGGED quantile bucket (cheap
                       premium), thresholds {10%, 20%, 30%};
  3  realized regime   sit out when the previous session's realized variance
                       over the 11:00 implied slice (a VRP proxy) is above its
                       expanding LAGGED quantile, thresholds {70%, 80%, 90%};
  4  calendar          the skips already known to fail: FOMC statement days
                       and month-end sessions (asl.fomc_and_monthend).

The gate is the repo's standard, and no rule is called adopted without it:
(a) a purged, embargoed walk-forward - each rule's threshold is chosen on an
expanding window that stops EMBARGO_DAYS before the scored day, and only the
out-of-sample day is scored; (b) a placebo - N_PLACEBO random skip masks with
the SAME sat-out count and the SAME run-length multiset (block-shuffled), so
the null is "a skip rule that sits out as often, and in runs as long, but for
no reason".  A rule is adopted only if its out-of-sample calendar-day Sharpe
beats always-sell AND its placebo p-value is below PLACEBO_ALPHA.

The script reproduces the published book numbers before it computes anything
new, and reproduces the 11:00 signal against the notebook's own persisted buy
share.  Nothing runs until that gate passes.

Run:  python writeup/intraday_proposals/30_skip_day_rule.py
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "notebooks"))
import atm_straddle_lib as asl  # noqa: E402

REPO = asl.find_repo(Path(__file__).resolve().parent)
HOLD = REPO / "results" / "atm_straddle_intraday_holdclose"
OUT = HOLD / "proposals" / "30"

ENTRY = "11:00"
CLOSE = "15:30"
ANN = float(np.sqrt(asl.PERIODS_PER_YEAR))

# The expanding-window warm-up this repo already uses for a causal
# expanding statistic: the diurnal remaining-share profile of section 5b
# (notebooks/_write_0dte_intraday_T_nb.py, `expanding(min_periods=63)`).
# Every expanding lagged quantile here, and the walk-forward's minimum
# training window, use the same number.
WARMUP_SESSIONS = 63

# The gate's embargo: days within EMBARGO_DAYS of the scored day are purged
# from the training window, so the threshold cannot be chosen on a day that
# shares a week with the day it is applied to.
EMBARGO_DAYS = 5

# The placebo: block-shuffled skip masks, and the level the gate reads.
N_PLACEBO = 2000
PLACEBO_SEED = 0
PLACEBO_ALPHA = 0.05

# Rule thresholds.  The implied-level rule sits out CHEAP premium (the low
# tail of the implied slice); the realized-regime rule sits out a HIGH
# realized-over-implied ratio (the high tail).
IV_QUANTILE_GRID = (0.10, 0.20, 0.30)
VRP_QUANTILE_GRID = (0.70, 0.80, 0.90)

# Published reference numbers, reproduced before anything new is computed.
# Books: writeup/make_dh_causal_standalone_tex.py (mid and crossed-at-the-
# model-mark) and results/.../proposals/27/q1_exit_quote.csv (the fully
# crossed exit at the QUOTED 15:30 ask).  Signal: the 11:00 buy share of
# results/.../rule_by_entry_hhmm_hitrate.csv, which is the share of days
# with s_matched > 0 at that clock.
REFERENCE_BOOKS = (
    ("n whole sample", 865.0, 0),
    ("hold Sharpe_ann mid", 4.200674, 6),
    ("hold Sharpe_ann crossed", 3.582391, 6),
    ("exit-15:30 model mark Sharpe_ann mid", 4.695674, 6),
    ("exit-15:30 quoted ask crossed Sharpe_ann whole", 2.252546754992813, 12),
    ("exit-15:30 quoted ask crossed Sharpe_ann daily era", 2.5131322639023277, 12),
)


# ------------------------------------------------------------- helpers ----
def _load_module(path: Path, name: str):
    """Import a script by path without touching it (the repo's read-only import)."""
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None, path
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def sharpe(x: np.ndarray) -> float:
    """Annualized mean/sd of a daily series, one expiration day one return."""
    v = np.asarray(x, dtype=float)
    v = v[np.isfinite(v)]
    if v.size < 2:
        return float("nan")
    sd = float(v.std(ddof=1))
    if not (sd > 0):
        return float("nan")
    return float(v.mean()) / sd * ANN


def maxdd(x: np.ndarray) -> float:
    """Worst peak-to-trough of the cumulative SUM path, peak seeded at 0."""
    v = np.asarray(x, dtype=float)
    v = v[np.isfinite(v)]
    if v.size < 1:
        return float("nan")
    path = np.cumsum(v)
    peak = np.maximum.accumulate(np.concatenate(([0.0], path)))[1:]
    return float((path - peak).min())


def expanding_lagged_quantile(s: pd.Series, p: float, min_periods: int) -> pd.Series:
    """Quantile p of the STRICTLY PRIOR days, expanding, after a warm-up.

    Same shape as the section-5b remaining-share profile: an expanding
    statistic with a minimum count, shifted one day, so the value used on
    day d is a function of days before d only.  Warm-up days are NaN.
    """
    return s.expanding(min_periods=min_periods).quantile(p).shift(1)


def expanding_sharpe(x: np.ndarray) -> np.ndarray:
    """out[k] = annualized Sharpe of x[:k], for every k (NaN where undefined).

    One pass, so the walk-forward can price every candidate on every
    expanding training window without an O(n^2) re-scan.
    """
    v = np.asarray(x, dtype=float)
    n = v.size
    cs = np.concatenate(([0.0], np.cumsum(v)))
    cs2 = np.concatenate(([0.0], np.cumsum(v * v)))
    k = np.arange(n + 1, dtype=float)
    out = np.full(n + 1, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(k > 0, cs / np.where(k > 0, k, 1.0), np.nan)
        var = np.where(
            k > 1, (cs2 - k * mean**2) / np.where(k > 1, k - 1.0, 1.0), np.nan
        )
        sd = np.sqrt(np.where(np.isfinite(var) & (var > 0), var, np.nan))
        out = mean / sd * ANN
    return out


def show(df: pd.DataFrame, title: str) -> None:
    print(f"\n{title}")
    with pd.option_context("display.width", 250, "display.max_columns", 60):
        print(df.to_string(index=False))


# ---------------------------------------------------------------- gate ----
def gate(p27, pkg: pd.DataFrame):
    """Rebuild the three books and assert every published number first."""
    std = p27._standalone()
    r_hold, r_mark, r_hold_x, _ = std.hold_mark_1100(pkg)
    idx = r_hold.dropna().index

    # the fully crossed exit at the QUOTED 15:30 ask, rebuilt with proposal
    # 27's own construction (its helpers are imported, never edited)
    tape = p27.build_tape(pkg)
    sl = tape["sl"].reindex(idx)
    entry = tape["entry"].reindex(idx).to_numpy(float)
    bid_entry = tape["bid_entry"].reindex(idx).to_numpy(float)
    rows = tape["dates"].get_indexer(idx)
    Sg, dlt, dS_f = (tape[k][rows] for k in ("Sg", "dlt", "dS_f"))
    npos_f = dlt.copy()
    npos_f[:, tape["j15"] :] = 0.0
    hedge_f, _ = p27.hedge_and_cost(npos_f, dS_f, Sg, 0.0)
    q = p27.quotes_1530(p27.CHAIN)
    exp = pd.DatetimeIndex(sl["expiration"])
    assert (exp == pd.DatetimeIndex(idx)).all(), (
        "the 11:00 row's expiration is not its own date"
    )
    qc = p27.leg_quotes(q, exp, sl["K_c"].to_numpy(float), "C")
    qp = p27.leg_quotes(q, exp, sl["K_p"].to_numpy(float), "P")
    no_row = qc["bid"].isna().to_numpy() | qp["bid"].isna().to_numpy()
    sent_c = (qc["bid"].to_numpy() == 0) & (qc["ask"].to_numpy() == 0)
    sent_p = (qp["bid"].to_numpy() == 0) & (qp["ask"].to_numpy() == 0)
    live = ~(no_row | sent_c | sent_p)
    ask_q = np.where(live, qc["ask"].to_numpy() + qp["ask"].to_numpy(), np.nan)
    r_exit_x_q = pd.Series((-(ask_q - bid_entry) + hedge_f) / entry, index=idx)
    print(
        f"15:30 quotes on the two 11:00 strikes: {int(live.sum())} of {len(idx)} days live"
    )

    era = r_exit_x_q.index >= p27.DAILY_0DTE
    got = {
        "n whole sample": float(idx.size),
        "hold Sharpe_ann mid": sharpe(r_hold.reindex(idx).to_numpy()),
        "hold Sharpe_ann crossed": sharpe(r_hold_x.reindex(idx).to_numpy()),
        "exit-15:30 model mark Sharpe_ann mid": sharpe(r_mark.reindex(idx).to_numpy()),
        "exit-15:30 quoted ask crossed Sharpe_ann whole": sharpe(r_exit_x_q.to_numpy()),
        "exit-15:30 quoted ask crossed Sharpe_ann daily era": sharpe(
            r_exit_x_q[era].to_numpy()
        ),
    }
    print("\nGATE - published book numbers")
    for label, want, dec in REFERENCE_BOOKS:
        have = got[label]
        assert abs(round(have, dec) - want) < 1e-9, (label, have, want)
        print(f"  {label:<52s} {have:>18.12f}  reference {want:.{dec}f}  OK")
    ref = pd.read_csv(OUT.parent / "27" / "q1_exit_quote.csv")
    ref = ref.loc[
        ref["book"] == "exit 15:30 at the quoted ask, entry at the bid (crossed)"
    ]
    for smp, ser in (("whole", r_exit_x_q), ("daily_era", r_exit_x_q[era])):
        row = ref.loc[ref["sample"] == smp].iloc[0]
        assert int(row["n"]) == int(ser.dropna().size), (
            smp,
            row["n"],
            ser.dropna().size,
        )
        assert abs(float(row["mean"]) - float(ser.mean())) < 1e-12, smp
        assert abs(float(row["sd"]) - float(ser.std(ddof=1))) < 1e-12, smp
    print("  proposals/27 q1_exit_quote.csv crossed-quoted exit matched to 1e-12")
    return {
        "exit_crossed_quoted": r_exit_x_q,
        "exit_mid_model": r_mark.reindex(idx),
        "hold_settle": r_hold.reindex(idx),
    }, idx


# ------------------------------------------------------------- signals ----
def panel_profile(
    panel: pd.DataFrame, clocks: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Section 5b's remaining-share profile, and the per-session realized tape.

    Panel stamps are bar-end labelled, so stamp 10:30..16:00 is trade clock
    10:00..15:30.  pi[d, t] is that bar's realized variance over the day's
    remaining-to-close sum; w is the expanding mean of pi over the STRICTLY
    prior sessions, seeded from the panel's own history.
    """
    pf = panel.reset_index()
    pf = pf[pf["in_fit"].to_numpy(dtype=bool)].copy()
    clock = pd.to_datetime(pf["t"], utc=True).dt.tz_convert(
        "America/New_York"
    ) - pd.Timedelta(minutes=30)
    pf["pdate"] = clock.dt.normalize().dt.tz_localize(None)
    pf["phhmm"] = clock.dt.strftime("%H:%M")
    prof = pf.pivot_table(
        index="pdate", columns="phhmm", values="rv_raw", aggfunc="mean"
    )
    prof = prof.sort_index()
    pi = pd.DataFrame(index=prof.index, columns=clocks, dtype=float)
    for i, c in enumerate(clocks):
        rem = prof[clocks[i:]].sum(axis=1)
        pi[c] = prof[c] / rem.replace(0.0, np.nan)
    w = pi.expanding(min_periods=WARMUP_SESSIONS).mean().shift(1)
    return prof, w


def signals_1100(pkg: pd.DataFrame, idx: pd.DatetimeIndex) -> pd.DataFrame:
    """Every 11:00 decision variable, on the book's own days, causal at 11:00."""
    work0 = pkg.copy()
    work0["t"] = pd.to_datetime(work0["timestamp"], utc=True)
    work0["date"] = pd.to_datetime(work0["date"])
    clocks = sorted(work0["hhmm"].unique())
    n_rem = {c: len(clocks) - i for i, c in enumerate(clocks)}

    paths = asl.yhat_paths(REPO)
    panels = {
        tag: asl.load_yhat_panel(paths[tag]).set_index("t") for tag in asl.MODEL_ORDER
    }
    base = panels["blk2"]
    # every panel carries the same realized tape; the profile is fit once,
    # on the panel of record, exactly as the notebook does
    for tag, pan in panels.items():
        shared = base.index.intersection(pan.index)
        a = base.loc[shared, "rv_raw"].to_numpy(float)
        b = pan.loc[shared, "rv_raw"].to_numpy(float)
        ok = np.isfinite(a) & np.isfinite(b)
        assert np.allclose(a[ok], b[ok], rtol=0, atol=0), (
            f"{tag} carries a different realized tape"
        )
    prof, w_slice = panel_profile(base, clocks)

    out = pd.DataFrame(index=idx)
    for tag in asl.MODEL_ORDER:
        p = panels[tag].reset_index()[["t", "rv_hat", "in_fit"]].copy()
        p["t"] = pd.to_datetime(p["t"], utc=True) - pd.Timedelta(minutes=30)
        work = work0.merge(p, on="t", how="left").dropna(subset=["R", "rv_hat"])
        assert bool(work["in_fit"].all()), (
            f"{tag}: a joined trade bar is outside the fit mask"
        )
        mi = pd.MultiIndex.from_arrays([work["date"], work["hhmm"]])
        work["w_slice"] = w_slice.stack().reindex(mi).to_numpy()
        work["h_rem"] = work["hhmm"].map(n_rem).astype(float) * 0.5
        work["slice"] = (
            work["iv_hourly"].astype(float) ** 2 * work["h_rem"] * work["w_slice"]
        )
        work["s_matched"] = work["rv_hat"] - work["slice"]
        at = work.loc[work["hhmm"] == ENTRY].drop_duplicates("date").set_index("date")
        out[f"s_{tag}"] = at["s_matched"].reindex(idx)
        if tag == "blk2":
            out["slice"] = at["slice"].reindex(idx)

    # previous session's realized variance over the 11:00 implied slice
    day_rv = prof.sum(axis=1, min_count=len(clocks)).dropna()
    pos = day_rv.index.searchsorted(pd.DatetimeIndex(idx), side="left") - 1
    prev = np.where(pos >= 0, day_rv.to_numpy(float)[np.clip(pos, 0, None)], np.nan)
    out["rv_prev"] = prev
    out["vrp"] = out["rv_prev"] / out["slice"]

    # the notebook's own persisted 11:00 buy share is the signal's gate
    hit = pd.read_csv(HOLD / "rule_by_entry_hhmm_hitrate.csv")
    row = hit.loc[hit["hhmm"] == ENTRY].iloc[0]
    got = float((out["s_blk2"] > 0).mean())
    assert int(row["n"]) == len(idx), (row["n"], len(idx))
    assert abs(got - float(row["buy share"])) < 1e-12, (got, float(row["buy share"]))
    print(
        f"\nGATE - 11:00 signal: s_matched > 0 on {int((out['s_blk2'] > 0).sum())} of {len(idx)} "
        f"days, share {got:.15f}; rule_by_entry_hhmm_hitrate.csv buy share "
        f"{float(row['buy share']):.15f}  OK"
    )
    print("GATE PASSED\n")
    for tag in asl.MODEL_ORDER:
        n_na = int(out[f"s_{tag}"].isna().sum())
        if n_na:
            print(
                f"  {tag}: {n_na} of {len(idx)} days without a signal (they sell, the default)"
            )
    print(
        f"11:00 implied slice: median {float(out['slice'].median()):.3e}; "
        f"previous-session realized variance: median {float(out['rv_prev'].median()):.3e}; "
        f"VRP proxy median {float(out['vrp'].median()):.3f}"
    )
    return out


# --------------------------------------------------------------- rules ----
def build_rules(sig: pd.DataFrame, idx: pd.DatetimeIndex, p27) -> list[dict]:
    """Every candidate skip mask, in families.  True = sit out that day.

    A day with no decision variable (a warm-up day for an expanding
    quantile, or a model with no forecast row) sells: the baseline book is
    always-sell, so a rule only ever REMOVES a day it has a reason to
    remove.  Each family's first entry is the null, "never sit out", which
    is what the walk-forward must beat to pick anything else.
    """
    n = len(idx)
    never = np.zeros(n, dtype=bool)
    rules: list[dict] = [
        {
            "family": "baseline",
            "name": "always sell (no skip)",
            "skip": never,
            "warmup": 0,
        }
    ]

    for tag in asl.MODEL_ORDER:
        s = sig[f"s_{tag}"].to_numpy(float)
        rules.append(
            {
                "family": "forecast",
                "name": f"sit out when s_matched(11:00) > 0, {asl.YHAT_LABEL[tag]}",
                "skip": np.isfinite(s) & (s > 0),
                "warmup": int((~np.isfinite(s)).sum()),
            }
        )

    slice11 = sig["slice"]
    for p in IV_QUANTILE_GRID:
        thr = expanding_lagged_quantile(slice11, p, WARMUP_SESSIONS)
        m = (slice11 <= thr).fillna(False).to_numpy(bool)
        rules.append(
            {
                "family": "implied_level",
                "name": f"sit out when the 11:00 implied slice <= its expanding {p:.0%} quantile",
                "skip": m,
                "warmup": int(thr.isna().sum()),
            }
        )

    vrp = sig["vrp"]
    for p in VRP_QUANTILE_GRID:
        thr = expanding_lagged_quantile(vrp, p, WARMUP_SESSIONS)
        m = (vrp > thr).fillna(False).to_numpy(bool)
        rules.append(
            {
                "family": "realized_regime",
                "name": f"sit out when prior-session RV / 11:00 slice > its expanding {p:.0%} quantile",
                "skip": m,
                "warmup": int(thr.isna().sum()),
            }
        )

    m = _rule_mod_deck(p27)
    sessions = pd.DatetimeIndex(
        pd.to_datetime(pd.read_parquet(m.DECK / "daily_blk2.parquet").index)
    )
    flags = asl.fomc_and_monthend(pd.DatetimeIndex(idx), REPO, sessions=sessions)
    is_fomc = flags["is_fomc"].fillna(False).to_numpy(bool)
    is_me = flags["is_me"].to_numpy(bool)
    print(
        f"calendar flags on the book's {len(idx)} days: {int(is_fomc.sum())} FOMC statement days, "
        f"{int(is_me.sum())} month-end sessions; FOMC knowledge horizon "
        f"{pd.Timestamp(flags.attrs['fomc_known_until']).date()}"
    )
    rules.append(
        {
            "family": "calendar",
            "name": "sit out on FOMC statement days",
            "skip": is_fomc,
            "warmup": 0,
        }
    )
    rules.append(
        {
            "family": "calendar",
            "name": "sit out on month-end sessions",
            "skip": is_me,
            "warmup": 0,
        }
    )
    return rules


def _rule_mod_deck(p27):
    """The DH rule-table module, only for its deck path (read, never edited)."""
    return p27._standalone()._rule_mod()


def rule_stats(r: np.ndarray, skip: np.ndarray, dates: pd.DatetimeIndex) -> dict:
    """Calendar-day statistics of one book under one skip mask.

    A sat-out day stays in the series as a zero, so every rule is scored on
    the same calendar days and the mean is directly comparable with
    always-sell (the deck's convention, proposals/27 Q3).
    """
    ok = np.isfinite(r)
    cal = np.where(skip, 0.0, r)[ok]
    traded = r[ok & ~skip]
    sat = r[ok & skip]
    sd = float(cal.std(ddof=1)) if cal.size >= 2 else float("nan")
    mean = float(cal.mean()) if cal.size else float("nan")
    d = dates[ok]
    worst_i = int(np.argmin(cal)) if cal.size else -1
    return {
        "n_days": int(cal.size),
        "n_traded": int(traded.size),
        "n_satout": int(sat.size),
        "frac_satout": float(sat.size / cal.size) if cal.size else float("nan"),
        "mean_traded": float(traded.mean()) if traded.size else float("nan"),
        "mean_calendar": mean,
        "sd": sd,
        "Sharpe_ann": mean / sd * ANN
        if (sd and np.isfinite(sd) and sd > 0)
        else float("nan"),
        "t": float(np.sqrt(cal.size)) * mean / sd
        if (sd and np.isfinite(sd) and sd > 0)
        else float("nan"),
        "MaxDD": maxdd(cal),
        "worst_day": float(cal.min()) if cal.size else float("nan"),
        "worst_date": str(pd.Timestamp(d[worst_i]).date()) if worst_i >= 0 else "",
        "mean_satout": float(sat.mean()) if sat.size else float("nan"),
    }


# ------------------------------------------------------- the walk-forward ----
def walk_forward(
    r: np.ndarray, cands: list[dict]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Expanding, embargoed threshold choice; only the out-of-sample day scores.

    For day i the training window is days [0, i - EMBARGO_DAYS); the
    candidate with the best in-window calendar-day Sharpe is applied to day
    i.  The first candidate is the null "never sit out", and np.argmax
    keeps the first maximum, so a tie goes to always-sell.  Days whose
    training window is shorter than WARMUP_SESSIONS have no decision and
    are not scored.
    """
    n = r.size
    curves = np.vstack([expanding_sharpe(np.where(c["skip"], 0.0, r)) for c in cands])
    curves = np.where(np.isfinite(curves), curves, -np.inf)
    scored = np.zeros(n, dtype=bool)
    pick = np.full(n, -1, dtype=int)
    skip = np.zeros(n, dtype=bool)
    for i in range(n):
        k = i - EMBARGO_DAYS
        if k < WARMUP_SESSIONS:
            continue
        c = int(np.argmax(curves[:, k]))
        pick[i] = c
        skip[i] = bool(cands[c]["skip"][i])
        scored[i] = True
    return scored, skip, pick


# ------------------------------------------------------------- placebo ----
def block_shuffled_masks(
    skip: np.ndarray, n_draw: int, rng: np.random.Generator
) -> np.ndarray:
    """n_draw masks with the SAME sat-out count and the SAME run lengths.

    The mask is cut into its maximal runs; the sat-out run lengths and the
    traded run lengths are each permuted, then re-interleaved starting from
    the same state the real mask starts in.  Every draw therefore sits out
    exactly as many days, in runs of exactly the same lengths, as the rule
    - it only sits out on different days.
    """
    m = np.asarray(skip, dtype=bool)
    n = m.size
    edges = np.flatnonzero(np.diff(m.astype(np.int8)) != 0) + 1
    runs = np.split(np.arange(n), edges)
    lens = np.array([len(r) for r in runs])
    states = np.array([bool(m[r[0]]) for r in runs])
    len_on = lens[states]
    len_off = lens[~states]
    first = bool(m[0])
    out = np.zeros((n_draw, n), dtype=bool)
    for b in range(n_draw):
        on = rng.permutation(len_on)
        off = rng.permutation(len_off)
        seq: list[tuple[bool, int]] = []
        i_on = i_off = 0
        state = first
        while i_on < on.size or i_off < off.size:
            if state and i_on < on.size:
                seq.append((True, int(on[i_on])))
                i_on += 1
            elif (not state) and i_off < off.size:
                seq.append((False, int(off[i_off])))
                i_off += 1
            state = not state
        pos = 0
        row = out[b]
        for st, ln in seq:
            if st:
                row[pos : pos + ln] = True
            pos += ln
        assert pos == n
    assert (out.sum(axis=1) == m.sum()).all(), (
        "a placebo mask changed the sat-out count"
    )
    return out


def placebo_row(r: np.ndarray, skip: np.ndarray, rng: np.random.Generator) -> dict:
    """Where the rule's calendar-day Sharpe falls among block-shuffled skips."""
    ok = np.isfinite(r)
    rr = r[ok]
    mm = np.asarray(skip, dtype=bool)[ok]
    if mm.sum() == 0 or mm.all():
        return {
            "n_placebo": 0,
            "sharpe_rule": sharpe(np.where(mm, 0.0, rr)),
            "placebo_mean": float("nan"),
            "placebo_sd": float("nan"),
            "placebo_q05": float("nan"),
            "placebo_q50": float("nan"),
            "placebo_q95": float("nan"),
            "pct_rank": float("nan"),
            "p_value": float("nan"),
        }
    s_rule = sharpe(np.where(mm, 0.0, rr))
    masks = block_shuffled_masks(mm, N_PLACEBO, rng)
    cal = np.where(masks, 0.0, rr[None, :])
    mean = cal.mean(axis=1)
    sd = cal.std(axis=1, ddof=1)
    s_plac = np.where(sd > 0, mean / np.where(sd > 0, sd, 1.0) * ANN, np.nan)
    good = s_plac[np.isfinite(s_plac)]
    ge = int((good >= s_rule).sum())
    return {
        "n_placebo": int(good.size),
        "sharpe_rule": s_rule,
        "placebo_mean": float(good.mean()),
        "placebo_sd": float(good.std(ddof=1)),
        "placebo_q05": float(np.quantile(good, 0.05)),
        "placebo_q50": float(np.quantile(good, 0.50)),
        "placebo_q95": float(np.quantile(good, 0.95)),
        "pct_rank": float((good < s_rule).mean()),
        "p_value": float((1 + ge) / (1 + good.size)),
    }


# ---------------------------------------------------------------- main ----
def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    p27 = _load_module(
        REPO / "writeup" / "intraday_proposals" / "27_tradeable_polish.py", "p30_p27"
    )
    pkg = pd.read_parquet(sorted((HOLD / "cache").glob("trade_*.parquet"))[-1])
    books, idx = gate(p27, pkg)
    sig = signals_1100(pkg, idx)
    rules = build_rules(sig, idx, p27)
    era_mask = np.asarray(idx >= p27.DAILY_0DTE, dtype=bool)
    samples = (("whole", np.ones(len(idx), dtype=bool)), ("daily_era", era_mask))
    book_order = ["exit_crossed_quoted", "exit_mid_model", "hold_settle"]

    # ------------------------------------------------------- rules.csv ----
    rows = []
    for bk in book_order:
        r_full = books[bk].to_numpy(float)
        for smp, m in samples:
            for rule in rules:
                row = {
                    "book": bk,
                    "sample": smp,
                    "family": rule["family"],
                    "rule": rule["name"],
                    "n_warmup_days": rule["warmup"],
                }
                row.update(rule_stats(r_full[m], rule["skip"][m], idx[m]))
                rows.append(row)
    rules_df = pd.DataFrame(rows)
    rules_df.to_csv(OUT / "rules.csv", index=False)
    cols = [
        "family",
        "rule",
        "n_days",
        "n_traded",
        "frac_satout",
        "mean_traded",
        "mean_calendar",
        "sd",
        "Sharpe_ann",
        "t",
        "MaxDD",
        "worst_day",
        "mean_satout",
    ]
    for bk in book_order:
        for smp, _ in samples:
            sub = rules_df[(rules_df["book"] == bk) & (rules_df["sample"] == smp)]
            show(sub[cols], f"rules.csv  |  book {bk}  |  sample {smp}")

    # --------------------------------------------------------- oos.csv ----
    fams = ["forecast", "implied_level", "realized_regime", "calendar"]
    base_rule = rules[0]
    oos_rows = []
    selected: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}
    for bk in book_order:
        r_full = books[bk].to_numpy(float)
        for fam in fams:
            cands = [base_rule] + [c for c in rules if c["family"] == fam]
            scored, wf_skip, pick = walk_forward(r_full, cands)
            selected[(bk, fam)] = (scored, wf_skip)
            names = np.array([c["name"] for c in cands])
            for smp, m in samples:
                sel = scored & m
                if not sel.any():
                    continue
                st = rule_stats(r_full[sel], wf_skip[sel], idx[sel])
                base = rule_stats(r_full[sel], base_rule["skip"][sel], idx[sel])
                picks = pd.Series(names[pick[sel]]).value_counts()
                oos_rows.append(
                    {
                        "block": "walk_forward_selected",
                        "book": bk,
                        "sample": smp,
                        "family": fam,
                        "rule": f"{fam}: threshold chosen on an expanding window, {EMBARGO_DAYS}-day embargo",
                        **st,
                        "Sharpe_always_sell": base["Sharpe_ann"],
                        "delta_Sharpe": st["Sharpe_ann"] - base["Sharpe_ann"],
                        "beats_always_sell": bool(
                            st["Sharpe_ann"] > base["Sharpe_ann"]
                        ),
                        "modal_pick": str(picks.index[0]),
                        "modal_pick_share": float(picks.iloc[0] / picks.sum()),
                    }
                )
        # every FIXED rule, scored on the same out-of-sample day set
        scored_any, _, _ = walk_forward(r_full, [base_rule])
        for smp, m in samples:
            sel = scored_any & m
            base = rule_stats(r_full[sel], base_rule["skip"][sel], idx[sel])
            for rule in rules:
                st = rule_stats(r_full[sel], rule["skip"][sel], idx[sel])
                oos_rows.append(
                    {
                        "block": "fixed_rule_oos",
                        "book": bk,
                        "sample": smp,
                        "family": rule["family"],
                        "rule": rule["name"],
                        **st,
                        "Sharpe_always_sell": base["Sharpe_ann"],
                        "delta_Sharpe": st["Sharpe_ann"] - base["Sharpe_ann"],
                        "beats_always_sell": bool(
                            st["Sharpe_ann"] > base["Sharpe_ann"]
                        ),
                        "modal_pick": rule["name"],
                        "modal_pick_share": 1.0,
                    }
                )
    oos_df = pd.DataFrame(oos_rows)
    oos_df.to_csv(OUT / "oos.csv", index=False)
    ocols = [
        "block",
        "family",
        "rule",
        "n_days",
        "n_traded",
        "frac_satout",
        "mean_calendar",
        "sd",
        "Sharpe_ann",
        "Sharpe_always_sell",
        "delta_Sharpe",
        "beats_always_sell",
        "t",
        "MaxDD",
        "worst_day",
        "mean_satout",
        "modal_pick",
        "modal_pick_share",
    ]
    print(
        f"\nout-of-sample day set: expanding training window, {EMBARGO_DAYS}-day embargo, "
        f"minimum {WARMUP_SESSIONS} training days, so the first "
        f"{WARMUP_SESSIONS + EMBARGO_DAYS} days are never scored"
    )
    for bk in book_order:
        for smp, _ in samples:
            sub = oos_df[(oos_df["book"] == bk) & (oos_df["sample"] == smp)]
            show(sub[ocols], f"oos.csv  |  book {bk}  |  sample {smp}")

    # ----------------------------------------------------- placebo.csv ----
    rng = np.random.default_rng(PLACEBO_SEED)
    prows = []
    for bk in book_order:
        r_full = books[bk].to_numpy(float)
        scored_any, _, _ = walk_forward(r_full, [base_rule])
        for smp, m in samples:
            # "full" pairs with rules.csv, "oos" with oos.csv: a placebo is
            # only read against a Sharpe computed on the same days.
            for day_set, sel in (("full", m), ("oos", m & scored_any)):
                for rule in rules:
                    if rule["family"] == "baseline":
                        continue
                    prows.append(
                        {
                            "book": bk,
                            "sample": smp,
                            "day_set": day_set,
                            "block": "fixed_rule",
                            "family": rule["family"],
                            "rule": rule["name"],
                            **placebo_row(r_full[sel], rule["skip"][sel], rng),
                        }
                    )
            for fam in fams:
                wf_scored, wf = selected[(bk, fam)]
                sel = m & wf_scored
                prows.append(
                    {
                        "book": bk,
                        "sample": smp,
                        "day_set": "oos",
                        "block": "walk_forward_selected",
                        "family": fam,
                        "rule": f"{fam}: threshold chosen on an expanding window, {EMBARGO_DAYS}-day embargo",
                        **placebo_row(r_full[sel], wf[sel], rng),
                    }
                )
    plac_df = pd.DataFrame(prows)
    plac_df.to_csv(OUT / "placebo.csv", index=False)
    pcols = [
        "day_set",
        "block",
        "family",
        "rule",
        "n_placebo",
        "sharpe_rule",
        "placebo_mean",
        "placebo_sd",
        "placebo_q05",
        "placebo_q50",
        "placebo_q95",
        "pct_rank",
        "p_value",
    ]
    print(
        f"\nplacebo: {N_PLACEBO} block-shuffled skip masks per rule (same sat-out count, same "
        f"run-length multiset), seed {PLACEBO_SEED}; "
        f"p = (1 + #(placebo Sharpe >= rule Sharpe)) / (1 + n_placebo)"
    )
    for bk in book_order:
        for smp, _ in samples:
            sub = plac_df[(plac_df["book"] == bk) & (plac_df["sample"] == smp)]
            show(sub[pcols], f"placebo.csv  |  book {bk}  |  sample {smp}")

    # ---------------------------------------------------------- gate ----
    key = ["book", "sample", "family", "rule"]
    oos_fix = oos_df[oos_df["block"].isin(["fixed_rule_oos", "walk_forward_selected"])]
    oos_fix = oos_fix[oos_fix["family"] != "baseline"]
    pl = plac_df.loc[plac_df["day_set"] == "oos"]
    g = oos_fix.merge(
        pl[key + ["p_value", "pct_rank", "sharpe_rule"]], on=key, how="left"
    )
    assert not g["p_value"].isna().all(), "the gate lost its placebo leg in the merge"
    same = g["sharpe_rule"].notna() & g["Sharpe_ann"].notna()
    assert np.allclose(
        g.loc[same, "sharpe_rule"], g.loc[same, "Sharpe_ann"], rtol=0, atol=1e-12
    ), "the placebo and the out-of-sample table disagree on the rule's own Sharpe"
    g["adopted"] = g["beats_always_sell"] & (g["p_value"] < PLACEBO_ALPHA)
    show(
        g[
            [
                "book",
                "sample",
                "block",
                "family",
                "rule",
                "Sharpe_ann",
                "Sharpe_always_sell",
                "beats_always_sell",
                "p_value",
                "adopted",
            ]
        ],
        f"GATE  |  adopted = out-of-sample beats always-sell AND placebo p < {PLACEBO_ALPHA}",
    )
    n_ad = int(g["adopted"].sum())
    print(f"\nrules adopted on any book or sample: {n_ad}")
    prim = g[(g["book"] == "exit_crossed_quoted") & g["adopted"]]
    print(f"rules adopted on the primary book (exit_crossed_quoted): {int(len(prim))}")
    if len(prim) == 0:
        print(
            "no skip rule survives the gate on the primary book: the seller sits out on no day"
        )
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
