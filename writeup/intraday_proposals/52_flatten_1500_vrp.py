"""52 - buying the legs back at 15:00: is any signal cheap enough to flatten on?

The question.  An earlier vintage of the intraday notebook attached, at every
stamp, the forecast of the bar that had JUST ENDED (causal, one bar stale).  On
that vintage the sign of the signal was weak and scaling the position by the
size of the signal (the unit-median rule: |s| over its expanding lagged median)
roughly doubled the Sharpe; once the attachment was corrected the sign alone
carried the trade and the scaling stopped helping.  The question asked here is
whether that stale, scaled signal is nevertheless a usable EXIT signal for the
short straddle: at 15:00, can it pick the days on which the two held legs can
be bought back cheaply enough that flattening there beats carrying the short
through the last hour into cash settlement?

The book.  Proposal 43's chain-built tape, unchanged: sell the nearest-OTM SPX
0DTE straddle at an entry clock E at the QUOTED BID, Black-76 delta-hedge the
held legs on the index every 30 minutes with each stamp's own mid-inverted
total volatility, and either

  hold      carry both legs to the 4pm cash settlement, the hedge running
            through the last bar, or
  flatten   buy both HELD legs back at flatten clock X at their QUOTED ASK and
            stop the hedge at X.

Proposal 43 has X = 15:30 only; this script generalises the flatten clock and
is gated to reproduce 43's tape exactly at X = 15:30.  E is 11:00 (the book of
record) and 13:30 (the afternoon hold design).  X is 15:00 (the question) and
15:30 (the reference the earlier proposals studied).

Every rule is a per-day flatten fraction phi in [0, 1]:

    r = r_hold + phi * G,        G = r_flatten(X) - r_hold,

so G is exactly what buying the legs back at X earns over holding them: the
settlement payoff of the held legs, net of the hedge over [X, 16:00], minus the
ask paid at X.  "Cheap enough" means E[G | the rule flattens] > 0.  G is also
reported with the buy-back at the held legs' MIDPOINT, which separates the
premium given up from the spread paid.

The signals, at flatten clock X, for each of the eight forecasts f:

  fresh   s = rv_hat_f issued AT X for the bar X..X+30  minus  slice(X)
  stale   s = rv_hat_f issued at X-30 for the bar X-30..X  minus  slice(X)
          (the one-bar-stale attachment, reproduced on purpose)

slice(X) is the window-matched implied slice of the intraday notebook's
section 5b (iv_hourly_used^2 x h_rem x w_slice), read from the same work frame
proposal 51 builds.  s > 0 says the option is cheap against the forecast.

The forms, none with a free parameter:

  sign     phi = 1{s > 0}
  scaled   phi = min(1, s+ / m),  m the expanding lagged median of |s| at that
           clock (minimum 63 sessions, strictly prior days): the unit-median
           scaling, capped at "fully flat" because the question is about
           flattening a short, not reversing it
  margin   phi = 1{s > m}: flatten only when the signal exceeds its own typical
           size

A day with no median yet (warm-up) or no quote on a held leg at X holds.

References: always hold (phi = 0) and always flatten at X (phi = 1).
Controls: the REVERSE rule (the same form on -s) and a placebo that flattens
on a block-shuffled set of days with the same count and the same run lengths
(proposal 31's vectorised draws), the rule's own fractions permuted onto them.

Pre-registered reading.  The primary unit is index points per contract.  A
cell PASSES only if the circular-block-bootstrap interval of its paired mean
difference excludes zero on the positive side against BOTH references AND its
placebo p-value is below 0.05.  The two HEADLINE cells are the ones the
question names: ridge forecast (blk2), stale signal, scaled form, X = 15:00,
E = 11:00 and E = 13:30.  All other cells are reported and counted; with C
cells about 0.025 x C pass by chance on the bootstrap leg alone.

Part A is forecast-free and runs on every chain session (2020-01-03 ..
2025-12-31, the 413 unseen sessions included): G by period and year, at the ask
and at the mid, with and without a 0.5 bp hedge cost.  Parts B and C need the
forecasts and so stop on 2024-04-30; every rule result is in-sample.

Gates, in order:
  GATE 43   proposal 43's own gates pass on the tape (gate_zero, gate_one).
  GATE X    this script's generalised book at X = 15:30 equals proposal 43's
            ``_clock_book`` hold and flatten series exactly, both entries.
  GATE S    the fresh 15:30 ridge signal has the deck's sign on every deck day,
            and stale(15:30) is fresh(15:00) by construction.
  GATE P    the fractional placebo reduces to proposal 31's masks when every
            fraction is one.

Run:  python writeup/intraday_proposals/52_flatten_1500_vrp.py
"""

from __future__ import annotations

import importlib.util
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT, ROOT / "notebooks", ROOT / "writeup"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import atm_straddle_lib as asl  # noqa: E402

HERE = Path(__file__).resolve().parent


def _load_module(path: Path, name: str) -> ModuleType:
    """The repo's read-only import: a script is imported by path, never edited."""
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None, path
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


p43 = _load_module(HERE / "43_causal_entry_over_time.py", "p43_causal_entry")
p31 = _load_module(HERE / "31_margin_skip_rule.py", "p31_margin_skip")

OUT = ROOT / "results" / "atm_straddle_intraday_holdclose" / "proposals" / "52"
DECK = ROOT / "results" / "atm_straddle_0dte_1530"

CLOCKS = p43.CLOCKS
ENTRY_CLOCKS: tuple[str, ...] = ("11:00", "13:30")
FLATTEN_CLOCKS: tuple[str, ...] = ("15:00", "15:30")
SIGNALS: tuple[str, ...] = ("stale", "fresh")
FORMS: tuple[str, ...] = ("sign", "scaled", "margin")
DIRECTIONS: tuple[str, ...] = ("rule", "reverse")
HEADLINE = {"forecast": "blk2", "signal": "stale", "form": "scaled", "flatten": "15:00"}

ANN = p43.ANN
BOOT_B = p43.BOOT_B
BOOT_BLOCK = p43.BOOT_BLOCK
BOOT_SEED = p43.BOOT_SEED
HEDGE_COST_BP = p43.HEDGE_COST_BP
ERA_START = p43.GATE_ERA_START
DECK_END = p43.DECK_END
OOS_START = p43.OOS_START
#: the unit-median scale's minimum history, proposal 19's (and 51's) value
UM_MIN_HIST = 63
N_PLACEBO = 2000
PLACEBO_SEED = 0
PLACEBO_ALPHA = 0.05
SPX_MULT = 100.0


# ------------------------------------------------------------------ helpers --
def sharpe(x: np.ndarray) -> float:
    v = np.asarray(x, float)
    v = v[np.isfinite(v)]
    if v.size < 3:
        return float("nan")
    sd = float(v.std(ddof=1))
    return float(v.mean() / sd * ANN) if sd > 0 else float("nan")


_IDX_CACHE: dict[int, np.ndarray] = {}


def boot_idx(n: int) -> np.ndarray:
    """One circular-block index per sample length, shared by every cell."""
    hit = _IDX_CACHE.get(n)
    if hit is None:
        hit = asl.circular_block_bootstrap_idx(
            np.random.default_rng([BOOT_SEED, n]), n, BOOT_BLOCK, BOOT_B
        )
        _IDX_CACHE[n] = hit
    return hit


def mean_ci(d: np.ndarray) -> tuple[float, float]:
    v = np.asarray(d, float)
    m = v[boot_idx(v.size)].mean(axis=1)
    lo, hi = np.percentile(m, [2.5, 97.5])
    return float(lo), float(hi)


def diff_row(d: np.ndarray, tag: str) -> dict[str, float]:
    v = np.asarray(d, float)
    t_hac, _ = asl.newey_west_t(v)
    lo, hi = mean_ci(v)
    return {
        f"{tag}_mean": float(v.mean()),
        f"{tag}_t_hac": float(t_hac),
        f"{tag}_ci_lo": lo,
        f"{tag}_ci_hi": hi,
    }


def write(df: pd.DataFrame, name: str, title: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / name, index=False)
    print(f"\nwrote {OUT / name}  ({len(df)} rows) - {title}")


# ---------------------------------------------------- held-leg quotes at X ---
def held_leg_quotes(
    stamp: pd.DataFrame, ch: dict[str, Any]
) -> dict[str, dict[str, np.ndarray]]:
    """Quoted bid/ask/mid at each flatten clock of the legs held from each entry."""
    idx = ch["dates"]
    out: dict[str, dict[str, np.ndarray]] = {}
    for x in FLATTEN_CLOCKS:
        ts = sorted(pd.DatetimeIndex(stamp.loc[idx, x]).unique())
        q = pd.read_parquet(
            p43.CHAIN,
            columns=["expiration", "timestamp", "strike", "cp", "bid", "ask"],
            filters=[("timestamp", "in", ts)],
        )
        # the session's own expiration at the session's own X stamp, as 43 joins
        want = pd.DataFrame(
            {"date": idx, "timestamp": [pd.Timestamp(stamp.loc[d, x]) for d in idx]}
        ).dropna(subset=["timestamp"])
        want["timestamp"] = want["timestamp"].astype(q["timestamp"].dtype)
        want["expiration"] = want["date"].astype(q["expiration"].dtype)
        q = q.merge(want, on=["expiration", "timestamp"], how="inner")
        b = q["bid"].astype(float)
        a = q["ask"].astype(float)
        dead = (b == 0.0) & (a == 0.0)  # bid == ask == 0: no quote
        q["leg_ask"] = a.mask(dead)
        q["leg_bid"] = b.mask(dead)
        q["strike"] = q["strike"].astype(float)
        q["cp"] = q["cp"].astype(str)
        q = q.dropna(subset=["leg_ask"]).drop_duplicates(["date", "strike", "cp"])
        amap = q.set_index(["date", "strike", "cp"])["leg_ask"]
        bmap = q.set_index(["date", "strike", "cp"])["leg_bid"]
        rec: dict[str, np.ndarray] = {}
        for e in ENTRY_CLOCKS:
            j = CLOCKS.index(e)
            kc, kp = ch["K_c"][:, j], ch["K_p"][:, j]
            ic = pd.MultiIndex.from_arrays([idx, kc, ["C"] * len(idx)])
            ip = pd.MultiIndex.from_arrays([idx, kp, ["P"] * len(idx)])
            ask = amap.reindex(ic).to_numpy(float) + amap.reindex(ip).to_numpy(float)
            bid = bmap.reindex(ic).to_numpy(float) + bmap.reindex(ip).to_numpy(float)
            rec[f"{e}_ask"] = ask
            rec[f"{e}_mid"] = 0.5 * (ask + bid)
        out[x] = rec
    return out


# ------------------------------------------------ the generalised book -------
def book(ch: dict[str, Any], quotes: dict[str, Any], e: str, x: str) -> pd.DataFrame:
    """Proposal 43's ``_clock_book`` with the flatten clock a parameter.

    Index points per contract throughout; the premium-unit series divide by
    the entry midpoint.  ``G`` is flatten-minus-hold, at the ask and at the mid.
    """
    j, ix = CLOCKS.index(e), CLOCKS.index(x)
    assert j < ix, (e, x)
    s, s_close = ch["S"], ch["S_close"]
    n_d, n_k = s.shape
    kc, kp = ch["K_c"][:, j], ch["K_p"][:, j]
    entry, bid = ch["entry"][:, j], ch["bid"][:, j]
    tot = ch["tot"].copy()
    tot[:, :j] = np.nan
    dlt = p43.pkg_delta_vec(tot, s, kc[:, None], kp[:, None])
    dlt[:, :j] = 0.0
    nxt = np.full_like(s, np.nan)
    nxt[:, :-1] = s[:, 1:]
    nxt[:, -1] = s_close
    d_s = np.where(np.isfinite(s) & np.isfinite(nxt), nxt - s, 0.0)
    d_s[:, :j] = 0.0
    d_s_flat = d_s.copy()
    d_s_flat[:, ix:] = 0.0
    hedge_hold = (dlt * d_s).sum(axis=1)
    hedge_flat = (dlt * d_s_flat).sum(axis=1)
    settle = np.maximum(s_close - kc, 0.0) + np.maximum(kp - s_close, 0.0)
    ok = np.isfinite(entry) & (entry > 0.0) & np.isfinite(bid) & (bid > 0.0)

    pos_flat = dlt.copy()
    pos_flat[:, ix:] = 0.0
    prev_h, prev_f = np.zeros(n_d), np.zeros(n_d)
    turn_h, turn_f = np.zeros(n_d), np.zeros(n_d)
    for k in range(j, n_k):
        turn_h += np.abs(dlt[:, k] - prev_h) * s[:, k]
        turn_f += np.abs(pos_flat[:, k] - prev_f) * s[:, k]
        prev_h, prev_f = dlt[:, k], pos_flat[:, k]
    turn_h += np.abs(prev_h) * s_close

    buy_ask = quotes[x][f"{e}_ask"]
    buy_mid = quotes[x][f"{e}_mid"]
    nan = np.full(n_d, np.nan)
    hold_pts = np.where(ok, -(settle - bid) + hedge_hold, nan)
    flat_pts = np.where(ok, -(buy_ask - bid) + hedge_flat, nan)
    flat_mid_pts = np.where(ok, -(buy_mid - bid) + hedge_flat, nan)
    cost_saved = np.where(ok, HEDGE_COST_BP * (turn_h - turn_f), nan)
    return pd.DataFrame(
        {
            "entry": np.where(ok, entry, nan),
            "hold_pts": hold_pts,
            "flat_pts": flat_pts,
            "G_pts": flat_pts - hold_pts,
            "G_mid_pts": flat_mid_pts - hold_pts,
            "G_costed_pts": flat_pts - hold_pts + cost_saved,
            "spread_pts": buy_ask - buy_mid,
            "buy_ask_pts": np.where(ok, buy_ask, nan),
            "settle_pts": np.where(ok, settle, nan),
            "hedge_tail_pts": np.where(ok, hedge_hold - hedge_flat, nan),
        },
        index=ch["dates"],
    )


# ------------------------------------------------------------- the signals ---
def um_median(s: np.ndarray) -> np.ndarray:
    """Expanding median of |s| over STRICTLY prior days, UM_MIN_HIST minimum."""
    return (
        pd.Series(np.abs(s))
        .expanding(min_periods=UM_MIN_HIST)
        .median()
        .shift(1)
        .to_numpy()
    )


def fraction(s: np.ndarray, form: str) -> np.ndarray:
    """The flatten fraction in [0, 1]; NaN signal or warm-up holds (0)."""
    s = np.asarray(s, float)
    if form == "sign":
        return np.where(np.isfinite(s) & (s > 0.0), 1.0, 0.0)
    m = um_median(s)
    ok = np.isfinite(s) & np.isfinite(m) & (m > 0.0)
    ratio = np.where(ok, np.maximum(s, 0.0) / np.where(ok, m, 1.0), 0.0)
    if form == "scaled":
        return np.minimum(ratio, 1.0)
    if form == "margin":
        return np.where(ratio > 1.0, 1.0, 0.0)
    raise ValueError(form)


def signal_table(work: pd.DataFrame, x: str) -> pd.DataFrame:
    """Per deck day: slice(X) and every forecast's fresh and stale value at X."""
    ix = CLOCKS.index(x)
    prev = CLOCKS[ix - 1]
    now = work[work["hhmm"] == x].set_index("date")
    was = work[work["hhmm"] == prev].set_index("date")
    out = pd.DataFrame(index=pd.DatetimeIndex(pd.to_datetime(now.index)))
    out["slice"] = now["slice"].to_numpy(float)
    was = was.reindex(now.index)
    for tag in asl.MODEL_ORDER:
        out[f"fcst_fresh_{tag}"] = now[f"rv_hat_{tag}"].to_numpy(float)
        out[f"fcst_stale_{tag}"] = was[f"rv_hat_{tag}"].to_numpy(float)
        out[f"fresh_{tag}"] = out[f"fcst_fresh_{tag}"] - out["slice"]
        out[f"stale_{tag}"] = out[f"fcst_stale_{tag}"] - out["slice"]
    return out


# --------------------------------------------------------------- placebo -----
def placebo_fractions(
    phi: np.ndarray, n_draw: int, rng: np.random.Generator
) -> np.ndarray:
    """(n_draw, n) placebo fraction paths.

    The active set {phi > 0} is block-shuffled by proposal 31's draws (same
    count, same run lengths); the rule's own positive fractions are then
    permuted onto each draw's active days.
    """
    act = phi > 0.0
    masks = p31.block_shuffled_masks(act, n_draw, rng)
    vals = phi[act]
    order = np.argsort(rng.random((n_draw, vals.size)), axis=1)
    out = np.zeros(masks.shape)
    out[masks] = vals[order].ravel()
    return out


def cell_task(job: dict[str, Any]) -> list[dict[str, Any]]:
    """One forecast's cells for one (entry, flatten) book, in its own process."""
    tag, e, x, i_tag = job["tag"], job["entry"], job["flatten"], job["i_tag"]
    hold, g, g_mid, g_cost = job["hold"], job["G"], job["G_mid"], job["G_costed"]
    entry, dates = job["entry_px"], pd.DatetimeIndex(job["dates"])
    quoted = np.isfinite(g)
    g0 = np.where(quoted, g, 0.0)
    g0_mid = np.where(quoted, g_mid, 0.0)
    g0_cost = np.where(quoted, g_cost, 0.0)
    flat_all = hold + g0
    samples = {"whole": np.ones(len(dates), bool), "era": dates >= ERA_START}
    rows: list[dict[str, Any]] = []
    for i_sig, sig in enumerate(SIGNALS):
        s = job[f"s_{sig}"]
        for i_form, form in enumerate(FORMS):
            for i_dir, direction in enumerate(DIRECTIONS):
                phi = fraction(s if direction == "rule" else -s, form)
                phi = np.where(quoted, phi, 0.0)
                r = hold + phi * g0
                for i_smp, (smp, m) in enumerate(samples.items()):
                    ph, act = phi[m], phi[m] > 0.0
                    rec: dict[str, Any] = {
                        "entry": e,
                        "flatten": x,
                        "forecast": tag,
                        "signal": sig,
                        "form": form,
                        "direction": direction,
                        "sample": smp,
                        "n_days": int(m.sum()),
                        "n_active": int(act.sum()),
                        "mean_fraction": float(ph.mean()),
                        "mean_pts": float(r[m].mean()),
                        "Sharpe_pts": sharpe(r[m]),
                        "Sharpe_units": sharpe(r[m] / entry[m]),
                        "hold_Sharpe_pts": sharpe(hold[m]),
                        "flat_Sharpe_pts": sharpe(flat_all[m]),
                        "G_on_active_pts": float(g0[m][act].mean())
                        if act.any()
                        else float("nan"),
                        "G_mid_on_active_pts": float(g0_mid[m][act].mean())
                        if act.any()
                        else float("nan"),
                        "G_on_idle_pts": float(g0[m][~act].mean())
                        if (~act).any()
                        else float("nan"),
                    }
                    rec.update(diff_row((phi * g0)[m], "d_hold"))
                    rec.update(diff_row(((phi - 1.0) * g0)[m], "d_flat"))
                    rec["d_hold_costed_mean"] = float((phi * g0_cost)[m].mean())
                    rec["d_hold_units_mean"] = float((phi * g0 / entry)[m].mean())
                    if act.any() and not act.all():
                        rng = np.random.default_rng(
                            [PLACEBO_SEED, i_tag, i_sig, i_form, i_dir, i_smp]
                        )
                        pf = placebo_fractions(ph, N_PLACEBO, rng)
                        rr = hold[m][None, :] + pf * g0[m][None, :]
                        mu, sd = rr.mean(axis=1), rr.std(axis=1, ddof=1)
                        s_pl = mu / sd * ANN
                        s_rule = sharpe(r[m])
                        rec["placebo_p_sharpe"] = float(
                            (1 + int((s_pl >= s_rule).sum())) / (1 + N_PLACEBO)
                        )
                        rec["placebo_p_mean"] = float(
                            (1 + int((mu >= r[m].mean()).sum())) / (1 + N_PLACEBO)
                        )
                        rec["placebo_q50_sharpe"] = float(np.median(s_pl))
                    else:
                        rec["placebo_p_sharpe"] = float("nan")
                        rec["placebo_p_mean"] = float("nan")
                        rec["placebo_q50_sharpe"] = float("nan")
                    rec["passes"] = bool(
                        rec["d_hold_ci_lo"] > 0.0
                        and rec["d_flat_ci_lo"] > 0.0
                        and rec["placebo_p_sharpe"] < PLACEBO_ALPHA
                    )
                    rows.append(rec)
    return rows


# -------------------------------------------------------------------- gates --
def gate_x(ch: dict[str, Any], quotes: dict[str, Any]) -> None:
    arrays = {
        k: ch[k] for k in ("S", "K_c", "K_p", "entry", "bid", "tot", "ask_c", "ask_p")
    }
    arrays["S_close"] = ch["S_close"]
    for e in ENTRY_CLOCKS:
        ref = p43._clock_book((CLOCKS.index(e), arrays))
        mine = book(ch, quotes, e, "15:30")
        ent = mine["entry"].to_numpy(float)
        for col, got in (
            ("hold", mine["hold_pts"].to_numpy(float) / ent),
            ("flatten", mine["flat_pts"].to_numpy(float) / ent),
        ):
            both = np.isnan(got) & np.isnan(ref[col])
            dev = float(np.max(np.where(both, 0.0, np.abs(got - ref[col]))))
            assert dev == 0.0, (e, col, dev)
        print(
            f"GATE X  entry {e}: the generalised book at X = 15:30 equals proposal "
            f"43's hold and flatten series exactly ({int(np.isfinite(ent).sum())} days)"
        )


def gate_s(sig: dict[str, pd.DataFrame]) -> None:
    deck = pd.read_parquet(DECK / "daily_blk2.parquet")
    days = pd.DatetimeIndex(pd.to_datetime(deck.index))
    mine = sig["15:30"]["fresh_blk2"].reindex(days).to_numpy(float)
    ref = deck["signal"].to_numpy(float)
    bad = int(((mine > 0.0) != (ref > 0.0)).sum())
    assert bad == 0, f"{bad} days disagree with the deck's 15:30 sign"
    a = sig["15:30"]["fcst_stale_blk2"]
    b = sig["15:00"]["fcst_fresh_blk2"]
    dev = float((a - b).abs().max())
    assert dev == 0.0, dev
    print(
        f"GATE S  fresh 15:30 ridge signal carries the deck's sign on all "
        f"{len(days)} deck days; stale(15:30) forecast is fresh(15:00)'s exactly"
    )


def gate_p() -> None:
    rng_a = np.random.default_rng([PLACEBO_SEED, 99])
    rng_b = np.random.default_rng([PLACEBO_SEED, 99])
    phi = (np.random.default_rng(7).random(400) < 0.3).astype(float)
    a = placebo_fractions(phi, 50, rng_a) > 0.0
    b = p31.block_shuffled_masks(phi > 0.0, 50, rng_b)
    assert (a == b).all()
    print("GATE P  the fractional placebo reduces to proposal 31's masks on 0/1 input")


# --------------------------------------------------------------- part A ------
def part_a(books: dict[tuple[str, str], pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for (e, x), b in books.items():
        ok = b["hold_pts"].notna() & b["G_pts"].notna()
        bb = b[ok]
        idx = pd.DatetimeIndex(bb.index)
        periods: dict[str, np.ndarray] = {
            "deck in-sample": np.asarray(idx <= DECK_END),
            "unseen": np.asarray(idx >= OOS_START),
            "all": np.ones(len(idx), bool),
        }
        for y in sorted(set(idx.year)):
            periods[str(y)] = np.asarray(idx.year == y)
        for name, m in periods.items():
            g = bb.loc[m]
            rec: dict[str, Any] = {
                "entry": e,
                "flatten": x,
                "period": name,
                "n": int(m.sum()),
                "no_quote_days": int((~ok).sum()) if name == "all" else 0,
                "hold_mean_pts": float(g["hold_pts"].mean()),
                "hold_Sharpe_pts": sharpe(g["hold_pts"].to_numpy(float)),
                "flat_mean_pts": float(g["flat_pts"].mean()),
                "flat_Sharpe_pts": sharpe(g["flat_pts"].to_numpy(float)),
                "buy_ask_mean_pts": float(g["buy_ask_pts"].mean()),
                "settle_mean_pts": float(g["settle_pts"].mean()),
                "hedge_tail_mean_pts": float(g["hedge_tail_pts"].mean()),
                "spread_mean_pts": float(g["spread_pts"].mean()),
                "G_mid_mean_pts": float(g["G_mid_pts"].mean()),
                "G_costed_mean_pts": float(g["G_costed_pts"].mean()),
                "frac_G_positive": float((g["G_pts"] > 0).mean()),
            }
            rec.update(diff_row(g["G_pts"].to_numpy(float), "G"))
            t_mid, _ = asl.newey_west_t(g["G_mid_pts"].to_numpy(float))
            rec["G_mid_t_hac"] = float(t_mid)
            rows.append(rec)
    return pd.DataFrame(rows)


# --------------------------------------------------------------- part C ------
def part_c(
    books: dict[tuple[str, str], pd.DataFrame], sig: dict[str, pd.DataFrame]
) -> pd.DataFrame:
    """G by tercile of the ridge signal over its lagged median: is there a slope?"""
    rows = []
    for (e, x), b in books.items():
        st = sig[x]
        days = st.index.intersection(
            b.index[b["hold_pts"].notna() & b["G_pts"].notna()]
        )
        for kind in SIGNALS:
            s = st.loc[days, f"{kind}_blk2"].to_numpy(float)
            m = um_median(s)
            z = s / m
            ok = np.isfinite(z)
            g = b.loc[days, "G_pts"].to_numpy(float)[ok]
            gm = b.loc[days, "G_mid_pts"].to_numpy(float)[ok]
            zz = z[ok]
            cuts = np.quantile(zz, [1 / 3, 2 / 3])
            bins = np.digitize(zz, cuts)
            for k, name in enumerate(("low (rich)", "middle", "high (cheap)")):
                sel = bins == k
                t_hac, _ = asl.newey_west_t(g[sel])
                rows.append(
                    {
                        "entry": e,
                        "flatten": x,
                        "signal": kind,
                        "tercile_of_s_over_median": name,
                        "n": int(sel.sum()),
                        "z_mean": float(zz[sel].mean()),
                        "G_ask_mean_pts": float(g[sel].mean()),
                        "G_ask_t_hac": float(t_hac),
                        "G_mid_mean_pts": float(gm[sel].mean()),
                        "frac_G_positive": float((g[sel] > 0).mean()),
                    }
                )
            rho = float(pd.Series(zz).corr(pd.Series(g), method="spearman"))
            rows.append(
                {
                    "entry": e,
                    "flatten": x,
                    "signal": kind,
                    "tercile_of_s_over_median": "spearman(z, G)",
                    "n": int(ok.sum()),
                    "z_mean": rho,
                }
            )
    return pd.DataFrame(rows)


# ------------------------------------------------------------------- main ----
def main() -> None:  # noqa: PLR0915
    OUT.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 80)
    pd.set_option("display.max_rows", 400)
    t_start = time.time()

    stamp, _ = p43.session_stamps()
    half = p43.half_sessions(stamp)
    sessions = pd.DatetimeIndex(stamp.index).difference(half)
    ch = p43.build_chain(stamp, sessions)
    books43 = p43.clock_books(ch)
    p43.gate_zero(books43, ch)
    p43.gate_one(books43)
    print("GATE 43 proposal 43's gate_zero and gate_one pass on this tape")
    quotes = held_leg_quotes(stamp, ch)
    gate_x(ch, quotes)
    gate_p()

    books = {
        (e, x): book(ch, quotes, e, x) for e in ENTRY_CLOCKS for x in FLATTEN_CLOCKS
    }

    a_tab = part_a(books)
    write(
        a_tab, "a_flatten_minus_hold.csv", "G = flatten(X) - hold, by period and year"
    )
    show_cols = [
        "entry",
        "flatten",
        "period",
        "n",
        "hold_mean_pts",
        "flat_mean_pts",
        "G_mean",
        "G_t_hac",
        "G_ci_lo",
        "G_ci_hi",
        "G_mid_mean_pts",
        "G_mid_t_hac",
        "G_costed_mean_pts",
        "spread_mean_pts",
        "hold_Sharpe_pts",
        "flat_Sharpe_pts",
    ]
    print("\n--- A. buying the held legs back at X instead of holding (index points)")
    print(a_tab[show_cols].round(4).to_string(index=False))

    # ------------------------------------------------------ parts B and C --
    p51 = _load_module(HERE / "51_accuracy_ladder_1530.py", "p51_accuracy_ladder")
    fr = p51.build_intraday_frames()
    work = fr["work"]["dh"].copy()
    work["date"] = pd.to_datetime(work["date"])
    sig = {x: signal_table(work, x) for x in FLATTEN_CLOCKS}
    gate_s(sig)

    jobs = []
    for e in ENTRY_CLOCKS:
        for x in FLATTEN_CLOCKS:
            b = books[(e, x)]
            days = sig[x].index.intersection(b.index[b["hold_pts"].notna()])
            bb = b.loc[days]
            for i_tag, tag in enumerate(asl.MODEL_ORDER):
                jobs.append(
                    {
                        "tag": tag,
                        "i_tag": i_tag,
                        "entry": e,
                        "flatten": x,
                        "dates": days.to_numpy(),
                        "hold": bb["hold_pts"].to_numpy(float),
                        "G": bb["G_pts"].to_numpy(float),
                        "G_mid": bb["G_mid_pts"].to_numpy(float),
                        "G_costed": bb["G_costed_pts"].to_numpy(float),
                        "entry_px": bb["entry"].to_numpy(float),
                        "s_fresh": sig[x].loc[days, f"fresh_{tag}"].to_numpy(float),
                        "s_stale": sig[x].loc[days, f"stale_{tag}"].to_numpy(float),
                    }
                )
    t0 = time.time()
    n_workers = min(len(jobs), os.cpu_count() or 1)
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        res = list(pool.map(cell_task, jobs))
    rules = pd.DataFrame([r for chunk in res for r in chunk])
    print(
        f"\n{len(rules)} cells on {n_workers} worker processes: {time.time() - t0:.1f} s"
    )
    write(rules, "b_rules.csv", "every flatten rule, its references, controls, placebo")

    head = rules[
        (rules["forecast"] == HEADLINE["forecast"])
        & (rules["signal"] == HEADLINE["signal"])
        & (rules["form"] == HEADLINE["form"])
        & (rules["flatten"] == HEADLINE["flatten"])
        & (rules["sample"] == "whole")
    ]
    hcols = [
        "entry",
        "direction",
        "n_days",
        "n_active",
        "mean_fraction",
        "hold_Sharpe_pts",
        "Sharpe_pts",
        "flat_Sharpe_pts",
        "G_on_active_pts",
        "G_mid_on_active_pts",
        "G_on_idle_pts",
        "d_hold_mean",
        "d_hold_t_hac",
        "d_hold_ci_lo",
        "d_hold_ci_hi",
        "d_flat_mean",
        "d_flat_ci_lo",
        "d_flat_ci_hi",
        "d_hold_costed_mean",
        "placebo_p_sharpe",
        "passes",
    ]
    print("\n--- B1. HEADLINE: ridge, stale signal, scaled fraction, flatten at 15:00")
    print(head[hcols].round(4).to_string(index=False))

    ridge = rules[
        (rules["forecast"] == "blk2")
        & (rules["sample"] == "whole")
        & (rules["direction"] == "rule")
    ]
    print("\n--- B2. ridge forecast, every signal x form x book (whole sample)")
    print(
        ridge[["entry", "flatten", "signal", "form", *hcols[2:]]]
        .round(4)
        .to_string(index=False)
    )

    main_cells = rules[(rules["direction"] == "rule") & (rules["sample"] == "whole")]
    rev_cells = rules[(rules["direction"] == "reverse") & (rules["sample"] == "whole")]
    count = (
        main_cells.groupby(["entry", "flatten", "signal", "form"])
        .agg(
            cells=("passes", "size"),
            passes=("passes", "sum"),
            d_hold_positive=("d_hold_mean", lambda v: int((v > 0).sum())),
            d_hold_ci_excl=("d_hold_ci_lo", lambda v: int((v > 0).sum())),
            median_d_hold=("d_hold_mean", "median"),
            median_d_flat=("d_flat_mean", "median"),
            median_placebo_p=("placebo_p_sharpe", "median"),
        )
        .reset_index()
    )
    write(count, "b_pass_counts.csv", "pass counts over the eight forecasts")
    print("\n--- B3. over the eight forecasts (whole sample, rule direction)")
    print(count.round(4).to_string(index=False))
    n_cells = len(main_cells)
    print(
        f"\ncells {n_cells}; passes {int(main_cells['passes'].sum())}; reverse-rule "
        f"passes {int(rev_cells['passes'].sum())}; bootstrap leg alone expects about "
        f"{0.025 * n_cells:.1f} by chance"
    )

    c_tab = part_c(books, sig)
    write(c_tab, "c_signal_terciles.csv", "G by tercile of the ridge signal")
    print("\n--- C. G by tercile of s over its lagged median (ridge)")
    print(c_tab.round(4).to_string(index=False))

    # agreement of the stale and fresh signs at each flatten clock
    for x in FLATTEN_CLOCKS:
        st = sig[x]
        agree = float(((st["fresh_blk2"] > 0) == (st["stale_blk2"] > 0)).mean())
        print(
            f"ridge sign agreement stale vs fresh at {x}: {agree:.3f}; buy share "
            f"fresh {float((st['fresh_blk2'] > 0).mean()):.3f}, stale "
            f"{float((st['stale_blk2'] > 0).mean()):.3f}"
        )
    print(
        f"\none index point per contract is ${SPX_MULT:.0f}.  total runtime "
        f"{time.time() - t_start:.1f} s"
    )


if __name__ == "__main__":
    main()
