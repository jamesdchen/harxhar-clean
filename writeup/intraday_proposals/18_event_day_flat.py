"""Proposal 18 - flat on FOMC statement days and month-end sessions at the close.

THE CANDIDATE, ONE RULE. The deck's 15:30 close trade (notebooks/atm_straddle_rv_iv.ipynb,
per-day files results/atm_straddle_0dte_1530/daily_<tag>.parquet) with the position

    q_t = 0            on FOMC statement days and on the last session of a calendar month
    q_t = sign(s_t)    on every other day

and nothing else changed: same package, same entry, same cash settlement, same forecast.
The flags come from asl.fomc_and_monthend(index, repo): is_fomc (nullable boolean; NA
beyond the knowledge horizon is treated as False here) and is_me (last session of each
calendar month, taken from the traded index).

PRIOR REGISTRATION. writeup/intraday_proposals/06_intraday_rules_one_trade_per_day.md
section (e) declared and ran this filter on the intraday notebook's close leg (864 days,
block-diagonal ridge) and closed with "needs more ... adopt only after an out-of-sample era
or a pre-registered forward test". This file is the deck-level test on the deck's own 866
days, at both fills, on all seven scored forecasts, with the placebos, the concentration
curve and the mechanism reading that 06 did not run.

WHAT IS AUDITED (every item prints a number):
  1  causality of the flags, the FOMC horizon, the 2020-03-16 emergency statement
  2  reproduction of the coordinator's numbers; seven forecasts, both fills; paired
     block-bootstrap intervals (percentile and basic); the rule on always short and on the
     two heaviside legs; the buy/sell mean decomposition; the information ratio
  3  multiplicity: the three sub-rules, the neighbour-day placebos, 2000 random 84-day
     flats
  4  concentration: leave-one-year-out, the k-worst-event-day curve, month-end vs FOMC
  5  mechanism: the last bar, the close bar and the close-bar implied on event days;
     the position mix; whether the baseline's calendar columns move its forecast
  6  the corollary (long the package on month-ends), pre-registered here, run once, NOT
     adopted, plus its genuine holdout on the twenty unscored months 2024-05 .. 2025-12

Re-runnable: python 18_event_day_flat.py. Reads only committed artefacts and the two
notebook caches (the 15:30/16:00 chain slice and the official-close table). Writes
results/atm_straddle_intraday/proposals/18/.
"""

import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, "C:/Users/james/CC Allowed/harxhar-0dte-professor/notebooks")
import atm_straddle_lib as asl  # noqa: E402

REPO = asl.find_repo(Path("C:/Users/james/CC Allowed/harxhar-0dte-professor"))
BOOKS = REPO / "results" / "atm_straddle_0dte_1530"
CACHE = BOOKS / "cache"
OUT = REPO / "results" / "atm_straddle_intraday" / "proposals" / "18"
OUT.mkdir(parents=True, exist_ok=True)
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 60)

MODELS = [t for t in asl.MODEL_ORDER if t != "blk2_inc"]  # the seven scored forecasts
ANN = np.sqrt(asl.PERIODS_PER_YEAR)
BLEN, BOOT_B, SEED = 21, 2000, 0


def sharpe(v):
    v = np.asarray(v, float)
    sd = v.std(ddof=1)
    return float(v.mean() / sd * ANN) if sd > 0 else float("nan")


def tstat(v):
    v = np.asarray(v, float)
    sd = v.std(ddof=1)
    return float(np.sqrt(len(v)) * v.mean() / sd) if sd > 0 else float("nan")


def boot_idx(n_obs, blen=BLEN, boot_b=BOOT_B, seed=SEED):
    return asl.circular_block_bootstrap_idx(
        np.random.default_rng(seed), n_obs, blen, boot_b
    )


def paired_dsharpe(a, b, idx):
    """Sharpe(a) - Sharpe(b) with percentile and basic intervals on the same day draws."""
    a, b = np.asarray(a, float), np.asarray(b, float)

    def s(x):
        return x.mean(axis=1) / x.std(axis=1, ddof=1) * ANN

    dd = s(a[idx]) - s(b[idx])
    hat = sharpe(a) - sharpe(b)
    lo, hi = (float(v) for v in np.percentile(dd, [2.5, 97.5]))
    return {
        "dSharpe": hat,
        "pct_lo": lo,
        "pct_hi": hi,
        "basic_lo": 2 * hat - hi,
        "basic_hi": 2 * hat - lo,
    }


def reading(lo, hi):
    return "excludes zero" if (lo > 0 or hi < 0) else "includes zero"


print("=" * 100)
print("0. FRAME AND GATE")
print("=" * 100)
books = {t: pd.read_parquet(BOOKS / f"daily_{t}.parquet") for t in MODELS}
common = books["blk2"].index
for t in MODELS:
    common = common.intersection(books[t].index)
print(f"forecasts {MODELS}")
print(
    f"days common to all seven: {len(common)}  {common.min().date()} .. {common.max().date()}"
)
px = books["blk2"].loc[common]
n = len(common)
Rv = px["R"].to_numpy(float)
EXIT = px["exit"].to_numpy(float)
BID = (px["bid_c"] + px["bid_p"]).to_numpy(float)
ASK = (px["ask_c"] + px["ask_p"]).to_numpy(float)
ENTRY = px["entry"].to_numpy(float)
YEAR = common.year.to_numpy()

# the deck's crossed fill: buy at the ask, sell at the bid, cash settlement; a flat day pays nothing
_bidS = pd.Series(BID, index=common)
_askS = pd.Series(ASK, index=common)
_exS = pd.Series(EXIT, index=common)


def crossed(q):
    q = pd.Series(np.asarray(q, float), index=common)
    signq = np.sign(q.replace(0.0, -1.0))
    r = asl.crossed_premium_return(signq, _exS, _bidS, _askS) * q.abs()
    return r.to_numpy(float)


def mid(q):
    return np.asarray(q, float) * Rv


POS = {
    t: np.where(books[t].loc[common, "signal"].to_numpy(float) > 0, 1.0, -1.0)
    for t in MODELS
}
q_sign = POS["blk2"]
print(
    "untradeable crossed rows (sign(s), blk2): "
    f"{asl.crossed_untradeable_count(pd.Series(q_sign, index=common), _bidS, _askS)}"
)
_gate = asl.rule_row(
    pd.Series(mid(q_sign), index=common), pd.Series(q_sign, index=common)
)
_ref = pd.read_csv(BOOKS / "rule_table_blk2.csv", index_col=0)
_gap = abs(float(_gate["Sharpe_ann"]) - float(_ref.loc["sign(s)", "Sharpe_ann"]))
print(
    f"gate: deck rule table sign(s) Sharpe {float(_ref.loc['sign(s)', 'Sharpe_ann']):.6f}, "
    f"rebuilt {float(_gate['Sharpe_ann']):.6f}, |diff| {_gap:.2e}"
)
assert _gap < 1e-9, "rule table not reproduced"

print()
print("=" * 100)
print("1. CAUSALITY OF THE FLAGS")
print("=" * 100)
with warnings.catch_warnings(record=True) as _w:
    warnings.simplefilter("always")
    flags = asl.fomc_and_monthend(common, REPO)
    _msgs = [str(m.message) for m in _w]
print(f"warnings raised by fomc_and_monthend: {len(_msgs)} {_msgs}")
print(
    f"FOMC knowledge horizon (flags.attrs): {pd.Timestamp(flags.attrs['fomc_known_until']).date()}"
)
print(
    f"is_fomc dtype {flags['is_fomc'].dtype}; NA rows inside the frame "
    f"{int(flags['is_fomc'].isna().sum())} (NA is treated as False)"
)
IS_FOMC = flags["is_fomc"].fillna(False).to_numpy(bool)
IS_ME = flags["is_me"].to_numpy(bool)
IS_EV = IS_FOMC | IS_ME
print(
    f"FOMC days {IS_FOMC.sum()}   month-end days {IS_ME.sum()}   "
    f"overlap {int((IS_FOMC & IS_ME).sum())}   event days {IS_EV.sum()} of {n}"
)
print("FOMC statement days in the frame:", [str(d.date()) for d in common[IS_FOMC]])
print(
    "sources: data/releases.parquet column 'fomc release' (bar-end stamps) up to its last flagged day,"
)
print(
    "         plus the hard-coded tuple asl.FOMC_STATEMENT_DAYS "
    "(2020-03-03, 2020-03-16, 2023-12-13 .. 2025-12-10)."
)

# release-file horizon, measured
_rel = pd.read_parquet(
    REPO / "data" / "releases.parquet", columns=["endbartime", "fomc release"]
)
_f = pd.to_numeric(_rel["fomc release"], errors="coerce").fillna(0.0) > 0
_fd = pd.to_datetime(_rel.loc[_f, "endbartime"]).dt.normalize()
print(
    f"releases.parquet FOMC flags: {_fd.nunique()} distinct days, "
    f"{_fd.min().date()} .. {_fd.max().date()} (the feed dies there)"
)
print(
    f"hard-coded tuple covers {min(asl.FOMC_STATEMENT_DAYS)} .. {max(asl.FOMC_STATEMENT_DAYS)}; "
    f"frame ends {common.max().date()}, so every scored day is inside the horizon"
)

# no realized data in the flags: both are pure calendar
_pert = asl.fomc_and_monthend(common, REPO)
_same = bool(
    (_pert["is_me"].to_numpy(bool) == IS_ME).all()
    and (_pert["is_fomc"].fillna(False).to_numpy(bool) == IS_FOMC).all()
)
print(
    "flags are a pure function of (dates, releases.parquet, the hard-coded tuple): "
    f"re-call identical {_same}"
)
print(
    "they read no price, no return, no realized variance and no forecast column - "
    "checked by inspection of asl.fomc_and_monthend"
)

# month-end flag against the exchange calendar, not just the traded index
_spot = pd.read_parquet(REPO / "data" / "spxw_spot.parquet")
_set = pd.to_datetime(_spot["timestamp"], utc=True).dt.tz_convert("America/New_York")
_sess = pd.DatetimeIndex(sorted(set(_set.dt.normalize().dt.tz_localize(None))))
_true_me = pd.DatetimeIndex(
    pd.Series(_sess).groupby([_sess.year, _sess.month]).max().values
)
_true_in = _true_me[(_true_me >= common.min()) & (_true_me <= common.max())]
_flagged = common[IS_ME]
print(
    f"month-end check: exchange last-sessions inside the frame {len(_true_in)}, flagged {len(_flagged)}, "
    f"mismatches {len(set(_true_in) ^ set(_flagged))}"
)

# 2020-03-16: the Sunday-afternoon emergency statement
_em = pd.Timestamp("2020-03-16")
print()
print(
    "2020-03-16 (emergency statement, announced Sunday 2020-03-15 in the afternoon, about 17:00 ET):"
)
print(
    f"  2020-03-03 in the traded frame: {pd.Timestamp('2020-03-03') in common}  "
    "(a Tuesday; SPXW listed M/W/F before 2022-06)"
)
print(
    "  2020-03-15 was a Sunday, not a session; the library flags the FIRST SESSION AFTER an "
    f"off-hours statement, i.e. {_em.date()}"
)
_i = int(np.where(common == _em)[0][0])
print(
    f"  at 15:30 on {_em.date()} the statement was about 22.5 hours old and public: "
    "the flag is known before the entry"
)
print(
    f"  that day: R {Rv[_i]:+.4f}  sign(s) position {q_sign[_i]:+.0f}  "
    f"sign(s) R' {q_sign[_i] * Rv[_i]:+.4f} (the frame's worst sign(s) day is {mid(q_sign).min():+.3f})"
)
_ev_no_em = IS_EV.copy()
_ev_no_em[_i] = bool(IS_ME[_i])
print(
    "  it was NOT scheduled in advance, so the strict pre-scheduled reading drops it: "
    f"event days {IS_EV.sum()} -> {_ev_no_em.sum()}"
)
for _lab, _fill in (("mid", mid), ("crossed", crossed)):
    _a = _fill(np.where(IS_EV, 0.0, q_sign))
    _b = _fill(np.where(_ev_no_em, 0.0, q_sign))
    print(
        f"  {_lab:8s} flat-on-events {sharpe(_a):+.4f}  without the emergency day "
        f"{sharpe(_b):+.4f}  difference {sharpe(_a) - sharpe(_b):+.4f}"
    )

print()
print("=" * 100)
print("2. REPRODUCTION, SEVEN FORECASTS, BOTH FILLS")
print("=" * 100)
IDX = boot_idx(n)
rules = {
    "sign(s)": q_sign,
    "flat on 84 event days": np.where(IS_EV, 0.0, q_sign),
    "flat on month-end only": np.where(IS_ME, 0.0, q_sign),
    "flat on FOMC only": np.where(IS_FOMC, 0.0, q_sign),
    "always short": -np.ones(n),
    "always short, flat on events": np.where(IS_EV, 0.0, -1.0),
    "heaviside long only": np.where(q_sign > 0, 1.0, 0.0),
    "heaviside long only, flat on events": np.where(
        IS_EV, 0.0, np.where(q_sign > 0, 1.0, 0.0)
    ),
    "heaviside short only": np.where(q_sign > 0, 0.0, -1.0),
    "heaviside short only, flat on events": np.where(
        IS_EV, 0.0, np.where(q_sign > 0, 0.0, -1.0)
    ),
}
rows = []
for name, q in rules.items():
    m, c = mid(q), crossed(q)
    rows.append(
        {
            "rule": name,
            "n": n,
            "n_flat": int((q == 0).sum()),
            "n_buy": int((q > 0).sum()),
            "mean mid": m.mean(),
            "std mid": m.std(ddof=1),
            "t mid": tstat(m),
            "Sharpe mid": sharpe(m),
            "mean crossed": c.mean(),
            "std crossed": c.std(ddof=1),
            "t crossed": tstat(c),
            "Sharpe crossed": sharpe(c),
            "worst": m.min(),
            "hit rate": float((m > 0).mean()),
        }
    )
repro = pd.DataFrame(rows).set_index("rule")
print(repro.round(4).to_string())
repro.to_csv(OUT / "repro_blk2.csv")

print()
print("coordinator's claims against this rebuild (block-diagonal ridge):")
for _lab, _wm, _wc in (
    ("sign(s)", 1.338, 0.870),
    ("flat on 84 event days", 1.783, 1.349),
    ("flat on month-end only", 1.667, 1.217),
    ("flat on FOMC only", 1.428, 0.974),
    ("always short", 0.204, -0.275),
    ("always short, flat on events", 0.710, 0.242),
):
    _gm = repro.loc[_lab, "Sharpe mid"]
    _gc = repro.loc[_lab, "Sharpe crossed"]
    print(
        f"  {_lab:32s} claimed {_wm:+.3f} / {_wc:+.3f}   rebuilt {_gm:+.4f} / {_gc:+.4f}   "
        f"|diff| {abs(_gm - _wm):.4f} / {abs(_gc - _wc):.4f}"
    )

print()
print(
    "per-forecast: sign(s) vs flat-on-events, both fills, paired block bootstrap "
    "(block 21, B 2000, rng 0)"
)
pf = []
for t in MODELS:
    q = POS[t]
    qf = np.where(IS_EV, 0.0, q)
    rec = {"forecast": asl.YHAT_LABEL[t], "tag": t}
    for _lab, _fill in (("mid", mid), ("crossed", crossed)):
        a, b = _fill(qf), _fill(q)
        rec[f"sign(s) {_lab}"] = sharpe(b)
        rec[f"flat {_lab}"] = sharpe(a)
        bi = paired_dsharpe(a, b, IDX)
        rec[f"dSharpe {_lab}"] = bi["dSharpe"]
        rec[f"pct {_lab}"] = f"[{bi['pct_lo']:+.3f}, {bi['pct_hi']:+.3f}]"
        rec[f"basic {_lab}"] = f"[{bi['basic_lo']:+.3f}, {bi['basic_hi']:+.3f}]"
        rec[f"pct reading {_lab}"] = reading(bi["pct_lo"], bi["pct_hi"])
        rec[f"basic reading {_lab}"] = reading(bi["basic_lo"], bi["basic_hi"])
        rec[f"t of daily diff {_lab}"] = tstat(a - b)
        # seed-free reading of the same draws: how much of the bootstrap mass is above zero
        _s = a[IDX].mean(axis=1) / a[IDX].std(axis=1, ddof=1) - b[IDX].mean(axis=1) / b[
            IDX
        ].std(axis=1, ddof=1)
        rec[f"boot share > 0 {_lab}"] = float((_s > 0).mean())
    pf.append(rec)
perf = pd.DataFrame(pf).set_index("forecast")
print(
    perf[["sign(s) mid", "flat mid", "dSharpe mid", "pct mid", "basic mid"]]
    .round(4)
    .to_string()
)
print(
    perf[
        [
            "sign(s) crossed",
            "flat crossed",
            "dSharpe crossed",
            "pct crossed",
            "basic crossed",
            "t of daily diff crossed",
            "boot share > 0 crossed",
        ]
    ]
    .round(4)
    .to_string()
)
_ex = int((perf["pct reading crossed"] == "excludes zero").sum())
_exb = int((perf["basic reading crossed"] == "excludes zero").sum())
print(
    f"crossed-spread gain range {perf['dSharpe crossed'].min():+.3f} .. "
    f"{perf['dSharpe crossed'].max():+.3f}; percentile intervals excluding zero {_ex} of 7; "
    f"basic intervals excluding zero {_exb} of 7"
)

print()
print(
    "how stable is that count? the same intervals under ten bootstrap seeds "
    "(0..9), crossed spread"
)
seed_rows = []
for t in MODELS:
    q = POS[t]
    a, b = crossed(np.where(IS_EV, 0.0, q)), crossed(q)
    hits = 0
    los = []
    for sd in range(10):
        bi = paired_dsharpe(a, b, boot_idx(n, seed=sd))
        los.append(bi["pct_lo"])
        hits += int(bi["pct_lo"] > 0)
    seed_rows.append(
        {
            "forecast": asl.YHAT_LABEL[t],
            "seeds whose percentile interval excludes zero": hits,
            "min lower bound": min(los),
            "max lower bound": max(los),
        }
    )
seedt = pd.DataFrame(seed_rows).set_index("forecast")
print(seedt.round(4).to_string())
seedt.to_csv(OUT / "seed_sensitivity.csv")
perf.to_csv(OUT / "per_forecast.csv")

print()
print("mean decomposition of sign(s) (blk2): contribution = share x mean, R' = q R")
dec = []
for _lab, _sel in (
    ("all 866 days", np.ones(n, bool)),
    ("non-event days", ~IS_EV),
    ("event days", IS_EV),
    ("month-end days", IS_ME),
    ("FOMC days", IS_FOMC),
):
    q, r = q_sign[_sel], Rv[_sel]
    buy = q > 0
    dec.append(
        {
            "days": _lab,
            "n": int(_sel.sum()),
            "buy share": float(buy.mean()),
            "E[R|buy]": float(r[buy].mean()) if buy.any() else np.nan,
            "contrib buy": float(buy.mean() * r[buy].mean()) if buy.any() else np.nan,
            "E[-R|sell]": float(-r[~buy].mean()) if (~buy).any() else np.nan,
            "contrib sell": float((~buy).mean() * -r[~buy].mean())
            if (~buy).any()
            else np.nan,
            "mean R'": float((q * r).mean()),
            "t": tstat(q * r),
            "mean R (package)": float(r.mean()),
        }
    )
decomp = pd.DataFrame(dec).set_index("days")
print(decomp.round(4).to_string())
decomp.to_csv(OUT / "mean_decomposition.csv")

print()
print(
    "information ratio against always short on the same 866 days, zeros on event days for BOTH legs"
)
ir_rows = {}
for _lab, _fill in (("mid", mid), ("crossed", crossed)):
    ir_rows[f"flat on events (both legs), {_lab}"] = asl.information_ratio(
        pd.Series(_fill(np.where(IS_EV, 0.0, q_sign)), index=common),
        pd.Series(_fill(np.where(IS_EV, 0.0, -1.0)), index=common),
    )
    ir_rows[f"unfiltered, {_lab}"] = asl.information_ratio(
        pd.Series(_fill(q_sign), index=common),
        pd.Series(_fill(-np.ones(n)), index=common),
    )
ir = pd.DataFrame(ir_rows).T
print(ir.round(4).to_string())
ir.to_csv(OUT / "information_ratio.csv")

print()
print("=" * 100)
print("3. MULTIPLICITY, NEIGHBOUR PLACEBOS AND RANDOM FLATS")
print("=" * 100)
prev_ev = np.roll(IS_EV, -1)  # flat on the session BEFORE each event day
prev_ev[-1] = False
next_ev = np.roll(IS_EV, 1)  # flat on the session AFTER each event day
next_ev[0] = False
subs = {
    "flat on 84 event days (the rule)": IS_EV,
    "flat on month-end only": IS_ME,
    "flat on FOMC only": IS_FOMC,
    "placebo: flat the session BEFORE each event day": prev_ev,
    "placebo: flat the session AFTER each event day": next_ev,
    "placebo: BEFORE, real event days excluded": prev_ev & ~IS_EV,
    "placebo: AFTER, real event days excluded": next_ev & ~IS_EV,
}
rng = np.random.default_rng(SEED)
draws = np.array(
    [rng.choice(n, size=int(IS_EV.sum()), replace=False) for _ in range(BOOT_B)]
)
rand_gain = {}
for _lab, _fill in (("mid", mid), ("crossed", crossed)):
    base = sharpe(_fill(q_sign))
    g = np.empty(BOOT_B)
    for j in range(BOOT_B):
        qq = q_sign.copy()
        qq[draws[j]] = 0.0
        g[j] = sharpe(_fill(qq)) - base
    rand_gain[_lab] = g
    print(
        f"random 84-day flats ({BOOT_B} draws, rng 0), {_lab}: median gain {np.median(g):+.4f}, "
        f"95th pct {np.percentile(g, 95):+.4f}, max {g.max():+.4f}"
    )

pl = []
for _lab, mask in subs.items():
    rec = {
        "rule": _lab,
        "n flat": int(mask.sum()),
        "overlap with the real flags": int((mask & IS_EV).sum()),
    }
    for _fl, _fill in (("mid", mid), ("crossed", crossed)):
        a, b = _fill(np.where(mask, 0.0, q_sign)), _fill(q_sign)
        bi = paired_dsharpe(a, b, IDX)
        rec[f"Sharpe {_fl}"] = sharpe(a)
        rec[f"gain {_fl}"] = bi["dSharpe"]
        rec[f"pct {_fl}"] = f"[{bi['pct_lo']:+.3f}, {bi['pct_hi']:+.3f}]"
        rec[f"reading {_fl}"] = reading(bi["pct_lo"], bi["pct_hi"])
        rec[f"placebo pct {_fl}"] = 100.0 * float(
            (rand_gain[_fl] < bi["dSharpe"]).mean()
        )
    pl.append(rec)
plac = pd.DataFrame(pl).set_index("rule")
print(plac.round(4).to_string())
plac.to_csv(OUT / "subrules_placebos.csv")
print(
    "(placebo pct = the percentile of the rule's gain in the distribution of "
    "2000 random 84-day flats)"
)

print()
print("=" * 100)
print("4. CONCENTRATION")
print("=" * 100)
q_flat = np.where(IS_EV, 0.0, q_sign)
loyo = []
for y in sorted(set(YEAR)):
    keep = YEAR != y
    inyr = YEAR == y
    rec = {
        "year": int(y),
        "n days": int(inyr.sum()),
        "event days": int(IS_EV[inyr].sum()),
    }
    for _lab, _fill in (("mid", mid), ("crossed", crossed)):
        rec[f"sign(s) within, {_lab}"] = sharpe(_fill(q_sign)[inyr])
        rec[f"flat within, {_lab}"] = sharpe(_fill(q_flat)[inyr])
        rec[f"gain within, {_lab}"] = sharpe(_fill(q_flat)[inyr]) - sharpe(
            _fill(q_sign)[inyr]
        )
        rec[f"gain leaving out, {_lab}"] = sharpe(_fill(q_flat)[keep]) - sharpe(
            _fill(q_sign)[keep]
        )
    loyo.append(rec)
loyo_t = pd.DataFrame(loyo).set_index("year")
print(loyo_t.round(4).to_string())

print()
print("k worst event days removed from the sample (both rules), k = 0 .. 10")
ev_pos = np.where(IS_EV)[0]
order = ev_pos[np.argsort(mid(q_sign)[ev_pos])]  # most negative sign(s) R' first
kw = []
for k in range(11):
    drop = np.zeros(n, bool)
    drop[order[:k]] = True
    keep = ~drop
    rec = {"k": k, "days left": int(keep.sum())}
    rec["worst removed"] = str(common[order[k - 1]].date()) if k else ""
    rec["its sign(s) R'"] = float(mid(q_sign)[order[k - 1]]) if k else np.nan
    for _lab, _fill in (("mid", mid), ("crossed", crossed)):
        rec[f"gain {_lab}"] = sharpe(_fill(q_flat)[keep]) - sharpe(_fill(q_sign)[keep])
    kw.append(rec)
kworst = pd.DataFrame(kw).set_index("k")
print(kworst.round(4).to_string())
loyo_t.to_csv(OUT / "concentration_by_year.csv")
kworst.to_csv(OUT / "concentration_k_worst.csv")

print()
print("month-end versus FOMC: how the gain splits (crossed spread)")
gm_all = sharpe(crossed(q_flat)) - sharpe(crossed(q_sign))
gm_me = sharpe(crossed(np.where(IS_ME, 0.0, q_sign))) - sharpe(crossed(q_sign))
gm_fo = sharpe(crossed(np.where(IS_FOMC, 0.0, q_sign))) - sharpe(crossed(q_sign))
print(
    f"  both {gm_all:+.4f}   month-end only {gm_me:+.4f} ({100 * gm_me / gm_all:.1f}% of the "
    f"joint gain)   FOMC only {gm_fo:+.4f} ({100 * gm_fo / gm_all:.1f}%)   "
    f"sum of the parts {gm_me + gm_fo:+.4f}"
)

print()
print(
    "distribution of the package return R on event days "
    "(R > 0 means the straddle settled above what it cost)"
)
qt = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
dist = []
for _lab, _sel in (
    ("all 866", np.ones(n, bool)),
    ("non-event", ~IS_EV),
    ("event", IS_EV),
    ("month-end", IS_ME),
    ("FOMC", IS_FOMC),
):
    r = Rv[_sel]
    rec = {
        "days": _lab,
        "n": int(_sel.sum()),
        "mean R": r.mean(),
        "t of mean R": tstat(r),
        "share R>0 (a long wins)": float((r > 0).mean()),
    }
    rec.update({f"q{int(100 * p):02d}": float(np.quantile(r, p)) for p in qt})
    rec["mean R', sign(s)"] = float(mid(q_sign)[_sel].mean())
    rec["t of R'"] = tstat(mid(q_sign)[_sel])
    rec["sign(s) short share"] = float((q_sign[_sel] < 0).mean())
    dist.append(rec)
distr = pd.DataFrame(dist).set_index("days")
print(distr.round(4).to_string())
distr.to_csv(OUT / "event_distribution.csv")
_me_r = Rv[IS_ME]
_trim = np.sort(_me_r)[:-3]
print(
    f"month-end mean R {_me_r.mean():+.4f}; with the three largest month-end R removed "
    f"{_trim.mean():+.4f}; median {np.median(_me_r):+.4f}"
)

print()
print("=" * 100)
print("5. MECHANISM")
print("=" * 100)
pan = asl.load_yhat_panel_mz(asl.yhat_paths(REPO)["a0"])
pan["date"] = pd.to_datetime(pan["date"])
# bar-end stamps: 15:30 is the 15:00->15:30 bar the forecast reads, 16:00 is the traded close bar
bar_last = pan[pan["mins"] == 930].drop_duplicates("date").set_index("date")["rv_raw"]
bar_close = pan[pan["mins"] == 960].drop_duplicates("date").set_index("date")["rv_raw"]
mech = (
    px[["iv_var", "rv_hat"]]
    .join(bar_last.rename("rv_last_bar"))
    .join(bar_close.rename("rv_close_bar"))
)
print(
    f"panel rows joined: last bar {int(mech['rv_last_bar'].notna().sum())} of {n}, "
    f"close bar {int(mech['rv_close_bar'].notna().sum())} of {n}"
)
_ivv = px["iv_var"].to_numpy(float)
mm = []
for _lab, _sel in (
    ("non-event", ~IS_EV),
    ("event", IS_EV),
    ("month-end", IS_ME),
    ("FOMC", IS_FOMC),
):
    g = mech[_sel]
    mm.append(
        {
            "days": _lab,
            "n": int(_sel.sum()),
            "median rv last bar (1e-6)": 1e6 * float(g["rv_last_bar"].median()),
            "median iv_var close bar (1e-6)": 1e6 * float(g["iv_var"].median()),
            "median rv close bar (1e-6)": 1e6 * float(g["rv_close_bar"].median()),
            "median rv_hat (1e-6)": 1e6 * float(g["rv_hat"].median()),
            "median close/last": float((g["rv_close_bar"] / g["rv_last_bar"]).median()),
            "median close/implied": float((g["rv_close_bar"] / g["iv_var"]).median()),
            "median rv_hat/implied": float((g["rv_hat"] / g["iv_var"]).median()),
            "share close bar > last bar": float(
                (g["rv_close_bar"] > g["rv_last_bar"]).mean()
            ),
            "share close bar > implied": float(
                (g["rv_close_bar"] > g["iv_var"]).mean()
            ),
        }
    )
mecht = pd.DataFrame(mm).set_index("days")
print(mecht.round(4).to_string())
mecht.to_csv(OUT / "mechanism_bars.csv")

print()
print(
    "what the package actually pays on: the squared terminal move r^2 = ((S_close - S)/S)^2, "
    "not the close bar's realized variance"
)
_r2 = (((px["S_close"] - px["S"]) / px["S"]) ** 2).to_numpy(float)
_rvcb = mech["rv_close_bar"].to_numpy(float)
tm = []
for _lab, _sel in (
    ("non-event", ~IS_EV),
    ("event", IS_EV),
    ("month-end", IS_ME),
    ("FOMC", IS_FOMC),
):
    tm.append(
        {
            "days": _lab,
            "n": int(_sel.sum()),
            "median r^2 (1e-6)": 1e6 * float(np.median(_r2[_sel])),
            "mean r^2 / mean close-bar RV": float(
                _r2[_sel].mean() / _rvcb[_sel].mean()
            ),
            "median r^2 / iv_var": float(np.median(_r2[_sel] / _ivv[_sel])),
            "share r^2 > iv_var": float((_r2[_sel] > _ivv[_sel]).mean()),
            "share close-bar RV > iv_var": float((_rvcb[_sel] > _ivv[_sel]).mean()),
        }
    )
tmt = pd.DataFrame(tm).set_index("days")
print(tmt.round(4).to_string())
tmt.to_csv(OUT / "mechanism_terminal_move.csv")

print()
print("position mix on event days, all seven forecasts (share short)")
pm = []
for t in MODELS:
    q = POS[t]
    pm.append(
        {
            "forecast": asl.YHAT_LABEL[t],
            "short share, all": float((q < 0).mean()),
            "short share, non-event": float((q[~IS_EV] < 0).mean()),
            "short share, month-end": float((q[IS_ME] < 0).mean()),
            "short share, FOMC": float((q[IS_FOMC] < 0).mean()),
            "mean R' event": float((q * Rv)[IS_EV].mean()),
            "mean R' non-event": float((q * Rv)[~IS_EV].mean()),
        }
    )
posmix = pd.DataFrame(pm).set_index("forecast")
print(posmix.round(4).to_string())
posmix.to_csv(OUT / "position_mix.csv")

print()
print(
    "do the baseline's calendar columns move its forecast on those days? "
    "rv_hat / iv_var by day type"
)
_rvc = mech["rv_close_bar"].to_numpy(float)
cal = []
for t in MODELS:
    rh = books[t].loc[common, "rv_hat"].to_numpy(float)
    ratio = rh / _ivv
    cal.append(
        {
            "forecast": asl.YHAT_LABEL[t],
            "median rv_hat/iv_var non-event": float(np.median(ratio[~IS_EV])),
            "median rv_hat/iv_var month-end": float(np.median(ratio[IS_ME])),
            "median rv_hat/iv_var FOMC": float(np.median(ratio[IS_FOMC])),
            "median rv_hat month-end / non-event": float(
                np.median(rh[IS_ME]) / np.median(rh[~IS_EV])
            ),
        }
    )
calt = pd.DataFrame(cal).set_index("forecast")
print(calt.round(4).to_string())
print(
    f"for scale: median iv_var month-end / non-event {np.median(_ivv[IS_ME]) / np.median(_ivv[~IS_EV]):.4f}; "
    f"median realized close bar month-end / non-event {np.median(_rvc[IS_ME]) / np.median(_rvc[~IS_EV]):.4f}"
)
calt.to_csv(OUT / "calendar_channel.csv")
_binc = (
    pd.read_parquet(BOOKS / "daily_blk2_inc.parquet")
    .loc[common, "rv_hat"]
    .to_numpy(float)
)
_b2 = books["blk2"].loc[common, "rv_hat"].to_numpy(float)
_flip = int(
    (
        np.sign(_b2[IS_FOMC] - _ivv[IS_FOMC]) != np.sign(_binc[IS_FOMC] - _ivv[IS_FOMC])
    ).sum()
)
print(
    "diagnostic, the ridge with and without the FOMC columns (blk2 vs blk2_inc): "
    f"median rv_hat ratio on FOMC days {float(np.median(_b2[IS_FOMC] / _binc[IS_FOMC])):.4f}, "
    f"on other days {float(np.median(_b2[~IS_FOMC] / _binc[~IS_FOMC])):.4f}; "
    f"positions differ on FOMC days {_flip} of {int(IS_FOMC.sum())}"
)

print()
print("=" * 100)
print("6. THE COROLLARY (pre-registered here, run once, NOT adopted) AND ITS HOLDOUT")
print("=" * 100)
print(
    "Formulated AFTER seeing the month-end mean R above. It therefore carries selection risk and"
)
print("is reported, not adopted. Two cells, no variants, no grid:")
print("  C1  q = +1 on month-end sessions, sign(s) elsewhere")
print("  C2  q = +1 on FOMC statement days, sign(s) elsewhere")
cor_rules = {
    "C1 long on month-ends": np.where(IS_ME, 1.0, q_sign),
    "C2 long on FOMC days": np.where(IS_FOMC, 1.0, q_sign),
}
cr = []
for name, q in cor_rules.items():
    rec = {"rule": name, "positions changed": int((q != q_sign).sum())}
    for _lab, _fill in (("mid", mid), ("crossed", crossed)):
        a = _fill(q)
        rec[f"Sharpe {_lab}"] = sharpe(a)
        for cmp_name, cmp_q in (("vs sign(s)", q_sign), ("vs flat", q_flat)):
            bi = paired_dsharpe(a, _fill(cmp_q), IDX)
            rec[f"{cmp_name} d{_lab}"] = bi["dSharpe"]
            rec[f"{cmp_name} pct {_lab}"] = (
                f"[{bi['pct_lo']:+.3f}, {bi['pct_hi']:+.3f}]"
            )
            rec[f"{cmp_name} reading {_lab}"] = reading(bi["pct_lo"], bi["pct_hi"])
    cr.append(rec)
corr = pd.DataFrame(cr).set_index("rule")
print(corr.round(4).to_string())
corr.to_csv(OUT / "corollary.csv")

print()
print(
    "HOLDOUT: the long-on-month-end leg needs NO forecast, so the twenty unscored months"
)
print(
    "2024-05 .. 2025-12 are a genuine out-of-sample test of the corollary (not of the flat rule)."
)
_ck = sorted(CACHE.glob("chain_15301600v2_*.parquet"))
assert _ck, "chain 15:30/16:00 cache missing - run the deck notebook once"
ch = pd.read_parquet(_ck[-1])
ch["timestamp"] = pd.to_datetime(ch["timestamp"], utc=True)
_et = ch["timestamp"].dt.tz_convert("America/New_York")
ch["hhmm"] = np.where(
    (_et.dt.hour == 15) & (_et.dt.minute == 30),
    "15:30",
    np.where((_et.dt.hour == 16) & (_et.dt.minute == 0), "16:00", "other"),
)
ch["et"] = _et
ch["et_date"] = _et.dt.normalize().dt.tz_localize(None)
ch["exp_date"] = pd.to_datetime(ch["expiration"]).dt.normalize()
ch = ch[ch["et_date"] == ch["exp_date"]]
e = ch[ch["hhmm"] == "15:30"].copy()
half = asl.early_close_days(e)
sess_all = pd.DatetimeIndex(sorted(e["et_date"].unique()))
_lo, _hi = pd.Timestamp("2024-05-01"), pd.Timestamp("2025-12-31")
me_all = pd.DatetimeIndex(
    pd.Series(sess_all).groupby([sess_all.year, sess_all.month]).max().values
)
me_hold = me_all[(me_all >= _lo) & (me_all <= _hi)]
print(
    f"0DTE sessions in the chain {len(sess_all)}; months in the window {len(me_hold)}; "
    f"half sessions among the month-ends (15:30 row after the close, dropped): "
    f"{[str(d.date()) for d in me_hold if d in set(half)]}"
)
me_hold = pd.DatetimeIndex([d for d in me_hold if d not in set(half)])
_gc = sorted(CACHE.glob("gspc_close_*.parquet"))
assert _gc, "official-close cache missing - run the deck notebook once"
close = pd.read_parquet(_gc[-1])["close"].astype(float)
close.index = pd.to_datetime(close.index).normalize()


def build_book(e_rows, days):
    """The deck's 15:30 package on the given sessions: nearest-OTM call and put on a
    live midpoint, entry = mid_c + mid_p, cash settlement against the official close."""
    e_ = e_rows[e_rows["et_date"].isin(days)]
    live = e_[np.isfinite(e_["mid"]) & (e_["mid"] > 0)].copy()
    live["S"] = live["underlying_price"].astype(float)
    sp = live.dropna(subset=["S"]).groupby("et_date")["S"].first()
    cc = live[live["cp"].astype(str).str.upper().str[0] == "C"].copy()
    pp = live[live["cp"].astype(str).str.upper().str[0] == "P"].copy()
    cc["S"] = cc["et_date"].map(sp)
    pp["S"] = pp["et_date"].map(sp)
    cc = cc[np.isfinite(cc["S"]) & (cc["strike"] >= cc["S"])].assign(
        k_gap=lambda z: z["strike"] - z["S"]
    )
    pp = pp[np.isfinite(pp["S"]) & (pp["strike"] <= pp["S"])].assign(
        k_gap=lambda z: z["S"] - z["strike"]
    )
    cpick = (
        cc.sort_values(["et_date", "k_gap", "strike"])
        .groupby("et_date", as_index=False)
        .first()
    )
    ppick = (
        pp.sort_values(["et_date", "k_gap", "strike"])
        .groupby("et_date", as_index=False)
        .first()
    )
    b = (
        cpick.merge(ppick, on="et_date", suffixes=("_c", "_p"))
        .set_index("et_date")
        .sort_index()
    )
    b["S_close"] = b.index.map(close)
    b["S"] = b["S_c"].astype(float)
    b["K_c"] = b["strike_c"].astype(float)
    b["K_p"] = b["strike_p"].astype(float)
    b["entry"] = b["mid_c"].astype(float) + b["mid_p"].astype(float)
    b["ask"] = b["ask_c"].astype(float) + b["ask_p"].astype(float)
    b["exit"] = np.maximum(b["S_close"] - b["K_c"], 0.0) + np.maximum(
        b["K_p"] - b["S_close"], 0.0
    )
    b = b[np.isfinite(b["entry"]) & (b["entry"] > 0) & np.isfinite(b["exit"])]
    b = b[
        np.maximum((b["K_c"] - b["S"]).abs(), (b["S"] - b["K_p"]).abs())
        <= asl.ATM_MAX_STRIKE_GAP
    ]
    b["R"] = b["exit"] / b["entry"] - 1.0
    b["R_crossed"] = np.where(b["ask"] > 0, b["exit"] / b["ask"] - 1.0, np.nan)
    return b


all_hold = pd.DatetimeIndex(
    [d for d in sess_all if _lo <= d <= _hi and d not in set(half)]
)
ctrl = build_book(e, all_hold)
h = ctrl.loc[ctrl.index.isin(me_hold)]
print(
    f"holdout window {_lo.date()} .. {_hi.date()}: sessions priced {len(ctrl)}, "
    f"month-end packages {len(h)} of {len(me_hold)} listed month-ends"
)
print(
    "  gate: this builder rebuilds the deck's own frame before it is used out of sample:"
)
_gb = build_book(e, common)
_j = _gb.join(px[["R"]].rename(columns={"R": "R_deck"}), how="inner")
print(
    f"        {len(_j)} of {n} deck days rebuilt, max |R rebuilt - R deck| "
    f"{float((_j['R'] - _j['R_deck']).abs().max()):.2e}"
)
for _lab, _col in (("MIDPOINT", "R"), ("CROSSED", "R_crossed")):
    v = h[_col].to_numpy(float)
    print(
        f"  long the month-end package, {_lab}: n {len(v)}  mean R {v.mean():+.4f}  "
        f"median {np.median(v):+.4f}  hit rate {float((v > 0).mean()):.4f}  t {tstat(v):+.3f}"
    )
_cv = ctrl["R"].to_numpy(float)
print(
    f"  control, EVERY session in the same window (no forecast needed): n {len(_cv)}  "
    f"mean R {_cv.mean():+.4f}  median {np.median(_cv):+.4f}  hit rate {float((_cv > 0).mean()):.4f}  "
    f"t {tstat(_cv):+.3f}"
)
_rngh = np.random.default_rng(SEED)
_dr = np.array(
    [
        _cv[_rngh.choice(len(_cv), size=len(h), replace=False)].mean()
        for _ in range(BOOT_B)
    ]
)
print(
    f"  placebo, {BOOT_B} random {len(h)}-session draws from the same window: median mean R "
    f"{np.median(_dr):+.4f}, 95th pct {np.percentile(_dr, 95):+.4f}; the month-end draw sits at the "
    f"{100 * float((_dr < h['R'].mean()).mean()):.1f}th percentile"
)
_hv = h["R"].to_numpy(float)
_big = h["R"].idxmax()
print(
    f"  concentration of the holdout: the largest day is {pd.Timestamp(_big).date()} at "
    f"R {float(h.loc[_big, 'R']):+.4f}, {100 * float(h.loc[_big, 'R']) / (len(_hv) * _hv.mean()):.1f}% of the "
    f"total; mean R without it {np.sort(_hv)[:-1].mean():+.4f} (n {len(_hv) - 1}), "
    f"without the two largest {np.sort(_hv)[:-2].mean():+.4f}"
)
print(
    f"  in-sample month-ends for comparison (n {int(IS_ME.sum())}): mean R {_me_r.mean():+.4f}  "
    f"median {np.median(_me_r):+.4f}  hit rate {float((_me_r > 0).mean()):.4f}; "
    f"in-sample non-month-end mean R {Rv[~IS_ME].mean():+.4f}"
)
h[["S", "K_c", "K_p", "entry", "ask", "S_close", "exit", "R", "R_crossed"]].round(
    4
).to_csv(OUT / "holdout_monthend.csv")
ctrl[["entry", "ask", "S_close", "exit", "R", "R_crossed"]].round(4).to_csv(
    OUT / "holdout_all_sessions.csv"
)
print(f"saved {OUT / 'holdout_monthend.csv'} and {OUT / 'holdout_all_sessions.csv'}")

print()
print("=" * 100)
print("FIGURE")
print("=" * 100)
q_long_me = cor_rules["C1 long on month-ends"]


def points(q, fill):
    """Index points earned per package: q (exit - entry) at the midpoint,
    exit - ask on a long and bid - exit on a short at the crossed spread."""
    q = np.asarray(q, float)
    if fill == "mid":
        return q * (EXIT - ENTRY)
    return np.where(q > 0, EXIT - ASK, np.where(q < 0, BID - EXIT, 0.0)) * np.abs(q)


fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.8), sharex=True)
for ax, _lab in zip(axes, ("mid", "crossed")):
    for name, q, ls in (
        ("sign(s)", q_sign, "-"),
        ("flat on FOMC / month-end", q_flat, "--"),
        ("long on month-ends", q_long_me, ":"),
    ):
        ax.plot(common, np.cumsum(points(q, _lab)), ls, lw=1.2, label=name)
    ax.axhline(0.0, color="0.7", lw=0.6)
    ax.set_title(
        f"block-diagonal ridge, {'midpoint' if _lab == 'mid' else 'crossed spread'}",
        fontsize=10,
    )
    ax.set_ylabel("cumulative index points")
    ax.legend(fontsize=7.5)
fig.suptitle("15:30 close trade: cumulative points, 866 days", fontsize=11)
fig.tight_layout()
fig.savefig(OUT / "cum_points_blk2.png", dpi=130, bbox_inches="tight")
plt.close(fig)
print(f"saved {OUT / 'cum_points_blk2.png'}")
print()
print("files written to", OUT)
