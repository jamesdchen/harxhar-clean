"""Rule-by-strategy tables for the twelve intraday entry windows, plus pooled.

The deck's standalone (writeup/rule_by_strategy_standalone.tex, built from
writeup/make_rule_by_strategy_tex.py) is one booktabs tabular with a panel per
rule -- "Short volatility (always short)" and sign(s) -- and the eight forecasts
down the rows.  This script builds the same object for every intraday entry
window 10:00, 10:30, ..., 15:30 of the intraday notebook's frame, and one for
the pooled daily sums, with ONE added column: the annualized Sharpe at the
crossed spread (entry at the touch, exit at the touch of the next stamp; the
15:30 window cash-settles and pays no exit spread; a re-pick that lands on the
same two strikes with the same sign is a hold, not a round trip, and is not
charged -- the intraday notebook's convention).

Construction (mirrors notebooks/_write_0dte_intraday_nb.py):

  * trade frame  = the latest results/atm_straddle_intraday/cache/trade_*.parquet,
    restricted to the deck's 866 expiration days;
  * forecast     = asl.load_yhat_panel_mz(...)["rv_hat"] for each of the eight
    tags in asl.MODEL_ORDER, joined on timestamp + 30 min (the panel is
    bar-end labelled);
  * implied slice = iv_hourly^2 * hours_to_close * w_slice, with w_slice the
    expanding per-clock mean of the panel's own realized bar variance over
    PRIOR sessions (in-fit rows back to 2001, shift(1), min 63 sessions),
    divided by its reverse cumulative sum over the remaining clocks.  At 15:30
    w_slice = 1 and the slice is iv_hourly^2 / 2 exactly, equal to the deck's
    iv_var on all 866 days (both asserted);
  * censored implied: the bars whose vendor implied volatility sits on a
    solver bracket node do NOT sit flat.  They take the deck's treatment
    (iv_hourly_15_30 in §8 of notebooks/_write_0dte_nb.py): the volatility
    that reproduces the package midpoint, recovered by bisection over the
    stamp's hours_to_expiration.  With that in place the 15:30 window IS the
    deck's close trade, and gate() asserts the whole ridge row reproduces it.

Writes
  results/atm_straddle_intraday/rule_by_strategy/<hhmm>/rule_by_strategy_*.csv
  results/atm_straddle_intraday/rule_by_strategy/pooled/rule_by_strategy_*.csv
  writeup/generated_intraday/table_rule_by_strategy_intraday_<hhmm>.tex
  writeup/rule_by_strategy_intraday_<hhmm>.tex        (standalone, compiled)
  writeup/rule_by_strategy_intraday_pooled.tex        (standalone, compiled)
  writeup/rule_by_strategy_intraday_index.tex         (all thirteen, compiled)
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pypdf import PdfReader

ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, str(ROOT / "notebooks"))

import atm_straddle_lib as asl  # noqa: E402

WRITEUP = ROOT / "writeup"
GEN = WRITEUP / "generated_intraday"
OUT = ROOT / "results" / "atm_straddle_intraday" / "rule_by_strategy"
DECK = ROOT / "results" / "atm_straddle_0dte_1530"

# Deck numbers of record for the 15:30 window (results/atm_straddle_0dte_1530/
# rule_by_strategy_*.csv).  The gate is on these, never on a loosened tolerance.
DECK_ALWAYS_SHORT_SHARPE = 0.203779128108568
DECK_SIGN_S_RIDGE_SHARPE = 1.3383216004229939

PANELS = [
    ("always_short", r"Short volatility (always short)"),
    ("sign_s", r"$\mathrm{sign}(s)$ ($\pm 1$ on the sign of $s$)"),
]

MODEL_TEX = {
    "all models": r"all models (no forecast)",
    "baseline (HAR + calendar OLS)": r"baseline (HAR + calendar OLS)",
    "block-diagonal ridge": r"block-diagonal ridge",
    "block-diagonal ridge, without the FOMC columns": (
        r"block-diagonal ridge, without the FOMC columns"
    ),
    "LightGBM": r"LightGBM$^{\dagger}$",
    "XGBoost": r"XGBoost$^{\dagger}$",
    "lasso (causally tuned)": r"lasso (causally tuned)$^{\dagger}$",
    "lasso (fixed 1e-4)": r"lasso ($\alpha=10^{-4}$)",
    "elastic net (causally tuned)": r"elastic net (causally tuned)$^{\dagger}$",
}

# (csv column, format, tex header) -- the deck's PAPER_COLS, plus the one
# addition at the end.
COLS: list[tuple[str, str, str]] = [
    ("n", "{:.0f}", r"$n$"),
    ("mean", "{:.3f}", r"mean"),
    ("std", "{:.3f}", r"std"),
    ("min", "{:.2f}", r"min"),
    ("max", "{:.2f}", r"max"),
    ("skew", "{:.2f}", r"skew"),
    ("ex_kurt", "{:.2f}", r"ex.\ kurt"),
    ("t_mean", "{:.2f}", r"$t$"),
    ("Sharpe_ann", "{:.2f}", r"Sharpe$_{\mathrm{ann}}$"),
    ("Sharpe_crossed", "{:.2f}", r"Sharpe$^{\times}_{\mathrm{ann}}$"),
]

PREAMBLE = r"""\documentclass{article}
\usepackage[landscape,margin=0.5in]{geometry}
\usepackage{booktabs}
\usepackage{amsmath}
\pagestyle{empty}
"""

FOOTNOTE = r"""\noindent\small One expiration day = one return; mid fill. Every $t$ here is the plain
$t=\sqrt{n}\cdot\mathrm{mean}/\mathrm{std}$, with no autocorrelation correction.
``ex.\ kurt'' is the bias-corrected sample excess kurtosis (Gaussian $=0$).
The short-volatility rule takes no forecast, so it is one row for all models.
Sharpe$_{\mathrm{ann}}=(\mathrm{mean}/\mathrm{std})\times\sqrt{252}$; every other column is daily.
The one column this table adds to the deck's is Sharpe$^{\times}_{\mathrm{ann}}$, the same
rule's annualized Sharpe at the \emph{crossed spread} instead of the midpoint; every other
column is the deck's.

\smallskip
\noindent\small Forecasts: baseline (HAR + calendar OLS); block-diagonal ridge (HAR block and exogenous block, separate penalties)
on the design carrying the FOMC calendar block, and the same ridge on the earlier design without those columns, as a diagnostic row;
LightGBM and XGBoost on the all-features design; lasso on the same design, causally tuned vs.\ fixed $\alpha=10^{-4}$;
elastic net (causally tuned). $\dagger$ marks a column still fitted on the earlier design panel; a run of those four on the
panel of record is pending.
The signal is $s_t=\widehat{RV}_t-\mathrm{IV}^{2}_{\mathrm{hr}}h_t w_t$: the forecast against the
implied variance of the same window. On the bars whose vendor implied volatility is a censored
solver node the deck's treatment is used here too --- the volatility that reproduces the package
midpoint, recovered by bisection over the remaining $h_t$ hours --- so every bar carries a
signal and no bar sits flat."""


# -------------------------------------------------------------- implied ----
def on_vendor_node(v: pd.Series) -> np.ndarray:
    """True where the vendor reported a bracket node rather than a solved vol."""
    v = pd.to_numeric(v, errors="coerce").astype(float)
    return (asl.censor_vendor_iv(v).isna() & v.notna()).to_numpy()


def iv_hourly_reinverted(work: pd.DataFrame) -> pd.Series:
    """Vendor hourly implied volatility, re-inverted on the censored bars.

    The deck's treatment (notebooks/_write_0dte_nb.py, iv_hourly_15_30 in §8):
    a bar whose call or put leg sits on a solver bracket node carries no
    volatility, so recover the one that reproduces the package MIDPOINT by
    bisection and read it back in the vendor's hourly convention.  The deck
    trades only 15:30, where the package covers the last half hour; at an
    intraday clock the package still covers the remaining session, so
    hours_remaining is that stamp's hours_to_expiration -- which the frame
    already carries as h_rem (asserted against data/spxw_chain.parquet).
    """
    iv = work["iv_hourly"].astype(float).copy()
    capped = on_vendor_node(work["impl_volatility_c"]) | on_vendor_node(
        work["impl_volatility_p"]
    )
    cap = capped & (work["entry"].to_numpy(float) > 0)
    idx = np.flatnonzero(cap)
    hrs = work["h_rem"].to_numpy(float)
    chain = pd.read_parquet(
        ROOT / "data" / "spxw_chain.parquet",
        columns=["expiration", "timestamp", "strike", "cp", "hours_to_expiration"],
    )
    chain = chain[chain["cp"] == "C"]
    got = (
        work.iloc[idx][["expiration", "timestamp", "K_c"]]
        .merge(
            chain,
            left_on=["expiration", "timestamp", "K_c"],
            right_on=["expiration", "timestamp", "strike"],
            how="left",
        )["hours_to_expiration"]
        .to_numpy(float)
    )
    assert np.allclose(got, hrs[idx]), (
        "hours_to_expiration disagrees with the clock's hours to the close"
    )
    inv = np.array(
        [
            asl.bsm_invert_package_vol(
                float(work["S"].iloc[i]),
                float(work["K_c"].iloc[i]),
                float(work["K_p"].iloc[i]),
                float(work["entry"].iloc[i]),
                hours_remaining=hrs[i],
            )
            for i in idx
        ]
    )
    rec = np.array(
        [
            asl.hourly_iv_from_total_vol(s, hours_remaining=h)
            for s, h in zip(inv, hrs[idx])
        ]
    )
    iv.iloc[idx] = rec
    print(
        f"censored vendor implied on {len(idx)} bars "
        f"({int((work['hhmm'].to_numpy()[idx] == '15:30').sum())} of them at 15:30); "
        f"re-inverted from the package midpoint on {int(np.isfinite(rec).sum())} of them"
    )
    assert bool(np.isfinite(iv.to_numpy()).all()), (
        "a bar is still without an implied volatility after the re-inversion"
    )
    return iv


# ---------------------------------------------------------------- frame ----
def build_work() -> tuple[pd.DataFrame, list[str], pd.DataFrame]:
    """Trade bars on the deck's 866 days, with the slice and the eight rv_hat."""
    cache = ROOT / "results" / "atm_straddle_intraday" / "cache"
    trade = max(cache.glob("trade_*.parquet"), key=lambda p: p.stat().st_mtime)
    print("trade cache:", trade.name)
    pkg = pd.read_parquet(trade)

    deck = pd.read_parquet(DECK / "daily_blk2.parquet")
    deck.index = pd.to_datetime(deck.index)
    days = pd.DatetimeIndex(deck.index)
    print("deck days", len(days), days.min().date(), "->", days.max().date())

    work = pkg[pkg["date"].isin(days)].copy()
    work["t"] = pd.to_datetime(work["timestamp"], utc=True)
    work = work.sort_values("t").reset_index(drop=True)
    clocks = sorted(work["hhmm"].unique())
    print("bars", len(work), "days", work["date"].nunique(), "clocks", clocks)

    # Causal diurnal profile, seeded from the forecast panel's own history
    # (every in-fit session bar back to 2001), so the frame carries no warm-up.
    pan = asl.load_yhat_panel_mz(asl.yhat_paths(ROOT)["blk2"])
    pf = pan[pan["in_fit"].to_numpy(dtype=bool)].copy()
    clock = pf["et"] - pd.Timedelta(minutes=30)
    pf["pdate"] = clock.dt.normalize().dt.tz_localize(None)
    pf["phhmm"] = clock.dt.strftime("%H:%M")
    prof = pf.pivot_table(
        index="pdate", columns="phhmm", values="rv_raw", aggfunc="mean"
    ).sort_index()
    prof_exp = prof.expanding(min_periods=63).mean().shift(1)
    rem_sum = prof_exp[clocks[::-1]].cumsum(axis=1)[clocks]
    w_slice = prof_exp / rem_sum
    assert bool(np.isclose(w_slice["15:30"].dropna().to_numpy(), 1.0).all()), (
        "w must be 1 at 15:30"
    )
    mi = pd.MultiIndex.from_arrays([work["date"], work["hhmm"]])
    work["w_slice"] = w_slice.stack().reindex(mi).to_numpy()
    assert bool(np.isfinite(work["w_slice"]).all()), "a scored bar has no slice"
    print(
        "diurnal profile on",
        int(prof.index.size),
        "panel sessions",
        prof.index.min().date(),
        "->",
        prof.index.max().date(),
    )

    n_rem = {c: len(clocks) - i for i, c in enumerate(clocks)}
    work["h_rem"] = work["hhmm"].map(n_rem).astype(float) * 0.5
    work["iv_hourly_used"] = iv_hourly_reinverted(work).to_numpy()
    iv2 = work["iv_hourly_used"].astype(float) ** 2
    work["slice"] = iv2 * work["h_rem"] * work["w_slice"]
    at15 = work["hhmm"] == "15:30"
    chk = work.loc[at15]
    dev = float(
        (chk["slice"] / (chk["iv_hourly_used"].astype(float) ** 2 * 0.5) - 1.0)
        .abs()
        .max()
    )
    assert dev == 0.0, f"15:30 slice is not iv_hourly^2/2 (max deviation {dev})"
    print("15:30 slice equals iv_hourly^2/2 exactly on all", len(chk), "days")
    # Against the deck's own iv_var: exact on the quoted days, and on the three
    # re-inverted ones down to the float32 rounding of the package midpoint the
    # two frames store (entry 31.150001 against 31.150000, and so on).
    rel = float(
        (chk.set_index("date")["slice"] / deck["iv_var"].astype(float) - 1.0)
        .abs()
        .max()
    )
    assert rel < 1e-6, (
        f"the 15:30 slice differs from the deck's iv_var by {rel} relative"
    )
    print(
        f"15:30 slice equals the deck's iv_var on all 866 days (max rel diff {rel:.3g})"
    )

    paths = asl.yhat_paths(ROOT)
    for tag in asl.MODEL_ORDER:
        d = asl.load_yhat_panel_mz(paths[tag])[["t", "rv_hat"]].copy()
        d["t"] = pd.to_datetime(d["t"], utc=True) - pd.Timedelta(minutes=30)
        joined = work[["t"]].merge(d, on="t", how="left")["rv_hat"].to_numpy()
        work["rv_hat_" + tag] = joined
        miss = int((~np.isfinite(work["rv_hat_" + tag])).sum())
        print(f"  {tag:9s} joined; bars with no forecast: {miss}")
    return work, clocks, deck


def positions(work: pd.DataFrame, tag: str) -> np.ndarray:
    """sign(s) over the whole frame.

    The q = 0 branch is the guard for a bar with no signal at all; with the
    censored implied re-inverted from the package midpoint and every forecast
    joined, it does not fire on this frame (build_work asserts both).
    """
    s = work["rv_hat_" + tag].to_numpy(float) - work["slice"].to_numpy(float)
    return np.where(s > 0, 1.0, np.where(np.isfinite(s), -1.0, 0.0))


# --------------------------------------------------------------- crossed ----
def crossed_points(work: pd.DataFrame, q: np.ndarray) -> pd.Series:
    """Per-bar P&L in index points at the crossed spread.

    Entry at the ask when long and at the bid when short; exit at the touch of
    the next stamp, except the 15:30 bar, which cash-settles at the official
    close and pays no exit spread.  A bar whose next stamp re-picks the SAME
    two strikes with the same sign is held through: no exit, no re-entry, no
    spread at that boundary.  A bar with no quote on the side actually used (a
    zero fill price) cannot be priced at the spread and is NaN.
    """
    ask_e = (work["ask_c"] + work["ask_p"]).to_numpy(float)
    bid_e = (work["bid_c"] + work["bid_p"]).to_numpy(float)
    ask_x = (work["ask_c_nxt"] + work["ask_p_nxt"]).to_numpy(float)
    bid_x = (work["bid_c_nxt"] + work["bid_p_nxt"]).to_numpy(float)
    is_last = work["is_last"].to_numpy(dtype=bool)
    same_k = (
        (work["K_c"].shift(-1) == work["K_c"])
        & (work["K_p"].shift(-1) == work["K_p"])
        & (work["date"].shift(-1) == work["date"])
        & ~work["is_last"]
    ).to_numpy(dtype=bool)

    q = np.asarray(q, float)
    long, short = q > 0, q < 0
    nxt_q = np.append(q[1:], 0.0)
    hold = same_k & (np.sign(nxt_q) == np.sign(q)) & (q != 0)
    held_in = np.concatenate([[False], hold[:-1]])
    entry = work["entry"].to_numpy(float)
    exit_ = work["exit"].to_numpy(float)
    entry_px = np.where(
        held_in, entry, np.where(long, ask_e, np.where(short, bid_e, entry))
    )
    exit_px = np.where(
        is_last,
        exit_,
        np.where(hold, exit_, np.where(long, bid_x, np.where(short, ask_x, exit_))),
    )
    untradeable = ~held_in & ((long & ~(ask_e > 0)) | (short & ~(bid_e > 0)))
    pts = np.where(untradeable, np.nan, q * (exit_px - entry_px))
    return pd.Series(pts, index=work.index)


# ----------------------------------------------------------------- rows ----
def summary_row(
    work: pd.DataFrame, q: np.ndarray, mask: np.ndarray, pooled: bool
) -> pd.Series:
    """One rule-table row for the bars in `mask`, at the midpoint and crossed."""
    date = work["date"]
    days = pd.DatetimeIndex(sorted(date[mask].unique()))
    qm = np.where(mask, q, 0.0)
    rp = pd.Series(qm * work["R"].to_numpy(float), index=work.index)
    cross = pd.Series(
        crossed_points(work, qm).to_numpy() / work["entry"].to_numpy(float),
        index=work.index,
    )
    if pooled:
        r = rp.groupby(date).sum().reindex(days)
        size = pd.Series(qm, index=work.index).groupby(date).sum().reindex(days)
        rc = cross.groupby(date).sum().reindex(days)
    else:
        idx = np.flatnonzero(mask)
        r = pd.Series(rp.to_numpy()[idx], index=days)
        size = pd.Series(qm[idx], index=days)
        rc = pd.Series(cross.to_numpy()[idx], index=days)
    row = asl.rule_row(r, size)
    x = rc.dropna()
    sd = float(x.std(ddof=1)) if len(x) >= 2 else float("nan")
    row["Sharpe_crossed"] = (
        float(x.mean()) / sd * np.sqrt(asl.PERIODS_PER_YEAR)
        if (sd and sd > 0)
        else float("nan")
    )
    row["n_crossed"] = float(len(x))
    return row


def build_tables(
    work: pd.DataFrame, clocks: list[str]
) -> dict[str, dict[str, pd.DataFrame]]:
    """{window key -> {rule stem -> frame indexed by model label}}."""
    hhmm = work["hhmm"].to_numpy()
    pos = {tag: positions(work, tag) for tag in asl.MODEL_ORDER}
    tables: dict[str, dict[str, pd.DataFrame]] = {}
    for key in [c.replace(":", "") for c in clocks] + ["pooled"]:
        pooled = key == "pooled"
        mask = np.ones(len(work), bool) if pooled else (hhmm == f"{key[:2]}:{key[2:]}")
        short = summary_row(work, -np.ones(len(work)), mask, pooled)
        sign_rows = {
            asl.YHAT_LABEL[tag]: summary_row(work, pos[tag], mask, pooled)
            for tag in asl.MODEL_ORDER
        }
        tables[key] = {
            "always_short": pd.DataFrame({"all models": short}).T,
            "sign_s": pd.DataFrame(sign_rows).T,
        }
    return tables


# ------------------------------------------------------------------ tex ----
def render(tabs: dict[str, pd.DataFrame], source: str) -> list[str]:
    ncol = len(COLS) + 1
    lines = [
        "% AUTO-GENERATED by writeup/make_rule_by_strategy_intraday_tex.py"
        " -- do not edit.",
        f"% Source: {source}",
        r"\begingroup\small\setlength{\tabcolsep}{4.5pt}",
        r"\begin{tabular}{l" + "r" * len(COLS) + "}",
        r"\toprule",
        "& " + " & ".join(h for _, _, h in COLS) + r" \\",
    ]
    for stem, label in PANELS:
        df = tabs[stem]
        lines.append(r"\midrule")
        lines.append(r"\multicolumn{%d}{l}{\emph{%s}} \\" % (ncol, label))
        for name, row in df.iterrows():
            cells = [MODEL_TEX.get(str(name), str(name))]
            for col, fmt, _ in COLS:
                cells.append(fmt.format(float(row[col])))
            lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\endgroup", ""]
    return lines


def window_title(key: str) -> str:
    if key == "pooled":
        return (
            r"0DTE nearest-OTM package: per-rule return summary, pooled over the twelve "
            r"entry windows 10:00--15:30 ET (daily sums), models compared on the same 866 days"
        )
    tail = "cash-settled at the close" if key == "1530" else "one-bar hold"
    return (
        r"0DTE nearest-OTM package: per-rule return summary, entry %s:%s ET, %s, "
        r"models compared on the same 866 days" % (key[:2], key[2:], tail)
    )


def short_title(key: str) -> str:
    if key == "pooled":
        return "Rule table by strategy --- pooled over the twelve entry windows"
    tail = "cash-settled at the close" if key == "1530" else "one-bar hold"
    return f"Rule table by strategy --- entry {key[:2]}:{key[2:]} ET, {tail}"


def data_note(key: str, n_days: int) -> str:
    if key == "pooled":
        held = (
            r"Each row is the daily sum over the twelve entry windows 10:00--15:30 ET: "
            r"one number per expiration day, then the usual statistics on that daily series."
        )
    elif key == "1530":
        held = (
            r"One trade a day: enter the nearest-OTM package at 15:30 ET and let it "
            r"cash-settle at the official close."
        )
    else:
        held = (
            r"One trade a day: enter the nearest-OTM package at %s:%s ET and exit thirty "
            r"minutes later at the next stamp, in the same two strikes."
            % (key[:2], key[2:])
        )
    return (
        r"\noindent\small %s "
        r"%d expiration days, 2020-01-03 to 2024-04-30 (the deck's frame). "
        r"Fills are at the quoted midpoint everywhere except the last column, which is the "
        r"same rule filled at the crossed spread: entry at the touch (the ask when long, the "
        r"bid when short) and exit at the touch of the next stamp; the 15:30 leg cash-settles "
        r"and pays no exit spread, and a re-pick that lands on the same two strikes with the "
        r"same sign is a hold, not a round trip, so it is not charged." % (held, n_days)
    )


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
    print("wrote", path.relative_to(ROOT))


LOG_BAD = ("Undefined", "not found")
AUX_SUFFIXES = (".aux", ".log", ".out")


def compile_tex(stem: str) -> None:
    """One pdflatex pass; the log is scanned, then the aux files are removed."""
    r = subprocess.run(
        ["pdflatex", "-interaction=nonstopmode", stem + ".tex"],
        cwd=WRITEUP,
        capture_output=True,
        text=True,
    )
    pdf = WRITEUP / (stem + ".pdf")
    log = WRITEUP / (stem + ".log")
    text = log.read_text(encoding="utf-8", errors="replace") if log.exists() else ""
    bad = [ln for ln in text.splitlines() if any(b in ln for b in LOG_BAD)]
    assert r.returncode == 0 and pdf.exists(), f"{stem}: pdflatex rc={r.returncode}"
    assert not bad, f"{stem}: log carries {LOG_BAD}:\n" + "\n".join(bad[:5])
    for suf in AUX_SUFFIXES:  # keep only the .tex and the .pdf beside the standalone
        (WRITEUP / (stem + suf)).unlink(missing_ok=True)
    print(f"  pdflatex {stem}: rc=0, pdf written, log clean, aux files removed")


def _pdf_text(path: Path) -> str:
    """Whitespace-collapsed text of a PDF."""
    return " ".join(
        " ".join(p.extract_text() or "" for p in PdfReader(str(path)).pages).split()
    )


def _row_numbers(key: str) -> list[str]:
    """The numeric cells of each data row of one generated tabular, in order."""
    src = (GEN / f"table_rule_by_strategy_intraday_{key}.tex").read_text(
        encoding="utf-8"
    )
    rows = []
    for line in src.splitlines():
        line = line.strip()
        if not line.endswith(r"\\") or line.startswith((r"\multicolumn", "&")):
            continue
        cells = [c.strip() for c in line[:-2].split("&")]
        rows.append(" ".join(cells[1:]))
    return rows


def check_bundle_matches_standalones(keys: list[str]) -> None:
    """Every window's table in the bundle is the one in its own standalone."""
    bundle = _pdf_text(WRITEUP / "rule_by_strategy_intraday_index.pdf")
    for key in keys:
        alone = _pdf_text(WRITEUP / f"rule_by_strategy_intraday_{key}.pdf")
        rows = _row_numbers(key)
        assert len(rows) == 9, (key, len(rows))
        for row in rows:
            assert row in alone, f"{key}: row not rendered in its standalone: {row}"
            assert row in bundle, f"{key}: row not rendered in the bundle: {row}"
    print(
        f"bundle vs standalones: all {len(keys)} windows, "
        f"{9 * len(keys)} data rows, render identically"
    )


# ----------------------------------------------------------------- main ----
def gate(work: pd.DataFrame, deck: pd.DataFrame, tables: dict[str, Any]) -> None:
    """The 15:30 window against the deck's numbers of record.

    With the censored implied re-inverted the way the deck does it, the 15:30
    leg IS the deck's close trade -- same strikes, same entry, same forecast,
    same implied, w = 1 -- so the positions must agree on EVERY day and the
    whole ridge row must reproduce the deck's, not merely come close.
    """
    at15 = (work["hhmm"] == "15:30").to_numpy()
    c15 = work[at15].set_index("date")
    pos = positions(work, "blk2")[at15]
    j = pd.DataFrame({"pos_nb": pos}, index=c15.index).join(deck[["pos"]], how="inner")
    assert len(j) == len(deck), (len(j), len(deck))
    dis = j.index[j["pos_nb"].to_numpy() != j["pos"].to_numpy(float)]
    assert len(dis) == 0, (
        "the 15:30 sign(s) positions disagree with the deck on "
        f"{len(dis)} days: {', '.join(str(d.date()) for d in dis)}"
    )
    print(f"GATE positions: equal to the deck's on all {len(j)} days, none differing")

    got = float(tables["1530"]["always_short"].loc["all models", "Sharpe_ann"])
    assert abs(got - DECK_ALWAYS_SHORT_SHARPE) < 1e-6, (got, DECK_ALWAYS_SHORT_SHARPE)
    print(
        f"GATE always short at 15:30: {got:.9f} vs the deck's "
        f"{DECK_ALWAYS_SHORT_SHARPE:.9f} -- match"
    )
    sgn = float(tables["1530"]["sign_s"].loc["block-diagonal ridge", "Sharpe_ann"])
    assert abs(sgn - DECK_SIGN_S_RIDGE_SHARPE) < 1e-6, (sgn, DECK_SIGN_S_RIDGE_SHARPE)
    print(
        f"GATE sign(s) ridge at 15:30: {sgn:.9f} vs the deck's "
        f"{DECK_SIGN_S_RIDGE_SHARPE:.9f} -- match"
    )

    # The whole 15:30 panel against the deck's own CSV, every shared column.
    cols = [c for c, _, _ in COLS if c != "Sharpe_crossed"]
    for stem, csv in (("sign_s", "sign_s"), ("always_short", "always_short")):
        ref = pd.read_csv(DECK / f"rule_by_strategy_{csv}.csv", index_col=0)
        mine = tables["1530"][stem]
        assert list(mine.index) == list(ref.index), (list(mine.index), list(ref.index))
        dev = float((mine[cols] - ref[cols]).abs().to_numpy().max())
        assert dev < 1e-6, f"the 15:30 {stem} panel differs from the deck's by {dev}"
        print(f"GATE 15:30 {stem} panel vs the deck's CSV: max |diff| {dev:.3g}")


def main() -> None:
    work, clocks, deck = build_work()
    tables = build_tables(work, clocks)
    gate(work, deck, tables)

    keys = [c.replace(":", "") for c in clocks] + ["pooled"]
    for key in keys:
        for stem, _ in PANELS:
            dst = OUT / key / f"rule_by_strategy_{stem}.csv"
            dst.parent.mkdir(parents=True, exist_ok=True)
            tables[key][stem].to_csv(dst)
        src = (
            f"results/atm_straddle_intraday/rule_by_strategy/{key}/"
            f"rule_by_strategy_*.csv"
        )
        write(
            GEN / f"table_rule_by_strategy_intraday_{key}.tex",
            "\n".join(render(tables[key], src)),
        )
        n_days = int(tables[key]["always_short"].loc["all models", "n"])
        body = "\n".join(
            [
                f"% Standalone render of generated_intraday/"
                f"table_rule_by_strategy_intraday_{key}.tex",
                f"% Build: pdflatex -interaction=nonstopmode "
                f"rule_by_strategy_intraday_{key}.tex",
                PREAMBLE,
                r"\begin{document}",
                r"\begin{center}",
                r"\textbf{%s}\\[1.5ex]" % window_title(key),
                r"\input{generated_intraday/table_rule_by_strategy_intraday_%s}" % key,
                r"\end{center}",
                r"\vspace{1ex}",
                data_note(key, n_days),
                r"\vspace{1ex}",
                FOOTNOTE,
                r"\end{document}",
                "",
            ]
        )
        write(WRITEUP / f"rule_by_strategy_intraday_{key}.tex", body)
        compile_tex(f"rule_by_strategy_intraday_{key}")

    # The bundle: the same thirteen generated tabulars, in order, one section
    # each with its own data note; the shared footnote once, at the end.
    idx = [
        "% AUTO-GENERATED by writeup/make_rule_by_strategy_intraday_tex.py.",
        "% Build: pdflatex -interaction=nonstopmode rule_by_strategy_intraday_index.tex",
        PREAMBLE,
        r"\begin{document}",
        r"\begin{center}\textbf{\large 0DTE nearest-OTM package: rule table by strategy,",
        r"every intraday entry window}\end{center}",
    ]
    for key in keys:
        n_days = int(tables[key]["always_short"].loc["all models", "n"])
        idx += [
            r"\section*{%s}" % short_title(key),
            r"\begin{center}",
            r"\input{generated_intraday/table_rule_by_strategy_intraday_%s}" % key,
            r"\end{center}",
            data_note(key, n_days),
            r"\clearpage",
        ]
    idx += [
        r"\section*{Notes}",
        FOOTNOTE,
        r"\end{document}",
        "",
    ]
    write(WRITEUP / "rule_by_strategy_intraday_index.tex", "\n".join(idx))
    compile_tex("rule_by_strategy_intraday_index")
    pages = len(PdfReader(str(WRITEUP / "rule_by_strategy_intraday_index.pdf")).pages)
    print(f"  bundle rule_by_strategy_intraday_index.pdf: {pages} pages")
    check_bundle_matches_standalones(keys)

    print()
    print("sign(s) Sharpe by forecast -- 15:30 and pooled, mid and crossed")
    rep = pd.DataFrame(
        {
            "15:30 mid": tables["1530"]["sign_s"]["Sharpe_ann"],
            "15:30 crossed": tables["1530"]["sign_s"]["Sharpe_crossed"],
            "pooled mid": tables["pooled"]["sign_s"]["Sharpe_ann"],
            "pooled crossed": tables["pooled"]["sign_s"]["Sharpe_crossed"],
        }
    )
    print(rep.to_string(float_format=lambda x: f"{x:+.4f}"))
    print()
    print("always short, Sharpe mid / crossed")
    ash = pd.DataFrame(
        {
            "15:30": tables["1530"]["always_short"].loc["all models"],
            "pooled": tables["pooled"]["always_short"].loc["all models"],
        }
    ).T[["n", "mean", "std", "t_mean", "Sharpe_ann", "Sharpe_crossed"]]
    print(ash.to_string(float_format=lambda x: f"{x:+.4f}"))


if __name__ == "__main__":
    main()
