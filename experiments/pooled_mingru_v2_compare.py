"""MinGRU v2 (RULE_gru_v2.txt ADDENDUM 2): full set against the HAR set within the MinGRU, by day type, beside Table 1's
six classes -- the comparison worker of the v2 scoring (2026-10-28; author: merge, seed-average and score the finished
v2 walks of the four arms, CPU only, nothing in the paper).

STATUS OF THE INPUTS  The HAR-set walks are final.  The full-set walks ran on the PRE-FOMC-FIX design caches (pmf_<set>,
         exported 2026-10-22, before the FOMC patch D13): every full-set number here is a PRE-FOMC-FIX PREVIEW.  Its
         apples-to-apples context is the six classes' PRE-PATCH full sets; the patched rows are shown beside them.

SCORED SET  Table 1's 33,527 bars on 2,579 days (2014-01-02 .. 2024-04-30, a positive 16:00 print, early closes dropped,
         every bar with a finite positive forecast of Table 1's 13 rows), built from each context run exactly as
         writeup/make_rehaul_channels.py builds it (asserted equal across the runs); the MinGRU must forecast every one
         of those bars.  plain = pred_adj^2 B; QLIKE = r - ln r - 1, r = RV / plain, RV = data/core_stats.parquet sumret2
         (asserted = rv_actual); squared error alongside.
TESTS    full vs HAR (a = full, b = HAR; negative = the full set more accurate): % change of the mean loss and
         src.evaluation.diebold_mariano.dm_test (HLN) on the daily mean of the day's bar losses, Newey-West lag 8 (the
         channels script's HAC_LAG, every span), two-sided; the automatic lag (3 on the 81 EVENT days) alongside.
         Spans: every bar; the 16:00 bar; ORDINARY days; EVENT days.  Benchmarks: the OLS on the HAR set (D2) for ridge,
         lasso, elastic net (as Section 6); each tree's and the MinGRU's own HAR-set fit.
DAY TYPES EVENT = the 81 scheduled FOMC statement days of the span (make_rehaul_channels.statement_stamps' rule, the
         statement in the bar ending 14:30); ORDINARY = the other 2,498.  Statement afternoons = the EVENT bars ending
         14:30, 15:00, 15:30, 16:00.  Gain share (Section 6.2, make_rehaul_channels item 2): 100 x sum over the EVENT
         afternoon bars of (HAR loss - full loss) / the same sum over every bar.
ECONOMICS  The sign strategy of Section 7 on the 866 option days (make_rehaul_economics.load / add_pooled_linear: the
         OLS row's days, iv_var, rv_unclipped; the GUI quote book's straddle; q = +1 if the 16:00 plain > iv_var else
         -1; mid = q R; bid / ask = master_table_gui.rows._crossed_return), the paper's Sharpe ratio (D19,
         hac_inference.paper_sharpe) on the 614 post-warmup days, every difference tested by Ledoit-Wolf
         (hac_inference.paper_sharpe_test, block 10, 4,999 resamples; the paired block bootstrap p alongside); the
         short straddle = make_rehaul_economics' always_short (mid -R, bid / ask 1 - exit / bid).
GATES    (asserted; gates.json) patched W 3,000 against results/channels_2026-10-26 (item3ii joint rows, item2 shares)
         and results/ablations_2014start_w3000/scores_fomcfix; pre-patch W 3,000 against .../scores and
         results/rehaul_robust_window_final; W 3,500 against results/rehaul_robust_window_final (accuracy and every
         full - HAR Sharpe row).  The arms' own scores / econ files are compared where their columns allow.

usage: python pooled_mingru_v2_compare.py <main tree (read-only)> <mgv2 scratch> <out results dir> [W ...]
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
sys.dont_write_bytecode = True  # the main tree is read-only: no __pycache__ there
MAIN = Path(sys.argv[1]).resolve()
for _p in (MAIN, MAIN / "notebooks", MAIN / "writeup"):
    sys.path.insert(0, str(_p))

import atm_straddle_lib as asl  # noqa: E402
import hac_inference as hi  # noqa: E402
from master_table_gui import rows as gui  # noqa: E402

from src.data import loading  # noqa: E402
from src.evaluation.diebold_mariano import dm_test  # noqa: E402

for _m in (asl, hi, gui, loading):
    assert Path(str(_m.__file__)).resolve().is_relative_to(MAIN), _m.__file__
MG = Path(sys.argv[2])
OUTROOT = Path(sys.argv[3])
WINDOWS = tuple(int(x) for x in sys.argv[4:]) or (3000, 3500)
RES = MAIN / "results"
FIRST, LAST = pd.Timestamp("2014-01-02"), pd.Timestamp("2024-04-30")
N_BARS, N_DAYS, N_EVENT, BARS_A_DAY, HAC_LAG = 33527, 2579, 81, 13, 8
PM_FROM = 14 * 60 + 30
LIN, TREES = ("ridge", "lasso", "enet"), ("lgbm", "xgb", "rf")
CLASSES = LIN + TREES
SET_ARM = {"all_features": "full", "baseline": "har"}
SPANS = ("all", "c16", "c16_2020", "ord", "event")
SUB_FIRST, SUB_DAYS = pd.Timestamp("2020-01-03"), 1082  # Table 1's printed 16:00 QLIKE: 2020-01-03 .. 2024-04-30
PREVIEW = "PRE-FOMC-FIX PREVIEW"
RUNS = {
    (3000, "patched"): (
        "window_sweep_2014start_fomcfix/w3000",
        "pooled_trees_final_2014start_fomcfix_w3000",
        "pooled_rf_skx_2014start_fomcfix_w3000",
    ),
    (3000, "pre-patch"): (
        "window_sweep_2014start_2026-10-22/w3000",
        "pooled_trees_final_2014start_w3000",
        "pooled_rf_skx_2014start_w3000",
    ),
    (3500, "pre-patch"): (
        "window_sweep_2014start_2026-10-22/w3500",
        "pooled_trees_final_2014start_w3500",
        "pooled_rf_skx_2014start_w3500",
    ),
}
ARMS = {"baseline": "har", "all_features": "full"}
t0 = time.time()
GATES: dict = {"fail": [], "checked": [], "notes": []}


def say(msg: str) -> None:
    print(f"[{time.time() - t0:6.0f} s] {msg}", flush=True)


def chk(ok: bool, msg: str) -> None:
    (GATES["checked"] if ok else GATES["fail"]).append(msg)
    if not ok:
        say(f"GATE FAIL: {msg}")


def close(a: float, b: float, rtol: float) -> bool:
    return bool(np.isclose(float(a), float(b), rtol=rtol, atol=0.0))


# ---------------------------------------------------------------------------------------------------- data
def core_rv() -> pd.Series:
    c = pd.read_parquet(MAIN / "data" / "core_stats.parquet", columns=["endbartime", "sumret2"])
    s = pd.Series(c["sumret2"].to_numpy(float), index=pd.DatetimeIndex(pd.to_datetime(c["endbartime"])))
    return s[~s.index.duplicated()].sort_index()


def evaluation_days(rv: pd.Series) -> pd.DatetimeIndex:
    """make_rehaul_channels.evaluation_days: a positive 16:00 print, early closes dropped."""
    early = pd.DatetimeIndex(pd.to_datetime(list(asl.EARLY_CLOSE_DATES)))
    s = rv[(rv.index.hour == 16) & (rv.index.minute == 0)]
    d = pd.DatetimeIndex(s.index.normalize())
    keep = (s.to_numpy() > 0) & np.isfinite(s.to_numpy()) & ~d.isin(early)
    days = d[keep]
    days = days[(days >= FIRST) & (days <= LAST)]
    assert len(days) == N_DAYS and days[0] == FIRST and days[-1] == LAST, (len(days), days[0], days[-1])
    return days


def statement_stamps() -> pd.DatetimeIndex:
    """make_rehaul_channels.statement_stamps: the scheduled FOMC statements (bar end stamps)."""
    rel = pd.read_parquet(MAIN / "data" / "releases.parquet", columns=["endbartime", loading.FOMC_RAW_COL])
    t = pd.DatetimeIndex(pd.to_datetime(rel["endbartime"]))
    raw = np.nan_to_num(pd.to_numeric(rel[loading.FOMC_RAW_COL], errors="coerce").to_numpy(float), nan=0.0)
    st = t[raw > 0].append(pd.DatetimeIndex(pd.to_datetime(list(loading.FOMC_MISSING_STATEMENTS))))
    st = st[~pd.DatetimeIndex(st.normalize()).isin(list(loading.UNSCHEDULED_FOMC_DAYS))]
    return st.unique().sort_values()


def long_panel(path: Path, models: dict[str, str]) -> dict:
    """(class, arm) -> (pred_adj, B, rv_actual, plain) Series from a long forecasts file (make_rehaul_channels.long_panel
    plus the file's plain, checked against pred_adj^2 B)."""
    cols = ["model", "predictor_set", "date", "pred_adj", "B", "rv_actual", "plain"]
    d = pd.read_parquet(path, columns=cols)
    d = d[d["model"].isin(list(models)) & (pd.to_datetime(d["date"]) >= FIRST)]
    out = {}
    for (m, s), f in d.groupby(["model", "predictor_set"]):
        ix = pd.DatetimeIndex(pd.to_datetime(f["date"])).as_unit("ns")
        assert ix.is_unique, (path, m, s)
        out[(models[m], SET_ARM[s])] = tuple(
            pd.Series(f[c].to_numpy(float), index=ix) for c in ("pred_adj", "B", "rv_actual", "plain")
        )
    return out


def context_run(W: int, design: str) -> dict:
    lin, tr, rf = (RES / x / "forecasts.parquet" for x in RUNS[(W, design)])
    raw: dict = {}
    raw.update(long_panel(lin, {"ridge": "ridge", "lasso": "lasso", "enet": "enet", "ols": "ols"}))
    raw.update(long_panel(tr, {"lgbm": "lgbm", "xgb": "xgb"}))
    raw.update(long_panel(rf, {"rf_skx": "rf"}))
    assert len(raw) == 13, sorted(raw)
    return raw


# ---------------------------------------------------------------------------------------------------- scoring
class Scorer:
    """make_rehaul_channels.Scorer with the 16:00 bar as a span of its own and the automatic-lag DM alongside."""

    def __init__(self, idx: pd.DatetimeIndex, rv: np.ndarray, event_days: pd.DatetimeIndex):
        self.idx, self.rv = idx, rv
        day = pd.DatetimeIndex(idx.normalize())
        self.days, self.code = np.unique(day.to_numpy(), return_inverse=True)
        self.is_event_day = pd.DatetimeIndex(self.days).isin(event_days)
        ev = self.is_event_day[self.code]
        mins = np.asarray(idx.hour * 60 + idx.minute)
        c16 = mins == 16 * 60
        self.mask = {
            "all": np.ones(len(idx), bool),
            "c16": c16,
            "c16_2020": c16 & np.asarray(day >= SUB_FIRST),
            "event": ev,
            "ord": ~ev,
        }
        assert int(self.mask["c16"].sum()) == N_DAYS and int(self.mask["c16_2020"].sum()) == SUB_DAYS
        self.pm = mins >= PM_FROM
        self.L: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}

    def add(self, key: tuple[str, str], plain: np.ndarray) -> None:
        assert np.isfinite(plain).all() and (plain > 0).all(), key
        r = self.rv / plain
        self.L[key] = (r - np.log(r) - 1.0, (self.rv - plain) ** 2)

    def daily(self, x: np.ndarray, sp: str) -> np.ndarray:
        m = self.mask[sp]
        s = np.bincount(self.code[m], weights=x[m], minlength=len(self.days))
        n = np.bincount(self.code[m], minlength=len(self.days))
        return s[n > 0] / n[n > 0]

    def mean(self, key: tuple[str, str], sp: str, loss: int = 0) -> float:
        return float(self.L[key][loss][self.mask[sp]].mean())

    def comp(self, a: tuple[str, str], b: tuple[str, str], sp: str) -> dict:
        out: dict = dict(span=sp, n_bars=int(self.mask[sp].sum()), n_days=int(len(self.daily(self.L[a][0], sp))))
        for li, nm in ((0, "qlike"), (1, "mse")):
            xa, xb = self.L[a][li], self.L[b][li]
            da, db = self.daily(xa, sp), self.daily(xb, sp)
            r = dm_test(da, db, hac_lag=HAC_LAG)
            ra = dm_test(da, db)
            qa, qb = float(xa[self.mask[sp]].mean()), float(xb[self.mask[sp]].mean())
            out.update(
                {
                    f"{nm}_full": qa,
                    f"{nm}_har": qb,
                    f"pct_{nm}": 100.0 * (qa / qb - 1.0),
                    f"dm_{nm}": float(r["dm"]),
                    f"p_{nm}": float(r["p"]),
                    f"dm_{nm}_autolag": float(ra["dm"]),
                    f"p_{nm}_autolag": float(ra["p"]),
                }
            )
            assert int(r["hac_lag"]) == HAC_LAG
            out["autolag"] = int(ra["hac_lag"])
        out["hac_lag"] = HAC_LAG
        out["gain_sum"] = float((self.L[b][0] - self.L[a][0])[self.mask[sp]].sum())
        return out

    def shares(self, a: tuple[str, str], b: tuple[str, str]) -> dict:
        """make_rehaul_channels item 2 (Section 6.2): shares of the every-bar gain sum (HAR loss - full loss)."""
        gain = self.L[b][0] - self.L[a][0]
        tot = float(gain.sum())
        ev = float(gain[self.mask["event"]].sum())
        evpm = float(gain[self.mask["event"] & self.pm].sum())
        ordpm = float(gain[self.mask["ord"] & self.pm].sum())
        return dict(
            total_gain_sum=tot,
            event_gain_sum=ev,
            event_pm_gain_sum=evpm,
            share_event_pct=100.0 * ev / tot,
            share_event_pm_pct=100.0 * evpm / tot,
            share_ord_pct=100.0 * (tot - ev) / tot,
            share_ord_pm_pct=100.0 * ordpm / tot,
            event_bars_pct=100.0 * self.mask["event"].mean(),
            event_pm_bars_pct=100.0 * (self.mask["event"] & self.pm).mean(),
        )


def har_of(c: str) -> tuple[str, str]:
    return ("ols", "har") if c in LIN else (c, "har")


# ---------------------------------------------------------------------------------------------------- MinGRU arms
SPAN_ALIASES = {
    "all": ("every_bar", "all", "bar"),
    "c16": ("bar_1600", "1600", "c16", "close"),
    "c16_2020": ("bar_1600_2020_2024", "1600_2020", "c16_2020"),
    "ord": ("ordinary_days", "ord", "ordinary"),
    "event": ("event_days", "event", "ev"),
    "event_pm": ("event_1430_1600", "event_pm", "event_pm_1430_1600"),
}


def arm_score_row(a: pd.DataFrame, sp: str) -> pd.Series | None:
    """The row of span sp in an arm worker's scores file (span names differ between workers)."""
    if "span" not in a.columns or "qlike" not in a.columns or "mse" not in a.columns:
        return None
    x = a[a["span"].astype(str).isin(SPAN_ALIASES[sp])]
    return x.iloc[0] if len(x) == 1 else None


def arm_econ_row(a: pd.DataFrame, col: str) -> tuple[float, float, float] | None:
    """(D19 ratio, short straddle's D19 ratio, LW p of the difference) of side col (mid / x) from an arm worker's econ
    file: long (a prices / fills column) or wide (sr_mid / sr_ba ...) layouts."""
    mid = col == "mid"
    for side_col in ("prices", "fills", "side"):
        if side_col in a.columns:
            want = ("mid",) if mid else ("bid/ask", "bid_ask", "ba", "x")
            x = a[a[side_col].astype(str).isin(want)]
            if len(x) != 1:
                return None
            r = x.iloc[0]
            for sr, ss, pp in (
                ("sr_sign_d19", "sr_short_d19", "p_lw"),
                ("sharpe_d19", "sharpe_d19_short", "p_lw_vs_short"),
            ):
                if {sr, ss, pp} <= set(a.columns):
                    return float(r[sr]), float(r[ss]), float(r[pp])
            return None
    tag = "mid" if mid else "ba"
    if {"key", f"sr_{tag}", f"lw_p_{tag}_vs_short"} <= set(a.columns):  # one row a key, the short straddle's own row
        me = a[a["key"].astype(str).str.startswith("mingru")]
        sh = a[a["key"].astype(str) == "always_short"]
        if len(me) != 1 or len(sh) != 1:
            return None
        return (
            float(me.iloc[0][f"sr_{tag}"]),
            float(sh.iloc[0][f"sr_{tag}"]),
            float(me.iloc[0][f"lw_p_{tag}_vs_short"]),
        )
    cols = (f"sr_{tag}", f"sr_{tag}_short", f"p_lw_{tag}_vs_short")
    if set(cols) <= set(a.columns) and len(a) == 1:
        return tuple(float(a.iloc[0][c]) for c in cols)  # type: ignore[return-value]
    return None


def arm_forecasts(W: int, s: str) -> tuple[pd.DataFrame | None, Path | None]:
    d = MG / f"w{W}_{s}" / "out"
    cands = [p for p in sorted(d.glob("forecasts_*.parquet")) if "seed" not in p.name]
    if not cands:
        return None, None
    p = cands[0]
    f = pd.read_parquet(p)
    if "predictor_set" in f.columns:
        f = f[f["predictor_set"] == s]
    if "model" in f.columns:
        assert set(f["model"]) == {"mingru"}, set(f["model"])
    return f, p


# ---------------------------------------------------------------------------------------------------- economics
class Econ:
    """The sign strategy of Section 7 on the 866 option days (make_rehaul_economics.load / add_pooled_linear)."""

    def __init__(self) -> None:
        daily = pd.read_parquet(
            MAIN / "results" / "close_master_table" / "master_table_daily.parquet",
            columns=["key", "day", "iv_var", "rv_unclipped"],
        )
        daily = daily[daily["key"] == "sub_ols_baseline"].copy()
        daily["day"] = pd.to_datetime(daily["day"]).dt.normalize().astype("datetime64[ns]")
        ref = daily.set_index("day").sort_index()
        q = pd.read_parquet(gui._QUOTE_BOOK, columns=["R", "exit", "bid_c", "bid_p", "ask_c", "ask_p"])
        q.index = pd.to_datetime(q.index).normalize().astype("datetime64[ns]")
        book = pd.DataFrame(
            {
                "R": q["R"].astype(float),
                "ex": q["exit"].astype(float),
                "bid": (q["bid_c"] + q["bid_p"]).astype(float),
                "ask": (q["ask_c"] + q["ask_p"]).astype(float),
            }
        )
        g = ref.join(book, how="inner")
        assert len(g) == 866, len(g)
        self.days = pd.DatetimeIndex(g.index)
        self.pw = self.days[gui._SIZING_WARMUP :]
        assert len(self.pw) == 614
        self.iv = g["iv_var"].to_numpy(float)
        self.rv = g["rv_unclipped"].to_numpy(float)
        self.R, self.ex, self.bid, self.ask = (g[c].to_numpy(float) for c in ("R", "ex", "bid", "ask"))
        assert np.isfinite(self.iv).all() and np.isfinite(self.rv).all()
        self.ret: dict[str, dict[str, Any]] = {
            "short": {"mid": -self.R, "x": 1.0 - self.ex / self.bid},
        }

    def plain16(self, plain: pd.Series, rv: pd.Series, key: str) -> np.ndarray:
        at = plain[(plain.index.hour == 16) & (plain.index.minute == 0)]
        rr = rv.reindex(at.index)
        at.index = at.index.normalize()
        rr.index = rr.index.normalize()
        p = at.reindex(self.days).to_numpy(float)
        assert np.isfinite(p).all() and (p > 0).all(), (key, int((~np.isfinite(p)).sum()))
        assert np.allclose(rr.reindex(self.days).to_numpy(float), self.rv, rtol=1e-12, atol=0.0), key
        return p

    def add(self, key: str, p: np.ndarray) -> None:
        q = np.where(p > self.iv, 1.0, -1.0)
        self.ret[key] = {
            "mid": q * self.R,
            "x": gui._crossed_return(q, self.ex, self.bid, self.ask),
            "q": q,
            "qlike866": float(np.mean(self.rv / p - np.log(self.rv / p) - 1.0)),
        }

    def pwv(self, key: str, col: str) -> np.ndarray:
        return np.asarray(self.ret[key][col], float)[gui._SIZING_WARMUP :]

    def sr(self, key: str, col: str) -> float:
        return float(hi.paper_sharpe(self.pwv(key, col), dates=self.pw)["sharpe"])

    def test(self, a: str, b: str, col: str) -> dict:
        r = hi.paper_sharpe_test(self.pwv(a, col), self.pwv(b, col), self.pw)
        return dict(sr_a=r["sr_a"], sr_b=r["sr_b"], delta=r["delta"], p_lw=r["p_lw"], p_boot=r["p_boot"], T=r["T"])


# ---------------------------------------------------------------------------------------------------- tables
SPN = {
    "all": "every bar",
    "c16": "16:00 bar",
    "c16_2020": "16:00 bar 2020-24 (Table 1's column)",
    "ord": "ORDINARY days",
    "event": "EVENT days (81)",
}
MNAME = {
    "ridge": "Ridge",
    "lasso": "Lasso",
    "enet": "Elastic net",
    "lgbm": "LightGBM",
    "xgb": "XGBoost",
    "rf": "Random forest",
    "mingru": "MinGRU v2",
}


def fp(p: float) -> str:
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def cell(r: pd.Series, loss: str = "qlike") -> str:
    return f"{r[f'pct_{loss}']:+.2f} ({r[f'dm_{loss}']:+.2f}, {fp(r[f'p_{loss}'])})"


def tables_md(W: int, A: pd.DataFrame, SHx: pd.DataFrame, LV: pd.DataFrame, E: pd.DataFrame, short: dict) -> str:
    """The tables of SUMMARY.md (generated; every MinGRU full-set number labelled PRE-FOMC-FIX PREVIEW)."""
    L: list[str] = []
    m = A[A.model == "mingru"]
    if len(m):
        L += [
            f"### MinGRU v2, W = {W:,}: full set vs HAR set ({PREVIEW} for every full-set number)",
            "",
            "Negative % / DM = the full set more accurate. DM: HLN, daily mean loss, NW lag 8 (EVENT: automatic lag "
            "in brackets).",
            "",
            "| span | QLIKE full [PREVIEW] | QLIKE HAR | % QLIKE (DM, p) [PREVIEW] | % MSE (DM, p) [PREVIEW] |",
            "|---|---|---|---|---|",
        ]
        for sp in SPANS:
            r = m[m.span == sp].iloc[0]
            extra = f" [lag {int(r['autolag'])}: {r['dm_qlike_autolag']:+.2f}, {fp(r['p_qlike_autolag'])}]"
            L.append(
                f"| {SPN[sp]} | {r['qlike_full']:.5f} | {r['qlike_har']:.5f} | {cell(r)}"
                f"{extra if sp == 'event' else ''} | {cell(r, 'mse')} |"
            )
        s = SHx[SHx.model == "mingru"].iloc[0]
        L += [
            "",
            f"Gain shares [{PREVIEW}] (Section 6.2's definition; sum over bars of HAR loss - full loss = "
            f"{s['total_gain_sum']:.2f}): EVENT days {s['share_event_pct']:.1f} %, EVENT afternoons (bars ending "
            f"14:30-16:00) {s['share_event_pm_pct']:.1f} %, ORDINARY days {s['share_ord_pct']:.1f} % "
            f"(EVENT afternoons hold {s['event_pm_bars_pct']:.2f} % of the bars).",
            "",
        ]
    else:
        L += [f"### MinGRU v2, W = {W:,}: an arm is missing -- no full vs HAR comparison", ""]
    L += [
        f"### Context, W = {W:,}: full vs HAR, % QLIKE (DM, p); linear classes against the OLS on the HAR set (D2)",
        "",
        "| model | design | every bar | 16:00 | 16:00 2020-24 | ORDINARY | EVENT | EVENT pm share % |",
        "|---|---|---|---|---|---|---|---|",
    ]
    order = [(c, d) for d in ("patched", "pre-patch") for c in CLASSES] + [("mingru", "v2")]
    for c, d in order:
        x = A[(A.model == c) & (A.design == d)]
        if not len(x):
            continue
        sh = SHx[(SHx.model == c) & (SHx.design == d)].iloc[0]
        lab = f"v2 [{PREVIEW}]" if c == "mingru" else d
        cells = " | ".join(cell(x[x.span == sp].iloc[0]) for sp in SPANS)
        L.append(f"| {MNAME[c]} | {lab} | {cells} | {sh['share_event_pm_pct']:.1f} |")
    L += [
        "",
        f"### QLIKE levels, W = {W:,} (every bar / 16:00)",
        "",
        "| model | design | full every bar | full 16:00 | HAR every bar | HAR 16:00 |",
        "|---|---|---|---|---|---|",
    ]
    for c, d in [(c, d) for d in ("patched", "pre-patch") for c in ("ols",) + CLASSES] + [("mingru", "mingru")]:
        x = LV[(LV.model == c) & (LV.design == d)]
        if not len(x):
            continue
        f_ = x[x.arm == "full"]
        h_ = x[x.arm == "har"]
        fa = f"{f_.iloc[0]['qlike_all']:.5f} | {f_.iloc[0]['qlike_c16']:.5f}" if len(f_) else "- | -"
        ha = f"{h_.iloc[0]['qlike_all']:.5f} | {h_.iloc[0]['qlike_c16']:.5f}" if len(h_) else "- | -"
        lab = f"v2 (full: {PREVIEW})" if c == "mingru" else d
        L.append(f"| {MNAME.get(c, 'OLS')} | {lab} | {fa} | {ha} |")
    if len(E):
        L += [
            "",
            f"### Economics, W = {W:,}: the sign strategy, paper Sharpe ratio (D19) on the 614 post-warmup days; "
            "differences tested by Ledoit-Wolf (p; paired block bootstrap p_boot alongside)",
            "",
            f"Short straddle: mid {short['mid']:.3f}, bid/ask {short['x']:.3f}.",
            "",
            "| model | design | test | side | SR a | SR b | delta | p (LW) | p_boot |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        for _, r in E.iterrows():
            lab = r["design"] + (f" [{r['label']}]" if r["label"] else "")
            L.append(
                f"| {MNAME[r['model']]} | {lab} | {r['test']} | {'mid' if r['side'] == 'mid' else 'bid/ask'} | "
                f"{r['sr_a']:.3f} | {r['sr_b']:.3f} | {r['delta']:+.3f} | {fp(r['p_lw'])} | {fp(r['p_boot'])} |"
            )
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------------------------------------------- main
def main() -> None:
    rv_core = core_rv()
    days = evaluation_days(rv_core)
    st = statement_stamps()
    raws = {k: context_run(*k) for k in RUNS if k[0] in WINDOWS}
    say(f"context runs read: {sorted(raws)}")

    # the scored set of every context run (make_rehaul_channels), asserted equal across runs
    idx = None
    for k, raw in raws.items():
        ii = None
        for key, (p, b, _, pl) in raw.items():
            plain = p.to_numpy() ** 2 * b.to_numpy()
            ok = np.isfinite(plain) & (plain > 0)
            chk(
                bool(np.allclose(plain[ok], pl.to_numpy()[ok], rtol=1e-12, atol=0)),
                f"{k} {key}: plain = pred_adj^2 B",
            )
            good = p.index[ok]
            ii = good if ii is None else ii.intersection(good)
        assert ii is not None
        ii = ii[pd.DatetimeIndex(ii.normalize()).isin(days)]
        t = rv_core.reindex(ii).to_numpy()
        ii = ii[(t > 0) & np.isfinite(t)].sort_values()
        per_day = pd.Series(1, index=ii.normalize()).groupby(level=0).size()
        chk(len(ii) == N_BARS and per_day.size == N_DAYS and bool((per_day == BARS_A_DAY).all()), f"{k}: scored set")
        if idx is None:
            idx = ii
        chk(bool(ii.equals(idx)), f"{k}: scored set equal across runs")
    assert idx is not None
    rv = rv_core.reindex(idx).to_numpy(float)
    per_day = pd.Series(1, index=idx.normalize()).groupby(level=0).size()
    st_span = st[pd.DatetimeIndex(st.normalize()).isin(per_day.index)]
    event_days = pd.DatetimeIndex(st_span.normalize().unique())
    chk(len(event_days) == N_EVENT, f"EVENT days {len(event_days)}")
    chk(sorted(set(st_span.strftime("%H:%M"))) == ["14:30"], "statement stamps 14:30")
    S = Scorer(idx, rv, event_days)
    GATES.update(
        n_bars=len(idx),
        n_days=int(per_day.size),
        n_event_days=len(event_days),
        n_event_bars=int(S.mask["event"].sum()),
        n_event_pm_bars=int((S.mask["event"] & S.pm).sum()),
    )
    B0 = None
    for k, raw in raws.items():
        for key, (p, b, ra, _) in raw.items():
            chk(bool(np.allclose(ra.reindex(idx).to_numpy(), rv, rtol=1e-12, atol=0)), f"{k} {key}: rv_actual = RV")
            bb = b.reindex(idx).to_numpy(float)
            if B0 is None:
                B0 = bb
            chk(bool(np.array_equal(bb, B0)), f"{k} {key}: B equal to the first template's")
            S.add((f"{k[0]}|{k[1]}|{key[0]}", key[1]), p.reindex(idx).to_numpy(float) ** 2 * bb)
    assert B0 is not None
    del raws

    def ctx(W: int, design: str, c: str) -> tuple[str, str]:
        return (f"{W}|{design}|{c}", "full")

    def ctx_h(W: int, design: str, c: str, own: bool = False) -> tuple[str, str]:
        h = (c, "har") if own else har_of(c)
        return (f"{W}|{design}|{h[0]}", "har")

    # ------------------------------------------------------------------------------------------- context comparisons
    rows, shrows = [], []
    for W, design in RUNS:
        if W not in WINDOWS:
            continue
        for c in CLASSES:
            for sp in SPANS:
                rows.append(
                    dict(
                        W=W,
                        model=c,
                        design=design,
                        benchmark="ols|har" if c in LIN else f"{c}|har",
                        label="",
                        **S.comp(ctx(W, design, c), ctx_h(W, design, c), sp),
                    )
                )
            shrows.append(
                dict(W=W, model=c, design=design, label="", **S.shares(ctx(W, design, c), ctx_h(W, design, c)))
            )
    CT = pd.DataFrame(rows)
    SH = pd.DataFrame(shrows)

    # ------------------------------------------------------------------------------------------- gates (context)
    chg = RES / "channels_2026-10-26"
    if 3000 in WINDOWS:
        j = pd.read_csv(chg / "item3ii_ablation_daytype.csv")
        j = j[j["kind"] == "joint"]
        for c in CLASSES:
            for sp in ("all", "event", "ord"):
                r = j[(j["cls"] == c) & (j["span"] == sp)].iloc[0]
                x = CT[(CT.W == 3000) & (CT.design == "patched") & (CT.model == c) & (CT.span == sp)].iloc[0]
                for a_, b_ in (
                    ("pct_qlike", "pct_qlike"),
                    ("dm_qlike", "dm_qlike"),
                    ("p_qlike", "p_qlike"),
                    ("dm_mse", "dm_mse"),
                    ("qlike_full", "qlike_a"),
                    ("qlike_har", "qlike_b"),
                ):
                    chk(close(x[a_], r[b_], 1e-9), f"channels joint {c} {sp} {a_}: {x[a_]} vs {r[b_]}")
        sh = pd.read_csv(chg / "item2_shares.csv")
        for c in CLASSES:
            r = sh[sh["cls"] == c].iloc[0]
            x = SH[(SH.W == 3000) & (SH.design == "patched") & (SH.model == c)].iloc[0]
            for col in ("total_gain_sum", "share_event_pct", "share_event_pm_pct", "share_ord_pct"):
                chk(close(x[col], r[col], 1e-9), f"channels item2 {c} {col}: {x[col]} vs {r[col]}")
        for design, sub, spans in (
            ("patched", "scores_fomcfix", ("all", "c16", "c16_2020", "ord", "event")),
            ("pre-patch", "scores", ("all", "c16", "c16_2020")),
        ):
            m = pd.read_csv(RES / "ablations_2014start_w3000" / sub / "metrics.csv")
            dm = pd.read_csv(RES / "ablations_2014start_w3000" / sub / "dm.csv")
            for c in CLASSES:
                for sp in spans:
                    span = "ev" if sp == "event" else sp
                    x = CT[(CT.W == 3000) & (CT.design == design) & (CT.model == c) & (CT.span == sp)].iloc[0]
                    for arm in ("full", "har"):
                        cls_ = "ols" if (arm == "har" and c in LIN) else c  # a linear class's benchmark: the OLS
                        mm = m[(m.cls == cls_) & (m.arm == arm) & (m.span == span)].iloc[0]
                        chk(close(x[f"qlike_{arm}"], mm["qlike"], 1e-12), f"{sub} metrics {cls_} {arm} {sp} qlike")
                        chk(close(x[f"mse_{arm}"], mm["mse"], 1e-12), f"{sub} metrics {cls_} {arm} {sp} mse")
                    vs = "ols" if c in LIN else c
                    for loss in ("qlike", "mse"):
                        r = dm[
                            (dm.cls == c)
                            & (dm.arm == "full")
                            & (dm.vs_cls == vs)
                            & (dm.vs_arm == "har")
                            & (dm.span == span)
                            & (dm.loss == loss)
                        ].iloc[0]
                        chk(
                            close(x[f"dm_{loss}_autolag"], r["dm"], 1e-9) and int(r["hac_lag"]) == x["autolag"],
                            f"{sub} dm {c} {sp} {loss}: {x[f'dm_{loss}_autolag']} vs {r['dm']}",
                        )
    if 3000 in WINDOWS:
        t1 = pd.read_csv(RES / "oos_final_2014start" / "table1_w3000" / "metrics.csv")
        for _, r in t1.iterrows():
            key = (f"3000|patched|{r['model']}", SET_ARM[r["set"]])
            chk(
                close(S.mean(key, "all"), r["qlike"], 1e-12) and close(S.mean(key, "c16_2020"), r["qlike_1600"], 1e-12),
                f"Table 1 metrics {key}: every-bar QLIKE and qlike_1600 (2020-24)",
            )
    rw = RES / "rehaul_robust_window_final"
    rdm = pd.read_csv(rw / "dm_full_vs_har.csv")
    rmet = pd.read_csv(rw / "metrics.csv")
    for W in WINDOWS:
        for c in CLASSES:
            for bars, sp in (("bar", "all"), ("close", "c16")):
                r = rdm[(rdm.W == W) & (rdm.model == c) & (rdm.bars == bars)].iloc[0]
                own = S.comp(ctx(W, "pre-patch", c), ctx_h(W, "pre-patch", c, own=True), sp)
                chk(
                    close(own["dm_qlike"], r["dm"], 1e-9) and close(own["pct_qlike"], r["pct"], 1e-9),
                    f"robust dm_full_vs_har W{W} {c} {bars}: {own['dm_qlike']} vs {r['dm']}",
                )
            for arm, s in (("full", "all_features"), ("har", "baseline")):
                r = rmet[(rmet.W == W) & (rmet.model == c) & (rmet.set == s)].iloc[0]
                key = (f"{W}|pre-patch|{c}", arm)
                chk(close(S.mean(key, "all"), r["qlike"], 1e-12), f"robust metrics W{W} {c} {s} qlike")
                chk(close(S.mean(key, "c16"), r["qlike_1600"], 1e-12), f"robust metrics W{W} {c} {s} qlike_1600")
    say(f"context gates: {len(GATES['checked'])} passed, {len(GATES['fail'])} failed")

    # ------------------------------------------------------------------------------------------- the MinGRU
    econ = Econ()
    rsh = pd.read_csv(rw / "sharpe.csv")
    mg_rows, mg_sh, econ_rows, arm_info = [], [], [], []
    for W in WINDOWS:
        # economics gate: the six classes' full - HAR rows of the robustness run (pre-patch), from their forecasts
        raw = context_run(W, "pre-patch")
        for c in CLASSES:
            for arm in ("full", "har"):
                p, b, ra, _ = raw[(c, arm)]
                econ.add(f"ctx{W}_{c}_{arm}", econ.plain16(p**2 * b, ra, f"{W} {c} {arm}"))
            for col, side in (("mid", "mid"), ("x", "x")):
                r = rsh[(rsh.test == "full - HAR") & (rsh.W == W) & (rsh.model == c) & (rsh.side == side)].iloc[0]
                x = econ.test(f"ctx{W}_{c}_full", f"ctx{W}_{c}_har", col)
                chk(
                    close(x["sr_a"], r["sr_a"], 1e-9)
                    and close(x["sr_b"], r["sr_b"], 1e-9)
                    and close(x["p_lw"], r["p"], 1e-12)
                    and close(x["p_boot"], r["p_boot"], 1e-12),
                    f"robust sharpe W{W} {c} {side}: {x} vs {r.to_dict()}",
                )
                econ_rows.append(dict(W=W, model=c, design="pre-patch", test="full - HAR", side=col, label="", **x))
                for arm in ("full", "har"):
                    y = econ.test(f"ctx{W}_{c}_{arm}", "short", col)
                    econ_rows.append(
                        dict(W=W, model=c, design="pre-patch", test=f"{arm} - short", side=col, label="", **y)
                    )
        del raw
        fs = {}
        for s, arm in ARMS.items():
            f, path = arm_forecasts(W, s)
            arm_info.append(dict(W=W, set=s, path=str(path) if path else "", present=f is not None))
            if f is None:
                say(f"W {W} {s}: no forecasts (arm not done) -- comparisons needing it are skipped")
                continue
            ix = pd.DatetimeIndex(pd.to_datetime(f["date"])).as_unit("ns")
            assert ix.is_unique, (W, s)
            p = pd.Series(f["pred_adj"].to_numpy(float), index=ix)
            b = pd.Series(f["B"].to_numpy(float), index=ix)
            # the arms' compact files carry no rv_actual: RV = core_stats sumret2 (= every context run's rv_actual)
            ra = (
                pd.Series(f["rv_actual"].to_numpy(float), index=ix) if "rv_actual" in f.columns else rv_core.reindex(ix)
            )
            plain = p**2 * b
            if "plain" in f.columns:
                chk(
                    bool(np.allclose(plain.to_numpy(), f["plain"].to_numpy(float), rtol=1e-12, atol=0)),
                    f"MinGRU W{W} {s}: plain = pred_adj^2 B",
                )
            miss = idx.difference(ix)
            chk(len(miss) == 0, f"MinGRU W{W} {s}: forecasts every scored bar (missing {len(miss)})")
            pv = plain.reindex(idx).to_numpy(float)
            chk(bool(np.isfinite(pv).all() and (pv > 0).all()), f"MinGRU W{W} {s}: finite positive on the scored set")
            chk(
                bool(np.allclose(ra.reindex(idx).to_numpy(), rv, rtol=1e-12, atol=0)),
                f"MinGRU W{W} {s}: RV = core_stats sumret2",
            )
            # B from the design export pmf_rows.csv (decimal text): equal to the templates' B to ~1e-12 relative
            bmax = float(np.max(np.abs(b.reindex(idx).to_numpy(float) / B0 - 1.0)))
            chk(bmax <= 1e-11, f"MinGRU W{W} {s}: B = the templates' B (max rel diff {bmax:.2e})")
            S.add((f"{W}|mingru|mingru", arm), pv)
            fs[arm] = (plain, ra)
            comb = MG / "combine" / f"w{W}" / "forecasts.parquet"  # walk-combine's record copy (both arms)
            if comb.is_file():
                cf = pd.read_parquet(comb)
                cf = cf[cf["predictor_set"] == s]
                ci = pd.DatetimeIndex(pd.to_datetime(cf["date"])).as_unit("ns")
                same = ci.equals(ix) and all(
                    np.array_equal(cf[c].to_numpy(float), f[c].to_numpy(float)) for c in ("pred_adj", "B", "plain")
                )
                chk(same, f"MinGRU W{W} {s}: the arm's compact forecasts = walk-combine's forecasts.parquet (exact)")
                chk(
                    bool(
                        np.allclose(
                            cf["rv_actual"].to_numpy(float),
                            rv_core.reindex(ci).to_numpy(),
                            rtol=1e-12,
                            atol=0,
                            equal_nan=True,
                        )
                    ),
                    f"MinGRU W{W} {s}: walk-combine rv_actual = core_stats sumret2",
                )
            else:
                GATES["notes"].append(f"MinGRU W{W} {s}: no walk-combine forecasts.parquet to compare")
            # the arm worker's own scores (same scored set and spans; each worker names its spans its own way)
            sc = MG / f"w{W}_{s}" / "out" / f"scores_w{W}_{s}.csv"
            if sc.is_file():
                a_sc = pd.read_csv(sc)
                key = (f"{W}|mingru|mingru", arm)
                for sp in SPANS + ("event_pm",):
                    r = arm_score_row(a_sc, sp)
                    if r is None:
                        GATES["notes"].append(f"MinGRU W{W} {s} {sp}: no matching span in the arm's scores file")
                        continue
                    m_ = S.mask["event"] & S.pm if sp == "event_pm" else S.mask[sp]
                    q_, e_ = (float(S.L[key][i][m_].mean()) for i in (0, 1))
                    chk(
                        close(q_, r["qlike"], 1e-12) and close(e_, r["mse"], 1e-12),
                        f"MinGRU W{W} {s} {sp}: QLIKE / MSE = the arm's scores file",
                    )
            arm_info[-1].update(
                n_rows=len(f),
                first=str(ix.min()),
                last=str(ix.max()),
                b_max_rel_diff=bmax,
            )
            econ.add(f"mg{W}_{arm}", econ.plain16(plain, ra, f"MinGRU {W} {arm}"))
            ec = MG / f"w{W}_{s}" / "out" / f"econ_w{W}_{s}.csv"
            a_ec = pd.read_csv(ec) if ec.is_file() else None
            for col in ("mid", "x"):
                y = econ.test(f"mg{W}_{arm}", "short", col)
                ae = arm_econ_row(a_ec, col) if a_ec is not None else None
                if ae is None:
                    GATES["notes"].append(f"MinGRU W{W} {s} {col}: the arm's econ file not compared (absent / format)")
                else:  # the arm worker's own D19 ratio and LW test against the short straddle
                    chk(
                        close(y["sr_a"], ae[0], 1e-10)
                        and close(y["sr_b"], ae[1], 1e-10)
                        and close(y["p_lw"], ae[2], 1e-12),
                        f"MinGRU W{W} {s} {col}: D19 ratio and LW p vs short = the arm's econ file",
                    )
                econ_rows.append(
                    dict(
                        W=W,
                        model="mingru",
                        design="v2",
                        test=f"{arm} - short",
                        side=col,
                        label=PREVIEW if arm == "full" else "final",
                        **y,
                    )
                )
        if len(fs) == 2:
            key_f, key_h = (f"{W}|mingru|mingru", "full"), (f"{W}|mingru|mingru", "har")
            for sp in SPANS:
                mg_rows.append(
                    dict(
                        W=W,
                        model="mingru",
                        design="v2",
                        benchmark="mingru|har",
                        label=PREVIEW,
                        **S.comp(key_f, key_h, sp),
                    )
                )
            mg_sh.append(dict(W=W, model="mingru", design="v2", label=PREVIEW, **S.shares(key_f, key_h)))
            for col in ("mid", "x"):
                econ_rows.append(
                    dict(
                        W=W,
                        model="mingru",
                        design="v2",
                        test="full - HAR",
                        side=col,
                        label=PREVIEW,
                        **econ.test(f"mg{W}_full", f"mg{W}_har", col),
                    )
                )
            for arm in ("full", "har"):
                kk = f"mg{W}_{arm}"
                arm_info.append(
                    dict(
                        W=W,
                        set=arm,
                        long_share_pw=float((econ.pwv(kk, "q") > 0).mean()),
                        qlike866=econ.ret[kk]["qlike866"],
                    )
                )
        # MinGRU levels next to the classes (each arm alone, whichever exist)
    MG_ = pd.DataFrame(mg_rows)
    ALL = pd.concat([CT, MG_], ignore_index=True)
    SHALL = pd.concat([SH, pd.DataFrame(mg_sh)], ignore_index=True)
    EC = pd.DataFrame(econ_rows)
    # the short straddle's own ratio
    short = {col: econ.sr("short", col) for col in ("mid", "x")}

    # levels: every-bar / 16:00 / ORD / EVENT QLIKE of each arm
    lev = []
    for key in sorted(S.L):
        W_, design, c = key[0].split("|")
        lev.append(
            dict(
                W=int(W_),
                design=design,
                model=c,
                arm=key[1],
                label=PREVIEW if (c == "mingru" and key[1] == "full") else "",
                **{f"qlike_{sp}": S.mean(key, sp) for sp in SPANS},
                **{f"mse_{sp}": S.mean(key, sp, 1) for sp in SPANS},
            )
        )
    LEV = pd.DataFrame(lev)

    GATES["pass"] = not GATES["fail"]
    GATES["short_straddle_sr"] = short
    GATES["arms"] = arm_info
    GATES["n_checked"] = len(GATES["checked"])
    for W in WINDOWS:
        out = OUTROOT / f"pooled_mingru_v2_2014start_w{W}" / "score"
        out.mkdir(parents=True, exist_ok=True)
        ALL[ALL.W == W].to_csv(out / "compare_full_vs_har.csv", index=False)
        SHALL[SHALL.W == W].to_csv(out / "compare_gain_shares.csv", index=False)
        LEV[LEV.W == W].to_csv(out / "compare_levels.csv", index=False)
        if len(EC):
            EC[EC.W == W].to_csv(out / "compare_econ.csv", index=False)
        g = dict(GATES)
        g["arms"] = [a for a in arm_info if a["W"] == W]
        (out / "compare_gates.json").write_text(json.dumps(g, indent=1, default=str), encoding="utf-8")
        (out / "compare_tables.md").write_text(
            tables_md(
                W, ALL[ALL.W == W], SHALL[SHALL.W == W], LEV[LEV.W == W], EC[EC.W == W] if len(EC) else EC, short
            ),
            encoding="utf-8",
        )
    say(f"gates: {len(GATES['checked'])} passed, {len(GATES['fail'])} failed")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    show = ["W", "model", "design", "span", "pct_qlike", "dm_qlike", "p_qlike", "pct_mse", "dm_mse", "p_mse"]
    print(ALL[show].round(4).to_string(index=False))
    print(
        SHALL[["W", "model", "design", "share_event_pct", "share_event_pm_pct", "share_ord_pct"]].round(2).to_string()
    )
    if len(EC):
        print(EC.round(4).to_string(index=False))
    print("short straddle", short)
    assert GATES["pass"], GATES["fail"]


if __name__ == "__main__":
    main()
