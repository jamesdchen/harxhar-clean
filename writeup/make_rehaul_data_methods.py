"""make_rehaul_data_methods.py -- the Data, Features and Methods tables of the rehauled paper (2026-10-15).

Writes six float files into writeup/generated/, each \\input by writeup/rehaul/0{3,4,5}_*_floats.tex:

  rehaul_dm_sources.tex     D1  data sources and spans                    tab:rh_dm_sources
  rehaul_dm_samples.tex     D2  samples (grid, pooled, per bar, pre-tune, test)   tab:rh_dm_samples
  rehaul_dm_design.tex      F1  design composition (22 / 628 / 574 columns)      tab:rh_dm_design
                                (2026-10-19, base 5 across the paper: the pooled count was 1,144 at base 2)
  rehaul_dm_dictionary.tex  F2  feature dictionary, 47 series (longtable)        tab:rh_dm_dictionary
  rehaul_dm_hyper.tex       M1  hyperparameters per model                        tab:rh_dm_hyper
  rehaul_dm_tuning.tex      M2  tuning protocol per model family                 tab:rh_dm_tuning

Every number is computed here, never typed: from the bar files (data/*.parquet through src/data/loading.py), the
SPXW chain, the 16:00 design cache, the pooled panel cache, the transform code (src/features/transforms/target.py,
src/backtest/executor.py, run on the data), the specs' and drivers' constants (read with ast from their source, so
no spec is executed) and the studies' result files.  The values the outline and the generated macros quote are
asserted (writeup/PAPER_REHAUL_OUTLINE_2026-10-15.md sections 3-5; writeup/generated/close_main_numbers.tex).

Inputs that are gitignored / local (the script stops with the path if one is missing):
  data/*.parquet                                              the bar files and the SPXW chain
  results/linear_pretune_2026-10-06/_work/design/design_bar1600_{all_features,baseline}.npz   (the design cache;
      cross-checked against the committed export results/close_studies_2026-10-03/forecasts/design_bar1600_all_features.parquet)
  ../harxhar-clean/results/prep_cache_all_features_b2.npz     the pooled panel (names and y only are read)
      -- 2026-10-19 (user): parked -- base 5 across the paper; the pooled panel is now the base-5 one, read from
      results/pooled_trees_base5_2026-10-19_48/panel_full.json (committed: names and row count of the base-5 pooled
      panel before the E6 / E9 cut; experiments/pooled_trees_base5.py design --sample 48)
  results/puffer_forecast_2026-10-12/_work/{walk_softplus,walk_softplus_rinit_q,pwalk_wdgrid}/   MinGRU job files

    python writeup/make_rehaul_data_methods.py
"""

from __future__ import annotations

import ast
import json
import math
import re
import sys
import warnings
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.data import loading as LD  # noqa: E402
from src.features.extractors.har import resolve_har_lags  # noqa: E402
from src.features.transforms import target as TG  # noqa: E402

GEN = REPO / "writeup" / "generated"
DATA = REPO / "data"
RES = REPO / "results"
CLEAN = REPO.parent / "harxhar-clean"
DESIGN_DIRS = (
    RES
    / "close_design"
    / "_work",  # experiments/capture_design_close.py's default cache
    RES / "linear_pretune_2026-10-06" / "_work" / "design",
    RES / "trees_pretune_2026-10-06" / "_work" / "design",
)
DESIGN_EXPORT = (
    RES
    / "close_studies_2026-10-03"
    / "forecasts"
    / "design_bar1600_all_features.parquet"
)
# 2026-10-19 (user): parked -- base 5 across the paper. Was the base-2 panel of the earlier pooled fits:
# POOLED_CACHE = CLEAN / "results" / "prep_cache_all_features_b2.npz"
POOLED_PANEL = REPO / "results" / "pooled_trees_base5_2026-10-19_48" / "panel_full.json"


def pooled_panel() -> tuple[list[str], int]:
    """(names, rows) of the base-5 pooled panel (experiments/pooled_trees_base5.py, before the E6 / E9 cut)."""
    d = json.loads(need(POOLED_PANEL).read_text(encoding="utf-8"))
    return [str(c) for c in d["names"]], int(d["n_rows"])
PF = REPO / "experiments" / "puffer_forecast"
PFR = RES / "puffer_forecast_2026-10-12"
PFW = PFR / "_work"
MACROS = REPO / "writeup" / "generated" / "close_main_numbers.tex"
CHECKS: list[str] = []

# the values the outline quotes (PAPER_REHAUL_OUTLINE_2026-10-15.md); each is asserted against the computation
OUTLINE: dict[str, Any] = dict(
    rv_prints=303_444,
    grid=303_442,
    burn_in=3_125,
    pooled_rows=300_317,
    pooled_window=24_000,
    perbar_sessions=3_469,
    perbar_window=2_000,
    forecast_rows=1_469,
    pre_sessions=3_947,
    pre_start="2004-04-13",
    val_first="2016-01-06",
    val_last="2020-01-02",
    trade_days=866,
    expirations_to_end=872,
    early_closes_to_end=6,
    sessions_1600_test=1_087,
    expirations=1_291,
    fomc_events=246,
    n_series=47,
    p_base=22,
    p_all=628,
    # 2026-10-19 (user): parked -- base 5 across the paper. Was: p_pooled=1_144 (base 2, 12 horizons, pre-FOMC panel)
    p_pooled=574,
    n_active=7,
    mingru_origins=1_088,
)


def check(ok: bool, what: str) -> None:
    if not ok:
        raise AssertionError(what)
    CHECKS.append(what)


def need(p: Path) -> Path:
    if not p.exists():
        raise SystemExit(f"missing input: {p}")
    return p


# ============================================================================ formatting
def n_(x: int) -> str:
    """LaTeX thousands separator."""
    return f"{int(x):,}".replace(",", "{,}")


def d_(t) -> str:
    return pd.Timestamp(t).strftime("%Y-%m-%d")


def D_(t) -> str:
    """A date for a table cell: never broken at its hyphens."""
    return r"\mbox{" + d_(t) + "}"


def tt(s: str) -> str:
    return r"\texttt{" + s.replace("_", r"\_") + "}"


def ttb(s: str) -> str:
    """texttt with a permitted line break after every underscore (long column names in narrow cells)."""
    return r"\texttt{" + s.replace("_", r"\_\allowbreak{}") + "}"


def gm(x, pow10: bool = False) -> str:
    """A value as math-mode content: a power of ten as 10^k (always when pow10, else when |k| >= 4), a small or
    large number in scientific form, None as 'none', anything else as %g."""
    if x is None:
        # 2026-10-15 (table legibility): parked -- "none" for an unbounded depth read as a missing value
        # return r"\text{none}"
        return r"\text{unrestricted}"
    x = float(x)
    if x > 0:
        k = math.log10(x)
        if abs(k - round(k)) < 1e-9 and (pow10 or abs(round(k)) >= 4):
            return rf"10^{{{int(round(k))}}}"
    s = f"{x:g}"
    if "e" in s:
        m, e = s.split("e")
        return rf"{m}\times10^{{{int(e)}}}" if m != "1" else rf"10^{{{int(e)}}}"
    return s


def g_(x, pow10: bool = False) -> str:
    s = gm(x, pow10)
    return f"${s}$" if ("^" in s or "\\" in s) else s


def gset(vals, pow10: bool = False) -> str:
    return r"$\{" + r", \allowbreak ".join(gm(v, pow10) for v in vals) + r"\}$"


def colspec(permille: list[int], align: str | None = None) -> str:
    """Every column a p{} whose widths are shares (per mille) of the text width less the inner column gaps, so the
    tabular is exactly \\textwidth wide (outer gaps removed with @{})."""
    assert sum(permille) == 1000, sum(permille)
    k = 2 * (len(permille) - 1)
    cols = [
        rf"p{{\dimexpr(\textwidth-{k}\tabcolsep)*{w}/1000\relax}}" for w in permille
    ]
    return "@{}" + "".join(cols) + "@{}"


HEAD = (
    "% generated by writeup/make_rehaul_data_methods.py -- do not edit by hand; change the generator and re-run.\n"
    "\\providecommand{\\rdmL}{\\raggedright\\let\\\\\\tabularnewline}\n"
    "\\providecommand{\\rdmR}{\\raggedleft\\let\\\\\\tabularnewline}\n"
)


def row(cells: list[str], align: str) -> str:
    """align: one letter per column, l (ragged right) or r (ragged left)."""
    out = []
    for c, a in zip(cells, align):
        out.append((r"\rdmL " if a == "l" else r"\rdmR ") + c)
    return " & ".join(out) + r" \\"


def group(title: str, ncol: int) -> str:
    return rf"\multicolumn{{{ncol}}}{{@{{}}l}}{{\textit{{{title}}}}} \\"


def table(
    label: str,
    caption: str,
    spec: str,
    body: list[str],
    notes: str,
    size: str = r"\small",
    sep: str = "4pt",
) -> str:
    lines = [
        HEAD,
        r"\begin{table}[tbp]",
        r"\centering",
        size,
        rf"\setlength{{\tabcolsep}}{{{sep}}}",
        r"\begin{threeparttable}",
        rf"\caption{{{caption}}}\label{{{label}}}",
        rf"\begin{{tabular}}{{{spec}}}",
        *body,
        r"\end{tabular}",
        r"\begin{tablenotes}[flushleft]",
        r"\footnotesize",
        rf"\item \textit{{Notes:}} {notes}",
        r"\end{tablenotes}",
        r"\end{threeparttable}",
        r"\end{table}",
        "",
    ]
    return "\n".join(lines)


# 2026-10-15 (table legibility): finance-journal tables (NOTATION_GUIDE_2026-10-15.md section 8): short caption, a
# few-sentence \textit{Notes:}, ragged-right p{} columns for words, r columns for numbers, panels, \small at most.
def P_(frac: float) -> str:
    """A ragged-right paragraph column `frac` of the text width wide (array's >{} -- makecell loads array)."""
    return rf">{{\raggedright\arraybackslash}}p{{{frac:.3f}\textwidth}}"


def pcols(
    permille: list[int], width: str = r"\linewidth", sep: str = r"\tabcolsep"
) -> str:
    """Ragged-right p{} columns whose widths are shares (per mille) of `width` less the inner column gaps, so the
    table is exactly `width` wide (outer gaps removed with @{})."""
    assert sum(permille) == 1000, sum(permille)
    k = 2 * (len(permille) - 1)
    return (
        "@{}"
        + "".join(
            rf">{{\raggedright\arraybackslash}}p{{\dimexpr({width}-{k}{sep})*{w}/1000\relax}}"
            for w in permille
        )
        + "@{}"
    )


def cells(c: list[str]) -> str:
    return " & ".join(c) + r" \\"


def panel_row(title: str, ncol: int) -> str:
    """A finance-style panel heading spanning the table."""
    return rf"\multicolumn{{{ncol}}}{{@{{}}l}}{{\textit{{{title}}}}} \\"


def mc(n: int, text: str, align: str = "c") -> str:
    return rf"\multicolumn{{{n}}}{{{align}}}{{{text}}}"


def stack(*lines: str, align: str = "c") -> str:
    """A multi-line header cell (makecell)."""
    return rf"\makecell[{align}b]{{" + r" \\ ".join(lines) + "}"


def write(name: str, text: str) -> None:
    p = GEN / name
    p.write_bytes(text.encode("utf-8"))
    print(f"wrote {p.relative_to(REPO)}")


# ============================================================================ source constants (ast; no execution)
def consts(path: Path, names: tuple[str, ...], extra: dict | None = None) -> dict:
    """The module-level UPPER_CASE assignments of a source file, executed in order in a namespace holding only numpy,
    math and the builtins (a statement that needs anything else -- an env read, an import -- is skipped and the last
    literal value stands).  Returns the requested names; asserts each was found."""
    tree = ast.parse(need(path).read_text(encoding="utf-8"))
    ns: dict = {"np": np, "math": math, **(extra or {})}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            targets = [node.target]
        else:
            continue
        ok = True
        for t in targets:
            elts = t.elts if isinstance(t, ast.Tuple) else [t]
            ok &= all(isinstance(e, ast.Name) and e.id == e.id.upper() for e in elts)
        if not ok:
            continue
        value = node.value
        assert value is not None
        stmt = ast.Module(
            body=[ast.Assign(targets=targets, value=value)], type_ignores=[]
        )
        ast.fix_missing_locations(stmt)
        try:
            exec(compile(stmt, str(path), "exec"), ns)  # noqa: S102 -- literal constants of our own source
        except Exception:  # noqa: BLE001 -- statements that need the runtime are skipped by design
            continue
    miss = [k for k in names if k not in ns]
    assert not miss, (path, miss)
    return {k: ns[k] for k in names}


def macros() -> dict[str, str]:
    m = {}
    for k, v in re.findall(
        r"\\newcommand\{\\(\w+)\}\{(.*)\}", need(MACROS).read_text(encoding="utf-8")
    ):
        m[k] = v
    return m


def jobs(path: Path) -> list[dict]:
    """A MinGRU job file: one job per line, space-separated key=value."""
    out = []
    for line in need(path).read_text().splitlines():
        if line.strip():
            out.append(dict(kv.split("=", 1) for kv in line.split()))
    return out


# ============================================================================ data
def sources() -> dict:
    """D1: one record per source file, from the files themselves."""
    S: dict = {}

    def bar(f: str) -> pd.DataFrame:
        d = pd.read_parquet(need(DATA / f"{f}.parquet"))
        d["endbartime"] = pd.to_datetime(d["endbartime"])
        return d

    def span(d: pd.DataFrame, cols: list[str]) -> tuple:
        m = d[cols].notna().any(axis=1)
        t = d.loc[m, "endbartime"]
        return t.min(), t.max(), int(m.sum())

    core = bar("core_stats")
    S["core_cols"] = [c for c in core.columns if c != "endbartime"]
    S["rv"] = span(core, ["sumret2"])
    check(S["rv"][2] == OUTLINE["rv_prints"], f"RV prints {S['rv'][2]}")
    check(
        d_(S["rv"][0]) == "1998-01-05" and d_(S["rv"][1]) == "2024-04-30",
        "RV span 1998-01-05 .. 2024-04-30",
    )
    xs = {}
    for f in ("ewstock_stats", "vwstock_stats"):
        d = bar(f)
        cols = [c for c in d.columns if c != "endbartime"]
        xs[f] = (len(cols), *span(d, cols))
    check(
        xs["ewstock_stats"][1:] == xs["vwstock_stats"][1:]
        and xs["ewstock_stats"][0] == xs["vwstock_stats"][0],
        "EW and VW cross-section files share columns count, span and print count",
    )
    S["xs"] = xs["ewstock_stats"]
    ewc = [
        c
        for c in pd.read_parquet(DATA / "ewstock_stats.parquet").columns
        if c != "endbartime"
    ]
    S["xs_moments"] = sum(c.startswith("sum") for c in ewc)
    check(
        S["xs_moments"] == 6 and len(ewc) - S["xs_moments"] == 5,
        "cross-section: 6 moments + 5 liquidity columns",
    )
    st = bar("spy_and_sentiment")
    S["att"] = span(st, ["stocktwits_attention"])
    S["sent"] = span(st, ["stocktwits_sentiment", "stocktwits_sentcount"])
    check(
        span(st, ["stocktwits_sentiment"]) == span(st, ["stocktwits_sentcount"]),
        "sentiment and sentcount coincide",
    )
    vv = bar("vix_and_voldemand")
    for c in ("vix", "vix3m", "vvix"):
        S[c] = span(vv, [c])
    vd = [c for c in vv.columns if c.startswith("voldemand")]
    S["vd"] = span(vv, vd)
    check(
        all(span(vv, [c]) == S["vd"] for c in vd),
        "the four voldemand series print on the same bars",
    )
    S["n_vd"] = len(vd)
    rel = bar("releases")
    ev = rel.loc[pd.to_numeric(rel[LD.FOMC_RAW_COL]) > 0, "endbartime"]
    S["fomc"] = (ev.min(), ev.max(), len(ev))
    check(
        len(ev) == OUTLINE["fomc_events"] and set(ev.dt.strftime("%H:%M")) == {"14:30"},
        "246 FOMC bars, all 14:30",
    )
    S["release_flags"] = [
        c for c in rel.columns if c not in ("endbartime", LD.FOMC_RAW_COL)
    ]
    ch = pd.read_parquet(
        need(DATA / "spxw_chain.parquet"),
        columns=["expiration", "timestamp", "early_close"],
    )
    e = ch.groupby("expiration")["early_close"].max()
    S["chain"] = (ch["timestamp"].min(), ch["timestamp"].max(), len(ch), len(e))
    S["exp"] = e
    check(len(e) == OUTLINE["expirations"], f"{len(e)} SPXW expirations")
    check(
        d_(e.index.min()) == "2020-01-03" and d_(e.index.max()) == "2025-12-31",
        "chain expirations 2020-01-03 .. 2025-12-31",
    )
    return S


def panel() -> dict:
    """D2: the grid, the pooled panel, the per-bar design, the pre-tune rows, the test sample."""
    P: dict = {}
    df = LD.load_raw_data(str(DATA), allow_missing=True)
    P["df"] = df
    t = df["t"]
    P["grid"] = (t.iloc[0], t.iloc[-1], len(df))
    check(len(df) == OUTLINE["grid"], f"grid {len(df)} print-bars")
    burn = resolve_har_lags()[-1]
    check(burn == OUTLINE["burn_in"], f"burn-in {burn}")
    P["burn"] = burn
    P["pooled"] = (t.iloc[burn], t.iloc[-1], len(df) - burn)
    check(len(df) - burn == OUTLINE["pooled_rows"], f"pooled panel {len(df) - burn}")
    # 2026-10-19 (user): parked -- base 5 across the paper. Was: z = np.load(need(POOLED_CACHE), allow_pickle=True); len(z["y"])
    check(
        pooled_panel()[1] == len(df) - burn,
        "the pooled panel cache has one row per pooled-panel bar",
    )
    W = OUTLINE["pooled_window"]
    uw = consts(
        REPO / "src" / "unification.py",
        ("DEFAULT_WINDOW_BARS",),
        {"PERIODS_PER_DAY": TG.PERIODS_PER_DAY},
    )
    check(
        uw["DEFAULT_WINDOW_BARS"] == W, "pooled window 24,000 bars (src/unification.py)"
    )
    P["pooled_window"] = uw["DEFAULT_WINDOW_BARS"]
    P["pooled_fc"] = (t.iloc[burn + W], t.iloc[-1], len(df) - burn - W)
    # the tree development span: pooled rows strictly before the first bar of 2003 (experiments/tree_menu_dev.py)
    tm = consts(
        REPO / "experiments" / "tree_menu_dev.py", ("WINDOW", "SCREEN_REFIT", "BOXES")
    )
    rf = json.loads(
        need(RES / "rf_devmenu_2026-10-06" / "budget400" / "rf_config.json").read_text()
    )
    dev_end = int(rf["provenance"]["dev_end_row"])
    tp = t.iloc[burn:].reset_index(drop=True)
    check(
        tp.iloc[dev_end - 1] < pd.Timestamp("2003-01-01") <= tp.iloc[dev_end],
        "dev span = pooled rows before 2003",
    )
    P["dev"] = (tp.iloc[0], tp.iloc[dev_end - 1], dev_end, tp.iloc[tm["WINDOW"]])
    P["tm"] = tm
    P["rf"] = rf
    # per-bar 16:00: the burn-in applies to the one-bar series
    hm = t.dt.strftime("%H:%M")
    t16 = t[hm == "16:00"].reset_index(drop=True).iloc[burn:].reset_index(drop=True)
    P["perbar"] = (t16.iloc[0], t16.iloc[-1], len(t16))
    check(
        len(t16) == OUTLINE["perbar_sessions"] and d_(t16.iloc[0]) == "2010-07-12",
        "per-bar design 3,469 from 2010-07-12",
    )
    # pre-tune: the 16:00 rows of the last-hour design (two bars a session; burn-in on that series)
    tl = t[hm.isin(["15:30", "16:00"])].reset_index(drop=True).iloc[burn:]
    pre = tl[tl.dt.strftime("%H:%M") == "16:00"]
    pre = pre[pre < pd.Timestamp("2020-01-03")].reset_index(drop=True)
    clp = consts(
        REPO / "experiments" / "close_linear_pretune.py",
        (
            "TW",
            "TUNE_PER",
            "VAL_TAIL",
            "EMBARGO",
            "PRE_FOLDS",
            "PRE_FOLD_LEN",
            "R_BB",
            "MARGIN_ALPHA",
            "RETUNE_FOLDS",
            "FIRST_TRADE_DAY",
            "LAST_TRADE_DAY",
        ),
    )
    P["clp"] = clp
    nval = clp["PRE_FOLDS"] * clp["PRE_FOLD_LEN"]
    P["pre"] = (pre.iloc[0], pre.iloc[-1], len(pre), pre.iloc[-nval], nval)
    check(
        len(pre) == OUTLINE["pre_sessions"] and d_(pre.iloc[0]) == OUTLINE["pre_start"],
        "pre-tune 3,947 from 2004-04-13",
    )
    check(
        d_(pre.iloc[-nval]) == OUTLINE["val_first"]
        and d_(pre.iloc[-1]) == OUTLINE["val_last"],
        "pre-tune validation 2016-01-06 .. 2020-01-02",
    )
    facts = pd.read_csv(
        need(RES / "linear_pretune_2026-10-06" / "paper_facts.csv")
    ).set_index("item")["value"]
    P["facts"] = facts
    check(
        int(facts["pre-2020 16:00 sessions in the design"]) == len(pre)
        and facts["pre-tune validation: first session"] == d_(pre.iloc[-nval]),
        "pre-tune facts agree with paper_facts.csv",
    )
    # the design cache: the same 3,469 sessions, W = 2,000, 1,469 forecast rows
    dz = None
    for d in DESIGN_DIRS:
        if (d / "design_bar1600_all_features.npz").is_file():
            dz = d
            break
    if dz is None:
        raise SystemExit(
            f"missing input: design_bar1600_all_features.npz in any of {[str(d) for d in DESIGN_DIRS]}"
        )
    za = np.load(dz / "design_bar1600_all_features.npz", allow_pickle=False)
    zb = np.load(dz / "design_bar1600_baseline.npz", allow_pickle=False)
    P["design_dir"] = dz
    P["names_all"] = [str(v) for v in za["names"]]
    P["names_base"] = [str(v) for v in zb["names"]]
    dd = pd.to_datetime(za["date"])
    Wd = int(za["W"])
    check(
        np.array_equal(dd.to_numpy(), t16.to_numpy()),
        "design dates = the 16:00 rows after the burn-in, one for one",
    )
    check(
        Wd == OUTLINE["perbar_window"] and len(dd) - Wd == OUTLINE["forecast_rows"],
        "W = 2,000, 1,469 forecast rows",
    )
    check(
        np.array_equal(zb["date"], za["date"]),
        "baseline and all_features designs share their rows",
    )
    # 2026-10-15 (table legibility): the calendar dummies constant at the closing interval (dropped by the window
    # screen at every closing-interval fit; NOTATION_GUIDE_ADDENDA A.3)
    Xa = za["X"]
    cal_cols = [c for c in P["names_all"] if c in CAL_TEX]
    P["cal_const"] = [
        c
        for c in cal_cols
        if np.nanmax(Xa[:, P["names_all"].index(c)])
        == np.nanmin(Xa[:, P["names_all"].index(c)])
    ]
    check(
        P["cal_const"] == ["hour", "is_overnight", "is_open", "is_close"],
        "hour, overnight, open and close dummies are constant on every closing-interval row",
    )
    import pyarrow.parquet as pq

    ex_names = [
        c
        for c in pq.read_schema(need(DESIGN_EXPORT)).names
        if c not in ("row", "date", "forecast_row", "y", "baseline", "true_raw")
    ]
    check(
        ex_names == P["names_all"],
        "design cache names = the committed export's columns",
    )
    P["fc"] = (dd[Wd], dd[-1], len(dd) - Wd)
    # the test sample: SPXW expirations to the last 16:00 row, early closes out
    e = SRC["exp"]
    last = pd.Timestamp(clp["LAST_TRADE_DAY"])
    to_end = e[e.index <= last]
    P["exp_to_end"] = len(to_end)
    P["ec_to_end"] = int(to_end.sum())
    days = to_end.index[~to_end.to_numpy().astype(bool)]
    fcd = pd.DatetimeIndex(dd[Wd:]).normalize()
    check(
        set(days).issubset(set(fcd)),
        "every full-session expiration to 2024-04-30 has a 16:00 forecast row",
    )
    check(
        len(to_end) == OUTLINE["expirations_to_end"]
        and int(to_end.sum()) == OUTLINE["early_closes_to_end"]
        and len(days) == OUTLINE["trade_days"],
        "866 = 872 expirations - 6 early closes",
    )
    P["days"] = days
    P["later"] = (
        int((e.index > last).sum()),
        int((~e[e.index > last].astype(bool)).sum()),
    )
    check(P["later"][0] + len(to_end) == len(e), "later expirations complete the chain")
    daily = pd.read_parquet(
        need(RES / "close_master_table" / "master_table_daily.parquet"),
        columns=["key", "day"],
    )
    one = daily[daily["key"] == daily["key"].iloc[0]]
    check(
        set(pd.to_datetime(one["day"]).dt.normalize()) == set(days),
        "the 866 days = the master table's trade days",
    )
    first = pd.Timestamp(clp["FIRST_TRADE_DAY"])
    P["test16"] = int((fcd >= first).sum())
    check(
        P["test16"] == OUTLINE["sessions_1600_test"]
        and d_(days.min()) == clp["FIRST_TRADE_DAY"],
        "1,087 16:00 sessions in the test period, first trade day 2020-01-03",
    )
    # 2026-10-15 (user): the evaluation span of the accuracy statistics -- every trading day on which all rolling
    # forecasts of tab:rh_oos exist, early closes dropped (experiments/rolling_eval_2026_10_15.py common_span())
    sys.path.insert(0, str(REPO / "experiments"))
    import rolling_eval_2026_10_15 as rolling

    span = rolling.common_span()
    P["span"] = span
    P["span_ec"] = int(
        ((rolling.EARLY >= span.min()) & (rolling.EARLY <= span.max())).sum()
    )
    check(
        set(days).issubset(set(span)) and d_(span.min()) == clp["FIRST_TRADE_DAY"],
        "the 866 option-data days lie in the evaluation span, which starts on the first trade day",
    )
    nyse = rolling.nyse_sessions(str(span.min().date()), str(span.max().date()))
    check(
        len(span) == len(nyse) - P["span_ec"] and set(span) <= set(nyse),
        "the evaluation span = every NYSE session of 2020-01-03 .. 2024-04-30 less the early closes",
    )
    return P


# ============================================================================ features
def series_bucket() -> dict[str, str]:
    order = (
        "moments",
        "liquidity",
        "market_ew",
        "market_vw",
        "sentiment",
        "implied_vol",
        "vol_demand",
        "fomc",
    )
    m: dict[str, str] = {}
    for b in order:
        for s in LD.SUBGROUPS[b]:
            check(s not in m, f"{s} in one bucket only")
            m[s] = b
    check(
        set(m) == set(LD.ALL_FEATURES) and len(m) == OUTLINE["n_series"],
        "8 buckets partition the 47 series",
    )
    return m


# 2026-10-15 (table legibility): parked -- group names as in the Predictors section (04_features.tex).
# BUCKET_LABEL = {
#     "moments": "ES return moments", "liquidity": "Liquidity", "market_ew": "Market, equal-weighted",
#     "market_vw": "Market, value-weighted", "sentiment": "Sentiment", "implied_vol": "Implied volatility",
#     "vol_demand": "Volatility demand", "fomc": "FOMC calendar",
# }
BUCKET_LABEL = {
    "moments": "Realized moments",
    "liquidity": "Liquidity",
    "market_ew": "Equal-weighted market",
    "market_vw": "Value-weighted market",
    "sentiment": "Sentiment",
    "implied_vol": "Implied volatility",
    "vol_demand": "Volatility demand",
    "fomc": "FOMC",
}

# ---------------------------------------------------------------------------- predictor names (2026-10-15)
# The series symbols of NOTATION_GUIDE_ADDENDA_2026-10-15.md section A.4 (drop-in dictionary), which follows the core
# guide's section 5 and the Data / Predictors sections; checked below over both designs' column names.
SERIES_TEX = {
    "sumret": r"r",
    "sumabsret": r"\mathrm{RAV}",
    "sumret3": r"\mathrm{RSK}",
    "sumret4": r"\mathrm{RQ}",
    "sumpret2": r"\mathrm{RS}^{+}",
    "sumbipow": r"\mathrm{BV}",
    "sumautocov": r"\mathrm{AC}",
    "sumvolume": r"\mathrm{Volume}",
    "numobs": r"N",
    "turnover": r"\mathrm{TO}",
    "buyturnover": r"\mathrm{TO}_{\mathrm{buy}}",
    "sellturnover": r"\mathrm{TO}_{\mathrm{sell}}",
    "effspread": r"\mathrm{EffSpread}",
    "spread": r"\mathrm{Spread}",
    "ofi": r"\mathrm{OFI}",
    "sumret2": r"\mathrm{RV}",
    "stocktwits_attention": r"\mathrm{Attn}",
    "stocktwits_sentiment": r"\mathrm{Sent}",
    "stocktwits_sentcount": r"\mathrm{NSent}",
    "vix": r"\mathrm{VIX}",
    "vvix": r"\mathrm{VVIX}",
    "vix3m": r"\mathrm{VIX3M}",
    "voldemand_spx_open_and_close": r"\mathrm{VD}^{\mathrm{SPX}}_{\mathrm{OC}}",
    "voldemand_spx_open_only": r"\mathrm{VD}^{\mathrm{SPX}}_{\mathrm{O}}",
    "voldemand_all_open_and_close": r"\mathrm{VD}^{\mathrm{All}}_{\mathrm{OC}}",
    "voldemand_all_open_only": r"\mathrm{VD}^{\mathrm{All}}_{\mathrm{O}}",
    "fomc_release": r"\mathrm{FOMC}^{\mathrm{rel}}",
    "fomc_day": r"\mathrm{FOMC}^{\mathrm{day}}",
    "fomc_until_inv": r"\mathrm{FOMC}^{\mathrm{next}}",
    "fomc_since_inv": r"\mathrm{FOMC}^{\mathrm{last}}",
}
CAL_TEX = {
    "DOW_0": r"\mathrm{Mon}",
    "DOW_1": r"\mathrm{Tue}",
    "DOW_2": r"\mathrm{Wed}",
    "DOW_3": r"\mathrm{Thu}",
    "DOW_4": r"\mathrm{Fri}",
    "hour": r"\mathrm{Hour}",
    "is_overnight": r"\mathrm{Night}",
    "is_open": r"\mathrm{Open}",
    "is_close": r"\mathrm{Close}",
    "is_opex": r"\mathrm{OpEx}",
    "is_opex_week": r"\mathrm{OpExWk}",
    "is_quad_witch": r"\mathrm{QW}",
    "is_rebalance_close": r"\mathrm{QW}\cdot\mathrm{Close}",
    "is_month_end": r"\mathrm{ME}",
    "is_quarter_end": r"\mathrm{QE}",
    "days_to_opex": r"\mathrm{DOpEx}",
}
GROUP_TEX = {"ewstock": r"\mathrm{EW}", "vwstock": r"\mathrm{VW}"}
GROUP_WORDS = {"ewstock": "Equal-weighted", "vwstock": "Value-weighted"}


def _sup(base: str, extra: str) -> str:
    """Append `extra` to the superscript of `base` (brace-matched): RS^{+} -> RS^{+,EW}; TO_{buy} -> TO^{EW}_{buy}."""
    i = base.find("^{")
    if i >= 0:
        depth = 0
        for j in range(i + 1, len(base)):
            depth += {"{": 1, "}": -1}.get(base[j], 0)
            if depth == 0:
                return base[:j] + "," + extra + base[j:]
    i = base.find("_{")
    if i >= 0:
        return base[:i] + "^{" + extra + "}" + base[i:]
    return base + "^{" + extra + "}"


def series_sym(s: str) -> str:
    """Math symbol (no $) of a bare series code name, e.g. 'sumret2_ewstock' -> RV^{EW}."""
    if s in SERIES_TEX:
        return SERIES_TEX[s]
    stem, _, grp = s.rpartition("_")
    return _sup(SERIES_TEX[stem], GROUP_TEX[grp])


# 2026-10-19 (user): the draft names each series by its raw parquet name for now. \ser{raw}{symbol} and
# \serx{raw}{raw sup/sub}{symbol} switch on \ifrawseries (paper_rehaul.tex preamble; \rawseriesfalse restores the
# symbols). The fallbacks keep a standalone \input of a generated file working.
SER_PROVIDE = "\\providecommand{\\ser}[2]{#2}\\providecommand{\\serx}[3]{#3}\n"


def series_tex(s: str) -> str:
    """The series' name in the draft (no $, math mode): \\ser{raw name}{symbol}."""
    return rf"\ser{{{s}}}{{{series_sym(s)}}}"


def column_tex(c: str) -> str:
    """Math name (with $) of any design column of the 16:00 design or the pooled panel."""
    if c in CAL_TEX:
        return f"${CAL_TEX[c]}$"
    if m := re.fullmatch(r"har_ma_(\d+)", c):
        return rf"$\bar y^{{({m.group(1)})}}$"
    if m := re.fullmatch(r"har_ma_(\d+)_x_(open|close)", c):
        return rf"$\bar y^{{({m.group(1)})}}\cdot\mathrm{{{m.group(2).title()}}}$"
    if m := re.fullmatch(r"adj_(.+)_ma_(\d+)", c):
        # 2026-10-19: the horizon stacks into the symbol's own superscript, so the raw form carries its own ^{(k)}
        sym = _sup(series_sym(m.group(1)), "(" + m.group(2) + ")")
        return rf"$\serx{{{m.group(1)}}}{{^{{({m.group(2)})}}}}{{{sym}}}$"
    if m := re.fullmatch(r"(.+)_(avail|active)_ma_(\d+)", c):
        f = r"\bar a" if m.group(2) == "avail" else r"\bar\chi"
        return f"${f}^{{({m.group(3)})}}({series_tex(m.group(1))})$"
    raise KeyError(c)


# What each series measures, in words (the Data section's symbols; the addenda's "words" column).  A constituent
# series (suffix _ewstock / _vwstock) is the stem's words prefixed with its weighting.
# 2026-10-17 (user, E5): descriptions from data_explainer_v11. Constituent moments are 5-minute;
# the E-mini moments stay one-minute. "all index options" -> all traded options. SPX includes the top ETFs.
DESC = {
    # 2026-10-19 (user): the defined series prints through \ser / \serx (raw names for now). Was: "$r_{t,i} = ...$",
    # "$\mathrm{BV}_{t,i} = ...$", "$\mathrm{AC}_{t,i} = ...$", "$N_{t,i} \le 30$"
    "sumret": r"Sum of one-minute changes in the log price of the front-month contract, $\serx{sumret}{_{t,i}}{r_{t,i}} = \sum_k r_{t,i,k}$",
    "sumabsret": r"Realized absolute variation, $\sum_k |r_{t,i,k}|$",
    "sumret3": r"Realized skewness, unnormalized, $\sum_k r_{t,i,k}^3$",
    "sumret4": r"Realized quarticity, unscaled, $\sum_k r_{t,i,k}^4$",
    "sumpret2": r"Positive realized semivariance, $\sum_k r_{t,i,k}^2\,\mathbf{1}\{r_{t,i,k} > 0\}$",
    "sumbipow": r"Bipower variation, $\ser{sumbipow}{\mathrm{BV}}_{t,i} = \sum_{k \ge 2} |r_{t,i,k}|\,|r_{t,i,k-1}|$",
    "sumautocov": r"First-order realized autocovariance, $\ser{sumautocov}{\mathrm{AC}}_{t,i} = \sum_{k \ge 2} r_{t,i,k}\, r_{t,i,k-1}$",
    # 2026-10-15 (notation, final consolidation): "ES" in words is "E-mini S&P 500 futures"
    # 2026-10-17: parked -- "E-mini S&P 500 futures contract volume"
    "sumvolume": "Number of front-month contracts traded in the half hour",
    "numobs": r"Number of non-missing one-minute intervals in the half hour, $\ser{numobs}{N}_{t,i} \le 30$",
    "turnover": "turnover, shares traded in the half hour divided by the previous day's shares outstanding",
    "buyturnover": "buyer-initiated turnover, the same ratio using estimated buy orders",
    "sellturnover": "seller-initiated turnover, the same ratio using estimated sell orders",
    "effspread": "effective spread, the mean percent difference between the trade price and the preceding midquote",
    "spread": "quoted spread, the mean percent difference between the bid and the ask",
    "ofi": r"order-flow imbalance, $(\mathrm{Buy} - \mathrm{Sell}) / (\mathrm{Buy} + \mathrm{Sell})$ of turnover",
    "sumret2": "realized variance of 5-minute log returns",
    "stocktwits_attention": "Number of StockTwits posts in the half hour",
    "stocktwits_sentiment": r"Score of tagged posts, $1$ if all are bullish and $-1$ if all are bearish",
    "stocktwits_sentcount": "Number of posts in the half hour that carry a bullish or bearish tag",
    "vix": "VIX index (30-day S\\&P 500 implied volatility)",
    "vvix": "VVIX index (implied volatility of VIX)",
    "vix3m": "VIX3M index (3-month S\\&P 500 implied volatility)",
    "voldemand_spx_open_and_close": "Net vega bought by non-market-makers, S\\&P 500 index and top S\\&P 500 ETF options, trades that open or close a position",
    "voldemand_spx_open_only": "Net vega bought by non-market-makers, S\\&P 500 index and top S\\&P 500 ETF options, trades that open a position",
    "voldemand_all_open_and_close": "Net vega bought by non-market-makers, all traded options, trades that open or close a position",
    "voldemand_all_open_only": "Net vega bought by non-market-makers, all traded options, trades that open a position",
    "fomc_release": "FOMC announcement-interval dummy (14:30 ET)",
    "fomc_day": "FOMC announcement-day dummy",
    "fomc_until_inv": r"$1/(1 + n^{+}_{t,i})$, $n^{+}_{t,i}$ intervals to the next announcement",
    "fomc_since_inv": r"$1/(1 + n^{-}_{t,i})$, $n^{-}_{t,i}$ intervals since the last announcement",
}
# the constituent versions of the ES moments: words only (the formula is the ES row's)
DESC_XS = {
    "sumret3": "realized skewness of 5-minute log returns, unnormalized",
    "sumret4": "realized quarticity of 5-minute log returns, unscaled",
    "sumabsret": "realized absolute variation of 5-minute log returns",
    "sumbipow": "bipower variation of 5-minute log returns",
    "sumpret2": "positive realized semivariance of 5-minute log returns",
}
# the source of each data file, in words (Table tab:rh_dm_sources)
SOURCE_WORDS = {
    "core_stats": "E-mini S\\&P 500 futures",
    "ewstock_stats": "Constituents",
    "vwstock_stats": "Constituents",
    "spy_and_sentiment": "StockTwits",
    "vix_and_voldemand": "Cboe",
    "releases": "FOMC calendar",
}


def desc_of(s: str) -> str:
    if s in DESC:
        return DESC[s][0].upper() + DESC[s][1:]
    stem, _, grp = s.rpartition("_")
    assert grp in GROUP_WORDS and (stem in DESC or stem in DESC_XS), s
    return f"{GROUP_WORDS[grp]} {DESC_XS.get(stem, DESC[stem])}"


CALENDAR_ORDER = (
    "DOW_0",
    "DOW_1",
    "DOW_2",
    "DOW_3",
    "DOW_4",
    "hour",
    "is_overnight",
    "is_open",
    "is_close",
)


def classify(names: list[str], bmap: dict[str, str]) -> dict:
    """Column kinds of a design: har / edge (HAR x open/close) / calendar / value / avail / active, per bucket."""
    C: dict = {
        "har": [],
        "edge": [],
        "calendar": [],
        "value": Counter(),
        "avail": Counter(),
        "active": Counter(),
        "series": {k: set() for k in ("value", "avail", "active")},
        "rungs": set(),
    }
    for c in names:
        if re.fullmatch(r"har_ma_\d+", c):
            C["har"].append(c)
        elif re.fullmatch(r"har_ma_\d+_x_(open|close)", c):
            C["edge"].append(c)
        elif m := re.fullmatch(r"adj_(.+)_ma_(\d+)", c):
            C["value"][bmap[m.group(1)]] += 1
            C["series"]["value"].add(m.group(1))
            C["rungs"].add(int(m.group(2)))
        elif m := re.fullmatch(r"(.+)_(avail|active)_ma_(\d+)", c):
            C[m.group(2)][bmap[m.group(1)]] += 1
            C["series"][m.group(2)].add(m.group(1))
        else:
            C["calendar"].append(c)
    return C


def design_tables(P: dict, bmap: dict[str, str]) -> dict:
    A = classify(P["names_all"], bmap)
    B = classify(P["names_base"], bmap)
    # 2026-10-19 (user): parked -- base 5 across the paper. Was:
    # Z = np.load(need(POOLED_CACHE), allow_pickle=True)
    # pooled_names = [str(s) for s in Z["names"]]
    pooled_names = pooled_panel()[0]
    Q = classify(pooled_names, bmap)
    k = len(resolve_har_lags())
    check(
        len(A["har"]) == k == 6 and len(A["calendar"]) == 16 and not A["edge"],
        "per bar: 6 HAR + 16 calendar, no edge",
    )
    check(
        B["har"] == A["har"]
        and B["calendar"] == A["calendar"]
        and len(P["names_base"]) == OUTLINE["p_base"],
        "baseline = the 22 backbone columns of all_features",
    )
    check(
        P["names_base"] == [c for c in P["names_all"] if c in set(P["names_base"])],
        "baseline columns in all_features order",
    )
    nser = {b: sum(1 for s in bmap if bmap[s] == b) for b in BUCKET_LABEL}
    for b in BUCKET_LABEL:
        check(
            A["value"][b] == k * nser[b] and A["avail"][b] == k * nser[b],
            f"{b}: value and avail ladders x {k}",
        )
    act = sorted(A["series"]["active"], key=list(bmap).index)
    check(
        len(act) == OUTLINE["n_active"] and sum(A["active"].values()) == k * len(act),
        "7 activity ladders",
    )
    check(
        len(P["names_all"]) == OUTLINE["p_all"] == 22 + 47 * 6 * 2 + 7 * 6,
        "628 = 22 + 47x6 + 47x6 + 7x6",
    )
    kq = len(Q["har"])
    check(
        # 2026-10-19 (user): parked -- base 5 across the paper. Was: kq == 12 ... "pooled: 12 HAR rungs, 24 edge, 16 calendar"
        kq == 6 and len(Q["edge"]) == 2 * kq and len(Q["calendar"]) == 16,
        "pooled: 6 HAR rungs, 12 edge, 16 calendar",
    )
    # 2026-10-19 (user): parked -- base 5 across the paper. Was:
    # check(Q["rungs"] == {2**i for i in range(12)}, "pooled ladder = base 2, 1 .. 2048")
    check(Q["rungs"] == set(resolve_har_lags()), "pooled ladder = base 5, 1 .. 3125 (the per-bar ladder)")
    check(A["rungs"] == set(resolve_har_lags()), "per-bar ladder = base 5, 1 .. 3125")
    check(
        not Q["series"]["value"] & set(LD.FOMC_FEATURES),
        # 2026-10-19 (user): parked -- the base-5 panel leaves FOMC out by choice. Was: "pooled panel of record is pre-FOMC",
        "pooled panel has no FOMC series",
    )
    check(
        # 2026-10-19 (user): parked -- base 5 across the paper. Was: == 52 + 43 * 12 * 2 + 5 * 12, "1,144 = 52 + 43x12x2 + 5x12"
        # (the 5 activity series were the old full-sample mode rule's; the base-5 panel is built under the causal
        # rule, CAUSAL_MODE_INFLATION, which flags numobs and the three StockTwits series: note S4a)
        len(pooled_names) == OUTLINE["p_pooled"] == 34 + 43 * 6 * 2 + 4 * 6,
        "574 = 34 + 43x6x2 + 4x6",
    )
    check(A["calendar"] == Q["calendar"], "same calendar columns per bar and pooled")
    # 2026-10-15 (table legibility): every column of both designs has a math name (the addenda's lookup)
    names_tex = [column_tex(c) for c in P["names_all"] + pooled_names]
    check(
        len(set(column_tex(c) for c in P["names_all"])) == len(P["names_all"])
        and len(set(column_tex(c) for c in pooled_names)) == len(pooled_names)
        and len(names_tex) > 0,
        "every design column has a distinct math name (series_tex / column_tex)",
    )
    return dict(
        A=A,
        Q=Q,
        k=k,
        kq=kq,
        nser=nser,
        act=act,
        cal=A["calendar"],
        cal_const=P["cal_const"],
        # 2026-10-15 (number macros): the pooled panel's composition, for rehaul_dm_numbers.tex
        n_pooled=len(pooled_names),
        n_series_pooled=len(Q["series"]["value"]),
        n_active_pooled=len(Q["series"]["active"]),
        n_edge_pooled=len(Q["edge"]),
    )


STAT = {  # what each raw stem measures (the file's own column naming; ES = E-mini S&P 500 futures, 1-min returns r)
    "sumret": r"bar return $\sum r$",
    "sumabsret": r"$\sum |r|$",
    "sumret2": r"realized variance $\sum r^2$",
    "sumret3": r"$\sum r^3$",
    "sumret4": r"$\sum r^4$",
    "sumpret2": r"upside semivariance $\sum r^2 \mathbf{1}\{r>0\}$",
    "sumbipow": "bipower variation",
    "sumautocov": r"first-order autocovariance $\sum r_i r_{i-1}$",
    "sumvolume": "contract volume",
    "numobs": r"1-min returns in the bar ($\le 30$)",
    "turnover": "turnover",
    "buyturnover": "buyer-initiated turnover",
    "sellturnover": "seller-initiated turnover",
    "effspread": "effective spread",
    "spread": "quoted spread",
    "ofi": r"order-flow imbalance $(b-s)/(b+s)$ of turnover",
    "stocktwits_attention": "message volume (attention)",
    "stocktwits_sentiment": "net sentiment score",
    "stocktwits_sentcount": "messages with a sentiment tag",
    "vix": "VIX level",
    "vvix": "VVIX level",
    "vix3m": "VIX3M level",
    "voldemand_spx_open_and_close": "SPX options, opening and closing",
    "voldemand_spx_open_only": "SPX options, opening only",
    "voldemand_all_open_and_close": "all options, opening and closing",
    "voldemand_all_open_only": "all options, opening only",
    "fomc_release": "1 on the 14:30 statement bar",
    "fomc_day": "1 on statement days",
    "fomc_until_inv": r"$1/(1+{}$bars to next statement$)$",
    "fomc_since_inv": r"$1/(1+{}$bars since last$)$",
}


def stat_of(s: str) -> str:
    if s in STAT:
        return STAT[s]
    stem, _, who = s.rpartition("_")
    assert who in ("ewstock", "vwstock") and stem in STAT, s
    return STAT[stem]


PROBE = pd.Series([-4.0, 0.25, 4.0, 16.0, 81.0])
CANDIDATES = {
    "sqrt": lambda x: np.sqrt(x),
    "signed sqrt": lambda x: np.sign(x) * np.sqrt(np.abs(x)),
    "cube root": lambda x: np.cbrt(x),
    "signed 4th root": lambda x: np.sign(x) * np.abs(x) ** 0.25,
    "identity": lambda x: x,
    "log": lambda x: np.log(np.clip(x, 1e-12, None)),
}
TRANSFORM_TEX = {
    "sqrt": r"$\sqrt{x}$",
    "signed sqrt": r"$\pm\sqrt{|x|}$",
    "cube root": r"$x^{1/3}$",
    "signed 4th root": r"$\pm|x|^{1/4}$",
    "identity": r"$x$",
    "log": r"$\log x$",
    "asinh": r"$\operatorname{asinh}(x/s_t)$",
}


def semantic_label(col: str, has_neg: bool) -> str:
    """Which branch target.apply_semantic_transform takes for this column, found by running it on a probe."""
    with np.errstate(invalid="ignore", divide="ignore"):
        out = TG.apply_semantic_transform(PROBE.copy(), col, has_neg).to_numpy(float)
        hits = [
            k
            for k, f in CANDIDATES.items()
            if np.allclose(out, f(PROBE.to_numpy(float)), equal_nan=True)
        ]
    assert len(hits) == 1, (col, hits)
    return hits[0]


def dictionary(P: dict, bmap: dict[str, str], D: dict) -> list[dict]:
    """F2: per series, the transform chain robust_transform applies (run on the data the executor hands it)."""
    import src.backtest.executor as EX

    check(
        not LD.RATIO_FILL_COLS,
        "the overnight 1.0 fill is a no-op (RATIO_FILL_COLS empty)",
    )
    df = P["df"].copy()
    df[LD.ALL_FEATURES] = df[
        LD.ALL_FEATURES
    ].ffill()  # load_and_transform: ffill(limit=None), the run's setting
    files: dict[str, str] = {}
    for f in (
        "core_stats",
        "ewstock_stats",
        "vwstock_stats",
        "spy_and_sentiment",
        "vix_and_voldemand",
    ):
        for c in pd.read_parquet(DATA / f"{f}.parquet").columns:
            files.setdefault(c, f)
    rows = []
    for s in LD.ALL_FEATURES:
        x = df[s]
        neg = bool((x.dropna() < 0).any())
        excl = TG.is_diurnal_excluded(s)
        diurnal = "none" if excl else ("s.d." if neg else "mean")
        transform = semantic_label(s, neg)
        if not excl and neg and not TG.has_name_stabilizer(s):
            # the asinh gate: robust_transform swaps in asinh when the s.d. divisor is pinned on >= half the rows
            _, base = TG.diurnal_adjust(
                x, df["time_of_day"], True, slot_band=TG.DIURNAL_SLOT_BAND_FRAC
            )
            pinned = float(np.isclose(base, base.min(), rtol=1e-9).mean())
            _, rb = TG.robust_transform(df, s)
            routed = bool((rb == 1.0).all())
            check(
                routed == (pinned >= TG.DIURNAL_PINNED_MAX),
                f"{s}: asinh routing agrees with the pinned share",
            )
            if routed:
                transform, diurnal = "asinh", "none"
        inflated, mode = EX.mode_inflated(x.to_numpy(dtype=np.float64))
        share = (
            float(np.mean(x.dropna().to_numpy() == mode)) if inflated else float("nan")
        )
        check(
            inflated == (s in D["act"]),
            f"{s}: activity flag = an _active ladder in the design",
        )
        if s in files:
            src = files[s]
        elif s in LD.OFI_PAIRS:
            src = files[LD.OFI_PAIRS[s][0]] + " (derived)"
        else:
            check(s in LD.FOMC_FEATURES, f"{s} source")
            src = "releases (derived)"
        raw = P["df"][s]
        nn = int((raw < 0).sum())
        fneg = (
            pd.Timestamp(P["df"].loc[raw < 0, "t"].iloc[0]).strftime("%Y-%m-%d %H:%M")
            if nn
            else ""
        )
        rows.append(
            dict(
                series=s,
                bucket=bmap[s],
                stat=stat_of(s),
                source=src,
                transform=transform,
                diurnal=diurnal,
                active=inflated,
                mode=mode,
                share=share,
                neg=neg,
                n_neg=nn,
                first_neg=fneg,
            )
        )
    R = pd.DataFrame(rows)
    check(int(R["active"].sum()) == OUTLINE["n_active"], "7 activity flags")
    check(
        set(R.loc[R["transform"] == "asinh", "series"])
        == set(LD.SUBGROUPS["vol_demand"]),
        "asinh = the voldemand family",
    )
    return rows


# ============================================================================ writers: data
# 2026-10-15 (table legibility): parked -- D1 rebuilt to the finance-table style (short caption, words not code paths, one row per series, 5-sentence notes); new version below
# def write_sources(S: dict) -> None:
#     rv0, rv1, nrv = S["rv"]
#     nx, x0, x1, nxs = S["xs"]
#     c0, c1, nq, ne = S["chain"]
#     V = r"\textit{[vendor]}"
#     body = [
#         r"\toprule",
#         row(["Source", "Content", "Freq.", "First", "Last", "Rows", "Role"], "lllllrl"),
#         r"\midrule",
#         row([rf"E-mini S\&P 500 futures (ES); {V}",
#              rf"sums of 1-min log returns per bar: realized variance $\sum r^2$ (the target) and {len(S['core_cols']) - 1} "
#              r"other statistics (return moments, bipower variation, autocovariance, volume, return count)",
#              "30 min", D_(rv0), D_(rv1), n_(nrv), "target; features"], "lllllrl"),
#         row([rf"Constituent cross-section, EW and VW; {V}",
#              rf"{nx} statistics each: {S['xs_moments']} return moments, turnover (total, buyer-, seller-initiated), "
#              r"effective and quoted spread", "30 min", D_(x0), D_(x1), n_(nxs) + r"\newline each", "features"], "lllllrl"),
#         row([rf"StockTwits; {V}", "attention;\\newline sentiment and sentiment count", "30 min",
#              D_(S["att"][0]) + r"\newline " + D_(S["sent"][0]), D_(S["att"][1]),
#              n_(S["att"][2]) + r"\newline " + n_(S["sent"][2]), "features"], "lllllrl"),
#         row([rf"Cboe volatility indices; {V}", "VIX\\newline VIX3M\\newline VVIX", "30 min",
#              r"\newline ".join(D_(S[c][0]) for c in ("vix", "vix3m", "vvix")),
#              r"\newline ".join(D_(S[c][1]) for c in ("vix", "vix3m", "vvix")),
#              r"\newline ".join(n_(S[c][2]) for c in ("vix", "vix3m", "vvix")), "features"], "lllllrl"),
#         row([rf"Option volume demand; {V}", rf"{S['n_vd']} series: SPX or all options; opening and closing, or opening only",
#              "30 min", D_(S["vd"][0]), D_(S["vd"][1]), n_(S["vd"][2]), "features"], "lllllrl"),
#         row([rf"FOMC statement calendar; {V}", "one bar per scheduled statement, 14:30 ET", "event",
#              D_(S["fomc"][0]), D_(S["fomc"][1]), n_(S["fomc"][2]), "features"], "lllllrl"),
#         row([rf"SPXW option chain, expiration days; {V}",
#              r"bid, ask, mid, underlying, implied volatility, delta, hours to expiry, early-close flag",
#              "30 min", D_(c0), D_(c1), n_(nq) + r"\newline quotes;\newline " + n_(ne) + r"\newline expirations",
#              r"trade: entry, exit,\newline implied variance"], "lllllrl"),
#         r"\bottomrule",
#     ]
#     notes = (
#         r"Bar files are keyed on the bar-end stamp (ET) and outer-merged on it (\texttt{src/data/loading.py}); "
#         r"Rows counts the 30-minute bars on which the series printed, before the grid and the weekend cut; "
#         r"First and Last are the first and last print. The chain is stamped in UTC and loaded on its own "
#         r"(\texttt{src/data/spxw.py}). The VIX family, the volume-demand series and the FOMC flag stop before the "
#         rf"sample ends ({D_(rv1)}) and are carried forward from their last print. "
#         rf"Merged but never a feature: the other release flags ({len(S['release_flags'])}), the SPXW slice and "
#         r"model-free implied variance, and the VIX representations. "
#         r"\textit{[vendor]}: no file or document names the vendor (the chain is called OptionMetrics only in a "
#         r"code comment)."
#     )
#     write("rehaul_dm_sources.tex", table("tab:rh_dm_sources", "Data sources and spans.",
#                                          colspec([155, 245, 75, 120, 120, 125, 160]), body, notes))


# 2026-10-15 (table legibility): parked -- D2 rebuilt in panels (pooled intraday / closing interval / out-of-sample) with the guide's terms; new version below
# def write_samples(P: dict) -> None:
#     g0, g1, ng = P["grid"]
#     p0, p1, np_ = P["pooled"]
#     f0, f1, nf = P["pooled_fc"]
#     b0, b1, nb = P["perbar"]
#     c0, c1, nc = P["fc"]
#     r0, r1, nr, v0, nv = P["pre"]
#     s0, s1, ns, s_fc = P["dev"]
#     clp = P["clp"]
#     days = P["days"]
#     W = OUTLINE["pooled_window"]
#     lat, lat_full = P["later"]
#     body = [
#         r"\toprule",
#         row(["Sample", "Start", "End", "Rows", "Window", "Role"], "lllrll"),
#         r"\midrule",
#         row(["Bar grid: bars on which RV printed, weekend cut", D_(g0), D_(g1), n_(ng) + " bars", "--",
#              "construction"], "lllrll"),
#         row([rf"Pooled panel: after the {n_(P['burn'])}-bar burn-in", D_(p0), D_(p1), n_(np_) + " bars",
#              rf"{n_(W)} bars (trees, MinGRU); {n_(clp['TW'] * TG.PERIODS_PER_DAY)} (linear)", "pooled designs"], "lllrll"),
#         row(["Pooled forecasts", D_(f0), D_(f1), n_(nf) + " bars", "--", "pooled scoring"], "lllrll"),
#         row(["Tree development span (pooled rows before 2003)", D_(s0), D_(s1), n_(ns) + " bars",
#              rf"{n_(P['tm']['WINDOW'])} bars, refit every {P['tm']['SCREEN_REFIT']}",
#              r"shipped tree settings"], "lllrll"),
#         row(["Per-bar 16:00 design", D_(b0), D_(b1), n_(nb) + " sessions", rf"{n_(clp['TW'])} sessions",
#              "per-bar designs"], "lllrll"),
#         row(["Per-bar forecast rows", D_(c0), D_(c1), n_(nc) + " sessions", "--", "walk-forward"], "lllrll"),
#         row(["Linear pre-tune rows (16:00 rows of the last-hour design)", D_(r0), D_(r1), n_(nr) + " sessions",
#              rf"{n_(clp['TW'])} sessions", "pre-tuning data"], "lllrll"),
#         row([rf"Pre-tune validation folds ({clp['PRE_FOLDS']} $\times$ {clp['PRE_FOLD_LEN']})", D_(v0), D_(r1),
#              n_(nv) + " sessions", rf"{n_(clp['TW'])}-session fit, {clp['EMBARGO']}-session embargo",
#              "penalty choice"], "lllrll"),
#         row(["Test period: 16:00 rows", D_(clp["FIRST_TRADE_DAY"]), D_(c1), n_(P["test16"]) + " sessions", "--",
#              "out of sample"], "lllrll"),
#         row([rf"Trade days: SPXW expirations ({n_(P['exp_to_end'])}) less early closes ({P['ec_to_end']})",
#              D_(days.min()), D_(days.max()), n_(len(days)) + " days", "--", "scoring, trading"], "lllrll"),
#         row(["Later SPXW expirations", D_(SRC["exp"].index[SRC["exp"].index > pd.Timestamp(clp['LAST_TRADE_DAY'])].min()),
#              D_(SRC["exp"].index.max()), n_(lat) + rf" ({n_(lat_full)} full)", "--", "held back"], "lllrll"),
#         r"\bottomrule",
#     ]
#     notes = (
#         rf"Dates are bar-end dates (ET). The burn-in is the longest HAR rung ({n_(P['burn'])} rows) and is applied to "
#         r"each series as built: the half-hour grid (pooled), the one-bar 16:00 series (per bar) and the two-bar "
#         rf"last-hour series (pre-tune), which is why the per-bar design starts in {b0.year} and the pre-tune "
#         rf"borrows the last-hour rows. The pooled linear models' window is the per-bar {n_(clp['TW'])} sessions, "
#         r"in bars. Full: expirations that are not early closes."
#     )
#     write("rehaul_dm_samples.tex", table("tab:rh_dm_samples", "Samples.", colspec([265, 120, 120, 165, 185, 145]),
#                                          body, notes))


# ============================================================================ writers: features
# 2026-10-15 (table legibility): parked -- F1 rebuilt with the guide's terms (HAR components, missing-data / activity indicators, HAR benchmark, full predictor set) and no code names; new version below
# def write_design(D: dict) -> None:
#     A, Q, k, kq, nser = D["A"], D["Q"], D["k"], D["kq"], D["nser"]
#     al = "lrrrrrr"
#     body = [
#         r"\toprule",
#         r" & & \multicolumn{4}{c}{Per bar (16:00)} & \rdmR Pooled \\",
#         r"\cmidrule(lr){3-6}\cmidrule(l){7-7}",
#         row(["Block", "Series", "Value", "Avail.", "Active", "Total", "Total"], al),
#         r"\midrule",
#         row(["HAR ladder of the target", "1", str(len(A["har"])), "--", "--", str(len(A["har"])), str(len(Q["har"]))], al),
#         row([r"HAR $\times$ open / close", "--", "--", "--", "--", "--", str(len(Q["edge"]))], al),
#         row(["Calendar", "--", "--", "--", "--", str(len(A["calendar"])), str(len(Q["calendar"]))], al),
#         r"\addlinespace",
#         row([r"\textbf{HAR + calendar}", "", "", "", "", rf"\textbf{{{len(A['har']) + len(A['calendar'])}}}",
#              str(len(Q["har"]) + len(Q["edge"]) + len(Q["calendar"]))], al),
#         r"\midrule",
#     ]
#     tv = ta = tc = tq = 0
#     for b, lab in BUCKET_LABEL.items():
#         v, a, c = A["value"][b], A["avail"][b], A["active"][b]
#         q = Q["value"][b] + Q["avail"][b] + Q["active"][b]
#         tv, ta, tc, tq = tv + v, ta + a, tc + c, tq + q
#         body.append(row([lab, str(nser[b]), str(v), str(a), str(c) if c else "--", str(v + a + c),
#                          str(q) if q else "--"], al))
#     body += [
#         r"\addlinespace",
#         row([r"Exogenous", str(sum(nser.values())), str(tv), str(ta), str(tc), str(tv + ta + tc), n_(tq)], al),
#         r"\midrule",
#         row([r"\textbf{All features}", "", "", "", "", rf"\textbf{{{tv + ta + tc + len(A['har']) + len(A['calendar'])}}}",
#              rf"\textbf{{{n_(tq + len(Q['har']) + len(Q['edge']) + len(Q['calendar']))}}}"], al),
#         r"\bottomrule",
#     ]
#     act = ", ".join(ttb(s) for s in D["act"])
#     pr = sorted(set(LD.ALL_FEATURES) - Q["series"]["value"])
#     check(set(pr) == set(LD.FOMC_FEATURES), "the pooled panel lacks exactly the FOMC series")
#     backbone_q = len(Q["har"]) + len(Q["edge"]) + len(Q["calendar"])
#     with_fomc = backbone_q + 2 * kq * sum(nser.values()) + kq * len(D["act"])
#     words = {12: "twelve", 6: "six"}
#     notes = (
#         rf"Value: trailing means $\bar x_{{t,k}}$ of the adjusted series, $k \in \{{{', '.join(str(r) for r in resolve_har_lags())}\}}$ "
#         rf"bars of the one-bar series per bar, $k = 1, 2, 4, \dots, 2048$ bars pooled, all lagged one bar. Avail.: "
#         r"the same ladder of the is-observed flag (taken before the forward fill). Active: the same ladder of an "
#         rf"off-the-mode flag for the {len(D['act'])} mode-inflated series ({act}). Calendar: weekday dummies, hour, "
#         r"overnight / open / close session flags, option-expiration day and week, quad witching, rebalancing close, "
#         rf"month and quarter end, days to expiration. Per bar, the {words.get(2 * k, str(2 * k))} HAR $\times$ open / "
#         r"close columns are zero or copies of the HAR ladder at 16:00 and are dropped. The pooled panel of record "
#         rf"predates the FOMC bucket ({len(pr)} series) and so has no FOMC columns; with them it has {n_(with_fomc)}. "
#         r"At each fit, columns constant on the training window or copies of an earlier column are dropped."
#     )
#     write("rehaul_dm_design.tex", table("tab:rh_dm_design", "Design composition: columns by block.",
#                                         colspec([300, 100, 120, 120, 120, 120, 120]), body, notes))


# 2026-10-15 (table legibility): parked -- F2 rebuilt as a variable-definition table (math names, words, source, transform; landscape); new version below
# def write_dictionary(rows: list[dict]) -> None:
#     import inspect
#
#     import src.backtest.executor as EX
#
#     cs = colspec([220, 300, 190, 130, 80, 80])
#     al = "llllll"
#     order = list(BUCKET_LABEL)
#     rows = sorted(rows, key=lambda r: (order.index(r["bucket"]), LD.ALL_FEATURES.index(r["series"])))
#     head = row(["Series", "Raw statistic", "Source file", "Transform", "Diurnal", "Active"], al)
#     act = [r for r in rows if r["active"]]
#     modes = "; ".join(f"{ttb(r['series'])} {r['mode']:g} ({100 * r['share']:.0f}\\%)" for r in act)
#     ww = inspect.signature(TG.rolling_winsorize).parameters["window"].default
#     check(f"else {ww}" in inspect.getsource(TG.robust_transform), "exog winsor window = robust_transform's default")
#     band = TG.DIURNAL_SLOT_BAND_FRAC
#     signed_id = [r["series"] for r in rows if r["transform"] == "identity" and r["neg"]]
#     notes = (
#         r"\textit{Notes:} One row per exogenous series, by bucket. Derived programmatically: the bucket from "
#         r"\texttt{src/data/loading.py}; the transform from the branch \texttt{apply\_semantic\_transform} takes for the "
#         r"column name and sign (\texttt{src/features/transforms/target.py}); the diurnal adjustment and the asinh "
#         r"routing from \texttt{robust\_transform} run on the forward-filled panel; the activity flag from the "
#         rf"executor's mode-inflation rule (modal value on at least {100 * EX.ZERO_INFLATED_FRAC:.0f}\,\% of rows), "
#         r"checked against the design's \texttt{\_active} columns. Diurnal: mean, divided by the trailing same-slot "
#         rf"mean ({TG.DIURNAL_WINDOW} observations, at least {TG.DIURNAL_MIN_PERIODS}, lagged one bar); s.d., divided by "
#         r"the trailing same-slot standard deviation (floored); both banded to "
#         rf"$[{band:g}, {1 / band:g}]$ times the slot's trailing {TG.DIURNAL_SLOT_REF_WINDOW}-observation median; none: "
#         r"the VIX family, sentiment and the FOMC flags. asinh: a signed series whose s.d.\ divisor sits on its floor "
#         rf"on at least {100 * TG.DIURNAL_PINNED_MAX:.0f}\,\% of rows skips the division and takes "
#         r"$\operatorname{asinh}(x/s_t)$, $s_t$ a causal EWMA of $|x|$. $x$: no transform (signed series and "
#         r"\texttt{sumabsret}); a series counts as signed if it has any negative value"
#         + "".join(
#             f" ({tt(r['series'])} is signed by {r['n_neg']} negative print{'s' if r['n_neg'] > 1 else ''}, "
#             f"{r['first_neg']}, and so takes $x$ and s.d.\\ where its EW twin takes $\\log x$ and mean)"
#             for r in rows if r["series"] in signed_id and "spread" in r["series"])
#         + rf". After the transform every series is winsorised at rolling {ww}-bar "
#         rf"{100 * TG.WINSOR_LOWER_Q:.0f}\,/\,{100 * TG.WINSOR_UPPER_Q:.0f}\,\% bounds, set to its median where "
#         r"unobserved, laddered and scaled. Modes of the active series (share of rows): " + modes + "."
#     )
#     out = [
#         HEAD,
#         r"\begingroup",
#         r"\footnotesize",
#         r"\setlength{\tabcolsep}{4pt}",
#         r"\setlength{\LTcapwidth}{\textwidth}",
#         rf"\begin{{longtable}}{{{cs}}}",
#         r"\caption{Feature dictionary: the 47 exogenous series.}\label{tab:rh_dm_dictionary}\\",
#         r"\toprule",
#         head,
#         r"\midrule",
#         r"\endfirsthead",
#         r"\multicolumn{6}{@{}l}{\textit{Table \ref{tab:rh_dm_dictionary} (continued)}}\\",
#         r"\toprule",
#         head,
#         r"\midrule",
#         r"\endhead",
#         r"\midrule",
#         r"\multicolumn{6}{r@{}}{\textit{continued on the next page}}\\",
#         r"\endfoot",
#         r"\bottomrule",
#         r"\multicolumn{6}{@{}p{\textwidth}@{}}{\rdmL\rule{0pt}{2.5ex}" + notes + r"}\\",
#         r"\endlastfoot",
#     ]
#     cur = None
#     for r in rows:
#         if r["bucket"] != cur:
#             if cur is not None:
#                 out.append(r"\addlinespace")
#             cur = r["bucket"]
#             nb = sum(1 for x in rows if x["bucket"] == cur)
#             out.append(group(f"{BUCKET_LABEL[cur]} ({nb})", 6))
#         out.append(row([ttb(r["series"]), r["stat"], ttb(r["source"].replace(" (derived)", "")) +
#                         (" (derived)" if "derived" in r["source"] else ""),
#                         TRANSFORM_TEX[r["transform"]], r["diurnal"], "yes" if r["active"] else "no"], al))
#     out += [r"\end{longtable}", r"\endgroup", ""]
#     write("rehaul_dm_dictionary.tex", "\n".join(out))


# ============================================================================ methods
def tree_settings(P: dict) -> dict:
    T = consts(
        REPO / "specs" / "causal_tune_trees.py",
        ("LGBM_PARAMS", "SEED", "REFIT_EVERY", "POOLED_WINDOW_ROWS"),
    )
    X = json.loads(need(RES / "trees_xgbtop_1600" / "xgb_config.json").read_text())
    XT = consts(REPO / "specs" / "causal_tune_trees_xgbtop.py", ("XGB_PARAMS",))[
        "XGB_PARAMS"
    ]
    R4 = consts(REPO / "specs" / "causal_tune_trees_rf400.py", ("RF_PARAMS",))[
        "RF_PARAMS"
    ]
    rf = P["rf"]
    scale = OUTLINE["perbar_window"] / T["POOLED_WINDOW_ROWS"]
    check(
        XT == X["config"]["params"], "xgbtop spec params = xgb_config.json (xgb_opt_00)"
    )
    check(
        abs(
            XT["min_child_weight"] * scale
            - X["per_bar_params_train_win_2000"]["min_child_weight"]
        )
        < 1e-12,
        "XGBoost min_child_weight scaled x 2000/24000",
    )
    check(
        R4["min_samples_leaf"]
        == rf["min_samples_leaf_scaled_to_2000_rows"]
        == max(1, round(rf["params"]["min_samples_leaf"] * scale)),
        "forest leaf 12 -> 1",
    )
    check(
        {k: v for k, v in R4.items() if k != "min_samples_leaf"}
        == {k: v for k, v in rf["params"].items() if k != "min_samples_leaf"},
        "rf400 spec = rf_config.json",
    )
    M = macros()
    check(
        M["cmTreeProvRfTrees"] == str(rf["params"]["n_estimators"])
        and M["cmTreeProvRfDepth"] == str(rf["params"]["max_depth"])
        and M["cmTreeProvRfFeatures"] == f"{rf['params']['max_features']:.2f}"
        and M["cmTreeProvRfSamples"] == f"{rf['params']['max_samples']:.2f}"
        and M["cmTreeProvRfTrials"] == str(rf["trials"]["complete_considered"])
        and M["cmTreeProvRfLeaf"] == str(R4["min_samples_leaf"]),
        r"forest settings = the \cmTreeProvRf* macros",
    )
    check(
        M["cmTreeProvLgbmLeaf"]
        == str(max(1, round(T["LGBM_PARAMS"]["min_child_samples"] * scale))),
        "LightGBM leaf 98 -> 8 = the macro",
    )
    check(int(rf["trial"]) == 255, "forest = trial 255")
    return dict(
        T=T, X=XT, Xs=X["per_bar_params_train_win_2000"], R=R4, rf=rf, scale=scale, M=M
    )


def linear_picks(P: dict) -> dict:
    f = P["facts"]
    M = macros()
    L = {}
    for est, tag in (("ridge", "Ridge"), ("lasso", "Lasso"), ("enet", "Enet")):
        for b, bt in (("baseline", "Base"), ("all_features", "All")):
            pen = float(f[f"{est} {b}: pre-tuned penalty"])
            fired = f[f"{est} {b}: retunes fired"]
            want = M[f"cmPtPen{tag}{bt}"]
            k = int(round(math.log10(pen)))
            check(want == rf"\ensuremath{{10^{{{k}}}}}", rf"\cmPtPen{tag}{bt} = 10^{k}")
            L[(est, b)] = (pen, fired)
    G = consts(
        REPO / "specs" / "causal_tune_linear.py",
        ("ESTIMATOR_GRIDS", "TUNE_PER", "VAL_TAIL", "EMBARGO"),
    )
    grids = {k: [a for _, a, _ in v] for k, v in G["ESTIMATOR_GRIDS"].items()}
    check(
        grids["ridge"] == [10.0**k for k in range(-2, 4)]
        or np.allclose(grids["ridge"], np.logspace(-2, 3, 6)),
        "ridge grid 1e-2 .. 1e3",
    )
    check(
        np.allclose(grids["reclasso"], np.logspace(-6, -2, 5))
        and np.allclose(grids["reclasticnet"], np.logspace(-6, -2, 5)),
        "lasso / enet grid 1e-6 .. 1e-2",
    )
    check(
        {l1 for _, _, l1 in G["ESTIMATOR_GRIDS"]["reclasticnet"]} == {0.5},
        "elastic net rho = 0.5",
    )
    check(
        float(f["backbone penalty ratio (HAR + calendar / exogenous)"])
        == P["clp"]["R_BB"]
        == 0.01,
        "R_BB = 0.01",
    )
    check(
        f["retunes fired, total"].startswith("4 of") and M["cmPtFiredTotal"] == "4",
        "4 retunes fired",
    )
    return dict(L=L, grids=grids, G=G, M=M)


def mingru() -> dict:
    """The MinGRU grids from the job files the walks ran, the picks from their selection files."""
    out = {}

    def grid(jl: list[dict], keys: tuple[str, ...]) -> dict:
        vals = {k: sorted({float(j[k]) for j in jl if k in j}) for k in keys}
        return {k: v for k, v in vals.items() if v}

    # per bar, the base walk (softplus, MSE): the 12-configuration grid of phase2.py
    ph = consts(PF / "phase2.py", ("BLOCK", "W", "SEEDS", "GRID", "COMMON"))
    base_jobs = []
    for k in range(5):
        base_jobs += jobs(PFW / "walk_softplus" / f"tune_block{k}.txt")
    gb = grid(base_jobs, ("L", "H", "lr"))
    check(
        {(int(g["L"]), int(g["H"]), g["lr"]) for g in ph["GRID"]}
        == {(int(j["L"]), int(j["H"]), float(j["lr"])) for j in base_jobs},
        "base walk grid = phase2.GRID",
    )
    seeds = sorted({int(j["seed"]) for j in base_jobs})
    check(tuple(seeds) == ph["SEEDS"] and len(seeds) == 5, "5 seeds, phase2.SEEDS")
    check(
        len(base_jobs) == 5 * 2 * len(ph["GRID"]) * len(seeds),
        "base walk: 5 blocks x 2 designs x 12 x 5 seeds",
    )
    out["base"] = dict(
        grid=gb,
        n=len(ph["GRID"]),
        seeds=seeds,
        common=ph["COMMON"],
        sel=pd.read_csv(need(PFR / "walk_softplus_p100" / "selection.csv")),
    )
    # per bar, shortcut + ridge init, trained / early-stopped / selected on QLIKE (qlike_study.walk_inputs)
    qj = []
    for k in range(5):
        qj += jobs(PFW / "walk_softplus_rinit_q" / f"tune_block{k}.txt")
    gq = grid(qj, ("L", "H", "lr", "wd_enc", "wd_lin", "lr_lin"))
    nq = math.prod(len(v) for v in gq.values())
    cfg = {
        (j["data"], j["L"], j["H"], j["lr"], j["wd_enc"], j["wd_lin"], j["lr_lin"])
        for j in qj
        if j["w1"] == qj[0]["w1"]
    }
    check(
        len(cfg) == 2 * nq == 2 * 216,
        "QLIKE walk: 216 configurations per design, full factorial",
    )
    check(
        {j["loss"] for j in qj} == {"qlike"} and {j["es"] for j in qj} == {"qlike"},
        "trained and early-stopped on QLIKE",
    )
    check(
        {j["link"] for j in qj} == {"softplus"} and {j["lin"] for j in qj} == {"1"},
        "softplus link, linear shortcut",
    )
    bq = json.loads(need(PFW / "walk_softplus_rinit_q" / "blocks.json").read_text())
    origins = [o for b in bq["blocks"] for o in b["origins"]]
    dates = pd.read_csv(need(PFW / "dates.csv"))["date"]
    check(
        len(origins) == OUTLINE["mingru_origins"]
        and d_(dates[origins[0]]) == "2020-01-02"
        and all(b["n_origins"] <= bq["block"] for b in bq["blocks"])
        and len(bq["blocks"]) == 5,
        "MinGRU walk: 1,088 origins from 2020-01-02 in 5 blocks of 250",
    )
    out["q"] = dict(
        grid=gq,
        n=nq,
        common=bq["common"],
        seeds=bq["seeds"],
        blocks=bq,
        sel=pd.read_csv(need(PFR / "qlike_study" / "score_rinitq" / "selection.csv")),
        first=dates[origins[0]],
    )
    # pooled (pooled_walk.py), the wd-grid walk
    pw = consts(PF / "pooled_walk.py", ("W", "VAL", "EMB", "GRID", "WDS", "COMMON"))
    pj = []
    for k in range(5):
        pj += jobs(PFW / "pwalk_wdgrid" / f"tune_block{k}.txt")
    gp = grid(pj, ("L", "H", "lr", "wd"))
    check(
        math.prod(len(v) for v in gp.values())
        == len(pw["GRID"]) * len(pw["WDS"])
        == 48,
        "pooled grid 8 x 6 wd = 48",
    )
    plan = json.loads(need(PFW / "pwalk_wdgrid" / "plan.json").read_text())
    src = (PF / "pooled_walk.py").read_text(encoding="utf-8")
    mref = re.search(r"range\(0, len\(origins\), (\d+)\)", src)
    check(mref is not None, "pooled refit spacing in pooled_walk.prep")
    assert mref is not None
    sel = pd.read_csv(need(PFR / "pooled_wd_study" / "selection.csv"))
    out["pooled"] = dict(
        grid=gp,
        n=48,
        common=pw["COMMON"],
        seeds=plan["seeds"],
        W=pw["W"],
        VAL=pw["VAL"],
        EMB=pw["EMB"],
        refit=int(mref.group(1)),
        sel=sel[sel["walk"] == "wdgrid"],
    )
    # the pooled ridge-init QLIKE walk, if its inputs exist (running at the time of writing)
    rq = PFW / "pwalk_rinitq_all_features"
    if (rq / "tune_block0.txt").is_file():
        rj = jobs(rq / "tune_block0.txt")
        out["pooled_q"] = dict(
            grid=grid(rj, ("L", "H", "lr", "wd_enc", "wd_lin", "lr_lin")),
            seeds=sorted({int(j["seed"]) for j in rj}),
        )
    # 2026-10-15 (table legibility): the pooled MinGRU of record is the ridge-initialized QLIKE walk
    # (make_rehaul_results.py SOURCES: pooled_rinit_study/runs/pwalk_rinitq_composite); its grid from both designs'
    # job files, its per-block picks from the ten pooled_rinit_study/runs/prq_<design>_b<k>_sel_*/selection.json
    keys6 = ("L", "H", "lr", "wd_enc", "wd_lin", "lr_lin")
    pq_jobs: dict[str, list[dict]] = {}
    for des in ("baseline", "all_features"):
        pq_jobs[des] = []
        for k in range(5):
            pq_jobs[des] += jobs(PFW / f"pwalk_rinitq_{des}" / f"tune_block{k}.txt")
    gpq = grid(pq_jobs["all_features"], keys6)
    check(
        grid(pq_jobs["baseline"], keys6) == gpq,
        "pooled QLIKE walk: same grid for both designs",
    )
    cfg_pq = {tuple(j[k] for k in keys6) for j in pq_jobs["all_features"]}
    npq = math.prod(len(v) for v in gpq.values())
    check(len(cfg_pq) == npq, "pooled QLIKE walk: full factorial grid")
    check(
        {j["loss"] for js in pq_jobs.values() for j in js} == {"qlike"}
        and {j["es"] for js in pq_jobs.values() for j in js} == {"qlike"},
        "pooled walk trained and stopped on QLIKE",
    )
    sel_rows = []
    for des in ("baseline", "all_features"):
        for k in range(5):
            hit = sorted(
                (PFR / "pooled_rinit_study" / "runs").glob(
                    f"prq_{des}_b{k}_sel_*/selection.json"
                )
            )
            check(len(hit) == 1, f"one selection file for pooled QLIKE {des} block {k}")
            s = json.loads(hit[0].read_text())[des]
            check(
                s["criterion"] == "qlike",
                f"pooled QLIKE {des} block {k} selected on QLIKE",
            )
            sel_rows.append(
                dict(block=k, design=des, cfg=s["cfg"], n_val=int(s["n_val"]))
            )
    pplan = json.loads(
        need(PFW / "pwalk_rinitq_all_features" / "plan.json").read_text()
    )
    pblocks = (
        pplan["blocks"]
        if isinstance(pplan["blocks"], list)
        else ast.literal_eval(pplan["blocks"])
    )
    nper = [len(b["origins"]) for b in pblocks]
    check(
        len(pblocks) == 5 and len(set(nper[:-1])) == 1 and nper[-1] <= nper[0],
        "pooled QLIKE walk: 5 blocks, equal but for a shorter last one",
    )
    check(
        all(r["n_val"] == pplan["common"]["val_tail"] for r in sel_rows),
        "pooled QLIKE validation tail = plan",
    )
    out["pooled_rq"] = dict(
        grid=gpq,
        n=npq,
        common=pplan["common"],
        seeds=pplan["seeds"],
        W=pplan["W"],
        VAL=pplan["common"]["val_tail"],
        EMB=pplan["common"]["embargo"],
        refit=out["pooled"]["refit"],
        blocks=len(pblocks),
        per_block=nper[0],
        sel=pd.DataFrame(sel_rows),
    )
    return out


# 2026-10-15 (table legibility): parked -- M1 rebuilt with the guide's symbols and readable parameter names (no library argument names, no code tags); new version below
# def write_hyper(P: dict, TS: dict, LP: dict, NN: dict) -> None:
#     """M1, in two floats: (i) the linear models and the shipped tree settings, (ii) what is searched in the walk
#     (the re-tuned trees and the MinGRU grids).  One row per hyperparameter."""
#     T, X, Xs, R, rf = TS["T"], TS["X"], TS["Xs"], TS["R"], TS["rf"]
#     lg = T["LGBM_PARAMS"]
#     L = LP["L"]
#     check(sum(int(v[1].split(" of ")[0]) for v in L.values()) == int(LP["M"]["cmPtFiredTotal"])
#           and sum(int(v[1].split(" of ")[1]) for v in L.values()) == int(LP["M"]["cmPtFiredOf"]),
#           r"retunes fired per arm sum to \cmPtFiredTotal of \cmPtFiredOf")
#     al = "lllll"
#     sc = TS["scale"]
#     oj = consts(REPO / "specs" / "causal_tune_trees_optuna_jobs.py",
#                 ("SPACES", "TPE_SEED_BASE", "ROUNDS_CAP_MULT", "XGB_SUBSAMPLE_LO", "XGB_REG_LAMBDA_LO"))
#     op = consts(REPO / "specs" / "causal_tune_trees_optuna.py", ("N_TRIALS_DEFAULT", "BUDGETS"))
#     ts = consts(REPO / "specs" / "causal_tune_trees_tuned.py", ("N_CANDIDATES", "GRID_SEED", "RF_MAX_FEATURES",
#                                                                  "RF_MAX_DEPTH", "RF_MIN_LEAF"))
#     xlo = min(oj["XGB_REG_LAMBDA_LO"], 10.0 ** math.floor(math.log10(X["reg_lambda"])))
#     check(re.search(r"min\(OJ\.XGB_REG_LAMBDA_LO, 10\.0 \*\* math\.floor\(math\.log10\(shipped\)\)\)",
#                     (REPO / "specs" / "causal_tune_trees_optuna_xgbtop.py").read_text(encoding="utf-8")) is not None
#           and xlo == 1e-5, "xgbtop reg_lambda floor 1e-5")
#     tdoc = (REPO / "specs" / "causal_tune_trees_tuned.py").read_text(encoding="utf-8")
#     msz = re.search(r"The full products are (\d+) \(LightGBM\), (\d+)\s*\n?\s*\(XGBoost\) and (\d+) \(RF\)", tdoc)
#     check(msz is not None and int(msz.group(3)) == len(ts["RF_MAX_FEATURES"]) * len(ts["RF_MAX_DEPTH"]) * len(ts["RF_MIN_LEAF"]),
#           "random-search grid sizes (RF = 4 x 4 x 7)")
#     box = {fam: {n: (lo, hi) for n, lo, hi, _, _ in P["tm"]["BOXES"][fam]} for fam in ("lgbm", "xgb")}
#     box["rf"] = {b[0]: (b[1], b[2]) for b in rf["provenance"]["box"]}
#     nblk = len(NN["q"]["sel"])
#     check(nblk == len(NN["base"]["sel"]) == len(NN["pooled"]["sel"]) == 5 * 2, "10 (block, design) picks per network walk")
#
#     def rng(lo, hi) -> str:
#         s = f"[{gm(lo)}, {gm(hi)}]"
#         return f"${s}$" if ("^" in s or "\\" in s) else s
#
#     def num(v) -> str:
#         """Integers as is; below 1e-3 in scientific form (2 digits); [0.1, 1) with 2 decimals (as the macros);
#         otherwise 3 significant digits."""
#         if isinstance(v, str):
#             return v
#         v = float(v)
#         if v == int(v) and abs(v) < 1e6:
#             return str(int(v))
#         if abs(v) < 1e-3:
#             return g_(float(f"{v:.2g}"))
#         if 0.1 <= abs(v) < 1:
#             return f"{v:.2f}"
#         return f"{v:.3g}"
#
#     def grid10(gr) -> str:
#         ks = [int(round(math.log10(a))) for a in gr]
#         check(ks == list(range(ks[0], ks[-1] + 1)), "a one-per-decade grid")
#         return rf"$\{{10^{{{ks[0]}}}, 10^{{{ks[1]}}}, \dots, 10^{{{ks[-1]}}}\}}$ ({len(gr)})"
#
#     def lin_chosen(est: str) -> str:
#         return (rf"HAR + cal.\ {g_(L[(est, 'baseline')][0], True)};"
#                 rf" all {g_(L[(est, 'all_features')][0], True)}")
#
#     # ------------------------------------------------------------------ (i) linear + shipped trees
#     body = [
#         r"\toprule",
#         row(["Model", "Hyperparameter", "Grid or search space", "Chosen", "Chosen by"], al),
#         r"\midrule",
#         group("Linear, per bar", 5),
#         row(["OLS", "none", "--", "--", "minimum-norm least squares"], al),
#         row(["Ridge", r"$\alpha$", grid10(LP["grids"]["ridge"]), lin_chosen("ridge"),
#              rf"pre-tune (mean validation MSE, {P['clp']['PRE_FOLDS']} folds), then the margin retune"], al),
#         row(["Lasso", r"$\alpha$", grid10(LP["grids"]["reclasso"]), lin_chosen("lasso"), "as ridge"], al),
#         row(["Elastic net", r"$\alpha$", grid10(LP["grids"]["reclasticnet"]), lin_chosen("enet"), "as ridge"], al),
#         row(["", r"$\rho$ (L1 share)", "fixed", "0.5", "set a priori"], al),
#         row(["Block penalty", r"$R_{BB}$, backbone / exogenous", "fixed", g_(P["clp"]["R_BB"]), "set a priori"], al),
#     ]
#     shipped = {
#         "lgbm": ("LightGBM", lg, ("num_leaves", "learning_rate", "n_estimators", "min_child_samples", "feature_fraction",
#                                   "bagging_fraction", "lambda_l1", "lambda_l2"),
#                  {"min_child_samples": max(1, round(lg["min_child_samples"] * sc))}, [r"\texttt{lgbm\_opt\_00}"]),
#         "xgb": ("XGBoost", X, ("max_depth", "learning_rate", "n_estimators", "min_child_weight", "subsample",
#                                "colsample_bytree", "reg_alpha", "reg_lambda", "gamma"),
#                 {"min_child_weight": Xs["min_child_weight"]}, [r"\texttt{xgb\_opt\_00}"]),
#         "rf": ("Random forest", rf["params"], ("n_estimators", "max_depth", "max_features", "max_samples", "min_samples_leaf"),
#                {"min_samples_leaf": R["min_samples_leaf"]},
#                [r"\texttt{rf\_opt\_00}", rf"trial {rf['trial']} of {rf['trials']['complete_considered']}"]),
#     }
#     body += [r"\addlinespace", group(r"Trees, per bar: shipped settings from the 1998--2002 pooled development search", 5)]
#     for fam, (name, prm, keys, scaled, entry) in shipped.items():
#         if fam != "lgbm":
#             body.append(r"\addlinespace")
#         for i, k in enumerate(keys):
#             ch = num(prm[k])
#             if k in scaled:
#                 ch = rf"{num(prm[k])} $\to$ {num(scaled[k])}"
#             by = ["top entry of the menu", *entry]
#             body.append(row([name if i == 0 else "", ttb(k), rng(*box[fam][k]), ch, by[i] if i < len(by) else ""], al))
#     body.append(r"\bottomrule")
#     # fixed parameters outside the boxes, read from the shipped dicts
#     fixed_lg = {k: v for k, v in lg.items() if k not in shipped["lgbm"][2]}
#     fixed_xg = {k: v for k, v in X.items() if k not in shipped["xgb"][2]}
#     fixed_rf = {k: v for k, v in rf["params"].items() if k not in shipped["rf"][2]}
#     fx = "; ".join(
#         f"{nm} " + ", ".join(f"{tt(k)} = {v}" for k, v in d.items())
#         for nm, d in (("LightGBM", fixed_lg), ("XGBoost", fixed_xg), ("forest", fixed_rf)))
#     fired = "; ".join(
#         f"{lab} {L[(e, 'baseline')][1].split(' of ')[0]} and {L[(e, 'all_features')][1].split(' of ')[0]}"
#         for e, lab in (("ridge", "ridge"), ("lasso", "lasso"), ("enet", "elastic net")))
#     notes_a = (
#         r"Linear: sklearn units, one grid value per decade (\texttt{specs/causal\_tune\_linear.py}); picks are the "
#         r"pre-tuned penalties on the HAR + calendar (HAR + cal.) and all-features designs. The margin retune at the "
#         rf"{L[('ridge', 'baseline')][1].split(' of ')[1]} block starts then switched (HAR + cal.\ and all): {fired}. "
#         r"$R_{BB}$ multiplies the penalty on the HAR + calendar columns; the intercept is unpenalised. Trees: the "
#         r"search box of the development menu (LightGBM and XGBoost: \texttt{experiments/tree\_menu\_dev.py}; forest: "
#         rf"its {rf['trials']['nominal_budget']}-trial study), searched by Optuna on the pooled "
#         rf"{n_(P['tm']['WINDOW'])}-bar window and ranked by development-span MSE; $a \to b$: a leaf minimum tuned on "
#         rf"{n_(T['POOLED_WINDOW_ROWS'])} rows, scaled to the {n_(OUTLINE['perbar_window'])}-row per-bar window. "
#         rf"Fixed: {fx}. Seed {T['SEED']}."
#     )
#
#     # ------------------------------------------------------------------ (ii) searched in the walk
#     body_b = [
#         r"\toprule",
#         row(["Model", "Hyperparameter", "Search space or grid", "Chosen", "Chosen by"], al),
#         r"\midrule",
#         group(rf"Trees, per bar, re-tuned in the walk: Optuna TPE, {op['N_TRIALS_DEFAULT']} trials, trial 0 = shipped", 5),
#     ]
#     names = {"lgbm": "LightGBM", "xgb": "XGBoost", "rf": "Random forest"}
#     for fam in ("lgbm", "xgb", "rf"):
#         for i, (k, v) in enumerate(oj["SPACES"][fam].items()):
#             if v[0] == "cat":
#                 space = gset(v[1])
#             else:
#                 lo = xlo if (fam == "xgb" and k == "reg_lambda") else v[1]
#                 space = rng(lo, v[2]) + (" (log)" if v[3] else "")
#             by = ["validation MSE", r"(\texttt{tp25\_q}: QLIKE)"] if fam == "lgbm" else ["as LightGBM"]
#             body_b.append(row([names[fam] if i == 0 else "", ttb(k), space,
#                                "per tuning point" if i == 0 else "", by[i] if i < len(by) else ""], al))
#     body_b.append(row(["Random search", "the same axes", rf"{ts['N_CANDIDATES']} fixed candidates from "
#                        rf"{n_(int(msz.group(1)))} / {n_(int(msz.group(2)))} / {msz.group(3)} configurations",
#                        "per tuning point", "validation MSE"], al))
#     b, q, p = NN["base"], NN["q"], NN["pooled"]
#
#     def nn_rows(label: list[str], d: dict, keys: tuple[tuple[str, str, str], ...], rule: list[str]) -> list[str]:
#         cnt = pick_counts(d["sel"])
#         out = []
#         for i, (gk, sk, lab) in enumerate(keys):
#             v, m = cnt[sk]
#             out.append(row([label[i] if i < len(label) else "", lab, gset(d["grid"][gk]),
#                             f"{g_(v)} ({m}/{len(d['sel'])})", rule[i] if i < len(rule) else ""], al))
#         return out
#
#     base_keys = (("L", "L", "$L$, sequence length"), ("H", "H", "$H$, width"), ("lr", "lr", "learning rate"))
#     body_b += [r"\addlinespace", group("MinGRU network", 5)]
#     body_b += nn_rows(["Per bar, base", f"({b['n']} configurations)"], b, base_keys,
#                       ["lowest validation MSE", f"of the {len(b['seeds'])}-seed mean"])
#     body_b.append(r"\addlinespace")
#     body_b += nn_rows(["Per bar, QLIKE", f"({q['n']} configurations)"], q, base_keys + (
#         ("wd_enc", "we", r"$w_{\mathrm{enc}}$, encoder decay"), ("wd_lin", "wl", r"$w_{\mathrm{lin}}$, shortcut L2"),
#         ("lr_lin", "ll", r"shortcut learning rate")), ["lowest validation QLIKE", f"of the {len(q['seeds'])}-seed mean"])
#     body_b.append(r"\addlinespace")
#     body_b += nn_rows(["Pooled", f"({p['n']} configurations)"], p, base_keys + (("wd", "wd", "weight decay"),),
#                       ["lowest validation MSE", f"of the {len(p['seeds'])}-seed mean"])
#     body_b.append(r"\bottomrule")
#     bc = b["common"]
#     notes = (
#         r"Re-tuned trees: log: a log-uniform axis; early stopping with patience $\lceil 1/\mathrm{lr}\rceil$ and a "
#         rf"cap of {oj['ROUNDS_CAP_MULT']}$\times$ the shipped shrinkage budget; best of the first "
#         rf"{', '.join(str(x) for x in op['BUDGETS'])} trials recorded; the XGBoost \texttt{{reg\_lambda}} floor is "
#         rf"{g_(xlo)} so that the shipped value is inside; other parameters stay shipped. Random search: the shipped "
#         rf"configuration and {ts['N_CANDIDATES'] - 1} drawn once (seed {ts['GRID_SEED']}) from the LightGBM / XGBoost / "
#         r"forest product grids. MinGRU: in parentheses, the grid size per design; base, the softplus network trained "
#         r"on squared error; QLIKE, the network with the linear shortcut started at the window's ridge, trained, "
#         r"early-stopped and selected on QLIKE; pooled, the 48-bar network. $w_{\mathrm{enc}}$: decay on the encoder's "
#         r"input columns, scaled by the linear penalty factors; $w_{\mathrm{lin}}$: the shortcut's L2 penalty. Fixed: "
#         rf"{bc['layers']} layer, batch {bc['batch']}, at most {bc['max_epochs']} epochs (pooled "
#         rf"{p['common']['max_epochs']}), patience {bc['patience']} (pooled {p['common']['patience']}), Muon momentum "
#         rf"{bc['momentum']}, gradient clip {bc['max_grad_norm']}, cosine schedule, softplus output. Chosen: the modal "
#         rf"pick over the {nblk // 2} blocks $\times$ 2 designs, with its count."
#     )
#     cs = colspec([150, 210, 225, 165, 250])
#     ta = table("tab:rh_dm_hyper", "Hyperparameters per model (i): linear models and the shipped tree settings.",
#                cs, body, notes_a)
#     tb = table("tab:rh_dm_hyper_search", "Hyperparameters per model (ii): searched in the walk.", cs, body_b, notes)
#     write("rehaul_dm_hyper.tex", ta + "\n" + tb.replace(HEAD, ""))


def pick_counts(sel: pd.DataFrame) -> dict[str, tuple[float, int]]:
    """The modal pick of each hyperparameter over the (block, design) cells of a selection file, with its count."""
    P = []
    for cfg in sel["cfg"]:
        d = {}
        for tok in cfg.split("_"):
            m = re.fullmatch(r"(L|H|lr|we|wl|ll)([\d.e-]+)", tok)
            if m:
                d[m.group(1)] = float(m.group(2))
        P.append(d)
    if "wd" in sel.columns:
        for d, w in zip(P, sel["wd"]):
            d["wd"] = float(w)
    out = {}
    for k in P[0]:
        v, cnt = Counter(d[k] for d in P).most_common(1)[0]
        out[k] = (v, cnt)
    return out


# 2026-10-15 (table legibility): parked -- M2 rebuilt with the guide's terms, numeric cells and panels; seed values moved to the notes; new version below
# def write_tuning(P: dict, TS: dict, NN: dict) -> None:
#     clp = P["clp"]
#     T = TS["T"]
#     op = consts(REPO / "specs" / "causal_tune_trees_optuna.py", ("PATHS",))
#     oj = consts(REPO / "specs" / "causal_tune_trees_optuna_jobs.py", ("TPE_SEED_BASE",))
#     ts = consts(REPO / "specs" / "causal_tune_trees_tuned.py", ("GRID_SEED",))
#     su = consts(REPO / "experiments" / "score_unification.py", ("TREE_TUNE_EVERY", "TREE_EMBARGO", "TREE_TAIL"))
#     pl = consts(REPO / "experiments" / "pool_linear_pretune.py", ("TW_DAYS", "PERIODS_PER_DAY", "PRE_FOLDS",
#                                                                    "PRE_FOLD_SESS", "PRE_EMBARGO_SESS"))
#     lin = consts(REPO / "specs" / "causal_tune_linear.py", ("TUNE_PER", "VAL_TAIL", "EMBARGO"))
#     tp = sorted({v[0] for v in op["PATHS"].values()})
#     q, p = NN["q"], NN["pooled"]
#     nexp = consts(REPO / "src" / "unification.py", ("N_TREE_EXPERTS",))["N_TREE_EXPERTS"]
#     mb = re.search(r"TREE_REFIT_EVERY=(\d+)", (REPO / "specs" / "causal_tune_trees.py").read_text(encoding="utf-8"))
#     check(mb is not None, "the pooled bank's TREE_REFIT_EVERY (specs/causal_tune_trees.py docstring)")
#     bank_every = int(mb.group(1))
#     check(lin["VAL_TAIL"] == clp["VAL_TAIL"] == 125 and lin["EMBARGO"] == clp["EMBARGO"] == 25, "125 / 25 split")
#     check(q["common"]["val_tail"] == clp["VAL_TAIL"] and q["common"]["embargo"] == clp["EMBARGO"], "MinGRU per-bar 125 / 25")
#     check(pl["TW_DAYS"] == clp["TW"], "pooled linear window = 2,000 sessions")
#     W = n_(clp["TW"])
#     ppd = pl["PERIODS_PER_DAY"]
#     al = "lllllll"
#     s0 = q["seeds"]
#     body = [
#         r"\toprule",
#         row(["Family", "Window", "Refit", "Tuning", "Validation tail / embargo", "Criterion", "Seeds"], al),
#         r"\midrule",
#         group("Per bar (16:00 design)", 7),
#         row(["OLS", f"{W} sessions", "every session", "none", "--", "--", "none"], al),
#         row(["Ridge, lasso, elastic net", f"{W} sessions", "every session",
#              rf"pre-tune once ({clp['PRE_FOLDS']} folds $\times$ {clp['PRE_FOLD_LEN']} sessions); margin retune every "
#              rf"{clp['TUNE_PER']} sessions",
#              rf"pre-tune: {clp['PRE_FOLD_LEN']} / {clp['EMBARGO']} sessions; retune: {clp['RETUNE_FOLDS']} $\times$ "
#              rf"{clp['VAL_TAIL']} / {clp['EMBARGO']}",
#              rf"validation MSE; a switch needs HAC $t < -z_{{{clp['MARGIN_ALPHA']}/m}}$", "none"], al),
#         row(["Trees, shipped", f"{W} sessions", f"every {T['REFIT_EVERY']} or 1 sessions", "none (frozen)",
#              "--", "--", f"{T['SEED']}"], al),
#         row(["Trees, random search", f"{W} sessions", f"every {T['REFIT_EVERY']} sessions", f"every {clp['TUNE_PER']} sessions",
#              f"{clp['VAL_TAIL']} / {clp['EMBARGO']} sessions", "validation MSE", f"{T['SEED']}; grid {ts['GRID_SEED']}"], al),
#         row(["Trees, Optuna", f"{W} sessions", "every session",
#              "every " + ", ".join(str(x) for x in tp) + " sessions", f"{clp['VAL_TAIL']} / {clp['EMBARGO']} sessions",
#              r"validation MSE (\texttt{tp25\_q}: QLIKE)", f"{T['SEED']}; TPE {oj['TPE_SEED_BASE']} + row"], al),
#         row(["MinGRU", f"{W} sessions", "every session, cold",
#              rf"every {q['blocks']['block']} sessions ({len(q['blocks']['blocks'])} blocks)",
#              f"{q['common']['val_tail']} / {q['common']['embargo']} sessions",
#              r"validation MSE or QLIKE of the seed mean; epochs frozen per block",
#              f"{len(s0)} ({s0[0]}--{str(s0[-1])[-2:]})"], al),
#         r"\addlinespace",
#         group(rf"Pooled (all {ppd} bars)", 7),
#         row(["Ridge, lasso, elastic net", rf"{n_(pl['TW_DAYS'] * ppd)} bars", "every bar",
#              rf"pre-tune once ({pl['PRE_FOLDS']} folds $\times$ {n_(pl['PRE_FOLD_SESS'] * ppd)} bars); margin retune every "
#              rf"{lin['TUNE_PER']} bars",
#              rf"pre-tune: {n_(pl['PRE_FOLD_SESS'] * ppd)} / {n_(pl['PRE_EMBARGO_SESS'] * ppd)} bars; retune: "
#              rf"{lin['VAL_TAIL']} / {lin['EMBARGO']}", "as per bar", "none"], al),
#         row([rf"Tree bank ({nexp} experts)", f"{n_(OUTLINE['pooled_window'])} bars", "every bar",
#              f"configurations frozen; expert chosen every {su['TREE_TUNE_EVERY']} bars",
#              f"{su['TREE_TAIL']} / {su['TREE_EMBARGO']} bars",
#              r"tail QLIKE: the best expert, or exponential weights", f"{T['SEED']}"], al),
#         row(["MinGRU", f"{n_(p['W'])} bars", f"every {p['refit']} sessions", f"{len(q['blocks']['blocks'])} blocks, as per bar",
#              f"{n_(p['VAL'])} / {p['EMB']} bars", "validation MSE of the seed mean",
#              f"{len(p['seeds'])} ({p['seeds'][0]}--{str(p['seeds'][-1])[-2:]})"], al),
#         r"\bottomrule",
#     ]
#     v0, v1 = P["pre"][3], P["pre"][1]
#     notes = (
#         r"Every split is forward: fit block, embargo, then the validation tail, all inside the trailing window; "
#         r"the test period is never used for a choice. Validation tail / embargo in sessions or bars; the linear "
#         rf"pre-tune folds tile {D_(v0)} to {D_(v1)}. Seeds: none, the fit is deterministic. "
#         r"$m$: the number of challengers (the two grid neighbours). "
#         r"Fit, tuning and selection use squared error on the fit scale ($\sqrt{RV/B}$) unless QLIKE is named. "
#         r"Trees, Optuna: a fresh TPE study at each tuning point; the configuration in force at a row is the best of "
#         r"the latest point. MinGRU: each seed is refitted for its own best epoch, and the forecast is the seed mean. "
#         rf"The tree bank's experts refit at every bar (part of the bank every {bank_every}th bar)."
#     )
#     write("rehaul_dm_tuning.tex", table("tab:rh_dm_tuning", "Tuning protocol per model family.",
#                                         colspec([135, 95, 110, 195, 150, 195, 120]), body, notes))


# ============================================================================ writers (2026-10-15, table legibility)
# NOTATION_GUIDE_2026-10-15.md section 8 (finance-journal tables) and section 6b; series symbols from
# NOTATION_GUIDE_ADDENDA_2026-10-15.md section A; per-table fixes from TABLE_FIXES_FROM_FLOATS_2026-10-15.md.
NA = "--"
TREE_WORDS = {  # library argument -> the parameter in words (and its Methods symbol); the library name stays in code
    "num_leaves": r"Leaves per tree, $J$",
    "learning_rate": r"Learning rate, $\eta$",
    "n_estimators": r"Boosting rounds, $B$",
    "min_child_samples": r"Minimum leaf size, $n_{\min}$",
    "min_child_weight": r"Minimum leaf size, $n_{\min}$",
    "feature_fraction": "Share of predictors per tree",
    "colsample_bytree": "Share of predictors per tree",
    "bagging_fraction": "Share of observations per tree",
    "subsample": "Share of observations per tree",
    "lambda_l1": r"$L_1$ penalty on leaf weights, $\lambda_1^{\mathrm{tree}}$",
    "reg_alpha": r"$L_1$ penalty on leaf weights, $\lambda_1^{\mathrm{tree}}$",
    "lambda_l2": r"$L_2$ penalty on leaf weights, $\lambda_2^{\mathrm{tree}}$",
    "reg_lambda": r"$L_2$ penalty on leaf weights, $\lambda_2^{\mathrm{tree}}$",
    "gamma": r"Penalty per leaf, $\gamma$",
    "max_depth": "Maximum depth",
    "max_features": "Share of predictors per split",
    "max_samples": "Bootstrap sample share",
    "min_samples_leaf": r"Minimum leaf size, $n_{\min}$",
}
RF_WORDS = {"n_estimators": r"Number of trees, $N$"}
FIXED_WORDS = {
    "bagging_freq": "bagging frequency",
    "tree_method": "tree method",
    "bootstrap": "bootstrap sampling",
}
FIXED_VALUE_WORDS = {  # (library, setting, value) -> words, for the hyperparameter table's notes
    ("LightGBM", "bagging_freq", 1): "bagging at every round",
    ("XGBoost", "tree_method", "hist"): "histogram-based split finding",
    ("random forest", "bootstrap", True): "bootstrap samples",
}
MODEL = {"lgbm": "LightGBM", "xgb": "XGBoost", "rf": "Random forest"}
TRANSFORM_G = {  # the variance-stabilizing map g of the Predictors section
    "sqrt": r"$\sqrt{v}$",
    "signed sqrt": r"$\sign(v)\,|v|^{1/2}$",
    "cube root": r"$v^{1/3}$",
    "signed 4th root": r"$\sign(v)\,|v|^{1/4}$",
    "identity": r"$v$",
    "log": r"$\log v$",
    "asinh": r"$\operatorname{asinh}(u/\kappa)$",
}
ADJ_WORDS = {"mean": "mean", "s.d.": "s.d.", "none": "none"}


def tree_word(fam: str, k: str) -> str:
    return RF_WORDS[k] if (fam == "rf" and k in RF_WORDS) else TREE_WORDS[k]


def num(v) -> str:
    """Integers as is; below 1e-3 in scientific form (2 digits); [0.1, 1) with 2 decimals (as the macros);
    otherwise 3 significant digits."""
    if isinstance(v, str):
        return v
    v = float(v)
    if v == int(v) and abs(v) < 1e6:
        return str(int(v))
    if abs(v) < 1e-3:
        return g_(float(f"{v:.2g}"))
    if 0.1 <= abs(v) < 1:
        return f"{v:.2f}"
    return f"{v:.3g}"


def rng(lo, hi, log: bool = False) -> str:
    s = f"[{gm(lo)}, {gm(hi)}]"
    return (f"${s}$" if ("^" in s or "\\" in s) else s) + (" (log)" if log else "")


def grid10(gr) -> str:
    ks = [int(round(math.log10(a))) for a in gr]
    check(ks == list(range(ks[0], ks[-1] + 1)), "a one-per-decade grid")
    return (
        rf"$\{{10^{{{ks[0]}}}, 10^{{{ks[1]}}}, \dots, 10^{{{ks[-1]}}}\}}$ ({len(gr)})"
    )


def ftable(
    label: str,
    caption: str,
    spec: str,
    body: list[str],
    notes: list[str],
    sep: str = "4pt",
) -> str:
    """A finance-style float: \\small body, short caption, \\textit{Notes:} of a few sentences below."""
    return table(label, caption, spec, body, " ".join(notes), size=r"\small", sep=sep)


def write_sources(S: dict) -> None:
    """D1 (tab:rh_dm_sources): one row per source series; words, not file names."""
    rv0, rv1, nrv = S["rv"]
    nx, x0, x1, nxs = S["xs"]
    c0, c1, nq, ne = S["chain"]
    V = r"\textit{[vendor]}"
    spec = "@{}" + P_(0.20) + P_(0.32) + "llr" + P_(0.11) + "@{}"
    body = [
        r"\toprule",
        cells(["Source", "Series", "First", "Last", "Obs.", "Use"]),
        r"\midrule",
        cells(
            [
                rf"E-mini S\&P 500 futures (ES), {V}",
                rf"$\RV_{{t,i}}$ and {len(S['core_cols']) - 1} other statistics of the one-minute log returns "
                r"$r_{t,i,k}$ (moments, bipower variation, autocovariance, volume, return count)",
                D_(rv0),
                D_(rv1),
                n_(nrv),
                "Target, predictors",
            ]
        ),
        cells(
            [
                rf"Index constituents, equal- and value-weighted, {V}",
                rf"{nx} statistics per weighting: {S['xs_moments']} realized moments, turnover (total, buyer- and "
                r"seller-initiated), effective and quoted spreads",
                D_(x0),
                D_(x1),
                n_(nxs),
                "Predictors",
            ]
        ),
        cells(
            [
                rf"StockTwits, {V}",
                "Attention (message volume)",
                D_(S["att"][0]),
                D_(S["att"][1]),
                n_(S["att"][2]),
                "Predictors",
            ]
        ),
        cells(
            [
                "",
                "Sentiment score; sentiment-tagged messages",
                D_(S["sent"][0]),
                D_(S["sent"][1]),
                n_(S["sent"][2]),
                "Predictors",
            ]
        ),
        cells(
            [
                rf"Cboe, {V}",
                "VIX",
                D_(S["vix"][0]),
                D_(S["vix"][1]),
                n_(S["vix"][2]),
                "Predictors",
            ]
        ),
        cells(
            [
                "",
                "VIX3M",
                D_(S["vix3m"][0]),
                D_(S["vix3m"][1]),
                n_(S["vix3m"][2]),
                "Predictors",
            ]
        ),
        cells(
            [
                "",
                "VVIX",
                D_(S["vvix"][0]),
                D_(S["vvix"][1]),
                n_(S["vvix"][2]),
                "Predictors",
            ]
        ),
        cells(
            [
                "",
                rf"Net volatility demand, {S['n_vd']} series: SPX or all index options; opening and closing, "
                r"or opening trades only",
                D_(S["vd"][0]),
                D_(S["vd"][1]),
                n_(S["vd"][2]),
                "Predictors",
            ]
        ),
        cells(
            [
                rf"FOMC calendar, {V}",
                "Scheduled announcement times, 14:30 ET",
                D_(S["fomc"][0]),
                D_(S["fomc"][1]),
                n_(S["fomc"][2]),
                "Predictors",
            ]
        ),
        cells(
            [
                rf"SPXW options on expiration days, {V}",
                rf"Bid, ask, midquote, underlying level, implied volatility, delta, time to expiration, early-close "
                rf"flag; {n_(ne)} expirations",
                D_(c0),
                D_(c1),
                n_(nq),
                "Straddle positions",
            ]
        ),
        r"\bottomrule",
    ]
    notes = [
        r"Each series is recorded at the end of its 30-minute interval (ET) and merged on that interval-end time "
        r"stamp; the SPXW quotes are time-stamped in UTC and loaded separately.",
        r"First and Last: the first and last interval with an observation; Obs.: the number of such intervals before "
        r"the intraday grid is formed (constituents: per weighting; FOMC: announcements; SPXW: quotes).",
        r"The VIX indices, the volatility-demand series and the FOMC calendar end before the futures sample "
        rf"({D_(rv1)}); after their last observation the availability indicator $a_{{t,i}}$ is zero and the value "
        r"is imputed (Sections~\ref{sec:rh_data} and~\ref{sec:rh_predictors}).",
        rf"The data files also contain {len(S['release_flags'])} other release indicators, an SPXW-implied variance "
        r"and other VIX series, none of them used as a predictor.",
        rf"{V}: the vendor is not yet named in any data file or document (the option chain is attributed to "
        r"OptionMetrics only in a code comment).",
    ]
    write(
        "rehaul_dm_sources.tex",
        ftable("tab:rh_dm_sources", "Data sources.", spec, body, notes),
    )


def write_samples(P: dict) -> None:
    """D2 (tab:rh_dm_samples): panels intraday / closing interval / out-of-sample."""
    g0, g1, ng = P["grid"]
    p0, p1, np_ = P["pooled"]
    f0, f1, nf = P["pooled_fc"]
    b0, b1, nb = P["perbar"]
    c0, c1, nc = P["fc"]
    r0, r1, nr, v0, nv = P["pre"]
    s0, s1, ns, s_fc = P["dev"]
    clp = P["clp"]
    days = P["days"]
    W = OUTLINE["pooled_window"]
    lat, lat_full = P["later"]
    ppd = TG.PERIODS_PER_DAY
    later = SRC["exp"].index[SRC["exp"].index > pd.Timestamp(clp["LAST_TRADE_DAY"])]
    spec = "@{}" + P_(0.50) + "llrr@{}"
    body = [
        r"\toprule",
        cells(["Sample", "Start", "End", "Obs.", "Window $W$"]),
        r"\midrule",
        # 2026-10-15 (table legibility): panel order as core guide section 8 (closing-interval first); the grid row,
        # which underlies both samples, above the panels (NOTATION_GUIDE_ADDENDA C.3)
        cells(
            [
                r"Intraday grid: 30-minute intervals with an observed $\RV_{t,i}$, outside the weekend break",
                D_(g0),
                D_(g1),
                n_(ng),
                NA,
            ]
        ),
        r"\addlinespace",
        panel_row("Panel A: Closing-interval models (trading days)", 5),
        cells(
            [
                "Closing-interval observations, after the burn-in",
                D_(b0),
                D_(b1),
                n_(nb),
                n_(clp["TW"]),
            ]
        ),
        cells(["Rolling-window forecasts", D_(c0), D_(c1), n_(nc), NA]),
        cells(
            [
                "Hyperparameter-selection pre-sample (linear models)",
                D_(r0),
                D_(r1),
                n_(nr),
                n_(clp["TW"]),
            ]
        ),
        cells(
            [
                rf"\quad Validation samples ({clp['PRE_FOLDS']} $\times$ {clp['PRE_FOLD_LEN']} days)",
                D_(v0),
                D_(r1),
                n_(nv),
                n_(clp["TW"]),
            ]
        ),
        r"\addlinespace",
        panel_row("Panel B: Intraday models (30-minute intervals)", 5),
        cells(
            [
                rf"Estimation sample, after the {n_(P['burn'])}-interval burn-in",
                D_(p0),
                D_(p1),
                n_(np_),
                n_(W),
            ]
        ),
        cells(["Rolling-window forecasts", D_(f0), D_(f1), n_(nf), NA]),
        cells(
            [
                rf"Tree hyperparameter pre-sample (before {s1.year + 1})",
                D_(s0),
                D_(s1),
                n_(ns),
                n_(P["tm"]["WINDOW"]),
            ]
        ),
        r"\addlinespace",
        # 2026-10-15 (user: "Out of sample period doesn't make sense for forecasts because we're performing rolling
        # backtests ..."): Panel C names the evaluation span of the accuracy statistics (every rolling forecast of
        # tab:rh_oos exists; experiments/rolling_eval_2026_10_15.py common_span()) and the option-data subsample of the
        # straddle economics.  Parked -- was:
        # panel_row("Panel C: Out-of-sample period (trading days)", 5),
        # cells(["All trading days (closing-interval forecasts)", D_(clp["FIRST_TRADE_DAY"]), D_(c1),
        #        n_(P["test16"]), NA]),
        # cells([rf"Evaluation days ($T_{{\mathrm{{oos}}}}$): SPXW expiration days ({n_(P['exp_to_end'])}) less "
        #        rf"early-close days ({P['ec_to_end']})", D_(days.min()), D_(days.max()), n_(len(days)), NA]),
        panel_row("Panel C: Evaluation samples (trading days)", 5),
        cells(
            [
                rf"Evaluation span ($T_{{\mathrm{{oos}}}}$): trading days on which every forecast exists, less "
                rf"early-close days ({P['span_ec']})",
                D_(P["span"].min()),
                D_(P["span"].max()),
                n_(len(P["span"])),
                NA,
            ]
        ),
        cells(
            [
                rf"\quad Option-data subsample ($T_{{\mathrm{{opt}}}}$): SPXW expiration days "
                rf"({n_(P['exp_to_end'])}) less early-close days ({P['ec_to_end']})",
                D_(days.min()),
                D_(days.max()),
                n_(len(days)),
                NA,
            ]
        ),
        cells(
            [
                rf"Later SPXW expiration days, held out ({n_(lat_full)} full trading days)",
                D_(later.min()),
                D_(SRC["exp"].index.max()),
                n_(lat),
                NA,
            ]
        ),
        r"\bottomrule",
    ]
    notes = [
        r"Dates are interval-end dates (ET); Obs.\ and the rolling estimation window $W$ are in the units of each "
        r"panel.",
        rf"The burn-in, {n_(P['burn'])} intervals, is the longest HAR horizon and is applied to each series as built "
        r"(the closing-interval series, the intraday grid and, for the linear pre-sample, the series of the two "
        rf"intervals of the last hour), so the closing-interval sample starts in {b0.year} and the pre-sample in "
        rf"{r0.year}.",
        # 2026-10-15 (finish): parked -- the computed 96{,}000 and 2{,}000 -> \rhdmPooledLinWindow (write_numbers) and
        # \cmPtFitWin (checked below), and the pointer -> Appendix B.1 (sec:rh_exp_samples), where the gap and the
        # validation sample are stated. Was:
        # rf"The pooled linear models use $W = {n_(clp['TW'] * ppd)}$ intervals ({n_(clp['TW'])} trading days); the "
        # rf"tree pre-sample is re-estimated every {P['tm']['SCREEN_REFIT']} intervals.",
        # rf"Each validation sample follows a {clp['EMBARGO']}-day gap and a {n_(clp['TW'])}-day estimation sample "
        # r"(Section~\ref{sec:rh_methods_tuning}).",
        # 2026-10-18 (user, second-list 17): parked -- the 250 is the screening stride of the one-off tree search, not
        # a re-estimation of a pre-sample (hand edit mirrored in generated/rehaul_dm_samples.tex). Was:
        # rf"tree pre-sample is re-estimated every {P['tm']['SCREEN_REFIT']} intervals.",
        r"The pooled linear models use $W = \rhdmPooledLinWindow$ intervals (\cmPtFitWin{} trading days); the "
        rf"tree hyperparameters are chosen once, by an Optuna search on the {P['dev'][0].year}--{P['dev'][1].year} "
        rf"development sample that fits each candidate on the previous {n_(P['tm']['WINDOW'])} intervals and scores "
        rf"it on the next {P['tm']['SCREEN_REFIT']}, stepping forward {P['tm']['SCREEN_REFIT']} intervals at a time; "
        r"the reported forecasts refit every interval.",
        rf"Each validation sample follows a {clp['EMBARGO']}-day gap and a \cmPtFitWin-day estimation sample "
        r"(Section~\ref{sec:rh_exp_samples}).",
        # "Full trading days: expiration days that are not early-close days.",  # 2026-10-15: + the evaluation span
        r"Full trading days: expiration days that are not early-close days. Panel C: the evaluation span holds every "
        r"trading day on which all rolling forecasts exist (Section~\ref{sec:rh_results_eval}); the straddle economics "
        r"use its option-data subsample (Section~\ref{sec:rh_econ}).",
    ]
    check(
        macros()["cmPtFitWin"] == n_(clp["TW"]),
        r"\cmPtFitWin = the closing-interval window (samples-table notes)",
    )
    check(
        pooled_lin_window(clp) == clp["TW"] * ppd,
        r"\rhdmPooledLinWindow = 2,000 trading days x 48 intervals (samples-table notes)",
    )
    write(
        "rehaul_dm_samples.tex",
        ftable(
            "tab:rh_dm_samples", "Estimation and evaluation samples.", spec, body, notes
        ),
    )


def write_design(D: dict) -> None:
    """F1 (tab:rh_dm_design): columns of the HAR benchmark and the full predictor set, by predictor group."""
    import src.backtest.executor as EX

    A, Q, k, kq, nser = D["A"], D["Q"], D["k"], D["kq"], D["nser"]

    # 2026-10-15 (table legibility): parked -- bold totals (bold marks 5% significance in the paper's tables)
    # def B(s) -> str:
    #     return rf"\textbf{{{s}}}"
    def B(s) -> str:
        return str(s)

    body = [
        r"\toprule",
        r" & & \multicolumn{4}{c}{Closing-interval model} & "
        + mc(1, stack("Pooled", "intraday"))
        + r" \\",
        r"\cmidrule(lr){3-6}\cmidrule(l){7-7}",
        cells(
            [
                "Predictor group",
                "Series",
                stack("Series", "averages", r"$u^{(k)}$"),
                stack("Missing-data", "indicators", r"$\bar a^{(k)}$"),
                stack("Activity", "indicators", r"$\bar\chi^{(k)}$"),
                "Total",
                "Total",
            ]
        ),
        r"\midrule",
        cells(
            [
                r"HAR components $\bar y^{(k)}$",
                "1",
                str(len(A["har"])),
                NA,
                NA,
                str(len(A["har"])),
                str(len(Q["har"])),
            ]
        ),
        cells(
            [
                r"HAR components $\times$ open, close",
                NA,
                NA,
                NA,
                NA,
                NA,
                str(len(Q["edge"])),
            ]
        ),
        cells(
            [
                # 2026-10-15 (consolidation): \bm d -> \bm D (d_t is the DM loss differential)
                r"Calendar dummies $\bm D$",
                NA,
                NA,
                NA,
                NA,
                str(len(A["calendar"])),
                str(len(Q["calendar"])),
            ]
        ),
        r"\addlinespace",
        cells(
            [
                B("HAR benchmark"),
                NA,
                NA,
                NA,
                NA,
                B(len(A["har"]) + len(A["calendar"])),
                B(len(Q["har"]) + len(Q["edge"]) + len(Q["calendar"])),
            ]
        ),
        r"\midrule",
    ]
    tv = ta = tc = tq = 0
    for b, lab in BUCKET_LABEL.items():
        v, a, c = A["value"][b], A["avail"][b], A["active"][b]
        q = Q["value"][b] + Q["avail"][b] + Q["active"][b]
        tv, ta, tc, tq = tv + v, ta + a, tc + c, tq + q
        body.append(
            cells(
                [
                    lab,
                    str(nser[b]),
                    str(v),
                    str(a),
                    str(c) if c else NA,
                    str(v + a + c),
                    n_(q) if q else NA,
                ]
            )
        )
    body += [
        r"\addlinespace",
        cells(
            [
                "Additional predictors",
                str(sum(nser.values())),
                str(tv),
                str(ta),
                str(tc),
                str(tv + ta + tc),
                n_(tq),
            ]
        ),
        r"\midrule",
        cells(
            [
                B("Full predictor set"),
                NA,
                NA,
                NA,
                NA,
                B(tv + ta + tc + len(A["har"]) + len(A["calendar"])),
                B(n_(tq + len(Q["har"]) + len(Q["edge"]) + len(Q["calendar"]))),
            ]
        ),
        r"\bottomrule",
    ]
    pr = sorted(set(LD.ALL_FEATURES) - Q["series"]["value"])
    check(
        set(pr) == set(LD.FOMC_FEATURES),
        "the pooled panel lacks exactly the FOMC series",
    )
    backbone_q = len(Q["har"]) + len(Q["edge"]) + len(Q["calendar"])
    with_fomc = backbone_q + 2 * kq * sum(nser.values()) + kq * len(D["act"])
    words = {12: "twelve", 6: "six", 4: "four", 16: "16"}
    cc = D["cal_const"]
    notes = [
        r"$u^{(k)}$: series $u$ averaged over the previous $k$ intervals; $\bar a^{(k)}(u)$ and $\bar\chi^{(k)}(u)$: "
        r"the same averages of its availability (missing-data) and activity indicators (Section~\ref{sec:rh_predictors} "
        r"and Table~\ref{tab:rh_dm_dictionary}).",
        # 2026-10-19 (user): parked -- base 5 across the paper. Was:
        # rf"Horizons: $k \in \{{{', '.join(str(r) for r in resolve_har_lags())}\}}$ in the closing-interval model and "
        # rf"$k \in \{{1, 2, 4, \dots, {2 ** (kq - 1)}\}}$ in the intraday model; activity indicators exist "
        rf"Horizons: $k \in \{{{', '.join(str(r) for r in sorted(Q['rungs']))}\}}$ in both models; activity indicators exist "
        rf"for the {len(D['act'])} series whose mode accounts for at least {100 * EX.ZERO_INFLATED_FRAC:.0f}\% of "
        r"intervals.",
        rf"In the closing-interval model, the {words.get(2 * k, str(2 * k))} interactions of the HAR components "
        r"with the opening- and closing-interval dummies are zero or copies of $\bar y^{(k)}$ and are omitted, and "
        rf"{words.get(len(cc), str(len(cc)))} of the {len(A['calendar'])} calendar dummies (hour, overnight, open, "
        r"close) are constant and are dropped at every estimation.",
        # 2026-10-19 (user): parked -- the base-5 pooled panel leaves the FOMC group out by choice (it no longer
        # predates it). Was: rf"The intraday panel predates the FOMC group ({len(pr)} series) and has no FOMC predictors; with "
        rf"The intraday panel leaves out the FOMC group ({len(pr)} series) and has no FOMC predictors; with "
        rf"them it would have {n_(with_fomc)} columns.",
        "At every estimation, predictors constant within the rolling window or exact copies of another are dropped.",
    ]
    spec = "@{}lrrrrrr@{}"
    write(
        "rehaul_dm_design.tex",
        ftable(
            "tab:rh_dm_design", "Composition of the predictor sets.", spec, body, notes
        ),
    )


def write_dictionary(rows: list[dict], D: dict) -> None:
    """F2 (tab:rh_dm_dictionary): predictor definitions, a landscape longtable for Appendix A.

    No code names in the table: each row's data field (the series' name in the source data files) is a % comment
    after the row, for replication."""
    import inspect

    order = list(BUCKET_LABEL)
    rows = sorted(
        rows,
        key=lambda r: (order.index(r["bucket"]), LD.ALL_FEATURES.index(r["series"])),
    )
    ww = inspect.signature(TG.rolling_winsorize).parameters["window"].default
    check(
        f"else {ww}" in inspect.getsource(TG.robust_transform),
        "exog winsor window = robust_transform's default",
    )

    def source(r: dict) -> str:
        f = r["source"].replace(" (derived)", "")
        check(f in SOURCE_WORDS, f"{r['series']}: source file {f} has a name in words")
        return SOURCE_WORDS[f]

    # 2026-10-15 (table legibility): parked -- the typewriter "Data field" column (core guide section 8 and addenda
    # A.1: no code names in a table); each row's data field is written as a % comment after it instead.
    # ncol = 7
    # spec = pcols([105, 320, 110, 90, 100, 115, 160])
    ncol = 6
    # all p{} columns summing to the (landscape) line width: a p{\linewidth} notes row over natural-width l columns
    # came out 88pt too wide
    # 2026-10-19 (user): parked -- Variable 110 -> 120, Description 440 -> 430: the raw names (\\ser, raw for now) need
    # the width ("sellturnover_" ran 0.6pt over). Was: spec = pcols([110, 440, 120, 100, 110, 120])
    spec = pcols([120, 430, 120, 100, 110, 120])
    head = cells(
        [
            "Variable",
            "Description",
            "Source",
            # 2026-10-15 (user): "periodicity adjustment" -> "diurnal adjustment" (the symbol s^2_{t,i} stays)
            stack("Diurnal", "adjustment", align="l"),
            stack("Transform", "$g(v)$", align="l"),
            stack("Predictors,", r"$k \in \mathcal K$", align="l"),
        ]
    )
    signed_spread = [
        r
        for r in rows
        if r["transform"] == "identity" and r["neg"] and "spread" in r["series"]
    ]
    check(
        len(signed_spread) == 1 and signed_spread[0]["series"] == "spread_vwstock",
        "the one spread series signed by a stray negative print is the VW quoted spread",
    )
    sp = signed_spread[0]
    twin = next(
        r for r in rows if r["series"] == sp["series"].replace("vwstock", "ewstock")
    )
    check(
        twin["transform"] == "log" and twin["diurnal"] == "mean",
        "its EW twin takes log and the mean",
    )
    kq = D["kq"]
    notes = [
        r"\textit{Notes:} Each series $u$ enters as $u^{(k)}$, its adjusted, transformed and winsorized value averaged "
        r"over the previous $k$ intervals, and as $\bar a^{(k)}(u)$, the share of those intervals on which it was "
        r"observed; series with an activity indicator also enter as $\bar\chi^{(k)}(u)$, the share off their mode "
        r"(the mode and its share of intervals are in the description).",
        # 2026-10-19 (user): parked -- base 5 across the paper. Was:
        # rf"$\mathcal K = \{{{', '.join(str(r) for r in resolve_har_lags())}\}}$ in the closing-interval model and "
        # rf"$\{{1, 2, 4, \dots, {2 ** (kq - 1)}\}}$ in the intraday model, which is every group here except "
        # r"FOMC (Section~\ref{sec:rh_data}).",
        rf"$\mathcal K = \{{{', '.join(str(r) for r in resolve_har_lags())}\}}$ in the closing-interval model and "
        r"in the intraday model, which is every group here except FOMC (Section~\ref{sec:rh_data}).",
        r"Diurnal adjustment: division by the trailing same-interval mean (mean) or standard deviation (s.d.); "
        r"$g$: the variance-stabilizing map; the volatility-demand series take $\operatorname{asinh}(u_{t,i} / "
        r"\kappa_{t,i})$ in place of both (Section~\ref{sec:rh_predictors}).",
        rf"A series is signed if it has any negative value: ${series_tex(sp['series'])}$ has "
        + (
            "a single negative value"
            if sp["n_neg"] == 1
            else f"{sp['n_neg']} negative values"
        )
        + (" (" if sp["n_neg"] == 1 else " (first ")
        + rf"{sp['first_neg']} ET) and so takes $g(v) = v$ and the s.d.\ adjustment, where "
        rf"${series_tex(twin['series'])}$ takes $\log v$ and the mean.",
        r"Unnormalized and unscaled: without the factors $\sqrt{N_{t,i}} / \RV_{t,i}^{3/2}$ (skewness), "
        r"$N_{t,i}/3$ (quarticity) and $\pi/2$ (bipower variation).",
    ]
    out = [
        HEAD + SER_PROVIDE,
        r"\begin{landscape}",
        r"\begingroup",
        r"\small",
        r"\setlength{\tabcolsep}{3.5pt}",
        r"\setlength{\LTcapwidth}{\linewidth}",
        rf"\begin{{longtable}}{{{spec}}}",
        r"\caption{Predictor definitions.}\label{tab:rh_dm_dictionary}\\",
        r"\toprule",
        head,
        r"\midrule",
        r"\endfirsthead",
        rf"\multicolumn{{{ncol}}}{{@{{}}l}}{{\textit{{Table \ref{{tab:rh_dm_dictionary}} (continued)}}}}\\",
        r"\toprule",
        head,
        r"\midrule",
        r"\endhead",
        r"\midrule",
        rf"\multicolumn{{{ncol}}}{{r@{{}}}}{{\textit{{continued on the next page}}}}\\",
        r"\endfoot",
        r"\bottomrule",
        rf"\multicolumn{{{ncol}}}{{@{{}}p{{\linewidth}}@{{}}}}{{\footnotesize\raggedright\rule{{0pt}}{{2.5ex}}"
        + " ".join(notes)
        + r"}\\",
        r"\endlastfoot",
    ]
    cur = None
    for r in rows:
        if r["bucket"] != cur:
            if cur is not None:
                out.append(r"\addlinespace")
            cur = r["bucket"]
            nb = sum(1 for x in rows if x["bucket"] == cur)
            # 2026-10-17 (user, E5): group clause from data_explainer_v11. Parked -- fomc-only extra.
            # extra = "; closing-interval model only" if cur == "fomc" else ""
            extra = {
                "moments": "; front-month E-mini, one-minute log-price changes",
                "market_ew": "; simple averages of S\\&P 500 constituents",
                "market_vw": "; previous day's market capitalization",
                "vol_demand": "; net vega bought by non-market-makers",
                "fomc": "; closing-interval model only, scheduled calendar",
            }.get(cur, "")
            out.append(panel_row(f"{BUCKET_LABEL[cur]} ({nb}{extra})", ncol))
        desc = desc_of(r["series"])
        if r["active"]:
            desc += rf"; mode {r['mode']:g} on {100 * r['share']:.0f}\% of intervals"
        pred = r"$u^{(k)}$, $\bar a^{(k)}$" + (
            r", $\bar\chi^{(k)}$" if r["active"] else ""
        )
        out.append(
            cells(
                [
                    # 2026-10-19 (user): raw names for now; \ser in text mode, so a long raw name breaks at its
                    # underscores. Was: f"${series_tex(r['series'])}$"
                    rf"\ser{{{r['series']}}}{{${series_sym(r['series'])}$}}",
                    desc,
                    source(r),
                    ADJ_WORDS[r["diurnal"]],
                    TRANSFORM_G[r["transform"]],
                    pred,
                ]
            )
            + f"  % data field: {r['series']}"
        )
    out += [r"\end{longtable}", r"\endgroup", r"\end{landscape}", ""]
    write("rehaul_dm_dictionary.tex", "\n".join(out))


def write_hyper(P: dict, TS: dict, LP: dict, NN: dict) -> None:
    """M1, two floats: (i) tab:rh_dm_hyper, the linear penalties and the fixed tree hyperparameters; (ii)
    tab:rh_dm_hyper_search, what is re-selected in the rolling-window exercise (trees, MinGRU grids)."""
    T, X, Xs, R, rf = TS["T"], TS["X"], TS["Xs"], TS["R"], TS["rf"]
    lg = T["LGBM_PARAMS"]
    L = LP["L"]
    check(
        sum(int(v[1].split(" of ")[0]) for v in L.values())
        == int(LP["M"]["cmPtFiredTotal"])
        and sum(int(v[1].split(" of ")[1]) for v in L.values())
        == int(LP["M"]["cmPtFiredOf"]),
        r"retunes fired per arm sum to \cmPtFiredTotal of \cmPtFiredOf",
    )
    sc = TS["scale"]
    oj = consts(
        REPO / "specs" / "causal_tune_trees_optuna_jobs.py",
        (
            "SPACES",
            "TPE_SEED_BASE",
            "ROUNDS_CAP_MULT",
            "XGB_SUBSAMPLE_LO",
            "XGB_REG_LAMBDA_LO",
        ),
    )
    op = consts(
        REPO / "specs" / "causal_tune_trees_optuna.py", ("N_TRIALS_DEFAULT", "BUDGETS")
    )
    ts = consts(
        REPO / "specs" / "causal_tune_trees_tuned.py",
        ("N_CANDIDATES", "GRID_SEED", "RF_MAX_FEATURES", "RF_MAX_DEPTH", "RF_MIN_LEAF"),
    )
    xlo = min(oj["XGB_REG_LAMBDA_LO"], 10.0 ** math.floor(math.log10(X["reg_lambda"])))
    check(
        re.search(
            r"min\(OJ\.XGB_REG_LAMBDA_LO, 10\.0 \*\* math\.floor\(math\.log10\(shipped\)\)\)",
            (REPO / "specs" / "causal_tune_trees_optuna_xgbtop.py").read_text(
                encoding="utf-8"
            ),
        )
        is not None
        and xlo == 1e-5,
        "xgbtop reg_lambda floor 1e-5",
    )
    tdoc = (REPO / "specs" / "causal_tune_trees_tuned.py").read_text(encoding="utf-8")
    msz = re.search(
        r"The full products are (\d+) \(LightGBM\), (\d+)\s*\n?\s*\(XGBoost\) and (\d+) \(RF\)",
        tdoc,
    )
    check(
        msz is not None
        and int(msz.group(3))
        == len(ts["RF_MAX_FEATURES"])
        * len(ts["RF_MAX_DEPTH"])
        * len(ts["RF_MIN_LEAF"]),
        "random-search grid sizes (RF = 4 x 4 x 7)",
    )
    assert msz is not None
    box = {
        fam: {n: (lo, hi, lg_) for n, lo, hi, lg_, _ in P["tm"]["BOXES"][fam]}
        for fam in ("lgbm", "xgb")
    }
    box["rf"] = {b[0]: (b[1], b[2], b[3]) for b in rf["provenance"]["box"]}
    nblk = len(NN["q"]["sel"])
    check(
        nblk == len(NN["base"]["sel"]) == len(NN["pooled"]["sel"]) == 5 * 2,
        "10 (block, design) picks per network walk",
    )
    dev0, dev1 = P["dev"][0].year, P["dev"][1].year

    # ------------------------------------------------------------------ (i) linear penalties + fixed tree settings
    spec = "@{}l" + P_(0.31) + "lrr@{}"
    body = [
        r"\toprule",
        r" & & & \multicolumn{2}{c}{Value} \\",
        r"\cmidrule(l){4-5}",
        cells(
            [
                "Model",
                "Hyperparameter",
                "Search space",
                stack("HAR", "benchmark", align="r"),
                stack("Full", "predictor set", align="r"),
            ]
        ),
        r"\midrule",
        panel_row("Panel A: Penalized linear models, selected on the pre-sample", 5),
        cells(["OLS", NA, NA, NA, NA]),
    ]
    for est, name in (("ridge", "Ridge"), ("lasso", "Lasso"), ("enet", "Elastic net")):
        gk = {"ridge": "ridge", "lasso": "reclasso", "enet": "reclasticnet"}[est]
        body.append(
            cells(
                [
                    name,
                    r"Penalty, $\lambda$",
                    grid10(LP["grids"][gk]),
                    g_(L[(est, "baseline")][0], True),
                    g_(L[(est, "all_features")][0], True),
                ]
            )
        )
    body += [
        cells(["", r"Mixing weight, $\alpha$", "fixed", "0.5", "0.5"]),
        cells(
            [
                "All three",
                r"Penalty weight on the HAR benchmark, $\underline\omega$",
                "fixed",
                "1",
                g_(P["clp"]["R_BB"]),
            ]
        ),
        r"\addlinespace",
        panel_row(
            rf"Panel B: Tree ensembles, fixed hyperparameters (selected on the {dev0}--{dev1} pre-sample)",
            5,
        ),
    ]
    shipped = {
        "lgbm": (
            lg,
            (
                "num_leaves",
                "learning_rate",
                "n_estimators",
                "min_child_samples",
                "feature_fraction",
                "bagging_fraction",
                "lambda_l1",
                "lambda_l2",
            ),
            {"min_child_samples": max(1, round(lg["min_child_samples"] * sc))},
        ),
        "xgb": (
            X,
            (
                "max_depth",
                "learning_rate",
                "n_estimators",
                "min_child_weight",
                "subsample",
                "colsample_bytree",
                "reg_alpha",
                "reg_lambda",
                "gamma",
            ),
            {"min_child_weight": Xs["min_child_weight"]},
        ),
        "rf": (
            rf["params"],
            (
                "n_estimators",
                "max_depth",
                "max_features",
                "max_samples",
                "min_samples_leaf",
            ),
            {"min_samples_leaf": R["min_samples_leaf"]},
        ),
    }
    for fam, (prm, keys, scaled) in shipped.items():
        if fam != "lgbm":
            body.append(r"\addlinespace")
        for i, k in enumerate(keys):
            v = num(prm[k]) + (f" ({num(scaled[k])})" if k in scaled else "")
            body.append(
                cells(
                    [
                        MODEL[fam] if i == 0 else "",
                        tree_word(fam, k),
                        rng(*box[fam][k]),
                        mc(2, v, "c"),
                    ]
                )
            )
    body.append(r"\bottomrule")
    fixed = {
        "LightGBM": {k: v for k, v in lg.items() if k not in shipped["lgbm"][1]},
        "XGBoost": {k: v for k, v in X.items() if k not in shipped["xgb"][1]},
        "random forest": {
            k: v for k, v in rf["params"].items() if k not in shipped["rf"][1]
        },
    }
    # 2026-10-15 (table legibility): parked -- library names and a typewriter value in the notes; the fixed
    # settings are now written in words (FIXED_VALUE_WORDS), each checked against the shipped dicts.
    # check(all(k in FIXED_WORDS for d in fixed.values() for k in d), "every fixed library setting has a name in words")
    # def fval(v) -> str: ...
    # fx = "; ".join(f"{nm} " + ", ".join(f"{FIXED_WORDS[k]} {fval(v)}" for k, v in d.items()) ...)
    check(
        {(nm, k, v) for nm, d in fixed.items() for k, v in d.items()}
        == set(FIXED_VALUE_WORDS),
        "the fixed library settings are exactly bagging every round, histogram splits, bootstrap samples",
    )
    impl = {
        nm: FIXED_VALUE_WORDS[(nm, k, v)]
        for nm, d in fixed.items()
        for k, v in d.items()
    }
    fired_n = {
        e: (
            int(L[(e, "baseline")][1].split(" of ")[0]),
            int(L[(e, "all_features")][1].split(" of ")[0]),
        )
        for e in ("ridge", "lasso", "enet")
    }
    fired_of = sum(int(v[1].split(" of ")[1]) for v in L.values())
    fired = ", ".join(
        f"{lab} {fired_n[e][0]} and {fired_n[e][1]}"
        for e, lab in (("ridge", "ridge"), ("lasso", "lasso"), ("enet", "elastic net"))
    )
    notes_a = [
        r"Panel A: Section~\ref{sec:rh_methods_models} states the scale of $\lambda$. One grid value "
        r"per decade (grid size in parentheses), selected by mean validation MSE over the "
        rf"{P['clp']['PRE_FOLDS']} folds of the pre-sample (Table~\ref{{tab:rh_dm_samples}}).",
        rf"At each of the {L[('ridge', 'baseline')][1].split(' of ')[1]} re-selection points a grid neighbor may "
        rf"replace it (Section~\ref{{sec:rh_methods_tuning}}); this happened {sum(map(sum, fired_n.values()))} times "
        rf"in {fired_of} re-selections ({fired}, for the HAR benchmark and the full predictor set).",
        # 2026-10-15 (finish): parked -- the computed "trial 255 of 400", 24{,}000 and 2{,}000 -> \rhdmRfTrial of
        # \cmTreeProvRfTrials, \rhdmPooledWindow and \cmPtFitWin (same values, checked below). Was:
        # r"Panel B: the best trial of an Optuna search on the pooled intraday pre-sample "
        # rf"({n_(P['tm']['WINDOW'])}-interval window; random forest: trial {rf['trial']} of "
        # rf"{rf['trials']['complete_considered']}), ranked by MSE and then held fixed for both predictor sets; (log): "
        # r"searched on a logarithmic scale.",
        # rf"$n_{{\min}}$ was selected on {n_(T['POOLED_WINDOW_ROWS'])} intervals and is rescaled to the "
        # rf"{n_(OUTLINE['perbar_window'])}-day window (rescaled value in parentheses).",
        r"Panel B: the best trial of an Optuna search on the intraday pre-sample "
        r"(\rhdmPooledWindow-interval window; random forest: trial \rhdmRfTrial{} of "
        r"\cmTreeProvRfTrials), ranked by MSE and then held fixed for both predictor sets; (log): "
        r"searched on a logarithmic scale.",
        r"$n_{\min}$ was selected on \rhdmPooledWindow{} intervals and is rescaled to the "
        r"\cmPtFitWin-day window (rescaled value in parentheses).",
        rf"Implementations: LightGBM \citep{{Ke2017}} with {impl['LightGBM']}, XGBoost \citep{{Chen2016}} with "
        rf"{impl['XGBoost']} and the scikit-learn random forest with {impl['random forest']}; random seeds are fixed.",
    ]
    # the macros of the notes above carry the values this run computes
    check(
        P["tm"]["WINDOW"] == T["POOLED_WINDOW_ROWS"] == P["pooled_window"],
        r"Optuna pre-sample window = leaf-rescaling denominator = \rhdmPooledWindow",
    )
    check(
        str(rf["trial"]) == TYPED["rhdmRfTrial"]
        and TS["M"]["cmTreeProvRfTrials"] == str(rf["trials"]["complete_considered"]),
        r"random forest: trial \rhdmRfTrial{} of \cmTreeProvRfTrials",
    )
    check(
        TS["M"]["cmPtFitWin"] == n_(OUTLINE["perbar_window"]),
        r"\cmPtFitWin = the closing-interval window (hyperparameter-table notes)",
    )
    ta = ftable(
        "tab:rh_dm_hyper",
        "Hyperparameters: penalized regressions and tree ensembles.",
        spec,
        body,
        notes_a,
    )

    # ------------------------------------------------------------------ (ii) re-selected in the rolling window
    # 2026-10-15 (user): parked -- NN content parked for now. Panel B (the MinGRU networks) is parked, and with it the
    # "Selected (count)" column, which only the network rows filled (every tree row read "--"); the panel headings
    # lose their letters, since one panel remains.
    # spec_b = "@{}l" + P_(0.36) + "lr@{}"
    # body_b = [r"\toprule", cells(["Model", "Hyperparameter", "Search space", "Selected (count)"]), r"\midrule",
    #           panel_row(rf"Panel A: Tree ensembles, closing-interval models ({op['N_TRIALS_DEFAULT']} Optuna trials "
    #                     r"per re-selection)", 4)]
    spec_b = "@{}l" + P_(0.40) + "l@{}"
    body_b = [
        r"\toprule",
        cells(["Model", "Hyperparameter", "Search space"]),
        r"\midrule",
        panel_row(
            rf"Tree ensembles, closing-interval models ({op['N_TRIALS_DEFAULT']} Optuna trials per re-selection)",
            3,
        ),
    ]
    # 2026-10-15 (consolidation): the forest's depth axis as actually searched. The Optuna jobs' default
    # RF_MAX_DEPTH_CHOICES (2, 4, 8, 16, 32, None) is patched by specs/causal_tune_trees_optuna_rfp.py (depth_ladder,
    # :93; applied at :131-136) to {d/8, d/4, d/2, d, 2d} + None around the fixed depth d (rf_opt_00, d = 20), which
    # gives (3, 5, 10, 20, 40, None), the grid the picks run's trials record. The ladder is recomputed here from d.
    rfp_src = (REPO / "specs" / "causal_tune_trees_optuna_rfp.py").read_text(
        encoding="utf-8"
    )
    check(
        "def depth_ladder" in rfp_src and "(0.125, 0.25, 0.5, 1.0, 2.0)" in rfp_src,
        "the forest's depth axis is re-centred by depth_ladder (factors 1/8 .. 2)",
    )
    d_rf = int(rf["params"]["max_depth"])
    rf_depths = (
        *sorted(
            {
                v
                for v in (
                    math.floor(d_rf * f + 0.5) for f in (0.125, 0.25, 0.5, 1.0, 2.0)
                )
                if v >= 1
            }
        ),
        None,
    )
    check(
        d_rf in rf_depths and len(rf_depths) == 6,
        f"forest depth ladder around d = {d_rf}: {rf_depths}",
    )
    for fam in ("lgbm", "xgb", "rf"):
        if fam != "lgbm":
            body_b.append(r"\addlinespace")
        for i, (k, v) in enumerate(oj["SPACES"][fam].items()):
            if fam == "rf" and k == "max_depth":
                space = gset(
                    rf_depths
                )  # the patched axis (depth_ladder), not the jobs' default
            elif v[0] == "cat":
                space = gset(v[1])
            else:
                lo = xlo if (fam == "xgb" and k == "reg_lambda") else v[1]
                space = rng(lo, v[2], v[3])
            # 2026-10-15 (user): parked -- NN content parked for now (the "Selected" column, NA for every tree row)
            # body_b.append(cells([MODEL[fam] if i == 0 else "", tree_word(fam, k), space, NA]))
            body_b.append(
                cells([MODEL[fam] if i == 0 else "", tree_word(fam, k), space])
            )
    # 2026-10-15 (user): parked -- NN content parked for now: the Selected cell of the random-search row and the
    # Panel B heading.
    # body_b += [r"\addlinespace", cells(["Random search", "The same hyperparameters",
    #                                     rf"{ts['N_CANDIDATES']} candidates", NA]),
    #            r"\addlinespace",
    #            panel_row("Panel B: MinGRU network (ridge-initialized, trained and selected on QLIKE)", 4)]
    body_b += [
        r"\addlinespace",
        cells(
            [
                "Random search",
                "The same hyperparameters",
                rf"{ts['N_CANDIDATES']} candidates",
            ]
        ),
    ]
    q, pr = NN["q"], NN["pooled_rq"]
    check(len(pr["sel"]) == nblk, "10 (block, design) picks in the pooled QLIKE walk")

    def nn_rows(
        label: str, d: dict, keys: tuple[tuple[str, str, str], ...]
    ) -> list[str]:
        cnt = pick_counts(d["sel"])
        out = [mc(4, label, "@{}l") + r" \\"]
        for gk, sk, lab in keys:
            v, m = cnt[sk]
            out.append(
                cells(["", lab, gset(d["grid"][gk]), f"{g_(v)} ({m}/{len(d['sel'])})"])
            )
        return out

    base_keys = (
        ("L", "L", r"Look-back length, $\ell$"),
        ("H", "H", r"Hidden width, $H$"),
        ("lr", "lr", r"Learning rate, $\eta_0$"),
    )
    q_keys = base_keys + (
        ("wd_enc", "we", r"Encoder weight decay, $\nu_{\mathrm{enc}}$"),
        ("wd_lin", "wl", r"Shortcut weight decay, $\nu_{\mathrm{lin}}$"),
        ("lr_lin", "ll", r"Shortcut learning rate, $\eta_{\mathrm{lin}}$"),
    )
    # 2026-10-15 (table legibility): parked -- the two squared-error walks are development specifications (the
    # networks of record are the ridge-initialized QLIKE walks, make_rehaul_results.py SOURCES); they belong to the
    # appendix MinGRU development table.
    # body_b += nn_rows(f"MinGRU, closing-interval model, trained on squared error ({b['n']} configurations)", b,
    #                   base_keys)
    # body_b += nn_rows(f"MinGRU, pooled intraday model, trained on squared error ({p['n']} configurations)", p,
    #                   base_keys + (("wd", "wd", r"Weight decay, $\nu$"),))
    # 2026-10-15 (user): parked -- NN content parked for now (Panel B rows). The network checks below still run.
    # body_b += nn_rows(f"Closing-interval model ({q['n']} configurations)", q, q_keys)
    # body_b.append(r"\addlinespace")
    # body_b += nn_rows(f"Pooled intraday model ({pr['n']} configurations)", pr, q_keys)
    nn_panel = nn_rows(f"Closing-interval model ({q['n']} configurations)", q, q_keys)
    nn_panel += nn_rows(f"Intraday model ({pr['n']} configurations)", pr, q_keys)
    check(
        len(nn_panel) == 2 + 2 * len(q_keys),
        "the parked network panel still builds (two models x six hyperparameters)",
    )
    body_b.append(r"\bottomrule")
    bc = q["common"]
    check(
        all(
            k in bc
            for k in (
                "layers",
                "batch",
                "max_epochs",
                "patience",
                "momentum",
                "max_grad_norm",
            )
        )
        and bc["layers"] == pr["common"]["layers"]
        and bc["batch"] == pr["common"]["batch"]
        and bc["momentum"] == pr["common"]["momentum"]
        and bc["max_grad_norm"] == pr["common"]["max_grad_norm"],
        "MinGRU walks of record share layers, batch, momentum and clip",
    )
    nsets = len(set(q["sel"]["design"]))
    check(
        nsets == len(set(pr["sel"]["design"])) == 2,
        "two predictor sets in each MinGRU walk of record",
    )
    # 2026-10-15 (user): parked -- NN content parked for now: the "Panel A:" prefix (one panel remains), the Panel B
    # note (the network's fixed settings) and the "Selected" note (the network picks).
    # notes_b = [rf"Panel A: at each re-selection, ...", ..., r"Panel B: the network of Section~\ref{...}; fixed: ...",
    #            rf"Selected: the modal choice over the {nblk // nsets} blocks and {nsets} predictor sets, with its count."]
    notes_b = [
        rf"At each re-selection, {op['N_TRIALS_DEFAULT']} trials of Optuna's TPE sampler start from the "
        r"fixed hyperparameters of Table~\ref{tab:rh_dm_hyper} and are ranked by validation MSE (in one "
        r"specification, QLIKE); hyperparameters not listed stay fixed, and $B$ is set by early stopping with "
        rf"patience $\lceil 1/\eta \rceil$ and capped at $\lceil {oj['ROUNDS_CAP_MULT']} B^{{\mathrm{{fix}}}} "
        r"\eta^{\mathrm{fix}} / \eta \rceil$; (log): log-uniform.",
        rf"Random search: the fixed configuration and {ts['N_CANDIDATES'] - 1} candidates drawn once from product "
        rf"grids of {n_(int(msz.group(1)))} (LightGBM), {n_(int(msz.group(2)))} (XGBoost) and {msz.group(3)} "
        r"(random forest) configurations.",
        # 2026-10-15 (user): parked -- NN content parked for now
        # r"Panel B: the network of Section~\ref{sec:rh_methods_models}; fixed: "
        # rf"{bc['layers']} MinGRU layer, batch size {bc['batch']}, at most $E_{{\max}} = {bc['max_epochs']}$ epochs "
        # rf"(pooled intraday model: {pr['common']['max_epochs']}) with patience {bc['patience']} "
        # rf"({pr['common']['patience']}), Muon momentum {bc['momentum']}, gradient-norm clip {bc['max_grad_norm']}, "
        # r"cosine learning-rate schedule.",
        # rf"Selected: the modal choice over the {nblk // nsets} blocks and {nsets} predictor sets, with its count.",
    ]
    tb = ftable(
        "tab:rh_dm_hyper_search",
        "Hyperparameter search in the rolling-window exercise.",
        spec_b,
        body_b,
        notes_b,
    )
    # 2026-10-15 (user): parked -- "park all tables that I haven't mentioned in this chat"; tab:rh_dm_hyper_search
    # now goes to its own file (generated/rehaul_dm_hyper_search.tex, not \input), tab:rh_dm_hyper stays.
    # Was: write("rehaul_dm_hyper.tex", ta + "\n" + tb.replace(HEAD, ""))
    write("rehaul_dm_hyper.tex", ta)
    write("rehaul_dm_hyper_search.tex", tb)


def write_tuning(P: dict, TS: dict, NN: dict) -> None:
    """M2 (tab:rh_dm_tuning): windows, re-estimation, re-selection and criteria, by model class."""
    clp = P["clp"]
    T = TS["T"]
    op = consts(REPO / "specs" / "causal_tune_trees_optuna.py", ("PATHS",))
    oj = consts(REPO / "specs" / "causal_tune_trees_optuna_jobs.py", ("TPE_SEED_BASE",))
    ts = consts(REPO / "specs" / "causal_tune_trees_tuned.py", ("GRID_SEED",))
    # 2026-10-15 (consolidation): parked -- only the parked bank-combination row below used these.
    # su = consts(
    #     REPO / "experiments" / "score_unification.py",
    #     ("TREE_TUNE_EVERY", "TREE_EMBARGO", "TREE_TAIL"),
    # )
    pl = consts(
        REPO / "experiments" / "pool_linear_pretune.py",
        (
            "TW_DAYS",
            "PERIODS_PER_DAY",
            "PRE_FOLDS",
            "PRE_FOLD_SESS",
            "PRE_EMBARGO_SESS",
        ),
    )
    lin = consts(
        REPO / "specs" / "causal_tune_linear.py", ("TUNE_PER", "VAL_TAIL", "EMBARGO")
    )
    tp = sorted({v[0] for v in op["PATHS"].values()})
    # 2026-10-15 (user): parked -- NN content parked for now (the pooled MinGRU row read NN["pooled_rq"]; the
    # closing-interval walk's 125 / 25 check below still reads NN["q"])
    # q, pr = NN["q"], NN["pooled_rq"]
    q = NN["q"]
    nexp = consts(REPO / "src" / "unification.py", ("N_TREE_EXPERTS",))[
        "N_TREE_EXPERTS"
    ]
    mb = re.search(
        r"TREE_REFIT_EVERY=(\d+)",
        (REPO / "specs" / "causal_tune_trees.py").read_text(encoding="utf-8"),
    )
    check(
        mb is not None,
        "the pooled bank's TREE_REFIT_EVERY (specs/causal_tune_trees.py docstring)",
    )
    assert mb is not None
    bank_every = int(mb.group(1))
    check(
        lin["VAL_TAIL"] == clp["VAL_TAIL"] == 125
        and lin["EMBARGO"] == clp["EMBARGO"] == 25,
        "125 / 25 split",
    )
    check(
        q["common"]["val_tail"] == clp["VAL_TAIL"]
        and q["common"]["embargo"] == clp["EMBARGO"],
        "MinGRU per-bar 125 / 25",
    )
    check(pl["TW_DAYS"] == clp["TW"], "pooled linear window = 2,000 sessions")
    W = n_(clp["TW"])
    ppd = pl["PERIODS_PER_DAY"]
    x = r"$\times$"
    vg = f"{clp['VAL_TAIL']} / {clp['EMBARGO']}"
    # 2026-10-15 (user): parked -- NN content parked for now: the MinGRU rows of both panels, and the seed column $S$
    # (seeds averaged), which only the network filled (trees 1, linear --). The last cell of every row below is
    # dropped with it; nrow() keeps the parked rows' code readable.
    # spec = "@{}" + P_(0.20) + "r" + P_(0.125) + P_(0.145) + P_(0.16) + P_(0.165) + "r@{}"
    spec = "@{}" + P_(0.20) + "r" + P_(0.125) + P_(0.145) + P_(0.16) + P_(0.195) + "@{}"

    def nrow(c: list[str]) -> str:
        """A row without its parked seed cell."""
        check(len(c) == 7, "a tuning row as built: six cells and the parked seed cell")
        return cells(c[:6])

    body = [
        r"\toprule",
        cells(
            [
                "Model",
                "$W$",
                stack("Re-estimated", "every", align="l"),
                stack("Re-selected", "every", align="l"),
                stack("Validation", "sample / gap", align="l"),
                "Criterion",
                # "$S$",  # 2026-10-15 (user): parked -- NN content parked for now
            ]
        ),
        r"\midrule",
        panel_row("Panel A: Closing-interval models (trading days)", 6),
        nrow(["OLS", W, "1", NA, NA, NA, NA]),
        nrow(
            [
                "Ridge, Lasso, Elastic net",
                W,
                "1",
                rf"once (pre-sample); then {clp['TUNE_PER']}",
                rf"{clp['PRE_FOLDS']} {x} {clp['PRE_FOLD_LEN']} / {clp['EMBARGO']}; then {clp['RETUNE_FOLDS']} {x} "
                f"{vg}",
                rf"MSE; switch if $\DM < -z_{{{clp['MARGIN_ALPHA']}/|\mathcal C|}}$",
                NA,
            ]
        ),
        nrow(
            [
                "Trees, fixed hyperparameters",
                W,
                f"{T['REFIT_EVERY']} or 1",
                NA,
                NA,
                NA,
                "1",
            ]
        ),
        nrow(
            [
                "Trees, random search",
                W,
                f"{T['REFIT_EVERY']}",
                f"{clp['TUNE_PER']}",
                vg,
                "MSE",
                "1",
            ]
        ),
        nrow(
            [
                "Trees, Optuna",
                W,
                "1",
                ", ".join(str(v) for v in tp),
                vg,
                "MSE (one specification: QLIKE)",
                "1",
            ]
        ),
        # 2026-10-15 (user): parked -- NN content parked for now
        # nrow(["MinGRU", W, "1", rf"{q['blocks']['block']} ({len(q['blocks']['blocks'])} blocks)",
        #       f"{q['common']['val_tail']} / {q['common']['embargo']}", "QLIKE of the seed mean",
        #       str(len(q["seeds"]))]),
        r"\addlinespace",
        panel_row("Panel B: Intraday models (30-minute intervals)", 6),
        nrow(
            [
                "Ridge, Lasso, Elastic net",
                n_(pl["TW_DAYS"] * ppd),
                "1",
                rf"once (pre-sample); then {lin['TUNE_PER']}",
                rf"{pl['PRE_FOLDS']} {x} {n_(pl['PRE_FOLD_SESS'] * ppd)} / {n_(pl['PRE_EMBARGO_SESS'] * ppd)}; then "
                rf"{lin['VAL_TAIL']} / {lin['EMBARGO']}",
                "as in Panel A",
                NA,
            ]
        ),
        # 2026-10-15 (consolidation): parked -- this row described the 20-configuration bank's combination (best
        # configuration or exponential weights, re-selected every TREE_TUNE_EVERY), but the paper's pooled trees are
        # single arms of that bank, one fixed configuration per method (specs/causal_tune_trees.py docstring:
        # tree_expert_00 = LightGBM, tree_expert_16 = XGBoost; results/pooled_trees_2026-10-07 for the forest),
        # as Methods (05_methods.tex, pooled intraday trees) states.
        # cells([rf"Trees, fixed hyperparameters ({nexp} configurations)", n_(OUTLINE["pooled_window"]), "1",
        #        f"{su['TREE_TUNE_EVERY']}", f"{su['TREE_TAIL']} / {su['TREE_EMBARGO']}",
        #        "QLIKE: best configuration, or exponential weights", "1"]),
        nrow(
            [
                r"Trees, fixed hyperparameters (one configuration per method)",
                n_(OUTLINE["pooled_window"]),
                "1",
                NA,
                NA,
                NA,
                "1",
            ]
        ),
        # 2026-10-15 (table legibility): parked -- the squared-error pooled walk (development); the row now shows the
        # pooled MinGRU of record, the ridge-initialized QLIKE walk (NN["pooled_rq"]).
        # cells(["MinGRU", n_(p["W"]), f"{p['refit']} days",
        #        rf"{q['blocks']['block']} days ({len(q['blocks']['blocks'])} blocks)",
        #        f"{n_(p['VAL'])} / {p['EMB']}", "MSE of the seed mean", str(len(p["seeds"]))]),
        # 2026-10-15 (user): parked -- NN content parked for now
        # nrow(["MinGRU", n_(pr["W"]), f"{pr['refit']} days",
        #       rf"{pr['per_block']} re-estimations ({pr['blocks']} blocks)", f"{n_(pr['VAL'])} / {pr['EMB']}",
        #       "QLIKE of the seed mean", str(len(pr["seeds"]))]),
        r"\bottomrule",
    ]
    # 2026-10-15 (table legibility): parked -- the seed values (now "All random seeds are fixed"; the values stay in
    # the code: T['SEED'], oj['TPE_SEED_BASE'] plus an offset per specification, ts['GRID_SEED']) and the
    # pre-sample validation dates (in tab:rh_dm_samples).
    check(
        isinstance(T["SEED"], int)
        and isinstance(oj["TPE_SEED_BASE"], int)
        and isinstance(ts["GRID_SEED"], int),
        "every tree seed is a fixed integer",
    )
    notes = [
        r"Every split is forward: the estimation sample, a gap and the validation sample at the end of the rolling "
        r"window $\mathcal W_t$ (Section~\ref{sec:rh_methods_tuning}), so the out-of-sample period is never used for "
        r"a choice.",
        r"Re-estimated and re-selected every: the spacing of successive estimations and hyperparameter selections, "
        r"in the units of the panel; $\mathcal C$: the grid neighbors that challenge the selected penalty.",
        # 2026-10-15 (user): parked -- NN content parked for now: the seed column's definition and the network's
        # re-estimation clause. Was: "...unless QLIKE is named; $S$: the number of random seeds averaged; all random
        # seeds are fixed." and "Optuna starts a new search at each re-selection; the MinGRU network is re-estimated
        # from scratch at each origin, each seed at its early-stopping epoch; the pooled trees use ..."
        r"Criteria are computed on the scale of $\tilde y_{t,i}$ unless QLIKE is named; all random seeds are fixed.",
        r"Optuna starts a new search at each re-selection; the pooled trees use one fixed configuration per method (for "
        rf"LightGBM and XGBoost, single arms of a {nexp}-configuration development bank), re-estimated at every "
        rf"interval (part of them only every {bank_every}th).",
    ]
    write(
        "rehaul_dm_tuning.tex",
        ftable(
            "tab:rh_dm_tuning",
            "Hyperparameter tuning, by model class.",
            spec,
            body,
            notes,
        ),
    )


# ============================================================================ number macros
# 2026-10-15 (number macros): the sample counts and constants the Data, Predictors and Methods sections (and the
# Introduction) still typed, as \rhdm... macros in a macro-only file loaded once by the preamble of paper_rehaul.tex.
# TYPED holds each value as the text typed it on 2026-10-15; the run stops if a computed value differs, so a change
# in the data or the code cannot move a number silently (update the text and TYPED together).
NUMBERS_FILE = "rehaul_dm_numbers.tex"
BUCKETS_NUMBERS = (
    GEN / "buckets_numbers.tex"
)  # \bkSize<Bucket>: the group sizes 7/14/6/6/3/3/4/4 (make_buckets_tex.py)
BK_SIZE = {
    "moments": "bkSizeMom",
    "liquidity": "bkSizeLiq",
    "market_ew": "bkSizeMew",
    "market_vw": "bkSizeMvw",
    "sentiment": "bkSizeSent",
    "implied_vol": "bkSizeIv",
    "vol_demand": "bkSizeVd",
    "fomc": "bkSizeFomc",
}
TYPED = {
    "rhdmNIntervals": "48",
    "rhdmGridRows": "303{,}442",
    "rhdmBurnIn": "3{,}125",
    "rhdmPooledRows": "300{,}317",
    "rhdmPooledWindow": "24{,}000",
    "rhdmPooledLinWindow": "96{,}000",
    "rhdmCloseDays": "3{,}469",
    "rhdmExpirationsOos": "872",
    "rhdmEarlyCloses": "6",
    "rhdmHeldOut": "413",
    "rhdmHeldOutExpirations": "419",
    "rhdmSpxwExpirations": "1{,}291",
    # 2026-10-19 (user): parked -- base 5 across the paper. Was: rhdmNColsPooled 1{,}144, rhdmNHorizonsPooled 12,
    # rhdmNEdgePooled 24, rhdmNActivePooled 5 (the base-2 panel of the earlier pooled fits)
    "rhdmNColsPooled": "574",
    "rhdmNSeriesPooled": "43",
    "rhdmNHorizonsPooled": "6",
    "rhdmNHorizonsClose": "6",
    "rhdmNCalendar": "16",
    "rhdmNEdgePooled": "12",
    "rhdmNActive": "7",
    "rhdmNActivePooled": "4",
    "rhdmValTail": "125",
    "rhdmRetuneFolds": "two",
    "rhdmRfTrial": "255",
}
WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five"}
# 2026-10-15 (finish): the pooled linear models' window, 2,000 trading days x 48 intervals = 96,000, had no macro
# (typed with a FLAG in 03_data.tex, 06_results.tex and 10_appendix.tex); \rhdmPooledLinWindow reads it from the code.
POOL_LINEAR = REPO / "experiments" / "pool_linear_pretune.py"
_POOLED_LIN_W: dict[str, int] = {}


def pooled_lin_window(clp: dict) -> int:
    """The pooled linear models' rolling estimation window W, in intervals: TW_DAYS x PERIODS_PER_DAY of
    experiments/pool_linear_pretune.py:65-67 (W = 2,000 trading days x 48 = 96,000)."""
    if "W" not in _POOLED_LIN_W:
        pl = consts(POOL_LINEAR, ("TW_DAYS", "PERIODS_PER_DAY", "W"))
        check(
            pl["W"] == pl["TW_DAYS"] * pl["PERIODS_PER_DAY"],
            "pool_linear_pretune.py: W = TW_DAYS x PERIODS_PER_DAY",
        )
        check(
            pl["TW_DAYS"] == clp["TW"],
            "pooled linear window in trading days = the closing-interval window (2,000)",
        )
        check(
            pl["PERIODS_PER_DAY"] == TG.PERIODS_PER_DAY == 48,
            "48 intervals a day (pool_linear_pretune.py = src/features/transforms/target.py)",
        )
        assert pl["W"] == 96_000, f"pooled linear window {pl['W']} != 96,000"
        _POOLED_LIN_W["W"] = int(pl["W"])
    return _POOLED_LIN_W["W"]


def preamble_number_files() -> list[Path]:
    """The number files paper_rehaul.tex loads in its preamble (the \\input{generated/...} lines before
    \\begin{document}), this generator's own file excluded."""
    pre = (REPO / "writeup" / "paper_rehaul.tex").read_text(encoding="utf-8")
    pre = pre.split(r"\begin{document}")[0]
    out = []
    for line in pre.splitlines():
        m = re.match(r"\s*\\input\{generated/([\w-]+)\}", line)
        if m and m.group(1) + ".tex" != NUMBERS_FILE:
            out.append(GEN / (m.group(1) + ".tex"))
    return out


def defined_macros(p: Path) -> dict[str, str]:
    """name -> value of every \\newcommand / \\providecommand in a number file (values to the first unbraced '}')."""
    out: dict[str, str] = {}
    for m in re.finditer(
        r"\\(?:newcommand|providecommand)\{\\(\w+)\}\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}",
        need(p).read_text(encoding="utf-8"),
    ):
        out.setdefault(m.group(1), m.group(2))
    return out


def write_numbers(P: dict, D: dict, TS: dict, LP: dict) -> None:
    """rehaul_dm_numbers.tex: macro definitions only (no tables)."""
    clp, M = P["clp"], TS["M"]
    ppd = TG.PERIODS_PER_DAY
    clocks = P["df"]["t"].dt.strftime("%H:%M").nunique()
    check(
        clocks == ppd == 48,
        f"{ppd} half-hour intervals a day: every clock time occurs on the grid",
    )
    check(
        P["pooled_window"] == TS["T"]["POOLED_WINDOW_ROWS"],
        "the pooled window (src/unification.py) = the trees' leaf-rescaling denominator (specs/causal_tune_trees.py)",
    )
    check(
        clp["VAL_TAIL"] == LP["G"]["VAL_TAIL"] and clp["EMBARGO"] == LP["G"]["EMBARGO"],
        "validation sample and gap: close_linear_pretune.py = specs/causal_tune_linear.py",
    )
    check(
        M["cmPtEmbargo"] == str(clp["EMBARGO"]),
        r"the gap is \cmPtEmbargo (close_main_numbers.tex); no new macro",
    )
    lat, lat_full = P["later"]
    check(
        lat - lat_full
        == int(
            SRC["exp"][SRC["exp"].index > pd.Timestamp(clp["LAST_TRADE_DAY"])].sum()
        ),
        "held-out expirations less their early closes = the held-out full trading days",
    )
    # the group sizes are \bkSize* of buckets_numbers.tex (macro-only; loaded by the preamble): same counts here
    bk = defined_macros(BUCKETS_NUMBERS)
    check(
        all(bk[BK_SIZE[b]] == str(D["nser"][b]) for b in BUCKET_LABEL),
        r"\bkSize* (buckets_numbers.tex) = the series per group, from src/data/loading.py SUBGROUPS",
    )
    vals = {
        "rhdmNIntervals": (
            str(ppd),
            "intraday intervals a day, M (half hours; the intraday model)",
        ),
        "rhdmGridRows": (
            n_(P["grid"][2]),
            "intervals on the intraday grid (observed RV, weekend break out)",
        ),
        "rhdmBurnIn": (n_(P["burn"]), "burn-in intervals, the longest HAR horizon"),
        "rhdmPooledRows": (
            n_(P["pooled"][2]),
            "intraday sample, intervals after the burn-in",
        ),
        "rhdmPooledWindow": (
            n_(P["pooled_window"]),
            "pooled rolling estimation window W, intervals",
        ),
        "rhdmPooledLinWindow": (
            n_(pooled_lin_window(clp)),
            "pooled linear models' rolling estimation window W, intervals (2,000 trading days x 48)",
        ),
        "rhdmCloseDays": (
            n_(P["perbar"][2]),
            "closing-interval sample, trading days after the burn-in",
        ),
        "rhdmExpirationsOos": (
            str(P["exp_to_end"]),
            # "SPXW expirations to the end of the out-of-sample period",  # 2026-10-15: parked -- the wording
            "SPXW expirations to the end of the evaluation span (2024-04-30)",
        ),
        "rhdmEarlyCloses": (
            str(P["ec_to_end"]),
            # "of which early-close days (T_oos = expirations less these)",  # 2026-10-15: parked -- the wording
            "of which early-close days (T_opt, the option-data subsample = expirations less these)",
        ),
        "rhdmHeldOut": (
            str(lat_full),
            "later SPXW expirations held out: full trading days",
        ),
        "rhdmHeldOutExpirations": (
            str(lat),
            "later SPXW expirations held out: all, early closes included",
        ),
        "rhdmSpxwExpirations": (
            n_(len(SRC["exp"])),
            "SPXW expirations in the option file",
        ),
        "rhdmNColsPooled": (
            n_(D["n_pooled"]),
            "columns of the intraday model's full predictor set",
        ),
        "rhdmNSeriesPooled": (
            str(D["n_series_pooled"]),
            "exogenous series in the pooled panel (pre-FOMC)",
        ),
        "rhdmNHorizonsPooled": (
            str(D["kq"]),
            # 2026-10-19 (user): parked -- base 5 across the paper. Was: "HAR horizons, intraday model (base 2)",
            "HAR horizons, intraday model (base 5)",
        ),
        "rhdmNHorizonsClose": (
            str(D["k"]),
            "HAR horizons, closing-interval model (base 5)",
        ),
        "rhdmNCalendar": (str(len(D["cal"])), "calendar-dummy columns (both models)"),
        "rhdmNEdgePooled": (
            str(D["n_edge_pooled"]),
            "HAR x opening/closing-interval dummy columns (pooled)",
        ),
        "rhdmNActive": (
            str(len(D["act"])),
            "series with an activity indicator, closing-interval model",
        ),
        "rhdmNActivePooled": (
            str(D["n_active_pooled"]),
            "series with an activity indicator, pooled panel",
        ),
        "rhdmValTail": (
            str(clp["VAL_TAIL"]),
            "validation sample n_val, observations (the gap is \\cmPtEmbargo)",
        ),
        "rhdmRetuneFolds": (
            WORDS[clp["RETUNE_FOLDS"]],
            "validation folds of the linear light re-tune",
        ),
        "rhdmRfTrial": (
            str(TS["rf"]["trial"]),
            "the random forest's fixed configuration: Optuna trial number",
        ),
    }
    check(
        D["n_pooled"]
        == D["kq"]
        + len(D["cal"])
        + D["n_edge_pooled"]
        + 2 * D["n_series_pooled"] * D["kq"]
        + D["n_active_pooled"] * D["kq"],
        # 2026-10-19 (user): parked -- base 5. Was: "1,144 = 12 + 16 + 24 + 43 x 12 + 43 x 12 + 5 x 12 (the Predictors section's display)",
        "574 = 6 + 16 + 12 + 43 x 6 + 43 x 6 + 4 x 6 (the Predictors section's display)",
    )
    check(set(vals) == set(TYPED), "every \\rhdm macro has its typed value")
    for k, (v, _) in vals.items():
        check(v == TYPED[k], rf"\{k} = {v} (typed {TYPED[k]})")
    # no name clash with the other number files of the preamble
    others: dict[str, str] = {}
    for p in preamble_number_files():
        for name in defined_macros(p):
            others.setdefault(name, p.name)
    clash = sorted(k for k in vals if k in others)
    check(not clash, f"no \\rhdm name defined in another preamble number file: {clash}")
    lines = [
        "% AUTO-GENERATED by writeup/make_rehaul_data_methods.py -- do not edit; re-run the script.",
        "% Macro definitions only (no tables); loaded once by the preamble of paper_rehaul.tex. Sample counts and",
        "% constants of the Introduction and the Data, Predictors and Methods sections, each computed from the data,",
        "% the specs and the result files (asserted against the values the text typed on 2026-10-15). The group sizes",
        "% are \\bkSize<Group> (buckets_numbers.tex), the gap \\cmPtEmbargo (close_main_numbers.tex).",
    ]
    lines += [rf"\newcommand{{\{k}}}{{{v}}}  % {what}" for k, (v, what) in vals.items()]
    write(NUMBERS_FILE, "\n".join(lines) + "\n")


# ============================================================================ main
SRC: dict = {}

# 2026-10-15 (table legibility): the data-audit fixes landed in src/ today ("fix all 8"; results/data_audit_2026-10-15/,
# results/data_fix_2026-10-15/) behind module flags, each of which "off reproduces the pre-2026-10-15 chain".  The
# paper's results, and the Data and Predictors sections, still describe the pre-fix chain, so these tables document
# that chain: the flags below are switched off for this run only (src/ is not edited).  Set DOCUMENT_PRE_FIX_CHAIN =
# False once the results are re-run on the fixed chain, and re-baseline the checks that move (the asinh routing, the
# signed VW spread, the FOMC flags' transform).
DOCUMENT_PRE_FIX_CHAIN = True
PRE_FIX_FLAGS = (
    (LD, ("NULL_INVALID_PRINTS", "FOMC_CALENDAR_FIX", "SPLICE_VIX_FREE_FEED")),
    (
        TG,
        (
            "SIGN_BY_NAME",
            "FOMC_FLAGS_IDENTITY",
            "CAUSAL_SD_FLOOR",
            "CAUSAL_ASINH_ROUTING",
        ),
    ),
)


def pin_documented_chain() -> None:
    if not DOCUMENT_PRE_FIX_CHAIN:
        return
    off = []
    for mod, names in PRE_FIX_FLAGS:
        for nm in names:
            if hasattr(mod, nm):
                setattr(mod, nm, False)
                off.append(nm)
    print(
        f"documenting the pre-2026-10-15 chain: {', '.join(off) or 'no fix flags present'} off"
    )


def main() -> None:
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    pin_documented_chain()
    SRC.update(sources())
    P = panel()
    bmap = series_bucket()
    D = design_tables(P, bmap)
    rows = dictionary(P, bmap, D)
    TS = tree_settings(P)
    LP = linear_picks(P)
    NN = mingru()
    write_sources(SRC)
    write_samples(P)
    write_design(D)
    write_dictionary(rows, D)
    write_hyper(P, TS, LP, NN)
    write_tuning(P, TS, NN)
    write_numbers(P, D, TS, LP)
    print(f"{len(CHECKS)} checks passed")
    for c in CHECKS:
        print("  ok:", c)


if __name__ == "__main__":
    main()
