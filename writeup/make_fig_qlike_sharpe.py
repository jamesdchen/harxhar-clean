"""Paper figure: QLIKE against Sharpe, HAR + calendar vs all features (fig_qlike_sharpe).

The GUI's QLIKE-vs-Sharpe scatter (writeup/master_table_gui/static/index.html, chart code)
redrawn for the paper on the move from the HAR + calendar-indicator inputs (``baseline``) to
all features (``all_features``), across the models.  The lean set is not in this figure.

DATA.  The paper model set (results/close_master_table/paper_model_set.csv) restricted to the
input sets baseline and all_features and to the forecasts whose twin exists on the other set
(the same key with the input-set token ``_baseline`` <-> ``_all_features`` swapped).  Per-bar
linear models are the pre-tuned ``subpt_*`` rows; LSTMs are excluded.  Every model that exists
on both input sets is shown -- nothing is dropped to make the picture cleaner; the caption
gives the counts.  The HAR + calendar OLS (``sub_ols_baseline``, no all-features twin) is
drawn as the reference point the shaded quadrant hangs from.  QLIKE = ``qlike_raw`` (the
plain forecast f^2 B), Sharpe = the sign(s) Sharpe ratio, mid fill (``Sharpe_mid``), both from
results/close_master_table/master_table.csv; the per-day frame master_table_daily.parquet
reproduces both (gated) and carries the daily losses and returns the pair statistics need.

PAIR STATISTICS (all features minus HAR + calendar, the same 866 trade days).
  QLIKE    difference of the means; Diebold-Mariano statistic of the daily QLIKE difference
           (src.evaluation.diebold_mariano.dm_test, Harvey-Leybourne-Newbold corrected), at the
           master table's own setting (DM_LAG = None: dm_test's automatic Newey-West bandwidth,
           6 on 866 days, as master_table_close.score_one and make_featimp_lean_tex); the count
           at lag 14 is written beside it as a sensitivity (\\fqsQLowerSigLagFourteen).
  Sharpe   difference of the sign(s) mid Sharpe ratios; Newey-West HAC t of the daily
           mid-return difference (atm_straddle_lib.newey_west_t, lag floor(1.5 n^(1/3)) = 14,
           as the master table), two-sided normal p (master_table_close.hac_p).
           2026-10-21 (user, decision #2): parked -- the 866-day lag-14 t. Now writeup/hac_inference.py on
           the 614 post-warmup days (t > WARMUP = 252): t_hac is the LLSW (2018) t of the mean daily
           mid-return difference (bandwidth 33) with its fixed-b p (t_p); the Sharpe-change markers and
           counts (\\fqsShHigherSig, \\fqsShLowerSig) use the Ledoit-Wolf (2008) p (lw_p). The plotted
           Sharpe change is still the master table's (866 days).
  ES(1)    the elementary score at the trade's threshold: c = 1 on the implied-variance-scaled
           forecast and realized variance (close_murphy_splits_headline.elementary(f / iv, y / iv, 1);
           y the ACTUAL realized variance rv_unclipped since 2026-10-15, data audit #10; QLIKE too),
           mean over days; the change is all features minus HAR + calendar.
  Across pairs, the Pearson correlation of the ES(1) change with the HAC t of the P&L change,
  and the same for the QLIKE change: why Sharpe can rise when QLIKE does not -- the trade
  reads one point of the forecast distribution (above or below the implied variance), QLIKE
  integrates every threshold.  Significance at SIG = 5 % two-sided.  Point estimates
  otherwise; no intervals (2026-10-06, user decision).

DESIGN (mirrors the GUI chart).  One colour per estimator (the GUI's CHART_EST: OLS ink, ridge
blue, lasso violet, elastic net green, LightGBM orange, XGBoost aqua, random forest magenta);
pooled 48-bar fits are squares, per-bar (16:00 bar) fits circles; HAR + calendar points are
open, all-features points filled, and each pair is joined by a line that fades in from the
HAR + calendar end (faint) to the all-features end (dark); the light shaded quadrant is lower
QLIKE and higher Sharpe than the HAR + calendar OLS.  No title (the caption carries it).

SECOND FIGURE (fig_qlike_sharpe_change): the same pairs as changes, one point per model:
x = the QLIKE change in % of the HAR + calendar value, y = the Sharpe (mid) change, both all
features minus HAR + calendar; zero lines, the quadrant x < 0, y > 0 (lower QLIKE and higher
Sharpe) shaded, the count of models in each quadrant in its corner (\\fqsQuad... macros); an
ink ring marks a DM-significant QLIKE change, an ink triangle above the point a HAC-significant
Sharpe change, and the significant points carry a number the key below the axes resolves.

Re-runnable: if the pooled pre-tuned rows arrive later (MACROS_POOLED_DONE), the model set
file changes and a re-run picks them up; nothing here names a pooled key.

Writes writeup/figures/fig_qlike_sharpe.pdf and fig_qlike_sharpe_change.pdf (vector) with .png
previews, and
writeup/generated/fig_qlike_sharpe_numbers.tex (\\fqs... macros).

Run:  python writeup/make_fig_qlike_sharpe.py
"""

from __future__ import annotations

import math
import os
import re
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.patheffects as pe  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as mticker  # noqa: E402
import matplotlib.transforms as mtransforms  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.colors import to_rgb  # noqa: E402
from matplotlib.legend_handler import HandlerBase  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "notebooks", ROOT / "experiments", ROOT / "writeup"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import atm_straddle_lib as asl  # noqa: E402, F401  (2026-10-21: no longer used for the t; kept for the parked line)
import close_murphy_splits_headline as cms  # noqa: E402
import hac_inference as hi  # noqa: E402  (2026-10-21, decision #2)
import master_table_close as mtc  # noqa: E402
from src.evaluation.diebold_mariano import dm_test  # noqa: E402

MODEL_SET = mtc.OUT / "paper_model_set.csv"
MASTER = mtc.OUT / "master_table.csv"
DAILY = mtc.OUT / "master_table_daily.parquet"
FIG = ROOT / "writeup" / "figures" / "fig_qlike_sharpe.pdf"
PNG = FIG.with_suffix(".png")
FIG2 = ROOT / "writeup" / "figures" / "fig_qlike_sharpe_change.pdf"
PNG2 = FIG2.with_suffix(".png")
TEX = ROOT / "writeup" / "generated" / "fig_qlike_sharpe_numbers.tex"

BASE, ALL = "baseline", "all_features"
TOKEN = re.compile(r"_(baseline|all_features)(?=_|$)")
OLS_KEY = "sub_ols_baseline"  # the GUI's BASELINE_KEY: HAR + calendar OLS, 16:00 bar
N_DAYS = 866  # the deck's trade days, 2020-01-03 .. 2024-04-30 (asserted)
SIG = 0.05  # two-sided level of the DM / HAC t counts (statistics, not intervals)
DM_LAG = None  # dm_test's automatic bandwidth, the master table's DM (6 on 866 days)
DM_LAG_CHECK = 14  # sensitivity: the HAC t's lag
GATE_REL = 1e-9
WARMUP = 252  # 2026-10-21 (user, decision #2): tests on the days t > WARMUP (= master_table_gui.rows._SIZING_WARMUP)

# the GUI's CHART_EST (index.html), estimator -> colour; the order is the legend's
EST_NAME = {
    "ols": "OLS",
    "ridge": "ridge",
    "lasso": "lasso",
    "enet": "elastic net",
    "lgbm": "LightGBM",
    "xgb": "XGBoost",
    "rf": "random forest",
    "rfdev": "random forest",  # 2026-10-06: the dev-tuned forest of the T1 / T10 rows (paper set)
}
EST_COLOR = {
    "OLS": "#1c1c1c",
    "ridge": "#2a78d6",
    "lasso": "#4a3aa7",
    "elastic net": "#008300",
    "LightGBM": "#eb6834",
    "XGBoost": "#1baf7a",
    "random forest": "#e87ba4",
}
INK = "#1c1c1c"
MUTED = "#5e5a54"
LINK = "#7d766a"  # the GUI's connector grey
GRID = "#ece7de"
FRAME = "#c9c2b6"
# the GUI's gradient stops: faint at the HAR + cal. end, dark at the all-features end
LINK_ALPHA = (0.08, 0.9)
LINK_STEPS = 24
MS_CIRCLE = 26.0  # scatter area (pt^2), per-bar circles
MS_SQUARE = 22.0  # pooled squares, about the circle's area


# --------------------------------------------------------------------------- helpers
def estimator_of(key: str) -> str:
    """The GUI's estimator_of (rows.py), extended to the pre-tuned prefixes (subpt_, poolpt_)."""
    parts = key.split("_")
    tok = parts[-1] if parts[0] == "subtree" and parts[-1] in EST_NAME else parts[1]
    if tok not in EST_NAME:
        raise SystemExit(f"no estimator for {key!r}: extend EST_NAME")
    return EST_NAME[tok]


def is_pooled(key: str, family: str) -> bool:
    """Pooled 48-bar fit (the GUI's FIT_POOLED): the pooled twins and any later pooled rows."""
    return key.startswith("pool") or "pooled" in family.lower()


def twin(key: str, to: str) -> str:
    m = TOKEN.search(key)
    assert m is not None, key
    return key[: m.start()] + "_" + to + key[m.end() :]


def fnum(x: float, d: int = 2, plus: bool = False) -> str:
    """A number for LaTeX text or math: true minus sign, optional explicit plus."""
    if not math.isfinite(float(x)):
        return "---"
    s = f"{abs(float(x)):.{d}f}"
    if float(x) < 0 and float(s) != 0.0:
        return r"\ensuremath{-}" + s
    if plus and float(s) != 0.0:
        return r"\ensuremath{+}" + s
    return s


def gate(name: str, err: float, tol: float) -> None:
    ok = err <= tol
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {err:.2e} (tol {tol:.0e})")
    if not ok:
        raise SystemExit(f"gate failed: {name}")


# --------------------------------------------------------------------------- data
def load() -> tuple[pd.DataFrame, pd.Series, dict[str, pd.DataFrame]]:
    """The pairs (one row per model: keys, family, estimator, fit) with their statistics."""
    ms = pd.read_csv(MODEL_SET)
    mt = pd.read_csv(MASTER).set_index("key")
    ms = ms[ms["input_set"].isin([BASE, ALL])].copy()
    lstm = ms["key"].str.startswith("lstm") | ms["family"].str.contains(
        "LSTM", case=False
    )
    ms = ms[~lstm]
    # per-bar linear = the pre-tuned subpt_* rows; the OLS stays as the reference point only
    old_linear = (ms["family"] == "per-bar linear") & ~ms["key"].str.startswith(
        "subpt_"
    )
    assert OLS_KEY in set(ms["key"]), f"{OLS_KEY} is not in the model set"
    ms = ms[~old_linear]
    fam = ms.set_index("key")["family"].astype(str)
    keys = set(fam.index)
    pairs, unpaired = [], []
    for k in sorted(keys):
        m = TOKEN.search(k)
        if m is None:
            raise SystemExit(f"{k}: no input-set token")
        other = twin(k, ALL if m.group(1) == BASE else BASE)
        if other not in keys:
            unpaired.append(k)
        elif m.group(1) == BASE:
            pairs.append((k, other))
    print(
        f"model set: {len(keys)} forecasts on {BASE} / {ALL}; {len(pairs)} pairs; unpaired: {unpaired}"
    )

    need = sorted({x for p in pairs for x in p} | {OLS_KEY})
    missing = [k for k in need if k not in mt.index]
    assert not missing, f"model-set keys missing from master_table.csv: {missing}"
    assert (mt.loc[need, "table"] == "A").all()
    D = pd.read_parquet(DAILY)
    D = D[D["key"].isin(need)]
    fr = {k: g.set_index("day").sort_index() for k, g in D.groupby("key")}
    assert set(fr) == set(need), sorted(set(need) - set(fr))
    days = fr[OLS_KEY].index
    assert len(days) == N_DAYS, len(days)
    print("gates (the per-day frame reproduces master_table.csv):")
    gate(
        "same 866 days, every forecast",
        float(sum(not f.index.equals(days) for f in fr.values())),
        0,
    )
    gate(
        "mean daily qlike_raw = master qlike_raw",
        max(abs(fr[k]["qlike_raw"].mean() / mt.loc[k, "qlike_raw"] - 1) for k in need),
        GATE_REL,
    )
    gate(
        # "daily qlike_raw = QLIKE(target, plain)",  # 2026-10-15 (user): parked -- scored against the trimmed target; scores now use actual RV (data audit #10)
        "daily qlike_raw = QLIKE(rv_unclipped, plain)",
        max(
            float(
                np.abs(
                    cms.qlike(f["rv_unclipped"].to_numpy(), f["plain"].to_numpy())
                    - f["qlike_raw"]
                ).max()
            )
            for f in fr.values()
        ),
        1e-12,
    )
    gate(
        "Sharpe of daily ret_mid = master Sharpe_mid",
        max(
            abs(mtc.sharpe(fr[k]["ret_mid"].to_numpy()) / mt.loc[k, "Sharpe_mid"] - 1)
            for k in need
        ),
        GATE_REL,
    )
    gate(
        "q = sign(plain - iv_var)",
        max(
            float(np.abs(np.where(f["plain"] > f["iv_var"], 1.0, -1.0) - f["q"]).max())
            for f in fr.values()
        ),
        0,
    )
    gate(
        "target, rv_unclipped and iv_var shared by every forecast",
        max(
            float(np.abs(f[c].to_numpy() / fr[OLS_KEY][c].to_numpy() - 1).max())
            for f in fr.values()
            for c in ("target", "rv_unclipped", "iv_var")
        ),
        GATE_REL,
    )

    def es1(f: pd.DataFrame) -> float:
        s = f["iv_var"].to_numpy(float)
        return float(
            cms.elementary(
                # f["plain"].to_numpy(float) / s, f["target"].to_numpy(float) / s, 1.0  # 2026-10-15 (user): parked -- scored against the trimmed target; scores now use actual RV (data audit #10)
                f["plain"].to_numpy(float) / s,
                f["rv_unclipped"].to_numpy(float) / s,
                1.0,
            ).mean()
        )

    rows = []
    for b, a in pairs:
        fb, fa = fr[b], fr[a]
        dm = dm_test(
            fa["qlike_raw"].to_numpy(), fb["qlike_raw"].to_numpy(), hac_lag=DM_LAG
        )
        dm14 = dm_test(
            fa["qlike_raw"].to_numpy(), fb["qlike_raw"].to_numpy(), hac_lag=DM_LAG_CHECK
        )
        # 2026-10-21 (user, decision #2): parked -- 866 days, lag 14. Was:
        # t, lag = asl.newey_west_t(fa["ret_mid"].to_numpy() - fb["ret_mid"].to_numpy())
        ra, rb = fa["ret_mid"].to_numpy(float), fb["ret_mid"].to_numpy(float)
        assert len(ra) == len(rb) == N_DAYS
        fbt = hi.llsw_test(ra[WARMUP:] - rb[WARMUP:])
        assert fbt["T"] == N_DAYS - WARMUP == 614 and fbt["S"] == 33, fbt
        t = fbt["t"]
        rows.append(
            {
                "key_base": b,
                "key_all": a,
                "family": fam[b],
                "est": estimator_of(b),
                "pooled": is_pooled(b, fam[b]),
                "q_base": float(mt.loc[b, "qlike_raw"]),
                "q_all": float(mt.loc[a, "qlike_raw"]),
                "sh_base": float(mt.loc[b, "Sharpe_mid"]),
                "sh_all": float(mt.loc[a, "Sharpe_mid"]),
                "dm": float(dm["dm"]),
                "dm_p": float(dm["p"]),
                "dm_lag": int(dm["hac_lag"]),
                "dm_p14": float(dm14["p"]),
                "t_hac": float(t),
                "t_p": float(fbt["p"]),  # 2026-10-21 (user, decision #2): fixed-b p. Was: mtc.hac_p(t)
                "hac_s": int(fbt["S"]),  # was "hac_lag": int(lag)
                "lw_p": hi.lw_sharpe_test(ra[WARMUP:], rb[WARMUP:])["p"],
                "es_base": es1(fb),
                "es_all": es1(fa),
            }
        )
    p = pd.DataFrame(rows)
    assert (p["est"] == p["key_all"].map(estimator_of)).all()
    p["dq"] = p["q_all"] - p["q_base"]
    p["dq_pct"] = 100.0 * (p["q_all"] / p["q_base"] - 1.0)
    p["dsh"] = p["sh_all"] - p["sh_base"]
    p["des"] = p["es_all"] - p["es_base"]
    ols = mt.loc[OLS_KEY, ["qlike_raw", "Sharpe_mid"]].astype(float)
    return p, ols, fr


# --------------------------------------------------------------------------- numbers
def quadrants(p: pd.DataFrame) -> dict[str, int]:
    """Pairs per quadrant of (QLIKE change, Sharpe change); no exact zero (asserted in numbers)."""
    ql, sh = p["dq"] < 0, p["dsh"] > 0
    q = {
        "better_both": int((ql & sh).sum()),
        "sh_only": int((~ql & sh).sum()),
        "q_only": int((ql & ~sh).sum()),
        "worse_both": int((~ql & ~sh).sum()),
    }
    assert sum(q.values()) == len(p)
    return q


def numbers(p: pd.DataFrame, ols: pd.Series) -> list[tuple[str, str, str]]:
    """(macro, value, comment) for the caption."""
    assert not (p["dq"] == 0).any() and not (p["dsh"] == 0).any(), (
        "a tie: decide how to count it"
    )
    ql, qs, qs14 = p["dq"] < 0, p["dm_p"] < SIG, p["dm_p14"] < SIG
    # 2026-10-21 (user, decision #2): Sharpe-change significance from the Ledoit-Wolf p. Was: p["t_p"] < SIG
    sh, ss = p["dsh"] > 0, p["lw_p"] < SIG
    lin = (~p["pooled"]) & p["key_base"].str.startswith("subpt_")
    tree = p["key_base"].str.startswith("subtree")
    assert int(lin.sum() + p["pooled"].sum() + tree.sum()) == len(p), (
        "a pair outside linear / pooled / tree"
    )
    shade_all = int(
        ((p["q_all"] < ols["qlike_raw"]) & (p["sh_all"] > ols["Sharpe_mid"])).sum()
    )
    shade_base = int(
        ((p["q_base"] < ols["qlike_raw"]) & (p["sh_base"] > ols["Sharpe_mid"])).sum()
    )
    r_es = float(np.corrcoef(p["des"], p["t_hac"])[0, 1])
    r_q = float(np.corrcoef(p["dq"], p["t_hac"])[0, 1])
    dm_lags, hac_lags = sorted(set(p["dm_lag"])), sorted(set(p["hac_s"]))  # 2026-10-21: hac_s (was hac_lag)
    assert len(dm_lags) == 1 and len(hac_lags) == 1, (dm_lags, hac_lags)
    out = [
        ("fqsPairs", str(len(p)), "models on both input sets (pairs drawn)"),
        (
            "fqsPairsPerBarLinear",
            str(int(lin.sum())),
            "of which per-bar linear (pre-tuned)",
        ),
        (
            "fqsPairsPooled",
            str(int(p["pooled"].sum())),
            "of which pooled 48-bar linear",
        ),
        ("fqsPairsTree", str(int(tree.sum())), "of which per-bar trees"),
        (
            "fqsPoints",
            str(2 * len(p) + 1),
            "points drawn: both ends of every pair + the HAR + cal. OLS",
        ),
        ("fqsDays", str(N_DAYS), "trade days"),
        ("fqsQLower", str(int(ql.sum())), "QLIKE lower with all features"),
        ("fqsQLowerSig", str(int((ql & qs).sum())), "... and DM significant at 5 %"),
        ("fqsQHigher", str(int((~ql).sum())), "QLIKE higher with all features"),
        ("fqsQHigherSig", str(int((~ql & qs).sum())), "... and DM significant at 5 %"),
        (
            "fqsQLowerSigLagFourteen",
            str(int((ql & qs14).sum())),
            "sensitivity: QLIKE lower, DM at lag 14 significant",
        ),
        (
            "fqsQHigherSigLagFourteen",
            str(int((~ql & qs14).sum())),
            "sensitivity: QLIKE higher, DM at lag 14 significant",
        ),
        ("fqsShHigher", str(int(sh.sum())), "Sharpe (mid) higher with all features"),
        (
            "fqsShHigherSig",
            str(int((sh & ss).sum())),
            "... and Ledoit-Wolf p < 0.05 (614 post-warmup days)",
        ),
        ("fqsShLower", str(int((~sh).sum())), "Sharpe (mid) lower with all features"),
        (
            "fqsShLowerSig",
            str(int((~sh & ss).sum())),
            "... and Ledoit-Wolf p < 0.05 (614 post-warmup days)",
        ),
        (
            "fqsQLowerShHigher",
            str(int((ql & sh).sum())),
            "both: QLIKE lower and Sharpe higher",
        ),
        (
            "fqsMedDQ",
            fnum(p["dq"].median(), 4, True),
            "median QLIKE change (all features minus HAR + cal.)",
        ),
        (
            "fqsMedDQPct",
            fnum(p["dq_pct"].median(), 1, True),
            "median QLIKE change, % of the HAR + cal. QLIKE",
        ),
        ("fqsMedDSh", fnum(p["dsh"].median(), 2, True), "median Sharpe (mid) change"),
        (
            "fqsESLower",
            str(int((p["des"] < 0).sum())),
            "elementary score at the threshold (c = 1) lower with all features",
        ),
        (
            "fqsCorrES",
            fnum(r_es, 2),
            "Pearson corr. across pairs: ES(c = 1) change vs HAC t of the P&L change",
        ),
        (
            "fqsCorrQ",
            fnum(r_q, 2),
            "Pearson corr. across pairs: QLIKE change vs HAC t of the P&L change",
        ),
        ("fqsInShade", str(shade_all), "all-features points in the shaded quadrant"),
        ("fqsInShadeBase", str(shade_base), "HAR + cal. points in the shaded quadrant"),
        ("fqsOlsQ", fnum(ols["qlike_raw"], 4), "QLIKE of the HAR + cal. OLS"),
        ("fqsOlsSh", fnum(ols["Sharpe_mid"], 2), "Sharpe (mid) of the HAR + cal. OLS"),
        (
            "fqsDMLag",
            str(dm_lags[0]),
            "Newey-West lag of the DM statistic (the master table's)",
        ),
        # 2026-10-21 (user, decision #2): parked -- \\fqsHACLag (14, 866 days). Was:
        # ("fqsHACLag", str(hac_lags[0]), "Newey-West lag of the HAC t (the master table's)"),
        (
            "fqsHACS",
            str(hac_lags[0]),
            "bandwidth S of the t of the mean P&L change (LLSW 2018, 614 post-warmup days)",
        ),
        ("fqsSigPct", f"{100 * SIG:.0f}", "two-sided level of the counts, %"),
    ]
    # the change figure's quadrants (fig_qlike_sharpe_change): QLIKE change < 0 = better
    quad = quadrants(p)
    out += [
        (
            "fqsQuadBetterBoth",
            str(quad["better_both"]),
            "change figure: lower QLIKE, higher Sharpe",
        ),
        (
            "fqsQuadShOnly",
            str(quad["sh_only"]),
            "change figure: higher QLIKE, higher Sharpe",
        ),
        (
            "fqsQuadQOnly",
            str(quad["q_only"]),
            "change figure: lower QLIKE, lower Sharpe",
        ),
        (
            "fqsQuadWorseBoth",
            str(quad["worse_both"]),
            "change figure: higher QLIKE, lower Sharpe",
        ),
    ]
    # the Spearman versions, for the record (stdout only)
    rho_es = float(p[["des", "t_hac"]].corr("spearman").iloc[0, 1])
    rho_q = float(p[["dq", "t_hac"]].corr("spearman").iloc[0, 1])
    print(
        f"Spearman across pairs (record only): ES(1) change vs HAC t {rho_es:+.2f}; QLIKE change vs HAC t {rho_q:+.2f}"
    )
    return out


def write_tex(nums: list[tuple[str, str, str]], p: pd.DataFrame) -> None:
    fams = ", ".join(sorted(set(p.loc[p["pooled"], "family"]))) or "none"
    lines = [
        "% AUTO-GENERATED by writeup/make_fig_qlike_sharpe.py -- do not edit.",
        "% Figure writeup/figures/fig_qlike_sharpe.pdf: QLIKE against Sharpe (mid), HAR + calendar vs all",
        "% features, every model of results/close_master_table/paper_model_set.csv on both input sets",
        "% (per-bar linear = the pre-tuned subpt_* rows; LSTMs excluded); numbers from master_table.csv",
        "% and master_table_daily.parquet.  Changes are all features minus HAR + calendar, on the same days.",
        "% DM: dm_test on the daily QLIKE difference; t: hac_inference.llsw_test on the daily mid-return",
        "% difference (614 post-warmup days, bandwidth 33, fixed-b p); Sharpe changes: Ledoit-Wolf p;",
        "% ES(c = 1): close_murphy_splits_headline.elementary on f / iv, y / iv.  Point estimates.",
        f"% pooled pairs: family {fams}",
    ]
    for name, val, comment in nums:
        lines.append(f"\\newcommand{{\\{name}}}{{{val}}}  % {comment}")
    TEX.parent.mkdir(parents=True, exist_ok=True)
    with open(TEX, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"wrote {TEX.relative_to(ROOT)} ({len(nums)} macros)")


# --------------------------------------------------------------------------- figure
def _style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "font.size": 9,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.linewidth": 0.8,
            "axes.edgecolor": FRAME,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelcolor": MUTED,
            "ytick.labelcolor": MUTED,
            "text.color": INK,
            "axes.labelcolor": INK,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _fade_segments(
    x0: float, y0: float, x1: float, y1: float
) -> tuple[np.ndarray, np.ndarray]:
    """A line from (x0, y0) to (x1, y1) whose opacity rises from LINK_ALPHA[0] to LINK_ALPHA[1].

    LINK_STEPS nested segments, segment j from the j-th step point to (x1, y1), each with the
    alpha that makes the composited opacity on step j the ramp's value.  Every segment ends at
    the dark end, so the line has no seams between abutting pieces (the butt-joint gaps PDF
    viewers show for a chain of short segments); the opacity rises in small steps.
    """
    u = np.arange(LINK_STEPS) / LINK_STEPS
    xs, ys = x0 + (x1 - x0) * u, y0 + (y1 - y0) * u
    seg = np.stack(
        [np.column_stack([xs, ys]), np.tile([x1, y1], (LINK_STEPS, 1))], axis=1
    )
    target = LINK_ALPHA[0] + (LINK_ALPHA[1] - LINK_ALPHA[0]) * (u + 0.5 / LINK_STEPS)
    left = np.concatenate([[1.0], 1.0 - target[:-1]])  # transmittance before segment j
    rgba = np.tile((*to_rgb(LINK), 0.0), (LINK_STEPS, 1))
    rgba[:, 3] = 1.0 - (1.0 - target) / left
    return seg, rgba


class _PairHandle:
    """Legend key: open marker, fading line, filled marker."""


class _PairHandler(HandlerBase):
    def create_artists(
        self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans
    ):
        y = height / 2.0 - ydescent
        x0, x1 = -xdescent + 2.5, -xdescent + width - 2.5
        seg, rgba = _fade_segments(x0, y, x1, y)
        lc = LineCollection(
            seg, colors=rgba, linewidths=1.1, capstyle="butt", transform=trans
        )
        a = Line2D(
            [x0],
            [y],
            marker="o",
            ls="",
            ms=4.6,
            mfc="white",
            mec=MUTED,
            mew=0.9,
            transform=trans,
        )
        b = Line2D(
            [x1],
            [y],
            marker="o",
            ls="",
            ms=4.6,
            mfc=MUTED,
            mec="white",
            mew=0.5,
            transform=trans,
        )
        return [lc, a, b]


def _domain(v: np.ndarray, pad: float = 0.06) -> tuple[float, float]:
    lo, hi = float(np.min(v)), float(np.max(v))
    d = (hi - lo) * pad if hi > lo else 0.5
    return lo - d, hi + d


def figure(p: pd.DataFrame, ols: pd.Series) -> None:
    _style()
    fig, ax = plt.subplots(figsize=(6.2, 3.9))
    xs = np.concatenate([p["q_base"], p["q_all"], [ols["qlike_raw"]]])
    ys = np.concatenate([p["sh_base"], p["sh_all"], [ols["Sharpe_mid"]]])
    xd, yd = _domain(xs), _domain(ys)
    ax.set_xlim(*xd)
    ax.set_ylim(*yd)
    # lower QLIKE and higher Sharpe than the HAR + calendar OLS: left of it and above it
    ax.add_patch(
        Rectangle(
            (xd[0], ols["Sharpe_mid"]),
            ols["qlike_raw"] - xd[0],
            yd[1] - ols["Sharpe_mid"],
            facecolor=INK,
            alpha=0.06,
            edgecolor="none",
            zorder=0,
        )
    )
    ax.grid(True, color=GRID, linewidth=0.6, zorder=0.5)
    ax.set_axisbelow(True)
    ax.tick_params(length=2.5, width=0.6, color=FRAME)
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
    ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=6, steps=[1, 2, 5, 10]))
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
    ax.set_xlabel("QLIKE (lower is better)")
    ax.set_ylabel("Sharpe ratio, mid, annualised (higher is better)")

    segs, cols = [], []
    for r in p.itertuples():
        s, c = _fade_segments(r.q_base, r.sh_base, r.q_all, r.sh_all)
        segs.append(s)
        cols.append(c)
    ax.add_collection(
        LineCollection(
            np.concatenate(segs),
            colors=np.concatenate(cols),
            linewidths=1.0,
            capstyle="butt",
            zorder=2,
        )
    )
    for pooled, marker, size in ((False, "o", MS_CIRCLE), (True, "s", MS_SQUARE)):
        sub = p[p["pooled"] == pooled]
        if not len(sub):
            continue
        col = sub["est"].map(EST_COLOR).to_list()
        ax.scatter(
            sub["q_base"],
            sub["sh_base"],
            s=size,
            marker=marker,
            facecolors="white",
            edgecolors=col,
            linewidths=0.9,
            zorder=3,
        )
        ax.scatter(
            sub["q_all"],
            sub["sh_all"],
            s=size,
            marker=marker,
            facecolors=col,
            edgecolors="white",
            linewidths=0.5,
            zorder=4,
        )
    # the HAR + calendar OLS: open ink circle (it is on HAR + calendar), ringed and labelled
    ox, oy = float(ols["qlike_raw"]), float(ols["Sharpe_mid"])
    ax.scatter(
        [ox],
        [oy],
        s=MS_CIRCLE,
        marker="o",
        facecolors="white",
        edgecolors=INK,
        linewidths=1.0,
        zorder=5,
    )
    ax.scatter(
        [ox],
        [oy],
        s=120,
        marker="o",
        facecolors="none",
        edgecolors=INK,
        linewidths=0.8,
        zorder=5,
    )
    # the label sits in the empty low-QLIKE / low-Sharpe corner, with a leader to the ring
    fy = (oy - yd[0]) / (yd[1] - yd[0])
    ax.annotate(
        "HAR + cal. OLS",
        (ox, oy),
        xytext=(0.015, fy - 0.16),
        textcoords="axes fraction",
        arrowprops={
            "arrowstyle": "-",
            "color": INK,
            "lw": 0.6,
            "shrinkA": 1.5,
            "shrinkB": 6.5,
        },
        ha="left",
        va="top",
        fontsize=7.5,
        fontweight="bold",
        color=INK,
        zorder=6,
        path_effects=[pe.withStroke(linewidth=2.5, foreground="white")],
    )

    seen = set(p["est"]) | {"OLS"}
    est_handles = [
        Line2D([], [], marker="o", ls="", ms=5, mfc=c, mec=c, label=n)
        for n, c in EST_COLOR.items()
        if n in seen
    ]
    enc_handles = [
        _PairHandle(),
        Line2D([], [], marker="o", ls="", ms=5, mfc=MUTED, mec=MUTED),
        Line2D([], [], marker="s", ls="", ms=4.6, mfc=MUTED, mec=MUTED),
        Patch(facecolor=INK, alpha=0.06, edgecolor="none"),
    ]
    enc_labels = [
        "HAR + cal. ind. features\n(open) to all features\n(filled), same model",
        "per-bar fit (16:00 bar)",
        "pooled fit (all 48 bars)",
        "lower QLIKE and higher\nSharpe than HAR + cal. OLS",
    ]
    if not p["pooled"].any():
        del enc_handles[2], enc_labels[2]
    # legends outside the axes (right), so later rows cannot land under them
    fig.subplots_adjust(left=0.10, right=0.70, top=0.98, bottom=0.12)
    leg1 = fig.legend(
        handles=est_handles,
        loc="upper left",
        bbox_to_anchor=(0.715, 0.985),
        ncol=1,
        frameon=False,
        handletextpad=0.3,
        borderaxespad=0.0,
        labelspacing=0.45,
    )
    fig.add_artist(leg1)
    fig.legend(
        handles=enc_handles,
        labels=enc_labels,
        handler_map={_PairHandle: _PairHandler()},
        loc="lower left",
        bbox_to_anchor=(0.715, 0.12),
        ncol=1,
        frameon=False,
        handlelength=2.6,
        handletextpad=0.4,
        borderaxespad=0.0,
        labelspacing=0.7,
    )
    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        FIG,
        bbox_inches="tight",
        pad_inches=0.03,
        metadata={"CreationDate": None, "ModDate": None},
    )
    fig.savefig(PNG, dpi=200, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print(f"wrote {FIG.relative_to(ROOT)} and {PNG.relative_to(ROOT)}")


# --------------------------------------------------------------------------- figure 2: changes
def _signed(v: float, d: int) -> str:
    """A signed number for figure text, with a true minus sign; zero is plain 0."""
    if not round(float(v), 9):
        return "0"
    return f"{float(v):+.{d}f}".replace("-", "−")


def describe(key: str, est: str, pooled: bool) -> str:
    """A short name of a model for the change figure's key (input-set token dropped)."""
    if pooled:
        return f"pooled {est}"
    if key.startswith("subpt_"):
        return f"per-bar {est}, pre-tuned"
    m = re.match(r"subtree_optuna(q?)_tp(\d+)(?:_k(\d+))?_", key)
    if m:
        every = "every session" if m.group(2) == "1" else f"every {m.group(2)} sessions"
        pick = ", QLIKE pick" if m.group(1) else ""
        return f"{est}, Optuna retuned {every}, best of {m.group(3) or 50}{pick}"
    daily = ", refit daily" if "_daily_" in key else ""
    m = re.match(r"subtree_tuned(q?)_", key)
    if m:
        return f"{est}, random search{', QLIKE pick' if m.group(1) else ''}{daily}"
    if key.startswith("subtree_"):
        return f"{est}, untuned{daily}"
    return est


def figure_change(p: pd.DataFrame) -> None:
    """One point per model: x = QLIKE change (%), y = Sharpe (mid) change, all features minus HAR + cal."""
    _style()
    fig, ax = plt.subplots(figsize=(6.2, 4.3))
    x, y = p["dq_pct"].to_numpy(), p["dsh"].to_numpy()
    xd = _domain(np.append(x, 0.0), 0.06)
    # extra room above and below the points for the quadrant counts in the corners
    lo, hi = min(float(y.min()), 0.0), max(float(y.max()), 0.0)
    yd = (lo - 0.24 * (hi - lo), hi + 0.17 * (hi - lo))
    ax.set_xlim(*xd)
    ax.set_ylim(*yd)
    # lower QLIKE and higher Sharpe with all features: left of zero and above zero
    ax.add_patch(
        Rectangle(
            (xd[0], 0.0),
            -xd[0],
            yd[1],
            facecolor=INK,
            alpha=0.06,
            edgecolor="none",
            zorder=0,
        )
    )
    ax.axvline(0.0, color=MUTED, linewidth=0.8, zorder=1)
    ax.axhline(0.0, color=MUTED, linewidth=0.8, zorder=1)
    ax.grid(True, color=GRID, linewidth=0.6, zorder=0.5)
    ax.set_axisbelow(True)
    ax.tick_params(length=2.5, width=0.6, color=FRAME)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=8, steps=[1, 2, 5, 10]))
    ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=8, steps=[1, 2, 5, 10]))
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: _signed(v, 0)))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: _signed(v, 1)))
    ax.set_xlabel("Change in QLIKE, % of HAR + cal. (lower is better)")
    ax.set_ylabel("Change in Sharpe, mid (higher is better)")

    dm_sig = p["dm_p"] < SIG
    t_sig = p["lw_p"] < SIG  # 2026-10-21 (user, decision #2): Ledoit-Wolf. Was: p["t_p"] < SIG
    for pooled, marker, size in ((False, "o", MS_CIRCLE), (True, "s", MS_SQUARE)):
        sub = p[p["pooled"] == pooled]
        if len(sub):
            ax.scatter(
                sub["dq_pct"],
                sub["dsh"],
                s=size,
                marker=marker,
                facecolors=sub["est"].map(EST_COLOR).to_list(),
                edgecolors="white",
                linewidths=0.5,
                zorder=4,
            )
    # DM-significant QLIKE change: an ink ring; HAC-significant Sharpe change: an ink triangle above
    ax.scatter(
        p.loc[dm_sig, "dq_pct"],
        p.loc[dm_sig, "dsh"],
        s=95,
        marker="o",
        facecolors="none",
        edgecolors=INK,
        linewidths=0.9,
        zorder=5,
    )
    up = mtransforms.offset_copy(ax.transData, fig=fig, x=0.0, y=6.5, units="points")
    ax.scatter(
        p.loc[t_sig, "dq_pct"],
        p.loc[t_sig, "dsh"],
        s=16,
        marker="^",
        facecolors=INK,
        edgecolors="white",
        linewidths=0.4,
        zorder=6,
        transform=up,
    )
    # the significant points carry a number; the key below the axes names them
    sig = p[dm_sig | t_sig].sort_values("dq_pct")
    key_lines = []
    for i, r in enumerate(sig.itertuples(), start=1):
        ax.annotate(
            str(i),
            (r.dq_pct, r.dsh),
            xytext=(6.5, -1.0),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=7.5,
            fontweight="bold",
            color=INK,
            zorder=7,
            path_effects=[pe.withStroke(linewidth=2.2, foreground="white")],
        )
        what = []
        if r.dm_p < SIG:
            what.append(f"DM {_signed(r.dm, 2)}")
        # 2026-10-21 (user, decision #2): Ledoit-Wolf. Was: if r.t_p < SIG: what.append(f"HAC t {_signed(r.t_hac, 2)}")
        if r.lw_p < SIG:
            what.append(f"Ledoit-Wolf p {r.lw_p:.3f}")
        key_lines.append(
            f"{i}  {describe(r.key_base, r.est, r.pooled)} ({', '.join(what)})"
        )
    # the count of models in each quadrant, in its corner
    quad = quadrants(p)
    halo = [pe.withStroke(linewidth=2.5, foreground="white")]
    corners = (
        (
            0.015,
            0.985,
            "left",
            "top",
            quad["better_both"],
            "better on both",
        ),
        (0.985, 0.985, "right", "top", quad["sh_only"], "higher Sharpe only"),
        (0.015, 0.015, "left", "bottom", quad["q_only"], "lower QLIKE only"),
        (
            0.985,
            0.015,
            "right",
            "bottom",
            quad["worse_both"],
            "worse on both",
        ),
    )
    for fx, fy, ha, va, n, corner_label in corners:
        ax.text(
            fx,
            fy,
            f"{n} {'model' if n == 1 else 'models'}\n{corner_label}",
            transform=ax.transAxes,
            ha=ha,
            va=va,
            fontsize=7.5,
            color=INK,
            zorder=8,
            path_effects=halo,
        )

    est_handles = [
        Line2D([], [], marker="o", ls="", ms=5, mfc=c, mec=c, label=n)
        for n, c in EST_COLOR.items()
        if n in set(p["est"])
    ]
    enc = [
        (
            Line2D([], [], marker="o", ls="", ms=5, mfc=MUTED, mec=MUTED),
            "per-bar fit (16:00 bar)",
        ),
        (
            Line2D([], [], marker="s", ls="", ms=4.6, mfc=MUTED, mec=MUTED),
            "pooled fit (all 48 bars)",
        ),
        (
            Line2D([], [], marker="o", ls="", ms=8.5, mfc="none", mec=INK, mew=0.9),
            f"QLIKE change DM-\nsignificant at {100 * SIG:.0f} %",
        ),
        (
            Line2D([], [], marker="^", ls="", ms=4.5, mfc=INK, mec=INK),
            f"Sharpe change significant\nat {100 * SIG:.0f} % (Ledoit-Wolf)",  # 2026-10-21: was "HAC-significant"
        ),
        (
            Patch(facecolor=INK, alpha=0.06, edgecolor="none"),
            "lower QLIKE and\nhigher Sharpe",
        ),
    ]
    if not p["pooled"].any():
        del enc[1]
    fig.subplots_adjust(
        left=0.10, right=0.70, top=0.98, bottom=0.11 + 0.034 * len(key_lines)
    )
    leg1 = fig.legend(
        handles=est_handles,
        loc="upper left",
        bbox_to_anchor=(0.715, 0.985),
        ncol=1,
        frameon=False,
        handletextpad=0.3,
        borderaxespad=0.0,
        labelspacing=0.45,
    )
    fig.add_artist(leg1)
    fig.legend(
        handles=[h for h, _ in enc],
        labels=[t for _, t in enc],
        loc="lower left",
        bbox_to_anchor=(0.715, fig.subplotpars.bottom),
        ncol=1,
        frameon=False,
        handletextpad=0.4,
        borderaxespad=0.0,
        labelspacing=0.7,
    )
    if key_lines:
        fig.text(
            0.02,
            0.01,
            "\n".join(key_lines),
            ha="left",
            va="bottom",
            fontsize=7.5,
            color=INK,
            linespacing=1.35,
        )
    fig.savefig(
        FIG2,
        bbox_inches="tight",
        pad_inches=0.03,
        metadata={"CreationDate": None, "ModDate": None},
    )
    fig.savefig(PNG2, dpi=200, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print(f"wrote {FIG2.relative_to(ROOT)} and {PNG2.relative_to(ROOT)}")


# --------------------------------------------------------------------------- main
def main() -> None:
    p, ols, _ = load()
    show = p[["key_base", "est", "pooled", "dq", "dm", "dm_p", "dsh", "t_hac", "des"]]
    with pd.option_context("display.width", 200, "display.max_rows", 200):
        print(show.to_string(index=False, float_format=lambda v: f"{v:+.4f}"))
    nums = numbers(p, ols)
    for name, val, comment in nums:
        print(f"  \\{name} = {val}   ({comment})")
    write_tex(nums, p)
    figure(p, ols)
    figure_change(p)


if __name__ == "__main__":
    main()
