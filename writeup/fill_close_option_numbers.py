"""Resolve the close-option prose numbers from the deck's own artifacts (audit + fill).

The @TOKEN@ slots this script filled on 2026-09-04 are consumed; re-running it
with --dry prints every resolved value so a reader can check each number in the
close-option sections against the deck's CSVs, parquets and run log.

Every value is read from a file the deck wrote: the rule-table CSVs, the
regression / sign-split / recalibration / fixed-fraction CSVs, the per-day
parquets, and the executed run log.  Re-runnable: after the deck re-executes,
run this again and the paper follows.

Usage:  python fill_close_option_numbers.py [--dry]
"""

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
RES = REPO / "results" / "atm_straddle_0dte_1530"
WRITEUP = REPO / "writeup"
LOG = REPO / "notebooks" / "atm_straddle_rv_iv.run.log"

TAGS = {
    "a0": "baseline (HAR + calendar OLS)",
    "blk2": "block-diagonal ridge",
    "lgbm": "LightGBM",
    "xgb": "XGBoost",
    "lasso_t": "lasso (causally tuned)",
    "lasso_f": "lasso (fixed 1e-4)",
    "enet": "elastic net (causally tuned)",
}
B2 = "block-diagonal ridge"

WORDS = {
    0: "none",
    1: "one",
    2: "two",
    3: "three",
    4: "four",
    5: "five",
    6: "six",
    7: "seven",
    8: "eight",
    9: "nine",
    10: "ten",
    11: "eleven",
    12: "twelve",
}


def sig(v, nd):
    """Signed fixed-decimal, explicit + for positives."""
    return ("%+." + str(nd) + "f") % v


def m(v, nd):
    """Math-mode number with a real minus sign."""
    s = ("%." + str(nd) + "f") % v
    return "$" + s.replace("-", "-") + "$"


log = LOG.read_text(encoding="utf-8", errors="replace")


def logsearch(pat, what):
    hit = re.search(pat, log)
    if not hit:
        print("!! run log: could not find %s  (pattern %r)" % (what, pat))
        sys.exit(1)
    return hit


# ------------------------------------------------------------------ tables
ss = pd.read_csv(RES / "rule_by_strategy_sign_s.csv", index_col=0)
asr = pd.read_csv(RES / "rule_by_strategy_always_short.csv", index_col=0).iloc[0]
reg = pd.read_csv(RES / "regression_R_on_signal.csv").set_index("model")
spl = pd.read_csv(RES / "sameday_sign_split.csv").set_index("model")
rec = pd.read_csv(RES / "recalibration_mean_vs_median.csv").set_index("model")
ff = {t: pd.read_csv(RES / ("fixedfrac_summary_%s.csv" % t), index_col=0) for t in TAGS}
D = {t: pd.read_parquet(RES / ("daily_%s.parquet" % t)) for t in TAGS}

n_days = int(asr["n"])
b = D["blk2"]
assert len(b) == n_days, (len(b), n_days)
assert list(ss.index) == list(TAGS.values()), list(ss.index)

V = {}
V["NDAYS"] = str(n_days)

# --- control -------------------------------------------------------------
V["AS_MEAN"] = "%.3f" % asr["mean"]
V["AS_STD"] = "%.3f" % asr["std"]
V["AS_T"] = "%.2f" % asr["t_mean"]
V["AS_SH"] = "%.2f" % asr["Sharpe_ann"]

# --- sign(s) panel -------------------------------------------------------
V["SS_MEAN_LO"] = "%.3f" % ss["mean"].min()
V["SS_MEAN_HI"] = "%.3f" % ss["mean"].max()
V["SS_T_LO"] = "%.2f" % ss["t_mean"].min()
V["SS_T_HI"] = "%.2f" % ss["t_mean"].max()
V["B2_MEAN"] = "%.3f" % ss.loc[B2, "mean"]
V["B2_T"] = "%.2f" % ss.loc[B2, "t_mean"]
V["B2_SH"] = "%.2f" % ss.loc[B2, "Sharpe_ann"]

# claim: six of the seven are significant at conventional levels
n_sig = int((ss["t_mean"].abs() >= 1.96).sum())
assert n_sig == 6, "PROSE says six of seven significant; table says %d" % n_sig
# claim: every one of the seven improves on the control
assert (ss["Sharpe_ann"] > asr["Sharpe_ann"]).all(), (
    "not every sign(s) row beats always short"
)

# claim: the ridge does not lead; name every row above it
above = ss[ss["Sharpe_ann"] > ss.loc[B2, "Sharpe_ann"]].sort_values(
    "Sharpe_ann", ascending=False
)
NICE = {
    "lasso (fixed 1e-4)": "the fixed lasso",
    "LightGBM": "LightGBM",
    "XGBoost": "XGBoost",
    "lasso (causally tuned)": "the causally tuned lasso",
    "elastic net (causally tuned)": "the causally tuned elastic net",
    "baseline (HAR + calendar OLS)": "the baseline",
}
if len(above) == 0:
    print("!! the ridge now LEADS the table - the prose must be rewritten by hand")
    sys.exit(1)
parts = ["%s (%.2f)" % (NICE[k], v) for k, v in above["Sharpe_ann"].items()]
phrase = parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + " and " + parts[-1]
V["ABOVE_B2"] = phrase + (" scores higher" if len(parts) == 1 else " all score higher")

# --- buy / sell shares ---------------------------------------------------
V["BUY_LO"] = "%d" % round(ss["pct_buy"].min())
V["BUY_HI"] = "%d" % round(ss["pct_buy"].max())
V["SELL_LO"] = "%d" % round(100 - ss["pct_buy"].max())
V["SELL_HI"] = "%d" % round(100 - ss["pct_buy"].min())
V["B2_SELL"] = "%d" % round(100 - ss.loc[B2, "pct_buy"])


def stats(x):
    x = np.asarray(x, float)
    n = len(x)
    mu = x.mean()
    sd = x.std(ddof=1)
    return n, np.sqrt(n) * mu / sd, (mu / sd) * np.sqrt(252)


# --- COVID-quarter split -------------------------------------------------
idx = pd.DatetimeIndex(b.index)
covid = idx <= pd.Timestamp("2020-03-31")
n_c, t_c, sh_c = stats(b["R_p"][~covid])
assert sh_c > ss.loc[B2, "Sharpe_ann"], (
    "dropping the COVID quarter no longer RAISES the Sharpe"
)
V["NCOVID"] = "%d" % int(covid.sum())
V["NPOSTCOVID"] = "%d" % n_c
V["COVID_SH"] = "%.2f" % sh_c
V["COVID_T"] = "%.2f" % t_c

# --- era split at 2022-05-16 --------------------------------------------
cut = pd.Timestamp("2022-05-16")
n1, t1, s1 = stats(b["R_p"][idx < cut])
n2, t2, s2 = stats(b["R_p"][idx >= cut])
assert s1 > 0 and s2 > 0, "the era split is no longer positive on both tapes"
assert abs(t1) < 1.96 <= abs(t2), (
    "the era split no longer earns its significance after the date"
)
V["NPRE"], V["PRE_SH"], V["PRE_T"] = "%d" % n1, "%.2f" % s1, "%.2f" % t1
V["NPOST"], V["POST_SH"], V["POST_T"] = "%d" % n2, "%.2f" % s2, "%.2f" % t2
tree_flip = all(
    stats(D[t]["R_p"][pd.DatetimeIndex(D[t].index) < cut])[2]
    > stats(D[t]["R_p"][pd.DatetimeIndex(D[t].index) >= cut])[2]
    for t in ("lgbm", "xgb")
)
assert tree_flip, "the two tree forecasts no longer split the other way at 2022-05-16"

# --- settlement pins -----------------------------------------------------
pin = b["R"] <= -1 + 1e-12
V["NPIN"] = "%d" % int(pin.sum())
V["PIN_PCT"] = "%.1f" % (100 * pin.mean())
V["AS_OFFPIN"] = sig((-b["R"])[~pin].mean(), 2)
rp = b["R_p"]
V["B2_PIN"] = sig(rp[pin].mean(), 2)
V["B2_OFFPIN"] = sig(rp[~pin].mean(), 2)
V["B2_PINSHARE"] = "%d" % round(100 * rp[pin].sum() / rp.sum())
assert abs((-b["R"])[pin].mean() - 1.0) < 1e-9, (
    "always short no longer collects +1 on pin days"
)

# --- correlation / effective count --------------------------------------
M = pd.DataFrame({TAGS[t]: D[t]["R_p"] for t in TAGS})
C = M.corr()
iu = np.triu_indices(len(TAGS), 1)
pw = C.values[iu]
ev = np.linalg.eigvalsh(C.values)
effn = (ev.sum() ** 2) / (ev**2).sum()
V["CORR_MEAN"] = "%.2f" % pw.mean()
V["CORR_LO"] = "%.2f" % pw.min()
V["CORR_HI"] = "%.2f" % pw.max()
V["EFFN"] = "%.1f" % effn
V["JAC_MEAN"] = "%.2f" % float(
    logsearch(r"mean pairwise Jaccard ([0-9.]+)", "mean pairwise Jaccard").group(1)
)

# --- sign split ----------------------------------------------------------
V["B2_RBUY"] = sig(spl.loc[B2, "mean_R|s>0"], 2)
V["B2_RSELL"] = sig(spl.loc[B2, "mean_R|s<=0"], 2)
V["B2_DIFF"] = "%.2f" % spl.loc[B2, "diff"]
V["B2_DIFF_T"] = "%.2f" % spl.loc[B2, "t_diff"]
V["DIFF_T_LO"] = "%.2f" % spl["t_diff"].min()
V["DIFF_T_HI"] = "%.2f" % spl["t_diff"].max()
assert (spl["diff"] > 0).all(), (
    "the sign split is no longer positive for every forecast"
)

# --- OLS slope -----------------------------------------------------------
assert (reg["b"] < 0).all(), "the OLS slope is no longer negative for every forecast"
V["SLOPE_T_LO"] = m(reg["t_b"].min(), 1)
V["SLOPE_T_HI"] = m(reg["t_b"].max(), 1)

# --- fixed fraction ------------------------------------------------------
f2 = ff["blk2"]
sgn = f2.loc["sign(s)"]
ash = f2.loc["always short"]
V["B2_W"] = "%.1f" % sgn["terminal"]
V["B2_G"] = sig(sgn["g_ann"], 2)
V["B2_DD"] = "%d" % round(100 * abs(sgn["maxDD_frac"]))
V["B2_WORST"] = "%d" % round(100 * (1 - sgn["worst_day_factor"]))
V["B2_RUIN"] = "%.2f" % sgn["ruin_bound_f"]
V["AS_RUIN"] = "%.2f" % ash["ruin_bound_f"]
V["AS_W"] = "%.2f" % ash["terminal"]
term = pd.Series({t: ff[t].loc["sign(s)", "terminal"] for t in TAGS})
assert (term > 1).all(), "not every sign(s) portfolio compounds positively"
V["W_LO"] = "%.1f" % term.min()
V["W_HI"] = "%.1f" % term.max()

yrs = logsearch(
    r"per-year annualized log-growth, block-diagonal ridge, sign\(s\):\n((?:\s*\d{4}\s+[-+][0-9.]+\n)+)",
    "per-year log-growth for sign(s)",
).group(1)
peryear = {int(a): float(v) for a, v in re.findall(r"(\d{4})\s+([-+][0-9.]+)", yrs)}
assert len(peryear) == 5, peryear
neg = {y: v for y, v in peryear.items() if v <= 0}
V["NPOSYEARS"] = WORDS[5 - len(neg)]
if not neg:
    V["YEAREXC"] = ""
    V["NPOSYEARS"] = "every one"
else:
    V["YEAREXC"] = (
        " (" + ", ".join("%d is $%.2f$" % (y, v) for y, v in sorted(neg.items())) + ")"
    )

# --- median map ----------------------------------------------------------
assert (rec["Sharpe_median"] < rec["Sharpe_mean"]).all(), (
    "the median map no longer loses on all seven"
)
assert (rec["percentile_interval"] == "includes zero").all(), (
    "a median-map interval now excludes zero"
)
V["MEDMAP_FROM"] = "%.2f" % rec.loc[B2, "Sharpe_mean"]
V["MEDMAP_TO"] = "%.2f" % rec.loc[B2, "Sharpe_median"]

# --- early-close sessions ------------------------------------------------
V["NEARLY"] = WORDS[
    int(
        logsearch(
            r"half-session days dropped \(15:30 row after the close\):\s*(\d+)",
            "early-close day count",
        ).group(1)
    )
].capitalize()
V["NEARLYSCORED"] = WORDS[
    int(
        logsearch(
            r"; (\d+) of them carry a 16:00 forecast row", "scored half sessions"
        ).group(1)
    )
]

# --- implied-volatility censoring ---------------------------------------
h = logsearch(
    r"legs inside the vendor bounds on ([\d,]+) days: model price / quoted midpoint[^,]*, "
    r"median ([0-9.]+), 5th-95th pct ([0-9.]+)-([0-9.]+)",
    "vendor-bound price ratio",
)
V["IVRATIO"], V["IVRATIO_LO"], V["IVRATIO_HI"] = h.group(2), h.group(3), h.group(4)
n_cens = int(
    logsearch(
        r"days with a picked leg on a censoring node:\s*(\d+)", "censored day count"
    ).group(1)
)
V["NCENS"] = WORDS[n_cens] if n_cens <= 12 else str(n_cens)

drop = logsearch(
    r"sign\(s\) with the \d+ censored-implied days dropped instead of re-inverted \(per model\):\n"
    r"\s*n_days\s+Sharpe_ann\s+t\n((?:.+\n)+?)---",
    "drop-censored table",
).group(1)
dropsh = {}
for line in drop.strip().splitlines():
    hit = re.match(r"\s*(.+?)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*$", line)
    if hit:
        dropsh[hit.group(1).strip()] = float(hit.group(3))
assert set(dropsh) == set(TAGS.values()), sorted(dropsh)
gap = max(abs(dropsh[k] - ss.loc[k, "Sharpe_ann"]) for k in dropsh)
V["IVDROP_MAX"] = "%.2f" % (np.ceil(gap * 100) / 100)

# ------------------------------------------------------------------ apply
FILES = [
    "sections/results_close_option.tex",
    "sections/methods_close_option.tex",
    "sections/introduction.tex",
    "sections/conclusion.tex",
    "rule_by_strategy_standalone.tex",
]

print("=== resolved values ===")
for k in sorted(V):
    print("  %-14s %s" % (k, V[k]))

if "--dry" in sys.argv:
    print("\n(dry run: no files written)")
    sys.exit(0)

used = set()
for rel in FILES:
    p = WRITEUP / rel
    raw = p.read_bytes()
    crlf = raw.count(b"\r\n") > 0
    txt = raw.decode("utf-8")
    if crlf:
        txt = txt.replace("\r\n", "\n")
    for k, v in V.items():
        tok = "@%s@" % k
        if tok in txt:
            used.add(k)
            txt = txt.replace(tok, v)
    left = re.findall(r"@[A-Z0-9_]+@", txt)
    if left:
        print("!! unresolved tokens in %s: %s" % (rel, sorted(set(left))))
        sys.exit(1)
    if crlf:
        txt = txt.replace("\n", "\r\n")
    p.write_bytes(txt.encode("utf-8"))
    print("wrote %s (%s)" % (rel, "CRLF" if crlf else "LF"))

unused = set(V) - used
if unused:
    print("note: values computed but not placed: %s" % sorted(unused))
print("FILL DONE")
