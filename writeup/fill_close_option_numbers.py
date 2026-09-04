"""Resolve the close-option prose numbers from the deck's own artifacts (audit + fill).

Every value below is read from a file the deck wrote: the rule-table CSVs, the
paired-test / regression / sign-split / within-side-slope / recalibration /
fill-variant / cliff / fixed-fraction CSVs, the per-day parquets, and the
executed run log.  Nothing is retyped from memory.

Two modes:

  python fill_close_option_numbers.py          fill the @TOKEN@ slots in FILES
  python fill_close_option_numbers.py --dry    resolve and AUDIT, writing nothing

In audit mode the script also checks that every resolved value still appears
somewhere in FILES.  A value that has gone MISSING means the deck moved a number
and the prose did not follow, which is exactly the drift this script exists to
catch.  (Short values such as ``866`` can of course match coincidentally; the
check is a tripwire, not a proof.)

Re-runnable: after the deck re-executes, re-tokenize the sentence that moved and
run this again, and the paper follows.
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

# Eight forecast columns, in the library's MODEL_ORDER.  "blk2_inc" is the
# diagnostic row (the same ridge on the earlier panel); the four marked
# "earlier panel" in the manifest carry a dagger in the generated table.
TAGS = {
    "a0": "baseline (HAR + calendar OLS)",
    "blk2": "block-diagonal ridge",
    "blk2_inc": "block-diagonal ridge, without the FOMC columns",
    "lgbm": "LightGBM",
    "xgb": "XGBoost",
    "lasso_t": "lasso (causally tuned)",
    "lasso_f": "lasso (fixed 1e-4)",
    "enet": "elastic net (causally tuned)",
}
B2 = "block-diagonal ridge"
B2_INC = "block-diagonal ridge, without the FOMC columns"
ALWAYS_SHORT = "always short (no forecast)"

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
pair = pd.read_csv(RES / "paired_tests.csv")
wside = pd.read_csv(RES / "within_side_slope.csv").set_index("model")
variants = pd.read_csv(RES / "pnl_variants_blk2.csv")
cliff = pd.read_csv(RES / "forecast_shift_cliff.csv", index_col=0)
ff = {t: pd.read_csv(RES / ("fixedfrac_summary_%s.csv" % t), index_col=0) for t in TAGS}
D = {t: pd.read_parquet(RES / ("daily_%s.parquet" % t)) for t in TAGS}

n_days = int(asr["n"])
b = D["blk2"]
assert len(b) == n_days, (len(b), n_days)
assert list(ss.index) == list(TAGS.values()), list(ss.index)
assert list(wside.index) == list(TAGS.values()), list(wside.index)

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
V["B2_MEAN"] = "%.3f" % ss.loc[B2, "mean"]
V["B2_T"] = "%.2f" % ss.loc[B2, "t_mean"]
V["B2_SH"] = "%.2f" % ss.loc[B2, "Sharpe_ann"]

# claim: every one of the eight improves on the control
assert (ss["Sharpe_ann"] > asr["Sharpe_ann"]).all(), (
    "not every sign(s) row beats always short"
)
# claim: each portfolio's OWN t clears 1.96 for N of the eight -- named in the
# prose precisely so it is not mistaken for the paired statistic.
V["NOWNSIG"] = WORDS[int((ss["t_mean"].abs() >= 1.96).sum())]

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
    B2_INC: "the same ridge without the FOMC columns",
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

# --- paired tests --------------------------------------------------------
lev = pair[pair["comparison"] == "Sharpe ratio of the portfolio (level)"].set_index(
    "model"
)
vs_as = pair[pair["comparison"] == "sign(s) minus always short"].set_index("model")
vs_pan = pair[pair["comparison"] == "FOMC panel minus earlier panel, same ridge"].iloc[
    0
]
assert set(vs_as.index) == set(TAGS.values()), sorted(vs_as.index)
assert ALWAYS_SHORT in lev.index, sorted(lev.index)

V["B2_SH_SE"] = "%.2f" % lev.loc[B2, "sharpe_se"]
V["B2_SH_LO"] = "%.2f" % lev.loc[B2, "sharpe_lo"]
V["B2_SH_HI"] = "%.2f" % lev.loc[B2, "sharpe_hi"]

assert (vs_as["dSharpe"] > 0).all(), "the paired improvement is no longer unanimous"
V["PAIR_DS_LO"] = "%.2f" % vs_as["dSharpe"].min()
V["PAIR_DS_HI"] = "%.2f" % vs_as["dSharpe"].max()
V["PAIR_T_LO"] = "%.2f" % vs_as["t_plain"].min()
V["PAIR_T_HI"] = "%.2f" % vs_as["t_plain"].max()
V["PAIR_HACT_LO"] = "%.2f" % vs_as["t_hac"].min()
V["PAIR_HACT_HI"] = "%.2f" % vs_as["t_hac"].max()
V["B2_PAIR_DS"] = "%.2f" % vs_as.loc[B2, "dSharpe"]
V["B2_PAIR_T"] = "%.2f" % vs_as.loc[B2, "t_hac"]

incl = vs_as["pct_lo"] <= 0
V["NPCTZERO"] = WORDS[int(incl.sum())]
excl = vs_as[~incl]
assert len(excl) == 1 and excl.index[0] == "XGBoost", (
    "the set of intervals excluding zero is no longer {XGBoost}: %s" % list(excl.index)
)
V["XGB_PCT_LO"] = "%.3f" % float(excl["pct_lo"].iloc[0])
V["XGB_PCT_W"] = "%.2f" % float(excl["pct_hi"].iloc[0] - excl["pct_lo"].iloc[0])

# FOMC columns' contribution: the two ridge rows, paired
V["FOMC_DS"] = sig(vs_pan["dSharpe"], 2)
V["FOMC_T"] = "%.2f" % vs_pan["t_hac"]
V["FOMC_LO"] = m(vs_pan["pct_lo"], 2)
V["FOMC_HI"] = m(vs_pan["pct_hi"], 2)
assert vs_pan["pct_lo"] < 0 < vs_pan["pct_hi"], (
    "the FOMC-panel difference is now resolved; the prose says it is not"
)

# sign agreement with the baseline, over the seven non-baseline columns
agree = vs_as["same_position_as_baseline"].drop(TAGS["a0"])
V["AGREE_LO"] = "%d" % round(100 * agree.min())
V["AGREE_HI"] = "%d" % round(100 * agree.max())


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

# --- magnitude: within-side slopes --------------------------------------
assert (wside["b_sell"] < 0).all() and (wside["t_sell"] <= -1.96).all(), (
    "the sell-side slope is no longer negative and resolved for every column"
)
V["SELL_T_LO"] = "%.1f" % abs(wside["t_sell"].max())  # smallest |t|
V["SELL_T_HI"] = "%.1f" % abs(wside["t_sell"].min())  # largest |t|
V["SELL_T_B2"] = m(abs(wside.loc[B2, "t_sell"]), 1)
V["NBUYRES"] = WORDS[int((wside["t_buy"].abs() >= 1.96).sum())]

# --- crossed spread ------------------------------------------------------
pv = variants.set_index(["rule", "variant"])
V["XS_B2_MID"] = "%.2f" % pv.loc[("sign(s)", "mid premium R"), "Sharpe_ann"]
V["XS_B2"] = "%.2f" % pv.loc[("sign(s)", "crossed spread"), "Sharpe_ann"]
V["XS_B2_T"] = "%.2f" % pv.loc[("sign(s)", "crossed spread"), "t_mean"]
V["XS_AS_MID"] = "%.2f" % pv.loc[("always short", "mid premium R"), "Sharpe_ann"]
V["XS_AS"] = m(pv.loc[("always short", "crossed spread"), "Sharpe_ann"], 2)
assert (
    pv.loc[("sign(s)", "crossed spread"), "Sharpe_ann"]
    > 0
    > pv.loc[("always short", "crossed spread"), "Sharpe_ann"]
), "the crossed spread no longer separates the rule from the control"

# --- look-ahead cliff ----------------------------------------------------
V["CLIFF_M1"] = "%.2f" % cliff.loc[B2, "bar-1"]
V["CLIFF_0"] = "%.2f" % cliff.loc[B2, "bar+0"]
V["CLIFF_P1"] = "%.2f" % cliff.loc[B2, "bar+1"]
V["CLIFF_CEIL"] = "%.2f" % cliff.loc[B2, "realized"]
assert abs(cliff.loc[B2, "bar+0"] - ss.loc[B2, "Sharpe_ann"]) < 1e-9, (
    "the k=0 cliff column no longer equals the rule table"
)
stale_cols = ["bar-%d" % k for k in range(1, 12)]
enet = TAGS["enet"]
V["ENET_STALE"] = "%.2f" % cliff.loc[enet, stale_cols].max()
V["ENET_0"] = "%.2f" % cliff.loc[enet, "bar+0"]
beats = [
    k for k in TAGS.values() if cliff.loc[k, stale_cols].max() > cliff.loc[k, "bar+0"]
]
assert beats == [enet], (
    "the set of columns whose best stale shift beats their trade is no longer "
    "{elastic net}: %s" % beats
)

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
    "the median map no longer loses on all eight"
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
inside = logsearch(
    r"half sessions inside the scored range:\s*(\d+).*?; (\d+) of them carry a "
    r"16:00 forecast row",
    "half sessions inside the scored range",
)
V["NEARLYINSIDE_CAP"] = WORDS[int(inside.group(1))].capitalize()
V["NEARLYSCORED"] = WORDS[int(inside.group(2))]

# --- unscored tail, annualization, FOMC feed -----------------------------
V["NUNSCORED"] = logsearch(
    r"expiration days with no forecast row:\s*(\d+)", "unscored expiration days"
).group(1)
ann = logsearch(
    r"this frame trades ([0-9.]+) days a year, so a calendar-time convention would "
    r"multiply every Sharpe ratio by sqrt\([0-9./]+\) = ([0-9.]+)",
    "annualization convention",
)
V["TPY"] = ann.group(1)
V["ANNSCALE"] = ann.group(2)
V["NFOMCDEAD"] = logsearch(
    r"(\d+) of (?:the )?866 scored days.*?after (?:it|2023-11-01)"
    r"|FOMC feed ends 2023-11-01[^\n]*?(\d+) of 866",
    "scored days after the FOMC feed end",
).group(1)

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
    print("  %-16s %s" % (k, V[k]))

if "--dry" in sys.argv:
    print("\n=== audit: is every resolved value still in the prose? ===")
    blob = "\n".join(
        (WRITEUP / rel).read_bytes().decode("utf-8").replace("\r\n", "\n")
        for rel in FILES
    )
    missing = [k for k, v in V.items() if v and v not in blob]
    for k in sorted(missing):
        print("  MISSING  %-16s %s" % (k, V[k]))
    print(
        "  %d of %d resolved values found in the prose"
        % (len(V) - len(missing), len(V))
    )
    print("\n(dry run: no files written)")
    sys.exit(1 if missing else 0)

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
