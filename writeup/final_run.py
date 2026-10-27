"""The switch between the printed pooled runs and the FINAL pooled rerun (2026-10-23).

One module, read by the three pooled loaders (pooled_linear_13.py, pooled_trees_13.py, pooled_mingru_13.py) and by
the generators that print a pooled number or the pooled design (make_rehaul_oos_rerun.py, make_rehaul_economics.py,
make_rehaul_results.py, make_rehaul_oos_table.py, make_rehaul_data_methods.py).  With the switch off (the default)
every one of them reads the printed runs and writes what it wrote before, byte for byte.

SWITCH   REHAUL_FINAL_RUN=1 (environment): the final design of the pooled 13-bar intraday models (decided 2026-10-23,
         writeup/rehaul/99_author_notes.tex "Window and sample"): first evaluation day 2 January 2014 (Table 1's
         every-bar span 2014-01-02 .. 2024-04-30; the option-trading sample of Section 7 is unchanged), penalties and
         configurations tuned on the last pre-sample fold (fold 3, validation year 2013), a rolling window of
         REHAUL_FINAL_WINDOW trading days (3,000 = 39,000 bars, the main window; 3,500 the robustness window), refit at
         every bar, the final specification (LATE_START_WARMUP, UNOBSERVED_NEUTRAL, EXCLUDE_FOMC_RELEASE,
         WALK_KKT_RESEED, STANDARDIZE_PENALTY with the availability screen).

INPUTS   (each overridable by its environment variable, e.g. to point a loader at a stand-in)
  REHAUL_LINEAR_RUN   results/window_sweep_2014start_2026-10-22/w<W>         ridge, lasso, elastic net, OLS
  REHAUL_TREES_RUN    results/pooled_trees_final_2014start_w<W>              LightGBM, XGBoost (model keys lgbm, xgb)
  REHAUL_RF_RUN       results/pooled_rf_skx_2014start_w<W>                   the random forest on oneDAL
  REHAUL_RF_KEY       rf_skx                                                 its model key in that forecasts.parquet
  REHAUL_MINGRU_RUN   results/pooled_mingru_final_2014start_w<W>             the pooled MinGRU (not printed: NN parked)

OUTPUTS  REHAUL_GEN_DIR (default writeup/generated), REHAUL_FIG_DIR (default writeup/figures), REHAUL_RES_DIR (the CSV
         files a generator writes under results/; default: each generator's own directory, switched to a new
         directory under the final design so no printed run's CSV is overwritten).

STAND-INS  writeup/tests/final_run_standins.py builds labelled stand-ins (the printed runs cut to the final bars);
         guard_standin() refuses them unless REHAUL_ALLOW_STANDIN=1 and GEN / FIG are outside writeup/.

PENDING  (2026-10-23, owner: "where forest or MinGRU numbers are not finished, cells read pending") Under the switch, a
         final run with no forecasts.parquet is PENDING: the random forest (RF_RUN; RF_PENDING, PENDING_TREES = ("rf",))
         and the pooled MinGRU (MINGRU_RUN; MINGRU_PENDING).  A pending arm is never read from a printed run and never
         stops a generator: its loader raises Pending on a read, the generators skip it and print PENDING_TEX
         (\\rhPending{}, defined by PENDING_DEF at the top of every generated file that uses it) in each of its table
         cells and macros, mark each such table row with ROW_MARK (writeup/tests/check_pending.py), leave it out of the
         figures (or label it "pending"), and compute every count, aggregate or comparison that spans arms over the
         FINISHED arms only.  REHAUL_REQUIRE_ALL=1 restores the hard failure (no arm is pending; a missing run raises in
         its loader as before).  With the switch off nothing is pending.
         The pooled forecasts the paper counts (its scope on 2026-10-23): the six pooled linear models (ridge, lasso,
         elastic net on the HAR benchmark and on the full set; POOL_LINEAR x POOL_SETS) and the six pooled trees
         (LightGBM, XGBoost, the random forest; POOL_TREES x POOL_SETS), POOL_N_ALL = 12.  The pooled OLS is Table 1's
         reference row, not one of them.  The pooled MinGRU is parked (NN content): it is outside the 12, so it is
         neither counted nor named as pending in the paper's text; MINGRU_PENDING only gates its loader.
         Companion macros (make_rehaul_results.py, rehaul_results_numbers.tex): \\rhrPoolN and every other count
         over the pooled forecasts / classes count the finished ones; \\rhrPoolNDone (= \\rhrPoolN), \\rhrPoolNAll
         (12), \\rhrPoolNPending, \\rhrPoolPending (the pending arms as text, "the random forest"; empty when none),
         \\rhrPoolPendingParen (" (the random forest: pending)"; empty when none); the generators' own denominators
         get an ...All twin (e.g. \\rhrAfPoolPairsAll, \\rhorAfPairsAll).

REGENERATE (once the final runs land; from the repository root, Git Bash; PowerShell: $env:REHAUL_FINAL_RUN = "1")
  export REHAUL_FINAL_RUN=1
  python writeup/final_run.py                      # every input "ok" or "pending"
  python writeup/tests/check_final_loaders.py      # the loaders' gates and the cross-run bar / RV checks
  python writeup/make_rehaul_economics.py          # Section 7 (\\rhe..., E1-E4, figures)
  python writeup/make_rehaul_results.py            # Section 6 (\\rhr..., T1-T3, figures)
  python writeup/make_rehaul_oos_table.py          # Table 2 and \\rhoos... (reads rehaul_results_numbers.tex)
  python writeup/make_rehaul_oos_rerun.py          # Table 1 (--span 2014) and \\rhor...
  python writeup/make_rehaul_data_methods.py       # samples / tuning tables and \\rhdm...
  python writeup/tests/check_pending.py            # no generated file carries a number for a pending arm
  # the generators below do not read the switch (run them as they are; 2026-10-23, coordinator)
  python writeup/make_rehaul_disclosures.py
  python writeup/make_rehaul_pooled_extras.py
  python writeup/make_rehaul_robust_window.py
  python writeup/make_rehaul_fi_pooled.py          # Section 8 (pooled variable importance)
  python writeup/make_rehaul_fip_lc.py             # Section 8 (the trees across windows, 1,000-3,500 days; 2026-10-23)
  cd writeup && pdflatex paper_rehaul && bibtex paper_rehaul && pdflatex paper_rehaul && pdflatex paper_rehaul
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
WRITEUP = REPO / "writeup"
RESULTS = REPO / "results"


def _on(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _path(name: str, default: Path) -> Path:
    v = os.environ.get(name, "").strip()
    return Path(v).resolve() if v else default


FINAL: bool = _on("REHAUL_FINAL_RUN")

# ---------------------------------------------------------------------------------------------- the final design
WINDOW_DAYS = int(os.environ.get("REHAUL_FINAL_WINDOW", "3000"))
assert WINDOW_DAYS in (3000, 3500), (
    f"REHAUL_FINAL_WINDOW={WINDOW_DAYS}: the final design's windows are 3000 (main) and 3500 (robustness)"
)
BARS_DAY = 13  # 10:00 .. 16:00 ET bar ends
WINDOW_BARS = WINDOW_DAYS * BARS_DAY  # 39,000 at W = 3,000
FIRST_DAY = pd.Timestamp("2014-01-02")  # first evaluation day (--first-day of every final walk)
LAST_DAY = pd.Timestamp("2024-04-30")
# the linear walk starts at the 250-solve boundary at or before the first 2014-01-02 row (as the printed run's
# 2011-12-28 14:00 before 2012-01-03): 1 + 13 + 13 bars of 27, 30 and 31 December 2013
LINEAR_WALK_START = pd.Timestamp("2013-12-27 16:00")
N_LINEAR_ROWS = 33_997  # rows of each linear arm (results/window_sweep_2014start_2026-10-22/w3000, w3500)
N_BARS = 33_970  # bars of each tree / forest / MinGRU arm, 2014-01-02 10:00 .. 2024-04-30 16:00 (RULE_rf_skx.txt)
TUNE_FOLDS = (3,)  # the pre-sample folds the penalties and configurations are chosen on (--first-fold 3)
VAL_YEAR = 2013  # fold 3's validation year (2013-01-02 15:30 .. 2013-12-31 16:00)
# Table 1's subperiods (\rhorSubA... / \rhorSubB...): the first evaluation day to the end of 2019, and 2020-2024
SUB_A = (FIRST_DAY, pd.Timestamp("2019-12-31"))
SUB_B = (pd.Timestamp("2020-01-03"), LAST_DAY)

# ---------------------------------------------------------------------------------------------- the FOMC-patched runs
# 2026-10-26 (author: the FOMC-patched design adopted; results/fomcfix_2026-10-26/INTEGRATION.md): REHAUL_FOMCFIX=1
# (with REHAUL_FINAL_RUN=1, W = 3,000 only) points the three final inputs at the runs on the patched design, cell B of
# results/fomcfix_2026-10-26/RULE.txt (FOMC_SCHEDULE_2020_FIX and FOMC_PROXIMITY_NO_WINSOR on, holiday sessions kept):
# FOMCFIX_RUNS below.  It sets REHAUL_LINEAR_RUN / REHAUL_TREES_RUN / REHAUL_RF_RUN where they are unset, so every reader
# of those variables (and make_rehaul_robust_window.py's path check) sees the patched runs.  Off: as before.
# (2026-10-26: on by default under REHAUL_FINAL_RUN=1 at W = 3,000; see FOMCFIX below. Was: "Off (default): as before.")
# 2026-10-26 (coordinator, D13 regeneration): the patched design is the DEFAULT under REHAUL_FINAL_RUN=1 at W = 3,000,
# so a later regeneration cannot silently revert to the pre-patch runs; REHAUL_FOMCFIX=0 (or false / no / off) is the
# explicit opt-out.  At REHAUL_FINAL_WINDOW=3500 there is no patched run: the default stays off there, and an explicit
# REHAUL_FOMCFIX=1 still stops at the assert below.  Was: FOMCFIX: bool = _on("REHAUL_FOMCFIX")
FOMCFIX_OPTOUT: bool = os.environ.get("REHAUL_FOMCFIX", "").strip().lower() in {"0", "false", "no", "off"}
FOMCFIX: bool = _on("REHAUL_FOMCFIX") or (FINAL and not FOMCFIX_OPTOUT and WINDOW_DAYS == 3000)
FOMCFIX_RUNS = {
    "REHAUL_LINEAR_RUN": RESULTS / "window_sweep_2014start_fomcfix" / "w3000",
    "REHAUL_TREES_RUN": RESULTS / "pooled_trees_final_2014start_fomcfix_w3000",
    "REHAUL_RF_RUN": RESULTS / "pooled_rf_skx_2014start_fomcfix_w3000",
}
if FOMCFIX:
    assert FINAL, "REHAUL_FOMCFIX=1 needs REHAUL_FINAL_RUN=1"
    assert WINDOW_DAYS == 3000, (
        f"REHAUL_FOMCFIX=1: the patched runs exist for W = 3,000 only (REHAUL_FINAL_WINDOW={WINDOW_DAYS})"
    )
    for _k, _v in FOMCFIX_RUNS.items():
        os.environ.setdefault(_k, str(_v))

# ---------------------------------------------------------------------------------------------- inputs
LINEAR_RUN = _path(
    "REHAUL_LINEAR_RUN",
    RESULTS / "window_sweep_2014start_2026-10-22" / f"w{WINDOW_DAYS}",
)
TREES_RUN = _path("REHAUL_TREES_RUN", RESULTS / f"pooled_trees_final_2014start_w{WINDOW_DAYS}")
RF_RUN = _path("REHAUL_RF_RUN", RESULTS / f"pooled_rf_skx_2014start_w{WINDOW_DAYS}")
RF_KEY = os.environ.get("REHAUL_RF_KEY", "rf_skx").strip() or "rf_skx"
MINGRU_RUN = _path("REHAUL_MINGRU_RUN", RESULTS / f"pooled_mingru_final_2014start_w{WINDOW_DAYS}")

# ---------------------------------------------------------------------------------------------- outputs
GEN = _path("REHAUL_GEN_DIR", WRITEUP / "generated")
FIG = _path("REHAUL_FIG_DIR", WRITEUP / "figures")
_RES_OUT = os.environ.get("REHAUL_RES_DIR", "").strip()


def res_dir(default: Path, final: Path | None = None) -> Path:
    """Where a generator writes its CSV files: REHAUL_RES_DIR/<name> when set, else `final` under the final design
    (a new directory, so no CSV of a printed run is overwritten) and `default` otherwise."""
    d = final if (FINAL and final is not None) else default
    return Path(_RES_OUT).resolve() / d.name if _RES_OUT else d


def rel(p: Path) -> str:
    """p relative to the repository when it is inside it (log lines), else the absolute path."""
    try:
        return str(Path(p).resolve().relative_to(REPO))
    except ValueError:
        return str(p)


def guard_standin(meta: dict, run: Path) -> None:
    """A stand-in run (forecasts.json "standin", written by writeup/tests/final_run_standins.py) is read only with
    REHAUL_ALLOW_STANDIN=1 and never into writeup/generated/, so no stand-in number can reach the paper's files."""
    if not meta.get("standin"):
        return
    if not _on("REHAUL_ALLOW_STANDIN"):
        raise SystemExit(f"{run} is a STAND-IN ({meta['standin']}); set REHAUL_ALLOW_STANDIN=1 for a test run")
    if GEN.resolve() == (WRITEUP / "generated").resolve() or FIG.resolve() == (WRITEUP / "figures").resolve():
        raise SystemExit(
            f"{run} is a STAND-IN: set REHAUL_GEN_DIR and REHAUL_FIG_DIR to a scratch directory (not writeup/)"
        )
    if str(run) not in _WARNED:
        _WARNED.add(str(run))
        print(f"WARNING: reading the STAND-IN {run} ({meta['standin']})")


_WARNED: set[str] = set()


# ---------------------------------------------------------------------------------------------- pending arms
# 2026-10-23 (owner): a final run without forecasts is PENDING (module docstring, PENDING); REHAUL_REQUIRE_ALL=1 makes
# every arm required again (a missing run then raises in its loader, as before).
REQUIRE_ALL: bool = _on("REHAUL_REQUIRE_ALL")


class Pending(LookupError):
    """Read of a model arm whose final run has no forecasts yet: the caller prints it as pending (never a printed
    run's number)."""


def _no_forecasts(run: Path) -> bool:
    return not (run / "forecasts.parquet").is_file()


RF_PENDING: bool = FINAL and not REQUIRE_ALL and _no_forecasts(RF_RUN)
MINGRU_PENDING: bool = FINAL and not REQUIRE_ALL and _no_forecasts(MINGRU_RUN)
PENDING_TREES: tuple[str, ...] = ("rf",) if RF_PENDING else ()  # pooled_trees_13 estimator keys

# LaTeX of a pending cell / macro value, its definition (top of every generated file that uses it), and the comment
# that ends each table row whose cells are pending (writeup/tests/check_pending.py checks those rows hold no number)
PENDING_TEX = r"\rhPending{}"
PENDING_DEF = r"\providecommand{\rhPending}{\textit{pending}}"
ROW_MARK = "% rhPending row"

# the pooled forecasts the paper counts (scope 2026-10-23; module docstring): 6 linear + 6 trees; the MinGRU is parked
POOL_LINEAR = ("enet", "lasso", "ridge")
POOL_TREES = ("lgbm", "xgb", "rf")
POOL_SETS = ("baseline", "all_features")
POOL_N_ALL = (len(POOL_LINEAR) + len(POOL_TREES)) * len(POOL_SETS)  # 12
# how the text names a pending arm (the pooled MinGRU is parked, so it is never named)
PENDING_NAME = {"rf": "the random forest"}


def pending(est: str) -> bool:
    """True when the pooled estimator `est` (a POOL_LINEAR / POOL_TREES key) has no final forecasts yet."""
    return est in PENDING_TREES


def pool_classes(done: bool = True) -> list[str]:
    """The pooled model classes (estimators) the paper counts, POOL_LINEAR then POOL_TREES; the finished ones only
    unless done=False."""
    return [e for e in POOL_LINEAR + POOL_TREES if not (done and pending(e))]


def pool_n(done: bool = True) -> int:
    """The pooled forecasts the paper counts (class x predictor set): the finished ones, or all (done=False)."""
    return len(pool_classes(done)) * len(POOL_SETS)


def pending_text() -> str:
    """The pending pooled classes as text ("the random forest"); empty when none is pending."""
    names = [PENDING_NAME[e] for e in PENDING_TREES]
    return " and ".join(names)


def cell(value: str, est: str) -> str:
    """`value` (a formatted table cell or macro value), or PENDING_TEX when the pooled estimator `est` is pending."""
    return PENDING_TEX if pending(est) else value


def tag() -> str:
    """One log line: which inputs the loaders read."""
    if not FINAL:
        return "pooled runs: PRINTED (REHAUL_FINAL_RUN off)"
    pend = [n for n, p in (("forest", RF_PENDING), ("MinGRU", MINGRU_PENDING)) if p]
    return (
        f"pooled runs: FINAL design{' (FOMC-patched)' if FOMCFIX else ' (pre-patch: REHAUL_FOMCFIX off)'}, "
        f"W = {WINDOW_DAYS:,} days, first day {FIRST_DAY.date()}; linear "
        f"{rel(LINEAR_RUN)}, trees {rel(TREES_RUN)}, forest {rel(RF_RUN)} [{RF_KEY}], MinGRU {rel(MINGRU_RUN)}"
        + (f"; PENDING: {', '.join(pend)}" if pend else "")
        + ("; REHAUL_REQUIRE_ALL" if REQUIRE_ALL else "")
    )


if __name__ == "__main__":
    print(tag())
    print(f"GEN {GEN}\nFIG {FIG}\nRES {_RES_OUT or '(generators own directories)'}")
    for name, p, pend in (
        ("linear", LINEAR_RUN, False),
        ("trees", TREES_RUN, False),
        ("forest", RF_RUN, RF_PENDING),
        ("MinGRU", MINGRU_RUN, MINGRU_PENDING),
    ):
        fc = p / "forecasts.parquet"
        state = "ok     " if fc.is_file() else ("pending" if pend else "MISSING")
        print(f"  {name:7s} {state} {fc}")
    if FINAL:
        print(
            f"pooled forecasts counted: {pool_n()} finished of {pool_n(False)}"
            + (f" ({pending_text()} pending)" if pending_text() else "")
            + "; the pooled MinGRU is parked (outside the count)"
        )
