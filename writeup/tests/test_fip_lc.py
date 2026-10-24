"""Tests of writeup/make_rehaul_fip_lc.py, Section 8's tree ensembles across training windows (2026-10-23).

Synthetic: the extra gain's Newey-West t is the paper's DM statistic before the HLN correction (one variance estimator
for both), and the text phrases.  On the generated files (skipped when the generator has not been run): every \\rhlc
macro the paper uses is defined, no \\rhlc name is defined by another number file of the preamble, and the self-check
reproduced every committed cross-window value.

Run:  python -m pytest writeup/tests/test_fip_lc.py   (or: python writeup/tests/test_fip_lc.py)
"""

from __future__ import annotations

import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
WRITEUP = ROOT / "writeup"
for _p in (ROOT, ROOT / "notebooks", WRITEUP, WRITEUP / "tests"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import check_pending as cp  # noqa: E402  (macros(): the definitions of a number file)
import make_rehaul_fip_lc as lc  # noqa: E402

from src.evaluation.diebold_mariano import dm_test  # noqa: E402

GEN = WRITEUP / "generated"
NUMBERS = GEN / "rehaul_fip_lc_numbers.tex"
TABLE = GEN / "rehaul_fip_lc_table.tex"
SELFCHECK = ROOT / "results" / "rehaul_fip_lc_final" / "selfcheck.csv"
_USE = re.compile(r"\\(rhlc[A-Za-z]+)")


def _ar1(T: int, phi: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    e = rng.standard_normal(T)
    x = np.empty(T)
    x[0] = e[0]
    for t in range(1, T):
        x[t] = phi * x[t - 1] + e[t]
    return 0.05 + x


def test_nw_is_dm_before_hln():
    for T, phi, seed in ((2579, 0.3, 0), (2579, -0.2, 1), (500, 0.6, 2)):
        x = _ar1(T, phi, seed)
        r = lc.nw(x)
        d = dm_test(x, np.zeros(T))
        assert r["lag"] == d["hac_lag"] and r["T"] == d["T"] == T
        assert math.isclose(r["mean"], d["mean_diff"], rel_tol=1e-12)
        assert math.isclose(r["t"] * math.sqrt((T - 1) / T), d["dm"], rel_tol=1e-10)
        assert math.isclose(
            r["p"], math.erfc(abs(r["t"]) / math.sqrt(2.0)), rel_tol=1e-12
        )
    assert (
        lc.nw(_ar1(2579, 0.0, 3))["lag"] == 8
    )  # the lag the paper prints for 2,579 days


def test_who():
    assert lc.who(["lgbm", "xgb", "rf"], 3) == "all three trees"
    assert (
        lc.who(["rf", "lgbm"], 3) == "LightGBM and the random forest"
    )  # the order of TREES
    assert lc.who(["rf"], 3) == "the random forest"
    assert lc.who([], 3) == "none of the trees"
    assert lc.who(["lgbm", "xgb"], 2) == "both trees"


def test_sig_helpers():
    r = pd.Series({"p": 0.01, "extra_gain": -1e-3})
    assert lc.sig_neg(r) and not lc.sig_pos(r)
    assert not lc.sig_neg(pd.Series({"p": 0.2, "extra_gain": -1e-3}))
    assert lc.sig_pos(pd.Series({"p": 0.04, "extra_gain": 2e-3}))
    assert not lc.sig_neg(None) and not lc.sig_pos(None)


@pytest.mark.skipif(not NUMBERS.is_file(), reason="make_rehaul_fip_lc.py not run")
def test_every_used_macro_is_defined():
    defined = cp.macros(NUMBERS.read_text(encoding="utf-8"))
    used: set[str] = set()
    for f in [*(WRITEUP / "rehaul").glob("*.tex"), TABLE]:
        text = re.sub(
            r"(?<!\\)%.*$", "", f.read_text(encoding="utf-8"), flags=re.MULTILINE
        )
        used |= set(_USE.findall(text))
    assert used, "the paper uses no \\rhlc macro"
    assert used <= set(defined), sorted(used - set(defined))
    assert all(v.strip() for n, v in defined.items() if n.startswith("rhlc")), (
        "an empty \\rhlc value"
    )


@pytest.mark.skipif(not NUMBERS.is_file(), reason="make_rehaul_fip_lc.py not run")
def test_names_not_defined_elsewhere():
    pre = (
        (WRITEUP / "paper_rehaul.tex")
        .read_text(encoding="utf-8")
        .split(r"\begin{document}")[0]
    )
    pre = re.sub(r"(?<!\\)%.*$", "", pre, flags=re.MULTILINE)
    files = re.findall(r"\\input\{generated/([^}]+)\}", pre)
    assert "rehaul_fip_lc_numbers" in files
    ours = {
        n
        for n in cp.macros(NUMBERS.read_text(encoding="utf-8"))
        if n.startswith("rhlc")
    }
    for name in files:
        if name == "rehaul_fip_lc_numbers":
            continue
        other = set(
            cp.macros(
                (GEN / f"{name}.tex").read_text(encoding="utf-8", errors="replace")
            )
        )
        assert not ours & other, (name, sorted(ours & other))


@pytest.mark.skipif(not SELFCHECK.is_file(), reason="make_rehaul_fip_lc.py not run")
def test_selfcheck_reproduced_the_committed_values():
    sc = pd.read_csv(SELFCHECK)
    assert sc["ok"].astype(bool).all()
    cross = sc[sc["source"].str.contains("_cross_window/")]
    # the two cross-window comparisons: 24 QLIKE, 48 DM, 24 gap and 24 extra-gain rows, every row compared
    assert not cross["column"].str.startswith("(row").any()
    assert cross.groupby("source")["key"].nunique().sum() == 120
    assert sc["source"].str.contains("rehaul_robust_window_final").any()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
