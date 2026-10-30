"""Tests of the macro-release scoring and tables after ablations addendum L (iii) (2026-10-30: the macro analysis
apples to apples with FOMC's): experiments/ablate_score.py's day split (rel / nrel, rel_<r> / nrel_<r>, FOMC's
ORDINARY / EVENT split; no quiet span) and the forest's full_mac reference, and writeup/make_rehaul_ablations.py's
macros, Panel F and tab:rh_ablation_macro on synthetic scores (no forecast file, no TeX run).  A test on the real
calendar runs when src/data/loading.py exports it (it reads data/core_stats.parquet and data/releases.parquet).

Run:  python -m pytest writeup/tests/test_ablation_macro_iii.py
"""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
for _p in (str(ROOT), str(ROOT / "experiments"), str(ROOT / "writeup")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


SC = _load("ablate_score_t", ROOT / "experiments" / "ablate_score.py")
GN = _load("make_rehaul_ablations_t", ROOT / "writeup" / "make_rehaul_ablations.py")
import src.data.loading as LD  # noqa: E402

RELS = ("cpi", "emp", "ppi", "gdp", "trade", "retail")


# ------------------------------------------------------------------------------------------------------- the scorer
def test_is_mac_and_release_of() -> None:
    assert all(SC.is_mac(a) for a in ("add_mac", "add_rm_fomc_mac", "full_mac", "add_mac_cpi", "add_rm_fomc_mac_gdp"))
    assert not any(SC.is_mac(a) for a in ("add_fomc", "add_rm_fomc", "full", "har", "full_fix", "add_rm_fomc_vd"))
    assert SC.mac_release_of("add_mac_trade") == "trade"
    assert SC.mac_release_of("add_rm_fomc_mac_retail") == "retail"
    assert SC.mac_release_of("add_mac") is None and SC.mac_release_of("full_mac") is None


@pytest.fixture
def synth(monkeypatch):
    """Ten scored days of three bars; calendars and FOMC days set by hand."""
    days = pd.bdate_range("2021-03-01", periods=10)
    idx = pd.DatetimeIndex([d + pd.Timedelta(hours=h) for d in days for h in (10, 13, 16)])
    cal = {r: pd.DatetimeIndex([]) for r in RELS}
    cal["cpi"] = pd.DatetimeIndex([days[1], days[3]])
    cal["emp"] = pd.DatetimeIndex([days[3], days[5], pd.Timestamp("2021-06-01")])  # one day off the scored span
    cal["gdp"] = pd.DatetimeIndex([days[7]])
    for r in ("ppi", "trade", "retail"):
        cal[r] = pd.DatetimeIndex([days[8]])
    ev = pd.DatetimeIndex([days[3], days[6]])
    monkeypatch.setattr(SC, "N_DAYS", len(days))
    monkeypatch.setattr(SC, "macro_calendar", lambda: (RELS, cal, "synthetic"))
    monkeypatch.setattr(SC, "event_days", lambda d: ev)
    return idx, days, ev


def test_macro_spans_split_as_fomc(synth) -> None:
    idx, days, ev = synth
    checks: dict = {}
    sp = SC.macro_spans(idx, days, checks)
    assert "quiet" not in sp and not any(k.startswith("quiet") for k in sp)
    assert set(sp) == {"rel", "nrel", *(f"{s}_{r}" for s in ("rel", "nrel") for r in RELS)}
    rel_days = {days[i] for i in (1, 3, 5, 7, 8)}
    assert set(sp["rel"].normalize()) == rel_days
    assert set(sp["nrel"].normalize()) == set(days) - rel_days  # FOMC day 6 is a non-release day
    for s in ("", *(f"_{r}" for r in RELS)):  # each pair partitions the scored bars
        a, b = sp[f"rel{s}"], sp[f"nrel{s}"]
        assert a.intersection(b).empty and a.union(b).equals(idx)
    mr = checks["macro_releases"]
    assert "quiet_days" not in mr and "quiet_bars" not in mr
    assert mr["release_days"] == 5 and mr["nonrelease_days"] == 5
    assert mr["release_days_by_release"]["emp"] == 2 and mr["nonrelease_days_by_release"]["emp"] == 8
    assert mr["release_and_fomc_days"] == 1 and mr["nonrelease_and_fomc_days"] == 1
    assert mr["release_and_fomc_days_by_release"] == {"cpi": 1, "emp": 1, "ppi": 0, "gdp": 0, "trade": 0, "retail": 0}
    assert mr["release_bars"] + mr["nonrelease_bars"] == len(idx)
    assert not mr["counts_asserted"]


@pytest.mark.skipif(not hasattr(LD, "macro_release_days"), reason="src/data/loading.py has no macro calendar")
def test_real_calendar_counts() -> None:
    """The features' calendar on Table 1's scored days: 590 release days, 1,989 non-release days (asserted inside)."""
    rv = SC.core_rv()
    days = SC.evaluation_days(rv)
    idx = pd.DatetimeIndex([d + pd.Timedelta(hours=16) for d in days])
    checks: dict = {}
    SC.macro_spans(idx, days, checks)
    mr = checks["macro_releases"]
    assert mr["counts_asserted"] and mr["release_days"] == SC.N_REL_ANY == 590
    assert mr["nonrelease_days"] == SC.N_DAYS - SC.N_REL_ANY == 1989
    assert mr["release_and_fomc_days"] + mr["nonrelease_and_fomc_days"] == mr["fomc_days"] == SC.N_EVENT
    assert sum(mr["release_and_fomc_days_by_release"].values()) >= mr["release_and_fomc_days"]


# ---------------------------------------------------------------------------------------------------- the generator
CHECKS = {
    "macro_releases": dict(
        source="synthetic",
        releases=list(RELS),
        release_days=590,
        nonrelease_days=1989,
        release_days_by_release={r: 100 + i for i, r in enumerate(RELS)},
        nonrelease_days_by_release={r: 2479 - i for i, r in enumerate(RELS)},
        release_and_fomc_days=24,
        release_and_fomc_days_by_release={r: i for i, r in enumerate(RELS)},
        nonrelease_and_fomc_days=57,
        fomc_days=81,
    )
}
SPANS = ("all", "nrel", "rel", "c16", *(f"{s}_{r}" for s in ("nrel", "rel") for r in RELS))


def scores(have: list[tuple[str, str]], ref_of) -> tuple[pd.DataFrame, pd.DataFrame, set]:
    """Synthetic metrics / dm frames as the scorer writes them: every arm on every span; the DM of each macro arm
    against its reference."""
    rng = np.random.default_rng(1)
    m = [dict(cls=c, arm=a, span=s, qlike=0.2 + 0.01 * rng.random(), mse=1e-9) for c, a in have for s in SPANS]
    d = []
    for c, a in have:
        b = ref_of(c, a)
        if b is not None and (c, b) in have:
            d += [
                dict(cls=c, arm=a, vs_cls=c, vs_arm=b, span=s, loss="qlike", dm=rng.normal(), p=rng.random())
                for s in SPANS
            ]
    M, D = pd.DataFrame(m), pd.DataFrame(d)
    return M.set_index(["cls", "arm", "span"]), D.set_index(["cls", "arm", "vs_cls", "vs_arm", "span"]), set(have)


def ref_of(c: str, a: str) -> str | None:
    if a == "full_mac":
        return "full_fix" if c == "rf" else "full"
    if a == "add_mac" or a.startswith("add_mac_"):
        return "har"
    if a.startswith("add_rm_fomc_mac"):
        return "add_rm_fomc"
    return None


def arms(c: str) -> list[str]:
    base = ["har", "add_rm_fomc", "full"] + (["full_fix"] if c == "rf" else [])
    mac = [f"add_rm_fomc_mac_{r}" for r in RELS] + ["add_rm_fomc_mac"]
    if c != "ols":
        mac += [f"add_mac_{r}" for r in RELS] + ["add_mac", "full_mac"]
    return base + mac


FULL = [(c, a) for c in GN.CLS for a in arms(c)]


def parts(tex: str) -> list[list[str]]:
    """The body rows (\\quad ...) of each float of the table."""
    return [
        [ln for ln in t.split("\\midrule", 1)[1].split("\\bottomrule", 1)[0].splitlines() if ln.startswith("\\quad")]
        for t in tex.split("\\begin{table}")[1:]
    ]


def test_macro_table_complete() -> None:
    q, dq, have = scores(FULL, ref_of)
    tex = GN.macro_table(q, dq, have, CHECKS)
    pa, pb = parts(tex)  # Panel A (vs HAR), Panel B (vs HAR + RM + FOMC, the continuation float)
    assert len(pa) == len(pb) == 3 * (len(RELS) + 1)  # three day types; a row per release, then macro
    rows = pa + pb
    assert all(r.count("&") == len(GN.CLS) for r in rows)
    assert all(re.search(r" \((\$-\$)?\d+\.\d\)", r) for r in rows)  # "pct (dm)" cells
    # the OLS: dashes in Panel A only (not walked on HAR + a calendar), none elsewhere
    assert all(r.split("&")[1].strip() == "--" for r in pa) and not any(r.split("&")[1].strip() == "--" for r in pb)
    assert sum(r.count("--") for r in rows) == len(pa)
    assert "pending" not in tex and "quiet" not in tex.lower()
    assert tex.count("\\begin{") == tex.count("\\end{") and tex.count("{") == tex.count("}")
    assert tex.count("\\label{tab:rh_ablation_macro}") == 1 and tex.count("\\ContinuedFloat") == 1
    for r in RELS:
        assert f"\\rhabNRelFomcDays{GN.REL_TEX[r]}{{}}" in tex and f"\\rhabNRelDays{GN.REL_TEX[r]}{{}}" in tex
    for sub in GN.MAC_TAB_SPANS.values():
        assert tex.count(f"{{{sub}}} \\\\") == 2  # the sub-heading, once in each panel
    # Panel A's CPI row on release days is add_mac_cpi against HAR on rel_cpi
    cpi_rel = [r for r in pa if r.startswith("\\quad CPI &")][2]
    for i, c in enumerate(GN.CLS[1:], start=2):
        assert cpi_rel.split("&")[i].strip(" \\").startswith(GN.fmt(GN.gain(q, c, "add_mac_cpi", "har", "rel_cpi"), 1))


def test_macro_table_staged_and_forest_reference() -> None:
    # the forest without its full set on the macro cache, LightGBM without HAR + CPI: dashes and the pending sentence
    have_l = [(c, a) for c, a in FULL if (c, a) not in {("rf", "full_fix"), ("lgbm", "add_mac_cpi")}]
    q, dq, have = scores(have_l, ref_of)
    tex = GN.macro_table(q, dq, have, CHECKS)
    pa, pb = parts(tex)
    cpi = [r for r in pa if r.startswith("\\quad CPI &")]
    assert len(cpi) == 3 and all(r.split("&")[5].strip() == "--" for r in cpi)  # LightGBM column
    assert tex.count("pending") == 1 and "the LightGBM column are pending" in tex.split("\\ContinuedFloat")[0]
    assert not any("--" in r.split("&", 2)[2] for r in pb)  # Panel B complete
    f = "\n".join(GN.panel_f(q, dq, have, GN.CLS))
    fm = [ln for ln in f.split("Full set + macro", 1)[1].splitlines() if ln.startswith("\\quad")]
    assert len(fm) == 3 and all(ln.split("&")[7].strip(" \\") == "--" for ln in fm)  # forest: no full_fix, no cell
    assert "the Forest column are pending" in f
    # with full_fix the forest's full_mac is against it, not against the full set
    q2, dq2, have2 = scores(FULL, ref_of)
    f2 = "\n".join(GN.panel_f(q2, dq2, have2, GN.CLS))
    pc = GN.gain(q2, "rf", "full_mac", "full_fix", "all")
    assert f"{GN.fmt(pc, 1)} (" in f2.split("Full set + macro", 1)[1].splitlines()[1].split("&")[7]
    assert "pending" not in f2


def test_panel_f_rows_and_spans() -> None:
    q, dq, have = scores(FULL, ref_of)
    f = GN.panel_f(q, dq, have, GN.CLS)
    labs = [ln.split("&")[0].strip() for ln in f if ln.startswith("\\quad")]
    assert labs == ["\\quad every bar", "\\quad non-release days", "\\quad release days"] * 3
    assert all(ln.count("&") == len(GN.CLS) for ln in f if ln.startswith("\\quad"))
    t = "\n".join(f)
    assert "\\rhabNNrelDays{}" in t and "\\rhabNQuietDays" not in t and "FOMC days &" not in t


def test_mac_macros_names() -> None:
    q, dq, have = scores(FULL, ref_of)
    lines = GN.mac_macros(q, dq, CHECKS, have)
    names = [ln.split("}", 1)[0].removeprefix("\\newcommand{\\rhab") for ln in lines if ln.startswith("\\newcommand")]
    assert len(names) == len(set(names)) and all(n.isalpha() for n in names)
    for n in (
        "NNrelDays",
        "NRelFomcDaysGdp",
        "NNrelDaysCpi",
        "RidgeAddMacCpiNrelPct",
        "RfAddMacRetailRelDm",
        "XgbAddRmFomcMacGdpNrelP",
        "RfFullMacVsFullRelPct",
        "LgbmAddMacNrelQlike",
    ):
        assert n in names, n
    assert not any("Quiet" in n or n.endswith(("EvPct", "EvQlike")) for n in names)
    assert not any(n.startswith("OlsAddMac") for n in names)  # the OLS is not walked on HAR + release / macro
    # the forest's full_mac is against full_fix
    v = next(ln for ln in lines if ln.startswith("\\newcommand{\\rhabRfFullMacVsFullAllPct}"))
    assert f"{{{GN.fmt(GN.gain(q, 'rf', 'full_mac', 'full_fix', 'all'), 1)}}}" in v


def test_old_scores_refused() -> None:
    old = {"macro_releases": dict(CHECKS["macro_releases"], quiet_days=1932)}
    old["macro_releases"].pop("nonrelease_days")
    with pytest.raises(AssertionError, match="before addendum L"):
        GN.mac_releases(old)
