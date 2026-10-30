"""Tests of writeup/style/prose_check.py, the code checks of the paper-prose skill's audit stage (2026-10-29).

Each check gets cases that must fire and cases that must not (B7 of writeup/reports/Academic writing skills
comparison.md): ", none significantly" fires V6, and so do the live tails of 08_feature_importance.tex (:224 and :261
when written); the field whitelist ("the leverage effect", "robust standard errors", "statistically significant") never
fires AP-V; a typed "12.3%" fires V2 and "\\rhabX{}\\%" does not. Diff mode: a rewrite that drops a \\ref fires V16, a
change outside --target fires V14, an \\auth{...} change fires V13. Also the masking (comments, \\iffalse, \\auth),
line ranges, generator strings, severities from the pattern files, the JSON rows and the exit status.

Run:  python -m pytest writeup/tests/test_prose_check.py
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
STYLE = ROOT / "writeup" / "style"
if str(STYLE) not in sys.path:
    sys.path.insert(0, str(STYLE))

import prose_check as pc  # noqa: E402

SEV, SEV_SOURCE = pc.load_severities()
LIVE_FI = ROOT / "writeup" / "rehaul" / "08_feature_importance.tex"
ROW_KEYS = {"mode", "id", "file", "line", "severity", "candidate", "match", "message"}


def lint(tex: str, name: str = "x.tex", rng: tuple[int, int] | None = None) -> list[pc.Hit]:
    return pc.lint_docs(pc.docs_for(Path(name), tex), name, rng, SEV, conclusion="conclusion" in name)


def rows_of(rows: list[pc.Hit], rid: str, candidate: bool | None = None) -> list[pc.Hit]:
    return [r for r in rows if r.id == rid and (candidate is None or r.candidate is candidate)]


def gate_hits(rows: list[pc.Hit]) -> list[pc.Hit]:
    return [r for r in rows if r.severity == "gate" and not r.candidate]


# ----------------------------------------------------------------------------------------------------------- V6


def test_v6_fires_on_none_significantly() -> None:
    rows = lint("The removals change the loss by at most \\rhX\\%, none significantly.\n")
    v6 = rows_of(rows, "V6")
    assert len(v6) == 1
    assert v6[0].match == ", none significantly"
    assert v6[0].candidate is True and v6[0].line == 1
    assert v6[0].severity == SEV["V6"]
    assert not gate_hits(rows)  # a candidate never fails the run by itself


def test_v6_fires_on_a_tail_before_a_parenthesis_split_across_lines() -> None:
    tex = (
        "Against HAR the full set changes the loss of the ridge by \\rhabRidgeX{} and \\rhabEnetX\\%, none\n"
        "  significantly ($p \\geq{}$\\rhabLinPMin), and that of LightGBM by \\rhabLgbmX\\%.\n"
    )
    v6 = rows_of(lint(tex), "V6")
    assert [r.line for r in v6] == [1]


@pytest.mark.parametrize(
    ("tex", "match"),
    [
        ("Beyond them, only boosted trees gain, slightly.\n", ", slightly"),
        ("Three classes beat HAR -- and only barely.\n", "-- and only barely"),
        ("The other groups add little, if at all.\n", ", if at all"),
        ("The window counts bars; neither does the gap.\n", "; neither does the gap"),
    ],
)
def test_v6_fires_on_other_tails(tex: str, match: str) -> None:
    assert [r.match for r in rows_of(lint(tex), "V6")] == [match]


@pytest.mark.parametrize(
    "tex",
    [
        "On ordinary days, none of the removals is significant at the \\rhLevel\\% level.\n",
        "The trees gain on every-bar loss, every-bar QLIKE included.\n",
        "We fit the rolling window, and the HAR benchmark.\n",
        "No removal is significant (smallest $p = $ \\rhPMin).\n",
        "The rule bans the tail ``none significantly'' in agent text.\n",
        "% The loss changes by at most \\rhX\\%, none significantly.\n",
    ],
)
def test_v6_does_not_fire(tex: str) -> None:
    assert not rows_of(lint(tex), "V6")


FROZEN_TAILS = r"""
  gradient-boosted trees' ordinary-day loss by at most \rhchOrdLoOtherTreeMaxAbs\%, none significantly. The
  redundancy reading holds in all \rhchLinOrdNRedTest{} linear classes.
  ridge, lasso and elastic net by \rhabRidgeFullVsAddRmFomcAllPct, \rhabLassoFullVsAddRmFomcAllPct{} and
  \rhabEnetFullVsAddRmFomcAllPct\%, none significantly ($p \geq{}$\rhabLinFullVsAddRmFomcAllPMin), and that of
  LightGBM and XGBoost by \rhabLgbmFullVsAddRmFomcAllPct{} and \rhabXgbFullVsAddRmFomcAllPct\%.
"""


def test_v6_fires_on_the_frozen_feature_importance_tails() -> None:
    v6 = rows_of(lint(FROZEN_TAILS), "V6")
    assert sorted(r.line for r in v6) == [2, 5]


def test_v6_fires_on_the_live_feature_importance_tails() -> None:
    """08_feature_importance.tex:224 and :261 (B7); if the lines moved, the sentences are found by their text."""
    text = LIVE_FI.read_text(encoding="utf-8").replace("\r\n", "\n")
    lines = text.split("\n")
    live = [n for n, ln in enumerate(lines, start=1) if "none significantly" in ln and not ln.lstrip().startswith("%")]
    if not live:
        pytest.skip("no live 'none significantly' tail left in 08_feature_importance.tex")
    fired = {r.line for r in rows_of(pc.lint_docs(pc.docs_for(LIVE_FI), "fi", None, SEV), "V6")}
    assert set(live) <= fired
    for n in (224, 261):
        if n <= len(lines) and "none significantly" in lines[n - 1]:
            assert n in fired


# ----------------------------------------------------------------------------------------------------------- V2


def test_v2_typed_percent_fires_and_macro_does_not() -> None:
    typed = rows_of(lint("The full set lowers QLIKE by 12.3\\% on ordinary days.\n"), "V2")
    assert [(r.match, r.candidate, r.severity) for r in typed] == [("12.3%", False, "gate")]
    assert not rows_of(lint("The full set lowers QLIKE by \\rhabX{}\\% on ordinary days.\n"), "V2")


def test_v2_exemptions() -> None:
    tex = "\n".join(
        [
            "Corsi (2009) proposed HAR(3) for daily data; Section 3 describes the 2020--2024 sample.",
            "See Table~\\ref{tab:rh_a1} and \\citet{corsi2009}, \\citep[p.~12]{Patton2011}.\\label{sec:s2}",
            "The loss is $L_{t,1} = x_2^{3}$ on the S\\&P~500 and in the COVID-19 crash.",
            "% A comment with 12.5 percent.",
            "\\iffalse The loss fell by 4.4\\%. \\fi",
            "\\auth{Q-2026-10-29-1}{He wrote 3.8\\% here.}",
        ]
    )
    assert not gate_hits(rows_of(lint(tex), "V2"))


def test_v2_candidates() -> None:
    rows = lint("All six classes beat HAR at 15:30 ($p = 0.021$) in a two-sided test.\n")
    v2 = rows_of(rows, "V2")
    assert {r.match for r in v2} == {"six", "15:30", "0.021"}
    assert all(r.candidate for r in v2)


def test_v2_quotations_and_cited_findings() -> None:
    assert not rows_of(lint("The author asked for ``the 12 closing bars'' in the table.\n"), "V2")
    cited = rows_of(lint("\\citet{Bollerslev2016} find a 40\\% gain over HAR.\n"), "V2")
    assert [(r.match, r.candidate) for r in cited] == [("40%", True)]
    ours = rows_of(lint("The gain is significant at the 5\\% level under the \\citet{DM1995} test.\n"), "V2")
    assert [(r.match, r.candidate) for r in ours] == [("5%", False)]


# --------------------------------------------------------------------------------------------------------- AP-V


@pytest.mark.parametrize(
    "phrase", ["the leverage effect", "robust standard errors", "statistically significant", "the significance level"]
)
def test_apv_field_whitelist_never_fires(phrase: str) -> None:
    rows = lint(f"We report {phrase} in every class.\n")
    assert not [r for r in rows if r.id.startswith("AP-V")]


def test_apv_hard_flag_and_density_rows() -> None:
    rows = lint("We delve into the data. We leverage the data and its insights.\n")
    v1 = rows_of(rows, "AP-V1")
    assert [(r.match, r.severity, r.candidate) for r in v1] == [("delve", SEV["AP-V1"], False)]
    v2 = rows_of(rows, "AP-V2")
    assert len(v2) == 1 and v2[0].severity == "rate"
    assert v2[0].per_1000 is not None and v2[0].per_1000 > 0
    assert v2[0].match == "2 hits"


def test_apv_quotation_not_flagged() -> None:
    assert not rows_of(lint("The catalog bans ``delve'' and ``tapestry'' in agent text.\n"), "AP-V1")


# ----------------------------------------------------------------------------------------------------------- V3


def test_v3_listed_terms_fire() -> None:
    rows = lint("The per-bar OLS uses all features over the test window.\n")
    assert {r.match.lower() for r in rows_of(rows, "V3", candidate=False)} == {"per-bar", "all features", "test window"}


@pytest.mark.parametrize(
    "tex",
    [
        "The closing-interval OLS uses the full predictor set over the evaluation span.\n",
        "The pooled intraday model has one observation per bar.\n",
        "Added to the HAR\n  benchmark set, the realized moments lower the loss against the random walk.\n",
        "Against the heterogeneous autoregressive (HAR) benchmark, the full set lowers QLIKE.\n",
    ],
)
def test_v3_does_not_fire(tex: str) -> None:
    assert not rows_of(lint(tex), "V3")


def test_v3_term_with_an_ordinary_sense_is_a_candidate() -> None:
    assert [r.candidate for r in rows_of(lint("A family of \\rhK{} comparisons is tested.\n"), "V3")] == [True]
    assert [r.candidate for r in rows_of(lint("The linear model family gains.\n"), "V3")] == [False]


def test_v3_bare_benchmark_is_a_candidate_and_math_clash_fires() -> None:
    rows = lint("It beats the benchmark. The interaction is $\\Phi_{t,AB}$.\n")
    assert [r.candidate for r in rows_of(rows, "V3") if r.match.lower() == "benchmark"] == [True]
    assert any(r.match.startswith("\\Phi_{t,AB") and not r.candidate for r in rows_of(rows, "V3"))


# ----------------------------------------------------------------------------------------------------- V5, V9, V10


def test_v5_lexicon_candidates() -> None:
    rows = rows_of(lint("We score every regular-hours bar; the random forest is clearly the largest.\n"), "V5")
    assert {r.match for r in rows} == {"every", "clearly", "largest"}  # "forest" is not a superlative
    assert all(r.candidate and r.severity == SEV["V5"] for r in rows)


@pytest.mark.parametrize(
    ("title", "fires"),
    [
        ("Where the gain comes from: FOMC statement days", True),
        ("Which predictors carry the gain?", True),
        ("What remains", True),
        ("Forecast accuracy", False),
        ("Related work", False),
        ("Predictor-group ablations", False),
    ],
)
def test_v9_titles(title: str, fires: bool) -> None:
    rows = rows_of(lint(f"\\subsection{{{title}}}\n\\label{{sec:x}}\nText.\n"), "V9")
    assert bool(rows) is fires
    assert all(r.candidate and r.line == 1 for r in rows)


@pytest.mark.parametrize(
    ("tex", "fires"),
    [
        ("The results underscore the role of the FOMC calendar.\n", True),
        ("This decomposition highlights the realized moments.\n", True),
        ("Table~\\ref{tab:rh_ablation} reports the removals.\n", False),
        ("\\citet{Corsi2009} highlight the role of the HAR terms.\n", False),
        ("We show the loss of each class.\n", False),
    ],
)
def test_v10(tex: str, fires: bool) -> None:
    assert bool(rows_of(lint(tex), "V10")) is fires


# --------------------------------------------------------------------------------------------------------- AP-S


def test_aps_rows() -> None:
    tex = (
        "The gain is large---and it is close. The calendar serves as a proxy, highlighting the FOMC days.\n"
        "\\item \\textbf{In short.} Two groups carry the gain.\n"
    )
    rows = lint(tex)
    em = rows_of(rows, "AP-S1")
    assert len(em) == 1 and em[0].severity == "rate" and em[0].per_1000 is not None
    assert [(r.match, r.candidate) for r in rows_of(rows, "AP-S2")] == [("highlighting", False)]
    assert rows_of(rows, "AP-S7")[0].severity == "rate"
    assert [r.line for r in rows_of(rows, "AP-S6")] == [2]
    length = rows_of(rows, pc.LENGTH_ID)
    assert len(length) == 1 and length[0].severity == "rate" and length[0].match == "sentence length"
    assert not rows_of(lint(tex, name="09_conclusion.tex"), "AP-S6")


def test_aps_genuine_three_item_roadmap_is_not_counted() -> None:
    rows = lint("Sections~\\ref{a}, \\ref{b} and \\ref{c} give the details.\n")
    assert not rows_of(rows, "AP-S8")


# ------------------------------------------------------------------------------------------------------ masking


def test_comments_iffalse_and_auth_are_never_reported() -> None:
    tex = "\n".join(
        [
            "% Was: the results underscore 12.3\\%, none significantly.",
            "\\iffalse",
            "The results underscore 12.3\\%, none significantly. \\ifdefined\\rhX\\else\\input{x}\\fi",
            "\\fi",
            "\\auth{Q-2026-10-29-9}{The results underscore 12.3\\%, none significantly.}",
            "The full set lowers the loss by 4.4\\%.",
        ]
    )
    rows = [r for r in lint(tex) if r.id != pc.LENGTH_ID]
    assert {(r.id, r.line) for r in rows} == {("V2", 6)}


def test_line_range_limits_the_rows() -> None:
    tex = "The loss falls by 1.5\\%.\nThe loss falls by 2.5\\%.\nThe loss falls by 3.5\\%.\n"
    assert [r.line for r in rows_of(lint(tex, rng=(2, 2)), "V2")] == [2]
    assert pc.parse_target("C:/x/y.tex:10-20") == ("C:/x/y.tex", (10, 20))
    assert pc.parse_target("C:\\x\\y.tex") == ("C:\\x\\y.tex", None)


def test_generator_strings(tmp_path: Path) -> None:
    gen = tmp_path / "make_x.py"
    gen.write_text(
        '"""Docstring with 12.3% and a tail, none significantly."""\n'
        "\n"
        'OUT = "results/x_1.csv"\n'
        "\n"
        "\n"
        "def note(max_abs: float) -> str:\n"
        '    head = f"\\\\textit{{Notes:}} the loss falls by at most {max_abs:.1f}\\\\% on the days of the table."\n'
        '    tail = r"""\\item Removing a group changes the loss by 4.2\\%,\n'
        'none significantly (Table~\\ref{tab:x})."""\n'
        "    return head + tail\n",
        encoding="utf-8",
    )
    rows = pc.lint([str(gen)], SEV)
    assert [(r.id, r.line, r.match) for r in rows_of(rows, "V2", candidate=False)] == [("V2", 8, "4.2%")]
    assert [r.line for r in rows_of(rows, "V6")] == [9]


# ------------------------------------------------------------------------------------------------- severities


def test_severities_come_from_the_pattern_files() -> None:
    assert SEV["V2"] == "gate" and SEV["V5"] == "review" and SEV["AP-V2"] == "rate"
    if (pc.PATTERNS_DIR / "numbers" / "V2-numbers-from-macros.md").read_text(encoding="utf-8").count("**Severity:**"):
        assert SEV_SOURCE["V2"] == "V2-numbers-from-macros.md"


def test_severity_parsing(tmp_path: Path) -> None:
    (tmp_path / "claims").mkdir()
    (tmp_path / "claims" / "V6-tails.md").write_text("# V6\n\n**Severity:** review. Text.\n", encoding="utf-8")
    (tmp_path / "AP-S-structural.md").write_text(
        "**Severity:** per row.\n\n| Id | Marker | Flag | Severity |\n|---|---|---|---|\n"
        "| AP-S2 | tails | Rate | gate (at drafting) |\n| AP-S9 | bold | none | none |\n",
        encoding="utf-8",
    )
    sev, source = pc.load_severities(tmp_path)
    assert sev["V6"] == "review" and source["V6"] == "V6-tails.md"
    assert sev["AP-S2"] == "gate"
    assert sev["AP-S9"] == pc.DEFAULT_SEVERITY["AP-S9"] and source["AP-S9"] == "built-in"
    assert sev["V2"] == "gate" and source["V2"] == "built-in"


# ------------------------------------------------------------------------------------------------ CLI and JSON


def test_lint_cli_json_rows_and_exit_status(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    bad = tmp_path / "bad.tex"
    bad.write_text("Intro.\nThe loss falls by 12.3\\%, none significantly---as expected.\n", encoding="utf-8")
    assert pc.main(["lint", f"{bad}:2-2", "--json"]) == 1
    rows = json.loads(capsys.readouterr().out)
    assert rows and all(ROW_KEYS <= set(r) for r in rows)
    assert all(r["mode"] == "lint" and r["line"] == 2 for r in rows)
    assert all(("per_1000" in r) == (r["severity"] == "rate") for r in rows)
    ok = tmp_path / "ok.tex"
    ok.write_text("The loss falls by \\rhX\\%, none significantly.\n", encoding="utf-8")
    assert pc.main(["lint", str(ok)]) == 0
    out = capsys.readouterr().out
    assert re.search(r"ok\.tex:1 V6 gate candidate: ", out)


# ------------------------------------------------------------------------------------------------------- diff


BASE = (
    "The full set lowers the loss by \\rhX\\% (Table~\\ref{tab:a}).\n"
    "The trees do not lower the loss.\n"
    "The calendar carries the gain (\\citet{Lucca2015}).\n"
    "\\auth{Q-2026-10-29-1}{The author's own words stay.}\n"
)


def diff(new: str, **kw: object) -> list[pc.Hit]:
    return pc.diff_texts("f.tex", BASE, new, SEV, **kw)  # type: ignore[arg-type]


def test_diff_unchanged_text_has_no_rows() -> None:
    assert diff(BASE, ranges=[(1, 1)], log=[]) == []


def test_diff_v16_dropped_ref_fires() -> None:
    rows = diff(BASE.replace(" (Table~\\ref{tab:a})", ""))
    v16 = rows_of(rows, "V16", candidate=False)
    assert len(v16) == 1 and "refs -tab:a" in v16[0].message and v16[0].line == 1
    assert v16[0].severity == "gate" and v16[0].mode == "diff"


def test_diff_v16_negation_change_is_reported() -> None:
    rows = diff(BASE.replace("do not lower", "barely lower"))
    v16 = rows_of(rows, "V16")
    assert [(r.candidate, r.line) for r in v16] == [(True, 2)]
    assert "-not" in v16[0].message


def test_diff_v14_change_outside_target_fires() -> None:
    new = BASE.replace("carries the gain", "carries most of the gain")
    assert [r.line for r in rows_of(diff(new, ranges=[(1, 1)]), "V14")] == [3]
    assert not rows_of(diff(new, ranges=[(3, 3)]), "V14")


def test_diff_v13_auth_change_fires() -> None:
    rows = diff(BASE.replace("The author's own words stay.", "The author's words stay."))
    v13 = rows_of(rows, "V13")
    assert len(v13) == 1 and v13[0].line == 4 and not v13[0].candidate
    assert not rows_of(rows, "V16") and not rows_of(rows, "V14")  # nothing is reported inside \auth


def test_diff_v12_edit_log() -> None:
    new = BASE.replace("carries the gain", "carries most of the gain")
    assert [r.line for r in rows_of(diff(new, log=[]), "V12")] == [3]
    logged = [pc.LogEntry("writeup/f.tex", 3, "delete the quantifier", "V5")]
    assert not rows_of(diff(new, log=logged), "V12")
    no_rule = [pc.LogEntry("f.tex", 3, "delete the quantifier", "")]
    assert rows_of(diff(new, log=no_rule), "V12")


def _git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@example.com", "-c", "core.autocrlf=false", *args],
        cwd=repo,
        check=True,
        capture_output=True,
    )


def test_diff_cli_in_a_git_repo(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    repo = tmp_path / "repo"
    (repo / "writeup" / "overleaf_sync").mkdir(parents=True)
    f = repo / "writeup" / "s.tex"
    f.write_text(BASE, encoding="utf-8")
    _git(repo, "init", "-q")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "base")
    # A pull commit (it commits map_pulled.json) writes line 3.
    f.write_text(BASE.replace("carries the gain", "carries the statement-day gain"), encoding="utf-8")
    (repo / "writeup" / "overleaf_sync" / "map_pulled.json").write_text("{}", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "Overleaf pull")
    pulled = f.read_text(encoding="utf-8")
    f.write_text(
        pulled.replace(" (Table~\\ref{tab:a})", "")
        .replace("statement-day gain", "gain on statement days")
        .replace("own words", "words"),
        encoding="utf-8",
    )
    record = tmp_path / "author.json"
    record.write_text(json.dumps([{"text": "The author's own words stay.", "file": "writeup/s.tex"}]), encoding="utf-8")
    code = pc.main(["diff", "--base", "HEAD", "--target", f"{f}:1-1", "--author-record", str(record), "--json", str(f)])
    rows = json.loads(capsys.readouterr().out)
    assert code == 1
    got = {(r["id"], r["line"], r["candidate"]) for r in rows}
    assert ("V16", 1, False) in got  # the dropped \ref
    assert ("V14", 3, False) in got  # changed outside the target
    assert ("V15", 3, True) in got  # the base sentence came in through a pull
    assert ("V13", 4, False) in got  # the \auth text changed
    assert sum(1 for r in rows if r["id"] == "V13") == 2  # and the recorded author string is gone
    assert all(ROW_KEYS <= set(r) and r["mode"] == "diff" for r in rows)
    assert pc.main(["diff", "--base", "no-such-rev", str(f)]) == 2
