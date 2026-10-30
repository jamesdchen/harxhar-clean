"""Tests of writeup/tools/regen.py, the regeneration-and-impact tool (2026-10-29).

Synthetic strings only (no generator, git or TeX run): the macro parser (nested braces, escaped braces, comments, the
\\providecommand{\\x}{}\\renewcommand{\\x}{v} form, a value over two lines), the classification of a change (numeric
or not, p-values crossing 0.05 / 0.10, sign flips), the live-line scan (comment text, \\iffalse ... \\fi with nested
conditionals and an \\else), the pdflatex log parser, and the dependency map (every key a generator, no cycle).

Run:  python -m pytest writeup/tests/test_regen.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / "writeup" / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import regen  # noqa: E402


# ------------------------------------------------------------------------------------------------- the macro parser
def test_parse_newcommand_nested_braces() -> None:
    m = regen.parse_macros(r"\newcommand{\rhA}{\textbf{1{,}234} and {x}}  % a comment {")
    assert m == {"rhA": r"\textbf{1{,}234} and {x}"}


def test_parse_provide_then_renew() -> None:
    t = r"\providecommand{\rhB}{}\renewcommand{\rhB}{0.123}  % note" + "\n" + r"\providecommand{\rhC}{7}"
    assert regen.parse_macros(t) == {"rhB": "0.123", "rhC": "7"}


def test_parse_providecommand_keeps_first_definition() -> None:
    # \providecommand does nothing when the name is defined; \renewcommand always sets
    t = r"\providecommand{\x}{a}\providecommand{\x}{b}\renewcommand{\y}{1}\providecommand{\y}{2}"
    assert regen.parse_macros(t) == {"x": "a", "y": "1"}


def test_parse_forms_args_star_escapes_comments() -> None:
    t = "\n".join(
        [
            r"\newcommand*{\rhStar}{s}",
            r"\newcommand\rhBare{b}",
            r"\providecommand{\ser}[2]{#2}",
            r"\newcommand{\rhEsc}{a \{ b \} 5\%}",
            r"% \newcommand{\rhCommented}{1}",
            r"\newcommand{\rhTwoLines}{first % dropped",
            r"second}",
        ]
    )
    m = regen.parse_macros(t)
    assert m["rhStar"] == "s" and m["rhBare"] == "b" and m["ser"] == "#2"
    assert m["rhEsc"] == r"a \{ b \} 5\%"
    assert "rhCommented" not in m
    assert m["rhTwoLines"] == "first \nsecond"


def test_strip_comment_escaped_percent() -> None:
    assert regen.strip_comment(r"12\% of it % comment") == r"12\% of it "
    assert regen.strip_comment(r"a \\% comment after a line break") == r"a \\"


# ------------------------------------------------------------------------------------------- classifying a change
@pytest.mark.parametrize(
    ("old", "new", "nonnumeric"),
    [
        ("0.123", "0.125", False),
        ("33{,}527", "33{,}528", False),
        ("12.3\\%", "$-$4.5\\%", False),
        ("0.012", r"\ensuremath{<}0.001", False),
        ("yes", "no", True),
        ("LightGBM", "XGBoost", True),
        ("3 of 6", "three of 6", True),
        (r"\rhPending{}", "0.41", True),
        ("0.41", "", True),
        ("significant (p = 0.01)", "not significant (p = 0.20)", True),
    ],
)
def test_nonnumeric(old: str, new: str, nonnumeric: bool) -> None:
    assert regen.is_nonnumeric(old, new) is nonnumeric


@pytest.mark.parametrize(
    ("value", "parsed"),
    [
        ("0.034", (0.034, False)),
        (r"\ensuremath{<}0.001", (0.001, True)),
        ("$<$0.01", (0.01, True)),
        ("$p = 0.2$", (0.2, False)),
        ("1.7", None),
        ("--", None),
        ("0.03 and 0.04", None),
    ],
)
def test_p_value(value: str, parsed: tuple[float, bool] | None) -> None:
    assert regen.p_value(value) == parsed


@pytest.mark.parametrize(
    ("name", "old", "new", "crossed"),
    [
        ("rhorDMLassoRidgeP", "0.04", "0.06", (0.05,)),
        ("rhorDMLassoRidgeP", "0.04", "0.12", (0.05, 0.10)),
        ("rhorDMLassoRidgeP", "0.06", "0.09", ()),
        ("rhorDMLassoRidgeP", r"\ensuremath{<}0.001", "0.07", (0.05,)),
        ("rhorDMLassoRidgeP", "0.050", "0.049", (0.05,)),  # p < t is significant; 0.050 is not
        ("rhorDMLassoRidgeT", "0.04", "0.06", ()),  # not a p-value name, no bound in the value
        ("rhxSomething", r"\ensuremath{<}0.001", "0.2", (0.05, 0.10)),  # a bound marks it as a p-value
        ("rhxShareP", "12.5", "0.03", ()),  # 12.5 is no p-value: nothing to compare
    ],
)
def test_p_crossings(name: str, old: str, new: str, crossed: tuple[float, ...]) -> None:
    assert regen.p_crossings(name, old, new) == crossed


def test_p_bound_undecided() -> None:
    # "p < 0.2" cannot be placed against 0.10 or 0.05, so a move from it to 0.01 crosses nothing that can be shown
    assert regen.below((0.2, True), 0.10) is None
    assert regen.below((0.001, True), 0.05) is True
    assert regen.p_crossings("xP", r"\ensuremath{<}0.2", "0.01") == ()


@pytest.mark.parametrize(
    ("old", "new", "flip"),
    [
        ("$-$0.12", "0.15", True),
        ("0.15", "-0.15", True),
        (r"\ensuremath{-}1.2", "$-$1.3", False),
        ("0.00", "$-$0.00", False),  # a rounded zero
        ("1.2", "1.3", False),
        ("yes", "$-$1", False),
    ],
)
def test_sign_flip(old: str, new: str, flip: bool) -> None:
    assert regen.sign_flip(old, new) is flip


# ------------------------------------------------------------------------------------------------ the live lines
def _live(text: str) -> dict[int, str]:
    return {no: live for no, live, _ in regen.live_lines(text)}


def test_live_skips_comments() -> None:
    t = "% \\rhA in a comment\nText \\rhB % and \\rhC\n  % indented comment \\rhD\n"
    live = _live(t)
    assert list(live) == [2]
    assert "\\rhB" in live[2] and "\\rhC" not in live[2]


def test_live_skips_iffalse_nested() -> None:
    t = "\n".join(
        [
            "before \\rhA",  # 1 live
            "\\iffalse",  # 2
            "\\rhB \\ifdefined\\rhX inner \\fi",  # 3 skipped, nested conditional closed
            "\\rhC",  # 4 skipped
            "\\fi after \\rhD",  # 5 live after the \fi
            "% \\iffalse (a comment: opens nothing)",  # 6
            "\\rhE",  # 7 live
            "mid \\rhF \\iffalse \\rhG \\fi \\rhH",  # 8: F and H live, G skipped
        ]
    )
    live = _live(t)
    assert sorted(live) == [1, 5, 7, 8]
    assert "\\rhD" in live[5] and "\\rhB" not in "".join(live.values())
    assert "\\rhF" in live[8] and "\\rhH" in live[8] and "\\rhG" not in live[8]


def test_live_iffalse_else_and_not_conditionals() -> None:
    t = "\n".join(
        [
            "\\iffalse",  # 1
            "$a \\iff b$ \\rhA",  # 2 skipped (\iff opens nothing)
            "\\else \\rhB",  # 3 live
            "\\ifdefined\\rhY y\\fi \\rhC",  # 4 live
            "\\fi \\rhD",  # 5 live
            "\\newif\\ifdraft \\rhE",  # 6 live: \newif\ifdraft opens nothing
        ]
    )
    live = _live(t)
    assert sorted(live) == [3, 4, 5, 6]
    assert "\\rhB" in live[3] and "\\rhD" in live[5]


def test_usage_index_and_impact_lines(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    w = tmp_path / "writeup"
    (w / "rehaul").mkdir(parents=True)
    (w / "paper_rehaul.tex").write_text("\\input{rehaul/01}\n\\rhA{} once\n", encoding="utf-8")
    (w / "rehaul" / "01.tex").write_text("\\rhAB and \\rhA\n% \\rhA\n\\iffalse\n\\rhA\n\\fi\n", encoding="utf-8")
    monkeypatch.setattr(regen, "REPO", tmp_path)
    monkeypatch.setattr(regen, "WRITEUP", w)
    idx = regen.usage_index()
    assert [(f, n) for f, n, _ in idx["rhA"]] == [("writeup/paper_rehaul.tex", 2), ("writeup/rehaul/01.tex", 1)]
    assert [(f, n) for f, n, _ in idx["rhAB"]] == [("writeup/rehaul/01.tex", 1)]


# --------------------------------------------------------------------------------------------- the pdflatex log
def test_parse_log() -> None:
    log = "\n".join(
        [
            "(./rehaul/06_results.tex",
            "! Undefined control sequence.",
            "l.42 ... the gain is \\rhrMissingMacro",
            "                                       {} percent",
            "LaTeX Warning: Reference `tab:rh_gone' on page 12 undefined on input line 99.",
            "Package natbib Warning: Citation `Smith2020' on page 3 undefined on input line 7.",
            "! LaTeX Error: File `figures/x.pdf' not found.",
            "Output written on paper_rehaul.pdf (119 pages, 2345678 bytes).",
        ]
    )
    blg = 'Warning--I didn\'t find a database entry for "Jones2019"'
    c = regen.parse_log(log, blg)
    assert c.pages == 119
    assert len(c.errors) == 2 and c.errors[0].startswith("Undefined control sequence")
    assert c.undef_cs == ["\\rhrMissingMacro  (l.42 ... the gain is \\rhrMissingMacro)"]
    assert c.undef_refs == ["tab:rh_gone"]
    assert c.undef_cites == ["Jones2019", "Smith2020"]


# ---------------------------------------------------------------------------------------------- the dependency map
def test_dependency_map() -> None:
    on_disk = regen.discover()
    for name, g in regen.GENERATORS.items():
        assert name in on_disk, f"GENERATORS names make_rehaul_{name}.py, which is not on disk"
        for d in g.after:
            assert d in regen.GENERATORS and d != name, (name, d)
    waves = regen.run_order(sorted(regen.GENERATORS))
    pos = {n: i for i, w in enumerate(waves) for n in w}
    assert all(pos[d] < pos[n] for n, g in regen.GENERATORS.items() for d in g.after)


def test_run_order_cycle(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(regen, "GENERATORS", {"a": regen.Gen(after=("b",)), "b": regen.Gen(after=("a",))})
    with pytest.raises(SystemExit, match="cycle"):
        regen.run_order(["a", "b"])


def test_resolve_names() -> None:
    disk = ["channels", "ladder", "results"]
    assert regen.resolve_names("make_rehaul_ladder.py, channels", disk) == ["channels", "ladder"]
    assert regen.resolve_names(None, disk) == disk
    with pytest.raises(SystemExit):
        regen.resolve_names("nope", disk)


def test_parse_pythons(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("REGEN_PYTHON", raising=False)
    p = regen.parse_pythons(["C:/py/python.exe", "disclosures=C:/other/python.exe", "make_rehaul_fi.py=D:/x.exe"])
    assert p == {"": "C:/py/python.exe", "disclosures": "C:/other/python.exe", "fi": "D:/x.exe"}
    assert regen.parse_pythons(None) == {"": sys.executable}
    monkeypatch.setenv("REGEN_PYTHON", "E:/env/python.exe")
    assert regen.parse_pythons(["fi=D:/x.exe"]) == {"": "E:/env/python.exe", "fi": "D:/x.exe"}


def test_downstream_skipped(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(regen, "GENERATORS", {"a": regen.Gen(), "b": regen.Gen(after=("a",)), "c": regen.Gen()})
    assert regen.downstream_skipped(["a"], ["a", "b", "c"]) == {"b": ["a"]}
    assert regen.downstream_skipped(["a", "b"], ["a", "b", "c"]) == {}


def test_dependents_transitive(monkeypatch: pytest.MonkeyPatch) -> None:
    g = {"e": regen.Gen(), "m": regen.Gen(after=("e",)), "h": regen.Gen(after=("e", "m")), "x": regen.Gen()}
    monkeypatch.setattr(regen, "GENERATORS", g)
    assert regen.dependents("e", list(g)) == {"m", "h"}
    assert regen.dependents("m", list(g)) == {"h"}
    assert regen.dependents("x", list(g)) == set()
    assert regen.start_order(["x", "m", "e"], list(g)) == ["e", "m", "x"]


def test_start_order_real_map() -> None:
    names = sorted(regen.GENERATORS)
    first = regen.start_order(names, names)[:2]
    assert set(first) == {"economics", "results"}


# ------------------------------------------------------------------------------------------- report and summary
def test_report_and_summary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(regen, "REPORT", tmp_path / "regen_report.md")
    log = regen.REPO / "writeup" / "tools" / "regen_logs" / "x.log"
    runs = [
        regen.Run("ladder", (), 0, 3.0, log),
        regen.Run("ladder", ("--prepatch",), 1, 2.0, log, "Traceback ...\nValueError: boom"),
        regen.Run("fi", (), None, 0.0, log, "", "not run: py lacks pandas<3 (has 3.0.2) (give one: --python fi=PATH)"),
    ]
    f = "writeup/generated/rehaul_x_numbers.tex"
    d = regen.FileDiff(f, 3)
    d.changed = [regen.classify(f, "rhxAP", "0.04", "0.06"), regen.classify(f, "rhxVerdict", "yes", "no")]
    d.removed = [regen.Change(f, "rhxGone", "1", None)]
    imp = regen.Impact({"rhxAP": [("writeup/rehaul/06_results.tex", 12, r"p = \rhxAP")]}, [(f, "+2 -2", "macros")], [])
    comp = regen.parse_log("Output written on paper_rehaul.pdf (9 pages, 1 bytes).")
    text = regen.write_report(["--compile"], "HEAD", "0" * 40, runs, {"holiday": ["economics"]}, [d], imp, comp, 5.0)
    assert (tmp_path / "regen_report.md").read_text(encoding="utf-8") == text
    assert "**ladder --prepatch failed** (exit 1)" in text and "ValueError: boom" in text
    assert "| fi | (none) | not run |" in text
    assert r"`\rhxAP` (rehaul_x_numbers.tex): `0.04` -> `0.06` (crosses 0.05)" in text
    assert "### Non-numeric changes (words, verdicts, yes / no): 1" in text
    assert r"- writeup/rehaul/06_results.tex:12: p = \rhxAP" in text
    assert "- holiday (reads economics)" in text
    s = "\n".join(regen.summary(runs, [d], imp, comp, "HEAD", 5.0))
    assert "1 FAILED: ladder" in s and "1 NOT RUN (interpreter): fi" in s
    assert "2 changed, 0 added, 1 removed in 1 of 1 files; 1 non-numeric, 1 p-value crossings" in s
    assert "compile: 9 pages, 0 errors" in s
