"""Round trip with the real flattener on a temporary copy of writeup/ (2026-10-20): SPEC.md section 6, end to end.

Steps, all inside a temporary directory (the real writeup/ is only read, and checked unchanged at the end):
1. copy paper_rehaul.tex, rehaul/, generated/, figures/ and references.bib to <tmp>/writeup/;
2. flatten the copy (flatten.py --entry <copy>/paper_rehaul.tex --map <tmp>/map.json --out <tmp>/overleaf);
3. unflatten the unchanged main.tex: "no changes", no file modified;
4. make scripted Overleaf edits in main.tex:
   * one prose word in 01_introduction, on a line with a number macro;
   * two words around a number macro on another introduction line (the macro must be re-linked);
   * the trading-day count 1{,}082 in the abstract (macro \\rhoosDays);
   * a cell of Table 1 (generated/rehaul_results_oos.tex);
   * a new sentence with a digit, on its own line, in 09_conclusion;
   * a word inside a comment of 06_results_floats;
   and, before unflatten, change 01_introduction.tex in the copy in another place (concurrent change);
5. unflatten with --root on the copy and check every outcome, the report and that generated/ is untouched;
6. flatten the copy again: the applied edits show, the conflicting ones are back to the generated values, and an
   unflatten of the new main.tex reports "no changes";
7. compile the copy's paper_rehaul.tex (pdflatex, bibtex, pdflatex x2) and check the PDF has the new word;
8. flatten --check on the edited copy: the new main.tex builds to the same PDF text as the paper;
9. the real writeup/ sources are unchanged.

    python writeup/overleaf_sync/test_roundtrip.py          (set UNFLATTEN_KEEP=1 to keep the temporary directory)

Needs flatten.py next to this file, git, pdflatex, bibtex and pdftotext. Standard library only.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
WRITEUP = HERE.parent
FLATTEN = HERE / "flatten.py"
UNFLATTEN = HERE / "unflatten.py"
DATE = "2026-10-20"
PARK = f"% {DATE} (overleaf): parked -- edited in Overleaf. Was:"
COPY_ITEMS = ["paper_rehaul.tex", "rehaul", "generated", "figures", "references.bib"]

# The scripted Overleaf edits, as (old, new) replacements of text that occurs exactly once in main.tex.
PROSE_OLD = "Each intraday model is estimated across all $M = "
PROSE_NEW = "Each intraday model is fitted across all $M = "
RELINK_OLD = "benchmark across every intraday model class ({pairs} classes in total, with"
RELINK_NEW = "benchmark in every intraday model class ({pairs} classes overall, with"
NUMBER_OLD = "($T_{\\mathrm{oos}} = 1{,}082$ trading days). We compare"
NUMBER_NEW = "($T_{\\mathrm{oos}} = 1{,}083$ trading days). We compare"
TABLE_OLD = "OLS\\tnote{a} & 4.82 & 0.1291 & 74.7"
TABLE_NEW = "OLS\\tnote{a} & 4.82 & 0.1290 & 74.7"
INSERT_AFTER = "(the 15:30--16:00 ET closing half hour).\n"
INSERT_LINE = "  \\item The forecasts are scored on 30 minutes of each trading day.\n"
COMMENT_OLD = "% 2026-10-15 (user): first results table; generator writeup/make_rehaul_oos_table.py"
COMMENT_NEW = "% 2026-10-15 (user): the first results table; generator writeup/make_rehaul_oos_table.py"
CONCURRENT_OLD = "Section~\\ref{sec:rh_related} reviews the related literature."
CONCURRENT_NEW = "Section~\\ref{sec:rh_related} reviews the related literature on intraday forecasting."


def sha_tree(paths: list[Path]) -> dict[str, str]:
    out = {}
    for top in paths:
        for p in ([top] if top.is_file() else sorted(top.rglob("*"))):
            if p.is_file():
                out[str(p)] = hashlib.sha256(p.read_bytes()).hexdigest()
    return out


def run(cmd: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, errors="replace")


@unittest.skipUnless(FLATTEN.exists(), f"{FLATTEN} does not exist yet")
class RoundTrip(unittest.TestCase):
    def setUp(self):
        self.real_before = self.real_sha()
        self.tmp = Path(tempfile.mkdtemp(prefix="unflatten_roundtrip_"))
        self.root = self.tmp / "writeup"
        self.root.mkdir()
        for item in COPY_ITEMS:
            src = WRITEUP / item
            (shutil.copytree if src.is_dir() else shutil.copy2)(src, self.root / item)
        self.out = self.tmp / "overleaf"
        self.map = self.tmp / "map.json"
        self.results: list[tuple[str, bool]] = []

    def tearDown(self):
        width = max(len(name) for name, _ in self.results) if self.results else 0
        print("\nround trip:")
        for name, ok in self.results:
            print(f"  {name:<{width}}  {'pass' if ok else 'FAIL'}")
        if os.environ.get("UNFLATTEN_KEEP"):
            print(f"kept {self.tmp}")
        else:
            shutil.rmtree(self.tmp, ignore_errors=True)
        # Never touch the real sources: every hand-written and generated file still has its hash from the start.
        self.assertEqual(self.real_sha(), self.real_before, "the real writeup/ sources changed")

    def real_sha(self) -> dict[str, str]:
        return sha_tree([WRITEUP / "paper_rehaul.tex", WRITEUP / "rehaul", WRITEUP / "generated"])

    # -- helpers

    def flatten(self) -> str:
        if self.out.exists():
            shutil.rmtree(self.out)
        res = run([sys.executable, str(FLATTEN), "--out", str(self.out), "--map", str(self.map),
                   "--entry", str(self.root / "paper_rehaul.tex")])
        self.assertEqual(res.returncode, 0, f"flatten failed:\n{res.stdout}\n{res.stderr}")
        return (self.out / "main.tex").read_text(encoding="utf-8")

    def unflatten(self, name: str, *extra: str) -> tuple[int, str, str]:
        """Run unflatten.py on the copy; its report goes to <tmp>/REPORT_<name>.md."""
        report_path = self.tmp / f"REPORT_{name}.md"
        res = run([sys.executable, str(UNFLATTEN), "--main", str(self.out / "main.tex"), "--map", str(self.map),
                   "--root", str(self.root), "--date", DATE, "--report", str(report_path), *extra])
        report = report_path.read_text(encoding="utf-8") if report_path.exists() else ""
        return res.returncode, report, res.stdout + res.stderr

    def source(self, rel: str) -> str:
        return (self.root / rel).read_text(encoding="utf-8")

    @contextlib.contextmanager
    def case(self, name: str):
        """A subTest that also records pass/fail for the summary printed at the end."""
        with self.subTest(name):
            ok = False
            try:
                yield
                ok = True
            finally:
                self.results.append((name, ok))

    def compile_copy(self) -> tuple[bool, str]:
        steps = [["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "paper_rehaul.tex"],
                 ["bibtex", "paper_rehaul"],
                 ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "paper_rehaul.tex"],
                 ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "paper_rehaul.tex"]]
        for cmd in steps:
            res = run(cmd, cwd=self.root)
            if res.returncode != 0:
                return False, f"{' '.join(cmd)} failed:\n{res.stdout[-3000:]}"
        return (self.root / "paper_rehaul.pdf").exists(), ""

    # -- the test

    def test_round_trip(self):
        hand = [self.root / "paper_rehaul.tex", self.root / "rehaul"]
        generated = [self.root / "generated"]
        flat = self.flatten()
        gen_before = sha_tree(generated)

        with self.case("round trip, no edit"):
            hand_before = sha_tree(hand)
            code, report, log = self.unflatten("no_edit")
            self.assertEqual(code, 0, log)
            self.assertIn("No changes", report)
            self.assertEqual(sha_tree(hand), hand_before)

        # Scripted Overleaf edits.
        pairs = self.macro_value("rhrAfPoolPairs")
        relink_old = RELINK_OLD.format(pairs=pairs)
        relink_new = RELINK_NEW.format(pairs=pairs)
        edits = [(PROSE_OLD, PROSE_NEW), (relink_old, relink_new), (NUMBER_OLD, NUMBER_NEW), (TABLE_OLD, TABLE_NEW),
                 (INSERT_AFTER, INSERT_AFTER + INSERT_LINE), (COMMENT_OLD, COMMENT_NEW)]
        edited = flat
        for old, new in edits:
            self.assertEqual(edited.count(old), 1, f"{old!r} occurs {edited.count(old)} times in main.tex")
            edited = edited.replace(old, new)
        (self.out / "main.tex").write_text(edited, encoding="utf-8", newline="\n")

        # Concurrent change to a source file after the flatten, in another place of the introduction.
        intro = self.root / "rehaul/01_introduction.tex"
        text = intro.read_text(encoding="utf-8")
        self.assertEqual(text.count(CONCURRENT_OLD), 1)
        intro.write_text(text.replace(CONCURRENT_OLD, CONCURRENT_NEW), encoding="utf-8", newline="\n")
        paper_before = self.source("paper_rehaul.tex")

        code, report, log = self.unflatten("edits")

        with self.case("exit status 2 (conflicts)"):
            self.assertEqual(code, 2, log)

        with self.case("prose edit"):
            intro_new = self.source("rehaul/01_introduction.tex")
            self.assertIn(f"  {PARK}\n  % \\item Each intraday model is estimated across all $M = \\rhdmNIntervals$",
                          intro_new)
            self.assertIn("  \\item Each intraday model is fitted across all $M = \\rhdmNIntervals$ half-hour",
                          intro_new)

        with self.case("prose edit around a macro (re-link)"):
            self.assertIn("benchmark in every intraday model class (\\rhrAfPoolPairs{} classes overall, with "
                          "\\rhrAfPoolLowerSig{}", intro_new)

        with self.case("number edit"):
            self.assertEqual(self.source("paper_rehaul.tex"), paper_before)
            self.assertIn("Number conflict", report)
            self.assertIn("`\\rhoosDays`", report)
            self.assertIn("`writeup/make_rehaul_oos_table.py`", report)

        with self.case("table edit"):
            self.assertIn("Generated conflict", report)
            self.assertIn("`generated/rehaul_results_oos.tex`", report)
            self.assertIn("`writeup/make_rehaul_oos_table.py`", report)

        with self.case("generated files untouched"):
            self.assertEqual(sha_tree(generated), gen_before)

        with self.case("inserted sentence"):
            concl = self.source("rehaul/09_conclusion.tex")
            self.assertIn(INSERT_AFTER + INSERT_LINE, concl)
            self.assertNotIn("(overleaf): parked", concl)
            warnings = report.split("## Warnings", 1)[1].split("## Next step", 1)[0]
            self.assertIn("rehaul/09_conclusion.tex", warnings)
            self.assertIn("`30`", warnings)

        with self.case("edit inside a comment"):
            floats = self.source("rehaul/06_results_floats.tex")
            self.assertIn(COMMENT_NEW, floats)
            self.assertNotIn("(overleaf): parked", floats)

        with self.case("concurrent change (three-way merge)"):
            self.assertIn(CONCURRENT_NEW, intro_new)
            self.assertNotIn("<<<<<<<", intro_new)
            self.assertIn("three-way merged", report)
            self.assertNotIn("Merge conflict", report)

        with self.case("re-flatten"):
            flat2 = self.flatten()
            self.assertIn(PROSE_NEW, flat2)
            self.assertIn(relink_new, flat2)
            self.assertIn(INSERT_AFTER + INSERT_LINE, flat2)
            self.assertIn(COMMENT_NEW, flat2)
            self.assertIn(CONCURRENT_NEW, flat2)
            self.assertIn(NUMBER_OLD, flat2)  # the number conflict is overwritten by the generated value
            self.assertIn(TABLE_OLD, flat2)
            code2, report2, log2 = self.unflatten("reflatten")
            self.assertEqual(code2, 0, log2)
            self.assertIn("No changes", report2)

        with self.case("paper still compiles"):
            ok, why = self.compile_copy()
            self.assertTrue(ok, why)
            txt = run(["pdftotext", "-layout", str(self.root / "paper_rehaul.pdf"), "-"]).stdout
            self.assertIn("fitted across all", txt)
            self.assertIn("30 minutes of each trading day", txt)
            log = (self.root / "paper_rehaul.log").read_text(encoding="latin-1")
            self.assertNotIn("Undefined control sequence", log)

        with self.case("re-flattened main.tex builds like the paper (flatten --check)"):
            res = run([sys.executable, str(FLATTEN), "--out", str(self.tmp / "overleaf_check"),
                       "--map", str(self.tmp / "map_check.json"), "--entry", str(self.root / "paper_rehaul.tex"),
                       "--check"])
            self.assertEqual(res.returncode, 0, res.stdout[-3000:] + res.stderr[-3000:])
            self.assertIn("--check PASSED", res.stdout)

        with self.case("real writeup/ untouched"):
            self.assertEqual(self.real_sha(), self.real_before)

        self.assertTrue(all(ok for _, ok in self.results), self.results)

    def macro_value(self, name: str) -> str:
        return json.loads(self.map.read_text(encoding="utf-8"))["macros"][name]["expanded"]


if __name__ == "__main__":
    unittest.main(verbosity=2)
