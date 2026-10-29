r"""Flatten the rehauled paper into one Overleaf file, main.tex, plus a map of where each of its characters came from.

Spec: writeup/overleaf_sync/SPEC.md (2026-10-20), sections 1-4. The paper of record, writeup/paper_rehaul.tex, loads
its macro-only number files (writeup/generated/*_numbers.tex) with \input lines in the preamble and its section files
(writeup/rehaul/*.tex) with \input lines in the body; some section files \input generated floats
(writeup/generated/rehaul_*.tex). This script writes

  DIR/main.tex        the paper as one file. Every live \input is inlined between a "%% >>> begin X.tex (...)" and a
                      "%% <<< end X.tex" marker line; every number macro (an argument-free macro defined in a number
                      file) is replaced by its literal value, fully expanded; the number-file \input lines of the
                      preamble and the \ifdefined\X\else\input{generated/<number file>}\fi guard lines are dropped;
                      everything else, comments included, is copied verbatim. A header comment at the top names the
                      commit it was made from.
  DIR/references.bib  a copy (every .bib named by a live \bibliography).
  DIR/<figure>        every figure of a live \includegraphics, at its path relative to writeup/ (default extensions
                      .pdf, .png, .jpg, .eps, and \graphicspath).
  map.json            (default writeup/overleaf_sync/map.json) the segment map of SPEC section 4: the flat text, every
                      source text read, every number macro with its value and expansion, and segments that say where
                      each stretch of main.tex came from (src / macro / gen / synthetic / drop).
                      writeup/overleaf_sync/unflatten.py reads it to bring Overleaf edits back to the sources.

TeX lexing (SPEC section 2). A % preceded by an even number of backslashes starts a comment that runs to the end of
the line; comments are copied verbatim (no expansion, no inlining). A control word is a backslash and the maximal run
of ASCII letters, so \rhfiP and \rhfiPredGap are different names. A number-macro call followed at once by {} consumes
the {}; written without braces it swallows the spaces and tabs after it and, when the next line is not blank, the end
of line too (and that line's indentation, which TeX skips at the start of a line). If the written expansion ends in a
control word that the next character would extend (a letter, or a blank after a consumed {}), {} is appended.

Number files are the \input targets under generated/ in the preamble of the entry file, in that order; nothing is
hard-coded. Their definitions are read with TeX's rules: \providecommand defines only an undefined name, \newcommand
and \renewcommand (re)define, so \providecommand{\X}{}\renewcommand{\X}{value} gives "value". Definitions with
arguments (\ser [2], \serx [3]) are not number macros; they stay as the preamble defines them.

Before anything is written the segment invariants are checked: sorted segments that tile main.tex, src text equal to
the source text, macro text equal to the expansion, and every character of every inlined file accounted for once.
Output is deterministic: the same sources give a byte-identical main.tex and map (except the map's "created").

--check builds main.tex (in a temporary copy of DIR's main.tex, .bib files and figures) and paper_rehaul.tex (in a
temporary copy of writeup/, so the tracked build files are not touched), each with pdflatex, bibtex, pdflatex x2 (the
first two passes in -draftmode; the two builds in parallel). It requires identical pdftotext output, equal page counts, no live \input / \include / \InputIfFileExists and no live
number-macro control word in main.tex, no "Undefined control sequence" in the main.tex log, and no warning in that
log that the paper's log does not have (line numbers ignored).

CLI
    python writeup/overleaf_sync/flatten.py --out DIR [--map writeup/overleaf_sync/map.json]
        [--entry writeup/paper_rehaul.tex] [--check] [--keep-builds] [--name main.tex] [--bib-suffix SUFFIX]

Examples
    python writeup/overleaf_sync/flatten.py --out /tmp/overleaf_flat
    python writeup/overleaf_sync/flatten.py --out /tmp/overleaf_flat --check
    python writeup/overleaf_sync/flatten.py --out /tmp/overleaf_flat --map /tmp/overleaf_flat/map.json

Exit status: 0 on success, 1 when --check fails, 2 when the sources cannot be flattened (the reason is printed).
Standard library only; --check needs pdflatex, bibtex, pdftotext and pdfinfo on PATH.
"""

from __future__ import annotations

import argparse
import collections
import concurrent.futures
import datetime
import difflib
import functools
import hashlib
import json
import os
import posixpath
import re
import shutil
import string
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
WRITEUP = SCRIPT_DIR.parent
DEFAULT_ENTRY = WRITEUP / "paper_rehaul.tex"
DEFAULT_MAP = SCRIPT_DIR / "map.json"
SCRIPT_NAME = "writeup/overleaf_sync/flatten.py"

LETTERS = frozenset(string.ascii_letters)
BLANKS = " \t"
GRAPHICS_EXTS = (".pdf", ".png", ".jpg", ".eps")
KNOWN_GRAPHICS_EXTS = GRAPHICS_EXTS + (".jpeg", ".mps")
DEFINERS = ("newcommand", "providecommand", "renewcommand")
# Live control words that a flat file cannot keep or that this flattener cannot inline faithfully.
UNSUPPORTED = ("include", "includeonly", "InputIfFileExists", "endinput")
# Control words that read the next token without expanding it: a number macro right after one cannot be replaced.
NON_EXPANDING = (
    "def",
    "gdef",
    "edef",
    "xdef",
    "let",
    "futurelet",
    "ifdefined",
    "ifx",
    "string",
    "meaning",
    "noexpand",
    "show",
    "detokenize",
    "unexpanded",
    "newcommand",
    "renewcommand",
    "providecommand",
    "DeclareRobustCommand",
)
# A guard line: \ifdefined\NAME\else\input{generated/<number file>}\fi
GUARD = re.compile(
    r"\\ifdefined[ \t]*\\([A-Za-z]+)[ \t]*\\else[ \t]*\\input[ \t]*\{([^{}\\%]*)\}[ \t]*\\fi(?![A-Za-z])"
)
# A generator named in the first comment lines of a generated file.
GENERATOR = re.compile(r"(?<![\w/.])((?:writeup/)?make_\w+\.py|experiments/\w+\.py)")
# Names the preamble defines before a number file is loaded (for \providecommand, which keeps an existing name).
PREAMBLE_DEF = re.compile(
    r"\\(?:(?:new|renew|provide)command|DeclareRobustCommand|DeclareMathOperator)\*?\s*\{?\s*\\([A-Za-z]+)"
    r"|\\(?:def|gdef|edef|xdef|let)\s*\\([A-Za-z]+)"
)
_PLAIN = re.compile(r"[^\\%]+")


class FlattenError(Exception):
    """The sources cannot be flattened faithfully; the message says where and why."""


# ----------------------------------------------------------------------------------------------------------------
# TeX lexing (SPEC section 2)
# ----------------------------------------------------------------------------------------------------------------


def next_token(text: str, i: int, end: int) -> tuple[str, int, int]:
    r"""The lexical item that starts at text[i], as (kind, start, stop).

    kind is "comment" (an unescaped % to the end of the line, newline included), "cw" (a control word: a backslash
    and the maximal run of ASCII letters), "cs" (a control symbol: a backslash and one other character) or "text" (a
    run of characters that are neither a backslash nor a percent sign). Scanning left to right, a % starts a comment
    exactly when an even number of backslashes precede it, since an odd number makes it the control symbol \%.
    """
    c = text[i]
    if c == "%":
        j = text.find("\n", i, end)
        return "comment", i, (end if j < 0 else j + 1)
    if c == "\\":
        j = i + 1
        while j < end and text[j] in LETTERS:
            j += 1
        if j > i + 1:
            return "cw", i, j
        return "cs", i, min(i + 2, end)
    return "text", i, _PLAIN.match(text, i, end).end()


def skip_blank(text: str, i: int, end: int) -> int:
    """Skip whitespace and comments (used to read the arguments of \\input, \\newcommand and friends)."""
    while i < end:
        if text[i] in " \t\r\n":
            i += 1
        elif text[i] == "%":
            i = next_token(text, i, end)[2]
        else:
            break
    return i


def read_group(text: str, i: int, end: int) -> int:
    """text[i] is "{": the index just after its matching "}" (escaped braces and comments do not count)."""
    depth = 0
    j = i
    while j < end:
        kind, s, e = next_token(text, j, end)
        if kind == "text":
            for k in range(s, e):
                if text[k] == "{":
                    depth += 1
                elif text[k] == "}":
                    depth -= 1
                    if depth == 0:
                        return k + 1
        j = e
    raise FlattenError(f"unbalanced braces from offset {i}")


def read_bracket(text: str, i: int, end: int) -> int:
    """text[i] is "[": the index just after the matching "]" at brace depth 0."""
    j = i + 1
    while j < end:
        kind, s, e = next_token(text, j, end)
        if kind == "text":
            k = s
            while k < e:
                if text[k] == "{":
                    k = read_group(text, k, end)
                    break
                if text[k] == "]":
                    return k + 1
                k += 1
            else:
                j = e
                continue
            j = k
            continue
        j = e
    raise FlattenError(f"unclosed [ from offset {i}")


def ends_in_control_word(s: str) -> bool:
    """True when s ends in a control word (a run of letters after an odd number of backslashes)."""
    j = len(s)
    while j > 0 and s[j - 1] in LETTERS:
        j -= 1
    if j == len(s):
        return False
    k = j
    while k > 0 and s[k - 1] == "\\":
        k -= 1
    return (j - k) % 2 == 1


def call_extent(text: str, j: int, end: int) -> tuple[int, bool]:
    """Where the source of a number-macro call whose control word ends at j stops, and whether it consumed "{}".

    TeX skips the blanks after a control word. A single end of line after them is dropped as well, and the next
    line's indentation is skipped too (a new line starts in TeX's state N), unless that next line is blank: then the
    newlines stay, so that the paragraph break survives.
    """
    if j + 2 <= end and text.startswith("{}", j):
        return j + 2, True
    k = j
    while k < end and text[k] in BLANKS:
        k += 1
    if k < end and text[k] == "\n":
        m = k + 1
        while m < end and text[m] in BLANKS:
            m += 1
        if m < end and text[m] != "\n":
            return m, False
    return k, False


def line_bounds(text: str, i: int) -> tuple[int, int]:
    """(start, end) of the line holding text[i]; end is the index of its newline (or len(text))."""
    ls = text.rfind("\n", 0, i) + 1
    le = text.find("\n", i)
    return ls, (len(text) if le < 0 else le)


def blank_comments(text: str) -> str:
    """text with the characters of every comment replaced by spaces (newlines kept), so offsets stay valid."""
    parts = []
    i, n = 0, len(text)
    while i < n:
        kind, s, e = next_token(text, i, n)
        parts.append(re.sub(r"[^\n]", " ", text[s:e]) if kind == "comment" else text[s:e])
        i = e
    return "".join(parts)


def live_control_words(text: str):
    """Yield (name, start, stop) for every control word outside comments."""
    i, n = 0, len(text)
    while i < n:
        kind, s, e = next_token(text, i, n)
        if kind == "cw":
            yield text[s + 1 : e], s, e
        i = e


def line_of(text: str, i: int) -> int:
    return text.count("\n", 0, i) + 1


class SourceLocation:
    """file:line of an offset, for messages; the line number is counted only when the message is formatted."""

    def __init__(self, rel: str, text: str, i: int):
        self.rel, self.text, self.i = rel, text, i

    def __str__(self) -> str:
        return f"{self.rel}:{line_of(self.text, self.i)}"


# ----------------------------------------------------------------------------------------------------------------
# Number files
# ----------------------------------------------------------------------------------------------------------------


@dataclass
class Definition:
    cmd: str  # newcommand / providecommand / renewcommand
    name: str
    nargs: int
    value: str  # the raw text between the braces of the body
    start: int


@dataclass
class Macro:
    name: str
    value: str
    file: str
    generator: str | None
    expanded: str | None = None


def parse_definitions(text: str, rel: str) -> list[Definition]:
    """Every \\newcommand / \\providecommand / \\renewcommand (with or without *) of a number file.

    A number file holds definitions only: any other live material would be lost when its \\input line is dropped, so
    it is an error.
    """
    defs = []
    i, n = 0, len(text)
    while i < n:
        kind, s, e = next_token(text, i, n)
        if kind == "comment" or (kind == "text" and not text[s:e].strip()):
            i = e
            continue
        if kind != "cw" or text[s + 1 : e] not in DEFINERS:
            raise FlattenError(
                f"{rel}:{line_of(text, s)}: live material other than a definition ({text[s:e][:40]!r}); a number "
                f"file must hold definitions only, since flatten.py drops its \\input line"
            )
        cmd, j = text[s + 1 : e], e
        if j < n and text[j] == "*":
            j += 1
        j = skip_blank(text, j, n)
        braced = j < n and text[j] == "{"
        if braced:
            j = skip_blank(text, j + 1, n)
        kind2, s2, e2 = next_token(text, j, n) if j < n else ("", j, j)
        if kind2 != "cw":
            raise FlattenError(f"{rel}:{line_of(text, s)}: cannot read the name defined by \\{cmd}")
        name, j = text[s2 + 1 : e2], e2
        if braced:
            j = skip_blank(text, j, n)
            if j >= n or text[j] != "}":
                raise FlattenError(f"{rel}:{line_of(text, s)}: cannot read the name defined by \\{cmd}")
            j += 1
        j = skip_blank(text, j, n)
        nargs = 0
        if j < n and text[j] == "[":
            k = read_bracket(text, j, n)
            try:
                nargs = int(text[j + 1 : k - 1].strip())
            except ValueError:
                raise FlattenError(f"{rel}:{line_of(text, s)}: bad argument count for \\{name}") from None
            j = skip_blank(text, k, n)
            if j < n and text[j] == "[":  # optional-argument default
                nargs = max(nargs, 1)
                j = skip_blank(text, read_bracket(text, j, n), n)
        if j >= n or text[j] != "{":
            raise FlattenError(f"{rel}:{line_of(text, s)}: cannot read the body of \\{name}")
        g = read_group(text, j, n)
        defs.append(Definition(cmd, name, nargs, text[j + 1 : g - 1], s))
        i = g
    return defs


def find_generator(text: str) -> str | None:
    """The generator named in the first comment lines of a generated file (SPEC section 1), or None.

    The first comment lines are the comment block at the top of the file (blank lines allowed); the first
    writeup/make_*.py or experiments/*.py named there is the generator. A bare make_*.py is taken to live in writeup/.
    """
    for line in text.split("\n")[:20]:
        stripped = line.strip()
        if not stripped:
            continue
        if not stripped.startswith("%"):
            break
        m = GENERATOR.search(stripped)
        if m:
            g = m.group(1)
            return g if "/" in g else "writeup/" + g
    return None


# ----------------------------------------------------------------------------------------------------------------
# The flattener
# ----------------------------------------------------------------------------------------------------------------


@dataclass
class Segment:
    a: int
    b: int
    kind: str  # src | macro | gen | synthetic | drop
    file: str | None
    src: tuple[int, int] | None
    macro: str | None
    generator: str | None

    def as_json(self) -> dict:
        return {
            "flat": [self.a, self.b],
            "kind": self.kind,
            "file": self.file,
            "src": list(self.src) if self.src is not None else None,
            "macro": self.macro,
            "generator": self.generator,
        }


class Flattener:
    def __init__(self, entry: Path, flat_name: str = "main.tex", bib_suffix: str = ""):
        self.root = entry.resolve().parent
        # 2026-10-20 (user): output names, so the flat paper can sit next to an existing Overleaf project's files
        # (main2.tex, references2.bib) without replacing them. bib_suffix renames every .bib of a live \bibliography.
        self.flat_name = flat_name
        self.bib_suffix = bib_suffix
        self.entry_rel = entry.name
        self.texts: dict[str, str] = {}  # rel path -> text as read
        self.kinds: dict[str, str] = {}  # rel path -> hand | generated | numbers
        self.generators: dict[str, str | None] = {}
        self.number_files: list[str] = []
        self.macros: dict[str, Macro] = {}
        self.out: list[str] = []
        self.pos = 0
        self.flat_tail = ""  # the last characters written (to see whether the flat text ends in a control word)
        self.segments: list[Segment] = []
        self.inline_count: collections.Counter[str] = collections.Counter()
        self.stack: list[str] = []  # files being inlined (cycle check)
        self.figures: list[str] = []
        self.bibs: list[str] = []
        self.graphicspath: list[str] = []
        self.warnings: list[str] = []

    # -- files --------------------------------------------------------------------------------------------------

    def read(self, rel: str, kind: str) -> str:
        if rel not in self.texts:
            path = self.root / rel
            if not path.is_file():
                raise FlattenError(f"missing file {rel} (relative to {self.root})")
            with open(path, encoding="utf-8", newline="") as fh:
                self.texts[rel] = fh.read()
            self.kinds[rel] = kind
            self.generators[rel] = find_generator(self.texts[rel]) if kind != "hand" else None
        return self.texts[rel]

    def resolve_input(self, arg: str, where: object) -> str:
        """The file an \\input{arg} reads, relative to the root: arg.tex if it exists, else arg (TeX's order)."""
        p = posixpath.normpath(arg.strip())
        candidates = [p] if p.endswith(".tex") else [p + ".tex", p]
        for c in candidates:
            if (self.root / c).is_file():
                return c
        raise FlattenError(f"{where}: \\input{{{arg}}} names no file (tried {', '.join(candidates)})")

    @staticmethod
    def input_argument(text: str, e: int, where: object) -> tuple[str, int]:
        """The braced argument of the \\input control word that ends at e, and the index after its "}"."""
        n = len(text)
        j = e
        while j < n and text[j] in " \t\n":  # LaTeX's \input looks past blanks for a brace
            j += 1
        if j >= n or text[j] != "{":
            raise FlattenError(f"{where}: \\input without braces (the TeX primitive form) is not supported")
        g = read_group(text, j, n)
        arg = text[j + 1 : g - 1].strip()
        if not arg or "\\" in arg:
            raise FlattenError(f"{where}: cannot resolve \\input{{{arg}}}")
        return arg, g

    # -- number files and macros ----------------------------------------------------------------------------------

    def load_numbers(self) -> None:
        """Find the number files in the entry's preamble and read their definitions with TeX's rules."""
        text = self.read(self.entry_rel, "hand")
        doc = self.document_start(text)
        live = blank_comments(text)
        preamble_defs = [(m.start(), m.group(1) or m.group(2)) for m in PREAMBLE_DEF.finditer(live, 0, doc)]
        for name, s, e in live_control_words(text[:doc]):
            if name != "input":
                continue
            where = f"{self.entry_rel}:{line_of(text, s)}"
            arg, _ = self.input_argument(text, e, where)
            rel = self.resolve_input(arg, where)
            if not rel.startswith("generated/"):
                continue  # a hand-written preamble file: inlined like any other \input
            if rel in self.number_files:
                continue
            self.number_files.append(rel)
            ntext = self.read(rel, "numbers")
            before = {nm for pos, nm in preamble_defs if pos < s}
            for d in parse_definitions(ntext, rel):
                if d.nargs:
                    # \ser / \serx: not number macros. Dropping the \input drops this definition, which is safe only
                    # when the preamble already defines the name.
                    if d.name not in before:
                        raise FlattenError(
                            f"{rel}:{line_of(ntext, d.start)}: \\{d.name} takes arguments and is not defined in the "
                            f"preamble before the \\input; dropping the number file would lose it"
                        )
                    continue
                defined = d.name in self.macros or d.name in before
                if d.cmd == "providecommand" and defined:
                    if d.name in before and d.name not in self.macros:
                        self.warnings.append(
                            f"{rel}: \\providecommand{{\\{d.name}}} is a no-op (the preamble defines it); not a "
                            f"number macro"
                        )
                    continue
                if d.cmd == "newcommand" and defined:
                    self.warnings.append(
                        f"{rel}:{line_of(ntext, d.start)}: \\newcommand{{\\{d.name}}} of a defined name; TeX keeps "
                        f"the earlier definition, and so does flatten.py"
                    )
                    continue
                if d.name in before:  # \renewcommand of a preamble name: from here on it is a number macro
                    self.warnings.append(f"{rel}: \\{d.name} redefines a preamble command")
                self.macros[d.name] = Macro(d.name, d.value, rel, self.generators[rel])
        for name in self.macros:
            self.expansion(name, ())

    def document_start(self, text: str) -> int:
        for name, s, e in live_control_words(text):
            if name == "begin" and text.startswith("{document}", e):
                return s
        raise FlattenError(f"{self.entry_rel}: no live \\begin{{document}}")

    def expansion(self, name: str, stack: tuple[str, ...]) -> str:
        """The flat expansion of a number macro: its value with the number macros inside expanded recursively."""
        m = self.macros[name]
        if m.expanded is None:
            if name in stack:
                raise FlattenError(f"number macros expand in a cycle: \\{' -> \\'.join(stack + (name,))}")
            m.expanded = self.expand_text(m.value, stack + (name,), f"value of \\{name} ({m.file})")
        return m.expanded

    def expand_text(self, raw: str, stack: tuple[str, ...], where: str) -> str:
        """Expand the number macros in a macro value with the call rules of the source text.

        Comments in a value are not tokens of the value: each is dropped with its newline, and the indentation of
        the line after it, which TeX skips.
        """
        out: list[str] = []
        i, n = 0, len(raw)
        while i < n:
            kind, s, e = next_token(raw, i, n)
            if kind == "comment":
                i = e
                while i < n and raw[i] in BLANKS:
                    i += 1
                continue
            if kind == "cw" and raw[s + 1 : e] in self.macros:
                k, braced = call_extent(raw, e, n)
                exp = self.expansion(raw[s + 1 : e], stack)
                exp += self.brace_after("".join(out)[-256:], exp, raw, k, braced, stack, where)
                out.append(exp)
                i = k
                continue
            out.append(raw[s:e])
            i = e
        return "".join(out)

    def next_flat_char(self, text: str, k: int, stack: tuple[str, ...]) -> str:
        """The first character the flat text will have at source offset k (a number macro there is expanded)."""
        for _ in range(64):
            if k >= len(text):
                return ""
            if text[k] == "\\":
                kind, s, e = next_token(text, k, len(text))
                if kind == "cw" and text[s + 1 : e] in self.macros:
                    exp = self.expansion(text[s + 1 : e], stack)
                    if exp:
                        return exp[0]
                    k = call_extent(text, e, len(text))[0]
                    continue
            return text[k]
        return ""

    def brace_after(
        self, prefix: str, exp: str, text: str, k: int, braced: bool, stack: tuple[str, ...], where: object
    ) -> str:
        """ "{}" when the written expansion would leave a control word that the next character extends; else "".

        The next character extends the control word when it is a letter (SPEC rule) or, after a call that consumed
        "{}", a blank, which TeX would otherwise skip although the source had a space token there. An expansion that
        starts with a letter right after a control word would merge with it; that has no faithful flat form.
        """
        if exp and exp[0] in LETTERS and ends_in_control_word(prefix):
            raise FlattenError(
                f"{where}: a number macro whose value starts with a letter follows a control word directly; write "
                f"a space or {{}} between them in the source"
            )
        if not ends_in_control_word(exp if exp else prefix):
            return ""
        nxt = self.next_flat_char(text, k, stack)
        if nxt in LETTERS or (braced and nxt in " \t\n"):
            return "{}"
        return ""

    # -- emitting ----------------------------------------------------------------------------------------------

    def emit(
        self,
        text: str,
        kind: str,
        file: str | None = None,
        src: tuple[int, int] | None = None,
        macro: str | None = None,
        generator: str | None = None,
    ) -> None:
        """Append text to the flat file and record its segment (consecutive verbatim stretches merge)."""
        a, b = self.pos, self.pos + len(text)
        last = self.segments[-1] if self.segments else None
        if (
            kind in ("src", "gen")
            and macro is None
            and last is not None
            and last.kind == kind
            and last.macro is None
            and last.file == file
            and last.b == a
            and last.src[1] == src[0]
        ):
            last.b, last.src = b, (last.src[0], src[1])
        else:
            self.segments.append(Segment(a, b, kind, file, src, macro, generator))
        if text:
            self.out.append(text)
            self.pos = b
            self.flat_tail = (self.flat_tail + text)[-256:]

    def verbatim(self, rel: str, s: int, e: int) -> None:
        if e > s:
            kind = "src" if self.kinds[rel] == "hand" else "gen"
            self.emit(self.texts[rel][s:e], kind, rel, (s, e), None, self.generators[rel])

    def drop(self, rel: str, run: int, s: int, e: int) -> int:
        """Drop source text[s:e] (a number-file \\input or a guard): the whole line when nothing else is on it.

        Returns the offset where the next verbatim stretch starts.
        """
        text = self.texts[rel]
        ls, le = line_bounds(text, s)
        if run <= ls and not text[ls:s].strip(BLANKS) and not text[e:le].strip(BLANKS):
            s, e = ls, min(le + 1, len(text))
        self.verbatim(rel, run, s)
        self.emit("", "drop", rel, (s, e), None, self.generators[rel])
        return e

    # -- the pass over a file ----------------------------------------------------------------------------------

    def process(self, rel: str) -> None:
        """Write the flat text of one hand-written or generated file, recursing into its live \\input's."""
        text = self.texts[rel]
        n = len(text)
        kind = self.kinds[rel]
        run = 0  # start of the pending verbatim stretch
        last_cw, last_cw_end = "", -1
        i = 0
        while i < n:
            tok, s, e = next_token(text, i, n)
            if tok != "cw":
                i = e
                continue
            name = text[s + 1 : e]
            where = SourceLocation(rel, text, s)
            if name in self.macros:
                if last_cw in NON_EXPANDING and re.fullmatch(r"\*?\s*\{?\s*", text[last_cw_end:s]):
                    raise FlattenError(f"{where}: \\{last_cw} reads \\{name} unexpanded; it cannot be replaced")
                if e < n and text[e] == "@":
                    raise FlattenError(f"{where}: \\{name}@... depends on the catcode of @; rename it")
                self.verbatim(rel, run, s)
                k, braced = call_extent(text, e, n)
                exp = self.macros[name].expanded
                exp += self.brace_after(self.flat_tail, exp, text, k, braced, (), where)
                if kind == "hand":
                    self.emit(exp, "macro", rel, (s, k), name, self.macros[name].generator)
                else:
                    self.emit(exp, "gen", rel, (s, k), name, self.generators[rel])
                run = i = k
                last_cw = ""
                continue
            if name == "input":
                arg, g = self.input_argument(text, e, where)
                target = self.resolve_input(arg, where)
                if target in self.number_files:
                    run = i = self.drop(rel, run, s, g)
                else:
                    self.verbatim(rel, run, s)
                    self.emit("", "drop", rel, (s, g), None, self.generators[rel])
                    self.inline(target, text, g, where)
                    run = i = g
                last_cw = ""
                continue
            if name == "ifdefined":
                m = GUARD.match(text, s)
                if m:
                    target = self.resolve_input(m.group(2), where)
                    if target in self.number_files:
                        run = i = self.drop(rel, run, s, m.end())
                        last_cw = ""
                        continue
            if name in UNSUPPORTED:
                raise FlattenError(f"{where}: live \\{name} is not supported by flatten.py")
            if name == "includegraphics":
                self.note_figure(text, e, where)
            elif name == "graphicspath":
                j = skip_blank(text, e, n)
                if j < n and text[j] == "{":
                    g = read_group(text, j, n)
                    self.graphicspath = re.findall(r"\{([^{}]*)\}", text[j + 1 : g - 1])
            elif name == "bibliography":
                j = skip_blank(text, e, n)
                if j < n and text[j] == "{":
                    g = read_group(text, j, n)
                    names = []
                    for b in text[j + 1 : g - 1].split(","):
                        b = b.strip()
                        if b:
                            b = posixpath.normpath(b if b.endswith(".bib") else b + ".bib")
                            names.append(b)
                            if b not in self.bibs:
                                self.bibs.append(b)
                    if self.bib_suffix and names:
                        # The written argument is dropped and the renamed list is synthetic text, so an Overleaf
                        # edit of it is a structural conflict, not an edit of the source's \bibliography.
                        self.verbatim(rel, run, j + 1)
                        self.emit("", "drop", rel, (j + 1, g - 1), None, self.generators[rel])
                        self.emit(",".join(self.out_bib(b)[: -len(".bib")] for b in names), "synthetic")
                        run = g - 1
                        last_cw, last_cw_end = name, e
                        i = g - 1
                        continue
            last_cw, last_cw_end = name, e
            i = e
        self.verbatim(rel, run, n)

    def inline(self, target: str, parent: str, call_end: int, where: object) -> None:
        """Inline a file between marker lines; TeX reads the file, then the rest of the calling line."""
        if target in self.stack:
            raise FlattenError(f"{where}: \\input cycle through {target}")
        kind = "generated" if target.startswith("generated/") else "hand"
        self.read(target, kind)
        if self.kinds[target] == "numbers":
            raise FlattenError(f"{where}: {target} is a number file")
        self.inline_count[target] += 1
        gen = self.generators[target]
        label = "hand-written" if kind == "hand" else f"generated by {gen or 'an unknown generator'}"
        self.emit(f"%% >>> begin {target} ({label})\n", "synthetic")
        self.stack.append(target)
        self.process(target)
        self.stack.pop()
        # The end marker is a comment line. When only blanks remain on the calling line (the usual case: the \input
        # stands on its own line), the marker ends with that line's blanks and newline, which stay src text of the
        # calling file: a marker with a newline of its own would put an empty line, a \par, where TeX reads a space.
        # Otherwise the marker ends with its own newline and the rest of the calling line follows on the next line.
        le = parent.find("\n", call_end)
        rest = parent[call_end : (len(parent) if le < 0 else le)]
        marker = ("" if self.flat_tail.endswith("\n") else "\n") + f"%% <<< end {target}"
        if rest.strip(BLANKS):
            marker += "\n"
        self.emit(marker, "synthetic")

    def note_figure(self, text: str, e: int, where: object) -> None:
        """Record the file of a live \\includegraphics[...]{path}, resolved as graphicx resolves it."""
        n = len(text)
        j = e
        if j < n and text[j] == "*":
            j += 1
        j = skip_blank(text, j, n)
        while j < n and text[j] == "[":
            j = skip_blank(text, read_bracket(text, j, n), n)
        if j >= n or text[j] != "{":
            return
        g = read_group(text, j, n)
        path = text[j + 1 : g - 1].strip()
        if "\\" in path:
            self.warnings.append(f"{where}: \\includegraphics{{{path}}} uses a macro; figure not copied")
            return
        ext = posixpath.splitext(path)[1].lower()
        names = [path] if ext in KNOWN_GRAPHICS_EXTS else [path + x for x in GRAPHICS_EXTS] + [path]
        for name in names:  # graphicx: extensions outer, search path inner
            for d in [""] + self.graphicspath:
                cand = posixpath.normpath(posixpath.join(d, name))
                if (self.root / cand).is_file():
                    if cand.startswith("../") or posixpath.isabs(cand):
                        self.warnings.append(
                            f"{where}: figure {cand} lies outside {self.root.name}/; Overleaf cannot keep that "
                            f"path, not copied"
                        )
                    elif cand not in self.figures:
                        self.figures.append(cand)
                    return
        self.warnings.append(f"{where}: \\includegraphics{{{path}}}: no such file; not copied")

    def out_bib(self, rel: str) -> str:
        """The name a .bib file is written under in the output directory (bib_suffix before the extension)."""
        return posixpath.splitext(rel)[0] + self.bib_suffix + ".bib"

    # -- the whole document ------------------------------------------------------------------------------------

    def run(self, header: str) -> str:
        """Flatten the entry file below the given header; returns the flat text."""
        self.load_numbers()
        self.emit(header, "synthetic")
        self.inline_count[self.entry_rel] += 1
        self.stack.append(self.entry_rel)
        self.process(self.entry_rel)
        self.stack.pop()
        if not self.bibs:
            self.bibs.append("references.bib")
        return "".join(self.out)

    # -- invariants (SPEC section 4) ---------------------------------------------------------------------------

    def validate(self, flat: str) -> None:
        """Check the segment invariants of SPEC section 4 (and full source coverage); raise on any violation."""
        problems: list[str] = []

        def bad(msg: str) -> None:
            if len(problems) < 20:
                problems.append(msg)

        pos = 0
        last_a = -1
        covered: dict[str, list[tuple[int, int]]] = collections.defaultdict(list)
        for idx, sg in enumerate(self.segments):
            tag = f"segment {idx} ({sg.kind}, flat {sg.a}..{sg.b})"
            if sg.a < last_a:
                bad(f"{tag}: not sorted by flat start")
            last_a = sg.a
            if sg.b < sg.a:
                bad(f"{tag}: negative length")
            if sg.b > sg.a:
                if sg.a != pos:
                    bad(f"{tag}: does not start where the previous segment ends ({pos})")
                pos = sg.b
            elif sg.a != pos:
                bad(f"{tag}: zero-length segment off the tiling ({pos})")
            piece = flat[sg.a : sg.b]
            if sg.kind == "synthetic":
                if sg.file is not None or sg.src is not None:
                    bad(f"{tag}: synthetic text with a source")
                continue
            if sg.file not in self.texts or sg.src is None:
                bad(f"{tag}: unknown source")
                continue
            s, e = sg.src
            source = self.texts[sg.file]
            if not 0 <= s <= e <= len(source):
                bad(f"{tag}: source range {s}..{e} outside {sg.file}")
                continue
            covered[sg.file].append((s, e))
            if sg.kind == "src":
                if self.kinds[sg.file] != "hand":
                    bad(f"{tag}: src segment from {sg.file}, which is not hand-written")
                if piece != source[s:e]:
                    bad(f"{tag}: flat text differs from {sg.file}[{s}:{e}]")
            elif sg.kind == "drop":
                if sg.a != sg.b:
                    bad(f"{tag}: drop with flat text")
            elif sg.kind in ("macro", "gen"):
                if sg.kind == "gen" and (self.kinds[sg.file] != "generated" or not sg.generator):
                    bad(f"{tag}: gen segment without a generated file and its generator")
                if sg.macro is None:
                    if sg.kind == "macro":
                        bad(f"{tag}: macro segment without a macro")
                    elif piece != source[s:e]:
                        bad(f"{tag}: flat text differs from {sg.file}[{s}:{e}]")
                    continue
                m = self.macros.get(sg.macro)
                if m is None:
                    bad(f"{tag}: unknown macro {sg.macro}")
                    continue
                if piece not in (m.expanded, m.expanded + "{}"):
                    bad(f"{tag}: flat text {piece!r} is not the expansion of \\{sg.macro}")
                call = source[s:e]
                after = source[s + 1 + len(sg.macro) : s + 2 + len(sg.macro)]
                if not call.startswith("\\" + sg.macro) or (after and after in LETTERS):
                    bad(f"{tag}: source {call!r} is not a call of \\{sg.macro}")
                if m.expanded != self.expand_text(m.value, (sg.macro,), sg.macro):
                    bad(f"{tag}: stored expansion of \\{sg.macro} is stale")
            else:
                bad(f"{tag}: unknown kind")
        if pos != len(flat):
            bad(f"segments end at {pos}, the flat text at {len(flat)}")
        # Every character of every inlined file is accounted for once each time the file is inlined.
        for rel, count in self.inline_count.items():
            ranges = sorted(covered.get(rel, []))
            total = sum(e - s for s, e in ranges)
            if total != count * len(self.texts[rel]):
                bad(f"{rel}: segments cover {total} source characters, expected {count} x {len(self.texts[rel])}")
            elif count == 1:
                at = 0
                for s, e in ranges:
                    if s != at:
                        bad(f"{rel}: source characters {at}..{s} not accounted for")
                        break
                    at = e
        if problems:
            raise FlattenError("segment invariants violated:\n  " + "\n  ".join(problems))

    # -- the map -----------------------------------------------------------------------------------------------

    def map_document(self, flat: str, commit: str | None, dirty: bool) -> dict:
        def sha(t: str) -> str:
            return hashlib.sha256(t.encode("utf-8")).hexdigest()

        order = list(self.inline_count) + [f for f in self.number_files if f not in self.inline_count]
        return {
            "version": 1,
            "created": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "entry": self.entry_rel,
            "source_commit": commit,
            "source_dirty": dirty,
            "flat_file": self.flat_name,
            "flat_sha256": sha(flat),
            "flat_text": flat,
            "sources": {
                rel: {"sha256": sha(self.texts[rel]), "text": self.texts[rel], "kind": self.kinds[rel]} for rel in order
            },
            "macros": {
                m.name: {"value": m.value, "expanded": m.expanded, "file": m.file, "generator": m.generator}
                for m in self.macros.values()
            },
            "segments": [sg.as_json() for sg in self.segments],
        }


def dump_map(doc: dict) -> str:
    """The map as JSON with one source, one macro and one segment a line (stable, diff-friendly)."""

    def j(v) -> str:
        return json.dumps(v, ensure_ascii=False)

    parts = []
    for k, v in doc.items():
        if k in ("sources", "macros"):
            body = ",\n".join(f"  {j(kk)}: {j(vv)}" for kk, vv in v.items())
            parts.append(f" {j(k)}: {{\n{body}\n }}")
        elif k == "segments":
            body = ",\n".join(f"  {j(sv)}" for sv in v)
            parts.append(f" {j(k)}: [\n{body}\n ]")
        else:
            parts.append(f" {j(k)}: {j(v)}")
    text = "{\n" + ",\n".join(parts) + "\n}\n"
    if json.loads(text) != doc:
        raise FlattenError("map JSON does not round-trip")
    return text


# ----------------------------------------------------------------------------------------------------------------
# git, header
# ----------------------------------------------------------------------------------------------------------------


def git(root: Path, *args: str) -> str | None:
    try:
        r = subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return None
    return r.stdout if r.returncode == 0 else None


def git_state(root: Path, rels: list[str]) -> tuple[str | None, bool]:
    """(HEAD sha, dirty): dirty when a tracked file differs from HEAD or a source read is not tracked."""
    sha = git(root, "rev-parse", "HEAD")
    if sha is None:
        return None, False
    tracked_changes = git(root, "status", "--porcelain", "--untracked-files=no") or ""
    untracked = git(root, "ls-files", "--others", "--", *rels) or ""
    return sha.strip(), bool(tracked_changes.strip() or untracked.strip())


def header_text(entry: str, commit: str | None, dirty: bool, name: str = "main.tex") -> str:
    """The synthetic header comment at the top of the flat file (SPEC section 3, rule 5)."""
    rev = (commit or "unknown") + ("+dirty" if dirty else "")
    rule = "%% " + "=" * 113 + "\n"
    return (
        rule + f"%% {name} -- GENERATED by {SCRIPT_NAME} from {entry}\n"
        f"%% at commit {rev}.\n"
        "%% A flat copy of the paper for Overleaf: the section files and generated tables are inlined between\n"
        '%% "%% >>> begin" / "%% <<< end" marker lines, and every number macro is replaced by its value.\n'
        "%% Edit it like any LaTeX file: writeup/overleaf_sync/unflatten.py brings the edits back to the hand-written\n"
        "%% sources (paper_rehaul.tex, rehaul/*.tex), parking the replaced text in dated comments.\n"
        "%% Numbers and tables come from generators (writeup/make_*.py, experiments/*.py): an edit to them here is\n"
        "%% flagged as a conflict and not applied; change the generator instead. Leave the marker lines alone.\n" + rule
    )


# ----------------------------------------------------------------------------------------------------------------
# --check
# ----------------------------------------------------------------------------------------------------------------

WARNING_START = re.compile(
    r"(LaTeX Warning|LaTeX Font Warning|Package \S+ Warning|Class \S+ Warning|pdfTeX warning|"
    r"Overfull \\[hv]box|Underfull \\[hv]box|Missing character)"
)


def normalize_warning(line: str) -> str:
    line = re.sub(r"on input line \d+", "on input line N", line)
    line = re.sub(r"at lines? \d+(--\d+)?", "at line N", line)
    line = re.sub(r"detected at line \d+", "detected at line N", line)
    return line.strip()


def log_warnings(log: str) -> collections.Counter[str]:
    out: collections.Counter[str] = collections.Counter()
    for line in log.splitlines():
        m = WARNING_START.search(line)
        if m:
            out[normalize_warning(line[m.start() :])] += 1
    return out


@functools.cache
def is_miktex() -> bool:
    try:
        r = subprocess.run(["pdflatex", "--version"], capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return False
    return "MiKTeX" in r.stdout


def build(workdir: Path, job: str) -> tuple[Path, str]:
    """pdflatex (draft), bibtex, pdflatex (draft), pdflatex in workdir; returns (pdf, final log text)."""
    env = dict(os.environ, max_print_line="10000", error_line="254", half_error_line="238")
    # 2026-10-27: MiKTeX ignores those variables (TeX Live reads them) and wraps the log at 79 columns, so a warning
    # "... on input line NNNN." broke at a different place in the two logs and counted as new; it takes them as flags.
    flags = ["-max-print-line=10000", "-error-line=254", "-half-error-line=238"] if is_miktex() else []
    latex = ["pdflatex", "-interaction=nonstopmode", *flags, f"{job}.tex"]
    # 2026-10-28 (author: update Overleaf faster): the first two passes in -draftmode (aux files written, no PDF);
    # only the last pass writes the PDF, whose text and log the check compares.
    draft = ["pdflatex", "-interaction=nonstopmode", "-draftmode", *flags, f"{job}.tex"]
    for cmd in (draft, ["bibtex", job], draft, latex):
        subprocess.run(
            cmd,
            cwd=workdir,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=1200,
        )
    log = workdir / f"{job}.log"
    return workdir / f"{job}.pdf", (log.read_text(encoding="latin-1") if log.is_file() else "")


def pdf_text_and_pages(pdf: Path) -> tuple[str, int]:
    txt = pdf.with_suffix(".txt")
    subprocess.run(["pdftotext", str(pdf), str(txt)], check=True, capture_output=True)
    info = subprocess.run(["pdfinfo", str(pdf)], check=True, capture_output=True, text=True).stdout
    pages = int(re.search(r"^Pages:\s+(\d+)", info, re.M).group(1))
    return txt.read_text(encoding="utf-8", errors="replace"), pages


def check(fl: Flattener, out_dir: Path, flat: str, keep: bool) -> bool:
    """The acceptance test of SPEC section 3. Prints a report; True when everything passes."""
    ok = True
    print("--check")
    # Static checks on main.tex: no live \input / \include / \InputIfFileExists, no live number-macro control word.
    forbidden = collections.Counter()
    leftover = collections.Counter()
    for name, _, _ in live_control_words(flat):
        if name in ("input", "include", "InputIfFileExists"):
            forbidden[name] += 1
        if name in fl.macros:
            leftover[name] += 1
    print(f"  live \\input / \\include / \\InputIfFileExists in main.tex: {dict(forbidden) or 'none'}")
    print(f"  live number-macro control words in main.tex: {dict(leftover.most_common(10)) or 'none'}")
    ok &= not forbidden and not leftover

    tmp = Path(tempfile.mkdtemp(prefix="flatten_check_"))
    try:
        flat_dir = tmp / "flat"
        flat_dir.mkdir()
        for rel in [fl.flat_name] + [fl.out_bib(b) for b in fl.bibs] + fl.figures:
            src = out_dir / rel
            if src.is_file():
                (flat_dir / rel).parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(src, flat_dir / rel)
        paper_dir = tmp / fl.root.name
        shutil.copytree(fl.root, paper_dir, ignore=shutil.ignore_patterns("__pycache__", "*.zip", SCRIPT_DIR.name))
        job = Path(fl.entry_rel).stem
        for ext in (".aux", ".bbl", ".blg", ".out", ".toc", ".lof", ".lot", ".log", ".pdf", ".fls"):
            (paper_dir / (job + ext)).unlink(missing_ok=True)  # both builds start from scratch
        print(f"  building main.tex in {flat_dir} and {fl.entry_rel} in {paper_dir} ...", flush=True)
        # 2026-10-28 (author: update Overleaf faster): the two builds run at the same time (separate directories)
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            f_main = pool.submit(build, flat_dir, Path(fl.flat_name).stem)
            f_paper = pool.submit(build, paper_dir, job)
            main_pdf, main_log = f_main.result()
            paper_pdf, paper_log = f_paper.result()
        if not main_pdf.is_file() or not paper_pdf.is_file():
            print(f"  FAIL: no PDF (main.pdf: {main_pdf.is_file()}, {job}.pdf: {paper_pdf.is_file()})")
            return False
        main_txt, main_pages = pdf_text_and_pages(main_pdf)
        paper_txt, paper_pages = pdf_text_and_pages(paper_pdf)
        same_text = main_txt == paper_txt
        same_pages = main_pages == paper_pages
        print(f"  pages: main.pdf {main_pages}, {job}.pdf {paper_pages} ({'equal' if same_pages else 'DIFFERENT'})")
        print(f"  pdftotext identical: {'yes' if same_text else 'NO'} ({len(main_txt)} vs {len(paper_txt)} characters)")
        if not same_text:
            diff = list(
                difflib.unified_diff(
                    paper_txt.splitlines(), main_txt.splitlines(), f"{job}.txt", "main.txt", lineterm="", n=1
                )
            )
            print("    " + "\n    ".join(diff[:60]))
        ok &= same_text and same_pages

        undefined = main_log.count("Undefined control sequence")
        errors_main = collections.Counter(ln for ln in main_log.splitlines() if ln.startswith("! "))
        errors_paper = collections.Counter(ln for ln in paper_log.splitlines() if ln.startswith("! "))
        print(f"  'Undefined control sequence' in main.log: {undefined}")
        new_errors = errors_main - errors_paper
        if new_errors:
            print(f"  errors in main.log only: {dict(new_errors)}")
        ok &= undefined == 0 and not new_errors
        new_warn = log_warnings(main_log) - log_warnings(paper_log)
        gone_warn = log_warnings(paper_log) - log_warnings(main_log)
        print(f"  warnings in main.log but not in {job}.log: {sum(new_warn.values()) or 'none'}")
        for w, c in sorted(new_warn.items())[:30]:
            print(f"    {c} x {w}")
        if gone_warn:
            print(f"  (warnings in {job}.log but not in main.log: {sum(gone_warn.values())})")
            for w, c in sorted(gone_warn.items())[:10]:
                print(f"    {c} x {w}")
        ok &= not new_warn
    finally:
        if keep:
            print(f"  build directories kept in {tmp}")
        else:
            shutil.rmtree(tmp, ignore_errors=True)
    print(f"--check {'PASSED' if ok else 'FAILED'}")
    return ok


# ----------------------------------------------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Flatten writeup/paper_rehaul.tex into one Overleaf main.tex plus a segment map.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="See writeup/overleaf_sync/SPEC.md.",
    )
    ap.add_argument("--out", required=True, type=Path, help="output directory for main.tex, references.bib, figures")
    ap.add_argument("--map", type=Path, default=DEFAULT_MAP, help="map file (default: %(default)s)")
    ap.add_argument("--entry", type=Path, default=DEFAULT_ENTRY, help="paper to flatten (default: %(default)s)")
    ap.add_argument("--check", action="store_true", help="build both PDFs and compare their text (SPEC section 3)")
    ap.add_argument("--keep-builds", action="store_true", help="with --check: keep the temporary build directories")
    ap.add_argument("--name", default="main.tex", help="file name of the flat paper in --out (default: %(default)s)")
    ap.add_argument(
        "--bib-suffix",
        default="",
        help="write each .bib as <name><suffix>.bib and point \\bibliography at it (e.g. 2: references2.bib)",
    )
    ap.add_argument(
        "--figures-out",
        type=Path,
        default=None,
        help="write the figures copied into --out, one path relative to --out a line (overleaf_git.sh commits them)",
    )
    args = ap.parse_args(argv)

    fl = Flattener(args.entry, args.name, args.bib_suffix)
    try:
        # The header names the commit, and the dirty flag depends on which sources are read: flatten once to learn
        # them, then again with the final header (the second pass is cheap and gives the same segments).
        probe = Flattener(args.entry, args.name, args.bib_suffix)
        probe.run("")
        commit, dirty = git_state(probe.root, list(probe.texts))
        flat = fl.run(header_text(f"{fl.root.name}/{fl.entry_rel}", commit, dirty, fl.flat_name))
        fl.validate(flat)
        doc = fl.map_document(flat, commit, dirty)
        map_text = dump_map(doc)
    except FlattenError as exc:
        print(f"flatten.py: {exc}", file=sys.stderr)
        return 2

    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / fl.flat_name, "w", encoding="utf-8", newline="") as fh:
        fh.write(flat)
    copied_figures: list[str] = []
    for rel, out_rel in [(b, fl.out_bib(b)) for b in fl.bibs] + [(f, f) for f in fl.figures]:
        src = fl.root / rel
        if not src.is_file():
            fl.warnings.append(f"{rel}: not found; not copied")
            continue
        (out_dir / out_rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, out_dir / out_rel)
        if rel in fl.figures:
            copied_figures.append(out_rel)
    # 2026-10-24: the caller commits the figures from this list (they were copied but never committed, so Overleaf
    # compiled the flat paper without them).
    if args.figures_out is not None:
        with open(args.figures_out, "w", encoding="utf-8", newline="\n") as fh:
            fh.write("".join(f"{f}\n" for f in copied_figures))
    args.map.parent.mkdir(parents=True, exist_ok=True)
    with open(args.map, "w", encoding="utf-8", newline="") as fh:
        fh.write(map_text)

    kinds = collections.Counter(sg.kind for sg in fl.segments)
    calls = [sg for sg in fl.segments if sg.macro is not None]
    print(f"wrote {out_dir / fl.flat_name}: {len(flat.encode('utf-8'))} bytes, {flat.count(chr(10))} lines")
    print(f"wrote {args.map}: {len(fl.segments)} segments ({', '.join(f'{k} {v}' for k, v in sorted(kinds.items()))})")
    print(
        f"number files: {len(fl.number_files)}; number macros resolved: {len(fl.macros)}; calls replaced: "
        f"{len(calls)} ({len({sg.macro for sg in calls})} distinct macros)"
    )
    print(
        f"inlined: {len(fl.inline_count) - 1} files; bib: {', '.join(fl.out_bib(b) for b in fl.bibs)}; "
        f"figures: {len(fl.figures)}"
    )
    print(f"source commit {commit}{'+dirty' if dirty else ''}")
    for w in fl.warnings:
        print(f"warning: {w}")
    if args.check:
        return 0 if check(fl, out_dir, flat, args.keep_builds) else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
