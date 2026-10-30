"""Code checks for the paper-prose skill's audit stage (B6-B8 of writeup/reports/Academic writing skills comparison.md).

The script runs the Detection checks of the pattern files in writeup/style/patterns/ whose lines code can check, so the
drafter no longer grades itself: code finds the hits and the candidates, and the auditors confirm the candidates.

    python writeup/style/prose_check.py lint TARGET... [--json]
    python writeup/style/prose_check.py diff --base REV [--target FILE:START-END ...] [--log EDITLOG.jsonl]
                                             [--author-record FILE] [--json] FILE...

TARGET is FILE or FILE:START-END (1-based, inclusive lines). FILE is a .tex file, or a generator .py file, in which
only string literals holding LaTeX prose are read (docstrings and plain strings are not).

lint checks (one id each; per-hit rows unless the row's severity is rate):
    V2   typed digits in live text outside macros, math, \\ref, \\cite, \\label and the C2 whitelist (gate); number
         words ("six classes") and numbers typed into math after a relation ("$p = 0.021$") as candidates
    V3   names the notation guide replaces (its section 5, the 6b amendments, the addenda's B.1-B.3 clashes) (gate);
         a bare "benchmark" as a candidate
    V5   Rule NT lexicon: universals, completeness adverbs, superlatives, boosters (candidates)
    V6   clause-final tails of at most five words after , ; : or a dash, holding a quantifier or verdict (candidates)
    V9   section titles with a colon, a question mark or a clause (candidates)
    V10  an inanimate subject with an evaluative verb (candidates)
    AP-V vocabulary rows AP-V1..AP-V3, minus the field whitelist (AP-V2 is a rate)
    AP-S structural rows AP-S1, AP-S2, AP-S4..AP-S8 (AP-S3 is reported as V6), and the sentence-length row "AP-S"

diff checks, sentence-aligned between FILE at REV (git show) and the working tree:
    V16  per changed sentence block: the multisets of macros, \\cite keys and \\ref labels must be equal (gate);
         negations, modals and direction words are diffed and reported (candidates)
    V14  sentences changed outside every --target range (only when --target is given)
    V13  every \\auth{...} span unchanged; with --author-record, every recorded author string present byte for byte
    V12  with --log (JSONL of {file, line, op, rule_id}): every changed sentence has a logged op with a rule id
    V15  candidates: changed sentences whose base lines were last written by an Overleaf pull commit (git blame; a pull
         commit is one that commits writeup/overleaf_sync/map_pulled.json)

Both modes skip %-comments, \\iffalse...\\fi blocks, comment environments and the preamble, and never report inside
\\auth{...}. Output: one line per row, "file:line ID severity message"; with --json a JSON list of objects
{"mode", "id", "file", "line", "severity", "candidate", "match", "message"}, rate rows adding "per_1000".
Severity comes from each pattern file's "**Severity:**" line (anti-pattern files: the Severity column of each row),
else from DEFAULT_SEVERITY. Candidates go to an auditor and never fail the run by themselves. Exit status: 1 if any
non-candidate gate row, 2 on a usage or git error, else 0. Standard library only.
"""

from __future__ import annotations

import argparse
import bisect
import difflib
import io
import json
import re
import subprocess
import sys
import tokenize
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

PATTERNS_DIR = Path(__file__).resolve().parent / "patterns"
BOUNDARY = "\x1e"  # paragraph, item, title or environment break in the prose view: no sentence crosses it
MATH = "<math>"

# Fallback when a pattern file has no Severity line yet (or no Severity column, for the anti-pattern rows).
DEFAULT_SEVERITY: dict[str, str] = {
    "V1": "review",
    "V2": "gate",
    "V3": "gate",
    "V4": "gate",
    "V5": "review",
    "V6": "gate",
    "V7": "review",
    "V8": "review",
    "V9": "review",
    "V10": "review",
    "V11": "review",
    "V12": "gate",
    "V13": "gate",
    "V14": "gate",
    "V15": "gate",
    "V16": "gate",
    "AP-V1": "gate",
    "AP-V2": "rate",
    "AP-V3": "review",
    "AP-S1": "rate",
    "AP-S2": "review",
    "AP-S3": "review",
    "AP-S4": "rate",
    "AP-S5": "rate",
    "AP-S6": "review",
    "AP-S7": "rate",
    "AP-S8": "rate",
    "AP-S9": "review",
    "AP-S10": "review",
}
# The sentence-length item of AP-S ("Measured or catalogued, with no rule": rate, dashboard only, never a gate) has no
# row id; it is reported as "AP-S" and is always a rate.
LENGTH_ID = "AP-S"
LONG_SENTENCE = 40  # words; the length row counts sentences longer than this

_SEV_LINE = re.compile(r"\*\*Severity:\*\*\s*(gate|review|rate)\b", re.I)
_SEV_WORD = re.compile(r"\b(gate|review|rate)\b", re.I)
_PATTERN_FILE = re.compile(r"^(AP-[A-Z]|V\d+)-")
_ROW_ID = re.compile(r"^AP-[A-Z]\d+$")


def load_severities(pattern_dir: Path = PATTERNS_DIR) -> tuple[dict[str, str], dict[str, str]]:
    """Severity of every check id, and where it came from (a pattern file's name, or "built-in")."""
    sev = dict(DEFAULT_SEVERITY)
    source = {k: "built-in" for k in sev}
    if not pattern_dir.is_dir():
        return sev, source
    for md in sorted(pattern_dir.rglob("*.md")):
        m = _PATTERN_FILE.match(md.name)
        if not m:
            continue
        pid = m.group(1)
        try:
            text = md.read_text(encoding="utf-8")
        except OSError:
            continue
        if pid.startswith("AP-"):
            for rid, s in _table_severities(text).items():
                sev[rid] = s
                source[rid] = md.name
            continue
        hit = _SEV_LINE.search(text)
        if hit:
            sev[pid] = hit.group(1).lower()
            source[pid] = md.name
    return sev, source


def _table_severities(text: str) -> dict[str, str]:
    """{row id: severity} from the markdown tables of an anti-pattern file that have a Severity column."""
    out: dict[str, str] = {}
    col: int | None = None
    for raw in text.splitlines():
        line = raw.strip()
        if not line.startswith("|"):
            col = None
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if col is None:
            names = [c.strip("*` ").lower() for c in cells]
            col = names.index("severity") if "severity" in names else -1
            continue
        if col < 0 or col >= len(cells) or not set(line) - set("|-: "):
            continue
        rid = cells[0].strip("*` ")
        word = _SEV_WORD.search(cells[col])
        if _ROW_ID.match(rid) and word:
            out[rid] = word.group(1).lower()
    return out


# --------------------------------------------------------------------------------------------------------- sources


@dataclass
class Src:
    """Text to check, with the 1-based source line of each character."""

    text: str
    line: list[int]


def tex_src(text: str) -> Src:
    lines: list[int] = []
    for n, chunk in enumerate(text.split("\n"), start=1):
        lines.extend([n] * (len(chunk) + 1))
    return Src(text, lines[: len(text)])


_PY_ESCAPES = {"\\": "\\", "'": "'", '"': '"', "n": "\n", "t": " ", "r": "", "a": "", "b": "", "f": "", "v": ""}
PY_VALUE = "\\pyvalue"  # stands for a formatted value {expr} of an f-string: a computed number, never a typed one


_PY_MESSAGE_CALLS = {"print", "warn", "warning", "info", "debug", "error", "exception", "critical", "log", "exit"}


def py_srcs(text: str) -> list[Src]:
    """The string literals of a generator script that hold LaTeX prose, as sources with their own line numbers.

    Implicitly concatenated literals form one source. Docstrings and bare string statements are skipped, and so are
    console messages (strings inside print(...), logging calls, exceptions, assert and raise statements) and every
    literal without a LaTeX command or with fewer than four words of text.
    """
    try:
        toks = list(tokenize.generate_tokens(io.StringIO(text).readline))
    except (tokenize.TokenError, SyntaxError, IndentationError):
        return []
    out: list[Src] = []
    statement_start = {tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT, tokenize.ENCODING}
    prev: tokenize.TokenInfo | None = None
    calls: list[str] = []  # the callee of each open bracket ("" for a bracket that is not a call)
    keyword = ""  # the first name of the current statement
    i = 0
    while i < len(toks):
        tok = toks[i]
        if tok.type != tokenize.STRING:
            if tok.type == tokenize.OP and tok.string in "([{":
                callee = prev.string if prev is not None and prev.type == tokenize.NAME and tok.string == "(" else ""
                calls.append(callee)
            elif tok.type == tokenize.OP and tok.string in ")]}" and calls:
                calls.pop()
            if tok.type == tokenize.NAME and (prev is None or prev.type in statement_start):
                keyword = tok.string
            if tok.type not in (tokenize.NL, tokenize.COMMENT):
                prev = tok
            i += 1
            continue
        message = keyword in ("assert", "raise") or any(
            c in _PY_MESSAGE_CALLS or c.endswith(("Error", "Exception", "Warning")) for c in calls
        )
        group = [tok]
        j = i + 1
        while j < len(toks) and toks[j].type in (tokenize.STRING, tokenize.NL, tokenize.COMMENT):
            if toks[j].type == tokenize.STRING:
                group.append(toks[j])
            j += 1
        nxt = toks[j] if j < len(toks) else None
        bare = (prev is None or prev.type in statement_start) and (
            nxt is None or nxt.type in (tokenize.NEWLINE, tokenize.ENDMARKER)
        )
        if not bare and not message:
            src = _py_group(group)
            if _is_latex_prose(src.text):
                out.append(src)
        prev = group[-1]
        i = j
    return out


def _py_group(group: list[tokenize.TokenInfo]) -> Src:
    chars: list[str] = []
    lines: list[int] = []
    for tok in group:
        prefix = re.match(r"[A-Za-z]*", tok.string).group()  # type: ignore[union-attr]
        rest = tok.string[len(prefix) :]
        quote = rest[:3] if rest[:3] in ('"""', "'''") else rest[:1]
        body = rest[len(quote) : len(rest) - len(quote)]
        raw = "r" in prefix.lower()
        fmt = "f" in prefix.lower()
        ln = tok.start[0]
        k = 0
        while k < len(body):
            ch = body[k]
            if not raw and ch == "\\" and k + 1 < len(body):
                nx = body[k + 1]
                if nx == "\n":
                    k += 2
                    ln += 1
                    continue
                if nx in _PY_ESCAPES:
                    for c in _PY_ESCAPES[nx]:
                        chars.append(c)
                        lines.append(ln)
                    k += 2
                    continue
            if fmt and ch in "{}":
                if body.startswith(ch * 2, k):
                    chars.append(ch)
                    lines.append(ln)
                    k += 2
                    continue
                if ch == "{":
                    depth, p = 0, k
                    while p < len(body):
                        depth += {"{": 1, "}": -1}.get(body[p], 0)
                        if depth == 0:
                            break
                        p += 1
                    for c in PY_VALUE + "{}":
                        chars.append(c)
                        lines.append(ln)
                    ln += body.count("\n", k, p)
                    k = p + 1
                    continue
            chars.append(ch)
            lines.append(ln)
            if ch == "\n":
                ln += 1
            k += 1
    return Src("".join(chars), lines)


def _is_latex_prose(text: str) -> bool:
    if not re.search(r"\\[A-Za-z]{2,}|\\%|\$|~", text):
        return False
    plain = re.sub(r"\$[^$]*\$|\\[A-Za-z]+|[{}]", " ", text)
    return len(re.findall(r"\b[A-Za-z]{2,}\b", plain)) >= 4 and " " in text


# ------------------------------------------------------------------------------------------------------- live text


def _line_spans(text: str) -> list[tuple[int, int]]:
    return [m.span() for m in re.finditer(r"[^\n]*\n|[^\n]+$", text)]


def _comment_at(line: str) -> int:
    i = line.find("%")
    while i >= 0:
        k = i
        while k > 0 and line[k - 1] == "\\":
            k -= 1
        if (i - k) % 2 == 0:
            return i
        i = line.find("%", i + 1)
    return -1


def _comment_spans(text: str) -> list[tuple[int, int]]:
    """%-comments, each through the end of its line (the newline included, as TeX eats it), and comment envs."""
    spans = []
    for a, b in _line_spans(text):
        c = _comment_at(text[a:b])
        if c >= 0:
            spans.append((a + c, b))
    for m in re.finditer(r"\\begin\{comment\}.*?(?:\\end\{comment\}|\Z)", text, re.S):
        spans.append(m.span())
    return spans


_CTRL = re.compile(r"\\([A-Za-z]+)")


def _iffalse_spans(text: str) -> list[tuple[int, int]]:
    """\\iffalse ... \\fi blocks (nested conditionals counted; a top-level \\else ends the skipped part)."""
    spans = []
    start, depth, skip_next = -1, 0, False
    for m in _CTRL.finditer(text):
        name = m.group(1)
        if skip_next:
            skip_next = False
            continue
        if name in ("newif", "let"):
            skip_next = True
            continue
        if start < 0:
            if name == "iffalse":
                start, depth = m.start(), 0
            continue
        if name.startswith("if") and name != "ifthenelse":
            depth += 1
        elif name == "fi":
            if depth == 0:
                spans.append((start, m.end()))
                start = -1
            else:
                depth -= 1
        elif name == "else" and depth == 0:
            spans.append((start, m.end()))
            start = -1
    if start >= 0:
        spans.append((start, len(text)))
    return spans


def _blank(chars: list[str], spans: list[tuple[int, int]]) -> None:
    for a, b in spans:
        for k in range(a, b):
            chars[k] = " "


def live_text(text: str) -> str:
    """The text with comments, \\iffalse blocks and the preamble blanked, newlines turned into spaces and each blank
    line (a paragraph break) turned into BOUNDARY. Same length as text, so offsets map to source lines."""
    chars = list(text)
    comments = _comment_spans(text)
    _blank(chars, comments)
    iff = _iffalse_spans("".join(chars))
    _blank(chars, iff)
    skipped = list(comments) + list(iff)
    joined = "".join(chars)
    begin = joined.find("\\begin{document}")
    if begin >= 0:
        skipped.append((0, begin + len("\\begin{document}")))
    end = joined.find("\\end{document}")
    if end >= 0:
        skipped.append((end, len(text)))
    if begin >= 0 or end >= 0:
        _blank(chars, skipped[len(comments) + len(iff) :])
    merged: list[tuple[int, int]] = []
    for a, b in sorted(skipped):
        if merged and a <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], b))
        else:
            merged.append((a, b))
    firsts = [a for a, _ in merged]

    def inside(pos: int) -> bool:
        k = bisect.bisect_right(firsts, pos) - 1
        return k >= 0 and pos < merged[k][1]

    for a, b in _line_spans(text):
        if text[a:b].strip() == "" and text[b - 1 : b] == "\n" and not inside(a):
            chars[b - 1] = BOUNDARY
    return "".join(ch if ch not in "\n\r\t" else " " for ch in chars)


@dataclass
class AuthSpan:
    start: int
    end: int
    ident: str | None
    text: str


def _group_end(s: str, k: int) -> int:
    """Index just past the brace group opening at s[k] == '{' (or len(s) if unbalanced)."""
    depth = 0
    p = k
    while p < len(s):
        ch = s[p]
        if ch == "\\":
            p += 2
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return p + 1
        p += 1
    return len(s)


def auth_spans(live: str) -> list[AuthSpan]:
    """\\auth{id}{text} (or \\auth{text}) spans of live text."""
    out = []
    for m in re.finditer(r"\\auth(?![A-Za-z])\s*\{", live):
        a = m.end() - 1
        b = _group_end(live, a)
        first = live[a + 1 : b - 1]
        k = b
        while k < len(live) and live[k] == " ":
            k += 1
        if k < len(live) and live[k] == "{":
            c = _group_end(live, k)
            out.append(AuthSpan(m.start(), c, first.strip(), live[k + 1 : c - 1]))
        else:
            out.append(AuthSpan(m.start(), b, None, first))
    return out


# ----------------------------------------------------------------------------------------------------------- detex

MATH_ENVS = {
    "equation", "equation*", "align", "align*", "alignat", "alignat*", "gather", "gather*", "multline", "multline*",
    "eqnarray", "eqnarray*", "displaymath", "math", "flalign", "flalign*", "dmath", "dmath*",
}  # fmt: skip
SKIP_ENVS = {
    "tabular", "tabular*", "tabularx", "longtable", "longtable*", "array", "verbatim", "verbatim*", "lstlisting",
    "minted", "tikzpicture", "comment", "thebibliography", "filecontents", "filecontents*", "alltt", "Verbatim",
}  # fmt: skip
ENV_ARGS = {"minipage": 1, "multicols": 1, "adjustbox": 1, "wrapfigure": 2, "subfigure": 1, "list": 2}
SECTION_CMDS = {"part", "chapter", "section", "subsection", "subsubsection", "paragraph", "subparagraph"}
BLOCK_CMDS = {"caption", "footnote", "footnotetext", "thanks", "title"}
TRANSPARENT = {
    "textbf", "textit", "emph", "textsc", "textrm", "textsf", "textup", "textmd", "textnormal", "textsl", "underline",
    "uline", "mbox", "fbox", "enquote", "hl", "makebox", "framebox", "shortstack", "centerline", "boldsymbol",
    "textsuperscript", "textsubscript", "mathrm", "mathit", "mathbf", "mathsf", "mathtt", "text", "operatorname",
}  # fmt: skip
KEEP_NTH = {
    "ser": (2, 1), "serx": (3, 2), "textcolor": (2, 1), "colorbox": (2, 1), "href": (2, 1), "multicolumn": (3, 2),
    "multirow": (3, 2), "parbox": (2, 1), "raisebox": (2, 1), "resizebox": (3, 2), "scalebox": (2, 1),
    "rotatebox": (2, 1), "fcolorbox": (3, 2), "foreignlanguage": (2, 1),
}  # fmt: skip
REF_CMDS = {"ref", "eqref", "pageref", "autoref", "cref", "Cref", "nameref", "vref", "Vref", "cpageref", "labelcref"}
CITE_CMDS = {
    "cite", "citet", "citep", "citealt", "citealp", "citeauthor", "citeyear", "citeyearpar", "citetalias",
    "citepalias", "Citet", "Citep", "Citealt", "Citealp", "Citeauthor", "nocite", "parencite", "textcite", "autocite",
    "footcite",
}  # fmt: skip
TOKEN_CMDS = {"texttt": "<code>", "url": "<url>", "path": "<code>", "sername": "<code>", "nolinkurl": "<url>"}
DROP_CMDS = {
    "label": 1, "input": 1, "include": 1, "includegraphics": 1, "hspace": 1, "vspace": 1, "editc": 3, "cline": 1,
    "setlength": 2, "addtolength": 2, "setcounter": 2, "addtocounter": 2, "stepcounter": 1, "refstepcounter": 1,
    "rule": 2, "usepackage": 1, "RequirePackage": 1, "documentclass": 1, "bibliography": 1, "bibliographystyle": 1,
    "color": 1, "definecolor": 3, "cellcolor": 1, "rowcolor": 1, "arrayrulecolor": 1, "phantom": 1, "hphantom": 1,
    "vphantom": 1, "setstretch": 1, "linespread": 1, "fontsize": 2, "captionsetup": 1, "setlist": 1, "geometry": 1,
    "newgeometry": 1, "pagestyle": 1, "thispagestyle": 1, "enlargethispage": 1, "bibitem": 1, "specialrule": 3,
    "hyphenation": 1, "graphicspath": 1, "addbibresource": 1, "newcounter": 1, "newtheorem": 2, "pdfbookmark": 2,
    "hypersetup": 1, "settowidth": 2, "newlength": 1, "includepdf": 1, "cmidrule": 1, "addlinespace": 0,
    "theoremstyle": 1, "vskip": 0, "hskip": 0, "kern": 0, "numberwithin": 2, "crefname": 3, "Crefname": 3,
}  # fmt: skip
DEF_CMDS = {"newcommand", "renewcommand", "providecommand", "DeclareRobustCommand", "DeclareMathOperator"}
NULL_CMDS = {
    "noindent", "centering", "raggedright", "raggedleft", "small", "footnotesize", "scriptsize", "tiny", "normalsize",
    "large", "Large", "LARGE", "huge", "Huge", "clearpage", "newpage", "cleardoublepage", "pagebreak", "nopagebreak",
    "linebreak", "nolinebreak", "newline", "medskip", "smallskip", "bigskip", "vfill", "hfill", "allowbreak",
    "nobreak", "protect", "relax", "ignorespaces", "maketitle", "tableofcontents", "toprule", "midrule",
    "bottomrule", "hline", "fi", "else", "or", "expandafter", "makeatletter", "makeatother", "phantomsection",
    "FloatBarrier", "onehalfspacing", "doublespacing", "singlespacing", "normalfont", "itshape", "bfseries", "upshape",
    "rmfamily", "sffamily", "ttfamily", "em", "bf", "it", "rm", "sc", "selectfont", "frenchspacing", "sloppy",
    "fussy", "null", "strut", "unskip", "xspace", "endinput", "begingroup", "endgroup", "bgroup", "egroup",
    "displaystyle", "textstyle", "leavevmode", "indent", "morecmidrules", "endhead", "endfirsthead", "endfoot",
    "endlastfoot", "tabularnewline", "printbibliography", "listoffigures", "listoftables", "nonumber", "notag",
    "hfil", "vfil", "break", "smallbreak", "medbreak", "bigbreak", "endcsname", "ContinuedFloat", "immediate",
}  # fmt: skip
SPACE_CMDS = {"quad", "qquad", "enspace", "enskip", "thinspace", "negthinspace", "space", "hss", "nbsp"}
SYMBOL_CMDS = {
    "ldots": "...", "dots": "...", "textellipsis": "...", "cdots": "...", "textendash": "--", "textemdash": "---",
    "S": "\u00a7", "P": "\u00b6", "textasciitilde": "~", "textunderscore": "_", "textbackslash": "\\", "textbar": "|",
    "textless": "<", "textgreater": ">", "LaTeX": "LaTeX", "TeX": "TeX", "dag": "\u2020", "ddag": "\u2021",
    "textdagger": "\u2020", "textdaggerdbl": "\u2021", "textdollar": "$", "aa": "\u00e5", "ss": "\u00df",
    "o": "\u00f8", "ae": "\u00e6", "i": "i", "j": "j", "l": "l", "textquoteright": "'", "textquoteleft": "`",
    "textquotedblleft": "``", "textquotedblright": "''", "lq": "`", "rq": "'", "textperiodcentered": "\u00b7",
    "textbullet": "\u2022", "pounds": "\u00a3", "euro": "\u20ac",
}  # fmt: skip
ACCENT_LETTERS = set("cvuHrkdbt")
_NAME = re.compile(r"[A-Za-z]+")
_SPECIAL = re.compile(r"[\\${}~]")


class _Detex:
    """Turns live LaTeX into a prose view. Each prose character keeps the source offsets it came from: off (start)
    and upto (end), so a sentence of the prose view maps back to its source span and lines."""

    def __init__(self, s: str) -> None:
        self.s = s
        self.n = len(s)
        self.out: list[str] = []
        self.off: list[int] = []
        self.upto: list[int] = []
        self.math: list[tuple[int, int]] = []
        self.titles: list[tuple[str, int, int, int]] = []  # (command, prose start, prose end, source offset)

    def emit(self, text: str, at: int, upto: int | None = None) -> None:
        self.out.extend(text)
        self.off.extend([at] * len(text))
        self.upto.extend([upto if upto is not None else at + 1] * len(text))

    def run(self, i: int, end: int) -> None:
        s = self.s
        while i < end:
            m = _SPECIAL.search(s, i, end)
            j = m.start() if m else end
            if j > i:
                self.out.extend(s[i:j])
                self.off.extend(range(i, j))
                self.upto.extend(range(i + 1, j + 1))
                i = j
                continue
            c = s[i]
            if c == "\\":
                i = self.command(i, end)
            elif c == "$":
                i = self.dollar(i, end)
            elif c == "~":
                self.emit(" ", i)
                i += 1
            else:  # a brace: groups are transparent
                i += 1

    # -- argument parsing
    def group(self, k: int, end: int) -> tuple[int, int, int] | None:
        s = self.s
        while k < end and s[k] == " ":
            k += 1
        if k >= end or s[k] != "{":
            return None
        b = min(_group_end(s, k), end)
        return (k + 1, max(k + 1, b - 1), b)

    def skip_opts(self, k: int, end: int, spaces: bool = False) -> int:
        s = self.s
        while True:
            q = k
            if spaces:
                while q < end and s[q] == " ":
                    q += 1
            if q >= end or s[q] != "[":
                return k
            depth, p = 0, q
            limit = min(end, q + 600)
            while p < limit:
                ch = s[p]
                if ch == "\\":
                    p += 2
                    continue
                if ch == "{":
                    p = _group_end(s, p)
                    continue
                if ch == "[":
                    depth += 1
                elif ch == "]":
                    depth -= 1
                    if depth == 0:
                        break
                p += 1
            if p >= limit:
                return k
            k = p + 1

    def star_opts(self, k: int, end: int, spaces: bool = False) -> int:
        if k < end and self.s[k] == "*":
            k += 1
        return self.skip_opts(k, end, spaces)

    def skip_groups(self, k: int, end: int, count: int) -> int:
        for _ in range(count):
            g = self.group(k, end)
            if g is None:
                break
            k = g[2]
        return k

    def skip_token(self, k: int, end: int) -> int:
        s = self.s
        while k < end and s[k] == " ":
            k += 1
        if k < end and s[k] == "\\":
            m = _NAME.match(s, k + 1)
            return m.end() if m else min(k + 2, end)
        if k < end and s[k] == "{":
            return min(_group_end(s, k), end)
        return min(k + 1, end)

    def env_end(self, env: str, k: int, end: int) -> int:
        depth = 1
        for m in re.finditer(r"\\(begin|end)\s*\{" + re.escape(env) + r"\}", self.s[k:end]):
            depth += 1 if m.group(1) == "begin" else -1
            if depth == 0:
                return k + m.end()
        return end

    # -- math
    def emit_math(self, a: int, b: int, content: str) -> int:
        self.math.append((a, b))
        self.emit(MATH, a, b)
        tail = re.sub(r"(\\(label|tag)\{[^}]*\}|\\nonumber|\\notag|\\\\|\s)+$", "", content)
        if tail.endswith((".", ",")) and not tail.endswith("\\."):
            self.emit(tail[-1], b - 1, b)
        return b

    def dollar(self, i: int, end: int) -> int:
        s = self.s
        if s.startswith("$$", i):
            k = s.find("$$", i + 2, end)
            if k < 0:
                self.emit("$", i)
                return i + 1
            return self.emit_math(i, k + 2, s[i + 2 : k])
        p = i + 1
        while p < end:
            ch = s[p]
            if ch == "\\":
                p += 2
                continue
            if ch == "$" or ch == BOUNDARY:
                break
            p += 1
        if p >= end or s[p] != "$":
            self.emit("$", i)
            return i + 1
        return self.emit_math(i, p + 1, s[i + 1 : p])

    # -- commands
    def command(self, i: int, end: int) -> int:
        s = self.s
        if i + 1 >= end:
            return end
        c = s[i + 1]
        j = i + 2
        if not c.isalpha():
            if c == "\\":
                self.emit(" ", i)
                return self.star_opts(j, end)
            if c in "%&$#_{}":
                self.emit(c, i, j)
                return j
            if c in ",;: >":
                self.emit(" ", i, j)
                return j
            if c in "([":
                closer = "\\)" if c == "(" else "\\]"
                k = s.find(closer, j, end)
                if k < 0:
                    return j
                return self.emit_math(i, k + 2, s[j:k])
            return j  # \! \/ \@ \- and accents (the accented letter follows)
        m = _NAME.match(s, i + 1)
        assert m is not None
        name, j = m.group(), m.end()
        return self.dispatch(name, i, j, end)

    def dispatch(self, name: str, i: int, j: int, end: int) -> int:
        s = self.s
        if name in ("begin", "end"):
            g = self.group(j, end)
            if g is None:
                return j
            env = s[g[0] : g[1]].strip()
            if name == "end":
                self.emit(BOUNDARY, i, g[2])
                return g[2]
            if env in MATH_ENVS:
                k = self.env_end(env, g[2], end)
                return self.emit_math(i, k, s[g[2] : k].rsplit("\\end", 1)[0])
            if env in SKIP_ENVS:
                self.emit(BOUNDARY, i, g[2])
                return self.env_end(env, g[2], end)
            self.emit(BOUNDARY, i, g[2])
            k = self.skip_opts(g[2], end)
            return self.skip_groups(k, end, ENV_ARGS.get(env, 0))
        if name == "auth":
            g = self.group(j, end)
            if g is None:
                return j
            k = g[2]
            g2 = self.group(k, end)
            if g2 is not None:
                k = g2[2]
            self.emit("<auth>", i, k)
            return k
        if name in SECTION_CMDS or name in BLOCK_CMDS:
            k = self.star_opts(j, end)
            g = self.group(k, end)
            if g is None:
                return k
            self.emit(BOUNDARY, i, g[0])
            a = len(self.out)
            self.run(g[0], g[1])
            if name in SECTION_CMDS:
                self.titles.append((name, a, len(self.out), i))
            self.emit(BOUNDARY, g[1], g[2])
            return g[2]
        if name in TRANSPARENT or (name in ACCENT_LETTERS and s[j : j + 1] == "{"):
            return self.star_opts(j, end)
        if name in KEEP_NTH:
            count, keep = KEEP_NTH[name]
            k = self.star_opts(j, end)
            groups = []
            for _ in range(count):
                k = self.skip_opts(k, end)
                g = self.group(k, end)
                if g is None:
                    break
                groups.append(g)
                k = g[2]
            if len(groups) > keep:
                self.run(groups[keep][0], groups[keep][1])
            return k
        if name in REF_CMDS:
            k = self.star_opts(j, end)
            g = self.group(k, end)
            k = g[2] if g else k
            self.emit("<ref>", i, k)
            return k
        if name in CITE_CMDS:
            k = self.star_opts(j, end)
            k = self.skip_opts(k, end)
            g = self.group(k, end)
            k = g[2] if g else k
            self.emit("<cite>", i, k)
            return k
        if name in TOKEN_CMDS:
            k = self.star_opts(j, end)
            g = self.group(k, end)
            k = g[2] if g else k
            self.emit(TOKEN_CMDS[name], i, k)
            return k
        if name == "verb":
            if j < end:
                close = s.find(s[j], j + 1, end)
                k = close + 1 if close > 0 else end
                self.emit("<code>", i, k)
                return k
            return j
        if name == "rhPending":
            g = self.group(j, end) if s[j : j + 1] == "{" else None
            k = g[2] if g else j
            self.emit("<pending>", i, k)
            return k
        if name in DROP_CMDS:
            k = self.star_opts(j, end)
            if name == "cmidrule" and s[k : k + 1] == "(":
                close = s.find(")", k, end)
                k = close + 1 if close > 0 else k
            return self.skip_groups(k, end, DROP_CMDS[name])
        if name in DEF_CMDS:
            k = self.star_opts(j, end)
            k = self.skip_token(k, end)
            k = self.skip_opts(k, end)
            return self.skip_groups(k, end, 1)
        if name in ("newenvironment", "renewenvironment"):
            k = self.star_opts(j, end)
            k = self.skip_groups(k, end, 1)
            k = self.skip_opts(k, end)
            return self.skip_groups(k, end, 2)
        if name in ("def", "gdef", "edef", "xdef"):
            k = self.skip_token(j, end)
            brace = s.find("{", k, end)
            return min(_group_end(s, brace), end) if brace >= 0 else k
        if name in ("let", "ifx"):
            return self.skip_token(self.skip_token(j, end), end)
        if name in ("newif", "ifdefined", "ifundefined"):
            return self.skip_token(j, end)
        if name in ("csname", "ifcsname"):
            k = s.find("\\endcsname", j, end)
            return k + len("\\endcsname") if k >= 0 else j
        if name == "ensuremath":
            g = self.group(j, end)
            return self.emit_math(i, g[2], s[g[0] : g[1]]) if g else j
        if name in ("item", "par", "appendix"):
            self.emit(BOUNDARY, i, j)
            return self.skip_opts(j, end, spaces=True) if name == "item" else j
        if name in NULL_CMDS or name.startswith("if"):
            return j
        if name in SPACE_CMDS:
            self.emit(" ", i, j)
            return j
        if name in SYMBOL_CMDS:
            self.emit(SYMBOL_CMDS[name], i, j)
            return j + 2 if s.startswith("{}", j) else j
        # Any other macro: a number macro (\rh..., \cm...) or a shared symbol (\DM, \SR); it stays as a word.
        k = j + 2 if s.startswith("{}", j) else j
        self.emit("\\" + name, i, k)
        return k


# ------------------------------------------------------------------------------------------------------- documents


@dataclass
class Doc:
    src: Src
    live: str
    prose: str
    off: list[int]
    upto: list[int]
    masked: str  # prose with quotations blanked (same length)
    words_masked: str  # masked, with macros and <tokens> blanked too: for word lists
    sents: list[tuple[int, int]]
    titles: list[tuple[str, int, int, int]]
    math: list[tuple[int, int]]
    auths: list[AuthSpan]
    words: list[int] = field(default_factory=list)  # prose index of each word

    def line(self, k: int) -> int:
        if not self.off:
            return 1
        k = min(max(k, 0), len(self.off) - 1)
        return self.src.line[min(self.off[k], len(self.src.line) - 1)]

    def src_line(self, offset: int) -> int:
        return self.src.line[min(max(offset, 0), len(self.src.line) - 1)] if self.src.line else 1


ABBREV = {
    "al.", "cf.", "vs.", "fig.", "figs.", "eq.", "eqs.", "sec.", "secs.", "no.", "nos.", "approx.", "resp.", "jr.",
    "dr.", "mr.", "ms.", "prof.", "st.", "inc.", "ltd.", "p.", "pp.", "vol.", "ch.", "app.", "ed.", "eds.", "viz.",
    "ca.", "ibid.", "e.g.", "i.e.", "etc.", "approx.", "col.", "cols.",
}  # fmt: skip
_END = re.compile(r"[.?!]+")
_QUOTES = re.compile(r"``[^\x1e]{0,400}?''|\"[^\"\x1e]{1,400}\"|`[^`'\x1e]{1,200}'")
_TOKENS = re.compile(r"\\[A-Za-z]+|<(?:math|ref|cite|code|url|auth|pending)>")
_WORD = re.compile(r"(?<![<A-Za-z\\])\\?[A-Za-z][A-Za-z'\-]*")


def _is_abbrev(p: str, a: int, dot: int) -> bool:
    k = dot
    while k > a and not p[k - 1].isspace():
        k -= 1
    tok = p[k : dot + 1].lstrip("([`'\"")
    return tok.lower() in ABBREV or bool(re.fullmatch(r"(?:[A-Za-z]\.){2,}", tok))


def sentences(p: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []

    def add(a: int, b: int) -> None:
        while a < b and p[a].isspace():
            a += 1
        while b > a and p[b - 1].isspace():
            b -= 1
        if b > a and re.search(r"[A-Za-z0-9]", p[a:b]):
            spans.append((a, b))

    for blk in re.finditer(r"[^\x1e]+", p):
        a, b = blk.span()
        start = a
        for m in _END.finditer(p, a, b):
            j = m.end()
            while j < b and p[j] in ")]'\"":
                j += 1
            if j < b and not p[j].isspace():
                continue
            k = j
            while k < b and p[k].isspace():
                k += 1
            if k < b and not (p[k].isupper() or p[k].isdigit() or p[k] in "\\<`([$"):
                continue
            if m.group() == "." and _is_abbrev(p, a, m.start()):
                continue
            add(start, j)
            start = k
        add(start, b)
    return spans


def analyze(src: Src) -> Doc:
    live = live_text(src.text)
    auths = auth_spans(live)
    dx = _Detex(live)
    dx.run(0, len(live))
    prose = "".join(dx.out)
    masked = list(prose)
    for m in _QUOTES.finditer(prose):
        for k in range(*m.span()):
            masked[k] = " "
    masked_s = "".join(masked)
    wm = list(masked_s)
    for m in _TOKENS.finditer(masked_s):
        for k in range(*m.span()):
            wm[k] = " "
    doc = Doc(
        src=src,
        live=live,
        prose=prose,
        off=dx.off,
        upto=dx.upto,
        masked=masked_s,
        words_masked="".join(wm),
        sents=sentences(prose),
        titles=dx.titles,
        math=dx.math,
        auths=auths,
    )
    doc.words = [m.start() for m in _WORD.finditer(prose)]
    return doc


def docs_for(path: Path, text: str | None = None) -> list[Doc]:
    if text is None:
        text = path.read_text(encoding="utf-8", errors="replace")
    text = text.replace("\r\n", "\n")
    if path.suffix == ".py":
        return [analyze(s) for s in py_srcs(text)]
    return [analyze(tex_src(text))]


# ---------------------------------------------------------------------------------------------------- lint checks


@dataclass
class RawHit:
    id: str
    line: int
    match: str
    message: str
    candidate: bool = False


def _ctx(text: str, a: int, b: int, width: int = 30) -> str:
    left = text[max(0, a - width) : a]
    right = text[b : b + width]
    return re.sub(r"\s+", " ", f"{left}[{text[a:b]}]{right}").replace(BOUNDARY, " ").strip()


# V2 -------------------------------------------------------------------------------------------------------------
# The C2 whitelist (report section 8.3 C2: years, "HAR(3)", "Section 3"), each entry with its reason. The entries
# marked False extend the documented list.
V2_WHITELIST: list[tuple[str, str, str, bool]] = [
    ("year", r"(?<![\d.,])(?:1[89]|20)\d\d(?:s|'s)?(?![\d%]|[.,]\d)", "a year (C2 whitelist)", True),
    ("model order", r"\b[A-Za-z][A-Za-z-]*\(\d+(?:,\s*\d+)*\)", "a model's order, as in HAR(3) (C2 whitelist)", True),
    (
        "cross-reference",
        r"\b(?:Sections?|Tables?|Figures?|Fig\.|Appendix|Appendices|Panels?|Equations?|Eqs?\.|Columns?|Chapters?"
        r"|Footnotes?|Parts?|Steps?|Rows?)\s+\d+(?:\.\d+)*(?:\s*(?:--|-|,|and|to)\s*\d+(?:\.\d+)*)*",
        "a numbered cross-reference, as in Section 3 (C2 whitelist)",
        True,
    ),
    (
        "name",
        r"\bS\s*&\s*P\s*\d+|\bCOVID-19\b|\b\d+DTE\b|\bRussell\s+\d+",
        "digits that are part of a proper name (S&P 500, COVID-19, 0DTE), not a quantity",
        False,
    ),
]
_V2_WL = [re.compile(rx, re.I if name == "cross-reference" else 0) for name, rx, _, _ in V2_WHITELIST]
_DIGITS = re.compile(r"(?<![A-Za-z0-9_\\])(?:\d{1,2}:\d\d(?![\d:])|\d+(?:[.,]\d+)*)")
_CLOCK = re.compile(r"\d{1,2}:\d\d")
# A sentence that reports a cited work's finding ("\citet{x} find ..."): its numbers may be the cited work's (B5).
_CITED_FINDING = re.compile(
    r"<cite>\s+(?:[A-Za-z]+\s+){0,2}(?:find|finds|found|show|shows|showed|report|reports|reported|document|documents"
    r"|documented|estimate|estimates|estimated|obtain|obtains|obtained|attribute|attributes)\b"
)
_NUMWORDS = re.compile(
    r"\b(two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen"
    r"|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundreds?|thousands?|millions?"
    r"|billions?|dozens?|twice|thrice)\b(?!-(?:sided|tailed|step|stage|way|sample|dimensional)\b)",
    re.I,
)
_MATH_NUM = re.compile(
    r"(?:=|\\approx|\\simeq|\\leq?|\\geq?|\\neq?|<|>)\s*(?:\{\})?\s*[-+]?\s*(\d+(?:\.\d+)?|\.\d+)(?![\d.]*\s*[_^])"
)


def _in_script(text: str, pos: int) -> bool:
    depth = 0
    for k in range(pos - 1, -1, -1):
        ch = text[k]
        if ch == "}":
            depth += 1
        elif ch == "{":
            if depth == 0:
                return k > 0 and text[k - 1] in "_^"
            depth -= 1
    return False


def check_v2(doc: Doc) -> list[RawHit]:
    p = doc.prose
    exempt = [False] * len(p)
    for rx in _V2_WL:
        for m in rx.finditer(p):
            for k in range(*m.span()):
                exempt[k] = True
    hits = []
    starts = [a for a, _ in doc.sents]
    for m in _DIGITS.finditer(doc.masked):  # quotations are not flagged (B5)
        if all(exempt[m.start() : m.end()]):
            continue
        shown = m.group() + ("%" if p[m.end() : m.end() + 1] == "%" else "")
        if _CLOCK.fullmatch(m.group()):
            hits.append(
                RawHit(
                    "V2",
                    doc.line(m.start()),
                    shown,
                    f'typed clock time "{shown}" ({_ctx(p, m.start(), m.end())}): a design time (session, entry, '
                    "release) is not a hit; a result (a peak or worst bar) is a macro",
                    candidate=True,
                )
            )
            continue
        k = bisect.bisect_right(starts, m.start()) - 1
        cites = k >= 0 and bool(_CITED_FINDING.search(p, doc.sents[k][0], doc.sents[k][1]))
        hits.append(
            RawHit(
                "V2",
                doc.line(m.start()),
                shown,
                f'typed number "{shown}" in live text ({_ctx(p, m.start(), m.end())}); use a generated macro, '
                "or \\rhPending{<macro name>} until it exists"
                + ("; the sentence cites: a number reported from the cited work is not a hit" if cites else ""),
                candidate=cites,
            )
        )
    for m in _NUMWORDS.finditer(doc.masked):
        hits.append(
            RawHit(
                "V2",
                doc.line(m.start()),
                m.group(),
                f'number word "{m.group()}" ({_ctx(p, m.start(), m.end())}): a count is a number; take it from a '
                "count macro unless it is not a quantity",
                candidate=True,
            )
        )
    for a, b in doc.math:
        content = doc.live[a:b]
        for m in _MATH_NUM.finditer(content):
            val = m.group(1)
            if val in ("0", "1") or _in_script(content, m.start()):
                continue
            shown = re.sub(r"\s+", " ", m.group()).strip()
            hits.append(
                RawHit(
                    "V2",
                    doc.src_line(a + m.start(1)),
                    val,
                    f'number typed in math after a relation ("{shown}"): a result is a macro; a formula constant is '
                    "not a hit",
                    candidate=True,
                )
            )
    return hits


# V3 -------------------------------------------------------------------------------------------------------------
# (regex over the prose view, the notation guide's name, source, candidate). Case-insensitive; quotations and code
# are skipped. A term that also has an ordinary sense ("a family of K comparisons", "one observation per bar") is a
# hit only in the guide's sense, or a candidate where the sense is unclear.
V3_TERMS: list[tuple[str, str, str, bool]] = [
    (
        r"\bper-bar\b|\bper bar (?:models?|OLS|forecasts?|fits?|designs?|regressions?)\b",
        "closing-interval model",
        "notation guide section 5",
        False,
    ),
    (r"\bpooled models?\b", "pooled intraday model", "section 5", False),
    (r"\bbaselines?\b", "HAR benchmark", "section 5", False),
    (r"\ball[ _-]features\b", "full predictor set", "section 5", False),
    (r"\b(?:f )?arms?\b|\bsubpt\b|\bsub_[a-z]\w*", "specification, or describe it", "section 5", False),
    (r"\bwalk[- ]forward\b", "rolling-window out-of-sample exercise; re-estimated daily", "section 5", False),
    (
        r"\btest (?:period|window)s?\b|\bout-of-sample periods?\b",
        "evaluation span $\\mathcal T_{\\mathrm{oos}}$ (or the option-data subsample $\\mathcal T_{\\mathrm{opt}}$)",
        "section 5; 6b",
        False,
    ),
    (r"\btrade days?\b", "trading days", "section 5; addenda B.3 #29", False),
    (r"\bshipped\b|\buntuned\b", "fixed hyperparameters", "section 5; addenda B.3 #31", False),
    (r"\bpre-?tun(?:e|ed|ing)\b", "hyperparameter selection on a pre-sample", "section 5", False),
    (r"\bfit blocks?\b", "estimation sample", "section 5", False),
    (r"\bvalidation tails?\b", "validation sample", "section 5", False),
    (r"\bembargo(?:es|ed)?\b", "gap", "section 5", False),
    (r"\b(?:replication )?gates?\b", "replication check", "section 5", True),
    (r"\b(?:HAR|avail\w*|missing-data) ladders?\b", "HAR components; missing-data indicators", "section 5", False),
    (r"\bladders?\b", "HAR components; missing-data indicators", "section 5", True),
    (r"\bavail\b|_avail\b", "availability indicator $a_{t,i}$", "section 5", False),
    (r"_active\b", "activity indicator $\\chi_{t,i}$", "section 5; addenda B.1 #5", False),
    (r"\bplain forecasts?\b", "forecast", "section 5", False),
    (
        r"\bridge-init\w*|\bQLIKE-tuned MinGRU\b",
        "MinGRU network (ridge-initialized, trained and selected on QLIKE)",
        "section 5",
        False,
    ),
    (
        r"\b(?:model|linear|tree|MinGRU|network) famil(?:y|ies)\b|\bfamil(?:y|ies) of (?:models|estimators)\b",
        "model class",
        "section 5",
        False,
    ),
    (r"\bfamil(?:y|ies)\b", "model class", "section 5", True),
    (
        r"\bstraddle trades?\b",
        "at-the-money straddle position, opened at 15:30, held unhedged",
        "section 5; B.1 #6",
        False,
    ),
    (
        r"\bdelta-neutral\b",
        "at-the-money straddle, held unhedged (approximately delta-neutral at entry)",
        "B.1 #6",
        False,
    ),
    (r"\bcrossed\b", "at quoted bid/ask prices", "section 5", False),
    (r"\binterim\b", "preliminary", "section 5", False),
    (r"\badj_\w+|\b\w+_ma_\d+\b", "the series symbol $x^{(k)}_t$", "section 5", False),
    (
        r"\b(?:opening|closing)[- ]session\b|\bsession (?:dumm|interaction)\w*",
        "opening- and closing-interval dummies",
        "addenda B.3 #30",
        False,
    ),
    (r"\bfeature importance\b", "variable importance", "6b (Section 8's title)", False),
]
_V3_RX = [(re.compile(rx, re.I), name, srcref, cand) for rx, name, srcref, cand in V3_TERMS]
# Symbol clashes of addenda B.1/B.2 and the final consolidation, read inside math (case-sensitive).
V3_MATH: list[tuple[str, str, str]] = [
    (r"(?<![A-Za-z\\])q\s*=\s*\\(?:fqs|rhr|rhe|rhch)\w*Lag", "$q_{\\mathrm{NW}}$ ($q_t$ is the position)", "B.1 #1"),
    (r"\\bm\s*\{?d\}?_", "$\\bm D_t$ for the calendar dummies ($d_t$ is the loss differential)", "B.1 #3"),
    (r"\\Phi_\{?\s*t\s*,\s*AB", "$\\phi_{t,AB}$ ($\\Phi$ is the normal distribution function)", "B.1 #9"),
    (r"\\mathrm\{ES\}_(?:t\b|\{t\})", "$\\mathrm{ES}_{\\theta,t}$", "final consolidation; B.1 #10"),
    (r"\\sign\s*\(\s*s_t", "$q_t = \\sign(\\hat\\sigma^2_t - \\mathrm{IV}^2_t)$ (no $s_t$)", "B.1 #11"),
    (r"_\{\s*s\s*\\in\s*\\mathcal\s*\{?W", "$t'$ as the window row index", "B.1 #12"),
    (r"\\mathcal\s*\{?T\}?_b(?![A-Za-z])", "$f_b$, $b = 1, \\dots, B$ trees", "B.1 #14"),
    (r"(?<![A-Za-z\\])x\^\{\(k\)\}_t", "$\\bar u^{(k)}_t$", "B.1 #15"),
    (r"(?<![A-Za-z])R\s*\\times\s*C(?![A-Za-z])", "$d_{\\mathrm{out}} \\times d_{\\mathrm{in}}$", "B.1 #16"),
    (r"\\RV\^\{\(k\)\}", "Corsi's $\\RV^{(d)}, \\RV^{(w)}, \\RV^{(m)}$", "B.1 #8"),
    (r"\\sum_\{k\s*=\s*1\}\^\{?30\}?", "$\\sum_{k=1}^{N_{t,c}}$", "B.2 #21"),
]
_V3_MATH_RX = [(re.compile(rx), name, srcref) for rx, name, srcref in V3_MATH]
_BENCHMARK = re.compile(r"\bbenchmarks?\b(?!\s+forecasts?\b)", re.I)
_BENCHMARK_OK = re.compile(r"(?:\bHAR\)?|short-straddle|\bstraddle)\s*$", re.I)


def check_v3(doc: Doc) -> list[RawHit]:
    t = doc.words_masked
    hits = []
    covered: list[tuple[int, int]] = []
    for rx, name, srcref, cand in sorted(_V3_RX, key=lambda x: x[3]):  # the guide's senses first
        for m in rx.finditer(t):
            if cand and any(a < m.end() and m.start() < b for a, b in covered):
                continue
            covered.append(m.span())
            sense = f" ({_ctx(doc.prose, m.start(), m.end())}), if it names the guide's object" if cand else ""
            hits.append(
                RawHit(
                    "V3",
                    doc.line(m.start()),
                    m.group(),
                    f'"{m.group()}"{sense} is not the notation guide\'s name: use "{name}" ({srcref})',
                    candidate=cand,
                )
            )
    for m in _BENCHMARK.finditer(t):
        if _BENCHMARK_OK.search(t[max(0, m.start() - 40) : m.start()]):
            continue
        hits.append(
            RawHit(
                "V3",
                doc.line(m.start()),
                m.group(),
                f'bare "{m.group()}" ({_ctx(doc.prose, m.start(), m.end())}): the guide keeps "HAR benchmark" for the '
                'predictor set, "benchmark forecast" for the OLS and "short-straddle benchmark" for the strategy '
                "(final consolidation; addenda B.3 #32)",
                candidate=True,
            )
        )
    for a, b in doc.math:
        content = doc.live[a:b]
        for rx, name, srcref in _V3_MATH_RX:
            for m in rx.finditer(content):
                hits.append(
                    RawHit(
                        "V3",
                        doc.src_line(a + m.start()),
                        m.group(),
                        f'symbol "{m.group()}" clashes in the notation guide: use {name} ({srcref})',
                    )
                )
        if content.strip("$ ") == "q" and re.search(r"truncation\s*$", doc.live[max(0, a - 40) : a]):
            hits.append(RawHit("V3", doc.src_line(a), "$q$", "lag truncation $q$: use $q_{\\mathrm{NW}}$ (B.1 #1)"))
    return hits


# V5 -------------------------------------------------------------------------------------------------------------
_EST_EXCEPT = {
    "forest", "forests", "test", "tests", "interest", "interests", "rest", "request", "requests", "suggest",
    "suggests", "honest", "modest", "harvest", "earnest", "manifest", "manifests", "contest", "protest", "digest",
    "invest", "invests", "arrest", "west", "nest", "guest", "quest", "chest", "lest", "zest", "jest", "crest",
    "attest", "attests", "detest", "infest", "divest", "vest", "unrest", "backtest", "backtests", "retest", "pretest",
    "subtest", "bequest", "conquest", "inquest", "behest", "midwest", "priest", "forecast", "est", "harvests",
}  # fmt: skip
V5_LEXICON: list[tuple[str, str]] = [
    ("universal", r"\b(?:all|every|each|any|no|none|always|never|uniformly|without exception)\b"),
    ("completeness adverb", r"\b(?:entirely|completely|fully)\b"),
    (
        "superlative",
        r"\bthe (?:most|least)\b|\b(?:best|worst|only|unique|uniquely)\b|\b(?:rank(?:s|ed)?|comes?|came|places?"
        r"|placed) first\b|\bthe first to\b|\b[a-z]{2,}est\b",
    ),
    (
        "booster",
        r"\b(?:clearly|crucially|fundamentally|importantly|indeed|very|remarkably|strikingly|substantially"
        r"|dramatically)\b",
    ),
]
_V5_RX = [(cls, re.compile(rx, re.I)) for cls, rx in V5_LEXICON]


def check_v5(doc: Doc) -> list[RawHit]:
    t = doc.words_masked
    hits = []
    for cls, rx in _V5_RX:
        for m in rx.finditer(t):
            w = m.group()
            if cls == "superlative" and w.lower() in _EST_EXCEPT:
                continue
            hits.append(
                RawHit(
                    "V5",
                    doc.line(m.start()),
                    w,
                    f'Rule NT {cls} "{w}" ({_ctx(doc.prose, m.start(), m.end())}): NEEDED?, then TRUE?',
                    candidate=True,
                )
            )
    return hits


# V6 -------------------------------------------------------------------------------------------------------------
_V6_SEP = re.compile(r"[,;:](?=\s)|\s--?-?(?=\s)|---|\u2014|\u2013")
_V6_TERM = re.compile(r"[.?!](?!\d)|[;()]|:(?=\s)|\s--?-?\s|---|\u2014|\u2013|,(?=\s)")
_V6_STRONG = re.compile(
    r"\b(?:none|neither|nor|not one|(?:in)?significant(?:ly)?|barely|if at all|at all|if any|nowhere near"
    r"|not close|not even close|slightly|marginally|modestly|narrowly|at best)\b",
    re.I,
)
_V6_WEAK = re.compile(r"^(?:(?:and|but|or|yet)\s+)?(?:no|not|all|every|only)\b(?!-)", re.I)
_V6_MAX_WORDS = 5


def check_v6(doc: Doc) -> list[RawHit]:
    t = doc.masked
    hits = []
    for a, b in doc.sents:
        for m in _V6_SEP.finditer(t, a, b):
            q = m.end()
            term = _V6_TERM.search(t, q, b)
            stop = term.start() if term else b
            tail = t[q:stop].strip()
            words = re.findall(r"[^\s]+", tail)
            if not 1 <= len(words) <= _V6_MAX_WORDS:
                continue
            by_comma = term is not None and term.group().startswith(",")
            strong = _V6_STRONG.search(tail)
            weak = len(words) <= 3 and not by_comma and _V6_WEAK.match(tail)
            if not (strong or weak):
                continue
            k = q + (len(t[q:stop]) - len(t[q:stop].lstrip()))
            shown = re.sub(r"\s+", " ", m.group().strip() + " " + tail).strip()
            hits.append(
                RawHit(
                    "V6",
                    doc.line(k),
                    shown,
                    f'clause-final tail "{shown}" ({_ctx(doc.prose, k, stop)}): state it as a claim of its own with '
                    "its number and test, or delete it",
                    candidate=True,
                )
            )
    return hits


# V9 -------------------------------------------------------------------------------------------------------------
_V9_CLAUSE = re.compile(
    r"^(?:what|where|which|why|how|when|who|whether|does|do|did|is|are|can|should)\b|\b(?:is|are|was|were|be|been"
    r"|does|do|did|has|have|had|can|could|will|would|should|may|might|must|matters?|helps?|beats?|fails?"
    r"|holds?|comes?|remains?|carries|carry|drives?|explains?|adds?|lowers?|raises?|improves?|wins?|loses?|pays?"
    r"|shows?|reveals?|tells?)\b",
    re.I,
)


def check_v9(doc: Doc) -> list[RawHit]:
    hits = []
    for cmd, a, b, at in doc.titles:
        title = re.sub(r"\s+", " ", doc.prose[a:b]).strip()
        if "<auth>" in title:
            continue
        forms = []
        if ":" in title:
            forms.append("a colon")
        if "?" in title:
            forms.append("a question mark")
        if _V9_CLAUSE.search(title):
            forms.append("a clause")
        if forms:
            hits.append(
                RawHit(
                    "V9",
                    doc.src_line(at),
                    title,
                    f'\\{cmd} title "{title}" has {" and ".join(forms)}: use the field\'s noun phrase for the topic '
                    "and keep the \\label",
                    candidate=True,
                )
            )
    return hits


# V10 ------------------------------------------------------------------------------------------------------------
_V10_VERB = (
    r"(?:underscor(?:e|es|ed)|highlight(?:s|ed)?|showcas(?:e|es|ed)|reveal(?:s|ed)?|demonstrat(?:e|es|ed)"
    r"|emphasi[sz](?:e|es|ed)|illustrat(?:e|es|ed)|attest(?:s|ed)?|testif(?:y|ies|ied)|speaks? to|points? to"
    r"|sheds? light)"
)
_V10 = re.compile(
    r"\b(?:results?|analys[ie]s|findings?|evidence|estimates?|decompositions?|exercises?|patterns?|comparisons?"
    r"|ablations?|tests?|data|numbers|this|these|those|which|it|(?:Tables?|Figures?|Panels?)\s+<ref>)\s+(?:\w+ly\s+)?"
    + _V10_VERB
    + r"\b",
    re.I,
)


def check_v10(doc: Doc) -> list[RawHit]:
    hits = []
    for m in _V10.finditer(doc.masked):
        hits.append(
            RawHit(
                "V10",
                doc.line(m.start()),
                m.group(),
                f'inanimate subject with an evaluative verb: "{m.group()}" ({_ctx(doc.prose, m.start(), m.end())}); '
                "state the number or finding instead",
                candidate=True,
            )
        )
    return hits


# AP-V -----------------------------------------------------------------------------------------------------------
# Field whitelist (AP-V; report section 8.4 S2): never flagged.
_APV_WHITELIST = re.compile(
    r"\bleverage effects?\b|\brobust standard errors?\b|\b(?:statistically )?significan(?:t|tly|ce)(?: levels?)?\b"
    r"|\bcausal(?:ly|ity)?\b|\bempirical(?:ly)?\b|\brobust to\b|\brobustness\b",
    re.I,
)
AP_V: list[tuple[str, str, str]] = [
    (
        "AP-V1",
        r"\bdelv(?:e|es|ed|ing)\b|\bunderscor(?:e|es|ed|ing)\b|\bshowcas(?:e|es|ed|ing)\b|\bintricat(?:e|ely)\b"
        r"|\bintricac(?:y|ies)\b|\bmeticulous(?:ly)?\b|\bpivotal\b|\brealms?\b|\btapestr(?:y|ies)\b|\btestaments?\b"
        r"|\bmultifaceted\b|\bgarner(?:s|ed|ing)?\b",
        "hard-flag word",
    ),
    (
        "AP-V2",
        r"\badditionally\b|\bnotably\b|\bcrucial(?:ly)?\b|\bcomprehensive(?:ly)?\b|\binsights?\b"
        r"|\benhanc(?:e|es|ed|ing|ement)\b|\bhighlighting\b"
        # The report's list is open-ended; these catalog words are counted with the field whitelist applied.
        r"|\bleverag(?:e|es|ed|ing)\s+(?:the|our|this|these|a|an|its|their)\b"
        r"|\brobust\s+(?:results?|evidence|findings?|performance|gains?|improvements?|framework|approach)\b"
        r"|\b(?:landscape|navigat(?:e|es|ing)|foster(?:s|ed|ing)?|bolster(?:s|ed|ing)?|seamless(?:ly)?|nuanced"
        r"|holistic|paramount|noteworthy|invaluable)\b",
        "density word",
    ),
    ("AP-V3", r"\butili[sz](?:e|es|ed|ing|ation)\b|\bhuge\b|!", "prescriptive marker"),
]
_APV_RX = [(rid, re.compile(rx, re.I), label) for rid, rx, label in AP_V]


def check_apv(doc: Doc) -> list[RawHit]:
    t = _APV_WHITELIST.sub(lambda m: " " * len(m.group()), doc.words_masked)
    hits = []
    for rid, rx, label in _APV_RX:
        for m in rx.finditer(t):
            hits.append(
                RawHit(
                    rid,
                    doc.line(m.start()),
                    m.group(),
                    f'{label} "{m.group()}" ({_ctx(doc.prose, m.start(), m.end())})',
                )
            )
    return hits


# AP-S -----------------------------------------------------------------------------------------------------------
_APS2_DOC = {"highlighting", "ensuring", "underscoring", "showcasing"}
_APS2 = re.compile(
    r",\s+(highlighting|ensuring|underscoring|showcasing|emphasi[sz]ing|illustrating|demonstrating|reflecting"
    r"|suggesting|indicating|confirming|revealing|reinforcing|signal(?:l)?ing)\b",
    re.I,
)
_APS4 = re.compile(r"\bnot\b[^.;:?!\x1e]{1,80}?\bbut\b|\brather than\b|\bit'?s not\b[^.;\x1e]{1,60}?,\s*it'?s\b", re.I)
_APS5 = re.compile(
    r"\bit is (?:worth (?:noting|mentioning)|important to (?:note|stress|emphasi[sz]e))\b|\bit should be (?:noted|"
    r"stressed|emphasi[sz]ed)\b|\bin other words\b|\bneedless to say\b|\bit bears (?:noting|mentioning)\b",
    re.I,
)
_APS6 = re.compile(
    r"(?:In short|In summary|In sum|Overall|To summari[sz]e|To sum up|Taken together|All in all|In conclusion"
    r"|In a nutshell)\b",
    re.I,
)
_APS7 = re.compile(r"\b(?:serves?|served|serving|stands?|stood|standing) as\b", re.I)
_TOKEN = r"[^\s,;:.()\x1e]+"
_ITEM = rf"(?:{_TOKEN} ){{0,2}}{_TOKEN}"  # a list item of at most three words
_APS8 = re.compile(rf"(?<![,;] )(?<![,;])\b({_ITEM}), ({_ITEM}),? (?:and|or) ({_TOKEN})")


def check_aps(doc: Doc, conclusion: bool) -> list[RawHit]:
    t = doc.masked
    p = doc.prose
    hits = []
    for m in re.finditer(r"---|\u2014", t):
        hits.append(RawHit("AP-S1", doc.line(m.start()), m.group(), f"em dash ({_ctx(p, m.start(), m.end())})"))
    for m in _APS2.finditer(t):
        w = m.group(1).lower()
        hits.append(
            RawHit(
                "AP-S2",
                doc.line(m.start()),
                m.group().strip(", "),
                f'participial tail ", {m.group(1)} ..." ({_ctx(p, m.start(), m.end())}): delete the participial tail',
                candidate=w not in _APS2_DOC,
            )
        )
    for m in _APS4.finditer(t):
        hits.append(RawHit("AP-S4", doc.line(m.start()), _short(m.group()), '"not X but Y" / "rather than"'))
    for m in _APS5.finditer(t):
        hits.append(RawHit("AP-S5", doc.line(m.start()), m.group(), f'signposting "{m.group()}"'))
    if not conclusion:
        for a, b in doc.sents:
            opener = _APS6.match(t, a, b)
            if opener:
                hits.append(
                    RawHit(
                        "AP-S6",
                        doc.line(a),
                        opener.group(),
                        f'summary opener "{opener.group()}" outside the conclusion ({_ctx(p, a, opener.end())})',
                    )
                )
    for m in _APS7.finditer(t):
        hits.append(RawHit("AP-S7", doc.line(m.start()), m.group(), f'"{m.group()}"'))
    for a, b in doc.sents:
        for m in _APS8.finditer(t, a, b):
            items = [g.strip() for g in m.groups()]
            if any(re.search(r"\b(?:and|or)\b", x) for x in items[:2]):
                continue
            if all(re.fullmatch(r"(?:Sections?|Tables?|Figures?|Appendix)?\s*<ref>", x) for x in items):
                continue  # a roadmap or an exhibit list is not a flag
            hits.append(RawHit("AP-S8", doc.line(m.start()), _short(m.group()), "list of three"))
    return hits


def _short(s: str, n: int = 60) -> str:
    s = re.sub(r"\s+", " ", s).strip()
    return s if len(s) <= n else s[: n - 3] + "..."


def lint_raw(doc: Doc, conclusion: bool) -> list[RawHit]:
    return (
        check_v2(doc)
        + check_v3(doc)
        + check_v5(doc)
        + check_v6(doc)
        + check_v9(doc)
        + check_v10(doc)
        + check_apv(doc)
        + check_aps(doc, conclusion)
    )


# ---------------------------------------------------------------------------------------------------------- rows


@dataclass
class Hit:
    mode: str
    id: str
    file: str
    line: int
    severity: str
    candidate: bool
    match: str
    message: str
    per_1000: float | None = None

    def as_dict(self) -> dict[str, object]:
        d: dict[str, object] = {
            "mode": self.mode,
            "id": self.id,
            "file": self.file,
            "line": self.line,
            "severity": self.severity,
            "candidate": self.candidate,
            "match": self.match,
            "message": self.message,
        }
        if self.per_1000 is not None:
            d["per_1000"] = self.per_1000
        return d

    def text(self) -> str:
        flag = "candidate: " if self.candidate else ""
        return f"{self.file}:{self.line} {self.id} {self.severity} {flag}{self.message}"


def _in(line: int, rng: tuple[int, int] | None) -> bool:
    return rng is None or rng[0] <= line <= rng[1]


def lint_docs(
    docs: list[Doc], display: str, rng: tuple[int, int] | None, sev: dict[str, str], conclusion: bool = False
) -> list[Hit]:
    """Rows for the lines of rng (None: the whole file) of the documents of one file."""
    rows: list[Hit] = []
    rated: dict[str, list[RawHit]] = {}
    n_words = 0
    lengths: list[int] = []
    first_line: int | None = None
    for doc in docs:
        n_words += sum(1 for w in doc.words if _in(doc.line(w), rng))
        for a, b in doc.sents:
            if _in(doc.line(a), rng):
                lengths.append(len(_WORD.findall(doc.prose[a:b])))
                first_line = doc.line(a) if first_line is None else first_line
        for h in lint_raw(doc, conclusion):
            if not _in(h.line, rng):
                continue
            s = sev.get(h.id, "review")
            if s == "rate":
                rated.setdefault(h.id, []).append(h)
            else:
                rows.append(Hit("lint", h.id, display, h.line, s, h.candidate, h.match, h.message))
    for rid, hs in sorted(rated.items()):
        per = round(1000.0 * len(hs) / n_words, 2) if n_words else 0.0
        lines = sorted({h.line for h in hs})
        shown = ", ".join(str(x) for x in lines[:12]) + (", ..." if len(lines) > 12 else "")
        rows.append(
            Hit(
                "lint",
                rid,
                display,
                lines[0],
                "rate",
                False,
                f"{len(hs)} hits",
                f"{len(hs)} in {n_words} words ({per} per 1,000); lines {shown}; first: {hs[0].message}",
                per,
            )
        )
    if lengths:
        mean = sum(lengths) / len(lengths)
        sd = (sum((x - mean) ** 2 for x in lengths) / len(lengths)) ** 0.5
        long = sum(1 for x in lengths if x > LONG_SENTENCE)
        per = round(1000.0 * long / n_words, 2) if n_words else 0.0
        rows.append(
            Hit(
                "lint",
                LENGTH_ID,
                display,
                first_line or (rng[0] if rng else 1),
                "rate",
                False,
                "sentence length",
                f"{len(lengths)} sentences, mean {mean:.1f} words, sd {sd:.1f} (CV {sd / mean if mean else 0:.2f}); "
                f"{long} longer than {LONG_SENTENCE} words ({per} per 1,000 words); dashboard only, never a gate",
                per,
            )
        )
    return rows


_RANGE = re.compile(r"^(?P<path>.+?):(?P<a>\d+)(?:-(?P<b>\d+))?$")


def parse_target(spec: str) -> tuple[str, tuple[int, int] | None]:
    m = _RANGE.match(spec)
    if not m:
        return spec, None
    a = int(m.group("a"))
    b = int(m.group("b")) if m.group("b") else a
    return m.group("path"), (min(a, b), max(a, b))


def _display(path: str) -> str:
    return path.replace("\\", "/")


def lint(targets: list[str], sev: dict[str, str]) -> list[Hit]:
    cache: dict[str, list[Doc]] = {}
    rows: list[Hit] = []
    for spec in targets:
        path, rng = parse_target(spec)
        p = Path(path)
        if not p.is_file():
            raise FileNotFoundError(path)
        key = str(p.resolve())
        if key not in cache:
            cache[key] = docs_for(p)
        rows += lint_docs(cache[key], _display(path), rng, sev, conclusion="conclusion" in p.name.lower())
    return rows


# ------------------------------------------------------------------------------------------------------------ diff

NEGATIONS = {"not", "no", "none", "never", "neither", "nor", "nothing", "nobody", "nowhere", "without", "cannot"}
MODALS = {"can", "cannot", "could", "may", "might", "must", "shall", "should", "will", "would"}
DIRECTIONS = {
    "higher", "lower", "lowers", "lowered", "lowering", "raise", "raises", "raised", "raising", "more", "less",
    "fewer", "increase", "increases", "increased", "increasing", "decrease", "decreases", "decreased", "decreasing",
    "rise", "rises", "rose", "risen", "fall", "falls", "fell", "fallen", "above", "below", "larger", "smaller",
    "greater", "exceed", "exceeds", "exceeded", "improve", "improves", "improved", "worsen", "worsens", "worsened",
    "reduce", "reduces", "reduced", "better", "worse", "outperform", "outperforms", "outperformed", "underperform",
    "underperforms", "underperformed", "positive", "negative", "up", "down", "gain", "gains", "hurt", "hurts",
}  # fmt: skip
_LAYOUT_MACROS = {"allowbreak", "nobreak", "linebreak", "newline", "relax", "ignorespaces"}
_CITE_RX = re.compile(r"\\(?:" + "|".join(sorted(CITE_CMDS)) + r")\*?(?:\[[^\]]*\]){0,2}\{([^}]*)\}")
_REF_RX = re.compile(r"\\(?:" + "|".join(sorted(REF_CMDS)) + r")\*?\{([^}]*)\}")


@dataclass
class DSent:
    norm: str
    l0: int
    l1: int
    macros: Counter[str]
    cites: Counter[str]
    refs: Counter[str]
    neg: Counter[str]
    modal: Counter[str]
    direction: Counter[str]


def _no_auth(live: str, auths: list[AuthSpan]) -> str:
    chars = list(live)
    _blank(chars, [(a.start, a.end) for a in auths])
    return "".join(chars)


def diff_sentences(docs: list[Doc]) -> list[DSent]:
    out: list[DSent] = []
    for doc in docs:
        src = _no_auth(doc.live, doc.auths)
        for a, b in doc.sents:
            s0 = doc.off[a]
            s1 = max(doc.upto[a:b])
            chunk = src[s0:s1]
            words = [w.lower() for w in re.findall(r"[A-Za-z]+(?:'t)?", _TOKENS.sub(" ", doc.prose[a:b]))]
            neg = Counter(w for w in words if w in NEGATIONS or w.endswith("n't"))
            macros = Counter(
                m
                for m in re.findall(r"\\([A-Za-z]+)", chunk)
                if m not in _LAYOUT_MACROS and m not in CITE_CMDS and m not in REF_CMDS
            )
            cites = Counter(k.strip() for m in _CITE_RX.findall(chunk) for k in m.split(",") if k.strip())
            refs = Counter(k.strip() for m in _REF_RX.findall(chunk) for k in m.split(",") if k.strip())
            out.append(
                DSent(
                    norm=re.sub(r"\s+", " ", doc.prose[a:b]).strip(),
                    l0=doc.src_line(s0),
                    l1=doc.src_line(max(s0, s1 - 1)),
                    macros=macros,
                    cites=cites,
                    refs=refs,
                    neg=neg,
                    modal=Counter(w for w in words if w in MODALS),
                    direction=Counter(w for w in words if w in DIRECTIONS),
                )
            )
    return out


def _sum(sents: list[DSent], attr: str) -> Counter[str]:
    total: Counter[str] = Counter()
    for s in sents:
        total.update(getattr(s, attr))
    return total


def _delta(old: Counter[str], new: Counter[str], prefix: str = "") -> str:
    gone = old - new
    added = new - old
    parts = [f"-{prefix}{k}" + (f" x{v}" if v > 1 else "") for k, v in sorted(gone.items())]
    parts += [f"+{prefix}{k}" + (f" x{v}" if v > 1 else "") for k, v in sorted(added.items())]
    return " ".join(parts)


@dataclass
class LogEntry:
    file: str
    line: int
    op: str
    rule_id: str


def load_log(path: Path) -> list[LogEntry]:
    out = []
    for n, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw.strip():
            continue
        try:
            d = json.loads(raw)
            out.append(
                LogEntry(
                    str(d.get("file", "")), int(d.get("line", 0)), str(d.get("op") or ""), str(d.get("rule_id") or "")
                )
            )
        except (ValueError, TypeError, AttributeError) as exc:
            raise ValueError(f"{path}:{n}: not a JSON object with file, line, op, rule_id ({exc})") from exc
    return out


@dataclass
class AuthorRecord:
    text: str
    file: str | None = None
    ident: str | None = None


def load_author_records(path: Path) -> list[AuthorRecord]:
    """JSON (a list of strings or of {text, file?, id?}, or {"records": [...]}), JSONL (one such object per line),
    or plain text (each non-empty line is one record)."""
    raw = path.read_text(encoding="utf-8").replace("\r\n", "\n")

    def rec(x: object) -> AuthorRecord:
        if isinstance(x, str):
            return AuthorRecord(x)
        if isinstance(x, dict):
            text = x.get("text", x.get("author_text", x.get("words")))
            if not isinstance(text, str):
                raise ValueError(f"{path}: author record without a text: {x}")
            f = x.get("file")
            i = x.get("id", x.get("decision"))
            return AuthorRecord(text, str(f) if f else None, str(i) if i else None)
        raise ValueError(f"{path}: not an author record: {x!r}")

    if path.suffix == ".jsonl":
        return [rec(json.loads(ln)) for ln in raw.splitlines() if ln.strip()]
    if path.suffix == ".json" or raw.lstrip().startswith(("[", "{")):
        data = json.loads(raw)
        if isinstance(data, dict):
            data = data.get("records", [data])
        return [rec(x) for x in data]
    return [AuthorRecord(ln) for ln in raw.splitlines() if ln.strip()]


def _same_file(a: str, b: str) -> bool:
    pa, pb = Path(a), Path(b)
    try:
        if pa.resolve() == pb.resolve():
            return True
    except OSError:
        pass
    x, y = a.replace("\\", "/").lstrip("./"), b.replace("\\", "/").lstrip("./")
    return x == y or x.endswith("/" + y) or y.endswith("/" + x)


def _overlaps(l0: int, l1: int, ranges: list[tuple[int, int]]) -> bool:
    return any(a <= l1 and l0 <= b for a, b in ranges)


def diff_texts(
    display: str,
    old_text: str,
    new_text: str,
    sev: dict[str, str],
    ranges: list[tuple[int, int]] | None = None,
    log: list[LogEntry] | None = None,
    suffix: str = ".tex",
    pulled_lines: set[int] | None = None,
) -> list[Hit]:
    """V16, V14 (ranges given), V12 (log given), V13 (\\auth spans) and V15 (pulled_lines: base lines last written by
    an Overleaf pull) for one file, base text against working text."""
    path = Path("x" + suffix)
    old_docs = docs_for(path, old_text)
    new_docs = docs_for(path, new_text)
    old = diff_sentences(old_docs)
    new = diff_sentences(new_docs)
    rows: list[Hit] = []

    def add(rid: str, line: int, match: str, message: str, candidate: bool = False) -> None:
        rows.append(Hit("diff", rid, display, line, sev.get(rid, "gate"), candidate, match, message))

    def anchor(j: int) -> int:
        if j < len(new):
            return new[j].l0
        return new[-1].l1 if new else 1

    def logged(lines: list[tuple[int, int]]) -> bool:
        assert log is not None
        return any(
            e.op.strip()
            and e.rule_id.strip()
            and _same_file(e.file, display)
            and any(a <= e.line <= b for a, b in lines)
            for e in log
        )

    sm = difflib.SequenceMatcher(None, [s.norm for s in old], [s.norm for s in new], autojunk=False)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        olds, news = old[i1:i2], new[j1:j2]
        line = news[0].l0 if news else anchor(j1)
        old_lines = f"base lines {olds[0].l0}-{olds[-1].l1}" if olds else "no base sentence"
        # V16: content of the changed block.
        if olds and news:
            parts = []
            for attr, label, prefix in (("macros", "macros", "\\"), ("cites", "cite keys", ""), ("refs", "refs", "")):
                d = _delta(_sum(olds, attr), _sum(news, attr), prefix)
                if d:
                    parts.append(f"{label} {d}")
            if parts:
                add("V16", line, "; ".join(parts), f"rewrite changes {'; '.join(parts)} ({old_lines})")
            words = []
            for attr, label in (("neg", "negations"), ("modal", "modals"), ("direction", "direction words")):
                d = _delta(_sum(olds, attr), _sum(news, attr))
                if d:
                    words.append(f"{label} {d}")
            if words:
                add(
                    "V16",
                    line,
                    "; ".join(words),
                    f"rewrite changes {'; '.join(words)} ({old_lines}): kept only if the operation targeted them",
                    candidate=True,
                )
        elif olds:
            lost = [
                f"{label} {_delta(_sum(olds, attr), Counter(), prefix)}"
                for attr, label, prefix in (
                    ("macros", "macros", "\\"),
                    ("cites", "cite keys", ""),
                    ("refs", "refs", ""),
                )
                if _sum(olds, attr)
            ]
            if lost:
                add(
                    "V16",
                    line,
                    "; ".join(lost),
                    f"{len(olds)} sentence(s) deleted ({old_lines}), dropping {'; '.join(lost)}",
                    candidate=True,
                )
        # V14: changed sentences outside every target range.
        if ranges is not None:
            if news:
                for s in news:
                    if not _overlaps(s.l0, s.l1, ranges):
                        add("V14", s.l0, _short(s.norm), f'sentence changed outside the target: "{_short(s.norm, 90)}"')
            elif not _overlaps(line, line, ranges):
                add("V14", line, _short(olds[0].norm), f"sentence(s) deleted outside the target ({old_lines})")
        # V12: every changed sentence has a logged op with a rule id.
        if log is not None:
            if news:
                for s in news:
                    if not logged([(s.l0, s.l1)]):
                        add(
                            "V12",
                            s.l0,
                            _short(s.norm),
                            "changed sentence with no logged op and rule id in the edit log",
                        )
            elif not logged([(line, line)] + [(s.l0, s.l1) for s in olds]):
                add(
                    "V12",
                    line,
                    _short(olds[0].norm),
                    f"deleted sentence(s) ({old_lines}) with no logged op and rule id",
                )
        # V15: the base sentence came in through an Overleaf pull.
        if pulled_lines:
            for s in olds:
                if any(ln in pulled_lines for ln in range(s.l0, s.l1 + 1)):
                    add(
                        "V15",
                        line,
                        _short(s.norm),
                        f"changed sentence was last written by an Overleaf pull (base line {s.l0}): text typed on "
                        "Overleaf is not restyled",
                        candidate=True,
                    )
                    break
    # V13: every \auth span of the base is unchanged.
    old_auth = [a for d in old_docs for a in d.auths]
    new_auth = [a for d in new_docs for a in d.auths]
    new_by_id = {a.ident: a.text for a in new_auth if a.ident is not None}
    new_texts = Counter(a.text for a in new_auth)
    line_of = {id(a): d.src_line(a.start) for d in old_docs for a in d.auths}
    for a in old_auth:
        if a.ident is not None:
            if a.ident not in new_by_id:
                add("V13", line_of[id(a)], a.ident, f"\\auth{{{a.ident}}} span removed (base line {line_of[id(a)]})")
            elif new_by_id[a.ident] != a.text:
                new_line = next(d.src_line(b.start) for d in new_docs for b in d.auths if b.ident == a.ident)
                add("V13", new_line, a.ident, f'text inside \\auth{{{a.ident}}} changed: was "{_short(a.text, 90)}"')
        elif new_texts[a.text] > 0:
            new_texts[a.text] -= 1
        else:
            add("V13", line_of[id(a)], _short(a.text), f"\\auth span changed or removed (base line {line_of[id(a)]})")
    return rows


def check_author_records(
    records: list[AuthorRecord], texts: dict[str, tuple[str, str]], sev: dict[str, str]
) -> list[Hit]:
    """V13: each recorded author string appears byte for byte in its file (or in one of the files) of the diff.
    texts maps display path -> (base text, working text)."""
    rows = []
    for r in records:
        files = [f for f in texts if r.file is None or _same_file(r.file, f)]
        if r.file is not None and not files:
            continue  # the record belongs to a file outside this diff
        if any(r.text in texts[f][1] for f in files):
            continue
        where = files[0] if files else (next(iter(texts)) if texts else "?")
        old = texts.get(where, ("", ""))[0]
        k = old.find(r.text)
        line = old.count("\n", 0, k) + 1 if k >= 0 else 1
        was = f" (present at base line {line})" if k >= 0 else ""
        label = f" [{r.ident}]" if r.ident else ""
        rows.append(
            Hit(
                "diff",
                "V13",
                where,
                line,
                sev.get("V13", "gate"),
                False,
                _short(r.text),
                f'recorded author text{label} not found byte for byte{was}: "{_short(r.text, 90)}"',
            )
        )
    return rows


def _git(args: list[str], cwd: Path) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(["git", *args], cwd=str(cwd), capture_output=True, check=False)


def git_show(rev: str, path: Path) -> str | None:
    """The file at rev, or None if it did not exist there."""
    r = _git(["show", f"{rev}:./{path.name}"], path.resolve().parent)
    if r.returncode != 0:
        return None
    return r.stdout.decode("utf-8", errors="replace").replace("\r\n", "\n")


def pulled_base_lines(rev: str, path: Path) -> set[int]:
    """Base lines whose last commit (git blame at rev) committed writeup/overleaf_sync/map_pulled.json (a pull)."""
    cwd = path.resolve().parent
    r = _git(["blame", "--porcelain", rev, "--", path.name], cwd)
    if r.returncode != 0:
        return set()
    by_sha: dict[str, list[int]] = {}
    for raw in r.stdout.decode("utf-8", errors="replace").splitlines():
        m = re.match(r"^([0-9a-f]{40}) \d+ (\d+)", raw)
        if m:
            by_sha.setdefault(m.group(1), []).append(int(m.group(2)))
    out: set[int] = set()
    for sha, lines in by_sha.items():
        show = _git(["show", "--name-only", "--format=", sha], cwd)
        names = show.stdout.decode("utf-8", errors="replace").splitlines()
        if any(n.strip().endswith("overleaf_sync/map_pulled.json") for n in names):
            out.update(lines)
    return out


def diff(
    base: str,
    files: list[str],
    target_specs: list[str],
    log_path: str | None,
    author_path: str | None,
    sev: dict[str, str],
) -> list[Hit]:
    targets = [parse_target(t) for t in target_specs]
    log = load_log(Path(log_path)) if log_path else None
    records = load_author_records(Path(author_path)) if author_path else []
    rows: list[Hit] = []
    texts: dict[str, tuple[str, str]] = {}
    for f in files:
        p = Path(f)
        if not p.is_file():
            raise FileNotFoundError(f)
        new_text = p.read_text(encoding="utf-8", errors="replace").replace("\r\n", "\n")
        old_text = git_show(base, p)
        display = _display(f)
        ranges = None
        if targets:
            ranges = [rng if rng else (1, 10**9) for path, rng in targets if _same_file(path, f)]
        old = old_text if old_text is not None else ""
        pulled = pulled_base_lines(base, p) if old_text is not None and old_text != new_text else set()
        texts[display] = (old, new_text)
        if old == new_text:
            continue
        rows += diff_texts(display, old, new_text, sev, ranges, log, p.suffix, pulled)
    rows += check_author_records(records, texts, sev)
    return rows


# ------------------------------------------------------------------------------------------------------------- CLI


def _split_targets(targets: list[str], files: list[str]) -> tuple[list[str], list[str]]:
    """--target takes one or more FILE:START-END values; a value without a range is a FILE, and a FILE with a range
    is also a target."""
    t_out: list[str] = []
    f_out: list[str] = []
    for t in targets:
        (t_out if parse_target(t)[1] is not None else f_out).append(t)
    for f in files:
        path, rng = parse_target(f)
        if rng is not None:
            t_out.append(f)
        f_out.append(path)
    for t in t_out:
        path = parse_target(t)[0]
        if not any(_same_file(path, f) for f in f_out):
            f_out.append(path)
    seen: list[str] = []
    for f in f_out:
        if not any(_same_file(f, s) for s in seen):
            seen.append(f)
    return t_out, seen


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="prose_check.py",
        description=(__doc__ or "").split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = ap.add_subparsers(dest="mode", required=True)
    pl = sub.add_parser("lint", help="pattern checks over files or line ranges")
    pl.add_argument("targets", nargs="+", metavar="TARGET", help="FILE or FILE:START-END (.tex, or generator .py)")
    pl.add_argument("--json", action="store_true", help="print a JSON list of rows")
    pd = sub.add_parser("diff", help="sentence-aligned checks of FILE at REV against the working tree")
    pd.add_argument("--base", required=True, metavar="REV", help="git revision before the edit")
    pd.add_argument("--target", action="extend", nargs="+", default=[], metavar="FILE:START-END")
    pd.add_argument("--log", metavar="EDITLOG.jsonl", help="edit log, one {file, line, op, rule_id} per line")
    pd.add_argument(
        "--author-record",
        metavar="FILE",
        help="author strings: .json list (strings or {text, file?, id?}), .jsonl, or text (one per line)",
    )
    pd.add_argument("--json", action="store_true", help="print a JSON list of rows")
    pd.add_argument("files", nargs="*", metavar="FILE")
    return ap


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
        except (AttributeError, ValueError):
            pass
    args = build_parser().parse_args(argv)
    sev, _ = load_severities()
    try:
        if args.mode == "lint":
            rows = lint(args.targets, sev)
        else:
            targets, files = _split_targets(args.target, args.files)
            if not files:
                print("prose_check.py diff: no FILE given", file=sys.stderr)
                return 2
            check = _git(
                ["rev-parse", "--verify", "--quiet", f"{args.base}^{{commit}}"], Path(files[0]).resolve().parent
            )
            if check.returncode != 0:
                print(f"prose_check.py diff: unknown revision {args.base!r}", file=sys.stderr)
                return 2
            rows = diff(args.base, files, targets, args.log, args.author_record, sev)
    except (FileNotFoundError, ValueError, OSError) as exc:
        print(f"prose_check.py: {exc}", file=sys.stderr)
        return 2
    rows.sort(key=lambda r: (r.file, r.line, r.id, r.match))
    if args.json:
        print(json.dumps([r.as_dict() for r in rows], indent=1, ensure_ascii=False))
    else:
        for r in rows:
            print(r.text())
    return 1 if any(r.severity == "gate" and not r.candidate for r in rows) else 0


if __name__ == "__main__":
    sys.exit(main())
