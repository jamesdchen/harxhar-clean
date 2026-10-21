"""Bring Overleaf edits of the flat main.tex back to the hand-written sources (2026-10-20).

flatten.py turns the paper (paper_rehaul.tex, its rehaul/*.tex sections and the generated/*.tex floats and number
files) into one flat main.tex with literal numbers, and records in a map (map.json, SPEC.md section 4) where every
character of main.tex came from. People edit that main.tex in Overleaf. This script reads the edited main.tex and
the map and:

* diffs the edited text against the flatten-time text (map.flat_text) by lines, one hunk for each changed block,
  each hunk trimmed to the characters that changed;
* applies a hunk that lies in one hand-written file (paper_rehaul.tex or rehaul/*.tex) to that file. Number
  macros whose literal values survive in the new text are linked back to their calls (\\rhoosDays{} and so on);
* parks the replaced source lines in a dated comment instead of deleting them:
      % <date> (overleaf): parked -- edited in Overleaf. Was:
      % <old line>
      <new line>
  (comment-only edits and new lines are written directly);
* never writes a generated file (generated/*): edits there, or to numbers that came from generated macros, are
  reported as conflicts with the generator to change;
* merges three ways (git merge-file) when a source file changed in the repository after the flatten;
* warns when an applied edit types new digits into prose (every number in prose should come from a macro);
* writes a markdown report of everything above.

Conflict kinds (none of them is applied):
* number:     the edit changes a literal that came from a number macro (its expansion no longer appears in the new
              text, or digits were typed right against it);
* generated:  the edit touches text inlined from a generated float;
* structural: the edit touches flattener text (header, begin/end markers), spans source text the flatten dropped
              (an \\input line, a guard line), or spans two source files;
* merge:      the file changed since the flatten and git merge-file could not merge the edit cleanly.

Details that SPEC section 5 leaves open, as implemented here:
* an insertion strictly inside a segment belongs to that segment (the attach rule of SPEC 5.2 covers only
  insertions at segment boundaries);
* digits typed right against a macro's value (1{,}082 -> 1{,}0825, which trimming turns into an insertion next to
  the macro) are a number conflict, not an applied insertion;
* the digit warning covers every applied edit that adds a number not in the old text, outside comments and outside
  \\label / \\ref / \\cite-style arguments, and names existing macros with that value;
* each parked old line keeps its own indentation before the "% ";
* a hunk is one changed block of lines, so edits on adjacent lines share a fate: a number edit next to a prose edit
  blocks both, and deleting a line that holds a macro value is a number conflict (its value no longer appears);
* the map is checked against SPEC section 4 before anything is done, and a bad map stops the run (exit 1);
* a byte-order mark, CRLF line endings or a dropped final newline in the edited main.tex are not edits.

CLI:
    python writeup/overleaf_sync/unflatten.py --main PATH/main.tex [--map writeup/overleaf_sync/map.json]
        [--root writeup] [--date YYYY-MM-DD] [--dry-run] [--strict] [--report writeup/overleaf_sync/REPORT.md]

    --main     the main.tex edited in Overleaf (required)
    --map      the map written by the flatten Overleaf started from (default: <root>/overleaf_sync/map.json)
    --root     the writeup/ directory the map's source paths are relative to (default: the writeup/ directory that
               holds this script); tests point it at a temporary copy
    --date     the date written in parking comments (default: today, UTC)
    --dry-run  write nothing except the report
    --strict   apply nothing when any conflict exists
    --report   the markdown report (default: <root>/overleaf_sync/REPORT.md)

Exit status: 0 when there is no conflict (including "no changes"), 2 when there is at least one conflict (the
non-conflicting edits are still applied unless --strict), 1 on an error (unreadable main.tex, invalid map, bad
arguments).

Examples:
    python writeup/overleaf_sync/unflatten.py --main ../harxhar-paper-overleaf/main.tex --dry-run   # read REPORT.md
    python writeup/overleaf_sync/unflatten.py --main ../harxhar-paper-overleaf/main.tex
    python writeup/overleaf_sync/unflatten.py --main /tmp/x/main.tex --map /tmp/x/map.json --root /tmp/x/writeup \\
        --date 2026-10-20 --report /tmp/x/REPORT.md

Standard-library Python 3 only; git must be on PATH for the three-way merge. Spec: writeup/overleaf_sync/SPEC.md.
"""

from __future__ import annotations

import argparse
import bisect
import datetime
import difflib
import hashlib
import json
import re
import subprocess
import sys
import tempfile
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_ROOT = SCRIPT_DIR.parent  # writeup/

CLOSING_NOTE = (
    "run flatten.py again; conflicting edits to numbers or tables will be overwritten in `main.tex` by the next "
    "flatten unless the generator changes."
)
PARK_HEADER = "% {date} (overleaf): parked -- edited in Overleaf. Was:"

# Digits glued to a number: "{,}5", ".5", ",5" or "5" right after a number, "5", "5." or "5{,}" right before one.
_GLUE_AFTER = re.compile(r"(?:\{,\}|[.,])?\d")
_GLUE_BEFORE = re.compile(r"\d(?:\{,\}|[.,])?$")
# A number as it is typed in prose: 1{,}082, 0.9, 3,5 or 12.
_NUMBER = re.compile(r"\d+(?:(?:\{,\}|[.,])\d+)*")
# Commands whose braced argument holds names, not prose numbers (labels, references, citations, files, colors).
_NAME_ARG_COMMANDS = re.compile(
    r"\\(?:label|ref|eqref|pageref|autoref|cref|Cref|nameref|cite[a-zA-Z]*|url|href|input|include|"
    r"includegraphics|begin|end|bibliography|bibliographystyle|textcolor|colorbox|color|editc)\*?"
    r"(?:\[[^\]]*\])*\{[^{}]*\}"
)


# ----------------------------------------------------------------------------------------------------- errors


class UnflattenError(Exception):
    """An error that stops the run before anything is written (exit status 1)."""


class _Parser(argparse.ArgumentParser):
    """argparse exits 2 on bad arguments, which this script uses for conflicts; exit 1 instead."""

    def error(self, message: str) -> None:  # type: ignore[override]
        self.print_usage(sys.stderr)
        self.exit(1, f"{self.prog}: error: {message}\n")


# ------------------------------------------------------------------------------------------------ the map


@dataclass
class Segment:
    """One entry of map.segments: flat range [fa, fb) of main.tex and where it came from."""

    fa: int
    fb: int
    kind: str  # src | macro | gen | synthetic | drop
    file: str | None
    s: int | None
    e: int | None
    macro: str | None
    generator: str | None
    index: int  # position in map.segments, for error messages


@dataclass
class FlatMap:
    """The parsed map file plus lookup tables over its segments."""

    raw: dict
    path: Path
    flat_text: str
    sources: dict  # rel path -> {"sha256", "text", "kind"}
    macros: dict  # name -> {"value", "expanded", "file", "generator"}
    segments: list[Segment]  # every segment, in map order (sorted by fa)
    spans: list[Segment] = field(default_factory=list)  # nonzero-length segments, tiling the flat text
    span_starts: list[int] = field(default_factory=list)
    points: list[Segment] = field(default_factory=list)  # zero-length segments (drops, empty macros)

    def kind_of(self, rel: str | None) -> str | None:
        entry = self.sources.get(rel) if rel is not None else None
        return entry.get("kind") if entry else None

    def file_generator(self, rel: str | None) -> str | None:
        """The generator of a generated file: from its gen segments, else from its first comment lines (SPEC 1)."""
        if rel is None:
            return None
        for g in self.segments:
            if g.file == rel and g.kind == "gen" and g.generator:
                return g.generator
        text = self.sources.get(rel, {}).get("text", "")
        for line in split_lines(text)[:8]:
            if not is_comment_line(line):
                break
            m = re.search(r"((?:writeup/make_|experiments/)[\w./-]*?\.py)\b", line)
            if m:
                return m.group(1)
        return None

    def source_text(self, rel: str) -> str:
        return self.sources[rel]["text"]


def load_map(path: Path) -> FlatMap:
    """Read map.json, check it against SPEC section 4 and build the lookup tables."""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise UnflattenError(f"cannot read the map {path}: {exc}") from exc
    for key in ("flat_text", "sources", "macros", "segments"):
        if key not in raw:
            raise UnflattenError(f"map {path} has no '{key}'")
    segments = []
    for i, seg in enumerate(raw["segments"]):
        try:
            fa, fb = seg["flat"]
            src = seg.get("src")
            s, e = (src if src is not None else (None, None))
            segments.append(
                Segment(int(fa), int(fb), seg["kind"], seg.get("file"), s, e, seg.get("macro"),
                        seg.get("generator"), i)
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise UnflattenError(f"map segment {i} is malformed: {seg!r} ({exc})") from exc
    fmap = FlatMap(raw, path, raw["flat_text"], raw["sources"], raw["macros"], segments)
    problems = validate_map(fmap)
    if problems:
        shown = "\n  ".join(problems[:20])
        more = f"\n  ... and {len(problems) - 20} more" if len(problems) > 20 else ""
        raise UnflattenError(f"the map {path} does not satisfy SPEC section 4:\n  {shown}{more}")
    fmap.spans = [g for g in segments if g.fb > g.fa]
    fmap.span_starts = [g.fa for g in fmap.spans]
    fmap.points = [g for g in segments if g.fb == g.fa]
    return fmap


def validate_map(fmap: FlatMap) -> list[str]:
    """Every violated invariant of SPEC section 4, as readable strings (empty when the map is sound).

    Editing sources through a map that does not describe the flat text would corrupt them, so a bad map stops the
    run.
    """
    out: list[str] = []
    raw, flat = fmap.raw, fmap.flat_text
    if raw.get("version") != 1:
        out.append(f"version is {raw.get('version')!r}, expected 1")
    sha = hashlib.sha256(flat.encode("utf-8")).hexdigest()
    if raw.get("flat_sha256") not in (None, sha):
        out.append("flat_sha256 does not match flat_text")
    pos = 0
    last_fa = -1
    for g in fmap.segments:
        where = f"segment {g.index} ({g.kind}, flat [{g.fa}, {g.fb}])"
        if g.fa < last_fa:
            out.append(f"{where}: not sorted by flat start")
        last_fa = g.fa
        if g.fb < g.fa:
            out.append(f"{where}: negative length")
            continue
        if g.kind not in ("src", "macro", "gen", "synthetic", "drop"):
            out.append(f"{where}: unknown kind")
            continue
        if g.fb > g.fa:
            if g.fa != pos:
                out.append(f"{where}: does not start where the previous segment ends ({pos})")
            pos = g.fb
        if g.kind == "drop" and g.fa != g.fb:
            out.append(f"{where}: a drop segment must have zero length")
        if g.kind == "synthetic":
            continue
        if g.file is None or g.file not in fmap.sources:
            out.append(f"{where}: file {g.file!r} is not in sources")
            continue
        text = fmap.sources[g.file].get("text", "")
        if g.kind in ("src", "macro", "drop"):
            if g.s is None or g.e is None or not (0 <= g.s <= g.e <= len(text)):
                out.append(f"{where}: bad source range {g.s}, {g.e}")
                continue
        if g.kind == "src":
            if fmap.kind_of(g.file) != "hand":
                out.append(f"{where}: src segment in a file of kind {fmap.kind_of(g.file)!r}")
            if flat[g.fa:g.fb] != text[g.s:g.e]:
                out.append(f"{where}: flat text differs from {g.file}[{g.s}:{g.e}]")
        elif g.kind == "macro":
            m = fmap.macros.get(g.macro or "")
            if m is None:
                out.append(f"{where}: macro {g.macro!r} is not in macros")
                continue
            piece = flat[g.fa:g.fb]
            if piece not in (m["expanded"], m["expanded"] + "{}"):
                out.append(f"{where}: flat text {piece!r} is not the expansion of \\{g.macro}")
            if not re.match(r"\\" + re.escape(g.macro or "") + r"(?![A-Za-z])", text[g.s:g.e]):
                out.append(f"{where}: {g.file}[{g.s}:{g.e}] is not a call of \\{g.macro}")
    if pos != len(flat):
        out.append(f"segments cover [0, {pos}) of a flat text of length {len(flat)}")
    return out


# --------------------------------------------------------------------------------------------- text helpers


def split_lines(text: str) -> list[str]:
    """Lines with their newlines, split at '\\n' only (str.splitlines also splits at \\f, \\x1c, ...)."""
    lines = text.split("\n")
    out = [ln + "\n" for ln in lines[:-1]]
    if lines[-1]:
        out.append(lines[-1])
    return out


def offsets_of(lines: list[str]) -> list[int]:
    """Character offset of the start of every line, plus the total length at the end."""
    out = [0]
    for ln in lines:
        out.append(out[-1] + len(ln))
    return out


def line_no(text: str, pos: int) -> int:
    """1-based line number of character position pos."""
    return text.count("\n", 0, pos) + 1


def strip_comment(line: str) -> str:
    """The line without its TeX comment: a '%' preceded by an even number of backslashes starts one."""
    for m in re.finditer("%", line):
        i = m.start()
        n = 0
        while i - n - 1 >= 0 and line[i - n - 1] == "\\":
            n += 1
        if n % 2 == 0:
            return line[:i]
    return line


def is_comment_line(line: str) -> bool:
    return re.match(r"^\s*%", line) is not None


def prose_numbers(text: str) -> Counter:
    """The typed numbers of text outside comments and outside label/reference/citation/file arguments."""
    live = "".join(strip_comment(ln.rstrip("\n")) + "\n" for ln in split_lines(text))
    live = _NAME_ARG_COMMANDS.sub(" ", live)
    return Counter(_NUMBER.findall(live))


def short(text: str, limit: int = 90) -> str:
    """One-line abbreviation of text for the report."""
    flat = re.sub(r"\s+", " ", text).strip()
    return flat if len(flat) <= limit else flat[: limit - 3] + "..."


def md_code(text: str) -> str:
    """text as a markdown code span, whatever backticks it contains."""
    if text == "":
        return "(empty)"
    run = max((len(m) for m in re.findall(r"`+", text)), default=0)
    fence = "`" * (run + 1)
    pad = " " if text.startswith("`") or text.endswith("`") else ""
    return f"{fence}{pad}{text}{pad}{fence}"


def md_block(text: str, limit: int = 1500) -> str:
    """text as a fenced markdown block, cut at limit characters."""
    if len(text) > limit:
        text = text[:limit] + "\n[... cut ...]"
    run = max((len(m) for m in re.findall(r"`+", text)), default=0)
    fence = "`" * max(3, run + 1)
    return f"{fence}tex\n{text.rstrip(chr(10))}\n{fence}"


# -------------------------------------------------------------------------------------------------- hunks


@dataclass
class Hunk:
    """One changed block of main.tex, trimmed: base text [a, b) of map.flat_text becomes new."""

    number: int  # 1-based, in main.tex order
    a: int
    b: int
    new: str
    edited_pos: int  # character position of the hunk in the edited main.tex
    edited_line: int  # its line number there


def compute_hunks(base: str, edited: str) -> list[Hunk]:
    """SPEC 5.1: line diff of base against edited, a hunk for each non-equal block, common prefix/suffix trimmed."""
    bl, el = split_lines(base), split_lines(edited)
    boff, eoff = offsets_of(bl), offsets_of(el)
    hunks: list[Hunk] = []
    matcher = difflib.SequenceMatcher(None, bl, el, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        a, b = boff[i1], boff[i2]
        old, new = base[a:b], edited[eoff[j1]:eoff[j2]]
        p = 0
        while p < len(old) and p < len(new) and old[p] == new[p]:
            p += 1
        q = 0
        while q < len(old) - p and q < len(new) - p and old[len(old) - 1 - q] == new[len(new) - 1 - q]:
            q += 1
        pos = eoff[j1] + p
        hunks.append(Hunk(len(hunks) + 1, a + p, b - q, new[p:len(new) - q], pos, line_no(edited, pos)))
    return hunks


# ------------------------------------------------------------------------------------------- classification


@dataclass
class Conflict:
    """An edit that is not applied, with what the report says about it."""

    kind: str  # number | generated | structural | merge
    hunks: list[Hunk]
    old: str
    new: str
    details: list[str]  # markdown bullet texts: the macro or generator to change, the reason
    context: str = ""  # the edited main.tex line(s) around the hunk


@dataclass
class Edit:
    """An edit to apply: source range [s, e) of file becomes new (in the flatten-time text of the file)."""

    file: str
    s: int
    e: int
    new: str
    hunk: Hunk
    old: str


class Classifier:
    """SPEC 5.2: decides, for each hunk, apply (an Edit) or conflict (a Conflict)."""

    def __init__(self, fmap: FlatMap, edited: str):
        self.m = fmap
        self.base = fmap.flat_text
        self.edited = edited

    # -- segment lookup

    def spans_overlapping(self, a: int, b: int) -> list[Segment]:
        """Nonzero-length segments meeting [a, b) (a < b)."""
        m = self.m
        i = max(bisect.bisect_right(m.span_starts, a) - 1, 0)
        out = []
        while i < len(m.spans) and m.spans[i].fa < b:
            if m.spans[i].fb > a:
                out.append(m.spans[i])
            i += 1
        return out

    def span_ending_at(self, pos: int, kinds: tuple[str, ...]) -> Segment | None:
        m = self.m
        i = bisect.bisect_left(m.span_starts, pos) - 1
        if 0 <= i < len(m.spans) and m.spans[i].fb == pos and m.spans[i].kind in kinds:
            return m.spans[i]
        return None

    def span_starting_at(self, pos: int, kinds: tuple[str, ...]) -> Segment | None:
        m = self.m
        i = bisect.bisect_left(m.span_starts, pos)
        if i < len(m.spans) and m.spans[i].fa == pos and m.spans[i].kind in kinds:
            return m.spans[i]
        return None

    def span_containing(self, pos: int) -> Segment | None:
        """The nonzero-length segment with fa < pos < fb, if any."""
        m = self.m
        i = bisect.bisect_right(m.span_starts, pos) - 1
        if 0 <= i < len(m.spans) and m.spans[i].fa < pos < m.spans[i].fb:
            return m.spans[i]
        return None

    def points_inside(self, a: int, b: int) -> list[Segment]:
        """Zero-length segments strictly inside (a, b)."""
        return [g for g in self.m.points if a < g.fa < b]

    def is_generated(self, g: Segment) -> bool:
        """A segment whose text belongs to a generated file (never edited back)."""
        return g.kind == "gen" or (g.kind in ("src", "macro") and self.m.kind_of(g.file) != "hand")

    # -- reporting helpers

    def context(self, h: Hunk) -> str:
        """The edited main.tex lines that hold the hunk."""
        start = self.edited.rfind("\n", 0, h.edited_pos) + 1
        end = self.edited.find("\n", h.edited_pos + len(h.new))
        return self.edited[start:len(self.edited) if end < 0 else end]

    def generator_of(self, g: Segment) -> str | None:
        if g.generator:
            return g.generator
        if g.kind == "macro" and g.macro in self.m.macros:
            return self.m.macros[g.macro].get("generator")
        return None

    def macro_detail(self, g: Segment, why: str) -> str:
        info = self.m.macros.get(g.macro or "", {})
        value = info.get("value", "")
        expanded = info.get("expanded", "")
        shown = md_code(expanded) if value == expanded else f"{md_code(expanded)} (raw value {md_code(value)})"
        gen = info.get("generator") or g.generator
        return (
            f"Macro {md_code(chr(92) + str(g.macro))}, old value {shown}, defined in "
            f"{md_code(str(info.get('file')))}; generator to change: "
            f"{md_code(gen) if gen else '(none named in the number file)'}. {why}"
        )

    def shown(self, h: Hunk, reach: int = 40) -> tuple[str, str]:
        """Old and new text of a hunk for the report, both widened to whole tokens with the same unchanged context
        (a trimmed "2" -> "3" reads as "1{,}082$" -> "1{,}083$")."""
        lo = h.a
        while lo > 0 and h.a - lo < reach and not self.base[lo - 1].isspace():
            lo -= 1
        hi = h.b
        while hi < len(self.base) and hi - h.b < reach and not self.base[hi].isspace():
            hi += 1
        left, right = self.base[lo:h.a], self.base[h.b:hi]
        return left + self.base[h.a:h.b] + right, left + h.new + right

    def conflict(self, kind: str, h: Hunk, details: list[str]) -> Conflict:
        old, new = self.shown(h)
        return Conflict(kind, [h], old, new, details, self.context(h))

    # -- the decision

    def classify(self, h: Hunk) -> Edit | Conflict:
        if h.a == h.b:
            segs = self.attach_insertion(h.a)
            if not segs:
                return self.unattached_insertion(h)
        else:
            segs = self.spans_overlapping(h.a, h.b)
        points = self.points_inside(h.a, h.b) if h.a < h.b else []

        # Generated material wins: the actionable fix is its generator.
        gens = [g for g in segs if self.is_generated(g)]
        if gens:
            details, seen = [], set()
            for g in gens:
                gen = self.m.file_generator(g.file) or (g.generator if g.kind == "gen" else None)
                if g.file not in seen:
                    seen.add(g.file)
                    details.append(
                        f"Generated file {md_code(str(g.file))}; generator to change: "
                        f"{md_code(gen) if gen else '(none named in its first comment lines)'}. "
                        "Generated files are never edited by hand."
                    )
                if g.macro in self.m.macros:  # a number inside the float: its own generator may be another script
                    details.append(self.macro_detail(g, "The edit touches this number inside the float."))
            return self.conflict("generated", h, details)

        reasons = []
        if any(g.kind == "synthetic" for g in segs):
            reasons.append("The edit touches text written by flatten.py (the header comment or a begin/end marker "
                           "of an inlined file), which has no source.")
        for g in points:
            if g.kind == "drop" and g.file is not None and g.s is not None and g.e is not None and g.e > g.s:
                dropped = self.m.source_text(g.file)[g.s:g.e]
                reasons.append(f"The edit spans source text that the flatten removed: {md_code(short(dropped))} "
                               f"in {md_code(g.file)}.")
        files = sorted({g.file for g in segs if g.file is not None})
        if len(files) > 1:
            reasons.append("The edit spans more than one source file: " + ", ".join(md_code(f) for f in files) + ".")
        if not reasons:
            for x, y in zip(segs, segs[1:]):
                if x.e != y.s:
                    reasons.append(f"The edit spans two pieces of {md_code(str(x.file))} that are not contiguous in "
                                   "the source.")
                    break
        if reasons:
            return self.conflict("structural", h, reasons)

        # Empty-valued macros strictly inside the edit: their calls cannot be placed again.
        empty = [g for g in points if g.kind == "macro"]
        if empty:
            return self.conflict("number", h, [self.macro_detail(
                g, "Its value is empty, so the edit gives no position to put the call back at.") for g in empty])

        glued = self.glued_macro(h)
        if glued is not None:
            return self.conflict("number", h, [self.macro_detail(
                glued, "Digits were typed right against its value, which changes the number.")])

        return self.relink(h, segs)

    def attach_insertion(self, pos: int) -> list[Segment]:
        """SPEC 5.2: an insertion attaches to the src segment ending at pos, else the one starting at pos.

        An insertion strictly inside a segment belongs to that segment. As a last resort a hand-written macro
        segment next to pos takes it (for example an insertion between a number and an end marker).
        """
        for g in (self.span_ending_at(pos, ("src",)), self.span_starting_at(pos, ("src",)),
                  self.span_containing(pos)):
            if g is not None:
                return [g]
        for g in (self.span_ending_at(pos, ("macro",)), self.span_starting_at(pos, ("macro",))):
            if g is not None and not self.is_generated(g):
                return [g]
        return []

    def unattached_insertion(self, h: Hunk) -> Conflict:
        """An insertion with no hand-written segment next to it: between generated text and markers."""
        near = [g for g in (self.span_ending_at(h.a, ("gen", "macro", "synthetic")),
                            self.span_starting_at(h.a, ("gen", "macro", "synthetic"))) if g is not None]
        gens = [g for g in near if self.is_generated(g)]
        if gens:
            g = gens[0]
            gen = self.m.file_generator(g.file) or self.generator_of(g)
            return self.conflict("generated", h, [
                f"The text is inserted inside or at the edge of generated file {md_code(str(g.file))}; generator "
                f"to change: {md_code(gen) if gen else '(none named)'}. To add text next to the float, add it in "
                "the hand-written file that inputs it."])
        return self.conflict("structural", h, [
            "The text is inserted between pieces of flattener text (header or markers), which have no source."])

    def glued_macro(self, h: Hunk) -> Segment | None:
        """A hand-written number macro whose value the hunk extends with digits without overlapping it.

        Trimming can turn 1{,}082 -> 1{,}0825 into an insertion of "5" right after the macro's range, which the
        attach rule would apply as "\\rhoosDays{}5". That changes the number, so it is a number conflict.
        """
        old = self.base[h.a:h.b]
        before = self.span_ending_at(h.a, ("macro",))
        if (before is not None and self.base[before.fa:before.fb].rstrip("{}")[-1:].isdigit()
                and _GLUE_AFTER.match(h.new) and not _GLUE_AFTER.match(old)):
            return before
        after = self.span_starting_at(h.b, ("macro",))
        if (after is not None and self.base[after.fa:after.fb][:1].isdigit()
                and _GLUE_BEFORE.search(h.new) and not _GLUE_BEFORE.search(old)):
            return after
        return None

    def relink(self, h: Hunk, segs: list[Segment]) -> Edit | Conflict:
        """SPEC 5.2, apply branch: the source range and the new source text, macros linked back to their calls."""
        first, last = segs[0], segs[-1]
        rel = first.file
        assert rel is not None
        if h.a == h.b and first.kind == "src":
            # Insertion: map the position into the segment (also when it sits at the segment's start or end).
            s = e = first.s + (h.a - first.fa)
            fa = fb = h.a
        else:
            if first.kind == "macro":
                fa, s = first.fa, first.s
            else:
                fa, s = h.a, first.s + (h.a - first.fa)
            if last.kind == "macro":
                fb, e = last.fb, last.e
            else:
                fb, e = h.b, last.s + (h.b - last.fa)
        new_flat = self.base[fa:h.a] + h.new + self.base[h.b:fb]
        source = self.m.source_text(rel)

        # Search the expansions left to right and put each macro's call back in its place.
        pieces, cursor, missing = [], 0, []
        for g in (x for x in segs if x.kind == "macro"):
            flat_piece = self.base[g.fa:g.fb]
            expanded = self.m.macros[g.macro]["expanded"]
            found, used = -1, ""
            for candidate in dict.fromkeys((flat_piece, expanded)):
                found = new_flat.find(candidate, cursor)
                if found >= 0:
                    used = candidate
                    break
            if found < 0:
                missing.append(g)
                continue
            pieces.append(new_flat[cursor:found])
            pieces.append(source[g.s:g.e])
            cursor = found + len(used)
        pieces.append(new_flat[cursor:])
        if missing:
            c = self.conflict("number", h, [self.macro_detail(
                g, "Its value no longer appears in the edited text.") for g in missing])
            return c
        return Edit(rel, s, e, "".join(pieces), h, source[s:e])


# ------------------------------------------------------------------------------------------------- applying


@dataclass
class Applied:
    """One block written to a source file (after merging edits on the same or adjacent lines)."""

    file: str
    line: int  # 1-based line in the written file where the block starts
    parked: bool
    old: str  # the old whole lines
    new: str  # the new whole lines
    written: str  # what replaced the old lines: parking comment (if any) plus the new lines
    hunks: list[Hunk]
    merged: bool = False  # the file went through git merge-file
    line_exact: bool = True  # False when the line is the flatten-time line (merged file, block not located)


@dataclass
class FilePlan:
    """The outcome for one source file: its new text, or a merge conflict."""

    file: str
    path: Path
    edits: list[Edit]
    blocks: list[Applied]
    new_text: str | None = None  # None when nothing is to be written
    newline: str = "\n"
    conflict: Conflict | None = None


def block_bounds(text: str, s: int, e: int, new: str) -> tuple[int, int]:
    """SPEC 5.3: the whole source lines [ls, le) that the edit [s, e) -> new touches.

    A pure insertion of complete lines at a line start touches no old line (ls == le).
    """
    ls = text.rfind("\n", 0, s) + 1
    if e > s:
        nl = text.find("\n", e - 1)
    elif (s == 0 or text[s - 1] == "\n") and (new.endswith("\n") or s == len(text)):
        return s, s
    else:
        nl = text.find("\n", s)
    return ls, (len(text) if nl < 0 else nl + 1)


def patch_text(text: str, edits: list[Edit], date: str) -> tuple[str, list[Applied]]:
    """Apply edits (non-overlapping, any order) to text, parking replaced lines; return the text and the blocks."""
    edits = sorted(edits, key=lambda x: (x.s, x.e))
    groups: list[tuple[int, int, list[Edit]]] = []
    for ed in edits:
        ls, le = block_bounds(text, ed.s, ed.e, ed.new)
        if groups and ls <= groups[-1][1]:  # same or adjacent line: one block
            gls, gle, members = groups[-1]
            if ed.s < members[-1].e:
                raise UnflattenError(f"internal: overlapping edits in {ed.file} at {ed.s}")
            groups[-1] = (gls, max(gle, le), members + [ed])
        else:
            groups.append((ls, le, [ed]))

    out: list[str] = []
    blocks: list[Applied] = []
    cursor = 0
    newlines = 0  # newlines written so far, for the report's line numbers
    for ls, le, members in groups:
        out.append(text[cursor:ls])
        newlines += text.count("\n", cursor, ls)
        old_block = text[ls:le]
        parts, pos = [], ls
        for ed in members:
            parts.append(text[pos:ed.s])
            parts.append(ed.new)
            pos = ed.e
        parts.append(text[pos:le])
        new_block = "".join(parts)
        old_lines = split_lines(old_block)
        nonblank = [ln for ln in old_lines if ln.strip()]
        direct = not nonblank or all(is_comment_line(ln) for ln in nonblank)
        if direct:
            written = new_block
        else:
            # <indent>% <date> (overleaf): parked -- edited in Overleaf. Was:
            # <indent>% <old line without its indent>   (each old line keeps its own indent before the %)
            indent = re.match(r"[ \t]*", nonblank[0]).group(0)
            parked = [indent + PARK_HEADER.format(date=date) + "\n"]
            for ln in old_lines:
                body = ln.rstrip("\n")
                own = re.match(r"[ \t]*", body).group(0)
                parked.append(own + "% " + body[len(own):] + "\n" if body.strip() else indent + "%\n")
            written = "".join(parked) + new_block
        out.append(written)
        blocks.append(Applied(members[0].file, newlines + 1, not direct, old_block, new_block, written,
                              [ed.hunk for ed in members]))
        newlines += written.count("\n")
        cursor = le
    out.append(text[cursor:])
    return "".join(out), blocks


def file_state(path: Path, entry: dict) -> tuple[bool, str | None, str]:
    """(unchanged since the flatten?, current text or None if missing, newline convention of the file)."""
    try:
        data = path.read_bytes()
    except FileNotFoundError:
        return False, None, "\n"
    newline = "\r\n" if b"\r\n" in data else "\n"
    try:
        text = data.decode("utf-8").replace("\r\n", "\n")
    except UnicodeDecodeError as exc:
        raise UnflattenError(f"{path} is not UTF-8: {exc}") from exc
    want = entry.get("sha256")
    shas = {hashlib.sha256(data).hexdigest(), hashlib.sha256(text.encode("utf-8")).hexdigest()}
    return (want in shas or text == entry.get("text")), text, newline


def merge_three_way(current: str, base: str, patched: str) -> tuple[str | None, str]:
    """git merge-file -p current base patched: (merged text or None, a note on what happened)."""
    with tempfile.TemporaryDirectory(prefix="unflatten_merge_") as tmp:
        paths = []
        for name, text in (("current.tex", current), ("base.tex", base), ("overleaf.tex", patched)):
            p = Path(tmp) / name
            p.write_text(text, encoding="utf-8", newline="\n")
            paths.append(str(p))
        cmd = ["git", "merge-file", "-p", "-L", "current", "-L", "flatten base", "-L", "overleaf", *paths]
        try:
            res = subprocess.run(cmd, capture_output=True)
        except FileNotFoundError:
            return None, "git is not on PATH, so the three-way merge could not run."
    if res.returncode == 0:
        return res.stdout.decode("utf-8"), "merged cleanly"
    if 0 < res.returncode < 128:
        return None, f"git merge-file reported {res.returncode} conflict(s)."
    return None, f"git merge-file failed (status {res.returncode}): {res.stderr.decode('utf-8', 'replace').strip()}"


def safe_target(root: Path, rel: str) -> Path:
    """The path of a hand-written source under root; refuses generated/ and anything outside root."""
    p = Path(rel)
    if p.is_absolute() or ".." in p.parts or (p.parts and p.parts[0] == "generated"):
        raise UnflattenError(f"refusing to write {rel!r}: not a hand-written source under the root")
    full = (root / p).resolve()
    if root.resolve() not in full.parents:
        raise UnflattenError(f"refusing to write {rel!r}: outside {root}")
    return full


def plan_file(root: Path, fmap: FlatMap, rel: str, edits: list[Edit], date: str) -> FilePlan:
    """SPEC 5.3-5.4: the new text of one source file, in place or through a three-way merge."""
    path = safe_target(root, rel)
    entry = fmap.sources[rel]
    base = entry["text"]
    patched, blocks = patch_text(base, edits, date)
    plan = FilePlan(rel, path, edits, blocks)
    unchanged, current, newline = file_state(path, entry)
    plan.newline = newline
    if current is None:
        plan.conflict = Conflict("merge", [ed.hunk for ed in edits], "", "", [
            f"{md_code(rel)} no longer exists under the root, so its edits could not be applied."])
        return plan
    if unchanged:
        plan.new_text = patched
        return plan
    merged, note = merge_three_way(current, base, patched)
    if merged is None:
        plan.conflict = Conflict("merge", [ed.hunk for ed in edits], "", "", [
            f"{md_code(rel)} changed after the flatten and the three-way merge of the Overleaf edits did not "
            f"merge cleanly: {note} The file is left untouched; apply these edits by hand."])
        return plan
    plan.new_text = merged
    for blk in plan.blocks:
        # Locate the block in the merged text for the report; keep the flatten-time line if it is not unique.
        blk.merged = True
        hits = [m.start() for m in re.finditer(re.escape(blk.written), merged)] if blk.written else []
        if len(hits) == 1:
            blk.line = line_no(merged, hits[0])
        else:
            blk.line_exact = False
    return plan


# -------------------------------------------------------------------------------------------------- warnings


@dataclass
class Warning_:
    file: str
    line: int
    numbers: list[str]
    hunk: Hunk
    candidates: dict[str, list[str]]  # typed number -> number macros with exactly that value


def digit_warnings(edits: list[Edit], plans: dict[str, FilePlan], fmap: FlatMap) -> list[Warning_]:
    """SPEC 5.5: an applied edit that types new numbers into prose (outside comments) is flagged.

    When a typed number equals the value of existing number macros, the warning names them (at most three).
    """
    by_value: dict[str, list[str]] = {}
    for name, info in sorted(fmap.macros.items()):
        by_value.setdefault(info.get("expanded", ""), []).append(name)
    out = []
    for ed in edits:
        added = prose_numbers(ed.new) - prose_numbers(ed.old)
        if not added:
            continue
        plan = plans.get(ed.file)
        line = next((b.line for b in (plan.blocks if plan else []) if ed.hunk in b.hunks), 0)
        numbers = sorted(added.elements())
        candidates = {n: by_value[n][:3] for n in numbers if n in by_value}
        out.append(Warning_(ed.file, line, numbers, ed.hunk, candidates))
    return out


# ---------------------------------------------------------------------------------------------------- report


def write_report(path: Path | None, lines: list[str]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip("\n") + "\n", encoding="utf-8")


def render_report(args, fmap: FlatMap, status: str, applied: list[Applied], conflicts: list[Conflict],
                  warnings: list[Warning_], held: list[Applied], n_hunks: int) -> list[str]:
    raw = fmap.raw
    commit = raw.get("source_commit") or "unknown"
    dirty = "+dirty" if raw.get("source_dirty") else ""
    out = [
        "# Overleaf unflatten report",
        "",
        f"* Edited file: {md_code(str(args.main))}",
        f"* Map: {md_code(str(fmap.path))} (flatten of commit {md_code(commit + dirty)}, created "
        f"{raw.get('created', '?')})",
        f"* Source root: {md_code(str(args.root))}",
        f"* Parking date: {args.date}",
        f"* Mode: {'dry run (only this report is written)' if args.dry_run else 'apply'}"
        f"{', strict' if args.strict else ''}",
        f"* Result: {status}",
        "",
    ]
    if n_hunks == 0:
        out += ["No changes: the edited `main.tex` equals the text of the flatten recorded in the map. No file was "
                "modified.", ""]
        out += ["## Next step", "", CLOSING_NOTE]
        return out

    title = "## Applied edits" if not args.dry_run else "## Edits that would be applied (dry run)"
    out += [title, ""]
    if not applied:
        out.append("None.")
    for blk in applied:
        where = f"{blk.file}:{blk.line}"
        if blk.merged:
            where += " (three-way merged" + ("" if blk.line_exact else "; line at flatten time") + ")"
        how = "parked and replaced" if blk.parked else "written directly"
        lines = ", ".join(str(h.edited_line) for h in blk.hunks)
        out.append(f"* {md_code(where)} ({how}; main.tex line {lines}): "
                   f"{md_code(short(blk.old))} -> {md_code(short(blk.new))}")
    out.append("")

    if held:
        out += ["## Edits held back by --strict", "", "These edits would apply cleanly but were not written "
                "because conflicts exist:", ""]
        for blk in held:
            out.append(f"* {md_code(blk.file + ':' + str(blk.line))}: {md_code(short(blk.old))} -> "
                       f"{md_code(short(blk.new))}")
        out.append("")

    out += ["## Conflicts (not applied)", ""]
    if not conflicts:
        out.append("None.")
    for i, c in enumerate(conflicts, 1):
        lines = ", ".join(str(h.edited_line) for h in c.hunks)
        out += [f"### {i}. {c.kind.capitalize()} conflict -- main.tex line {lines}", ""]
        for d in c.details:
            out.append(f"* {d}")
        if c.kind != "merge":
            out.append(f"* Old text: {md_code(short(c.old, 300))}")
            out.append(f"* New text: {md_code(short(c.new, 300))}")
            if c.context:
                out += ["* Edited line(s) in main.tex:", "", md_block(c.context)]
        else:
            for h in c.hunks:
                out.append(f"* main.tex line {h.edited_line}: {md_code(short(fmap.flat_text[h.a:h.b], 120))} -> "
                           f"{md_code(short(h.new, 120))}")
        out.append("")

    out += ["## Warnings", ""]
    if not warnings:
        out.append("None.")
    for w in warnings:
        hint = "".join(
            f" {md_code(n)} is the value of " + ", ".join(md_code(chr(92) + c) for c in cands) + "."
            for n, cands in w.candidates.items()
        )
        out.append(f"* {md_code(w.file + ':' + str(w.line))} (main.tex line {w.hunk.edited_line}): new number(s) "
                   f"{', '.join(md_code(n) for n in w.numbers)} typed into prose. The house rule wants every number "
                   f"in prose from a generated macro: add one to the generator and use it here.{hint}")
    out += ["", "## Next step", "", CLOSING_NOTE]
    return out


# ------------------------------------------------------------------------------------------------------ main


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    ap = _Parser(description="Apply Overleaf edits of the flat main.tex back to the hand-written sources.")
    ap.add_argument("--main", required=True, type=Path, help="the main.tex edited in Overleaf")
    ap.add_argument("--map", type=Path, default=None, help="map.json of the flatten (default <root>/overleaf_sync/)")
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="the writeup/ directory (default: %(default)s)")
    ap.add_argument("--date", default=None, help="date for parking comments, YYYY-MM-DD (default: today, UTC)")
    ap.add_argument("--dry-run", action="store_true", help="write nothing except the report")
    ap.add_argument("--strict", action="store_true", help="apply nothing when any conflict exists")
    ap.add_argument("--report", type=Path, default=None, help="markdown report (default <root>/overleaf_sync/)")
    args = ap.parse_args(argv)
    args.root = args.root.resolve()
    if args.map is None:
        args.map = args.root / "overleaf_sync" / "map.json"
    if args.report is None:
        args.report = args.root / "overleaf_sync" / "REPORT.md"
    if args.date is None:
        args.date = datetime.datetime.now(datetime.timezone.utc).date().isoformat()
    elif not re.fullmatch(r"\d{4}-\d{2}-\d{2}", args.date):
        ap.error(f"--date must be YYYY-MM-DD, got {args.date!r}")
    else:
        try:
            datetime.date.fromisoformat(args.date)
        except ValueError:
            ap.error(f"--date is not a valid date: {args.date!r}")
    return args


def read_edited(path: Path, base: str) -> str:
    """The edited main.tex, with editor artifacts that are not edits undone: CRLF line endings, a byte-order mark,
    a final newline that was dropped."""
    try:
        text = path.read_bytes().decode("utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise UnflattenError(f"cannot read {path}: {exc}") from exc
    if text.startswith("\ufeff") and not base.startswith("\ufeff"):
        text = text[1:]
    if "\r\n" in text and "\r\n" not in base:
        text = text.replace("\r\n", "\n")  # Overleaf or a Windows checkout changed the line endings
    if base.endswith("\n") and text and not text.endswith("\n"):
        text += "\n"
    return text


def run(args: argparse.Namespace) -> int:
    fmap = load_map(args.map)
    edited = read_edited(args.main, fmap.flat_text)

    # 1. Diff.
    hunks = [] if edited == fmap.flat_text else compute_hunks(fmap.flat_text, edited)
    if not hunks:
        write_report(args.report, render_report(args, fmap, "no changes (exit 0)", [], [], [], [], 0))
        print("unflatten: no changes; nothing modified.")
        return 0

    # 2. Classify.
    clf = Classifier(fmap, edited)
    edits: list[Edit] = []
    conflicts: list[Conflict] = []
    for h in hunks:
        res = clf.classify(h)
        (edits if isinstance(res, Edit) else conflicts).append(res)

    # 3-4. Plan every file in memory (parking, merging) before writing anything, so --strict can hold all back.
    by_file: dict[str, list[Edit]] = {}
    for ed in edits:
        by_file.setdefault(ed.file, []).append(ed)
    plans: dict[str, FilePlan] = {}
    for rel in sorted(by_file):
        plan = plan_file(args.root, fmap, rel, by_file[rel], args.date)
        plans[rel] = plan
        if plan.conflict is not None:
            conflicts.append(plan.conflict)
    conflicts.sort(key=lambda c: min(h.number for h in c.hunks))

    ready = [p for p in plans.values() if p.conflict is None and p.new_text is not None]
    write = not args.dry_run and not (args.strict and conflicts)
    applied = [b for p in ready for b in p.blocks]
    held = applied if (args.strict and conflicts) else []
    if held:
        applied = []
    if write:
        for p in ready:
            p.path.write_text(p.new_text, encoding="utf-8", newline=p.newline)

    # 5. Warnings, for the edits that are (or would be) applied.
    applied_edits = [ed for p in ready for ed in p.edits] if not held else []
    warnings = digit_warnings(applied_edits, plans, fmap)

    # 6. Report.
    code = 2 if conflicts else 0
    status = (f"{len(hunks)} changed block(s) in main.tex: {len(applied)} block(s) "
              f"{'applied' if write else 'to apply'}, {len(conflicts)} conflict(s), {len(warnings)} warning(s); "
              f"exit {code}.")
    if args.strict and conflicts:
        status += " Nothing was written (--strict)."
    write_report(args.report, render_report(args, fmap, status, applied, conflicts, warnings, held, len(hunks)))
    print(f"unflatten: {status}")
    print(f"unflatten: report written to {args.report}")
    return code


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        return run(args)
    except UnflattenError as exc:
        print(f"unflatten: error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
