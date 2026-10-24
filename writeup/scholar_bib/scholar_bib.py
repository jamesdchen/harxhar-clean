"""Replace writeup/references.bib entries with BibTeX copied from Google Scholar (2026-10-24, user).

    python writeup/scholar_bib/scholar_bib.py list              # writes writeup/scholar_bib/SEARCH_LIST.md
    python writeup/scholar_bib/scholar_bib.py import            # dry run: what the pasted entries would replace
    python writeup/scholar_bib/scholar_bib.py import --apply    # write references.bib (replaced entries are kept)

Workflow: open each link in SEARCH_LIST.md, click "Cite" -> "BibTeX", and paste the entry into
writeup/scholar_bib/scholar_paste.bib (any order, Scholar's own keys). `import` matches every pasted entry to the
paper's citation key by title (first author and year as tie-breakers), so nothing in the paper changes: the entry
takes our key. Scholar's bibliographic fields (type, author, title, journal/booktitle, volume, number, pages, year,
publisher) replace ours; fields Scholar does not give (doi, url, note, howpublished, ...) are kept from our entry.
Titles: plainnat lowercases titles except braced text, and Scholar gives sentence case without braces, so our
brace-protected title is kept when the words are the same; when they differ, Scholar's title is used with its
acronyms braced (warning). Warnings: Scholar truncating authors ("and others"; our author list is then kept), a
different year or entry type,
and pasted entries that match nothing or match ambiguously (those are skipped). Every replaced entry is appended,
with a dated comment, to writeup/scholar_bib/replaced_entries.bib (parked, not deleted).
"""

from __future__ import annotations

import argparse
import datetime as dt
import difflib
import re
import subprocess
import sys
import urllib.parse
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent
WRITEUP = HERE.parent
REPO = WRITEUP.parent
BIB = WRITEUP / "references.bib"
PASTE = HERE / "scholar_paste.bib"
REPLACED = HERE / "replaced_entries.bib"
SEARCH_LIST = HERE / "SEARCH_LIST.md"
BASE_COMMIT = "b3ca310"  # references.bib before the related-work additions of 2026-10-24
SCHOLAR_FIELDS = ("author", "title", "journal", "booktitle", "volume", "number", "pages", "year", "publisher")
FIELD_ORDER = SCHOLAR_FIELDS + ("note", "howpublished", "url", "doi")


@dataclass
class Entry:
    etype: str
    key: str
    fields: dict[str, str] = field(default_factory=dict)  # name (lower case) -> value without outer braces/quotes
    start: int = 0
    end: int = 0  # text[start:end] is the raw entry


def _match_brace(text: str, i: int) -> int:
    """Index just past the brace that closes the one at text[i]."""
    depth = 0
    while i < len(text):
        c = text[i]
        if c == "\\":
            i += 2
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1
    raise ValueError("unbalanced braces")


def _parse_fields(body: str) -> dict[str, str]:
    out: dict[str, str] = {}
    i = 0
    while i < len(body):
        m = re.compile(r"\s*,?\s*([A-Za-z][A-Za-z0-9_\-]*)\s*=\s*").match(body, i)
        if not m:
            break
        name = m.group(1).lower()
        j = m.end()
        if j < len(body) and body[j] == "{":
            k = _match_brace(body, j)
            val = body[j + 1 : k - 1]
        elif j < len(body) and body[j] == '"':
            k = j + 1
            while k < len(body) and not (body[k] == '"' and body[k - 1] != "\\"):
                k += 1
            val = body[j + 1 : k]
            k += 1
        else:
            k = j
            while k < len(body) and body[k] not in ",\n":
                k += 1
            val = body[j:k].strip()
        out[name] = re.sub(r"\s+", " ", val).strip()
        i = k
    return out


def parse_bib(text: str) -> list[Entry]:
    entries = []
    for m in re.finditer(r"(?m)^\s*@([A-Za-z]+)\s*\{\s*([^,\s]+)\s*,", text):
        if m.group(1).lower() in ("comment", "string", "preamble"):
            continue
        brace = text.index("{", m.start())
        end = _match_brace(text, brace)
        body = text[m.end() : end - 1]
        entries.append(Entry(m.group(1).lower(), m.group(2), _parse_fields(body), m.start(), end))
    return entries


def norm_title(t: str) -> str:
    t = re.sub(r"\\[A-Za-z]+\s*", " ", t).replace("&", " and ")
    t = re.sub(r"[{}$\\]", "", t)
    return re.sub(r"[^a-z0-9]+", " ", t.lower()).strip()


def first_surname(author: str) -> str:
    a = re.split(r"\s+and\s+", author.strip())[0]
    a = a.split(",")[0] if "," in a else (a.split()[-1] if a.split() else "")
    return norm_title(a)


def _rel(p: Path) -> str:
    try:
        return str(p.relative_to(REPO))
    except ValueError:
        return str(p)


def protect_caps(title: str) -> str:
    """Brace tokens that plainnat would lowercase but must keep their case: acronyms (two or more capitals, e.g.
    HAR, GARCH, VIX, S\\&P) and tokens with an inner capital; tokens already inside braces are left alone."""
    out, depth, i = [], 0, 0
    for m in re.finditer(r"[{}]|[^\s{}]+", title):
        tok = m.group(0)
        out.append(title[i : m.start()])
        i = m.end()
        if tok == "{":
            depth += 1
        elif tok == "}":
            depth -= 1
        elif depth == 0 and (len(re.findall(r"[A-Z]", tok)) >= 2 or re.search(r"[a-z][A-Z]", tok)):
            core = re.match(r"^([^A-Za-z0-9\\]*)(.*?)([^A-Za-z0-9]*)$", tok)
            if core:
                tok = f"{core.group(1)}{{{core.group(2)}}}{core.group(3)}"
        out.append(tok)
    out.append(title[i:])
    return "".join(out)


def scholar_query_title(t: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[{}]", "", t.replace("\\&", "&").replace("\\", ""))).strip()


def render(e: Entry) -> str:
    names = [n for n in FIELD_ORDER if n in e.fields] + sorted(n for n in e.fields if n not in FIELD_ORDER)
    w = max(len(n) for n in names) if names else 0
    lines = [f"@{e.etype}{{{e.key},"] + [f"  {n.ljust(w)} = {{{e.fields[n]}}}," for n in names] + ["}"]
    return "\n".join(lines)


def cited_keys() -> set[str]:
    keys: set[str] = set()
    files = [WRITEUP / "paper_rehaul.tex", *sorted((WRITEUP / "rehaul").glob("*.tex"))]
    files += sorted((WRITEUP / "generated").glob("*.tex"))
    for f in files:
        for line in f.read_text(encoding="utf-8", errors="replace").splitlines():
            line = re.split(r"(?<!\\)%", line, maxsplit=1)[0]
            for m in re.finditer(r"\\cite[a-zA-Z]*\*?(?:\[[^\]]*\])*\{([^}]*)\}", line):
                keys.update(k.strip() for k in m.group(1).split(",") if k.strip())
    return keys


def base_keys() -> set[str]:
    try:
        old = subprocess.run(
            ["git", "-C", str(REPO), "show", f"{BASE_COMMIT}:writeup/references.bib"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=True,
        ).stdout
    except (subprocess.CalledProcessError, FileNotFoundError):
        return set()
    return {e.key for e in parse_bib(old)}


def cmd_list() -> int:
    entries = {e.key: e for e in parse_bib(BIB.read_text(encoding="utf-8"))}
    cited, old = cited_keys(), base_keys()
    new_today = [k for k in entries if k not in old] if old else []
    rest = sorted(k for k in cited if k in entries and k not in new_today)
    missing = sorted(k for k in cited if k not in entries)

    def row(k: str) -> str:
        e = entries[k]
        t = scholar_query_title(e.fields.get("title", ""))
        q = f'"{t}" {first_surname(e.fields.get("author", "")).title()}'
        url = "https://scholar.google.com/scholar?q=" + urllib.parse.quote_plus(q)
        venue = re.sub(r"[{}\\]", "", e.fields.get("journal", e.fields.get("booktitle", e.fields.get("note", ""))))
        return f"| `{k}` | [{t}]({url}) | {first_surname(e.fields.get('author', '')).title()}, {e.fields.get('year', '')} | {venue[:60]} |"

    head = "| Key | Search on Google Scholar (click, then Cite -> BibTeX) | First author, year | Venue (ours) |\n|---|---|---|---|"
    out = [
        "# Citations to replace with Google Scholar BibTeX",
        "",
        "Workflow: click a title, open the right result's **Cite -> BibTeX**, and paste the entry into",
        "`writeup/scholar_bib/scholar_paste.bib` (any order; keep Scholar's key). Then run",
        "`python writeup/scholar_bib/scholar_bib.py import` (dry run) and `... import --apply`.",
        "The importer maps each pasted entry to the key below by title, so the paper needs no edits.",
        "If Scholar lists a working paper and a journal version, take the journal version unless the text",
        "cites the working paper on purpose (e.g. `RahimikiaPoon2020`, kept as the 2020 SSRN paper for priority).",
        "",
        f"## 1. Added to the bibliography on 2026-10-24 ({len(new_today)}): check these first",
        "",
        head,
        *[row(k) for k in new_today],
        "",
        f"## 2. Every other entry the paper cites ({len(rest)})",
        "",
        head,
        *[row(k) for k in rest],
    ]
    if missing:
        out += ["", "## Cited but not in references.bib", "", *[f"- `{k}`" for k in missing]]
    SEARCH_LIST.write_text("\n".join(out) + "\n", encoding="utf-8", newline="\n")
    print(
        f"wrote {_rel(SEARCH_LIST)}: {len(new_today)} added today, {len(rest)} other cited, "
        f"{len(missing)} cited but missing"
    )
    return 0


def best_match(p: Entry, ours: list[Entry]) -> tuple[Entry | None, float, str]:
    pt, pa, py = (
        norm_title(p.fields.get("title", "")),
        first_surname(p.fields.get("author", "")),
        p.fields.get("year", ""),
    )
    scored = []
    for e in ours:
        r = difflib.SequenceMatcher(None, pt, norm_title(e.fields.get("title", ""))).ratio()
        bonus = (0.05 if pa and pa == first_surname(e.fields.get("author", "")) else 0.0) + (
            0.02 if py and py == e.fields.get("year", "") else 0.0
        )
        scored.append((r + bonus, r, e))
    scored.sort(key=lambda s: -s[0])
    if not scored:
        return None, 0.0, "no entries"
    s0, r0, e0 = scored[0]
    if r0 < 0.80:
        return None, r0, f"best title match only {r0:.2f} ({e0.key})"
    if len(scored) > 1 and scored[1][0] > s0 - 0.03:
        return None, r0, f"ambiguous: {e0.key} vs {scored[1][2].key}"
    return e0, r0, ""


def cmd_import(apply: bool) -> int:
    if not PASTE.exists() or not parse_bib(PASTE.read_text(encoding="utf-8")):
        print(f"nothing to import: paste Scholar BibTeX into {_rel(PASTE)}")
        return 1
    raw = BIB.read_bytes().decode("utf-8")
    nl = "\r\n" if "\r\n" in raw else "\n"
    text = raw.replace("\r\n", "\n")
    ours = parse_bib(text)
    pasted = parse_bib(PASTE.read_text(encoding="utf-8").replace("\r\n", "\n"))
    plan: dict[str, tuple[Entry, Entry]] = {}
    for p in pasted:
        e, r, why = best_match(p, ours)
        if e is None:
            print(f"SKIP  {p.key}: {why}")
            continue
        if e.key in plan:
            print(f"SKIP  {p.key}: {e.key} already matched by {plan[e.key][1].key}")
            continue
        plan[e.key] = (e, p)
        print(f"MATCH {p.key} -> {e.key} (title similarity {r:.2f})")
    stamp = dt.date.today().isoformat()
    replaced_log = []
    for key, (e, p) in sorted(plan.items(), key=lambda kv: -kv[1][0].start):
        new = Entry(p.etype, key, dict(e.fields))
        for f in SCHOLAR_FIELDS:
            new.fields.pop(f, None)
        for f, v in p.fields.items():
            if f in SCHOLAR_FIELDS or f not in new.fields:
                new.fields[f] = v
        # plainnat lowercases titles except braced text: keep our (brace-protected) title when the words are the
        # same; otherwise take Scholar's and brace its acronyms, and say so.
        if norm_title(p.fields.get("title", "")) == norm_title(e.fields.get("title", "")):
            new.fields["title"] = e.fields.get("title", new.fields.get("title", ""))
        elif "title" in p.fields:
            new.fields["title"] = protect_caps(p.fields["title"])
            print(f"WARN  {key}: title wording differs from ours; Scholar's title used with acronyms braced, check it")
        if re.search(r"\band others\b", p.fields.get("author", "")):
            new.fields["author"] = e.fields.get("author", p.fields["author"])
            print(f"WARN  {key}: Scholar truncated the authors ('and others'); our author list is kept")
        if p.fields.get("year") != e.fields.get("year"):
            print(f"WARN  {key}: year {e.fields.get('year')} -> {p.fields.get('year')} (working paper vs published?)")
        if p.etype != e.etype:
            print(f"WARN  {key}: entry type @{e.etype} -> @{p.etype}")
        for f in sorted(set(e.fields) | set(new.fields)):
            if e.fields.get(f) != new.fields.get(f):
                print(f"      {key}.{f}: {e.fields.get(f)!r} -> {new.fields.get(f)!r}")
        replaced_log.append(
            f"% {stamp}: replaced in references.bib by the Google Scholar entry {p.key}\n"
            + text[e.start : e.end]
            + "\n"
        )
        text = text[: e.start] + render(new) + text[e.end :]
    print(f"{len(plan)} of {len(pasted)} pasted entries matched")
    if not apply:
        print("dry run: nothing written (add --apply to write references.bib)")
        return 0
    BIB.write_bytes(text.replace("\n", nl).encode("utf-8"))
    with REPLACED.open("a", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(reversed(replaced_log)))
    print(f"wrote {_rel(BIB)}; replaced entries kept in {_rel(REPLACED)}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list", help="write SEARCH_LIST.md")
    imp = sub.add_parser("import", help="match scholar_paste.bib to our keys (dry run unless --apply)")
    imp.add_argument("--apply", action="store_true")
    args = ap.parse_args(argv)
    return cmd_list() if args.cmd == "list" else cmd_import(args.apply)


if __name__ == "__main__":
    sys.exit(main())
