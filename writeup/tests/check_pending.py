"""No generated file carries a number for a pending arm (2026-10-23, writeup/final_run.py PENDING).

Run in the environment of the generators it checks (REHAUL_FINAL_RUN, REHAUL_FINAL_WINDOW, REHAUL_RF_RUN, ...), after
them, on their output directory (final_run.GEN, or --gen DIR):

  python writeup/tests/check_pending.py [--gen DIR]

Under the switch, a pooled random forest without final forecasts is pending (final_run.RF_PENDING); the pooled MinGRU
is parked and printed by no generator, so only the forest's cells and macros can be pending in a generated file.  The
check fails (exit 1, one line a failure) unless:

 1. MARKED ROWS (every .tex file of the directory): a table line that ends with final_run.ROW_MARK holds no digit
    outside LaTeX commands, \\tnote{...}, \\multicolumn{n}{...}, \\addlinespace[...] and the row end \\\\[...]: its
    cells are \\rhPending{} (or empty).
 2. FOREST ROWS (FOREST_ROWS: the tables of the paper's generators that print the pooled forest, the pooled section
    of each): with the forest pending, each of its rows there, and each continuation line under it (a line whose
    first cell is empty, e.g. the t-statistics), is marked (and passes 1); a table may also leave the pending forest
    out.  With the forest finished, no row is marked, the rows are there and each holds a number.
 3. FOREST MACROS (FOREST_MACRO, in the macro files of the paper's generators): with the forest pending each is
    \\rhPending{}; with it finished none is.
 4. \\rhPending: every file that uses \\rhPending defines it (final_run.PENDING_DEF); in the paper's generators' files
    it is used only when an arm is pending.
 5. COUNTS (when the macro files are there): \\rhrPoolN = \\rhrPoolNDone = final_run.pool_n(), \\rhrPoolNAll =
    final_run.POOL_N_ALL, \\rhrPoolNPending = their difference, \\rhrPoolPending = final_run.pending_text();
    \\rhorAfPairs = the finished classes of Table 1, \\rhorAfPairsAll = 6, \\rhorAfTreeN, \\rhorPending = the pending
    text; \\rhePoolNDone = \\rhrPoolN, \\rhePoolNAll = \\rhrPoolNAll, \\rheDefN / \\rheDefNAll (the pooled classes,
    finished / all).  Each macro file is checked when it is in the directory.

With the switch off nothing is pending, so 2-4 check that no pending mark, cell or macro is there.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import final_run as fr  # noqa: E402

# the files the paper's pooled generators write (make_rehaul_results / _economics / _oos_table / _oos_rerun /
# _data_methods); other generators' files get checks 1 and the definition half of 4 only
OWN = re.compile(
    r"^(rehaul_results_|rehaul_econ_|rehaul_oos_numbers|rehaul_oos_rerun_numbers|rehaul_dm_|rehaul_app_hac_robust)"
)
# file -> (start of the pooled section, its end, the row label); None start: the whole table; the section may recur
_FIRST = r"^\s*(\\quad\s*)?Random forest\b"  # the forest in the first cell
_SECOND = r"^[^&]*&\s*Random forest\b"  # ... in the second (the rank tables)
FOREST_ROWS: dict[str, tuple[str | None, str | None, str]] = {
    "rehaul_results_oos.tex": (r"Tree ensembles", None, _FIRST),  # Table 1
    "rehaul_results_t1_axis1.tex": (r"Panel B: Intraday", None, _FIRST),
    "rehaul_results_t2_pooled.tex": (None, None, _FIRST),
    "rehaul_results_t3_ranking.tex": (r"Panel B: Intraday", r"Panel C", _SECOND),
    "rehaul_results_t3_full.tex": (r"Panel B: Intraday", r"Panel C", _SECOND),
    "rehaul_app_hac_robust.tex": (r"Panel A: Pooled", r"Panel D", _FIRST),
    "rehaul_econ_e1_control.tex": (r"Panel A: Pooled intraday", r"Panel B", _FIRST),
    "rehaul_econ_e4_sizing.tex": (
        r"Pooled intraday models",
        r"Closing-interval models",
        _FIRST,
    ),
    "rehaul_econ_sharpe_defs.tex": (None, None, _FIRST),
}
# macros of the pooled forest (names in the generators' styles); the closing-interval forest's (...RfClose...) are
# other runs and are not matched
FOREST_MACRO = re.compile(
    r"^(rhr\w*Rf\w*Pool\w*|rhr\w*Pool\w*Rf\w*|rhe\w*Rf\w*Pool\w*|rhe\w*Pool\w*Rf\w*|rheDef\w*Rf|rhor(Dm|P)Rf[A-Z])$"
)
_DEF = re.compile(r"\\(?:providecommand|newcommand|renewcommand)\{\\([A-Za-z]+)\}")
_STRIP = [
    (re.compile(r"%.*$"), ""),
    (re.compile(r"\\tnote\{[^}]*\}"), ""),
    (re.compile(r"\\multicolumn\{\d+\}\{[^}]*\}"), ""),
    (re.compile(r"\\(addlinespace|vspace)\[[^\]]*\]"), ""),
    (re.compile(r"\\\\\[[^\]]*\]"), ""),
    (re.compile(r"\\[A-Za-z]+"), ""),
]


def macros(text: str) -> dict[str, str]:
    """name -> value of every \\providecommand / \\newcommand / \\renewcommand of the text (the last one wins; an empty
    \\providecommand{\\X}{} before a \\renewcommand is skipped); the value is read with brace matching."""
    out: dict[str, str] = {}
    for m in _DEF.finditer(text):
        i = m.end()
        if i >= len(text) or text[i] != "{":
            continue
        depth, j = 0, i
        while j < len(text):
            depth += {"{": 1, "}": -1}.get(text[j], 0)
            if depth == 0:
                break
            j += 1
        val = text[i + 1 : j]
        if val == "" and text.startswith(r"\renewcommand", j + 1):
            continue
        out[m.group(1)] = val
    return out


def has_digit(line: str) -> bool:
    for rx, rep in _STRIP:
        line = rx.sub(rep, line)
    return any(c.isdigit() for c in line)


def forest_rows(
    lines: list[str], spec: tuple[str | None, str | None, str]
) -> list[list[str]]:
    """The forest's rows of the pooled sections (each: its line and its continuation lines)."""
    start, end, label = spec
    inside, rows = start is None, []
    for i, ln in enumerate(lines):
        if start is not None and re.search(start, ln):
            inside = True
            continue
        if end is not None and re.search(end, ln):
            inside = False
            continue
        if inside and re.search(label, ln):
            row = [ln]
            k = i + 1
            while k < len(lines) and re.match(r"^\s*&", lines[k]):
                row.append(lines[k])
                k += 1
            rows.append(row)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--gen", type=Path, default=fr.GEN, help="directory of the generated .tex files"
    )
    gen = ap.parse_args().gen
    rf = fr.RF_PENDING
    any_pending = bool(fr.PENDING_TREES)
    print(fr.tag())
    print(f"checking {gen} (forest {'PENDING' if rf else 'finished'})")
    bad: list[str] = []
    n_marked = n_rows = n_mac = 0
    files = sorted(gen.glob("*.tex"))
    for f in files:
        text = f.read_text(encoding="utf-8", errors="replace")
        lines = text.splitlines()
        own = bool(OWN.match(f.name))
        # 1. marked rows hold no number
        for i, ln in enumerate(lines, 1):
            if ln.rstrip().endswith(fr.ROW_MARK):
                n_marked += 1
                if has_digit(ln[: ln.rindex(fr.ROW_MARK)]):
                    bad.append(
                        f"{f.name}:{i}: a pending row carries a number: {ln.strip()[:120]}"
                    )
                if own and not any_pending:
                    bad.append(f"{f.name}:{i}: a pending row, but no arm is pending")
        # 4. \rhPending is defined where used, and used (in the paper's generators' files) only when pending
        body = text.replace(fr.PENDING_DEF, "")
        if r"\rhPending" in body:
            if fr.PENDING_DEF not in text:
                bad.append(f"{f.name}: uses \\rhPending without {fr.PENDING_DEF}")
            if own and not any_pending:
                bad.append(f"{f.name}: uses \\rhPending, but no arm is pending")
        # 2. the forest's rows
        if f.name in FOREST_ROWS:
            rows = forest_rows(lines, FOREST_ROWS[f.name])
            if not rf and not rows:
                bad.append(
                    f"{f.name}: no row of the pooled forest found (the check's section spec is stale)"
                )
            for row in rows:
                n_rows += 1
                for ln in row:
                    marked = ln.rstrip().endswith(fr.ROW_MARK)
                    if rf and not marked:
                        bad.append(
                            f"{f.name}: the pending forest's row is not marked: {ln.strip()[:120]}"
                        )
                    if not rf and marked:
                        bad.append(
                            f"{f.name}: the finished forest's row is marked pending: {ln.strip()[:120]}"
                        )
                if not rf and not has_digit(row[0]):
                    bad.append(
                        f"{f.name}: the finished forest's row holds no number: {row[0].strip()[:120]}"
                    )
        # 3. the forest's macros
        if own:
            mac = macros(text)
            for name, val in mac.items():
                if not FOREST_MACRO.match(name):
                    continue
                n_mac += 1
                is_p = val.strip() == fr.PENDING_TEX
                if rf and not is_p:
                    bad.append(
                        f"{f.name}: \\{name} = {val[:60]!r}, the pending forest's macro carries a value"
                    )
                if not rf and is_p:
                    bad.append(
                        f"{f.name}: \\{name} is pending, but the forest is finished"
                    )
    # 5. the counts, each checked in the macro file that defines it (skipped when that file is not in the directory,
    # e.g. a scratch run of one generator; a file that is there must define its macros)
    n_done, n_all = fr.pool_n(), fr.POOL_N_ALL
    want = {
        "rehaul_results_numbers.tex": {
            "rhrPoolN": str(n_done),
            "rhrPoolNDone": str(n_done),
            "rhrPoolNAll": str(n_all),
            "rhrPoolNPending": str(n_all - n_done),
            "rhrPoolPending": fr.pending_text(),
        },
        "rehaul_oos_rerun_numbers.tex": {
            "rhorPending": fr.pending_text(),
            "rhorAfPairsAll": str(len(fr.POOL_LINEAR) + len(fr.POOL_TREES)),
            "rhorAfPairs": str(len(fr.pool_classes())),
            "rhorAfTreeN": str(len(fr.POOL_TREES) - len(fr.PENDING_TREES)),
        },
        "rehaul_econ_numbers.tex": {
            "rhePoolNDone": str(n_done),
            "rhePoolNAll": str(n_all),
            "rheDefN": str(len(fr.pool_classes())),
            "rheDefNAll": str(len(fr.POOL_LINEAR) + len(fr.POOL_TREES)),
        },
    }
    n_cnt = 0
    for fname, names in want.items():
        f = gen / fname
        if not f.is_file():
            continue
        mac = macros(f.read_text(encoding="utf-8", errors="replace"))
        for name, v in names.items():
            if name not in mac:
                if fr.FINAL:
                    bad.append(f"{fname}: \\{name} is not defined")
                continue
            n_cnt += 1
            if mac[name].strip() != v:
                bad.append(f"{fname}: \\{name} = {mac[name]!r}, expected {v!r}")
    print(
        f"{len(files)} files: {n_marked} pending rows, {n_rows} pooled-forest rows, {n_mac} pooled-forest macros, "
        f"{n_cnt} count macros checked"
    )
    for b in bad:
        print("FAIL", b)
    print("ok" if not bad else f"{len(bad)} failure(s)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
