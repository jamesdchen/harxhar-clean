#!/usr/bin/env python3
"""Regenerate the paper's generated files and report what moved (2026-10-29).

One run does what was done by hand after every generator change:

1. RUN      the selected writeup/make_rehaul_*.py generators (default: every one on disk), each in every mode the
            committed outputs come from (GENERATORS below), under REHAUL_FINAL_RUN=1, up to --jobs at a time; a
            generator that reads another one's output runs after it.  Each run's stdout / stderr go to
            writeup/tools/regen_logs/; a failure stops nothing else and is reported.  A generator whose interpreter
            lacks what it needs (numba; pandas < 3) is not run and is reported (--python NAME=PATH gives it one).
2. MACROS   every macro of writeup/generated/*.tex against --base (default HEAD): \\newcommand{\\x}{v},
            \\renewcommand, \\providecommand{\\x}{}\\renewcommand{\\x}{v}; values with nested braces.  Counts changed /
            added / removed per file; lists the non-numeric changes (words, verdicts, yes / no), the p-values whose old
            and new values lie on opposite sides of 0.05 or 0.10, and the sign flips.
3. IMPACT   for each changed or removed macro, the live lines of writeup/paper_rehaul.tex and writeup/rehaul/*.tex that
            use it (comment text and \\iffalse ... \\fi blocks skipped); the generated files whose content changed
            (git diff --numstat against --base; line endings ignored).
4. COMPILE  (--compile) copies paper_rehaul.tex, references.bib, rehaul/, generated/ and figures/ to a temporary
            directory and runs pdflatex, bibtex, pdflatex, pdflatex there (never in writeup/); reports the errors, the
            undefined references, citations and control sequences, and the page count.

The whole report goes to writeup/tools/regen_report.md; a short summary is printed.  Standard library only.

Run (from anywhere; the repository is the one this file is in):
    python writeup/tools/regen.py                              # every generator, then diff + impact against HEAD
    python writeup/tools/regen.py --only ladder,channels       # these generators only (names without make_rehaul_)
    python writeup/tools/regen.py --no-run --compile           # the current tree: diff + impact + compile
    python writeup/tools/regen.py --no-run --base HEAD~3       # what moved since HEAD~3
    python writeup/tools/regen.py --python fi=PATH --python disclosures=PATH   # those under another interpreter
    python writeup/tools/regen.py --list                       # the generators, their modes and run order

Exit codes: 0 done (every generator ran clean; the compile, if asked, has no error); 1 a generator failed or the
compile has errors; 2 bad usage (unknown generator, bad --base).
"""

from __future__ import annotations

import argparse
import functools
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Iterator
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
WRITEUP = REPO / "writeup"
GEN_DIR = WRITEUP / "generated"
TOOLS = WRITEUP / "tools"
REPORT = TOOLS / "regen_report.md"
LOG_DIR = TOOLS / "regen_logs"
PAPER = "paper_rehaul"


# ---------------------------------------------------------------------------------------------------------------------
# The generators.  Key: the name after make_rehaul_ (writeup/make_rehaul_<key>.py).  modes: one argument list per run,
# the runs the committed writeup/generated/ files come from (each "_prepatch" / "_everybar" file is its own mode); a
# generator's modes run one after another.  after: the generators whose OUTPUT this one reads (a file under
# writeup/generated/ or results/, named in the comment above the entry; a gate's cross-check counts); it starts once
# they have finished.  Everything else runs in parallel: no two generators write the same file (2026-10-29: each one
# run alone in a scratch worktree, the files it wrote listed, and each one's source searched for the others' outputs).  needs: what the generator needs of its interpreter, "module" or "module<N" (major version below N):
# src/backtest/executor.py imports numba, which the miniconda base env lacks; make_rehaul_fi.py's fig_inter writes into
# DataFrame.to_numpy(), read-only under pandas 3.  Under an interpreter that falls short the generator is not run and
# the report says so (give it one with --python <name>=PATH).  A make_rehaul_*.py on disk that is not listed here runs
# once, with no arguments and no ordering, and the report says so: add it here.
# ---------------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Gen:
    modes: tuple[tuple[str, ...], ...] = ((),)
    after: tuple[str, ...] = ()
    needs: tuple[str, ...] = ()
    note: str = ""


GENERATORS: dict[str, Gen] = {
    "ablations": Gen(),
    "appendix": Gen(),
    "beyond_factor": Gen(),
    "beyond_factor_lin": Gen(),
    "beyond_ladder": Gen(),
    # results/oos_final_2014start/table1_w3000/{metrics,dm}.csv
    "channels": Gen(after=("oos_rerun",)),
    "ci_har": Gen(),
    # the gates read rehaul_oos_rerun_numbers, rehaul_results_oos, rehaul_results_numbers, rehaul_results_oos_last30,
    # rehaul_robust_w3500_numbers (.tex) and results/oos_final_2014start/, results/rehaul_robust_window_final/metrics.csv
    "close_accuracy": Gen(after=("oos_rerun", "oos_table", "results", "robust_window")),
    "complexity": Gen(modes=((), ("--grid", "results/ladder_2026-10-26/grid", "--tag", "_prepatch"))),
    "data_methods": Gen(needs=("numba",)),
    "disclosures": Gen(needs=("numba",)),
    # rehaul_econ_main_numbers.tex, rehaul_econ_decomp_numbers.tex
    "econ_beta": Gen(after=("economics", "econ_main", "econ_decomp")),
    # rehaul_econ_numbers.tex, rehaul_results_numbers.tex
    "econ_decomp": Gen(after=("economics", "results")),
    # rehaul_econ_e1_control, rehaul_econ_numbers, rehaul_app_hac_robust, rehaul_results_numbers (.tex)
    "econ_main": Gen(after=("economics", "results")),
    "economics": Gen(),
    "factor": Gen(),
    "fi": Gen(needs=("pandas<3",)),
    # rehaul_live_numbers.tex
    "fi_live": Gen(modes=((), ("--prepatch",)), after=("live_windows",)),
    "fi_pooled": Gen(modes=((), ("--prepatch",))),
    # rehaul_ablation_numbers.tex
    "fi_reconcile": Gen(after=("ablations",)),
    # results/rehaul_robust_window_final/{metrics,dm_window}.csv
    "fip_lc": Gen(after=("robust_window",)),
    # rehaul_econ_numbers.tex, rehaul_econ_main_numbers.tex
    "holiday": Gen(after=("economics", "econ_main")),
    "ladder": Gen(modes=((), ("--prepatch",), ("--everybar",))),
    "live_windows": Gen(needs=("numba",)),
    # results/ablations_2014start_w3000/scores_fomcfix/ only (the scorer's files and robust2019.csv)
    "macro_summary": Gen(),
    # rehaul_econ_numbers, rehaul_results_numbers, rehaul_results_oos (.tex), results/oos_final_2014start/
    "mcs": Gen(after=("economics", "results", "oos_rerun")),
    "oos_rerun": Gen(),  # --span 2014: its default under REHAUL_FINAL_RUN=1
    # rehaul_results_numbers.tex
    "oos_table": Gen(after=("results",)),
    "pooled_extras": Gen(),
    "results": Gen(),
    "robust_window": Gen(),
    # rehaul_econ_numbers, rehaul_oos_rerun_numbers, rehaul_results_oos, rehaul_robust_w3500_numbers,
    # rehaul_close_accuracy_numbers, rehaul_econ_main_numbers (.tex)
    "train_arms": Gen(after=("economics", "oos_rerun", "robust_window", "close_accuracy", "econ_main")),
    "tuning": Gen(),
    # rehaul_econ_numbers.tex
    "window_sweep": Gen(after=("economics",)),
}

# Environment of every generator run (over the caller's).  REHAUL_FINAL_RUN=1: the final pooled design the paper
# prints (writeup/final_run.py; the FOMC-patched runs are its default at W = 3,000); generators that do not read the
# switch ignore it.  MPLBACKEND: no figure window.  PYTHONIOENCODING: a generator's print of a non-ASCII character
# cannot fail on the pipe.
ENV = {"REHAUL_FINAL_RUN": "1", "MPLBACKEND": "Agg", "PYTHONIOENCODING": "utf-8"}

P_THRESHOLDS = (0.05, 0.10)
GIT = ("git", "-c", "core.safecrlf=false", "-c", "core.longpaths=true")


# ============================================================================================ step 1: run the generators
@dataclass
class Run:
    gen: str
    args: tuple[str, ...]
    rc: int | None  # None: timed out or could not start
    secs: float
    log: Path
    err_tail: str = ""
    note: str = ""

    @property
    def ok(self) -> bool:
        return self.rc == 0


def discover() -> list[str]:
    """The generator keys on disk (writeup/make_rehaul_*.py), sorted."""
    return sorted(p.stem.removeprefix("make_rehaul_") for p in WRITEUP.glob("make_rehaul_*.py"))


def spec(name: str) -> Gen:
    return GENERATORS.get(name, Gen(note="not in GENERATORS: one run, no arguments, no ordering"))


def resolve_names(only: str | None, on_disk: list[str]) -> list[str]:
    """The generators to run: --only's comma-separated names (with or without make_rehaul_ / .py), else all on disk."""
    if not only:
        return on_disk
    names = []
    for raw in only.split(","):
        n = raw.strip().removesuffix(".py").removeprefix("writeup/").removeprefix("make_rehaul_")
        if not n:
            continue
        if n not in on_disk:
            raise SystemExit(f"regen: no generator writeup/make_rehaul_{n}.py (on disk: {', '.join(on_disk)})")
        names.append(n)
    return sorted(set(names))


def run_order(names: list[str]) -> list[list[str]]:
    """The generators in waves: each wave's dependencies (GENERATORS[.].after, within `names`) are in earlier waves.
    For display and for the cycle check; the runner itself starts a generator as soon as its own dependencies end."""
    left = {n: {d for d in spec(n).after if d in names} for n in names}
    waves: list[list[str]] = []
    done: set[str] = set()
    while left:
        wave = sorted(n for n, deps in left.items() if deps <= done)
        if not wave:
            raise SystemExit(f"regen: dependency cycle among {sorted(left)}")
        waves.append(wave)
        done.update(wave)
        for n in wave:
            del left[n]
    return waves


def dependents(name: str, names: list[str]) -> set[str]:
    """The generators of `names` that read `name`'s output, directly or through another one."""
    out: set[str] = set()
    frontier = [name]
    while frontier:
        cur = frontier.pop()
        for n in names:
            if cur in spec(n).after and n not in out:
                out.add(n)
                frontier.append(n)
    return out


def start_order(ready: list[str], names: list[str]) -> list[str]:
    """The ready generators in the order they get a free slot: the ones most others wait for first (economics and
    results, the start of the longest chain), then by name."""
    return sorted(ready, key=lambda n: (-len(dependents(n, names)), n))


def downstream_skipped(names: list[str], on_disk: list[str]) -> dict[str, list[str]]:
    """Generators NOT selected that read the output of a selected one (their files may now be stale)."""
    out: dict[str, list[str]] = {}
    for n in on_disk:
        if n not in names:
            hit = [d for d in spec(n).after if d in names]
            if hit:
                out[n] = hit
    return out


_NEEDS_CHECK = """
import importlib.metadata as md, importlib.util as u, sys
bad = []
for need in sys.argv[1:]:
    mod, _, below = need.partition("<")
    if u.find_spec(mod) is None:
        bad.append(need)
    elif below and int(md.version(mod).split(".")[0]) >= int(below):
        bad.append(need + " (has " + md.version(mod) + ")")
print("|".join(bad))
"""


@functools.cache
def missing_modules(python: str, needs: tuple[str, ...]) -> tuple[str, ...]:
    """The entries of `needs` ("module" or "module<N") that interpreter `python` does not meet (all if it fails)."""
    if not needs:
        return ()
    try:
        p = subprocess.run([python, "-c", _NEEDS_CHECK, *needs], capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.TimeoutExpired):
        return needs
    return tuple(x for x in p.stdout.strip().split("|") if x) if p.returncode == 0 else needs


def parse_pythons(values: list[str] | None) -> dict[str, str]:
    """--python values: PATH (every generator) or NAME=PATH (one generator) -> {"": default, name: path}.  The
    default before any PATH: $REGEN_PYTHON, else the interpreter running regen.py."""
    out = {"": os.environ.get("REGEN_PYTHON", "").strip() or sys.executable}
    for v in values or []:
        name, sep, path = v.partition("=")
        if sep and not os.path.exists(v):
            out[name.strip().removeprefix("make_rehaul_").removesuffix(".py")] = path
        else:
            out[""] = v
    return out


def run_one(name: str, pythons: dict[str, str], timeout: float) -> list[Run]:
    """Every mode of one generator, in order; stdout and stderr of each go to LOG_DIR/<name>[__mode].log."""
    env = {**os.environ, **ENV}
    python = pythons.get(name, pythons[""])
    lack = missing_modules(python, spec(name).needs)
    runs = []
    for args in spec(name).modes:
        tag = "__".join(a.strip("-").replace("/", "_") for a in args if a.startswith("--"))
        log = LOG_DIR / (name + (f"__{tag}" if tag else "") + ".log")
        cmd = [python, str(WRITEUP / f"make_rehaul_{name}.py"), *args]
        if lack:
            note = f"not run: {python} lacks {', '.join(lack)} (give one: --python {name}=PATH)"
            log.write_text(f"$ {' '.join(cmd)}\n# {note}\n", encoding="utf-8")
            runs.append(Run(name, args, None, 0.0, log, "", note))
            continue
        t0 = time.monotonic()
        try:
            p = subprocess.run(cmd, cwd=REPO, env=env, capture_output=True, timeout=timeout)
            rc, out, err, note = p.returncode, p.stdout, p.stderr, ""
        except subprocess.TimeoutExpired as e:
            rc, out, err, note = None, e.stdout or b"", e.stderr or b"", f"timed out after {timeout:.0f} s"
        except OSError as e:
            rc, out, err, note = None, b"", str(e).encode(), "could not start"
        secs = time.monotonic() - t0
        so, se = out.decode("utf-8", "replace"), err.decode("utf-8", "replace")
        log.write_text(
            f"$ {' '.join(cmd)}\n# cwd {REPO}\n# exit {rc} after {secs:.1f} s {note}\n\n=== stdout\n{so}\n=== stderr\n{se}",
            encoding="utf-8",
        )
        tail = "\n".join((se.strip() or so.strip()).splitlines()[-12:]) if rc != 0 else ""
        runs.append(Run(name, args, rc, secs, log, tail, note))
    return runs


def run_generators(names: list[str], jobs: int, pythons: dict[str, str], timeout: float) -> list[Run]:
    """Step 1.  Runs `names` (each generator's modes in sequence), at most `jobs` generators at a time, a generator
    starting once the generators it reads (within `names`) have finished, whether or not they succeeded."""
    run_order(names)  # the cycle check
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    deps = {n: {d for d in spec(n).after if d in names} for n in names}
    finished: dict[str, list[Run]] = {}
    running: dict[Future[list[Run]], str] = {}
    t0 = time.monotonic()
    with ThreadPoolExecutor(max_workers=max(1, jobs)) as pool:
        while deps or running:
            ready = start_order([n for n, d in deps.items() if d <= finished.keys()], names)
            for n in ready[: max(1, jobs) - len(running)]:
                del deps[n]
                running[pool.submit(run_one, n, pythons, timeout)] = n
            done, _ = wait(running, return_when=FIRST_COMPLETED)
            for fut in done:
                n = running.pop(fut)
                runs = fut.result()
                failed_deps = [d for d in spec(n).after if d in finished and not all(r.ok for r in finished[d])]
                if failed_deps:
                    runs[0].note = (runs[0].note + "; " if runs[0].note else "") + (
                        f"read the output of failed {', '.join(failed_deps)}"
                    )
                finished[n] = runs
                state = "ok" if all(r.ok for r in runs) else "FAILED"
                print(
                    f"  [{len(finished)}/{len(names)}] {n}: {state} ({sum(r.secs for r in runs):.0f} s, "
                    f"{len(runs)} run{'s' if len(runs) > 1 else ''}; {time.monotonic() - t0:.0f} s elapsed)",
                    flush=True,
                )
    return [r for n in names for r in finished[n]]


# ================================================================================================ step 2: macro diff
DEF_RE = re.compile(
    r"\\(newcommand|renewcommand|providecommand)\*?\s*(?:\{\s*\\([A-Za-z@]+)\s*\}|\\([A-Za-z@]+))"
    r"\s*(?:\[\d\]\s*(?:\[[^\]]*\]\s*)?)?\{"
)


def strip_comment(line: str) -> str:
    """The line up to its first unescaped % (TeX ignores the rest)."""
    i = line.find("%")
    while i >= 0:
        k = i
        while k > 0 and line[k - 1] == "\\":
            k -= 1
        if (i - k) % 2 == 0:  # an even number of backslashes before it: a real comment
            return line[:i]
        i = line.find("%", i + 1)
    return line


def close_brace(text: str, i: int) -> int:
    """Index of the } that closes the group whose content starts at text[i]; -1 if it never closes.  Escaped braces
    (\\{, \\}) do not count."""
    depth, j = 1, i
    while j < len(text):
        c = text[j]
        if c == "\\":
            j += 2
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return j
        j += 1
    return -1


def parse_macros(text: str) -> dict[str, str]:
    """Macro name (no backslash) -> value, as LaTeX leaves it after reading `text`: \\newcommand and \\renewcommand
    set the value; \\providecommand sets it only when the name is not defined yet (so \\providecommand{\\x}{}
    \\renewcommand{\\x}{v} gives v).  Comments are dropped first; a value may hold nested braces and span lines."""
    body = "\n".join(strip_comment(ln) for ln in text.splitlines())
    out: dict[str, str] = {}
    pos = 0
    while (m := DEF_RE.search(body, pos)) is not None:
        end = close_brace(body, m.end())
        if end < 0:
            break
        name, value = m.group(2) or m.group(3), body[m.end() : end]
        if m.group(1) != "providecommand" or name not in out:
            out[name] = value
        pos = end + 1
    return out


@dataclass
class Change:
    file: str
    name: str
    old: str | None  # None: added
    new: str | None  # None: removed
    nonnumeric: bool = False
    p_cross: tuple[float, ...] = ()
    sign_flip: bool = False


@dataclass
class FileDiff:
    file: str
    n_new: int
    changed: list[Change] = field(default_factory=list)
    added: list[Change] = field(default_factory=list)
    removed: list[Change] = field(default_factory=list)


_NUM = re.compile(r"\d+(?:\.\d+)?|\.\d+")
_DECOR = re.compile(r"\\(?:ensuremath|textless|textgreater|textminus|lt|gt|le|ge|leq|geq|,|!|;|:)(?![A-Za-z])")


def skeleton(value: str) -> str:
    """The value with every number replaced by # and the number decorations (braces, $, signs, <, thin spaces, the
    {,} thousands separator) dropped: two values with the same skeleton differ in numbers only."""
    s = _DECOR.sub("", value.replace("{,}", ""))
    s = re.sub(r"[\s{}$~<>+\-\u2212]", "", s)
    return _NUM.sub("#", s)


def is_nonnumeric(old: str, new: str) -> bool:
    """True when the change is more than a number moving: words, verdicts, yes / no, a cell turning pending or empty."""
    if not (re.search(r"\d", old) and re.search(r"\d", new)):
        return old.strip() != new.strip()
    return skeleton(old) != skeleton(new)


def p_value(value: str) -> tuple[float, bool] | None:
    """(p, is_bound) of a printed p-value: '0.034', '\\ensuremath{<}0.001' (bound: p < 0.001), '$<$0.01', 'p = 0.2';
    None when the value is not one number in [0, 1]."""
    s = _DECOR.sub(lambda m: "<" if m.group(0) in ("\\textless", "\\lt") else "", value)
    s = re.sub(r"[\s{}$~]", "", s).removeprefix("p").removeprefix("=")
    m = re.fullmatch(r"(<)?(\d*\.?\d+)", s)
    if not m:
        return None
    x = float(m.group(2))
    return (x, bool(m.group(1))) if 0.0 <= x <= 1.0 else None


def is_p_macro(name: str, value: str) -> bool:
    """A p-value macro: the name ends in P (or Pval), or the value is a bound such as \\ensuremath{<}0.001."""
    pv = p_value(value)
    return pv is not None and (bool(re.search(r"(?:P|Pval|PVal|pval)$", name)) or pv[1])


def below(pv: tuple[float, bool], t: float) -> bool | None:
    """Is the p-value below threshold t?  None when a bound cannot tell (p < 0.2 against 0.10)."""
    x, bound = pv
    if bound:
        return True if x <= t else None
    return x < t


def p_crossings(name: str, old: str, new: str) -> tuple[float, ...]:
    """The thresholds of P_THRESHOLDS that a p-value macro's old and new values lie on opposite sides of."""
    if not (is_p_macro(name, old) or is_p_macro(name, new)):
        return ()
    po, pn = p_value(old), p_value(new)
    if po is None or pn is None:
        return ()
    out = []
    for t in P_THRESHOLDS:
        a, b = below(po, t), below(pn, t)
        if a is not None and b is not None and a != b:
            out.append(t)
    return tuple(out)


def is_negative(value: str) -> bool:
    """Does the value's first number carry a minus ($-$0.12, -0.12, \\ensuremath{-}0.1, \\textminus 3)?"""
    s = re.sub(r"\\(?:ensuremath|textbf|mathbf|text)(?![A-Za-z])|[\s{}$~]", "", value)
    return s.startswith(("-", "\u2212", "\\textminus", "\\minus"))


def sign_flip(old: str, new: str) -> bool:
    """A number that changed sign: both values hold a non-zero number, one negative and one not."""
    nz = re.compile(r"[1-9]")
    return bool(nz.search(old) and nz.search(new)) and is_negative(old) != is_negative(new)


def classify(file: str, name: str, old: str, new: str) -> Change:
    return Change(file, name, old, new, is_nonnumeric(old, new), p_crossings(name, old, new), sign_flip(old, new))


def git(*args: str, check: bool = True, stdin: bytes | None = None) -> bytes:
    p = subprocess.run([*GIT, *args], cwd=REPO, capture_output=True, input=stdin)
    if check and p.returncode != 0:
        raise SystemExit(f"regen: git {' '.join(args)} failed: {p.stderr.decode('utf-8', 'replace').strip()}")
    return p.stdout


def resolve_base(base: str) -> str:
    p = subprocess.run([*GIT, "rev-parse", "--verify", "--quiet", base + "^{commit}"], cwd=REPO, capture_output=True)
    if p.returncode != 0:
        raise SystemExit(f"regen: --base {base!r} is not a commit")
    return p.stdout.decode().strip()


def base_texts(base: str, rel_dir: str) -> dict[str, str]:
    """File name -> text of every .tex file directly in `rel_dir` at commit `base` (one git cat-file process)."""
    names = [
        n
        for n in git("ls-tree", "--name-only", f"{base}:{rel_dir}", check=False).decode("utf-8").splitlines()
        if n.endswith(".tex")
    ]
    if not names:
        return {}
    raw = git("cat-file", "--batch", stdin="".join(f"{base}:{rel_dir}/{n}\n" for n in names).encode("utf-8"))
    out, pos = {}, 0
    for n in names:
        nl = raw.index(b"\n", pos)
        head = raw[pos:nl].split()
        if len(head) < 3 or head[1] != b"blob":
            pos = nl + 1
            continue
        size = int(head[2])
        out[n] = raw[nl + 1 : nl + 1 + size].decode("utf-8", "replace")
        pos = nl + 1 + size + 1
    return out


def diff_macros(base: str) -> list[FileDiff]:
    """Step 2.  Every writeup/generated/*.tex at `base` against the working tree, macro by macro."""
    old_texts = base_texts(base, "writeup/generated")
    new_names = {p.name for p in GEN_DIR.glob("*.tex")}
    diffs = []
    for fname in sorted(new_names | old_texts.keys()):
        a = parse_macros(old_texts.get(fname, ""))
        b = parse_macros((GEN_DIR / fname).read_text("utf-8", "replace")) if fname in new_names else {}
        rel = f"writeup/generated/{fname}"
        d = FileDiff(rel, len(b))
        d.changed = [classify(rel, k, a[k], b[k]) for k in b if k in a and a[k] != b[k]]
        d.added = [Change(rel, k, None, b[k]) for k in b if k not in a]
        d.removed = [Change(rel, k, a[k], None) for k in a if k not in b]
        diffs.append(d)
    return diffs


# ================================================================================================== step 3: impact
_TOKEN = re.compile(r"\\([A-Za-z@]+)")
# names that begin with "if" but open no TeX conditional (no matching \fi)
NOT_CONDITIONALS = frozenset(
    (
        "iff",
        "ifthenelse",
        "ifdef",
        "ifndef",
        "ifundef",
        "ifcsdef",
        "ifcsundef",
        "ifdefstring",
        "ifstrequal",
        "ifstrempty",
        "ifblank",
        "ifbool",
        "iftoggle",
        "ifboolexpr",
        "ifnumcomp",
        "ifdimcomp",
        "ifnumequal",
        "ifpackageloaded",
        "ifclassloaded",
    )
)


def live_lines(text: str) -> Iterator[tuple[int, str, str]]:
    """(line number, live part, whole line) of every line TeX reads: the part of a line from an unescaped % is a
    comment, and everything between \\iffalse and its \\fi is skipped (conditionals nested inside are counted; an
    \\else at the block's own level makes the rest of the block live).  Lines with no live text are left out."""
    skip = 0  # > 0: inside an \iffalse block; the depth of conditionals open there
    else_open: list[int] = []  # per \iffalse whose \else branch is live: the conditionals opened since that \else
    for no, raw in enumerate(text.splitlines(), 1):
        line = strip_comment(raw)
        parts: list[str] = []
        last = 0
        prev = ""
        for m in _TOKEN.finditer(line):
            name, defining = m.group(1), prev == "newif"
            prev = name
            if defining:  # \newif\ifx defines a conditional, opens none
                continue
            opens = name.startswith("if") and name not in NOT_CONDITIONALS
            if skip:
                if opens:
                    skip += 1
                elif name == "fi":
                    skip -= 1
                    last = m.end() if skip == 0 else last
                elif name == "else" and skip == 1:
                    skip, last = 0, m.end()
                    else_open.append(0)
                continue
            if name == "iffalse":
                parts.append(line[last : m.start()])
                skip = 1
            elif else_open and opens:
                else_open[-1] += 1
            elif else_open and name == "fi":
                if else_open[-1] == 0:
                    else_open.pop()
                    parts.append(line[last : m.start()])
                    last = m.end()
                else:
                    else_open[-1] -= 1
        if not skip:
            parts.append(line[last:])
        live = "".join(parts)
        if live.strip():
            yield no, live, raw


def paper_files() -> list[Path]:
    return [WRITEUP / f"{PAPER}.tex", *sorted((WRITEUP / "rehaul").glob("*.tex"))]


def usage_index() -> dict[str, list[tuple[str, int, str]]]:
    """Control-sequence name -> [(file, line, line text)] over the live lines of the paper's sources."""
    idx: dict[str, list[tuple[str, int, str]]] = {}
    for p in paper_files():
        rel = p.relative_to(REPO).as_posix()
        for no, live, raw in live_lines(p.read_text("utf-8", "replace")):
            for name in dict.fromkeys(_TOKEN.findall(live)):
                idx.setdefault(name, []).append((rel, no, raw.strip()))
    return idx


@dataclass
class Impact:
    uses: dict[str, list[tuple[str, int, str]]]  # macro -> live uses
    stat: list[tuple[str, str, str]]  # (generated file, "+added -deleted" lines, kind: table / figure / macros / ...)
    untracked: list[str]


def file_kind(path: Path) -> str:
    """What a generated file holds: a table, a figure, macro definitions, or (deleted / other)."""
    if not path.is_file():
        return "deleted"
    if path.suffix != ".tex":
        return path.suffix.lstrip(".") or "other"
    t = path.read_text("utf-8", "replace")
    if "tabular" in t:
        return "table"
    if "includegraphics" in t:
        return "figure"
    return "macros" if parse_macros(t) else "other"


def impact(diffs: list[FileDiff], base: str) -> Impact:
    """Step 3.  The live uses of every changed or removed macro, and the generated files whose content changed
    (git diff --numstat, line endings ignored as git does under core.autocrlf)."""
    idx = usage_index()
    names = [c.name for d in diffs for c in d.changed + d.removed]
    uses = {n: idx[n] for n in dict.fromkeys(names) if n in idx}
    raw = git("diff", "--no-color", "--no-ext-diff", "--numstat", base, "--", "writeup/generated").decode("utf-8")
    stat = []
    for ln in raw.splitlines():
        parts = ln.split("\t", 2)
        if len(parts) == 3:
            add, dele, path = parts
            counts = "binary" if add == "-" else f"+{add} -{dele}"
            stat.append((path, counts, file_kind(REPO / path)))
    untracked = git("ls-files", "--others", "--exclude-standard", "--", "writeup/generated").decode().split()
    return Impact(uses, stat, untracked)


# ================================================================================================= step 4: compile
@dataclass
class Compile:
    dir: Path
    rcs: list[tuple[str, int | None]]
    pages: int | None
    errors: list[str]
    undef_refs: list[str]
    undef_cites: list[str]
    undef_cs: list[str]
    secs: float
    rerun: bool = False


def compile_paper(keep: bool, timeout: float) -> Compile:
    """Step 4.  pdflatex, bibtex, pdflatex x2 on a copy of the paper's sources in a temporary directory."""
    t0 = time.monotonic()
    tmp = Path(tempfile.mkdtemp(prefix="regen_tex_"))
    for f in (f"{PAPER}.tex", "references.bib"):
        shutil.copy2(WRITEUP / f, tmp / f)
    for d in ("rehaul", "generated", "figures"):
        shutil.copytree(WRITEUP / d, tmp / d, ignore=shutil.ignore_patterns("*.aux", "*.log", "__pycache__"))
    tex = ["pdflatex", "-interaction=nonstopmode", "-max-print-line=10000", f"{PAPER}.tex"]
    rcs: list[tuple[str, int | None]] = []
    for step, cmd in (("pdflatex", tex), ("bibtex", ["bibtex", PAPER]), ("pdflatex", tex), ("pdflatex", tex)):
        try:
            p = subprocess.run(cmd, cwd=tmp, capture_output=True, timeout=timeout, stdin=subprocess.DEVNULL)
            rcs.append((step, p.returncode))
        except (subprocess.TimeoutExpired, OSError):
            rcs.append((step, None))
            break
    log_path = tmp / f"{PAPER}.log"
    log = log_path.read_text("utf-8", "replace") if log_path.is_file() else ""
    blg_path = tmp / f"{PAPER}.blg"
    blg = blg_path.read_text("utf-8", "replace") if blg_path.is_file() else ""
    res = parse_log(log, blg)
    res.dir, res.rcs, res.secs = tmp, rcs, time.monotonic() - t0
    if not keep:
        shutil.rmtree(tmp, ignore_errors=True)
    return res


def parse_log(log: str, blg: str = "") -> Compile:
    """The errors (lines starting with !, with their l.<n> context), undefined references / citations / control
    sequences and the page count of a pdflatex log (and the bibtex .blg's missing entries)."""
    lines = log.splitlines()
    errors, undef_cs = [], []
    for i, ln in enumerate(lines):
        if not ln.startswith("!"):
            continue
        ctx = next((x.strip() for x in lines[i + 1 : i + 12] if re.match(r"l\.\d+", x)), "")
        errors.append(f"{ln[1:].strip()}" + (f"  [{ctx[:160]}]" if ctx else ""))
        if ln.startswith("! Undefined control sequence"):
            for x in lines[i + 1 : i + 12]:
                m = re.search(r"(\\[A-Za-z@]+)\s*$", x)
                if m:
                    undef_cs.append(m.group(1) + (f"  ({ctx[:120]})" if ctx else ""))
                    break
    refs = re.findall(r"Reference `([^']+)' on page \d+ undefined", log)
    cites = re.findall(r"Citation `([^']+)' on page \d+ undefined", log)
    cites += re.findall(r"I didn't find a database entry for \"([^\"]+)\"", blg)
    m = re.search(r"Output written on .*?\((\d+) pages?", log)
    return Compile(
        dir=Path(),
        rcs=[],
        pages=int(m.group(1)) if m else None,
        errors=errors,
        undef_refs=sorted(set(refs)),
        undef_cites=sorted(set(cites)),
        undef_cs=list(dict.fromkeys(undef_cs)),
        secs=0.0,
        rerun="Label(s) may have changed" in log,
    )


# ==================================================================================================== the report
def rel(p: Path) -> str:
    """p relative to the repository when it is inside it, else as given."""
    try:
        return p.resolve().relative_to(REPO).as_posix()
    except ValueError:
        return str(p)


def short(v: str | None, n: int = 90) -> str:
    if v is None:
        return "(none)"
    v = " ".join(v.split())
    return v if len(v) <= n else v[: n - 3] + "..."


def md(v: str) -> str:
    """A value inside a Markdown table cell or code span."""
    return v.replace("|", "\\|").replace("`", "'")


def write_report(
    argv: list[str],
    base: str,
    base_sha: str,
    runs: list[Run] | None,
    skipped_down: dict[str, list[str]],
    diffs: list[FileDiff],
    imp: Impact,
    comp: Compile | None,
    wall: float,
) -> str:
    changed = [c for d in diffs for c in d.changed]
    head = git("rev-parse", "--short", "HEAD").decode().strip()
    L = [
        "# Regeneration report",
        "",
        f"{datetime.now():%Y-%m-%d %H:%M}; `python writeup/tools/regen.py {' '.join(argv)}`; base `{base}` "
        f"({base_sha[:10]}), HEAD {head}; wall time {wall:.0f} s.  Written by writeup/tools/regen.py.",
        "",
        "## 1. Generators",
        "",
    ]
    if runs is None:
        L.append("Not run (--no-run): the diff below is of the files as they are.")
    else:
        L += ["| generator | arguments | exit | seconds | log | note |", "|---|---|---|---|---|---|"]
        for r in runs:
            state = "ok" if r.ok else ("not run" if r.note.startswith("not run") else f"**{r.rc}**")
            args = "`" + " ".join(r.args) + "`" if r.args else "(none)"
            L.append(f"| {r.gen} | {args} | {state} | {r.secs:.0f} | {rel(r.log)} | {md(r.note)} |")
        for r in runs:
            if not r.ok and not r.note.startswith("not run"):
                what = " ".join((r.gen, *r.args))
                L += ["", f"**{what} failed** ({r.note or f'exit {r.rc}'}):", "", "```"]
                L += [r.err_tail or "(no output)", "```"]
        if skipped_down:
            L += ["", "Not rerun although they read the output of a generator that ran (their files may be stale):"]
            L += [f"- {n} (reads {', '.join(d)})" for n, d in sorted(skipped_down.items())]
        unmapped = sorted({r.gen for r in runs if r.gen not in GENERATORS})
        if unmapped:
            L += ["", f"Not in the GENERATORS map of regen.py (one run, no ordering): {', '.join(unmapped)}."]
    L += ["", f"## 2. Macros against {base}", ""]
    moved = [d for d in diffs if d.changed or d.added or d.removed]
    L.append(
        f"{sum(d.n_new for d in diffs)} macros in {len(diffs)} files; {len(changed)} changed, "
        f"{sum(len(d.added) for d in diffs)} added, {sum(len(d.removed) for d in diffs)} removed, in {len(moved)} files."
    )
    if moved:
        L += ["", "| file | macros | changed | added | removed |", "|---|---|---|---|---|"]
        L += [f"| {d.file} | {d.n_new} | {len(d.changed)} | {len(d.added)} | {len(d.removed)} |" for d in moved]
    for title, sel in (
        ("Non-numeric changes (words, verdicts, yes / no)", [c for c in changed if c.nonnumeric]),
        ("p-values crossing " + " or ".join(f"{t:g}" for t in P_THRESHOLDS), [c for c in changed if c.p_cross]),
        ("Sign flips", [c for c in changed if c.sign_flip]),
    ):
        L += ["", f"### {title}: {len(sel)}"] + ([""] if sel else [])
        for c in sel:
            extra = f" (crosses {', '.join(f'{t:g}' for t in c.p_cross)})" if c.p_cross else ""
            L.append(f"- `\\{c.name}` ({Path(c.file).name}): `{md(short(c.old))}` -> `{md(short(c.new))}`{extra}")
    L += ["", f"### All changed macros: {len(changed)}"] + ([""] if changed else [])
    L += [f"- `\\{c.name}` ({Path(c.file).name}): `{md(short(c.old))}` -> `{md(short(c.new))}`" for c in changed]
    for title, key in (("Added", "added"), ("Removed", "removed")):
        sel = [c for d in diffs for c in getattr(d, key)]
        if sel:
            L += ["", f"### {title}: {len(sel)}", ""]
            L += [
                f"- `\\{c.name}` ({Path(c.file).name}): `{md(short(c.new if key == 'added' else c.old))}`" for c in sel
            ]
    L += ["", "## 3. Impact", ""]
    removed = {c.name for d in diffs for c in d.removed}
    L.append(
        f"{len(imp.uses)} of {len(changed) + len(removed)} changed or removed macros are used in live lines of "
        f"{PAPER}.tex and rehaul/*.tex ({sum(len(v) for v in imp.uses.values())} lines)."
    )
    by_name = {c.name: c for d in diffs for c in d.changed + d.removed}
    for name, hits in imp.uses.items():
        c = by_name[name]
        what = "REMOVED" if c.new is None else f"`{md(short(c.old, 50))}` -> `{md(short(c.new, 50))}`"
        L += ["", f"`\\{name}` {what}", ""]
        L += [f"- {f}:{n}: {md(short(t, 220))}" for f, n, t in hits]
    unused = [n for n in by_name if n not in imp.uses]
    if unused:
        L += ["", f"Changed or removed, used in no live line of the paper: {', '.join(unused)}."]
    L += ["", f"### Generated files whose content changed (git diff --numstat {base}): {len(imp.stat)}"]
    if imp.stat:
        L += ["", "| file | lines | holds |", "|---|---|---|"] + [f"| {p} | {c} | {k} |" for p, c, k in imp.stat]
    if imp.untracked:
        L += ["", "New, untracked: " + ", ".join(imp.untracked)]
    L += ["", "## 4. Compile", ""]
    if comp is None:
        L.append("Not compiled (no --compile).")
    else:
        L.append(
            f"{comp.pages} pages; {len(comp.errors)} errors, {len(comp.undef_refs)} undefined references, "
            f"{len(comp.undef_cites)} undefined citations, {len(comp.undef_cs)} undefined control sequences; "
            f"{comp.secs:.0f} s; "
            + ", ".join(f"{s} exit {rc}" for s, rc in comp.rcs)
            + "."
            + (
                " The last pass still says 'Label(s) may have changed' (as with the usual four passes)."
                if comp.rerun
                else ""
            )
        )
        for title, items in (
            ("Errors", comp.errors),
            ("Undefined references", comp.undef_refs),
            ("Undefined citations", comp.undef_cites),
            ("Undefined control sequences", comp.undef_cs),
        ):
            if items:
                L += ["", f"### {title}", ""] + [f"- {md(x)}" for x in items]
    text = "\n".join(L) + "\n"
    REPORT.write_text(text, encoding="utf-8", newline="\n")
    return text


def summary(
    runs: list[Run] | None, diffs: list[FileDiff], imp: Impact, comp: Compile | None, base: str, wall: float
) -> list[str]:
    changed = [c for d in diffs for c in d.changed]
    out = []
    if runs is not None:
        skipped = sorted({r.gen for r in runs if r.note.startswith("not run")})
        bad = sorted({r.gen for r in runs if not r.ok} - set(skipped))
        n_gen = len({r.gen for r in runs})
        out.append(
            f"generators: {n_gen} ({len(runs)} runs), {n_gen - len(bad) - len(skipped)} ok"
            + (f", {len(bad)} FAILED: {', '.join(bad)}" if bad else "")
            + (f", {len(skipped)} NOT RUN (interpreter): {', '.join(skipped)}" if skipped else "")
        )
        out += [f"  {r.gen}: {r.note}" for r in runs if r.note.startswith("not run") and r.args == spec(r.gen).modes[0]]
    out.append(
        f"macros vs {base}: {len(changed)} changed, {sum(len(d.added) for d in diffs)} added, "
        f"{sum(len(d.removed) for d in diffs)} removed in {sum(1 for d in diffs if d.changed or d.added or d.removed)} "
        f"of {len(diffs)} files; {sum(c.nonnumeric for c in changed)} non-numeric, "
        f"{sum(bool(c.p_cross) for c in changed)} p-value crossings, {sum(c.sign_flip for c in changed)} sign flips"
    )
    out.append(
        f"impact: {len(imp.uses)} macros in {sum(len(v) for v in imp.uses.values())} live lines; "
        f"{len(imp.stat)} generated files changed ({sum(k == 'table' for _, _, k in imp.stat)} tables)"
        + (f", {len(imp.untracked)} new untracked" if imp.untracked else "")
    )
    if comp is not None:
        out.append(
            f"compile: {comp.pages} pages, {len(comp.errors)} errors, {len(comp.undef_refs)} undefined references, "
            f"{len(comp.undef_cites)} undefined citations, {len(comp.undef_cs)} undefined control sequences"
            + (f" (kept: {comp.dir})" if comp.dir.exists() else "")
        )
    out.append(f"report: {rel(REPORT)} (wall {wall:.0f} s)")
    return out


# ====================================================================================================== main
def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--only", help="comma-separated generators to run (ladder,channels or make_rehaul_ladder.py,...)")
    ap.add_argument("--no-run", action="store_true", help="skip step 1: diff + impact (+ compile) of the current tree")
    ap.add_argument("--compile", action="store_true", help="step 4: build the paper in a temporary copy")
    ap.add_argument("--base", default="HEAD", help="the revision to diff the macros against (default HEAD)")
    ap.add_argument("--jobs", type=int, default=4, help="generators at once (default 4; each loads its own frames)")
    ap.add_argument(
        "--python",
        action="append",
        metavar="[NAME=]PATH",
        help="interpreter of the generators (default: $REGEN_PYTHON, else this one); NAME=PATH for one generator; "
        "repeatable",
    )
    ap.add_argument("--timeout", type=float, default=3600, help="seconds per generator run / per TeX pass")
    ap.add_argument("--keep-build", action="store_true", help="keep the temporary TeX directory")
    ap.add_argument("--list", action="store_true", help="print the generators, modes and run order, and exit")
    a = ap.parse_args(argv)
    on_disk = discover()
    try:
        names = resolve_names(a.only, on_disk)
        base_sha = resolve_base(a.base)
    except SystemExit as e:
        print(e, file=sys.stderr)
        return 2
    if a.list:
        for i, wave in enumerate(run_order(names), 1):
            for n in wave:
                g = spec(n)
                modes = " | ".join(" ".join(m) or "(no arguments)" for m in g.modes)
                print(
                    f"wave {i}  {n:18s} {modes}"
                    + (f"   after {', '.join(g.after)}" if g.after else "")
                    + (f"   needs {', '.join(g.needs)}" if g.needs else "")
                )
        return 0
    t0 = time.monotonic()
    runs = None
    if not a.no_run:
        waves = run_order(names)
        print(f"regen: {len(names)} generators, {len(waves)} waves, {a.jobs} at a time; logs in {LOG_DIR}")
        runs = run_generators(names, a.jobs, parse_pythons(a.python), a.timeout)
    diffs = diff_macros(base_sha)
    imp = impact(diffs, base_sha)
    comp = None
    if a.compile:
        print("regen: compiling a copy of the paper (pdflatex, bibtex, pdflatex x2) ...", flush=True)
        comp = compile_paper(a.keep_build, a.timeout)
    wall = time.monotonic() - t0
    skipped = downstream_skipped(names, on_disk) if runs is not None else {}
    write_report(argv, a.base, base_sha, runs, skipped, diffs, imp, comp, wall)
    for line in summary(runs, diffs, imp, comp, a.base, wall):
        print(line)
    for n, d in sorted(skipped.items()):
        print(f"  note: {n} not rerun, reads {', '.join(d)}")
    failed = runs is not None and not all(r.ok for r in runs)
    return 1 if failed or (comp is not None and (comp.errors or comp.pages is None)) else 0


if __name__ == "__main__":
    sys.exit(main())
