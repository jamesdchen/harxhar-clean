"""Pack the Overleaf drop zip from what the paper actually reads (2026-10-15).

pack-overleaf.ps1 globbed main.tex, sections/, generated/ and figures/, which misses the
other generated_* folders and the figures main.tex takes from ../results/ (Overleaf cannot
read outside the project). This packer compiles main.tex once with -recorder in a scratch
directory, takes every input file under the repository from the .fls, adds references.bib,
and writes them into the zip with forward-slash paths. A file outside writeup/ (../results/x)
is stored at results/x and every packed .tex has "../results/" rewritten to "results/"; the
repository's files are not changed. Then the zip is unpacked into a second scratch directory
and compiled from scratch (pdflatex, bibtex, pdflatex x3) to prove it is complete.

    python writeup/pack_overleaf.py            -> writeup/overleaf-paper-<date>.zip

Stdlib only; needs pdflatex and bibtex (MiKTeX) on PATH.
"""

from __future__ import annotations

import datetime
import re
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

WRITEUP = Path(__file__).resolve().parent
REPO = WRITEUP.parent


def run(cmd: list[str], cwd: Path) -> int:
    return subprocess.run(
        cmd, cwd=cwd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    ).returncode


def recorded_inputs(build: Path) -> list[Path]:
    """Every file under the repository that pdflatex read (from main.fls)."""
    run(
        [
            "pdflatex",
            "-recorder",
            "-interaction=nonstopmode",
            f"-output-directory={build}",
            "main.tex",
        ],
        WRITEUP,
    )
    out: set[Path] = set()
    for line in (
        (build / "main.fls").read_text(encoding="utf-8", errors="replace").splitlines()
    ):
        if not line.startswith("INPUT "):
            continue
        p = Path(line[6:].strip())
        p = (WRITEUP / p).resolve() if not p.is_absolute() else p.resolve()
        try:
            p.relative_to(REPO)
        except ValueError:
            continue  # TeX distribution files
        if (
            p.suffix in (".aux", ".out", ".toc", ".bbl", ".blg", ".log")
            or not p.is_file()
        ):
            continue
        out.add(p)
    out.add(WRITEUP / "references.bib")
    return sorted(out)


def arcname(p: Path) -> str:
    try:
        return p.relative_to(WRITEUP).as_posix()
    except ValueError:
        return p.relative_to(REPO).as_posix()  # ../results/x -> results/x


def check(zpath: Path) -> tuple[int, int, str]:
    """Unpack into a clean directory and compile: (exit code, pages, problems)."""
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        with zipfile.ZipFile(zpath) as z:
            z.extractall(d)
        run(["pdflatex", "-interaction=nonstopmode", "main.tex"], d)
        run(["bibtex", "main"], d)
        code = 0
        for _ in range(3):
            code = run(["pdflatex", "-interaction=nonstopmode", "main.tex"], d)
        log = (d / "main.log").read_text(encoding="utf-8", errors="replace")
        m = re.search(r"Output written on .*?\((\d+) pages", log.replace("\n", ""))
        probs = [
            f"errors {len(re.findall(r'^! ', log, flags=re.M))}",
            f"missing files {log.count('not found')}",
            f"undefined refs {len(re.findall(r'Reference .* undefined', log))}",
            f"undefined cites {len(re.findall(r'Citation .* undefined', log))}",
        ]
        return code, int(m.group(1)) if m else 0, ", ".join(probs)


def main() -> None:
    stamp = datetime.date.today().isoformat()
    dest = WRITEUP / f"overleaf-paper-{stamp}.zip"
    with tempfile.TemporaryDirectory() as td:
        files = recorded_inputs(Path(td))
    with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED) as z:
        for p in files:
            name = arcname(p)
            if p.suffix == ".tex":
                text = p.read_text(encoding="utf-8").replace("../results/", "results/")
                z.writestr(name, text)
            else:
                z.write(p, name)
    outside = [arcname(p) for p in files if not str(p).startswith(str(WRITEUP))]
    print(
        f"{dest}  {len(files)} files ({len(outside)} from outside writeup/: {outside})"
    )
    code, pages, probs = check(dest)
    print(f"clean compile of the zip: exit {code}, {pages} pages, {probs}")
    if code != 0 or pages == 0:
        sys.exit("the zip does not compile on its own")


if __name__ == "__main__":
    main()
