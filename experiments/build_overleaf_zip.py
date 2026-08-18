"""Idempotent packaging script for the paper's Overleaf upload.

Collects, from ``writeup/``:

- ``main.tex``
- the ``.bib`` file ``main.tex`` actually uses (resolved from its
  ``\\bibliography{...}`` command)
- ``pending_macros.tex``
- every ``*.tex`` under ``writeup/sections/``
- every ``*.tex`` under ``writeup/generated/``
- every image referenced via ``\\includegraphics{...}`` or the second
  argument of ``\\treefigorpending{...}{...}`` in that collected tex
  tree, resolved relative to ``writeup/``

into ``writeup/overleaf_YYYY-MM-DD.zip``, preserving the ``writeup/``-relative
directory layout so Overleaf's ``\\input``/``\\includegraphics`` paths resolve
unchanged. Build artifacts (aux/log/bbl/blg/out/pdf/synctex) and any
COLD_SESSION/MEETING_PREP markdown are never included.

Re-running with unchanged sources overwrites the same zip deterministically
(sorted entries, source mtimes) -- the script is idempotent.

Usage::

    python experiments/build_overleaf_zip.py [--out PATH] [--no-verify]

With verification (the default), the zip is extracted to a scratch temp
dir and compiled standalone with pdflatex (+bibtex if references.bib is
present) to confirm it is self-contained.
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
WRITEUP_DIR = REPO_ROOT / "writeup"
SECTIONS_DIR = WRITEUP_DIR / "sections"
GENERATED_DIR = WRITEUP_DIR / "generated"
MAIN_TEX = WRITEUP_DIR / "main.tex"
PENDING_MACROS = WRITEUP_DIR / "pending_macros.tex"

INCLUDEGRAPHICS_RE = re.compile(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]*)\}")
TREEFIGORPENDING_RE = re.compile(r"\\treefigorpending\{[^}]*\}\{([^}]*)\}")
BIBLIOGRAPHY_RE = re.compile(r"\\bibliography\{([^}]*)\}")

IMAGE_EXTENSION_FALLBACKS = (".pdf", ".png", ".jpg", ".jpeg", ".eps")

EXCLUDED_SUFFIXES = {".aux", ".log", ".bbl", ".blg", ".out", ".pdf", ".synctex"}


def is_excluded(path: Path) -> bool:
    """Defense-in-depth filter: never let a build artifact or a
    COLD_SESSION/MEETING_PREP markdown file into the package, even if some
    future glob would otherwise pick one up."""
    name = path.name
    if name.endswith(".synctex.gz"):
        return True
    if path.suffix.lower() in EXCLUDED_SUFFIXES:
        return True
    if name.startswith("COLD_SESSION") or name.startswith("MEETING_PREP"):
        return True
    return False


def find_bib_path(main_tex: Path) -> Path | None:
    """Resolve the .bib file main.tex's \\bibliography{...} points at."""
    text = main_tex.read_text(encoding="utf-8")
    m = BIBLIOGRAPHY_RE.search(text)
    if not m:
        return None
    # \bibliography can list multiple comma-separated names; take the first.
    name = m.group(1).split(",")[0].strip()
    if not name.endswith(".bib"):
        name += ".bib"
    return WRITEUP_DIR / name


def collect_core_files() -> list[Path]:
    if not MAIN_TEX.is_file():
        raise SystemExit(f"missing required file: {MAIN_TEX}")
    files = [MAIN_TEX]

    bib_path = find_bib_path(MAIN_TEX)
    if bib_path is None:
        raise SystemExit(f"no \\bibliography{{...}} command found in {MAIN_TEX}")
    if not bib_path.is_file():
        raise SystemExit(f"main.tex references bib file that does not exist: {bib_path}")
    files.append(bib_path)

    if not PENDING_MACROS.is_file():
        raise SystemExit(f"missing required file: {PENDING_MACROS}")
    files.append(PENDING_MACROS)

    if SECTIONS_DIR.is_dir():
        files.extend(sorted(SECTIONS_DIR.rglob("*.tex")))
    if GENERATED_DIR.is_dir():
        files.extend(sorted(GENERATED_DIR.rglob("*.tex")))

    return [f for f in files if not is_excluded(f)]


def resolve_image(ref: str) -> Path | None:
    """Resolve an \\includegraphics / \\treefigorpending path relative to
    writeup/. Returns the existing Path, or None if nothing matched
    (literal path, or one of the common LaTeX-omitted-extension guesses)."""
    ref = ref.strip()
    candidate = (WRITEUP_DIR / ref).resolve()
    if candidate.is_file():
        return candidate
    if not Path(ref).suffix:
        for ext in IMAGE_EXTENSION_FALLBACKS:
            alt = (WRITEUP_DIR / (ref + ext)).resolve()
            if alt.is_file():
                return alt
    return None


def collect_images(tex_files: list[Path]) -> tuple[list[Path], list[str]]:
    """Scan the collected tex tree for image references. Returns
    (found_paths, missing_refs)."""
    refs: set[str] = set()
    for tex_file in tex_files:
        if tex_file.suffix.lower() != ".tex":
            continue
        text = tex_file.read_text(encoding="utf-8", errors="replace")
        refs.update(INCLUDEGRAPHICS_RE.findall(text))
        refs.update(TREEFIGORPENDING_RE.findall(text))

    # Drop macro-parameter placeholders (e.g. "#2") picked up from a
    # \newcommand/\providecommand *definition* body -- such as main.tex's
    # own \treefigorpending macro, whose body contains a literal
    # \includegraphics[width=#1]{#2} -- these are not real image refs.
    refs = {r for r in refs if "#" not in r}

    found: list[Path] = []
    missing: list[str] = []
    for ref in sorted(refs):
        resolved = resolve_image(ref)
        if resolved is not None and not is_excluded(resolved):
            found.append(resolved)
        elif resolved is None:
            missing.append(ref)
    return sorted(set(found)), missing


def arcname_for(path: Path) -> str:
    rel = path.resolve().relative_to(WRITEUP_DIR.resolve())
    return rel.as_posix()


def build_zip(out_path: Path) -> tuple[list[Path], list[str]]:
    core_files = collect_core_files()
    all_tex_for_scan = [f for f in core_files if f.suffix.lower() == ".tex"]
    images, missing_images = collect_images(all_tex_for_scan)

    all_files = sorted(set(core_files) | set(images), key=arcname_for)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in all_files:
            zf.write(f, arcname=arcname_for(f))

    return all_files, missing_images


def print_manifest(zip_path: Path, files: list[Path], missing_images: list[str]) -> None:
    total_bytes = sum(f.stat().st_size for f in files)
    top_level = sorted({arcname_for(f).split("/", 1)[0] for f in files})

    print("=== manifest ===")
    print(f"zip path       : {zip_path}")
    print(f"zip size       : {zip_path.stat().st_size} bytes")
    print(f"file count     : {len(files)}")
    print(f"total src bytes: {total_bytes}")
    print("top-level entries:")
    for entry in top_level:
        print(f"  {entry}")
    if missing_images:
        print(f"missing images ({len(missing_images)}) -- referenced but not found on disk, NOT fatal:")
        for m in missing_images:
            print(f"  {m}")
    else:
        print("missing images : none")
    print("files:")
    for f in files:
        print(f"  {arcname_for(f)}")


def find_scratch_base() -> Path | None:
    """Locate this session's Claude scratchpad dir
    (%LOCALAPPDATA%/Temp/claude/<mangled-repo-path>/<session>/scratchpad),
    if one exists and is writable. Returns None otherwise so the caller
    falls back to tempfile.mkdtemp()."""
    local_appdata = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
    mangled = re.sub(r"[:\\/ ]+", "-", str(REPO_ROOT))
    claude_dir = Path(local_appdata) / "Temp" / "claude" / mangled
    if not claude_dir.is_dir():
        return None
    candidates = sorted(
        (p for p in claude_dir.glob("*/scratchpad") if p.is_dir()),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def make_verify_tempdir() -> Path:
    base = find_scratch_base()
    if base is not None:
        try:
            d = Path(tempfile.mkdtemp(prefix="overleaf_verify_", dir=str(base)))
            return d
        except OSError:
            pass
    return Path(tempfile.mkdtemp(prefix="overleaf_verify_"))


def run_cmd(cmd: list[str], cwd: Path) -> int:
    print(f"  $ {' '.join(cmd)}")
    proc = subprocess.run(
        cmd,
        cwd=str(cwd),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=300,
    )
    if proc.returncode != 0:
        tail = "\n".join(proc.stdout.splitlines()[-20:])
        print(f"  (exit {proc.returncode}) last 20 lines of stdout:\n{tail}")
    return proc.returncode


def verify_zip(zip_path: Path) -> bool:
    """Extract the zip standalone and compile it, verifying it is
    self-contained. Returns True iff the compile looks clean (no pdflatex
    error lines, no bibtex failure, no undefined refs/citations)."""
    extract_dir = make_verify_tempdir()
    print(f"=== verify: extracting to {extract_dir} ===")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(extract_dir)

    has_bib = (extract_dir / "references.bib").is_file()

    exit_codes: list[tuple[str, int]] = []
    if shutil.which("pdflatex") is None:
        print("pdflatex not found on PATH -- cannot verify")
        return False

    pdflatex_cmd = ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "main.tex"]
    exit_codes.append(("pdflatex#1", run_cmd(pdflatex_cmd, extract_dir)))
    if has_bib:
        if shutil.which("bibtex") is None:
            print("bibtex not found on PATH -- cannot verify bibliography step")
            return False
        exit_codes.append(("bibtex", run_cmd(["bibtex", "main"], extract_dir)))
        exit_codes.append(("pdflatex#2", run_cmd(pdflatex_cmd, extract_dir)))
        exit_codes.append(("pdflatex#3", run_cmd(pdflatex_cmd, extract_dir)))
    else:
        exit_codes.append(("pdflatex#2", run_cmd(pdflatex_cmd, extract_dir)))

    log_path = extract_dir / "main.log"
    error_lines = 0
    undefined_refs = 0
    undefined_cites = 0
    page_count: int | None = None
    if log_path.is_file():
        log_text = log_path.read_text(encoding="utf-8", errors="replace")
        error_lines = sum(1 for line in log_text.splitlines() if line.startswith("!"))
        undefined_refs = len(re.findall(r"LaTeX Warning: Reference .*? undefined", log_text))
        undefined_cites = len(re.findall(r"LaTeX Warning: Citation .*? undefined", log_text))
        m = re.search(r"Output written on main\.pdf \((\d+) pages?", log_text)
        if m:
            page_count = int(m.group(1))
    else:
        print(f"no main.log produced at {log_path}")

    print("=== verify: results ===")
    for name, code in exit_codes:
        print(f"  {name} exit code   : {code}")
    print(f"  '!' lines in log     : {error_lines}")
    print(f"  page count           : {page_count if page_count is not None else 'NOT FOUND'}")
    print(f"  undefined references : {undefined_refs}")
    print(f"  undefined citations  : {undefined_cites}")
    print(f"  extracted at         : {extract_dir}")

    ok = (
        all(code == 0 for _, code in exit_codes)
        and error_lines == 0
        and page_count is not None
        and undefined_refs == 0
        and undefined_cites == 0
    )
    print(f"  VERIFY {'PASSED' if ok else 'FAILED'}")
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Override the output zip path (default: writeup/overleaf_YYYY-MM-DD.zip)",
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip the standalone-compile verification step",
    )
    args = parser.parse_args()

    today = dt.date.today().isoformat()
    out_path = args.out if args.out is not None else WRITEUP_DIR / f"overleaf_{today}.zip"
    out_path = out_path.resolve()

    files, missing_images = build_zip(out_path)
    print_manifest(out_path, files, missing_images)

    if args.no_verify:
        print("--no-verify: skipping compile verification")
        return 0

    ok = verify_zip(out_path)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
