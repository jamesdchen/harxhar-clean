"""Keep executed outputs when a writer regenerates an ipynb."""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf


def _src(cell) -> str:
    s = cell.source
    if isinstance(s, list):
        s = "".join(s)
    return s


def carry_outputs(new_nb, dest: Path) -> int:
    """Copy outputs from dest onto new_nb code cells with identical source.

    Markdown-only regenerations then keep plots and prints. Returns how
    many code cells kept their outputs.
    """
    dest = Path(dest)
    if not dest.exists():
        return 0
    old = nbf.read(dest, as_version=4)
    kept = 0
    for a, b in zip(old.cells, new_nb.cells):
        if a.cell_type != "code" or b.cell_type != "code":
            continue
        if _src(a) != _src(b):
            continue
        outs = a.get("outputs")
        if not outs:
            continue
        b["outputs"] = outs
        if "execution_count" in a:
            b["execution_count"] = a["execution_count"]
        kept += 1
    return kept


def code_cells_complete(nb) -> bool:
    """True if every nonempty code cell already has outputs."""
    for c in nb.cells:
        if c.cell_type != "code":
            continue
        if not _src(c).strip():
            continue
        if not c.get("outputs"):
            return False
    return True
