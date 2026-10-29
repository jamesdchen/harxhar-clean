"""Revert an edit made in the Overleaf editor to a review layer (beats.tex, beats_top.tex) when every change lies inside
the layer's generated header -- the leading block of %% lines of its last flatten (2026-10-28, author: stray keystrokes
there kept blocking the review-layer push).  Any other edit is left for the author.

Usage: layer_header_fix.py OVERLEAF_DIR SYNC_DIR
Exit 0: nothing to do, or header-only edits reverted (each printed; the caller commits them); exit 2: some layer has an
edit outside its header (printed; left alone)."""

from __future__ import annotations

import difflib
import json
import sys
from pathlib import Path

LAYERS = {"beats.tex": "map_beats.json", "beats_top.tex": "map_beats_top.json"}


def main() -> int:
    out_dir, sync = Path(sys.argv[1]), Path(sys.argv[2])
    status = 0
    for flat, map_name in LAYERS.items():
        f = out_dir / flat
        if not f.is_file() or not (sync / map_name).is_file():
            continue
        last = json.loads((sync / map_name).read_text(encoding="utf-8"))["flat_text"].replace("\r\n", "\n")
        cur = f.read_text(encoding="utf-8").lstrip("﻿").replace("\r\n", "\n")
        if cur.rstrip("\n") == last.rstrip("\n"):
            continue
        old, new = last.split("\n"), cur.split("\n")
        head = 0
        while head < len(old) and old[head].startswith("%%"):
            head += 1
        ops = [op for op in difflib.SequenceMatcher(a=old, b=new, autojunk=False).get_opcodes() if op[0] != "equal"]
        if head and all(i1 < head and i2 <= head for _, i1, i2, _, _ in ops):
            f.write_text(last if last.endswith("\n") else last + "\n", encoding="utf-8", newline="\n")
            print(f"layer_header_fix: {flat}: edit inside the generated header reverted ({len(ops)} change(s))")
        else:
            print(f"layer_header_fix: {flat}: edit outside the generated header; left for the author")
            status = 2
    return status


if __name__ == "__main__":
    sys.exit(main())
