"""Build (regenerate) the quant pack manifest — the "sweep docs at pack build" step.

The user's standing rule (memory 2026-07-07): the quant pack SWEEPS the lab's
writeup docs at build time. domain-packs.md defines no dedicated sweep mechanism,
so the pack-side realization is the manifest integrity set: this build step
gathers (a) the pack's own declaration/template/check files and (b) the swept
writeup docs (``sweep.json`` globs, referenced IN PLACE via ``../../writeup/...``
relpaths), recomputes every file's raw-bytes SHA-256, and writes ``manifest.json``.

Binding the resulting manifest (``hpc-agent pack-bind``) SEALS which lab docs the
domain standards were drafted from: edit a swept writeup doc and the on-disk sha
no longer matches, so the next bind/gate reads drift and revokes every clearance
signed under the old standards (domain-packs.md, "Re-bind = drift"). Re-running
this build re-sweeps and moves the shas — the honest "edit standards -> rebuild
-> re-bind" flow, at pack-build granularity.

Run from anywhere:  python packs/quant/build_quant_pack.py [--check]
  (no args)  rewrite packs/quant/manifest.json from the on-disk shas
  --check    fail (exit 1) if the manifest is stale, writing nothing (CI use)

sweep.json is the build RECIPE, not sealed content; it is deliberately absent
from the integrity set (a Makefile is not one of its own targets).
"""

from __future__ import annotations

import glob
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

PACK_ROOT = Path(__file__).resolve().parent
SWEEP_CONFIG = PACK_ROOT / "sweep.json"
MANIFEST = PACK_ROOT / "manifest.json"


def _raw_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rel_from_pack_root(abspath: str) -> str:
    """Pack-root-relative POSIX relpath (keeps ``../../writeup/...`` for swept docs)."""
    return Path(os.path.relpath(abspath, PACK_ROOT)).as_posix()


def _sweep_docs(patterns: list[str]) -> list[str]:
    """Resolve each ``../../writeup/*.md`` glob against the pack root -> sorted relpaths."""
    hits: set[str] = set()
    for pat in patterns:
        for abspath in glob.glob(str(PACK_ROOT / pat)):
            if Path(abspath).is_file():
                hits.add(_rel_from_pack_root(abspath))
    return sorted(hits)


def build_manifest_dict() -> dict[str, Any]:
    cfg = json.loads(SWEEP_CONFIG.read_text(encoding="utf-8"))
    relpaths = sorted({*cfg["pack_files"], *_sweep_docs(cfg["sweep"])})
    files = [{"path": rel, "sha256": _raw_sha(PACK_ROOT / rel)} for rel in relpaths]
    return {
        "name": cfg["name"],
        "version": cfg["version"],
        "files": files,
        "seams": cfg["seams"],
        "fills_slots": cfg["fills_slots"],
    }


def _serialize(manifest: dict[str, Any]) -> str:
    return json.dumps(manifest, indent=2, sort_keys=True) + "\n"


def main(argv: list[str]) -> int:
    manifest = build_manifest_dict()
    text = _serialize(manifest)
    if "--check" in argv:
        current = MANIFEST.read_text(encoding="utf-8") if MANIFEST.exists() else ""
        if current != text:
            print("quant pack manifest is STALE — run: python packs/quant/build_quant_pack.py")
            return 1
        print("quant pack manifest is current.")
        return 0
    MANIFEST.write_text(text, encoding="utf-8")
    swept = [f["path"] for f in manifest["files"] if f["path"].startswith("../")]
    print(f"wrote {MANIFEST}")
    print(f"  files sealed: {len(manifest['files'])}  (swept writeup docs: {len(swept)})")
    for s in swept:
        print(f"    swept: {s}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
