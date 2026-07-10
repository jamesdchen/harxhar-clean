"""Quant pack CHECK script — runs caller-side, emits a pack receipt.

The domain check runs ENTIRELY outside core (domain-packs.md DP2 — core never
imports or executes a pack file), then records its mechanical verdict as a
sha-bound CODE receipt via ``hpc-agent pack-record-receipt``. The pack's own CI
(or the experiment env) runs this; core only ever weighs the resulting receipt,
which reads stale the instant any checked byte drifts.

The check here is STRUCTURAL, not a research judgment: the S4 audit template must
carry every expected ``hpc-audit-section`` slug as an order-preserving presence.
The slug list is the SIGNED run10 5-slug inventory (data-selection ->
target-construction -> feature-construction -> baseline -> metrics). No research
content is asserted; the verdict is a mechanical boolean the receipt records.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

_PACK = "quant"
_SLOT = "rv-audit"
_TEMPLATE_REL = "packs/quant/templates/quant_audit.py"

# The signed run10 5-slug inventory (order-preserving). Post-signature swap to the
# 12-slug specs/audit_template_rv.py inventory happens together with the seat swap.
_EXPECTED_SECTIONS = (
    "data-selection",
    "target-construction",
    "feature-construction",
    "baseline",
    "metrics",
)


def check_sections(template: Path) -> bool:
    """Every expected section slug appears, in order (a structural presence check)."""
    text = template.read_text(encoding="utf-8")
    cursor = 0
    for slug in _EXPECTED_SECTIONS:
        marker = f"hpc-audit-section: {slug}"
        idx = text.find(marker, cursor)
        if idx < 0:
            return False
        cursor = idx + len(marker)
    return True


def main() -> int:
    experiment_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.cwd()
    passed = check_sections(experiment_dir / _TEMPLATE_REL)
    spec = {
        "pack": _PACK,
        "slot": _SLOT,
        "checked": [_TEMPLATE_REL],
        "passed": passed,
        "evidence": {"checker": "check_sections", "sections_in_order": passed},
    }
    spec_path = experiment_dir / ".hpc" / "quant_receipt_spec.json"
    spec_path.parent.mkdir(parents=True, exist_ok=True)
    spec_path.write_text(json.dumps(spec), encoding="utf-8")
    return subprocess.call(  # noqa: S603 — fixed argv, illustrative caller-side call
        [
            "hpc-agent",
            "pack-record-receipt",
            "--experiment-dir",
            str(experiment_dir),
            "--spec",
            str(spec_path),
        ]
    )


if __name__ == "__main__":
    raise SystemExit(main())
