"""2026-10-15: live pins the pre-audit src chain (forecast.PIN_PRE_AUDIT_SRC) in every copy of src/."""

from __future__ import annotations

import re
from pathlib import Path

from live.close_signal import forecast as F

REPO = Path(__file__).resolve().parents[3]


def test_the_pin_is_on_by_default_and_covers_the_vix_splice() -> None:
    assert F.PIN_PRE_AUDIT_SRC is True
    assert "SPLICE_VIX_FREE_FEED" in F.PINNED_AUDIT_FLAGS["src/data/loading.py"]


def test_copy_code_sets_every_audit_flag_false_in_the_copy_only(tmp_path: Path) -> None:
    F._copy_code(REPO, tmp_path / "root")
    for rel, names in F.PINNED_AUDIT_FLAGS.items():
        copied = (tmp_path / "root" / rel).read_text(encoding="utf-8")
        original = (REPO / rel).read_text(encoding="utf-8")
        for name in names:
            assert re.search(rf"^{name}: bool = False", copied, re.MULTILINE), (
                rel,
                name,
            )
            assert re.search(rf"^{name}: bool = True$", original, re.MULTILINE), (
                rel,
                name,
            )


def test_every_flag_of_the_audit_block_is_pinned() -> None:
    """A flag added to the executor's 2026-10-15 block must be added to the pin list too."""
    text = (REPO / "src/backtest/executor.py").read_text(encoding="utf-8")
    block = text[
        text.index('# -- 2026-10-15 (user: "fix all 8"') : text.index(
            "def early_close_post_bars"
        )
    ]
    flags = set(re.findall(r"^([A-Z_]+): bool = True$", block, re.MULTILINE))
    assert flags == set(F.PINNED_AUDIT_FLAGS["src/backtest/executor.py"])
