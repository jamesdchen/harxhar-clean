"""Tests of writeup/deliberation/digest.py, the escalation filter of the question protocol (2026-10-29).

Small synthetic rounds in a temporary folder: a hard trigger forces T3 despite high confidence; more than three T3
items leave the extras queued, and the next round shows them; a LOW T2 item is hidden from the digest; malformed input
fails and writes nothing; re-running a round adds no log line. Also checks that the module's thresholds mirror the
"Machine constants" block of writeup/deliberation/PROTOCOL.md.

Run:  python -m pytest writeup/tests/test_digest.py
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]
DELIB = ROOT / "writeup" / "deliberation"
if str(DELIB) not in sys.path:
    sys.path.insert(0, str(DELIB))

import digest  # noqa: E402

LENSES = ("evidence", "author_record", "referee")


def _adj(item_id: str, **over: Any) -> dict[str, Any]:
    """A valid T1 adjudication, LOW importance; keyword arguments override fields."""
    d: dict[str, Any] = {
        "id": item_id,
        "date": "2026-10-30",
        "question": f"Question {item_id}?",
        "adjudicated_by": "adjudicator",
        "kind": "evidence",
        "solvers": [
            {"lens": lens, "file": f"{item_id}.{lens}.md", "answer": "Yes.", "confidence": 0.9, "tier": "T1"}
            for lens in LENSES
        ],
        "agreement": "unanimous",
        "refutation": {"outcome": "failed", "note": "No later dated decision overrides the answer."},
        "lock_conflict": "none",
        "final_answer": "Yes; apply it.",
        "answer_short": f"Answer for {item_id}.",
        "tier": "T1",
        "confidence": 0.9,
        "importance": "LOW",
        "touches": ["wording"],
        "consequence": 2,
        "hard_trigger": [],
        "evidence": ["writeup/INTEGRATION_PLAN_2026-10-26.md:54"],
        "actions": [{"kind": "edit_text", "target": "writeup/rehaul/01_introduction.tex"}],
        "consequences": "One sentence of the introduction.",
        "recommendation": "Apply it.",
        "default": "Applied as recommended.",
    }
    d.update(over)
    return d


def _t3(item_id: str, importance: str, consequence: int) -> dict[str, Any]:
    return _adj(
        item_id,
        kind="taste",
        tier="T3",
        confidence=0.5,
        agreement="split",
        importance=importance,
        touches=["abstract"] if importance == "HIGH" else ["appendix"],
        consequence=consequence,
        question_for_author=f"Ask {item_id}?",
        t3_reason="The record is silent on this.",
    )


def _router(item_id: str, **over: Any) -> dict[str, Any]:
    d = _adj(item_id, adjudicated_by="router", kind="mechanical", tier="T0", confidence=1.0, **over)
    for key in ("solvers", "agreement", "refutation"):
        d.pop(key)
    return d


def _write(folder: Path, *items: dict[str, Any]) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    for d in items:
        (folder / f"{d['id']}.adjudication.json").write_text(json.dumps(d, indent=2), encoding="utf-8")
    return folder


def _log(folder: Path) -> list[dict[str, Any]]:
    path = folder.parent / "log.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _digest(folder: Path) -> str:
    return (folder / "digest.md").read_text(encoding="utf-8")


def test_hard_trigger_forces_t3_despite_high_confidence(tmp_path: Path) -> None:
    rnd = _write(
        tmp_path / "deliberation" / "2026-10-30",
        _adj("Q-2026-10-30-01", confidence=0.97, hard_trigger=["outward"]),
        _adj("Q-2026-10-30-02", confidence=0.97, actions=[{"kind": "run_compute", "target": "forest arm"}]),
        _adj(
            "Q-2026-10-30-03",
            confidence=0.97,
            actions=[{"kind": "run_compute", "target": "ladder rerun", "approved_by": 'author 2026-10-28: "go"'}],
        ),
    )
    assert digest.main([str(rnd)]) == 0
    log = {e["id"]: e for e in _log(rnd)}
    assert log["Q-2026-10-30-01"]["tier"] == "T3"
    assert log["Q-2026-10-30-01"]["tier_declared"] == "T1"
    assert log["Q-2026-10-30-01"]["status"] == "escalated"
    assert log["Q-2026-10-30-01"]["hard_trigger"] == ["outward"]
    assert log["Q-2026-10-30-02"]["tier"] == "T3"  # compute nobody approved: the action alone triggers
    assert log["Q-2026-10-30-02"]["hard_trigger"] == ["compute"]
    assert log["Q-2026-10-30-03"]["tier"] == "T1"  # the author approved this run
    assert log["Q-2026-10-30-03"]["status"] == "applied"
    text = _digest(rnd)
    assert "`Q-2026-10-30-01 yes`" in text
    assert "hard trigger: it is outward-facing" in text
    assert "Question Q-2026-10-30-01?" in text  # no question_for_author given: the question itself is asked


def test_more_than_three_t3_leave_extras_queued_for_the_next_round(tmp_path: Path) -> None:
    first = _write(
        tmp_path / "deliberation" / "2026-10-30",
        _t3("Q-2026-10-30-01", "LOW", 5),
        _t3("Q-2026-10-30-02", "HIGH", 3),
        _t3("Q-2026-10-30-03", "LOW", 4),
        _t3("Q-2026-10-30-04", "HIGH", 5),
        _t3("Q-2026-10-30-05", "LOW", 1),
    )
    assert digest.main([str(first)]) == 0
    text = _digest(first)
    order = re.findall(r"^### \d\. .*\((Q-[\d-]+)\)$", text, flags=re.M)
    assert order == ["Q-2026-10-30-04", "Q-2026-10-30-02", "Q-2026-10-30-01"]  # HIGH first, then by consequence
    assert "2 more question(s) for you are queued for the next digest: Q-2026-10-30-03, Q-2026-10-30-05." in text
    status = {e["id"]: e["status"] for e in _log(first)}
    assert [k for k, v in sorted(status.items()) if v == "queued"] == ["Q-2026-10-30-03", "Q-2026-10-30-05"]
    assert sum(v == "escalated" for v in status.values()) == digest.MAX_T3_PER_DIGEST

    second = _write(tmp_path / "deliberation" / "2026-10-31", _adj("Q-2026-10-31-01"))
    assert digest.main([str(second)]) == 0
    text2 = _digest(second)
    assert re.findall(r"^### \d\. .*\((Q-[\d-]+)\)$", text2, flags=re.M) == ["Q-2026-10-30-03", "Q-2026-10-30-05"]
    assert "Still waiting for your answer from earlier digests: Q-2026-10-30-01 (2026-10-30)" in text2
    carried = [e for e in _log(second) if e["round"] == "2026-10-31" and e["id"].startswith("Q-2026-10-30")]
    assert sorted(e["id"] for e in carried) == ["Q-2026-10-30-03", "Q-2026-10-30-05"]
    assert all(e["status"] == "escalated" for e in carried)


def test_low_t2_is_hidden_and_high_t2_gets_one_line(tmp_path: Path) -> None:
    rnd = _write(
        tmp_path / "deliberation" / "2026-10-30",
        _adj("Q-2026-10-30-01", kind="taste", tier="T2", answer_short="Low taste call."),
        # declared LOW, but it touches a main-text table: importance is raised mechanically
        _adj("Q-2026-10-30-02", tier="T2", confidence=0.7, touches=["main_text_table"], answer_short="Table call."),
        _adj("Q-2026-10-30-03"),
        _router("Q-2026-10-30-04"),
    )
    assert digest.main([str(rnd)]) == 0
    text = _digest(rnd)
    assert "Q-2026-10-30-01" not in text
    assert "Low taste call." not in text
    assert text.count("Q-2026-10-30-02") == 1
    assert "- Q-2026-10-30-02: Table call." in text
    assert "T0: 1, T1: 1, minor T2: 1;" in text
    log = {e["id"]: e for e in _log(rnd)}
    assert log["Q-2026-10-30-01"]["status"] == "provisional"
    assert log["Q-2026-10-30-02"]["importance"] == "HIGH"
    assert log["Q-2026-10-30-04"]["status"] == "applied"


@pytest.mark.parametrize(
    ("over", "tier"),
    [
        ({"confidence": 0.7}, "T2"),  # T1 needs confidence >= 0.8
        ({"agreement": "majority"}, "T2"),  # T1 needs every solver agreeing
        ({"refutation": {"outcome": "succeeded", "note": "D20 overrides."}}, "T2"),
        ({"kind": "taste"}, "T2"),
        ({"confidence": 0.59}, "T3"),
        ({"agreement": "split"}, "T3"),
        ({"lock_conflict": "unresolved"}, "T3"),
        (
            {"tier": "T3", "confidence": 0.95, "question_for_author": "Ask?", "t3_reason": "Taste."},
            "T3",
        ),  # never lowered
        ({}, "T1"),
    ],
)
def test_thresholds(over: dict[str, Any], tier: str) -> None:
    d = _adj("Q-2026-10-30-01", **over)
    assert digest.validate(d, "fixture") == []
    assert digest.classify(d)[0] == tier


def _broken_missing_field(d: dict[str, Any]) -> str:
    d.pop("recommendation")
    return json.dumps(d)


def _broken_tier(d: dict[str, Any]) -> str:
    d["tier"] = "T4"
    return json.dumps(d)


def _broken_id(d: dict[str, Any]) -> str:
    d["id"] = "Q-2026-10-30-99"
    return json.dumps(d)


def _broken_lens(d: dict[str, Any]) -> str:
    d["solvers"] = [s for s in d["solvers"] if s["lens"] != "referee"]
    return json.dumps(d)


def _broken_key(d: dict[str, Any]) -> str:
    d["hard_triggers"] = d.pop("hard_trigger")
    return json.dumps(d)


def _broken_json(d: dict[str, Any]) -> str:
    return json.dumps(d)[:-5]


def _broken_router(d: dict[str, Any]) -> str:
    return json.dumps(_router(d["id"], hard_trigger=["spend"]))


@pytest.mark.parametrize(
    ("breaker", "message"),
    [
        (_broken_missing_field, "field 'recommendation': missing"),
        (_broken_tier, "field 'tier': must be one of"),
        (_broken_id, "but the file name says"),
        (_broken_lens, "needs exactly one 'referee' solver"),
        (_broken_key, "field 'hard_triggers': not a field"),
        (_broken_json, "not valid JSON"),
        (_broken_router, "the router may only settle T0 items"),
    ],
)
def test_malformed_input_fails_and_writes_nothing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], breaker: Any, message: str
) -> None:
    rnd = _write(tmp_path / "deliberation" / "2026-10-30", _adj("Q-2026-10-30-01"))
    (rnd / "Q-2026-10-30-02.adjudication.json").write_text(breaker(_adj("Q-2026-10-30-02")), encoding="utf-8")
    assert digest.main([str(rnd)]) == 1
    err = capsys.readouterr().err
    assert "Q-2026-10-30-02.adjudication.json" in err
    assert message in err
    assert not (rnd / "digest.md").exists()
    assert not (rnd.parent / "log.jsonl").exists()


def test_malformed_input_exits_non_zero_from_the_command_line(tmp_path: Path) -> None:
    rnd = _write(tmp_path / "deliberation" / "2026-10-30", _adj("Q-2026-10-30-01", confidence="high"))
    proc = subprocess.run(
        [sys.executable, str(DELIB / "digest.py"), str(rnd)], capture_output=True, text=True, check=False
    )
    assert proc.returncode == 1
    assert "field 'confidence'" in proc.stderr


def test_folder_without_adjudications_is_refused_and_untouched(tmp_path: Path) -> None:
    rnd = tmp_path / "deliberation" / "2026-10-29"
    rnd.mkdir(parents=True)
    (rnd / "digest.md").write_text("hand-written\n", encoding="utf-8")
    assert digest.main([str(rnd)]) == 2
    assert (rnd / "digest.md").read_text(encoding="utf-8") == "hand-written\n"


def test_rerun_does_not_duplicate_log_lines(tmp_path: Path) -> None:
    rnd = _write(
        tmp_path / "deliberation" / "2026-10-30",
        _adj("Q-2026-10-30-01"),
        _adj("Q-2026-10-30-02", tier="T2", kind="taste", importance="HIGH"),
        *(_t3(f"Q-2026-10-30-{n:02d}", "LOW", n - 2) for n in range(3, 8)),
    )
    assert digest.main([str(rnd)]) == 0
    first_log, first_digest = _log(rnd), _digest(rnd)
    assert digest.main([str(rnd)]) == 0
    assert digest.main([str(rnd)]) == 0
    assert _log(rnd) == first_log
    assert _digest(rnd) == first_digest
    keys = [(e["id"], e["round"]) for e in first_log]
    assert len(keys) == len(set(keys)) == 7


def test_constants_mirror_protocol() -> None:
    text = (DELIB / "PROTOCOL.md").read_text(encoding="utf-8")

    def value(name: str) -> str:
        found = re.findall(rf"^{name} = (.+)$", text, flags=re.M)
        assert len(found) == 1, f"PROTOCOL.md must state {name} once in its machine-constants block"
        return found[0].strip()

    def names(name: str) -> list[str]:
        return [x.strip() for x in value(name).split(",")]

    assert float(value("T1_MIN_CONFIDENCE")) == digest.T1_MIN_CONFIDENCE
    assert float(value("T2_MIN_CONFIDENCE")) == digest.T2_MIN_CONFIDENCE
    assert int(value("MAX_T3_PER_DIGEST")) == digest.MAX_T3_PER_DIGEST
    assert names("HARD_TRIGGERS") == list(digest.HARD_TRIGGERS)
    assert names("HIGH_TOUCHES") == list(digest.HIGH_TOUCHES)
    assert names("LOW_TOUCHES") == list(digest.LOW_TOUCHES)
