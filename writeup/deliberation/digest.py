#!/usr/bin/env python3
"""Escalation filter for one deliberation round (writeup/deliberation/PROTOCOL.md; shapes in SCHEMA.md).

Reads every ``<id>.adjudication.json`` in a round folder and validates it. It then recomputes each item's tier and
importance from the protocol's thresholds and hard triggers. The recomputation can only raise what the adjudicator
declared, never lower it. It writes ``<folder>/digest.md``, the only text the author is sent, and appends the round's
items to ``log.jsonl``. Re-running is safe: a ``(id, round)`` pair already in the log is never written again.

Standard library only.

Run:
    python writeup/deliberation/digest.py writeup/deliberation/<round>             # writes digest.md, appends log
    python writeup/deliberation/digest.py writeup/deliberation/<round> --dry-run   # validates and prints only

Exit codes: 0 done; 1 a file is malformed (nothing is written); 2 bad usage (no folder, or no adjudication files).
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------------------------------------------------
# Protocol constants. They mirror PROTOCOL.md section 3 ("Machine constants"); writeup/tests/test_digest.py fails when
# the two drift apart. Change both together, and only on the author's word (PROTOCOL.md section 12).
# ---------------------------------------------------------------------------------------------------------------------
T1_MIN_CONFIDENCE = 0.8  # T0/T1 need at least this, plus unanimous solvers and a failed refutation
T2_MIN_CONFIDENCE = 0.6  # below this an item is T3
MAX_T3_PER_DIGEST = 3  # T3 questions shown per digest; the rest stay queued for the next digest

TIERS = ("T0", "T1", "T2", "T3")
HARD_TRIGGERS: dict[str, str] = {
    "compute": "launches compute beyond the runs you already approved",
    "outward": "is outward-facing (collaborators, Overleaf comments or suggestions, submission, disclosure, publishing)",
    "config": "changes CLAUDE.md, settings or permissions",
    "delete_data": "deletes data",
    "spend": "spends money",
    "irreversible": "cannot be undone by a git revert",
}
# Action kind -> the hard trigger it implies (None: no trigger). run_compute is exempt when approved_by cites the
# author's approval of that run.
ACTION_TRIGGERS: dict[str, str | None] = {
    "edit_text": None,
    "edit_code": None,
    "record_decision": None,
    "log_only": None,
    "run_compute": "compute",
    "outward": "outward",
    "edit_config": "config",
    "delete_data": "delete_data",
    "spend": "spend",
}
HIGH_TOUCHES = (
    "abstract",
    "introduction",
    "headline_claim",
    "main_text_number",
    "main_text_table",
    "design",
    "collaborators",
)
LOW_TOUCHES = ("wording", "appendix", "process")
KIND_FLOOR = {"mechanical": 0, "evidence": 1, "taste": 2}
LENSES_REQUIRED = ("evidence", "author_record", "referee")
LENSES_EXTRA = ("reviewer",)
AGREEMENTS = ("unanimous", "majority", "split")
REFUTATIONS = ("failed", "succeeded")
LOCK_CONFLICTS = ("none", "resolved", "unresolved")
ADJUDICATED_BY = ("adjudicator", "router")
IMPORTANCE = ("HIGH", "LOW")
STATUSES = ("applied", "provisional", "escalated", "vetoed", "queued")
STATUS_BY_TIER = {"T0": "applied", "T1": "applied", "T2": "provisional"}
ANSWER_SHORT_MAX = 200
CONSEQUENCE_MIN, CONSEQUENCE_MAX = 1, 5

ID_RE = re.compile(r"^Q-\d{4}-\d{2}-\d{2}-\d{2,3}$")
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
SUFFIX = ".adjudication.json"

_TEXT = ("id", "date", "question", "final_answer", "answer_short", "recommendation", "default", "consequences")
_ALLOWED_KEYS = frozenset(
    _TEXT
    + (
        "adjudicated_by",
        "kind",
        "tier",
        "confidence",
        "importance",
        "touches",
        "consequence",
        "lock_conflict",
        "precedence_note",
        "hard_trigger",
        "evidence",
        "actions",
        "solvers",
        "agreement",
        "refutation",
        "question_for_author",
        "t3_reason",
        "alternatives",
        "rationale",
    )
)
_ACTION_KEYS = frozenset(("kind", "target", "detail", "approved_by"))
_SOLVER_KEYS = frozenset(("lens", "file", "answer", "confidence", "tier"))
_REFUTATION_KEYS = frozenset(("outcome", "note"))


class UsageError(Exception):
    """Bad invocation: exit code 2."""


class Malformed(Exception):
    """A file departs from SCHEMA.md: exit code 1, nothing written."""


@dataclass
class Item:
    """One adjudicated item with the tier and importance the protocol assigns it."""

    data: dict[str, Any]
    path: Path
    round: str  # the round folder the adjudication lives in
    tier: str  # effective tier (never below the declared one)
    importance: str  # effective importance
    triggers: list[str]  # declared hard triggers plus those implied by the actions
    escalated_by: list[str]  # reasons the effective tier is above the declared one
    t3_reasons: list[str]  # plain-language reasons the item goes to the author (T3 only)

    @property
    def id(self) -> str:
        return str(self.data["id"])

    @property
    def consequence(self) -> int:
        return int(self.data["consequence"])


# ---------------------------------------------------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------------------------------------------------
def _is_number(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def validate(d: Any, where: str) -> list[str]:
    """Return every way ``d`` departs from SCHEMA.md's adjudication shape (empty list: valid)."""
    if not isinstance(d, dict):
        return [f"{where}: the file must hold one JSON object"]
    errs: list[str] = []

    def bad(key: str, msg: str) -> None:
        errs.append(f"{where}: field '{key}': {msg}")

    def text(obj: dict[str, Any], key: str, label: str, *, required: bool = True) -> bool:
        if key not in obj:
            if required:
                bad(label, "missing")
            return False
        if not isinstance(obj[key], str) or not obj[key].strip():
            bad(label, "must be a non-empty string")
            return False
        return True

    def choice(obj: dict[str, Any], key: str, label: str, options: tuple[str, ...]) -> bool:
        if key not in obj:
            bad(label, "missing")
            return False
        if obj[key] not in options:
            bad(label, f"must be one of {', '.join(options)} (got {obj[key]!r})")
            return False
        return True

    def str_list(key: str, *, non_empty: bool, options: tuple[str, ...] | None = None, required: bool = True) -> None:
        if key not in d:
            if required:
                bad(key, "missing")
            return
        v = d[key]
        if not isinstance(v, list) or not all(isinstance(x, str) and x.strip() for x in v):
            bad(key, "must be a list of non-empty strings")
            return
        if non_empty and not v:
            bad(key, "must not be empty")
        if options is not None:
            unknown = [x for x in v if x not in options]
            if unknown:
                bad(key, f"unknown value(s) {', '.join(unknown)}; allowed: {', '.join(options)}")

    for key in sorted(set(d) - _ALLOWED_KEYS):
        bad(key, "not a field of the adjudication schema (typo?)")

    for key in _TEXT:
        text(d, key, key)
    if isinstance(d.get("id"), str) and not ID_RE.match(d["id"]):
        bad("id", f"must look like Q-2026-10-29-01 (got {d['id']!r})")
    if isinstance(d.get("date"), str) and not DATE_RE.match(d["date"]):
        bad("date", f"must be YYYY-MM-DD (got {d['date']!r})")
    short = d.get("answer_short")
    if isinstance(short, str):
        if "\n" in short:
            bad("answer_short", "must be one line")
        if len(short) > ANSWER_SHORT_MAX:
            bad("answer_short", f"must be at most {ANSWER_SHORT_MAX} characters (has {len(short)})")
    for key in ("precedence_note", "rationale"):
        text(d, key, key, required=False)

    adjudicated_by_ok = choice(d, "adjudicated_by", "adjudicated_by", ADJUDICATED_BY)
    choice(d, "kind", "kind", tuple(KIND_FLOOR))
    tier_ok = choice(d, "tier", "tier", TIERS)
    choice(d, "importance", "importance", IMPORTANCE)
    choice(d, "lock_conflict", "lock_conflict", LOCK_CONFLICTS)

    if "confidence" not in d:
        bad("confidence", "missing")
    elif not _is_number(d["confidence"]) or not 0.0 <= float(d["confidence"]) <= 1.0:
        bad("confidence", f"must be a number from 0 to 1 (got {d['confidence']!r})")
    if "consequence" not in d:
        bad("consequence", "missing")
    elif (
        not isinstance(d["consequence"], int)
        or isinstance(d["consequence"], bool)
        or not CONSEQUENCE_MIN <= d["consequence"] <= CONSEQUENCE_MAX
    ):
        bad("consequence", f"must be an integer from {CONSEQUENCE_MIN} to {CONSEQUENCE_MAX} (got {d['consequence']!r})")

    str_list("touches", non_empty=True, options=HIGH_TOUCHES + LOW_TOUCHES)
    str_list("hard_trigger", non_empty=False, options=tuple(HARD_TRIGGERS))
    str_list("evidence", non_empty=True)
    str_list("alternatives", non_empty=False, required=False)

    if "actions" not in d:
        bad("actions", "missing (use [] when nothing is to be done)")
    elif not isinstance(d["actions"], list):
        bad("actions", "must be a list of objects")
    else:
        for n, a in enumerate(d["actions"]):
            label = f"actions[{n}]"
            if not isinstance(a, dict):
                bad(label, "must be an object")
                continue
            for key in sorted(set(a) - _ACTION_KEYS):
                bad(f"{label}.{key}", "not a field of an action")
            choice(a, "kind", f"{label}.kind", tuple(ACTION_TRIGGERS))
            text(a, "target", f"{label}.target")
            text(a, "detail", f"{label}.detail", required=False)
            text(a, "approved_by", f"{label}.approved_by", required=False)

    if tier_ok and d["tier"] == "T3":
        text(d, "question_for_author", "question_for_author")
        text(d, "t3_reason", "t3_reason")
    else:
        text(d, "question_for_author", "question_for_author", required=False)
        text(d, "t3_reason", "t3_reason", required=False)

    if adjudicated_by_ok and d["adjudicated_by"] == "router":
        if d.get("solvers") or "agreement" in d or "refutation" in d:
            bad("adjudicated_by", "a router item carries no solvers, agreement or refutation")
        if d.get("kind") != "mechanical":
            bad("kind", "the router settles only mechanical items; send this item to the solvers")
        if d.get("tier") != "T0":
            bad("tier", "the router may only assign T0; send this item to the solvers")
    elif adjudicated_by_ok:
        choice(d, "agreement", "agreement", AGREEMENTS)
        ref = d.get("refutation")
        if not isinstance(ref, dict):
            bad("refutation", "missing or not an object; the adjudicator must attempt a refutation")
        else:
            for key in sorted(set(ref) - _REFUTATION_KEYS):
                bad(f"refutation.{key}", "not a field of a refutation")
            choice(ref, "outcome", "refutation.outcome", REFUTATIONS)
            text(ref, "note", "refutation.note")
        solvers = d.get("solvers")
        if not isinstance(solvers, list):
            bad("solvers", "missing or not a list")
        else:
            lenses: list[str] = []
            for n, s in enumerate(solvers):
                label = f"solvers[{n}]"
                if not isinstance(s, dict):
                    bad(label, "must be an object")
                    continue
                for key in sorted(set(s) - _SOLVER_KEYS):
                    bad(f"{label}.{key}", "not a field of a solver summary")
                if choice(s, "lens", f"{label}.lens", LENSES_REQUIRED + LENSES_EXTRA):
                    lenses.append(s["lens"])
                text(s, "file", f"{label}.file")
                text(s, "answer", f"{label}.answer")
                choice(s, "tier", f"{label}.tier", TIERS)
                if not _is_number(s.get("confidence")) or not 0.0 <= float(s["confidence"]) <= 1.0:
                    bad(f"{label}.confidence", "must be a number from 0 to 1")
            for lens in LENSES_REQUIRED:
                if lenses.count(lens) != 1:
                    bad("solvers", f"needs exactly one '{lens}' solver (has {lenses.count(lens)})")
    return errs


# ---------------------------------------------------------------------------------------------------------------------
# The escalation filter
# ---------------------------------------------------------------------------------------------------------------------
def action_trigger(action: dict[str, Any]) -> str | None:
    """The hard trigger an action implies; compute the author already approved implies none."""
    trigger = ACTION_TRIGGERS[action["kind"]]
    if trigger == "compute" and str(action.get("approved_by", "")).strip():
        return None
    return trigger


def classify(d: dict[str, Any]) -> tuple[str, str, list[str], list[str], list[str]]:
    """Effective (tier, importance, hard triggers, escalated_by, t3_reasons) of a valid adjudication."""
    declared = TIERS.index(d["tier"])
    triggers = sorted(set(d["hard_trigger"]) | {t for a in d["actions"] if (t := action_trigger(a)) is not None})
    conf = float(d["confidence"])
    floors: list[tuple[int, str]] = []
    if d["kind"] == "evidence":
        floors.append((1, "an evidence question is at least T1"))
    if d["kind"] == "taste":
        floors.append((2, "a matter of taste is at least T2"))
    if d["adjudicated_by"] == "adjudicator":
        if d["agreement"] == "majority":
            floors.append((2, "the solvers were not unanimous"))
        if d["agreement"] == "split":
            floors.append((3, "the solvers are still split after adjudication"))
        if d["refutation"]["outcome"] == "succeeded":
            floors.append((2, "the refutation overturned the solvers' majority answer"))
    if conf < T1_MIN_CONFIDENCE:
        floors.append((2, f"confidence {conf:.2f} is below {T1_MIN_CONFIDENCE}"))
    if conf < T2_MIN_CONFIDENCE:
        floors.append((3, f"confidence {conf:.2f} is below {T2_MIN_CONFIDENCE}"))
    if d["lock_conflict"] == "unresolved":
        floors.append((3, "your own decisions conflict and no later one settles it"))
    for t in triggers:
        floors.append((3, f"hard trigger: it {HARD_TRIGGERS[t]}"))

    tier = TIERS[max([declared] + [f for f, _ in floors])]
    escalated_by = [reason for f, reason in floors if f > declared]
    t3_reasons: list[str] = []
    if tier == "T3":
        if d["tier"] == "T3":
            t3_reasons.append(str(d.get("t3_reason", "")).strip())
        t3_reasons += [reason for f, reason in floors if f == 3]
    importance = "HIGH" if d["importance"] == "HIGH" or any(t in HIGH_TOUCHES for t in d["touches"]) else "LOW"
    return tier, importance, triggers, escalated_by, t3_reasons


def load_item(path: Path, round_name: str) -> tuple[Item | None, list[str]]:
    """Read, validate and classify one adjudication file."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError) as exc:
        return None, [f"{path}: cannot be read ({exc})"]
    except json.JSONDecodeError as exc:
        return None, [f"{path}: not valid JSON ({exc})"]
    errs = validate(data, str(path))
    expected = path.name[: -len(SUFFIX)]
    if isinstance(data, dict) and isinstance(data.get("id"), str) and data["id"] != expected:
        errs.append(f"{path}: field 'id': is {data['id']!r} but the file name says {expected!r}")
    if errs:
        return None, errs
    tier, importance, triggers, escalated_by, t3_reasons = classify(data)
    if data["adjudicated_by"] == "router" and tier != "T0":
        why = "; ".join(escalated_by)
        return None, [
            f"{path}: the router may only settle T0 items, but this one is {tier} ({why}); send it to solvers"
        ]
    return Item(data, path, round_name, tier, importance, triggers, escalated_by, t3_reasons), []


def read_log(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    """The log's entries in order (a missing file is an empty log)."""
    if not path.exists():
        return [], []
    entries: list[dict[str, Any]] = []
    errs: list[str] = []
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            errs.append(f"{path}:{n}: not valid JSON ({exc})")
            continue
        if not isinstance(obj, dict) or not all(isinstance(obj.get(k), str) for k in ("id", "round", "status")):
            errs.append(f"{path}:{n}: a log line needs string fields 'id', 'round' and 'status'")
            continue
        if obj["status"] not in STATUSES:
            errs.append(f"{path}:{n}: status {obj['status']!r} is not one of {', '.join(STATUSES)}")
            continue
        entries.append(obj)
    return entries, errs


def _display(path: Path) -> str:
    try:
        return path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _rel(path: Path, base: Path) -> str:
    try:
        return path.resolve().relative_to(base.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def log_line(item: Item, status: str, round_name: str, log_dir: Path) -> dict[str, Any]:
    d = item.data
    return {
        "id": item.id,
        "round": round_name,
        "date": d["date"],
        "source": "digest",
        "status": status,
        "question": d["question"],
        "answer": d["final_answer"],
        "answer_short": d["answer_short"],
        "tier": item.tier,
        "tier_declared": d["tier"],
        "confidence": d["confidence"],
        "importance": item.importance,
        "hard_trigger": item.triggers,
        "escalated_by": item.escalated_by,
        "evidence": d["evidence"],
        "actions": d["actions"],
        "adjudication": _rel(item.path, log_dir),
    }


def render_digest(
    round_name: str,
    shown: list[Item],
    queued: list[Item],
    t2_high: list[Item],
    counts: dict[str, int],
    waiting: list[dict[str, Any]],
    log_display: str,
) -> str:
    out: list[str] = [f"# Deliberation digest, round {round_name}", ""]
    logged = counts["T0"] + counts["T1"] + counts["T2_LOW"]
    out += [
        f"Questions for you: {len(shown)}. Applied provisionally, yours to veto: {len(t2_high)}. "
        f"Settled without you and logged: {logged}.",
        "",
        "## Questions for you",
        "",
    ]
    if not shown:
        out += ["None this round.", ""]
    for n, item in enumerate(shown, start=1):
        d = item.data
        ask = str(d.get("question_for_author") or d["question"]).strip()
        out += [f"### {n}. {ask} ({item.id})", ""]
        out.append(f"- Recommendation: {d['recommendation'].strip()}")
        out.append(f"- What it changes: {d['consequences'].strip()}")
        out.append(f"- Default if you do not answer: {d['default'].strip()}")
        conf_line = f"- Confidence in the recommendation: {float(d['confidence']):.2f}."
        if d.get("alternatives"):
            conf_line += " Alternatives: " + "; ".join(a.strip() for a in d["alternatives"]) + "."
        out.append(conf_line)
        reasons = list(dict.fromkeys(r.strip().rstrip(".") for r in item.t3_reasons if r.strip()))
        reasons = reasons or ["the adjudicator judged it yours"]
        out.append(f"- Why it comes to you: {'; '.join(reasons)}.")
        if item.round != round_name:
            out.append(f"- Deliberated in round {item.round}; held back from an earlier digest by the limit of three.")
        out.append(f"- Answer in one line: `{item.id} yes` to take the recommendation, or `{item.id} <your answer>`.")
        out.append("")
    if queued:
        ids = ", ".join(i.id for i in queued)
        out += [f"{len(queued)} more question(s) for you are queued for the next digest: {ids}.", ""]
    out += ["## Applied provisionally (reply `veto <ID>` to undo one)", ""]
    if not t2_high:
        out += ["None this round.", ""]
    for item in t2_high:
        out.append(f"- {item.id}: {item.data['answer_short'].strip()}")
    if t2_high:
        out.append("")
    out += [
        "## Logged only",
        "",
        f"T0: {counts['T0']}, T1: {counts['T1']}, minor T2: {counts['T2_LOW']}; every item is in {log_display}.",
    ]
    if waiting:
        listed = ", ".join(f"{e['id']} ({e['round']})" for e in waiting)
        out.append(f"Still waiting for your answer from earlier digests: {listed}.")
    return "\n".join(out) + "\n"


def run(folder: Path, log_path: Path, *, dry_run: bool = False) -> tuple[str, list[dict[str, Any]]]:
    """Process one round. Returns (digest text, log lines appended). Raises UsageError or Malformed."""
    if not folder.is_dir():
        raise UsageError(f"{folder} is not a folder")
    files = sorted(folder.glob("*" + SUFFIX))
    if not files:
        raise UsageError(f"{folder} holds no *{SUFFIX} files; nothing to digest (digest.md left untouched)")
    round_name = folder.name
    log_dir = log_path.parent

    errs: list[str] = []
    items: list[Item] = []
    for path in files:
        item, item_errs = load_item(path, round_name)
        errs += item_errs
        if item is not None:
            items.append(item)
    log_entries, log_errs = read_log(log_path)
    errs += log_errs

    # State before this round: the last log line per id, ignoring this round's own digest lines (keeps re-runs stable).
    current_ids = {i.id for i in items}
    prior: dict[str, dict[str, Any]] = {}
    for e in log_entries:
        if e.get("source", "digest") == "digest" and e["round"] == round_name:
            continue
        prior[e["id"]] = e
    carried: list[Item] = []
    waiting: list[dict[str, Any]] = []
    for item_id, e in prior.items():
        if item_id in current_ids:
            continue
        if e["status"] == "queued" and e.get("tier") == "T3":
            ref = e.get("adjudication")
            if not isinstance(ref, str) or not (log_dir / ref).is_file():
                errs.append(f"{log_path}: queued item {item_id} points to a missing adjudication file ({ref!r})")
                continue
            ref_path = log_dir / ref
            item, item_errs = load_item(ref_path, ref_path.parent.name)
            errs += item_errs
            if item is not None:
                carried.append(item)
        elif e["status"] == "escalated":
            waiting.append(e)
    if errs:
        raise Malformed("\n".join(errs))

    t3 = sorted(
        (i for i in items + carried if i.tier == "T3"),
        key=lambda i: (i.importance != "HIGH", -i.consequence, i.round, i.id),
    )
    shown, queued = t3[:MAX_T3_PER_DIGEST], t3[MAX_T3_PER_DIGEST:]
    shown_ids = {i.id for i in shown}
    t2_high = sorted(
        (i for i in items if i.tier == "T2" and i.importance == "HIGH"), key=lambda i: (-i.consequence, i.id)
    )
    counts = {
        "T0": sum(i.tier == "T0" for i in items),
        "T1": sum(i.tier == "T1" for i in items),
        "T2_LOW": sum(i.tier == "T2" and i.importance == "LOW" for i in items),
    }
    digest = render_digest(round_name, shown, queued, t2_high, counts, waiting, _display(log_path))

    existing = {(e["id"], e["round"]) for e in log_entries if e.get("source", "digest") == "digest"}
    new_lines: list[dict[str, Any]] = []
    for item in sorted(items, key=lambda i: i.id):
        status = STATUS_BY_TIER.get(item.tier) or ("escalated" if item.id in shown_ids else "queued")
        if (item.id, round_name) not in existing:
            new_lines.append(log_line(item, status, round_name, log_dir))
    for item in sorted(carried, key=lambda i: i.id):
        if item.id in shown_ids and (item.id, round_name) not in existing:
            new_lines.append(log_line(item, "escalated", round_name, log_dir))

    if not dry_run:
        (folder / "digest.md").write_text(digest, encoding="utf-8", newline="\n")
        if new_lines:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            prefix = ""
            if log_path.exists():
                raw = log_path.read_bytes()
                if raw and not raw.endswith(b"\n"):
                    prefix = "\n"
            with log_path.open("a", encoding="utf-8", newline="\n") as fh:
                fh.write(prefix + "".join(json.dumps(line, ensure_ascii=False) + "\n" for line in new_lines))
    return digest, new_lines


def main(argv: list[str] | None = None) -> int:
    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if callable(reconfigure):
        reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("folder", type=Path, help="the round folder, e.g. writeup/deliberation/2026-10-30")
    parser.add_argument("--log", type=Path, default=None, help="log file (default: <folder>/../log.jsonl)")
    parser.add_argument("--dry-run", action="store_true", help="validate and print the digest; write nothing")
    args = parser.parse_args(argv)
    folder: Path = args.folder
    log_path: Path = args.log if args.log is not None else folder.parent / "log.jsonl"
    try:
        digest, new_lines = run(folder, log_path, dry_run=args.dry_run)
    except UsageError as exc:
        print(f"digest.py: {exc}", file=sys.stderr)
        return 2
    except Malformed as exc:
        print(f"digest.py: malformed input, nothing written:\n{exc}", file=sys.stderr)
        return 1
    if args.dry_run:
        print(digest, end="")
        print(f"[dry run] {len(new_lines)} log line(s) would be appended to {_display(log_path)}.")
        return 0
    shown_path, log_shown = _display(folder / "digest.md"), _display(log_path)
    print(f"digest.py: wrote {shown_path}; appended {len(new_lines)} line(s) to {log_shown}")
    decisions = folder.parent.parent / "decisions.yaml"
    to_record = [line["id"] for line in new_lines if line["status"] in ("applied", "provisional")]
    if decisions.exists() and to_record:
        print(f"decisions.yaml exists: record {', '.join(to_record)} there with source 'inferred' and their evidence.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
