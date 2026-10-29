# Deliberation record shapes

The JSON shapes used by the question protocol (`PROTOCOL.md`). `digest.py` validates §2 strictly: unknown keys are errors, so a misspelt field cannot be silently dropped. §1 and §3 are written by agents and read by `digest.py` and the `deliberate` skill.

**Common conventions:**

- **Encoding.** All files are UTF-8.
- **Dates.** Dates are `YYYY-MM-DD` in the author's local time (America/Los_Angeles).
- **Paths.** Paths are repo-relative with forward slashes unless stated otherwise.
- **Evidence pointers.** Each is a short string in one of these forms:
  - `path:line` (or `path:line-line`);
  - `commit <sha>`;
  - a dated author quote, `author 2026-10-26 18:58: "<words verbatim>"`;
  - `transcript <session-id> <UTC timestamp>`;
  - a URL.

---

## 1. `queue.jsonl`: raised items

Append-only, one JSON object per line. Raisers write it; nobody edits or deletes a line.

```json
{"id": "Q-2026-10-30-01", "raised_by": "agent: apply_edit for the author's 30 Oct message on Section 6", "date": "2026-10-30", "question": "Does the author's 'every bar' edit to Table 1's note also govern the appendix table tab:rh_oos_vsols?", "context": ["writeup/rehaul/06_results.tex:217", "writeup/INTEGRATION_PLAN_2026-10-26.md:51"], "stakes": "Which tables carry 16:00 columns; touches a main-text table.", "blocking": false, "origin": "chat edit 2026-10-30 09:12"}
```

| Field | Type | Required | Meaning |
|---|---|---|---|
| `id` | string | yes | Format `Q-<date raised>-<NN>`, with NN counting up from `01` per date (three digits after 99). Take 1 + the largest NN already used for that date in `queue.jsonl`. Never reused |
| `raised_by` | string | yes | Which agent or task raised it, enough to find the context |
| `date` | string | yes | Date raised |
| `question` | string | yes | The question as the raiser would have asked the author, in one or two sentences |
| `context` | list of strings | yes | Paths, `path:line` pointers, commits or quotes the solvers should start from (may be empty) |
| `stakes` | string | yes | What the answer changes and where, in one line. It is the raiser's guess; the adjudicator decides importance |
| `blocking` | boolean | no | `true` if the raiser's work stops until it is answered |
| `origin` | string | no | Where the question came from: a `% CONFLICT` comment, §8.9 Qn, a chat edit, a lock check |
| `related` | list of strings | no | Ids of earlier items this one continues (a re-raise after a veto, a failed application, new evidence) |

**Open items.** An item is open while `log.jsonl` has no line with its id.

---

## 2. `<round>/<id>.adjudication.json`: one per item

Written by the adjudicator, or by the router for T0 items. The file name must be `<id>.adjudication.json`, and the round folder is `writeup/deliberation/<round>/`. One JSON object per file.

```json
{
  "id": "Q-2026-10-30-01",
  "date": "2026-10-30",
  "question": "Does the author's 'every bar' edit to Table 1's note also govern the appendix table tab:rh_oos_vsols?",
  "adjudicated_by": "adjudicator",
  "kind": "evidence",
  "solvers": [
    {"lens": "evidence", "file": "Q-2026-10-30-01.evidence.md", "answer": "No: the appendix table is D16's home for 16:00 results.", "confidence": 0.85, "tier": "T1"},
    {"lens": "author_record", "file": "Q-2026-10-30-01.author_record.md", "answer": "No: D16 keeps 16:00 accuracy in app:rh_closing.", "confidence": 0.9, "tier": "T1"},
    {"lens": "referee", "file": "Q-2026-10-30-01.referee.md", "answer": "No: an appendix may report the closing bar if labelled.", "confidence": 0.8, "tier": "T1"}
  ],
  "agreement": "unanimous",
  "refutation": {"outcome": "failed", "note": "Searched for a later dated decision moving app:rh_closing; none after D16 (26 Oct)."},
  "lock_conflict": "none",
  "precedence_note": "D16 (26 Oct) is later than and more specific than D1 (25 Oct) for 16:00 results.",
  "final_answer": "No. The edit governs main-text tables only; tab:rh_oos_vsols keeps its 16:00 columns in App. rh_closing.",
  "answer_short": "The 'every bar' note edit stays main-text only; the appendix closing table is unchanged.",
  "rationale": "All three lenses read D16 the same way, and the refutation found nothing later.",
  "tier": "T1",
  "confidence": 0.88,
  "importance": "LOW",
  "touches": ["appendix"],
  "consequence": 2,
  "hard_trigger": [],
  "evidence": ["writeup/INTEGRATION_PLAN_2026-10-26.md:51", "author 2026-10-26 17:56: \"why is the 1600 bar still in the main paper??\""],
  "actions": [{"kind": "log_only", "target": "writeup/deliberation/log.jsonl", "detail": "Nothing to change."}],
  "consequences": "None outside the log; the appendix table and its note stay as they are.",
  "recommendation": "Leave the appendix table as it is.",
  "default": "The appendix table stays as it is."
}
```

| Field | Type | Required | Meaning |
|---|---|---|---|
| `id` | string | yes | The queue id; must equal the file name's stem |
| `date` | string | yes | Adjudication date |
| `question` | string | yes | The queued question (verbatim or tightened) |
| `adjudicated_by` | `adjudicator` \| `router` \| `legacy` | yes | `router` only for T0 items settled without solvers. `legacy` only for items of the hand-run first round, dated up to 2026-10-29 (see below) |
| `kind` | `mechanical` \| `evidence` \| `taste` | yes | Floors: `evidence` is at least T1, `taste` at least T2 (`PROTOCOL.md` §3) |
| `solvers` | list | adjudicator: yes; router: absent | One summary per solver: `lens` (`evidence`, `author_record`, `referee`, or `reviewer`), `file` (relative to the round folder), `answer` (one or two sentences), `confidence` (0-1), `tier`. Exactly one each of `evidence`, `author_record` and `referee`; any number of `reviewer` |
| `agreement` | `unanimous` \| `majority` \| `split` | adjudicator: yes; router: absent | Agreement on the substance of the answer, after adjudication. `majority` means the adjudicator resolved the minority with a checked reason. `split` means it could not |
| `refutation` | object | adjudicator: yes; router: absent | `outcome`: `failed` (the answer survived) or `succeeded` (the majority answer was overturned and `final_answer` differs from it); `note`: what was checked |
| `lock_conflict` | `none` \| `resolved` \| `unresolved` | yes | Whether the author's own decisions conflict on this item, and whether precedence (`PROTOCOL.md` §5) settles it |
| `precedence_note` | string | no | Which decision governs, and why |
| `final_answer` | string | yes | The adjudicated answer, complete enough to act on |
| `answer_short` | string | yes | One line, at most 200 characters; the digest's T2 line |
| `rationale` | string | no | Why this answer, tier and confidence |
| `tier` | `T0`-`T3` | yes | The tier the adjudicator declares. `digest.py` may raise it, never lower it |
| `confidence` | number 0-1 | yes | Probability that `final_answer` is what the author would choose on the evidence (`PROTOCOL.md` §3) |
| `importance` | `HIGH` \| `LOW` | yes | Raised to HIGH by `digest.py` if any `touches` code is a HIGH code |
| `touches` | list, non-empty | yes | Codes from `PROTOCOL.md` §6: `abstract`, `introduction`, `headline_claim`, `main_text_number`, `main_text_table`, `design`, `collaborators`, `wording`, `appendix`, `process` |
| `consequence` | integer 1-5 | yes | Orders T3 items (`PROTOCOL.md` §6 scale) |
| `hard_trigger` | list | yes (may be `[]`) | Codes: `compute`, `outward`, `config`, `delete_data`, `spend`, `irreversible`. `digest.py` adds those implied by `actions` |
| `evidence` | list, non-empty | yes | Evidence pointers (conventions above) |
| `actions` | list | yes (may be `[]`) | What applying the answer does, as objects: `kind`, `target` (path or object), optional `detail`, optional `approved_by` |
| `consequences` | string | yes | Where-used consequences: every file, table, macro, section and generator the answer changes |
| `recommendation` | string | yes | What the adjudicator recommends (for T0-T2, the action applied) |
| `default` | string | yes | What holds if the author never answers. For a hard-trigger item, always inaction |
| `alternatives` | list of strings | no | The options not recommended, one line each |
| `question_for_author` | string | if `tier` is `T3` | The one question the author is asked, answerable in one line. If `digest.py` raises an item to T3 and this field is absent, the digest asks `question` |
| `t3_reason` | string | if `tier` is `T3` | In plain words, why only the author can settle it |

**Action kinds** and the hard trigger each implies:

| `kind` | Use for | Implied trigger |
|---|---|---|
| `edit_text` | Paper, beat, review-layer or report text, through the normal editing rules (the routine Overleaf push included) | none |
| `edit_code` | Generators, checks, tools | none |
| `record_decision` | An entry in `decisions.yaml` | none |
| `log_only` | Nothing beyond the log | none |
| `run_compute` | Any run beyond a quick read-only script | `compute`, unless `approved_by` cites the author's approval of this run |
| `outward` | Messages to collaborators, Overleaf comments or suggestion decisions, submission, disclosure text, publishing | `outward` |
| `edit_config` | `CLAUDE.md`, `.claude/settings*.json`, permissions | `config` |
| `delete_data` | Deleting data or results | `delete_data` |
| `spend` | Anything that costs money | `spend` |

**Legacy items** carry `adjudicated_by: "legacy"`. They re-enter the 29 Oct round, which ran before this schema, from its Markdown adjudication.

- **Solvers.** `solvers` lists the solvers the item actually had (at least one), under their own lens names (`process_risk`, `process_goals` and so on). A solver that stated no tier or confidence has `null` there.
- **Everything else** is as for adjudicator items: agreement and refutation are required, and the same floors apply.
- **Date limit.** `digest.py` rejects `legacy` on any item dated after 2026-10-29.

**Router items** carry `adjudicated_by: "router"`, `kind: "mechanical"` and `tier: "T0"`, and no `solvers`, `agreement` or `refutation`. The decisive source goes first in `evidence`. If such an item would land above T0, `digest.py` rejects it.

---

## 3. `log.jsonl`: resolved items and later events

Append-only, one JSON object per line. An item's current state is its **last** line.

- **Digest lines.** `digest.py` writes one line per item per round (`source: "digest"`). It never writes a second line with the same `(id, round)`, so re-running a round adds nothing.
- **Event lines.** The coordinator appends author answers, vetoes and failed applications as new lines (`source: "author"` or `"agent"`). Each copies the core fields from the item's previous line and changes `status`.

```json
{"id": "Q-2026-10-30-01", "round": "2026-10-30", "date": "2026-10-30", "source": "digest", "status": "applied", "question": "...", "answer": "No. The edit governs main-text tables only; ...", "answer_short": "...", "tier": "T1", "tier_declared": "T1", "confidence": 0.88, "importance": "LOW", "hard_trigger": [], "escalated_by": [], "evidence": ["writeup/INTEGRATION_PLAN_2026-10-26.md:51"], "actions": [{"kind": "log_only", "target": "writeup/deliberation/log.jsonl"}], "adjudication": "2026-10-30/Q-2026-10-30-01.adjudication.json"}
{"id": "Q-2026-10-30-04", "round": "2026-10-30", "date": "2026-10-31", "source": "author", "status": "vetoed", "author_words": "veto Q-2026-10-30-04, keep the old title", "commit": "<sha of the revert>", "question": "...", "answer": "...", "tier": "T2", "confidence": 0.7, "importance": "HIGH", "evidence": ["..."], "actions": [{"kind": "edit_text", "target": "writeup/rehaul/06_results.tex"}], "adjudication": "2026-10-30/Q-2026-10-30-04.adjudication.json"}
```

| Field | Type | On | Meaning |
|---|---|---|---|
| `id` | string | all | Item id |
| `round` | string | all | The round folder the line belongs to. Event lines use the round of the item's last digest line |
| `date` | string | all | Digest lines: the adjudication date. Event lines: the date of the event |
| `source` | `digest` \| `author` \| `agent` | all | Who wrote the line |
| `status` | `applied` \| `provisional` \| `escalated` \| `vetoed` \| `queued` | all | See below |
| `question`, `answer`, `tier`, `confidence`, `importance`, `evidence`, `actions` | as in §2 | all | `tier` and `importance` are the effective values after `digest.py`'s floors |
| `answer_short`, `tier_declared`, `hard_trigger`, `escalated_by`, `adjudication` | as in §2 | digest lines | `escalated_by`: the reasons the effective tier is above the declared one. `adjudication`: path relative to `writeup/deliberation/` |
| `author_words` | string | author lines | The author's message, verbatim (answer or veto) |
| `commit` | string | event lines | The commit that applied or reverted the item |
| `note` | string | agent lines | For example, why an application failed |

**Status:**

| Status | Set when |
|---|---|
| `applied` | T0 or T1 (digest line); or an author's T3 answer was applied (author line) |
| `provisional` | T2, applied provisionally, open to veto |
| `escalated` | T3, shown in a digest, awaiting the author |
| `queued` | T3 beyond the three shown. `digest.py` carries it into the next digest automatically. An agent line with `queued` and a `note` means the application failed; re-raise the item in `queue.jsonl` with `related` |
| `vetoed` | The author vetoed it; its commit is reverted |

**Validation.** `digest.py` checks only that each log line is JSON with string `id`, `round` and `status` fields and a known `status`. A malformed line stops the run before anything is written.

---

## 4. Solver files: `<round>/<id>.<lens>.md`

Markdown, not validated by code. The adjudicator relies on these sections, in this order:

1. **Header.** The id, lens and date; a statement that the solver read no other solver's file and made no edit, commit or run; and the sources read.
2. **Answer.** The answer in a few sentences, complete enough to act on.
3. **Confidence.** A number from 0 to 1, with one line on what would change it.
4. **Tier.** T0-T3, with the threshold of `PROTOCOL.md` §3 it meets, any hard trigger, and the importance codes it touches.
5. **Evidence.** `file:line`, commits and dated quotes, verbatim, with the decisive ones first.
6. **If adopted.** Where-used consequences: files, generators, macros, tables and sections, and what changes in each.
7. **Alternatives.** The options considered and why each was rejected.
8. **T3 question.** Only if the tier is T3: one question answerable in one line, a recommendation and a default.
