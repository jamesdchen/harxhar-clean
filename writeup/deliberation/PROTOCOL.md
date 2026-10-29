# Question-escalation protocol

The single source of rules for how agents on this paper work through questions before any of them reaches the author.
It implements §8.10 ("Question tiers and deliberation") and the §8.6 rule "No question goes to the author without a
saved deliberation" of `writeup/reports/AI layered paper writing system.md`. Where the two differ, this file governs
the mechanics, and the author's decisions (A1 onward) govern everything.

- **Data shapes:** `SCHEMA.md`.
- **The mechanical filter:** `digest.py`. Its constants mirror §3 below, and `writeup/tests/test_digest.py` fails when they drift.
- **The procedure an agent runs:** the `deliberate` skill (`.claude/skills/deliberate/SKILL.md`).

Written 29 October 2026.

---

## 0. The author's words

> **A6 (29 Oct 2026):** "we may need a tier system for questions if we ever make parts of it more automatic. plenty of tokens should be burned arriving at the most likely correct conclusions before every coming to me with any questions"

> **A7 (29 Oct 2026):** "Working through the questions should be somehow encoded in the process. There needs to be a system of escalation where only the important stuff reaches me."

Two consequences follow.

- **A6: effort before asking.** No agent asks the author anything until three independent solvers and an adjudicator have tried to answer it from the repo. Deliberation tokens are cheap; the author's attention is not.
- **A7: a filter, not a courtesy.** The filter is mechanical. `digest.py` recomputes every tier and importance from the thresholds below, can only raise what the adjudicator declared, and writes the only text the author is sent.

---

## 1. Scope

**What counts as a question.** Anything an agent would otherwise ask the author:

- lock conflicts, flipped and uncertain claims (§8.5);
- ambiguous chat edits (§8.5 step 1);
- `% CONFLICT` comments;
- the open questions of §8.9;
- any choice an agent cannot settle within its task.

**The rule.** No agent asks the author directly. It raises the item in `queue.jsonl` (§9) and runs the `deliberate` skill, or leaves the item for the next round. Work that does not depend on the answer continues meanwhile.

**Not questions:**

- **Answering the author.** Replying to a question the author asked in chat is not raising one.
- **Clear instructions.** An unambiguous chat instruction is applied, locked and pushed without a question (§8.6).
- **Explicit approval.** An instruction from the author naming a specific action is that action's approval, hard triggers included. For example, "run it" approves that run. No round is needed for it.

---

## 2. Roles

| Role | Who | Does | Never |
|---|---|---|---|
| **Raiser** | Any agent with a question | Appends one queue entry: id, question, context paths, stakes (`SCHEMA.md` §1) | Asks the author; answers its own question by guessing |
| **Router** | The coordinating agent, applying the T0 test | Settles an item mechanically when a single cited source decides it with no interpretation. That source is a lock or decision, a generated macro or table value, a pre-registered `RULE_*.txt`, a check script's output, or this protocol. Writes a router adjudication (T0) | Settles anything needing judgment, or anything with a hard trigger; those go to the solvers |
| **Solver: EVIDENCE** | Fresh agent, blind to the other solvers | Answers from what the repo computes and tests: `results/` (SUMMARYs, `RULE_*.txt`), `writeup/generated/` macros and tables, generators (`writeup/make_*.py`), `writeup/tests/` and check scripts, run logs, and JFEc norms (author guidelines, page budget, replication policy, accepted-paper practice) | Edits, commits, pushes, compute, contacting anyone |
| **Solver: AUTHOR RECORD** | Fresh agent, blind | Answers what the author would most likely say. Sources: decisions D1-D20 in `writeup/INTEGRATION_PLAN_2026-10-26.md`; `writeup/decisions.yaml` once it exists; dated `(user\|author, <date>: ...)` comments in the paper sources; commit messages; the author's own chat messages in the Claude Code transcripts (`C:\Users\james\.claude\projects\C--Users-james-CC-Allowed-harxhar-0dte-professor\*.jsonl`); the A-decisions box of the report; earlier outcomes and vetoes in `log.jsonl` | Same |
| **Solver: REFEREE** | Fresh agent, blind | Judges correctness and defensibility to a JFEc referee and editor and to the realized-volatility literature (HAR, QLIKE and robust losses, DM and MCS inference, intraday RV). Sources include `writeup/references.bib`, `writeup/research_notes/`, and published JFEc precedents (read-only web) | Same |
| **Adjudicator** | Fresh agent that was not a solver this round | Compares the blind answers, tries to refute the majority answer against the repo, applies the precedence rule (§5), and assigns kind, tier, confidence, importance, consequence, hard triggers and actions. Writes `<id>.adjudication.json` | Same; also never softens a solver's T3 without a stated, checked reason |
| **Coordinator** | The agent running the `deliberate` skill | Queues items, routes, spawns solvers and the adjudicator, runs `digest.py`, sends the digest, applies outcomes, records vetoes and answers | Sends the author anything other than `digest.md`'s content; applies a hard-trigger action without the author's answer |
| **Author** | The author | Answers T3 items in one line each; vetoes any outcome in one line | (is never asked a question that skipped deliberation) |
| **Outside reviewer** (optional) | A different-model reviewer at milestones (§11) | Adds a fourth, independent answer (`reviewer` lens) | Replaces one of the three fixed lenses |

**Process questions.** For questions about tooling, sync, Overleaf and process rather than paper content, the lenses keep their names and read as follows:

- **EVIDENCE:** the mechanics. Code, tests, logs, vendor documentation and vendor source code.
- **AUTHOR RECORD:** unchanged.
- **REFEREE:** outside defensibility. Co-author norms, journal and publisher policy (JFEc, OUP), and what a skeptical outside reader would object to.

**Independence.** Solvers do not read each other's files for the item, and they do not read the adjudication. They may read earlier rounds' outcomes in `log.jsonl` as part of the record.

**Effort (A6).** Each solver spends what it takes to reach the most likely correct answer: it reads the sources rather than inferring them. A solver that answers "ask the author" without having tried has not done its job.

---

## 3. Tiers and thresholds

| Tier | What qualifies (all conditions must hold) | What happens |
|---|---|---|
| **T0 mechanical** | A single cited lock, generated fact or rule decides it, with no interpretation (kind `mechanical`); confidence >= 0.8; no hard trigger | Act and log |
| **T1 evidence-resolvable** | Data, results, the literature or the author's recorded decisions settle it. **All solvers agree** (`unanimous`), the adjudicator's **refutation failed**, confidence **>= 0.8**, no unresolved lock conflict, no hard trigger | Act, keep it reversible, log it, record it in `decisions.yaml` (source `inferred`, with evidence) |
| **T2 inferred preference** | **Either** a matter of taste that the author's dated quotes and past choices imply (kind `taste`, confidence >= 0.6), **or** confidence **0.6 <= c < 0.8**, **or** a majority answer that survived adjudication or an answer the refutation corrected. No split, no unresolved lock conflict, no hard trigger | Apply provisionally, marked `inferred`; the author can veto in one line |
| **T3 author judgment** | **Any** of: solvers **still split after adjudication**; confidence **< 0.6**; the author's own locks **conflict with no precedence** (§5); a **hard trigger** (§4); a matter of taste the record does not settle (the adjudicator declares T3) | Ask once, batched, with a recommendation, confidence, alternatives, consequences and a default (§7) |

**Mechanical floors.** `digest.py` computes the effective tier as the **highest** of the tier the adjudicator declared and each floor below. It never lowers a declared tier.

| Condition | Floor |
|---|---|
| kind `evidence` | T1 |
| kind `taste` | T2 |
| agreement `majority`, or refutation `succeeded` | T2 |
| confidence < 0.8 | T2 |
| confidence < 0.6 | T3 |
| agreement `split` | T3 |
| lock conflict `unresolved` | T3 |
| any hard trigger, declared or implied by an action | T3 |

**Machine constants.** `digest.py` must hold the same values (checked by `writeup/tests/test_digest.py`):

```text
T1_MIN_CONFIDENCE = 0.8
T2_MIN_CONFIDENCE = 0.6
MAX_T3_PER_DIGEST = 3
HARD_TRIGGERS = compute, outward, config, delete_data, spend, irreversible
HIGH_TOUCHES = abstract, introduction, headline_claim, main_text_number, main_text_table, design, collaborators
LOW_TOUCHES = wording, appendix, process
```

**Confidence** is the adjudicator's probability that the final answer is the one the author would give on reading the evidence. It is not the share of solvers that agreed. A unanimous answer resting on one unverified fact is not 0.9.

**Router items** can only be T0. If a router item would land above T0 (a hard trigger, confidence below 0.8), `digest.py` rejects it, and the item goes to the solvers.

---

## 4. Hard triggers: always T3, whatever the confidence

| Code | Trigger | Implied by action kind |
|---|---|---|
| `compute` | Launching compute beyond runs the author already approved: cluster jobs, model fits, new local analysis arms | `run_compute` without `approved_by` |
| `outward` | Anything outward-facing: messages to collaborators, Overleaf comments, accepting or rejecting Overleaf suggestions, submission, disclosure text, publishing | `outward` |
| `config` | Changes to `CLAUDE.md`, settings or permissions | `edit_config` |
| `delete_data` | Deleting data | `delete_data` |
| `spend` | Spending money | `spend` |
| `irreversible` | Any other action a `git revert` cannot undo (§8.10: "irreversible or outward-facing") | (declared only) |

**Rules:**

1. **Union.** The adjudicator declares triggers in `hard_trigger`, and `digest.py` adds every trigger implied by the item's `actions`. The union decides.
2. **Approved compute.** A `run_compute` action is exempt only when `approved_by` cites the author's approval of that specific run (dated quote or commit), for example `author 2026-10-28 17:04: "go"`. Approval of a different or smaller run does not carry over.
   - **What counts as compute.** Any run beyond a quick read-only script: cluster jobs, model fits, and new local analysis arms, even from existing forecasts.
   - **Basis:** the author's "Don't launch anything new without asking" (28 Oct 2026, said of the cluster runs; quoted in `2026-10-29/adjudication.md`). This protocol reads it as covering new local arms too.
3. **Defaults are inaction.** The default of a hard-trigger item is always inaction: nothing is sent, launched, deleted, spent or reconfigured until the author answers.
4. **The routine Overleaf push is not a trigger.** `CLAUDE.md` requires a push to Overleaf after every paper update (author's standing order, 22 Oct). That push is part of the normal editing rules. Overleaf comments, suggestions and messages to collaborators remain `outward`.

---

## 5. Precedence

Applied by the adjudicator and by the AUTHOR RECORD solver.

1. **A later dated author decision supersedes an earlier one on the same scope.** Read decisions by scope first. Decisions that govern different things compose and do not conflict. Example: D1 sets the target, D2 the benchmark, D18 the question.
2. **The author's words outrank everything else.** That includes collaborators' Overleaf edits (until the author adopts them, A5), `inferred` records, agent proposals and the report's inferences.
3. **Inferred records never supersede an author's words.** A T1 or T2 outcome cannot override a dated author decision, however much later it is. A veto is an author decision.
4. **Dating.** Transcript timestamps are UTC. Convert them to the author's local time (America/Los_Angeles) before comparing them with dated comments and commits. Within one day, order by timestamp where one exists.
5. **No precedence.** Precedence fails when:
   - the two decisions share a date with no recoverable order;
   - either is undated;
   - the later one's scope is ambiguous, so it is unclear whether it covers the earlier one's case.

   In these cases `lock_conflict` is `unresolved`, and the item is T3.

---

## 6. Importance (the escalation filter)

| Importance | Touches (codes) |
|---|---|
| **HIGH** | `abstract`, `introduction`, `headline_claim` (headline claims wherever stated, conclusion included), `main_text_number`, `main_text_table` (tables and figures), `design` (sample, benchmark, span, training design), `collaborators` |
| **LOW** | `wording`, `appendix` (appendix-only), `process` (process internals: tooling, records, this protocol's operation) |

- **Mechanical rule.** An item is HIGH if the adjudicator says so or if any of its `touches` is a HIGH code. `digest.py` applies this rule.
- **Uncertain cases.** When unsure, choose HIGH: it costs one line in the digest, while a hidden HIGH item costs the author a surprise.

**Consequence** (1-5) orders T3 items within an importance class:

| Score | Meaning |
|---|---|
| 5 | Changes the paper's story or headline, or is irreversible or outward-facing |
| 4 | Changes a main-text result, table or design parameter |
| 3 | Changes a section's argument, or several places |
| 2 | A local change in one place |
| 1 | Cosmetic, or process-only |

---

## 7. What the author sees

`digest.py` writes `<round>/digest.md`, and the coordinator sends **only its content**. It holds:

1. **T3 questions, at most three per digest.** They are ordered HIGH first, then by consequence, then oldest round first. Each carries:
   - the question;
   - the recommendation;
   - what it changes (where-used consequences);
   - the default;
   - the confidence and alternatives;
   - why it comes to the author;
   - a one-line answer format.

   Further T3 items stay `queued`, and `digest.py` carries them into the next digest without re-deliberating them.
2. **HIGH T2 items, one line each.** Each is applied provisionally unless vetoed.
3. **One count line.** It counts T0, T1 and LOW T2 items and gives the path of `log.jsonl`. Those items are only logged.
4. **A reminder line** naming T3 items from earlier digests that are still unanswered (ids only).

Nothing else. No preamble, no summary of the deliberation and no solver output reach the author unless the author asks for them.

---

## 8. Answers, defaults and vetoes

**Answering a T3.**

- **Format.** One line: `<ID> yes` takes the recommendation; `<ID> <words>` gives a different answer.
- **Recording.** The coordinator records the words verbatim and dated as an author decision: a `log.jsonl` line with source `author` and status `applied`, plus `decisions.yaml` with source `chat` once that file exists.
- **Applying.** It then applies the answer through the normal editing rules.

**Defaults.**

- **While a T3 is unanswered,** its default governs agent work. Agents may build on it, reversibly and unlocked, and list what depends on it.
- **Hard-trigger items** default to inaction (§4).
- **Unanswered T3s** are not re-asked in full. They appear on the reminder line until answered. They are re-deliberated (under a new id that names the old one) only if new evidence changes the recommendation.

**Veto.**

- **Format.** One line in chat: `veto <ID>`, optionally followed by words. It applies to any applied or provisional item.
- **What the coordinator does:**
  1. Reverts the item's commit. Each applied item gets its own commit, with the id at the start of the subject, so a veto is one `git revert`.
  2. Appends a `log.jsonl` line with status `vetoed` and the author's message verbatim in `author_words`.
  3. Marks the `inferred` entry superseded in `decisions.yaml` and records the veto as an author decision, once that file exists.
  4. Pushes per `CLAUDE.md` if the paper changed.
- **No comeback.** A vetoed answer can never come back as T1 or T2. Only new words from the author reopen it.
- **Open conflicts.** If the revert leaves a conflict open, the coordinator raises a new item that cites the veto.
- **Vetoing a T0** disputes the lock it followed from. It is handled as an author edit of that lock (A3).

---

## 9. Records

| File | Written by | Content |
|---|---|---|
| `writeup/deliberation/queue.jsonl` | Raisers, the coordinator | Raised items, append-only (`SCHEMA.md` §1). An item is open while it has no line in `log.jsonl` |
| `writeup/deliberation/<round>/<id>.<lens>.md` | Solvers | One file per item and lens: `evidence`, `author_record`, `referee` (and `reviewer` if used) |
| `writeup/deliberation/<round>/<id>.adjudication.json` | Router or adjudicator | `SCHEMA.md` §2 |
| `writeup/deliberation/<round>/digest.md` | `digest.py` | The only text sent to the author |
| `writeup/deliberation/log.jsonl` | `digest.py`; the coordinator for author answers, vetoes and failed applications | Append-only, one line per event (`SCHEMA.md` §3). An item's current state is its last line |
| `writeup/decisions.yaml` (once it exists) | The coordinator | T0-T2 outcomes with source `inferred` and their evidence; author answers and vetoes as author decisions |

**Round folders.** `<round>` is the deliberation date, `YYYY-MM-DD`. A second round on the same day uses `YYYY-MM-DD-2`.

**Credentials.** No record ever holds a credential. Mask `olp_*` tokens wherever a transcript or script output is quoted.

---

## 10. A round, in order

The `deliberate` skill holds the full procedure and the prompt templates.

1. **Queue.** Append the items to `queue.jsonl`.
2. **Route.** T0 items get a router adjudication and are done.
3. **Solve.** Three blind solvers per remaining item, in parallel. Each writes `<id>.<lens>.md`.
4. **Adjudicate.** One adjudicator writes `<id>.adjudication.json` and validates it with `digest.py --dry-run`.
5. **Digest.** Run `python writeup/deliberation/digest.py writeup/deliberation/<round>`.
6. **Send.** Send the author the content of `digest.md`, and nothing else.
7. **Apply.** Apply T0, T1 and T2 actions through the repo's normal editing rules, one commit per item. Record answers and vetoes as they arrive.

---

## 11. Known limits

**Shared blind spots.** The three solvers usually run on one model and can share its blind spots, so agreement among them is weaker evidence than agreement among independent people.

- **Milestones.** The protocol allows a different-model reviewer, as a fourth `reviewer` answer, at the end of a layer and before submission (§8.6: "in another model family at milestones").
- **A paid outside vendor** is a `spend` trigger. It needs the author's answer to §8.9 Q5 first.
- **A cheaper partial step:** inside Claude Code, one solver may run on a different Claude model. That is a different model, not a different family.

**Tokens on trivial questions.** The router settles mechanical items before any solver runs. The thresholds are inference (I, A6) and untested.

**Calibration.** The veto rate is the calibration signal.

- **Any vetoed T1** prompts a review of what the solvers missed.
- **A T2 veto rate above roughly one in five** over several rounds means `T1_MIN_CONFIDENCE` and `T2_MIN_CONFIDENCE` should rise. Changing them is the author's call (§12).

**No router script yet.** Until `locks.py` and `claims/claims.yml` exist (Phase 2 of §8.7), the coordinator applies the T0 test by hand. Its test: could a script with access to the one cited file decide this? If not, the item goes to the solvers.

**The first live round (29 Oct) predates this file.** It used five solvers across two lens sets (content items; process items Q10-Q13). Its adjudication and digest were written by hand as Markdown, with no item ids or JSON. `digest.py` refuses a folder without adjudication JSON, so it never overwrites that round's `digest.md`. Its outcomes enter `log.jsonl` only if they are re-expressed as items under this schema.

---

## 12. Changing this protocol

The thresholds, hard triggers and importance codes are part of the author's process, so they change only on the author's word. `PROTOCOL.md` §3 and `digest.py`'s constants change in the same commit; the test enforces it. A proposal to change them is raised as an item like any other; the adjudicator declares it T3, because only the author may decide it.
