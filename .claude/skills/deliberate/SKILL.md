---
name: deliberate
description: "Runs the author's question-escalation protocol for the harxhar paper (writeup/deliberation/PROTOCOL.md). Agents must use it instead of asking the author directly: every question, lock conflict, ambiguous edit, flipped claim or open item is queued, worked through by three blind solvers and an adjudicator, and filtered by digest.py. Only the digest reaches the author: at most three T3 questions and one line per HIGH T2 outcome. Use it whenever you are about to ask the author something, or when writeup/deliberation/queue.jsonl has open items. Arguments: optional new questions to raise, or queue ids to restrict the round to."
---

# deliberate: work through questions before the author sees them

The author's words: "plenty of tokens should be burned arriving at the most likely correct conclusions before every coming to me with any questions" (A6, 29 Oct 2026). And: "There needs to be a system of escalation where only the important stuff reaches me." (A7, 29 Oct 2026).

**Sources of rules.**

- **Rules:** `writeup/deliberation/PROTOCOL.md`. Read it before the first round in a session; on any conflict it wins over this file.
- **Shapes:** `writeup/deliberation/SCHEMA.md`.
- **Mechanical filter:** `writeup/deliberation/digest.py`.

**Paths.** Repo root: `C:\Users\james\CC Allowed\harxhar-0dte-professor` (Git Bash: `/c/Users/james/CC Allowed/harxhar-0dte-professor`). `python` below means any Python 3.10+, for example `/c/Users/james/miniconda3/python.exe`.

## Coordinator limits (all steps)

- **Never ask the author directly.** Send the author nothing but the content of `digest.md` (step 6), and answer only what the author asks.
- **Hard triggers wait for the author.** Never carry out an action with a hard trigger (PROTOCOL.md §4) without the author's one-line answer: compute beyond approved runs, anything outward-facing, CLAUDE.md, settings or permissions, deleting data, spending money, anything a git revert cannot undo.
- **Solvers and adjudicator are read-only.** They edit nothing outside their own output files: no commits, no pushes, no compute.
- **Leave other rounds alone.** Never write in another round's folder. `writeup/deliberation/2026-10-29/` is the hand-written first round.
- **Credentials.** Never print one. Mask `olp_*` tokens.
- **Subagents.** If you are a subagent without the Agent tool, do step 1 only and return the queued ids to your caller; the main session runs the round. When you return a digest to a caller instead of the author, say "relay verbatim".

## Step 1: queue

1. **Open items.** Read `writeup/deliberation/queue.jsonl` and `writeup/deliberation/log.jsonl`. An item is open when its id has no line in the log.
2. **New questions first.** For each question in the caller's arguments, write a queue entry (SCHEMA.md §1) and append it **before** anything else runs, so the item survives a crash. Fill `context` with the paths you already know, and `stakes` with one line on what changes and where.
3. **Round folder.** The round is `writeup/deliberation/<today>/`. Use `<today>-2` if today's folder already holds a digested round. Create the folder.
4. **Scope.** If the caller named ids, the round covers those; otherwise every open item.
5. **Nothing open:** say so in one line and stop.

```bash
python - "<question>" "<raised_by>" "<stakes>" "<context path 1>" "<context path 2>" <<'EOF'
import json, sys, datetime
from pathlib import Path
q = Path("writeup/deliberation/queue.jsonl")
today = datetime.date.today().isoformat()
used = [int(json.loads(l)["id"].rsplit("-", 1)[1]) for l in q.read_text(encoding="utf-8").splitlines()
        if l.strip() and json.loads(l)["id"].startswith(f"Q-{today}-")]
item = {"id": f"Q-{today}-{(max(used) + 1 if used else 1):02d}", "raised_by": sys.argv[2], "date": today,
        "question": sys.argv[1], "context": sys.argv[4:], "stakes": sys.argv[3]}
with q.open("a", encoding="utf-8", newline="\n") as fh:
    fh.write(json.dumps(item, ensure_ascii=False) + "\n")
print(item["id"])
EOF
```

## Step 2: router (T0)

For each item, apply the T0 test.

**The test.** One cited source decides the answer with no interpretation, and no hard trigger is involved. The source may be:

- a decision (D1-D20 in `writeup/INTEGRATION_PLAN_2026-10-26.md`, or `writeup/decisions.yaml` once it exists);
- a generated macro or table value in `writeup/generated/`;
- a pre-registered `RULE_*.txt`;
- a check script's output;
- PROTOCOL.md itself.

Ask: could a script with that one file decide it?

- **If yes:** write a router adjudication (template below), and the item skips the solvers.
- **If there is any doubt:** send it to the solvers. A6 prefers spent tokens to a wrong T0.

## Step 3: solvers (three per item, blind, in parallel)

**Spawning.** For each remaining item, spawn three agents in **one message** with the Agent tool, using `subagent_type: "general-purpose"` (they need Read, Grep, Bash, Write and read-only web). Use one agent per lens: `evidence`, `author_record`, `referee`.

- **Several items:** put all their solver calls in the same message, or in batches of about twelve.
- **Related items** that share their sources may go to one solver per lens, as on 29 Oct. That solver still writes one file per item, and answers each item on its own.
- **Blindness:** never paste one solver's output into another's prompt.

**Afterwards.**

- **Missing files.** Check that every `<id>.<lens>.md` exists, and re-run any solver that failed or wrote nothing.
- **Stray edits.** Compare `git status --porcelain` from before and after. If a solver changed anything outside the round folder, confirm with `git diff` that the change is the solver's (other sessions edit this repo too), restore it, and tell the adjudicator.

**Solver prompt template.** Fill every `{...}`, and paste the matching lens block from the next section.

```text
You are the {LENS_NAME} solver for item {ID}, in a deliberation under the author's question protocol.

Repo: C:\Users\james\CC Allowed\harxhar-0dte-professor (Git Bash: /c/Users/james/CC Allowed/harxhar-0dte-professor),
branch {BRANCH}. Today is {DATE}. Read first: writeup/deliberation/PROTOCOL.md (sections 2-6) and the "Author
decisions" box (A1 onward) at the top of writeup/reports/AI layered paper writing system.md.

The item (writeup/deliberation/queue.jsonl):
- id: {ID}
- question: {QUESTION}
- raised by {RAISED_BY} on {RAISED_DATE}
- start from: {CONTEXT}
- stakes, as the raiser saw them: {STAKES}

Your lens:
{LENS_BLOCK}

The author's instruction for this work (A6): "plenty of tokens should be burned arriving at the most likely correct
conclusions before every coming to me with any questions". Reach the most likely correct answer yourself. Read the
sources; do not infer what a file says from its name or from a summary. Propose T3 only when the thresholds below
require it, never as a way to avoid deciding.

Independence: two other solvers work this item with other lenses. Do not open writeup/deliberation/{ROUND}/{ID}.*.md
other than your own, any *.adjudication.json, or any digest.md of this round. Earlier outcomes in
writeup/deliberation/log.jsonl are part of the record; read them.

Rules: write only your output file. No other edits, no commits, no git or Overleaf push or pull, no compute (no
cluster jobs, no model fits, nothing that runs over a minute; small read-only parsing scripts in your scratchpad are
fine), no messages or comments to anyone, no settings changes. Never print credentials; mask any olp_* token.
Read-only web search and fetch are allowed.

Tiers (PROTOCOL.md section 3). You propose one; the adjudicator decides:
- T0: one cited lock, generated fact or rule decides it, with no interpretation.
- T1: data, results, the literature or the author's recorded decisions settle it, at confidence >= 0.8.
- T2: a matter of taste that the author's dated quotes and past choices imply, or confidence from 0.6 to below 0.8.
- T3: confidence < 0.6; the author's own decisions conflict and no later one settles it (PROTOCOL.md section 5); a
  matter of taste the record does not settle; or a hard trigger: compute beyond runs the author approved, anything
  outward-facing (collaborators, Overleaf comments or suggestions, submission, disclosure, publishing), CLAUDE.md,
  settings or permissions, deleting data, spending money, or anything a git revert cannot undo.
Importance codes (PROTOCOL.md section 6): HIGH = abstract, introduction, headline_claim, main_text_number,
main_text_table, design, collaborators. LOW = wording, appendix, process.

Write writeup/deliberation/{ROUND}/{ID}.{LENS}.md with these sections, in order (SCHEMA.md section 4):
  # {ID} solver {LENS_NAME} ({DATE})
  One line: you read no other solver's file and made no edit, commit or run. Then the sources you read.
  ## Answer          a few sentences, complete enough to act on
  ## Confidence      0-1, and one line on what would change it
  ## Tier            T0-T3, the threshold it meets, any hard trigger, the importance codes it touches
  ## Evidence        file:line, commit <sha>, author <local date time>: "<verbatim words>", URLs; decisive first.
                     Transcript times are UTC: convert to America/Los_Angeles. Quote the author; never paraphrase.
  ## If adopted      where-used consequences: every file, generator, macro, table and section, and what changes
  ## Alternatives    options considered, and why each loses
  ## T3 question     only if your tier is T3: one question the author can answer in one line, your recommendation,
                     and the default if the author never answers (for a hard trigger, the default is inaction)

Reply with one line: short answer | confidence | tier | importance codes.
```

### Lens blocks

**EVIDENCE** (`{LENS}` = `evidence`, `{LENS_NAME}` = `EVIDENCE`):

```text
EVIDENCE: what the repo computes and tests. Read the results (results/**/SUMMARY*.md, ADDENDUM*.md and the
pre-registered RULE_*.txt, which bind and cannot be revised after the fact), the generated macros and tables
(writeup/generated/), the generators that write them (writeup/make_*.py, experiments/), the tests and checks
(writeup/tests/), and the git log for which run produced what. For a number, give the macro name and its value. For
JFEc norms (page budget, abstract limit, replication policy, what accepted papers report), cite the journal's
instructions or an accepted paper. For a process question (sync, Overleaf, tooling), the evidence is the code, its
tests and logs, and vendor documentation or source code.
```

**AUTHOR RECORD** (`{LENS}` = `author_record`, `{LENS_NAME}` = `AUTHOR RECORD`):

```text
AUTHOR RECORD: what the author has said and chosen, so that your answer is the one he would most likely give. Read:
the "Author decisions" box (A1 onward) of the report; D1-D20 in writeup/INTEGRATION_PLAN_2026-10-26.md, and
writeup/decisions.yaml if it exists; writeup/deliberation/log.jsonl (author answers and vetoes there are author
decisions); dated comments of the form "(user|author, <date>: ...)" in writeup/paper_rehaul.tex, writeup/rehaul/*.tex
and writeup/beats/; commit messages (git log --format='%h %ad%n%B' --date=iso); and the author's own chat messages in
C:\Users\james\.claude\projects\C--Users-james-CC-Allowed-harxhar-0dte-professor\*.jsonl. The transcripts are large:
stream-parse them with a small Python script and keep only what the author typed, including messages queued while an
agent worked. Skip tool results, agent hand-backs, system reminders and command expansions. Where a reply answers an
agent's question, read the question too. Build a dated precedence timeline of the decisions this item turns on (local
time), then apply PROTOCOL.md section 5: a later dated author decision supersedes an earlier one on the same scope;
decisions on different scopes compose; inferred records never outrank the author's words.
```

**REFEREE** (`{LENS}` = `referee`, `{LENS_NAME}` = `REFEREE`):

```text
REFEREE: a Journal of Financial Econometrics referee and editor. Is the answer correct, and can the paper defend it to
a referee who knows the realized-volatility forecasting literature (HAR and its extensions, QLIKE and robust losses,
Diebold-Mariano and model-confidence-set inference, intraday and overnight RV, machine-learning RV forecasts)? Check
claims against the paper's own generated numbers and against the literature: writeup/references.bib,
writeup/research_notes/, and published papers (read-only web). For each option, name the objection a referee would
raise and whether the paper can answer it. For a process question, read the lens as outside defensibility: co-author
norms, journal and publisher policy (JFEc; OUP's AI-disclosure rules), and what a skeptical outside reader would
object to.
```

**Correlated errors.** The three solvers normally share one model, and with it its blind spots (PROTOCOL.md §11).

- **Milestones.** At the end of a layer and before submission, add a fourth solver with lens `reviewer` from a different model family.
- **Paid vendors.** A paid outside vendor is a `spend` trigger and waits for the author's answer to §8.9 Q5.
- **A cheaper partial step.** Run one of the three solvers with the Agent tool's `model` parameter set to a different Claude model. That is a different model, not a different family.

## Step 4: adjudicator

**Pipelined (author, 29 Oct 2026: "pipeline yes").** Do not wait for the whole round. As soon as one item's three solver files exist (or, for related items that shared solvers, all of that group's files), check that item's files and stray edits as in step 3's "Afterwards", then spawn its adjudicator at once while other items' solvers are still running. Group related items that shared solvers under one adjudicator. Each adjudicator validates with `--dry-run` and ignores errors in items that are not its own and not yet adjudicated. Run step 5 (the digest) only when every item of the round has its adjudication.

**Spawning.** Spawn one fresh `general-purpose` agent that was not a solver this round, per item or per related group.

**Validation.** When it finishes, run `python writeup/deliberation/digest.py writeup/deliberation/<round> --dry-run`.

- **Exit 1:** send the errors back to the same agent (SendMessage) until the run exits 0.
- **Formatting fixes:** you may fix pure formatting errors yourself, such as a misspelt key or a missing `[]`.
- **Substance:** never change a tier, a confidence, an answer or a trigger yourself.

**Adjudicator prompt template.**

```text
You are the adjudicator for deliberation round {ROUND} under the author's question protocol. You were not a solver.

Repo: C:\Users\james\CC Allowed\harxhar-0dte-professor, branch {BRANCH}. Today is {DATE}. Read first, in full:
writeup/deliberation/PROTOCOL.md and writeup/deliberation/SCHEMA.md (section 2).

Items (from writeup/deliberation/queue.jsonl):
{for each item: id, question, raised_by, date, context, stakes}
Solver files: writeup/deliberation/{ROUND}/<id>.evidence.md, <id>.author_record.md, <id>.referee.md
(and <id>.reviewer.md where present).
{Any stray-edit note from the coordinator.}

For each item:
1. Read the solver files in full. Summarize each answer in one or two sentences, with its confidence and tier.
2. Compare them on substance, not wording. Where they differ, find the decisive point.
3. Refute. Take the majority answer (or the unanimous one) and try hard to break it against the repo:
   - Re-open the cited lines and check that they say what the solver claims.
   - Search for a later dated author decision the solvers missed: log.jsonl vetoes and answers, the newest commits,
     the newest transcript messages, and dated comments.
   - Check every number against writeup/generated/.
   - Look for a lock, forbid or named exception the answer would break.
   Record what you checked in refutation.note. If the refutation succeeds, change the final answer; the item can
   then be T2 at best.
4. Precedence (PROTOCOL.md section 5). Where the author's decisions conflict, say which governs and why in
   precedence_note. If none governs, set lock_conflict to "unresolved".
5. Classify:
   - kind and agreement.
   - confidence: your probability that the final answer is the one the author would choose on the evidence; it is
     not the vote share.
   - the declared tier (section 3).
   - touches and importance (section 6; when unsure, HIGH).
   - consequence, 1-5.
   - hard triggers (section 4). Make every action's kind honest: a run is run_compute, and a message, comment or
     suggestion decision on Overleaf is outward. Set approved_by only when you can quote the author approving that
     very run.
6. Write writeup/deliberation/{ROUND}/<id>.adjudication.json (SCHEMA.md section 2). The author reads
   question_for_author, recommendation, consequences, default, alternatives and answer_short exactly as you write
   them, so use plain language with no internal jargon.
   - For T3: give one question answerable in one line, t3_reason in plain words, and a default. The default of a
     hard-trigger item is inaction.

Do not lower a solver's T3 without a checked reason in rationale. Do not invent a middle answer that no solver gave
unless your refutation forces it.

Then validate: python writeup/deliberation/digest.py writeup/deliberation/{ROUND} --dry-run, and fix every error until
it exits 0. Never run it without --dry-run.

Rules: write only the adjudication JSON files. No other edits, no commits, no pushes, no compute, no messages. Never
print credentials.

Reply with a table: id | short final answer | declared tier | confidence | importance | hard triggers.
```

**Router adjudication template (step 2).** Write it to `writeup/deliberation/<round>/<id>.adjudication.json`:

```json
{
  "id": "Q-2026-10-30-02",
  "date": "2026-10-30",
  "question": "<the queued question>",
  "adjudicated_by": "router",
  "kind": "mechanical",
  "lock_conflict": "none",
  "final_answer": "<the answer the source dictates>",
  "answer_short": "<one line>",
  "tier": "T0",
  "confidence": 1.0,
  "importance": "LOW",
  "touches": ["appendix"],
  "consequence": 1,
  "hard_trigger": [],
  "evidence": ["<the one decisive source, e.g. writeup/INTEGRATION_PLAN_2026-10-26.md:51>"],
  "actions": [{"kind": "edit_text", "target": "<path>", "detail": "<what changes>"}],
  "consequences": "<where-used consequences>",
  "recommendation": "<the action>",
  "default": "<the action>"
}
```

## Step 5: digest

Run `python writeup/deliberation/digest.py writeup/deliberation/<round>`. It must exit 0.

**What it does.**

- Re-validates every adjudication.
- Raises tiers and importance mechanically (hard triggers, thresholds).
- Picks at most three T3 questions (HIGH first, then by consequence) and carries earlier queued T3s forward.
- Writes `<round>/digest.md`.
- Appends the round to `log.jsonl`. Re-running adds nothing.

## Step 6: send

**What to send.** Send the author the content of `digest.md`, verbatim, as the whole message: no preamble, no summary of the deliberation.

**Quiet rounds.** A round with no questions and no provisional lines yields a three-line digest. Send it anyway, so the author knows the round ran and where the log is.

## Step 7: apply and record

**Order.** Apply T0, then T1, then T2 (HIGH and LOW alike), using the logged actions. Apply nothing for T3 items. Until a T3 is answered its default governs further work, reversibly; for a hard-trigger item the default is inaction.

**Everything goes through the repo's normal editing rules.**

- **§8.6 of the report:**
  - never type a digit (numbers come from macros);
  - text inside `\auth{...}` is never touched;
  - verdict words come only from predicates;
  - a decision's words are never edited.
- **CLAUDE.md, before a commit:** run `ruff check --fix`, `ruff format` and `mypy --ignore-missing-imports` in parallel on staged `.py` files. Check `git remote -v` before any push.
- **CLAUDE.md, after a paper update:** run `bash writeup/overleaf_sync/overleaf_git.sh push --check`, then `git push` the map.json commit it makes. Never print the token.
  - **Refused push.** CLAUDE.md's rule for a refused push ("run pull --dry-run, report, and ask") stands until the author changes it. Changing it is a `config` item.
- **One commit per item.** The subject starts with the item id (`Q-2026-10-30-03: <answer_short>`) and ends with the session's attribution trailers. That way a veto is one `git revert`.
- **Decision record.** When `writeup/decisions.yaml` exists, record each T0-T2 outcome there with source `inferred`, the item id and its evidence. `digest.py` prints the ids to record.

**Failed applications.** If an action fails a check (a lock check, the build, a refused push), do not force it.

1. Undo it.
2. Append a log line with `source: "agent"`, `status: "queued"` and a `note`.
3. Re-raise the item in `queue.jsonl` with `related: [<old id>]`.

**When the author replies.**

- **`<ID> yes` or `<ID> <words>`:**
  1. Append a log line with `source: "author"`, `status: "applied"` and `author_words` (verbatim).
  2. Record it as an author decision (in `decisions.yaml` with source `chat`, once that file exists).
  3. Apply it, commit and push as above.
  4. Reply with one short line: what changed, and the commit.
- **`veto <ID> [words]`:**
  1. `git revert` the item's commit.
  2. Append a log line with `source: "author"`, `status: "vetoed"`, `author_words` (verbatim) and `commit` (the revert).
  3. Mark the `inferred` entry superseded and add the veto as an author decision.
  4. Push if the paper changed.
  5. If the revert leaves a conflict open, raise a new item citing the veto. A vetoed answer never returns as T1 or T2.

**Appending an event line.** Copy the core fields from the item's last log line:

```bash
python - "<ID>" "<status>" "<author words or note>" "<commit sha>" "<author|agent>" <<'EOF'
import json, sys, datetime
from pathlib import Path
log = Path("writeup/deliberation/log.jsonl")
last = [json.loads(l) for l in log.read_text(encoding="utf-8").splitlines() if l.strip()
        and json.loads(l)["id"] == sys.argv[1]][-1]
keep = ("id", "round", "question", "answer", "tier", "confidence", "importance", "evidence", "actions", "adjudication")
line = {k: last[k] for k in keep if k in last}
line.update({"date": datetime.date.today().isoformat(), "source": sys.argv[5], "status": sys.argv[2],
             "author_words" if sys.argv[5] == "author" else "note": sys.argv[3], "commit": sys.argv[4]})
with log.open("a", encoding="utf-8", newline="\n") as fh:
    fh.write(json.dumps(line, ensure_ascii=False) + "\n")
EOF
```
