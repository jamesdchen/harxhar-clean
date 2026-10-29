# Solver: process questions Q10–Q13, risk-and-mechanics lens (2026-10-29)

Lens: what can go wrong with git, Overleaf and the existing sync code, and which answer is safest and simplest.
Independent of the other solver(s); written before seeing their work. No edits, commits, pushes or pulls were made.

Read: the report `writeup/reports/AI layered paper writing system.md` (A1–A6 box, §2.4, §8.3 C26/C28/C29, §8.5, §8.6,
§8.9, §8.10); the notes `precedent_scholarly_overleaf.md`, `toggleable_layers.md`, `review_interfaces.md`; the sync code
`writeup/overleaf_sync/` (`overleaf_git.example.sh`, `autopush.sh`, `layer_header_fix.py`, `unflatten.py`, `SPEC.md`,
`README.md`); the local Overleaf clone's log (`../harxhar-overleaf`, read-only); Overleaf's docs; and Overleaf's
open-source code (`overleaf/overleaf` on GitHub: git-bridge, project-history, document-updater, web).

---

## 0. Mechanical facts these answers rest on

| # | Fact | Source | Strength |
|---|---|---|---|
| F1 | The git bridge's snapshot is a list of `[content, path]` pairs for docs plus attachments. It has no field for comments or tracked changes, so **git carries neither**. | git-bridge `snapshot/getforversion/SnapshotData.java`, `SnapshotFile.java` (`srcs`/`atts`; `contents = jsonArray.get(0)`, `path = jsonArray.get(1)`) | D (code) |
| F2 | Project-history builds a project snapshot's doc content with `file.getContent({ filterTrackedDeletes: true })`. Tracked deletions are dropped and tracked insertions kept, so **a clone shows every pending suggestion as if accepted**. | project-history `app/js/SnapshotManager.js` `getProjectSnapshot` (route `GET /project/:id/version/:version`) | D for project-history. Inference that the git bridge's web-side module (closed source, Server Pro) uses this route. The older docstore model also keeps tracked deletes outside the doc lines, so both paths give the same visible-text semantics. |
| F3 | A pushed file that replaces an existing doc is applied as a **minimal diff**, not a replacement: `setDoc` diffs the old and new text (`diffAsTextOperation` / `DiffCodec.diffAsShareJsOp`) and applies only the ops. Comments and suggestions on untouched text survive. Those on changed text are lost or displaced. | document-updater `DocumentManager.setDoc` | D (code) |
| F4 | An external write applied through `upsertDoc` is **recorded as tracked changes when track changes is on for the writing user** (`trackChanges = isTrackChangesEnabledForUser(project.track_changes, userId)`, which is true when the setting is "everyone" or on for that user). If that holds for git pushes, every agent push becomes suggestions under the token owner's name, the author's. | web `ProjectEntityUpdateHandler.upsertDoc`, `ProjectHelper.isTrackChangesEnabledForUser` | D for the TPDS path (GitHub/Dropbox sync). Strong inference for the git bridge (same upsert path; its web module is closed). |
| F5 | The bridge rejects a push as **non-fast-forward when the project changed since the clone's last pull** (`OutOfDateException` → `REJECTED_NONFASTFORWARD`). | git-bridge `git/handler/hook/WriteLatexPutHook.java:70-72`, `SnapshotPostExceptionBuilder` ("outOfDate") | D (code) |
| F6 | Commit author of an Overleaf-side commit: "the author of the most recent change will be listed as the author of the commit". Commits are made only when someone pulls. | [Overleaf: Git integration](https://docs.overleaf.com/integrations-and-add-ons/git-integration-and-github-synchronization/git-integration) | D (docs). The report (§8.5 "Editor attribution") cites the GitHub-sync bundling rule instead and concludes "unknown". For the git bridge the editor is the last editor of each batch, exact whenever one person edited between pulls. |
| F7 | "A file rename … becomes a delete and create … causing the deletion of tracked changes and comments." | same page | D (docs) |
| F8 | All 4 Overleaf-side edits from the author's account (`jamesdchen01`, 29 Oct) were **stray keystrokes** typed into `beats_top.tex` line 3: `%% at commit let's03e6513…`, `I n360d45d…`, `doe s0e76d4…`, plus one line split. `autopush.sh` reverted them all. No Overleaf-side commit has touched `main2.tex`. | `git -C ../harxhar-overleaf log` (commits 85500a8, aba8c24, be31a78, ae201a4) | D (repo) |
| F9 | Collaborators are senior: the advisor (Chris Jones, USC Marshall) and coauthors Austin Pollok and Mihai Cucuringu. | `writeup/MEETING_PREP_SEED_2026-08-18.md:9` | D (repo) |
| F10 | claudeleaf logs in by a cached browser session (`~/.claudeleaf`) and drives "Overleaf's live collaboration channel (ShareJS over Socket.IO)". `trackChanges` "requires track changes on the project (a premium feature)". Comments can be created and listed, "but not resolved or deleted". It edits the live document. Its README makes no statement on terms of service or protocol stability (36 stars, 11 commits). | [claudeleaf README](https://github.com/lonetis/claudeleaf) | D (README) |

**Two defects in the current sync code bear on every item below.** Both are T0: fix and log, nobody needs to be asked.

- **D-a: after any real pull of `main2.tex`, `push` deadlocks.** `pull` runs `unflatten.py` and records nothing (`overleaf_git.example.sh:218-237`; `unflatten.py` never writes the map). `push` then compares Overleaf's `main2.tex` with `map.json`'s `flat_text` and dies while they differ (`:117-119`). `--first` dies too when the file exists (`:120-122`). So once a collaborator edits `main2.tex`, no push succeeds until someone hand-edits `map.json` or reverts Overleaf. This path has never run in production: no Overleaf-side commit touched `main2.tex`, and no test drives `overleaf_git.sh`. **Fix:** at pull time, record a watermark: the Overleaf commit SHA and the text consumed, including any held hunks. `push` (and `push-beats`, once layers flow back) then accepts Overleaf content equal to the last flatten *or* to the last pulled text.
- **D-b: `map.json` advances before the push lands.** `flatten.py --map "$MAP"` rewrites the repo's `map.json` (`:134-135`) before `og push` (`:151`). If a collaborator edits while the multi-minute `--check` build runs, the push is rejected (F5), and `og` does not retry on that error. The Overleaf clone keeps an unpushed commit, so the next `pull --ff-only` fails and the loop is stuck. `map.json` meanwhile holds a flat text Overleaf never received. If someone recovers by resetting the clone and pulling, `unflatten` diffs Overleaf against that never-pushed base and **reapplies reversals of the agent's own last edits as "Overleaf edits"**. **Fix:** write the map to a temporary path and move it into place only after the push succeeds. Run `--check` before taking the guard, so checkout → guard → commit → push takes seconds. On a non-fast-forward rejection: reset the clone to origin, pull and unflatten, re-flatten, and retry a bounded number of times.

A third precondition for A5's "pull before every push" in `autopush.sh`: `unflatten.py` writes into the author's working tree and merges into files that may hold an agent's uncommitted edits (SPEC §5 step 4). Auto-pull only when `git status --porcelain -- writeup` is clean. Commit each pull on its own (`Overleaf pull <overleaf-sha>: N applied, M held; editor <commit author>`). That commit is also the provenance used in Q11.

---

## Q10. Collaborator authority

**Answer: keep the default, with five refinements.** A collaborator's Overleaf edit is applied as text and recorded. It locks only on the author's adoption. An edit that would break a lock is held. Do **not** let named coauthors' edits lock directly.

Refinements:

1. **"Held" must not stall the loop.** A held hunk is not applied to the source. It is recorded verbatim (Overleaf SHA, editor, date, the lock it breaks), and the pull's watermark advances past it (D-a). The next push therefore **reverts it on Overleaf**, which is what SPEC §5 already does to number and table conflicts. The text is not lost. It is parked at that spot in the source as a dated comment, `% <date> (overleaf, held: conflicts with D16): <their text>`, using the existing park convention (`unflatten.py:93`). The collaborator sees the comment in `main2.tex` on Overleaf, and the author gets one conflict line in `flips.md`/chat.
   - The alternatives are worse. Blocking pushes until the author decides stalls every chat edit (A3's loop). Carrying the held text into the pushed flat file breaks `push --check`'s rule that Overleaf's PDF and the source PDF are identical, and C30's page mapping relies on that rule.
2. **"Would break a lock" means the deterministic checks only.** That covers a macro-valued number (already a conflict), a generated region (already a conflict), any edit inside an `\auth{}` span (C26), a `forbids` hit, a flipped governed predicate or direction word, and any change to the words of a locked beat claim.
   - Cheapest mechanism for `\auth`: have flatten emit `\auth` spans as their own map segment kind, so `unflatten` classifies an edit inside one as a conflict at plan time instead of applying it and failing C26 afterwards. This also keeps `\auth{ID}{…}` wrappers out of the text coauthors read.
   - A meaning change outside these checks is applied, not held.
3. **Collaborator text is frozen but not locked.** No regeneration or agent pass rewrites it (the report's own "keep approved and collaborator prose frozen", §8.5 "Avoid"). An author chat edit or a lock change may still change it. When one does, the old text is parked, and the agent's one-line report names it ("also rewrote a sentence edited on Overleaf by <editor> on <date>").
4. **Record the editor as the Overleaf commit author** (F6), marked "last editor of the batch", not "unknown". Since `autopush` pulls before every push, batches are short and the name is usually exact.
5. **One digest line per pull**, e.g. "Overleaf: 2 edits applied in §4.2 (editor X), 1 held (D16)". This is not a review card. It is how stray keystrokes (F8) and coauthor meaning changes get seen at all. Adoption is a one-line chat message, which may cover many edits at once ("adopt Chris's edits in §4").

**Confidence:** 0.8. **Tier:** T2. The default is the reversible, author-retains-control option, and A2 ("through *my* reading and correction") and A5's "the *option* of flowing back" imply it. The author can veto in one line ("Chris's edits lock directly").

**Decisive evidence:**
- **Attribution.** Direct locking by named coauthors cannot be implemented reliably on the git channel. The bridge names only the last editor of each batch (F6), so a mixed batch is misattributed.
- **Stray keystrokes.** Every Overleaf edit from a "named" account so far, all of them the author's own, was a stray keystroke (F8). Under direct locking those would have become locks.
- **Unconfirmed parameters.** A lock carries machine parameters (`binds`) that an agent infers. For chat edits the author confirms ambiguous ones (§8.5 step 3). For an Overleaf edit nobody would, so an auto-lock would bind generators to an inference no human checked. That is the D13 failure class in reverse.
- **Precedence.** Two humans able to lock means conflicting locks with no precedence, which is A6's own T3 trigger. A single lock authority keeps precedence defined.
- **No stall.** D-a shows that "held" needs a watermark, or the loop stalls.

**What would change it:**
- The author saying a named coauthor (for example the advisor) has equal authority. Then add a per-editor standing adoption rule, applied only to batches with a single editor and never to the author's own account.
- Overleaf exposing per-change attribution to the git channel.
- The author preferring that held edits block pushes rather than be reverted on Overleaf with a visible note.

---

## Q11. Does A4 cover collaborators' text?

**Answer: yes for the scrub and Rule NT, no for the C26 byte-lock.** Text that arrives through the Overleaf pull is exempt from every automatic agent rewrite: scrub stages S2–S8 and Rule NT deletion. Unlike A4 text, it is not byte-locked. The author's chat edits and lock propagation may still change it, with the old text parked and the change reported (Q10 refinement 3). When the author adopts it, it becomes A4 text (`\auth`, C26).

At L4, where an unbacked literal universal is a gate failure, a failure inside collaborator text is not auto-fixed. It becomes a conflict for the author. NEEDED? findings on collaborator text are logged and not applied.

**Mechanism:** provenance from the pull commits (`git blame` lines last changed by an `Overleaf pull …` commit), excluded in S1 and rejected in S8. Do not use in-text markup. A `\collab{}` wrapper would appear in the flat file coauthors edit, and their next edit across it would become a structural conflict. With one sentence per line, the protection is per sentence. Before the reflow it is per paragraph line, which errs toward protecting too much.

**Confidence:** 0.75. **Tier:** T2. The protective core is effectively T1, for the first reason below. Whether coauthor text should ever be scrubbed for register consistency is taste, and the author can veto in one line.

**Decisive evidence:**
- **The pull cannot tell the author's own Overleaf typing from a coauthor's.** A4's literal scope is "text the author typed" (A4 box). A5 says the author edits on Overleaf "on occasion". The git bridge cannot separate the two within a batch (F6, F8). The only way to guarantee A4 mechanically for the author's own Overleaf text is to exempt all Overleaf-pulled text.
- **Risk asymmetry.** Rule NT deletes words automatically with no reader (C18). Applied to the advisor's or a coauthor's sentences (F9), that is a silent outward-facing edit to a senior coauthor's text. Exemption is reversible: the author can say "scrub that paragraph". The opposite default is not reversible in effect, because the coauthor sees their text changed.
- **Not a byte-lock.** Giving unadopted collaborator text a full C26 lock would make it stronger than the author's own agent-drafted locked text and would break Q10's adoption rule.

**What would change it:**
- The author saying that register consistency outranks coauthor wording. Then scrub collaborator text, but log each change in the pull digest.
- A reliable per-editor attribution channel. Then protection could be limited to the author's own Overleaf text plus adopted text.

---

## Q12. Tracked changes and comments on Overleaf

**What the git bridge actually carries:** file text only (F1).
- **Comments:** never carried, in either direction.
- **Tracked changes:** pending suggestions are exported as if accepted. Insertions are present and deletions absent (F2).
- **Pushes:** applied as minimal diffs (F3), and become suggestions if track changes is on for the pushing account (F4).

So "accept/reject before each pull" cannot be enforced, "detect and report" is impossible from git, and "never push over open suggestions" cannot be checked. None of the three options in §8.9 Q12 is implementable on the git channel.

**Rule:**
1. **Read Overleaf's visible text and nothing else.** The pull takes pending suggestions as edits (as if accepted) and treats them exactly like any collaborator edit: applied as text, recorded, held if it breaks a lock. A later rejection on Overleaf arrives at the next pull as an ordinary reverting edit. The repo converges with no discipline asked of coauthors and no new channel. If Overleaf's export turns out to exclude pending insertions instead, the same rule still holds: suggestions then simply arrive when accepted. The mechanism does not depend on which way F2 resolves.
2. **Keep track changes off for the git-token account.** Otherwise each agent push becomes a stack of suggestions in the author's name. Someone must then accept them one by one, which is A1's rejected per-edit review moved to Overleaf. A coauthor who rejects one, believing it the author's suggestion, sends a reversal of a locked author edit back through the pull, where it is held and re-pushed in a loop. Git cannot detect this setting, because the export looks identical either way, so check it once, read-only, in the project's Review panel. **Conditional T3:** if "track changes for everyone" is on, the per-user setting cannot override it (`isTrackChangesEnabledForUser` returns true for "everyone"). Turning it off, or asking coauthors to, affects them, so that one item goes to the author.
3. **Comments are never an input** (already resolved, §8.9 Q1). Preserve them:
   - minimal-diff pushes (C28; F3 shows the server already applies diffs, so the real threats are whole-file reflows and renames);
   - fixed flat file names (F7);
   - the one-sentence-per-line reflow done once, before coauthors start;
   - no agent rewrite of a span except under an author edit or a lock change.
4. **Optional cheap guard (T0):** hold any pulled insertion that reproduces text the agent's own pushes removed since the last pull, as a "possible resurrected suggestion". Rejecting a tracked deletion that a push displaced reinserts stale text at the displaced point (F3), and that text would otherwise flow back as a coauthor edit.
5. **Transparency:** add one line to the flatten's synthetic header, which coauthors see in `main2.tex`: "suggestions are read as accepted; comments are not read; edits that conflict with the author's decisions are parked as comments."

**Confidence:** 0.75 (F2's git-bridge path and F4's applicability to the bridge are strong inferences from open code, not direct tests). **Tier:** T1, with the conditional T3 in rule 2.

**Decisive evidence:** F1 (the snapshot has only `[content, path]`), F2 (`filterTrackedDeletes: true`), F3 (`setDoc` diffs), F4 (`upsertDoc` tracks external writes when track changes is on), and Overleaf's own advice "we do not recommend mixing active use of Git and the use of track changes and/or comments".

**What would change it:**
- A 5-minute test in a throwaway Overleaf project (never the paper's): make a tracked insertion and deletion, clone, and inspect. Only the documented semantics would change; the rule would not.
- Overleaf exposing comment or range metadata to git.
- The author wanting coauthors' comments read by agents. That would need the live channel (see Q13) and reopens §8.9 Q1.

---

## Q13. Agent edit route

**Answer: keep the git route. Do not adopt claudeleaf, as either the write route or a hybrid.**

**Confidence:** 0.85. **Tier:** T1.

**Decisive evidence:**
- **A1.** Tracked-change suggestions must be accepted one at a time on Overleaf. That is per-edit review, the "way too much" content A1 rejected.
- **The suggestions' review gate would be fake.** The pull would read a bot's pending suggestions as accepted (F2) and unflatten them into the sources before anyone accepted them.
- **Source of record.** Agent edits belong in sources, generators and macros. A bot typing into `main2.tex` has to come back through `unflatten`, where numbers and tables cannot go (generated and number conflicts, SPEC §5). While suggestions are pending, `push --check`'s rule that Overleaf's PDF equals the source PDF breaks, and with it C30 and A3's PDF-to-source mapping (§8.5).
- **Operational risk.**
  - It uses Overleaf's undocumented live channel.
  - It adds a second cached-session credential.
  - It edits the live document while coauthors type, and revert is only through Overleaf history.
  - The tool is small and young (F10).
- **Its benefits are small once C28 lands.** Comment preservation: the git route is already applied as minimal diffs (F3), so comments survive anywhere the push does not change the text. Typed edits destroy comments on text they change, just as pushes do. Attribution: all agent edits are applied on the author's chat instructions, so the author's account is the right owner in Overleaf history. The repo's trailers carry the AI provenance OUP needs (§7.1).

**Read-only hybrid considered and rejected:** listing comments and pending suggestions over the live channel. Comments are not an input (§8.9 Q1), and pending suggestions are already handled by Q12's rule 1, so it buys nothing that justifies the protocol and credential risk.

**What would change it:**
- The author wanting coauthors to see agent edits as distinct from his own in Overleaf history. The cheapest route then is still git, pushing with a dedicated bot account's git token. That is outward-facing (a visible collaborator, possibly a paid seat), so it would be T3 and is not asked now, since nothing requires it.
- The author wanting coauthors' comments read by agents (see Q12).
- Overleaf removing or breaking the git bridge.

---

## Summary

| Item | Answer | Confidence | Tier |
|---|---|---|---|
| Q10 | Keep the adoption default. Held edits: watermark past them, revert on Overleaf by the next push, park them verbatim as a dated comment, one conflict line to the author. Collaborator text is frozen, not locked. Editor = Overleaf commit author. One digest line per pull. No direct locking by named coauthors. | 0.8 | T2 |
| Q11 | Yes for the scrub and Rule NT (all Overleaf-pulled text is exempt, by pull-commit provenance). No C26 byte-lock until the author adopts it. | 0.75 | T2 |
| Q12 | Git carries text only: no comments, suggestions exported as if accepted. Rule: pull the visible text as ordinary collaborator edits; keep track changes off for the token account; protect comments by minimal diffs and fixed names. | 0.75 | T1 (conditional T3 if "track changes for everyone" is on) |
| Q13 | Keep the git route; no claudeleaf, no hybrid. | 0.85 | T1 |
| D-a, D-b, clean-tree auto-pull | Sync-code fixes: pull watermark; map written only after the push lands, with non-fast-forward retry; auto-pull only on a clean tree, committed alone | 0.85 | T0 |

**T3 questions:** none outright. One conditional: *"Overleaf has 'track changes for everyone' turned on, which turns every agent push into suggestions under your name. Turn it off for the project (it affects your coauthors), or leave it and have the agent stop auto-pushing main2.tex? Recommended default: turn it off and tell the coauthors in one line."* Ask it only if the read-only check finds the setting on.
