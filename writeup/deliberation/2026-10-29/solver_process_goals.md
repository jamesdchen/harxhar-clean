# Solver: author goals and co-author norms (Q10–Q13)

Lens: the author's stated goals (A1–A6) and collaboration norms, plus how co-authors normally work on Overleaf. Written 29 Oct 2026 for the adjudicator under A6. Nothing was edited, committed, pushed or pulled.

Sources: the report `writeup/reports/AI layered paper writing system.md` (commit 99c572e: the Author-decisions box, §2.4, §8.5, §8.6, §8.9, §8.10) and its notes (`precedent_scholarly_overleaf.md`, `toggleable_layers.md`, `review_interfaces.md`, `precedent_single_sourcing.md`). Also the author's chat messages, from a stream scan of user messages across the nine project transcripts (timestamps in UTC; tokens masked); `writeup/overleaf_sync/README.md`; the local Overleaf clone `../harxhar-overleaf` (git log only); Overleaf's open-source code and docs; claudeleaf's README; and one co-authoring etiquette guide.

## Summary

| Item | Answer | Tier | Confidence |
|---|---|---|---|
| Q10 | Keep the default: the author is the only one whose edits lock, and collaborator edits are applied as text and recorded. Four refinements: a held edit stays visible and is never silently reverted; adoption happens at the author's single read, or by one word for later meaning changes; the author's own Overleaf account is not a locking channel; no named-coauthor locking by default | T2 | 0.80 |
| Q11 | Protect all human-typed Overleaf text from the scrub and Rule NT, the author's and collaborators' alike. Don't extend the full verbatim lock: the author's chat edits and the mechanical propagation of his decisions can still change it. Record its provenance in a sidecar, never as markup in the text | T1 core, T2 boundaries | 0.80 |
| Q12 | Collaborators work as they normally do. The pull mirrors what Overleaf compiles: pending suggestions come in as text, under Q10's rules. Agents never accept, reject or resolve anything on Overleaf unless the author tells them to in chat. Pushes stay minimal-diff. Collaborators' comments and suggestions are read-only inputs to the digest | T2 | 0.75 |
| Q13 | Keep git for every agent write. Don't use suggestion mode for agent edits, and don't add a bot account now. The live editor (the author's Chrome session) is for reading review state only, plus accepting or rejecting on the author's explicit word. Revisit only if minimal-diff pushes are seen to destroy comments | T1 | 0.85 |

None of the four is T3. Two facts below correct the report, and one needs the author's one-line OK because it changes his CLAUDE.md (see "Cross-item findings").

---

## Cross-item findings (facts the four answers rest on)

**F1. No collaborator has edited on Overleaf yet, and the author's Overleaf edits so far are stray keystrokes.** The local Overleaf clone has five commits made on the Overleaf side (committer `node`, author `jamesdchen01 <jamesdchen01@gmail.com>`). One is the August project's initial content (13 Aug). The other four, all on 28 Oct (US time), are one-line edits to `beats_top.tex`'s generated `%%` header: "at commit **doe s**0e76…", "at commit **I n**360d…", "at commit **let's**03e6…", and a stray line break. Each looks like chat typing that landed in the editor, and each was reverted (01ec92a, a3987d9, b301469; the line-break case is the one the coordinator asked about: "The stray line break in `beats_top.tex` on Overleaf: OK to overwrite it?", 2026-10-29T00:28Z). So the author's Overleaf account has not been a deliberate edit channel, and no collaborator edit or comment has reached the bridge.

**F2. The git bridge does record who edited, per Overleaf version.** This corrects §8.5's "the editor field therefore defaults to 'unknown'", which was carried over from the GitHub-sync docs. In Overleaf's git-bridge source, each snapshot carries the editor: `SnapshotInfo` has fields `versionId, comment, user (WLUser), createdAt`, with constructors taking `name, email`, and the default comment is "Update on" plus the service name ([git-bridge SnapshotInfo.java](https://raw.githubusercontent.com/overleaf/overleaf/main/services/git-bridge/src/main/java/uk/ac/ic/wlgitbridge/snapshot/getsavedvers/SnapshotInfo.java)). The local "Update on Overleaf." commits carry the Overleaf user as author and `node` as committer, which matches. **Unverified:** whether edits by two people between pulls are split into separate commits; one collaborator test edit settles it. Agent pushes made with the author's token presumably appear in Overleaf's history under the author's account (also unverified).

**F3. The git bridge sees Overleaf's "accept-all" text.** This settles the report's "whether the git bridge exposes tracked changes … is unverified", subject to one test. Overleaf's project-history service builds the snapshot "Used by git bridge to get the state of the project" (`getProjectSnapshot`) with `file.getContent({ filterTrackedDeletes: true })` ([SnapshotManager.js](https://raw.githubusercontent.com/overleaf/overleaf/main/services/project-history/app/js/SnapshotManager.js); route `/project/:project_id/version/:version`, [Router.js](https://raw.githubusercontent.com/overleaf/overleaf/main/services/project-history/app/js/Router.js)). In the pulled text, pending tracked insertions are present and pending tracked deletions are already gone. Comments and suggestion metadata live in separate "ranges" and never reach git. Overleaf's compiler and its project copies show the same view: "If you make a copy of your project, all tracked changes are automatically applied in the copied version" ([Overleaf: Track changes](https://docs.overleaf.com/collaborating/track-changes)). **Caveat:** this is the open-source code. The hosted overleaf.com bridge is assumed, not shown, to match. One test settles it: a tracked insertion and a tracked deletion, then `pull --dry-run`.

**F4. Overleaf's review tools are available, and project-level.** Track changes is premium, and "All collaborators invited to a project owned by an Overleaf account with this premium feature" can use it. The project has the git bridge, itself premium, so its owner has premium. The owner can force a collaborator's changes to be tracked. A "Reviewer" role, for named collaborators only, "can only make and respond to comments and make tracked changes" ([Track changes](https://docs.overleaf.com/collaborating/track-changes); [Reviewing and reviewers](https://docs.overleaf.com/collaborating/reviewing-and-reviewers)). Git pushes "can result in the loss or displacement of track changes and comments" (§2.4).

**F5. Who the collaborators are, and how their judgment has entered the paper.** They are the author's professors and a postdoc mentor. Their judgments have always reached the paper through the author's own messages:
- "my prof suggested clipping would help" (2026-10-02T07:03Z);
- "professor suggested we do an MCS instead of this lean stuff" (2026-10-08T23:52Z);
- "my professor wants to tell the following story: …" (2026-10-13T06:19Z);
- "fuck the headline. we need to write the paper to the prof's specification" (2026-10-14T02:47Z);
- "my prof wants to answer the following questions: …" (2026-10-22T21:24Z);
- "I'm thinking I point them to the overleaf" (2026-10-22T22:01Z);
- "it's just a meeting with my postdoc mentor" (2026-10-26T21:33Z).

The paper also has decisions owned by the professor: `\editc{orange}{E30}{Decision: open, for the professor. …}` in `rehaul/03_data.tex:241`. The author wanted `main2.tex` editable on Overleaf on the day of a 5 pm meeting: "I still can't edit main2" (2026-10-28T23:24Z), then "you have access to overleaf on chrome. make it work" (23:26Z). The coordinator then declined to change the main-document setting "since it affects your co-authors too" (23:27Z). The author is the pen-holder, and the professors set direction.

**F6. Co-author etiquette.** I found no source specific to economics or finance on Overleaf editing roles; this is a gap. A general etiquette guide ([CASRAI, "Track-changes etiquette for co-authored manuscripts", updated 25 Aug 2026](https://casrai.org/guides/track-changes-etiquette-co-authored-manuscripts)) gives four norms:
- "Never accept or reject another author's tracked change unilaterally."
- The corresponding author is the default tie-breaker.
- "Agree on one tool and one mode."
- Every tracked change is settled and every thread closed before submission, and threads are resolved rather than deleted.

The common economics pattern is that the junior author holds the pen while senior co-authors steer in meetings and line-edit. That is practitioner knowledge, not sourced here, but F5 shows this project follows it.

**F7. A standing rule must change, and only the author can change it.** The author's CLAUDE.md says: "If push refuses (Overleaf edits not pulled back), run `pull --dry-run`, report, and ask." Under A5 and A6, every collaborator edit would then become a question, which contradicts A1 ("way too much") and A6 (ask only T3). The later decisions (28–29 Oct) take precedence over the 22 Oct rule in substance. But agents cannot amend CLAUDE.md on their own authority, so the digest should carry a one-line request. Proposed wording: "Pull and apply per A5; ask only on held conflicts".

---

## Q10. Collaborator authority

**Answer: keep the default's core, refined.** The author is the only one whose edits lock. A collaborator's Overleaf edit is applied as text and recorded with its source: Overleaf, the date, and the bridge's commit author (F2). It becomes a lock only if the author adopts it. Named co-authors do not lock directly. Refinements:

1. **"Held" never means silently reverted on Overleaf.** As specified, a held edit is not applied to the source. The next flatten would then push the locked text back, and the professor's edit would vanish from Overleaf with no trace. That is a unilateral rejection of a co-author's edit (F6), and the professor cannot see `flips.md`. Instead, the held text goes into the source verbatim inside a visible marker, e.g. `\heldedit{OL-17}{D1}{<their text>}`. Review builds and `main2.tex` print the locked text with the held proposal beside it: "held: conflicts with D1; awaiting James". The final build treats an unresolved `\heldedit` as an error, like `\rhPending`. The lock stays intact in every build, the collaborator sees the edit arrived, and the author resolves it with one chat line. That resolution is an author judgment per instance and goes into his batch.
2. **Adoption costs the author no extra reading.**
   - A collaborator edit made before the author's single read of a layer is adopted by that read. Under A2 and §8.5, a beat the author has read, with his corrections applied, becomes a locked claim, and that includes text a collaborator wrote.
   - After a layer is locked, wording-only edits need no adoption.
   - Meaning-changing edits that break no lock are applied and listed in the digest by the drift gate (C29), one line each. The author replies "adopt" or "drop", and silence leaves them applied but unlocked.
   - Meaning-changing edits that contradict a lock follow refinement 1.
3. **The author's own Overleaf account is not a locking channel.** By F1, every Overleaf-side edit from his account has been a stray keystroke. Auto-locking that account would lock typos. Authority comes only from chat (A3). Edits that don't change the rendered text (whitespace, line breaks, keystrokes in a generated header) are absorbed or reverted as T0 and logged, as the header auto-revert already does. A prose fragment from the author's account that matches chat typing is reverted and listed in the digest (T1).
4. **No named-coauthor locking by default, but it is available as a standing rule.** If the author ever says "adopt whatever Prof X writes", edits whose commit author is that account lock directly, and the rule is recorded as a decision. Attribution by version (F2) supports this once one test edit confirms it.
5. **Prerequisite:** the CLAUDE.md amendment in F7.

**Why this and not direct locking:**
- **A1 (28 Oct):** "a human's judgement at any top layer … stay locked in". The same message frames it as "if *I* decide …". **A2:** "through *my* reading and correction". **A3:** "I … send you edits". Read together, the locking human is the author.
- **Two lock authorities would contradict each other with no precedence rule.** That is A6's definition of T3, so they would raise the author's question load, the opposite of A1 and A6.
- **The professors' judgments already lock through the author (F5).** He relays them, and a relayed instruction is a chat edit that locks with the professor named in its words. The default formalizes how the project already works.
- **A5 (28 Oct):** "edits do still need to have the *option* of flowing back". It requires flow-back, not authority.

**Tier: T2.** It weighs the professors' edits, which is the author's relationship and taste, but his quotes and the relay pattern imply the answer. It is reversible: the text is applied either way, and lock status can be granted later. The held marker is a draft artifact, not an outward action.

**What would change it:**
- The author says a named co-author's edits are final. Then refinement 4 becomes the rule for that person.
- The bridge test shows per-person attribution is unreliable. Then refinement 4 needs the Overleaf history, read through Chrome.
- Collaborators object to visible held markers. Then the alternative is a note posted on Overleaf in the author's name, which is outward-facing and T3.

---

## Q11. Does A4 cover collaborators' text?

**Answer.**
- **By its words, no.** A4 answered "Q9: whether Rule NT applies to text you typed yourself" with "no" (2026-10-29T06:37Z), so it covers the author's own text.
- **Extend its scrub protection to all human-typed Overleaf text.** The scrub stages that rewrite text (register, cadence, voice) and Rule NT's automatic deletions never alter text a person typed on Overleaf, the author's or a collaborator's.
- **Do not extend A4's full verbatim lock** (the byte-identity of C26).

**Boundaries:**
- **Still changed by the author's chat edits.** A collaborator sentence the author edits by chat becomes his `\auth` text.
- **Still changed by mechanical propagation of the author's locked decisions.** A1 requires a decision to be "reflected everywhere".
  - Numbers re-linked to their macros with the rendered text unchanged: T0 (`unflatten.py` already does this).
  - A scope phrase reworded minimally to match a decision: applied and listed in the digest.
  - A meaning-level clash with a lock: surfaced as a conflict.
- **Still checked by the gates.** A typed digit matching no macro, or an unbacked universal at L4, becomes a conflict for the author, never an automatic deletion. Scrub findings on human text are logged, not served (A1).

**Mechanism:**
- **Provenance goes in a sidecar** (file plus sentence hash → `overleaf`, editor, date), not in an `\auth`-style wrapper. The flatten copies source lines verbatim into `main2.tex`. Wrapping a collaborator's sentence would rewrite the very line that holds their Overleaf comment or suggestion, which a git push can then lose (F4).
- **S1 reads the sidecar** and excludes those spans from S2–S7 and S4b.

**Why:**
- **Purpose.** The scrub targets agent-written prose: its markers are LLM fingerprints (§4.1–§4.3), and the report's A4 scope box says "Rule NT governs only agent-written text" (§4.3.7). Human co-author text is not the target.
- **A4 cannot be guaranteed otherwise.** A5's effect says the author edits on Overleaf "on occasion". The bridge attributes edits by version, not by span (F2), and per-person splitting is unverified. A rule that scrubs collaborators' text but not the author's would scrub the author's own Overleaf text whenever attribution fails. Protecting all human Overleaf text is the only rule that never breaks the binding A4.
- **Co-author norm (F6).** An automated pass silently rewording the author's professors' sentences is a unilateral change to a co-author's text. The reputational cost falls on the author.
- **Cost.** A human sentence keeps a redundant universal or an awkward rhythm, which is the same cost A4 already accepts for the author's text (§8.8). The author can fix any of it with one chat edit.

**Tier: T1 for the core.** It follows from the binding A4, the bridge's documented attribution limits and the report's own Rule NT scope. **T2 for the boundaries**: propagation is applied, not asked; gates raise conflicts.

**What would change it:**
- The author says "polish everyone's text but mine". The scrub then runs on collaborator text, with the edits listed in the digest.
- Per-span attribution proves reliable, and the author wants collaborators scrubbed.
- A collaborator is found to paste AI-generated text, flagged by the S1–S3 markers. That reopens the question once.

---

## Q12. Tracked changes and comments on Overleaf

**Answer (rule):**
1. **Impose no process on collaborators.** Do not ask them to accept or reject before each pull: the autopush loop pulls every 30 seconds, the professors would be working for the tooling, and nothing enforces it. They use Overleaf as they normally do, with direct edits, suggestions and comments.
2. **Mirror what Overleaf compiles.** The bridge exports the accept-all view (F3), so a pending suggestion arrives exactly like a direct edit and goes through Q10's rules, recorded as source `overleaf (pending suggestion)` when the detection in point 6 has run. This is also what Overleaf's PDF shows, so the PDF the author reads already shows suggestions as text. Keeping the source equal to Overleaf's text is what `push --check` requires. A push that removed a pending suggestion would itself reject a co-author's suggestion unilaterally. A later accept changes nothing in git. A later reject arrives as an ordinary edit, and is held under Q10 if a lock formed in between.
3. **Agents never accept, reject or resolve anything on Overleaf, and never change project settings** (track-changes toggles, roles, main document), unless the author says so in chat (e.g. "accept Prof X's suggestions in 6.3"). Any write made through the author's live session appears as the author, so without his instruction it would be an outward action in his name. This matches the coordinator's own 28 Oct precedent ("I didn't change that setting, since it affects your co-authors too") and the norm in F6.
4. **Protect review metadata on push.** Pushes are minimal-diff with no reflow (C28, with the one announced sentence-per-line reflow done before collaborators start). A push changes a line holding a pending suggestion or comment anchor only when the author's edit or a lock requires it, and that is noted in the digest.
5. **Comments are read-only inputs to the digest, not an edit channel.** Collaborators' comments don't appear in the PDF the author reads (A3), so without this he would miss them. Each open comment becomes one digest line: who, where, the comment, and a proposed edit. The author's chat reply turns any of them into an edit. Agents never reply to or resolve comments on Overleaf.
6. **Detection mechanism.** Before a push that touches a file collaborators have edited or annotated, an agent reads the pending suggestions and open comments read-only from Overleaf's review panel in the author's Chrome session, which the author sanctioned for Overleaf on 28 Oct (2026-10-28T23:26Z). Build this at the first collaborator edit or comment, not before (F1).
7. **Before submission.** The submission is built from the repo, so Overleaf metadata never ships. The L4 checklist still requires no pending suggestions or open comments on Overleaf, so every co-author input has been considered (F6).

**Tier: T2.** The facts are T1 (F3, F4). The rule is inferred from A1, A3, A5 and the co-author norms. Nothing here acts on Overleaf's review state or on collaborators.

**What would change it:**
- The F3 test on overleaf.com shows the hosted bridge differs, for example tracked deletions exported as present. Point 2 would then have to subtract pending deletions using point 6's read.
- Comments are displaced even by minimal-diff pushes. Q13 reopens.
- The author says he will work the Overleaf review panel himself. The comment digest is then dropped.
- The author wants collaborators set to the Reviewer role. That is outward-facing and T3; not recommended.

---

## Q13. Agent edit route

**Answer:**
- **Keep git for every agent write.** No suggestion mode for agent edits, and no bot account now.
- **The hybrid is read-only.** The live editor, in the author's Chrome session, is used to read review state before pushes (Q12, points 5–6) and to accept or reject suggestions on the author's explicit chat instruction. It is never used to write paper content.

**Why:**
- **The author named the route.** A3 (28 Oct): "then you use the overleaf git to update what I'm looking at". His standing rule (CLAUDE.md, set 22 Oct: "no, override the rule. always push to overleaf when updating", 2026-10-22T22:37Z) runs `overleaf_git.sh push --check`. He asked for speed ("can overleaf be updated faster?", 2026-10-29T00:13Z), and the git route now takes about 41 s end to end, including the full-PDF identity check (2026-10-29T00:28Z).
- **Suggestion mode is per-edit review.** Every agent edit would become an accept/reject item, which A1 rejected ("no cards. the content that I need to review … is way too much"). The author doesn't work in Overleaf's editor (A3), so suggestions would sit pending. Overleaf compiles pending insertions (F3), so the PDF would look the same anyway. Hundreds of agent suggestions would also bury the professors' own.
- **Verification and atomicity.** The git route has `push --check` (flat and source builds give identical PDF text) and the `map.json` guard against pushing over unpulled edits. Live typing has neither. It would also replay large regenerated content (about 1.4 MB in `main2.tex`, about 1 MB per layer) keystroke by keystroke, without atomic multi-file updates.
- **Maturity and risk.** claudeleaf has 36 stars and dates from 2026. Its edits "apply immediately" ("Use Overleaf's history to review or revert changes"). Its cached session "is a credential". Its suggestions need premium. It cannot resolve or delete comments ([claudeleaf](https://github.com/lonetis/claudeleaf)).
- **Attribution gains little.** Agent pushes appear under the author's account in Overleaf (F2, unverified). But OUP's disclosure is generated from git trailers (§7.1, BLUF item 7), not from Overleaf's history, and the author answers for his agents' edits to his co-authors either way. The flatten commits already state their source commit ("Flatten from harxhar-clean <sha>").

**Tier: T1.** The author's recorded words settle it (A3, the CLAUDE.md rule, A1). A bot account would be T3: it adds a member to the co-authors' project and creates a stored credential. It is not recommended, so nothing is asked.

**What would change it:**
- In the first collaborator round, minimal-diff git pushes are seen to displace or destroy collaborators' comments or suggestions.
- The professors ask for agent edits to be distinguishable in Overleaf's history.
- The bridge's outages (repeated 503s on 28 Oct) become chronic.

Any of these reopens live-channel writes from a bot account, as direct edits (not suggestions) on the files collaborators annotate. That is a T3 question. The recommended default would be to keep git, with the live route only for annotated spans.

---

## Residual T3 questions

None from this solver. One item still needs the author because it changes his CLAUDE.md, not because the answer is in doubt (F7). Suggested digest line: "OK to change the Overleaf rule from 'report and ask on unpulled edits' to 'pull and apply per A5; ask only on held conflicts'?" Default if no reply: keep asking on each refused push, which is the current behaviour, until he says yes.

## Tests to run before relying on F2 and F3 (T0, reversible)

1. Have one collaborator make one direct edit. Check that the bridge commit's author is their account.
2. Make one tracked insertion and one tracked deletion. Run `pull --dry-run` and check that the insertion is present and the deletion absent.
3. On a scratch file, add one comment, then push a one-line change elsewhere in the file. Check that the comment survives a minimal-diff push.
