# Deliberation digest, round 2026-10-29c

Questions for you: 1. Applied provisionally, yours to veto: 4. Settled without you and logged: 2.

## Questions for you

### 1. May I add one line to the project section of your global CLAUDE.md telling every agent that writes or rewrites paper text (sections, beats, bullets, captions, table notes) to read your style guide, writeup/style/voice.md, first? Today no agent reads it, so your 29 Oct rule against tails like 'none significantly' does not reach drafting. (Q-2026-10-29-25)

- Recommendation: Yes. Add this line after the 'Writing-process reference' line: **Voice guide** (user, 2026-10-29: the style-guide rule V6; added on the author's yes to Q-2026-10-29-25): before writing or rewriting any paper text (sections, beats, bullets, captions, table notes, including the captions and notes written by `writeup/make_*.py`, and the wording of applied deliberation outcomes), read `writeup/style/voice.md` and follow its rules, and name that file in the prompt of any subagent you give such work. It governs agent-written text only: text the author supplied stays verbatim (A4), coauthors' Overleaf text is not restyled, and it is no reason to restyle text outside the edit at hand.
- What it changes: Changes only C:/Users/james/.claude/CLAUDE.md: one new line in the project section, about 100 words, read in every session; the file is outside git, so undoing it means deleting that line (a dated backup is taken first). No paper text changes when it is added. From then on every new session and every subagent that writes or rewrites paper text reads writeup/style/voice.md (rules V1-V6) first. Existing tails of the banned kind (for example 08_feature_importance.tex:224 and 261, A_results.tex:180 and 442, A_closing.tex:216) stay until an edit touches them or the section's review flags them; nothing is rewritten in bulk. One clash will surface: an earlier settled outcome (Q-2026-10-29-15) words 06_results.tex:146 as '... higher in all six classes, in none significantly', which is the kind of tail your new rule bans; it has not been applied yet, and this item does not decide it. The guide exists only on this branch (commit 2f61ba3), so agents in other worktrees will not find it until that commit is merged.
- Default if you do not answer: Nothing in CLAUDE.md changes. Agents read the style guide only when their task prompt names it; the coordinator names it in the drafting prompts it sends, which needs no config change.
- Confidence in the recommendation: 0.85. Alternatives: A rule file in the repo (.claude/rules/) that loads the guide whenever an agent opens a paper file: loads more reliably, but it is a new kind of config file that git ignores here; the fallback if agents ignore the pointer; No CLAUDE.md change, the coordinator names the guide in each drafting prompt: misses fresh sessions that apply your chat edits; Copy or import the whole guide into CLAUDE.md: about 70 lines in every session of every project, and two copies that drift apart; A repo-level CLAUDE.md: splits your project rules across two places and is committed separately on each branch.
- Why it comes to you: Only you can change your CLAUDE.md; hard trigger: it changes CLAUDE.md, settings or permissions.
- Answer in one line: `Q-2026-10-29-25 yes` to take the recommendation, or `Q-2026-10-29-25 <your answer>`.

## Applied provisionally (reply `veto <ID>` to undo one)

- Q-2026-10-29-30: Your ladder is now the every-bar one (figure, Panel B, intro, conclusion); the monthly ladder stays, labelled, for the forest, the 16:00 bar and the window grid.
- Q-2026-10-29-31: Ladder sentences on every-bar numbers: top 10 keep 96-97% of the boosted trees' gain and 104-115% of the linear models'; gain in about 10 of 46 series; 'concentrated', 'do best' dropped.
- Q-2026-10-29-26: Sec. 8: 'attribution ranks the parts the wrong way round' becomes 'the trees' attribution understates the FOMC calendar'; the 'converse' clause goes; the opening matches.
- Q-2026-10-29-27: Sec. 8 'In short': 'the series a model uses most are not the ones that make it more accurate' becomes the FOMC fact (1.4-2.3% of the trees' attribution, 65-68% of their gain).

## Logged only

T0: 2, T1: 0, minor T2: 0; every item is in writeup/deliberation/log.jsonl.
Still waiting for your answer from earlier digests: Q-2026-10-29-09 (2026-10-29), Q-2026-10-29-20 (2026-10-29), Q-2026-10-29-21 (2026-10-29).
