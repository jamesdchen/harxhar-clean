# V11. Take the literature's register, not its sentences

**Family:** sentences. **Stage:** drafting (S0); S5, S6 and S8 at the edit stage. **Basis:** the author's request (24
Oct 2026); D for the copying risk (Lee et al.); I for phrase frames (no published tool, S6); T-genre for the n-gram
check (S8).

## Intent

Match how the target journals state standard things through abstracted phrase frames and corpus statistics, never by
copying or closely paraphrasing a sentence from a paper.

## Also known as

Phrase frames (S6); corpus calibration (S5).

## Motivation

The author, 24 Oct 2026: "as you read the literature, please make note of the prose and make the language of the paper
replicate it. so far it's pretty good, it just needs a few tweaks". LLMs reproduce verbatim, paraphrased and idea-level
content, and fine-tuning reduced verbatim copying but increased paraphrase reuse (Lee et al., §4.4). JFEc screens
manuscripts "with services provided by third parties" (§4.4).

## Applicability

Any sentence written with a paper open for style, and standard statements (Diebold–Mariano, the model confidence set,
QLIKE, Mincer–Zarnowitz). The corpus and the frames do not exist yet; which papers, through which access, is open
(§8.9 Q4). Until they exist, write the statement from the method's definition in your own words and cite the method.
A term enters the paper's register only if it appears in at least five corpus papers, so one author's habits do not
become the norm (§4.4).

## Structure / Rewrite

| Instead of | Write |
|---|---|
| a sentence pasted or closely paraphrased from a paper read for style | the paper's own sentence about its own design, with the method's `\cite` |
| a phrase adopted from one paper's habit | a phrase the corpus shows in at least five papers, or the plain form |
| (once S6 exists) a corpus sentence as an exemplar | the abstracted frame from `writeup/style/frames/`, with its bibkey and page |

## Consequences

Costs: slower than borrowing a good sentence, and until the corpus exists, register rests on the writer's reading.
Buys: no shared spans for a similarity screen to find, and a register set by the field rather than by one author.

## Detection

S8 (not built): the longest word n-gram shared with the corpus, through hashed 8-grams; rare shared spans are flagged.
S5 scores over-used terms against the corpus. By hand: compare a sentence written with a source open against that
source for shared runs of words.

## Related patterns

V3, V9, V12, V16.

## Source

The author's words of 24 Oct 2026, transcribed in `writeup/deliberation/2026-10-29c/Q-2026-10-29-25.evidence.md`, item
6; report §4.4 (inference: "Feed the LLM abstracted phrase frames ... never raw corpus sentences"; "Keep a term only if
it appears in at least five corpus papers"), §8.4 S5, S6 and S8; `writeup/NOTATION_GUIDE_2026-10-15.md`, preamble.
