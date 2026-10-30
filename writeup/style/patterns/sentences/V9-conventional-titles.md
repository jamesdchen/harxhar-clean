# V9. Conventional section titles

**Family:** sentences. **Stage:** drafting (S0). **Basis:** the author's rule (28 Oct 2026).

**Severity:** review. A confirmed hit makes the drafter replace the title with the field's noun phrase for its topic, keeping its `\label`.
**Checked by:** code + auditor (sentences slice). The code lists titles with a colon or a question mark as candidates; the auditor confirms them and reads for full clauses.

## Intent

Title sections, subsections and appendices as papers in the field do: a short noun phrase naming the topic, not a
claim, a question or a phrase with a colon.

## Also known as

Conventional titles (commit 3f57938).

## Motivation

The author, 28 Oct 2026 (17:10 PDT): "make the section and subsection titles normal like you'd find in the literature,
not with ai prose". The commit that applied it (3f57938) shows the forms he removed: a colon reveal, a question and a
full clause. The notation guide records the same aim for the whole paper: "use standard symbols and language. Model it
after the papers in the financial and financial economic fields."

## Applicability

`\section`, `\subsection` and `\subsubsection` titles in the main text and the appendices. Not beat labels, which state
claims (V8). The author's words cover titles, not captions. Not titles the author supplied (V13). Labels (`\label`)
never change when a title does (notation guide house rules; 3f57938 kept them and parked the old titles).

## Structure / Rewrite

Before and after, from commit 3f57938:

| Instead of | Write |
|---|---|
| `\section{Results}` | `\section{Empirical results}` |
| `\subsection{Accuracy on every bar}` | `\subsection{Forecast accuracy}` |
| `\subsection{Where the gain comes from: FOMC statement days and ordinary days}` | `\subsection{FOMC statement days}` |
| `\subsection{Which predictors carry the gain?}` | `\subsection{Predictor-group ablations}` |
| `\subsubsection{The FOMC calendar: one group, few bars}` | `\subsubsection{FOMC calendar}` |
| `\subsubsection{What remains}` | `\subsubsection{Other predictor groups}` |

## Consequences

Costs: a title no longer carries the section's finding; the first beat does. Buys: titles read like the field's, and a
reader finds a section by its topic.

## Detection

If `writeup/style/prose_check.py` exists, it lists the titles with a colon or a question mark. A reviewer treats as
candidates titles with a colon, a question mark or a full clause, the three forms the 28 Oct edit removed.

## Related patterns

V3 (titles use the guide's names), V8, V11.

## Source

The author's words of 28 Oct 2026, transcribed in `writeup/deliberation/2026-10-29c/Q-2026-10-29-25.evidence.md`, item
6; commit 3f57938 ("Conventional section and subsection titles (author)"); `writeup/NOTATION_GUIDE_2026-10-15.md`,
preamble.
