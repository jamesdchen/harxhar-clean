# Economics and finance writing guides compared with the paper-prose skill

Scope: the operational rules of the main economics and finance writing guides, classified as structure (S),
introductions, abstracts, conclusions and titles (I), claims (C), numbers (N), sentences and words (W), tables and
figures (T), and voice (V), and mapped against `.claude/skills/paper-prose/SKILL.md`, `writeup/style/voice.md` and the
pattern files V1–V16, AP-V and AP-S. The notation guide (`writeup/NOTATION_GUIDE_2026-10-15.md` §6 prose, §8 tables) is
counted where the skill reaches it.

Already in the report, not repeated here: Cochrane's root-out-every-passive rule and the JFEc voice question (report
§4.3.2); "It should be noted that", "In other words", "use" not "utilize", no adjectives on own results, "punchline in
Table 12", "No number should appear in a table that is not discussed in the text" and Head's formula (report §4.5, §2.3).
Some of these reached the skill (AP-S5, AP-V3) and some did not (the table rule, Head's formula); the mapping in Q5
shows which.

Sources read in full (retrieved 29 Oct 2026; page numbers are the printed ones):
- Cochrane, "Writing Tips for Ph.D. Students", 8 June 2005, 12 pp.
- Thomson, "The Young Person's Guide to Writing Economic Theory", JEL 37 (March 1999), pp. 157–183. This is the article
  the book grew out of; the 2nd edition of the book (2011) was not accessible.
- Varian, "How to Build an Economic Model in Your Spare Time".
- Shapiro, "Four Steps to an Applied Micro Paper", 19 Sept 2022.
- Head, "The Introduction Formula".
- Bellemare, "The Conclusion Formula", 6 Oct 2016.
- Goldsmith-Pinkham, "Writing and thinking with AI assistance", Substack, 12 Apr 2026.
- Harvey, "Reflections on Editing the Journal of Finance, 2006–2012".
- Stulz, JFE "Tips for Authors".
- The AER style guide and the JFEc author instructions.

McCloskey's book was not accessible: rule titles come from the Google Books table of contents, and the wording from
secondary summaries, marked as such.

## Q1. John Cochrane, "Writing Tips for Ph.D. Students" (2005): the rules

### Takeaway
Cochrane's guide is the most operational in the field: about 45 rules, most checkable by hand or by grep. Its core is
structural ("triangular" order, nothing before the main result the reader does not need, no travelogue). The skill has
taken his phrase bans (AP-S5, AP-V3) but none of his structure, table-discussion, figure, "naked this",
previews-and-recalls or economic-magnitude rules.

### Cited Findings
**Structure (S)**
- Write down "the one central and novel contribution of your paper" in one paragraph, and make it concrete — [Cochrane 2005, p. 1](https://static1.squarespace.com/static/5e6033a4ea02d801f37e15bb/t/5eda74919c44fa5f87452697/1591374993570/phd_paper_writing.pdf)
- Write in "'triangular' or 'newspaper' style, not in 'joke' or 'novel' style", and "put the punchline right up front" — [Cochrane, p. 1](https://static1.squarespace.com/static/5e6033a4ea02d801f37e15bb/t/5eda74919c44fa5f87452697/1591374993570/phd_paper_writing.pdf)
- "A good paper is not a travelogue of your search process"; "Do not motivate the specification that worked with all your failures" — [Cochrane, pp. 1, 3](https://static1.squarespace.com/static/5e6033a4ea02d801f37e15bb/t/5eda74919c44fa5f87452697/1591374993570/phd_paper_writing.pdf)
- "There should be nothing before the main result that a reader does not need to know in order to understand the main result" — [Cochrane, p. 3](https://static1.squarespace.com/static/5e6033a4ea02d801f37e15bb/t/5eda74919c44fa5f87452697/1591374993570/phd_paper_writing.pdf)
- After the main result come intuition graphs and tables, then "limited responses to potential criticisms and robustness checks. Most of those should end up in your web appendix" — [Cochrane, p. 3](https://static1.squarespace.com/static/5e6033a4ea02d801f37e15bb/t/5eda74919c44fa5f87452697/1591374993570/phd_paper_writing.pdf)
- At the micro level: "first describe what you do, then explain it, compare it to alternatives, and compare it to others' procedures" ("I adjust income by the square root of household size" comes first, then why) — [Cochrane, pp. 4–5](https://static1.squarespace.com/static/5e6033a4ea02d801f37e15bb/t/5eda74919c44fa5f87452697/1591374993570/phd_paper_writing.pdf)
- "Previews and recalls are a good sign of poor organization. 'As we will see in Table 6' 'Recall from section 2'..." — [Cochrane, p. 5](https://static1.squarespace.com/static/5e6033a4ea02d801f37e15bb/t/5eda74919c44fa5f87452697/1591374993570/phd_paper_writing.pdf)
- Footnotes are not for parenthetical comments: "If it's important, put it in the text. If it's not important, delete it." Use them for "Long lists of references, simple bits of algebra, or other documentation" — [Cochrane, p. 5](https://static1.squarespace.com/static/5e6033a4ea02d801f37e15bb/t/5eda74919c44fa5f87452697/1591374993570/phd_paper_writing.pdf)
- "Keep the paper as short as possible"; "Final papers should be no more than 40 pages" — [Cochrane, p. 4](https://static1.squarespace.com/static/5e6033a4ea02d801f37e15bb/t/5eda74919c44fa5f87452697/1591374993570/phd_paper_writing.pdf)
- "The most important thing in writing is to keep track of what your reader knows and doesn't know" — [Cochrane, p. 6](https://static1.squarespace.com/static/5e6033a4ea02d801f37e15bb/t/5eda74919c44fa5f87452697/1591374993570/phd_paper_writing.pdf)

**Introductions, abstracts, conclusions (I)**
- Abstract: "Most journals allow 100-150 words"; "You should not mention other literature in the abstract"; "Say what you find, not what you look for" — [Cochrane, p. 1](https://static1.squarespace.com/static/5e6033a4ea02d801f37e15bb/t/5eda74919c44fa5f87452697/1591374993570/phd_paper_writing.pdf)
- The introduction starts with the contribution and gives "the fact behind that result. 'In a regression of x on y, controlling for z, the coefficient is q.'" — [Cochrane, p. 2](https://static1.squarespace.com/static/5e6033a4ea02d801f37e15bb/t/5eda74919c44fa5f87452697/1591374993570/phd_paper_writing.pdf)
- No philosophical first sentence ("Financial economists have long wondered if markets are efficient"); this is "clearing your throat" — [Cochrane, p. 2](https://static1.squarespace.com/static/5e6033a4ea02d801f37e15bb/t/5eda74919c44fa5f87452697/1591374993570/phd_paper_writing.pdf)
- "Three pages is a good upper limit for the introduction" — [Cochrane, p. 2](https://static1.squarespace.com/static/5e6033a4ea02d801f37e15bb/t/5eda74919c44fa5f87452697/1591374993570/phd_paper_writing.pdf)
- Roadmap: "I don't write a 'roadmap' paragraph ... Make your own mind up about this question; but realize it's not mandatory" — [Cochrane, p. 2](https://static1.squarespace.com/static/5e6033a4ea02d801f37e15bb/t/5eda74919c44fa5f87452697/1591374993570/phd_paper_writing.pdf)
- Literature after the contribution, set off so readers can skip it, aimed at "the 2 or 3 closest current papers"; "Be generous in your citations" — [Cochrane, p. 2](https://static1.squarespace.com/static/5e6033a4ea02d801f37e15bb/t/5eda74919c44fa5f87452697/1591374993570/phd_paper_writing.pdf)
- Conclusions "short and sweet. Do not restate all of your findings. One statement in the abstract, one in the introduction and once more in the body"; "don't speculate"; a paragraph on limitations is allowed. Omitting the conclusion "was too radical for editors and referees" — [Cochrane, pp. 3–4](https://static1.squarespace.com/static/5e6033a4ea02d801f37e15bb/t/5eda74919c44fa5f87452697/1591374993570/phd_paper_writing.pdf)
- "Strike 'I leave x for future research.'"; "Don't start your paper with a cute quotation" — [Cochrane, pp. 8–9](https://static1.squarespace.com/static/5e6033a4ea02d801f37e15bb/t/5eda74919c44fa5f87452697/1591374993570/phd_paper_writing.pdf)

**Claims (C)**
- "Don't use adjectives to describe your work: 'striking results' 'very significant' coefficients ... If the work merits adjectives, the world will give them to you"; no double adjectives ("very novel") — [Cochrane, p. 7](https://static1.squarespace.com/static/5e6033a4ea02d801f37e15bb/t/5eda74919c44fa5f87452697/1591374993570/phd_paper_writing.pdf)
- "'It is easy to show that' means that it isn't"; "strike 'A comment is in order at this point.'" — [Cochrane, p. 7](https://static1.squarespace.com/static/5e6033a4ea02d801f37e15bb/t/5eda74919c44fa5f87452697/1591374993570/phd_paper_writing.pdf)
- "Give the stylized facts in the data that drive your result, not just estimates and p values". His example: Fama and French (1996) persuade by showing expected returns against betas "even though the GRS test is a disaster" — [Cochrane, p. 10](https://static1.squarespace.com/static/5e6033a4ea02d801f37e15bb/t/5eda74919c44fa5f87452697/1591374993570/phd_paper_writing.pdf)
- "Explain the economic magnitude of the central numbers, not just their statistical significance"; "every important number should include a standard error" — [Cochrane, pp. 10–11](https://static1.squarespace.com/static/5e6033a4ea02d801f37e15bb/t/5eda74919c44fa5f87452697/1591374993570/phd_paper_writing.pdf)
- "Describe the source of variation in the data that drives your estimates, for every single number you present" — [Cochrane, pp. 9–10](https://static1.squarespace.com/static/5e6033a4ea02d801f37e15bb/t/5eda74919c44fa5f87452697/1591374993570/phd_paper_writing.pdf)
- "Much bad writing comes down to trying to avoid responsibility for what you're saying" — [Cochrane, p. 7](https://static1.squarespace.com/static/5e6033a4ea02d801f37e15bb/t/5eda74919c44fa5f87452697/1591374993570/phd_paper_writing.pdf)

**Numbers (N)**
- "Use the correct number of significant digits ... 4.56783 with a standard error of 0.6789 should be 4.6 with a standard error of 0.7. Two to three significant digits are plenty" — [Cochrane, p. 6](https://static1.squarespace.com/static/5e6033a4ea02d801f37e15bb/t/5eda74919c44fa5f87452697/1591374993570/phd_paper_writing.pdf)
- "Use sensible units. Percentages are good." — [Cochrane, p. 6](https://static1.squarespace.com/static/5e6033a4ea02d801f37e15bb/t/5eda74919c44fa5f87452697/1591374993570/phd_paper_writing.pdf)
- Signs: "one direction is enough"; in "goes up (down) ... goes down (up)", "the stuff in the parentheses is distracting" — [Cochrane, p. 9](https://static1.squarespace.com/static/5e6033a4ea02d801f37e15bb/t/5eda74919c44fa5f87452697/1591374993570/phd_paper_writing.pdf)
- "A fellow graduate student must be able to ... reproduce every number in it from instructions given in the paper" — [Cochrane, p. 5](https://static1.squarespace.com/static/5e6033a4ea02d801f37e15bb/t/5eda74919c44fa5f87452697/1591374993570/phd_paper_writing.pdf)

**Tables and figures (T)**
- Each table has "a self-contained caption so that a skimming reader can understand the fact presented", but "Don't go nuts here; some captions are longer than the paper". A regression table's caption gives the equation and the left-hand variable — [Cochrane, pp. 5–6](https://static1.squarespace.com/static/5e6033a4ea02d801f37e15bb/t/5eda74919c44fa5f87452697/1591374993570/phd_paper_writing.pdf)
- "No number should appear in a table that is not discussed in the text ... 'Table 5 shows summary statistics' (period) is not ok" — [Cochrane, p. 6](https://static1.squarespace.com/static/5e6033a4ea02d801f37e15bb/t/5eda74919c44fa5f87452697/1591374993570/phd_paper_writing.pdf)
- Figures: "Label the axes. Use sensible units. Don't use dotted line types that are invisible when reproduced. Don't use dashes for very volatile series" — [Cochrane, p. 6](https://static1.squarespace.com/static/5e6033a4ea02d801f37e15bb/t/5eda74919c44fa5f87452697/1591374993570/phd_paper_writing.pdf)

**Sentences and words (W)**
- "Use the normal sentence structure: subject, verb, object"; "Keep down the number of clauses in your sentences"; "Every sentence should have a subject, verb and object" — [Cochrane, pp. 7–9](https://static1.squarespace.com/static/5e6033a4ea02d801f37e15bb/t/5eda74919c44fa5f87452697/1591374993570/phd_paper_writing.pdf)
- "Don't repeat things ... 'In other words' is a sign of trouble" — [Cochrane, p. 4](https://static1.squarespace.com/static/5e6033a4ea02d801f37e15bb/t/5eda74919c44fa5f87452697/1591374993570/phd_paper_writing.pdf)
- "everything before the 'that' should be deleted"; "'It should be noted that' is particularly obnoxious"; "Search for 'that'" — [Cochrane, p. 7](https://static1.squarespace.com/static/5e6033a4ea02d801f37e15bb/t/5eda74919c44fa5f87452697/1591374993570/phd_paper_writing.pdf)
- "Clothe the naked 'this.' ... 'This' should always have something following it. 'This regression shows that....' is fine" — [Cochrane, pp. 7–8](https://static1.squarespace.com/static/5e6033a4ea02d801f37e15bb/t/5eda74919c44fa5f87452697/1591374993570/phd_paper_writing.pdf)
- Greek letters: define them "in an easy-to-find place", give them a name, and remind the reader of the name ("the elasticity of substitution σ equals 3") unless it was named "in the last paragraph or two" — [Cochrane, p. 8](https://static1.squarespace.com/static/5e6033a4ea02d801f37e15bb/t/5eda74919c44fa5f87452697/1591374993570/phd_paper_writing.pdf)
- Hyphens, quoting the JFE style sheet: hyphenate "true compound modifiers before the noun (e.g., after-tax income, risk-free rate ...) unless part of the compound modifier is an adverb ending in 'ly' (e.g., ... equally weighted index)" — [Cochrane, p. 8](https://static1.squarespace.com/static/5e6033a4ea02d801f37e15bb/t/5eda74919c44fa5f87452697/1591374993570/phd_paper_writing.pdf)
- Write "models in which", not "models where"; "Don't abbreviate authors' names, 'FF show that ...'"; don't overuse italics; avoid jargon; be concrete — [Cochrane, pp. 7–9](https://static1.squarespace.com/static/5e6033a4ea02d801f37e15bb/t/5eda74919c44fa5f87452697/1591374993570/phd_paper_writing.pdf)

**Voice (V)**
- "'I' is fine. Don't use the royal 'we' on a sole-authored paper"; "Table 5 presents estimates" is acceptable, "though a purist might object" — [Cochrane, p. 6](https://static1.squarespace.com/static/5e6033a4ea02d801f37e15bb/t/5eda74919c44fa5f87452697/1591374993570/phd_paper_writing.pdf)
- "Present tense is usually best. You can say 'Fama and French 1993 find that'"; "keep the tense consistent" — [Cochrane, p. 7](https://static1.squarespace.com/static/5e6033a4ea02d801f37e15bb/t/5eda74919c44fa5f87452697/1591374993570/phd_paper_writing.pdf)
- "we are primarily writers. Economics and finance papers are essays" — [Cochrane, p. 12](https://static1.squarespace.com/static/5e6033a4ea02d801f37e15bb/t/5eda74919c44fa5f87452697/1591374993570/phd_paper_writing.pdf)

### Inferences
- Cochrane works at three levels: the paper (order), the exhibit (tables are discussed, figures are legible) and the
  sentence (bans). The skill copies only the sentence level. The paper and exhibit levels are where his rules are both
  checkable and most relevant to a results-heavy JFEc paper with many ablation tables.
- "No number in a table that is not discussed" is the mirror of V2. V2 makes every number in the text come from the
  results; Cochrane makes every number in the results reach the text. For generated tables, a check could require each
  table label to be `\ref`'d by at least one sentence that states a finding with a macro.
- Several micro rules are greppable and cheap, and the current main text mostly passes them. A rough grep of
  `writeup/rehaul/*.tex` on 29 Oct 2026 found 0 previews or recalls, 0 "up (down)" parentheticals, 0 sentences starting
  with a math token, 15 `\footnote` calls and 1 naked "This is" (`A_econ.tex:433`). As rules they would mostly stop
  regressions.
- "Give the stylized facts, not just estimates and p values" has a direct reading here: show the loss differences (by
  day type, by year) behind each Diebold–Mariano test, not the test alone.

### Gaps
- None material; the 2005 PDF was read in full. Cochrane's later blog posts on writing were not checked.

## Q2. Deirdre McCloskey, *Economical Writing* (3rd ed. 2019; 1st as *Economic Inquiry* 1985)

### Takeaway
The 35 rules of the 3rd edition are verified as chapter titles with page numbers, but the book's wording could not be
read; rule content comes from secondary summaries. Her rules that bear on the skill are four: Avoid Boilerplate
(including the table-of-contents paragraph), Avoid Elegant Variation, Make Tables, Graphs, Displayed Equations, and
Labels on Images Readable by Themselves, and Avoid This, That, These, Those. The first and last are missing from the
skill; elegant variation is covered by V3, but only for estimators.

### Cited Findings
- 3rd-edition table of contents (rule number: title, page), relevant entries: 11 "Speak to an Audience of Human Beings"
  (39); 12 "Avoid Boilerplate" (41); 13 "Control Your Tone" (46); 14 "A Paragraph Should Have a Point" (52); 15 "Make
  Tables, Graphs, Displayed Equations, and Labels on Images Readable by Themselves" (55); 16 "Footnotes and Other
  Scholarly Tics Are Pedantic" (58); 17 "Make Your Writing Cohere" (60); 19 "Write in Complete Sentences" (65); 20
  "Avoid Elegant Variation" (66); 23 "The Order Around Switch Until It Good Sounds" (78); 25 "Use Verbs, Active Ones"
  (83); 26 "Avoid Words That Bad Writers Love" (86); 27 "Be Concrete" (92); 28 "Be Plain" (94); 29 "Avoid Cheap
  Typographical Tricks" (98); 30 "Avoid This, That, These, Those" (101) — [Google Books TOC](https://books.google.com/books/about/Economical_Writing_Third_Edition.html?id=j9SWDwAAQBAJ)
- The rules first appeared as McCloskey, "Economical Writing", *Economic Inquiry* 23 (April 1985): 187–222 (citation only; text not accessed) — [search result listing](https://ageconsearch.umn.edu/record/136731/files/Latham_39_2.pdf)
- Boilerplate (secondary quotes of the 2019 book): "Never start a paper with that all-purpose filler for the bankrupt imagination, 'This paper …'" — [alexop.dev summary](https://alexop.dev/posts/10-rules-better-writing-economical-writing-book/). Another summary renders the rule as dropping "the table of contents paragraph because it betrays a bankrupt imagination" — [Sum & Some](https://www.sumandsome.com/summaries/economical-writing)
- Elegant variation (secondary): those who vary words for elegance "mistake the purpose of writing, believing it to be an opportunity for empty display. The seventh grade, they should realize, is over." — [alexop.dev summary](https://alexop.dev/posts/10-rules-better-writing-economical-writing-book/)
- A paragraph is "a more or less complete discussion of one topic" (secondary quote) — [alexop.dev summary](https://alexop.dev/posts/10-rules-better-writing-economical-writing-book/)
- A university handout "Adapted from Economical Writing" (secondary, not McCloskey's wording):
  - "It is far better to repeat a word than to use synonyms and confuse your reader";
  - "place the main idea (emphasis) of each sentence at its end";
  - "Minimize use of doublets";
  - "Avoid excessive use of This, That, These and Those ... try repeating the word it represents";
  - a bad-word list ("individuals" and "agents" when you mean people, "the existence of", "process", "structure",
    "implement", "comprise", "analyze" as a synonym for "discuss").

  — [WU Vienna handout](https://wu.ac.at/fileadmin/wu/d/economics/VW_Zentrum/Veranstaltungsunterlagen/Writing_Advice.pdf)
- Cochrane recommends McCloskey's *Rhetoric of Economics*, and Shapiro's note lists "McCloskey, D. 1985. Economical writing" as further reading — [Cochrane, p. 12](https://static1.squarespace.com/static/5e6033a4ea02d801f37e15bb/t/5eda74919c44fa5f87452697/1591374993570/phd_paper_writing.pdf); [Shapiro 2022, p. 3](https://ignaciomsarmiento.github.io/teaching/seminar/foursteps.pdf)

### Inferences
- McCloskey's "this" rule is stricter than Cochrane's. Cochrane allows "This regression shows"; McCloskey prefers "the"
  or the repeated noun. V10 flags "This decomposition highlights" for its evaluative verb, but neither rule's "this" is
  covered.
- "Avoid elegant variation" is the general form of V3. V3 covers only estimators, sets, samples and tests, so
  everyday synonyms ("loss" / "forecast error" / "accuracy"; "gain" / "improvement") can still vary.
- The end-focus advice ("main idea at its end"; secondary wording) is in mild tension with V1 ("subject and main verb
  early; scope, conditions and tests after them"). V1 puts the test in end position, McCloskey the point. V6 bans
  only clipped verdict tails after a finished clause, so it does not conflict with end focus in general.

### Gaps
- The book's text, including any "five-finger exercise", was not accessed. I found no source that confirms a
  "five-finger exercise" rule in *Economical Writing*; treat that as unverified.
- Exact wording of Rules 12, 15, 20 and 30 in the 3rd edition, and their page-level content, is unverified (secondary
  sources only).

## Q3. William Thomson, *A Guide for the Young Economist* (2nd ed. 2011), via its JEL 1999 precursor

### Takeaway
Thomson's guide is aimed at theory (notation, definitions, proofs), but four of its rules transfer directly:
- one name per concept, with the literature's synonyms listed once;
- consistency of person, voice and tense, with "we" preferred;
- no sentence starting with a symbol;
- the one-sentence-per-paragraph structure test.

He also contradicts the notation guide on the tense for prior work, and V9's example title on abbreviations in
headings.

### Cited Findings
- "By leafing through your article, a reader should be able to easily spot the main results, figure out most of the
  notation, and locate the crucial definitions" — [Thomson 1999, p. 158](https://static.uni-graz.at/fileadmin/_Persoenliche_Webseite/wendner_ronald/write_a_thesis/Thomson.pdf)
- "Your title should be as descriptive of your topic as possible. Devote time to your abstract" — [Thomson, p. 158](https://static.uni-graz.at/fileadmin/_Persoenliche_Webseite/wendner_ronald/write_a_thesis/Thomson.pdf)
- Introduction: "Do not start with a two- or three-page survey of the field"; the literature review "should not be a
  mere enumeration of previous articles ... give priority to the development of the ideas rather than to telling us who
  did what" — [Thomson, p. 158](https://static.uni-graz.at/fileadmin/_Persoenliche_Webseite/wendner_ronald/write_a_thesis/Thomson.pdf)
- Conclusion: "should not be a rehashing of the introduction", but "a compact summary of your results and a statement of
  the main lesson" leading to "specific open questions" belongs there — [Thomson, p. 158](https://static.uni-graz.at/fileadmin/_Persoenliche_Webseite/wendner_ronald/write_a_thesis/Thomson.pdf)
- Structure test: "summarize each of them [paragraphs] in one sentence. Does the string of these sentences make sense?
  It should." Repeat for subsections and sections — [Thomson, p. 159](https://static.uni-graz.at/fileadmin/_Persoenliche_Webseite/wendner_ronald/write_a_thesis/Thomson.pdf)
- "motivate your work, but do not over-motivate it, or your readers will get suspicious" — [Thomson, p. 159](https://static.uni-graz.at/fileadmin/_Persoenliche_Webseite/wendner_ronald/write_a_thesis/Thomson.pdf)
- Mnemonic labels instead of "Assumptions A1–A3 and B1–B4"; "Never use abbreviations in a section heading" — [Thomson, p. 161](https://static.uni-graz.at/fileadmin/_Persoenliche_Webseite/wendner_ronald/write_a_thesis/Thomson.pdf)
- "Refer to a given concept by only one name or phrase ... Indicate in parentheses next to your definition, or in a
  footnote, the other terms that appear in the literature"; after the concept is "baptized, only refer to it by its
  name" — [Thomson, p. 167](https://static.uni-graz.at/fileadmin/_Persoenliche_Webseite/wendner_ronald/write_a_thesis/Thomson.pdf)
- Figures: warn the reader that a figure exists before using it; "Label your figures as completely as possible"; "Avoid
  unnecessary arrows"; "Have one enumeration for each category of objects" — [Thomson, pp. 172–173](https://static.uni-graz.at/fileadmin/_Persoenliche_Webseite/wendner_ronald/write_a_thesis/Thomson.pdf)
- "search for the 'clearlys' and 'obviouslys' and make sure that what you claimed was clear and obvious is, if not clear
  and obvious, at least true" — [Thomson, p. 178](https://static.uni-graz.at/fileadmin/_Persoenliche_Webseite/wendner_ronald/write_a_thesis/Thomson.pdf)
- "Do not start a sentence with a piece of mathematical notation ... 'Let x designate an allocation' is what editors
  will prefer" — [Thomson, p. 180](https://static.uni-graz.at/fileadmin/_Persoenliche_Webseite/wendner_ronald/write_a_thesis/Thomson.pdf)
- "Do not switch back and forth between first person singular, first person plural, and passive forms"; the example
  ends "Section 5 concludes", and he asks "why do you let Section 5 conclude when you did all the work?"; "Between 'I'
  and 'we', I choose 'we'" — [Thomson, p. 180](https://static.uni-graz.at/fileadmin/_Persoenliche_Webseite/wendner_ronald/write_a_thesis/Thomson.pdf)
- "do not travel back and forth between present and future tenses ... using the present tense throughout, even in
  describing past literature, is just fine" — [Thomson, p. 180](https://static.uni-graz.at/fileadmin/_Persoenliche_Webseite/wendner_ronald/write_a_thesis/Thomson.pdf)
- "Let time elapse between revisions"; "Good writing requires rewriting, and rewriting again" — [Thomson, p. 182](https://static.uni-graz.at/fileadmin/_Persoenliche_Webseite/wendner_ronald/write_a_thesis/Thomson.pdf)
- Bellemare says Thomson's *Guide* "has nothing about how to write conclusions", but the 1999 article does have the
  conclusion paragraph quoted above (p. 158) — [Bellemare 2016](https://marcfbellemare.com/wordpress/12060); contradicted by [Thomson 1999, p. 158](https://static.uni-graz.at/fileadmin/_Persoenliche_Webseite/wendner_ronald/write_a_thesis/Thomson.pdf)

### Inferences
- Thomson's p. 167 rule is V3 plus one addition worth taking: give the literature's other names for a term once, at
  its definition ("the full predictor set (all features in ...)"), and never again.
- The paper's roadmap ends "Section~\ref{sec:rh_conclusion} concludes." (`writeup/rehaul/01_introduction.tex:202`).
  Thomson objects to that construction; V10 does not flag it (a reporting verb with an exhibit subject); Cochrane
  accepts "Table 5 presents estimates". The guides split, and the skill sides with Cochrane.
- V9's own example `\subsubsection{FOMC calendar}` breaks Thomson's "Never use abbreviations in a section heading".
  For field-standard acronyms (HAR, FOMC) this is a weak objection, but it applies to any acronym coined in the paper.

### Gaps
- The 2011 book's added chapters (introductions, talks, and whatever else it adds) were not accessed. I could not
  check whether the book changes the 1999 rules or adds rules on empirical papers. The article says it "does not cover
  the writing up of empirical work" (p. 157).

## Q4. Other guides: Goldsmith-Pinkham, Shapiro, Bellemare, Head, Varian, Sahm, Kiefer, finance editors, AEA, JFEc

### Takeaway
The introduction templates (Head, Shapiro) and the conclusion template (Bellemare) are the main thing the skill lacks.
They conflict with Cochrane's ordering (contribution first, no hook, optional roadmap, minimal conclusion). Finance
editors (Harvey at JF, Stulz at JFE) add the following:
- abstract and introduction pitched to a general reader, with the "new economic insight";
- "minimal data mining";
- no "technobabble";
- no insults toward prior work.

Goldsmith-Pinkham's material is about AI workflow, not a writing guide. Its concrete point, "Ask for comments, not
rewrites", is already in the skill's reviewer branch G. His personal style habits (em dashes, subordinate clauses)
contradict V1 and AP-S1.

### Cited Findings
**Head, "The Introduction Formula" (introduction)**
- Hook ("Attract the reader's interest by telling them that this paper relates to something interesting": Y matters,
  is puzzling, controversial, big or common). Avoid "The bait and switch" and "'all my friends are doing it'" — [Head](https://blogs.ubc.ca/khead/research/research-advice/formula)
- Question: "Tell the reader what this paper actually does"; the "'This paper addresses the question' sentence" comes at
  the end of paragraph 2 or the start of paragraph 3 — [Head](https://blogs.ubc.ca/khead/research/research-advice/formula)
- Antecedents: only prior work "critical for understanding the contribution"; establish "in a non-insulting way, that
  the prior work is incomplete" — [Head](https://blogs.ubc.ca/khead/research/research-advice/formula)
- Value-added: "approximately 3 contributions", "the most important" paragraph for referees, up to 3 paragraphs — [Head](https://blogs.ubc.ca/khead/research/research-advice/formula)
- Road-map: avoid an outline "so generic that it could apply to any paper"; mention pivotal "landmarks"; "keep this
  short" — [Head](https://blogs.ubc.ca/khead/research/research-advice/formula)
- Write the introduction first, then re-edit it "every time you compose other parts of the paper"; "If you can't write
  a good introduction, then you may be writing the wrong paper" — [Head](https://blogs.ubc.ca/khead/research/research-advice/formula)

**Shapiro, "Four Steps to an Applied Micro Paper" (2022) (structure, introduction)**
- Step 1, aspirational introduction: "Feel free to make up your results, within reason"; Step 3, the "Robot": "Don't
  try to convince the robot you are right. Just state your assumptions, methods, and findings. Be linear ... Be clear
  ... Be plain ... Be formal" — [Shapiro 2022, p. 1](https://ignaciomsarmiento.github.io/teaching/seminar/foursteps.pdf)
- Step 4, contractual introduction: "a contract between you and the reader"; "you will find you want claim some things
  that your research does not really deliver. When this happens, return to steps 2-3" — [Shapiro, p. 2](https://ignaciomsarmiento.github.io/teaching/seminar/foursteps.pdf)
- Introduction template (Appendix B): paragraphs 1–2 Motivation; 3–4 Challenges ("why your research question has not
  already been answered"); 5 This Paper; 6–7 Model; 8–9 Data; 10–11 Methods; 12–13 Findings ("connect clearly to the
  motivation"); 14–15 Literature ("two main ways your paper contributes ... how your paper differs from the most closely
  related recent work") — [Shapiro, p. 3](https://ignaciomsarmiento.github.io/teaching/seminar/foursteps.pdf)
- Official listing: [Shapiro notes and lectures](https://shapiro.scholars.harvard.edu/notes-and-lectures) (returned
  403 to automated fetch; the PDF copy above is a course mirror dated 19 Sept 2022)

**Bellemare, "The Conclusion Formula" (6 Oct 2016) (conclusion)**
- Four parts in order:
  - Summary: "it needs to be different enough from the abstract and the introduction";
  - Limitations: "the conclusion should also emphasize the limitations of your approach";
  - Implications for Policy: "don't make claims that are not supported by your results", with costs set against
    benefits;
  - Implications for Future Research.

  — [Bellemare](https://marcfbellemare.com/wordpress/12060)
- He teaches Head's formula ("Hook-Research Question-Antecedents-Value Added-Roadmap") and found no economics norm for
  conclusions — [Bellemare](https://marcfbellemare.com/wordpress/12060)

**Varian, "How to Build an Economic Model in Your Spare Time" (structure)**
- "the introduction is the most important part of the paper. You've got to grab the reader on the first page"; "make
  your paper look like your talk. Get to the point. Use examples. Keep it simple. Tell people why what you did is
  important after you've done it. Put the tedious stuff in the appendix. End with a summary" — [Varian, §1.10, p. 11](https://people.ischool.berkeley.edu/~hal/Papers/how.pdf)
- "Once you've made your point, stop"; readers "only remember about 10 pages of your paper" — [Varian, §1.11, p. 12](https://people.ischool.berkeley.edu/~hal/Papers/how.pdf)

**Goldsmith-Pinkham, "Writing and thinking with AI assistance" (12 Apr 2026) (process, voice)**
- A style-guide skill is "just a laundry list of rules ... text that gets dumped into the context window"; "more 'list
  of rules' than 'voice capture'" — [Goldsmith-Pinkham](https://paulgp.substack.com/p/writing-and-thinking-with-ai-assistance)
- LLMs produce "a kind of homogenized default voice"; "having a unique voice is important" — [Goldsmith-Pinkham](https://paulgp.substack.com/p/writing-and-thinking-with-ai-assistance)
- His own guide's patterns: "use subordinate clauses, I do a lot of em-dashes, translate coefficients into dollar terms
  when discussing economic significance" — [Goldsmith-Pinkham](https://paulgp.substack.com/p/writing-and-thinking-with-ai-assistance)
- "Ask for comments, not rewrites" — [Goldsmith-Pinkham](https://paulgp.substack.com/p/writing-and-thinking-with-ai-assistance)

**Finance editors (claims, introduction)**
- Harvey (JF editor 2006–2012, about 7,500 submissions), item 16: abstract and introduction "should clearly state the
  paper's new economic insight" and argue it is "big" enough; they "need to be pitched to the general reader, not a
  specialist" — [Harvey, p. 7](https://faculty.fuqua.duke.edu/~charvey/Research/Working_Papers/W111_Reflections_on_editing.pdf)
- Harvey, item 19, empirical papers should:
  - "explain why the finding is important";
  - "clearly differentiate the paper's approach from past approaches";
  - "not be shackled by past methods (we do X because another did X in the past)";
  - "convince the reader that there has been minimal data mining";
  - "check for robustness and influential observations".

  — [Harvey, p. 8](https://faculty.fuqua.duke.edu/~charvey/Research/Working_Papers/W111_Reflections_on_editing.pdf)
- Stulz (JFE tips):
  - "Do not fall into the trap of technobabble";
  - abstract, introduction and conclusion "clear enough that you could read them to a class of MBA students";
  - "use technical language only when it is necessary";
  - "Do not reference papers that are irrelevant";
  - "Avoid insults and slights".

  — [Stulz, JFE Tips for Authors, Q1](https://www.wu.ac.at/fileadmin/wu/d/i/ifr/Stulz-Tips_for_authors.pdf)
- Stulz: an abstract, introduction and conclusion that "reflect the contribution of the paper accurately" help the
  editor pick referees; referees may return a paper unreviewed for "poor and careless exposition" — [Stulz, Q9](https://www.wu.ac.at/fileadmin/wu/d/i/ifr/Stulz-Tips_for_authors.pdf)

**Journal style rules (tables, abstract)**
- JFEc:
  - "an abstract of not more than 100 words";
  - two to six keywords and JEL codes;
  - "should typically not exceed 40 double-spaced typewritten pages";
  - tables carry "a brief, descriptive title", and "Vertical rules should not be used";
  - for figures, "All explanatory material should be included in the legend";
  - Chicago Manual of Style, 15th edition;
  - manuscripts "may also be screened, including with services provided by third parties".

  — [JFEc General Instructions](https://academic.oup.com/jfec/pages/General_Instructions) (read through a fetch tool's
  summary; quotes as returned)
- AER (not binding on JFEc):
  - abstract "must not exceed 100 words";
  - "The introductory section should not receive a heading";
  - tables at most 9 columns;
  - "Do not abbreviate in column headings";
  - a zero before the decimal point;
  - "Do not use asterisks to denote significance of estimation results. Report the standard errors in parentheses."

  — [AER Style Guide](https://www.aeaweb.org/journals/aer/style-guide)

**Compilations (secondary)**
- A PhD workshop attributes rules to Goldin and Katz: "Write the damn abstract! Do not copy and paste from the intro";
  "Use 'I' if you're the only author. Use 'we' otherwise."; "Define the acronym the first time only, then just use
  it" — [McCarthy, Emory workshop](https://imccart.github.io/phd-workshop-content/year3/write-better.html)
- Nikolov's compilation: "Do not create your own acronyms"; in tables, "Don't use variable abbreviations from your
  statistical software"; "Evidence from..." titles are "clichéd" (citing Hamermesh) — [Nikolov, IZA DP 16276](https://repec.iza.org/dp16276.pdf)

### Inferences
- The introduction templates disagree on order:
  - Cochrane: contribution first, no hook, no throat-clearing.
  - Head: hook, question by paragraph 2–3, antecedents, about 3 contributions, roadmap.
  - Shapiro: motivation, challenges, then "This Paper" at paragraph 5.

  JFEc introductions are a register question for the corpus (report §4.4). The choice belongs to the author, through
  the deliberate queue, not to a pattern.
- Head's "approximately 3 contributions" means a triad in the value-added paragraph is the field's norm. AP-S8 counts
  triads per paragraph as an AI marker, so the contributions paragraph would need an exemption, or AP-S8 stays a pure
  rate.
- Shapiro's aspirational introduction ("make up your results") must never enter the paper sources. In this system it
  could live only outside `writeup/rehaul/`, or as an L0 draft with `\rhPending{}` placeholders, so that V2 holds.
- Harvey's "minimal data mining" fits the repo's pre-registered `RULE_*.txt` files (the X8 rule file of 29 Oct
  2026, commit 6acaee1). The prose could say that the specifications were fixed before computation. No current pattern
  asks for that sentence.
- V1 cites "Goldsmith-Pinkham's style guide" as part of its T-genre basis, but his guide's patterns (subordinate
  clauses, em dashes) are the opposite of V1 and AP-S1. The transferable part is the practice (a rules file referenced
  in prompts; comments, not rewrites), not his rules. V1's basis line could say so.
- "Translate coefficients into dollar terms" (Goldsmith-Pinkham) and "Explain the economic magnitude" (Cochrane p. 10)
  map onto this paper's straddle economics. A loss reduction would be read alongside the strategy's return or Sharpe
  ratio, where the paper has them.

### Gaps
- Claudia Sahm: I found no writing guide or writing-advice post; only biography and opinion columns came up.
- "Kiefer": I could not identify a Kiefer writing guide.
- Goldsmith-Pinkham has no public standalone writing or style guide that I found; his day-to-day style guide file is
  not public.
- I found no RFS editor statement on writing, and I could not retrieve the JF submission guidelines page (afajof.org
  returned no content).
- The full Goldin–Katz rule list was not retrieved (the workshop slide reproduces it only in part).
- Bellemare has no separate introduction formula; he teaches Head's.
- From memory, not fetched: Harvey's 2017 AFA presidential address ("The Scientific Outlook in Financial Economics",
  JF 72(4)) argues for multiple-testing adjustments in finance. It may bear on how the paper reports its many
  ablation tests, but was not verified here.

## Q5. Mapping: which rules the skill has, lacks, or contradicts

### Takeaway
Of 52 operational rules mapped:
- 13 are covered by a pattern;
- 14 are partly covered (in the notation guide or the report but not routed by the skill, or narrower);
- 22 are missing;
- 3 conflict with the skill or the notation guide.

What is missing is mostly paper-level and exhibit-level: section order, introduction, abstract and conclusion
templates, table discussion, figures, and the economic magnitude of results. A few cheap grep rules are missing too.
On two of the missing rules, the roadmap and the conclusion, the guides disagree among themselves. The three conflicts
are:
- present tense for prior work, against the notation guide's past tense;
- no abbreviations in section headings, against V9's "FOMC calendar" example;
- AER's ban on significance stars.

Goldsmith-Pinkham's em dashes and subordinate clauses also conflict with V1 and AP-S1; they are listed below the
table.

### Cited Findings
Verdicts: **C** covered, **P** partly, **M** missing, **X** contradicted. "NG" means the notation guide.

| # | Cat | Guide rule (short) | Source | Skill / repo | Verdict |
|---|---|---|---|---|---|
| 1 | S | Main result first; nothing before it the reader doesn't need | [Cochrane pp. 1, 3](https://static1.squarespace.com/static/5e6033a4ea02d801f37e15bb/t/5eda74919c44fa5f87452697/1591374993570/phd_paper_writing.pdf) | Report L0 (contribution) only; no pattern for section order | P |
| 2 | S | One central contribution in one concrete paragraph | [Cochrane p. 1](https://static1.squarespace.com/static/5e6033a4ea02d801f37e15bb/t/5eda74919c44fa5f87452697/1591374993570/phd_paper_writing.pdf) | Report §8.1 L0 "(Cochrane, Minto)"; not in skill | P |
| 3 | S | No travelogue of the search; failures not used as motivation | [Cochrane pp. 1, 3](https://static1.squarespace.com/static/5e6033a4ea02d801f37e15bb/t/5eda74919c44fa5f87452697/1591374993570/phd_paper_writing.pdf) | None | M |
| 4 | S | Robustness to (web) appendix; tedious stuff to appendix | [Cochrane p. 3](https://static1.squarespace.com/static/5e6033a4ea02d801f37e15bb/t/5eda74919c44fa5f87452697/1591374993570/phd_paper_writing.pdf); [Varian p. 11](https://people.ischool.berkeley.edu/~hal/Papers/how.pdf) | None in skill (paper already has an online appendix) | M |
| 5 | S | Describe what you do, then why, then alternatives | [Cochrane pp. 4–5](https://static1.squarespace.com/static/5e6033a4ea02d801f37e15bb/t/5eda74919c44fa5f87452697/1591374993570/phd_paper_writing.pdf) | None | M |
| 6 | S | No previews or recalls ("As we will see", "Recall from") | [Cochrane p. 5](https://static1.squarespace.com/static/5e6033a4ea02d801f37e15bb/t/5eda74919c44fa5f87452697/1591374993570/phd_paper_writing.pdf) | AP-S5 lists other signposts, not these | M |
| 7 | S | Footnotes only for skippable documentation | [Cochrane p. 5](https://static1.squarespace.com/static/5e6033a4ea02d801f37e15bb/t/5eda74919c44fa5f87452697/1591374993570/phd_paper_writing.pdf); [McCloskey R16](https://books.google.com/books/about/Economical_Writing_Third_Edition.html?id=j9SWDwAAQBAJ) | None | M |
| 8 | S | Paragraph has one point; one-sentence-per-paragraph test | [Thomson p. 159](https://static.uni-graz.at/fileadmin/_Persoenliche_Webseite/wendner_ronald/write_a_thesis/Thomson.pdf); [McCloskey R14](https://books.google.com/books/about/Economical_Writing_Third_Edition.html?id=j9SWDwAAQBAJ) | V8 (beat labels state claims); report §2.3 reverse outline | C |
| 9 | S | Track what the reader knows; be linear (define before use) | [Cochrane p. 6](https://static1.squarespace.com/static/5e6033a4ea02d801f37e15bb/t/5eda74919c44fa5f87452697/1591374993570/phd_paper_writing.pdf); [Shapiro p. 1](https://ignaciomsarmiento.github.io/teaching/seminar/foursteps.pdf) | NG §6 "never use a symbol before its definition"; not routed by the skill | P |
| 10 | S | Length: about 40 pages; stop once the point is made | [Cochrane p. 4](https://static1.squarespace.com/static/5e6033a4ea02d801f37e15bb/t/5eda74919c44fa5f87452697/1591374993570/phd_paper_writing.pdf); [JFEc](https://academic.oup.com/jfec/pages/General_Instructions); [Varian p. 12](https://people.ischool.berkeley.edu/~hal/Papers/how.pdf) | Report §7.2 (JFEc limits); not in skill | P |
| 11 | I | Abstract of 100 words or fewer | [JFEc](https://academic.oup.com/jfec/pages/General_Instructions); [Cochrane p. 1](https://static1.squarespace.com/static/5e6033a4ea02d801f37e15bb/t/5eda74919c44fa5f87452697/1591374993570/phd_paper_writing.pdf) | Report L0 "D for the ≤100-word limit (JFEc)" | C |
| 12 | I | Abstract: what you find, not what you look for; no literature | [Cochrane p. 1](https://static1.squarespace.com/static/5e6033a4ea02d801f37e15bb/t/5eda74919c44fa5f87452697/1591374993570/phd_paper_writing.pdf) | V7 scopes claims; no abstract rule | P |
| 13 | I | Introduction opens with the contribution and the fact behind it, not throat-clearing | [Cochrane p. 2](https://static1.squarespace.com/static/5e6033a4ea02d801f37e15bb/t/5eda74919c44fa5f87452697/1591374993570/phd_paper_writing.pdf) | V1/AP-S10 ban importance assertions; no first-sentence rule | P |
| 14 | I | Introduction template (hook, question, antecedents, about 3 contributions, roadmap) | [Head](https://blogs.ubc.ca/khead/research/research-advice/formula); [Shapiro p. 3](https://ignaciomsarmiento.github.io/teaching/seminar/foursteps.pdf) | Report §2.3, §4.5 mention Head; no skill branch | M |
| 15 | I | Introduction of 3 pages or fewer | [Cochrane p. 2](https://static1.squarespace.com/static/5e6033a4ea02d801f37e15bb/t/5eda74919c44fa5f87452697/1591374993570/phd_paper_writing.pdf) | None | M |
| 16 | I | Contractual introduction: claim only what the paper delivers | [Shapiro p. 2](https://ignaciomsarmiento.github.io/teaching/seminar/foursteps.pdf); [Stulz Q9](https://www.wu.ac.at/fileadmin/wu/d/i/ifr/Stulz-Tips_for_authors.pdf) | V7 + C19 (qualifier diff across sections) | C |
| 17 | I | Abstract and introduction for the general reader ("MBA students") | [Harvey p. 7](https://faculty.fuqua.duke.edu/~charvey/Research/Working_Papers/W111_Reflections_on_editing.pdf); [Stulz Q1](https://www.wu.ac.at/fileadmin/wu/d/i/ifr/Stulz-Tips_for_authors.pdf) | None (V3 imposes technical names everywhere) | M |
| 18 | I | Literature: after the contribution, set off, 2–3 closest papers; ideas, not enumeration | [Cochrane p. 2](https://static1.squarespace.com/static/5e6033a4ea02d801f37e15bb/t/5eda74919c44fa5f87452697/1591374993570/phd_paper_writing.pdf); [Thomson p. 158](https://static.uni-graz.at/fileadmin/_Persoenliche_Webseite/wendner_ronald/write_a_thesis/Thomson.pdf); [Shapiro p. 3](https://ignaciomsarmiento.github.io/teaching/seminar/foursteps.pdf) | None (paper has a separate `02_related_work.tex`) | M |
| 19 | I | Roadmap: none (Cochrane), customized (Head), or drop it (McCloskey, secondary) | [Cochrane p. 2](https://static1.squarespace.com/static/5e6033a4ea02d801f37e15bb/t/5eda74919c44fa5f87452697/1591374993570/phd_paper_writing.pdf); [Head](https://blogs.ubc.ca/khead/research/research-advice/formula); [Sum & Some](https://www.sumandsome.com/summaries/economical-writing) | Skill silent; paper has one ending "Section … concludes." | M (guides disagree) |
| 20 | I | Conclusion: short, no restating, no future research (Cochrane) vs summary, limitations, implications, future research (Bellemare) vs summary, lesson, open questions (Thomson) | [Cochrane pp. 3–4, 8](https://static1.squarespace.com/static/5e6033a4ea02d801f37e15bb/t/5eda74919c44fa5f87452697/1591374993570/phd_paper_writing.pdf); [Bellemare](https://marcfbellemare.com/wordpress/12060); [Thomson p. 158](https://static.uni-graz.at/fileadmin/_Persoenliche_Webseite/wendner_ronald/write_a_thesis/Thomson.pdf) | AP-S6 allows summary openers there; V7 scope | M (guides disagree) |
| 21 | I | Paper title descriptive; avoid "Evidence from …" | [Thomson p. 158](https://static.uni-graz.at/fileadmin/_Persoenliche_Webseite/wendner_ronald/write_a_thesis/Thomson.pdf); [Nikolov](https://repec.iza.org/dp16276.pdf) | V9 covers section titles only | P |
| 22 | C | No adjectives on own results; no "very" | [Cochrane p. 7](https://static1.squarespace.com/static/5e6033a4ea02d801f37e15bb/t/5eda74919c44fa5f87452697/1591374993570/phd_paper_writing.pdf) | AP-V3; V5 boosters | C |
| 23 | C | "Clearly"/"obviously" must at least be true | [Thomson p. 178](https://static.uni-graz.at/fileadmin/_Persoenliche_Webseite/wendner_ronald/write_a_thesis/Thomson.pdf) | V5 (NEEDED? then TRUE?); NG §6 bans "clearly" | C |
| 24 | C | "It is easy to show that" means it isn't | [Cochrane p. 7](https://static1.squarespace.com/static/5e6033a4ea02d801f37e15bb/t/5eda74919c44fa5f87452697/1591374993570/phd_paper_writing.pdf) | Not in the V5 lexicon or AP-S5 | M |
| 25 | C | Economic magnitude, not just statistical significance | [Cochrane p. 10](https://static1.squarespace.com/static/5e6033a4ea02d801f37e15bb/t/5eda74919c44fa5f87452697/1591374993570/phd_paper_writing.pdf); [Goldsmith-Pinkham](https://paulgp.substack.com/p/writing-and-thinking-with-ai-assistance) | V4 and NG §6 separate the words; nothing requires the translation | P |
| 26 | C | Every important number has a standard error or test | [Cochrane p. 11](https://static1.squarespace.com/static/5e6033a4ea02d801f37e15bb/t/5eda74919c44fa5f87452697/1591374993570/phd_paper_writing.pdf) | V4 "result, then its test"; NG §6 | C |
| 27 | C | Stylized facts behind the result, not only estimates and p-values | [Cochrane p. 10](https://static1.squarespace.com/static/5e6033a4ea02d801f37e15bb/t/5eda74919c44fa5f87452697/1591374993570/phd_paper_writing.pdf) | None | M |
| 28 | C | Source of variation behind each number | [Cochrane pp. 9–10](https://static1.squarespace.com/static/5e6033a4ea02d801f37e15bb/t/5eda74919c44fa5f87452697/1591374993570/phd_paper_writing.pdf) | V7 (sample, period, test) | P |
| 29 | C | No speculation; implications only where results support them | [Cochrane p. 4](https://static1.squarespace.com/static/5e6033a4ea02d801f37e15bb/t/5eda74919c44fa5f87452697/1591374993570/phd_paper_writing.pdf); [Bellemare](https://marcfbellemare.com/wordpress/12060) | V7 (hedges on interpretation only) | C |
| 30 | C | Prior work: non-insulting, generous, relevant only | [Head](https://blogs.ubc.ca/khead/research/research-advice/formula); [Stulz Q1](https://www.wu.ac.at/fileadmin/wu/d/i/ifr/Stulz-Tips_for_authors.pdf); [Cochrane p. 2](https://static1.squarespace.com/static/5e6033a4ea02d801f37e15bb/t/5eda74919c44fa5f87452697/1591374993570/phd_paper_writing.pdf) | Report §6 checks citation support, not framing | M |
| 31 | C | Show minimal data mining; not "we do X because another did X" | [Harvey p. 8](https://faculty.fuqua.duke.edu/~charvey/Research/Working_Papers/W111_Reflections_on_editing.pdf) | Pre-registered `RULE_*.txt` exist; no prose rule | M |
| 32 | C | Don't over-motivate | [Thomson p. 159](https://static.uni-graz.at/fileadmin/_Persoenliche_Webseite/wendner_ronald/write_a_thesis/Thomson.pdf) | AP-S10, V1 (importance assertions) | C |
| 33 | N | 2–3 significant digits; sensible units (percentages) | [Cochrane p. 6](https://static1.squarespace.com/static/5e6033a4ea02d801f37e15bb/t/5eda74919c44fa5f87452697/1591374993570/phd_paper_writing.pdf) | NG §8 fixes decimals per column (QLIKE 4, % 1, DM 2); not routed by skill | P |
| 34 | N | One direction for signs; no "up (down)" | [Cochrane p. 9](https://static1.squarespace.com/static/5e6033a4ea02d801f37e15bb/t/5eda74919c44fa5f87452697/1591374993570/phd_paper_writing.pdf) | None | M |
| 35 | N | Every number reproducible from the paper | [Cochrane p. 5](https://static1.squarespace.com/static/5e6033a4ea02d801f37e15bb/t/5eda74919c44fa5f87452697/1591374993570/phd_paper_writing.pdf) | V2 (macros, provenance) | C |
| 36 | T | Every table number discussed; no bare "Table X reports …" | [Cochrane p. 6](https://static1.squarespace.com/static/5e6033a4ea02d801f37e15bb/t/5eda74919c44fa5f87452697/1591374993570/phd_paper_writing.pdf) | Quoted in report §4.5; V10 allows "Table X reports" with no finding required | M |
| 37 | T | Self-contained caption and notes, without going overboard | [Cochrane pp. 5–6](https://static1.squarespace.com/static/5e6033a4ea02d801f37e15bb/t/5eda74919c44fa5f87452697/1591374993570/phd_paper_writing.pdf); [McCloskey R15](https://books.google.com/books/about/Economical_Writing_Third_Edition.html?id=j9SWDwAAQBAJ); [JFEc](https://academic.oup.com/jfec/pages/General_Instructions) | NG §8 (short title, notes of about 5 sentences); skill branch C does not route to NG §8 | P |
| 38 | T | Figures: label axes, units, no dotted or dashed lines for volatile series; warn before use | [Cochrane p. 6](https://static1.squarespace.com/static/5e6033a4ea02d801f37e15bb/t/5eda74919c44fa5f87452697/1591374993570/phd_paper_writing.pdf); [Thomson pp. 172–173](https://static.uni-graz.at/fileadmin/_Persoenliche_Webseite/wendner_ronald/write_a_thesis/Thomson.pdf) | NG §7 figures parked; nothing in skill | M |
| 39 | T | No software variable names; no abbreviations in column heads | [Nikolov](https://repec.iza.org/dp16276.pdf); [AER](https://www.aeaweb.org/journals/aer/style-guide) | NG §8 "No code names"; V3 | C |
| 40 | T | No significance asterisks; standard errors in parentheses | [AER](https://www.aeaweb.org/journals/aer/style-guide) | NG §8 uses stars ("finance convention"); JFEc silent | X (AER only) |
| 41 | W | One name per concept; list the literature's synonyms once at the definition | [Thomson p. 167](https://static.uni-graz.at/fileadmin/_Persoenliche_Webseite/wendner_ronald/write_a_thesis/Thomson.pdf); [McCloskey R20](https://books.google.com/books/about/Economical_Writing_Third_Edition.html?id=j9SWDwAAQBAJ) | V3 (estimators, sets, samples, tests only) | P |
| 42 | W | Subject–verb–object; few clauses; complete sentences | [Cochrane pp. 7–9](https://static1.squarespace.com/static/5e6033a4ea02d801f37e15bb/t/5eda74919c44fa5f87452697/1591374993570/phd_paper_writing.pdf); [McCloskey R19](https://books.google.com/books/about/Economical_Writing_Third_Edition.html?id=j9SWDwAAQBAJ) | V1 | C |
| 43 | W | Delete "It should be noted that", "In other words"; "Search for 'that'" | [Cochrane pp. 4, 7](https://static1.squarespace.com/static/5e6033a4ea02d801f37e15bb/t/5eda74919c44fa5f87452697/1591374993570/phd_paper_writing.pdf) | AP-S5 | C |
| 44 | W | Clothe the naked "this" (Cochrane) / avoid this, that, these, those (McCloskey) | [Cochrane pp. 7–8](https://static1.squarespace.com/static/5e6033a4ea02d801f37e15bb/t/5eda74919c44fa5f87452697/1591374993570/phd_paper_writing.pdf); [McCloskey R30](https://books.google.com/books/about/Economical_Writing_Third_Edition.html?id=j9SWDwAAQBAJ) | None (V10 only for evaluative verbs) | M |
| 45 | W | No sentence starting with a symbol ("Let x designate …") | [Thomson p. 180](https://static.uni-graz.at/fileadmin/_Persoenliche_Webseite/wendner_ronald/write_a_thesis/Thomson.pdf) | None | M |
| 46 | W | Name a symbol and remind the reader of its name when re-used | [Cochrane p. 8](https://static1.squarespace.com/static/5e6033a4ea02d801f37e15bb/t/5eda74919c44fa5f87452697/1591374993570/phd_paper_writing.pdf) | NG §6 "define at first use" | P |
| 47 | W | Plain words; no jargon or "technobabble"; economese list | [Cochrane p. 7](https://static1.squarespace.com/static/5e6033a4ea02d801f37e15bb/t/5eda74919c44fa5f87452697/1591374993570/phd_paper_writing.pdf); [Stulz Q1](https://www.wu.ac.at/fileadmin/wu/d/i/ifr/Stulz-Tips_for_authors.pdf); [WU handout](https://wu.ac.at/fileadmin/wu/d/economics/VW_Zentrum/Veranstaltungsunterlagen/Writing_Advice.pdf) | AP-V lists AI markers, not economese; V3 removes code jargon | P |
| 48 | W | Hyphenate compound modifiers, not "-ly" adverbs (JFE style sheet) | [Cochrane p. 8](https://static1.squarespace.com/static/5e6033a4ea02d801f37e15bb/t/5eda74919c44fa5f87452697/1591374993570/phd_paper_writing.pdf) | None | M |
| 49 | W | Minor bans: "FF" for authors; "where" for "in which"; cute quotation; doublets; "leave for future research"; italics overuse | [Cochrane pp. 8–9](https://static1.squarespace.com/static/5e6033a4ea02d801f37e15bb/t/5eda74919c44fa5f87452697/1591374993570/phd_paper_writing.pdf); [WU handout](https://wu.ac.at/fileadmin/wu/d/economics/VW_Zentrum/Veranstaltungsunterlagen/Writing_Advice.pdf) | None (AP-S9 covers bold headers only) | M |
| 50 | W | No abbreviations in section headings; no coined acronyms | [Thomson p. 161](https://static.uni-graz.at/fileadmin/_Persoenliche_Webseite/wendner_ronald/write_a_thesis/Thomson.pdf); [Nikolov](https://repec.iza.org/dp16276.pdf) | V9's example `\subsubsection{FOMC calendar}` | X (minor) |
| 51 | V | Present tense, even for prior literature | [Cochrane p. 7](https://static1.squarespace.com/static/5e6033a4ea02d801f37e15bb/t/5eda74919c44fa5f87452697/1591374993570/phd_paper_writing.pdf); [Thomson p. 180](https://static.uni-graz.at/fileadmin/_Persoenliche_Webseite/wendner_ronald/write_a_thesis/Thomson.pdf) | NG §6: "past tense for what others found" | X |
| 52 | V | Consistent persona; "we" over "I" (Thomson), but no royal "we" when sole-authored (Cochrane; Goldin–Katz) | [Thomson p. 180](https://static.uni-graz.at/fileadmin/_Persoenliche_Webseite/wendner_ronald/write_a_thesis/Thomson.pdf); [Cochrane p. 6](https://static1.squarespace.com/static/5e6033a4ea02d801f37e15bb/t/5eda74919c44fa5f87452697/1591374993570/phd_paper_writing.pdf); [McCarthy](https://imccart.github.io/phd-workshop-content/year3/write-better.html) | NG §6 "We"; the source says "the authors" (`paper_rehaul.tex:47–48`) | C (if multi-author) |

Rules outside the table:
- Active voice everywhere (Cochrane p. 6; McCloskey R25). The report already sets it against the JFEc corpus (§4.3.2),
  and the skill defers voice to the corpus bands (V10 Applicability). Status: open by design, not a gap.
- Goldsmith-Pinkham's own patterns, "use subordinate clauses, I do a lot of em-dashes", contradict V1 and AP-S1 — [Goldsmith-Pinkham](https://paulgp.substack.com/p/writing-and-thinking-with-ai-assistance)
- "Ask for comments, not rewrites" is covered by branch G ("Do not rewrite") and by V13–V15 — [Goldsmith-Pinkham](https://paulgp.substack.com/p/writing-and-thinking-with-ai-assistance)

### Inferences
**What the guides do better**
- **Paper and exhibit architecture.** Cochrane's order rule, Head's and Shapiro's paragraph templates, Bellemare's
  conclusion and Cochrane's table-discussion rule all work above the sentence. The skill has no branch for an
  introduction, an abstract or a conclusion, and its caption branch C does not route to NG §8.
- **Reader model.** "Keep track of what your reader knows" (Cochrane) and "Be linear" (Shapiro) generalize NG §6's
  symbol rule to concepts and results.
- **Economics content in claims.** Economic magnitude, stylized facts behind tests, source of variation, minimal data
  mining. The skill governs how a verdict is worded (V4–V7), not whether the economic reading is given.
- **Cheap greps.** Previews and recalls, naked "this", "It is easy to show that", "up (down)", a sentence starting with
  math, "FF", "leave for future research". Each is one regex row in AP-S.

**What to borrow** (candidates; ids are for the caller to assign)
1. A branch for an introduction, abstract or conclusion that routes to:
   - an "abstract states findings, no literature" rule (Cochrane p. 1);
   - a "first sentence states the contribution or a fact with a number" rule (Cochrane p. 2; constrains Head's hook
     against AP-S10);
   - "contributions relative to the 2–3 closest papers, stated without slights" (Head, Shapiro, Stulz).

   The choice among the Cochrane, Head and Shapiro orders, the roadmap and the conclusion content is an author
   decision for the deliberate queue.
2. A table-discussion rule: each table and figure is cited by a sentence that states a finding with a macro, never
   "Table X reports Y." alone (Cochrane p. 6). Checkable from `\ref` sites.
3. New AP-S rows:
   - previews and recalls;
   - "It is easy to show that", "Note that", "It is interesting to";
   - naked "This/That" plus a verb (live example: `A_econ.tex:433`, "This is a diagnostic, not a strategy of the
     paper.");
   - "(lower)" or "(down)" sign parentheticals;
   - "leave … for future research";
   - author initials.
4. A rule against starting a sentence with math (Thomson p. 180), and the remind-the-name rule for symbols re-used
   after a gap (Cochrane p. 8).
5. Widen V3 to every recurring concept, with the literature's synonyms given once at the definition (Thomson p. 167).
6. An economic-magnitude rule: a headline loss result is paired with its economic reading where the paper has one
   (Cochrane p. 10; Goldsmith-Pinkham), stated as a macro.
7. A "stylized facts first" rule for results sections: the loss differences behind a DM test appear in a table or
   figure the text discusses (Cochrane p. 10).
8. A figure-style row (axes, units, solid lines for volatile series), since realized-volatility series are the
   textbook "very volatile series" (Cochrane p. 6).
9. Route branch C to NG §8 (and NG §7 once figures are unparked).

**What to avoid**
- Encoding Cochrane's self-described outlier positions as hard rules: no conclusion ("too radical for editors and
  referees", pp. 3–4) and no roadmap ("Make your own mind up", p. 2).
- AER-specific formats for a JFEc paper (no asterisks, Roman-numeral headings, unheaded introduction). JFEc's
  instructions are silent on stars.
- Head's hook as written ("Y matters", "Y is big"). Drafted by an LLM, it invites the importance assertions AP-S10
  removes; constrain it to a fact with a number or citation.
- Shapiro's made-up results in the aspirational introduction anywhere in the paper sources (V2).
- Taking a guide author's personal voice (Goldsmith-Pinkham's em dashes, subordinate clauses) as a field norm.
- Readability-formula targets (see Q6). They reward short sentences and short words, and a draft tuned to them could
  narrow sentence-length spread, which the report flags as an LLM trait (§4.3.3).

### Gaps
- Counts are this note's own classification of 52 rules; another reader could split or merge rows differently.
- Authorship (sole or multiple) was not checked against the title page; row 52's verdict assumes multiple authors from
  "the authors" in `paper_rehaul.tex`.
- Whether JFEc articles use present or past tense for prior findings (row 51) and roadmap paragraphs (row 19) is a
  corpus question (report §4.4); no corpus figures exist yet.

## Q6. Which rules are prescriptive opinion and which have evidence behind them

### Takeaway
Every guide rule above is prescriptive (grade P in the report's scheme), backed by editorial experience. Harvey reports
about 7,500 JF submissions handled; the others are senior authors' practice. Evidence exists only for the general claim
that clearer writing helps, from one causal study and several correlational ones, and for the overgeneralization risk
already in the report. No study tests an individual economics-writing rule (roadmap, previews, naked "this", passive
voice, triangular order).

### Cited Findings
- Feld, Lines and Ross (2024, *JEBO* 217: 378–397): 30 economists each judged five original and five different
  language-edited PhD papers, never both versions of one paper. "Our results show that writing matters." Economists
  judged edited versions higher in quality, more likely to be accepted for a conference, and more likely to be
  accepted at a good journal — [IDEAS abstract](https://ideas.repec.org/a/eee/jeborg/v217y2024icp378-397.html)
- A blog summary of the 2022 working-paper version read the effect as mattering mostly for below-average writing,
  which differs from the published abstract's unconditional wording (secondary; not verified against the paper) — [blog via search](https://sex-drugs-economics.blogspot.com/2022/03/the-importance-of-writing-quality-for.html); contrast [IDEAS abstract](https://ideas.repec.org/a/eee/jeborg/v217y2024icp378-397.html)
- Dowling, Hammami and Zreik (2018, *Economics Letters* 173: 100–103): "Ease of readability of Economics Letters
  abstracts ... is positively related to future citations", especially for "mathematical and quantitative methods and
  macroeconomics papers" (correlational) — [IDEAS](https://ideas.repec.org/a/eee/ecolet/v173y2018icp100-103.html)
- McCannon (2019, *Economics Letters* 180: 76–79): readability of AER introductions, 2000–2009, has "a statistically
  significant relationship" with citations. The abstract's effect size reads "up to 15 of a standard deviation", which
  looks garbled in the IDEAS text (correlational) — [IDEAS](https://ideas.repec.org/a/eee/ecolet/v180y2019icp76-79.html)
- Marino Fages (2020, *Scientometrics*): "in economics, there is also an effect of readability on the probability of
  publishing in a Top 5 journal"; earlier citation estimates are "downward biased" (observational) — [Springer](https://link.springer.com/article/10.1007/s11192-019-03332-4)
- Hengel ("Publishing While Female", *EJ* 2022): female-authored papers score as better written on readability
  measures, and the gap widens during peer review (from a search summary; paper not opened) — [UCL copy](https://discovery.ucl.ac.uk/10148349/1/publishing_female.pdf)
- Editorial authority: Harvey handled "approximately 7,500 submissions" as JF editor (2006–2012) — [Harvey, p. 1](https://faculty.fuqua.duke.edu/~charvey/Research/Working_Papers/W111_Reflections_on_editing.pdf)
- Stulz states as practice, not data, that referees may return badly exposed papers without a report — [Stulz Q9](https://www.wu.ac.at/fileadmin/wu/d/i/ifr/Stulz-Tips_for_authors.pdf)

### Inferences
- Grades for the borrow list, in the report's scheme:
  - The structure, template, table and figure rules: P.
  - "Plain, readable prose": D for the general effect (Feld et al., economics papers, economist judges), but not for
    any single rule.
  - The claims-scope rules (contractual introduction; no unsupported implications): already D for the failure mode
    through the report's Peters & Chin-Yee evidence (V7).
- Readability formulas measure word and sentence length. The correlational studies support shorter, plainer
  introductions and abstracts. They do not support a uniform style target, which V12 already bans as a global
  instruction.
- Where a guide rule conflicts with corpus practice, the report's ordering applies (§4.5: corpus wins on register;
  prescription wins on throat-clearing and self-praise). The tense rule for prior work (row 51), roadmap use (row 19)
  and stars (row 40) are register questions for the corpus. Previews and recalls, naked "this" and adjectives on own
  results are throat-clearing questions, where the guides can decide.

### Gaps
- No experiment isolates a single writing rule in economics or finance papers.
- Hengel (2022) and the 2022 Feld et al. working paper were seen only through search summaries.
- No evidence was found on JFEc-specific referee responses to writing quality.
