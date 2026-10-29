# Cadence, grammatical voice and stance in LLM prose vs human financial-econometrics prose: evidence, metrics and corrections

Research date: 2026-10-28. Scope: 2023-2026 evidence, plus older applied-linguistics work on research articles (RAs). Context: an automated prose scrub for a Journal of Financial Econometrics paper drafted mostly by Claude agents. The sibling note `llm_prose_markers.md` already covers excess vocabulary, em dashes, "not X but Y", triads, signposting, summary closers, the basic Reinhart et al. ratios and the burstiness dispute. The sibling `corpus_style_matching.md` covers pybiber, PassivePy, corpus percentiles and the Cochrane/Head/JFEc style rules. This note does not repeat them. It extends them on voice, person, sentence rhythm, stance (absolutes, boosters, hedges) and the "is this quantifier needed?" test.

Overall summary:
- **Voice.** The best quantitative evidence (Reinhart et al., PNAS 2025) shows instruction-tuned models write *fewer* agentless passives than humans (GPT-4o about half the human rate) and far more nominalizations and participial clauses. Base models stay close to human rates.
- **Person.** Studies of ChatGPT academic and essay prose find *fewer* self-mentions than humans, not more. The LLM substitute for the passive therefore looks like an inanimate or nominal subject with an active verb ("This specification captures...", "The analysis reveals..."), not "we". The draft must be measured directly, because Claude agents may also have been told to write "we".
- **Cadence.** Measured findings are narrower sentence-length distributions with a thinner right tail, register levelling, and reused part-of-speech templates. "Punch" sentences, colon reveals and paragraph-final summaries are catalogued, not measured.
- **Stance.** LLMs *overgeneralize* scope (Peters & Chin-Yee 2025: odds ratio 4.85 vs human summaries). RLHF reward models penalize hedges (Zhou et al. 2024). Lexical-booster counts ("clearly", "definitely") are not consistently higher in ChatGPT essays.
- **Correction.** Asking a model to continue a human exemplar "in the same style" does not close the grammatical gap. Measuring, then making targeted sentence-level edits, then re-measuring is the defensible design.

## Q1. Voice and person in LLM prose: Reinhart et al. (PNAS 2025) and related studies. Do instruction-tuned models prefer active "we" where academics use agentless passives? Do RLHF or style guides cause it?

### Takeaway
Instruction tuning, not pretraining, moves LLM grammar away from human registers. GPT-4o uses agentless passives at about half the human rate and present participial clauses at 5.3x. The models with base versions (Llama 3) stay near human rates in their base form. No study found that LLMs swap passives for "we". Comparisons of ChatGPT and human academic and essay prose report *fewer* self-mentions and less authorial presence in the ChatGPT text. Reinhart et al. do not identify the cause of the tuning effect. A separate ACL 2024 study does trace a stance bias (against hedges) to RLHF reward models and preference data. I found no evidence that Strunk & White-style "use the active voice" guidance in tuning causes the voice pattern.

### Cited Findings

**Reinhart, Markey, Laudenbach, Pantusen, Yurko, Weinberg & Brown, "Do LLMs write like humans? Variation in grammatical and rhetorical styles" (PNAS 122(8), Feb 2025; arXiv 2410.16107)**
- Design: each human text from the HAP-E corpus was split into two consecutive chunks of about 500 words at sentence boundaries. Each LLM got the first chunk and was asked to "write 500 more words in the same style, tone, and diction". Its output was compared with the human second chunk using Biber's feature set. Models: GPT-4o, GPT-4o Mini, and Llama 3 8B and 70B in base and instruct versions. Genres: academic, news, fiction, spoken, blogs, TV/movie scripts. — [arXiv 2410.16107v2 HTML](https://arxiv.org/html/2410.16107v2)
- GPT-4o rates relative to human, with paired Cohen's d (the sibling note lists the ratios without the d values):
  - present participial clauses 5.3x (d = 1.38);
  - "that" clauses as subject 2.6x (d = 0.77);
  - nominalizations 2.1x (d = 1.23);
  - phrasal coordination 1.9x (d = 0.81).

  — [arXiv 2410.16107v2 HTML](https://arxiv.org/html/2410.16107v2)
- **Voice:** GPT-4o uses agentless passives at "roughly half the rate as human texts." The main text does not report by-passives, first- and second-person pronouns, past participial clauses or that-complement clauses; they are in SI Tables S5-S6, which I could not open (PNAS returned 403, and the arXiv PDF text could not be extracted). — [arXiv 2410.16107v2 HTML](https://arxiv.org/html/2410.16107v2)
- **Genre:** the main text has no academic-only breakdown. On the M4 out-of-domain check (GPT-3.5 *arXiv abstracts*), classifiers stayed above chance for instruction-tuned models. Classifiers trained on Llama base outputs reached only 50%. — [arXiv 2410.16107v2 HTML](https://arxiv.org/html/2410.16107v2)
- **Base vs instruct:** "The Llama base models use these features at rates closer to humans." Instruction tuning produces the more extreme differences. — [arXiv 2410.16107v2 HTML](https://arxiv.org/html/2410.16107v2)
- **Cause:** the authors attribute the gap to instruction tuning, not the training texts. They do not name RLHF or style guides. They suggest human preference ratings, or tuning tasks such as summarization, and say tuning processes are not publicly documented, so the cause cannot be determined. — [arXiv 2410.16107v2 HTML](https://arxiv.org/html/2410.16107v2)
- **Classifiers:** a seven-way random forest reached 66% accuracy (chance 14%). Single-LLM-vs-human forests reached 93-98%. — [arXiv 2410.16107v2 HTML](https://arxiv.org/html/2410.16107v2)

**Self-mention and authorial presence in ChatGPT academic prose (applied linguistics, 2024-2026)**
- Pham (2026, *Indonesian Journal of Applied Linguistics*) compared ChatGPT-generated and human-written linguistics research articles. Hedges and boosters were common in both. The ChatGPT corpus "lacked self-mentions" and had fewer attitude markers. (Search-result summary; full text not read.) — [IJAL article page](https://ijal.upi.edu/index.php/ijal/article/view/101)
- Jiang & Hyland (2026, *Applied Linguistics Review*) compared British A-level argumentative essays with ChatGPT essays on the same topics. ChatGPT used significantly fewer epistemic and attitudinal stance markers and showed far less authorial presence. — [UEA eprint (docx)](https://ueaeprints.uea.ac.uk/103026/1/Stance_in_AI_and_student_essays.docx)
- Crosthwaite (Edge Hill research-group slides, 2026) compared BAWE human texts with several GenAI models. Human writers used more explicit stance and engagement features. The ChatGPT essays had a narrower, more repetitive range of lexical bundles. (Slides, not peer reviewed.) — [Edge Hill CRG slides](https://sites.edgehill.ac.uk/crg/files/2026/03/EHU-CRG.20.Crosthwaite.Slides.pdf)
- El-Dakhs, Afzaal & Siyanova-Chanturia (*Corpus Pragmatics*; catalog date possibly 2026, unverified) compared 320 abstracts, half ChatGPT and half human. ChatGPT gave less background and methods detail and overemphasized findings and conclusions. It used fewer transition markers and overused frame markers, attitude markers, and hedging and boosting verbs. (Attribution and figures from a search summary of the catalog record.) — [catalog record](https://ersearch2.cvtisr.sk/vufind/SummonRecord/FETCH-LOGICAL-c244t-abbf34b2aa81c4472ef9f703588e2b1621deb79416a625275c554b50cb5e47d23/Details?lng=en)

**RLHF and stance (cause evidence for an adjacent feature)**
- Zhou, Hwang, Ren & Sap (ACL 2024) found that RLHF models emit more strengtheners than weakeners, while base and instruction-tuned (SFT) models show the opposite. Reward-model average scores: plain statements 4.03, strengtheners 0.82, weakeners -1.86. In the human preference data, weakeners were preferred 8% less often than strengtheners and 9% less often than plain text. This is the clearest documented case of RLHF shaping a style feature. It concerns certainty marking, not voice. — [arXiv 2401.06730v2](https://arxiv.org/html/2401.06730v2)

**Style-guide pressure on the human side**
- Nature: an editor's blog says active voice "has been Nature policy for as long as I can remember; it is enshrined in our style manual and is specifically recommended to all authors as part of our standard acceptance procedure". Passive-voice papers appear "at the author's insistence". Nature's author pages are widely quoted as saying "Nature journals prefer authors to write in the active voice ('we performed the experiment...')". I saw that sentence only on secondary sites, not on nature.com. — [Nature blog, "Active is better than passive or neutral"](https://blogs.nature.com/blog/active_is_better_than_passive/); [Editage (secondary quote)](https://www.editage.com/insights/how-to-effectively-use-active-and-passive-voice-in-research-writing)
- CASRAI guidance: passive "remains the traditional and still-common choice in Methods sections", while active voice is the default elsewhere. — [CASRAI: active vs passive voice](https://www.casrai.org/dictionary/term/active-voice-vs-passive-voice)
- Cochrane's economics-specific rules (root out passives; no royal "we" for single authors; "we" means "you (the reader) and I") are in `corpus_style_matching.md`. — [Cochrane, Writing Tips for PhD Students (PDF)](https://static1.squarespace.com/static/5e6033a4ea02d801f37e15bb/t/5eda74919c44fa5f87452697/1591374993570/phd_paper_writing.pdf)

### Inferences
- The "half as many agentless passives" result plus "2x nominalizations" and "fewer self-mentions" point to one profile: inanimate or nominal subjects with active verbs. Examples: "The estimator achieves...", "This decomposition reveals...", "The analysis shows...", "Table 3 demonstrates...". The opposite profile, "We + verb" everywhere, is a plausible *Claude-agent* artifact if agent instructions told it to "use active voice and we". The literature does not establish it. The scrub should measure the *subject-type distribution* (Q4) on the actual draft before deciding which direction to correct.
- Reinhart's setup gave every model a 500-word human exemplar plus an explicit "same style, tone, and diction" instruction, and the gaps persisted. That is direct evidence that *exemplar continuation alone does not fix register-level grammar* (see Q5).
- Human editorial pressure (Nature, Cochrane) pushes toward active voice, but the literature keeps passives in methods (CASRAI; Q2). Active-everywhere is therefore not "human-like" for a methods section. The target should come from the JFEc corpus, not a style guide.
- RLHF demonstrably shapes stance markers (Zhou et al.). By analogy it may shape voice, but that is unproven.

### Gaps
- Reinhart SI Tables S5-S6 (by-passives, first-person pronouns, past participial clauses, per-model and per-genre rates) could not be retrieved. Whether GPT-4o's first-person rate is above or below human in the *academic* genre is unknown to me.
- No study found that directly tests "LLM replaces agentless passive with 'we' in methods sections".
- No study on Claude models specifically (any version) for passive share or self-mention rate was found.
- A search summary mentioned a 25-vs-25 study in which both ChatGPT and human abstracts leaned on active voice, with ChatGPT using more simple present and humans more simple past. I could not tie this to a specific source URL, so it is unverified.
- No evidence was found on whether tuning data or constitutions include "use active voice" instructions.

## Q2. How do human finance and econometrics papers distribute voice and person by section? Corpus studies of economics RAs

### Takeaway
No finance- or econometrics-specific corpus study of passive share by section was found. The general research-article literature agrees that passives cluster in methods sections (and in data and procedure descriptions), while introductions and results favor active voice. Science articles run at about 30% passive clauses (Leong 2014). Economics writers are *more* visible through authorial pronouns than linguists or engineers (Galaidin & Bednárová-Gibová 2023; BELLS 2024). Their "we" is mostly inclusive, meaning reader plus writer (Harwood 2005). The JFEc target must therefore be measured from the user's own corpus. The literature only says what shape to expect.

### Cited Findings
- **Leong (2014, JCOM):** 60 science articles from six journals. About 30% of all clauses were passive, and the author suggests the share may be stabilising near 30%. About 29% of all passives were in the methodology section. Passives remain widely used there despite advice favouring active voice. — [Leong, "The passive voice in scientific writing" (informalscience.org)](https://www.informalscience.org/passive-voice-scientific-writing-current-norm-science-journals)
- **Section contrast:** a 2024 *Journal of Language and Education* comparison of Indonesian and UK applied-linguistics journals (34 articles) looked at Introductions and Methods, on the premise that "the Introduction typically contains active voice while the Method section typically contains passive voice". Both groups used more passives in Methods than in Introductions. — [DOAJ record](https://doaj.org/article/d65ecf3c547f4b1385c9bd98904499f2)
- **Harwood (2005), *Applied Linguistics* 26(3): 343-375:** covers Business & Management, Computing, Economics and Physics. "All instances of we in the Business and Management articles and all but one of the instances of we in the Economics articles are inclusive." Only a third of Computing instances and under 10% of Physics instances are inclusive. — [Harwood 2005 PDF (ResearchGate)](https://www.researchgate.net/profile/Marcia-Lygia-Casarin/post/Should_I_use_either_we_or_I_in_my_thesis/attachment/59d61d9579197b8077977f87/AS%3A271736107405312%401441798344619/download/We+Do+Not+Seem+to+Have+a+Theory.pdf)
- **Harwood (2005), *Journal of Pragmatics* 37(8): 1207-1231:** a qualitative study of self-promotional I and we in the same four disciplines. "I and we can publicize the writer and their work." — [Benjamins/abstract listing](https://benjamins.com/online/ebop/publications/42568)
- **Galaidin & Bednárová-Gibová (2023):** linguistics vs economics RA abstracts by native and non-native writers. Economics abstracts use authorial pronouns much more often than linguistics abstracts, and Anglo-American writers are more visible. (Abstract-level.) — [UwB repository](https://repozytorium.uwb.edu.pl/jspui/handle/11320/15411)
- **BELLS (2024):** authorial presence markers in linguistics, economics and technology articles in English, Serbian and German. The English economics subcorpus holds almost half of all markers in the whole corpus. (Abstract-level.) — [Belgrade BELLS 2024 16(3)](https://belgrade-bells.fil.bg.ac.rs/index.php/belgrade-bells/article/view/bells-2024-16-3)
- **Hyland's stance model**, the standard framework for these counts: stance = hedges, boosters, attitude markers and self-mentions (I, we, my, our). Engagement = reader pronouns, directives, questions, shared knowledge and asides. — [U Alberta WAC quick guide](https://ualberta.ca/en/centre-for-teaching-and-learning/media-library/wac/quick-guides/wacuofaframingyourresearchforreadersinteractionengagementstance.pdf); [LSP Journal 2014 (spoken-discourse application)](https://rauli.cbs.dk/index.php/lspcog/article/view/4236)

### Inferences
- Expected JFEc pattern, a hypothesis for the corpus to test:
  - Introduction and results: "We find / We show / We document" for claims, inanimate subjects for exhibits ("Table 4 reports").
  - Data: agentless passives for provenance and cleaning ("Prices are obtained from...", "Quotes with zero bid are removed").
  - Methods: a mix. "We estimate" for authorial choices; passives for standard procedures and definitions ("The HAR model is estimated by OLS", "Standard errors are computed with...").
  - Abstract: high "we" density.
- Harwood's economics "we" is mostly inclusive ("we can see that", "as we show"). Modern multi-author finance papers mostly use exclusive "we" (the authors). Harwood's corpus is old (pre-2005), so its proportions should not be used as a target.
- The scrub's section-level targets should be corpus percentiles per section type, computed with the same pipeline (Q4) on 20-50 recent JFEc/JoE/JBES realized-volatility papers. `corpus_style_matching.md` covers building that corpus.

### Gaps
- No corpus study of passive share or self-mention rates in *finance* or *financial econometrics* RAs was found. The best economics-specific sources (Harwood, Galaidin & Bednárová-Gibová, BELLS) give per-discipline counts I could not read in full.
- Ebrahimi & Chan (2015) analyse the grammatical subject in applied-linguistics and economics abstracts, which is directly relevant to subject-type distribution. I found only a reference, not results.
- Hyland (2001), the foundational self-mention study, has no economics subcorpus as far as I could confirm. Its per-1,000-word tables were not retrieved.
- Medical-RA figures (median passive share of 20-26% of sentences; 43% in 297 medical RAs) appeared in a search summary without a traceable source.

## Q3. Cadence: LLM sentence-length distributions, rhythm devices and homogenisation. What is measured and what is only catalogued?

### Takeaway
Measured findings:
- LLM sentence lengths are *less dispersed* than human ones, with a thinner right tail (Muñoz-Ortiz et al.; Ju, Blix & Williams 2025).
- LLM text levels registers ("register leveling").
- LLM text reuses part-of-speech templates from pretraining far more than human text does (76% vs 35%; Shaib et al. 2024).
- Feedback-tuned models homogenise co-written text in content as well as words (Padmakumar & He 2024).
- RLHF reward models are length-biased.

Not measured in any study I found, only catalogued: alternating long/short "punch" sentences, colon or dash "reveals", paragraph-final "This matters because" sentences, sentence-initial repetition, and anaphora/parallelism. Triads and "not X but Y" are catalogued, and the latter is counted in Antislop (see the sibling note). The burstiness metric dispute is also in the sibling note.

### Cited Findings
- **Muñoz-Ortiz, Gómez-Rodríguez & Vilares, "Contrasting Linguistic Patterns in Human and LLM-Generated News Text" (arXiv 2308.09067):**
  - "Human texts exhibit more scattered sentence length distributions, more variety of vocabulary, a distinct use of dependency and constituent types, shorter constituents, and more optimized dependency distances."
  - The six LLMs "exhibit a similar distribution across different sentence lengths, presenting less variation when compared to human-generated sentences."
  - The authors suggest "register leveling", where genre-distinctive features become less pronounced.
  - Domain: news. — [arXiv 2308.09067](https://arxiv.org/pdf/2308.09067)
- **Ju, Blix & Williams, "Domain Regeneration: How well do LLMs match syntactic properties of text domains?" (arXiv 2505.07784, 2025):** regenerated text has narrower distributions, "the original human distributions are also wider (i.e., the regenerated distributions have less variance) for all data sources". The human distributions "appear to have a longer and heavier right tail than their model generated counterparts." (Quotes via search snippet; PDF text extraction failed.) — [arXiv 2505.07784](https://arxiv.org/pdf/2505.07784)
- **"Temporal Flattening in LLM-Generated Text" (arXiv 2604.12097, 2026):** average sentence length gave the strongest signal in its classifier. Its focus is year-to-year variability within an author, so it is only partly relevant. (Snippet only.) — [arXiv 2604.12097](https://arxiv.org/pdf/2604.12097)
- **Shaib et al., "Detection and Measurement of Syntactic Templates in Generated Text" (EMNLP 2024, pp. 6416-6431):**
  - 76% of templates in model-generated text also appear in pretraining data, against 35% for human-authored text.
  - Templates "are not overwritten during fine-tuning".
  - Example: in an OLMo-Instruct sample, 33 of 35 POS sequences (95%) matched the Dolma pretraining corpus. — [ACL Anthology PDF](https://aclanthology.org/2024.emnlp-main.368.pdf); [arXiv 2407.00211v1](https://arxiv.org/html/2407.00211v1)
- **Padmakumar & He, "Does Writing with Language Models Reduce Content Diversity?" (ICLR 2024; arXiv 2309.05196):**
  - Design: 38 Upwork writers wrote argumentative essays with GPT-3 (base), with InstructGPT (feedback-tuned), or alone.
  - Only InstructGPT produced a statistically significant reduction in diversity, including diversity of *key points*, not just vocabulary. GPT-3-assisted essays did not differ from solo essays.
  - The effect is "mainly attributable to InstructGPT contributing less diverse text".
  - The authors link it to prior findings that human-feedback tuning lowers output entropy. — [ICLR 2024 proceedings](https://proceedings.iclr.cc/paper_files/paper/2024/hash/02dec8877fb7c6aa9a79f81661baca7c-Abstract-Conference.html); [arXiv 2309.05196v3](https://arxiv.org/html/2309.05196v3)
- **Length and verbosity bias from preference tuning:**
  - Reward models show "a systematic tendency to favor longer responses by conflating verbosity with quality". — [Kim et al., arXiv 2511.12573](https://arxiv.org/html/2511.12573v1)
  - RLHF-tuned models "generate significantly longer answers" without a matching quality gain. — [Park et al., "Disentangling Length from Quality in DPO", arXiv 2403.19159](https://arxiv.org/html/2403.19159v2)
  - "Do Chatbot LLMs Talk Too Much?" (arXiv 2601.00624, 2026) finds an order-of-magnitude spread in median excess length across assistants. Failure modes include "vacuum-filling" on ambiguous inputs and explanation overhead. — [arXiv 2601.00624](https://arxiv.org/html/2601.00624v1)

### Inferences
- Reading "certain cadence" through the measured findings: medium-length sentences cluster, few very long ones occur (thin right tail), and clause architecture repeats (templates, participial tails). In econometrics prose, the human right tail is the long definitional or conditional sentence ("Under assumptions A1-A3, and provided that..."). The human short sentences are mostly terse exhibit pointers, not rhetorical punches.
- The "long sentence, then short punchy sentence" pattern that readers notice in Claude prose is plausible, but I found no published measurement of it. It can be measured cheaply (Q4: short-sentence share, alternation index, short-sentence position in the paragraph). Calibrate against the corpus before flagging.
- Register levelling matters most here. A JFEc methods section and an introduction should differ in voice mix and sentence length. LLM drafts may show *less between-section variation* than human papers. This is testable by comparing the spread of section-level metrics in the draft and in corpus papers.

### Gaps
- No peer-reviewed study measures in LLM vs human academic prose:
  - "punch" short sentences or long/short alternation;
  - colon or dash reveals;
  - paragraph-final summary or "This matters because" sentences;
  - sentence-initial repetition ("This...", "This...").

  These are catalogue-level only (Wikipedia "Signs of AI writing", practitioner lists; see sibling note).
- Sentence-length autocorrelation in LLM vs human prose: no study found. Long-range-correlation work exists for literary texts, but I did not verify any LLM comparison.
- All measured sentence-length findings are from news or mixed web domains, not research articles.

## Q4. Measurement: which voice and cadence metrics can be computed on a LaTeX draft, and which are robust at about 10k words?

### Takeaway
Use per-sentence or per-1,000-word *rates*. Compare each with corpus percentiles computed with the same pipeline. Stability is governed by event counts:
- Draft-level passive share and "we" rate are stable at 10k words. Passive share is about ±4 pp at 400 sentences.
- Section-level passive share is noisy, about ±11 pp at 60 sentences.
- Rare Biber features (by-passives, that-clauses as subject) and lag-1 sentence-length autocorrelation are informative only at draft level or across a corpus.
- Sentence-length dispersion must be length-normalised: report the coefficient of variation and quantiles, not the raw SD. Text length is a known confounder.

### Cited Findings
- Text length is "an unavoidable confounder in quantitative text-analytic corpus-linguistic studies" (Liimatta 2024, *Challenges in Corpus Linguistics*). — [Liimatta, "Text length and short texts" (De Gruyter Brill)](https://www.degruyterbrill.com/document/doi/10.1075/scl.118.07lii/html)
- Biber (1990, *Literary and Linguistic Computing* 5(4): 257-269) studies corpus design issues, including sample size per text, and concludes that existing corpora are adequate for many analyses of variation. I could not verify the often-cited claim that 1,000-word samples reliably represent common features. — [NAU record](https://experts.nau.edu/en/publications/methodological-issues-regarding-corpus-based-analyses-of-linguist/); [LLC 5(4) issue](https://academic.oup.com/dsh/issue/5/4)
- Templates: Shaib et al. measure "templatedness" as repeated POS n-gram sequences. This is computable on one draft with a POS tagger. — [arXiv 2407.00211v2](https://arxiv.org/html/2407.00211v2)
- Already covered in the siblings, not repeated here:
  - PassivePy accuracy (about 98% agreement with human coding);
  - pybiber's 67 features and the MDA;
  - the percentile-band approach;
  - the length-confounded burstiness critique (NeurIPS 2026);
  - the Antislop "not X but Y" regex. — see `corpus_style_matching.md` Q1 and `llm_prose_markers.md` Q2.

### Inferences

Proposed metric suite. Compute on detexed text with math, citations, `\ref` and table bodies stripped. Report draft-level values and per-section values (Intro, Data, Methods, Results, Conclusion) against corpus p10-p90 bands.

| Metric | Implementation | 10k-word robustness | Notes |
|---|---|---|---|
| Passive share of finite clauses, split agentless vs by-passive | PassivePy, or spaCy deps (`nsubjpass`/`auxpass`; `agent` child = by-phrase) | Good at draft level (n≈400 sentences, p≈0.25: SE≈2.2 pp). Weak per section (n≈60: SE≈5.6 pp) | Reinhart: LLM agentless passive about 0.5x human |
| Self-mention rate ("we/our/us" per 1k words), split by verb class (find/show vs estimate/compute) | regex + spaCy head verb | Good (likely 50-150 tokens per 10k) | Literature: LLMs under-mention; Claude agents may over-mention |
| Subject-type distribution of main clauses: we / passive subject / inanimate concrete NP (table, model, estimator) / nominalization head (-tion, -ment, -ity) / dummy it/there / "This/These" + noun | spaCy `nsubj` head lemma + suffix and word lists | Good at draft level for the 3-4 largest categories | The key metric for the "voice" impression (Q1 inference) |
| Sentence length: mean, CV (SD/mean), p10/p50/p90, share > 40 words, share < 10 words | sentence splitter after detex | Good for mean, quantiles and CV. Tail shares need about 300+ sentences | Measured LLM signature: narrow spread, thin right tail |
| Long/short alternation: lag-1 autocorrelation of length; share of sign changes in successive length differences; position of short sentences (paragraph-final share) | arithmetic on the length sequence | Lag-1 ρ SE ≈ 1/√n ≈ 0.05 at n=400, so only abs(ρ) > 0.1 is detectable. Useless per paragraph | Uncalibrated: no published LLM baseline |
| Sentence-opening distribution: entropy of first token and first POS bigram; share opening with "This/These", "We", discourse adverbials; same-opener runs in consecutive sentences | spaCy | Good at draft level | Catalogued sign (sentence-initial repetition) |
| POS-template repetition: share of POS 5- or 6-grams occurring 3+ times in the draft | spaCy POS + n-gram counts | Fair. Compare with corpus papers of equal length | Shaib et al. |
| Present participial tails, phrasal coordination, nominalization density | pybiber, or spaCy rules | Good (frequent features) | Reinhart ratios 5.3x / 1.9x / 2.1x. Econometrics is nominal-heavy, so the corpus baseline is essential |
| Rare Biber features (that-clause as subject, by-passive) | pybiber | Poor: ~10 events per 10k words, Poisson SE ~30% | Report, do not gate |
| Absolutes, superlatives, boosters, hedges per 1k words (see Q6) | lexicon + spaCy determiner/adverb parse | Good for the pooled class, poor per item | Gate on needed/true checks, not rates |

- Every number above that depends on n is a standard binomial or Poisson approximation, not a figure from a source. Corpus percentiles should be computed per paper, then the draft placed in that distribution. Pooled sentences overstate precision (also noted in `corpus_style_matching.md`).
- The CV of sentence length, together with the share of sentences over 40 words, sidesteps the length-confound critique of raw-SD burstiness.

### Gaps
- No validation study of PassivePy or spaCy passive detection on *LaTeX-derived econometrics prose* was found. Math-heavy sentences ("$\hat\beta$ is estimated...") may parse badly, so the detex step needs spot checks.
- No published human baseline exists for alternation or autocorrelation in RAs.
- I did not verify whether Shaib et al. ship a reusable package for template counting.

## Q5. Correction: does instructing a model to match a target passive share or sentence-length distribution work? Exemplar-based vs constraint-based rewriting, and meaning change

### Takeaway
Global instructions and exemplars are weak tools for grammar-level register:
- Reinhart's models had a human exemplar plus a "same style" instruction and still diverged.
- Em dashes persist under explicit suppression (sibling note).
- Instructions *can* move stance-marker frequencies a lot, but they overshoot (Zhou et al.: epistemic markers go from 5% to 71% of answers when requested).
- Trained per-axis rewriters work for voice and length (StyleRemix), but need open-weight fine-tuning.
- Decoding-time bans work for lexical patterns (Antislop, sibling note).

For a Claude-API pipeline the defensible design is measure, then flag specific sentences, then apply explicit sentence-level operations, then re-measure, then check meaning. I found no direct study of meaning drift in voice conversion in academic prose.

### Cited Findings
- **Exemplar plus instruction does not converge grammar:** Reinhart's generation prompt was the human first chunk plus "write 500 more words in the same style, tone, and diction". Instruction-tuned outputs still had 5.3x participial clauses and half the agentless passives (GPT-4o). — [arXiv 2410.16107v2](https://arxiv.org/html/2410.16107v2)
- **Instructions move stance frequencies, with overshoot:** epistemic markers appeared in about 5-6% of answers with a base prompt, 16% with chain-of-thought, and 71% when the prompt explicitly asked for certainty expressions. — [Zhou et al., arXiv 2401.06730v2](https://arxiv.org/html/2401.06730v2)
- **Accuracy instructions do not stop overgeneralization:** "Even when explicitly prompted for accuracy, most LLMs produced broader generalizations." — [Peters & Chin-Yee, arXiv 2504.00025](https://arxiv.org/abs/2504.00025)
- **StyleRemix (Fisher et al., EMNLP 2024):**
  - LoRA adapters are trained per style axis and direction. There are seven axes, including length, formality, grade level, sarcasm, writing type and "voice (passive or active)", for 16 style elements.
  - Adapters are merged at inference to rewrite text.
  - Training data: the DiSC parallel dataset, 1,500 texts rewritten in 16 directions (24K total).
  - The authors report that it beats baselines and much larger LLMs on authorship obfuscation, by automatic and human evaluation. — [arXiv 2408.15666](https://arxiv.org/html/2408.15666v1); [ACL Anthology 2024.emnlp-main.241](https://aclanthology.org/2024.emnlp-main.241)
- Covered in the siblings, not repeated:
  - Antislop decoding bans;
  - Freeburg's em-dash persistence under instruction;
  - the "Can You Make It Sound Like You?" post-editing study;
  - phrase-frame retrieval rather than passage imitation (`corpus_style_matching.md` Q3).

### Inferences
- **Do not instruct a global target** such as "use 25% passives" or "vary sentence length". Models cannot count their own clauses reliably, and Zhou et al. show requested features overshoot. Instead:
  1. Compute the metrics (Q4) and find the sections outside the corpus band.
  2. Pick the specific sentences driving the deviation, for example methods sentences with "we" plus a routine procedural verb (compute, winsorize, obtain, construct), or inanimate-subject sentences with evaluative verbs (reveals, demonstrates, underscores).
  3. Send each flagged sentence with its paragraph and a *named operation*:
     - "convert to agentless passive";
     - "replace the inanimate agent with 'we' or a passive";
     - "merge with the next sentence";
     - "split at the conjunction";
     - "delete the participial tail";
     - "delete the quantifier" (Q7).
  4. Re-measure, and stop when the section is inside the band. Do not aim for the corpus median, which would itself be a homogenisation.
- **Section-specific voice rules** (to be confirmed against the corpus):
  - Data and procedure sentences: agentless passive is acceptable. "Returns are sampled at 5-minute frequency" is better than "We sample returns...". Repeated "we + routine verb" runs are a scrub target.
  - Authorial choices and findings keep "we": "We choose the 30-minute window because...", "We find that...".
  - Exhibit pointers use concrete inanimate subjects: "Table 4 reports...".
  - Evaluative verbs with inanimate subjects ("The results underscore/highlight/reveal...") are the LLM signature. Rewrite them to state the number.
- **Meaning-preservation checks after each edit** (mechanical, cheap):
  1. The multiset of numerals, LaTeX macros, `\cite` keys and `\ref` labels is unchanged.
  2. Negations and modal verbs are unchanged unless the operation targeted them.
  3. For voice conversions, the agent was not invented. A passive-to-active rewrite must not attribute an action to "we" that the data vendor or a cited paper did ("Data are provided by OptionMetrics" must not become "We provide data...").
  4. Optionally, a bidirectional NLI or LLM-judge equivalence check on the sentence pair given its paragraph.
- Exemplars still help as *frames* (retrieved corpus sentences of the same function, per `corpus_style_matching.md`), not as continuation prompts.

### Gaps
- No study was found that evaluates LLM rewriting toward a *numeric* target (passive share, sentence-length quantiles) in academic prose, or that measures how often voice conversion changes meaning or agency.
- StyleRemix's per-axis accuracy for the voice axis specifically was not checked. Its domain is general text, not RAs.
- Whether Claude follows sentence-level operations more reliably than global style instructions is my inference, not a published result.

## Q6. Absolutes, superlatives, boosters and hedges: evidence that LLM prose overuses universals and boosters or underuses hedges; detection; literal vs rhetorical universals (scope addition)

### Takeaway
The strongest evidence concerns *scope inflation*, not booster words:
- LLM summaries of science are about five times as likely as human summaries to contain broad generalizations (Peters & Chin-Yee 2025, OR 4.85, 95% CI [3.06, 7.70]). Newer models do worse, and accuracy prompts do not stop it.
- RLHF reward models prefer plain or certain statements to hedged ones (Zhou et al. 2024). They push toward unhedged categorical assertions.
- Lexical-booster counts are mixed. ChatGPT-3.5 student-style essays used hedges such as "may" heavily and no "clearly/definitely". ChatGPT abstracts overused boosting and hedging verbs and attitude markers.

No corpus study counting "always/never/every/all/entirely/fundamentally/the most" in LLM vs human academic prose was found. The economics hedging literature exists (Varttala 2001; economics-abstract studies) but gives no finance-specific numbers I could verify. For this paper, the decisive design point is the split between *literal, checkable universals* (true only if a computed count equals the total) and *rhetorical absolutes* (Q7).

### Cited Findings

**Overgeneralization**
- **Peters & Chin-Yee, "Generalization bias in large language model summarization of scientific research" (*Royal Society Open Science* 12(4): 241776, April 2025; arXiv 2504.00025, 28 Mar 2025):**
  - Design: 10 LLMs, including ChatGPT-4o, ChatGPT-4.5, DeepSeek, LLaMA 3.3 70B and Claude 3.7 Sonnet. 4,900 summaries were compared with their source texts.
  - DeepSeek, ChatGPT-4o and LLaMA 3.3 70B overgeneralized "in 26 to 73% of cases".
  - LLM summaries "were nearly five times more likely to contain broad generalizations (OR = 4.85, 95% CI [3.06, 7.70])" than human-written summaries.
  - "Newer models tended to perform worse in generalization accuracy than earlier ones."
  - The mechanism named in the abstract: LLMs "may omit details that limit the scope of research conclusions".
  - Suggested mitigations: lower temperature, and benchmarking for generalization accuracy. — [arXiv 2504.00025](https://arxiv.org/abs/2504.00025); [journal record (DOAJ mirror)](https://library.kab.ac.ug/Record/doaj-art-0cfbd3437abe465db828960ce8924d2e/Details)
  - Explanations in the press release: Chin-Yee suggested models reproduce overgeneralization already common in science writing. Peters suggested models favour fluency and generality because users prefer helpful-sounding answers. — [Healthcare in Europe (Utrecht press release)](https://healthcare-in-europe.com/en/news/generative-ai-llm-exaggeration-science.html)

**RLHF and certainty**
- Strengtheners appeared in 20% of generations on average, weakeners in 14%. Six of nine models used strengtheners significantly more than weakeners.
- Only 53% of strengthener-marked answers were correct.
- RLHF models flip the base/SFT pattern toward strengtheners. Reward models score plain 4.03 > strengthener 0.82 > weakener -1.86. — [Zhou et al., ACL 2024, arXiv 2401.06730v2](https://arxiv.org/html/2401.06730v2)

**LLM vs human stance-marker counts (mixed)**
- Shalevska (2024) compared 100 ChatGPT-3.5 essays with 100 human essays in AntConc. The AI texts used some hedges ("may") at high frequency and had *no* boosters like "clearly" or "definitely". Human texts used a wider range of both. (The citation's volume and issue are inconsistent across listings.) — [UKLO eprint PDF](https://eprints.uklo.edu.mk/10825/1/HEDGES%2BAND%2BBOOSTERS%2BIN%2BAI%2BAND%2BHUMAN%2BWRITING%2BA%2BCOMPARATIVE%2BANALYSIS.pdf)
- El-Dakhs et al. (abstracts): ChatGPT overused attitude markers and hedging and boosting verbs, and overemphasized findings. — [catalog record](https://ersearch2.cvtisr.sk/vufind/SummonRecord/FETCH-LOGICAL-c244t-abbf34b2aa81c4472ef9f703588e2b1621deb79416a625275c554b50cb5e47d23/Details?lng=en)
- Jiang & Hyland (2026): fewer epistemic and attitudinal stance markers in ChatGPT essays. — [UEA eprint](https://ueaeprints.uea.ac.uk/103026/1/Stance_in_AI_and_student_essays.docx)
- Pham (2026): hedges and boosters common in both ChatGPT and human linguistics RAs. — [IJAL](https://ijal.upi.edu/index.php/ijal/article/view/101)

**Human baseline: hype and hedging in research writing**
- Hype adjectives in NIH grant-application abstracts: 72% contained at least one in 1985, 97% in 2020 (Millar et al., JAMA Netw Open 2022). A companion study finds hype rising in abstracts of NIH-funded publications, 1985-2020. Human academic prose is itself drifting toward promotion, so a corpus baseline must be recent. — [PMC9412227](https://pmc.ncbi.nlm.nih.gov/articles/PMC9412227); [PMC10739072](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10739072/)
- Economics hedging:
  - Varttala (2001, doctoral dissertation, Tampere) studies hedging in research articles in economics, medicine and technology. — [Tampere repository PDF](https://trepo.tuni.fi/bitstream/10024/67148/1/951-44-5195-3.pdf)
  - A corpus study compares lexical hedges in 100 British and 100 American economics RA abstracts from five leading journals (2019). — [Philology journal (2019)](https://philology-journal.ru/en/article/phil20190213/fulltext)
  - A 75-article study across economics, linguistics, medicine and natural sciences/engineering finds "noticeable disciplinary variation" in hedging through modal auxiliaries. — [Jurnal Humaniora](https://jurnal.ugm.ac.id/jurnal-humaniora/article/view/11412)
  - (All abstract-level; I did not read the frequency tables.)

### Inferences

**Detection (b):**
- Build a lexicon in four classes and parse each hit:
  1. Universal quantifiers and determiners: all, every, each, any, no, none, always, never, everywhere, without exception, uniformly, invariably.
  2. Completeness adverbs: entirely, completely, fully, wholly, totally.
  3. Superlatives and extreme degree: "the most/least", -est adjectives, best, largest, strongest, unique(ly), first, only.
  4. Boosters and emphatics: clearly, certainly, undoubtedly, crucially, fundamentally, importantly, indeed, of course, obviously, strongly, highly, very, remarkably, striking(ly), substantial(ly), dramatic(ally).
- Use spaCy to attach each hit to its scope: the quantified noun phrase (the `det`/`advmod` head) and the predicate.
- Add *generic-scope* detectors for the Peters & Chin-Yee failure mode:
  - bare plural or generic subjects in present tense where the evidence is sample-specific ("Option markets price variance risk..." vs "In our 2016-2024 SPX sample, ...");
  - dropped sample qualifiers in the abstract and conclusion compared with the results section;
  - modal-free claims about out-of-sample generality.
- Rates per 1k words should be reported against the corpus band. Gating should run on the per-instance tests below, not on rates.

**Literal vs rhetorical (c):**
- A *literal universal or superlative* asserts a checkable fact about a defined finite set: bars, days, classes, horizons, specifications, test results. It is TRUE only if a computed fact supports it:
  - "every bar of the sample is scored" ⇔ `n_scored == n_bars`;
  - "all six classes beat the benchmark" ⇔ `count(class: loss < bench) == 6` (with the significance qualifier, if any, matching the test);
  - "the largest gain is at the 30-minute horizon" ⇔ `argmax` over the reported table equals that horizon.
- In the paper pipeline this means generated macros, such as `\nScoredBars{}`/`\nBars{}` or a boolean `\allClassesBeat{}`, emitted by the results code. The scrub checks that each literal universal or superlative sentence cites such a macro or a table cell (see `results_traceability.md`).
- A *rhetorical absolute* ("fundamentally changes", "completely different", "always important", "the most natural approach") has no finite set behind it and no computation can back it. Delete it or replace it with a number.
- Hedging norms for finance and econometrics (inferred from the general RA literature, not from finance-specific sources):
  - Findings are scoped to sample, period, frequency and test ("at the 5% level", "in our sample", "for horizons up to one day").
  - Modal hedges ("may", "suggests") are reserved for interpretation and mechanism, not for the reported statistics themselves.
  - The scrub should therefore flag both directions: unhedged mechanism claims, and hedged statements of computed facts ("the loss appears to be lower" when the table shows it is).

### Gaps
- No corpus study of universal quantifiers, completeness adverbs or superlatives in LLM vs human *academic* prose was found. The author's observation is plausible and consistent with Peters & Chin-Yee and Zhou et al., but unmeasured.
- Peters & Chin-Yee per-model rates (including Claude 3.7 Sonnet) and the coding scheme for "generalization" were not retrieved. Secondary accounts claim accuracy prompts *increased* overgeneralization. The abstract says only that they did not prevent it, so that claim is unverified.
- Hyland's per-1,000-word hedge and booster tables by discipline (1998, 2005) were not retrieved. No finance or econometrics hedging corpus study was found.
- Ebrahimi & Chan and the economics-abstract hedging studies were read at abstract level only.

## Q7. The "NEEDED?" test: when a true universal or superlative adds nothing, and how to detect "implied by context" automatically (refinement)

### Takeaway
The principle is Grice's maxim of Quantity, second half: "Do not make your contribution more informative than is required."
- Editorial support is direct. Cochrane bans adjectives on one's own results ("striking", "very significant"), "In other words", and previews and recalls. The editorial literature favours scope-specific statements over emphasis.
- Empirical evidence that redundant boosters *reduce* credibility is weak or null. A hype RCT found no bias in students' merit ratings. Hedging-trust experiments are mixed and fragile in replication.
- The case for deletion therefore rests on information and on reader inference, not on measured credibility loss. A marked universal where none is needed invites the reader to infer a contrast (that some units might *not* have been covered) that does not exist.
- LLMs are pushed toward this redundancy: preference tuning rewards length and certainty, and assistants show measured excess length.
- Detection is feasible in two stages:
  1. a cheap structural test (does the quantifier's scope match a set already fully specified in the same or previous sentence, a definition, or a table?);
  2. a deletion-equivalence judgment (does removing the word change what the reader learns, given the paragraph and the definitions?).

  Only quantifiers that pass NEEDED go on to the TRUE check (Q6).

### Cited Findings
- **Grice's maxim of Quantity:**
  - The maxim has two submaxims: make your contribution as informative as required for the current purposes of the exchange, and do not make it more informative than required. — [Edinburgh pragmatics course: Grice's maxims](https://lel.ed.ac.uk/~hrohde/plc/implicature-3grice.html)
  - Some sources render it as "neither more nor less informative than is required". — [Encyclopedia.com: Conversational Implicature](https://www.encyclopedia.com/humanities/encyclopedias-almanacs-transcripts-and-maps/conversational-implicature)
  - Speakers can flout Quantity deliberately to generate implicatures, as in "damning with faint praise". — [Edinburgh pragmatics course](https://lel.ed.ac.uk/~hrohde/plc/implicature-3grice.html)
- **Economics editorial rules against emphasis and restatement** (Cochrane; full list in `corpus_style_matching.md` Q5):
  - no adjectives on one's own work ("striking results", "very significant");
  - "In other words" is "a sign of trouble";
  - previews and recalls signal poor organisation;
  - "It should be noted that" is "particularly obnoxious";
  - "There should be nothing before the main result that a reader does not need to know". — [Cochrane, Writing Tips for PhD Students (PDF)](https://static1.squarespace.com/static/5e6033a4ea02d801f37e15bb/t/5eda74919c44fa5f87452697/1591374993570/phd_paper_writing.pdf)
- **Credibility effects (weak evidence):**
  - A randomized trial (Peer Review Congress) gave students hyped and unhyped biomedical abstracts, with hype phrases such as "carefully designed" and "convincing evidence". Hype "did not appear to bias readers' evaluation of the scientific merit". One version of the project was listed as ongoing. — [Peer Review Congress abstract](https://peerreviewcongress.org/abstract/the-influence-of-promotional-language-on-evaluations-of-biomedical-literaturea-randomized-controlled-trial/)
  - Jensen (2008) found that hedged cancer-research news (study limitations reported) made both scientists and journalists seem more trustworthy (N=601, college students). — [Jensen 2008 (Houston Methodist record)](https://scholars.houstonmethodist.org/en/publications/including-limitations-in-news-coverage-of-cancer-research-effects/)
  - Ratcliff et al. (2018) reportedly found no effect of uncertainty on scientist credibility in a larger sample. — [Ratcliff et al. (SFU)](https://summit.sfu.ca/_flysystem/fedora/2024-07/Ratcliff-et-al_Covid-Uncertainty-Disclosure.pdf)
  - Gustafson & Rice (N=2,247) found only the "consensus uncertainty" frame had small negative effects. — [Gustafson & Rice (PDF)](https://rrice.faculty.comm.ucsb.edu/A145GustafsonRice2019UncertaintyFramesScienceCommunicationTopics.pdf)
- **Why LLMs add redundant emphasis or scope:**
  - Reward models conflate verbosity with quality. — [arXiv 2511.12573](https://arxiv.org/html/2511.12573v1)
  - RLHF lengthens answers without a matching quality gain. — [arXiv 2403.19159](https://arxiv.org/html/2403.19159v2)
  - Assistants show measured excess length beyond a "minimal sufficient answer", including explanation overhead. — [arXiv 2601.00624](https://arxiv.org/html/2601.00624v1)
  - Reward models favour certainty over hedges. — [arXiv 2401.06730v2](https://arxiv.org/html/2401.06730v2)
  - LLMs drop scope limits when summarizing. — [arXiv 2504.00025](https://arxiv.org/abs/2504.00025)

### Inferences

**Why "true but unneeded" hurts** (reasoning from Quantity, not a measured effect):
- In expository prose a quantifier is informative when it contrasts with an expectation of exceptions. Examples: "every bar, including half-days and FOMC afternoons, is scored"; "all six classes beat the benchmark, although three were tuned on a different window"; "never" where readers expect occasional failures.
- When the method has already *defined* full coverage, adding "every" either says nothing or prompts the reader to ask what the exceptions could have been. Both cost attention.
- Superlatives ("the most robust") and completeness adverbs ("fully", "entirely") add an evaluative layer the table already shows.

**Two-test gate, in order:**
1. **NEEDED?** Keep the quantifier or superlative only if one of these holds:
   - (a) it narrows or contrasts scope against a plausible expectation (exceptions matter);
   - (b) the reader cannot get the scope from the definitions, the table or the previous two sentences;
   - (c) it is the paper's claim itself and is stated once, in the abstract, introduction or conclusion, with a number.
   Otherwise delete it, even if it is true.
2. **TRUE?** If kept, it must point to a computed fact (macro or table cell) per Q6.

**Automatic detection of "implied by context":**
- **Stage A, structural and cheap.** For each hit from the Q6 lexicon:
  - Extract the scope NP head (for example "bars", "classes", "horizons", "days", "specifications").
  - Search a *set registry* built from the paper: noun phrases defined with a cardinality or enumeration, such as "the six classes (HAR, ...)", "the sample of N bars", "horizons h ∈ {…}", table row and column headers, and `\label`ed definitions.
  - Search the same sentence and the previous two sentences for the same head noun with a numeral, an enumeration or a definite back-reference ("these six classes", "the sample").
  - If the scope NP refers to a registered set and no contrastive marker is present nearby ("including", "even", "although", "except", "despite", "with no exception for"), flag the hit as **REDUNDANT-CANDIDATE**. Superlatives that restate the argmax of a table cited in the same sentence get the same flag.
- **Stage B, deletion-equivalence judge**, run on candidates only, plus all completeness adverbs and boosters:
  - Give an LLM (or a bidirectional NLI model) the sentence, the version with the word deleted, the paragraph, and the relevant definitions or table caption.
  - Ask whether a careful reader learns anything different from the two versions, and if so what. Require a concrete answer naming the exception or contrast the word excludes.
  - "No difference" means delete. "Difference" passes the word to the TRUE check.
  - Bidirectional entailment between the two versions given the context is the formal criterion. The LLM judge is the practical implementation for math-heavy text where NLI models are unreliable.
- **Boosters** ("clearly", "crucially", "fundamentally", "importantly", "indeed", "very") almost never carry propositional content in results prose. Default them to delete, unless the deletion judge identifies a contrast. Cochrane's rules support this.
- **Logging:** record each deletion with its reason code (REDUNDANT-SCOPE, BOOSTER, SUPERLATIVE-RESTATES-TABLE, UNSUPPORTED-UNIVERSAL) so the author can audit the scrub.

### Gaps
- No experimental or corpus evidence was found that *redundant universal quantifiers* specifically reduce readability or perceived credibility in academic prose. The closest evidence (hype RCT, hedging-trust experiments) is null or mixed and comes from health news or biomedical abstracts.
- No study was found measuring LLM tendencies to restate already-established scope ("redundant emphasis") as a distinct phenomenon. The support is indirect (verbosity and length bias, certainty bias, catalogued "unnecessary exposition").
- No published system implementing deletion-equivalence or entailment tests for quantifier redundancy was found. The Stage A/B design above is my proposal.
- The exact SEP wording of Grice's maxims was not retrieved; the citations are a university course page and Encyclopedia.com.
- McCloskey's *Economical Writing* rules on intensifiers were not consulted directly.
