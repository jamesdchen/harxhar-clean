# AI-based paper review ("AI referee") tools and the evidence on their usefulness

Research date: 2026-10-28. Scope is 2023-2026. Many 2026 items are company blogs, X posts or arXiv preprints. Each one is labeled below. Vendor-reported numbers are marked **[vendor]**.

## Q1. What AI review systems exist, and how are they built (single model vs multi-agent, rubrics, verification)?

### Takeaway
The field has moved from single-prompt GPT-4 reviews (2023) to staged, tool-using, multi-model pipelines (2025-2026). These pipelines split the review by aspect, ground it in retrieved literature or code, and add a self-critique or second-model verification pass before output. Refine is the only system aimed at economics. As of August 2026 the AEA and the Econometric Society use it at the R&R and conditional-acceptance stages. No tool aimed at financial econometrics was found.

### Cited Findings

**Refine (refine.ink): economics, math and theory**
- Co-founded by Ben Golub (Northwestern economics) and Yann Calvó López. It flags "errors in logic and math, empirical strategy problems, clarity issues, and consistency issues." Golub describes its reports as roughly the level of a better-than-average PhD student. He recommends running it "before you circulate or submit papers" and asking for at most 10 comments per paper. Golub is a co-founder, so this source is not independent. — [Golub, Markus' Academy ep. 154, 2025-12-18](https://markusacademy.substack.com/p/modern-ai-for-economics-research)
- Stated limitation for empirical work (Dec 2025): "tables and figures often cannot be appropriately parsed" was the main limitation. Golub advised checking how well Refine handles your tables before relying on it. In empirical papers it flags clustering choices and questions IV logic. Uploads are not used for training or exposed to model providers, per contract. — [same](https://markusacademy.substack.com/p/modern-ai-for-economics-research)
- Product timeline [vendor]:
  - 2026-06-16: structured benchmark post.
  - 2026-07-17: SOC 2 Type 1.
  - 2026-07-23: reviews can include referee reports, response letters, data outputs and prior analyses, for documents up to 100,000 words.
  - 2026-08-28: Daniel Litt retrospective.
  - 2026-09-23: "Refine 5." — [Refine blog index](https://www.refine.ink/blog)
- Journal partnerships (announced 2026-08-06): the AEA and the Econometric Society use Refine for "AI-assisted technical verification." Its main use is a "pre-publication audit of technical execution and exposition." Editors may pass its technical comments to authors during refereeing. Refine also writes "Overall Feedback," which editors do not see. It is free for journals, subject to capacity. — [Refine blog, 2026-08-06](https://www.refine.ink/blog/economics-journal-partnerships)
- AEA terms (2026-08-14): used only for papers that have received, or are under serious consideration for, an R&R. It is a technical check only and does not assess contribution or mass-screen submissions. Zero data retention with providers. — [AEA member announcement](https://benny.aeaweb.org/news/member-announcements/2026-aug-14)
- Econometric Society terms: run at conditional acceptance to look for proof errors, numerical or logical inconsistencies, notation problems and clarity issues. Submitting to QE or TE implies consent. Manuscripts are not used for training. — [Econometric Society QE instructions](https://www.econometricsociety.org/publications/quantitative-economics/submissions/instructions-for-submitting-articles)
- AEA pilot: Erzo Luttmer reported about 90% of authors favored incorporating Refine into the editorial process. He framed it as "not a referee, but a detailed technical proofreader." This is an X post with no published pilot report. — [Luttmer on X](https://x.com/ErzoLuttmer/status/2086330632440041927)
- Architecture (models, agents, verification) is not disclosed in any Refine page I could access. — [Refine blog](https://www.refine.ink/blog/economics-journal-partnerships)
- Price: one reader comment reports about $30 per paper paid and a $50 list price for a single review. Litvak estimates Refine at about $8.77 per error caught versus about $0.04 for frontier models. — [Litvak, 2026-08-10](https://paullitvak.com/p/how-well-does-ai-peer-review-work)

**Stanford Agentic Reviewer (paperreview.ai; Yixing Jiang and Andrew Ng; launched about Nov 2025)**
- Pipeline:
  1. PDF to Markdown via LandingAI ADE.
  2. The agent generates search queries at several levels of specificity.
  3. Tavily search over arXiv.
  4. Relevance scoring of the hits.
  5. Summaries of the top related papers.
  6. Template-based review generation.
  7. Optional scoring on 7 dimensions (originality, importance, claim support, soundness of experiments, clarity, community value, contextualization), combined by a linear regression.
- The LLM is not named. It is English only. The score is shown only for ICLR. The page notes that arXiv grounding means accuracy is lower outside fields that post to arXiv. — [paperreview.ai tech overview](https://paperreview.ai/tech-overview)
- Ng framed it as a fast feedback loop for authors after a student was rejected 6 times over 3 years. — [Ng on X](https://x.com/AndrewYNg/status/1993001922773893273)
- A third-party review site says only the first 15 pages are analyzed. Not verified on the official page. — [manusights](https://manusights.com/blog/paperreview-ai-review-2026)

**AAAI-26 AI review system (conference-side; OpenAI gpt-5)**
- Pipeline:
  - Preprocessing: olmOCR converts PDFs to Markdown at 250 DPI.
  - Five sequential aspect stages: Story, Presentation, Evaluations, Correctness, Significance. Each stage sees the prior stages' outputs.
  - Tools: a Python code interpreter in the Evaluations and Correctness stages; web search, restricted to published venues, in Significance.
  - Synthesis: initial review, then a self-critique stage for unsupported claims and inconsistencies, then a final review.
  - A separate o4-mini critic, which does not see the paper, flags anonymity leaks, bias, structure and offensive content.
- Model settings: gpt-5 at high reasoning effort, 400k context, under $1 per paper.
- Deployment: one labeled AI review per paper with no score or recommendation, alongside at least 2 human reviews. — [Biswas et al., arXiv 2604.13940, 2026-04-15](https://arxiv.org/html/2604.13940v1)

**ICLR 2025 Review Feedback Agent (Zou group; reviewer-facing, not author-facing)**
- It "leveraged multiple LLMs" plus "multiple LLM-based guardrails" (reliability tests) that checked the feedback before posting. It never edited reviews. — [ICLR blog, 2025-04-15](https://blog.iclr.cc/2025/04/15/leveraging-llm-feedback-to-enhance-review-quality/); code at github zou-group per [arXiv 2504.09737](https://arxiv.org/abs/2504.09737?context=cs)
- From training knowledge, not re-verified here: the paper describes Claude 3.5 Sonnet in actor, aggregator, critic and formatter roles, targeting vague comments, misunderstandings answerable from the paper, and unprofessional remarks.

**Error-checkers (narrow, objective-error focus)**
- Black Spatula Project (Dec 2024-): an open-source effort, prompted by the black-plastic-utensil paper's arithmetic error. It asks how many errors LLMs detect, how serious they are, what the false-positive rate is, and how much human verification is needed. Its GitHub reports no published results. Main repo last updated Dec 2024; website Aug 2025. — [GitHub org](https://github.com/The-Black-Spatula-Project)
- "AI Correctness Checker" (Bianchi, Kwon, Izzo, Zhang, Zou; Dec 2025): GPT-5 detects candidate errors, a second LLM verifies each one and removes false positives, GPT-5-mini categorizes, and GPT-5 proposes fixes. It is restricted to objective errors (formulas, derivations, calculations, figures, tables) and excludes novelty, significance and writing. — [arXiv 2512.05925](https://arxiv.org/html/2512.05925v1)
- skepthical (NeurIPS 2026 poster): a multi-agent reviewer.
  - Retrieves cited papers to check citations.
  - Verifies math with computer algebra.
  - Audits numerical claims with code.
  - Uses a cross-model ensemble for general issues.
  - Deployed publicly at skepthical-ai.org. — [Zubeldia et al., NeurIPS 2026](https://neurips.cc/virtual/2026/poster/152885)
- Google Paper Assistant Tool (PAT; Jun 2026): an "agentic AI framework" for deep review and verification using "inference scaling." Piloted as a pre-submission tool at STOC and ICML. — [Jayaram et al., arXiv 2606.28277](https://arxiv.org/abs/2606.28277)
- Zhang and Abernethy (2025): frame LLMs as "manuscript quality checkers" rather than review writers. Test data is papers withdrawn from arXiv, with LLM-as-judge evaluation. o3 performed best at modest cost. Cites Black Spatula as inspiration. — [arXiv 2505.23824](https://arxiv.org/abs/2505.23824)

**Research and open-source reviewer models (mostly ML-venue)**
- MARG (D'Arcy et al., 2024): a leader agent, worker agents that each hold a chunk of the paper, and expert agents for sub-tasks. This works around context limits by splitting the paper. A survey reports 2.2x more "good" comments and 29% generic comments versus at least 60% for baselines. I could not verify those numbers in the original. — [arXiv 2401.04259](https://arxiv.org/pdf/2401.04259); [ADS abstract](https://ui.adsabs.harvard.edu/abs/2024arXiv240104259D/abstract)
- OpenReviewer (Idahl and Ahmadi; NAACL 2025 demo): Llama-3.1-8B fine-tuned on 79k expert reviews. It produces "more critical and realistic" reviews than GPT-4 or Claude 3.5, which tend to be overly positive (400 test papers). — [arXiv 2412.11948](https://arxiv.org/abs/2412.11948v3); [HF model card](https://huggingface.co/maxidl/Llama-OpenReviewer-8B)
- AI Scientist reviewer (Sakana, 2024): GPT-4o with 5 rounds of self-reflection, a 5-review ensemble and a one-shot example. 70% accuracy on 500 ICLR 2022 papers; balanced accuracy about 0.65 versus a 0.66 human baseline (NeurIPS 2021 consistency experiment). — [arXiv 2408.06292](https://arxiv.org/pdf/2408.06292)
- Later Nature version: 5 reviews plus a meta-review. With o4-mini, balanced accuracy 0.77 and accuracy 0.69. Pre-cutoff accuracy 0.691 versus 0.660 for humans, not significant. — [arXiv 2606.15497](https://arxiv.org/pdf/2606.15497)
- CycleReviewer (CycleResearcher, ICLR 2025): reports a 26.89% reduction in score-prediction MAE versus individual human reviewers. — [arXiv 2411.00816](https://arxiv.org/abs/2411.00816v3)
- Liang et al. (2023/2024): a single GPT-4 pass that produces structured feedback. From training knowledge: the sections are significance/novelty, reasons to accept, reasons to reject, and suggestions. — [arXiv 2310.01783](https://arxiv.org/abs/2310.01783v1)

**Other commercial and author-side tools (2026)**
- Reviewer3 has a "journal mode." It was tested by Litvak and had the lowest detection, 30/100. — [Litvak](https://paullitvak.com/p/how-well-does-ai-peer-review-work)
- Paperpal, Octym and Penelope.ai are listed in vendor comparison blogs as manuscript checkers (references, statistics, reporting, journal compliance). This is weak evidence. — [thesify blog](https://www.thesify.ai/blog/ai-tools-academic-peer-review); [peerreviewai.org comparison](https://peerreviewai.org/compare/ai-peer-review-tools)
- AI-Paper-Review (Di Wu, Jun 2026): an open web UI that samples several AI reviewers from a pool, then clusters and ranks their comments. Case study on 20 computer-architecture drafts. — [arXiv 2606.01013](https://arxiv.org/abs/2606.01013)

### Inferences
- Two families exist:
  - **Referee simulators** (paperreview.ai, AAAI, AI Scientist, CycleReviewer, OpenReviewer) that judge quality, novelty and scores.
  - **Technical verifiers** (Refine as journals use it, the AI Correctness Checker, skepthical, PAT) that hunt objective errors.
- For a JFE-class submission, the verifier family is more reliable and better evidenced. The simulator family is weakest exactly where economists care most: novelty and significance (see Q2).
- paperreview.ai's arXiv-only grounding and ICLR-trained scoring make its scores largely uninformative for a financial-econometrics paper. Its text comments may still be useful.

### Gaps
- No evidence found of AI review at the Journal of Financial Econometrics, SoFiE or OUP. Did not search their policies exhaustively.
- Elsevier, Wiley, Springer Nature and Overleaf: no 2026 author-facing AI referee product found in one search. Publisher newsrooms were not checked directly.
- Refine's internal architecture (single model vs multi-agent, verification passes) is undisclosed. Refine's June 2026 benchmark methodology post on X returned HTTP 402 and could not be read.
- Could not verify the Thakkar et al. agent internals beyond the ICLR blog.

## Q2. What does controlled evidence say about accuracy and usefulness (agreement with referees, error-finding, false positives)?

### Takeaway
- **Usefulness to authors is consistently positive in surveys.** 57% found GPT-4 feedback helpful in 2023. In 2025-26, AI reviews were rated above human reviews on 6 of 9 criteria at AAAI-26, and about 90% of AEA pilot authors wanted Refine.
- **Agreement with human scores is about human-human level, which is low.** Spearman 0.42 vs 0.41; review-point overlap 31-39%.
- **Error-finding varies hugely by system and benchmark.** On real errata-level errors, 2025 LLMs had recall of 21% or less and precision of 6% or less. 2026 multi-stage or verifier systems reach about 60-70% recall on planted errors, with 1-3% false-positive rates in one benchmark and 83% human-verified precision in another.
- **Pooling systems is the biggest single lever.**
- **Omissions and novelty or significance judgments remain blind spots.**

### Cited Findings

**Usefulness and agreement**
- Liang et al. (NEJM AI 2024; arXiv Oct 2023), GPT-4 feedback:
  - Overlap with human reviewer points was 30.85% for Nature-family journals and 39.23% for ICLR, comparable to human-human overlap.
  - In a prospective survey of 308 researchers, 57.4% rated it helpful or very helpful, and 82.4% rated it more beneficial than feedback from at least some human reviewers.
  - Weak at in-depth method critique. — [NSF PAR copy](https://par.nsf.gov/biblio/10617069); [arXiv](https://arxiv.org/abs/2310.01783v1); [2 Minute Medicine](https://www.2minutemedicine.com/large-language-models-show-potential-to-provide-feedback-on-research-papers-on-a-large-scale/)
- Stanford Agentic Reviewer [developer-reported], on 147 held-out ICLR 2025 papers:
  - Spearman correlation: 0.42 between the AI and one human, versus 0.41 between two humans.
  - Acceptance AUC: 0.75 for the AI versus 0.84 for one human score. Not a fair comparison, since human scores drove decisions. — [paperreview.ai](https://paperreview.ai/tech-overview)
  - Critics note that matching noisy human scores shows imitation, not verification. — [Ng LinkedIn thread](https://www.linkedin.com/posts/andrewyng_releasing-a-new-agentic-reviewer-for-research-activity-7398773018317074432-vMct)
- AAAI-26 pilot (22,977 papers; 5,834 survey responses; voluntary, so self-selection is possible):
  - AI reviews were preferred on 6 of 9 criteria (Likert, AI minus human, all significant at α=0.01):
    - Identifying technical errors: +0.67
    - Raising unconsidered points: +0.61
    - Presentation suggestions: +0.54
    - Research-design suggestions: +0.49
    - Thoroughness: +0.48
  - AI reviews were worse on:
    - Overemphasizing minor issues: −0.38
    - Making technical errors itself: −0.22
    - Wrong or unhelpful suggestions: −0.11
  - 53.9% found the AI review useful versus 20.2% not. Only 13.8% said it changed their evaluation.
  - Stated weaknesses: misreading equations and tables, poor prioritization, weak novelty and significance judgment, verbosity.
  - Citation audit: 1,356 citations checked, no fabricated citations confirmed. — [arXiv 2604.13940](https://arxiv.org/html/2604.13940v1)
- ICLR 2025 RCT (Thakkar et al.; feedback to reviewers, not authors):
  - 26.6% of treated reviewers updated their reviews and incorporated 12,222 feedback items.
  - Updated reviews were about 80 words longer.
  - In blinded assessment, the feedback improved review quality in 89% of cases.
  - Author rebuttals were 48 words longer (p≤0.001).
  - No significant effect on acceptance.
  - Group sizes are reported inconsistently: 18,946 reviews received feedback (text) versus 22,467 treated and 22,364 control (figure). — [ICLR blog](https://blog.iclr.cc/2025/04/15/leveraging-llm-feedback-to-enhance-review-quality/); [arXiv 2504.09737](https://arxiv.org/abs/2504.09737?context=cs)
- AEA pilot: about 90% of authors favored adopting Refine (X post; no formal report). — [Luttmer](https://x.com/ErzoLuttmer/status/2086330632440041927)

**Error-finding benchmarks**
- SPOT (May 2025): 83 published papers with 91 errata- or retraction-level errors.
  - No LLM exceeded 21.1% recall or 6.1% precision; o3 was best.
  - Confidence was uniformly low.
  - Across 8 runs, models rarely rediscovered the same errors. — [arXiv 2505.11855](https://arxiv.org/abs/2505.11855)
- AAAI-26 SPECS benchmark: 120 papers with 783 planted, compiling LaTeX perturbations.
  - Recall: single-prompt baseline 0.43; aspect-targeted stages 0.61; final staged review 0.64 (+0.21, p≈6e-30).
  - For Significance, the targeted stage found more (0.53) than the final review (0.45). Synthesis dropped true findings. — [arXiv 2604.13940](https://arxiv.org/html/2604.13940v1)
- Bianchi et al. (Dec 2025), GPT-5 detect-then-verify checker on 2,500 ICLR, NeurIPS and TMLR papers:
  - Precision 83.2%: human experts confirmed 263 of 316 flagged issues in 60 papers.
  - Recall 60% on 90 injected errors.
  - Average 4.66 flagged objective mistakes per paper. 24-36% of papers had a potentially substantive one.
  - Fixes were offered for 86.3% of verified mistakes, and 75.8% of those fixes were correct.
  - Error mix: math/formula 54%, text 31%, table/figure 9%, cross-reference 5%.
  - Common false positives came from non-standard notation and OCR errors. — [arXiv 2512.05925](https://arxiv.org/html/2512.05925v1)
- Zubeldia et al. (NeurIPS 2026): 15 astrophysics preprints, 177 injected errors, 12 reviewers.
  - Three frontier LLMs differ by 48.6 points in detection under the same free-form prompt. A structured prompt roughly halves the spread.
  - Claude Code went from free-form to structured prompting: +32.2 points.
  - skepthical had the best detection, 70.6±6.2%:
    - General issues: 82.2%
    - Citations: 46.7%
    - Numerical: 80.0%; structured Claude Code reached 91.1%
    - Math: 73.8%
  - False-positive rates: free-form Opus 4.7 had 1.0%; skepthical had 2.8% and was the only reviewer with no FP failure mode across math, numerics and citations. — [NeurIPS poster](https://neurips.cc/virtual/2026/poster/152885)
- Litvak (Aug 2026): 100 planted errors in 10 open-access psychology papers, matched by an LLM judge. Systems: GPT-5.4/5.5, Claude Sonnet and Opus 4.6, Opus 4.8, Gemini 3.1 Pro and 3 Flash, Reviewer3, Refine.ink.
  - Best single system was GPT-5.5 at 71/100; worst was Reviewer3 at 30/100.
  - Pooled: 91 for the foundation models alone and 93 for all systems. The two figures are not reconciled in the post.
  - Refine contributed the most unique catches. It was strongest on methods design, statistics and internal consistency; weakest on causal inference, reporting completeness and generalizability.
  - The 7 errors no system caught were all omissions (deleted information).
  - False positives were not measured, though the author calls them "an issue." — [Litvak](https://paullitvak.com/p/how-well-does-ai-peer-review-work); [Marginal Revolution, 2026-08-11](https://marginalrevolution.com/marginalrevolution/2026/08/how-well-does-ai-peer-review-work.html); data at [github Dawes-Institute/ai-peer-review-benchmark](https://github.com/Dawes-Institute/ai-peer-review-benchmark)
- PaperAudit-Bench (EMNLP 2026): 220 papers with more than 2,200 synthetic errors in 8 categories.
  - "Sparse long-context auditing remains challenging."
  - A post-trained Qwen3-14B reached 55.8% F1 at Detection@1.
  - Conditioning reviews on explicit, manuscript-grounded error evidence gave more conservative assessments that aligned better with humans. — [arXiv 2601.19916](https://www.arxiv.org/abs/2601.19916)
- Google PAT: a 34% improvement over zero-shot recall on SPOT math errors (relative; absolute numbers not in the abstract). — [arXiv 2606.28277](https://arxiv.org/abs/2606.28277)

**Economics and math specifically**
- Litt retrospective [vendor post; author-labeled]: Refine on 20 of Litt's published or accepted math papers.
  - Of 278 comments: 248 correct, 24 partially correct, 6 incorrect. 97.8% flagged a real issue.
  - Severity: 217 were correctable errors in statements, proof steps, formulas, citations or hypotheses. 5 required technical corrections to main results. None overturned a paper's central message.
  - Recall was not measured.
  - Refine was better at spotting errata than at fixing them; its fixes could be overzealous. — [Refine blog, 2026-08-28](https://www.refine.ink/blog/retrospective-math-benchmark)
- Refine head-to-head benchmarks [vendor; judging method not visible to me]:
  - 150 economics preprints: 90.4% win rate over 1,349 matches against 9 other AI reviewers. 94.8% versus standalone LLMs; 85.0% versus scaffolded systems.
  - Refine 5 versus GPT-6 Astra and Claude Fable 5.1: won 197 of 215 matches on 108 papers across 10 fields. — [Refine blog index](https://www.refine.ink/blog); [search summary of X article](https://x.com/RefineDotInk/article/2068026653717008511)
- Toda (Jun 2026): Gemini, Refine, Claude and ChatGPT were asked to check 4 published economic theory papers with known errors. ChatGPT Pro did best and sometimes built counterexamples. "No model located a true error without substantial human guidance." Data contamination is a caveat. — [arXiv 2606.05383](https://arxiv.org/abs/2606.05383)
- Commentary on risks: offloading technical checks may reduce human reviewer care, and authors may feel pressured to pay for the same AI checks up front (capture). — [Dworkin, Abundance & Growth, 2026-08-13](https://www.abundanceandgrowth.org/p/ai-peer-review-goes-pro)

### Inferences
- **Precision depends heavily on scope.**
  - Objective, checkable claims (math, arithmetic, table-text consistency, citations) with a verification pass reach roughly 80-98% valid comments: Bianchi 83%, Litt 97.8% valid-or-partial, skepthical FP 2.8%.
  - Open-ended referee critique (novelty, significance, identification) is where false or overstated comments concentrate. The AAAI results on minor-issue overemphasis and weak significance judgment show this.
- **Recall per run is far from complete and is stochastic.** SPOT found low rediscovery across runs. Litvak's pooling went from 71 to 93. A clean single AI review is weak evidence of no errors. Several independent runs or models should be unioned.
- **Omissions are the shared blind spot:** a missing robustness check, a data detail, or a disclosure. Checklist-driven prompts ("is X reported?") are the obvious complement. This is inferred from Litvak's omission result and the structured-prompt gains in Zubeldia.
- **Table parsing matters for an empirical finance paper.** OCR/PDF artifacts are a top false-positive source (Bianchi), and Refine flagged table parsing as its main empirical limitation (Dec 2025). Feeding LaTeX source and machine-readable result files (CSV) instead of the PDF should help. Refine accepts "data outputs" since Jul 2026, and AAAI used LaTeX-level perturbations.

### Gaps
- No controlled study of AI review on empirical finance or econometrics papers was found. All quantitative error benchmarks are ML, astrophysics, psychology or math.
- Refine's precision and recall on empirical economics papers have not been independently measured. Its benchmark judging protocol could not be read.
- No author-side randomized trial was found (AI pre-submission feedback → acceptance or quality outcomes). The ICLR RCT treated reviewers.
- The Di Wu (2606.01013) quantitative results on coverage of human-raised issues were not in the abstract.

## Q3. Which design patterns raise precision (and recall)?

### Takeaway
The evidence supports five patterns:
1. Aspect- or section-targeted stages rather than one prompt.
2. Structured, checklist-style prompts.
3. Tool-grounded verification: code for numbers, CAS for algebra, retrieval for citations.
4. A separate verifier or critic pass that filters candidates before the human sees them.
5. Pooling across independent runs or different model families.

"Write as referee at journal X" persona prompting is advised by practitioners but untested in the controlled studies found.

### Cited Findings
- **Aspect-targeted stages beat a single prompt.** Recall went from 0.43 to 0.61 to 0.64 at AAAI-26. Final synthesis can drop true positives: for Significance, the targeted stage scored 0.53 and the final review 0.45. Keep stage-level findings visible. — [arXiv 2604.13940](https://arxiv.org/html/2604.13940v1)
- **Self-critique pass for unsupported claims.** AAAI-26 used initial review → self-critique → revised review. A separate small critic screened for anonymity leaks, bias and structure. Humans hand-checked the first 30% of outputs before scaling. — [same](https://arxiv.org/html/2604.13940v1)
- **Detect-then-verify with a second LLM** reached 83.2% human-confirmed precision. Restrict scope to objective errors. — [Bianchi et al.](https://arxiv.org/html/2512.05925v1)
- **Guardrail "reliability tests"** gated every feedback item before posting in the ICLR 2025 agent. — [ICLR blog](https://blog.iclr.cc/2025/04/15/leveraging-llm-feedback-to-enhance-review-quality/)
- **Structured prompts and tools.** Structured prompting halved the cross-model spread and added 32 points within Claude Code. Tool-verified agents (retrieved citations, CAS, code audits of numbers) had the lowest-risk false-positive profile. — [Zubeldia et al.](https://neurips.cc/virtual/2026/poster/152885)
- **Evidence-grounded review.** Conditioning on explicit, manuscript-grounded error evidence gives more conservative, human-aligned reviews. — [PaperAudit-Bench](https://www.arxiv.org/abs/2601.19916)
- **Ensembles and multiple models.**
  - AI Scientist: 5-review ensemble plus 5 self-reflection rounds; the later version used 5 reviews plus a meta-review (best balanced accuracy 0.77). — [arXiv 2408.06292](https://arxiv.org/pdf/2408.06292); [arXiv 2606.15497](https://arxiv.org/pdf/2606.15497)
  - Pooling across models gives far higher recall (71 to 91-93 of 100). — [Litvak](https://paullitvak.com/p/how-well-does-ai-peer-review-work)
  - SPOT shows low cross-run overlap, so single runs are unreliable. — [arXiv 2505.11855](https://arxiv.org/abs/2505.11855)
- **Inference scaling** (multiple passes) surfaces deeper errors (+34% relative recall on SPOT math). — [Google PAT](https://arxiv.org/abs/2606.28277)
- **Multi-agent chunking.** MARG splits the paper across worker agents with a leader and experts, reducing generic comments. Survey-reported figures were not verified. — [arXiv 2401.04259](https://arxiv.org/pdf/2401.04259)
- **Fine-tuning on real reviews** counters LLM positivity bias. — [OpenReviewer](https://arxiv.org/abs/2412.11948v3)
- **Literature grounding for novelty.** Retrieving and summarizing related work grounds novelty comments. It only works where the literature is on arXiv. — [paperreview.ai](https://paperreview.ai/tech-overview)
- **Comment caps.** Golub advises asking for at most 10 comments. — [Markus' Academy](https://markusacademy.substack.com/p/modern-ai-for-economics-research)
- **Role and audience prompts.** Golub advises rigor-focused prompts specifying the reader, for example "a senior econometrician writing for a top journal." — [same](https://markusacademy.substack.com/p/modern-ai-for-economics-research)
- **Clustering and ranking across reviewers.** AI-Paper-Review samples several AI reviewers and clusters and ranks their comments to de-duplicate. — [arXiv 2606.01013](https://arxiv.org/abs/2606.01013)

### Inferences
- A practical high-precision architecture for this user, built with Claude agents:
  1. **Generation:** N independent reviewers. Use different models or seeds where possible, and aspect-specialized roles: identification/econometrics, data and sample construction, table-text consistency, math/derivations, literature/positioning, exposition. Each reviewer gets a structured checklist and the LaTeX source plus result CSVs.
  2. **Clustering:** merge duplicate comments into issues. Agreement count is a confidence signal.
  3. **Adversarial verification:** for each issue, a separate verifier tries to refute it against the source. It recomputes numbers from CSVs or code, checks the cited table cell, and reads the cited paper. It labels each issue confirmed, refuted or judgment-call, with a quoted evidence locator.
  4. **Presentation:** show only confirmed issues and judgment-calls, ranked by severity.
- This mirrors AAAI's stages plus Bianchi's detect-then-verify plus skepthical's tool checks plus Litvak's pooling.
- Persona prompts ("referee at JFEC / J. Econometrics") are cheap to add. They are best treated as one of the N reviewers, not as a precision mechanism: no controlled evidence was found.

### Gaps
- No controlled study isolates the effect of "imagine a referee at journal X" persona prompting.
- No study directly compares adversarial (refute-this) verification with neutral verification. The evidence is for verification passes in general.
- No published false-positive rate exists for a pooled multi-model setup. Pooling raises recall but probably raises FP volume too, which makes the verifier stage necessary.

## Q4. How should an author use these tools in an iterative pre-submission loop, and triage comments efficiently?

### Takeaway
No study evaluates author-side triage workflows directly. The pieces support a loop of four steps:
1. Cheap, pooled, verified AI checks on every draft.
2. A paid specialist pass (Refine) at milestones, feeding in prior reports and response letters.
3. The human reads only a ranked, evidence-linked, verified issue list.
4. AI-generated fixes are treated as drafts, because fix accuracy (about 76%) lags detection accuracy.

### Cited Findings
- Golub recommends Refine before circulating or submitting, with at most 10 comments. For the workflow: stage the work, start fresh chats instead of arguing after a wrong answer, and pass "handoff reports" between sessions. — [Markus' Academy](https://markusacademy.substack.com/p/modern-ai-for-economics-research)
- Refine can ingest referee reports, response letters, data outputs and prior analyses (up to 100k words, since Jul 2026). This enables round-over-round review: checking whether a response letter actually addresses each referee point. [vendor] — [Refine blog index](https://www.refine.ink/blog)
- Refine separates journal-visible technical comments from "Overall Feedback." This is a natural two-bucket triage: objective fixes versus judgment calls. — [Refine blog](https://www.refine.ink/blog/economics-journal-partnerships)
- Fix quality: 75.8% of AI-proposed fixes for verified mistakes were correct. — [Bianchi et al.](https://arxiv.org/html/2512.05925v1) Litt found Refine better at spotting than fixing, with overzealous fixes. — [Refine/Litt](https://www.refine.ink/blog/retrospective-math-benchmark)
- AAAI respondents flagged overemphasis on minor issues and verbosity, which argues for severity ranking and caps. Only 13.8% said the AI review changed their evaluation. — [arXiv 2604.13940](https://arxiv.org/html/2604.13940v1)
- Paperreview.ai positions itself as a submit, revise, resubmit loop with author verification of each point. — [paperreview.ai](https://paperreview.ai/tech-overview)
- Cost structure: frontier models cost about $0.04 per caught error and Refine about $8.77. Refine adds unique catches. — [Litvak](https://paullitvak.com/p/how-well-does-ai-peer-review-work)
- Bianchi et al. recommend treating flags as "potential mistakes" until a human verifies them. Non-standard notation and OCR are FP sources. — [arXiv 2512.05925](https://arxiv.org/html/2512.05925v1)
- AEA and Econometric Society now run Refine at R&R or conditional acceptance. Running it pre-submission therefore anticipates what editors at those societies will see. — [AEA](https://benny.aeaweb.org/news/member-announcements/2026-aug-14); [Econometric Society](https://www.econometricsociety.org/publications/quantitative-economics/submissions/instructions-for-submitting-articles)
- Gaming concern: once authors can run the screen themselves, a passed AI screen certifies cost paid, not quality. — [Fourie, arXiv 2607.13844](https://arxiv.org/abs/2607.13844)

### Inferences
- **Suggested triage output per issue** (designed to minimize human minutes):
  - Severity: blocker / major / minor.
  - Type: objective / judgment.
  - Verifier status: confirmed / refuted / unsure, with the refuted ones hidden by default.
  - Agreement count across reviewers.
  - Exact locator: section, equation, table cell, line.
  - Quoted evidence plus a recomputed value where applicable.
  - Proposed patch as a diff.
- The human then:
  - Accepts or rejects objective items in batch.
  - Reads judgment items individually; these are where AI precision is lowest.
  - Never sees refuted items unless sampling them for audit.
- **Loop cadence:**
  - Every substantive draft: run the cheap pooled-and-verified pipeline.
  - Before circulating and before submission: run Refine as an independent outside check. Its catches only partly overlap with frontier models (Litvak).
  - After referee reports arrive: feed reports plus the draft response letter to check point-by-point coverage.
  - Track issue IDs across rounds so resolved items do not resurface. Log refuted items as known false-positive patterns to suppress in later runs.
- **Financial-econometrics specifics to put in reviewer checklists** (inferred from the error types tools catch or miss):
  - Table-text number consistency, recomputed from the CSVs.
  - Standard-error and inference choices (HAC/clustering, overlapping returns).
  - Look-ahead or sample-construction leakage in forecasting.
  - Multiple-testing and data-snooping disclosure.
  - Out-of-sample protocol description.
  - Omitted robustness details. Omissions are the universal blind spot, so ask explicit "is X reported?" questions.
- **Human time budget:** the AAAI ratio of one AI review per paper, with the AI not changing most evaluations, suggests AI output is most valuable as an error list rather than a verdict. Do not iterate the paper toward an AI score (paperreview.ai scores are ICLR-calibrated).

### Gaps
- No empirical study of how authors triage AI comments or of the time saved. All triage recommendations above are inferences.
- No evidence on whether iterating against AI reviewers overfits the paper to AI preferences (raised as a concern in the AAAI survey, unquantified).
- No public pricing page for Refine was confirmed. Price figures come from a reader comment.
