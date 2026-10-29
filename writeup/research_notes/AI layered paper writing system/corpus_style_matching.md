# Corpus-based style matching: measuring a target literature's prose and checking a draft against it

Research date: 2026-10-28. Target use: a JFEc paper (realized-volatility forecasting, HAR, ML) drafted largely by Claude agents; wanted is an automated "research the literature's prose, then scrub the draft" step that runs in Python on a Windows laptop. Dates of sources are given inline. "Inference" items are my reasoning, not sourced facts.

## Q1. Corpus comparison statistics: which ones suit flagging over- and under-used words/phrases in a draft?

### Takeaway
For a ranked, signed, variance-aware list of words or n-grams that the draft over- or under-uses relative to the corpus, the best tool is Monroe, Colaresi and Quinn's (2008) log-odds ratio with an informative Dirichlet prior, scored as a z-score. Log-likelihood (G2) keyness paired with an effect size (Log Ratio or %DIFF) is the standard corpus-linguistics alternative. Burrows' Delta and Biber-style multidimensional (MD) feature profiles give one overall "distance from the register" number rather than word-level flags. Sentence-length and passive-voice rates are best handled as distributions: place the draft at a percentile of the per-paper distribution in the corpus.

### Cited Findings
**Fightin' Words (Monroe, Colaresi and Quinn 2008, *Political Analysis* 16(4): 372–403, DOI 10.1093/pan/mpn018)**
- The two-group log-odds ratio with prior is δ̂_w^(i−j) = log[(y_w^i + α_w) / (n^i + α_0 − y_w^i − α_w)] − log[(y_w^j + α_w) / (n^j + α_0 − y_w^j − α_w)] (eq. 16). Without the prior, this reduces to the observed log-odds ratio. — [Monroe et al. 2008, p. 385 (PDF)](http://languagelog.ldc.upenn.edu/myl/Monroe.pdf); [DOI](https://doi.org/10.1093/pan/mpn018)
- Variance is approximately σ²(δ̂^(i−j)) ≈ 1/(y_w^i + α_w) + 1/(y_w^j + α_w) (eq. 20). This depends on the absolute frequency of the word in both groups and its implied prior frequency. The approximation is "reasonable for documents of moderate size (at 1000 words only the fourth decimal place is affected)". — [Monroe et al. 2008, pp. 385–386](http://languagelog.ldc.upenn.edu/myl/Monroe.pdf)
- The evaluation statistic is the z-score ζ̂_w = δ̂_w / √σ²(δ̂_w) (eqs. 21–22). The paper's Figure 4 uses the "familiar cutoff at 1.96". — [Monroe et al. 2008, p. 386](http://languagelog.ldc.upenn.edu/myl/Monroe.pdf)
- Without a prior, "the dominant words [are] still the same list of obscurities… estimates for infrequently spoken words have higher variance than frequently spoken ones". With a flat-prior z-score, function words such as *the, of, be, not, my* "still appear to be too prominent". — [Monroe et al. 2008, pp. 385–386](http://languagelog.ldc.upenn.edu/myl/Monroe.pdf)
- Informative prior: α_kw = α_k0 · π̂^MLE = y · α_0/n (eq. 23). The prior is proportional to the word's frequency in a background corpus, and α_0 sets the amount of shrinkage (the authors use a "prior sample" of 500 words). This pushes function words out of the top lists. Footnote 18 says the prior can instead come from "general word frequency information from other sources altogether". — [Monroe et al. 2008, pp. 387–388](http://languagelog.ldc.upenn.edu/myl/Monroe.pdf)
- A χ² test per word gives "nearly identical" rankings to the z-test (the approach of Gentzkow and Shapiro 2006), but "the χ² value itself does not have any obvious interpretation as an evaluation weight" and has no direction. — [Monroe et al. 2008, p. 386](http://languagelog.ldc.upenn.edu/myl/Monroe.pdf)
- A Laplace-prior variant shrinks most word contrasts to exactly zero, which suits sparse feature selection. It is computationally harder, though, and the authors call it "less than ideal for feature evaluation". — [Monroe et al. 2008, pp. 389–390](http://languagelog.ldc.upenn.edu/myl/Monroe.pdf)

**Keyness (log-likelihood, effect sizes)**
- Log-likelihood (G2) for a word with frequencies a and b in corpora of sizes c and d: E1 = c(a+b)/(c+d), E2 = d(a+b)/(c+d), and G2 = 2[a ln(a/E1) + b ln(b/E2)]. Critical values are 3.84 (p<0.05), 6.63 (p<0.01), 10.83 (p<0.001) and 15.13 (p<0.0001). The UCREL calculator also reports %DIFF, a Bayes Factor (BIC), ELL, Relative Risk, Log Ratio and Odds Ratio. Log Ratio is the binary log of relative risk, with a 0.5 adjustment for zero counts. — [UCREL Log-likelihood and effect size calculator](https://ucrel.lancs.ac.uk/llwizard.html)
- Gabrielatos and Marchi argue that keyness should be an effect size, not a significance test. LL "measures statistical significance, not frequency difference" and is sensitive to word frequencies and corpus sizes. %DIFF is the normalised-frequency difference divided by the reference frequency, ×100. Rankings by %DIFF and by LL differ substantially. — [Gabrielatos and Marchi, keyness slides (Lancaster eprint)](https://eprints.lancs.ac.uk/id/eprint/51449/4/Gabrielatos_Marchi_Keyness.pdf)
- The keyflux Python library offers %DIFF but recommends sorting by Log Ratio, because in large corpora LL flags far too many keywords to rank usefully. The R package KeynessMeasures computes LL, the effect size of LL, Log Ratio and %DIFF. — [keyflux keyness walkthrough](https://keyflux.readthedocs.io/en/latest/tutorials/keyness_walkthrough/); [KeynessMeasures](https://rdrr.io/github/amacanovic/KeynessMeasures/man/keyness_measure_calculator.html)
- quanteda's `textstat_keyness` takes a target document or group against a reference built from all other documents. Measures are `chi2` (default, Yates correction), `lr` (G2, Williams correction), `exact` (Fisher, slow) and `pmi`. The output is signed and includes counts and p-values. — [quanteda textstat_keyness reference (documents quanteda 2.1.2)](https://quanteda.io/reference/textstat_keyness.html)

**Stylometry and register analysis**
- Burrows' Delta compares the writing style of documents, typically using z-scored frequencies of the most frequent words. `faststylometry` (Python, `pip install faststylometry`) provides `calculate_burrows_delta`, which returns a pandas DataFrame, plus `calibrate` and `predict_proba` for probability-of-same-author estimates. One index reports a NumPy/Python 3.12 compatibility issue in an older release. — [faststylometry on PyPI](https://pypi.org/project/faststylometry); [Fast Data Science](https://fastdatascience.com/natural-language-processing/fast-stylometry-python-library/)
- `pybiber` (Python ≥3.10, needs a spaCy model such as `en_core_web_sm`, input as a Polars DataFrame with `doc_id` and `text`) extracts the 67 lexicogrammatical and functional features of Biber (1988). It also runs Biber's multidimensional analysis (`mda`), with scree and group-mean plots. `PybiberPipeline().run_from_folder(...)` runs everything from a folder of text files. An R sibling, `mda.biber`, starts from normalised feature counts. — [pybiber docs](https://browndw.github.io/pybiber); [pybiber on PyPI](https://pypi.org/project/pybiber/); [mda.biber on CRAN](https://cran.r-project.org/web/packages/mda.biber/readme/README.html)
- `PassivePy` (Sepehri, Mirshafiee and Markowitz, *Journal of Consumer Psychology*) finds passive voice with spaCy at sentence or corpus level. It returns a binary flag, the matched spans and the proportion of passive sentences. It reports 98% agreement with human coding (0.97 accuracy on CoLA, 0.98 on a CrowdSource set). — [PassivePy on PyPI](https://pypi.org/project/PassivePy)

**Detecting LLM-flavoured vocabulary (relevant because the draft is Claude-written)**
- Kobak, González-Márquez, Horvát and Lause, "Delving into LLM-assisted writing in biomedical publications through excess vocabulary", *Science Advances* 11(27), eadt3813 (2 July 2025). The paper applies an excess-mortality-style "excess word" method to 2010–2024 PubMed abstracts. It estimates that at least 13.5% of 2024 abstracts were LLM-processed (up to about 40% in some subcorpora; the preprint said 10% and 30%). Marker style words include "delves" (excess ratio r = 28.0), "showcasing", "crucial" and "pivotal". The code is public at berenslab/llm-excess-vocab. — [arXiv 2406.07016v5](https://arxiv.org/abs/2406.07016v5)

### Inferences
- **Which method for which job:**
  - *Word/n-gram over- and under-use (main flagger):* Fightin' Words z-scores, with the draft as group i and the corpus as group j. The prior should come from a larger background, either the pooled draft+corpus or a broader econ/finance open-access corpus, as footnote 18 allows. Run it separately on unigrams, bigrams, trigrams and POS-tag n-grams. Unlike raw LL, the prior and variance handle the very unequal sizes (a draft of about 10–15k words against a corpus of about 1M words).
  - *Second opinion / reporting:* G2 for significance plus Log Ratio for magnitude. Sort by Log Ratio and filter on G2 ≥ 10.83, following the keyflux advice.
  - *TF-IDF:* good for finding topical terms (which HAR variants and which estimators get discussed), poor for style. It has no reference-corpus contrast and no variance, so it is not recommended as a flagger. I did not read Monroe et al.'s section on tf-idf; see Gaps.
  - *Global "does this read like JFEc RV papers" scores:* Burrows' Delta of the draft against the corpus centroid, computed over the top 200–500 function words, and pybiber MD dimension scores. Track these across draft revisions; they should fall toward within-corpus values.
  - *Sentence length, passive share, hedges, first-person rate:* compute each per corpus paper and report the draft's percentile (for example, flag if outside the 10th–90th percentile band). Do not test pooled sentences, because papers differ.
- **Under-use has low statistical power** against a short draft. A phrase expected 2–3 times in the draft that never appears gives a small |z|. Under-use output should therefore be "suggestions", for example canonical phrase frames such as "the HAR model of Corsi (2009)" or "the model confidence set of Hansen, Lunde and Nason (2011)". Over-use output can be "warnings".
- **Dispersion filter (my addition):** keep only corpus terms that appear in at least k papers (for example k ≥ 5, or ≥ 10% of papers). Otherwise one prolific author's idiosyncrasies become the "norm".
- **Math/LaTeX noise must be stripped** before counting: equations, citations, table cells and reference lists. Otherwise symbols and author names dominate the keyness lists.
- The Kobak marker list should seed a fixed "AI-ism" blacklist that is checked whatever the corpus statistics say. A Claude draft will over-use these relative to pre-2023 RV papers, and Fightin' Words should rediscover many of them, which makes a useful sanity check.

### Gaps
- I verified Monroe et al. eqs. 15–23 and the critiques on pp. 385–391 only. I did not read the paper's earlier comparison of raw frequency difference, tf-idf and other baselines (section 3, pp. ~376–384), so I make no claim about its exact critique of tf-idf.
- I did not retrieve the primary Gabrielatos and Marchi (2012) paper or Gabrielatos (2018) chapter, only slides and secondary tool documentation.
- I could not confirm whether `textstat_keyness` now lives in `quanteda.textstats`. The fetched reference page documents quanteda 2.1.2. From memory (unverified), it moved there in quanteda v3.
- I found no published validation of Fightin' Words or keyness specifically for "draft against target-journal corpus" style correction. The method transfer is my inference.

## Q2. Tools: Vale, proselint, LanguageTool, textstat, spaCy, AntConc, quanteda, scattertext. Can a corpus-derived word list become Vale rules automatically?

### Takeaway
Yes. Vale rules are plain YAML files. An `existence` rule takes a `tokens:` list and a `substitution` rule takes a `swap:` map (regex keys and `$1` capture groups allowed). A Python script can therefore write corpus-derived over-use lists and curated swaps straight into a `styles/JFEcRV/` folder. The Python-native components are spaCy, pybiber, PassivePy, textstat, faststylometry, scattertext and sentence-transformers. They cover the statistics. Vale, proselint and LanguageTool act as the rule-based "linters" on top. Vale does not list LaTeX as a supported format, so the `.tex` draft needs converting to Markdown or plain text first.

### Cited Findings
**Vale**
- An existence rule looks like this (verbatim from the docs):
  `extends: existence` / `message: Consider removing '%s'` / `level: warning` / `ignorecase: true` / `tokens: [appears to be, arguably]`.
  Tokens are phrases or regexes, compiled into one word-bounded group. Other fields are `nonword`, `raw`, `exceptions`, `action` and `vocab`. — [Vale docs: existence](https://docs.vale.sh/checks/existence)
- A substitution rule has a `swap:` map of `observed: expected`. Keys may be regexes, for example `'(?:give|gave) rise to': lead to`, and capture groups may be reused (`$1`). Messages take one or two `%s`. Other fields are `ignorecase`, `exceptions`, `capitalize` and `vocab`. — [Vale docs: substitution](https://docs.vale.sh/checks/substitution.md)
- The check types are existence, substitution, occurrence, repetition, consistency, conditional, capitalization, metric, readability, spelling, sequence and script. Configuration uses `StylesPath`, `Packages`, `BasedOnStyles` and `Vocabularies` (accept/reject lists). — [Vale docs index (llms.txt)](https://docs.vale.sh/llms.txt)
- The supported formats listed are Markdown, MyST, Quarto, R Markdown, Jupyter, AsciiDoc, MDX, HTML, reStructuredText, Sphinx, Typst, XML, Org, DITA, QDoc, plain text and code. LaTeX is not in the list. — [Vale docs index (llms.txt)](https://docs.vale.sh/llms.txt)

**proselint**
- proselint has check groups for clichés, jargon and bureaucratese (`industrial_language.*`), `hedging`, `weasel_words`, redundancy, misc, spelling, typography, terms, `social_awareness` and uncomparables (for example "very unique"). Run it from the CLI with `proselint check text.md` (with `--output-format json`) or through a Python API (`LintFile(...).lint()`). Checks are switched on and off hierarchically in `proselint.json`, and `per_file_checks` applies settings by glob. Licence BSD-3-Clause. The repo is active, with a 2026 footer and 1,455 commits. — [proselint GitHub](https://github.com/amperser/proselint)

**LanguageTool**
- `language_tool_python` starts a local LanguageTool Java server by default. The current README requires Java ≥17 (the default download is LT 6.8; older versions required Java 8). The server is about 200 MB. A `remote_server` URL or `LanguageToolPublicAPI` avoids Java but sends the text off-machine. — [language_tool_python GitHub](https://github.com/jxmorris12/language_tool_python)

**textstat**
- `textstat` 0.7.13 (18 Feb 2026, MIT) provides `flesch_reading_ease`, `flesch_kincaid_grade`, `gunning_fog`, `smog_index`, `coleman_liau_index`, `dale_chall_readability_score`, `text_standard`, `sentence_count`, `lexicon_count`, `syllable_count`, `polysyllabcount`, `difficult_words` and more. — [textstat on PyPI](https://pypi.org/project/textstat/)

**scattertext**
- Kessler, "Scattertext: a Browser-Based Tool for Visualizing how Corpora Differ", ACL System Demonstrations 2017 (arXiv 1703.00565). It finds distinguishing terms and draws an interactive HTML scatter plot. Demos cover log-odds ratio with an informative Dirichlet prior, Scaled F-Score, bi-normal separation, Cohen's d and Hedges' g, Cliff's delta, G², and others. `corpus.get_df()` returns term data. Apache-2.0, version 0.2.3. — [scattertext GitHub](https://github.com/JasonKessler/scattertext)

**Retrieval of similar passages**
- Sentence Transformers (SBERT), installed with `pip install -U sentence-transformers` (Python ≥3.10, PyTorch ≥1.11), computes embeddings for semantic search. It includes `semantic_search()` and `paraphrase_mining()` utilities. v6.2 was current at the access date. — [sbert.net](https://sbert.net/)

**Other**
- `pybiber`, `PassivePy` and `faststylometry`: see Q1.
- quanteda `textstat_keyness` (R): see Q1. — [quanteda](https://quanteda.io/reference/textstat_keyness.html)

### Inferences
**Tool fit for a Windows laptop**
- Everything except LanguageTool is pure Python plus a spaCy model, so it installs with pip.
- LanguageTool needs a Java 17+ JRE. It is optional, since grammar checking matters less than register.
- Vale is a single Go binary (installation not researched; see Gaps).
- AntConc and quanteda are useful for interactive exploration but unnecessary for the automated pipeline.

**Recipe sketch (implementable in Python)**
1. **Corpus.** Collect about 60–150 open-access RV-forecasting papers (see Q4 for sourcing). Convert PDF to text. Strip equations, citations, tables, captions and references. Segment by section (abstract / intro / data / results / conclusion), because the norms differ by section.
2. **Store only statistics.** Per paper, keep:
   - unigram–4-gram counts after lowercasing, and POS-tag trigram counts (spaCy);
   - the sentence-length vector and passive share (PassivePy);
   - first-person rate ("we", "I", "our");
   - hedge and booster counts;
   - pybiber's 67 feature rates.
   Also keep a small set of short "phrase frames", for example `we find that … (DM|Diebold–Mariano) test`, with the provenance of each.
3. **Background prior.** Use a broader econ/finance open-access text pool, or the pooled corpus, for the Fightin' Words α.
4. **Score the draft.** Compute z-scores for 1–3-grams, keep terms present in at least k corpus papers, and set thresholds:
   - z ≥ +3 → "over-used" (warning);
   - z ≤ −3 with corpus document frequency ≥ 30% → "literature-standard phrase you never use" (suggestion).
5. **Emit Vale rules automatically.** A Python script writes YAML with pyyaml into `styles/JFEcRV/`:
   - `OverUsed.yml` (existence, from the z ≥ +3 list);
   - `AIisms.yml` (existence, from the Kobak markers plus a hand list);
   - `Cochrane.yml` (existence/substitution, see Q5);
   - `Swaps.yml` (substitution, curated pairs such as `utilize: use`, `'leverag(e|es|ing)': use`, `showcase: show`);
   - a `metric`/`occurrence` rule for sentence length.
   Run `vale --output=JSON` on a pandoc-converted Markdown copy of the `.tex`, then map line numbers back.
6. **Global dashboard.** Report Delta distance to the corpus centroid; percentiles of mean sentence length, passive share and "we" rate; MD dimension scores; textstat grade.
7. **LLM rewrite pass.** See Q3. Retrieve the nearest corpus sentences per flagged draft sentence with sentence-transformers, give them as phrase frames, and require no shared 7+-word sequence with the corpus.
8. **Iterate.** Re-score after the rewrite. Accept a change only if the flags drop and the facts are unchanged (numbers, citations and claims diffed).

**Minimal code core for steps 4–5 (my construction, not from a source)**
```python
import numpy as np, yaml
def fightin_words(y_d, y_c, y_bg, a0=1000):
    # y_*: aligned count vectors (draft, corpus, background)
    a = a0 * y_bg / y_bg.sum()
    nd, nc = y_d.sum(), y_c.sum()
    ld = np.log((y_d + a) / (nd + a0 - y_d - a))
    lc = np.log((y_c + a) / (nc + a0 - y_c - a))
    return (ld - lc) / np.sqrt(1/(y_d + a) + 1/(y_c + a))   # z, eq. 16/20/22
# rule = {"extends": "existence", "message": "Over-used vs RV literature: '%s'",
#         "level": "warning", "ignorecase": True, "tokens": overused_terms}
# yaml.safe_dump(rule, open("styles/JFEcRV/OverUsed.yml", "w"))
```

### Gaps
- I did not research Vale's installation on Windows, its `TokenIgnores`/`BlockIgnores` keys for skipping LaTeX math, or whether any community package parses `.tex` directly. pandoc conversion is my assumption.
- I did not research AntConc or spaCy pipeline specifics (versions, models) beyond what pybiber and PassivePy require.
- I did not research PDF-to-text tools (PyMuPDF, pdftotext, GROBID) or their equation and reference stripping, though this matters for corpus quality.
- The sentence-transformers page fetch did not confirm CPU-only operation explicitly. CPU use is standard but unverified here.

## Q3. LLM style transfer from exemplar passages (few-shot): effectiveness, risks, and retrieval of how the literature phrases things

### Takeaway
The research evidence covers *authorship* style transfer on informal text, mostly Reddit, not academic register. There, plain few-shot prompting with an author's texts works but is beaten by tuned approaches; TinyStyler's authors report that it outperforms GPT-4. The main risk is copying. LLMs reproduce verbatim, paraphrased and idea-level content from source text, so exemplar-conditioned rewriting needs an automatic overlap check. LLM-typical vocabulary is detectable at scale, and JFEc screens manuscripts with third-party integrity services.

### Cited Findings
- Patel, Andrews and Callison-Burch, "Low-Resource Authorship Style Transfer: Can Non-Famous Authors Be Imitated?" (arXiv 2212.08986, revised to Nov 2024) define transfer to authors with little text, using 16 Reddit posts per target author. Their STYLL baseline is pure in-context learning: prompt an LLM with several of the target author's texts, with no training. They propose automatic metrics based on authorship-representation models. — [arXiv 2212.08986](https://arxiv.org/abs/2212.08986)
- A follow-up (ASTRAPOP, arXiv 2403.08043) treats in-context prompting as the baseline and reports that a two-stage tune-and-optimise method beats it on low-resource individual-author transfer. — [arXiv 2403.08043v2](https://arxiv.org/pdf/2403.08043v2)
- TinyStyler (Horvitz, Patel et al., Findings of EMNLP 2024), an ~800M-parameter model using authorship embeddings, is reported by its authors to outperform GPT-4 on authorship style transfer. — [arXiv 2406.15586](https://arxiv.org/abs/2406.15586v1)
- Lee, Le, Chen and Lee, "Do Language Models Plagiarize?" (WWW 2023), tested GPT-2 on 210,000 generated texts against their top-10 most similar training documents. They found verbatim, paraphrase and idea plagiarism beyond simple memorisation. Larger models and some decoding choices plagiarised more. Fine-tuning reduced verbatim copying but *increased* paraphrase and idea reuse. Detection code is public. — [arXiv 2203.07618](https://arxiv.org/abs/2203.07618v2)
- LLM marker words ("delves", "showcasing", "crucial", "pivotal") rose sharply in 2024 scientific abstracts, by enough to estimate the share of LLM-processed papers. — [Kobak et al. 2025, arXiv 2406.07016v5](https://arxiv.org/abs/2406.07016v5)
- JFEc's instructions say manuscripts "may also be screened, including with services provided by third parties" for integrity issues, and offer an optional AI-driven Paperpal Preflight check. The General Instructions page states no policy on LLM use in writing. — [JFEc General Instructions](https://academic.oup.com/jfec/pages/General_Instructions)
- Sentence Transformers provides `semantic_search()` and `paraphrase_mining()` for retrieving the corpus passages nearest a draft sentence. — [sbert.net](https://sbert.net/)

### Inferences
- **Evidence gap for this use case.** No study I found tests few-shot "write like JFEc/JoE RV papers" register transfer, or measures whether it improves referee perception. The authorship-transfer results suggest few-shot prompting moves surface style. Fine-tuned or embedding-guided methods do better, but they are overkill for one paper.
- **Recommended design: retrieval of phrase *frames*, not passages.** For each draft sentence type (for example a Diebold–Mariano or MCS result statement), retrieve 5–10 nearest corpus sentences with sentence-transformers. Abstract them into templates, for example "X significantly outperforms Y at the 5% level (DM test)" or "is included in the 90% MCS", and give the LLM the templates plus summary statistics such as the corpus median sentence length and "we" rate. Raw paragraphs should not go to the LLM. This lowers the verbatim-copy risk that Lee et al. document and keeps exemplar text short.
- **Mandatory post-check.** After any LLM rewrite, compute the longest common word n-gram between each rewritten sentence and the corpus (a hashed 7- or 8-gram index stored in step 2 of the recipe), ignoring boilerplate n-grams that appear in many papers. Flag any rare shared span. Separately, re-run the Kobak/AI-ism list, because LLM rewriting can *add* marker words while removing others.
- **Content-preservation guard.** Diff numbers, citations, model names and claim polarity before and after the rewrite. Style transfer is known to drift content, and in a forecasting paper a changed number or a flipped "outperforms" is far worse than a style flag. This is my inference; I found no source specific to academic text.
- **Mechanical rewrites first.** Deterministic Vale swaps are reproducible and auditable. The LLM should handle only the sentences that remain flagged.

### Gaps
- I found no empirical study of LLM few-shot style transfer to an academic-journal register, and none measuring plagiarism rates when exemplars are in the prompt (as opposed to training data). Lee et al. studied GPT-2 training-data reuse only.
- Sources disagree on the year of STYLL (2022 preprint against a 2023 citation). I did not verify the canonical venue.
- I did not research OUP's general (publisher-wide) AI-authorship policy. Only the JFEc General Instructions page was checked, and it is silent on LLM use.

## Q4. Copyright-safe corpus building

### Takeaway
The safest pattern is to download open-access versions (arXiv, SSRN, author sites) or licensed copies for personal, non-commercial research. Keep the raw files local and never redistribute them. Persist only non-expressive statistics (counts, n-gram frequencies, feature rates) plus very short phrase frames. US case law supports transformative, non-substitutive text mining; EU and UK law has explicit TDM exceptions for research. Copies obtained from pirated sources are the clear legal risk.

### Cited Findings
- **US, *Authors Guild v. Google*** (2d Cir., 16 Oct 2015): digitising books, building search and showing snippets were non-infringing fair uses. The court noted that the engine enables "text mining" and "data mining" research and that snippets do not substitute for the books. It relied on *Authors Guild v. HathiTrust*, 755 F.3d 87 (2d Cir. 2014), which held that text-searchable digital libraries are fair use. (Sources are secondary commentary, not the opinion itself.) — [Mondaq summary](https://www.mondaq.com/unitedstates/copyright/438238/transforming-works-the-second-circuit-rules-that-the-google-library-project-digitization-is-a-transformative-fair-use-in-authors-guild-v-google-inc); [CaseMine commentary](https://www.casemine.com/commentary/us/transformative-fair-use-in-digital-libraries:-second-circuit-upholds-google-books/view)
- **US, *Bartz v. Anthropic*** (N.D. Cal., Judge Alsup, 23 June 2025):
  - Training LLMs on books was "spectacularly" transformative fair use.
  - Digitising lawfully purchased print books was fair use.
  - Downloading over seven million pirated books into a permanent central library was *not* fair use.
  - A class was certified in August 2025, and a settlement reported at $1.5B followed in September 2025.
  - In *Kadrey v. Meta* (Judge Chhabria, 25 June 2025), a different fair-use analysis was applied.
  — [ChatGPT Is Eating the World (24 Jun 2025)](https://chatgptiseatingtheworld.com/2025/06/24/judge-alsup-grants-partial-summary-judgment-to-anthropic-ruling-training-copies-were-fair-use-but-judge-rules-no-fair-use-in-pirated-copies-of-books-used-to-build-a-central-library-they-are-infring/); [Publishers Weekly](https://www.publishersweekly.com/pw/by-topic/digital/copyright/article/98089-federal-judge-rules-ai-training-is-fair-use-in-anthropic-copyright-case.html); [Kilpatrick settlement alert (Sep 2025)](https://ktslaw.com/insights/alert/2025/9/parties-reach-a-landmark-settlement-in-the-bartz-v-anthropic-litigation)
- **EU, DSM Directive:** Article 3 is a mandatory TDM exception for research organisations and cultural-heritage institutions doing scientific research. Article 4 is a broader any-purpose exception that rightsholders can opt out of. — [Kluwer Copyright Blog](https://legalblogs.wolterskluwer.com/copyright-blog/the-uk-government-moves-forward-with-a-text-and-data-mining-exception-for-all-purposes/)
- **UK, s.29A CDPA:** lawful users may copy for computational analysis for *non-commercial research* only, and contracts cannot override this. In February 2023 the government said it would not proceed with an all-purposes TDM exception. A 2026 reaffirmation is reported by CASRAI but I did not verify it against a primary source. — [Reed Smith (Feb 2024)](https://www.reedsmith.com/en/perspectives/ai-in-entertainment-and-media/2024/02/text-and-data-mining-in-uk); [CASRAI on s.29A](https://casrai.org/wp/tag/section-29a-cdpa/)
- **OUP (JFEc's publisher):** library guides report that OUP permits non-commercial TDM of its journals, with Data.Mining@oup.com as the contact. Some institutional agreements spell this out (for example Sheffield's read-and-publish deal). OUP's contact page refers to a TDM Reservation Policy and the TDM Reservation Protocol (TDMRep). These are mostly third-party and partly dated guides. — [WUR library guide](https://wur.nl/en/show/text-and-data-mining-what-you-need-to-know-about-publisher-policies.htm); [Sheffield library](https://sheffield.ac.uk/library/research/text-and-data-mining-tdm/tdm-university-collections-and-subscribed-resources); [UChicago guide](https://guides.lib.uchicago.edu/textmining/journals); [OUP contact page](https://academic.oup.com/pages/contact-us)
- **Elsevier** (relevant for *Journal of Econometrics* and *Journal of Banking & Finance*, among others): its TDM licence covers subscription content for non-commercial research. For open-access content, the author's chosen licence governs. — [Elsevier TDM licence](https://www.elsevier.com/about/policies-and-standards/text-and-data-mining/license)
- **arXiv:** per arXiv help text quoted in a 2021 thread (dated), most articles carry the default non-exclusive licence, which does not let arXiv authorise others to redistribute them. Builders of indexes and tools on the full text should link back to arXiv. Bulk files live in a requester-pays S3 bucket without per-article licence fields, so licences must come from the rate-limited OAI API. bioRxiv uses a similar requester-pays model, and its authors consent to TDM at submission. — [arxiv-api Google Group](https://groups.google.com/g/arxiv-api/c/4a7i2AJLYMA); [bioRxiv TDM](https://www.biorxiv.org/tdm)

### Inferences
- **Practical rules for this project:**
  1. Download PDFs individually from arXiv (q-fin.ST, q-fin.RM, econ.EM, stat.AP), SSRN, RePEc/IDEAS, and author or university pages, or through the user's university subscription (OUP and Elsevier non-commercial TDM).
  2. Keep raw PDFs and text in a gitignored local folder, never in the repo or in any published artifact.
  3. Commit only the derived statistics: counts, n-gram tables (possibly hashed for n ≥ 5), feature rates and per-paper sentence-length summaries.
  4. Keep phrase frames to a few words with variables abstracted out.
  5. Never use shadow-library copies. *Bartz* draws exactly that line.
- Storing only aggregate counts and hashed long n-grams is non-expressive and cannot substitute for the papers. This fits the transformative, non-substitutive reasoning of *Google Books* and *HathiTrust* comfortably. It is my reading, not legal advice.
- Corpus selection should follow *style* norms of the target outlets (JFEc, *Journal of Econometrics*, JBES, *Journal of Applied Econometrics*, *International Journal of Forecasting*, RFS for HAR-type work), not only topic. Working-paper versions can be less polished than published versions, so a corpus built mostly from SSRN/arXiv may slightly under-represent copy-edited journal style.

### Gaps
- I did not retrieve OUP's own TDM policy text, only library summaries, and I did not check whether OUP's TDMRep opt-out applies to non-commercial academic mining.
- I did not check SSRN's terms of use for automated download (scraping SSRN may breach its terms even when individual downloads are allowed).
- Jurisdiction is unknown. If the user is US-based at a university (the repo mentions CARC, a USC cluster, but I did not confirm), US fair use plus the university licence terms apply. The EU and UK exceptions apply only there.
- I verified no primary court opinions; all case descriptions come from law-firm or press summaries.

## Q5. Economics/finance-specific style guidance (Cochrane, McCloskey, Thomson, Shapiro, Bellemare, Head, JFEc/OUP guidelines)

### Takeaway
Cochrane's "Writing Tips for Ph.D. Students" (dated 8 June 2005) is the most rule-like source and translates almost line by line into Vale rules: banned phrases, active voice, no adjectives on one's own results, and so on. Head's "Introduction Formula" gives a section template (hook → question → antecedents → value-added → roadmap). JFEc's binding constraints include an abstract of at most 100 words, 2–6 keywords, JEL codes, author-date references, Chicago Manual of Style (15th ed.), a typical maximum of about 40 double-spaced pages, and a replication package. The prescriptive guides sometimes conflict with corpus norms (for example passive voice and "we"), so the pipeline should report both and let the author choose.

### Cited Findings
**Cochrane, "Writing Tips for Ph.D. Students" (8 June 2005)** — [landing page](https://www.johnhcochrane.com/research-all/writing-tips-for-phd-studentsnbsp); [PDF](https://static1.squarespace.com/static/5e6033a4ea02d801f37e15bb/t/5eda74919c44fa5f87452697/1591374993570/phd_paper_writing.pdf)

*Structure*
- State the one central contribution concretely and write "triangular"/"newspaper" style, not "joke" or "novel" style: "Readers don't stick around to find the punchline in Table 12."
- Abstract: 100–150 words, no literature, "Say what you find, not what you look for."
- The introduction starts with the contribution, with no "clearing your throat" openers such as "Financial economists have long wondered…" or "The finance literature has long been interested in x". "Three pages is a good upper limit for the introduction." A roadmap paragraph is "not mandatory".
- The literature review comes after the contribution and sets the paper against "the 2 or 3 closest current papers".
- "There should be nothing before the main result that a reader does not need to know in order to understand the main result."
- Conclusions are short. Strike "I leave x for future research."
- Papers should run to no more than 40 pages.

*Phrase-level rules (Vale-ready)*
- "In other words" is "a sign of trouble". Previews and recalls ("As we will see in Table 6", "Recall from section 2") signal poor organisation.
- Use active voice: "Search for 'is' and 'are' in the document to root out every single passive sentence."
- "I" is fine for a sole author, but no royal "we". "We" means "you (the reader) and I".
- Prefer present tense and subject–verb–object sentences.
- No adjectives on one's own work ("striking results", "very significant"; not "very novel").
- "Use" not "utilize"; "several" not "diverse".
- "It should be noted that" is "particularly obnoxious". Delete everything before the "that" in "It is easy to show that", and strike "A comment is in order at this point."
- "Clothe the naked 'this'."
- Follow the JFE hyphen rule for compound modifiers.
- Never write "illustrative test" or "illustrative empirical work".
- Don't "assume" things about a model.
- Write "in which", not "where", for models.
- Don't abbreviate authors' names ("FF").
- No cute opening quotations. Don't overuse italics. Avoid "up (down)" constructions.

*Tables and results*
- Captions should be self-contained.
- "No number should appear in a table that is not discussed in the text." "Table 5 shows summary statistics" (with a full stop and nothing more) is not acceptable.
- Two to three significant digits.
- Explain economic, not only statistical, significance. "Every important number should include a standard error."

**Head, "The Introduction Formula"** (UBC blog; no post date shown)
- The order is Hook → Question (the "This paper addresses the question" sentence at the end of paragraph 2 or the start of paragraph 3) → Antecedents (the essential prior work, deficient but not insulted) → Value-Added (about three contributions, up to three paragraphs) → Road-map (short and tailored, not generic).
- Avoid "the bait and switch" and "all my friends are doing it" hooks.
- "If you can't write a good introduction, then you may be writing the wrong paper."
— [Keith Head, The Introduction Formula](https://blogs.ubc.ca/khead/research/research-advice/formula)

**Other guides**
- **McCloskey, *Economical Writing*, 3rd ed.:** *Thirty-Five Rules for Clear and Persuasive Prose* (University of Chicago Press, 2019; appendix by Stephen T. Ziliak). — [RePEc record](https://econpapers.repec.org/RePEc:ucp:bkecon:9780226448077); [Bookshop](https://bookshop.org/p/books/economical-writing-third-edition-thirty-five-rules-for-clear-and-persuasive-prose-deirdre-nansen-mccloskey/2e8d82a7d69715da)
- **Thomson, *A Guide for the Young Economist*, 2nd ed.:** MIT Press, July 2011, 184 pp. It covers writing papers (titles, abstracts, exposition), talks and refereeing; the writing chapter gained new material in this edition. A reviewer compared it to Strunk and White. — [Penguin AU listing](https://www.penguin.com.au/books/a-guide-for-the-young-economist-second-edition-9780262515894); [Porchlight](https://www.porchlightbooks.com/products/guide-for-the-young-economist-second-edition-william-thomson-9780262515894)
- **Bellemare, "How to Write Applied Papers in Economics" (2020):** has a dedicated conclusion section. A third-party checklist summarises it as: restate the question, method and findings briefly; state contributions and policy implications; give future directions; keep it to about 1–2 pages. That last point conflicts with Cochrane's advice to drop "future research". — [Bellemare PDF mirror](https://dayuwu.github.io/files/BellemareHowToPaper.pdf); [third-party checklist (Oct 2025)](https://thomassoseco.blogspot.com/2025/10/how-to-write-applied-papers-in.html)

**JFEc (OUP) General Instructions** — [JFEc General Instructions](https://academic.oup.com/jfec/pages/General_Instructions)
- "Submitted papers should typically not exceed 40 double-spaced typewritten pages."
- The abstract is "not more than 100 words" with "two to six keywords". Authors provide JEL codes.
- Author-date references. Chicago Manual of Style, 15th ed. Footnotes go at the end of the text.
- Empirical and numerical papers must upload source files and data for editorial replicability review as a condition of acceptance (for submissions from 13 March 2023).
- Manuscripts may be screened by third-party services. LaTeX submissions use a locally generated PDF (OUP "Modern Small" template recommended).
- Figures need at least 300 dpi with alt text. There is a US$65 submission fee for non-SoFiE members.

### Inferences
- **Turn Cochrane into a `Cochrane.yml` Vale style:**
  - *existence:* "it should be noted", "it is easy to show", "a comment is in order", "in other words", "as we will see", "recall from section", "for future research", "illustrative", "striking", "very significant", "very novel", "interesting(ly)";
  - *substitution:* `utilize→use`, `diverse→several`, `where→in which` after "model(s)";
  - a regex for naked "This"/"These" followed by a verb;
  - a passive-voice share check through PassivePy rather than Vale.
  Head's formula becomes a structural check on the introduction (a question sentence by paragraph 3, a list of about three contributions, a roadmap of at most one short paragraph), best done by an LLM judge with a rubric rather than regex.
- **Resolving prescription against corpus norms.** Cochrane would remove nearly all passives and the royal "we". RV-forecasting papers are mostly multi-author and likely use "we" and a fair number of passives in methods sections; this is a hypothesis for the corpus statistics to test. Rule: corpus norms win on register features (pronouns, passive share, hedging, sentence length), and Cochrane/McCloskey win on throat-clearing, self-praise adjectives and AI-isms. Those are negative in both sources and also flagged by the Kobak list.
- **JFEc hard limits as code.** An abstract word count of at most 100 (not Cochrane's 150), a keyword count of 2–6, JEL codes present, and a page budget. These are trivial to assert in a pre-submission script and should block a build, unlike the style flags, which are advisory.

### Gaps
- I did not read the full texts of McCloskey, Thomson or Bellemare. Their specific rules (for example McCloskey's word-level rules) are not captured here, and Bellemare's conclusion advice is from a third-party summary.
- I found no source for a Jesse Shapiro writing guide (his widely shared slide decks on applied-micro papers and talks were not retrieved) and none for a Goldin/Katz writing guide. Both remain unverified.
- I did not check OUP's publisher-wide author guidance (house style, AI-use policy) beyond the JFEc page.
- Whether JFEc copy-editing uses US or UK spelling is not stated. CMS 15th ed. implies US conventions; this is an inference.
