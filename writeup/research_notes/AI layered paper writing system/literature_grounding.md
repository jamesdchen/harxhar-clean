# Literature grounding and citation verification for an AI-assisted paper-writing loop

Notes compiled 2026-10-28. Context: a Journal of Financial Econometrics paper on realized-volatility forecasting, written with Claude Code agents. Agents maintain `writeup/references.bib`, and SSRN often blocks automated access. Each finding carries the date of its source where one is known. Figures from search-result summaries rather than the primary page are marked "(secondary)".

---

## Q1. Retrieval and reading systems (PaperQA2, OpenScholar, Elicit, Undermind, Consensus, scite, Semantic Scholar/OpenAlex APIs, Connected Papers, ResearchRabbit): what they do and how accurate they are

### Takeaway
Only two retrieve-and-read systems have published, benchmarked accuracy and can run locally from Python: PaperQA2 (FutureHouse, open source, works over a local PDF folder and with Claude through LiteLLM) and OpenScholar (AI2/UW, open source, 45M open-access papers, Nature 2026). The commercial tools (Elicit, Undermind, Consensus, scite) have thin or disputed independent accuracy evidence and are poorly suited to a scripted local loop. For finance and econometrics, where much of the canon is paywalled, PaperQA2 over a locally held PDF set is the most direct fit. OpenScholar's open-access-only datastore is likely to miss much of the journal literature.

### Cited Findings
**PaperQA2 (FutureHouse; arXiv 2409.13740, Sept 2024)**
- The paper reports that PaperQA2 matches or beats subject-matter experts on three literature tasks: retrieval, summarization, and contradiction detection. — [Skarlinski et al., arXiv 2409.13740](https://arxiv.org/abs/2409.13740v2)
- LitQA2 has 248 multiple-choice questions. Answers are meant to sit in a paper's main text rather than its abstract, ideally in only one paper, and an "insufficient information" option is allowed. Averaged over three runs, PaperQA2 scored 85.2% ± 1.1% precision and 66.0% ± 1.2% accuracy, ahead of the other RAG systems tested. — [arXiv 2409.13740](https://arxiv.org/abs/2409.13740v1) (secondary summary of abstract)
- Its cited, Wikipedia-style summaries ("WikiCrow") were judged more accurate than the existing human-written Wikipedia entries. On biology papers it found on average 2.34 ± 1.99 contradictions per paper, and human experts validated 70% of them. — [arXiv 2409.13740](https://arxiv.org/abs/2409.13740v2) (secondary)
- Install with `pip install "paper-qa>=5"` (Python 3.11+). Running `pqa ask '...'` inside a folder indexes the PDFs recursively and caches the index. In Python, use `Settings(agent={"index": {"paper_directory": ...}})`. The project moved to calendar versioning (e.g. `v2025.12.17`) after 5.29.1 (Dec 2025). — [paper-qa GitHub README](https://github.com/Future-House/paper-qa)
- Any LiteLLM-supported model works, including Anthropic Claude through `ANTHROPIC_API_KEY`. The defaults are OpenAI models, and the README warns that small (7B) local models do poorly. — [paper-qa README](https://github.com/Future-House/paper-qa)
- Metadata clients: Crossref (BibTeX), Semantic Scholar (citation counts, licenses), Unpaywall (OA PDF links) and OpenAlex, plus a retraction check. A manifest CSV (file, DOI, title) improves metadata accuracy, and Crossref/S2 API keys are recommended for 100+ papers. Answers cite source passages by document plus page. — [paper-qa README](https://github.com/Future-House/paper-qa)
- Modes and presets: the default `ask` runs an agent (`ToolSelector`) that searches, gathers evidence and answers in any order. Presets are `high_quality` (evidence_k = 15), `fast`, `wikicrow`, and `contracrow` for contradiction detection. — [paper-qa README](https://github.com/Future-House/paper-qa)

**OpenScholar (AI2/UW; arXiv 2411.14199 v1 21 Nov 2024; Nature, published 4 Feb 2026 per Ai2/Nature Asia)**
- OpenScholar is a retrieval-augmented LM over a datastore of 45 million open-access papers. Its ScholarQABench has 2,967 expert-written queries and 208 long-form answers across CS, physics, neuroscience and biomedicine. Code, models, datastore, data and demo are all open-sourced. — [arXiv 2411.14199](https://arxiv.org/abs/2411.14199)
- The abstract says GPT-4o hallucinates citations "78 to 90% of the time" and that OpenScholar "achieves citation accuracy on par with human experts". — [arXiv 2411.14199](https://arxiv.org/abs/2411.14199)
- Ai2's blog gives a narrower figure: on biomedical questions, GPT-4o hallucinated more than 90% of the papers it cited. — [Ai2 blog](https://allenai.org/blog/openscholar)
- Running GPT-4o inside the OpenScholar pipeline raised citation F1 on ScholarQA-CS from 0.1 to 39.5 and improved correctness by 12%. Experts preferred OpenScholar-8B and OpenScholar-GPT-4o answers over expert-written ones 51% and 70% of the time, against 32% for GPT-4o. — [arXiv 2411.14199 HTML](https://arxiv.org/html/2411.14199v1) (secondary)
- Publication: Nature, 4 Feb 2026, open access. The arXiv page still lists no journal reference, and I could not open the Nature page itself (auth redirect). — [Nature Asia highlight](https://www.natureasia.com/ja-jp/nature/pr-highlights/15460); [Ai2 blog](https://allenai.org/blog/openscholar)
- The developers' stated limitations: it does not always retrieve the most relevant papers, and results for the GPT-4o variant may be hard to reproduce as the underlying model changes. — [arXiv 2411.14199 HTML](https://arxiv.org/html/2411.14199v1) (secondary)

**Elicit / Undermind / Consensus (commercial)**
- There is no head-to-head independent accuracy study of the three. A 2025 study (Bernard et al.) found Elicit's results varied across repeated searches and that it missed articles. Two further studies (Lau & Golder 2025; Lagisz et al. 2026) are reported to give Elicit about 39–40% search sensitivity relative to traditional search, with inconsistent accuracy of supporting quotes and reasoning. I did not verify those directly. — [CASRAI Elicit vs Consensus](https://casrai.org/compare/elicit-vs-consensus) (secondary; not verified against the original studies)
- The same page found no independent peer-reviewed accuracy study of the Consensus Meter. — [CASRAI Elicit vs Consensus](https://casrai.org/compare/elicit-vs-consensus) (secondary)
- The most substantive Undermind source is a qualitative 2025 product review (Giustini, J. Canadian Health Libraries Assoc.) that calls it useful for starting a biomedical search. It contains no measured accuracy. — [CASRAI Undermind guide](https://www.casrai.org/guides/undermind-ai) (secondary)
- Secondary sources agree on positioning: Undermind for early-stage discovery, Elicit for structured extraction, Consensus for summarizing the consensus on a yes/no claim. — [AI Agent Index: Undermind vs Elicit](https://theaiagentindex.com/compare/undermind-vs-elicit) (commercial/editorial)

**scite (smart citations: supporting / contrasting / mentioning)**
- Method paper: Nicholson et al., "scite: A smart citation index that displays the context of citations and classifies their intent using deep learning," *Quantitative Science Studies* (2021). — [cited in Hypothesis evaluation](https://journals.iupui.edu/index.php/hypothesis/article/view/26528)
- Independent evaluation (Bakker, Theis-Mahon, Brown) on 324 citations from systematic reviews: overall accuracy was low, with F-measures from 0.0 to 0.58. scite tended to label supporting and contrasting citations as "mentioning". — [Hypothesis: Evaluating the Accuracy of scite](https://journals.indianapolis.iu.edu/index.php/hypothesis/article/view/26528)
- The developers' rebuttal (Rife, Nicholson, Uppala, Rosati) argues that the evaluation used definitions different from the classifier's training and a narrow sample (citations of retracted works). The authors then published a reply. — [Hypothesis rebuttal](https://journals.indianapolis.iu.edu/index.php/hypothesis/article/view/28018); [authors' reply](https://journals.indianapolis.iu.edu/index.php/hypothesis/article/view/28651)
- Developer-reported accuracy is 97% for mentioning, 64% for supporting and 59% for contrasting. — [HKUST Library trial note](https://library.hkust.edu.hk/sc/trial-scite/)

**Semantic Scholar and OpenAlex APIs (the programmable backbone)**
- Semantic Scholar: an API key comes with an introductory limit of 1 request per second on all endpoints. Unauthenticated requests share a 1000 RPS pool and may be throttled further, and rate-limited calls return HTTP 429. Third-party wrappers quote different keyless limits. — [Semantic Scholar API tutorial](https://webflow.semanticscholar.org/product/api/tutorial); [semanticscholar Python lib docs](https://semanticscholar.readthedocs.io/en/latest/usage.html) (figures conflict across sources)
- Semantic Scholar has a snippet-search endpoint that returns excerpts of about 500 words from a paper's title, abstract and body text. — [Semantic Scholar API tutorial](https://webflow.semanticscholar.org/product/api/tutorial) (secondary summary; I could not render the official `/api-docs/graph` page to confirm the path and fields)
- OpenAlex has required an API key since 13 Feb 2026, and keys are free. Keyless use is capped at a tiny daily budget: the docs say $0.01/day, and one blog notes a $0.10 discrepancy. A free key gets a $1/day budget, and all users are capped at 100 requests/s. Pricing is usage-based per endpoint type, and a `/rate-limit` endpoint reports usage. — [OpenAlex authentication docs](https://developers.openalex.org/api-reference/authentication); [openalex-users announcement](https://groups.google.com/g/openalex-users/c/rI1GIAySpVQ); [OpenAlex blog](https://blog.openalex.org/author/jason)

**Connected Papers / ResearchRabbit**
- No accuracy or API evidence was found for either in this session (see Gaps).

### Inferences
- For a JoFE paper, the related literature (HAR, realized measures, QLIKE/MCS/DM testing, 0DTE options) sits mostly in Elsevier/OUP/Wiley/Taylor & Francis journals plus SSRN working papers. Most of it is not open access, so OpenScholar's 45M-paper open-access datastore probably under-covers it. PaperQA2 over a local folder of legitimately obtained PDFs (from Zotero, a library proxy, or author copies) avoids the coverage problem and the SSRN blocking.
- PaperQA2's `contracrow` preset (contradiction detection) and page-level passage citations fit a "does source X say Y?" check directly. Running it with Claude via LiteLLM keeps the stack single-vendor.
- In a scripted loop, the commercial tools (Elicit, Undermind, Consensus) are useful at most as human-facing discovery aids. Their benchmarking is weak, and none was confirmed to have an API usable in an agent loop.
- scite's supporting/contrasting labels should not count as evidence of support. Independent accuracy on those two classes is low (F ≤ 0.58), and developer-reported figures are only about 60%.
- For metadata and graph queries, the free routes are Semantic Scholar (key, 1 RPS) and OpenAlex (key since Feb 2026, $1/day free budget). A local cache, such as a JSON file keyed by DOI, avoids re-spending the OpenAlex budget on every agent run.

### Gaps
- I could not open the Nature version of OpenScholar to check whether the numbers changed from the 2024 preprint (the margin over GPT-4o is reported as 5% in one place and 6.1% in another).
- Elicit's 39–40% sensitivity figure and the Bernard 2025 / Lau & Golder 2025 / Lagisz 2026 studies were not verified at source.
- Nothing found on Connected Papers or ResearchRabbit accuracy, API access, or 2025–2026 changes.
- Current Elicit, Consensus, Undermind and scite APIs, pricing, and MCP availability were not confirmed.
- Neither PaperQA2 nor OpenScholar was found to have been evaluated on finance or econometrics questions. All published benchmarks are biomedicine, CS, physics or neuroscience.

---

## Q2. Citation verification: checking that BibTeX entries exist (Crossref/DOI/OpenAlex), detecting hallucinated references, and checking that each citation supports its sentence; plus LLM citation-hallucination rates (2023–2026)

### Takeaway
Checking that a reference exists is a solved, scriptable problem. Several open-source tools (RefChecker, bibtex-verifier, CheckIfExist, clibib) match .bib entries against Crossref, OpenAlex and Semantic Scholar. The 2026 evidence says agents should fetch BibTeX deterministically rather than write it from memory: even search-enabled frontier models (GPT-5, Claude Sonnet 4.6, Gemini 3 Flash) produced fully correct entries only 50.9% of the time. Checking that a citation supports its sentence is much less mature: research prototypes (CiteAudit, SemanticCite, CiteGuard) reach roughly human-level on narrow benchmarks, but none is a turnkey tool. Hallucinated references are now being found in peer-reviewed venues (51+ NeurIPS 2025 accepted papers).

### Cited Findings
**Hallucination rates (chronological)**
- Walters & Wilder, *Scientific Reports* (Sept 2023) had ChatGPT write literature reviews on 42 topics and checked 636 citations from 84 papers. Fabricated: 55% (GPT-3.5) vs 18% (GPT-4). Among real citations, 43% (GPT-3.5) and 24% (GPT-4) had substantive errors. — [Walters & Wilder 2023, Sci Rep](https://www.nature.com/articles/s41598-023-41032-5.pdf)
- A 2025 JMIR Mental Health study found GPT-4o produced 176 citations across 6 reviews, of which 35 (19.9%) were fabricated. — [JMIR Mental Health 2025](https://mental.jmir.org/2025/1/e80371/PDF) (secondary summary)
- OpenScholar (Nov 2024 preprint, Nature 2026) reports that GPT-4o hallucinates citations 78–90% of the time when asked about up-to-date literature. — [arXiv 2411.14199](https://arxiv.org/abs/2411.14199)
- Naser (arXiv 2603.03299, Feb 2026) audited 10 commercial LLMs across 4 domains and 69,557 citations, checked against Crossref, OpenAlex and Semantic Scholar. Hallucination rates ran from 11.4% to 56.8% depending on model, domain and prompt. Newer generations did not reliably reduce hallucination, while larger models within a family helped. Citations produced by more than 3 LLMs were 95.6% accurate, and citations repeated across more than 2 prompt replications were 88.9% accurate. A string-feature classifier with no database lookup reached AUC 0.876 (CV) and 0.834 (leave-one-model-out). — [Naser, arXiv 2603.03299](https://arxiv.org/abs/2603.03299)
- Rao & Callison-Burch (arXiv 2604.03159, Apr 2026, v2 Aug 2026) built a 931-paper benchmark across 4 domains and 3 tiers (popular, low-citation, post-cutoff) and tested GPT-5, Claude Sonnet 4.6 and Gemini 3 Flash with web search, about 23,000 field observations. BibTeX field accuracy was 83.6%, but only 50.9% of entries were fully correct. Accuracy fell 27.7 pp from popular to recent papers, which points to reliance on parametric memory even with search available. Errors were omission, partial corruption, substitution and hallucination, in two failure modes: wholesale entry substitution and isolated field error. — [Rao & Callison-Burch, arXiv 2604.03159](https://arxiv.org/abs/2604.03159)
- Rao, Wong & Callison-Burch (arXiv 2604.03173, Apr 2026) tested 10 models and deep-research agents on DRBench (53,090 URLs) and 3 models on ExpertQA (168,021 URLs). 3–13% of citation URLs were hallucinated (never archived), and 5–18% did not resolve. Deep-research agents cite more but hallucinate at higher rates. — [arXiv 2604.03173](https://arxiv.org/abs/2604.03173)

**Hallucinated references in submitted and accepted papers**
- In ICLR 2026, GPTZero scanned about 300 of roughly 20,000 submissions and found 50 with at least one fabricated citation. Most of the 3–5 reviewers per paper missed them, and some flagged papers averaged 8/10 ratings. — [GPTZero ICLR 2026 investigation](https://gptzero.me/news/iclr-2026/)
- In NeurIPS 2025 accepted papers, GPTZero analysed 4,841 papers and found at least 100 confirmed hallucinated citations across 51 papers, each confirmed by a human expert. An academic analysis (arXiv 2602.05930) using GPTZero data counts at least 53 papers (about 1%). — [GPTZero NeurIPS](https://gptzero.me/news/neurips/); [Fortune, 21 Jan 2026](https://www.fortune.com/2026/01/21/neurips-ai-conferences-research-papers-hallucinations); [arXiv 2602.05930](https://arxiv.org/html/2602.05930) (counts differ)
- Failure patterns seen include a real NeurIPS 2023 paper cited with a wholly fictional author list, and a real arXiv ID attached to a different title and authors. Fortune reports that ICLR hired GPTZero to check future submissions, and NeurIPS and ICLR policies treat hallucinated citations as grounds for rejection or revocation. — [The Decoder](https://the-decoder.com/?p=31267); [Fortune](https://www.fortune.com/2026/01/21/neurips-ai-conferences-research-papers-hallucinations). GPTZero's claimed >99% accuracy has not been independently audited.

**Existence and metadata verification tools (scriptable)**
- **RefChecker** (`pip install academic-refchecker`, by Mark Russinovich) checks references against Semantic Scholar, OpenAlex and Crossref. It takes BibTeX, LaTeX, PDF or arXiv IDs, exports corrected BibTeX, has a web UI, and uses an LLM with web search to flag likely fabricated entries. "Unverified" means the reference was found in none of the databases. — [PyPI academic-refchecker](https://pypi.org/project/academic-refchecker/3.0.101/)
- **bibtex-verifier** (PyPI, MIT, GitHub Altman-conquer/bibtex-verifier) is a CLI that looks up the DOI in Crossref first and falls back to an OpenAlex fuzzy title search. It writes a Markdown report and checks metadata only, not claim support. — [PyPI bibtex-verifier](https://pypi.org/project/bibtex-verifier/)
- **CheckIfExist** (arXiv 2602.15871, early 2026) is an open-source web tool. It validates references against Crossref, Semantic Scholar and OpenAlex by string-similarity scoring and accepts batch BibTeX. — [arXiv 2602.15871](https://arxiv.org/html/2602.15871v1)
- **clibib** (Rao & Callison-Burch, MIT, ships with an agent skill) does deterministic BibTeX retrieval inside a two-stage architecture: search first, then a separate revision step. It raised field accuracy to 91.5% (+8.0 pp) and fully correct entries to 78.3%, with a 0.8% regression rate. Single-stage tool loops had a 4.8% regression rate. — [arXiv 2604.03159](https://arxiv.org/abs/2604.03159)
- **urlhealth** (open source, with data) checks URL liveness and uses the Wayback Machine to tell link rot from hallucination. Models with urlhealth access cut non-resolving citation URLs 6–79× to under 1%, with results depending on the model's tool-use ability. — [arXiv 2604.03173](https://arxiv.org/abs/2604.03173)
- **CiteAudit** (arXiv 2602.23452, Feb 2026, v3 1 May 2026; code at github.com/shiiiikw/CiteAudit) is a benchmark plus a multi-agent detector: metadata extraction, memory lookup, web retrieval, then judgment. It uses a human-validated multi-domain dataset and reports beating LLM and commercial baselines, though the abstract gives no figures. — [arXiv 2602.23452](https://arxiv.org/abs/2602.23452). A search-engine summary described CiteAudit as also checking claims against passages; the abstract itself describes reference verification only.
- A general caveat: a fabricated reference that carries a real DOI or a real title can pass a metadata-only check. — [summary of RefChecker/bibtex-verifier docs](https://pypi.org/project/bibtex-verifier/)

**Claim–citation support checking (does the cited paper back the sentence?)**
- **SemanticCite** analyses the cited source's full text and classifies support into four classes. Fine-tuned lightweight models perform comparably to large commercial systems at much lower cost. — [AITopics listing of SemanticCite (arXiv)](https://aitopics.org/doc/arxivorg:F3487F90)
- **CiteGuard** (arXiv 2510.17853, Oct 2025) recasts citation checking as attribution alignment with retrieval-backed validation. It scores 65.4% on CiteME, against 69.7% for humans. — [arXiv 2510.17853](https://arxiv.org/abs/2510.17853v2)
- **CiteVerify** (student project, not peer-reviewed) checks the claim against the cited paper's abstract first, escalates to full-text passages if needed, and returns a 3-class verdict. — [First Principles article](https://www.firstprinciples.org/article/fake-citation-detection-for-ai-research)
- Benchmarks: ALCE (2023) scores citation quality with metrics that correlate with human judgment. L-CiteEval (2024/25) covers 11 tasks with 8K–48K contexts and finds retrieval improves faithfulness. AuthorityBench (arXiv 2606.13104, 2026; 220k prompts) finds that the presence of a citation increases hallucination across 7 models. — [ALCE](https://ar5iv.labs.arxiv.org/html/2305.14627); [L-CiteEval arXiv 2410.02115](https://arxiv.org/pdf/2410.02115v1); [AuthorityBench arXiv 2606.13104](https://arxiv.org/html/2606.13104)
- PaperQA2's contradiction detection (`contracrow`) finds statements in a paper that conflict with the wider literature, and experts validated 70% of its flagged contradictions. — [arXiv 2409.13740](https://arxiv.org/abs/2409.13740v2); [paper-qa README](https://github.com/Future-House/paper-qa)

**Local repo state (checked 2026-10-28)**
- `writeup/references.bib` has 148 entries (`^@` lines). 73 lines contain a `doi =` field, 28 lines mention SSRN, and 12 mention arXiv/eprint. The Overleaf upload copy (`writeup/overleaf-upload/references.bib`) has 71 entries and 1 DOI line, so the two copies differ. — local files `C:\Users\james\CC Allowed\harxhar-0dte-professor\writeup\references.bib`, `...\writeup\overleaf-upload\references.bib` (grep line counts, not parsed fields)

### Inferences
- The Rao & Callison-Burch result applies directly to the paper's setup. Agents that write `references.bib` entries from memory, even with search, will leave about half the entries partially wrong, and recent or low-citation papers fare worst. 0DTE and recent SSRN papers fall in exactly that tier. The fix is to have agents never type BibTeX fields. They should fetch by DOI from Crossref (or by title from OpenAlex/S2) and only revise in a separate, gated second stage (the clibib pattern).
- About 75 of the 148 entries appear to lack a DOI field, and many of those are probably SSRN working papers or books. These are where a metadata check is weakest and where a deterministic re-fetch helps most.
- A practical local pipeline:
  - (1) Parse the .bib with `bibtexparser`.
  - (2) For entries with a DOI, use Crossref `/works/{doi}` and compare title, authors, year and journal by fuzzy match.
  - (3) For entries without a DOI, run an OpenAlex title search plus a Semantic Scholar title match.
  - (4) Flag mismatches and record a `verified: date, source, id` sidecar.
  - (5) Run RefChecker as a second, independent opinion.

  This can be a pre-commit or pre-Overleaf-push gate.
- Claim support is the hard part, and no off-the-shelf tool is reliable enough to gate on. A workable design: extract every `\cite{}` sentence from the .tex, retrieve evidence passages with PaperQA2 over the local PDF of each cited work (falling back to the abstract from S2/OpenAlex), and have Claude return a verdict (supported / partially / not supported / cannot assess) with the quoted passage and page. Only "not supported" goes to the human. This follows the abstract-first, full-text-escalation pattern from CiteVerify and SemanticCite.
- The Naser consensus result suggests a cheap extra screen for any citation an agent proposes: accept candidates only when they are independently retrievable from a database, never because the model "remembers" them.

### Gaps
- I did not fetch Crossref REST API documentation in this session (polite-pool `mailto`, the Metadata Plus key, rate limits), so those details need checking at source.
- Whether SSRN working-paper DOIs (10.2139/ssrn.*) resolve cleanly through Crossref and OpenAlex metadata was not verified.
- CiteAudit's and SemanticCite's quantitative accuracy and the SemanticCite arXiv ID and date were not confirmed from primary pages.
- There are no studies of citation hallucination specific to economics or finance. All the audits above are cross-domain, CS/ML, or medical.
- Chelli et al. (2024) systematic-review fabrication rates could not be confirmed. A secondary source attributes a range of 28.6% (GPT-4) to 91.4% (Bard) to a different author, so this needs verification.

---

## Q3. Mining how the literature phrases standard statements (e.g., reporting Diebold–Mariano tests or QLIKE results) from a retrieved corpus, under copyright constraints

### Takeaway
A tool built specifically for mining phrasing conventions from a target-journal corpus was not found. The building blocks exist: Semantic Scholar snippet search (~500-word excerpts of body text), PaperQA2 over a local PDF corpus, and arXiv bulk full text through S3 or Kaggle. Copyright allows mining for research in the EU, UK (non-commercial), US (fair use) and Japan, but redistributing the text is restricted. The safe pattern is to mine locally, keep only short n-gram or template statistics plus pointers, and never paste source sentences into the draft.

### Cited Findings
- Semantic Scholar snippet search returns about 500-word excerpts from a paper's title, abstract and body text, which makes it a direct way to query "how do papers phrase X" without holding the PDFs. — [Semantic Scholar API tutorial](https://webflow.semanticscholar.org/product/api/tutorial) (secondary; official endpoint page not rendered)
- PaperQA2 indexes any local folder of PDFs and returns evidence passages with document and page citations. The same machinery can answer phrasing queries such as "how do these papers report DM test results" over a curated JoFE/JoE/JBES corpus. — [paper-qa README](https://github.com/Future-House/paper-qa)
- arXiv full text is available in bulk through Kaggle and a requester-pays Amazon S3 bucket. arXiv asks people not to scrape the corpus and points to export.arxiv.org for harvesting. — [arXiv bulk data help](https://info.arxiv.org/help/bulk_data.html); [NCSU Libraries note on arXiv S3](https://www.lib.ncsu.edu/node/42245)
- Most arXiv papers carry the default non-exclusive licence, so arXiv cannot grant third parties the right to redistribute them. Tools built on the full text should link back rather than re-host. Per-article licences come from OAI-PMH metadata, which is rate-limited, because the S3 files do not carry them. — [arxiv-api Google Group thread](https://groups.google.com/g/arxiv-api/c/4a7i2AJLYMA); [arXiv bulk data help](https://info.arxiv.org/help/bulk_data.html)
- Legal basis for text and data mining: the EU 2019 CDSM Directive requires a research TDM exception (but database owners can override it by contract). The UK has a non-commercial TDM exception. In the US the route is fair use, as in HathiTrust, where the court found the research purpose strongly favoured fair use. Japan has had a TDM exception since 2009, amended in 2018. Researchers often limit themselves to openly licensed material, which skews corpora; for example, only 62% of UK PMC malaria articles are open to TDM. — [Creative Commons: The Limits of Copyright: TDM](https://creativecommons.org/?p=44748); [CACM: TDM of in-copyright works (2021)](https://cacm.acm.org/magazines/2021/11/256379-text-and-data-mining-of-in-copyright-works/abstract) (secondary aggregation; claims not attributed to a specific source sentence)
- Publishers run their own TDM programmes (e.g. Elsevier's TDM policy). — [Elsevier TDM policy](https://www.elsevier.com/about/policies/text-and-data-mining) (page not read in detail)

### Inferences
- The JoFE literature mostly lives in subscription journals, plus SSRN, which blocks automated access. A realistic phrasing corpus is therefore (a) PDFs the author already holds legitimately (Zotero, institutional access), indexed locally with PaperQA2 or a simple chunk index, plus (b) open arXiv econometrics papers (econ.EM, q-fin.ST) from arXiv bulk or the API, plus (c) S2 snippet search for papers not held locally.
- Output that respects copyright: store pattern-level abstractions and pointers, such as "DM statistic reported as t-type with HAC/Newey–West variance; sign convention: negative favours model 1; reported with p-value in parentheses, ** at 5%", each with bibkey and page. Do not store copied sentences. The writing agent then drafts in its own words from the pattern card. Short quotations for internal reference are a different matter from text that ends up in the manuscript.
- A plagiarism and overlap check should run on the final draft against the mined passages (n-gram overlap against the local corpus) so that mined phrasing does not leak verbatim into the paper.
- PaperQA2 query templates could target the econometric conventions that matter here: the DM test (HLN small-sample correction, one- vs two-sided), MCS (Hansen–Lunde–Nason; T_max vs T_R statistics, block bootstrap), QLIKE loss normalisation, and Mincer–Zarnowitz R². Each answer should be logged with page-level citations into the literature map (Q4).

### Gaps
- No published tool or study was found that mines phrasing conventions from a target-journal corpus for AI writing. This appears to be unbuilt.
- The licence terms of S2ORC and unarXive (full-text derived corpora) were not confirmed in this session.
- The exact Semantic Scholar terms of use on snippet text, and the snippet endpoint's exact path, fields and rate limit, were not confirmed from the official docs (the page did not render).
- No authoritative statement was found on whether SSRN's terms permit text mining of downloaded PDFs, or what its automated-access policy is.

---

## Q4. Keeping a literature map (paper -> claims it supports -> where cited in our draft) maintained by agents

### Takeaway
No existing tool maintains a paper → supported-claims → draft-location map. The nearest pieces are Zotero MCP servers that let Claude Code read and write a Zotero library and notes (one, Zoteus, advertises a "scholarly-context graph"), plus local chunk-index MCP servers for passage retrieval. The map will probably have to be a small repo-local ledger built from the verification and support-check pipeline (Q2) and checked in CI.

### Cited Findings
- **zotero-mcp** (Python; `claude mcp add zotero -- zotero-mcp serve`) exposes the Zotero library to Claude Code, including a `literature_review` tool that drafts a structured review template from a collection and a `paper_summary` tool. — [Glama: zotero-mcp](https://glama.ai/mcp/servers/54yyyu/zotero-mcp)
- **Zoteus** (TypeScript MCP server) offers search, writes, CSL citations, add-by-DOI, hybrid semantic search and a "scholarly-context graph", and works with Claude Code and other MCP clients. — [github.com/oscardvs/zoteus](https://github.com/oscardvs/zoteus)
- **zotero-mcp-lite** adds literature-review prompts, including a per-paper `literature_review(item_key)` deep-dive, and can write notes back into Zotero. — [PyPI zotero-mcp-lite](https://pypi.org/project/zotero-mcp-lite/0.1.3/)
- **zotero-chunk-mcp** (ccam80) embeds PDFs into ChromaDB and exposes `search_papers`, `search_topic`, `get_passage_context` and `get_index_stats` to Claude Code. It is a retrieval layer, not a claims map. — [LobeHub listing](https://lobehub.com/es/mcp/ccam80-zotero-chunk-mcp)
- **Claude Scholar** includes a literature-reviewer agent and `/research-init`, `/zotero-review` and `/zotero-notes` commands that use Zotero MCP. — [Claude Scholar MCP setup guide](https://gitcode.com/Dawngammad/claude-scholar/blob/main/MCP_SETUP.md)
- PaperQA2 caches its index and answers under `PQA_HOME` (default `~/.pqa/`), and answers carry passage-level citation keys. Those could serve as the evidence records for a claims ledger. — [paper-qa README](https://github.com/Future-House/paper-qa)
- clibib ships an "agent skill" for deterministic BibTeX retrieval, a pattern that carries over to a Claude Code skill keeping bib entries in sync. — [arXiv 2604.03159](https://arxiv.org/abs/2604.03159)

### Inferences
- Proposed ledger (repo-local, versioned, agent-maintained): `writeup/litmap.jsonl` (or YAML), one record per bibkey:
  - identity: `doi` / `openalex_id` / `s2_id`, `verified_meta: {source, date, match_score}`, `pdf_path` (local, not committed)
  - `claims`: a list of `{claim_id, our_sentence_hash, tex_file:line, verdict, evidence_quote_short, page, checked_by, date}`
  - `phrasing_cards`: links to Q3 pattern cards
- Maintenance loop, run by agents on each draft change:
  - (1) Scan the `.tex` for `\cite*{}` (citation keys plus the surrounding sentence) and diff against the ledger.
  - (2) Give new or changed sentences a support check (PaperQA2 or Claude against the local PDF or abstract).
  - (3) Re-verify new bibkeys' metadata against Crossref, OpenAlex and S2.
  - (4) Flag orphans: bib entries never cited, and cites whose sentence changed since the last verdict.
  - (5) Block the Overleaf push on "not supported", "unverified metadata", or a missing key.
- The pipeline's own outputs fill in the ledger fields, so the map stays current without separate manual upkeep. A sentence hash makes stale verdicts detectable whenever a sentence is edited.
- Zotero suits holding PDFs and human notes, but the claims map belongs in the repo, next to the .tex, so that git history and Overleaf sync keep it aligned with the draft. Zotero MCP can be the PDF and metadata source.
- The two `references.bib` copies (148 vs 71 entries) show why one source of truth plus a generated export matters. The ledger should key off the canonical `writeup/references.bib`.

### Gaps
- No published system or study was found on agent-maintained claim-to-citation maps for a manuscript, and no evaluation of such maps for preventing citation drift.
- How Zoteus's "scholarly-context graph" is built was not checked (README not read).
- Connected Papers and ResearchRabbit graph exports (which could seed the "related but uncited" side of the map) were not investigated, and neither was their API availability.
