# Precedents in single-source authoring, transclusion and document assembly for toggleable layers and define-once values and decisions

Scope note: researched on 2026-10-28. Dates and versions are given where the source states them. "(via search excerpt)" means the claim comes from a search-engine summary of the page, not from a full fetch, so it should be checked before it is quoted. "(repo)" means the claim comes from this repository's own files. Claims I believe from prior knowledge but did not verify this session are in the Gaps subsections, not in the findings.

---

## Q1. DITA and similar structured-authoring systems (conref, keys, conditional processing, maps; Flare, Antora/AsciiDoc, Paligo, Heretto, S1000D): how do they define once, build views by condition, and avoid stale reuse in shared content?

### Takeaway
Every mature system splits the work three ways:
- **Content units** are reused by reference, not by copying (DITA conref, Flare snippets, Paligo components).
- **Values** are bound indirectly through a named key or variable that is resolved at build time (DITA keys, Flare variables, AsciiDoc/Antora attributes).
- **Views** are selected by external condition files applied at build time (DITAVAL, Flare target conditions, AsciiDoc `ifdef`, S1000D applicability).

Stale reuse is avoided mainly by construction, because a reference always resolves to the current source. The remaining risk is the opposite one: a bad edit spreads everywhere. The documented defences are mostly organisational and tooling-based: "where used" lookups, dedicated shared folders, locked reuse, validity checks, and filter-before-resolve rules. A formal review gate specifically for shared content is rarely documented in what I could verify.

### Cited Findings

**DITA (OASIS) content reference (conref)**
- DITA v1.3 Errata02 (OASIS, dated 19 June 2018): "The DITA conref attributes provide mechanisms for reusing content." The spec presents them as covering reuse scenarios that "are difficult or impossible to implement using other XML-based inclusion mechanisms like XInclude and entities." — [DITA 1.3 Errata02, conref overview](https://docs.oasis-open.org/dita/dita/v1.3/errata02/os/complete/part3-all-inclusive/archSpec/base/conref.html)
- The spec lists `@conref`, `@conkeyref`, `@conrefend` (ranges) and `@conaction`, and says "These mechanisms can be used both to pull and push content." Validity rules exist so that "the results of content inclusion remain valid after resolution." Processors "compare the restrictions of each context" and "need to combine the attributes that are specified on the referencing and referenced element." — [DITA 1.3 Errata02, conref overview](https://docs.oasis-open.org/dita/dita/v1.3/errata02/os/complete/part3-all-inclusive/archSpec/base/conref.html)
- On conref processing: "A conref processor SHOULD NOT permit resolution of a reuse relationship that could be rendered invalid" under either side's rules. — [DITA 1.3 conref processing](https://docs.oasis-open.org/dita/dita/v1.3/os/part1-base/archSpec/base/conref-processing.html) (via search excerpt)
- Conref works at element granularity. It cannot cross element boundaries, so whole paragraphs and lists can be reused but not text fragments inside a paragraph. — [Stilo, "Are there limitations to conref in DITA?"](https://www.stilo.com/?p=81987) (via search excerpt; vendor FAQ)

**DITA keys (keyref, conkeyref): indirect addressing, i.e. "define once, bind in the map"**
- In DITA 1.2, `@conkeyref` "provides an indirect content reference to topic elements, map elements, or elements within maps or topics". Key references are resolved using key definitions from DITA maps. — [DITA 1.2 conkeyref](https://docs.oasis-open.org/dita/v1.2/os/spec/common/theconkeyrefattribute.html) (via search excerpt)
- In DITA 1.3, the value of `@conkeyref` is a key name plus an optional `/elementID`. — [DITA 1.3 langref @conkeyref](https://dita-lang.org/1.3/dita/langref/attributes/theconkeyrefattribute) (via search excerpt)
- Fallbacks: "If both @keyref and @href attributes are specified on an element, the @href value MUST be used as a fallback address when the key name is undefined." The same rule applies to `@conkeyref`, with `@conref` as the fallback. — [DITA 1.3 processing key references](https://docs.oasis-open.org/dita/dita/v1.3/os/part1-base/archSpec/base/processing-key-references-general.html) (via search excerpt)
- Duplicate keys: the first definition wins by document order. "How a processor finds the effective key definition depends on document order and the effect of filtering applied to the key definitions." Key scopes add precedence rules. In DITA 1.2, duplicates were ignored without a warning. — [DITA 1.3 processing key references](https://dita-lang.org/1.3/dita/archspec/base/processing-key-references-general); [DITA 1.2 overview of keys](https://docs.oasis-open.org/dita/v1.2/os/spec/archSpec/overview_of_keys.html) (both via search excerpt)
- **Order of filtering and key resolution is a known source of divergence.** DITA 1.2 says: "Processors should perform conditional processing before determining effective key definitions." DITA 1.3 keeps this as a SHOULD but admits "processors might determine effective key definitions before filtering", and the two orders can yield different bindings. — [DITA 1.2 overview of keys](https://docs.oasis-open.org/dita/v1.2/os/spec/archSpec/overview_of_keys.html); [DITA 1.3 processing key references](https://docs.oasis-open.org/dita/dita/v1.3/os/part1-base/archSpec/base/processing-key-references-general.html) (both via search excerpt)

**DITA conditional processing (profiling attributes plus DITAVAL)**
- The DITAVAL `<prop>` element takes `@att`, `@val` and `@action`. `@att` names a profiling attribute: `@props`, `@audience`, `@platform`, `@product`, `@otherprops`, `@deliveryTarget`, or a specialization of `@props`. — [DITA 1.3 DITAVAL prop](https://dita-lang.org/1.3/dita/langref/ditaval/ditaval-prop)
- The available actions are:
  - **include** (the default);
  - **exclude**, which removes content only "if all values in the attribute are excluded";
  - **passthrough**;
  - **flag**: "Include and flag the content on output (if the content has not been excluded)", styled with `@color`, `@backcolor` and `@style` (underline, bold and so on).
  — [DITA 1.3 DITAVAL prop](https://dita-lang.org/1.3/dita/langref/ditaval/ditaval-prop)
- Duplicate `<prop>` declarations are an error, and recovery is implementation-dependent. — [DITA 1.3 DITAVAL prop](https://dita-lang.org/1.3/dita/langref/ditaval/ditaval-prop)
- Branch filtering (DITA 1.3): "Without the branch filtering mechanism, the conditions specified in a DITAVAL document are applied globally. With branch filtering, the `<ditavalref>` element specifies a DITAVAL" for one branch of the map. — [OASIS DITA map glossary (2021 mailing-list PDF)](https://lists.oasis-open.org/archives/dita/202104/msg00009/map_glossary.pdf) (via search excerpt)

**DITA reuse practice, pitfalls and staleness**
- Oxygen XML gives this guidance (blog page, no article date, "© 2002-2026"):
  - Use conref for content reused in only one or two places.
  - For extensive reuse, "define a key and use conkeyref".
  - Avoid reusing inline elements if the content will be translated, except product names and constants.
  - Keep reused content in dedicated folders, because authors "might modify content which is potentially used in multiple places."
  — [Oxygen, DITA Reuse Strategies: Conclusions](https://blog.oxygenxml.com/presentation-reuse/conclusions.html)
- RWS/Tridion localization guidance warns against conref-ing nouns or adjectives inside sentences: the translator "cannot determine how to translate the article 'the' before the conref". It recommends reusing complete thoughts. — [RWS Community, Writing DITA with Localization in Mind](https://community.rws.com/product-groups/tridion/tridion-docs/b/weblog/posts/best-practices-writing-dita-with-localization-in-mind-tips-for-using-conrefs-and-keyrefs) (via search excerpt)
- Reusable content gets translated "without the context of the topics in which it is used". — [Prepare DITA for L10N: content reuse](https://prepare-dita-for-l10n.netlify.app/topics/c_content_reuse) (via search excerpt)
- "Locked" (verbatim) reuse: "locked components… can't be edited locally – any update must occur in the original source." — [RWS glossary, verbatim content reuse](https://www.rws.com/glossary/verbatim-content-reuse/) (via search excerpt)

**Ann Rockley, *Managing Enterprise Content***
- The book is *Managing Enterprise Content: A Unified Content Strategy*, by Rockley with Kostur and Manning (1st edition, 2002) and with Charles Cooper (2nd edition, 2012). Chapter 2 is "Fundamental concepts of reuse". — [O'Reilly, ch. 2](https://www.oreilly.com/library/view/managing-enterprise-content/0735713065/ch02.html); [O'Reilly, 2nd ed.](https://www.oreilly.com/library/view/managing-enterprise-content/9780132931656/copyright.html)
- Rockley's reuse taxonomy (2004 workshop announcement): "identical, derivative, nested, opportunistic, systematic". — [SIGIA-L, July 2004](https://asist-archive.ischool.illinois.edu/sigia-l/2004-July/010918.html)
- Nested reuse "allows authors to write a complete set of information in context, yet filter out specific pieces of the content where appropriate." This is the closest technical-communication precedent for "one source, filtered into short and long layers". — [The Rockley Report V1I2, Best Practices](https://rockley.com/TheRockleyReport/V1I2/Best%20Practices.htm) (via search excerpt)

**MadCap Flare**
- Each target is a separate output, for example an HTML5 or PDF build. Condition tags are applied to content and then included or excluded per target: "you do not need to create a separate project for each output that you want to produce." — [Flare 2024 help, Conditions and Single-Sourcing](https://help.madcapsoftware.com/flare2024/Content/Flare/Conditions/Other-Information/Conditions-Single-Sourcing.htm)
- Snippets are reusable chunks of formatted content (text, tables, images). Variables are for single words or very short phrases. — [Flare 2025 help, Single-Sourcing for eLearning](https://help.madcapsoftware.com/flare2025/Content/Flare/Learning-Development/General-Information/Single-Sourcing-eLearning.htm); [MadCap Flare single-sourcing page](https://www.madcapsoftware.com/products/flare/single-sourcing.aspx) (both via search excerpt)
- Regular conditions are resolved at the target level. "Snippet conditions" are resolved at the topic or master-page level, so one snippet can render differently in different places. — [MadCap single-sourcing page](https://www.madcapsoftware.com/products/flare/single-sourcing.aspx); [Flare Snippet Conditions Tutorial (2024 PDF)](https://docs.madcapsoftware.com/flare2024/Flare-Snippet-Conditions-Tutorial.pdf) (via search excerpt)

**AsciiDoc and Antora**
- AsciiDoc has three conditional preprocessor directives: `ifdef`, `ifndef` and `ifeval`. "If the condition evaluates to true, the lines the conditional encloses are included." These directives run before parsing: "A preprocessor directive merely adds lines to the reader or takes lines away", so they do not know about document structure. — [Asciidoctor docs, Conditionals](https://docs.asciidoctor.org/asciidoc/latest/directives/conditionals/)
- Antora defines attributes at three levels: the site playbook, the component version (`antora.yml`), and the page header.
  - **Hard set** in the playbook: "Hard set site attributes cannot be overridden from a component version descriptor or a page."
  - **Soft set** with a trailing `@`: lower levels can override the value.
  - **Hard unset** (`~`) and **soft unset** (`false`) work the same way for removing an attribute.
  - Attributes can reference earlier attributes.
  — [Antora docs, AsciiDoc attributes in playbook](https://docs.antora.org/antora/latest/playbook/asciidoc-attributes/)

**Paligo and Heretto (component content management systems, CCMS)**
- Paligo: "write the content once, in one place and reuse that content in multiple topics and publications". Reuse works at the topic level and the element level (paragraphs, warnings, procedures). It is combined with variables and filtering to "vary the content depending on product and still use the same topics." — [Paligo docs, About reusing content](https://docs.paligo.net/en/about-reusing-content.html)
- Paligo has "where used" features to find where content is reused. Changes to reused content apply wherever it is reused. A topic branch "keeps any reused content from the original topic and creates new versions of any content that only existed in the original topic". — [Paligo, How to find reusable content](https://docs.paligo.net/en/how-to-find-reusable-content.html); [Paligo, What happens when you create a branch](https://docs.paligo.net/en/what-happens-when-you-create-a-branch-.html) (both via search excerpt)
- Heretto: a customer style guide (Medallia) tells writers to "Put the source of the shared content in a folder named _Shared, and put that folder at the highest directory where it will be used." — [Medallia style guide, Reusing content in Heretto](https://docs.medallia.com/en/medallia-documentation/medallia-product-documentation-style-guide/tips-for-working-in-heretto/reusing-content) (via search excerpt; third-party, not Heretto's own docs)

**S1000D applicability (aerospace and defence technical data)**
- Applicability is split across three modules:
  - **ACT** (Applicability Cross-reference Table): the entry point, linked to the CCT and PCT. It declares product attributes, i.e. static characteristics that vary between products.
  - **CCT** (Conditions Cross-reference Table): declares conditions, such as modification status.
  - **PCT** (Products Cross-reference Table): "where the applicability is decided", i.e. the per-product-instance values.
  — [Adobe FrameMaker S1000D help, Applicability and Common Source Data](https://help.adobe.com/en_US/framemaker/s1000d/WS9ad62b8fb4f8a997-577b0a7d12e234e8c0d-8000.html); [PTC Windchill S1000D applicability](https://support.ptc.com/help/windchill/whc/whc_en/Windchill_Help_Center/s1000d_solution_applicability.html) (both via search excerpt)
- Applicability assertions are evaluated to "TRUE" by looking up the PCT (for a product attribute) or the CCT (for a condition), depending on `applicPropertyType`. — [SDL slides, "Creating applicability statements that work"](https://www.slideshare.net/slideshow/creating-applicability-statements-that-work/64933514) (via search excerpt; older talk, possibly dated)
- The open-source s1kd-tools ship an applicability checker, `s1kd-appcheck`, which can override the ACT/CCT/PCT and search for them recursively. — [s1kd-appcheck man page](https://gopherproxy.meulie.net/khzae.net/0/s1000d/s1kd-tools/docs/validating/s1kd-appcheck.txt) (via search excerpt)

### Inferences
- **Two lessons map directly onto the paper:**
  - The DITA "key" pattern (a named indirection bound in the map, with an `@href` fallback) is the generated-macro pattern.
  - The DITAVAL pattern (an external file that picks include, exclude or flag per condition value) is a layer switch.
- **The "flag" action is worth copying.** A review build can render layer-tagged material in colour instead of hiding it, so a collaborator sees every layer at once and can tell which is which. This is a recognised third mode beside "include" and "exclude", not an ad hoc hack.
- **Antora's hard-set/soft-set distinction is the right model for author decisions.** A decision set once at the top should be un-overridable by a section or layer, and only explicitly soft defaults should be overridable. In LaTeX, `\newcommand` (which errors on redefinition) behaves like hard-set and `\providecommand` like soft-set. This mapping is my inference, not something a source states.
- **DITA's spec openly admits that filter-then-resolve and resolve-then-filter give different results.** A layered paper should fix and document one order. The natural choice is: resolve number and decision macros first, which is layer-independent, and then select layers.
- **Reuse whole claims, not sentence fragments.** The DITA translation literature's rule ("reuse complete thoughts; inline reuse only for product names and constants") transfers to LLM-written layers. Numbers and named constants are safe inline. Reusing half-sentences across layers will produce grammatical seams, the agreement problem in another form.
- **Stale reuse is not the main risk in by-reference systems; blast radius is.** The defences found are "where used" lookups (Paligo), dedicated shared folders (Oxygen, Heretto/Medallia) and locked reuse (RWS). The paper's equivalent would be a "where used" report listing every location of a macro or decision before it is changed.

### Gaps
- I did not fetch the DITA pages on `@conaction` push semantics or `@conrefend` range rules; only the overview was read.
- I did not verify the status and date of DITA 2.0. Pages for it exist at dita-lang.org/2.0, but I did not confirm when it became an OASIS standard.
- DITA-OT specifics (how it implements filtering order, and its preprocessing stages) were not researched.
- I found no primary documentation of a formal review or approval workflow for shared components in Heretto or Paligo. Both are marketed as having review workflows, but I did not verify this, so it stays unconfirmed.
- I could not retrieve Rockley's exact definitions of "derivative" or "locked" reuse. "Locked reuse" is attested only in the RWS glossary.
- S1000D: I did not consult the specification text itself (issue 5.0/6.0), only vendor help. Whether s1kd-appcheck enumerates every product combination is unverified.

---

## Q2. Transclusion and block references (Nelson; Roam, Logseq, Obsidian; Notion synced blocks; Microsoft Loop; Coda; Google Docs variables; Word fields, properties, content controls and Quick Parts): which keep a single source with live updates, and how do they behave under collaboration and version control?

### Takeaway
- **Cloud and real-time tools** (Notion synced blocks, Microsoft Loop components, Google Docs variable chips) give true single-source live updates that can be edited from any instance. They live in proprietary databases with weak or contested version history, and they are not git-mergeable.
- **Plain-text vault tools** (Obsidian, Logseq) keep embeds live and the files in git. The cost is that block identifiers must be written into the source text, which creates merge noise and fragile IDs.
- **Word** offers several mechanisms with different update semantics. XML-mapped content controls stay in sync. Custom DocProperty fields need a manual field update. Collaborators can break any of these links by hand-editing.
- **Nelson's original idea** adds a requirement most tools drop: the reuse must be *knowable*, so the reader can see that content is transcluded and trace it to its origin.

### Cited Findings
- **Ted Nelson's definition:** transclusion is "the same content knowably in more than one place". Nelson distinguishes it from "transdelivery" (inclusion from elsewhere) and "transquotation" (a quotation that stays connected to its origin). The term was introduced in *Literary Machines*; sources give the year as 1980 or 1981 (the 1987 edition is also cited). Nelson says the idea was already in his 1965 description of hypertext. — [Wikipedia, Transclusion](https://en.wikipedia.org/wiki/Transclusion) (via search excerpt; secondary source, and the year conflicts between sources)
- A secondary gloss of the idea: "a sort of quotation without copying the content but borrowing it from the original source". — [Maggie Appleton, Transcopyright Dreams](https://maggieappleton.com/transcopyright-dreams) (via search excerpt)
- **Notion synced blocks:**
  - Any instance can be edited, and "that change will also appear in all other places where that synced block exists."
  - The original is marked "ORIGINAL".
  - Viewers without access to the original cannot see a copy's contents, and editing a copy requires edit access to the original.
  - "After you unsync, each block becomes independent."
  - With more than 10 copies, deleting the original or choosing "Unsync all" removes every copy, and Undo will not restore them.
  - The help page says nothing about version history for synced blocks.
  — [Notion Help, Synced blocks](https://www.notion.com/help/synced-blocks) (accessed 2026-10-28)
- **Microsoft Loop components:** Microsoft's developer guidance says that if there are multiple instances, "they remain in sync as updates are made". — [Microsoft Learn, Design Loop components](https://learn.microsoft.com/en-ie/MicrosoftTeams/platform/m365-apps/design-loop-components) (via search excerpt)
  - Version history is **contested**. In a Microsoft Tech Community AMA, staff said it is reachable through the backing OneDrive/SharePoint file. A user in the same thread cited an earlier Microsoft answer that components would have no version history because they are designed to change. — [Microsoft Loop AMA](https://techcommunity.microsoft.com/t5/microsoft-365-events/microsoft-loop-ama/ev-p/3691014) (via search excerpt)
  - University IT guidance says edits made in Teams appear in the original Outlook email and vice versa. — [CU System tech tip](https://www.cu.edu/blog/tech-tips/collaborate-real-time-loop-components-teams-and-new-outlook) (via search excerpt)
- **Google Docs variable chips:** announced in May 2023. They are inserted through Insert > Smart chips > Variables, or by typing "@". Changing the text of one instance changes every instance of that variable in the document. — [Google Workspace Updates, May 2023](https://workspaceupdates.googleblog.com/2023/05/variable-chips-google-docs.html); [Google Docs Help](https://support.google.com/docs/answer/13584275); [Online Tech Tips](https://www.online-tech-tips.com/?p=97197) (via search excerpt)
  - Secondary sources say the feature is limited to Workspace accounts and that the variable panel may disappear for personal accounts. I found no mechanism that propagates a value across separate documents. — [Zapier, Google Docs variables](https://zapier.com/blog/google-docs-variables/) (via search excerpt)
- **Obsidian embeds:** `![[note]]`, `![[note#heading]]` and `![[note#^block-id]]`. "Embedded files display their content inline in a note and stay up to date when the source file changes." Embedding a list requires adding `^my-list-id` as plain text in the source note. — [Obsidian Help, Embed files](https://obsidian.md/help/Linking+notes+and+files/Embed+files)
- **Logseq:** Logseq writes an `id::` property into the source block's file "the first time a block is referenced/embedded or its reference is copied". The database id is only stable if that line exists. — [Logseq forum, "Confused by how logseq treats the id property"](https://discuss.logseq.com/t/confused-by-how-logseq-treats-the-id-property/1558) (via search excerpt)
  - Users syncing with git report large merge conflicts when the same journal day page is created on several machines. Users recommend git because changes are visible and conflicts are merged manually. — [Logseq forum, Syncthing sync conflicts thread](https://discuss.logseq.com/t/syncthing-sync-support-automatically-merge-sync-conflicts/26717) (via search excerpt; which thread each statement came from is uncertain)
- **Word, XML-mapped content controls:** content controls can be mapped to a node in a custom XML part, and setting a mapping fills the control with that node's content. This is the standard way to repeat one value in sync. — [Microsoft Learn, Content controls in Word](https://learn.microsoft.com/office/client-developer/word/content-controls-in-word); [MSDN, XMLMapping.SetMapping](https://msdn2.microsoft.com/en-us/library/bb226172.aspx) (via search excerpt)
  - Re-importing an edited XML file creates a duplicate part instead of updating the existing mapping. The fix is to delete the part, re-import and re-map. — [Microsoft Tech Community thread](https://techcommunity.microsoft.com/discussions/word/how-to-add-a-new-field-to-existing-xml-mapping/4475504) (via search excerpt)
- **Word, DocProperty fields and Quick Parts:**
  - A custom document property inserted as a DocProperty field repeats a value, but after the property changes the fields must be updated. — [TechRepublic, custom document property to repeat content](https://www.techrepublic.com/article/use-custom-property-repeat-word/) (via search excerpt)
  - Built-in Document Property Quick Parts are content controls that update together. — [allthings.how, Document Property Quick Parts](https://allthings.how/how-to-use-document-property-quick-parts-in-microsoft-word) (via search excerpt)
  - Sources disagree on whether custom properties auto-update. They also warn that manual edits by collaborators can break the link.

### Inferences
- **Live update and git-friendliness pull against each other.** Notion, Loop and Google Docs resolve references in a server-side database, which is why updates are live and editable from any instance, and also why there is no diffable source. Obsidian and Logseq keep diffable text but must write anchor IDs into the source, and those IDs then become merge hazards.
- **The paper's approach sits on the safer side of this trade-off.** It uses macro names in source and literal values in the Overleaf flat copy (repo: `writeup/overleaf_sync/README.md`), so anchors are human-readable names chosen by the author, not tool-inserted UUIDs.
- **"Unsync" and "unlink field" are the failure mode to design against.** Every live-reuse tool lets a collaborator break the link by editing a copy: Notion's Unsync, a hand-edited Word field, a literal number typed on Overleaf. The repo's unflatten step reports edits to numbers as conflicts instead of accepting them (repo). That is the correct response, and it matches "locked reuse" in the technical-communication literature.
- **Nelson's "knowably" criterion is a useful bar for the paper.** Each reported number or decision should be traceable to its source from the rendered document. The provenance record (`map.json`) already supports this (repo). A review build could expose it, for example as a hover or a margin tag in the PDF.
- **Permissions on the original (Notion) map onto ownership of shared content.** Generated macro files and decision files should be writable only by the generator or the author, never by Overleaf collaborators.

### Gaps
- **Roam Research** block references `((uid))` and embeds, and Roam's version and collaboration behaviour: I did not research them this session.
- **Coda formulas** (named formulas and tables referenced across a doc, live recomputation): not researched.
- **Microsoft Loop:** I found no current primary Microsoft documentation on version history or on behaviour under concurrent editing, and the evidence is contested.
- **Word Quick Parts/AutoText** (Building Blocks): these insert copies, not live references. I believe this but did not verify it this session.
- **Google Docs "building blocks"** (templates such as meeting notes): not researched beyond variable chips.
- **Xanadu:** I did not consult Nelson's primary text (*Literary Machines*).

---

## Q3. Legal and contract document assembly (HotDocs, Docassemble, ContractExpress, Avvoka): how are answers defined once, how are clauses made conditional, and how is consistency kept across a long document?

### Takeaway
Document-assembly systems separate three things:
1. a **template**: the document with variables and conditional regions;
2. a **questionnaire or interview** that collects each answer once;
3. an **answer record** (answer file or answer document) that can be stored and reused.

Consistency in a long contract comes from every occurrence being a reference to the same variable, and every optional clause being guarded by a condition over those variables. Some engines refuse to assemble when a referenced variable is undefined.

### Cited Findings
- **ContractExpress (Thomson Reuters):**
  - "Each question in the questionnaire acts as a variable." The answer feeds the document text or drives a conditional span. — [ContractExpress Author Online, Variables](https://www.thomsonreuters.com/en-us/help/contract-express/author-online/variables.html) (via search excerpt)
  - "Answer documents" are produced per session, controlled by a launch parameter that points to answer templates. These use system meta-variables documented in "a dictionary template giving details of all the meta-variables". — [ContractExpress help, Answer documents](https://www.thomsonreuters.com/en-us/help/contract-express/template/template-reports/answer-documents) (via search excerpt)
  - ContractExpress also has an "include template" field for composing documents from sub-templates. — [ContractExpress help, Include template field](https://www.thomsonreuters.com/en-us/help/contract-express/template/include-template-field.html) (title only; content not read)
- **HotDocs:** "A conditional region is a set of instructions that tells HotDocs whether to include or exclude certain blocks of text." A condition compares a variable against a value, and conditional regions can be nested. — [HotDocs help, Conditional Region Overview](https://help.hotdocs.com/preview/help/Conditional_Region_Overview.htm) (via search excerpt)
- **Avvoka:** a Condition Builder turns plain-English conditions into automation rules. "SmartAutomation" detects "logic, clauses, and variables" in uploaded documents using generative AI. — [Avvoka, Smart automation](https://avvoka.com/smart-automation); [Avvoka help, How do I automate a document](https://help.avvoka.com/en/articles/2945741-how-do-i-automate-a-document) (both via search excerpt; vendor marketing)
- **Docassemble** (open source):
  - DOCX templates use a Jinja2 variant built on docx-template, with `{{ var }}` and inline or paragraph-level `{% if %}`; some advanced Jinja features may not work. — [Docassemble class notes, Jinja2](https://nonprofittechy.github.io/docassemble-class/docs/jinja2); [Gavel, Templating Word documents in Docassemble](https://www.gavel.io/automation/docassemble-templating-word-documents-in-docassemble) (both via search excerpt)
  - Docassemble must find a definition for every variable in the document, and an undefined one throws an error.
  - Questions are asked lazily: a variable is never asked if it is never referenced on a code path that runs. A `need` block forces collection.
  — [Docassemble interview logic](https://docassemble.org/assets/docassemble-logic.html) (via search excerpt)
- **CoQuill** (a non-Docassemble engine, given for comparison): the same variable can appear any number of times in a template; it is collected once and substituted everywhere. — [CoQuill, Template authoring](https://houfu.github.io/coquill/guides/template-authoring) (via search excerpt)

### Inferences
- **The answer file is the closest legal-tech analogue to an "author decisions" ledger.** It is a persisted, reusable record of choices, separate from the template, that fully determines the assembled document. The paper's decisions (for example, the repo's generated-file header noting "2026-10-06: recalibration removed / intervals removed, user decision", in `writeup/generated/close_main_numbers.tex`) would be easier to audit if they lived in one answer-file-like source. Today they are spread across comments in generator scripts. I have not audited where all decisions live, so this is an inference.
- **Docassemble's "error on undefined variable" is worth copying deliberately.** LaTeX already errors on an undefined control sequence. The risk is a stale or soft fallback (`\providecommand`) that hides a missing value. The fallback should be loud, in the DITA style: a visible "??" marker, not a silent old value.
- **Lazy questioning (Docassemble) is analogous to layer selection.** Only the variables a selected layer actually references need to be resolved. That allows a top-beats layer to compile before every appendix number exists, provided missing values render loudly.
- **Conditional clauses keyed on answers correspond to text keyed on decisions.** For example: if a decision says "intervals removed", every sentence that would report an interval should sit inside a condition on that decision, not be deleted by hand in each layer.

### Gaps
- I found no primary documentation confirming Docassemble's behaviour when one variable appears many times in a DOCX. Its once-only semantics are implied by its lazy-evaluation model, not stated in what I read.
- HotDocs answer files (`.anx`) and component files (`.cmp`) and their reuse across templates were not verified this session.
- I found no independent review or study of how these tools guarantee consistency in long documents, beyond vendor documentation.
- I found no source on review or redlining workflows inside these tools, for example how a lawyer reviews an assembled contract against the template.

---

## Q4. Literate and variable-driven documents (Quarto/knitr inline values, MyST substitutions and embeds, LaTeX datatool, generated-macro conventions)

### Takeaway
The scientific-publishing world has converged on two patterns:
- **Executable inline expressions**, evaluated at render time (Quarto/knitr `` `{r} x` ``, `` `{python} x` ``).
- **Precomputed named values**, defined once and substituted by key: MyST substitutions; LaTeX macro files written by Python or Stata (`stattotex`, `texdoc`, `statrep`); `datatool` databases loaded from files.

MyST additionally supports labelling a notebook output and embedding it anywhere in a project, with a hover link back to the cell. That is the nearest literate-programming realisation of "knowable" transclusion of results.

### Cited Findings
- **Quarto inline code** works across Jupyter, Knitr and OJS. Output is inserted as plain text by default, with Markdown escaped. "Inline expressions should be confined to simple values that you have pre-computed" in normal code cells. Inline expressions are not generally supported in YAML strings such as the title or `fig-cap`. In notebooks, they are evaluated only when Quarto executes the notebook. — [Quarto docs, Inline code](https://quarto.org/docs/computations/inline-code.html)
- **MyST-Parser substitutions** (Sphinx, Jinja2):
  - Defined globally in `conf.py` via `myst_substitutions`, or per file in front-matter. "Keys in the front-matter will override ones in the conf.py."
  - Substitutions can be nested, but "circular references are prohibited".
  - "Substitutions will only be assessed where you would normally use Markdown, e.g. not in code blocks", and they cannot be placed directly in URLs.
  — [MyST-Parser, Optional syntax: substitutions](https://myst-parser.readthedocs.io/en/latest/syntax/optional.html)
- **mystmd** labels notebook cell outputs (`#| label:`) and reuses them anywhere in a project:
  - `[](#label)` cross-references show "a hover-preview and link to the notebook cell directly".
  - `![](#label)` or `{embed}` embed the output.
  - `{figure}` and `{table}` wrap an output with a caption.
  - A `placeholder` image is used in static exports "when the output cannot be directly serialized".
  — [mystmd guide, Reuse Jupyter outputs](https://mystmd.org/guide/reuse-jupyter-outputs)
- **LaTeX `datatool`** (maintainer Nicola Talbot, v3.4.5, dated 2026-08-23, LPPL 1.3): "databases may be created using LaTeX commands or by importing external files". It provides commands "to examine database elements, and to convert formats", plus sorting, mail-merge-style row iteration and charts. — [CTAN, datatool](https://ctan.org/pkg/datatool)
- **Generated-macro tools:**
  - `stattotex` (PyPI) is "a simple function for automatically updating LaTeX documents with numbers or strings from Python", inspired by a Stata package of the same name. — [PyPI, stattotex](https://pypi.org/project/stattotex)
  - Ben Jann's `texdoc` and newer `sttex` connect Stata and LaTeX. — [RePEc, sttex (London Stata Users Group 2022)](https://ideas.repec.org/p/boc/lsug22/14.html) (via search excerpt)
  - A practitioner write-up describes exporting numbers from Stata or Python to a `.tex` file used in Overleaf, which "eliminates much of the copy-paste malebolge". — [A. H. Morris, texdoc to LaTeX](https://arthurhowardmorris.github.io/resources/textdoc_to_latex.html) (via search excerpt)
  - The `statrep` package does the same for SAS. — [MiKTeX, statrep](https://miktex.org/packages/statrep__doc) (via search excerpt)
- An agent-skill listing describes adding `\hypertarget` labels to code-generated values so each number can be traced to its source output. — [skillselion listing, backward-traceability](https://skillselion.com/skills/lingzhi227/agent-research-skills/backward-traceability) (via search excerpt; low-quality aggregator, unverified)
- **(repo)** The paper already follows the generated-macro convention. `writeup/generated/*.tex` files are headed "AUTO-GENERATED by writeup/make_*.py -- do not edit", name their CSV sources, and define values such as `\newcommand{\cmNdays}{866}`. The Overleaf copy is a flattened `main.tex` "with literal numbers" plus a `map.json` "record of where each character of main.tex came from". `unflatten.py` "reports edits to numbers and tables as conflicts". — `writeup/generated/close_main_numbers.tex`; `writeup/overleaf_sync/README.md` (dated 2026-10-20)

### Inferences
- **The repo's convention matches the most common pattern in the economics and stats tooling.** The pattern is: values precomputed in code, emitted as named macros, and inlined by name. It is also Quarto's own recommendation ("confine inline expressions to … pre-computed" values).
- **Inline execution inside the paper would be a step backwards** for Overleaf collaborators, who cannot run Python. The repo's flattened literal copy avoids this. It resembles mystmd's `placeholder` (a static rendering for environments that cannot execute).
- **MyST's front-matter-overrides-global precedence is the opposite of what author decisions need.** Decisions should be global and hard-set (Antora style, Q1), and only presentation defaults should be locally overridable.
- **`datatool` is a credible in-LaTeX alternative** if values should live in one CSV per result family instead of many macro files. It moves lookups into TeX, though, which makes errors harder to read and Overleaf compiles slower. That is my judgment and was not measured. Generated `\newcommand` files remain simpler to diff.

### Gaps
- I did not verify whether Overleaf currently supports knitr (`.Rtex`) compilation or Quarto. I believe Overleaf supports knitr, but this is unconfirmed this session.
- `pythontex` and `\directlua`-based approaches were not researched.
- MyST/mystmd behaviour when notebooks are re-executed (staleness of embedded outputs, caching) is not described on the page I read.
- Quarto's `freeze` and cache semantics, which govern staleness of inline values, were not covered on the inline-code page and were not researched.

---

## Q5. Has anyone built a system that combines conditional layers, define-once variables or decisions, and a review workflow? What should the layered LaTeX/Overleaf paper copy or avoid?

### Takeaway
The combination exists in commercial technical-documentation CCMS products:
- DITA with keys, DITAVAL and a CCMS such as Heretto or Paligo;
- MadCap Flare with targets, conditions, variables and snippets;
- S1000D with applicability and common-source databases;
- legal assembly with answer files and conditional clauses.

All of these target product manuals or contracts, not academic papers, and none that I found integrates with LaTeX, Overleaf or git-based round-tripping. I found no precedent that treats *author decisions* (methodological choices) as first-class variables that propagate to every place a result is reported. The closest analogues are legal answer files and DITA keys.

So the paper's design is a recombination of established parts. The parts worth copying are:
- indirection through named keys;
- external condition files with an include/exclude/flag mode;
- hard-set global values;
- loud fallbacks;
- "where used" impact reports;
- locked reuse for generated content;
- filter/resolve order fixed by rule.

### Cited Findings
- DITA provides by-reference reuse (conref and conkeyref), named-value indirection through maps (keys) and build-time condition files (DITAVAL with include, exclude, passthrough and flag) in one standard. — [DITA 1.3 conref overview](https://docs.oasis-open.org/dita/dita/v1.3/errata02/os/complete/part3-all-inclusive/archSpec/base/conref.html); [DITA 1.3 DITAVAL prop](https://dita-lang.org/1.3/dita/langref/ditaval/ditaval-prop)
- Flare combines condition tags (per target), snippets and variables in one project, to avoid "a separate project for each output". — [Flare 2024 help, Conditions and Single-Sourcing](https://help.madcapsoftware.com/flare2024/Content/Flare/Conditions/Other-Information/Conditions-Single-Sourcing.htm)
- Paligo combines reuse, variables and filtering, plus "where used" lookups and branching. — [Paligo, About reusing content](https://docs.paligo.net/en/about-reusing-content.html); [Paligo, How to find reusable content](https://docs.paligo.net/en/how-to-find-reusable-content.html) (via search excerpt)
- S1000D separates declared product attributes and conditions (ACT/CCT) from per-instance values (PCT), and evaluates applicability assertions against them. — [Adobe FrameMaker S1000D help](https://help.adobe.com/en_US/framemaker/s1000d/WS9ad62b8fb4f8a997-577b0a7d12e234e8c0d-8000.html) (via search excerpt)
- Legal assembly separates the template from the answers and keys conditional regions to those answers. — [HotDocs, Conditional Region Overview](https://help.hotdocs.com/preview/help/Conditional_Region_Overview.htm); [ContractExpress, Answer documents](https://www.thomsonreuters.com/en-us/help/contract-express/template/template-reports/answer-documents) (both via search excerpt)
- Antora's hard-set and soft-set attributes are a precedence mechanism that prevents lower levels from overriding a value set once at the top. — [Antora docs, AsciiDoc attributes in playbook](https://docs.antora.org/antora/latest/playbook/asciidoc-attributes/)
- Notion shows how live single-source reuse can fail at scale: unsyncing or deleting the original of a block with more than 10 copies removes every copy, unrecoverably. — [Notion Help, Synced blocks](https://www.notion.com/help/synced-blocks)

### Inferences (lessons; copy or avoid)

**Copy**
1. **Named indirection for every reported value and decision** (DITA keys, Flare variables, Antora attributes). The repo already does this for numbers. Extend it to *decisions*, so that a decision such as "intervals removed" is a named value that conditions text, not a comment in a generator.
2. **A single decisions file, analogous to an answer file** (ContractExpress, HotDocs), hard-set and not overridable by a section or layer (Antora hard-set). A layer that needs a different value should fail loudly.
3. **A layer-selection file separate from the source** (DITAVAL, Flare targets), with three modes: include, exclude, and flag. The flag mode makes a reviewer build that shows all layers, colour-coded.
4. **Fixed processing order.** Resolve values first, then filter layers. DITA documents that the two orders can disagree.
5. **Loud fallbacks.** DITA falls back from keyref to href, and Docassemble errors on undefined variables. Avoid silent stale defaults.
6. **A "where used" impact report** (Paligo) before any decision or value change, listing every sentence or table in every layer that cites it. Pair it with a dedicated, clearly marked folder for shared and generated content (Oxygen, Heretto/Medallia; the repo's `writeup/generated/` already serves this).
7. **Locked reuse for generated content** (RWS). Collaborator edits to numbers on Overleaf are reported as conflicts, not merged; the repo already does this. Prose edits flow back.
8. **"Knowable" transclusion** (Nelson; mystmd hover-to-source). A review build where each number links or tags back to its generator and CSV row, built from the `map.json` provenance that already exists.

**Avoid**
1. **Phrase-level reuse of prose fragments across layers.** DITA localisation experience shows it breaks grammar and context. Reuse whole claims; inline-reuse only numbers and named constants.
2. **Live cloud transclusion (Notion, Loop, Google Docs chips) as the source of record.** These tools have no git diff, contested version history, and permission coupling to the original.
3. **Tool-inserted opaque IDs in source text** (Logseq `id::` UUIDs, Obsidian `^id`) as anchors. Prefer author-chosen semantic macro names, which diff and merge cleanly.
4. **Overridable local substitutions for decisions** (MyST front-matter overriding the global value).
5. **Unbounded fan-out without impact review.** By-reference systems make propagation automatic, so the residual risk is a wrong edit spreading everywhere, not staleness.

### Gaps
- I found no academic-paper-specific system (LaTeX, Overleaf, Typst or Quarto manuscripts) that combines conditional layers with decision propagation and a review workflow. This is absence of evidence from a limited search (about 30 queries and fetches), not proof that none exists.
- Commercial CCMS review workflows (Heretto Review, Paligo Contributor/Reviewer roles, Oxygen Content Fusion) were not verified against primary documentation this session.
- I found no published empirical study of error rates or staleness in single-sourced documents versus copied ones. Rockley's book is the canonical practitioner reference, but I could not access its chapters to quote reuse-failure data.
