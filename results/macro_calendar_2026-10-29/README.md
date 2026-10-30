# Macro release calendars as scheduled in advance (29 Oct 2026)

For RULE addendum L (results/ablations_2014start_w3000/RULE.txt): the lookahead control of the macro-release calendar
features (src/data/loading.py, MACRO_CALENDARS).

- `scheduled.csv`: every release date of CPI, Employment Situation, PPI, GDP (advance), international trade and advance
  retail sales, 2002-2024, as scheduled in OMB's annual *Schedule of Release Dates for Principal Federal Economic
  Indicators* (issued the year before; OMB Statistical Policy Directive No. 3). 1,656 dates. All at 8:30 ET.
- `mismatch.csv`: the 28 releases that moved after being scheduled, each confirmed by an agency notice.
- `SOURCES.md`: every source with URL, the announcement date of each change, and the recommended as-of construction.
- `LITERATURE.md`: the announcement-effect and volatility-forecasting literature, and what a referee would expect.
- The parsing and comparison scripts and the downloaded OMB PDFs are not kept in the repo (SOURCES.md gives the method and URLs).
