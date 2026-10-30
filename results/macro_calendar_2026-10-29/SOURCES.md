# Scheduled-in-advance release calendars for six US macro releases, 2002-2024

Notes for the JFEc intraday RV paper (S&P 500 futures, sample 2014-2024, training back to ~2002).
Research date: 2026-10-29. Status: complete (full OMB-vs-ALFRED diff 2002-2024 in 6.3).

Series: CPI (BLS), Employment Situation (BLS), PPI (BLS), GDP adv/2nd/3rd (BEA),
International Trade in Goods and Services (Census+BEA joint), Advance Monthly Retail Sales (Census).

---------------------------------------------------------------------------------------------------
## 1. Legal backbone: why an "advance calendar" exists at all

* OMB Statistical Policy Directive No. 3 (1985; revised 2024) governs all Principal Federal Economic
  Indicators (PFEIs); all six series are PFEIs. Rules (quoted from OMB's annual schedule notices):
  agencies give OMB their schedule for the coming calendar year; "The last release in the calendar year
  will contain a schedule for the next calendar year"; "each release of a principal economic indicator
  will include an announcement of the date and time of the next release"; if a date must change after
  the schedule is issued, "the agency must announce and explain the change as soon as it is known".
  - Directive text (BLS copy): https://www.bls.gov/bls/statistical-policy-directive-3.pdf
  - 2024 revision (comment embargo 1h -> 30 min, effective 2024-02-19): https://www.federalregister.gov/d/2024-02972
  - OMB notice text quoted above (2000 schedule press release): https://obamawhitehouse.archives.gov/node/18394
* Consequence: three nested "as-of" calendars exist for every date t:
  (a) the OMB consolidated annual schedule issued the previous autumn,
  (b) agency calendars, which can change during the year with public notice,
  (c) the "next release" line printed in each release (e.g. Census advance retail Dec-2018, released
      2019-02-14: "Next release: TBD" — https://www2.census.gov/marts/adv1812.pdf ;
      Jan-2019 release, 2019-03-11: "Next release: April 1, 2019" — https://www2.census.gov/marts/adv1901.pdf).

---------------------------------------------------------------------------------------------------
## 2. Official sources of the calendar as scheduled in advance

### 2.1 OMB "Schedule of Release Dates for Principal Federal Economic Indicators for <year>" (BEST single source)
One PDF per year, issued Sept-Dec of the prior year; rows = indicators (all six are rows), columns = months,
cells = day of month. It is the only official document that is frozen as of issuance (agency web pages are
overwritten, see 2.5).
* 2017-2026 (statspolicy.gov):
  - https://statspolicy.gov/assets/fcsm/files/docs/OMB_final_pfei_schedule_of_release_dates_2017.pdf
  - https://statspolicy.gov/assets/fcsm/files/docs/OMB_pfei_schedule_of_release_dates_2018.pdf
  - https://statspolicy.gov/assets/fcsm/files/docs/OMB_pfei_schedule_of_releasedates_2019.pdf
  - https://statspolicy.gov/assets/fcsm/files/docs/OMB_pfei_schedule_of_release_dates_2020.pdf ... _2021, _2022, _2023, _2024, _2025
  - https://statspolicy.gov/assets/fcsm/files/docs/OMB_pfei_schedule_release_dates_cy2026.pdf
  (2019 PDF checked: Retail Jan 16 / Feb 15 / Mar 14 / Apr 16; Trade Jan 8 / Feb 5 / Mar 7 / Apr 3 / May 7;
   GDP Jan 30 / Feb 28 / Mar 28 / Apr 26; ES Jan 4; PPI Jan 15; CPI Jan 11 — i.e. pre-shutdown dates.)
* 2002-2007: index page https://georgewbush-whitehouse.archives.gov/omb/inforeg/prior_yrs_schedules.html
  (pei_calendar2003.pdf ... pei_calendar2007.pdf; 2002 at /omb/pubpress/pei_calendar2002.pdf; UNT mirror
  https://webarchive.library.unt.edu/eot2008/20090115003716/http://www.whitehouse.gov/omb/pubpress/pei_calendar2002.pdf)
* 2008: https://georgewbush-whitehouse.archives.gov/omb/pubpress/2007/text/092707_pei2008.html
* 2009: https://georgewbush-whitehouse.archives.gov/omb/pubpress/2008/090508_pei2009.html
* 2011-2013: listed on the archived OMB statistical-programs page (2013 schedule dated 2012-09-28):
  https://webarchive.library.unt.edu/web/20120925105143/http://www.whitehouse.gov/omb/inforeg_statpolicy
* 1980s-1990s: FRASER https://fraser.stlouisfed.org/title/schedule-release-dates-principal-federal-economic-indicators-9880
* Caveat: OMB tables give the DAY only (time only when non-standard); GDP row has one date per month with
  the quarter label.

### 2.2 BLS (CPI, Employment Situation, PPI)
* Year pages (2000-2025): https://www.bls.gov/schedule/<YEAR>/home.htm ; index https://www.bls.gov/bls/archived_sched.htm ;
  1957-2000 PDF https://www.bls.gov/bls/histreleasedates.pdf
* Per-release schedule pages: https://www.bls.gov/schedule/news_release/cpi.htm , .../empsit.htm , .../ppi.htm
* CAUTION: these pages are maintained ex post (they show realized dates). Proof: Wayback captures of
  bls.gov/schedule/news_release/ppi.htm on 2013-10-06 list Sept-2013 PPI on "Oct. 11, 2013" and Oct-2013 PPI on
  "Nov. 14, 2013"; the 2013-11-04 capture lists "Oct. 29, 2013" and "Nov. 21, 2013". Use Wayback captures
  dated before each event (CDX: http://web.archive.org/cdx/search/cdx?url=bls.gov/schedule/news_release/ppi.htm).
* 2013 shutdown page (previously scheduled vs revised, with times): https://www.bls.gov/bls/updated_release_schedule.htm
* Other BLS closure notice (2018-12-05 National Day of Mourning; none of the 3 BLS series moved):
  https://www.bls.gov/bls/news-release-dates-to-change-for-government-closure.htm
* 2025 lapse (outside sample): https://www.bls.gov/bls/2025-lapse-revised-release-dates.htm

### 2.3 BEA (GDP; trade jointly with Census)
* Annual schedule published in the December Survey of Current Business, e.g. "BEA 2019 News Release Schedule"
  (SCB vol. 98 no. 12, 2018-12-19): https://apps.bea.gov/scb/issues/2018/12-december/1218-news-releases-2019.htm ;
  2020: https://apps.bea.gov/scb/issues/2019/12-december/1219-news-releases-2020.htm
  CAUTION: the online 2019 page was edited after the shutdown (it now shows "Initial GDP ... February 28",
  Dec trade "March 6", Jan trade "TBD"), so it is NOT the as-published calendar.
* Old-style year pages, archived: http://www.bea.gov/newsreleases/<YEAR>rd.htm (e.g. UNT capture of 2013:
  https://webarchive.library.unt.edu/web/20130213203027/http://www.bea.gov/newsreleases/2013rd.htm ;
  2012: https://webarchive.library.unt.edu/web/20121006175631/http://www.bea.gov/newsreleases/2012rd.htm ;
  national-accounts sort: https://webarchive.library.unt.edu/web/20130216154801/http://www.bea.gov/newsreleases/news_release_sort_national.htm)
* Current: https://www.bea.gov/news/schedule ; schedule-change posts on the BEA blog (tag "Economic Release
  Schedule Update"): https://www.bea.gov/news/blog?tid=1136

### 2.4 Census (retail sales; trade jointly with BEA)
* Economic indicator calendar: https://www.census.gov/economic-indicators/calendar-listview.html ; past years
  e.g. https://www.census.gov/economic-indicators/calendar-listview-2020.html (page mislabeled "2021"; gives
  8:30 AM for all advance retail and trade rows); quick-view PDF pattern /econcards/assets/censusreleaseglance_<YEAR>.pdf
* Program-level date lists (archived): advance retail https://webarchive.library.unt.edu/web/20120916071730/http://www.census.gov/retail/marts/www/martsdates.pdf ;
  monthly retail https://webarchive.library.unt.edu/web/20130216171526/http://www.census.gov/retail/mrts/www/reldates.pdf
* Economics & Statistics Administration consolidated Commerce calendar 2013 (archived):
  https://webarchive.library.unt.edu/web/20130215105727/http://esa.doc.gov/2013-economic-indicator-release-calendar
* Shutdown tip sheets: 2013 (2013-11-15 update, CB13-TPS.99) https://www.census.gov/newsroom/archives/2013-pr/cb13-tps99.html ;
  2019 (orig. 2019-01-30, revised 2019-02-28) https://www.census.gov/newsroom/press-releases/2019/revised-indicator-calendar.html ;
  trade notice 2019-03-06 https://www.census.gov/foreign-trade/statistics/notices/20190306_revisedreleaseschedule.pdf
* CAUTION: same ex-post overwrite problem as BLS/BEA.

### 2.5 Wayback Machine = the practical "vintage calendar" archive
* For any agency calendar URL, the CDX API lists capture timestamps:
  http://web.archive.org/cdx/search/cdx?url=<url>&from=YYYYMMDD&to=YYYYMMDD ; raw capture:
  http://web.archive.org/web/<timestamp>id_/<url>. UNT's End-of-Term / web archive (webarchive.library.unt.edu)
  has dense 2008-2016 captures of whitehouse.gov/omb and bea.gov/census.gov.
* Capture density is uneven (e.g. bls.gov/bls/updated_release_schedule.htm: captures 2013-10-18 03:20 UTC,
  10-19, 10-20, then 12-03), so it brackets but does not pin announcement times.

---------------------------------------------------------------------------------------------------
## 3. FRED / ALFRED release-date data: ACTUAL dates, not scheduled

* API docs: fred/release/dates https://fred.stlouisfed.org/docs/api/fred/release_dates.html ;
  fred/releases/dates https://fred.stlouisfed.org/docs/api/fred/releases_dates.html ;
  real-time periods https://fred.stlouisfed.org/docs/api/fred/realtime_period.html
  - "release dates are published by data sources and do not necessarily represent when data will be
    available on the FRED or ALFRED websites".
  - include_release_dates_with_no_data (default false) "excludes release dates that do not have data. In
    particular, this excludes future release dates which may be available in the FRED release calendar".
    So scheduled dates exist only for the FUTURE; once a date passes it is kept only if data were released.
  - realtime_start/realtime_end for release dates act as a date window (defaults 1776-07-04 / 9999-12-31
    for fred/release/dates; first day of current year for fred/releases/dates). There is no "calendar as
    known at date X" vintage.
* ALFRED release-date download (no key needed): https://alfred.stlouisfed.org/release/downloaddates?rid=<RID>&ff=txt
  RIDs: CPI 10, Employment Situation 50, PPI 46, GDP 53, Trade 51, Advance Retail 9.
  The page states the list includes "the dates when any series from this release was revised" — so it has
  extra non-headline dates (e.g. CPI 2010-02-17, 2019-02-11; PPI 2010-02-14 (a Sunday), 2010-02-16;
  GDP 2013-11-01; ES 2003-10-10).
  Test (downloaded 2026-10-29): CPI rid=10 contains 2013-10-30, 2013-11-20 (actual) and NOT 2013-10-16 or
  2013-11-15 (scheduled). Trade rid=51 contains 2013-10-24, 2013-11-14, 2019-02-06, 2019-03-06, 2019-03-27,
  2019-04-17, 2019-05-09 (all actual). Conclusion: ALFRED = realized dates (+ revision noise).
* FRED web release calendar (fred.stlouisfed.org/releases/calendar?rid=10&y=2013) shows only 2025+.
* ALFRED vintage dates of a series (e.g. CPIAUCSL) = actual publication dates; same conclusion.

---------------------------------------------------------------------------------------------------
## 4. Academic / commercial datasets of announcement dates

* MMS International survey (used by ABDV 2003 AER, 2007 JIE; Balduzzi-Elton-Green 2001): dates/times of
  realized releases with consensus forecasts. ABDV (NBER w8959, https://www.nber.org/system/files/working_papers/w8959/w8959.pdf):
  "U.S. announcements times are known in advance, whereas only the day for the German announcements are
  known in advance". They use realized dates; no discussion of rescheduling (sample 1992-1998).
* Bloomberg ECO (ECO <GO>, "Economic Calendars"): actual release date+time, survey median; widely used
  (Kurov-Sancetta-Strasser-Wolfe 2019 JFQA; Bollerslev-Li-Xue 2018 RES; Gilbert-Scotti-Strasser-Vega 2017 JME).
  Records realized timestamps; a rescheduled release simply appears on its new date.
* Econoday / Haver / Refinitiv calendars: same (realized).
* Savor & Wilson (2013 JFQA, "...Evidence from Scheduled Economic Announcements"; WP
  https://faculty.wharton.upenn.edu/wp-content/uploads/2012/04/Draft20111128p_edited.pdf): "We obtain dates of
  pre-scheduled monthly macroeconomic news announcements from the Bureau of Labor Statistics from 1958 to
  2009" ... "pre-scheduled announcements, whose dates are known months in advance"; they "exclude any
  unscheduled [FOMC] announcements". BLS dates are realized dates, so the delayed 1995-96 shutdown releases
  enter as if pre-scheduled — not discussed.
* Lucca & Moench (2015 JF; SR512 https://www.newyorkfed.org/medialibrary/media/research/staff_reports/sr512.pdf):
  "Only the occurrence of scheduled meetings is known to investors in advance, and because we study returns
  ahead of monetary policy news we exclusively consider these meetings" — schedule from the Board's FOMC
  calendar; unscheduled meetings (conference calls) dropped. The cleanest "scheduled vs actual" treatment in
  the literature, but for FOMC only.
* Martens, van Dijk & de Pooter (2009 IJF 25:282-303, "Forecasting S&P 500 volatility: long memory, level
  shifts, leverage effects, day-of-the-week seasonality, and macroeconomic announcements"; text in de Pooter's
  dissertation https://repub.eur.nl/pub/10533/Dissertation_Michiel_de_Pooter.pdf ch. 3): announcement and
  "calm-before-the-storm" (day-before) dummies for EMP, PPI, CPI (following Ederington-Lee 1993) and FOMC; they
  set the FOMC dummy to 1 on the five UNSCHEDULED 1994-2003 moves (1994-04-18, 1998-10-15, 2001-01-03,
  2001-04-18, 2001-09-17) — i.e. a realized-date dummy used in out-of-sample forecasts, a look-ahead that the
  paper does not discuss. Pre-announcement dummies require knowing tomorrow's schedule: fine with an
  as-of calendar.
* Net: the macro-finance / RV-forecasting literature uses realized dates and justifies them by
  pre-scheduling; none I found builds an as-of-t calendar for BLS/BEA/Census releases or discusses the
  2013/2019 shutdowns. FOMC papers (Lucca-Moench; Savor-Wilson) are the only ones that separate scheduled
  from unscheduled events.

---------------------------------------------------------------------------------------------------
## 5. Release times
* All six: 8:30 a.m. Eastern (EST/EDT) throughout 2002-2024, including all rescheduled releases:
  - BLS 2013 revised table lists 8:30 a.m. for ES, CPI, PPI (https://www.bls.gov/bls/updated_release_schedule.htm).
  - BEA: "8:30 AM Eastern ... unless otherwise noted" (SCB schedule page); 2019 GDP initial "Feb. 28 at 8:30 a.m."
  - Census 2019 tip sheet: advance retail and trade rows all 8:30 a.m.; Census calendar 2020: 8:30 AM.
  - AP list of 2013 revised dates: trade, retail, PPI, CPI, GDP, ES all "8:30 a.m." (https://www.ksl.com/article/27358699).
  - Release headers: "FOR RELEASE AT 8:30 AM EST, THURSDAY, FEBRUARY 14, 2019" (adv1812.pdf).
* Only near-exception found: after EO 13343 (2004-06-07) BLS announced the May-2004 PPI for Thu 2004-06-10 at
  3:00 p.m.; it was postponed again on 06-09 and actually released Thu 2004-06-17 at 8:30 a.m. (never executed).
  Outside the sample: PPI 1999-01-12 released 5:00 p.m. (BLS PPI release-date table, footnote 1).
* Not 8:30 (for contrast): JOLTS, Census construction/new-home-sales/wholesale/business inventories 10:00;
  Industrial production 9:15; BLS regional 10:00.
* 2020-03-16: BEA/Census suspended press lock-ups; releases still posted at the scheduled time
  (https://www.census.gov/newsroom/press-releases/2020/suspension-prerelease-lockups-econ-indicators.html).
  DOL/BLS media lock-up closed March 2020 and discontinued permanently effective 2020-06-03; data posted on the web at 8:30 ET
  (https://www.bls.gov/bls/discontinuing-department-of-labor-media-lockup-effective-june-3.pdf). Release time unchanged.
* Census Advance Economic Indicators report (advance goods trade, 8:30 ET) exists since 2014; not one of the six but an extra 8:30 event,
  sometimes on days without any of the six.

---------------------------------------------------------------------------------------------------
## 6. Reschedulings 2002-2024 for the six series

### 6.1 October 2013 shutdown (lapse 2013-10-01 to late 2013-10-16)
Suspension known: DOL said 2013-09-30 the Oct 4 jobs report would not be released if the government closed
(https://www.bloomberg.com/news/articles/2013-09-30/jobs-report-won-t-be-released-oct-4-if-u-s-government-closed);
DOL statement 2013-10-03 "An alternative release date has not been scheduled"
(https://www.enewspf.com/?p=98007). Commerce (Census/BEA) sites went dark on 2013-10-01.

| Series | Ref. period | Originally scheduled | Actual (8:30 ET) | New date announced |
|---|---|---|---|---|
| Trade | Aug 2013 | Tue 2013-10-08 | Thu 2013-10-24 | Mon 2013-10-21 (Census revised calendar + BEA advisory) |
| PPI | Sep 2013 | Fri 2013-10-11 | Tue 2013-10-29 | Thu 2013-10-17 (BLS updated schedule) |
| Retail | Sep 2013 | Fri 2013-10-11 (Census martsdates.pdf, last revised 2012-10-12) | Tue 2013-10-29 | Mon 2013-10-21 (Census) |
| CPI | Sep 2013 | Wed 2013-10-16 | Wed 2013-10-30 | Thu 2013-10-17 (BLS) |
| ES | Sep 2013 | Fri 2013-10-04 | Tue 2013-10-22 | Thu 2013-10-17 (BLS; Calculated Risk post 16:38 ET) |
| GDP adv | Q3 2013 | Wed 2013-10-30 | Thu 2013-11-07 | Mon 2013-10-21 (BEA) |
| ES | Oct 2013 | Fri 2013-11-01 | Fri 2013-11-08 | Thu 2013-10-17 (BLS) |
| Trade | Sep 2013 | Tue 2013-11-05 | Thu 2013-11-14 | 2013-10-21 (Census/BEA; AP list 2013-10-23) |
| PPI | Oct 2013 | Thu 2013-11-14 | Thu 2013-11-21 | by Wed 2013-10-23 (DOL release 13-2065-NAT; TBD on BLS page 10-17/10-19) |
| Retail | Oct 2013 | Thu 2013-11-14 | Wed 2013-11-20 | Mon 2013-10-21 (Census; Bloomberg 10-21) |
| CPI | Oct 2013 | Fri 2013-11-15 | Wed 2013-11-20 | by Wed 2013-10-23 (AP list; TBD on BLS page 10-17/10-19) |
| GDP 2nd | Q3 2013 | Tue 2013-11-26 | Thu 2013-12-05 | Thu 2013-10-24 (BEA media advisory) |
| GDP 3rd | Q3 2013 | Fri 2013-12-20 | Fri 2013-12-20 | unchanged |
| Trade | Oct 2013 | Wed 2013-12-04 | Wed 2013-12-04 | unchanged |

Sources: BLS https://www.bls.gov/bls/updated_release_schedule.htm (Wayback 2013-10-18 03:20 UTC capture already
shows ES/CPI/PPI Sept and ES Oct revised; CPI/PPI Oct still blank); Calculated Risk 2013-10-17
https://www.calculatedriskblog.com/2013/10/bls-september-employment-report-to-be.html and 2013-10-18
https://calculatedriskblog.com/2013/10/data-schedule-update-cbo-will-release.html ; Census/BEA 2013-10-21:
https://www.calculatedriskblog.com/2013/10/tuesday-jobs-jobs-jobs.html ,
https://www.govexec.com/management/2013/10/census-bureau-reschedules-key-reports-following-shutdown/72329 ,
https://www.bloomberg.com/news/articles/2013-10-21/september-u-s-retail-sales-data-rescheduled-for-oct-29 ;
AP table of all revised dates with original dates and times, 2013-10-23 12:58:
https://www.ksl.com/article/27358699 ; BEA advisory 2013-10-24 (Wayback of
http://www.bea.gov/newsreleases/general/pdf/media_advisory_on_new_release_dates.pdf): "second estimate of GDP for the
third quarter of 2013 will now be released on Thursday, Dec. 5. It was originally scheduled ... Tuesday, Nov. 26 ...
New public release dates have previously been announced ... advance estimate of third quarter GDP, which will now
come out on Thursday, Nov. 7"; BEA schedule pages: Wayback 2013-10-17 (2013rd.htm, "dates below are still under
review") and 2013-10-30 (news_release_sort_national.htm: GDP adv Nov 7, 2nd Dec 5, 3rd Dec 20; last modified
2013-10-24). BEA homepage Wayback 2013-10-22 16:17 UTC links "New Release Dates for GDP, Personal Income and
Outlays and Monthly Trade Reports".
DOL release 13-2065-NAT (2013-10-23): https://www.dol.gov/newsroom/releases/opa/opa20132065 (403 to scripts).

### 6.2 Dec 2018 - Jan 2019 shutdown (lapse 2018-12-22 to 2019-01-25; Commerce closed, BLS funded)
BLS series (CPI, ES, PPI) unaffected. GDP/trade/retail:

| Series | Ref. period | Originally scheduled (OMB 2019) | Actual (8:30 ET) | New date announced |
|---|---|---|---|---|
| Trade | Nov 2018 | Tue 2019-01-08 | Wed 2019-02-06 | Wed 2019-01-30 (BEA blog + Census tip sheet) |
| Retail | Dec 2018 | Wed 2019-01-16 | Thu 2019-02-14 | Wed 2019-02-06 (Census banner: 01-31 and 02-06 14:48 UTC captures say only "Rescheduled November 2018 release dates have been posted"; 02-08 capture says "November and December 2018"; Reuters 2019-02-06 15:28) |
| GDP | Q4 2018 adv | Wed 2019-01-30 | Thu 2019-02-28 ("Initial", replaces adv+2nd) | Wed 2019-02-06 (BEA blog); delay confirmed 2019-01-29 with "no new release dates" |
| GDP | Q4 2018 2nd | Thu 2019-02-28 | merged into 02-28 Initial | 2019-02-06 |
| Trade | Dec 2018 | Tue 2019-02-05 | Wed 2019-03-06 | Wed 2019-02-06 (BEA blog); "TBD" on 2019-01-30 |
| Retail | Jan 2019 | Fri 2019-02-15 | Mon 2019-03-11 | between 2019-02-15 and 2019-02-28 (tip-sheet revision date); "Next release: TBD" in the 02-14 release; Reuters 02-15: "no date has been set" |
| Trade | Jan 2019 | Thu 2019-03-07 | Wed 2019-03-27 | by 2019-02-28 (Census tip sheet) |
| GDP 3rd | Q4 2018 | Thu 2019-03-28 | Thu 2019-03-28 (kept "third estimate" title) | unchanged (BEA blog 2019-03-22) |
| Retail | Feb 2019 | Thu 2019-03-14 | Mon 2019-04-01 | by 2019-02-28 |
| Trade | Feb 2019 | Wed 2019-04-03 | Wed 2019-04-17 | by 2019-02-28 (BEA blog 2019-04-02 restates) |
| Retail | Mar 2019 | Tue 2019-04-16 | Thu 2019-04-18 | by 2019-02-28 |
| GDP adv | Q1 2019 | Fri 2019-04-26 | Fri 2019-04-26 | unchanged |
| Trade | Mar 2019 | Tue 2019-05-07 | Thu 2019-05-09 | by 2019-02-28 |

Sources: OMB 2019 PFEI PDF (above); Census tip sheet https://www.census.gov/newsroom/press-releases/2019/revised-indicator-calendar.html
("revised after the original release date January 30 to include economic indicator data through the March
statistical month"; table of original vs revised dates and times); BEA blog 2019-01-30
https://www.bea.gov/news/blog/2019-01-30/new-dates-set-some-delayed-releases (Nov trade "Feb. 6 at 8:30 a.m.";
GDP adv "originally scheduled for release Wednesday, Jan. 30", Dec trade "originally ... Tuesday, Feb. 5" still TBD);
BEA blog 2019-02-06 https://www.bea.gov/news/blog/2019-02-06/new-dates-set-gdp-personal-income-and-international-trade ;
BEA blog 2019-03-22 https://www.bea.gov/news/blog/2019-03-22/gdp-back-schedule-third-estimate ;
BEA blog 2019-04-02 https://www.bea.gov/news/blog/2019-04-02/national-statistics-return-normal-timetable-april ;
Reuters 2019-01-29 (via Euronews) https://www.euronews.com/2019/01/29/us-delays-fourth-quarter-gdp-report-after-government-shutdown ;
Reuters 2019-02-06 (via Investing.com) https://investing.com/news/economic-indicators/us-december-retail-sales-report-to-be-released-february-14-1771462 ;
FXStreet running log https://www.fxstreet.com/amp/analysis/re-scheduling-the-us-economic-calendar-after-the-government-shutdown-end-201901291407 ;
Census retail releases adv1812/adv1901/adv1902 (www2.census.gov/marts/advYYMM.pdf) "Next release:" lines.

### 6.3 Full diff 2002-2024 and the remaining reschedulings (2004 PPI, 2004 day of mourning, 2010 snow)
Method: parsed all 23 OMB PFEI calendars 2002-2024 (1,656 scheduled dates = 6 series x 12 months x 23 years; no
blank or ambiguous cells) and checked each against ALFRED actual dates. 28 scheduled dates are missing from ALFRED;
an agency release confirms every one as a genuine reschedule (none is an ALFRED artifact or parse error). BLS series
re-checked against BLS news-release archive filenames (cpi_/empsit_/ppi_MMDDYYYY): the same 10 BLS changes, nothing
else. Scratch files (not in repo): ...\scratchpad\pfei_diff\out\scheduled.csv (all 1,656 OMB dates),
out\mismatch.csv, scripts\, downloads\omb\ (the PDFs).
The 28 = 12 changed rows of 6.1 + 10 changed rows of 6.2 + these 6:

| Series | Ref. period | Scheduled (OMB) | Actual (8:30 ET) | Cause / when announced |
|---|---|---|---|---|
| PPI | Jan 2004 | Thu 2004-02-19 | Thu 2004-03-18 | SIC->NAICS conversion problems. Postponement announced Tue 2004-02-17; new date announced about 1 day ahead (CNN/Money 2004-03-17: release "the next morning") |
| PPI | Feb 2004 | Fri 2004-03-12 | Thu 2004-04-01 | Postponement announced Fri 2004-03-05; the 03-18 release still says the date "will be announced as soon as possible"; new date set between 03-18 and 03-31 |
| PPI | Mar 2004 | Thu 2004-04-08 | Thu 2004-04-22 | 04-01 notice: "has not yet scheduled a revised release date"; new date set between 04-01 and 04-21 |
| PPI | May 2004 | Fri 2004-06-11 | Thu 2004-06-17 | 06-11 was the Reagan National Day of Mourning (EO 13343, issued 2004-06-07). BLS first moved it to Thu 06-10 at 3:00 p.m.; on Wed 06-09 it postponed "until further notice ... no earlier than Tuesday, June 15". New date announced at least 1 day ahead; not yet posted in the 06-16 05:29 ET capture, so most likely 06-15 or 06-16 |
| Trade | Apr 2004 | Fri 2004-06-11 | Mon 2004-06-14 | Same closure (BEA trad0404: "not to be released before 8:30 a.m. Monday, June 14, 2004"); announced after the 06-07 EO, exact day not pinned |
| Retail | Jan 2010 | Thu 2010-02-11 | Fri 2010-02-12 | Snowstorm closed federal offices Feb 8-11. Delay announced Wed 2010-02-10 (AP). Trade Dec-2009 went out as scheduled Wed 2010-02-10 |

Sources:
* BLS PPI notices via Wayback, http://www.bls.gov/ppi/delaynotice.htm (captures 2004-03-12, 04-01, 06-03). Quotes:
  "As announced on February 17, the release of the PPI for January 2004 has been delayed from the originally
  scheduled date of February 19, 2004 ... February data originally scheduled for Friday, March 12, must also be
  postponed ... we plan to announce them at least one day ahead of time on this web page and through news
  advisories". The 04-01 capture: "has not yet scheduled a revised release date for the March 2004 PPI, originally
  scheduled for April 8".
* http://www.bls.gov/ppi/may_ppi_notice.htm (captures 2004-06-10 to 06-19).
* BLS "Release dates for the Producer Price Index, 1971-2025", with delay footnotes:
  https://www.bls.gov/ppi/news-release/historical-release-dates-for-the-producer-price-index.pdf
* PPI release texts: https://www.bls.gov/news.release/history/ppi_03182004.txt , ppi_04012004.txt ,
  ppi_06172004.txt ("EMBARGOED UNTIL 8:30 A.M. (EDT), THURSDAY, JUNE 17, 2004").
* Mises Wire, 2004-03-14, quoting the BLS notices: https://mises.org/mises-wire/delayed-producer-prices
* Fox/Reuters, 2004-06-09 (the 3 p.m. plan, then indefinite delay):
  https://www.foxnews.com/story/ppi-release-delayed-due-to-another-calculation-problem
* CNN/Money, 2004-03-17: https://money.cnn.com/2004/03/17/news/economy/ppi_release/index.htm
* EO 13343: https://www.fedsmith.com/2004/06/07/executive-order-close-government-offices-june-11/
* BEA trade releases: https://apps.bea.gov/newsreleases/international/trade/2004/trad0404.htm ,
  .../2010/trad1209.htm , .../2013/trad0813.htm ("originally scheduled for Tuesday, November 5, 2013, will be
  released on Thursday, November 14").
* Census retail releases: https://www2.census.gov/retail/releases/historical/marts/adv0912.pdf (says Feb 11) and
  adv1001.pdf ("FRIDAY, FEBRUARY 12, 2010"); adv1308.pdf ("scheduled to be released October 11, 2013").
* AP, 2010-02-10: https://www.investmentnews.com/ria-news/white-out-snowstorm-forces-rare-delays-of-economic-reports/26960

Checked with no change for the six:
* Hurricane Isabel (2003-09-18/19) and Hurricane Sandy (2012-10-29/30): none of the six was scheduled on those days.
  ES 2012-11-02 was kept (DOL decision 2012-10-31:
  https://www.upi.com/Top_News/US/2012/10/31/BLS-says-it-will-issue-jobs-report-on-time/63801351711970/).
* 2018-12-05 Bush day of mourning, the Jan-2016 blizzard, and all of 2020 (COVID only ended the press lock-ups).
* BLS series during the 2018-19 lapse.
* ES on the Thursday before July 4 (2003, 2008, 2009, 2014, 2015, 2020) is already in the advance calendars.

### 6.4 OMB PFEI calendar URLs actually used
All 23 years were found. They are frozen documents, created Aug-Sep of the prior year.
* 2002: https://webarchive.library.unt.edu/eot2008/20090115003716id_/http://www.whitehouse.gov/omb/pubpress/pei_calendar2002.pdf
* 2003: https://web.archive.org/web/20030319063055id_/http://www.whitehouse.gov:80/omb/inforeg/pei_calendar2003.pdf
* 2004: https://web.archive.org/web/20031203184129id_/http://www.whitehouse.gov:80/omb/inforeg/pei_calendar2004.pdf
* 2005-2009: https://web.archive.org/web/{20040926075253,20050922081715,20060925071502,20071003053502,20080916032009}id_/http://www.whitehouse.gov/omb/inforeg/pei_calendar{2005..2009}.pdf
* 2010: https://webarchive.library.unt.edu/web/20120925105143id_/http://www.whitehouse.gov/sites/default/files/omb/assets/information_and_regulatory_affairs/pei_calendar2010.pdf
* 2011-2013: the same UNT prefix + /sites/default/files/omb/inforeg/statpolicy/{pei_calendar_2011.pdf, pfei-schedule-of-release-dates-2012.pdf, pfei-schedule-of-release-dates-2013.pdf}
* 2014: https://web.archive.org/web/20131019102936id_/http://www.whitehouse.gov/sites/default/files/omb/inforeg/statpolicy/pfei-schedule-of-release-dates-2014.pdf
* 2015-2016: https://obamawhitehouse.archives.gov/sites/default/files/omb/inforeg/statpolicy/pfei-schedule-of-release-dates-{2015,2016}.pdf
* 2017-2024: the statspolicy.gov URLs in 2.1.
  - 2024 original: https://web.archive.org/web/20231201175518id_/https://www.whitehouse.gov/wp-content/uploads/2023/09/pfei_schedule_release_dates_2024.pdf
  - 2018: the statspolicy copy was re-saved in 2025 and the original was not found; its rows are assumed unchanged.
* No year has a time note for any of the six. Time notes appear only for Personal Income (10 a.m. on GDP days), EIA
  gas storage, and USDA (noon).

---------------------------------------------------------------------------------------------------
## 7. Recommended look-ahead-free construction

### 7.1 Facts behind the design
* Exactly 28 of the 1,656 advance-scheduled dates moved in 2002-2024, across these episodes:
  - PPI 2004 (NAICS conversion, plus the Reagan closure), 4 dates.
  - Trade, 2004-06 (Reagan closure), 1 date.
  - Retail, 2010-02 (snow), 1 date.
  - October 2013 funding lapse, 12 dates.
  - 2018-19 funding lapse, 10 dates.
* Only the 2019 episode (GDP, trade, retail) falls in the 2014-2024 evaluation sample. All the others are in the
  training windows.
* The postponement and the new date were announced SEPARATELY, with a "date TBD" state in between:
  - 2013: postponed Oct 1, new dates Oct 17-24.
  - 2019: postponed Dec 22, new dates Jan 30, Feb 6 and by Feb 28.
  - PPI 2004: Feb 17 -> about Mar 17; Mar 5 -> late Mar; Apr 1 -> mid Apr; Jun 9 -> about Jun 15-16.
* In every case the old date was withdrawn publicly BEFORE it arrived, either because a funding lapse was in force
  or by an explicit notice.
* In every case the new date was public at least one day before it arrived. The shortest leads were PPI
  2004-03-18 (announced about 03-17), PPI 2004-06-17 (at least 1 day, per BLS's stated policy) and retail 2010-02-12
  (announced 02-10). All other leads were 3 days or more.
* Consequence: for a forecast whose information set is the previous close or later, the REALIZED date equals the
  date known at forecast time, for all six series in 2002-2024.
* The original OMB calendar used naively is wrong in the other direction. It would flag 28 phantom release days
  that everyone knew would not happen (e.g. 2013-10-04, 2019-01-30) and miss the 28 real ones.

### 7.2 Construction: an as-of calendar C(t)
1. Base: the OMB PFEI calendar for each year (out\scheduled.csv), all six at 8:30 ET. The year-y calendar is public
   from Aug-Sep of year y-1, so it is available for every forecast in the sample.
2. Event log, one row per change: series, reference period, old date, state change, new date or TBD, and
   announcement date (plus time if known). The rows are in 6.1-6.3. Each affected release passes through three
   states:
   - (a) Before the postponement notice: the old date.
   - (b) After the postponement notice, before the new date is announced: no date. The release is pending and its
     dummy is 0 on every day. The start of a funding lapse counts as a postponement notice for every release of the
     affected agencies, since releases stopped "until further notice": 2013-10-01 for BLS, Census and BEA;
     2018-12-22 for Census and BEA only.
   - (c) After the new-date announcement: the new date.
   This refines the rule "the old date until the postponement is announced, then the new date". State (b) is
   needed because no new date existed for 1-6 weeks.
3. Dummy: for a forecast made at time t, D_k(d | t) = 1 if C_k(t) puts release k on day d.
   - For intraday or 1-day-ahead models with t at or after the close of d-1, this reproduces realized dates exactly.
   - Implement it anyway, so that multi-step horizons use C(t) rather than realized dates: h-day HAR targets,
     "announcement within the next h days" regressors, and pre-announcement (day-before) dummies.
   - The two calendars differ only inside these windows: 2004-02-17..04-22, 2004-06-07..06-17, 2010-02-10..02-12,
     2013-10-01..12-05, 2018-12-22..2019-05-09.
4. Edge cases to state in the paper:
   - (i) GDP Q4-2018 advance, scheduled 2019-01-30. The lapse ended 01-25, so state (b) was arguably over; BEA
     confirmed the delay on 01-29 with no new date. Treat it as TBD from 2018-12-22.
   - (ii) GDP 2019-02-28 and 2019-03-28 kept their dates but changed content. Feb 28 became the combined "initial"
     estimate, and the Mar 28 release kept the "third estimate" title.
   - (iii) PPI 2004-06 had an intermediate schedule (Thu 06-10 at 3:00 p.m.) that lasted about 2 days and was never
     executed. Ignore it, or treat it as state (b).
5. Robustness checks:
   - Re-estimate with the 2013-10-01..12-06 and 2018-12-22..2019-05-10 windows dropped, or with a "post-lapse
     catch-up" dummy.
   - Re-estimate with realized-date dummies. The differences should be confined to those days.
6. Do NOT use FRED/ALFRED release dates. They are realized dates plus revision-only dates such as the CPI/PPI
   February seasonal-factor updates, retail annual revisions and GDP annual revisions (e.g. GDP 2013-11-01 and PPI
   2004-04-05 are not releases). Do NOT use current agency calendar pages as "scheduled": they are overwritten ex post.

### 7.3 Suggested wording for the paper
"Announcement dummies use the release calendar as it was publicly known at the forecast origin: OMB's annual
Schedule of Release Dates for Principal Federal Economic Indicators, updated with each agency's public postponement
and rescheduling notices on the day they were issued. Over 2002-2024, 28 of 1,656 scheduled releases of the six
indicators were moved (the 2004 PPI delays, closures for a day of mourning in 2004 and for snow in 2010, and the 2013
and 2018-19 funding lapses). Every new date was public at least one day in advance, so for one-step-ahead forecasts
the as-of calendar coincides with realized release dates."
