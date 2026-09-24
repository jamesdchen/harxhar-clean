"""The long-only 15:30 close signal for a hand-traded account.

Every SPXW session the job rebuilds the research panel's row for the day from
feeds, runs the SAME per-bar forecast arm the research measured (the
live-feasible ridge at the 16:00 bar, recalibrated by the causal MZ map), turns
the sign rule ``buy iff rv_hat > iv_var`` into a break-even straddle price the
operator compares against the quote on the screen, sizes the buy by premium,
and posts the instruction as a Google Calendar event.  The short side of
sign(s) is not traded: the book is the Heaviside part 1{s > 0} plus the
month-end long.

Modules: ``feeds`` (Yahoo adapters, delays measured), ``features`` (the
panel's 30-minute moments from 1-minute bars), ``ingest`` (the one-time history
files), ``state`` (committed parquet store), ``forecast`` (the spec arm on the
extended panel + MZ), ``signal`` (break-even price, sizing, the card),
``calendar_push`` (Google Calendar), ``schedule`` (the Actions guard),
``run`` (the entry point).
"""
