"""The daily entry point.

    python -m live.close_signal.run --wait-until 15:30:30 --input-mode free_substitute

Order of work: guard (session, 16:00 close, firing window) -> sleep to the
decision stamp -> fetch the feeds and measure their delays -> build today's
panel rows (the input mode decides how the 15:30 stamp is filled) -> append
state -> forecast (the spec arm on the extended panel + the causal MZ map) ->
break-even price, sizing, card -> Calendar event -> journal.  A ``FeedError``
or a forecast failure posts a "NO SIGNAL" event and exits non-zero, so a
silent miss is impossible.

Input modes (README):
    ibkr             real-time ES bars + Cboe prints through live/ibkr's broker --
                     the research construction; needs a running gateway (not wired
                     to a socket here: ``fetch_ibkr`` is the seam to fill)
    free_substitute  ^GSPC 1-minute (real-time on Yahoo) stands in for the delayed
                     ES bar on the 15:00-15:30 return moments; the Cboe prints are
                     the latest available (about 15:15).  A MODEL CHANGE -- to be
                     validated on the purchased history before it is trusted.
    free_delayed     wait until the 15:30 ES bar and Cboe prints have arrived
                     (about 15:45 on Yahoo) and mark the card LATE.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import pandas as pd

from live.close_signal import calendar_push, feeds, forecast, schedule
from live.close_signal.common import INPUT_MODES, stamp
from live.close_signal.features import panel_rows, substitute_last_bar
from live.close_signal.signal import (
    DEFAULT_LONG_FRACTION,
    DEFAULT_MONTH_END_FRACTION,
    build_instruction,
    render_card,
)
from live.close_signal.state import StateStore

REPO = Path(__file__).resolve().parents[2]
STATE_DIR = REPO / "live" / "close_signal" / "state"
FOMC_CSV = STATE_DIR / "fomc_statement_dates.csv"
DECISION = "15:30"
FORECAST_ROW = "16:00"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--session", default=None, help="YYYY-MM-DD (default: today in ET)")
    ap.add_argument(
        "--wait-until", default=None, help="ET clock to sleep to, e.g. 15:30:30"
    )
    ap.add_argument("--input-mode", choices=INPUT_MODES, default="free_substitute")
    ap.add_argument(
        "--delayed-deadline",
        default="15:50:00",
        help="free_delayed: give up after this ET clock",
    )
    ap.add_argument(
        "--capital",
        type=float,
        default=float(os.environ.get("CLOSE_SIGNAL_CAPITAL", "0") or 0),
    )
    ap.add_argument("--long-fraction", type=float, default=DEFAULT_LONG_FRACTION)
    ap.add_argument(
        "--month-end-fraction", type=float, default=DEFAULT_MONTH_END_FRACTION
    )
    ap.add_argument(
        "--calendar-id", default=os.environ.get("GCAL_CALENDAR_ID", "primary")
    )
    ap.add_argument(
        "--no-push", action="store_true", help="compute and journal, post nothing"
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="print the card, touch no state, post nothing",
    )
    ap.add_argument(
        "--skip-guard",
        action="store_true",
        help="run outside the firing window (tests, replays)",
    )
    ap.add_argument("--state-dir", default=str(STATE_DIR))
    ap.add_argument(
        "--scratch", default=str(REPO / "live" / "close_signal" / ".scratch")
    )
    ap.add_argument(
        "--self-test",
        default=None,
        help="YYYY-MM-DD: run the placeholder-invariance gate and exit",
    )
    return ap.parse_args(argv)


def _push(a: argparse.Namespace, session, decision: str, body: str, late: bool) -> str:
    if a.no_push or a.dry_run:
        print(body, flush=True)
        return ""
    service = calendar_push.make_service()
    ev = calendar_push.build_event(
        session, calendar_push.summary_for(decision, late), body
    )
    return calendar_push.push_event(service, a.calendar_id, ev, session)


def main(argv: list[str] | None = None) -> int:
    a = parse_args(argv)
    store = StateStore(Path(a.state_dir))
    if a.self_test:
        day = pd.Timestamp(a.self_test)
        out = forecast.assert_placeholder_invariance(
            REPO, store, day, Path(a.scratch), FOMC_CSV
        )
        print("placeholder invariance OK:", out)
        return 0
    now = datetime.now(schedule.ET_TZ)
    session = pd.Timestamp(a.session) if a.session else pd.Timestamp(now.date())
    if not a.skip_guard:
        ok, why = schedule.should_run(datetime.now(schedule.ET_TZ))
        print(("run: " if ok else "skip: ") + why, flush=True)
        if not ok:
            return 0
    if a.wait_until:
        s = schedule.seconds_until(a.wait_until)
        print(f"sleeping {s:.0f}s to {a.wait_until} ET", flush=True)
        time.sleep(s)
    flags = schedule.calendar_flags(session.date())
    t1530 = stamp(session, DECISION)
    run_at = feeds.now_et()
    notes: list[str] = []
    late = False
    try:
        es = feeds.es_minute_bars(session)
        cboe = feeds.cboe_prints(session)
        bars = {"es": es, **cboe}
        if a.input_mode == "free_delayed":
            deadline = stamp(session, a.delayed_deadline[:5]) + pd.Timedelta(
                seconds=int(a.delayed_deadline[6:8] or 0)
            )
            es = feeds.wait_for_stamp(
                lambda: feeds.es_minute_bars(session), t1530, deadline=deadline
            )
            cboe = {
                k: feeds.wait_for_stamp(
                    lambda k=k: feeds.cboe_prints(session)[k], t1530, deadline=deadline
                )
                for k in cboe
            }
            bars = {"es": es, **cboe}
            late = True
            notes.append("free_delayed: inputs complete after the stamp")
        elif a.input_mode == "ibkr":
            raise feeds.FeedError(
                "input_mode ibkr: the broker seam is not wired in this package yet"
            )
        rows = panel_rows(es.frame, {k: v.frame for k, v in cboe.items()}, session)
        if a.input_mode == "free_substitute":
            if not es.stamp_complete(t1530):
                spx = feeds.spx_minute_bars(session)
                bars["spx"] = spx
                if not spx.stamp_complete(t1530):
                    raise feeds.FeedError(
                        f"^GSPC bar ending {t1530} not complete either (last {spx.last_bar_start})"
                    )
                rows = substitute_last_bar(rows, spx.frame, t1530)
                notes.append(
                    f"15:30 ES bar substituted by ^GSPC (ES delay {es.delay_minutes:.1f} min) -- MODEL CHANGE"
                )
            for k, b in cboe.items():
                if not b.stamp_complete(t1530):
                    notes.append(
                        f"{k} print at 15:30 is the latest available ({b.last_bar_start:%H:%M})"
                    )
        else:
            for k, b in bars.items():
                if not b.stamp_complete(t1530):
                    raise feeds.FeedError(
                        f"{k}: bar ending {t1530} not delivered (last {b.last_bar_start})"
                    )
        if not a.dry_run:
            store.append_latency(feeds.latency_records(bars), run_at)
            store.append_panel(rows, source="yahoo_es")
        # A dry run works on a COPY of the state (the forecast appends a placeholder row).
        fc_store = store
        if a.dry_run:
            import shutil

            copy_dir = Path(a.scratch) / "state_copy"
            if copy_dir.exists():
                shutil.rmtree(copy_dir)
            shutil.copytree(store.root, copy_dir)
            fc_store = StateStore(copy_dir)
            fc_store.append_panel(rows, source="yahoo_es")
        fc = forecast.forecast_session(
            REPO, fc_store, session, Path(a.scratch), FOMC_CSV
        )
        spot_bars = bars.get("spx") or feeds.spx_minute_bars(session)
        spot_series = spot_bars.frame["close"]
        spot = float(
            spot_series[spot_series.index <= t1530 - pd.Timedelta(minutes=1)].iloc[-1]
        )
        instr = build_instruction(
            session=session.date(),
            spot=spot,
            rv_hat=fc["rv_hat"],
            flags=flags,
            capital=a.capital,
            long_fraction=a.long_fraction,
            month_end_fraction=a.month_end_fraction,
            input_mode=a.input_mode,
            late=late,
            notes=tuple(notes),
        )
        body = render_card(instr)
        eid = _push(a, session.date(), instr.decision, body, late)
        if not a.dry_run:
            store.append_journal(
                {
                    "run_at": run_at,
                    "session": session,
                    **instr.as_record(),
                    **{f"fc_{k}": v for k, v in fc.items()},
                    "event_id": eid,
                    "status": "ok",
                }
            )
        print(body, flush=True)
        return 0
    except (feeds.FeedError, RuntimeError, ValueError, AssertionError) as e:
        msg = f"NO SIGNAL for {session.date()}: {type(e).__name__}: {e}"
        print(msg, flush=True)
        traceback.print_exc()
        try:
            _push(a, session.date(), "NO_SIGNAL", msg, late)
        finally:
            if not a.dry_run:
                store.append_journal(
                    {
                        "run_at": run_at,
                        "session": session,
                        "status": "no_signal",
                        "error": msg,
                    }
                )
        return 2


if __name__ == "__main__":
    sys.exit(main())
