"""The daily entry point.

    python -m live.close_signal.run --wait-until 15:30:30 --input-mode free_substitute

Order of work: guard (session, 16:00 close, firing window) -> the idle time
before the decision stamp: the PREP event at 15:00 (the day, the budget, the
row near the index) and the PRECOMPUTE at 15:15 (the warm arm server and a
canary pass on the 15:00 panel) -> sleep to the decision stamp -> fetch the
feeds at once and measure their delays -> build today's panel rows (the input
mode decides how the 15:30 stamp is filled) -> append state -> forecast (the
13 spec arms in one shared pass on the extended panel + the causal MZ map) ->
break-even price, sizing, card -> Calendar event -> journal.  A ``FeedError``
or a forecast failure posts a "NO SIGNAL" event and exits non-zero, so a
silent miss is impossible; the prep and the precompute can only fail quietly
(printed, never blocking the card).

Why nothing of the forecast itself is computed before 15:30: the 15:30 row
moves full-sample statistics of the spec's transform (the diurnal std floor
of the signed moments, the median fills), so every arm's history -- and the
MZ map fitted on it -- changes with it (``fastpath_check.py`` measures it).
What can be done early is everything around the arithmetic: the interpreter,
the imports, the worker processes, the numba kernels, the code copy, the
vendor files, and a canary that exercises the whole path.

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
import shutil
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import pandas as pd

from live.close_signal import calendar_push, feeds, forecast, schedule
from live.close_signal.common import CBOE_COLS, INPUT_MODES, stamp
from live.close_signal.features import panel_rows, substitute_last_bar
from live.close_signal.signal import (
    DEFAULT_LONG_FRACTION,
    DEFAULT_MONTH_END_FRACTION,
    build_instruction,
    headline,
    render_card,
    render_prep,
)
from live.close_signal.state import StateStore

REPO = Path(__file__).resolve().parents[2]
STATE_DIR = REPO / "live" / "close_signal" / "state"
FOMC_CSV = STATE_DIR / "fomc_statement_dates.csv"
DECISION = "15:30"
FORECAST_ROW = "16:00"
#: When the card lands on the runner (the prep event tells the operator).
CARD_AT = "15:31"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--session", default=None, help="YYYY-MM-DD (default: today in ET)")
    ap.add_argument(
        "--wait-until", default=None, help="ET clock to sleep to, e.g. 15:30:30"
    )
    ap.add_argument(
        "--max-wait-min",
        type=float,
        default=None,
        help="refuse (exit 2, no card) a --wait-until sleep longer than this",
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
        "--prep-at",
        default=schedule.PREP_AT,
        help="ET clock of the prep event (a waiting run on today's session only)",
    )
    ap.add_argument(
        "--precompute-at",
        default=schedule.PRECOMPUTE_AT,
        help="ET clock to start the arm server and the canary pass",
    )
    ap.add_argument(
        "--full-arms",
        action="store_true",
        help="forecast through the path of record (13 spec processes, ~5 min) "
        "instead of the shared pass; the same forecast (fastpath_check.py)",
    )
    ap.add_argument(
        "--self-test",
        default=None,
        help="YYYY-MM-DD: run the placeholder-invariance gate and exit",
    )
    ap.add_argument(
        "--gate-deck",
        action="store_true",
        help="run the 13 research arms on the vendor panel alone and require the "
        "assembled table and the loader's rv_hat to equal the research table and "
        "the deck (the fidelity gate); exit",
    )
    ap.add_argument(
        "--workers",
        type=int,
        default=None,
        help="processes: spec arms at a time on the path of record "
        f"(default {forecast.WORKERS}), backtest workers of the shared pass "
        f"(default {forecast.FAST_WORKERS})",
    )
    ap.add_argument(
        "--reuse-arms",
        action="store_true",
        dest="reuse_arms",
        help="with --gate-deck: reuse the arm CSVs already in --scratch (the "
        "assembly, loader or tolerances changed, not the arms)",
    )
    return ap.parse_args(argv)


def _push(
    a: argparse.Namespace,
    session,
    decision: str,
    body: str,
    late: bool,
    headline: str = "",
) -> str:
    if a.no_push or a.dry_run:
        print(body, flush=True)
        return ""
    service = calendar_push.make_service()
    ev = calendar_push.build_event(
        session, calendar_push.summary_for(decision, late, headline), body
    )
    return calendar_push.push_event(service, a.calendar_id, ev, session)


def _et_now() -> datetime:
    return datetime.now(schedule.ET_TZ)


def _log(t0: float, what: str) -> None:
    print(
        f"[{_et_now():%H:%M:%S}] +{time.perf_counter() - t0:5.1f}s {what}", flush=True
    )


def _sleep_until(hhmmss: str, cap: str | None = None) -> None:
    """Sleep to ``hhmmss`` ET today, never past ``cap`` (the decision stamp)."""
    s = schedule.seconds_until(hhmmss)
    if cap is not None:
        s = min(s, schedule.seconds_until(cap))
    if s > 0:
        print(f"sleeping {s:.0f}s to {hhmmss} ET", flush=True)
        time.sleep(s)


def prep_spot(session: pd.Timestamp) -> tuple[float, str]:
    """The index now: ^GSPC's last complete minute, else ES's (a small basis)."""
    errors = []
    for name, fetch in (
        ("^GSPC", feeds.spx_minute_bars),
        ("ES=F", feeds.es_minute_bars),
    ):
        try:
            b = fetch(session)
            return feeds.last_complete_close(b), name
        except Exception as e:  # noqa: BLE001 -- try the next feed
            errors.append(f"{name}: {type(e).__name__}: {e}")
    raise feeds.FeedError("no spot for the prep: " + "; ".join(errors))


def post_prep(a: argparse.Namespace, session: pd.Timestamp) -> None:
    """The 15:00 prep event.  Never raises: a failure is printed and the day goes on.

    No journal row: a journal row means "this session has a card" to the
    dedupe guard (``schedule.already_journaled``).
    """
    try:
        spot, src = prep_spot(session)
        title, body = render_prep(
            session=session.date(),
            spot=spot,
            flags=schedule.calendar_flags(session.date()),
            capital=a.capital,
            long_fraction=a.long_fraction,
            month_end_fraction=a.month_end_fraction,
            card_at=CARD_AT,
        )
        if a.no_push or a.dry_run:
            print(f"prep (not posted; spot from {src}):\n{title}\n{body}", flush=True)
            return
        service = calendar_push.make_service()
        ev = calendar_push.build_event(
            session.date(), title, body, start_hhmm=a.prep_at[:5], kind="prep"
        )
        eid = calendar_push.push_event(
            service, a.calendar_id, ev, session.date(), kind="prep"
        )
        print(f"posted prep event {eid} (spot {spot:,.2f} from {src})", flush=True)
    except Exception:  # noqa: BLE001 -- the prep must never block or delay the card
        print("prep event FAILED (the card is unaffected):", flush=True)
        traceback.print_exc()


def precompute(
    a: argparse.Namespace, store: StateStore, session: pd.Timestamp
) -> forecast.FastContext | None:
    """The warm arm server + a canary pass on the panel as it stands (~15:15).

    Works on a COPY of the state (the real state is appended only at the
    stamp).  Bounded by the decision stamp: a canary still running 20 s before
    it is killed and the card path starts a cold server.  Never raises.
    """
    t0 = time.perf_counter()
    ctx: forecast.FastContext | None = None
    try:
        got = feeds.fetch_all(session)
        es = feeds.take(got, "es")
        cboe = {k: feeds.take(got, k) for k in CBOE_COLS}
        rows = panel_rows(es.frame, {k: v.frame for k, v in cboe.items()}, session)
        # only stamps whose ES bar is complete (a partial bar is not a row)
        done = [s for s in pd.to_datetime(rows["endbartime"]) if es.stamp_complete(s)]
        if not done:
            raise feeds.FeedError("no complete ES bar yet today")
        rows = rows[pd.to_datetime(rows["endbartime"]) <= max(done)]
        pre_dir = Path(a.scratch) / "pre_state"
        if pre_dir.exists():
            shutil.rmtree(pre_dir)
        shutil.copytree(store.root, pre_dir)
        pre = StateStore(pre_dir)
        pre.append_panel(rows, source="yahoo_es")
        _log(t0, f"precompute: panel through {max(done):%H:%M}")
        ctx = forecast.FastContext(
            REPO,
            Path(a.scratch),
            workers=a.workers or forecast.FAST_WORKERS,
            start_server=False,
        )
        ctx.start_server(
            ready_timeout=max(10.0, schedule.seconds_until(a.wait_until) - 20.0)
        )
        _log(t0, "precompute: arm server ready")
        left = schedule.seconds_until(a.wait_until) - 20.0
        out = forecast.canary(ctx, pre, session, FOMC_CSV, timeout=max(10.0, left))
        _log(
            t0,
            "precompute: canary pass done (matrix {seconds_matrix:.1f}s, backtests "
            "{seconds_backtests:.1f}s)".format(**out),
        )
    except Exception:  # noqa: BLE001 -- the card path starts cold instead
        print("precompute FAILED (the card path starts cold):", flush=True)
        traceback.print_exc()
        if ctx is not None and not ctx.healthy():
            ctx.close(kill=True)
            ctx = None
    return ctx


def before_the_stamp(
    a: argparse.Namespace, store: StateStore, session: pd.Timestamp
) -> forecast.FastContext | None:
    """The idle time of a waiting run on today's session: prep, then precompute."""
    kind, _ = schedule.prep_plan(session.date(), _et_now(), a.prep_at)
    if kind == "sleep":
        _sleep_until(a.prep_at, cap=a.wait_until)
    if kind in ("sleep", "now"):
        post_prep(a, session)
    else:
        print("prep: skipped (after the prep window)", flush=True)
    if a.full_arms:
        return None
    kind, _ = schedule.precompute_plan(
        session.date(), _et_now(), a.wait_until, a.precompute_at
    )
    if kind == "skip":
        print("precompute: skipped (too close to the stamp)", flush=True)
        return None
    if kind == "sleep":
        _sleep_until(a.precompute_at, cap=a.wait_until)
    return precompute(a, store, session)


def main(argv: list[str] | None = None) -> int:
    a = parse_args(argv)
    # absolute: the spec processes run with the ext root as working directory
    a.scratch = str(Path(a.scratch).resolve())
    store = StateStore(Path(a.state_dir))
    if a.gate_deck:
        out = forecast.assert_reproduces_deck(
            REPO,
            Path(a.scratch),
            workers=a.workers or forecast.WORKERS,
            reuse_arms=a.reuse_arms,
        )
        print("deck reproduction OK:", out)
        return 0
    if a.self_test:
        day = pd.Timestamp(a.self_test)
        out = forecast.assert_placeholder_invariance(
            REPO,
            store,
            day,
            Path(a.scratch),
            FOMC_CSV,
            workers=a.workers or forecast.WORKERS,
        )
        print("placeholder invariance OK:", out)
        return 0
    now = _et_now()
    session = pd.Timestamp(a.session) if a.session else pd.Timestamp(now.date())
    if not a.skip_guard:
        ok, why = schedule.should_run(_et_now())
        if ok and schedule.already_journaled(store.load_journal(), session.date()):
            ok, why = False, f"session {session.date()} already has a journal row"
        print(("run: " if ok else "skip: ") + why, flush=True)
        if not ok:
            return 0
    ctx: forecast.FastContext | None = None
    if a.wait_until:
        plan, s = schedule.wait_plan(
            session.date(), _et_now(), a.wait_until, a.max_wait_min
        )
        if plan == "refuse":
            print(
                f"not waiting {s / 3600:.1f} h to {a.wait_until} ET (--max-wait-min "
                f"{a.max_wait_min:g}): dispatch inside the day, or pass --session "
                "YYYY-MM-DD to replay a past session",
                flush=True,
            )
            return 2
        if plan == "sleep":
            try:
                ctx = before_the_stamp(a, store, session)
            except Exception:  # noqa: BLE001 -- nothing before the stamp may cost the card
                print(
                    "before the stamp FAILED (the card path starts cold):", flush=True
                )
                traceback.print_exc()
                ctx = None
            _sleep_until(a.wait_until)
        else:
            print(f"replaying {session.date()}: no wait", flush=True)
    try:
        return _card(a, store, session, ctx)
    finally:
        if ctx is not None:
            ctx.close()


def _card(
    a: argparse.Namespace,
    store: StateStore,
    session: pd.Timestamp,
    ctx: forecast.FastContext | None,
) -> int:
    """From the decision stamp to the posted card (or the NO SIGNAL card)."""
    t0 = time.perf_counter()
    flags = schedule.calendar_flags(session.date())
    t1530 = stamp(session, DECISION)
    run_at = feeds.now_et()
    notes: list[str] = []
    late = False
    try:
        got = feeds.fetch_all(session)  # one thread per symbol
        es = feeds.take(got, "es")
        cboe = {k: feeds.take(got, k) for k in CBOE_COLS}
        spx_now = got["spx"]
        _log(t0, "feeds fetched")
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
                spx = feeds.take(got, "spx")
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
            copy_dir = Path(a.scratch) / "state_copy"
            if copy_dir.exists():
                shutil.rmtree(copy_dir)
            shutil.copytree(store.root, copy_dir)
            fc_store = StateStore(copy_dir)
            fc_store.append_panel(rows, source="yahoo_es")
        _log(t0, "panel rows appended")
        if a.full_arms:
            fc = forecast.forecast_session(
                REPO,
                fc_store,
                session,
                Path(a.scratch),
                FOMC_CSV,
                workers=a.workers or forecast.WORKERS,
            )
        else:
            fc = forecast.forecast_session_fast(
                REPO,
                fc_store,
                session,
                Path(a.scratch),
                FOMC_CSV,
                ctx=ctx if ctx is not None and ctx.healthy() else None,
                workers=a.workers or forecast.FAST_WORKERS,
            )
        _log(t0, f"forecast rv_hat {fc['rv_hat']:.6g}")
        # the prefetched ^GSPC (its 15:29 bar is complete at 15:30:30); a failed
        # prefetch is fetched again here, as before
        spot_bars = bars.get("spx") or (
            spx_now
            if isinstance(spx_now, feeds.Bars1m)
            else feeds.spx_minute_bars(session)
        )
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
        eid = _push(a, session.date(), instr.decision, body, late, headline(instr))
        posted_at = feeds.now_et()
        _log(t0, f"posted calendar event {eid or '(dry run)'}")
        print(body, flush=True)
        if not a.dry_run:
            # the card is posted: a journal failure from here on must not
            # replace it with NO SIGNAL (it did on 2026-09-24)
            try:
                store.append_journal(
                    {
                        "run_at": run_at,
                        "session": session,
                        **instr.as_record(),
                        **{f"fc_{k}": v for k, v in fc.items()},
                        "event_id": eid,
                        "status": "ok",
                        "posted_at": posted_at,
                        "forecast_path": "record" if a.full_arms else "shared",
                    }
                )
            except Exception:  # noqa: BLE001
                print("card posted, journal write FAILED:", flush=True)
                traceback.print_exc()
                return 3
        return 0
    except Exception as e:  # noqa: BLE001 -- every failure must become a NO SIGNAL card
        # (the 2026-09-23 dispatch died on a pytz NonExistentTimeError outside
        # the earlier tuple: rc 1, no card, the operator none the wiser)
        msg = (
            f"NO SIGNAL, {session.date():%a %d %b %Y}: the job failed, so there is no "
            "forecast.\nDo not trade the close on this model today.\n\n"
            f"For the log: {type(e).__name__}: {e}"
        )
        print(msg, flush=True)
        traceback.print_exc()
        try:
            eid = _push(a, session.date(), "NO_SIGNAL", msg, late)
            print(f"posted calendar event {eid or '(dry run)'}", flush=True)
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
