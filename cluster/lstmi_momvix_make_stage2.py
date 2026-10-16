"""Plan stage 2 of the moments_vix intraday LSTM with the campaign's own planner.

cluster/lstm_intraday_make_stage2.py (as shipped to the intraday campaign's deployment root,
copied unedited into /scratch1/jc_905/harxhar-momvix-lstm) sizes the refit chunks from stage 1's
measured fit times for the buckets in its module constant BUCKETS; this wrapper sets that
constant to the one new input set (2026-10-06, user request) and runs it unchanged.

Usage (CARC, deployment root): python cluster/lstmi_momvix_make_stage2.py --root DIR --nt 20 --out FILE
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lstm_intraday_make_stage2 as plan  # noqa: E402

plan.BUCKETS = ("moments_vix",)  # type: ignore[assignment]  # one input set
# 2026-10-06 (main session: "can we speed this up?", CARC mostly idle): smaller chunks, so the one
# bucket's refits spread over more tasks at once.  Chunking does not change a number (the
# campaign's chunk gates: chunked = unchunked bit for bit, incl. mid-period starts and the cache).
plan.TARGET_SEC = 12 * 60

if __name__ == "__main__":
    plan.main()
