"""The full procedure from analysis/PROTOCOL.md, gated in order.

No step starts before the previous one passes. Steps that fail stop the run
rather than letting a later step quote a number resting on an unvalidated
foundation -- which is the specific failure this pipeline exists to prevent.

  1  build the composed-fix cache        (assumed done, verified here)
  2  prep_invariants across all caches   must be clean on A, B, E
  3  factorial_design                    the pre-registered design
  4  audit_paper_numbers                 every quoted number traces to a JSON

Run with nohup or as a background task; it prints a gate line per step.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATS = os.path.join(ROOT, "writeup", "stats")
CACHES = ["b2_mmap", "b2_mmap_indonly", "b2_mmap_fix"]


def run(step, argv, must_pass=True):
    print(f"\n{'#' * 78}\n# STEP {step}: {' '.join(argv)}\n{'#' * 78}",
          flush=True)
    t0 = time.time()
    r = subprocess.run([sys.executable] + argv, cwd=ROOT,
                       env={**os.environ, "OMP_NUM_THREADS": "4"},
                       capture_output=True, text=True)
    sys.stdout.write(r.stdout[-8000:])
    if r.stderr.strip():
        sys.stderr.write(r.stderr[-3000:])
    ok = r.returncode == 0
    print(f"\n[gate {step}] {'PASS' if ok else 'FAIL'} "
          f"(rc={r.returncode}, {time.time()-t0:.0f}s)", flush=True)
    if must_pass and not ok:
        print(f"[gate {step}] stopping: later steps would quote numbers "
              f"resting on this", flush=True)
        sys.exit(1)
    return ok


def main():
    t0 = time.time()
    # ---- step 1: the cache must exist
    missing = [c for c in CACHES
               if not os.path.exists(os.path.join(ROOT, "results", c, "X.npy"))]
    print(f"caches present: {[c for c in CACHES if c not in missing]}")
    if missing:
        print(f"  missing (skipped downstream): {missing}")
    if not os.path.exists(os.path.join(ROOT, "results", "b2_mmap_indonly",
                                       "X.npy")):
        print("[gate 1] FAIL: the composed-fix cache is not built")
        sys.exit(1)
    print("[gate 1] PASS")

    # ---- step 2: invariants
    run(2, ["analysis/prep_invariants.py"] +
        [os.path.join(ROOT, "results", c) for c in CACHES if c not in missing])
    inv = json.load(open(os.path.join(STATS, "prep_invariants.json")))
    key = "b2_mmap_indonly"
    if key in inv:
        cls = inv[key]["by_class"]
        bad = {k: v for k, v in cls.items() if k in ("A", "B", "E") and v}
        print(f"[gate 2] composed-fix cache violations by class: {cls}")
        if bad:
            print(f"[gate 2] FAIL: classes A/B/E must be clean, got {bad}")
            sys.exit(1)
        print("[gate 2] PASS: A, B, E clean")

    # ---- step 3: the pre-registered factorial
    run(3, ["analysis/factorial_design.py"])

    # ---- step 4: the number audit
    run(4, ["analysis/audit_paper_numbers.py"], must_pass=False)

    print(f"\n{'=' * 78}")
    print(f"PIPELINE COMPLETE ({time.time()-t0:.0f}s)")
    print("=" * 78)
    fd = os.path.join(STATS, "factorial_design.json")
    if os.path.exists(fd):
        d = json.load(open(fd))
        print(f"  best design per cache: {d.get('best_per_cache')}")
        print(f"  design x cache interaction: {d.get('interaction')}")
        if "headline_dm" in d:
            h = d["headline_dm"]
            print(f"  headline: {h['a']} vs {h['b']} on {h['cache']}: "
                  f"{h['diff']:+.5f}  t {h['t']:+.2f}  p {h['p']:.3g}")
    print("\n  H1-H4 and the falsification condition are in "
          "analysis/PROTOCOL.md section 3; read the cells against them "
          "rather than for whatever looks best.")


if __name__ == "__main__":
    main()
