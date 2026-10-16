"""Sentinel-B impact audit (read-only): the main results with and without the
bars whose persisted diurnal baseline is the "no history" sentinel B = 1.

Context (2026-10-06).  The main results are scored by
``experiments/score_unification.py --convention plain`` on the plain
back-transform f = yhat^2 * B (outputs results/unification_plain_2026-10-06/).
The target transform writes ``baseline = 1.0`` where no diurnal baseline exists
(src/features/transforms/target.py, ``baseline.replace(0, 1.0).fillna(1.0)``);
on those bars f = yhat^2 is ~6 orders of magnitude above a 30-minute variance
and the per-bar QLIKE is ~11-17.  The paper2 checkout (harxhar-clean, branch
paper2) carries an UNCOMMITTED exclusion in its scorer::

    def _valid_of(c):
        return c["valid"] & (c["baseline"] != 1.0)

used by ``_burnin_state`` and ``_score_chunk_causal``.  This script reproduces
the plain scoring twice from the SAME stored outputs the plain scorer reads --
once as-is ("with" the sentinel bars) and once with exactly that exclusion
("without") -- by importing score_unification READ-ONLY and monkeypatching the
two functions in memory.  Nothing is fitted; score_unification.py, the paper
and writeup/generated are not touched; the shared raw-harvest cache
(results/.harvest_cache) is READ, never written (cache misses re-read the npz).

Every output lands in results/sentinel_b_impact_2026-10-06/:

  sentinel_bars.csv           the sentinel bars (row_id, date, clock, rv, f, QLIKE,
                              prior same-slot count -> cause), per root
  <set>_<pass>/               full scorer outputs per pass (scores CSV, increments,
                              tree selection, tex/campaign_numbers.tex + tables,
                              score.log) plus arm_detail.csv / pair_detail.csv
                              (n, finite-pair DM sample size T, sentinel rows per
                              arm) and sentinel_rows.csv (row_ids per (root, arm))
  impact_main.csv             curated side-by-side: the main results' quantities
  macro_diff.csv              every \\unif macro, with vs without
  claims.csv                  sentence lines in the paper that use a changed macro
  parity.txt                  "with" pass vs the published plain run (must match)

CLI:
    python experiments/sentinel_b_impact.py all          # identify + 4 passes + compare
    python experiments/sentinel_b_impact.py run --set main --pass without
    python experiments/sentinel_b_impact.py compare
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import os
import pickle
import re
import subprocess
import sys
from collections import Counter, defaultdict

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SU_PATH = os.path.join(_ROOT, "experiments", "score_unification.py")
OUT = os.path.join(_ROOT, "results", "sentinel_b_impact_2026-10-06")
PUBLISHED = os.path.join(_ROOT, "results", "unification_plain_2026-10-06")
PUBLISHED_TEX = os.path.join(_ROOT, "writeup", "generated", "campaign_numbers.tex")
SECTIONS = os.path.join(_ROOT, "writeup", "sections")
# the stored per-chunk outputs live in the paper2 checkout (sibling of this repo)
CLEAN_RESULTS = os.path.join(os.path.dirname(_ROOT), "harxhar-clean", "results")
SENTINEL = 1.0
MIN_PERIODS = 5  # src/features/transforms/target.py DIURNAL_MIN_PERIODS

# The arm set of the published plain run (results/unification_plain_2026-10-06/
# score.log: 61 arms in unification_carc, 62 in unification_h2, + 2 tree
# synthetics per root).  One regex serves both roots, so the union reproduces it.
MAIN_ARMS = (
    "a0_ols_har a_bucket_moments a_bucket_liquidity a_bucket_market_ew "
    "a_bucket_market_vw a_bucket_sentiment a_bucket_implied_vol a_bucket_vol_demand "
    "a_bucket_all_features b1_ridge b2_lasso b1_ridge_tuned b2_lasso_tuned "
    "b3_enet_tuned br_tuned_moments br_tuned_liquidity br_tuned_market_ew "
    "br_tuned_market_vw br_tuned_sentiment br_tuned_implied_vol br_tuned_vol_demand "
    "br_tuned_all_features be_tuned_moments be_tuned_liquidity be_tuned_market_ew "
    "be_tuned_market_vw be_tuned_sentiment be_tuned_implied_vol be_tuned_vol_demand "
    "be_tuned_all_features b1_ridge_a0p1 b1_ridge_a0p3 b1_ridge_a3 b1_ridge_a10 "
    "b1_ridge_a30 b1_ridge_a100 b1_ridge_a300 blk2_user blk2_doc blk2_tuned "
    "blk4_prodBbExogTrailSd b1_ridge_tuned_fine blk2_gated_tuned "
    + " ".join(f"tree_expert_{k:02d}" for k in range(20))
).split()

SETS = {
    "main": {
        "roots": [
            os.path.join(CLEAN_RESULTS, "unification_carc"),
            os.path.join(CLEAN_RESULTS, "unification_h2"),
        ],
        "arms": "|".join(MAIN_ARMS),
        "published_scores": os.path.join(PUBLISHED, "unification_scores.csv"),
        "published_incr": os.path.join(PUBLISHED, "unification_increments.csv"),
    },
    "fomc": {
        "roots": [os.path.join(CLEAN_RESULTS, "unification_fomc1")],
        "arms": None,
        "published_scores": os.path.join(PUBLISHED, "fomc1", "unification_scores.csv"),
        "published_incr": os.path.join(
            PUBLISHED, "fomc1", "unification_increments.csv"
        ),
    },
}
PASSES = ("with", "without")


# ── read-only import of the plain scorer ─────────────────────────────────────


def _load_su():
    spec = importlib.util.spec_from_file_location("score_unification", _SU_PATH)
    su = importlib.util.module_from_spec(spec)
    sys.modules["score_unification"] = su
    spec.loader.exec_module(su)
    return su


def _patch_read_only_cache(su) -> None:
    """Cache HIT -> the pickled raw chunks; MISS -> re-read the npz files.
    Never writes results/.harvest_cache (the scorer's own _raw_chunks would)."""

    def _ro_raw_chunks(found, arm, root_label, cache):
        if cache.enabled:
            fp = su._chunk_fingerprint(found)
            try:
                with open(cache.path_for(arm, root_label), "rb") as fh:
                    blob = pickle.load(fh)
                if (
                    blob.get("version") == su._HARVEST_CACHE_VERSION
                    and blob.get("fingerprint") == fp
                    and blob.get("arm") == arm
                    and blob.get("root") == root_label
                ):
                    cache.hits += 1
                    return blob["chunks"]
            except (OSError, EOFError, ValueError, AttributeError, KeyError,
                    pickle.UnpicklingError):  # fmt: skip
                pass
        cache.rebuilt += 1
        return su._read_chunks(found)

    su._raw_chunks = _ro_raw_chunks


# per (root, arm): sentinel row_ids inside SCORED windows (burn-in excluded)
_SENT_ROWS: dict[tuple[str, str], list[np.ndarray]] = defaultdict(list)
_CAPT: list = []


def _patch_capture(su, exclude: bool) -> None:
    """Record the sentinel rows of every scored window; when ``exclude``,
    apply the paper2 rule valid &= (baseline != 1.0) before scoring (and in
    the burn-in state, as paper2 does)."""
    orig_score, orig_burn = su._score_chunk_causal, su._burnin_state
    orig_harvest, orig_tree = su._harvest_arm, su._assemble_tree_arms

    def _masked(c: dict) -> dict:
        c2 = dict(c)
        c2["valid"] = np.asarray(c["valid"], dtype=bool) & (
            np.asarray(c["baseline"], dtype=np.float64) != SENTINEL
        )
        return c2

    def score(c, prev, res):
        s = np.asarray(c["baseline"], dtype=np.float64) == SENTINEL
        if s.any():
            _SENT_ROWS[(res.root, res.arm)].append(np.asarray(c["row_id"])[s])
        return orig_score(_masked(c) if exclude else c, prev, res)

    def burn(c):
        return orig_burn(_masked(c) if exclude else c)

    def harvest(*a, **k):
        r = orig_harvest(*a, **k)
        _CAPT.append(r)
        return r

    def trees(by_root, sel_rows):
        out = orig_tree(by_root, sel_rows)
        _CAPT.extend(out)
        return out

    su._score_chunk_causal, su._burnin_state = score, burn
    su._harvest_arm, su._assemble_tree_arms = harvest, trees


# ── one scoring pass (own process: a full pass holds GBs of per-bar arrays) ──


def _paired(su, a, b) -> dict:
    """Paired comparison on the row_id intersection: n_common, the DM sample
    size T (finite loss differentials), mean differential, DM t."""
    _, ia, ib = np.intersect1d(a.row_id, b.row_id, assume_unique=True,
                               return_indices=True)  # fmt: skip
    if not len(ia):
        return {"n_common": 0, "T": 0, "dq": None, "dm": None}
    d = su.dm_test(a.loss[ia], b.loss[ib], h=1)
    dq, t = d.get("mean_diff", np.nan), d.get("dm", np.nan)
    return {
        "n_common": int(len(ia)),
        "T": int(d.get("T", 0)),
        "dq": float(dq) if np.isfinite(dq) else None,
        "dm": float(t) if np.isfinite(t) else None,
    }


def run_pass(set_name: str, pass_name: str) -> int:
    cfg = SETS[set_name]
    out_dir = os.path.join(OUT, f"{set_name}_{pass_name}")
    os.makedirs(os.path.join(out_dir, "tex"), exist_ok=True)
    su = _load_su()
    _patch_read_only_cache(su)
    _patch_capture(su, exclude=(pass_name == "without"))
    argv = [
        "--roots", *cfg["roots"],
        "--out", os.path.join(out_dir, "unification_scores.csv"),
        "--tex-dir", os.path.join(out_dir, "tex"),
        "--convention", "plain",
        # absent dir: plain mode never assembles the condvar arms anyway
        "--variance-sidecar-root", os.path.join(out_dir, "_no_sidecar"),
    ]  # fmt: skip
    if cfg["arms"]:
        argv += ["--arms", cfg["arms"]]
    rc = su.main(argv)

    sent = {k: np.unique(np.concatenate(v)) for k, v in _SENT_ROWS.items()}
    by_root: dict[str, dict[str, object]] = defaultdict(dict)
    for r in _CAPT:
        by_root[r.root][r.arm] = r
    # tree synthetics / composite arms: their sentinel rows are the experts'
    sent_any = {}
    for root, arms in by_root.items():
        rows = [v for (rt, _), v in sent.items() if rt == root]
        sent_any[root] = np.unique(np.concatenate(rows)) if rows else np.empty(0, int)

    with open(os.path.join(out_dir, "arm_detail.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["root", "arm", "n_rows", "n_valid", "n_qlike", "n_sentinel_rows",
                    "n_sentinel_scored_finite", "qlike", "dm_t_vs_a0", "T_vs_a0",
                    "delta_join"])  # fmt: skip
        for root, arms in by_root.items():
            a0 = arms.get(su.A0)
            for arm, r in arms.items():
                srows = np.intersect1d(r.row_id, sent_any[root])
                _, ia, _ = np.intersect1d(r.row_id, srows, return_indices=True)
                fin = int(np.isfinite(r.loss[ia]).sum()) if len(ia) else 0
                pr = _paired(su, r, a0) if a0 is not None and r is not a0 else None
                w.writerow([
                    root, arm, r.n_rows, r.n_valid, r.n_qlike, len(srows), fin,
                    su._fmt(r.qlike, ".10g"), su._fmt(r.dm_t, ".6g"),
                    pr["T"] if pr else (int(np.isfinite(r.loss).sum())),
                    su._fmt(r.delta_join, ".10g"),
                ])  # fmt: skip

    # paired comparisons the text cites (increments + tree-bank pairs), with T
    by_arm: dict[str, list] = defaultdict(list)
    for r in _CAPT:
        by_arm[r.arm].append(r)
    pairs = [(hi, lo) for hi, lo in su._INCREMENT_PAIRS if "_condvar" not in hi + lo]
    with open(os.path.join(out_dir, "pair_detail.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["pair", "root", "n_common", "T", "dq", "dm"])
        for hi_arm, lo_arm in pairs:
            hi = su._pick_primary(by_arm.get(hi_arm, []))
            lo = su._pick_primary(by_arm.get(lo_arm, []))
            if hi is None or lo is None or hi.root != lo.root or hi.incomplete \
                    or lo.incomplete:  # fmt: skip
                continue
            p = _paired(su, hi, lo)
            w.writerow([f"{hi_arm}-vs-{lo_arm}", hi.root, p["n_common"], p["T"],
                        su._fmt(p["dq"], ".10g"), su._fmt(p["dm"], ".6g")])  # fmt: skip
        hedge = su._pick_primary(by_arm.get("tree_hedge", []))
        if hedge is not None:
            ents = list(by_root[hedge.root].values())
            experts = [r for r in ents if r.arm in su._TREE_EXPERTS and r.n_rows
                       and not r.incomplete and r.qlike]  # fmt: skip
            best = min(experts, key=lambda r: r.qlike) if experts else None
            blk2 = by_root[hedge.root].get("blk2_user")
            for lab, other in (("tree_hedge-vs-best_expert", best),
                               ("tree_hedge-vs-blk2_user", blk2)):  # fmt: skip
                if other is None:
                    continue
                p = _paired(su, hedge, other)
                w.writerow([f"{lab}({other.arm})", hedge.root, p["n_common"], p["T"],
                            su._fmt(p["dq"], ".10g"), su._fmt(p["dm"], ".6g")])  # fmt: skip
    with open(os.path.join(out_dir, "sentinel_rows.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["root", "arm", "n_sentinel_scored", "row_ids"])
        for (root, arm), v in sorted(sent.items()):
            w.writerow([root, arm, len(v), " ".join(map(str, v.tolist()))])
    return rc


# ── identification of the sentinel bars (a0's stored chunks, every root) ────


def identify() -> None:
    import pandas as pd

    su = _load_su()
    _patch_read_only_cache(su)
    cache = su.HarvestCache(enabled=True)
    rows = []
    for set_name, cfg in SETS.items():
        for root in cfg["roots"]:
            lab = os.path.basename(os.path.normpath(root))
            found = su._chunk_files(os.path.join(root, su.A0))
            raw = su._raw_chunks(found, su.A0, lab, cache)
            idx = sorted(raw)

            def cat(k, raw=raw, idx=idx):
                return np.concatenate([raw[i][k] for i in idx])

            rid, t, B, rv, yh = (cat("row_id"), cat("t"), cat("baseline"),
                                 cat("rv_raw"), cat("yhat"))  # fmt: skip
            chunk = np.concatenate([np.full(len(raw[i]["row_id"]), i) for i in idx])
            ts = pd.to_datetime(t)
            slot = ts.strftime("%H:%M").to_numpy()
            # prior same-slot observations INSIDE the panel (cumcount per slot)
            prior = pd.Series(slot).groupby(slot).cumcount().to_numpy()
            scored = chunk != idx[0]  # the first present window is the burn-in
            sent = B == SENTINEL
            warm = prior < MIN_PERIODS
            # the panel's first rows inherit pre-panel history: restrict the
            # warm-up check to slots first seen after the panel's first day
            first_seen = pd.Series(ts).groupby(slot).transform("min").to_numpy()
            born_late = first_seen > ts.min() + pd.Timedelta(days=7)
            f = yh**2 * B
            q = su.qlike_per_bar(f, rv)
            print(f"[{lab}] rows={len(rid)} sentinel={sent.sum()} "
                  f"(scored {int((sent & scored).sum())}); new-slot warm-up bars="
                  f"{int((warm & born_late).sum())}; sentinel==warm-up of a late-born "
                  f"slot: {bool(np.array_equal(sent, warm & born_late))}")  # fmt: skip
            for i in np.where(sent)[0]:
                rows.append({
                    "root": lab, "row_id": int(rid[i]), "t": str(ts[i]),
                    "date": ts[i].strftime("%Y-%m-%d"), "weekday": ts[i].strftime("%a"),
                    "clock": slot[i], "chunk": int(chunk[i]), "scored": bool(scored[i]),
                    "baseline": B[i], "rv_raw": rv[i], "yhat_a0": yh[i],
                    "f_plain_a0": f[i], "qlike_a0": q[i],
                    "prior_same_slot_obs": int(prior[i]),
                    "slot_first_seen": str(first_seen[i])[:10],
                    "cause": (f"slot {slot[i]} first printed {str(first_seen[i])[:10]}; "
                              f"{int(prior[i])} prior same-slot obs < min_periods "
                              f"{MIN_PERIODS} -> rolling mean NaN -> fillna(1.0)"),
                })  # fmt: skip
    os.makedirs(OUT, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "sentinel_bars.csv"), index=False)
    one = df[df.root == "unification_carc"]
    print("clock x count:", dict(Counter(one.clock)))
    print("date x count:", dict(Counter(one.date)))
    print(f"a0 qlike over sentinel bars: n_finite={np.isfinite(one.qlike_a0).sum()} "
          f"mean={np.nanmean(one.qlike_a0):.3f} min={np.nanmin(one.qlike_a0):.2f} "
          f"max={np.nanmax(one.qlike_a0):.2f}")  # fmt: skip


# ── comparison ───────────────────────────────────────────────────────────────


def _read_macros(path: str) -> dict[str, str]:
    out = {}
    pat = re.compile(r"\\newcommand\{\\(\w+)\}\{(.*)\}\s*$")
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            m = pat.match(line.strip())
            if m:
                out[m.group(1)] = m.group(2)
    return out


def _num(s: str | None) -> float | None:
    if s is None:
        return None
    m = re.fullmatch(r"\$?([+-]?\d+(?:\.\d+)?)(?:\\%)?\$?", s.strip())
    return float(m.group(1)) if m else None


def _csv(path: str) -> list[dict]:
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def compare() -> None:
    import pandas as pd

    lines: list[str] = []
    # 1. parity: the "with" pass must reproduce the published plain run
    for set_name, cfg in SETS.items():
        mine = pd.read_csv(
            os.path.join(OUT, f"{set_name}_with", "unification_scores.csv")
        )
        pub = pd.read_csv(cfg["published_scores"])
        j = mine.merge(pub, on=["arm", "root"], suffixes=("", "_pub"))
        dq = (j.qlike - j.qlike_pub).abs().max()
        ddm = (j.dm_t_vs_a0 - j.dm_t_vs_a0_pub).abs().max()
        dn = (j.n_bars - j.n_bars_pub).abs().max()
        lines.append(f"{set_name}: {len(j)}/{len(pub)} published (arm, root) rows matched; "
                     f"max|dQLIKE|={dq:.3g} max|dDM|={ddm:.3g} max|dn|={dn}")  # fmt: skip
    mw = _read_macros(os.path.join(OUT, "main_with", "tex", "campaign_numbers.tex"))
    mp = _read_macros(PUBLISHED_TEX)
    diff = [k for k in mp if mw.get(k) != mp[k]]
    lines.append(f"campaign_numbers.tex: {len(mp)} published macros, {len(diff)} differ "
                 f"from the 'with' pass{': ' + ', '.join(diff[:10]) if diff else ''}")  # fmt: skip
    with open(os.path.join(OUT, "parity.txt"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))

    # 2. macro diff (every \unif macro)
    mo = _read_macros(os.path.join(OUT, "main_without", "tex", "campaign_numbers.tex"))
    mrows = []
    for k in mw:
        a, b = mw[k], mo.get(k)
        na, nb = _num(a), _num(b)
        flags = []
        if a != b:
            flags.append("changed")
        if k.endswith("DM") and na is not None and nb is not None:
            if (abs(na) >= 1.96) != (abs(nb) >= 1.96):
                flags.append("crosses|t|=1.96")
            if np.sign(na) != np.sign(nb) and na != 0 and nb != 0:
                flags.append("DM sign flip")
        if (
            (k.endswith("Delta") or k.endswith("DQ"))
            and na is not None
            and nb is not None
        ):
            if np.sign(na) != np.sign(nb):
                flags.append("delta sign flip")
        if re.search(r"N(Beat|BeatSig|SobolBeat)?$", k) and a != b:
            flags.append("count changed")
        mrows.append({"macro": k, "with": a, "without": b, "flags": ";".join(flags)})
    pd.DataFrame(mrows).to_csv(os.path.join(OUT, "macro_diff.csv"), index=False)

    # 3. claims: every section line that uses a changed macro
    changed = {r["macro"]: r for r in mrows if r["flags"]}
    crow = []
    for fn in (
        "introduction",
        "methods",
        "results",
        "appendix",
        "conclusion",
        "data_prep",
    ):
        path = os.path.join(SECTIONS, f"{fn}.tex")
        with open(path, encoding="utf-8") as fh:
            hidden = 0  # \iffalse ... \fi depth: parked text is not rendered
            for ln, text in enumerate(fh, 1):
                s = text.lstrip()
                if s.startswith("\\iffalse"):
                    hidden += 1
                    continue
                if hidden and s.startswith("\\fi"):
                    hidden -= 1
                    continue
                if hidden or s.startswith("%"):
                    continue
                text = re.split(r"(?<!\\)%", text, maxsplit=1)[0]
                used = re.findall(r"\\(unif\w+)", text)
                hit = [m for m in used if m in changed]
                if hit:
                    crow.append({
                        "file": f"{fn}.tex", "line": ln,
                        "changes": " | ".join(
                            f"{m}: {changed[m]['with']} -> {changed[m]['without']}"
                            f"{' [' + changed[m]['flags'] + ']' if changed[m]['flags'] != 'changed' else ''}"
                            for m in dict.fromkeys(hit)),
                        "text": text.strip()[:300],
                    })  # fmt: skip
    pd.DataFrame(crow).to_csv(os.path.join(OUT, "claims.csv"), index=False)

    # 4. curated side-by-side of the main results
    def detail(set_name, pass_name):
        d = pd.read_csv(os.path.join(OUT, f"{set_name}_{pass_name}", "arm_detail.csv"))
        return d.set_index(["root", "arm"])

    def pdetail(set_name, pass_name):
        d = pd.read_csv(os.path.join(OUT, f"{set_name}_{pass_name}", "pair_detail.csv"))
        return d.set_index("pair")

    main_rows = [
        ("benchmark OLS HAR+cal (a0)", "a0_ols_har", "Azero"),
        ("two-block ridge, fixed", "blk2_user", "BlkTwoUser"),
        ("two-block ridge, tuned", "blk2_tuned", "BlkTwoTuned"),
        ("tuned ridge (fine grid)", "b1_ridge_tuned_fine", "BOneRidgeTunedFine"),
        ("tuned ridge (coarse)", "b1_ridge_tuned", "BOneRidgeTuned"),
        ("tuned lasso", "b2_lasso_tuned", "BTwoLassoTuned"),
        ("tuned elastic net", "b3_enet_tuned", "BThreeEnetTuned"),
        ("joint ridge (alpha=1)", "b1_ridge", "BOneRidge"),
        ("lasso (fixed)", "b2_lasso", "BTwoLasso"),
        ("tree tuned (selector)", "tree_tuned", "TreeTuned"),
        ("tree hedge", "tree_hedge", "TreeHedge"),
    ]
    dw, do = detail("main", "with"), detail("main", "without")
    out_rows = []
    for label, arm, camel in main_rows:
        cands = [ix for ix in dw.index if ix[1] == arm]
        if not cands:
            continue
        # the macro's root: complete first, then most bars (= _pick_primary)
        root = sorted(cands, key=lambda ix: -dw.loc[ix, "n_valid"])[0][0]
        a, b = dw.loc[(root, arm)], do.loc[(root, arm)]
        out_rows.append({
            "quantity": label, "arm": arm, "root": root,
            "qlike_with": a.qlike, "qlike_without": b.qlike,
            "dqlike": b.qlike - a.qlike,
            "dm_vs_a0_with": a.dm_t_vs_a0, "dm_vs_a0_without": b.dm_t_vs_a0,
            "delta_with": mw.get(f"unif{camel}Delta"), "delta_without": mo.get(f"unif{camel}Delta"),
            "n_with": a.n_valid, "n_without": b.n_valid,
            "T_dm_with": a.T_vs_a0, "T_dm_without": b.T_vs_a0,
            "sentinel_rows_in_arm": a.n_sentinel_rows,
        })  # fmt: skip
    pw, po = pdetail("main", "with"), pdetail("main", "without")
    for pair in pw.index:
        if not any(s in pair for s in ("tree_hedge", "b2_lasso_tuned-vs", "b3_enet_tuned-vs",
                                       "b1_ridge_tuned_fine-vs")):  # fmt: skip
            continue
        a, b = pw.loc[pair], po.loc[pair]
        out_rows.append({
            "quantity": f"paired {pair}", "arm": pair, "root": a.root,
            "qlike_with": None, "qlike_without": None, "dqlike": None,
            "dm_vs_a0_with": a.dm, "dm_vs_a0_without": b.dm,
            "delta_with": a.dq, "delta_without": b.dq,
            "n_with": a.n_common, "n_without": b.n_common,
            "T_dm_with": a["T"], "T_dm_without": b["T"], "sentinel_rows_in_arm": None,
        })  # fmt: skip
    fw, fo = detail("fomc", "with"), detail("fomc", "without")
    for root, arm in fw.index:
        a, b = fw.loc[(root, arm)], fo.loc[(root, arm)]
        out_rows.append({
            "quantity": f"FOMC table: {arm}", "arm": arm, "root": root,
            "qlike_with": a.qlike, "qlike_without": b.qlike, "dqlike": b.qlike - a.qlike,
            "dm_vs_a0_with": a.dm_t_vs_a0, "dm_vs_a0_without": b.dm_t_vs_a0,
            "delta_with": a.delta_join, "delta_without": b.delta_join,
            "n_with": a.n_valid, "n_without": b.n_valid,
            "T_dm_with": a.T_vs_a0, "T_dm_without": b.T_vs_a0,
            "sentinel_rows_in_arm": a.n_sentinel_rows,
        })  # fmt: skip
    imp = pd.DataFrame(out_rows)
    imp.to_csv(os.path.join(OUT, "impact_main.csv"), index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print(imp.to_string(index=False, float_format=lambda x: f"{x:.5f}"))
    tb = [r for r in mrows if r["macro"].startswith("unifTreeBank")]
    print(
        "\ntree bank:",
        "; ".join(f"{r['macro'][12:]} {r['with']}->{r['without']}" for r in tb),
    )
    print(f"\n{sum(1 for r in mrows if r['flags'])} of {len(mrows)} macros change; "
          f"{len(crow)} section lines use one")  # fmt: skip


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--set", choices=tuple(SETS), required=True)
    r.add_argument("--pass", dest="pass_name", choices=PASSES, required=True)
    sub.add_parser("identify")
    sub.add_parser("compare")
    sub.add_parser("all")
    a = ap.parse_args(argv)
    if a.cmd == "run":
        return run_pass(a.set, a.pass_name)
    if a.cmd == "identify":
        identify()
        return 0
    if a.cmd == "compare":
        compare()
        return 0
    identify()
    for set_name in SETS:
        for p in PASSES:
            d = os.path.join(OUT, f"{set_name}_{p}")
            os.makedirs(d, exist_ok=True)
            with open(os.path.join(d, "score.log"), "w") as log:
                rc = subprocess.call(
                    [sys.executable, os.path.abspath(__file__), "run", "--set", set_name,
                     "--pass", p], stdout=log, stderr=subprocess.STDOUT, cwd=_ROOT)  # fmt: skip
            print(f"pass {set_name}/{p}: rc={rc}", flush=True)
    compare()
    return 0


if __name__ == "__main__":
    sys.exit(main())
