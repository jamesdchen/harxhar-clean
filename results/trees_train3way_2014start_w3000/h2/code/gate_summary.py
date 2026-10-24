"""Hoffman2 gates of the trees' three-way comparison (RULE.txt addendum C): a summary of gate/h2/<tag>/ (one CPU model a
tag) after experiments/trees_train3way.py accept has checked them (gate 1: the forest's RTH13 --ends 10 pieces against
the templates' forecasts, <= 1e-14 relative; gate 2: every row against the scored origins only, bit for bit).

Adds, for the record: each piece's CPU model, host, threads and seconds a refit; each Hoffman2 gate piece against CARC's
EPYC gate piece of the same slice (gate/g1, gate/g2 of the CARC run: bit-identical? max relative difference of
pred_adj); each tag against each other tag (which Hoffman2 models give the same forests bit for bit); and the walk
projection of the assigned lines on each tag's measured seconds a refit (the ALL48 forest full set, every-row gate piece,
at its threads: 20, or 8 for the pieces resubmitted at 8 threads on 2026-10-24 04:10 because Hoffman2 did not schedule
20-slot jobs; CARC's rf_skx gate (d): 8 = 20 threads bit for bit).  Writes <out>/gate_summary.json, <out>/gate_summary.txt, <out>/passed_cpus.txt (the accepted CPU models
whose longest assigned piece fits the queue's 24 h with margin) and <out>/walk_resources.txt (ARCH = their UGE arch
values, HRT = the walks' h_rt).

usage: python gate_summary.py --h2-gates DIR --carc-gates DIR --accepted JSON --tags FILE --design-root DIR
                              --taskfile FILE --out DIR
"""

from __future__ import annotations

import argparse
import json
import math
from argparse import Namespace
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

H_RT_MAX = 24.0  # Hoffman2 campus jobs (not highp): at most 24 h
FIT_MARGIN = 1.3  # the longest piece times this must fit in 22 h for a model to take walks
HRT_MARGIN = 1.5  # h_rt = the longest piece of the eligible models times this (at least 4 h, at most 24 h)


def compare(a: Path, b: Path) -> dict[str, Any]:
    """Two pieces of the same slice: dates, pred_adj / true_adj / B bit for bit, max relative difference of pred_adj."""
    fa, fb = pd.read_parquet(a).set_index("date"), pd.read_parquet(b).set_index("date")
    i = fa.index.intersection(fb.index)
    pa, pb = fa.loc[i, "pred_adj"].to_numpy(float), fb.loc[i, "pred_adj"].to_numpy(float)
    same = len(i) == len(fa) == len(fb) and all(
        np.array_equal(fa.loc[i, c].to_numpy(float), fb.loc[i, c].to_numpy(float))
        for c in ("pred_adj", "true_adj", "B")
    )
    rel = float(np.max(np.abs(pa - pb) / np.abs(pb))) if len(i) else float("nan")
    return dict(rows=len(i), identical=bool(same), n_differ=int(np.sum(pa != pb)), max_rel_diff=rel)


def meta(p: Path) -> dict[str, Any] | None:
    j = p.with_suffix(".json")
    return json.loads(j.read_text()) if j.is_file() else None


def norm(name: str, tag: str) -> str:
    return name.replace(f"_{tag}", "_TAG")


def carc_name(name: str, tag: str, threads: int) -> str:
    """The CARC gate piece of the same slice: g1 ends10_<tag>[_t8] -> ends10_t20 / ends10_t8; g2 the same name."""
    if "_ends10_" in name:
        return name.split("_ends10_")[0] + f"_ends10_t{threads}.parquet"
    return name


def assigned_refits(design_root: Path, taskfile: Path) -> list[int]:
    """Refits of each line of the Hoffman2 task file (the arm's origins in the line's block of trading days)."""
    import trees_train3way as TT  # the comparison's module (PYTHONPATH = the code tree's experiments)

    out = []
    for ln in taskfile.read_text().splitlines():
        if not ln.strip():
            continue
        arm, _est, s, part = ln.split()
        des = TT.Design(design_root, TT.ARMS[arm]["design"], s)
        rows, _ = TT.select(Namespace(ends=0, from_="", part=part, tag=""), design_root, des, arm)
        out.append(len(rows))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    for k in ("--h2-gates", "--carc-gates", "--accepted", "--tags", "--design-root", "--taskfile", "--out"):
        ap.add_argument(k, required=True)
    a = ap.parse_args()
    h2, carc, out = Path(a.h2_gates), Path(a.carc_gates), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tags: dict[str, dict[str, str]] = {}
    for ln in Path(a.tags).read_text().splitlines():
        if ln.strip() and not ln.startswith("#"):
            t, arch, rx = ln.split(None, 2)
            tags[t] = dict(arch=arch, cpu_regex=rx.strip())
    acc = json.loads(Path(a.accepted).read_text()) if Path(a.accepted).is_file() else {"accepted": {}, "checked": {}}
    accepted_tags = {v["tag"]: cpu for cpu, v in acc["accepted"].items()}
    refits = assigned_refits(Path(a.design_root), Path(a.taskfile))
    s: dict[str, Any] = dict(
        tags={},
        cross_tag={},
        assigned_lines=len(refits),
        assigned_refits=int(sum(refits)),
        max_refits_a_line=int(max(refits)) if refits else 0,
    )
    carc_every = carc / "g2" / "pieces" / "ALL48_rf_skx_all_features_every.parquet"
    cm = meta(carc_every)
    carc_sec = cm["wall_sec"] / max(1, cm["refits"]) if cm else float("nan")
    s["carc_sec_refit_all48_rf_full_t20"] = carc_sec
    s["carc_threads"] = cm["threads"] if cm else None
    s["carc_cpu"] = cm["cpu"] if cm else None
    for t in sorted(x.name for x in h2.iterdir() if x.is_dir()):
        rec: dict[str, Any] = dict(tags.get(t, {}), pieces={}, accept=acc["checked"].get(t))
        for sub in ("g1", "g2"):
            for p in sorted((h2 / t / sub / "pieces").glob("*.parquet")):
                m = meta(p)
                r: dict[str, Any] = dict(sub=sub, complete=m is not None)
                if m:
                    r |= dict(
                        rows=m["rows"],
                        refits=m["refits"],
                        wall_sec=m["wall_sec"],
                        sec_refit=m["wall_sec"] / max(1, m["refits"]),
                        threads=m["threads"],
                        cpu=m["cpu"],
                        cpus=m["cpus"],
                        host=m["host"],
                        uge_job=f"{m['slurm_job']}.{m['slurm_task']}",
                        versions=m["versions"],
                    )
                    q = carc / sub / "pieces" / carc_name(p.name, t, int(m["threads"]))
                    r["vs_carc"] = dict(piece=q.name, **compare(p, q)) if q.is_file() else None
                rec["pieces"][p.name] = r
        ev = rec["pieces"].get("ALL48_rf_skx_all_features_every.parquet", {})
        sec = ev.get("sec_refit")
        nt = int(ev.get("threads") or 0)
        rec["sec_refit_all48_rf_full"] = sec
        rec["threads_all48_rf_full"] = nt
        if sec and refits:
            rec["projection"] = dict(
                ratio_to_carc_20_threads=sec / carc_sec,
                cpu_seconds_a_refit=sec * nt,
                cpu_ratio_to_carc=sec * nt / (carc_sec * 20),
                longest_piece_hours=max(refits) * sec / 3600,
                wall_hours_all_lines=sum(refits) * sec / 3600,
                cpu_hours_all_lines=sum(refits) * sec * nt / 3600,
            )
        rec["accepted"] = t in accepted_tags
        s["tags"][t] = rec
    names = list(s["tags"])
    for t1, t2 in combinations(names, 2):
        res = {}
        for p in sorted((h2 / t1).glob("g*/pieces/*.parquet")):
            n = norm(p.name, t1)
            q = h2 / t2 / p.parent.relative_to(h2 / t1) / n.replace("_TAG", f"_{t2}")
            if q.is_file() and meta(p) and meta(q):
                res[n] = compare(p, q)
        s["cross_tag"][f"{t1} vs {t2}"] = dict(
            pieces=len(res), all_identical=bool(res) and all(v["identical"] for v in res.values()), detail=res
        )
    elig = {
        t: r
        for t, r in s["tags"].items()
        if r["accepted"] and r.get("projection") and r["projection"]["longest_piece_hours"] * FIT_MARGIN <= 22.0
    }
    s["eligible_for_walks"] = sorted(elig)
    (out / "passed_cpus.txt").write_text("".join(f"{accepted_tags[t]}\n" for t in sorted(elig)))
    if elig:
        hmax = max(r["projection"]["longest_piece_hours"] for r in elig.values())
        h = int(min(H_RT_MAX, max(4.0, math.ceil(hmax * HRT_MARGIN))))
        arch = "|".join(sorted(r["arch"] for r in elig.values()))
        (out / "walk_resources.txt").write_text(f"ARCH='{arch}'\nHRT={h}:00:00\n")
        s["walk_resources"] = dict(arch=arch, h_rt_hours=h)
    else:
        (out / "walk_resources.txt").write_text("")
    (out / "gate_summary.json").write_text(json.dumps(s, indent=1, default=str))
    lines = [
        f"Hoffman2 gates (forest rf_skx): {len(names)} tags; accepted {sorted(accepted_tags)}; eligible for walks "
        f"{sorted(elig)}; assigned {len(refits)} lines, {sum(refits)} refits (longest line {max(refits or [0])})",
        f"CARC EPYC ALL48 forest full set, 20 threads: {carc_sec:.1f} s a refit ({s['carc_cpu']})",
    ]
    for t, r in s["tags"].items():
        ac = r["accept"] or {}
        lines.append(
            f"[{t}] {r.get('arch')} cpus {ac.get('cpus')} accept pass={ac.get('pass')} gate1_rc={ac.get('gate1_rc')} "
            f"gate2_rc={ac.get('gate2_rc')}; ALL48 full s/refit={r['sec_refit_all48_rf_full']} "
            f"at {r['threads_all48_rf_full']} threads; "
            f"projection={r.get('projection')}"
        )
        for n, p in r["pieces"].items():
            vc = p.get("vs_carc") or {}
            lines.append(
                f"   {p['sub']} {n}: complete={p['complete']} rows={p.get('rows')} threads={p.get('threads')} "
                f"s/refit={p.get('sec_refit', float('nan')):.2f} host={p.get('host')} | vs CARC {vc.get('piece')}: "
                f"identical={vc.get('identical')} n_differ={vc.get('n_differ')} max_rel={vc.get('max_rel_diff')}"
            )
    for k, v in s["cross_tag"].items():
        lines.append(f"cross-tag {k}: {v['pieces']} pieces, all identical {v['all_identical']}")
    (out / "gate_summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
