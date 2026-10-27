"""What moved between the pre-patch and the FOMC-patched results (RULE.txt ADDENDUM 3 (6)), on the laptop.

Reads the pre-patch and patched outputs (ladder: scores.csv, readings.csv, rank_primary.csv; grid: scores.csv,
readings.csv; tree SHAP: m1_variables.csv, m1_classes.csv, m2_linear_shares.csv, m2_nonstandard_D.csv, m4_contrasts.csv
when both exist) and writes results/ladder_2026-10-26/fomcfix/moved.csv and moved.txt.

usage: python ff_compare.py [--shap-pre DIR --shap-post DIR]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
FF = HERE.parent
LAD = FF.parent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shap-pre", default=str(LAD.parent / "tree_shap_w3000_2014start_2026-10-23"))
    ap.add_argument("--shap-post", default=str(LAD.parent / "tree_shap_w3000_2014start_fomcfix_2026-10-26"))
    a = ap.parse_args()
    L: list[str] = []
    rows = []
    # ladder (W = 3,000), primary ranking
    pre, post = pd.read_csv(LAD / "scores.csv"), pd.read_csv(FF / "scores.csv")
    key = ["model", "ranking", "k", "span"]
    j = pre.dropna(subset=["k"]).merge(post.dropna(subset=["k"]), on=key, suffixes=("_pre", "_post"))
    j = j[j.ranking.isin(["P", "both"])].drop_duplicates(["model", "k", "span"])
    j["dqlike"] = j["qlike_post"] - j["qlike_pre"]
    for r in j.itertuples():
        rows.append(
            dict(
                what="ladder",
                model=r.model,
                span=r.span,
                k=int(r.k),
                pre=r.qlike_pre,
                post=r.qlike_post,
                delta=r.dqlike,
                retained_pre=r.retained_pct_pre,
                retained_post=r.retained_pct_post,
                series_pre=r.series_pre,
                series_post=r.series_post,
            )
        )
    L.append("LADDER (W = 3,000, primary ranking): QLIKE pre -> post (x 1e3 change), retained % pre -> post")
    for sp in ("all", "c16"):
        L.append(f"  span {sp}")
        for m, g in j[j.span == sp].groupby("model", sort=False):
            g = g.sort_values("k", ascending=False)
            cells = "  ".join(
                f"k{int(r.k)} {r.qlike_post:.4f} ({1e3 * r.dqlike:+.2f}; R {r.retained_pct_pre:.0f}->{r.retained_pct_post:.0f})"
                for r in g.itertuples()
            )
            L.append(f"    {m:7s} {cells}")
    rp, rq = pd.read_csv(LAD / "readings.csv"), pd.read_csv(FF / "readings.csv")
    rj = rp.merge(rq, on=["model", "ranking", "span"], suffixes=("_pre", "_post"))
    L.append("  readings (k90, verdict) pre -> post")
    for r in rj.itertuples():
        L.append(
            f"    {r.model:7s} {r.ranking:24s} {r.span:4s} k90 {r.k90_pre} -> {r.k90_post}; {r.verdict_pre} -> {r.verdict_post}"
        )
    kp, kq = pd.read_csv(LAD / "rank_primary.csv"), pd.read_csv(FF / "rank_primary.csv")
    L.append("  primary top 5 pre | post")
    for m in kq.model.unique():
        tp = kp[(kp.model == m) & (kp["rank"] <= 5)].sort_values("rank").series.tolist()
        tq = kq[(kq.model == m) & (kq["rank"] <= 5)].sort_values("rank").series.tolist()
        L.append(f"    {m:7s} {', '.join(tp)} | {', '.join(tq)}")
    # grid
    gpre, gpost = LAD / "grid", FF / "grid"
    if (gpost / "scores.csv").is_file():
        sp_, sq = pd.read_csv(gpre / "scores.csv"), pd.read_csv(gpost / "scores.csv")
        gj = sp_.merge(sq, on=["wdays", "model", "k", "span"], suffixes=("_pre", "_post"))
        for r in gj.itertuples():
            rows.append(
                dict(
                    what="grid",
                    model=r.model,
                    span=r.span,
                    k=int(r.k),
                    wdays=int(r.wdays),
                    pre=r.gain_pre,
                    post=r.gain_post,
                    delta=r.gain_post - r.gain_pre,
                )
            )
        rp2, rq2 = pd.read_csv(gpre / "readings.csv"), pd.read_csv(gpost / "readings.csv")
        r2 = rp2.merge(rq2, on=["model", "span"], suffixes=("_pre", "_post"))
        L.append(
            "\nGRID: best k by window (W 1,000 / 2,000 / 3,000 / 3,500) pre -> post; full-set gain x 1e3 pre -> post"
        )
        for r in r2.itertuples():
            gfp = " ".join(f"{1e3 * float(x):.1f}" for x in str(r.gain_full_by_W_pre).split())
            gfq = " ".join(f"{1e3 * float(x):.1f}" for x in str(r.gain_full_by_W_post).split())
            L.append(
                f"  {r.model:6s} {r.span:4s} k* {r.kstar_by_W_pre} ({r.kstar_reading_pre}) -> {r.kstar_by_W_post} "
                f"({r.kstar_reading_post}); full gain {gfp} -> {gfq}"
            )
    # tree SHAP
    sp0, sp1 = Path(a.shap_pre), Path(a.shap_post)
    if (sp1 / "m1_variables.csv").is_file():
        m0, m1 = pd.read_csv(sp0 / "m1_variables.csv"), pd.read_csv(sp1 / "m1_variables.csv")
        mj = m0.merge(m1, on=["est", "variable"], suffixes=("_pre", "_post"))
        mj["dshare"] = mj["share_post"] - mj["share_pre"]
        L.append("\nTREE SHAP M1: share of total mean |SHAP| (%), pre -> post, the FOMC series and the largest moves")
        for e, g in mj.groupby("est", sort=False):
            fo = g[g.variable.str.startswith("fomc")]
            L.append(
                f"  {e:7s} "
                + "; ".join(
                    f"{r.variable} {100 * r.share_pre:.2f} -> {100 * r.share_post:.2f}" for r in fo.itertuples()
                )
            )
            big = g.reindex(g["dshare"].abs().sort_values(ascending=False).index).head(4)
            L.append(
                "          largest moves: "
                + "; ".join(f"{r.variable} {100 * r.dshare:+.2f} pp" for r in big.itertuples())
            )
            for r in g.itertuples():
                rows.append(
                    dict(
                        what="shap_m1",
                        model=e,
                        k=None,
                        span=r.variable,
                        pre=r.share_pre,
                        post=r.share_post,
                        delta=r.dshare,
                    )
                )
        c0, c1 = pd.read_csv(sp0 / "m1_classes.csv"), pd.read_csv(sp1 / "m1_classes.csv")
        L.append("  classes (pre | post):")
        L.append("    " + c0.to_string(index=False).replace("\n", "\n    "))
        L.append("    " + c1.to_string(index=False).replace("\n", "\n    "))
        for f in ("m2_nonstandard_D.csv", "m4_contrasts.csv"):
            if (sp0 / f).is_file() and (sp1 / f).is_file():
                L.append(f"  {f} pre:")
                L.append("    " + pd.read_csv(sp0 / f).head(12).to_string(index=False).replace("\n", "\n    "))
                L.append(f"  {f} post:")
                L.append("    " + pd.read_csv(sp1 / f).head(12).to_string(index=False).replace("\n", "\n    "))
    pd.DataFrame(rows).to_csv(FF / "moved.csv", index=False)
    (FF / "moved.txt").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
