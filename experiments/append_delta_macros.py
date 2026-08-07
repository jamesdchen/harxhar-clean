"""One-shot: append \\unif<Camel>Delta macros to campaign_numbers.tex from the
scores CSV (same formula the scorer now emits natively; this bridges until its
next full run)."""

import csv

REG = {
    "a_bucket_moments": "Moments",
    "a_bucket_liquidity": "Liquidity",
    "a_bucket_market_ew": "MarketEw",
    "a_bucket_market_vw": "MarketVw",
    "a_bucket_sentiment": "Sentiment",
    "a_bucket_implied_vol": "ImpliedVol",
    "a_bucket_vol_demand": "VolDemand",
    "a_bucket_all_features": "AllFeatures",
    "b1_ridge": "BOneRidge",
    "b2_lasso": "BTwoLasso",
    "b1_ridge_tuned": "BOneRidgeTuned",
    "b2_lasso_tuned": "BTwoLassoTuned",
    "blk2_user": "BlkTwoUser",
    "blk3_user": "BlkThreeUser",
    "blk4_user": "BlkFourUser",
    "blk2_doc": "BlkTwoDoc",
    "blk3_doc": "BlkThreeDoc",
    "blk4_doc": "BlkFourDoc",
    "c4_product_alone_user": "ProductAloneUser",
    "c4_product_alone_doc": "ProductAloneDoc",
    "d3_transmission_alone_user": "TransAloneUser",
    "d3_transmission_alone_doc": "TransAloneDoc",
}

rows = {
    r["arm"]: float(r["qlike"])
    for r in csv.DictReader(open("results/unification_scores.csv"))
    if r.get("qlike")
}
a0 = rows["a0_ols_har"]

with open("writeup/generated/campaign_numbers.tex", "a", encoding="utf-8") as f:
    f.write(
        "% Delta macros (arm QLIKE minus a0) -- derived from"
        " unification_scores.csv;\n% the scorer emits these natively on its"
        " next run.\n"
    )
    for arm, camel in REG.items():
        if arm in rows:
            f.write(
                "\\newcommand{\\unif%sDelta}{$%+.5f$}\n" % (camel, rows[arm] - a0)
            )
        else:
            f.write(
                "\\newcommand{\\unif%sDelta}{\\pending{%s}}\n"
                % (camel, arm.replace("_", "\\_"))
            )
print("deltas appended")
