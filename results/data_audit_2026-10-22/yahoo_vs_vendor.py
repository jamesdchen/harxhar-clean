"""Does the Yahoo feed agree with the vendor VIX / VIX3M / VVIX on the rest of the sample? (read-only)

Hourly Yahoo starts 2023-12; before that only daily closes exist. Compare each Yahoo daily close with the vendor's
last print of that day at or before 16:15 ET (VIX's official close is 16:15; VIX3M/VVIX close 16:15 too), and with
the print at the 16:00 bar end. Also the hourly overlap (Dec 2023 - Feb 2024) at matched bar ends.
"""

from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(r"C:\Users\james\CC Allowed\harxhar-0dte-professor")
v = pd.read_parquet(REPO / "data" / "vix_and_voldemand.parquet", columns=["endbartime", "vix", "vix3m", "vvix"])
t = pd.to_datetime(v["endbartime"])
if t.dt.tz is not None:
    t = t.dt.tz_convert("America/New_York").dt.tz_localize(None)
v["t"] = t
v["day"] = v["t"].dt.normalize()
hhmm = v["t"].dt.strftime("%H:%M")
print("vendor span", v["t"].min(), v["t"].max(), "bar ends sample:", sorted(hhmm.unique())[:5], "...", sorted(hhmm.unique())[-5:])

rows = []
for s in ("vix", "vix3m", "vvix"):
    y = pd.read_parquet(REPO / "data" / "free_feed" / f"{s}_1d_yahoo.parquet")
    yd = pd.Series(y["close"].to_numpy(float), index=pd.to_datetime(y["t"]).dt.tz_localize(None).dt.normalize())
    obs = v[v[s].notna()]
    last_le = obs[obs["t"].dt.strftime("%H:%M") <= "16:15"].groupby("day").last()[s]
    at16 = obs[obs["t"].dt.strftime("%H:%M") == "16:00"].set_index("day")[s]
    # the vendor bar ending 16:30 holds the 16:15 official close, the print Yahoo's daily close is
    at1630 = obs[obs["t"].dt.strftime("%H:%M") == "16:30"].set_index("day")[s]
    for lab, vend in (("last<=16:15", last_le), ("16:00 bar", at16), ("16:30 bar", at1630)):
        if vend.empty:
            continue
        j = pd.concat([vend.rename("vendor"), yd.rename("yahoo")], axis=1, join="inner").dropna()
        rel = (j["yahoo"] / j["vendor"] - 1.0).abs()
        dv, dy = np.log(j["vendor"]).diff(), np.log(j["yahoo"]).diff()
        for yr, g in [("all", j)] + list(j.groupby(j.index.year)):
            r = rel.loc[g.index]
            rows.append(dict(series=s, vendor_point=lab, year=yr, days=len(g), exact=float((r < 1e-6).mean()),
                             within_0p5pct=float((r < 0.005).mean()), median_rel=float(r.median()),
                             p99_rel=float(r.quantile(0.99)),
                             corr_dlog=float(dv.loc[g.index].corr(dy.loc[g.index])) if len(g) > 2 else np.nan))
df = pd.DataFrame(rows)
pd.set_option("display.width", 200)
out = Path(__file__).with_name("yahoo_vs_vendor_daily.csv")
df.to_csv(out, index=False, float_format="%.6g")
print(df[df["year"] == "all"].round(5).to_string(index=False))
print()
for s in ("vix", "vix3m", "vvix"):
    g = df[(df["series"] == s) & (df["vendor_point"] == "last<=16:15") & (df["year"] != "all")]
    print(s, "by year (exact share / median rel / p99 rel):")
    print(g[["year", "days", "exact", "within_0p5pct", "median_rel", "p99_rel", "corr_dlog"]].round(5).to_string(index=False))
print("wrote", out)
