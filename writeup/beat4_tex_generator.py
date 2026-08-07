"""beat4.tex v2: quirk-scrubbed table 1 (+date range), NEW table 2 = bundle-combo factorial
(real forecast metrics from both clusters), pgfplots skill chart."""

import json

SCR = "C:/Users/james/AppData/Local/Temp/claude/C--Users-james-CC-Allowed-harxhar-clean/8d0e5424-ba3e-4191-a154-aa9669f6b8a4/scratchpad"
REPO = "C:/Users/james/CC Allowed/harxhar-clean"
chart = json.load(open(f"{SCR}/scratchpad_beat4_chart.json"))
mz = {json.loads(li)["bucket"]: json.loads(li) for li in open(f"{SCR}/scratchpad_beat4_mz.jsonl")}
s48 = json.load(open(f"{REPO}/results/vrp_stage2_fm_tw48k/summary.json"))
s144 = json.load(open(f"{REPO}/results/vrp_stage2_fm_tw144k/summary.json"))
c48 = {c["tid"]: c for c in s48["cells"]}
c144 = {c["tid"]: c for c in s144["cells"]}
SURV = c48[0]["survivors"]
SHORT = {"sumabsret": "absret", "sumvolume": "volume", "vd_voldemand_spx_open_only": "vdem\\_spx",
         "vd_voldemand_all_open_only": "vdem\\_all", "vd_vvix": "vvix", "sumbipow": "bipow",
         "sn_stocktwits_sentcount": "sentcount", "sn_stocktwits_attention": "attention"}

# ---------------- Table 1 (bucket battery) ----------------
ROWS = [
    ("all_buckets", r"\textbf{ALL buckets}", r"ridge $\alpha{=}30$", 0.14265, 1.213e-06, 4.571e-11, 0.6969, -13.0, "1.2 \\times 10^{-38}"),
    ("implied_vol", "implied\\_vol (VIX/VVIX)", r"ridge $\alpha{=}1$", 0.14669, 1.236e-06, 4.558e-11, 0.6978, -8.7, "2.3 \\times 10^{-18}"),
    ("moments", "moments", r"ridge $\alpha{=}1000$", 0.14678, 1.246e-06, 4.624e-11, 0.6935, -7.5, "4.6 \\times 10^{-14}"),
    ("liquidity", "liquidity", r"ridge $\alpha{=}1$", 0.14679, 1.250e-06, 4.599e-11, 0.6951, -10.8, "3.8 \\times 10^{-27}"),
    ("sentiment", "sentiment", r"enet $\alpha{=}10^{-3}$, $\ell_1{=}0.6$", 0.14726, 1.250e-06, 4.484e-11, 0.7027, -8.2, "2.9 \\times 10^{-16}"),
    ("market_vw", "market\\_vw", r"enet $\alpha{=}10^{-3}$, $\ell_1{=}0.2$", 0.14744, 1.253e-06, 4.558e-11, 0.6978, -8.7, "4.5 \\times 10^{-18}"),
    ("market_ew", "market\\_ew", r"ridge $\alpha{=}1000$", 0.14758, 1.257e-06, 4.593e-11, 0.6955, -9.3, "1.0 \\times 10^{-20}"),
    ("vol_demand", "vol\\_demand", r"enet $\alpha{=}10^{-3}$, $\ell_1{=}0.6$", 0.14822, 1.259e-06, 4.553e-11, 0.6982, -6.2, "5.0 \\times 10^{-10}"),
    ("har_only", r"\emph{HAR-only (baseline)}", "none (OLS)", 0.14863, 1.262e-06, 4.545e-11, 0.6987, None, None),
]


def sci(x):
    m, e = f"{x:.3e}".split("e")
    return f"${m} \\times 10^{{{int(e)}}}$"


t1 = []
for key, pretty, pen, q, mae_, mse_, r2, dm, p in ROWS:
    z = mz[key]
    qs = f"\\textbf{{{q:.5f}}}" if key == "all_buckets" else f"{q:.5f}"
    r2s = f"\\textbf{{{r2:.4f}}}" if key == "sentiment" else f"{r2:.4f}"
    dms = "(baseline)" if dm is None else f"${dm:+.1f}$, ${p}$"
    t1.append(f"{pretty} & {pen} & {qs} & {sci(mae_)} & {sci(mse_)} & {r2s} & {z['mz_beta']:.3f} & {z['mz_r2']:.4f} & {dms} \\\\")
T1 = "\n".join(t1)

# ---------------- Table 2 (bundle-combo factorial) ----------------
def combo_label(c):
    if not c["active"]:
        return r"\emph{HAR ladder $+$ hour dummies (baseline)}"
    if len(c["active"]) == 8:
        return "all eight bundles"
    return ", ".join(SHORT[a] for a in c["active"])


def pfmt(p):
    if p == 0.0 or p < 1e-300:
        return "$< 10^{-300}$"
    m, e = f"{p:.1e}".split("e")
    return f"${m} \\times 10^{{{int(e)}}}$"


def row2(tid, best=False):
    a, b = c48[tid], c144.get(tid)
    qs = f"\\textbf{{{a['qlike']:.5f}}}" if best else f"{a['qlike']:.5f}"
    dms = "(baseline)" if tid == 0 else f"${a['dm']:+.1f}$, {pfmt(a['dm_p'])}"
    q144 = f"{b['qlike']:.5f}" if b else "---"
    dm144 = "(baseline)" if tid == 0 else (f"${b['dm']:+.1f}$" if b else "---")
    return (f"{combo_label(a)} & {qs} & {sci(a['mae'])} & {a['oos_r2']:.4f} & {a['mz_r2']:.4f} & {dms} & {q144} & {dm144} \\\\")


t2 = []
# singles, ranked by tw48k qlike
singles = sorted([2**i for i in range(8)], key=lambda t: c48[t]["qlike"])
best_tid = min(c48, key=lambda t: c48[t]["qlike"])
t2.append(r"\multicolumn{8}{@{}l}{\emph{Main effects: the eight single-bundle cells (one bundle added to HAR), ranked by QLIKE}} \\")
t2.append(r"\midrule")
for tid in singles:
    t2.append(row2(tid))
t2.append(r"\midrule")
t2.append(r"\multicolumn{8}{@{}l}{\emph{Selected combination cells: best by QLIKE; best-six of the P\&L factorial; the full set}} \\")
t2.append(r"\midrule")
t2.append(row2(best_tid, best=True))
t2.append(row2(219))  # best-6 P&L cell
t2.append(row2(255))  # all-8
t2.append(r"\midrule")
t2.append(row2(0))
T2 = "\n".join(t2)

# exact first-order (two-way) interaction effects from the full 2^8, both windows
import itertools

import numpy as np


def interactions(cells):
    q = np.array([cells[t]["qlike"] for t in range(256)])
    x = np.array([[2 * ((t >> i) & 1) - 1 for i in range(8)] for t in range(256)])
    out = {}
    for i, j in itertools.combinations(range(8), 2):
        s = x[:, i] * x[:, j]
        out[(i, j)] = float(q[s > 0].mean() - q[s < 0].mean())
    return out

i48, i144 = interactions(c48), interactions(c144)
top = sorted(i48, key=lambda k: -abs(i48[k]))[:3]
int_line = "; ".join(
    f"{SHORT[SURV[i]]}$\\times${SHORT[SURV[j]]} ${i48[(i, j)]:+.3f}$/${i144[(i, j)]:+.3f}$"
    for i, j in top
)

# main effects footer
me48, me144 = s48["main_effects_qlike"], s144["main_effects_qlike"]
me_line = "; ".join(
    f"{SHORT[k]} ${me48[k]:+.3f}$/${me144[k]:+.3f}$"
    for k in sorted(me48, key=lambda k: me48[k])
)

# ---------------- chart ----------------
order = ["all_buckets", "implied_vol", "moments", "liquidity", "sentiment", "market_vw", "market_ew", "vol_demand"]
short = {"all_buckets": "ALL", "implied_vol": "IV", "moments": "MOM", "liquidity": "LIQ",
         "sentiment": "SENT", "market_vw": "MKTvw", "market_ew": "MKTew", "vol_demand": "VOLD"}
rowd = {r["bucket"]: r for r in chart["rows"]}
plots = []
for mk in ["QLIKE", "HMSE", "MAE", "MSE", "OOS-R2"]:
    pts = " ".join(f"({short[b]},{rowd[b]['skill'][mk] * 100:.2f})" for b in order)
    plots.append(f"\\addplot+[ybar, fill=c{mk.replace('-', '')}, draw=white, line width=0.3pt] coordinates {{{pts}}};")
PLOTS = "\n".join(plots)
SYMX = ",".join(short[b] for b in order)

tex = r"""\documentclass[11pt]{article}
\usepackage[margin=0.7in]{geometry}
\usepackage{booktabs}
\usepackage{pgfplots}
\pgfplotsset{compat=1.18}
\usepackage{xcolor}
\definecolor{cQLIKE}{HTML}{0072B2}
\definecolor{cHMSE}{HTML}{009E73}
\definecolor{cMAE}{HTML}{56B4E9}
\definecolor{cMSE}{HTML}{E69F00}
\definecolor{cOOSR2}{HTML}{D55E00}
\begin{document}

\begin{table}[t]
\centering
\caption{Every-bar (refit $=1$) exogenous-bucket battery vs.\ the HAR baseline. Train window
1000 trading days; fixed COVID-inclusive evaluation window \textbf{Jan.\ 2018 -- May 2025}
($n = 74{,}934$ thirty-minute bars; 1{,}843 trading days; identical rows for every cell); best
penalty per bucket selected by QLIKE; HAR block unpenalized (OLS via Frisch--Waugh--Lovell).
DM $=$ Diebold--Mariano test on per-bar QLIKE loss vs.\ HAR-only, reported as ($t$, $p$): the
DM statistic is the $t$-ratio of the mean loss differential to its Newey--West HAC standard
error (Harvey--Leybourne--Newbold small-sample correction), with two-sided $p$-value.
The $90\%$ model confidence set contains ALL buckets alone;
every other bucket is eliminated. Raw-variance MSE and OOS-$R^2$ are dominated by a small number
of COVID bars; QLIKE, HMSE and MAE are the level-robust rankers.}
\vspace{4pt}
\footnotesize
\setlength{\tabcolsep}{4pt}
\begin{tabular}{@{}llccccccc@{}}
\toprule
exogenous bucket & penalty & QLIKE & MAE & MSE & OOS-$R^2$ & MZ-$\beta$ & MZ-$R^2$ & DM vs.\ HAR ($t$, $p$) \\
\midrule
__T1__
\bottomrule
\end{tabular}
\end{table}

\begin{table}[t]
\centering
\caption{Intra-bucket bundle-combination factorial ($2^8$ over the stage-1 surviving bundles;
a bundle $=$ one variable's six-lag moving-average ladder plus its availability indicator).
Model: ridge $\alpha = 1$ on the square-root target with the RV-HAR ladder and intraday hour
dummies always included; every-bar refit. Primary window $tw = 48{,}000$ bars (1{,}000 trading
days): evaluation \textbf{Oct.\ 9, 2008 -- Apr.\ 30, 2024} ($n = 198{,}059$ bars). Replicate
window $tw = 144{,}000$ (3{,}000 days): evaluation \textbf{Apr.\ 25, 2016 -- Apr.\ 30, 2024}
($n = 102{,}059$ bars); its QLIKE level is not comparable across windows (different evaluation
spans) --- the ranking and DM are. DM ($t$-ratio, two-sided $p$; as in Table 1) is vs.\ the
HAR-baseline cell of the same window.
QLIKE levels here are not comparable to Table 1: this panel is the raw research build (no diurnal
adjustment or rank transform; overnight bars included), so the bundle gains include intraday
seasonality that Table 1's cache removes by construction --- within-table rankings and DM are the
deliverable. Exact factorial main effects on QLIKE, mean(bundle on) $-$ mean(bundle off) over
all $2^8$ cells ($tw48$k/$tw144$k): __MELINE__.
Largest first-order (two-way) interaction effects, same convention (positive $=$ partial
redundancy: the joint gain is less than additive): __INTLINE__.}
\vspace{4pt}
\scriptsize
\setlength{\tabcolsep}{2.5pt}
\begin{tabular}{@{}lccccccc@{}}
\toprule
& \multicolumn{5}{c}{$tw = 48{,}000$ (2008--2024)} & \multicolumn{2}{c}{$tw = 144{,}000$ (2016--2024)} \\
\cmidrule(lr){2-6}\cmidrule(l){7-8}
bundle combination & QLIKE & MAE & OOS-$R^2$ & MZ-$R^2$ & DM vs.\ HAR ($t$, $p$) & QLIKE & DM \\
\midrule
__T2__
\bottomrule
\end{tabular}
\end{table}

\begin{figure}[t]
\centering
\begin{tikzpicture}
\begin{axis}[
  ybar, bar width=4.5pt,
  width=\textwidth, height=7.2cm,
  symbolic x coords={__SYMX__},
  xtick=data, x tick label style={font=\footnotesize},
  enlarge x limits=0.08,
  ymajorgrids, grid style={gray!25},
  axis line style={gray!60}, tick style={gray!60},
  ylabel={improvement over HAR-only (\%; higher is better)},
  ylabel style={font=\small},
  legend style={at={(0.5,-0.16)}, anchor=north, legend columns=5, font=\footnotesize, draw=none},
  every axis plot/.append style={draw opacity=0},
]
__PLOTS__
\draw[black, thin] (rel axis cs:0,0.0) -- (rel axis cs:1,0.0);
\legend{QLIKE, HMSE, MAE, MSE, OOS-$R^2$}
\end{axis}
\end{tikzpicture}
\caption{Every metric of Table 1 mapped to one comparable scale: fractional improvement over
HAR-only, sign-oriented so that larger is better (the OOS-$R^2$ improvement is scaled by
$1-R^2_{\mathrm{HAR}}$). The level-robust metrics (QLIKE, HMSE, MAE) agree --- the full bucket
set wins decisively (DM $p = 1.2 \times 10^{-38}$) --- while raw-variance MSE and OOS-$R^2$
collapse toward zero or reverse sign, reflecting COVID-bar domination rather than forecast skill.}
\end{figure}

\end{document}
"""
tex = (tex.replace("__T1__", T1).replace("__T2__", T2).replace("__MELINE__", me_line)
       .replace("__INTLINE__", int_line)
       .replace("__SYMX__", SYMX).replace("__PLOTS__", PLOTS))
open(f"{SCR}/beat4.tex", "w", encoding="utf-8").write(tex)
print("wrote beat4.tex")
print("best tid:", best_tid, c48[best_tid]["active"])
