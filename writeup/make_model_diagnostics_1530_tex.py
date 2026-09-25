"""Write writeup/model_diagnostics_1530.tex from results/model_diagnostics_1530/ (run experiments/model_diagnostics_1530.py first)."""

import pandas as pd
from pathlib import Path

R = Path(__file__).resolve().parent.parent
OUT = R / "results" / "model_diagnostics_1530"
W = pd.read_csv(OUT / "weights_table.csv")
GC = pd.read_csv(OUT / "corr_group_contribution_vs_pnl.csv")
DD = pd.read_csv(OUT / "s_decomposition_corr.csv")
FC = pd.read_csv(OUT / "corr_feature_vs_pnl.csv")


def ci(r):
    return f"{r['corr']:+.3f} [{r['lo']:+.2f}, {r['hi']:+.2f}]"


def esc(s):
    return s.replace("&", "\\&").replace("_", "\\_").replace("%", "\\%")


# table 1: top features by how much they move the forecast
t1 = []
for _, r in W.head(12).iterrows():
    t1.append(
        f"{esc(r['plain'])} & {r['std_weight_mean']:+.3f} & {r['std_weight_sd']:.3f} & "
        f"{100 * r['frac_positive']:.0f}\\% & {r['contrib_sd']:.3f} \\\\"
    )

# table 2: group contributions vs P&L (mid, crossed) and decision
order = list(GC[(GC.split == "all") & (GC.target == "pnl_mid")]["series"])
keep = order[:8] + ["FOMC day flag", "forecast (sum)"]
t2 = []
for s in keep:
    cells = []
    for sp in ("all", "month_end", "other"):
        for tg in ("pnl_mid", "pnl_crossed"):
            r = GC[(GC.split == sp) & (GC.target == tg) & (GC.series == s)].iloc[0]
            star = "$^{*}$" if (r["lo"] > 0 or r["hi"] < 0) else ""
            cells.append(f"{r['corr']:+.2f}{star}")
    t2.append(f"{esc(s)} & " + " & ".join(cells) + " \\\\")

# table 3: the decision's two sides
t3 = []
for s in [
    "log forecast rv_hat",
    "log implied iv_var",
    "log(rv_hat / iv_var)",
    "log baseline (diurnal scale)",
    "log rv_hat - log baseline (the yhat part)",
    "log baseline - log iv_var",
]:
    cells = []
    for sp in ("all", "month_end", "other"):
        for tg in ("pnl_mid", "buy_decision"):
            r = DD[(DD.split == sp) & (DD.target == tg) & (DD.series == s)].iloc[0]
            star = "$^{*}$" if (r["lo"] > 0 or r["hi"] < 0) else ""
            cells.append(f"{r['corr']:+.2f}{star}")
    label = {
        "log forecast rv_hat": "log forecast $\\hat{rv}$",
        "log implied iv_var": "log implied variance",
        "log(rv_hat / iv_var)": "$\\log(\\hat{rv}/iv)$ (sign = decision)",
        "log baseline (diurnal scale)": "log diurnal scale of $\\hat{rv}$",
        "log rv_hat - log baseline (the yhat part)": "regression part of $\\log\\hat{rv}$",
        "log baseline - log iv_var": "log diurnal scale $-$ log implied",
    }[s]
    t3.append(f"{label} & " + " & ".join(cells) + " \\\\")

# counts of feature-level intervals excluding zero
cnt = []
for sp in ("all", "month_end", "other"):
    row = [sp.replace("_", "-")]
    for tg in ("pnl_mid", "pnl_crossed", "buy_decision"):
        s_ = FC[(FC.split == sp) & (FC.target == tg)]
        nw = int(((s_.lo_w > 0) | (s_.hi_w < 0)).sum())
        nc = int(((s_.lo_c > 0) | (s_.hi_c < 0)).sum())
        row.append(f"{nw} / {nc}")
    cnt.append(" & ".join(row) + " \\\\")


def regimes_tex() -> str:
    """Pages 5+: regime changes, rolling SHAP, the lasso's VIX path (experiments/model_diagnostics_1530_regimes.py)."""
    import json

    J = json.loads((OUT / "regimes.json").read_text(encoding="utf-8"))
    g = J["gates"]
    KB = pd.read_csv(OUT / "regimes_breaks_known_dates.csv")
    SB = pd.read_csv(OUT / "regimes_breaks_supwald.csv").set_index("reg")
    ST = pd.read_csv(OUT / "regimes_weight_steps_at_penalty_switches.csv")
    RT = pd.read_csv(OUT / "regimes_shap_share_by_regime.csv")
    VT = pd.read_csv(OUT / "regimes_lasso_vix_columns.csv")
    Dd = J["dates"]
    covid = f"{Dd['covid'][0]}..{Dd['covid'][1]}"

    def e(x):
        if x == 0:
            return "0"
        m, ex = f"{x:.1e}".split("e")
        return f"{m}\\times10^{{{int(ex)}}}"

    def pv(x):
        if pd.isna(x):
            return "--"
        s_ = f"{x:.3f}" if x >= 0.001 else "$<$0.001"
        return s_ + ("$^{*}$" if x < 0.05 else "")

    def kp(reg, date):
        return KB[(KB.reg == reg) & (KB.date == date)].iloc[0]

    def kv(reg, date):
        return kp(reg, date).pval

    dates = [c[1] for c in Dd["cand"]]
    heads = [
        c[0]
        .replace("penalty ", "pen. ")
        .replace("COVID crash ", "COVID ")
        .replace("every-day expirations", "every-day")
        for c in Dd["cand"]
    ]
    regs = [
        ("A0_actual", "target on forecast", "record"),
        ("A0_100.0", "", "100"),
        ("A0_1000.0", "", "1000"),
        ("A1_actual", "target on contributions", "record"),
        ("A1_100.0", "", "100"),
        ("A1_1000.0", "", "1000"),
        ("B0_mid", "mean return, mid", "record"),
        ("B0_crossed", "mean return, crossed", "record"),
        ("B1_actual", "return on contributions", "record"),
        ("B1_1000.0", "", "1000"),
    ]
    tb = []
    for rid, lab, pen in regs:
        sr = SB.loc[rid]
        cells = [pv(kv(rid, dt)) for dt in dates] + [pv(kv(rid, covid))]
        cells.append(f"{sr.sup_wald:.1f} ({sr.date_max}) {pv(sr.pval)}")
        tb.append(f"{lab} & {pen} & {int(sr.p)} & " + " & ".join(cells) + " \\\\")
        if rid in ("A0_1000.0", "A1_1000.0", "B0_crossed"):
            tb.append("\\midrule")

    tsw = []
    for dt in Dd["switches"]:
        r = ST[(ST.date == dt) & (ST.column == "ALL")].iloc[0]
        tsw.append(
            f"{dt} & {r.switch} & {r.step:.4f} & {r.penalty_part:.4f} & {r.data_part:.4f} & "
            f"{r.max_abs_step_other_days:.4f} & {r.median_abs_step_other_days:.5f} \\\\"
        )

    tr = []
    for _, r in RT.iterrows():
        tr.append(
            f"{r.start} .. {r.end} & {r.sessions} & {r.deck_days} & {r.penalty} & "
            + " & ".join(
                f"\\texttt{{{esc(r[f'top{i}'])}}} {100 * r[f'share{i}']:.0f}\\%"
                for i in (1, 2, 3)
            )
            + f" & \\texttt{{{esc(r['held1000_top2'])}}} / \\texttt{{{esc(r['held1000_top3'])}}} \\\\"
        )

    tv = []
    VV = VT[(VT.sessions_nonzero > 0) | (VT.held_sessions_nonzero_before_switch > 0)]
    for _, r in VV.iterrows():
        sgn = (
            "--"
            if pd.isna(r.frac_positive_when_nonzero)
            else f"{100 * r.frac_positive_when_nonzero:.0f}\\%"
        )
        mw = (
            "--"
            if pd.isna(r.mean_std_weight_when_nonzero)
            else f"{r.mean_std_weight_when_nonzero:+.4f}"
        )
        hw = (
            "--"
            if pd.isna(r.held_mean_std_weight_before_switch)
            else f"{r.held_mean_std_weight_before_switch:+.4f}"
        )
        fn = r.first_nonzero if isinstance(r.first_nonzero, str) else "--"
        tv.append(
            f"\\texttt{{{esc(r.column)}}} & {r.sessions_nonzero} & {r.deck_days_nonzero} & {sgn} & {mw} & {fn} & "
            f"{r.n_events} & {r.held_sessions_nonzero_before_switch} & {hw} \\\\"
        )

    kr = {rid: SB.loc[rid] for rid in SB.index}
    sw = Dd["switches"]
    ev = Dd["everyday"]
    st_all = ST[ST.column == "ALL"].set_index("date")
    la = J["lasso"]
    vx1 = VT.set_index("column")
    vr = RT[(RT.penalty.astype(str) == "100") & (RT.deck_days > 0)].iloc[0]
    covid_row = RT[RT.start == Dd["covid"][0]].iloc[0]
    dom63, dom126 = J["dom"]["63"], J["dom"]["126"]
    a0_held_min = min(
        kv(k, d_) for k in ("A0_100.0", "A0_1000.0") for d_ in (sw[2], ev)
    )

    return (
        r"""
\clearpage
\begin{center}{\large\bf Regime changes, which inputs dominate, and how the lasso picks out the VIX}\\[2pt]
{\small same forecast, same 16:00 bar; """
        + f"{J['n']:,}"
        + r""" refits (2018-06-25 .. 2024-04-30), of which """
        + str(J["deck_n"])
        + r""" are deck days}\end{center}

\textbf{1. Regime changes.} \emph{Question:} do the weights, and the trade's money, change at the known dates -- the penalty switches ("""
        + ", ".join(sw)
        + r"""), the Feb--Apr 2020 crash ("""
        + covid.replace("..", " .. ")
        + r""", from the session after the S\&P high on the deck, """
        + Dd["peak"]
        + r"""), the start of every-day expirations -- and is a change the penalty or the data?

\emph{Gate.} Two more re-runs of the same arm with the penalty held at one value: held at 100 it equals the arm of record on all """
        + f"{g['fixed100']['n_same']}"
        + r""" refits where the record used 100 (to $"""
        + e(g["fixed100"]["maxrel"])
        + r"""$), held at 1000 on all """
        + f"{g['fixed1000']['n_same']}"
        + r""" refits at 1000 (to $"""
        + e(g["fixed1000"]["maxrel"])
        + r"""$). The arm of record is the two held-penalty runs spliced at the switch dates, so a change that neither held run shows is the penalty.
\emph{Every-day expirations.} The first Tuesday/Thursday deck day is """
        + Dd["first_tt"]
        + r""" (a holiday-shifted expiry); every Tuesday session is a deck day from """
        + ev
        + r""" and every Thursday from """
        + Dd["thu_start"]
        + r"""; the tests use """
        + ev
        + r""".
\emph{Tests.} Known date: Chow test in Wald form with Newey--West covariance (lag $\lfloor 4(n/100)^{2/9}\rfloor$ = """
        + f"{J['hac']['refits']}"
        + r""" on refits, """
        + f"{J['hac']['deck']}"
        + r""" on deck days), $\chi^2(p)$, run only where each side holds at least 15\% of the sample; the Feb--Apr 2020 episode ("""
        + f"{int(kp('A0_actual', covid).n_after)}"
        + r""" refits, """
        + f"{int(kp('B0_mid', covid).n_after)}"
        + r""" deck days) by Chow's predictive $F$ test, the standard test for a stretch too short to fit on its own. Unknown date: Andrews' sup-Wald over the middle 70\% of the sample, critical values simulated from its limit (20,000 draws; $p=1$, 5\%: """
        + f"{J['crit95']['1']:.2f}"
        + r""", published 8.58--8.85). Regressions: the 16:00 target on the forecast (calibration, $p=2$); the target on the contributions of the six largest series plus the rest summed ($p=8$); the day's return per premium on a constant ($p=1$), and on the same contributions ($p=8$).

\begin{figure}[h]\centering
\includegraphics[width=\textwidth]{../results/model_diagnostics_1530/regimes_weights_counterfactual.png}
\caption{Standardized weights of the eight largest inputs (design column names, key under Figure 1): the arm of record (dotted black) and the same arm with the penalty held at 100 (green) or 1000 (orange). Red dashed = penalty switches; blue dotted = COVID start/end and every-day expirations; blue band = Feb--Apr 2020; grey = deck days.}
\end{figure}

\begin{table}[h]\centering\small
\caption{The step in all standardized weights at each penalty switch (Euclidean norm across inputs), split into the penalty (the two held runs on the same day) and one more day of data (the old penalty's run, day over day).}
\begin{tabular}{llrrrrr}\toprule
date & penalty & step & penalty part & data part & largest step, other days & median \\\midrule
"""
        + "\n".join(tsw)
        + r"""
\bottomrule\end{tabular}\end{table}

\begin{table}[h]\centering\scriptsize
\caption{Break tests: $p$-values at the known dates and for the COVID episode, and the unknown-date sup-Wald (statistic, date of its maximum, $p$). record = the arm of record; 100 / 1000 = penalty held. -- = not testable (before the deck, or under 15\% of the deck days on one side). $^{*}$ = $p<0.05$; with seven dates per row, a stray $^{*}$ is expected by chance.}
\setlength{\tabcolsep}{2.5pt}
\resizebox{\textwidth}{!}{\begin{tabular}{llr"""
        + "r" * (len(dates) + 1)
        + r"""l}\toprule
regression & pen. & $p$ & """
        + " & ".join(heads)
        + r""" & COVID & sup-Wald \\
 & & & """
        + " & ".join(dates)
        + r""" & episode & \\\midrule
"""
        + "\n".join(tb)
        + r"""
\bottomrule\end{tabular}}\end{table}

\begin{figure}[h]\centering
\includegraphics[width=\textwidth]{../results/model_diagnostics_1530/regimes_supwald_paths.png}
\caption{Wald statistic for a break at each date (the sup-Wald scan) with its 5\% critical value. Vertical lines as in Figure 4.}
\end{figure}

\textbf{Reading.}
\begin{itemize}\itemsep1pt
\item \emph{Every step in the weights is the penalty.} At the three switches all weights together jump by """
        + ", ".join(f"{st_all.loc[d_, 'step']:.3f}" for d_ in sw)
        + r""", of which the penalty is """
        + ", ".join(f"{st_all.loc[d_, 'penalty_part']:.3f}" for d_ in sw)
        + r"""; on every other refit the largest step is """
        + f"{st_all['max_abs_step_other_days'].iloc[0]:.3f}"
        + r""" and the median """
        + f"{st_all['median_abs_step_other_days'].iloc[0]:.4f}"
        + r""". Held at one penalty the weights drift smoothly through COVID and through every-day expirations (Figure 4).
\item \emph{The forecast's calibration does not break.} Target on forecast: sup-Wald """
        + f"{kr['A0_actual'].sup_wald:.2f}"
        + r""" ($p$ """
        + f"{kr['A0_actual'].pval:.2f}"
        + r"""); held at 100 / 1000: """
        + f"{kr['A0_100.0'].sup_wald:.2f} / {kr['A0_1000.0'].sup_wald:.2f}"
        + r""". The two known-date rejections in the arm of record ("""
        + f"{sw[2]}, $p$ {kv('A0_actual', sw[2]):.3f}; {ev}, $p$ {kv('A0_actual', ev):.3f}"
        + r""") vanish with the penalty held ($p\ge$ """
        + f"{a0_held_min:.2f}"
        + r"""): made by the penalty. The Feb--Apr 2020 episode rejects under every penalty ($p<0.001$); the predictive test reacts to the crash's larger misses as well as to a changed slope.
\item \emph{How the series' contributions map into realized variance does change, and not because of the penalty.} The sup-Wald rejects under each penalty ("""
        + f"{kr['A1_actual'].sup_wald:.1f}, {kr['A1_100.0'].sup_wald:.1f}, {kr['A1_1000.0'].sup_wald:.1f}; $p\\le$ {max(kr['A1_actual'].pval, kr['A1_100.0'].pval, kr['A1_1000.0'].pval):.3f}"
        + r"""), its maximum at """
        + f"{kr['A1_actual'].date_max}"
        + r""" (record, held 1000) or """
        + f"{kr['A1_100.0'].date_max}"
        + r""" (held 100): one date is not pinned down. The COVID episode differs under every penalty. At the known dates the answer depends on which penalty is held (e.g. """
        + f"{ev}: record {kv('A1_actual', ev):.3f}, held 100 {kv('A1_100.0', ev):.2f}, held 1000 {kv('A1_1000.0', ev):.3f}"
        + r"""), so none is a clean data break.
\item \emph{The trade's money has no break in level.} Mean return per premium: sup-Wald """
        + f"{kr['B0_mid'].sup_wald:.2f}"
        + r""" mid / """
        + f"{kr['B0_crossed'].sup_wald:.2f}"
        + r""" crossed ($p$ """
        + f"{kr['B0_mid'].pval:.2f} / {kr['B0_crossed'].pval:.2f}"
        + r"""); the COVID days' mean is not different ($p$ """
        + f"{kv('B0_mid', covid):.2f} / {kv('B0_crossed', covid):.2f}"
        + r"""). The return's slopes on the contributions do shift (sup-Wald """
        + f"{kr['B1_actual'].sup_wald:.1f}, $p$ {kr['B1_actual'].pval:.3f}, at {kr['B1_actual'].date_max}; held 1000: {kr['B1_1000.0'].sup_wald:.1f}, $p$ {kr['B1_1000.0'].pval:.3f}"
        + r""") -- the data, not the penalty -- but these are eight slopes on heavy-tailed returns that are near zero on either side (Table 2).
\item \emph{Not testable on the money:} """
        + sw[0]
        + r""" is before the deck; the COVID start/end and """
        + sw[1]
        + r""" leave under 15\% of the deck days on one side.
\end{itemize}

\clearpage
\textbf{2. Which series dominate (rolling SHAP).} \emph{Question:} which input series carry the forecast, and does that change across regimes?

\emph{Why the contributions are the SHAP values.} For a linear forecast $\hat y=\beta_0+\sum_j\beta_j x_j$ with the window mean $\bar x$ as the baseline (interventional SHAP), the SHAP value of input $j$ is exactly $\beta_j(x_j-\bar x_j)$, and the values sum to $\hat y$ minus the forecast at $\bar x$. \emph{Gate.} The \texttt{shap} library's LinearExplainer (the window mean as its one background row) on """
        + f"{J['shap']['n']}"
        + r""" refits (the first, the last, and each candidate date) equals $\beta_j(x_j-\bar x_j)$ to $"""
        + e(J["shap"]["dmax"])
        + r"""$, and its values plus the baseline equal the forecast to $"""
        + e(J["shap"]["smax"])
        + r"""$ (relative). A series' SHAP value is the sum over its columns (all lags and flags). Share = the series' mean $|$SHAP$|$ over a trailing window divided by the sum over series. The window is a choice: 63 sessions (a quarter), checked against 126 (a half-year).

\begin{figure}[h]\centering
\includegraphics[width=\textwidth]{../results/model_diagnostics_1530/regimes_shap_share.png}
\caption{Share of mean $|$SHAP$|$ by input series (design column names, key under Figure 1), trailing 63 sessions (top) and 126 sessions (bottom). Dashed = penalty switches; dotted = COVID start/end and every-day expirations.}
\end{figure}

\begin{table}[h]\centering\scriptsize
\caption{The three largest series by share of mean $|$SHAP$|$ within each regime (cut at the candidate dates; no rolling), and the second and third with the penalty held at 1000.}
\resizebox{\textwidth}{!}{\begin{tabular}{lrrlllll}\toprule
regime & refits & deck days & penalty & first & second & third & held 1000: second / third \\\midrule
"""
        + "\n".join(tr)
        + r"""
\bottomrule\end{tabular}}\end{table}

\textbf{Reading.}
\begin{itemize}\itemsep1pt
\item \emph{Realized variance dominates almost everywhere.} \texttt{har\_ma\_*} has the largest share on """
        + f"{100 * dom63['har_ma_*'] / sum(dom63.values()):.0f}"
        + r"""\% of refits (63-session window) and """
        + f"{100 * dom126['har_ma_*'] / sum(dom126.values()):.0f}"
        + r"""\% (126); absolute return (\texttt{adj\_sumabsret\_ma\_*}) leads on the rest. The two windows name the same leader on """
        + f"{100 * J['agree']:.1f}"
        + r"""\% of """
        + f"{J['agree_n']:,}"
        + r""" refits.
\item \emph{COVID.} In Feb--Apr 2020 absolute return leads ("""
        + f"{100 * covid_row.share1:.0f}"
        + r"""\%) ahead of realized variance ("""
        + f"{100 * covid_row.share2:.0f}"
        + r"""\%) and 4th-power returns ("""
        + f"{100 * covid_row.share3:.0f}"
        + r"""\%) -- the only regime where realized variance is not first.
\item \emph{The penalty-100 year} ("""
        + f"{vr.start} .. {vr.end}"
        + r""") puts VIX3M and VIX second and third ("""
        + f"{100 * vr.share2:.0f}\\%, {100 * vr.share3:.0f}\\%"
        + r"""); with the penalty held at 1000 the same year ranks absolute and signed returns second and third, like every other regime: the VIX's rise there is the penalty.
\item \emph{Every-day expirations} (from """
        + ev
        + r""") leave the ranking unchanged.
\end{itemize}

\clearpage
\textbf{3. The lasso: how the VIX is picked out.} \emph{Question:} when the same forecast is fit by the spec's recursive lasso instead of the ridge, which VIX columns does it keep, with what sign and size, and do they enter or leave at the regime dates?

\emph{Gate.} The lasso re-run (same bucket, bar, 2000-session window) equals the stored lasso forecast on all """
        + f"{g['lasso']['common']:,}"
        + r""" rows of the 16:00 bar (max relative difference $"""
        + e(g["lasso"]["maxrel"])
        + r"""$); coefficients $\cdot$ inputs $+$ intercept reproduce it to $"""
        + e(g["lasso"]["recon"])
        + r"""$. Its penalty (grid $10^{-6}$ .. $10^{-2}$, re-chosen every 250 refits) is """
        + f"{la['alpha0']:g}"
        + r""" (the top of the grid) until """
        + la["switches"][0][0]
        + r""" and """
        + f"{la['switches'][0][1]:g}"
        + r""" after. A second re-run with the penalty held at """
        + f"{la['held']['alpha']:g}"
        + r""" equals the lasso of record on its """
        + f"{g['lasso_fixed']['n_same']}"
        + r""" refits at that penalty (to $"""
        + e(g["lasso_fixed"]["maxrel"])
        + r"""$). The lasso keeps """
        + f"{la['nnz'][0]} to {la['nnz'][2]}"
        + r""" of the 244 columns (median """
        + f"{la['nnz'][1]}"
        + r"""); its forecast correlates """
        + f"{la['corr_ridge']:.3f}"
        + r""" with the ridge's.

\begin{figure}[h]\centering
\includegraphics[width=\textwidth]{../results/model_diagnostics_1530/regimes_lasso_vix.png}
\caption{The lasso's standardized weights on the VIX-family columns (design column names; colour = lag as in Figure 1): solid = lasso of record, dashed = penalty held at """
        + f"{la['held']['alpha']:g}"
        + r""" from the start. Thick red line = the lasso's penalty re-choice; other lines as in Figure 4. Only columns the lasso of record ever keeps are drawn.}
\end{figure}

\begin{table}[h]\centering\scriptsize
\caption{VIX-family columns in the lasso: refits (of """
        + f"{J['n']:,}"
        + r""") and deck days with a nonzero weight, share positive, mean standardized weight when nonzero, first nonzero refit, number of entries and exits; with the penalty held at """
        + f"{la['held']['alpha']:g}"
        + r""": refits nonzero before """
        + la["switches"][0][0]
        + r""" (of """
        + f"{la['held']['n_before']}"
        + r""") and the mean weight there.}
\resizebox{\textwidth}{!}{\begin{tabular}{lrrrrlrrr}\toprule
column & refits & deck days & positive & mean std.\ weight & first & entries/exits & held: refits before & held: mean \\\midrule
"""
        + "\n".join(tv)
        + r"""
\bottomrule\end{tabular}}\end{table}

\enlargethispage{3\baselineskip}
\textbf{Reading.}
\begin{itemize}\itemsep1pt
\item \emph{The VIX enters at the penalty re-choice, not at an event.} Before """
        + la["switches"][0][0]
        + r""" no VIX-family column in the lasso of record exceeds """
        + f"{la['pre_switch_max_abs_vix_std_weight']:.4f}"
        + r""" in standardized weight. On """
        + la["switches"][0][0]
        + r""", when the penalty drops to """
        + f"{la['switches'][0][1]:g}"
        + r""", \texttt{adj\_vix\_ma\_1} enters and stays on all """
        + f"{vx1.loc['adj_vix_ma_1', 'sessions_nonzero']}"
        + r""" later refits, always positive (mean """
        + f"{vx1.loc['adj_vix_ma_1', 'mean_std_weight_when_nonzero']:+.3f}"
        + r"""), with the longer VIX3M means negative (\texttt{adj\_vix3m\_ma\_625} """
        + f"{vx1.loc['adj_vix3m_ma_625', 'mean_std_weight_when_nonzero']:+.3f}"
        + r""", \texttt{adj\_vix3m\_ma\_125} """
        + f"{vx1.loc['adj_vix3m_ma_125', 'mean_std_weight_when_nonzero']:+.3f}"
        + r"""): the VIX's last bar against the 3-month index's longer averages.
\item \emph{The VIX was in the data before.} Held at """
        + f"{la['held']['alpha']:g}"
        + r""" from the start, the lasso keeps \texttt{adj\_vix\_ma\_1} on """
        + f"{vx1.loc['adj_vix_ma_1', 'held_sessions_nonzero_before_switch']}"
        + r""" of the """
        + f"{la['held']['n_before']}"
        + r""" earlier refits (mean """
        + f"{vx1.loc['adj_vix_ma_1', 'held_mean_std_weight_before_switch']:+.3f}"
        + r""") and \texttt{adj\_vix3m\_ma\_625} negative on """
        + f"{vx1.loc['adj_vix3m_ma_625', 'held_sessions_nonzero_before_switch']}"
        + r""", through COVID. The top-of-grid penalty kept the VIX out until 2021.
\item \emph{Entries and exits in between are small.} Of the """
        + f"{sum(la['events_by_cause'].values())}"
        + r""" VIX-family entries and exits, """
        + f"{la['events_by_cause'].get('penalty re-chosen', 0)}"
        + r""" are at the penalty re-choice and """
        + f"{la['events_by_cause'].get('warm update (data)', 0)}"
        + r""" in the day-to-day path; after a day-to-day entry no weight exceeds """
        + f"{la['warm_max_abs_std_weight_after']:.4f}"
        + r""". None marks COVID or every-day expirations.
\item \emph{The lasso leans on the VIX more than the ridge.} Share of mean $|$SHAP$|$ on deck days: VIX """
        + f"{100 * la['share']['adj_vix_ma_*']:.1f}"
        + r"""\% and VIX3M """
        + f"{100 * la['share']['adj_vix3m_ma_*']:.1f}"
        + r"""\% (ridge """
        + f"{100 * J['ridge_share_deck']['adj_vix_ma_*']:.1f}\\% and {100 * J['ridge_share_deck']['adj_vix3m_ma_*']:.1f}"
        + r"""\%); realized variance """
        + f"{100 * la['share']['har_ma_*']:.1f}"
        + r"""\% (ridge """
        + f"{100 * J['ridge_share_deck']['har_ma_*']:.1f}"
        + r"""\%).
\end{itemize}
Files: \texttt{experiments/model\_diagnostics\_1530\_regimes.py}; outputs \texttt{results/model\_diagnostics\_1530/regimes\_*}.
"""
    )


summary = (OUT / "summary.txt").read_text(encoding="utf-8")
print(summary)

tex = (
    r"""\documentclass[10pt]{article}
\usepackage[margin=0.8in]{geometry}
\usepackage{graphicx,booktabs,amsmath}
\usepackage[font=small]{caption}
\setlength{\parskip}{3pt}\setlength{\parindent}{0pt}
\begin{document}
\begin{center}{\large\bf The 15:30 forecast's weights, and when sign(s) makes money}\\[2pt]
{\small 866 deck days, 2020-01-03 .. 2024-04-30; forecast of record: per-bar ridge, live-feasible features, 2000-session window, bar ending 16:00}\end{center}

\textbf{Question.} Which inputs move the 15:30 variance forecast, how stable are their weights, and do the weights or the inputs' contributions line up with the days on which sign(s) (buy the nearest-OTM 0DTE package if $s=\hat{rv}-iv>0$, sell otherwise, settle at the close) makes money?

\textbf{Verification (before any reading).} The fitted coefficients are not stored, so the forecast was re-run with the research code and the coefficient vector recorded at every refit.
(1) The re-run equals the stored forecast on all 1,469 rows of the 16:00 bar (max relative difference $7.3\times10^{-12}$).
(2) Coefficients $\cdot$ inputs $+$ intercept reproduce every forecast to $2.1\times10^{-15}$, and forecast $=$ level $+\sum_j$ contribution$_j$ to $1.2\times10^{-15}$, where contribution$_j=\beta_j(x_j-\bar x_j)$ over the trailing window and the level is the window-mean target (to $4.9\times10^{-13}$).
(3) All 866 deck days join (bar-end 16:00 ET row $=$ forecast issued at 15:30); 52 are month-end sessions.
Standardized weight $=\beta_j\times$ trailing-window sd of $x_j$. Intervals: circular block bootstrap, block $\lceil n^{1/3}\rceil$, 2,000 draws; $^{*}$ = 95\% interval excludes 0.

\textbf{Money on this forecast.} Summed return per premium: mid $+102.98$ (month-ends $-10.67$, other days $+113.65$); crossed (buy at the ask, sell at the bid) $+74.15$ ($-12.94$ / $+87.09$). It buys 348 of 866 days and only 10 of the 52 month-ends.

\begin{figure}[h]\centering
\includegraphics[width=\textwidth]{../results/model_diagnostics_1530/weights_small_multiples.png}
\caption{Standardized weights over every refit (1,469 sessions), one panel per input series; solid = the series' rolling means (colour = window length in 30-minute bars). Panel titles are the design's column names; \texttt{*\_ma\_$k$} is the mean over the last $k$ 30-minute bars (\texttt{\_ma\_1} = the last bar), \texttt{adj\_} marks the series after its transform (e.g. log for the VIX) and diurnal adjustment, and the dotted lines are the \texttt{*\_avail\_ma\_$k$} / \texttt{*\_active\_ma\_$k$} is-present / is-nonzero flags. Key: \texttt{har\_ma} realized variance (the target); \texttt{sumret} signed return; \texttt{sumabsret} absolute return; \texttt{sumret3} cubed returns (skew); \texttt{sumret4} 4th-power returns (tails); \texttt{sumpret2} upside squared returns; \texttt{sumbipow} bipower variation; \texttt{sumautocov} return autocovariance; \texttt{sumvolume} ES volume (contracts); \texttt{numobs} ES trade count per bar; \texttt{vix}, \texttt{vvix}, \texttt{vix3m} the Cboe indices; \texttt{fomc\_release} FOMC statement bar; \texttt{fomc\_day} FOMC day; \texttt{fomc\_until\_inv} / \texttt{fomc\_since\_inv} 1 / days to the next / since the last FOMC. Grey band = the 866 deck days. 138 of the 244 design columns carry a nonzero weight on deck days; the rest are constant or duplicated within the window and are masked to zero.}
\end{figure}

\begin{table}[h]\centering\small
\caption{The 12 inputs that move the forecast most on the deck days (by sd of their contribution; forecast sd 0.357 in the same units).}
\begin{tabular}{lrrrr}\toprule
input & mean std.\ weight & sd & days $>0$ & contribution sd\\\midrule
"""
    + "\n".join(t1)
    + r"""
\bottomrule\end{tabular}\end{table}

\begin{figure}[h]\centering
\includegraphics[width=\textwidth]{../results/model_diagnostics_1530/corr_contribution_vs_pnl.png}
\caption{Correlation across deck days of each input series' summed contribution (rows named by the design's columns, key under Figure 1; \texttt{ybar} = the level, \texttt{yhat} = the forecast) to the forecast with the day's return per premium (blue mid, red crossed), 95\% block-bootstrap intervals.}
\end{figure}

\begin{table}[h]\centering\small
\caption{Series contribution vs the day's return per premium (the top 8 series by contribution sd, the FOMC-day flag, and the forecast itself).}
\begin{tabular}{lrrrrrr}\toprule
 & \multicolumn{2}{c}{all (866)} & \multicolumn{2}{c}{month-ends (52)} & \multicolumn{2}{c}{other (814)}\\
series & mid & crossed & mid & crossed & mid & crossed\\\midrule
"""
    + "\n".join(t2)
    + r"""
\bottomrule\end{tabular}

\vspace{4pt}
Feature level (138 inputs): intervals excluding 0, standardized weight / contribution (chance $\approx$ 7 of 138)\\[2pt]
\begin{tabular}{lccc}\toprule
split & mid P\&L & crossed P\&L & buy decision\\\midrule
"""
    + "\n".join(cnt)
    + r"""
\bottomrule\end{tabular}\end{table}

\begin{figure}[h]\centering
\includegraphics[width=\textwidth]{../results/model_diagnostics_1530/pnl_by_contribution_quintile.png}
\caption{Mean mid return per premium by quintile of each of the five largest series' contributions (named by the design's columns, key under Figure 1) (dark: all 866 days; light: without month-ends).}
\end{figure}

\begin{table}[h]\centering\small
\caption{What drives the decision: correlation of the two sides of $s$ with the day's mid return and with the buy decision ($+1$ buy, $-1$ sell). $\log\hat{rv}$ = log diurnal scale (not a regression output) $+$ the regression part.}
\begin{tabular}{lrrrrrr}\toprule
 & \multicolumn{2}{c}{all} & \multicolumn{2}{c}{month-ends} & \multicolumn{2}{c}{other}\\
 & mid & decision & mid & decision & mid & decision\\\midrule
"""
    + "\n".join(t3)
    + r"""
\bottomrule\end{tabular}\end{table}

\textbf{Reading.}
\begin{itemize}\itemsep1pt
\item \emph{The weights keep their sign and move in steps.} The penalty is re-chosen every 250 refits (1000, the top of the grid, on 712 of 866 deck days; 100 on the rest), and the inputs that move the forecast most keep one sign on every deck day: realized variance in the last bar ($+0.068$, sd 0.018), its 5- and 25-bar means ($+0.048$, $+0.047$), absolute return in the last bar ($+0.043$), the 5-bar signed return ($-0.038$). The forecast is a HAR on realized variance and absolute returns (series contribution sd 0.170 and 0.111), trimmed by long-window tail, bipower and upside terms (negative weights), with VIX adding and VIX3M subtracting (0.038 / 0.037).
\item \emph{Weights do not track money.} Across 138 inputs, 1 standardized-weight interval excludes zero for the mid return on all days (0 crossed; 0 on non-month-end days) -- below the $\approx$7 expected by chance. They do co-move with the buy/sell mix (37 of 138 exclude zero for the decision): both drift slowly over the sample, which is not a day-level link to the return.
\item \emph{Contributions barely track money.} 14 (mid) and 16 (crossed) of 138 contribution intervals exclude zero on all days, against $\approx$7 by chance; on non-month-end days 5 and 7, i.e.\ chance. At the series level only absolute return is outside zero on all days ($-0.049$ mid, $-0.058$ crossed): days on which recent absolute returns push the forecast up earn less. The forecast itself: $-0.014$ [$-0.07$, $+0.04$]. No quintile pattern is monotone (Figure 3).
\item \emph{The decision is set by the implied side and the diurnal scale, not by the weights.} $\log(\hat{rv}/iv)$ has variance 0.101 while each side has about 1 (correlation 0.952); of $\log\hat{rv}$'s variance 0.967, the diurnal scale carries 0.813 and the regression part 0.365 (covariance $-0.211$). The regression part correlates $+0.04$ with the decision (interval through zero); log implied variance $-0.25$ on non-month-end days. Of the 330 day-to-day decision changes, the forecast moved more than the implied on 134.
\item \emph{Month-ends.} This forecast sells 42 of the 52 month-ends and loses there. On month-ends the FOMC-day flag's contribution correlates $+0.20$ with the day's return (interval $+0.09$ .. $+0.35$); on the 10 month-end buys (4 winners, 6 losers) the winners had lower absolute-return and ES-volume contributions and higher tail contributions. Both rest on a handful of days.
\end{itemize}
Files: \texttt{experiments/model\_diagnostics\_1530.py} (capture, then analyze); outputs in \texttt{results/model\_diagnostics\_1530/}.
"""
    + regimes_tex()
    + r"""
\end{document}
"""
)
(R / "writeup" / "model_diagnostics_1530.tex").write_text(tex, encoding="utf-8")
for _blk in (t1, t2, t3, cnt):
    print("\n".join(_blk))
