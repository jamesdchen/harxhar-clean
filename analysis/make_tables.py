"""Emit LaTeX tables from the inference JSONs into writeup/stats/."""

from __future__ import annotations

import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATS = os.path.join(ROOT, "writeup", "stats")


def fmt_p(p: float) -> str:
    if p >= 0.001:
        return f"{p:.3f}"
    m, e = f"{p:.1e}".split("e")
    return f"${m}\\times 10^{{{int(e)}}}$"


def main():
    bat = json.load(open(os.path.join(STATS, "battery_inference.json")))
    tile = json.load(open(os.path.join(STATS, "tile_inference.json")))

    # ---- Table 1: pairwise bounds, tree vs linear ----
    rows = bat["pairwise_bounds_tree_vs_linear"]
    lines = [
        r"\begin{table}[H]", r"\centering",
        r"\caption{Pairwise model-class inference on the frozen battery, from the",
        r"reported HAC standard errors alone. For arms sharing an incumbent the",
        r"pairwise differential is exactly $L_A-L_B$, whose standard error is bounded",
        r"above by $se_A+se_B$ regardless of the correlation between the two arms'",
        r"differentials; the resulting $|t|$ is therefore a \emph{lower bound} and the",
        r"$p$ an \emph{upper bound}. A tighter bound assuming the two arms'",
        r"improvements co-move ($\rho\ge 0$) is reported in the text; it does not",
        r"change any verdict. Holm adjustment is applied across the nine matched",
        r"comparisons. Negative $t$ favours the tree.}",
        r"\label{tab:pairwise_bounds}",
        r"\begin{tabular}{lrrrrl}", r"\toprule",
        r"Feature set & $\Delta$QLIKE & $|t|\ge$ (free) & $p\le$ (free) "
        r"& $p_{\text{Holm}}\le$ & Verdict \\", r"\midrule",
    ]
    for r in rows:
        lines.append(
            f"\\texttt{{{r['bucket'].replace('_', chr(92) + '_')}}} & "
            f"{r['mean_diff']:+.5f} & {abs(r['t_lower_bound_free']):.2f} & "
            f"{fmt_p(r['p_upper_bound_free'])} & {fmt_p(r['p_holm_upper_bound'])} & "
            f"{'tree better' if r['reject_holm'] else 'unresolved'} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    open(os.path.join(STATS, "tab_pairwise_bounds.tex"), "w").write("\n".join(lines) + "\n")

    # ---- Table 2: tile DM contrasts ----
    cs = sorted(tile["contrasts"], key=lambda c: c["p"])
    lines = [
        r"\begin{table}[H]", r"\centering",
        r"\caption{Per-bar Diebold--Mariano tests on the campaign tile "
        f"($n = {tile['panel']['n']:,}$ bars, {tile['panel']['n_arms']} arms, identical "
        r"targets). HAC (Newey--West) standard errors with the",
        r"Harvey--Leybourne--Newbold small-sample correction; Holm adjustment across",
        r"the whole contrast family. Negative $t$ favours arm A. \textbf{Panel note:}",
        r"this is a campaign tile, not the 218{,}934-bar frozen battery; no level here",
        r"is comparable to a battery level, and what the tile tests is whether each",
        r"claim survives per-bar inference at all.}",
        r"\label{tab:tile_dm}",
        r"\footnotesize", r"\begin{tabular}{lrrrl}", r"\toprule",
        r"Contrast (A vs B) & $\Delta$QLIKE & $t$ & $p_{\text{Holm}}$ & Verdict \\",
        r"\midrule",
    ]
    for c in cs:
        v = ("A better" if c["t"] < 0 else "B better") if c["reject_holm"] else "unresolved"
        lab = c["label"].replace("&", r"\&").replace("_", r"\_")
        lines.append(f"{lab} & {c['delta']:+.5f} & {c['t']:.2f} & "
                     f"{fmt_p(c['p_holm'])} & {v} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    open(os.path.join(STATS, "tab_tile_dm.tex"), "w").write("\n".join(lines) + "\n")

    # ---- Table 3: MCS by family ----
    fams = [("penalty", "Penalty: shrinkage vs selection"),
            ("trees_vs_linear", "Model class on the kernel dictionary"),
            ("local_k", "Analog neighbourhood size"),
            ("cap_ladder", "Kernel truncation depth"),
            ("dim_reduction", "Dimension reduction"),
            ("top20", "The twenty best arms")]
    lines = [
        r"\begin{table}[H]", r"\centering",
        r"\caption{Hansen--Lunde--Nason model confidence sets on the campaign tile "
        f"($\\alpha = {tile['settings']['alpha_mcs']}$, "
        f"$B = {tile['settings']['B']:,}$ stationary-bootstrap replications, "
        r"$T_{\max}$ statistic). ``Kept'' counts arms that cannot be excluded from",
        r"the set of best models. Survivor counts for the top-twenty family are",
        r"unchanged when the bootstrap block length is varied from half to four times",
        r"its automatic value.}",
        r"\label{tab:mcs}",
        r"\begin{tabular}{llrrl}", r"\toprule",
        r"Family & Question & Arms & Kept & Eliminated \\", r"\midrule",
    ]
    for key, desc in fams:
        m = tile["mcs"].get(key)
        if not m:
            continue
        elim = ", ".join(f"\\texttt{{{e[0].replace('_', chr(92) + '_')}}}"
                         for e in m["eliminated"]) or "---"
        n_arms = len(m["surviving"]) + len(m["eliminated"])
        lines.append(f"\\texttt{{{key.replace('_', chr(92) + '_')}}} & {desc} & "
                     f"{n_arms} & {len(m['surviving'])} & {elim} \\\\")
    allm = tile["mcs"]["all_arms"]
    lines.append(f"\\texttt{{all}} & Every tile arm & "
                 f"{len(allm['surviving']) + len(allm['eliminated'])} & "
                 f"{len(allm['surviving'])} & (see text) \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    open(os.path.join(STATS, "tab_mcs.tex"), "w").write("\n".join(lines) + "\n")

    # ---- Table 4: Hawkes kernel diagnostics ----
    hk = json.load(open(os.path.join(STATS, "hawkes_kernel.json")))
    lines = [
        r"\begin{table}[H]", r"\centering",
        r"\caption{Kernel diagnostics by session. $\hat n = \sum_k w_k$ is the",
        r"discretised kernel integral of Equation~\eqref{eq:branching_identity}, the",
        r"branching-ratio analogue; a linear Hawkes process requires $\hat n \in [0,1)$.",
        r"``Neg.\ lags'' counts lags at which the effective kernel is negative, which a",
        r"linear Hawkes kernel cannot be, and ``neg.\ mass'' is their share of total",
        r"absolute kernel mass. $\hat\beta$ is the log-log slope over the positive part,",
        r"with the textbook OLS standard error; because the kernel values are nested",
        r"partial sums of one coefficient vector, that error understates the true",
        r"uncertainty. $\cos$ is the similarity of the session kernel to the pooled one.}",
        r"\label{tab:hawkes_kernel}",
        r"\begin{tabular}{lrrrrrrr}", r"\toprule",
        r"Session & Bars & $\hat n$ & Neg.\ lags & Neg.\ mass & $\hat\beta$ & SE & $\cos$ \\",
        r"\midrule",
    ]
    p0 = hk["pooled"]
    pw = p0.get("powerlaw") or {}
    lines.append(f"pooled & 242{{,}}934 & ${p0['branching_analogue']:+.4f}$ & "
                 f"{p0['n_negative_lags']}/12 & {p0['negative_mass_share'] * 100:.1f}\\% & "
                 f"{pw.get('beta', float('nan')):.2f} & {pw.get('se', float('nan')):.2f} & --- \\\\")
    lines.append(r"\midrule")
    for k, v in hk["sessions"].items():
        b = v.get("powerlaw") or {}
        lines.append(
            f"{k} & {v['n_bars']:,} & ${v['branching_analogue']:+.4f}$ & "
            f"{v['n_negative_lags']}/12 & {v['negative_mass_share'] * 100:.1f}\\% & "
            f"{b.get('beta', float('nan')):.2f} & {b.get('se', float('nan')):.2f} & "
            f"${v['cos_to_pooled']:+.2f}$ \\\\".replace(",", "{,}", 1))
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    open(os.path.join(STATS, "tab_hawkes_kernel.tex"), "w").write("\n".join(lines) + "\n")

    # ---- Table 5: the kernel experiments ----
    ke = json.load(open(os.path.join(STATS, "kernel_experiments.json")))
    order = ["divide/partial/5-95", "divide/marginal/5-95", "dummies/partial/5-95",
             "dummies/marginal/5-95", "divide/marginal/1-99", "divide/marginal/none",
             "dummies/marginal/none"]
    desc = {
        "divide/partial/5-95": "committed specification",
        "divide/marginal/5-95": "drop the exogenous block",
        "dummies/partial/5-95": "drop the rolling divisor",
        "dummies/marginal/5-95": "drop both",
        "divide/marginal/1-99": "winsorize at 1/99",
        "divide/marginal/none": "no winsorization",
        "dummies/marginal/none": "no divisor, no winsorization",
    }
    lines = [
        r"\begin{table}[H]", r"\centering",
        r"\caption{Kernel diagnostics under a rebuild from the raw parquet data,",
        r"varying the two design choices suspected of producing the archived kernel's",
        r"sign structure, plus a winsorization arm. Every cell has $n = 242{,}934$",
        r"rows. $\hat n = \sum_k w_k$ is the branching-ratio analogue; ``neg.\ mass''",
        r"is the share of absolute kernel mass carried by negative lags. QLIKE is a",
        r"single 60/40 split on the raw variance scale and is \emph{not} comparable to",
        r"the walk-forward levels reported elsewhere in this paper --- it is included",
        r"only so that degenerate cells are visible. \textbf{The first row does not",
        r"reproduce the archived kernel} (correlation $+0.65$, $\hat n$ $+1.00$ against",
        r"an archived $-0.038$), so no cell here explains the archived estimate; see",
        r"Section~\ref{sec:kernel_repro}.}",
        r"\label{tab:kernel_experiments}",
        r"\begin{tabular}{llrrrrr}", r"\toprule",
        r"Specification & Change & $\hat n$ & Neg.\ lags & Neg.\ mass & "
        r"$\hat\beta$ & QLIKE \\", r"\midrule",
    ]
    for k in order:
        r = ke[k]
        b = r.get("powerlaw") or {}
        lines.append(
            f"\\texttt{{{k.replace('_', chr(92) + '_')}}} & {desc[k]} & "
            f"${r['branching_analogue']:+.3f}$ & {r['n_negative_lags']}/12 & "
            f"{r['negative_mass_share'] * 100:.1f}\\% & "
            f"{b.get('beta', float('nan')):.2f} & {r['oos_qlike']:.4f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    open(os.path.join(STATS, "tab_kernel_experiments.tex"), "w").write("\n".join(lines) + "\n")

    print("wrote:")
    for f in ("tab_pairwise_bounds.tex", "tab_tile_dm.tex", "tab_mcs.tex",
              "tab_hawkes_kernel.tex", "tab_kernel_experiments.tex"):
        print("   ", os.path.join(STATS, f))


if __name__ == "__main__":
    main()
