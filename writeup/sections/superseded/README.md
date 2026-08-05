# Superseded sections

These files are retained for history. **None of them is inputted by
`main.tex`, and no number in any of them appears in the current paper.**
They are kept because earlier decks and internal tables quote them, and a
reader comparing versions needs to know why the numbers moved.

| File | Why withdrawn |
|---|---|
| `results.tex` | §5.2's tuned-tree table (LightGBM 0.1099, XGBoost 0.1264, RF 0.1342) came from an Optuna search whose objective was QLIKE **on the full walk-forward backtest** — hyperparameter selection on the evaluation panel, under the evaluation loss, with unequal trial counts across models (125 / 30 / 16). The apparent model-class reversal is a protocol artifact. It survives in the paper only as the cautionary result in `linear_vs_nonlinear.tex` §"A Cautionary Result". §5.1 and §5.3's subgroup tables differenced rows with materially different `N` (138,279 to 222,058), confounding information with calendar regime. |
| `discussion.tex` | §6.1–6.2 narrate the withdrawn tuned-tree result and assert the opposite of the current model-class finding. §6.3's exogenous discussion is superseded by `marginal_contribution.tex`. |
| `tree_story.tex` | On a third panel again (QLIKE 0.1316 vs ridge 0.1380, and "13 30-min bars per day" against the 48-bar convention used everywhere else). All figure slots are unbuilt placeholders. The TreeSHAP methodology is sound and worth salvaging onto the frozen panel; the numbers are not comparable. |
| `master_table.tex`, `master_table_full.tex` | `baseline_exog_sweep_2026_05_05` cache, and a naive baseline (0.1955) that disagrees with both `results.tex` (0.3518) and the current shared incumbent (0.13415). |
| `meeting_update.tex`, `meeting_update_2026_06.tex` | Status decks, not paper sections. |

## What replaced them

The current paper draws every number from one frozen battery of 98 arms at
`n = 218,934` identical rows (`writeup/metrics_table_causal_tune_plus_spectral.csv`),
under the causal hyperparameter-selection protocol described in
`sections/algorithm_design.tex` §"Causal Hyperparameter Selection".

## Salvage list

Worth re-running on the frozen panel rather than discarding:

1. **TreeSHAP attribution** (`tree_story.tex`) — the walk-forward SHAP pooling
   protocol is the right way to answer "which interactions produce the tree's
   edge", and that question is currently unanswered in the paper.
2. **PatchTST / deep sequence arms** — promised in `results.tex` and never
   delivered; the paper asserts deep sequence models fail without a table.
3. **Ridge–PCA subgroup analysis** — the PCA-fails result is load-bearing for
   the dense-but-weak claim and currently rests on the withdrawn panel.
