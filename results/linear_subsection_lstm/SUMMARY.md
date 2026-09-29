# A small LSTM on the per-bar design (G1), 2026-09-29

Sources: `lstm_qlike_1600.csv`, `lstm_trade_1530.csv`, `lstm_trade_vs.csv`, `lstm_vs_comparators.csv`, `lstm_fitspace_1600.csv`, `lstm_hyperparameter_path.csv`, `lstm_hyperparameter_share.csv`, `lstm_seed_spread.csv`, `lstm_join_gates.csv`, `gates/gate_summary.csv`, `reduce_gates.csv`; all tables printed in `lstm_score_tables.md`. Scripts: `specs/causal_tune_lstm.py` (+ `_jobs.py`), `experiments/gate_lstm.py`, `experiments/reduce_lstm_chunks.py`, `experiments/build_subsection_lstm_yhat.py`, `experiments/score_lstm_subsection.py`. Every number below is read from those files.

## Results in one paragraph
A small LSTM on the per-bar design, tuned and refitted exactly like the per-bar ridge and trees, forecasts the 15:30–16:00 variance worse than every linear and tree forecast. QLIKE vs the per-bar ridge: +23 % on the live-feasible inputs [+0.015, +0.041], +43 % on all features, +9 % on HAR + calendar. More inputs make it worse. The 15:30 sign(s) straddle trade is weaker (Sharpe 1.01 vs 1.90 mid on live-feasible; difference −0.89 [−1.92, +0.25]). Averaged half-and-half with the ridge, the QLIKE is +1.6 % / +6.0 % / +2.4 % vs the ridge alone. Its forecast correlates 0.87–0.98 with the ridge's and 0.801 with the target (ridge 0.848, live-feasible).

## Model and discipline
- One LSTM layer over the last L sessions of the per-bar design (the same columns as the input set, standardised causally with the training window's mean and sd), 16:00 bar only, one model per input set (HAR + calendar "baseline", live-feasible, all features).
- Target: the per-bar arms' own target (square root of the diurnally adjusted variance, winsorized), so the run stacks and scores with the same tools as the ridge and trees.
- Grid: L {5, 20} × hidden {16, 64} × dropout {0, 0.2} × Adam lr {1e-3, 1e-2}; the forecast is the average of 5 seeds. Tuned every 250 sessions on the 125-session tail after the 25-session embargo (MSE rule of record; the QLIKE rule also walked forward, "qsel"); refit every 10 sessions; 2000-session window; early stopping on the validation tail.
- Cluster: canary 12464876 (chunk 0 twice, byte-identical) → fleet 12464877 (18 chunks = 3 input sets × 6 time chunks, 42 s – 13 min each) → score 12464878; all 20 jobs completed. torch 2.12.1+cpu was already in the cluster environment.

## Gates (all PASS; `gates/gate_summary.csv`)
- Identity, all 3 input sets: design, target, stamps and column names match the linear spec's own data path (sha256); the target equals the stored ridge results exactly.
- Determinism: bit-identical locally; byte-identical on the cluster via the canary.
- Pool = serial, bit-identical. Chunked = unchunked, end to end, bit-identical (the reducer reads CSVs round-trip exact).
- Local smoke at campaign settings: 300 sessions, 3103 s with 4 workers.

## Results (research scorer = 16:00-bar recalibration; 1,406 shared stamps, 866 deck days) — `lstm_qlike_1600.csv`, `lstm_trade_1530.csv`, `lstm_trade_vs.csv`
| input set | LSTM QLIKE | vs ridge | 95 % interval (ΔQLIKE) | DM | Sharpe mid [95 %] | ridge Sharpe | difference [95 %] |
|---|---|---|---|---|---|---|---|
| live_feasible | 0.1468 | +23.1 % | [+0.0149, +0.0407] | 4.56 | 1.01 [0.11, 2.00] | 1.90 | −0.89 [−1.92, +0.25] |
| all_features | 0.1703 | +43.0 % | [+0.0354, +0.0698] | 6.63 | 0.53 [−0.41, 1.53] | 1.66 | −1.13 [−2.40, +0.10] |
| baseline | 0.1270 | +8.9 % | [+0.0055, +0.0159] | 4.32 | 0.61 [−0.38, 1.54] | 1.22 | −0.60 [−1.31, +0.06] |

- QLIKE: the LSTM loses to every linear and tree arm on live_feasible and all_features (every interval above zero; `lstm_vs_comparators.csv`). On baseline it is level with the trees (all tree intervals include zero).
- More inputs hurt it: QLIKE 0.127 → 0.147 → 0.170 from 34 to 244 to 640 columns, while the ridge stays flat.
- Crossed Sharpe: 0.54 / 0.06 / 0.14. The QLIKE-selected path is no better (0.88 / 0.37 / 0.93 mid).
- Trade differences whose intervals exclude zero: vs tuned all-features LightGBM −1.28 [−2.48, −0.09]; on baseline vs lasso −0.89 [−1.71, −0.15] and vs OLS −0.71 [−1.41, −0.05].
- LSTM + ridge, equal weights: never beats the ridge — QLIKE +1.6 % / +6.0 % / +2.4 % (baseline interval above zero) and lower Sharpe.
- Fit-space diagnostics (`lstm_fitspace_1600.csv`): calibrated (Mincer–Zarnowitz slope 0.97–1.00) but less correlated with the target (0.801 vs 0.848 on live_feasible); its forecast correlates 0.87–0.98 with the ridge's. Tuning mostly picks lr 0.01 and stops early (mean 15.9 epochs, none at the cap; `lstm_hyperparameter_share.csv`).
- Seeds (`lstm_seed_spread.csv`): the 5-seed average beats every single network; single-network Sharpe ranges 0.01–1.11 on all_features, so a one-seed result would be mostly noise.
- Extreme values: without the recalibration's error term the stored all-features ridge scores QLIKE 2.64 (one day 3,517) vs 0.119 recalibrated; the LSTM has no such extremes.

## Caveats
- Cross-platform: the laptop (torch 2.9.1) and the cluster choose different configurations at chunk 0, because the lr 1e-2 fits diverge; configuration-rank correlation 0.956, forecast correlation 0.975. Results are bit-reproducible on one software stack, not across stacks.
- The arms merged on the cluster differ from the local re-merge by at most one unit in the last place; the local ones are the ones scored and stacked.

## Outputs
- `results/spxw_pnl/yhat_lstm_<bucket>.parquet` and `yhat_lstm_qsel_<bucket>.parquet` (16:00 bar only; committed) — rows in the master table (`results/close_master_table/master_table.csv`, scored with the 16:00-bar recalibration, never the 13-bar notebook one).
- Cluster twins: `cluster/slurm/{ship_lstm_carc.sh, submit_lstm.sh, lstm_pack.sbatch, lstm_score.sbatch}`, `cluster/lstm_tasks_{canary,fleet}.txt`.
