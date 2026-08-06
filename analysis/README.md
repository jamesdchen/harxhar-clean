# analysis/ — module index

104 modules across several campaigns. This index covers the **alpha-manifestation study**
(2026-08-04 →, `writeup/alpha_manifestation_findings_2026-08-04.md`, graded claims in
`writeup/CLAIMS_LEDGER_2026-08-06.md`); older campaigns are indexed by their own handoffs
(`writeup/SESSION_HANDOFF_*.md`).

## The study's chain of custody (section order)

| Module | Sections | What it holds |
|---|---|---|
| `alpha_manifestation.py` | §§1–13 | original manifestation battery; `TW`, `dm_test` |
| `synthesis.py` | §§15–16 | §15.3 execution; `_p` cache helper, blockwise ridge |
| `geometry.py`, `composition.py` | §§14, 17 | state axes; partition-of-unity tests |
| `minimal_model.py` | §§18–22, 24 | two-stage + one-stage FINAL model; QLIKE machinery (`_qlike_series`, `_hac_mean_t`); dial stages |
| `pc_quadratics.py` | §§19, 23 | PC-quadratic frames, soft thresholds, signal map |
| `cross_section.py` | §20 | 36 xsec ratio features (Tier-3 carried) |
| `map_monitor.py` | §26 | rot-detector calibration; daily lead-lag flow; `_frame_and_scores` (THE frozen frame) |
| `cucuringu.py` | §§27–28, 30 | the full Cucuringu battery; phase stages; `_causal_phase`, intraday flow |
| `straddle_horizon.py` | §§29, 29.1–29.2, 32 | intraday H-ladder, EOD/overnight targets, embargoed walks, re-draw |
| `alpha_law.py` | §29.3 | per-horizon shrinkage prior |
| `organ_arm.py` | §29.4 | announcement organ port (needs `data/releases.parquet`) |
| `cadence_decomp.py` | §29.5 | per-bar increment by channel (coefficient-vintage hybrids) |
| `fast_rerun.py` | §§29–30 verdicts, §30.3 | consolidated fast-engine reruns; `RAWLAG=1` control |
| `vrp_eod.py` | AM-02 | Tier-1 VRP consumer (blocked on chain parquets) |
| `knn_arm.py` | §33 P1 | professor's kNN, 3-view head-to-head |
| `regime_atlas.py` | §34 | regime decomposition (A1 coefficient trajectory, A2/A3 labels) |
| `wf.py` | engines | `walk_forward` (rank-1), `walk_forward_embargo_blocked` (day-blocked, ~8×), `walk_forward_embargo_dualcadence` (Sherman–Morrison, exact, both cadences + vintage hybrids) |

Engine discipline: any comparison uses the SAME engine for both arms. Honest labels at horizon
H: training window [t−H−W, t−H) everywhere (`walk_forward_embargo*`).

## Prior-campaign modules touched by this study

`nl_sparsity.py` (products, `_pair_ic`, `base_columns`), `alpha_panel.py` (panel cache),
`multiplicity.py` (Romano–Wolf; AM-10 will extend), `supervised_metric_knn.py` +
`../src/models/spectral_knn.py` (the prior kNN record cited in §33), `heat_graph.py`,
`intensity_graph.py`, `noise_metric.py`, `tail_fix.py`, `voldemand_fix.py`.
