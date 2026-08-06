# caches/ — small model caches for cluster jobs (AM-13, §34.7)

Contents (all on the FIXED panel, `alpha_panel_tw250.npz`, which is 203MB and NOT committed —
rebuild it deterministically with `python analysis/alpha_panel.py`, ~8 min, then set
`ALPHA_PANEL_CACHE` to a directory containing it plus these files):

- `har_resid.npz`      — HAR walk-forward residual (feeds the frozen frame + selections)
- `final_onestage.npz` — 679-column per-bar forecast (`yhat_bar`; the §22 one-stage model)
- `final_699_perbar.npz` — 679+F20 per-bar forecast (§35c entry)
- `pool40_perbar.npz`  — 679+F40 per-bar forecast (the production 719 first moment, §38.2)
