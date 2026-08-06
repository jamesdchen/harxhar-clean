"""§34.7: the LSTM smear head — the study's first justified deep arm.

Trains directly on QLIKE-through-the-smear (the deliverable's loss), sparse inputs, long memory
handed over as features. Expanding walk-forward, biennial refits, eval 2016+, 3 seeds. Gates:
DM >= +2.0 vs the means+leverage smear and vs the means+leverage+probe smear.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.alpha_manifestation import TW  # noqa: E402
from analysis.alpha_panel import load_panel  # noqa: E402
from analysis.minimal_model import HOLDOUT, _hac_mean_t  # noqa: E402
from analysis.synthesis import _p  # noqa: E402

SEQ = 96
HID = 48
EPOCHS = 8
STRIDE = 4
SEEDS = (0, 1, 2)
REFIT_YEARS = 2
EVAL_START = "2016-01-01"


def build_frames():
    p = load_panel()
    ts = pd.Series(pd.to_datetime(p.t))
    n = len(ts)
    f = np.full(n, np.nan)
    f[TW:] = np.load(_p("final_onestage.npz"))["yhat_bar"]
    e2 = (p.y - f) ** 2
    day = ts.dt.normalize()
    day_codes = pd.factorize(day)[0]

    a_day = pd.Series(e2).groupby(day.values).mean()
    la = np.log(a_day + 1e-12)
    r_day = pd.Series(p.X[:, p.names.index("adj_sumret_ma_1")].astype(np.float64)).groupby(day.values).sum()
    rs = (r_day - r_day.mean()) / (r_day.std() + 1e-12)
    day_map = dict(zip(la.index, range(len(la))))
    didx = day.map(day_map).to_numpy()
    l1 = la.shift(1).to_numpy()[didx]
    m5 = la.shift(1).rolling(5).mean().to_numpy()[didx]
    m21 = la.shift(1).rolling(21).mean().to_numpy()[didx]
    r1 = rs.shift(1).to_numpy()[didx]
    ar1 = rs.abs().shift(1).to_numpy()[didx]
    slot = (ts.dt.hour * 2 + ts.dt.minute // 30).to_numpy()
    rel = pd.read_parquet("data/releases.parquet")
    rel["endbartime"] = pd.to_datetime(rel["endbartime"])
    flags = pd.DataFrame({"t": ts}).merge(rel, left_on="t", right_on="endbartime",
                                          how="left").drop(columns=["t", "endbartime"]).fillna(0.0)
    rel_today = (flags.sum(axis=1) > 0).astype(float).groupby(day_codes).transform("max").to_numpy()
    le2 = np.log(np.where(np.isfinite(e2), e2, np.nan) + 1e-12)
    le2_lag = np.full(n, np.nan)
    le2_lag[1:] = le2[:-1]
    F = np.column_stack([
        le2_lag, l1, m5, m21, r1, ar1,
        np.sin(2 * np.pi * slot / 48), np.cos(2 * np.pi * slot / 48), rel_today,
    ])
    mu = np.nanmean(F[: 2 * TW], axis=0)
    sdv = np.nanstd(F[: 2 * TW], axis=0) + 1e-12
    F = np.nan_to_num((F - mu) / sdv)
    true_raw = p.y**2 * p.baseline
    return ts, F, f, p.y, p.baseline, true_raw, e2, day, day_codes


def main() -> None:
    import torch
    import torch.nn as nn
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "3")))

    ts, F, f, y, B, true_raw, e2, day, day_codes = build_frames()
    n = len(F)
    valid = np.isfinite(f) & np.isfinite(y) & (B > 0) & (true_raw > 0)
    years = ts.dt.year.to_numpy()

    device = "cpu"
    Ft = torch.tensor(F, dtype=torch.float32)
    f2B = torch.tensor(np.where(valid, f**2, 1.0), dtype=torch.float32)
    Bt = torch.tensor(np.where(valid, B, 1.0), dtype=torch.float32)
    tr_t = torch.tensor(np.where(valid, true_raw, 1.0), dtype=torch.float32)

    class Head(nn.Module):
        def __init__(self):
            super().__init__()
            self.rnn = nn.LSTM(F.shape[1], HID, num_layers=1, batch_first=True)
            self.drop = nn.Dropout(0.1)
            self.out = nn.Linear(HID, 1)

        def forward(self, x):
            h, _ = self.rnn(x)
            return self.out(self.drop(h[:, -1])).squeeze(-1)

    def qlike_loss(shat, idx):
        pred_raw = (f2B[idx] + torch.exp(shat)) * Bt[idx]
        r = tr_t[idx] / pred_raw.clamp(min=1e-12)
        return (r - torch.log(r) - 1.0).mean()

    def windows(idx):
        return torch.stack([Ft[i - SEQ : i] for i in idx])

    eval_years = sorted(set(years[(ts >= EVAL_START).to_numpy()]))
    preds = {s: np.full(n, np.nan) for s in SEEDS}
    for seed in SEEDS:
        torch.manual_seed(seed)
        np.random.seed(seed)
        for y0 in range(eval_years[0], eval_years[-1] + 1, REFIT_YEARS):
            tr_mask = (years < y0) & valid & (np.arange(n) >= SEQ + 2 * TW)
            te_mask = (years >= y0) & (years < y0 + REFIT_YEARS) & valid & (np.arange(n) >= SEQ)
            tr_idx = np.flatnonzero(tr_mask)[::STRIDE]
            te_idx = np.flatnonzero(te_mask)
            if len(tr_idx) < 5000 or len(te_idx) == 0:
                continue
            model = Head()
            opt = torch.optim.Adam(model.parameters(), lr=1e-3)
            for ep in range(EPOCHS):
                perm = np.random.permutation(len(tr_idx))
                tot = 0.0
                for b0 in range(0, len(perm), 256):
                    bi = tr_idx[perm[b0 : b0 + 256]]
                    xb = windows(torch.tensor(bi))
                    opt.zero_grad()
                    loss = qlike_loss(model(xb), torch.tensor(bi))
                    loss.backward()
                    opt.step()
                    tot += float(loss) * len(bi)
                print(f"seed {seed} refit {y0} epoch {ep} loss {tot/len(tr_idx):.5f}", flush=True)
            model.eval()
            with torch.no_grad():
                for b0 in range(0, len(te_idx), 4096):
                    bi = te_idx[b0 : b0 + 4096]
                    preds[seed][bi] = np.exp(model(windows(torch.tensor(bi))).numpy())
            print(f"seed {seed} refit {y0} done ({len(te_idx)} eval bars)", flush=True)
    np.savez_compressed(_p("lstm_smear_preds.npz"), **{f"seed{s}": preds[s] for s in SEEDS})

    # score vs the linear smears on the common evaluation span
    from analysis.smear_scoring import linear_smears  # built alongside
    sm_lev, sm_probe = linear_smears()
    late = (ts >= HOLDOUT).to_numpy()

    def qseries(sm):
        ok = valid & np.isfinite(sm) & (sm > 0)
        pr = (f**2 + sm) * B
        q = np.full(n, np.nan)
        r = true_raw[ok] / pr[ok]
        q[ok] = r - np.log(r) - 1.0
        return q

    q_lev, q_probe = qseries(sm_lev), qseries(sm_probe)
    for seed in SEEDS:
        q_l = qseries(preds[seed])
        for bname, qb in (("means+lev", q_lev), ("means+lev+probe", q_probe)):
            d = qb - q_l
            md = np.isfinite(d)
            g = _hac_mean_t(d[md], 480)
            print(f"seed {seed} vs {bname:16s}: QLIKE {np.nanmean(q_l):.5f} vs "
                  f"{np.nanmean(qb[md]):.5f}   DM {g:+.2f} "
                  f"(2020+ {_hac_mean_t(d[md & late], 480):+.2f})  "
                  f"{'PASS' if g >= 2.0 else 'FAIL'}", flush=True)


if __name__ == "__main__":
    main()
