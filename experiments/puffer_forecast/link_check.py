"""Softplus output link (user, 2026-10-13): loss-kernel check and gate-5 inputs.

    python experiments/puffer_forecast/link_check.py inputs          # _work/gate5/: losscheck bins, job files
    python experiments/puffer_forecast/link_check.py losscheck RUN   # RUN/losscheck_<loss>_softplus.out vs float64

The kernel (forecast_train losscheck IN OUT softplus) forms a = c + s o with c = softplus^-1(mu), s = sd / sigmoid(c),
f = softplus(a) (a > 20 ? a : log1p(exp(a))), and returns the loss and dL/do = dL/df sigmoid(a) s.  The inputs cover
random a around c and extremes from a = -80 (f ~ 2e-35 -> 0) to a = 80, across the a = 20 branch point, each with
targets 1e-4 .. 100, as (o, o + h, o - h) triplets for the kernel's own central differences.
"""

from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import reference as RF  # noqa: E402

REPO = HERE.parents[1]
WORK = REPO / "results" / "puffer_forecast_2026-10-12" / "_work"
G5 = WORK / "gate5"
MU, SD, DELTA, EPS, CAP = 0.95, 0.44, 0.6, 1e-3, 25.0
LOSSES = ((0, "mse"), (1, "huber"), (2, "qlike"))


def link_cs(mu: float, sd: float) -> tuple[float, float]:
    """the kernel's c and s (computed in double, stored as float32)."""
    c = np.log(np.expm1(np.float64(np.float32(mu))))
    s = np.float64(np.float32(sd)) * (1.0 + np.exp(-c))
    return float(np.float32(c)), float(np.float32(s))


def write_losscheck_inputs() -> None:
    rng = np.random.default_rng(12)
    c, s = link_cs(MU, SD)
    a = list(c + 2.0 * rng.normal(size=200))
    y_l = list(np.exp(rng.normal(-0.1, 0.5, size=200)))
    a_ext = [
        -80.0,
        -40.0,
        -30.0,
        -20.0,
        -15.0,
        -10.0,
        -5.0,
        -2.0,
        -1.0,
        0.0,
        1.0,
        2.0,
        5.0,
        10.0,
        15.0,
        19.5,
        19.99,
        20.01,
        20.5,
        30.0,
        50.0,
        80.0,
    ]
    y_ext = [1e-4, 0.01, 0.16, 1.0, 7.25, 100.0]
    for aa in a_ext:
        for yy in y_ext:
            a.append(aa)
            y_l.append(yy)
    o = (np.asarray(a) - c) / s
    y = np.asarray(y_l, np.float64)
    h = 1e-2 * np.maximum(1.0, np.abs(o))
    o = np.concatenate([o, o + h, o - h]).astype(np.float32)
    y = np.concatenate([y, y, y]).astype(np.float32)
    G5.mkdir(parents=True, exist_ok=True)
    for kind, name in LOSSES:
        hdr = struct.pack("<ii5f", len(o), kind, MU, SD, DELTA, EPS, CAP)
        (G5 / f"losscheck_{name}_softplus.bin").write_bytes(
            hdr + o.tobytes() + y.tobytes()
        )


def losscheck(run: Path) -> pd.DataFrame:
    rows = []
    c, s = link_cs(MU, SD)
    lp = dict(link=1, link_c=c, link_s=s, mu_y=MU, sd_y=SD)
    for kind, name in LOSSES:
        raw = (G5 / f"losscheck_{name}_softplus.bin").read_bytes()
        n = struct.unpack("<i", raw[:4])[0]
        o = np.frombuffer(raw[28 : 28 + 4 * n], "<f4").astype(np.float64)
        y = np.frombuffer(raw[28 + 4 * n : 28 + 8 * n], "<f4").astype(np.float64)
        out = (run / f"losscheck_{name}_softplus.out").read_bytes()
        lk = np.frombuffer(out[: 4 * n], "<f4").astype(np.float64)
        gk = np.frombuffer(out[4 * n : 8 * n], "<f4").astype(np.float64)
        ck = np.frombuffer(out[8 * n : 12 * n], "<f4").astype(np.float64)
        m = n // 3
        base = slice(0, m)
        # the float64 reference at the kernel's float32 a (a = c + s o rounded as the kernel rounds it)
        a32 = (np.float32(c) + np.float32(s) * o.astype(np.float32)).astype(np.float64)
        f = np.logaddexp(0.0, a32)
        dfdo = s * RF.sig(a32)
        lr_, gfr, cr_ = RF.loss_ref(kind, f, y, DELTA, EPS, CAP)
        gr_ = gfr * dfdo
        # float64 central differences of the reference in o (validates the analytic chain dL/df * df/do)
        hr = 1e-6 * np.maximum(1.0, np.abs(o[:m]))
        f_p, _ = RF.link_ref(lp, o[:m] + hr)
        f_m, _ = RF.link_ref(lp, o[:m] - hr)
        fd_r = (
            RF.loss_ref(kind, f_p, y[:m], DELTA, EPS, CAP)[0]
            - RF.loss_ref(kind, f_m, y[:m], DELTA, EPS, CAP)[0]
        ) / (2 * hr)
        h = (o[m : 2 * m] - o[2 * m :]) / 2
        fd_k = (lk[m : 2 * m] - lk[2 * m :]) / (2 * h)
        a_b = a32[base]
        # FD validity: no kink (Huber |e| = delta, QLIKE cap) inside the +-h interval, and a curvature small
        # enough over h (|a| step s h < 1 around the softplus bend)
        smooth = ~cr_[base] & ~cr_[m : 2 * m] & ~cr_[2 * m :]
        if kind == 1:
            e = f[base] - y[base]
            smooth &= np.abs(np.abs(e) - DELTA) > 2 * np.abs(dfdo[base]) * h + 1e-12
        fin = np.isfinite(lr_[base])

        def rel(x, ref, sel):
            fl = 1e-3 * np.max(np.abs(ref[sel]))
            return np.abs(x[sel] - ref[sel]) / np.maximum(np.abs(ref[sel]), fl)

        okf = fin & (f[base] >= 1e-6)
        rows.append(
            dict(
                loss=name,
                n=m,
                n_fd=int(smooth.sum()),
                min_a=float(a_b.min()),
                max_a=float(a_b.max()),
                kernel_vs_ref_loss_max_rel=float(np.max(rel(lk[base], lr_[base], okf))),
                kernel_vs_ref_grad_max_rel=float(np.max(rel(gk[base], gr_[base], okf))),
                ref_grad_vs_ref_fd_max_rel=float(
                    np.max(rel(gr_[base], fd_r, fin & ~cr_[base]))
                ),
                kernel_grad_vs_kernel_fd_median_rel=float(
                    np.median(rel(gk[base], fd_k, smooth))
                ),
                kernel_grad_vs_kernel_fd_p95_rel=float(
                    np.quantile(rel(gk[base], fd_k, smooth), 0.95)
                ),
                f_kernel_min_positive=bool(np.all(f[base][a_b >= -80] > 0)),
                n_capped=int(ck[base].sum()),
                cap_flags_agree=bool(np.array_equal(ck[base] > 0, cr_[base])),
                all_finite=bool(np.isfinite(lk).all() and np.isfinite(gk).all()),
                grad_at_a_minus80_abs=float(np.max(np.abs(gk[base][a_b < -79]))),
                grad_at_a_plus80_vs_ref=float(
                    np.max(rel(gk[base], gr_[base], fin & (a_b > 79)))
                ),
            )
        )
    return pd.DataFrame(rows)


def kv(**k) -> str:
    return " ".join(f"{a}={b}" for a, b in k.items())


def write_jobs() -> None:
    """gate-5 job files: softplus gradcheck, the softplus overfit job, Phase 2c block 0 with link=softplus."""
    G5.mkdir(parents=True, exist_ok=True)
    grad = (WORK / "jobs_grad.txt").read_text().splitlines()
    (G5 / "grad_softplus.txt").write_bytes(
        (
            "\n".join(
                line.replace("id=grad_", "id=gradsp_", 1) + " link=softplus"
                for line in grad
                if line.strip()
            )
            + "\n"
        ).encode()
    )
    import overfit as O

    lines = [
        O.job("baseline", 64, 2, 20, 0.01, "_softplus", link="softplus"),
        O.job("baseline", 64, 2, 20, 0.01, "_identity"),
    ]
    (G5 / "overfit_softplus.txt").write_bytes(("\n".join(lines) + "\n").encode())
    tune = (WORK / "phase2c" / "tune_block0.txt").read_text().splitlines()
    (G5 / "tune_block0_softplus.txt").write_bytes(
        ("\n".join(t + " link=softplus" for t in tune if t.strip()) + "\n").encode()
    )
    m = json.loads((WORK / "phase2c" / "blocks.json").read_text())
    m["common"]["link"] = "softplus"
    (G5 / "blocks_softplus.json").write_bytes(json.dumps(m, indent=1).encode())
    # link=identity written out explicitly: must equal the same jobs without the key, byte for byte
    r500 = (
        (
            REPO
            / "results"
            / "puffer_forecast_2026-10-12"
            / "phase2c"
            / "runs"
            / "13082282"
            / "block0"
            / "jobs_refit.txt"
        )
        .read_text()
        .splitlines()[:500]
    )
    (G5 / "refit500_link_identity.txt").write_bytes(
        ("\n".join(r + " link=identity" for r in r500) + "\n").encode()
    )
    bad = (G5.parent / "gate4" / "ok_small.txt").read_text().strip()
    (G5 / "bad_link.txt").write_bytes(
        (bad.replace("id=ok", "id=bad") + " link=exp\n").encode()
    )
    (G5 / "softplus_pack.txt").write_bytes(
        (bad.replace("id=ok", "id=bad") + " link=softplus\n").encode()
    )
    # task-0 concurrency A/B (12 vs 72): the 72 round-1 small models for 100 epochs (lr schedule over 1,000)
    t0 = (WORK / "overfit_r1" / "task_00_c12.txt").read_text().splitlines()
    (G5 / "task0_100ep.txt").write_bytes(
        (
            "\n".join(
                t.replace("epochs=1000 ", "epochs=100 ", 1).replace(
                    " epochs=1000", " epochs=100"
                )
                for t in t0
                if t.strip()
            )
            + "\n"
        ).encode()
    )


if __name__ == "__main__":
    if sys.argv[1] == "inputs":
        write_losscheck_inputs()
        write_jobs()
        print(sorted(p.name for p in G5.iterdir()))
    elif sys.argv[1] == "losscheck":
        pd.set_option("display.width", 250)
        print(losscheck(Path(sys.argv[2])).T.to_string())
