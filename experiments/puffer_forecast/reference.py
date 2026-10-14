"""PufferLib-5.0 native MinGRU forecaster (2026-10-12, Phase 1): numpy reference and correctness checks.

A float64 numpy re-implementation of exactly what the native trainer computes, used to check it:
  network   PufferLib 5.0 src/algo.cu as built by build_arch: encoder x @ We^T (no bias) -> MinGRU layers
            (combined = h @ Wl^T split [hidden | gate | proj]; z = sigmoid(gate); h~ = hidden + 0.5 if
            hidden >= 0 else sigmoid(hidden); h_t = (1 - z) h_{t-1} + z h~, h_0 = 0; highway
            out_t = s h_t + (1 - s) x_t, s = sigmoid(proj)) -> decoder h @ Wd^T (2 columns: logit, value)
  loss      ours (experiments/puffer_forecast/native/forecast_train.cu fc_loss): f = mu_y + sd_y o,
            o = decoder column 0 at the last step; mse / huber / qlike (smooth floor eps, capped gradient)
  optimizer upstream Muon (algo.cu muon_step): global L2 clip at max_grad_norm (eps 1e-6), Nesterov
            momentum, per-matrix Frobenius normalisation (eps 1e-7), 5 Newton-Schulz steps with the
            upstream coefficients, scale sqrt(max(1, R / C)), w -= lr * update; lr = upstream cosine
            annealing over max_epochs
Checks (inputs: a run directory of cluster/puffer_forecast_phase1_carc.sbatch):
  losscheck   the loss kernel's value and dL/do vs the float64 formula on random and extreme (o, y),
              and vs the kernel's own central differences (excluding capped samples)
  gradcheck   one forward / backward of the native network (toy and both real designs): forward output,
              loss and every weight gradient vs the float64 reference; the reference's analytic gradient
              vs float64 central differences
  trajectory  the toy problem trained natively (12 epochs, every loss) vs the reference from the same
              initial weights and minibatch order: per-step loss and gradient norm, final forecasts
Writes checks_*.csv into results/puffer_forecast_2026-10-12/ and prints a summary.

    python experiments/puffer_forecast/reference.py RUN_DIR
"""

from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "results" / "puffer_forecast_2026-10-12"
NS = (
    (4.0848, -6.8946, 2.9270),
    (3.9505, -6.3029, 2.6377),
    (3.7418, -5.5913, 2.3037),
    (2.8769, -3.1427, 1.2046),
    (2.8366, -3.0525, 1.2012),
)


def sig(x):
    return 1.0 / (1.0 + np.exp(-x))


# ============================================================================ loss (float64 reference)
def loss_ref(kind: int, f, y, delta: float, eps: float, cap: float):
    """per-sample loss, dL/df and the cap flag (the kernel's formulas in float64)."""
    f = np.asarray(f, np.float64)
    y = np.asarray(y, np.float64)
    capped = np.zeros(f.shape, bool)
    if kind == 0:
        e = f - y
        return e * e, 2 * e, capped
    if kind == 1:
        e = f - y
        ae = np.abs(e)
        inner = ae <= delta
        return (
            np.where(inner, e * e, 2 * delta * ae - delta**2),
            np.where(inner, 2 * e, 2 * delta * np.sign(e)),
            capped,
        )
    eps2 = eps * eps
    fe2 = f * f + eps2
    y2 = np.maximum(y * y, eps2)
    logr = np.clip(np.log(y2) - np.log(fe2), -80.0, 80.0)
    r = np.exp(logr)
    g = 2 * f * (1 - r) / fe2
    capped = np.abs(g) > cap
    return r - logr - 1, np.where(capped, np.sign(g) * cap, g), capped


# ============================================================================ network (float64 reference)
def unpack(flat: np.ndarray, P: int, H: int, layers: int) -> dict:
    """weights_create order: encoder [H, P], decoder [2, H] (od + 1 rows), MinGRU layers [3H, H]."""
    i = 0
    out: dict = {}
    out["We"] = flat[i : i + H * P].reshape(H, P)
    i += H * P
    out["Wd"] = flat[i : i + 2 * H].reshape(2, H)
    i += 2 * H
    out["Wl"] = []
    for _ in range(layers):
        out["Wl"].append(flat[i : i + 3 * H * H].reshape(3 * H, H))
        i += 3 * H * H
    assert i == len(flat), (i, len(flat))
    return out


def pack(w: dict) -> np.ndarray:
    return np.concatenate(
        [w["We"].ravel(), w["Wd"].ravel()] + [x.ravel() for x in w["Wl"]]
    )


def forward(w: dict, x: np.ndarray) -> tuple[np.ndarray, list]:
    """x [B, T, P] -> dec [B, T, 2] and the cache for backward."""
    B, T, _ = x.shape
    h = x @ w["We"].T
    cache = []
    for Wl in w["Wl"]:
        H = Wl.shape[1]
        comb = h @ Wl.T
        hid, gate, proj = comb[..., :H], comb[..., H : 2 * H], comb[..., 2 * H :]
        z = sig(gate)
        ht = np.where(hid >= 0, hid + 0.5, sig(hid))
        s = sig(proj)
        hs = np.zeros((B, H))
        hprev = np.zeros((B, T, H))
        hcur = np.zeros((B, T, H))
        out = np.zeros((B, T, H))
        for t in range(T):
            hprev[:, t] = hs
            hs = hs + z[:, t] * (ht[:, t] - hs)
            hcur[:, t] = hs
            out[:, t] = s[:, t] * hs + (1 - s[:, t]) * h[:, t]
        cache.append(dict(x=h, hid=hid, z=z, ht=ht, s=s, hprev=hprev, hcur=hcur))
        h = out
    dec = h @ w["Wd"].T
    cache.append(dict(hlast=h))
    return dec, cache


def backward(w: dict, x: np.ndarray, cache: list, ddec: np.ndarray) -> dict:
    g = {"Wd": np.einsum("btk,bth->kh", ddec, cache[-1]["hlast"])}
    dh = ddec @ w["Wd"]
    gl = [None] * len(w["Wl"])
    for li in range(len(w["Wl"]) - 1, -1, -1):
        c = cache[li]
        Wl = w["Wl"][li]
        B, T, H = dh.shape
        dcomb = np.zeros((B, T, 3 * H))
        dx = np.zeros((B, T, H))
        carry = np.zeros((B, H))
        for t in range(T - 1, -1, -1):
            dout = dh[:, t]
            s, z, ht, hp, hc, hid, xt = (
                c["s"][:, t],
                c["z"][:, t],
                c["ht"][:, t],
                c["hprev"][:, t],
                c["hcur"][:, t],
                c["hid"][:, t],
                c["x"][:, t],
            )
            dproj = dout * (hc - xt) * s * (1 - s)
            dx[:, t] = dout * (1 - s)
            dtot = carry + dout * s
            dz = dtot * (ht - hp)
            dht = dtot * z
            dcomb[:, t, 2 * H :] = dproj
            dcomb[:, t, H : 2 * H] = dz * z * (1 - z)
            dcomb[:, t, :H] = np.where(hid >= 0, dht, dht * ht * (1 - ht))
            carry = dtot * (1 - z)
        gl[li] = np.einsum("btk,bth->kh", dcomb, c["x"])
        dh = dcomb @ Wl + dx
    g["Wl"] = gl
    g["We"] = np.einsum("bth,btp->hp", dh, x)
    return g


def link_ref(lp: dict, o):
    """f and df/do: identity f = mu + sd o; softplus (link 1, 2026-10-13) f = softplus(c + s o)."""
    if lp.get("link", 0) == 1:
        a = lp["link_c"] + lp["link_s"] * o
        return np.logaddexp(0.0, a), lp["link_s"] * sig(a)
    return lp["mu_y"] + lp["sd_y"] * o, np.full(np.shape(o), lp["sd_y"])


def batch_loss(w: dict, x, y, mask, lp: dict, n_real: int, want_grad: bool = True):
    dec, cache = forward(w, x)
    o = dec[:, -1, 0]
    f, dfdo = link_ref(lp, o)
    ls_, gf, capped = loss_ref(
        lp["loss_kind"], f, y, lp["huber_delta"], lp["qlike_eps"], lp["qlike_cap"]
    )
    L = float(np.sum(mask * ls_) / n_real)
    if not want_grad:
        return L, dec, None
    ddec = np.zeros_like(dec)
    ddec[:, -1, 0] = mask * gf * dfdo / n_real
    return L, dec, backward(w, x, cache, ddec)


def muon_step(
    w: dict, g: dict, mb: dict, lr: float, mu: float, max_norm: float
) -> float:
    names = ["We", "Wd"] + [("Wl", i) for i in range(len(w["Wl"]))]

    def get(d, k):
        return d[k[0]][k[1]] if isinstance(k, tuple) else d[k]

    def put(d, k, v):
        if isinstance(k, tuple):
            d[k[0]][k[1]] = v
        else:
            d[k] = v

    sumsq = sum(float(np.sum(get(g, k) ** 2)) for k in names)
    clip = min(max_norm / (np.sqrt(sumsq) + 1e-6), 1.0)
    for k in names:
        gk = get(g, k) * clip
        m = mu * get(mb, k) + gk
        put(mb, k, m)
        x = gk + mu * m
        R, C = x.shape
        x = x / max(np.sqrt(np.sum(x * x)), 1e-7)
        tall = R > C
        for a, b, c in NS:
            G = x.T @ x if tall else x @ x.T
            Bm = b * G + c * (G @ G)
            x = a * x + (x @ Bm if tall else Bm @ x)
        put(w, k, get(w, k) - lr * np.sqrt(max(1.0, R / C)) * x)
    return float(np.sqrt(sumsq))


# ============================================================================ checks
def read_f32(p: Path) -> np.ndarray:
    return np.frombuffer(p.read_bytes(), "<f4").astype(np.float64)


def losscheck(run: Path, inputs: Path) -> pd.DataFrame:
    rows = []
    for name, kind in (("mse", 0), ("huber", 1), ("qlike", 2)):
        raw = (inputs / f"losscheck_{name}.bin").read_bytes()
        n, k = struct.unpack("<ii", raw[:8])
        mu, sd, delta, eps, cap = struct.unpack("<5f", raw[8:28])
        o = np.frombuffer(raw[28 : 28 + 4 * n], "<f4").astype(np.float64)
        y = np.frombuffer(raw[28 + 4 * n : 28 + 8 * n], "<f4").astype(np.float64)
        out = (run / f"losscheck_{name}.out").read_bytes()
        lk = np.frombuffer(out[: 4 * n], "<f4").astype(np.float64)
        gk = np.frombuffer(out[4 * n : 8 * n], "<f4").astype(np.float64)
        ck = np.frombuffer(out[8 * n : 12 * n], "<f4").astype(np.float64)
        f = np.float64(np.float32(mu)) + np.float64(np.float32(sd)) * o
        lr_, gr_, cr_ = loss_ref(kind, f, y, delta, eps, cap)
        gr_ = gr_ * sd
        m = n // 3  # (base, +h, -h) triplets
        h = (o[m : 2 * m] - o[2 * m :]) / 2
        fd_k = (lk[m : 2 * m] - lk[2 * m :]) / (2 * h)
        # float64 central differences of the reference formula (validates the analytic gradient), own step
        hr = 1e-6 * np.maximum(1.0, np.abs(o[:m]))
        fd_r = (
            loss_ref(kind, f[:m] + sd * hr, y[:m], delta, eps, cap)[0]
            - loss_ref(kind, f[:m] - sd * hr, y[:m], delta, eps, cap)[0]
        ) / (2 * hr)
        base = slice(0, m)
        free = ~cr_[base]
        # FD validity: no kink (Huber |e| = delta, QLIKE cap) and, for QLIKE, |f| away from the eps floor
        # by more than the step, inside the +-h interval
        smooth = free & ~cr_[m : 2 * m] & ~cr_[2 * m :]
        if kind == 1:
            e = f[base] - y[base]
            smooth &= np.abs(np.abs(e) - delta) > 2 * h * sd
        if kind == 2:
            smooth &= np.abs(f[base]) > 20 * h * sd
        fin = np.isfinite(lr_[base])

        def rel(a, b, sel):
            # relative error with a floor of 1e-3 x the largest reference value (near-zero gradients)
            fl = 1e-3 * np.max(np.abs(b[sel]))
            return np.abs(a[sel] - b[sel]) / np.maximum(np.abs(b[sel]), fl)

        okf = fin & (np.abs(f[base]) >= 1e-3)
        rows.append(
            dict(
                loss=name,
                n=m,
                n_fd=int(smooth.sum()),
                kernel_vs_ref_loss_max_rel=float(np.max(rel(lk[base], lr_[base], fin))),
                kernel_vs_ref_grad_max_rel=float(np.max(rel(gk[base], gr_[base], fin))),
                # f = mu + sd o cancels in fp32 when |f| << mu: there the kernel sees a different f (the test's
                # input conditioning, not the loss); the same comparison for |f| >= 1e-3
                kernel_vs_ref_loss_max_rel_f_ge_1e3=float(
                    np.max(rel(lk[base], lr_[base], okf))
                ),
                kernel_vs_ref_grad_max_rel_f_ge_1e3=float(
                    np.max(rel(gk[base], gr_[base], okf))
                ),
                ref_grad_vs_ref_fd_max_rel=float(np.max(rel(gr_[base], fd_r, smooth))),
                kernel_grad_vs_kernel_fd_median_rel=float(
                    np.median(rel(gk[base], fd_k, smooth))
                ),
                kernel_grad_vs_kernel_fd_p95_rel=float(
                    np.quantile(rel(gk[base], fd_k, smooth), 0.95)
                ),
                n_capped=int(ck[base].sum()),
                cap_flags_agree=bool(np.array_equal(ck[base] > 0, cr_[base])),
                all_finite=bool(np.isfinite(lk).all() and np.isfinite(gk).all()),
                max_abs_grad=float(np.max(np.abs(gk[base]))),
            )
        )
    return pd.DataFrame(rows)


def gradcheck(run: Path) -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(3)
    for js in sorted((run / "gradcheck").glob("*.json")):
        jid = js.stem
        meta = json.loads(js.read_text())
        B, T, P, H, Lyr = meta["B"], meta["T"], meta["P"], meta["H"], meta["layers"]
        g = run / "gradcheck" / jid
        flat = read_f32(Path(f"{g}.params0.f32"))
        gn = read_f32(Path(f"{g}.grads.f32"))
        x = read_f32(Path(f"{g}.obs.f32")).reshape(B, T, P)
        y = read_f32(Path(f"{g}.y.f32"))
        mask = read_f32(Path(f"{g}.mask.f32"))
        decn = read_f32(Path(f"{g}.dec.f32")).reshape(B, T, 2)
        step = read_f32(Path(f"{g}.step.f32"))
        w = unpack(flat, P, H, Lyr)
        L, dec, gr = batch_loss(w, x, y, mask, meta, meta["n_real"])
        gref = pack(gr)
        # reference analytic gradient vs float64 central differences (a random subset of coordinates)
        idx = rng.choice(len(flat), size=min(len(flat), 300), replace=False)
        fd = np.empty(len(idx))
        for a, i in enumerate(idx):
            hh = 1e-6 * max(1.0, abs(flat[i]))
            fp = flat.copy()
            fp[i] += hh
            fm = flat.copy()
            fm[i] -= hh
            fd[a] = (
                batch_loss(
                    unpack(fp, P, H, Lyr), x, y, mask, meta, meta["n_real"], False
                )[0]
                - batch_loss(
                    unpack(fm, P, H, Lyr), x, y, mask, meta, meta["n_real"], False
                )[0]
            ) / (2 * hh)
        scale = np.max(np.abs(gref))
        rows.append(
            dict(
                job=jid,
                B=B,
                T=T,
                P=P,
                H=H,
                layers=Lyr,
                n_params=len(flat),
                loss_kind=meta["loss_kind"],
                fwd_max_abs_diff=float(np.max(np.abs(decn - dec))),
                loss_native=float(step[0] / meta["n_real"]),
                loss_ref=L,
                grad_native_vs_ref_max_abs_over_max=float(
                    np.max(np.abs(gn - gref)) / scale
                ),
                grad_native_vs_ref_rel_l2=float(
                    np.linalg.norm(gn - gref) / np.linalg.norm(gref)
                ),
                grad_ref_vs_fd_max_abs_over_max=float(
                    np.max(np.abs(gref[idx] - fd)) / scale
                ),
                grad_norm_native=float(np.linalg.norm(gn)),
                grad_norm_ref=float(np.linalg.norm(gref)),
            )
        )
    return pd.DataFrame(rows)


def trajectory(run: Path) -> pd.DataFrame:
    """The toy jobs (jobs_toy.txt) replayed in float64 from the native initial weights / minibatch order
    (the gradcheck dumps of the identically configured grad_toy_<loss> jobs)."""
    steps = pd.read_csv(run / "toy" / "steps.csv")
    jo = pd.read_csv(run / "toy" / "jobs_out.csv").set_index("id")
    preds = pd.read_csv(run / "toy" / "preds.csv")
    rows, traj = [], []
    for loss in ("mse", "qlike", "huber"):
        gid = run / "gradcheck" / f"grad_toy_{loss}"
        meta = json.loads(Path(f"{gid}.json").read_text())
        B, T, P, H, Lyr, S, N = (
            meta[k] for k in ("B", "T", "P", "H", "layers", "S", "N")
        )
        row0, n_ctx = meta["row0"], meta["n_ctx"]
        Z = read_f32(Path(f"{gid}.Z.f32")).reshape(n_ctx, P)
        yrow = read_f32(Path(f"{gid}.yrow.f32"))
        perm = np.frombuffer(Path(f"{gid}.perm.i32").read_bytes(), "<i4").reshape(
            -1, S * B
        )
        w = unpack(read_f32(Path(f"{gid}.params0.f32")), P, H, Lyr)
        mb = dict(
            We=np.zeros_like(w["We"]),
            Wd=np.zeros_like(w["Wd"]),
            Wl=[np.zeros_like(a) for a in w["Wl"]],
        )
        jid = f"toy_{loss}"
        E = int(jo.loc[jid, "epochs_run"])
        max_epochs = 40
        st = steps[steps["id"] == jid].sort_values(["epoch", "step"])
        lr0 = float(meta["lr"])
        k_all = 0
        for e in range(E):
            lr = 0.5 * lr0 * (1 + np.cos(np.pi * e / max_epochs))
            for k in range(S):
                s_idx = perm[e, k * B : (k + 1) * B]
                real = (s_idx >= 0) & (s_idx < N)
                r = row0 + np.where(real, s_idx, 0)
                x = Z[r[:, None] + np.arange(-T + 1, 1)[None, :]]
                yb = yrow[r]
                n_real = int(real.sum())
                L, _, g = batch_loss(w, x, yb, real.astype(float), meta, n_real)
                gnorm = muon_step(
                    w, g, mb, lr, float(meta["momentum"]), float(meta["max_grad_norm"])
                )
                nat = st.iloc[k_all]
                traj.append(
                    dict(
                        loss=loss,
                        epoch=e + 1,
                        step=k,
                        loss_native=nat["loss"],
                        loss_ref=L,
                        gn_native=nat["grad_norm"],
                        gn_ref=gnorm,
                    )
                )
                k_all += 1
        # final forecasts on the toy's forecast rows
        fr = preds[(preds["id"] == jid) & (preds["kind"] == "fc")].sort_values("row")
        rr = fr["row"].to_numpy() - 0  # toy w0 = 0: ctx row = design row
        x = Z[rr[:, None] + np.arange(-T + 1, 1)[None, :]]
        dec, _ = forward(w, x)
        f_ref = meta["mu_y"] + meta["sd_y"] * dec[:, -1, 0]
        t_ = pd.DataFrame([t for t in traj if t["loss"] == loss])
        rel = np.abs(t_["loss_native"] - t_["loss_ref"]) / np.abs(t_["loss_ref"])
        rows.append(
            dict(
                loss=loss,
                steps=len(t_),
                epochs=E,
                loss_first_native=float(t_["loss_native"].iloc[0]),
                loss_last_native=float(t_["loss_native"].iloc[-1]),
                loss_first_ref=float(t_["loss_ref"].iloc[0]),
                loss_last_ref=float(t_["loss_ref"].iloc[-1]),
                loss_max_rel_diff=float(rel.max()),
                loss_median_rel_diff=float(rel.median()),
                gn_max_rel_diff=float(
                    (np.abs(t_["gn_native"] - t_["gn_ref"]) / t_["gn_ref"]).max()
                ),
                final_fc_max_abs_diff=float(np.max(np.abs(fr["f"].to_numpy() - f_ref))),
            )
        )
    pd.DataFrame(traj).to_csv(OUT / "checks_trajectory_steps.csv", index=False)
    return pd.DataFrame(rows)


def main() -> None:
    run = Path(sys.argv[1])
    inputs = Path(sys.argv[2]) if len(sys.argv) > 2 else OUT / "_work"
    OUT.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)
    for name, fn in (
        ("losscheck", lambda: losscheck(run, inputs)),
        ("gradcheck", lambda: gradcheck(run)),
        ("trajectory", lambda: trajectory(run)),
    ):
        df = fn()
        df.to_csv(OUT / f"checks_{name}.csv", index=False)
        print(f"== {name}\n{df.to_string(index=False)}\n")


if __name__ == "__main__":
    main()
