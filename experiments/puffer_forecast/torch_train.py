"""PyTorch MinGRU forecaster, same stack as the native PufferLib trainer.

Matches experiments/puffer_forecast/reference.py: encoder x @ We^T, MinGRU + highway,
2-column decoder, last-step MSE in sqrt space (f = mu_y + sd_y * o), Muon with the
upstream Newton-Schulz coefficients, cosine LR, causal standardization, splitmix64
minibatches. Tensors live on CUDA. This is the like-for-like timing against
runs/13037177 scale_k1 (one L40S, one model at a time, the same epoch counts).

    python experiments/puffer_forecast/torch_train.py --jobs JOBS --epochs-from JOBS_OUT --out DIR
"""

from __future__ import annotations

import argparse
import csv
import math
import time
from pathlib import Path

import numpy as np
import torch

NS = (
    (4.0848, -6.8946, 2.9270),
    (3.9505, -6.3029, 2.6377),
    (3.7418, -5.5913, 2.3037),
    (2.8769, -3.1427, 1.2046),
    (2.8366, -3.0525, 1.2012),
)


def splitmix64(s: int) -> tuple[int, int]:
    s = (s + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    z = s
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    z = (z ^ (z >> 31)) & 0xFFFFFFFFFFFFFFFF
    return s, z


def parse_jobs(path: Path) -> list[dict]:
    jobs = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        kv = {}
        for tok in line.split():
            if "=" in tok:
                k, v = tok.split("=", 1)
                kv[k] = v
        jobs.append(kv)
    return jobs


def iget(j: dict, k: str, d: int) -> int:
    return int(j[k]) if k in j else d


def fget(j: dict, k: str, d: float) -> float:
    return float(j[k]) if k in j else d


class MinGRU(torch.nn.Module):
    def __init__(self, p: int, h: int, layers: int):
        super().__init__()
        self.We = torch.nn.Parameter(torch.empty(h, p))
        self.Wd = torch.nn.Parameter(torch.empty(2, h))
        self.Wl = torch.nn.ParameterList(
            torch.nn.Parameter(torch.empty(3 * h, h)) for _ in range(layers)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x @ self.We.T
        for wl in self.Wl:
            hid_n = wl.shape[1]
            comb = h @ wl.T
            hid, gate, proj = (
                comb[..., :hid_n],
                comb[..., hid_n : 2 * hid_n],
                comb[..., 2 * hid_n :],
            )
            z = torch.sigmoid(gate)
            ht = torch.where(hid >= 0, hid + 0.5, torch.sigmoid(hid))
            s = torch.sigmoid(proj)
            hs = torch.zeros(h.shape[0], hid_n, device=h.device, dtype=h.dtype)
            outs = []
            for t in range(h.shape[1]):
                hs = hs + z[:, t] * (ht[:, t] - hs)
                outs.append(s[:, t] * hs + (1.0 - s[:, t]) * h[:, t])
            h = torch.stack(outs, dim=1)
        return h @ self.Wd.T


def init_weights(net: MinGRU, seed: int) -> None:
    """Seeded fan-in uniform. The native fill is curand inside algo.cu; the op mix does not depend on it."""
    g = torch.Generator(device=net.We.device)
    g.manual_seed(seed)
    with torch.no_grad():
        for p in net.parameters():
            bound = 1.0 / math.sqrt(p.shape[1])
            p.uniform_(-bound, bound, generator=g)


def muon_step(params, moms, lr: float, mu: float, max_norm: float) -> torch.Tensor:
    grads = [p.grad for p in params]
    sumsq = torch.zeros((), device=grads[0].device, dtype=torch.float32)
    for g in grads:
        sumsq = sumsq + g.detach().float().square().sum()
    clip = torch.clamp(max_norm / (torch.sqrt(sumsq) + 1e-6), max=1.0)
    with torch.no_grad():
        for p, m, g in zip(params, moms, grads):
            gk = g.detach() * clip
            m.mul_(mu).add_(gk)
            x = gk + mu * m
            r, c = x.shape
            x = x / torch.clamp(torch.sqrt(torch.sum(x * x)), min=1e-7)
            tall = r > c
            for a, b, cc in NS:
                gram = x.T @ x if tall else x @ x.T
                bm = b * gram + cc * (gram @ gram)
                x = a * x + (x @ bm if tall else bm @ x)
            p.add_(x, alpha=-lr * math.sqrt(max(1.0, r / c)))
    return torch.sqrt(sumsq)


def load_raw(root: Path, name: str):
    d = root / name
    n, p = (int(x) for x in (d / "meta.txt").read_text().split())
    x = np.fromfile(d / "X.f32", dtype="<f4").reshape(n, p)
    y = np.fromfile(d / "y.f32", dtype="<f4")
    return x, y


def standardize(x: np.ndarray, y: np.ndarray, j: dict):
    w0, w1 = int(j["w0"]), int(j["w1"])
    w = w1 - w0
    tune = j.get("mode", "tune") == "tune"
    val_tail, embargo, ell, fc_n = (
        iget(j, "val_tail", 125),
        iget(j, "embargo", 25),
        int(j["L"]),
        iget(j, "fc_n", 10),
    )
    f_end = (w - val_tail - embargo) if tune else w
    row0 = ell - 1
    n = f_end - row0
    n_ctx = w if tune else w + fc_n
    block = x[w0 : w0 + f_end].astype(np.float64)
    mu = block.mean(axis=0)
    sd = np.sqrt(((block - mu) ** 2).sum(axis=0) / f_end)
    sd = np.where(sd > 0, sd, 1.0)
    ctx = x[w0 : w0 + n_ctx].astype(np.float64)
    z = (ctx - mu) / sd
    z = np.concatenate([z, np.ones((n_ctx, 1))], axis=1).astype(np.float32)
    yrow = np.zeros(n_ctx, np.float32)
    yrow[:w] = y[w0 : w0 + w]
    ytr = y[w0 + row0 : w0 + f_end].astype(np.float64)
    mu_y = float(ytr.mean())
    sd_y = float(np.sqrt(((ytr - mu_y) ** 2).sum() / len(ytr)))
    return (
        z,
        yrow,
        mu_y,
        sd_y,
        n,
        row0,
        (w - val_tail if tune else w),
        (val_tail if tune else fc_n),
    )


def build_perm(n: int, s: int, b: int, epochs: int, seed: int) -> np.ndarray:
    perm = np.full((epochs, s * b), -1, np.int64)
    rs = (0x5EED0000 + seed * 1000003) & 0xFFFFFFFFFFFFFFFF
    for e in range(epochs):
        idx = list(range(n))
        for i in range(n - 1, 0, -1):
            rs, z = splitmix64(rs)
            k = z % (i + 1)
            idx[i], idx[k] = idx[k], idx[i]
        perm[e, :n] = idx
    return perm


def sequences(z: torch.Tensor, ends: torch.Tensor, t: int) -> torch.Tensor:
    offs = torch.arange(-t + 1, 1, device=z.device)
    return z[ends[:, None] + offs[None, :]]


def fit_one(j: dict, raw, epochs: int, device: torch.device) -> dict:
    x, y = raw
    z_np, yrow_np, mu_y, sd_y, n, row0, eval_row0, n_eval = standardize(x, y, j)
    ell, h, layers = int(j["L"]), int(j["H"]), iget(j, "layers", 1)
    b = iget(j, "batch", 128)
    s = (n + b - 1) // b
    p = z_np.shape[1]
    t0 = time.perf_counter()
    z = torch.from_numpy(z_np).to(device)
    yrow = torch.from_numpy(yrow_np).to(device)
    perm = torch.from_numpy(build_perm(n, s, b, epochs, int(j["seed"]))).to(device)
    net = MinGRU(p, h, layers).to(device)
    init_weights(net, int(j["seed"]))
    params = [net.We, net.Wd, *list(net.Wl)]
    moms = [torch.zeros_like(p_) for p_ in params]
    lr0 = float(j["lr"])
    max_epochs = iget(j, "max_epochs", 200)
    mu = fget(j, "momentum", 0.95)
    max_norm = fget(j, "max_grad_norm", 1.0)
    lr_min = fget(j, "min_lr_ratio", 0.0) * lr0
    mu_t = torch.tensor(mu_y, device=device)
    sd_t = torch.tensor(sd_y, device=device)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t_setup = time.perf_counter() - t0

    t1 = time.perf_counter()
    for e in range(epochs):
        u = e / max_epochs
        lr = lr_min + 0.5 * (lr0 - lr_min) * (1.0 + math.cos(math.pi * u))
        for k in range(s):
            n_real = min(b, n - k * b)
            slots = perm[e, k * b : (k + 1) * b]
            real = (slots >= 0) & (slots < n)
            ends = row0 + torch.where(real, slots, torch.zeros_like(slots))
            xb = sequences(z, ends, ell)
            yb = yrow[ends]
            mask = real.to(xb.dtype)
            dec = net(xb)
            f = mu_t + sd_t * dec[:, -1, 0]
            loss = (mask * (f - yb).square()).sum() / n_real
            loss.backward()
            muon_step(params, moms, lr, mu, max_norm)
            net.zero_grad(set_to_none=True)
        with torch.no_grad():
            ends = torch.arange(eval_row0, eval_row0 + n_eval, device=device)
            dec = net(sequences(z, ends, ell))
            f = mu_t + sd_t * dec[:, -1, 0]
            (f - yrow[ends]).square().mean()
    if device.type == "cuda":
        torch.cuda.synchronize()
    t_train = time.perf_counter() - t1
    dev = next(net.parameters()).device
    return dict(
        id=j["id"],
        epochs=epochs,
        steps=epochs * s,
        t_setup=t_setup,
        t_train=t_train,
        epochs_per_s=epochs / t_train,
        device=str(dev),
        n_params=sum(p_.numel() for p_ in params),
    )


def selfcheck(device: torch.device) -> None:
    """Autograd of the last-step MSE against float64 central differences, on the GPU."""
    g = torch.Generator(device="cpu")
    g.manual_seed(0)
    b, t, p, h = 4, 3, 5, 4
    net = MinGRU(p, h, 1).to(device=device, dtype=torch.float64)
    with torch.no_grad():
        for w in net.parameters():
            w.copy_(torch.randn(w.shape, generator=g, dtype=torch.float64))
    x = torch.randn(b, t, p, generator=g, dtype=torch.float64, device="cpu").to(device)
    y = torch.randn(b, generator=g, dtype=torch.float64, device="cpu").to(device)
    dec = net(x)
    loss = (dec[:, -1, 0] - y).square().mean()
    loss.backward()
    grads = [w.grad.detach().clone() for w in net.parameters()]
    flat0 = torch.cat([w.detach().flatten() for w in net.parameters()])
    worst = 0.0
    for i in (0, flat0.numel() // 3, flat0.numel() // 2, flat0.numel() - 1):
        hh = 1e-6 * max(1.0, abs(float(flat0[i])))

        def loss_at(delta, idx=i):
            with torch.no_grad():
                k = 0
                for w in net.parameters():
                    n = w.numel()
                    if idx < k + n:
                        w.view(-1)[idx - k] += delta
                        break
                    k += n
            with torch.no_grad():
                out = (net(x)[:, -1, 0] - y).square().mean()
            with torch.no_grad():
                k = 0
                for w in net.parameters():
                    n = w.numel()
                    if idx < k + n:
                        w.view(-1)[idx - k] -= delta
                        break
                    k += n
            return float(out)

        fd = (loss_at(hh) - loss_at(-hh)) / (2 * hh)
        k = 0
        ag = 0.0
        for wgrad in grads:
            n = wgrad.numel()
            if i < k + n:
                ag = float(wgrad.view(-1)[i - k])
                break
            k += n
        worst = max(worst, abs(ag - fd) / max(1.0, abs(fd)))
    print(
        f"selfcheck device={next(net.parameters()).device} grad_vs_fd_max_rel={worst:.3e}",
        flush=True,
    )
    if worst > 1e-4:
        raise SystemExit(f"selfcheck failed: grad_vs_fd_max_rel={worst}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=Path, required=True)
    ap.add_argument("--epochs-from", type=Path, required=True)
    ap.add_argument("--data-root", type=Path, default=Path("data"))
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit(
            f"pytorch is not using a GPU ({torch.__version__}, cuda {torch.version.cuda})"
        )
    device = torch.device("cuda")
    print(
        f"torch {torch.__version__} cuda {torch.version.cuda} device {torch.cuda.get_device_name(0)}",
        flush=True,
    )
    selfcheck(device)
    native = {}
    with args.epochs_from.open(newline="") as f:
        for row in csv.DictReader(f):
            native[row["id"]] = row
    args.out.mkdir(parents=True, exist_ok=True)
    out_csv = args.out / "jobs_out.csv"
    fields = [
        "id",
        "epochs",
        "steps",
        "t_setup",
        "t_train",
        "epochs_per_s",
        "native_t_train",
        "device",
        "n_params",
    ]
    cache = {}
    rows = []
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fields)
        w.writeheader()
        for j in parse_jobs(args.jobs):
            jid = j["id"]
            if jid not in native:
                raise SystemExit(f"{jid} missing from {args.epochs_from}")
            key = j["data"]
            if key not in cache:
                cache[key] = load_raw(args.data_root, key)
            epochs = int(native[jid]["epochs_run"])
            rec = fit_one(j, cache[key], epochs, device)
            rec["native_t_train"] = float(native[jid]["t_train"])
            w.writerow(rec)
            f.flush()
            rows.append(rec)
            print(
                f"{jid} epochs={rec['epochs']} t_train={rec['t_train']:.3f}s "
                f"native={rec['native_t_train']:.4f}s device={rec['device']}",
                flush=True,
            )
    ours = sum(r["t_train"] for r in rows)
    nat = sum(r["native_t_train"] for r in rows)
    ep = sum(r["epochs"] for r in rows)
    print(
        f"TORCH_DONE models={len(rows)} epochs={ep} pytorch_train_s={ours:.3f} "
        f"native_train_s={nat:.3f} pytorch_over_native={ours / nat:.2f} "
        f"pytorch_epochs_per_s={ep / ours:.2f} native_epochs_per_s={ep / nat:.2f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
