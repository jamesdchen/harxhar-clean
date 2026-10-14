"""Speed benchmark (2026-10-13): PufferLib 5.0 native MinGRU trainer vs PyTorch eager, torch.compile and batched
PyTorch, on the same GPU in one job (cluster/puffer_forecast_bench_h2.sh).

The PyTorch model, Muon, standardisation, minibatch order and data loading are the other session's
experiments/puffer_forecast/torch_train.py, imported unchanged (its md5 is recorded); this file only adds the
variants and the timing.  Everything is fp32 (TF32 off, float32 matmul precision "highest"); every timed
region starts and ends with torch.cuda.synchronize(); compile / CUDA-graph warm-up runs on throwaway copies
of the state and is timed separately.

Suites
  A  the 24 jobs of inputs/jobs_scaling.txt (12 baseline + 12 all_features; L 5 / 20, H 16 / 64, three lrs;
     one seed), each for the epoch count the native trainer ran in THIS job (native K = 1 output), every epoch
     = the training minibatches + the validation forecast, as native tune jobs.
       torch_eager_k1      torch_train.fit_one's loop (MinGRU module, autograd, torch_train.muon_step)
       torch_compile_ro_k1 the same step with torch.compile(mode="reduce-overhead") (CUDA graphs) on the loss
                           (forward + AOT backward), the Muon step and the evaluation
       torch_compile_ma_k1 the same with mode="max-autotune" (CUDA graphs + autotuned kernels), under a budget
  B  K models trained TOGETHER (K = 8, 24, 48) per design: L 20, H 16, 30 epochs each (max_epochs 30, no
     early stopping), seeds 20261012 + k, lr cycling 0.003 / 0.01 / 0.03 -- one shape so the models can be
     batched; the native trainer runs the same K jobs concurrently (bench_B_<design>_K<K>.txt).
       torch_stacked_eager   stacked parameters [K, ...], batched matmuls, one autograd graph, batched Muon
       torch_stacked_compile the same under torch.compile(mode="reduce-overhead")
       torch_vmap_eager      torch.func.stack_module_state + vmap(grad(...)) over torch_train.MinGRU (K = 8)

    python torch_bench.py --root ROOT --native-a NATIVE_K1_DIR --out OUT [--ma-budget SECONDS]
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch_train as TT  # noqa: E402

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.set_float32_matmul_precision("highest")
DEV = torch.device(
    os.environ.get("BENCH_DEVICE", "cuda")
)  # cpu only for local logic tests


def sync() -> None:
    if DEV.type == "cuda":
        torch.cuda.synchronize()


def make_jobs(jobs_scaling: Path, out_dir: Path) -> None:
    """bench_A48.txt (the 24 scaling jobs twice: native K = 48 on the same jobs) and bench_B_<design>_K<K>.txt
    (K jobs of one shape: L 20, H 16, 30 epochs, no early stopping, seeds 20261012 + k, lr cycling)."""
    lines = [ln for ln in jobs_scaling.read_text().splitlines() if ln.strip()]
    dup = [ln.replace(ln.split()[0], ln.split()[0] + "_dup", 1) for ln in lines]
    (out_dir / "bench_A48.txt").write_bytes(("\n".join(lines + dup) + "\n").encode())
    ref: dict[str, dict[str, str]] = {}
    for ln in lines:
        kv = dict(t.split("=", 1) for t in ln.split())
        ref.setdefault(kv["data"], kv)
    for data, kv in ref.items():
        design = data.split("/")[-1]
        for K in (8, 24, 48):
            out = []
            for k in range(K):
                j = dict(kv)
                j.update(
                    id=f"benchB_{design}_K{K}_m{k}",
                    L="20",
                    H="16",
                    lr=("0.003", "0.01", "0.03")[k % 3],
                    seed=str(20261012 + k),
                    max_epochs="30",
                    patience="1000",
                )
                out.append(" ".join(f"{a}={b}" for a, b in j.items()))
            (out_dir / f"bench_B_{design}_K{K}.txt").write_bytes(
                ("\n".join(out) + "\n").encode()
            )


def cosine(lr0: float, e: int, max_epochs: int, min_ratio: float = 0.0) -> float:
    lo = min_ratio * lr0
    return lo + 0.5 * (lr0 - lo) * (1.0 + math.cos(math.pi * e / max_epochs))


# ============================================================================ single model
class Prepared:
    """One job's device data (torch_train.standardize / build_perm), shared by every variant."""

    def __init__(self, j: dict, raw, epochs: int):
        x, y = raw
        z, yrow, mu_y, sd_y, n, row0, eval_row0, n_eval = TT.standardize(x, y, j)
        self.j, self.epochs = j, epochs
        self.L, self.H, self.layers = int(j["L"]), int(j["H"]), TT.iget(j, "layers", 1)
        self.B = TT.iget(j, "batch", 128)
        self.n, self.row0 = n, row0
        self.S = (n + self.B - 1) // self.B
        self.P = z.shape[1]
        self.z = torch.from_numpy(z).to(DEV)
        self.yrow = torch.from_numpy(yrow).to(DEV)
        self.perm = torch.from_numpy(
            TT.build_perm(n, self.S, self.B, max(epochs, 1), int(j["seed"]))
        ).to(DEV)
        self.mu_t = torch.tensor(mu_y, device=DEV)
        self.sd_t = torch.tensor(sd_y, device=DEV)
        self.offs = torch.arange(-self.L + 1, 1, device=DEV)
        self.eval_ends = torch.arange(eval_row0, eval_row0 + n_eval, device=DEV)
        self.lr0 = float(j["lr"])
        self.max_epochs = TT.iget(j, "max_epochs", 200)
        self.mu = TT.fget(j, "momentum", 0.95)
        self.max_norm = TT.fget(j, "max_grad_norm", 1.0)

    def fresh(self):
        net = TT.MinGRU(self.P, self.H, self.layers).to(DEV)
        TT.init_weights(net, int(self.j["seed"]))
        params = [net.We, net.Wd, *list(net.Wl)]
        moms = [torch.zeros_like(p) for p in params]
        return net, params, moms


def single_forward(params, x):
    """torch_train.MinGRU.forward on a parameter list [We, Wd, *Wl] (compiled functions take tensors, not the
    module: dynamo guards on module identity would recompile for every new model)."""
    We, Wd, *Wl = params
    h = x @ We.T
    for wl in Wl:
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
    return h @ Wd.T


def loss_fn(params, z, yrow, slots, offs, mu_t, sd_t, n: int, row0: int, net=None):
    real = (slots >= 0) & (slots < n)
    ends = row0 + torch.where(real, slots, torch.zeros_like(slots))
    xb = z[ends[:, None] + offs[None, :]]
    yb = yrow[ends]
    mask = real.to(xb.dtype)
    dec = net(xb) if net is not None else single_forward(params, xb)
    f = mu_t + sd_t * dec[:, -1, 0]
    return (mask * (f - yb).square()).sum() / mask.sum()


def eval_fn(params, z, yrow, ends, offs, mu_t, sd_t, net=None):
    with torch.no_grad():
        x = z[ends[:, None] + offs[None, :]]
        dec = net(x) if net is not None else single_forward(params, x)
        f = mu_t + sd_t * dec[:, -1, 0]
        return (f - yrow[ends]).square().mean()


def muon_t(params, moms, lr_t, mu: float, max_norm: float):
    """torch_train.muon_step with the learning rate as a device scalar (no recompilation per epoch)."""
    with torch.no_grad():
        grads = [p.grad for p in params]
        sumsq = sum(g.square().sum() for g in grads)
        clip = torch.clamp(max_norm / (torch.sqrt(sumsq) + 1e-6), max=1.0)
        for p, m, g in zip(params, moms, grads):
            gk = g * clip
            m.mul_(mu).add_(gk)
            x = gk + mu * m
            r, c = x.shape
            x = x / torch.clamp(torch.sqrt(torch.sum(x * x)), min=1e-7)
            tall = r > c
            for a, b, cc in TT.NS:
                gram = x.T @ x if tall else x @ x.T
                bm = b * gram + cc * (gram @ gram)
                x = a * x + (x @ bm if tall else bm @ x)
            p.sub_(x * (lr_t * math.sqrt(max(1.0, r / c))))


def make_state(d: Prepared, compiled: bool):
    net, params, moms = d.fresh()
    init = [p.detach().clone() for p in params]
    lr_t = torch.zeros((), device=DEV)
    if compiled:
        for m in moms:
            torch._dynamo.mark_static_address(m)
        torch._dynamo.mark_static_address(lr_t)
    return dict(net=net, params=params, moms=moms, init=init, lr_t=lr_t)


def reset_state(st: dict) -> None:
    """Same tensors (same addresses: no CUDA-graph re-record), initial values, zero momentum."""
    with torch.no_grad():
        for p, i, m in zip(st["params"], st["init"], st["moms"]):
            p.copy_(i)
            p.grad = None
            m.zero_()


def train_single(
    d: Prepared, mode: str, fns: dict | None, epochs: int, st: dict
) -> float:
    """mode eager: torch_train.fit_one's step (MinGRU module, torch_train.muon_step); compiled: fns."""
    reset_state(st)
    net, params, moms, lr_t = st["net"], st["params"], st["moms"], st["lr_t"]
    sync()
    t0 = time.perf_counter()
    for e in range(epochs):
        lr = cosine(d.lr0, e, d.max_epochs)
        if mode == "eager":
            for k in range(d.S):
                loss = loss_fn(
                    params,
                    d.z,
                    d.yrow,
                    d.perm[e, k * d.B : (k + 1) * d.B],
                    d.offs,
                    d.mu_t,
                    d.sd_t,
                    d.n,
                    d.row0,
                    net=net,
                )
                loss.backward()
                TT.muon_step(params, moms, lr, d.mu, d.max_norm)
                net.zero_grad(set_to_none=True)
            eval_fn(params, d.z, d.yrow, d.eval_ends, d.offs, d.mu_t, d.sd_t, net=net)
        else:
            assert fns is not None
            lr_t.fill_(lr)
            for k in range(d.S):
                torch.compiler.cudagraph_mark_step_begin()
                loss = fns["loss"](
                    params,
                    d.z,
                    d.yrow,
                    d.perm[e, k * d.B : (k + 1) * d.B],
                    d.offs,
                    d.mu_t,
                    d.sd_t,
                    d.n,
                    d.row0,
                )
                loss.backward()
                fns["muon"](params, moms, lr_t, d.mu, d.max_norm)
                for p in params:
                    p.grad = None
            torch.compiler.cudagraph_mark_step_begin()
            fns["eval"](params, d.z, d.yrow, d.eval_ends, d.offs, d.mu_t, d.sd_t)
    sync()
    return time.perf_counter() - t0


def compiled_fns(mode: str) -> dict:
    kw = dict(mode=mode, dynamic=False)
    return dict(
        loss=torch.compile(loss_fn, **kw),
        muon=torch.compile(muon_t, **kw),
        eval=torch.compile(eval_fn, **kw),
    )


# ============================================================================ K models batched
class StackedData:
    """K jobs of one shape (suite B): shared window data, per-model minibatch order / lr."""

    def __init__(self, jobs: list[dict], raw, epochs: int):
        self.ds = [Prepared(j, raw, epochs) for j in jobs]
        d0 = self.ds[0]
        for d in self.ds:
            assert (d.P, d.L, d.H, d.n, d.S, d.B) == (
                d0.P,
                d0.L,
                d0.H,
                d0.n,
                d0.S,
                d0.B,
            )
            assert torch.equal(d.z, d0.z)
        self.K, self.d0, self.epochs = len(jobs), d0, epochs
        self.perm = torch.stack([d.perm for d in self.ds])  # [K, E, S*B]
        self.lr0 = torch.tensor([d.lr0 for d in self.ds], device=DEV)

    def fresh(self):
        nets = [d.fresh()[0] for d in self.ds]
        We = torch.nn.Parameter(torch.stack([n.We.detach() for n in nets]))
        Wd = torch.nn.Parameter(torch.stack([n.Wd.detach() for n in nets]))
        Wl = [
            torch.nn.Parameter(torch.stack([n.Wl[i].detach() for n in nets]))
            for i in range(len(nets[0].Wl))
        ]
        params = [We, Wd, *Wl]
        return params, [torch.zeros_like(p) for p in params]


def stacked_forward(params, x):
    """x [K, B, T, P] -> dec [K, B, T, 2]; torch_train.MinGRU.forward with a leading model axis."""
    We, Wd, *Wl = params
    h = torch.matmul(x, We.transpose(1, 2).unsqueeze(1))
    for wl in Wl:
        hn = wl.shape[2]
        comb = torch.matmul(h, wl.transpose(1, 2).unsqueeze(1))
        hid, gate, proj = comb[..., :hn], comb[..., hn : 2 * hn], comb[..., 2 * hn :]
        z = torch.sigmoid(gate)
        ht = torch.where(hid >= 0, hid + 0.5, torch.sigmoid(hid))
        s = torch.sigmoid(proj)
        hs = torch.zeros(h.shape[0], h.shape[1], hn, device=h.device, dtype=h.dtype)
        outs = []
        for t in range(h.shape[2]):
            hs = hs + z[:, :, t] * (ht[:, :, t] - hs)
            outs.append(s[:, :, t] * hs + (1.0 - s[:, :, t]) * h[:, :, t])
        h = torch.stack(outs, dim=2)
    return torch.matmul(h, Wd.transpose(1, 2).unsqueeze(1))


def stacked_loss(params, z, yrow, slots, offs, mu_t, sd_t, n: int, row0: int):
    """slots [K, B]; the sum over models of each model's mean squared error (independent gradients)."""
    real = (slots >= 0) & (slots < n)
    ends = row0 + torch.where(real, slots, torch.zeros_like(slots))
    xb = z[ends[..., None] + offs]
    yb = yrow[ends]
    mask = real.to(xb.dtype)
    dec = stacked_forward(params, xb)
    f = mu_t + sd_t * dec[:, :, -1, 0]
    return ((mask * (f - yb).square()).sum(1) / mask.sum(1)).sum()


def stacked_eval(params, z, yrow, ends, offs, mu_t, sd_t):
    with torch.no_grad():
        K = params[0].shape[0]
        x = z[ends[:, None] + offs[None, :]].unsqueeze(0).expand(K, -1, -1, -1)
        dec = stacked_forward(params, x)
        return (mu_t + sd_t * dec[:, :, -1, 0] - yrow[ends]).square().mean(1)


def stacked_muon(params, moms, lr_t, mu: float, max_norm: float):
    """torch_train.muon_step per model, batched over the leading axis (lr_t [K])."""
    with torch.no_grad():
        grads = [p.grad for p in params]
        sumsq = sum(g.square().sum(dim=(1, 2)) for g in grads)
        clip = torch.clamp(max_norm / (torch.sqrt(sumsq) + 1e-6), max=1.0)[
            :, None, None
        ]
        for p, m, g in zip(params, moms, grads):
            gk = g * clip
            m.mul_(mu).add_(gk)
            x = gk + mu * m
            r, c = x.shape[1], x.shape[2]
            x = x / torch.clamp(
                torch.sqrt(torch.sum(x * x, dim=(1, 2), keepdim=True)), min=1e-7
            )
            tall = r > c
            for a, b, cc in TT.NS:
                gram = (
                    torch.bmm(x.transpose(1, 2), x)
                    if tall
                    else torch.bmm(x, x.transpose(1, 2))
                )
                bm = b * gram + cc * torch.bmm(gram, gram)
                x = a * x + (torch.bmm(x, bm) if tall else torch.bmm(bm, x))
            p.sub_(x * (lr_t[:, None, None] * math.sqrt(max(1.0, r / c))))


def make_stacked_state(sd: StackedData, compiled: bool) -> dict:
    params, moms = sd.fresh()
    st = dict(
        params=params,
        moms=moms,
        init=[p.detach().clone() for p in params],
        lr_t=torch.zeros(sd.K, device=DEV),
    )
    if compiled:
        for m in moms:
            torch._dynamo.mark_static_address(m)
        torch._dynamo.mark_static_address(st["lr_t"])
    return st


def train_stacked(
    sd: StackedData, mode: str, fns: dict | None, epochs: int, st: dict
) -> float:
    d = sd.d0
    reset_state(st)
    params, moms, lr_t = st["params"], st["moms"], st["lr_t"]
    f_loss = fns["loss"] if fns else stacked_loss
    f_muon = fns["muon"] if fns else stacked_muon
    f_eval = fns["eval"] if fns else stacked_eval
    sync()
    t0 = time.perf_counter()
    for e in range(epochs):
        u = e / d.max_epochs
        lr_t.copy_(0.5 * sd.lr0 * (1.0 + math.cos(math.pi * u)))
        for k in range(d.S):
            if mode != "eager":
                torch.compiler.cudagraph_mark_step_begin()
            loss = f_loss(
                params,
                d.z,
                d.yrow,
                sd.perm[:, e, k * d.B : (k + 1) * d.B],
                d.offs,
                d.mu_t,
                d.sd_t,
                d.n,
                d.row0,
            )
            loss.backward()
            f_muon(params, moms, lr_t, d.mu, d.max_norm)
            for p in params:
                p.grad = None
        if mode != "eager":
            torch.compiler.cudagraph_mark_step_begin()
        f_eval(params, d.z, d.yrow, d.eval_ends, d.offs, d.mu_t, d.sd_t)
    sync()
    return time.perf_counter() - t0


def train_vmap(sd: StackedData, epochs: int) -> float:
    """torch.func: stack_module_state over K torch_train.MinGRU modules, vmap(grad(loss)) per step, batched Muon."""
    from torch.func import functional_call, grad, stack_module_state, vmap

    d = sd.d0
    nets = [x.fresh()[0] for x in sd.ds]
    pdict, _ = stack_module_state(nets)
    base = TT.MinGRU(d.P, d.H, d.layers).to("meta")
    names = list(pdict)

    def single_loss(p, slots, z, yrow, offs, mu_t, sd_t):
        real = (slots >= 0) & (slots < d.n)
        ends = d.row0 + torch.where(real, slots, torch.zeros_like(slots))
        xb = z[ends[:, None] + offs[None, :]]
        mask = real.to(xb.dtype)
        dec = functional_call(base, p, (xb,))
        f = mu_t + sd_t * dec[:, -1, 0]
        return (mask * (f - yrow[ends]).square()).sum() / mask.sum()

    g_fn = vmap(grad(single_loss), in_dims=(0, 0, None, None, None, None, None))
    params = [pdict[nm] for nm in names]
    moms = [torch.zeros_like(p) for p in params]
    lr_t = torch.zeros(sd.K, device=DEV)
    sync()
    t0 = time.perf_counter()
    for e in range(epochs):
        lr_t.copy_(0.5 * sd.lr0 * (1.0 + math.cos(math.pi * e / d.max_epochs)))
        for k in range(d.S):
            gs = g_fn(
                dict(zip(names, params)),
                sd.perm[:, e, k * d.B : (k + 1) * d.B],
                d.z,
                d.yrow,
                d.offs,
                d.mu_t,
                d.sd_t,
            )
            for p, nm in zip(params, names):
                p.grad = gs[nm]
            stacked_muon(params, moms, lr_t, d.mu, d.max_norm)
        with torch.no_grad():
            ev = vmap(
                lambda p: functional_call(
                    base, p, (d.z[d.eval_ends[:, None] + d.offs[None, :]],)
                )
            )(dict(zip(names, params)))
            (d.mu_t + d.sd_t * ev[:, :, -1, 0] - d.yrow[d.eval_ends]).square().mean(1)
    sync()
    return time.perf_counter() - t0


# ============================================================================ checks
def check_stacked_vs_single(raw, jobs: list[dict]) -> float:
    """3 models: stacked (bmm) vs 3 single models (torch_train ops), one epoch: max |param diff| / max |param|."""
    sd = StackedData(jobs[:3], raw, 1)
    params, moms = sd.fresh()
    d = sd.d0
    lr_t = torch.zeros(sd.K, device=DEV)
    singles = [x.fresh() for x in sd.ds]
    lr_t.copy_(sd.lr0)
    for k in range(d.S):
        loss = stacked_loss(
            params,
            d.z,
            d.yrow,
            sd.perm[:, 0, k * d.B : (k + 1) * d.B],
            d.offs,
            d.mu_t,
            d.sd_t,
            d.n,
            d.row0,
        )
        loss.backward()
        stacked_muon(params, moms, lr_t, d.mu, d.max_norm)
        for p in params:
            p.grad = None
        for i, (net, ps, ms) in enumerate(singles):
            li = loss_fn(
                ps,
                d.z,
                d.yrow,
                sd.perm[i, 0, k * d.B : (k + 1) * d.B],
                d.offs,
                d.mu_t,
                d.sd_t,
                d.n,
                d.row0,
                net=net,
            )
            li.backward()
            TT.muon_step(ps, ms, float(sd.lr0[i]), d.mu, d.max_norm)
            net.zero_grad(set_to_none=True)
    worst = 0.0
    for j_, p in enumerate(params):
        for i, (_, ps, _) in enumerate(singles):
            worst = max(
                worst,
                float(
                    (p[i] - ps[j_]).detach().abs().max() / ps[j_].detach().abs().max()
                ),
            )
    return worst


# ============================================================================ main
def md5(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


def main() -> None:  # noqa: C901 -- one benchmark
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument(
        "--native-a",
        type=Path,
        required=True,
        help="native K = 1 run dir of suite A (epoch counts)",
    )
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument(
        "--ma-budget",
        type=float,
        default=1500.0,
        help="seconds for the max-autotune variant",
    )
    ap.add_argument("--only", default="", help="comma list of variants (debug)")
    ap.add_argument(
        "--make-jobs",
        type=Path,
        default=None,
        help="write the bench job files into this dir, exit",
    )
    args = ap.parse_args()
    if args.make_jobs:
        make_jobs(args.root / "inputs" / "jobs_scaling.txt", args.make_jobs)
        return
    args.out.mkdir(parents=True, exist_ok=True)
    only = set(filter(None, args.only.split(",")))

    def run(v: str) -> bool:
        return not only or v in only

    info = dict(
        torch=torch.__version__,
        cuda=torch.version.cuda,
        device=torch.cuda.get_device_name(0) if DEV.type == "cuda" else "cpu",
        capability=torch.cuda.get_device_capability(0) if DEV.type == "cuda" else None,
        tf32=torch.backends.cuda.matmul.allow_tf32,
        torch_train_md5=md5(Path(TT.__file__)),
        torch_bench_md5=md5(Path(__file__)),
    )
    try:
        import triton

        info["triton"] = triton.__version__
    except Exception as exc:  # noqa: BLE001
        info["triton"] = f"unavailable: {exc}"
    print(json.dumps(info), flush=True)
    (args.out / "torch_info.json").write_text(json.dumps(info, indent=1))
    TT.selfcheck(DEV)
    rows: list[dict] = []
    out_csv = args.out / "torch_runs.csv"

    def emit(r: dict) -> None:
        rows.append(r)
        with out_csv.open("w", newline="") as f:
            w = csv.DictWriter(f, sorted({k for x in rows for k in x}))
            w.writeheader()
            w.writerows(rows)
        print(json.dumps(r), flush=True)

    raw_cache: dict = {}

    def raw_of(j: dict):
        if j["data"] not in raw_cache:
            raw_cache[j["data"]] = TT.load_raw(args.root, j["data"])
        return raw_cache[j["data"]]

    # ---------------------------------------------------------------- suite A (K = 1)
    jobs_a = TT.parse_jobs(args.root / "inputs" / "jobs_scaling.txt")
    nat = {r["id"]: r for r in csv.DictReader((args.native_a / "jobs_out.csv").open())}
    eps = {j["id"]: int(nat[j["id"]]["epochs_run"]) for j in jobs_a}
    # global warm-up: CUDA context, cuBLAS handles, allocator (one eager epoch of the first job, discarded)
    d0 = Prepared(jobs_a[0], raw_of(jobs_a[0]), 1)
    train_single(d0, "eager", None, 1, make_state(d0, False))
    variants_a = [
        ("torch_eager_k1", None),
        ("torch_compile_ro_k1", "reduce-overhead"),
        ("torch_compile_ma_k1", "max-autotune"),
    ]
    for vname, cmode in variants_a:
        if not run(vname):
            continue
        torch._dynamo.reset()
        torch._dynamo.config.cache_size_limit = 64
        fns = compiled_fns(cmode) if cmode else None
        mode = "compiled" if cmode else "eager"
        t_var0 = time.perf_counter()
        for j in jobs_a:
            if (
                cmode == "max-autotune"
                and time.perf_counter() - t_var0 > args.ma_budget
            ):
                emit(
                    dict(
                        suite="A",
                        variant=vname,
                        K=1,
                        id=j["id"],
                        skipped="max-autotune budget exhausted",
                    )
                )
                continue
            d = Prepared(j, raw_of(j), eps[j["id"]])
            st = make_state(d, bool(cmode))
            # warm-up (excluded): compile + CUDA-graph recording for compiled modes; 1 epoch for eager
            try:
                tw = time.perf_counter()
                train_single(d, mode, fns, 2 if cmode else 1, st)
                t_warm = time.perf_counter() - tw
                t = train_single(d, mode, fns, d.epochs, st)
            except Exception as exc:  # noqa: BLE001 -- a failed compile is a result, not a crash
                emit(
                    dict(
                        suite="A", variant=vname, K=1, id=j["id"], error=repr(exc)[:400]
                    )
                )
                continue
            emit(
                dict(
                    suite="A",
                    variant=vname,
                    K=1,
                    id=j["id"],
                    epochs=d.epochs,
                    steps=d.epochs * d.S,
                    t_train=t,
                    t_warmup=t_warm,
                    P=d.P,
                    L=d.L,
                    H=d.H,
                )
            )
    # ---------------------------------------------------------------- suite B (K models together)
    for design in ("baseline", "all_features"):
        for K in (8, 24, 48):
            jobs_b = TT.parse_jobs(args.root / "inputs" / f"bench_B_{design}_K{K}.txt")
            E = TT.iget(jobs_b[0], "max_epochs", 30)
            sd = StackedData(jobs_b, raw_of(jobs_b[0]), E)
            if run("check") and K == 8:
                emit(
                    dict(
                        suite="B",
                        variant="check_stacked_vs_single",
                        design=design,
                        K=3,
                        max_rel_param_diff_after_1_epoch=check_stacked_vs_single(
                            raw_of(jobs_b[0]), jobs_b
                        ),
                    )
                )
            if run("torch_stacked_eager"):
                st = make_stacked_state(sd, False)
                tw = time.perf_counter()
                train_stacked(sd, "eager", None, 1, st)
                t_warm = time.perf_counter() - tw
                t = train_stacked(sd, "eager", None, E, st)
                emit(
                    dict(
                        suite="B",
                        variant="torch_stacked_eager",
                        design=design,
                        K=K,
                        epochs=K * E,
                        steps=E * sd.d0.S,
                        t_train=t,
                        t_warmup=t_warm,
                    )
                )
            if run("torch_stacked_compile"):
                torch._dynamo.reset()
                kw = dict(mode="reduce-overhead", dynamic=False)
                fns = dict(
                    loss=torch.compile(stacked_loss, **kw),
                    muon=torch.compile(stacked_muon, **kw),
                    eval=torch.compile(stacked_eval, **kw),
                )
                st = make_stacked_state(sd, True)
                try:
                    tw = time.perf_counter()
                    train_stacked(sd, "compiled", fns, 2, st)
                    t_warm = time.perf_counter() - tw
                    t = train_stacked(sd, "compiled", fns, E, st)
                except Exception as exc:  # noqa: BLE001
                    emit(
                        dict(
                            suite="B",
                            variant="torch_stacked_compile_ro",
                            design=design,
                            K=K,
                            error=repr(exc)[:400],
                        )
                    )
                    continue
                emit(
                    dict(
                        suite="B",
                        variant="torch_stacked_compile_ro",
                        design=design,
                        K=K,
                        epochs=K * E,
                        steps=E * sd.d0.S,
                        t_train=t,
                        t_warmup=t_warm,
                    )
                )
            if run("torch_vmap_eager") and K == 8:
                try:
                    tw = time.perf_counter()
                    train_vmap(sd, 1)
                    t_warm = time.perf_counter() - tw
                    t = train_vmap(sd, E)
                    emit(
                        dict(
                            suite="B",
                            variant="torch_vmap_eager",
                            design=design,
                            K=K,
                            epochs=K * E,
                            steps=E * sd.d0.S,
                            t_train=t,
                            t_warmup=t_warm,
                        )
                    )
                except Exception as exc:  # noqa: BLE001
                    emit(
                        dict(
                            suite="B",
                            variant="torch_vmap_eager",
                            design=design,
                            K=K,
                            error=repr(exc)[:300],
                        )
                    )
    try:
        from torch._dynamo.utils import counters

        (args.out / "torch_counters.json").write_text(
            json.dumps({k: dict(v) for k, v in counters.items()}, indent=1, default=str)
        )
    except Exception:  # noqa: BLE001
        pass
    print("TORCH_BENCH_DONE", flush=True)


if __name__ == "__main__":
    main()
