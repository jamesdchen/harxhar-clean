"""The 13 per-bar arms in ONE pass of the spec: load and transform once, 13 backtests.

The path of record (``forecast.run_arms``) starts ``specs/causal_tune_linear.py``
13 times, once per bar.  Each process loads the ~330k-row panel, transforms it
and builds the HAR + calendar matrix (``run_executor`` up to the segment slice),
backtests its one bar, and then also runs the spec's OLS incumbent and metrics
table, which the forecast never reads.  With ``LAG_SCOPE=global`` the matrix
before the slice does not depend on the bar: it is the same function of the
same files in all 13 processes.  This module computes it once and hands the 13
sliced frames to the SAME ``_backtest_and_save`` with the SAME ``fit_predict``.

Nothing is re-derived.  The spec's own source supplies everything:

* its definitions -- imports, constants, ``RollingTunedLinear``,
  ``fit_predict_lin_tuned`` -- are executed from its AST, minus the two
  pinned demonstration cells (``load_raw_data`` / ``robust_transform`` on a
  ``df`` nothing later reads) and everything from the arm loop on;
* the arm's ``run_executor(...)`` keyword arguments are read off the call
  inside that loop and evaluated in the spec's namespace, after the loop's own
  ``RollingTunedLinear.grid`` and ``out_csv`` statements;
* ``run_executor`` itself runs, with ``SEGMENT=all`` over a segment table
  restricted to the 13 bars and with ``_backtest_and_save`` collected instead
  of called -- so the validation, ``load_and_transform``, the HAR build and the
  per-segment train window are the executor's code, not a copy of it.

The collected backtests run in spawned worker processes (the arm's environment:
``OMP_NUM_THREADS=1``, the HPC_KW_* settings), each of which executes the
spec's definitions itself; each is submitted the moment its segment is sliced.
The same workers also take the 14 per-column ``robust_transform`` calls and
the expiry extractor while the executor builds the matrix (``_Farm``: the
executor's functions on exactly the inputs they read, verified at the call,
inline otherwise).  Every results CSV lands where the spec puts it
(``$HPC_RESULT_DIR/causal_tune_linear/<estimator>/<bucket>/results_<bar>.csv``).
``fastpath_check.py`` is the gate: the CSVs, rv_hat and the card equal the
path of record's.

Two ways to run it, both with the ext root as the working directory:

    python -m live.close_signal.arms_shared            # one pass, then exit
    python -m live.close_signal.arms_shared --serve    # warm: one pass per stdin line

``--serve`` is what the daily run starts in its precompute phase (~15:15 ET):
the interpreter, the spec's imports and the worker pool are up, and a canary
pass on the 15:00 panel has run through every code path, before the 15:30 row
exists.  At 15:30:30 the run rewrites ``data/`` and sends one line; only the
matrix and the 13 backtests remain.  ``ArmServer`` is the parent's handle.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import subprocess
import sys
import threading
import time
import traceback
from concurrent.futures import Executor, ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path
from typing import IO, Any

SPEC_REL = Path("specs") / "causal_tune_linear.py"
#: The pinned demonstration cells: they load and transform a ``df`` that the
#: arms never read (run_executor re-does both internally).
DEMO_NAMES = frozenset({"df", "adj_rv", "rv_baseline"})
DEMO_CALLS = frozenset({"load_raw_data", "robust_transform"})
#: Every regular-hours bar, the order forecast.BARS uses.
BARS: tuple[str, ...] = tuple(
    f"bar{h:02d}{m:02d}" for h in range(10, 17) for m in (0, 30) if (h, m) <= (16, 0)
)
READY = "ARMS_SHARED_READY"
DONE = "ARMS_SHARED_DONE"
FAIL = "ARMS_SHARED_FAIL"


def _names(node: ast.AST) -> set[str]:
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}


def _calls(node: ast.AST) -> set[str]:
    return {
        n.func.id
        for n in ast.walk(node)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
    }


def spec_parts(spec_path: Path) -> tuple[ast.Module, list[ast.stmt], ast.Call]:
    """(definitions, the arm loop's setup statements, the arm's run_executor call)."""
    src = Path(spec_path).read_text(encoding="utf-8")
    tree = ast.parse(src, filename=str(spec_path))
    head: list[ast.stmt] = []
    loop: ast.For | None = None
    for node in tree.body:
        if (
            isinstance(node, ast.For)
            and isinstance(node.target, ast.Name)
            and node.target.id == "estimator"
        ):
            loop = node
            break
        if _names(node) & DEMO_NAMES or _calls(node) & DEMO_CALLS:
            continue
        head.append(node)
    if loop is None:
        raise RuntimeError(f"{spec_path}: no `for estimator in ...` arm loop")
    setup: list[ast.stmt] = []
    call: ast.Call | None = None
    for sub in ast.walk(loop):
        if isinstance(sub, ast.Assign) and len(sub.targets) == 1:
            t = sub.targets[0]
            if (isinstance(t, ast.Attribute) and t.attr == "grid") or (
                isinstance(t, ast.Name) and t.id == "out_csv"
            ):
                setup.append(sub)
        if (
            isinstance(sub, ast.Call)
            and isinstance(sub.func, ast.Name)
            and sub.func.id == "run_executor"
            and call is None
        ):
            call = sub
    if call is None or len(setup) != 2 or call.args:
        raise RuntimeError(
            f"{spec_path}: the arm loop no longer has the grid / out_csv / "
            "run_executor(**keywords) shape this driver reads"
        )
    return ast.Module(body=head, type_ignores=[]), setup, call


def spec_namespace(spec_path: Path) -> dict[str, Any]:
    """Execute the spec's definitions (its env-derived settings included)."""
    defs, _, _ = spec_parts(spec_path)
    ns: dict[str, Any] = {"__name__": "causal_tune_linear", "__file__": str(spec_path)}
    exec(compile(defs, str(spec_path), "exec"), ns)  # noqa: S102 -- the spec's own code
    return ns


def arm_kwargs(
    spec_path: Path, ns: dict[str, Any], estimator: str, bucket: str
) -> dict[str, Any]:
    """The arm's run_executor keywords, evaluated as the loop evaluates them."""
    _, setup, call = spec_parts(spec_path)
    ns["estimator"], ns["bucket"] = estimator, bucket
    exec(compile(ast.Module(body=setup, type_ignores=[]), str(spec_path), "exec"), ns)  # noqa: S102
    return {
        kw.arg: eval(  # noqa: S307 -- the spec's own expressions
            compile(ast.Expression(body=kw.value), str(spec_path), "eval"), ns
        )
        for kw in call.keywords
        if kw.arg is not None
    }


# ------------------------------------------------------------------ workers --
_WORKER: dict[str, Any] = {}


def _init_worker(spec_path: str, estimator: str) -> None:
    ns = spec_namespace(Path(spec_path))
    ns["RollingTunedLinear"].grid = ns["ESTIMATOR_GRIDS"][estimator]
    import src.backtest.executor as ex

    _WORKER.update(ns=ns, ex=ex)


def _ping(_: int) -> int:
    time.sleep(0.2)  # long enough that every worker takes one (and initializes)
    return os.getpid()


def _backtest(job: tuple[tuple[Any, ...], dict[str, Any]]) -> tuple[str, float]:
    """One collected ``_backtest_and_save`` call with this process's fit_predict."""
    args, kwargs = job
    t = time.perf_counter()
    ns, ex = _WORKER["ns"], _WORKER["ex"]
    args = (args[0], args[1], ns["fit_predict_lin_tuned"], *args[3:])
    ex._backtest_and_save(*args, **kwargs)
    return str(args[9]), time.perf_counter() - t


def _rt_task(col: str, kwargs: dict[str, Any], frame: dict[str, Any]) -> Any:
    """``robust_transform`` on the inputs it reads: the column, time_of_day, the index."""
    import pandas as pd

    tod = frame["tod_categories"][frame["tod_codes"]]
    df = pd.DataFrame(
        {col: frame["values"], frame["time_col"]: tod}, index=frame["index"]
    )
    return _WORKER["ex"].robust_transform(df, col, **kwargs)


def _expiry_task(t: Any) -> Any:
    """The executor's calendar + expiry extractors on the stamps alone.

    Both are functions of ``t`` (expiry reads the calendar's ``is_close``);
    returns the expiry columns in the order the extractor writes them, and
    the ``is_close`` it used.
    """
    import pandas as pd

    ex = _WORKER["ex"]
    frame = pd.DataFrame({"t": t})
    ex.add_calendar_features(frame)
    names = ex.add_expiry_features(frame)
    return (
        list(names),
        {n: frame[n].to_numpy() for n in names},
        frame["is_close"].to_numpy(),
    )


Key = tuple[str, tuple[tuple[str, Any], ...]]


class _Farm:
    """This pass's robust_transform calls and expiry extractor, run in the pool.

    Both are the executor's own functions, called in a worker on exactly the
    inputs they read, so their results are the inline results:

    * ``robust_transform(df, col, **kw)`` reads ``df[col]``, ``df[time_col]``
      and the index.  At a pass's first call, every call the PREVIOUS pass
      made (``plan``: column + keywords, in order) is submitted; each call
      then checks that its column, index and time_of_day equal what was sent,
      and runs inline otherwise -- and always on a cold server's first pass,
      which has no plan.
    * ``add_expiry_features(df)`` is a function of ``t`` and ``is_close``
      (itself a function of ``t``): submitted with the transforms, applied
      when the executor calls it if ``t`` and ``is_close`` are what the
      worker saw -- the same columns written in the same order.

    The HAR rolling means stay inline: shipping ~170 feature columns of 330k
    rows back costs about what computing them does.
    """

    def __init__(
        self, pool: Executor, workers: int, plan: list[Key], enabled: bool = True
    ) -> None:
        self.pool = pool
        self.workers = max(1, int(workers))
        self.plan = plan if enabled else []
        self.enabled = enabled
        self.calls: list[Key] = []
        self.futures: dict[Any, Any] = {}
        self.sent: dict[Any, Any] = {}
        self.inline: list[str] = []
        self.farmed: list[str] = []

    def _submit_plan(self, df: Any, time_col: str) -> None:
        import numpy as np
        import pandas as pd

        if "t" in df.columns:
            self.sent["__t__"] = df["t"].to_numpy(copy=True)
            self.futures["__expiry__"] = self.pool.submit(
                _expiry_task, self.sent["__t__"]
            )
        if time_col not in df.columns:
            return
        codes, cats = pd.factorize(df[time_col], sort=False)
        tod_codes = np.asarray(codes)
        tod_categories = np.asarray(cats, dtype=object)
        self.sent["__tod__"] = df[time_col].to_numpy()
        self.sent["__index__"] = df.index
        for key in self.plan:
            col, items = key
            if col not in df.columns:
                continue
            vals = df[col].to_numpy(copy=True)
            frame = {
                "values": vals,
                "time_col": time_col,
                "tod_codes": tod_codes,
                "tod_categories": tod_categories,
                "index": df.index,
            }
            self.sent[key] = vals
            self.futures[key] = self.pool.submit(_rt_task, col, dict(items), frame)

    def robust_transform(self, real: Any) -> Any:
        import numpy as np

        def rt(df: Any, col_name: str, *args: Any, **kwargs: Any) -> Any:
            key: Key = (col_name, tuple(sorted(kwargs.items())))
            self.calls.append(key)
            time_col = kwargs.get("time_col", "time_of_day")
            if args:
                self.inline.append(col_name)
                return real(df, col_name, *args, **kwargs)
            if len(self.calls) == 1 and self.plan:
                self._submit_plan(df, time_col)
            fut = self.futures.pop(key, None)
            if fut is not None:
                sent = self.sent[key]
                now = df[col_name].to_numpy()
                same = (
                    df.index.equals(self.sent["__index__"])
                    and now.dtype == sent.dtype
                    and np.array_equal(now, sent, equal_nan=True)
                    and np.array_equal(df[time_col].to_numpy(), self.sent["__tod__"])
                )
                if same:
                    self.farmed.append(col_name)
                    return fut.result()
            self.inline.append(col_name)
            return real(df, col_name, *args, **kwargs)

        return rt

    def expiry(self, real: Any) -> Any:
        import numpy as np

        def add_expiry_features(df: Any) -> Any:
            fut = self.futures.pop("__expiry__", None)
            if fut is not None and "is_close" in df.columns:
                names, cols, is_close = fut.result()
                if (
                    np.array_equal(df["t"].to_numpy(), self.sent["__t__"])
                    and np.array_equal(df["is_close"].to_numpy(), is_close)
                    and not any(n in df.columns for n in names)
                ):
                    for n in names:
                        df[n] = cols[n]
                    self.farmed.append("expiry")
                    return list(names)
            self.inline.append("expiry")
            return real(df)

        return add_expiry_features


def make_pool(root: Path, workers: int, estimator: str) -> ProcessPoolExecutor:
    """Spawned workers (a fresh interpreter each, this process's environment)."""
    return ProcessPoolExecutor(
        max_workers=max(1, int(workers)),
        mp_context=get_context("spawn"),
        initializer=_init_worker,
        initargs=(str(Path(root) / SPEC_REL), estimator),
    )


def warm(pool: ProcessPoolExecutor, workers: int) -> None:
    """Start and initialize every worker now rather than on the first backtest."""
    list(pool.map(_ping, range(max(1, int(workers)))))


def one_pass(
    root: Path,
    pool: Executor,
    bars: tuple[str, ...] = BARS,
    workers: int = 4,
    plan: list[Key] | None = None,
    farm: bool = True,
) -> dict[str, Any]:
    """All ``bars`` for the arm the HPC_KW_* environment names; timings + CSV paths.

    ``plan`` is the previous pass's robust_transform calls (``out["plan"]``);
    ``farm=False`` runs the transforms and the expiry extractor inline.  Each bar's
    backtest is submitted the moment ``run_executor`` yields its segment.
    """
    t0 = time.perf_counter()
    spec = Path(root) / SPEC_REL
    ns = spec_namespace(spec)  # re-read per pass: HPC_RESULT_DIR may have moved
    est, bkt = ns["ESTIMATOR"], ns["EXOG_BUCKET"]
    if not est or not bkt:
        raise RuntimeError("HPC_KW_ESTIMATOR and HPC_KW_EXOG_BUCKET must name one arm")
    if ns["SEGMENT"] != "all" or ns["LAG_SCOPE"] != "global":
        raise RuntimeError(
            "the shared pass needs HPC_KW_SEGMENT=all and HPC_KW_LAG_SCOPE=global "
            f"(got {ns['SEGMENT']!r}, {ns['LAG_SCOPE']!r}): only a global-lag matrix "
            "is the same for every bar"
        )
    kw = arm_kwargs(spec, ns, est, bkt)
    import src.backtest.executor as ex
    import src.backtest.segmentation as seg

    t_defs = time.perf_counter()
    stem = os.path.splitext(kw["output_file"])[0]
    futs: dict[str, Any] = {}
    farmer = _Farm(pool, workers, plan or [], enabled=farm)

    def collect(*a: Any, **k: Any) -> None:
        # the spec's fit_predict stays behind (each worker binds its own): it
        # lives in an exec'd namespace, so it is not importable, not picklable
        out_file = str(a[9])
        bar = next((b for b in bars if out_file == f"{stem}_{b}.csv"), None)
        if bar is None or bar in futs:
            raise RuntimeError(f"run_executor yielded an unexpected segment {out_file}")
        futs[bar] = pool.submit(_backtest, ((*a[:2], None, *a[3:]), k))

    patched: dict[str, Any] = {
        "_backtest_and_save": collect,
        "SEGMENT_DEFINITIONS": {b: seg.SEGMENT_DEFINITIONS[b] for b in bars},
    }
    # installed either way: with farm=False they only record the calls (the plan)
    patched["robust_transform"] = farmer.robust_transform(ex.robust_transform)
    patched["add_expiry_features"] = farmer.expiry(ex.add_expiry_features)
    real = {name: getattr(ex, name) for name in patched}
    for name, obj in patched.items():
        setattr(ex, name, obj)
    try:
        ns["run_executor"](**kw)
    except BaseException:
        for f in futs.values():
            f.cancel()
        raise
    finally:
        for name, obj in real.items():
            setattr(ex, name, obj)
    t_matrix = time.perf_counter()
    if sorted(futs) != sorted(bars):
        raise RuntimeError(f"run_executor yielded {sorted(futs)}, not the bars {bars}")
    done = {b: futs[b].result() for b in bars}
    t_end = time.perf_counter()
    return {
        "csvs": {b: done[b][0] for b in bars},
        "seconds_defs": t_defs - t0,
        "seconds_matrix": t_matrix - t_defs,
        "seconds_backtests": t_end - t_matrix,
        "seconds_per_backtest": {b: done[b][1] for b in bars},
        "seconds_total": t_end - t0,
        "transforms_farmed": farmer.farmed,
        "transforms_inline": farmer.inline,
        "plan": [[c, [list(i) for i in items]] for c, items in farmer.calls],
    }


def serve(root: Path, workers: int, stdin: IO[str], stdout: IO[str]) -> int:
    """Warm loop: READY, then one pass per JSON line {"result_dir", "bars", "farm"}.

    Each pass hands its robust_transform calls to the next as the plan to
    farm out (the canary's pass is the plan of the card's).
    """
    t0 = time.perf_counter()
    ns = spec_namespace(Path(root) / SPEC_REL)
    with make_pool(root, workers, ns["ESTIMATOR"]) as pool:
        warm(pool, workers)
        print(f"{READY} {time.perf_counter() - t0:.2f}", file=stdout, flush=True)
        plan: list[Key] = []
        for line in stdin:
            if not line.strip():
                continue
            try:
                cmd = json.loads(line)
                os.environ["HPC_RESULT_DIR"] = str(cmd["result_dir"])
                out = one_pass(
                    root,
                    pool,
                    tuple(cmd.get("bars") or BARS),
                    workers=workers,
                    plan=plan,
                    farm=bool(cmd.get("farm", True)),
                )
                plan = [
                    (c, tuple((k, v) for k, v in items)) for c, items in out["plan"]
                ]
                print(f"{DONE} {json.dumps(out)}", file=stdout, flush=True)
            except Exception:  # noqa: BLE001 -- the parent turns it into NO SIGNAL
                msg = json.dumps({"error": traceback.format_exc()})
                print(f"{FAIL} {msg}", file=stdout, flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", default=".", help="the ext root (src/, specs/, data/)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--bars", default=",".join(BARS))
    ap.add_argument("--serve", action="store_true")
    ap.add_argument("--out-json", default=None)
    a = ap.parse_args(argv)
    root = Path(a.root).resolve()
    if a.serve:
        return serve(root, a.workers, sys.stdin, sys.stdout)
    ns = spec_namespace(root / SPEC_REL)
    with make_pool(root, a.workers, ns["ESTIMATOR"]) as pool:
        out = one_pass(root, pool, tuple(a.bars.split(",")), workers=a.workers)
    print(
        "arms_shared: defs {seconds_defs:.2f}s matrix {seconds_matrix:.2f}s "
        "backtests {seconds_backtests:.2f}s total {seconds_total:.2f}s".format(**out),
        flush=True,
    )
    if a.out_json:
        Path(a.out_json).write_text(json.dumps(out, indent=1))
    return 0


# ------------------------------------------------------------------ parent --
def arm_env(
    bucket: str, estimator: str, train_win: int, result_dir: Path
) -> dict[str, str]:
    """The arm's environment (forecast.run_arm's), with SEGMENT=all."""
    env = dict(os.environ)
    repo = str(Path(__file__).resolve().parents[2])
    env.update(
        {
            "HPC_KW_SEGMENT": "all",
            "HPC_KW_LAG_SCOPE": "global",
            "HPC_KW_ESTIMATOR": estimator,
            "HPC_KW_EXOG_BUCKET": bucket,
            "HPC_KW_TRAIN_WIN": str(train_win),
            "HPC_KW_START": "0",
            "HPC_KW_END": "-1",
            "HPC_KW_HALO": "0",
            "HPC_RESULT_DIR": str(result_dir),
            "PYTHONUNBUFFERED": "1",
            "PYTHONIOENCODING": "utf-8",
            "OMP_NUM_THREADS": "1",
            "PYTHONPATH": os.pathsep.join(
                [repo, *[p for p in [env.get("PYTHONPATH", "")] if p]]
            ),
        }
    )
    return env


class ArmServer:
    """The parent's handle on a ``--serve`` process rooted at an ext root.

    Its output (the spec's prints, tqdm) goes to ``log``; the protocol lines
    are picked out of it.  ``run`` blocks until the pass is done or ``timeout``
    seconds pass, and raises on a failed pass, a dead server or a timeout.
    """

    def __init__(
        self,
        root: Path,
        env: dict[str, str],
        log: Path,
        workers: int = 4,
        python: str = sys.executable,
        ready_timeout: float = 300.0,
    ) -> None:
        self.root = Path(root)
        self.log_path = Path(log)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log = open(self.log_path, "a", encoding="utf-8")
        self._cv = threading.Condition()
        self._lines: list[str] = []
        self.proc = subprocess.Popen(
            [
                python,
                "-m",
                "live.close_signal.arms_shared",
                "--serve",
                "--workers",
                str(workers),
            ],
            cwd=str(self.root),
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            # its own process group, so kill() takes the workers with it
            start_new_session=sys.platform != "win32",
        )
        self._reader = threading.Thread(target=self._read, daemon=True)
        self._reader.start()
        try:
            self.ready_seconds = float(self._wait(READY, ready_timeout).split()[1])
        except BaseException:
            self.kill()
            raise

    def _read(self) -> None:
        assert self.proc.stdout is not None
        for line in self.proc.stdout:
            self._log.write(line)
            self._log.flush()
            # a worker's tqdm fragment (carriage returns, no newline) can share
            # the line: the protocol runs from its marker to the end of the line
            at = [i for i in (line.find(m) for m in (READY, DONE, FAIL)) if i >= 0]
            if at:
                with self._cv:
                    self._lines.append(line[min(at) :].rstrip("\n"))
                    self._cv.notify_all()
        with self._cv:
            self._lines.append("EOF")
            self._cv.notify_all()

    def _wait(self, want: str, timeout: float) -> str:
        deadline = time.monotonic() + timeout
        with self._cv:
            while True:
                if self._lines:
                    line = self._lines.pop(0)
                    if line.startswith(want):
                        return line
                    if line.startswith(FAIL):
                        raise RuntimeError(
                            "the shared arm pass failed:\n"
                            + json.loads(line[len(FAIL) :])["error"]
                        )
                    if line == "EOF":
                        raise RuntimeError(
                            f"the arm server exited (rc {self.proc.poll()}); log "
                            f"{self.log_path}:\n{self._tail()}"
                        )
                    continue
                left = deadline - time.monotonic()
                if left <= 0:
                    raise TimeoutError(
                        f"the arm server gave no {want} in {timeout:.0f}s"
                    )
                self._cv.wait(left)

    def _tail(self) -> str:
        try:
            return self.log_path.read_text(encoding="utf-8", errors="replace")[-2000:]
        except OSError:
            return ""

    def alive(self) -> bool:
        return self.proc.poll() is None

    def run(
        self,
        result_dir: Path,
        bars: tuple[str, ...] = BARS,
        timeout: float = 600.0,
        farm: bool = True,
    ) -> dict[str, Any]:
        if not self.alive():
            raise RuntimeError(f"the arm server is not running (rc {self.proc.poll()})")
        assert self.proc.stdin is not None
        cmd = {"result_dir": str(result_dir), "bars": list(bars), "farm": farm}
        self.proc.stdin.write(json.dumps(cmd) + "\n")
        self.proc.stdin.flush()
        return json.loads(self._wait(DONE, timeout)[len(DONE) :])

    def kill(self) -> None:
        """End the server and its worker processes now (an abandoned pass)."""
        try:
            if sys.platform == "win32":
                subprocess.run(
                    ["taskkill", "/T", "/F", "/PID", str(self.proc.pid)],
                    capture_output=True,
                    check=False,
                )
            else:
                import signal

                os.killpg(self.proc.pid, signal.SIGKILL)
        except Exception:  # noqa: BLE001 -- best effort; fall back to the process
            pass
        try:
            self.proc.kill()
            self.proc.wait(timeout=10)
        except Exception:  # noqa: BLE001
            pass
        finally:
            self._reader.join(timeout=5)
            self._log.close()

    def close(self, timeout: float = 60.0) -> None:
        try:
            if self.proc.stdin is not None and not self.proc.stdin.closed:
                self.proc.stdin.close()
            self.proc.wait(timeout=timeout)
        except Exception:  # noqa: BLE001 -- best effort at shutdown
            self.proc.kill()
        finally:
            self._reader.join(timeout=5)
            self._log.close()

    def __enter__(self) -> ArmServer:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


if __name__ == "__main__":
    sys.exit(main())
