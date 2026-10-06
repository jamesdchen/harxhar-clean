"""Reproduction check of the close studies of 2026-10-03 / 04 (results/close_studies_2026-10-03/).

Can the committed results be recreated?  This script answers by re-executing a cheap sample of every
kind of run from the COMMITTED code (a clean ``git archive HEAD`` extracted under the scratch folder,
so nothing untracked can leak in and no committed ``_work`` file or design-cache file can be
overwritten) and comparing it with what exists.

Checks (stage ``check``):
  env  the environment against environment.json (informative: a difference predicts which checks fail)
  a0   every cached design in results/close_design/_work against design_hashes.json
  a    capture_design_close.py rebuilds the bar1600 all_features design into a scratch folder
       (CLOSE_DESIGN_DIR): compared bitwise with the cache and with design_hashes.json
  a2   (informative) the same rebuild with numpy's AVX-512 code paths off and OpenBLAS forced to its
       Haswell (AVX2) kernels, i.e. what an AVX2 machine would compute with the same packages
  b    close_exogpen.py ``run`` (EXOGPEN_MODES=bb0) refits ridge_bb0 and lasso_bb0 through the C kernel,
       compiled afresh by gcc: compared with exog_penalty/_work/runs/*.npz and, scored, with headline.csv
  b2   (informative) the same two arms with the C kernel compiled for -march=haswell instead of native
  c    the first 3 refits of one LightGBM arm of each tree study, through that study's own code path:
         trees_datasize   lgbm_bar1600_w2000 (shipped configuration, k = 1; ``run`` with TDS_MAX_ANCHORS=3)
         trees_pretune    the walk's control / frozen_k10 / frozen_k45 arms (``walk``, CTP_WALK_ROWS=0,30),
                          and trial 0 of the pre-tune (4 fold fits of the shipped configuration)
         trees_lineartree lt_all_r10_lam10 (``smoke`` with CLT_SMOKE_ANCHORS=3)
         trees_morebars   lgbm_bars4_r8000 (the k = 4 pool; the ``gate`` stage's code path)
         trees_kfull      lgbm_bars13_r26000_barmin_seed43 (``gate`` code path, random_state 43)
       each compared bitwise with the study's _work forecast (and fit records)
  d    close_kfull_tests.py ``run existing`` with KFT_OUT in the scratch folder: every CSV compared
       with results/close_studies_2026-10-03/kfull_tests/, key numbers listed
  safe every committed results file and the design cache unchanged by the check (sizes, mtimes)

Stages:
  python experiments/close_studies_repro_check.py env      write requirements-lock.txt and environment.json
  python experiments/close_studies_repro_check.py hashes   write design_hashes.json from the design cache
  python experiments/close_studies_repro_check.py check [--only env,a0,a,a2,b,b2,c,d] [--out DIR]
                                                            run the checks; write repro_check.csv / .md
  python experiments/close_studies_repro_check.py kfull-run
        a trees_kfull worker that fits only the arms the committed manifest.csv lists as fitted (43 arms);
        close_trees_kfull.py ``run`` would also fit the 12 no-column arms that were never run, which
        changes the no-column family of kfull_tests.  Run two at once (claim files), as ``run``.
  python experiments/close_studies_repro_check.py diff [--rev HEAD]
        after a full rerun: every study CSV in the working tree against the committed one (time columns
        ignored), max |difference| of the numbers; also written to _repro_scratch/diff_vs_committed.csv
Scratch: results/close_studies_2026-10-03/_repro_scratch/ (gitignored; env REPRO_SCRATCH).
Compute: one child process at a time, every fit single-threaded (OMP / OPENBLAS / MKL / NUMBA threads 1,
UNIF_SCALE_PROCS=1, CTP_WORKERS=1).  About 15 minutes on the machine of environment.json.
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse  # noqa: E402
import hashlib  # noqa: E402
import io  # noqa: E402
import json  # noqa: E402
import platform  # noqa: E402
import re  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
STUDIES = REPO / "results" / "close_studies_2026-10-03"
CACHE = REPO / "results" / "close_design" / "_work"
SCR = Path(os.environ.get("REPRO_SCRATCH", str(STUDIES / "_repro_scratch")))
LOCK = STUDIES / "requirements-lock.txt"
ENV_JSON = STUDIES / "environment.json"
HASH_JSON = STUDIES / "design_hashes.json"
PY = sys.executable

# the distributions the study scripts import (traced from their imports); the lock adds their dependencies
TOP_DISTS = ("numpy", "pandas", "scipy", "scikit-learn", "lightgbm", "xgboost", "optuna", "numba",
             "pyarrow", "statsmodels", "matplotlib",
             "colorama")  # colorama: not a declared dependency on Linux, but colorlog (optuna) imports it when present
DESIGN_KEYS = ("X", "y", "names", "date", "baseline", "true_raw", "W")
# environment of every child process: single-threaded libraries, no process pools
CHILD_ENV = {"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
             "NUMBA_NUM_THREADS": "1", "UNIF_SCALE_PROCS": "1", "CTP_WORKERS": "1",
             "PYTHONDONTWRITEBYTECODE": "1", "MPLBACKEND": "Agg"}
# study env axes a child must not inherit from the caller's shell
STUDY_ENV_PREFIXES = ("TDS_", "TMB_", "TKF_", "CLT_", "CTP_", "KFT_", "EXOGPEN_", "CLOSE_DESIGN_DIR",
                      "NPY_DISABLE_CPU_FEATURES", "OPENBLAS_CORETYPE")
# numpy's AVX-512 dispatch targets (numpy 1.26); off = the code paths an AVX2 machine runs
NPY_AVX512 = "AVX512F AVX512CD AVX512_SKX AVX512_CLX AVX512_CNL AVX512_ICL"
SCORE_TOL = 1e-12  # scored numbers: the scorer's column order changes the last bits (forecasts.csv vs headline.csv)


# ============================================================================ small helpers
def sha(a) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def load_npz(path: Path) -> dict:
    z = np.load(path, allow_pickle=False)
    return {k: z[k] for k in z.files}


def git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(REPO), *args], check=True, capture_output=True, text=True).stdout


def cmd_out(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, check=True, capture_output=True, text=True).stdout.strip()
    except Exception as e:  # noqa: BLE001 -- recorded, never fatal
        return f"unavailable ({e})"


def bitwise(a, b) -> tuple[bool, float]:
    """(identical bytes, dtype and shape; max |a - b| over the finite pairs, inf when shapes differ)."""
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape:
        return False, float("inf")
    same = a.dtype == b.dtype and np.ascontiguousarray(a).tobytes() == np.ascontiguousarray(b).tobytes()
    if a.dtype.kind in "fiub" and b.dtype.kind in "fiub":
        fa, fb = a.astype(np.float64), b.astype(np.float64)
        m = np.isfinite(fa) & np.isfinite(fb)
        nan_same = bool(np.array_equal(np.isnan(fa), np.isnan(fb)))
        d = float(np.max(np.abs(fa[m] - fb[m]))) if m.any() else 0.0
        return same, (d if nan_same else float("inf"))
    return same, (0.0 if same else float("inf"))


def child_env(extra: dict | None = None) -> dict:
    env = {k: v for k, v in os.environ.items() if not k.startswith(STUDY_ENV_PREFIXES)}
    env.update(CHILD_ENV)
    env.update({k: str(v) for k, v in (extra or {}).items()})
    return env


def run(name: str, cmd: list[str], cwd: Path, extra_env: dict | None = None, timeout: int = 7200) -> tuple[int, float, Path]:
    """Run one child process (output to _repro_scratch/logs/<name>.log); (exit code, wall seconds, log)."""
    log = SCR / "logs" / f"{name}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with open(log, "w") as fh:
        fh.write(f"$ {' '.join(map(str, cmd))}\n# cwd {cwd}\n# env {json.dumps(extra_env or {}, default=str)}\n")
        fh.flush()
        try:
            rc = subprocess.run([str(c) for c in cmd], cwd=str(cwd), env=child_env(extra_env), stdout=fh,
                                stderr=subprocess.STDOUT, timeout=timeout).returncode
        except subprocess.TimeoutExpired:
            rc = -9
    sec = time.time() - t0
    print(f"  [{name}] exit {rc}, {sec:.0f}s", flush=True)
    return rc, sec, log


# ============================================================================ stage env
def _canon(n: str) -> str:
    return re.sub(r"[-_.]+", "-", n).lower()


def locked_distributions() -> list[tuple[str, str]]:
    """TOP_DISTS and every installed dependency they require (markers evaluated here, no extras)."""
    from importlib import metadata as md

    from packaging.requirements import Requirement

    seen: dict[str, tuple[str, str]] = {}
    todo = list(TOP_DISTS)
    while todo:
        n = todo.pop()
        if _canon(n) in seen:
            continue
        try:
            d = md.distribution(n)
        except md.PackageNotFoundError:
            continue  # an optional dependency that is not installed
        seen[_canon(n)] = (d.metadata["Name"], d.version)
        for r in d.requires or []:
            req = Requirement(r)
            if req.marker is not None and not req.marker.evaluate({"extra": ""}):
                continue
            todo.append(req.name)
    return sorted(seen.values(), key=lambda t: t[0].lower())


def cpu_info() -> dict:
    txt = Path("/proc/cpuinfo").read_text() if Path("/proc/cpuinfo").is_file() else ""

    def field(k: str) -> str:
        m = re.search(rf"^{k}\s*:\s*(.*)$", txt, re.M)
        return m.group(1).strip() if m else ""

    flags = field("flags").split()
    return dict(
        model_name=field("model name"), vendor=field("vendor_id"), family=field("cpu family"),
        model=field("model"), stepping=field("stepping"), logical_cores=os.cpu_count(),
        hypervisor="hypervisor" in flags, fma="fma" in flags, avx2="avx2" in flags, avx512f="avx512f" in flags,
        avx512_flags=sorted(f for f in flags if f.startswith("avx512")), flags=flags,
    )


def blas_runtime() -> list[dict]:
    """The BLAS / OpenMP libraries loaded by the studies' packages and the kernel OpenBLAS picked."""
    import threadpoolctl

    import lightgbm  # noqa: F401
    import scipy.linalg  # noqa: F401
    import sklearn  # noqa: F401
    import xgboost  # noqa: F401

    keep = ("user_api", "internal_api", "prefix", "version", "architecture", "threading_layer", "num_threads")
    out = []
    for d in threadpoolctl.threadpool_info():
        r = {k: d.get(k) for k in keep}
        r["file"] = Path(d.get("filepath", "")).name
        r["package"] = Path(d.get("filepath", "")).parent.name
        out.append(r)
    return out


def c_kernel_info() -> dict:
    out = dict(
        gcc=cmd_out(["gcc", "--version"]).splitlines()[0] if shutil.which("gcc") else "absent",
        march_native=re.sub(r"\s+", " ", cmd_out(["bash", "-c", "gcc -march=native -Q --help=target | grep -E '^\\s+-march='"])).strip(),
        compile_command="gcc -O3 -march=native -fPIC -shared -o <so> experiments/close_exogpen_kernel.c "
                        "-l:liblapack.so.3 -l:libblas.so.3 -lm (close_exogpen.kernel; the same for close_exogpen_cdcheck.c)",
    )
    for lib in ("liblapack.so.3", "libblas.so.3"):
        p = Path("/usr/lib/x86_64-linux-gnu") / lib
        real = str(p.resolve()) if p.exists() else "absent"
        pkg = cmd_out(["dpkg-query", "-S", real]).split(":")[0] if p.exists() else ""
        ver = cmd_out(["dpkg-query", "-W", "-f=${Version}", pkg]) if pkg else ""
        out[lib] = dict(resolves_to=real, package=pkg, package_version=ver)
    return out


def stage_env() -> None:
    try:  # numpy 1.x
        from numpy.core._multiarray_umath import __cpu_baseline__, __cpu_dispatch__, __cpu_features__
    except ImportError:  # numpy 2.x
        from numpy._core._multiarray_umath import __cpu_baseline__, __cpu_dispatch__, __cpu_features__

    dists = locked_distributions()
    pyv = platform.python_version()
    when = time.strftime("%Y-%m-%d")
    head = git("rev-parse", "HEAD").strip()
    lines = [
        "# Exact package versions of the close studies of 2026-10-03 / 04 (see REPRODUCE.md in this folder).",
        f"# Python {pyv} ({platform.python_implementation()}), {platform.system()} {platform.machine()}, glibc {platform.libc_ver()[1]}.",
        "# The packages the study scripts import and every dependency they require, as installed (pip freeze,",
        f"# filtered); written by experiments/close_studies_repro_check.py env on {when} at commit {head[:10]}.",
        f"# Install:  python{pyv.rsplit('.', 1)[0]} -m pip install -r results/close_studies_2026-10-03/requirements-lock.txt",
        "# System packages outside pip: gcc (13.3 here), liblapack3 + libblas3 (reference LAPACK / BLAS 3.12, the",
        "# C kernel links them), libgomp1 (LightGBM's OpenMP).",
        *(f"{n}=={v}" for n, v in dists),
    ]
    LOCK.write_text("\n".join(lines) + "\n")
    osr = Path("/etc/os-release").read_text() if Path("/etc/os-release").is_file() else ""
    m = re.search(r'^PRETTY_NAME="?([^"\n]*)"?$', osr, re.M)
    info = dict(
        written=time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        written_by="experiments/close_studies_repro_check.py env",
        git_commit=head,
        python=dict(version=pyv, implementation=platform.python_implementation(), build=list(platform.python_build()),
                    compiler=platform.python_compiler(), executable=sys.executable),
        packages={n: v for n, v in dists},
        numpy_show_config=np.show_config(mode="dicts"),
        numpy_simd=dict(baseline=list(__cpu_baseline__), dispatch=list(__cpu_dispatch__),
                        active=[k for k, v in __cpu_features__.items() if v]),
        blas_and_openmp_runtime=blas_runtime(),
        cpu=cpu_info(),
        c_kernel=c_kernel_info(),
        os=dict(pretty_name=m.group(1) if m else platform.platform(), kernel=platform.release(),
                machine=platform.machine(), glibc=platform.libc_ver()[1]),
        thread_settings={
            "set by every study script before numpy loads": {"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
                                                             "MKL_NUM_THREADS": "1"},
            "NUMBA_NUM_THREADS": "1 (close_trees_pretune.py and close_trees_pretune_jobs.py, set if unset)",
            "SLURM_CPUS_PER_TASK": "1 (the tree scripts set it: the tree spec's N_THREADS = 1)",
            "model threads": {"LightGBM": "num_threads=1", "XGBoost": "n_jobs=1", "random forest": "n_jobs=1"},
            "capture_design_close.py": "sets no thread variables (the 2026-10-03 / 04 captures ran with the library "
                                       "defaults on 4 cores); its scaling pool UNIF_SCALE_PROCS defaults to "
                                       "min(cores, 8) and does not change the design (the check rebuilds with 1)",
            "CTP_WORKERS": "close_trees_pretune.py fold pool, default 2; every fit single-threaded, so it does not "
                           "change any number",
            "concurrent processes": "2 for most stages; up to 4 for the last trees_datasize / trees_morebars / "
                                    "trees_kfull arms (each process single-threaded)",
        },
    )
    ENV_JSON.write_text(json.dumps(info, indent=1, default=str) + "\n")
    print(f"wrote {LOCK} ({len(dists)} packages) and {ENV_JSON}")


# ============================================================================ stage hashes
def design_hash(path: Path) -> dict:
    z = load_npz(path)
    arrays = {k: dict(sha256=sha(z[k]), dtype=str(z[k].dtype), shape=list(z[k].shape)) for k in DESIGN_KEYS if k in z}
    date = z["date"]
    return dict(file=path.name, arrays=arrays, rows=int(len(z["y"])), columns=int(z["X"].shape[1]), W=int(z["W"]),
                first_row=str(date[0]), last_row=str(date[-1]))


def stage_hashes() -> None:
    files = sorted(CACHE.glob("design_*.npz"))
    if not files:
        raise SystemExit(f"no design cache in {CACHE} (experiments/capture_design_close.py)")
    out = dict(
        written=time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        written_by="experiments/close_studies_repro_check.py hashes",
        cache=str(CACHE.relative_to(REPO)),
        hash_rule="sha256 of numpy.ascontiguousarray(a).tobytes() for each array of the .npz (stored dtype, C "
                  "order, little-endian); the .npz files themselves carry zip timestamps, so compare arrays, "
                  "not files",
        capture="python experiments/capture_design_close.py <segment> <bucket> (file design_<segment>_<bucket>.npz)",
        machine=dict(cpu=cpu_info()["model_name"], avx512f=cpu_info()["avx512f"], numpy=np.__version__,
                     pandas=pd.__version__),
        designs={},
    )
    for f in files:
        t0 = time.time()
        out["designs"][f.stem.removeprefix("design_")] = design_hash(f)
        print(f"  {f.name}: {time.time() - t0:.0f}s", flush=True)
    HASH_JSON.write_text(json.dumps(out, indent=1) + "\n")
    print(f"wrote {HASH_JSON} ({len(files)} designs)")


# ============================================================================ stage check
ROWS: list[dict] = []


def row(check: str, target: str, ok: bool | None, maxdiff: float | None, sec: float | None, detail: str = "") -> None:
    result = "info" if ok is None else ("pass" if ok else "fail")
    ROWS.append(dict(check=check, target=target, result=result, max_difference=maxdiff, seconds=None if sec is None else round(sec, 1), detail=detail))
    print(f"  {result:4s} {check}: {target} (max diff {maxdiff}) {detail}", flush=True)


def compare_npz(new: dict, old: dict, keys: list[str], old_slice=None) -> tuple[bool, float, list[str]]:
    """Every key bitwise; old_slice applies to the stored arrays (e.g. the first 3 refits)."""
    ok_all, mx, bad = True, 0.0, []
    for k in keys:
        o = old[k] if old_slice is None else old[k][old_slice]
        same, d = bitwise(new[k], o)
        ok_all &= same
        mx = max(mx, d)
        if not same:
            bad.append(f"{k} (max |diff| {d:.3g})")
    return ok_all, mx, bad


STUDY_DIRS = ("exog_penalty", "trees_datasize", "trees_pretune", "trees_lineartree", "trees_morebars", "trees_kfull",
              "kfull_tests")


def snapshot() -> dict:
    """(size, mtime) of every file of the seven study folders (committed results and _work) and of the design cache."""
    out = {}
    for base in (*(STUDIES / d for d in STUDY_DIRS), CACHE):
        for p in base.rglob("*"):
            if p.is_file() and "__pycache__" not in p.parts:
                st = p.stat()
                out[str(p.relative_to(REPO))] = (st.st_size, st.st_mtime_ns)
    return out


def make_checkout() -> tuple[Path, str]:
    """A clean copy of HEAD (git archive; no untracked or ignored file) with the design cache files the
    reruns read copied in (copies: a rerun can never write into the real cache)."""
    ck = SCR / "checkout"
    if ck.exists():
        shutil.rmtree(ck)
    ck.mkdir(parents=True)
    head = git("rev-parse", "HEAD").strip()
    arch = subprocess.Popen(["git", "-C", str(REPO), "archive", "--format=tar", "HEAD"], stdout=subprocess.PIPE)
    subprocess.run(["tar", "-x", "-C", str(ck)], stdin=arch.stdout, check=True)
    arch.stdout.close()
    if arch.wait() != 0:
        raise SystemExit("git archive HEAD failed")
    dst = ck / "results" / "close_design" / "_work"
    dst.mkdir(parents=True, exist_ok=True)
    for seg in ("bar1600_all_features", "bar1600_baseline", "last30_all_features", "lastbars4_all_features",
                "lastbars13_all_features"):
        shutil.copy2(CACHE / f"design_{seg}.npz", dst / f"design_{seg}.npz")
    return ck, head


def check_env() -> None:
    if not ENV_JSON.is_file():
        row("env", "environment.json", None, None, None, "environment.json missing (stage env)")
        return
    ref = json.loads(ENV_JSON.read_text())
    cur_pk = {n: v for n, v in locked_distributions()}
    diffs = []
    if ref["python"]["version"] != platform.python_version():
        diffs.append(f"python {platform.python_version()} vs {ref['python']['version']}")
    for n in sorted(set(ref["packages"]) | set(cur_pk)):
        if ref["packages"].get(n) != cur_pk.get(n):
            diffs.append(f"{n} {cur_pk.get(n)} vs {ref['packages'].get(n)}")
    cpu = cpu_info()
    for k in ("model_name", "avx2", "avx512f", "fma"):
        if cpu[k] != ref["cpu"][k]:
            diffs.append(f"cpu {k} {cpu[k]} vs {ref['cpu'][k]}")
    arch_now = {(d["prefix"], d.get("architecture")) for d in blas_runtime() if d["user_api"] == "blas"}
    arch_ref = {(d["prefix"], d.get("architecture")) for d in ref["blas_and_openmp_runtime"] if d["user_api"] == "blas"}
    if arch_now != arch_ref:
        diffs.append(f"OpenBLAS kernels {sorted(arch_now)} vs {sorted(arch_ref)}")
    ck = c_kernel_info()
    for k in ("gcc", "march_native"):
        if ck[k] != ref["c_kernel"][k]:
            diffs.append(f"{k} {ck[k]!r} vs {ref['c_kernel'][k]!r}")
    for lib in ("liblapack.so.3", "libblas.so.3"):
        if ck[lib]["package_version"] != ref["c_kernel"][lib]["package_version"]:
            diffs.append(f"{lib} {ck[lib]['package_version']} vs {ref['c_kernel'][lib]['package_version']}")
    row("env", "this machine vs environment.json (python, packages, CPU flags, OpenBLAS kernels, gcc, LAPACK)",
        None, None, None, "same" if not diffs else "differs: " + "; ".join(diffs))


def check_a0() -> None:
    if not HASH_JSON.is_file():
        row("a0", "design cache vs design_hashes.json", False, None, None, "design_hashes.json missing (stage hashes)")
        return
    ref = json.loads(HASH_JSON.read_text())["designs"]
    t0 = time.time()
    bad, n = [], 0
    for name, h in ref.items():
        f = CACHE / f"design_{name}.npz"
        if not f.is_file():
            bad.append(f"{name} missing")
            continue
        got = design_hash(f)
        n += 1
        bad += [f"{name}.{k}" for k, v in h["arrays"].items() if got["arrays"].get(k, {}).get("sha256") != v["sha256"]]
    row("a0", f"{n} cached designs (results/close_design/_work) vs design_hashes.json", not bad, None, time.time() - t0,
        "every array's sha256 equal" if not bad else "differ: " + ", ".join(bad))


def check_a(ck: Path, variant: str = "") -> None:
    """Rebuild design_bar1600_all_features from the checkout; variant 'avx2' = numpy AVX-512 off + OpenBLAS Haswell."""
    out = SCR / ("design_rebuild" + (f"_{variant}" if variant else ""))
    if out.exists():
        shutil.rmtree(out)
    env = {"CLOSE_DESIGN_DIR": out}
    if variant == "avx2":
        env |= {"NPY_DISABLE_CPU_FEATURES": NPY_AVX512, "OPENBLAS_CORETYPE": "Haswell"}
    rc, sec, log = run("a_capture" + (f"_{variant}" if variant else ""),
                       [PY, ck / "experiments" / "capture_design_close.py", "bar1600", "all_features"], ck, env)
    tag = "a2" if variant else "a"
    f = out / "design_bar1600_all_features.npz"
    if rc != 0 or not f.is_file():
        row(tag, "design_bar1600_all_features rebuild", False, None, sec, f"exit {rc}, log {log.relative_to(REPO)}")
        return
    new, old = load_npz(f), load_npz(CACHE / "design_bar1600_all_features.npz")
    ok, mx, bad = compare_npz(new, old, list(DESIGN_KEYS))
    ncols = int((np.abs(new["X"] - old["X"]).max(axis=0) > 0).sum()) if new["X"].shape == old["X"].shape else -1
    if variant:
        row(tag, "bar1600 all_features rebuilt with numpy AVX-512 off and OpenBLAS Haswell kernels vs the cache "
                 "(what an AVX2 machine computes)", None, mx, sec,
            "bitwise identical" if ok else f"differs: {', '.join(bad)}; {ncols} of {new['X'].shape[1]} columns of X differ")
        return
    row(tag, "design_bar1600_all_features.npz rebuilt (clean checkout, CLOSE_DESIGN_DIR, 1 thread) vs the cache: "
             + ", ".join(DESIGN_KEYS), ok, mx, sec, "bitwise identical" if ok else f"differ: {', '.join(bad)}")
    if HASH_JSON.is_file():
        ref = json.loads(HASH_JSON.read_text())["designs"]["bar1600_all_features"]["arrays"]
        bad2 = [k for k, v in ref.items() if sha(new[k]) != v["sha256"]]
        row(tag, "rebuilt design vs design_hashes.json (sha256 of every array)", not bad2, None, None,
            "all equal" if not bad2 else "differ: " + ", ".join(bad2))


def check_b(ck: Path) -> None:
    runs_new = ck / "results" / "close_studies_2026-10-03" / "exog_penalty" / "_work" / "runs"
    runs_old = STUDIES / "exog_penalty" / "_work" / "runs"
    rc, sec, log = run("b_exogpen_run", [PY, ck / "experiments" / "close_exogpen.py", "run", "ridge", "reclasso"], ck,
                       {"EXOGPEN_MODES": "bb0"})
    if rc != 0:
        row("b", "exog_penalty run ridge reclasso (EXOGPEN_MODES=bb0)", False, None, sec, f"exit {rc}, log {log.relative_to(REPO)}")
        return
    compiled = "compiled:" in log.read_text()
    keys = ["pred", "theta", "alpha_blk", "r_blk", "val_mse", "pen_g", "mpr_mse", "events", "n_reseed",
            "exact_pred", "locked", "maskout"]
    first = True
    for arm in ("ridge_bb0", "lasso_bb0"):
        new, old = load_npz(runs_new / f"{arm}.npz"), load_npz(runs_old / f"{arm}.npz")
        ok, mx, bad = compare_npz(new, old, keys)
        k = min(len(new["n_singular"]), len(old["n_singular"]))
        same_ns = np.array_equal(new["n_singular"][:k], old["n_singular"][:k])
        ok &= bool(same_ns)
        note = (f"C kernel compiled afresh by gcc: {compiled}; cpu {float(new['cpu_sec']):.0f}s vs {float(old['cpu_sec']):.0f}s stored; "
                f"n_singular {new['n_singular'].tolist()} vs {old['n_singular'].tolist()} stored (the stored run predates "
                f"the last three counters)")
        row("b", f"exog_penalty {arm}: 1469 forecasts, coefficients, alphas, masks vs _work/runs/{arm}.npz", ok,
            float(bitwise(new["pred"], old["pred"])[1]), sec if first else None,
            ("bitwise identical; " if ok else f"differ: {', '.join(bad)}; ") + note)
        first = False
    new, old = load_npz(runs_new / "ols_baseline.npz"), load_npz(runs_old / "ols_baseline.npz")
    ok, mx, bad = compare_npz(new, old, ["pred", "theta_bb"])
    row("b", "exog_penalty ols_baseline (HAR + calendar OLS, numpy lstsq) vs _work/runs/ols_baseline.npz", ok, mx, None,
        "bitwise identical" if ok else f"differ: {', '.join(bad)}")
    # scored with the study's scorer, against headline.csv
    rc, sec, log = run("b_exogpen_score", [PY, Path(__file__).resolve(), "_child", "exog_score", "--root", ck], ck)
    res_f = SCR / "results" / "exog_score.json"
    if rc != 0 or not res_f.is_file():
        row("b", "ridge_bb0 / lasso_bb0 reruns scored vs headline.csv", False, None, sec, f"exit {rc}, log {log.relative_to(REPO)}")
        return
    sc = json.loads(res_f.read_text())
    head = pd.read_csv(STUDIES / "exog_penalty" / "headline.csv").set_index("key")
    for arm in ("ridge_bb0", "lasso_bb0"):
        d = {c: abs(sc[arm][c] - float(head.loc[arm, c])) for c in ("qlike", "sharpe_mid", "sharpe_crossed")}
        mx = max(d.values())
        row("b", f"{arm} rerun scored (866 trade days) vs exog_penalty/headline.csv: QLIKE, Sharpe mid, Sharpe crossed",
            mx <= SCORE_TOL, mx, sec if arm == "ridge_bb0" else None,
            f"QLIKE {sc[arm]['qlike']:.6f} vs {float(head.loc[arm, 'qlike']):.6f}, Sharpe mid {sc[arm]['sharpe_mid']:.4f} vs "
            f"{float(head.loc[arm, 'sharpe_mid']):.4f} (tolerance {SCORE_TOL:g}: the scorer's column order moves the last bits)")


def check_b2(ck: Path) -> None:
    so_dir = SCR / "exog_haswell"
    so_dir.mkdir(parents=True, exist_ok=True)
    so = so_dir / "close_exogpen_kernel.so"
    src = ck / "experiments" / "close_exogpen_kernel.c"
    cc = ["gcc", "-O3", "-march=haswell", "-fPIC", "-shared", "-o", str(so), str(src),
          "-l:liblapack.so.3", "-l:libblas.so.3", "-lm"]
    if subprocess.run(cc).returncode != 0:
        row("b2", "C kernel compiled with -march=haswell", None, None, None, "gcc failed")
        return
    rc, sec, log = run("b2_exogpen_haswell", [PY, Path(__file__).resolve(), "_child", "exog_haswell", "--root", ck], ck)
    runs_old = STUDIES / "exog_penalty" / "_work" / "runs"
    if rc != 0:
        row("b2", "ridge_bb0 / lasso_bb0 with the kernel built for -march=haswell", None, None, sec, f"exit {rc}, log {log.relative_to(REPO)}")
        return
    parts, mxs = [], []
    for arm in ("ridge_bb0", "lasso_bb0"):
        new, old = load_npz(so_dir / f"{arm}.npz"), load_npz(runs_old / f"{arm}.npz")
        same, d = bitwise(new["pred"], old["pred"])
        rel = float(np.nanmax(np.abs(new["pred"] / old["pred"] - 1.0)))
        mxs.append(d)
        parts.append(f"{arm} {'bitwise identical' if same else f'differs (max rel. {rel:.2g})'}, alphas "
                     f"{'same' if np.array_equal(new['alpha_blk'], old['alpha_blk']) else 'differ'}")
    row("b2", "ridge_bb0 / lasso_bb0 with the C kernel compiled for -march=haswell (AVX2 + FMA) vs the stored runs",
        None, max(mxs), sec, "; ".join(parts))


def first_rows(n_refits: int = 3, every: int = 10, start: int = 130) -> slice:
    return slice(start, start + n_refits * every)


def check_c(ck: Path) -> None:
    W = STUDIES
    # ---- trees_datasize: lgbm_bar1600_w2000 (the shipped configuration, k = 1), the run stage itself
    out = SCR / "trees_datasize"
    if out.exists():
        shutil.rmtree(out)
    rc, sec, log = run("c_datasize", [PY, ck / "experiments" / "close_trees_datasize.py", "run", "lgbm_bar1600_w2000"], ck,
                       {"TDS_WORK": out, "TDS_MAX_ANCHORS": 3})
    if rc == 0:
        new, old = load_npz(out / "lgbm_bar1600_w2000.npz"), load_npz(W / "trees_datasize" / "_work" / "lgbm_bar1600_w2000.npz")
        rows_ = first_rows()
        ok, mx, bad = compare_npz({"pred": new["pred"][rows_]}, {"pred": old["pred"][rows_]}, ["pred"])
        ok2, _, bad2 = compare_npz(new, old, ["anchors", "n_train", "kept_n", "leaf_min"], old_slice=slice(0, 3))
        nf = int(np.isfinite(new["pred"]).sum())
        row("c", "trees_datasize lgbm_bar1600_w2000 (shipped configuration, k = 1), first 3 refits (30 forecasts) vs "
                 "_work/lgbm_bar1600_w2000.npz", ok and ok2 and nf == 30, mx, sec,
            ("bitwise identical" if ok and ok2 else f"differ: {', '.join(bad + bad2)}") + f"; {nf} forecasts written")
    else:
        row("c", "trees_datasize lgbm_bar1600_w2000, first 3 refits", False, None, sec, f"exit {rc}, log {log.relative_to(REPO)}")

    # ---- trees_pretune: walk control / frozen_k10 / frozen_k45 (and the retune paths, same fits before row 380)
    out = SCR / "trees_pretune"
    if out.exists():
        shutil.rmtree(out)
    run_dir = out / "local" / "lgbm"
    run_dir.mkdir(parents=True)
    old_dir = W / "trees_pretune" / "_work" / "local" / "lgbm"
    for s in (0, 1):  # the pre-tune's exports: the walk's input
        shutil.copy2(old_dir / f"pretune_s{s}.npz", run_dir / f"pretune_s{s}.npz")
    rc, sec, log = run("c_pretune_walk", [PY, ck / "experiments" / "close_trees_pretune.py", "walk"], ck,
                       {"CTP_WORK": out, "CTP_SEEDS": "0,1", "CTP_CHECKPOINTS": "10", "CTP_RETUNE_TRIALS": "10",
                        "CTP_WALK_ROWS": "0,30", "CTP_WORKERS": "1"})
    if rc == 0:
        first = True
        for arm, also in (("control", ()), ("frozen_k10", ()), ("frozen_k45", ("retune_any", "retune_margin"))):
            oks, mxs, bads = [], [], []
            for a in (arm, *also):
                new, old = load_npz(run_dir / f"walk_{a}_r10.npz"), load_npz(old_dir / f"walk_{a}_r10.npz")
                ok, mx, bad = compare_npz({"pred": new["pred"], "n_kept": new["n_kept"]},
                                          {"pred": old["pred"][:30], "n_kept": old["n_kept"][:3]}, ["pred", "n_kept"])
                s_new, s_old = json.loads(str(new["schedule"])), json.loads(str(old["schedule"]))
                ok &= s_new[0] == s_old[0]
                oks.append(ok)
                mxs.append(mx)
                bads += [f"{a}: {b}" for b in bad] + ([] if s_new[0] == s_old[0] else [f"{a}: configuration"])
            row("c", f"trees_pretune walk {arm}" + (f" (+ {', '.join(also)}: the same fits before row 380)" if also else "")
                + f", first 3 refits (30 forecasts) vs _work/local/lgbm/walk_{arm}_r10.npz", all(oks), max(mxs),
                sec if first else None, "bitwise identical, same configuration and rounds" if all(oks) else "differ: " + ", ".join(bads))
            first = False
    else:
        row("c", "trees_pretune walk, first 3 refits", False, None, sec, f"exit {rc}, log {log.relative_to(REPO)}")
    rc, sec, log = run("c_pretune_folds", [PY, Path(__file__).resolve(), "_child", "pretune_folds", "--root", ck], ck,
                       {"CTP_WORK": out, "CTP_WORKERS": "1"})
    if rc == 0:
        got = json.loads((SCR / "results" / "pretune_folds.json").read_text())
        oks, mxs = [], []
        for s in (0, 1):
            z = load_npz(old_dir / f"pretune_s{s}.npz")
            for k in ("val_mse", "val_qlike", "rounds", "n_kept"):
                same, d = bitwise(np.array(got[k], float), z[f"fold_{k}"][0])
                oks.append(same)
                mxs.append(d)
        row("c", "trees_pretune pre-tune trial 0 (the shipped configuration, 4 fold fits): validation MSE, QLIKE, "
                 "early-stopped rounds vs pretune_s0.npz / pretune_s1.npz", all(oks), max(mxs), sec,
            f"val MSE {np.mean(got['val_mse']):.5f}, rounds {got['rounds']}" + ("" if all(oks) else "; differ"))
    else:
        row("c", "trees_pretune pre-tune trial 0, 4 folds", False, None, sec, f"exit {rc}, log {log.relative_to(REPO)}")

    # ---- trees_lineartree: lt_all_r10_lam10 through the smoke stage (first CLT_SMOKE_ANCHORS refits)
    out = SCR / "trees_lineartree"
    if out.exists():
        shutil.rmtree(out)
    rc, sec, log = run("c_lineartree", [PY, ck / "experiments" / "close_trees_lineartree.py", "smoke", "lt_all_r10_lam10"], ck,
                       {"CLT_WORK": out, "CLT_SMOKE_ANCHORS": 3})
    if rc == 0:
        new = load_npz(out / "smoke" / "lt_all_r10_lam10.npz")
        old = load_npz(W / "trees_lineartree" / "_work" / "lt_all_r10_lam10.npz")
        rows_ = first_rows()
        ok, mx, bad = compare_npz({"pred": new["pred"][rows_]}, {"pred": old["pred"][rows_]}, ["pred"])
        ok2, _, bad2 = compare_npz(new, old, ["anchors", "kept_n", "coef_max", "share_linear", "mean_feat", "fit_lo", "fit_hi"],
                                   old_slice=slice(0, 3))
        row("c", "trees_lineartree lt_all_r10_lam10 (linear_tree, linear_lambda 10), first 3 refits (30 forecasts, leaf "
                 "coefficients) vs _work/lt_all_r10_lam10.npz", ok and ok2, mx, sec,
            "bitwise identical" if ok and ok2 else f"differ: {', '.join(bad + bad2)}")
    else:
        row("c", "trees_lineartree lt_all_r10_lam10, first 3 refits", False, None, sec, f"exit {rc}, log {log.relative_to(REPO)}")

    # ---- trees_morebars (k = 4 pool) and trees_kfull (k = 13, bar_end_minute, seed 43): their gate code paths
    for task, label, stored, keys in (
        ("morebars", "trees_morebars lgbm_bars4_r8000 (the k = 4 pool, 8000 rows)",
         W / "trees_morebars" / "_work" / "lgbm_bars4_r8000.npz", ["anchors", "n_train", "kept_n", "leaf_min", "n_sessions"]),
        ("kfull", "trees_kfull lgbm_bars13_r26000_barmin_seed43 (k = 13, column bar_end_minute, random_state 43)",
         W / "trees_kfull" / "_work" / "lgbm_bars13_r26000_barmin_seed43.npz",
         ["anchors", "n_train", "kept_n", "leaf_min", "n_sessions", "imp_split", "imp_gain", "imp_kept"]),
    ):
        rc, sec, log = run(f"c_{task}", [PY, Path(__file__).resolve(), "_child", task, "--root", ck], ck,
                           {"TMB_WORK": SCR / "trees_morebars", "TKF_WORK": SCR / "trees_kfull"})
        f = SCR / "results" / f"{task}.npz"
        if rc != 0 or not f.is_file():
            row("c", f"{label}, first 3 refits", False, None, sec, f"exit {rc}, log {log.relative_to(REPO)}")
            continue
        new, old = load_npz(f), load_npz(stored)
        rows_ = first_rows()
        ok, mx, bad = compare_npz({"pred": new["pred"][rows_]}, {"pred": old["pred"][rows_]}, ["pred"])
        ok2, _, bad2 = compare_npz(new, old, keys, old_slice=slice(0, 3))
        row("c", f"{label}, first 3 refits (30 forecasts{', importances' if task == 'kfull' else ''}) vs _work/{stored.name}",
            ok and ok2, mx, sec, "bitwise identical" if ok and ok2 else f"differ: {', '.join(bad + bad2)}")


def _csv_diff(a: Path, b: Path) -> tuple[bool, float, str]:
    """(identical bytes, max |numeric difference|, note)."""
    if a.read_bytes() == b.read_bytes():
        return True, 0.0, ""
    x, y = pd.read_csv(a), pd.read_csv(b)
    if x.shape != y.shape or list(x.columns) != list(y.columns):
        return False, float("inf"), f"shape / columns differ {x.shape} vs {y.shape}"
    mx, txt = 0.0, []
    for c in x.columns:
        if pd.api.types.is_numeric_dtype(x[c]) and pd.api.types.is_numeric_dtype(y[c]):
            same, d = bitwise(x[c].to_numpy(float), y[c].to_numpy(float))
            mx = max(mx, d)
            if not same:
                txt.append(c)
        elif not x[c].astype(str).equals(y[c].astype(str)):
            txt.append(c)
            mx = float("inf")
    return False, mx, "columns differ: " + ", ".join(txt[:6])


def check_d(ck: Path) -> None:
    out = SCR / "kfull_tests"
    if out.exists():
        shutil.rmtree(out)
    ref = STUDIES / "kfull_tests"
    # the suite reads the committed runs' forecasts (gitignored _work folders), so it runs from the working
    # tree; its code there must equal HEAD's
    code = ["experiments/close_kfull_tests.py", "experiments/close_kfull_testlib.py", "experiments/close_trees_morebars.py",
            "experiments/close_trees_datasize.py", "experiments/dense_vs_sparse_1530.py", "notebooks/atm_straddle_lib.py",
            "src/evaluation/diebold_mariano.py", "src/evaluation/model_confidence_set.py"]
    drift = [c for c in code if (REPO / c).read_bytes() != (ck / c).read_bytes()]
    rc, sec, log = run("d_kfull_tests", [PY, REPO / "experiments" / "close_kfull_tests.py", "run", "existing"], REPO,
                       {"KFT_OUT": out})
    if rc != 0:
        row("d", "kfull_tests run existing", False, None, sec, f"exit {rc}, log {log.relative_to(REPO)}")
        return
    csvs = sorted(p.name for p in ref.glob("*.csv"))
    same_n, mx, notes = 0, 0.0, []
    for n in csvs:
        if not (out / n).is_file():
            notes.append(f"{n} not written")
            mx = float("inf")
            continue
        same, d, note = _csv_diff(out / n, ref / n)
        same_n += same
        mx = max(mx, d)
        if not same:
            notes.append(f"{n}: {note} (max {d:.3g})")
    ri_new, ri_old = json.loads((out / "run_info.json").read_text()), json.loads((ref / "run_info.json").read_text())
    for k in ("cpu_sec", "wall_sec", "written"):
        ri_new.pop(k, None)
        ri_old.pop(k, None)
    ri_same = ri_new == ri_old

    def body(p: Path) -> list[str]:
        return [ln for ln in p.read_text().splitlines() if not ln.startswith("- CPU time of the run")]

    md_same = body(out / "SUMMARY.md") == body(ref / "SUMMARY.md")
    ok = same_n == len(csvs) and ri_same and md_same and not drift
    row("d", f"kfull_tests run existing (KFT_OUT scratch): {len(csvs)} CSVs, run_info.json, SUMMARY.md vs "
             "results/close_studies_2026-10-03/kfull_tests/", ok, mx, sec,
        f"{same_n} of {len(csvs)} CSVs byte-identical; run_info.json (without times) {'same' if ri_same else 'differs'}; "
        f"SUMMARY.md (without the CPU-time line) {'same' if md_same else 'differs'}; code = HEAD: "
        f"{'yes' if not drift else 'no: ' + ', '.join(drift)}" + ("; " + "; ".join(notes[:5]) if notes else ""))
    # key numbers, read from both copies
    keys = []
    for fn, sel, col, label in (
        ("forecasts.csv", dict(forecast="ridge_bb0"), "qlike", "QLIKE ridge_bb0"),
        ("seed_curve.csv", dict(family="bar", k=4), "qlike_of_seed_average", "QLIKE seed average k = 4 (bar column)"),
        ("multiple_comparisons.csv", dict(family="bar", loss="qlike", test="SPA (Hansen 2005)", mean_block=10.0), "p_consistent", "SPA p (bar, QLIKE)"),
        ("multiple_comparisons.csv", dict(family="bar", loss="qlike", test="Reality Check (White 2000)", mean_block=10.0), "p_upper", "Reality Check p (bar, QLIKE)"),
    ):
        try:
            vals = []
            for base in (out, ref):
                t = pd.read_csv(base / fn)
                m = np.ones(len(t), bool)
                for c, v in sel.items():
                    m &= (t[c] == v).to_numpy()
                vals.append(float(t.loc[m, col].iloc[0]))
            keys.append(f"{label} {vals[0]:.6g} (committed {vals[1]:.6g})")
        except Exception as e:  # noqa: BLE001 -- a missing key number is reported, not fatal
            keys.append(f"{label}: not read ({e})")
    row("d", "kfull_tests key numbers", all("not read" not in k for k in keys) and ok, None, None, "; ".join(keys))


# ============================================================================ children (run inside the checkout)
def _child_paths(root: Path) -> None:
    for p in (root / "specs", root / "notebooks", root / "experiments", root):
        sys.path.insert(0, str(p))
    here = str(Path(__file__).resolve().parent)
    if Path(here) != root / "experiments":
        sys.path[:] = [p for p in sys.path if p != here]  # never import a study module from the working tree
    os.chdir(root)


def child(task: str, root: Path) -> None:
    _child_paths(root)
    res_dir = SCR / "results"
    res_dir.mkdir(parents=True, exist_ok=True)
    if task == "exog_score":
        import close_exogpen as ce
        import dense_vs_sparse_1530 as dvs

        S = ce.Setup()
        assert Path(ce.__file__).resolve().is_relative_to(root.resolve()), ce.__file__
        dz = ce.forecast_dz(S.d)
        runs = ce.WORK / "runs"
        preds = {a: np.load(runs / f"{a}.npz")["pred"] for a in ("ridge_bb0", "lasso_bb0")}
        P = dvs.deck_panel([dvs.research_frame(dz, preds[a]) for a in preds], dvs.deck_frame())
        pt = dvs.point(P)
        out = {a: dict(qlike=float(pt["ql"][i]), sharpe_mid=float(pt["sh"][i]), sharpe_crossed=float(pt["shx"][i]),
                       n_days=int(P["ql"].shape[0])) for i, a in enumerate(preds)}
        (res_dir / "exog_score.json").write_text(json.dumps(out, indent=1))
    elif task == "exog_haswell":
        import close_exogpen as ce

        ce.WORK = SCR / "exog_haswell"  # kernel() loads WORK/close_exogpen_kernel.so (built -march=haswell, newer than the .c)
        S = ce.Setup()
        grids = ce.spec_grids()
        for est, arm, every in (("ridge", "ridge_bb0", 0), ("reclasso", "lasso_bb0", ce.CHECK_EVERY)):
            r = ce.c_run(S, est, "bb0", grids[est], check_every=every)
            np.savez_compressed(ce.WORK / f"{arm}.npz", pred=r["pred"], alpha_blk=r["alpha_blk"])
    elif task == "pretune_folds":
        import close_trees_pretune as C
        import close_trees_pretune_jobs as CJ

        info = C.prepare_arrays()
        jobs = C.fold_jobs(info, C.pretune_folds(info), C.shipped(), "pre")
        outs = [CJ.fit_fold(j) for j in jobs]
        (res_dir / "pretune_folds.json").write_text(json.dumps(
            {k: [float(o[k]) for o in outs] for k in ("val_mse", "val_qlike", "rounds", "n_kept", "sec")}, indent=1))
    elif task == "morebars":
        import close_trees_morebars as mb

        c = mb.c
        mb.check()
        spec = mb.ARMS["lgbm_bars4_r8000"]
        src, _ = mb.source(spec)
        n_fc = c.sources()["n_fc"]
        anchors = c.tree_anchors(n_fc)[: mb.GATE_ANCHORS]
        out = c.run_tree("lgbm", src, spec["rows"], n_fc, anchors)  # as close_trees_morebars.gate / run
        out["n_sessions"] = mb.sessions_in_windows(src, spec["rows"], anchors)
        np.savez_compressed(res_dir / "morebars.npz", **out)
    elif task == "kfull":
        import close_trees_kfull as kf

        c = kf.c
        kf.check()
        kf.install_recorder()
        arm = "lgbm_bars13_r26000_barmin_seed43"
        spec = kf.ARMS[arm]
        src, _ = kf.mb.source(spec)
        n_fc = c.sources()["n_fc"]
        anchors = c.tree_anchors(n_fc)[: kf.GATE_ANCHORS]
        make = c.make_tree

        def seeded(m: str, n_rows: int, _seed: int = spec["seed"]):  # as close_trees_morebars.run builds a seeded arm
            return make(m, n_rows).set_params(random_state=_seed)

        c.make_tree = seeded
        try:
            out = c.run_tree("lgbm", src, spec["rows"], n_fc, anchors)
        finally:
            c.make_tree = make
        out["n_sessions"] = kf.mb.sessions_in_windows(src, spec["rows"], anchors)
        np.savez_compressed(res_dir / "kfull.npz", **out)
    else:
        raise SystemExit(f"unknown child task {task!r}")


# ============================================================================ report
def fmt(v) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return ""
    if isinstance(v, float):
        return "0" if v == 0 else (f"{v:.3g}" if np.isfinite(v) else "inf")
    return str(v)


def write_report(out_dir: Path, head: str, t_all: float) -> None:
    df = pd.DataFrame(ROWS)
    df[["check", "target", "result", "max_difference", "seconds"]].to_csv(out_dir / "repro_check.csv", index=False)
    ref = json.loads(ENV_JSON.read_text()) if ENV_JSON.is_file() else {}
    pk = ref.get("packages", {})
    n_fail = int((df["result"] == "fail").sum())
    lines = [
        "# Reproduction check of the close studies (2026-10-03 / 04)",
        "",
        f"Written by `experiments/close_studies_repro_check.py check` on {time.strftime('%Y-%m-%d %H:%M %Z')}, commit "
        f"`{head[:10]}` (a clean `git archive HEAD` under `_repro_scratch/checkout`), {t_all / 60:.1f} min wall. "
        f"Python {platform.python_version()}, LightGBM {pk.get('lightgbm', '?')}, XGBoost {pk.get('xgboost', '?')}, numpy "
        f"{pk.get('numpy', '?')}, pandas {pk.get('pandas', '?')}; CPU {cpu_info()['model_name']} (AVX-512 "
        f"{'yes' if cpu_info()['avx512f'] else 'no'}). Every rerun went to the scratch folder; one child process at a time, "
        "single-threaded.",
        "",
        f"**{(df['result'] == 'pass').sum()} pass, {n_fail} fail, {(df['result'] == 'info').sum()} informative.** "
        "pass = bitwise identical (scores: within 1e-12); info = an informative check with no expected outcome.",
        "",
        "| check | target | result | max difference | seconds |",
        "|---|---|---|---|---|",
        *(f"| {r.check} | {r.target} | {r.result} | {fmt(r.max_difference)} | {fmt(r.seconds)} |" for r in df.itertuples()),
        "",
        "Details:",
        "",
        *(f"- **{r.check}**, {r.target}: {r.detail}" for r in df.itertuples() if r.detail),
        "",
        "Seconds = wall time of the rerun behind the row (blank: the row reads the rerun of the row above). "
        "Logs: `_repro_scratch/logs/`. Commands for a full rerun: `REPRODUCE.md`.",
    ]
    (out_dir / "repro_check.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {out_dir / 'repro_check.csv'} and {out_dir / 'repro_check.md'}: {n_fail} fail")


def stage_check(only: set[str], out_dir: Path) -> None:
    t_all = time.time()
    SCR.mkdir(parents=True, exist_ok=True)
    before = snapshot()
    if "env" in only:
        check_env()
    if "a0" in only:
        check_a0()
    ck, head = make_checkout()
    print(f"checkout of {head[:10]} at {ck}", flush=True)
    if "a" in only:
        check_a(ck)
    if "a2" in only:
        check_a(ck, "avx2")
    if "b" in only:
        check_b(ck)
    if "b2" in only:
        check_b2(ck)
    if "c" in only:
        check_c(ck)
    if "d" in only:
        check_d(ck)
    after = snapshot()
    changed = sorted(k for k in set(before) | set(after) if before.get(k) != after.get(k))
    row("safe", "committed results files and the design cache unchanged by the check (size and mtime of "
                f"{len(before)} files)", not changed, None, None, "unchanged" if not changed else "changed: " + ", ".join(changed[:10]))
    write_report(out_dir, head, time.time() - t_all)


# ============================================================================ kfull-run (full-rerun helper)
def stage_kfull_run() -> None:
    sys.path.insert(0, str(REPO / "experiments"))
    import close_trees_kfull as kf

    rel = "results/close_studies_2026-10-03/trees_kfull/manifest.csv"
    m = pd.read_csv(io.StringIO(git("show", f"HEAD:{rel}")))  # the committed manifest (run rewrites the file)
    want = set(m.loc[(m["status"] == "new") & (m["exists"] == "yes"), "arm"])
    kf.QUEUE = [a for a in kf.QUEUE if a in want]  # close_trees_kfull.run reads the module's QUEUE
    print(f"trees_kfull: {len(kf.QUEUE)} committed arms to fit (of {len(kf.ARMS)} defined)", flush=True)
    kf.run()


# ============================================================================ diff (after a full rerun)
TIME_COLS = re.compile(r"(cpu|wall|_sec|^sec$|fit_s|elapsed|written|seconds)", re.I)


def stage_diff(rev: str) -> None:
    """Every committed CSV of the seven study folders: the working-tree file (a rerun rewrote it) against
    the version at ``rev``, ignoring the columns that record time; max |difference| of the numbers."""
    rows = []
    for d in STUDY_DIRS:
        for rel in git("ls-tree", "-r", "--name-only", rev, f"results/close_studies_2026-10-03/{d}/").split():
            if not rel.endswith(".csv"):
                continue
            new_p = REPO / rel
            if not new_p.is_file():
                rows.append(dict(file=rel, status="missing in the working tree", max_abs_diff=np.nan))
                continue
            old = pd.read_csv(io.StringIO(git("show", f"{rev}:{rel}")))
            new = pd.read_csv(new_p)
            if list(old.columns) != list(new.columns) or len(old) != len(new):
                rows.append(dict(file=rel, status=f"shape differs {old.shape} -> {new.shape}", max_abs_diff=np.nan))
                continue
            mx, other = 0.0, []
            for c in old.columns:
                if TIME_COLS.search(str(c)):
                    continue
                if pd.api.types.is_numeric_dtype(old[c]) and pd.api.types.is_numeric_dtype(new[c]):
                    mx = max(mx, bitwise(new[c].to_numpy(float), old[c].to_numpy(float))[1])
                elif not old[c].astype(str).equals(new[c].astype(str)):
                    other.append(str(c))
            status = "identical" if mx == 0 and not other else ("numbers differ" if not other else f"text differs in {', '.join(other[:4])}")
            rows.append(dict(file=rel, status=status, max_abs_diff=mx))
    df = pd.DataFrame(rows)
    out = SCR / "diff_vs_committed.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    with pd.option_context("display.max_rows", None, "display.width", 200, "display.max_colwidth", 90):
        print(df.to_string(index=False))
    print(f"{(df['status'] == 'identical').sum()} of {len(df)} CSVs identical (time columns ignored); wrote {out}")


# ============================================================================ main
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["env", "hashes", "check", "kfull-run", "diff", "_child"])
    ap.add_argument("task", nargs="?", default="")
    ap.add_argument("--only", default="env,a0,a,a2,b,b2,c,d")
    ap.add_argument("--out", default=str(STUDIES))
    ap.add_argument("--root", default=str(REPO))
    ap.add_argument("--rev", default="HEAD", help="diff: the committed version to compare with")
    a = ap.parse_args()
    if a.stage == "env":
        stage_env()
    elif a.stage == "hashes":
        stage_hashes()
    elif a.stage == "check":
        stage_check({s.strip() for s in a.only.split(",") if s.strip()}, Path(a.out))
    elif a.stage == "kfull-run":
        stage_kfull_run()
    elif a.stage == "diff":
        stage_diff(a.rev)
    else:
        child(a.task, Path(a.root).resolve())
