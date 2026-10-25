# MinGRU v2, full W 3,500: Hoffman2 UGE L40S as a GPU type (RULE ADDENDUM 1 VALIDATION, ADDENDUM 3f)

Owner (James, relayed): "validate on hoffman". Validation agent, 2026-10-25 12:00-13:00 PDT. Branch
insurance-selling-2026-09-19-e34-gru-v2-h2val (off insurance-selling-2026-09-19-e34-gru-v2 at 7cfa29b). Nothing was
changed on CARC: two read-only sessions. No walk piece was submitted.

## Status at 12:59 PDT: no verdict yet

The rule's gate cannot be evaluated: gate job 15272363 has been pending since 12:25 without starting. Its canary,
15272364, waits on it. Everything else is ready: inputs, runtime and job scripts are in place and md5-checked.

| Item | State |
|---|---|
| VALIDATION gate (both sets, tune twice, plus the first walk origin) | **not run**: 15272363 `qw` since 12:25:00 |
| Byte identity with CARC's L40S gate 13346568 | not known: the gate has not run |
| G5 canary of full W 3,500 (120 refits, CONC 24) | **not run**: 15272364 `hqw`, held on 15272363; it runs only if the gate's verdict is PASS |
| Seconds a refit on Hoffman2 L40S | not measured |
| GPUs we can actually get | **0 now; at most 3 (g18 x 2, g20 x 1), all busy until ~20:36 / 22:22 tonight and ~04:40 tomorrow** |

### Why nothing starts: the L40S hosts are mostly reserved for owner groups (read from UGE, 12:41-12:59)

The read-only survey at 11:55 counted ~3 free GPUs (g20: 1, g23: 2). For our user (jamesdc1, group mcucurin) none of
them can be used:

| Host | GPUs (L40S 48 GB) | Rule that applies to us | Usable by us now |
|---|---|---|---|
| g18 | 2 | `@ameyer_gpu`: users outside ameyer get at most 2 GPUs **each** | 0: both held by maertens (15234669 tasks 13 and 14, h_rt 20 h, started 02:22 and 08:40) |
| g19 | 2 | RQS `sautet`: `users !@sautet,!@admin hosts g19 to cuda=0` | never (group sautet only) |
| g20 | 2 | RQS `kappel`: `users !@kappel hosts @kappel_gpu to cuda=1`, **one GPU shared by all** users outside kappel; `kappel/1` reads cuda=1/1 | 0: the shared GPU is held by maertens (15234669 task 12, started 00:36). The "free" GPU is kappel's |
| g21 | 2 (queue slots 1) | RQS `lindert`: `users {!@lindert,!@admin} hosts g21,g22 to slots=0` | never (lindert debug node) |
| g22-g30 | 2 each | queue instances disabled (`d`); `@lindert_gpu` (g21-g31) also caps all non-lindert users together at cuda=12 | 0 (disabled) |

Other limits: per user, cuda=4 on g* hosts (RQS `gpu`) and h_rt at most 24 h. `qsub -w p` (verify only, nothing
submitted) gives "no suitable queues" for our request with 2 slots and with 1 slot. There are no reservations
(`max_reservation 0`).

The free GPUs also have a queue. At 12:58 these L40S jobs were waiting, all with higher priority than ours (0.50000)
and all still `qw`:

| Job | User | Ask | Waiting since |
|---|---|---|---|
| QRLOGIN | leumas | 8 slots, gpu_model list | 02-26 |
| job.sh | wathena | 4 slots, h_rt 36 h (above the queue's 24 h cap) | 08-09 |
| 1-GPU job | jwang194 | 1G memory | 10-21 15:01 |
| 1-GPU job | shammami | | 10-21 17:02 |
| 1-GPU job | tanigawa | | 10-22 03:10 |
| 1-GPU job | shammami | | 10-22 11:31 |
| 1-GPU job | seungmo6 | | 10-22 14:44 |
| 1-GPU job | bchua | | 10-22 21:20 |
| 1-GPU job | khan | | 10-23 23:30 |
| arrays | maertens | 26+ more 20-h tasks | |
| arrays | akshaya | about 23,000 tasks of 8 slots | |

The scheduler's priority grows with waiting time (weight_waiting_time 100), so these jobs keep their lead over ours.
Every GPU we can reach turns over about once every 20 h. The earlier L40S test of 2026-10-23, 15260302, did not start
in 3 h 23 min either.

**Practical conclusion:** whatever the gate shows, Hoffman2 UGE L40S cannot supply 2-4 GPUs for this arm on any useful
timescale. On reachable capacity alone, the best case is 3 GPUs, and our turn comes after the jobs listed above. The
arm's type choice (ADDENDUM 3d: P100 vs CARC L40S) does not need to wait for this validation.

## What is set up on Hoffman2 (ready to run)

Root `/u/scratch/j/jamesdc1/pmf2v_h2` (scratch, 14-day purge). Files were copied CARC -> laptop -> Hoffman2 in one tar
stream at 12:11-12:16: 2.3 GB, 170 files. The manifest's md5s were computed on CARC; `md5sum -c` gives OK for all 170
on Hoffman2 (`inputs/carc_manifest.md5`, copy in `h2_l40s/carc_manifest.md5`).

| Path | Source on CARC |
|---|---|
| `bin/forecast_train` (md5 44ab6fff) | `/home1/jc_905/pmf/bin` |
| `inputs/gate_tune_prev.txt`, `inputs/gate_walk_prev.txt` | `/home1/jc_905/pmf/inputs` (the gate's jobs files) |
| `code/pooled_mingru_final_gate.py` | `/home1/jc_905/pmf/code` |
| `data/pm13_{baseline,all_features}/*` and the 4 `pm13_lin_init` files the gate's jobs name | `/home1/jc_905/pm13/data` |
| `ref/pm13/runs/...`: the P100 reference, i.e. the stored tune / walk runs 13303258, 13303117, 13303665 and 13303659, plus `out/pretune_qlike_folds.csv` | `/home1/jc_905/pm13` |
| `ref/carc/gate_l40s_{tune,refit,tune2}_13346568` and the P100 gate runs `gate_tune_prev_13319779` / `gate_walk_prev_13319780` | `/home1/jc_905/pmf/runs` |
| `data/pmf_all_features/*` (the full-set W 3,500 design) | `/home1/jc_905/pmf35/data` |
| `data/pmf_lin_init/all_features/w000/r50777..r50800_L8_wl0.003.f32` (the 24 canary bar inits) | `/project2/pollok_1603/jc_905/pmf2_35` |
| `inputs/v2w_all_features.000.txt`, `inputs/walkinit/all_features_w000.csv`, `inputs/init_index_v2.csv`, `code/pooled_mingru_v2.py` (md5 11504cdb), `code/pooled_mingru_v2_gpu_carc.sbatch` (md5 802e37ae) | `/project2/pollok_1603/jc_905/pmf2_35` |
| `carclib/` | see Runtime |

The canary file `inputs/pieces/canary/v2w_all_features.000.c24.txt` is the first 120 lines of
`v2w_all_features.000.txt`: rows 50,777-50,800 x 5 seeds, H 64, L 8, lr 0.003, wd 0.045674614 (tau 1), wd_enc 3,
E 40. Its md5 is bc606f55, equal to CARC's `head -120`.

### Runtime: the same bytes as CARC

Hoffman2's UGE hosts run CentOS 7.9 (glibc 2.17). CARC's runtime libraries need GLIBC_2.27: libcublasLt 12.9.1.4 and
libcurand 10.3.10.19 of `cuda/12.9.1`. Hoffman2's newest module is `cuda/12.3`, a different cuBLAS. The binary also
calls host libm (exp, log, cos, expm1), and glibc 2.17's versions of those differ from 2.28's.

`apptainer pull` is unavailable on the login node (no mksquashfs). So the runtime follows this repo's 2026-09-29
precedent (`cluster/optuna_h2_runtime_setup.sh`, "a runtime that is CARC's exactly"): the unmodified binary is started
by CARC's own loader, `carclib/ld-linux-x86-64.so.2 --library-path carclib`. Every library is CARC's file (md5 in the
manifest), and `--list` resolves every dependency from `carclib/`. Only libcuda, the NVIDIA driver, is Hoffman2's.

| Component | CARC (L40S b17-15 gate; v2 walk jobs) | Hoffman2 job |
|---|---|---|
| Trainer binary | forecast_train 44ab6fff (native_g9; SASS sm_60/70/80/86/89/90 + compute_90 PTX) | same file; `cuobjdump`: sm_89 SASS present, so the L40S runs native code with no JIT |
| CUDA runtime | static 12.9, inside the binary | same |
| cuBLAS / cuBLASLt | 12.9.1.4 (`cuda/12.9.1`), fddca2d0 / 2c63b8a4 | same files |
| cuRAND | 10.3.10.19, e236b586 | same file |
| libstdc++ / libgcc_s | gcc 14.3.0 (6.0.33), 54c89330 / eff7eb70 | same files |
| glibc: loader, libc, libm, libpthread, librt, libdl | Rocky 8.10, glibc-2.28-251.el8_10.40 | same files (CARC login node discovery2) |
| NVIDIA driver | 580.159.04 | Hoffman2's; not known until a job runs (recorded in env.txt) |
| OS / kernel | Rocky 8.10 | CentOS 7.9; only the driver and kernel come from it |
| Python for POST and the gate checks (not the trainer) | harxhar-r13: 3.12.15, numpy 2.3.3, pandas 2.3.3 | t3t's `venv_gb` in the Debian bookworm container: CPython 3.12.3, numpy 1.26.4, pandas 2.3.3. Different numpy; it is used only for checks (status, rows, finite / positive, a float64 QLIKE against a 0.1 % tolerance) |

### Job scripts (this branch)

- `cluster/pooled_mingru_v2_gate_gputype_h2.sh`: CARC's `pooled_mingru_final_gate_gputype_carc.sbatch` ported to UGE.
  - Runs the same jobs files, binary and command line (`--concurrent 6 --chunk 5 --pool`,
    CUDA_DEVICE_MAX_CONNECTIONS 32): tune, refit (the first walk origin), then tune again.
  - Stages the data node-local and md5-checks it against the manifest.
  - Refuses the job unless the GPU UGE assigned (CUDA_VISIBLE_DEVICES, CUDA_DEVICE_ORDER=PCI_BUS_ID) is
    `NVIDIA L40S` and every GPU on the node is an L40S.
  - Writes env.txt as CARC does, plus a full env file: driver, uuid, loader version, `--list`, and the libraries the
    trainer actually mapped (`/proc/<pid>/maps`).
  - Request: `-pe shared 2 -l gpu,L40S,cuda=1,h_rt=0:40:00,h_data=12G` (24G total, as CARC's gate). `cuda` is a JOB
    consumable, so this is 1 GPU.
- `experiments/pooled_mingru_v2_h2_gate_check.py`: called by the gate job.
  - Runs CARC's own checker (`pooled_mingru_final_gate.py`, CARC's copy), with its P100 reference paths pointed at
    `ref/pm13` and `data/`. It writes `gate_check.json` as on CARC.
  - Evaluates VALIDATION exactly as the rule states:
    - self-consistent: tune and tune2 `preds.csv` identical as sorted text, CARC's `cmp`;
    - per set: the seed-mean validation QLIKE within 0.1 % relative of the P100 reference, and the 3 best epochs equal
      to the P100's;
    - PASS only if both sets pass.
  - Compares all three runs byte for byte with CARC's L40S gate 13346568: preds, epochs, steps as sorted text;
    jobs_out without timing columns; params.
  - Writes `runs/gate_h2l40s_verdict_<job>.{json,txt}` (`VALIDATION PASS|FAIL`, `TYPE L40S|H2-L40S`). TYPE is L40S only
    if all three runs are byte-identical to CARC's; otherwise H2-L40S, a separate type.
  - Smoke-tested locally on a synthetic fixture.
- `cluster/pooled_mingru_v2_gpu_h2.sh`: CARC's `pooled_mingru_v2_gpu_carc.sbatch` (802e37ae) ported to UGE for the
  W 3,500 full-set arm.
  - Same staging, NOSTEPS, md5-checked copy-back, and POST (`pooled_mingru_v2.py post`, CARC's copy) into
    `RUN/summary`.
  - Same GPU refusal as the gate.
  - `REQUIRE_GATE=<id>`: runs only on `VALIDATION PASS`, and records `GPUTYPE=<the gate's TYPE>` in env.txt, so
    `walk-merge --gputype` keeps types apart.
  - Samples GPU utilisation and memory every 10 s.
  - No time-limit salvage: UGE sends no USR1.

### Jobs submitted (Hoffman2, 12:25 PDT)

    cd /u/scratch/j/jamesdc1/pmf2v_h2
    qsub -terse -N v2h2gt -pe shared 2 -l gpu,L40S,cuda=1,h_rt=0:40:00,h_data=12G code/pooled_mingru_v2_gate_gputype_h2.sh   -> 15272363
    qsub -terse -N v2h2c -hold_jid 15272363 -pe shared 2 -l gpu,L40S,cuda=1,h_rt=0:30:00,h_data=8G \
      -v JOBS=pieces/canary/v2w_all_features.000.c24.txt,TAG=v2wcanary_all_features_H2L40S,CONC=24,NOSTEPS=1,INITSUB=w000,REQUIRE_GATE=15272363 \
      code/pooled_mingru_v2_gpu_h2.sh                                                                                    -> 15272364

At most one L40S is used at a time. Both jobs were left queued at hand-back. If they run, the results will be:

- verdict: `runs/gate_h2l40s_verdict_15272363.{txt,json}`, check log `runs/gate_h2l40s_check_15272363.log`;
- gate runs: `runs/gate_h2l40s_{tune,refit,tune2}_15272363/`;
- canary: `runs/v2wcanary_all_features_H2L40S_15272364/`, with POST in `summary/checks.json` (pass = n_bad 0) and
  `wall_s` in env.txt (seconds a refit = wall_s / 120, the G7 definition).

To withdraw them: `qdel 15272364 15272363`.

## Projection (not measured on Hoffman2)

Arm: 33,970 bars x 5 seeds = 169,850 refits. Two rates from CARC bracket the projection, assuming Hoffman2's L40S runs
as fast as CARC's:

- **Low: 1.12 s a refit (52.8 GPU-h).** CARC's L40S measured 0.028 GPU-s a model-epoch at CONC 24 at W 3,500
  (G4 Stage 1, H <= 64, L <= 20); E 40 gives 1.12 s. This equals the "~1.12 s planned" A40 figure.
- **High: 1.73 s a refit (81.6 GPU-h).** ADDENDUM 3's L40S walk estimate, "the G7 P100 rate / 0.9 measured on these
  shapes".

| L40S on Hoffman2 | Wall-clock at 1.12 s | Wall-clock at 1.73 s |
|---|---|---|
| 2 | ~26.4 h | ~40.8 h |
| 3 (all we can reach) | ~17.6 h | ~27.2 h |
| 4 (per-user cap) | ~13.2 h | ~20.4 h |

Pieces must stay under the 24 h h_rt. These figures are pure GPU time, after the jobs ahead of us. At the current
turnover the queue wait alone is likely a day or more.

## Sessions

- CARC: 2, both read-only.
  - 12:02, info: ldd, library versions, gate files, sizes.
  - 12:11-12:16: the tar stream out.
- Hoffman2: 13, one at a time.
  - 12:04 info.
  - 12:07 setup: complexes, sm_89 check, failed `apptainer pull`.
  - 12:11-12:16 receive.
  - 12:24 manifest check, uploads, canary file, runtime `--list`.
  - 12:25 submit.
  - 12:40 poll.
  - 12:41 x 3 and 12:42: diagnosis (verify-mode `qsub -w`, nothing submitted).
  - 12:58 poll.
  - 12:58 and 12:59: RQS reads.
