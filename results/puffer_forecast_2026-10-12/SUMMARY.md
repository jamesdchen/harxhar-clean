# PufferLib 5.0 native MinGRU, supervised -- Phase 1 (2026-10-13)

Written by the main session from the build agent's report (the agent could not write this file itself).

- Install: PufferLib git branch `5.0` @ 6ffa5b10 (no PyPI release; PyPI ends at 3.0.0 without MinGRU), built WITHOUT
  PyTorch on CARC b18-16 (L40S, driver 580.159, CUDA 12.9.1, gcc 14.3); upstream RL trainer smoke test: cartpole
  solved at 7.6M env steps/s.
- Upstream pieces used as shipped: `src/algo.cu` MinGRU scan forward / backward + highway gate, bias-free cuBLAS
  encoder / decoder; Muon optimizer (`muon_step`), global clip `max_grad_norm` 1.0, cosine LR (`pufferl.cu`).
  PPO enters in `pufferl.cu` `train_epoch_gpu` (cache_imp_and_v, puff_advantage, ppo_loss_fwd_bwd -> grad_logits /
  grad_values -> arch_backward -> muon_step).
- Supervised path: `experiments/puffer_forecast/native/forecast_train.cu` includes `algo.cu` unchanged; our gather +
  loss kernels replace the three PPO calls; one fork `native/puffer_substrate.cuh` = pufferl.cu lines 1-291 without
  the RL-only includes (`native/substrate.patch`); fp32 build; each epoch a CUDA graph; many models concurrently.
- Output f in the sqrt fit space (plain = f^2 B); default loss (f - y)^2 (Huber variant available); QLIKE optional with
  eps = 1e-3 smooth floor on f and a per-sample |dL/df| cap at the 0.99 quantile of a constant-forecast baseline;
  inputs / target standardized per training window.
- Checks vs a float64 numpy reference: loss kernel max rel err <= 4.5e-5 (random + extreme inputs); gradient vs own
  finite differences median <= 7e-5 rel; network forward <= 3e-7, weight gradients <= 3.6e-7 rel L2; toy MinGRU,
  156 Muon steps, per-step loss within 1.5e-4 rel; bit-identical outputs across duplicate models, 1 / 8 / 24 / 48
  concurrent models and two jobs on the same GPU model (observed, upstream kernels not audited).
- Diagnostics: 0 non-finite samples in 76k steps; grad clip binds (MSE) 0.25 % baseline / 4.5 % all_features, mostly
  epoch 1 (QLIKE 0.4 % / 15 %); QLIKE cap binds 0.16 % / 0.26 % of samples.
- Speed (one L40S): 288 models in 8.8 s (569 model-epochs/s, 8.6k optimizer steps/s, 1.06M sequences/s); 0.11 s per
  early-stopped tune fit; concurrency gain only ~2.2x (kernel-launch bound); one window x design full grid
  (12 configs x 5 seeds, tune + refit) ~4 s.
- Accuracy, window ending 2019-12-31, fit-space MSE (validation tail 125 rows / next 10 rows):
  baseline minGRU 0.0781 / 0.0965 vs linear 0.0879 / 0.1020 (LightGBM next-6 0.1042);
  all_features minGRU 0.0827 / 0.1318 vs enet 0.0828, ridge 0.0823 / linear 0.115-0.122 (LightGBM 0.1335).
  Caveats: the validation tail also chose config and epoch; 10 days cannot rank; early stopping at 2-10 epochs; single
  seeds noisy.
- Walk-forward projection (1,088 sessions -> 866 trade days, cold refits, one L40S): refit every session with
  re-tune every 250 sessions 1.3 min / design (~2 h for 96 designs); full re-tune at every refit 56 min / design;
  refit every 10: 0.35 / 5.6 min; every 25: 0.29 / 2.3 min. Warm starts unnecessary (cold refit 4-6 epochs).
- Leftover: Hoffman2 job 15144888 (duplicate) -- qdel by the user. Run of record: `runs/13037177`.
