// forecast_train.cu -- SUPERVISED training of PufferLib 5.0's native MinGRU / highway network as the
// per-bar 16:00 variance forecaster (2026-10-12, Phase 1).  No PPO, no PyTorch.
//
// What is upstream (PufferLib 5.0, commit 6ffa5b10, compiled UNCHANGED from the checkout given by -I):
//   src/algo.cu   the network: encoder (cuBLAS linear), MinGRU layers with highway gates (fused scan
//                 kernels mingru_scan_forward / mingru_scan_backward, weight GEMMs on cuBLAS with the
//                 side-stream dW overlap), decoder (cuBLAS linear), build_arch / weights_create /
//                 weights_init / arch_reg_train / arch_backward, and the optimizer Muon (muon_init /
//                 muon_step: global L2 gradient clipping at max_grad_norm, Nesterov momentum,
//                 Newton-Schulz orthogonalisation, lr read from a device scalar)
//   src/ocean.cu  included by algo.cu (no custom encoder for this build)
// What is vendored: puffer_substrate.cuh = pufferl.cu lines 1-291 minus the RL-only includes (substrate.patch).
// What is ours (this file): the job runner, causal standardisation, the minibatch gather (the supervised
//   replacement of PufferLib's rollout buffers / select_copy), the forecasting-loss kernel (replaces
//   cache_imp_and_v + puff_advantage + ppo_loss_fwd_bwd: it writes grad_logits, zero grad_values, which
//   arch_backward consumes exactly as it consumes the PPO gradients), validation / early stopping, the
//   CUDA-graph capture of one training epoch per model (the mechanism of pufferl.cu train_impl), and
//   many models trained concurrently, each on its own stream with its own cuBLAS handles / workspaces.
//
// Output layout: f = mu_y + sd_y * o, o = decoder column 0 at the LAST step of the sequence, f the
// forecast in the repo's fit space (the adjusted target y; plain variance forecast = f^2 * B).
// Losses (ours): mse   (f - y)^2,                         dL/df = 2 (f - y)                (default)
//                huber e^2 if |e| <= d else 2 d |e| - d^2, dL/df = 2e or 2 d sign(e)      (d from the window)
//                qlike r - log r - 1, r = y^2 / (f^2 + eps^2)  (= RV / v with v = f^2 B, RV = y^2 B; the
//                      smooth floor eps keeps v > 0), dL/df = 2 f (1 - r) / (f^2 + eps^2), capped at
//                      |dL/df| <= c (c from the training window, set by the job; capped samples counted)
// Every per-sample loss / gradient is fp32; non-finite ones are zeroed and counted; every reduction in
// our kernels is the substrate's deterministic block tree-reduce (no atomics).
//
//   forecast_train run   --jobs FILE --out DIR [--concurrent K] [--chunk E] [--pool]
//        --pool (2026-10-13): K slots, each allocated and graph-captured once per shape and reused for the
//        next job of that shape (only the window's data, permutation, schedule, loss parameters and the
//        initial weights are re-loaded; initial weights and permutations are cached per seed), and a slot
//        is refilled as soon as its model stops; same numbers as the wave path (gated byte for byte)
//        --pack K [--pack-slots S] (2026-10-13, experiment): refit jobs packed K at a time into one
//        block-diagonal model (see 'packing' below); tune jobs through the pool
//   forecast_train losscheck IN OUT            (the loss kernel on given (o, y): gradient checks)
//   forecast_train gradcheck --jobs FILE --out DIR   (one forward / backward of each job, dumped)
#ifndef PRECISION_FLOAT
#error "forecast_train.cu is built with -DPRECISION_FLOAT only (fp32 parameters, activations and reductions)"
#endif
#include "puffer_substrate.cuh"
#include "algo.cu"

#include <algorithm>
#include <chrono>
#include <deque>
#include <tuple>
#include <map>
#include <string>
#include <vector>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #x, __FILE__, __LINE__, cudaGetErrorString(e_)); \
    exit(2); } } while (0)

static double now_s() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

// ============================================================================ loss (ours)
enum { LOSS_MSE = 0, LOSS_HUBER = 1, LOSS_QLIKE = 2 };

enum { LINK_IDENTITY = 0, LINK_SOFTPLUS = 1 };

struct LossParams {
    int kind;
    float mu_y, sd_y;     // target standardisation of the training rows (o -> f)
    float huber_delta;    // fit-space units
    float qlike_eps;      // smooth floor of |f| (fit-space units)
    float qlike_cap;      // |dL/df| cap (fit-space units)
    int link;             // LINK_IDENTITY: f = mu_y + sd_y o;  LINK_SOFTPLUS: f = softplus(link_c + link_s o)
    float link_c, link_s; // softplus: c = softplus^-1(mu_y), s = sd_y / sigmoid(c) (f ~ mu_y + sd_y o at o ~ 0)
};

// The forecast f from the decoder output o, and df/do (user, 2026-10-13: link=softplus keeps every seed's
// f > 0).  identity: f = mu_y + sd_y o, df/do = sd_y (the expressions of the earlier builds, byte for byte);
// softplus: a = c + s o, f = softplus(a) (stable: a > 20 ? a : log1p(exp(a))), df/do = s sigmoid(a).
__host__ __device__ inline float fc_link(const LossParams& p, float o, float* dfdo) {
    if (p.link == LINK_SOFTPLUS) {
        float a = p.link_c + p.link_s * o;
        float f = a > 20.0f ? a : log1pf(expf(a));
        float sg = a >= 0.0f ? 1.0f / (1.0f + expf(-a)) : expf(a) / (1.0f + expf(a));
        *dfdo = p.link_s * (a > 20.0f ? 1.0f : sg);
        return f;
    }
    *dfdo = p.sd_y;
    return p.mu_y + p.sd_y * o;
}

__host__ __device__ inline void fc_loss(const LossParams& p, float f, float y,
        float* l_out, float* g_out, int* capped) {
    float l, g;
    int cap = 0;
    if (p.kind == LOSS_MSE) {
        float e = f - y;
        l = e * e;
        g = 2.0f * e;
    } else if (p.kind == LOSS_HUBER) {
        float e = f - y, d = p.huber_delta, ae = fabsf(e);
        if (ae <= d) {
            l = e * e;
            g = 2.0f * e;
        } else {
            l = 2.0f * d * ae - d * d;
            g = 2.0f * d * copysignf(1.0f, e);
        }
    } else {
        float eps2 = p.qlike_eps * p.qlike_eps;
        float fe2 = f * f + eps2;                 // > 0: the variance forecast never reaches 0
        float y2 = fmaxf(y * y, eps2);            // guarded target
        float logr = logf(y2) - logf(fe2);
        logr = fminf(fmaxf(logr, -80.0f), 80.0f); // exponent clamp
        float r = expf(logr);
        l = r - logr - 1.0f;
        g = 2.0f * f * (1.0f - r) / fe2;
        if (fabsf(g) > p.qlike_cap) {
            g = copysignf(p.qlike_cap, g);
            cap = 1;
        }
    }
    *l_out = l;
    *g_out = g;
    *capped = cap;
}

// One block (256 threads): per-sample loss on the last step, grad_logits for every (b, t) (zero but the
// last step), the step's [sum loss, n capped, n non-finite] by the deterministic block reduce.
__global__ void fc_loss_train(const precision_t* __restrict__ dec, int T, int od1,
        const float* __restrict__ y, const float* __restrict__ mask, int B, const LossParams* __restrict__ lpd,
        float inv_n, float* __restrict__ grad_logits, float* __restrict__ step_out) {
    const LossParams p = *lpd;  // device memory: a captured graph serves every window of its shape
    __shared__ float sm[3 * 256];
    int tid = threadIdx.x;
    float sl = 0.0f, sc = 0.0f, sn = 0.0f;
    for (int b = tid; b < B; b += blockDim.x) {
        for (int t = 0; t < T - 1; t++) {
            grad_logits[b * T + t] = 0.0f;
        }
        float o = to_float(dec[((long)b * T + T - 1) * od1]);
        float dfdo;
        float f = fc_link(p, o, &dfdo);
        float l, g;
        int cap;
        fc_loss(p, f, y[b], &l, &g, &cap);
        float m = mask[b];
        if (!isfinite(l) || !isfinite(g) || !isfinite(o)) {
            l = 0.0f;
            g = 0.0f;
            sn += m;
            cap = 0;
        }
        grad_logits[b * T + T - 1] = m * g * dfdo * inv_n;
        sl += m * l;
        sc += m * (float)cap;
    }
    sm[tid] = sl;
    sm[256 + tid] = sc;
    sm[512 + tid] = sn;
    block_reduce_sum(sm, step_out, tid, blockDim.x, 3);
}

// Minibatch gather: obs[b, t, :] = Z[r_b - T + 1 + t, :] with r_b = row0 + s_b, s_b = perm[e, k*B + b]
// (or k*B + b without perm); s_b outside [0, n) is padding (mask 0, row0's sequence as filler).
__global__ void fc_gather(const float* __restrict__ Z, int P, const float* __restrict__ yrow,
        const int* __restrict__ perm, const int* __restrict__ epoch_ptr, int perm_stride,
        int step_off, int B, int T, int row0, int n, precision_t* __restrict__ obs,
        float* __restrict__ ymb, float* __restrict__ mask) {
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long tot = (long)B * T * P;
    if (idx >= tot) {
        return;
    }
    int c = (int)(idx % P);
    long bt = idx / P;
    int t = (int)(bt % T);
    int b = (int)(bt / T);
    int s = (perm != NULL) ? perm[(long)(*epoch_ptr) * perm_stride + step_off + b] : step_off + b;
    bool real = (s >= 0 && s < n);
    int r = row0 + (real ? s : 0);
    obs[idx] = from_float(Z[(long)(r - T + 1 + t) * P + c]);
    if (t == 0 && c == 0) {
        ymb[b] = (yrow != NULL) ? yrow[r] : 0.0f;
        mask[b] = real ? 1.0f : 0.0f;
    }
}

// The same gather with one block per sequence (2026-10-13 speed review, gate 4): row r's window is the T
// contiguous rows r - T + 1 .. r of Z, so obs[b] is one contiguous copy of T * P values (no per-element
// div / mod).
__global__ void fc_gather_seq(const float* __restrict__ Z, int P, const float* __restrict__ yrow,
        const int* __restrict__ perm, const int* __restrict__ epoch_ptr, int perm_stride,
        int step_off, int T, int row0, int n, precision_t* __restrict__ obs,
        float* __restrict__ ymb, float* __restrict__ mask) {
    int b = blockIdx.x;
    int s = (perm != NULL) ? perm[(long)(*epoch_ptr) * perm_stride + step_off + b] : step_off + b;
    bool real = (s >= 0 && s < n);
    int r = row0 + (real ? s : 0);
    const float* src = Z + (long)(r - T + 1) * P;
    precision_t* dst = obs + (long)b * T * P;
    int tp = T * P;
    for (int i = threadIdx.x; i < tp; i += blockDim.x) {
        dst[i] = from_float(src[i]);
    }
    if (threadIdx.x == 0) {
        ymb[b] = (yrow != NULL) ? yrow[r] : 0.0f;
        mask[b] = real ? 1.0f : 0.0f;
    }
}

// run-wide switches (cmd_run): cuBLAS math (0 fp32 = cuBLAS default; 1 TF32 for Muon's Newton-Schulz GEMMs
// only, --tf32-ns; 2 TF32 for every GEMM, --tf32) and the gather kernel (FT_GATHER_OLD=1: fc_gather, for
// timing A/Bs only; the two write the same bytes)
static int g_math = 0;
static int g_gather_old = 0;
static const char* math_name() {
    return g_math == 0 ? "fp32" : (g_math == 1 ? "tf32-ns" : "tf32");
}

static void gather(cudaStream_t st, const float* Z, int P, const float* yrow, const int* perm, const int* epoch_ptr,
        int perm_stride, int step_off, int B, int T, int row0, int n, precision_t* obs, float* ymb, float* mask) {
    if (g_gather_old) {
        long tot = (long)B * T * P;
        fc_gather<<<(int)((tot + 255) / 256), 256, 0, st>>>(Z, P, yrow, perm, epoch_ptr, perm_stride, step_off, B,
            T, row0, n, obs, ymb, mask);
    } else {
        fc_gather_seq<<<B, 256, 0, st>>>(Z, P, yrow, perm, epoch_ptr, perm_stride, step_off, T, row0, n, obs, ymb,
            mask);
    }
}

// Per-step log (1 thread): mean loss, global grad norm (Muon's pre-clip sum of squares), counts.
__global__ void fc_step_log(const float* __restrict__ step_out, const float* __restrict__ grad_sumsq,
        const int* __restrict__ epoch_ptr, int S, int k, float n_real, float* __restrict__ log_loss,
        float* __restrict__ log_gn, float* __restrict__ log_cap, float* __restrict__ log_nf,
        int* __restrict__ nf_total) {
    long i = (long)(*epoch_ptr) * S + k;
    log_loss[i] = step_out[0] / n_real;
    log_gn[i] = sqrtf(*grad_sumsq);
    log_cap[i] = step_out[1];
    log_nf[i] = step_out[2];
    *nf_total += (int)step_out[2];
}

__global__ void fc_epoch_begin(const float* __restrict__ lr_sched, const int* __restrict__ epoch_ptr,
        float* __restrict__ lr_dev) {
    *lr_dev = lr_sched[*epoch_ptr];
}

__global__ void fc_epoch_end(int* __restrict__ epoch_ptr) {
    *epoch_ptr += 1;
}

// Evaluation (1 block): f for each real row -> pred; [sum sq err, sum qlike] -> val log at epoch e; and,
// with track != 0, keep the best epoch (strictly lower validation MSE) and its predictions -- only while
// the run is alive under early stopping (e - best_epoch < patience: the LSTM's loop, which breaks at
// the first epoch with ep - best_ep >= patience), so epochs run past the stop never change the result.
__global__ void fc_eval(const precision_t* __restrict__ dec, int T, int od1,
        const float* __restrict__ y, int n, const LossParams* __restrict__ lpd, float* __restrict__ pred,
        const int* __restrict__ epoch_ptr, float* __restrict__ val_mse_log,
        float* __restrict__ val_ql_log, int track, int patience, float* __restrict__ best,
        int* __restrict__ best_epoch, float* __restrict__ best_pred) {
    __shared__ float sm[2 * 256];
    __shared__ float red[2];
    __shared__ int improved;
    const LossParams p = *lpd;
    int tid = threadIdx.x;
    float se = 0.0f, sq = 0.0f;
    LossParams pq = p;
    pq.kind = LOSS_QLIKE;
    pq.qlike_cap = 3.0e38f;
    for (int b = tid; b < n; b += blockDim.x) {
        float o = to_float(dec[((long)b * T + T - 1) * od1]);
        float dfdo;
        float f = fc_link(p, o, &dfdo);
        pred[b] = f;
        if (y != NULL) {
            float e = f - y[b];
            se += e * e;
            float l, g;
            int cap;
            fc_loss(pq, f, y[b], &l, &g, &cap);
            sq += l;
        }
    }
    sm[tid] = se;
    sm[256 + tid] = sq;
    block_reduce_sum(sm, red, tid, blockDim.x, 2);
    __syncthreads();
    if (tid == 0) {
        improved = 0;
        if (y != NULL) {
            int e = *epoch_ptr;
            float mse = red[0] / (float)n;
            val_mse_log[e] = mse;
            val_ql_log[e] = red[1] / (float)n;
            if (track && e - *best_epoch < patience && isfinite(mse) && mse < *best) {
                *best = mse;
                *best_epoch = e + 1;
                improved = 1;
            }
        }
    }
    __syncthreads();
    if (improved) {
        for (int b = tid; b < n; b += blockDim.x) {
            best_pred[b] = pred[b];
        }
    }
}

// ============================================================================ data and jobs
struct Data {
    int n = 0, p = 0;
    std::vector<float> X, y;
    std::vector<double> X64;  // the float64 design (X.f64): the window mask compares float64 bytes
    std::string dir;
};

static void load_x64(Data* d) {
    if (!d->X64.empty()) {
        return;
    }
    d->X64.resize((size_t)d->n * d->p);
    FILE* f = fopen((d->dir + "/X.f64").c_str(), "rb");
    if (!f || fread(d->X64.data(), 8, d->X64.size(), f) != d->X64.size()) {
        fprintf(stderr, "mask=1 needs %s/X.f64 (%d x %d float64)\n", d->dir.c_str(), d->n, d->p);
        exit(2);
    }
    fclose(f);
}

// The linear arms' identifiability mask (src/models/window_mask.py window_keep), the same rule on the
// same window: a column is CONSTANT if every row of the window has the bytes of its first row (float64
// bit patterns), a COPY if its bytes over the window equal those of an earlier kept column; kept = every
// other column, in column order.
static std::vector<int> window_keep(Data* d, int r0, int r1) {
    load_x64(d);
    int p = d->p;
    std::vector<int> keep;
    std::map<uint64_t, std::vector<int>> seen;
    for (int c = 0; c < p; c++) {
        const double* first = &d->X64[(size_t)r0 * p + c];
        bool cst = true;
        uint64_t h = 1469598103934665603ULL;
        for (int r = r0; r < r1; r++) {
            const double* v = &d->X64[(size_t)r * p + c];
            if (cst && memcmp(v, first, 8) != 0) {
                cst = false;
            }
            uint64_t b;
            memcpy(&b, v, 8);
            h = (h ^ b) * 1099511628211ULL;
        }
        if (cst) {
            continue;
        }
        bool dup = false;
        for (int k : seen[h]) {
            bool same = true;
            for (int r = r0; r < r1 && same; r++) {
                same = memcmp(&d->X64[(size_t)r * p + c], &d->X64[(size_t)r * p + k], 8) == 0;
            }
            if (same) {
                dup = true;
                break;
            }
        }
        if (dup) {
            continue;
        }
        seen[h].push_back(c);
        keep.push_back(c);
    }
    return keep;
}

static std::map<std::string, Data*> g_data;

// the window mask of (design, window): computed once, used by the pool's slot matching and trainer_host
static std::map<std::tuple<std::string, int, int>, std::vector<int>> g_keep_cache;

static const std::vector<int>& keep_for(Data* d, int w0, int w1) {
    auto key = std::make_tuple(d->dir, w0, w1);
    auto it = g_keep_cache.find(key);
    if (it != g_keep_cache.end()) {
        return it->second;
    }
    return g_keep_cache[key] = window_keep(d, w0, w1);
}

static Data* load_data(const std::string& dir) {
    auto it = g_data.find(dir);
    if (it != g_data.end()) {
        return it->second;
    }
    Data* d = new Data();
    d->dir = dir;
    FILE* fm = fopen((dir + "/meta.txt").c_str(), "r");
    if (!fm || fscanf(fm, "%d %d", &d->n, &d->p) != 2) {
        fprintf(stderr, "bad meta in %s\n", dir.c_str());
        exit(2);
    }
    fclose(fm);
    d->X.resize((size_t)d->n * d->p);
    d->y.resize(d->n);
    FILE* fx = fopen((dir + "/X.f32").c_str(), "rb");
    FILE* fy = fopen((dir + "/y.f32").c_str(), "rb");
    if (!fx || !fy || fread(d->X.data(), 4, d->X.size(), fx) != d->X.size()
            || fread(d->y.data(), 4, d->y.size(), fy) != d->y.size()) {
        fprintf(stderr, "bad arrays in %s\n", dir.c_str());
        exit(2);
    }
    fclose(fx);
    fclose(fy);
    g_data[dir] = d;
    return d;
}

struct Job {
    std::map<std::string, std::string> kv;
    std::string id, data, mode, loss;
    int w0, w1, val_tail, embargo, fc_n, L, H, layers, batch, max_epochs, patience, epochs, seed;
    float lr, min_lr_ratio, momentum, max_grad_norm, huber_q, qlike_eps, qlike_cap_q;
    int anneal, save_params, mask;
    float zclip;
    int xstd;
    int fc_train;
    float stop_loss;
    std::string link;
    std::string epochs_from, init_from;
    std::string get(const char* k, const char* dflt) const {
        auto it = kv.find(k);
        return it == kv.end() ? std::string(dflt) : it->second;
    }
};

static std::vector<Job> read_jobs(const char* path) {
    std::vector<Job> jobs;
    FILE* f = fopen(path, "r");
    if (!f) {
        fprintf(stderr, "cannot open %s\n", path);
        exit(2);
    }
    char line[8192];
    while (fgets(line, sizeof(line), f)) {
        if (line[0] == '#' || line[0] == '\n') {
            continue;
        }
        Job j;
        char* save = NULL;
        for (char* tok = strtok_r(line, " \t\r\n", &save); tok; tok = strtok_r(NULL, " \t\r\n", &save)) {
            char* eq = strchr(tok, '=');
            if (!eq) {
                continue;
            }
            *eq = 0;
            j.kv[tok] = eq + 1;
        }
        j.id = j.get("id", "");
        j.data = j.get("data", "");
        j.mode = j.get("mode", "tune");
        j.loss = j.get("loss", "mse");
        j.w0 = atoi(j.get("w0", "-1").c_str());
        j.w1 = atoi(j.get("w1", "-1").c_str());
        j.val_tail = atoi(j.get("val_tail", "125").c_str());
        j.embargo = atoi(j.get("embargo", "25").c_str());
        j.fc_n = atoi(j.get("fc_n", "10").c_str());
        j.L = atoi(j.get("L", "20").c_str());
        j.H = atoi(j.get("H", "64").c_str());
        j.layers = atoi(j.get("layers", "1").c_str());
        j.batch = atoi(j.get("batch", "128").c_str());
        j.max_epochs = atoi(j.get("max_epochs", "200").c_str());
        j.patience = atoi(j.get("patience", "20").c_str());
        j.epochs = atoi(j.get("epochs", "0").c_str());
        j.epochs_from = j.get("epochs_from", "");
        j.init_from = j.get("init_from", "");      // warm start: flat fp32 params file (upstream .bin layout)
        j.save_params = atoi(j.get("save_params", "0").c_str());
        j.mask = atoi(j.get("mask", "0").c_str());        // 1: the linear arms' window mask on [w0, w1)
        j.zclip = atof(j.get("zclip", "0").c_str());      // > 0: clip standardised inputs at +-zclip
        // 1 (default): standardise every input column on the window (the LSTM spec's rule); 0 (2026-10-13,
        // Phase 2c): feed the design's kept columns unchanged -- the design is already the regular stack's
        // causally prescaled design (run_executor prescale=True: rolling median / IQR with the executor's
        // scale guards, the *_avail_ma_* / *_active_ma_* fractions passed raw), which the linear arms use
        // as is; a second standardisation undoes those guards
        j.xstd = atoi(j.get("xstd", "1").c_str());
        // 1 (2026-10-13, opt-in; refit jobs): after the last epoch, forecast every training row with the final
        // weights (preds.csv kind "train": the exact in-sample fit), and the forecast rows once more through
        // that same path (kind "fc_trainpath", a check against the eval path's "fc")
        j.fc_train = atoi(j.get("fc_train", "0").c_str());
        // > 0 (2026-10-13, opt-in; refit jobs, --pool): stop at the first chunk end whose last epoch's training
        // loss (row-weighted mean of the minibatch losses) is <= stop_loss * sd_y^2 (sd_y: the trainer's
        // training-target sd); every epoch up to the stop is the run without it, byte for byte
        j.stop_loss = atof(j.get("stop_loss", "0").c_str());
        // identity (default): f = mu_y + sd_y o;  softplus (user, 2026-10-13): f = softplus(c + s o) > 0
        j.link = j.get("link", "identity");
        j.seed = atoi(j.get("seed", "73").c_str());
        j.lr = atof(j.get("lr", "0.015").c_str());
        j.min_lr_ratio = atof(j.get("min_lr_ratio", "0.0").c_str());
        j.anneal = atoi(j.get("anneal", "1").c_str());
        j.momentum = atof(j.get("momentum", "0.95").c_str());
        j.max_grad_norm = atof(j.get("max_grad_norm", "1.0").c_str());
        j.huber_q = atof(j.get("huber_q", "0.9").c_str());
        j.qlike_eps = atof(j.get("qlike_eps", "0.001").c_str());
        j.qlike_cap_q = atof(j.get("qlike_cap_q", "0.99").c_str());
        if (j.id.empty() || j.data.empty() || j.w0 < 0 || j.w1 <= j.w0) {
            fprintf(stderr, "malformed job line (id / data / w0 / w1)\n");
            exit(2);
        }
        // (2026-10-13 review) every key must be one this program reads: a misspelt key would otherwise
        // silently fall back to its default
        static const char* known[] = {"id", "data", "mode", "loss", "w0", "w1", "val_tail", "embargo", "fc_n",
            "L", "H", "layers", "batch", "max_epochs", "patience", "epochs", "epochs_from", "init_from",
            "save_params", "mask", "zclip", "xstd", "fc_train", "seed", "lr", "min_lr_ratio", "anneal",
            "momentum", "max_grad_norm", "huber_q", "qlike_eps", "qlike_cap_q", "phase", "stop_loss", "link"};
        for (auto& kv : j.kv) {
            bool ok = false;
            for (const char* k : known) {
                ok = ok || kv.first == k;
            }
            if (!ok) {
                fprintf(stderr, "job %s: unknown key %s\n", j.id.c_str(), kv.first.c_str());
                exit(2);
            }
        }
        if (j.link != "identity" && j.link != "softplus") {
            fprintf(stderr, "job %s: link must be identity or softplus\n", j.id.c_str());
            exit(2);
        }
        if (j.mode != "tune" && j.mode != "refit") {
            fprintf(stderr, "job %s: mode must be tune or refit\n", j.id.c_str());
            exit(2);
        }
        if (!(j.huber_q >= 0.0f && j.huber_q <= 1.0f) || !(j.qlike_cap_q >= 0.0f && j.qlike_cap_q <= 1.0f)) {
            fprintf(stderr, "job %s: huber_q and qlike_cap_q must lie in [0, 1]\n", j.id.c_str());
            exit(2);
        }
        if (j.mode == "tune" && j.val_tail < 1) {
            fprintf(stderr, "job %s: a tune job needs val_tail >= 1\n", j.id.c_str());
            exit(2);
        }
        if (j.fc_train && j.mode == "refit" && j.fc_n < 1) {
            fprintf(stderr, "job %s: fc_train=1 needs fc_n >= 1 (its rows go through the forecast path)\n",
                j.id.c_str());
            exit(2);
        }
        if (!(j.stop_loss >= 0.0f) || (j.stop_loss > 0.0f && j.mode == "tune")) {
            fprintf(stderr, "job %s: stop_loss must be >= 0, and only on refit jobs\n", j.id.c_str());
            exit(2);
        }
        if (j.save_params && j.mode == "tune") {  // a tune job trains up to chunk - 1 epochs past its stop
            fprintf(stderr, "job %s: save_params is for refit jobs only\n", j.id.c_str());
            exit(2);
        }
        if (j.L < 1 || j.batch < 1 || j.layers < 1 || j.H < 4 || j.max_epochs < 1) {
            fprintf(stderr, "job %s: L, batch, layers, max_epochs must be >= 1 and H >= 4\n", j.id.c_str());
            exit(2);
        }
        if (j.H % 4 != 0) {
            fprintf(stderr, "job %s: H must be a multiple of 4 (Muon's flat-offset layout)\n", j.id.c_str());
            exit(2);
        }
        jobs.push_back(j);
    }
    fclose(f);
    return jobs;
}

static float quantile(std::vector<double> v, double q) {
    std::sort(v.begin(), v.end());
    double pos = q * (v.size() - 1);
    size_t i = (size_t)pos;
    double fr = pos - i;
    return (float)(i + 1 < v.size() ? v[i] * (1 - fr) + v[i + 1] * fr : v[i]);
}

// splitmix64 (ours): the minibatch order of epoch e of a seed
static uint64_t splitmix64(uint64_t& s) {
    uint64_t z = (s += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

// ============================================================================ one model
struct Trainer {
    Job job;
    int tune = 0;
    // geometry
    int P = 0, T = 0, B = 0, S = 0, N = 0, row0 = 0, n_ctx = 0, n_eval = 0, eval_row0 = 0, B_eval = 0;
    int epochs_target = 0;
    LossParams lp;
    float mu_y = 0, sd_y = 1;
    // upstream objects
    cudaStream_t stream = NULL;
    cublasHandle_t h = NULL, hdw = NULL;
    void *ws = NULL, *wsdw = NULL;
    cudaStream_t dws = NULL;
    cudaEvent_t ev_ready = NULL, ev_done = NULL;
    Arch arch;
    Weights w;
    Allocator params = {}, grads = {}, acts = {}, ev_acts_alloc = {}, ev_grads = {};
    Activations tr_acts, ev_acts;
    Muon muon;
    Float master;
    Prec grad;
    // our buffers
    float *d_Z = NULL, *d_y = NULL, *d_ymb = NULL, *d_mask = NULL, *d_gl = NULL, *d_gv = NULL;
    precision_t *d_obs = NULL, *d_state = NULL, *d_eobs = NULL, *d_estate = NULL;
    float *d_eymb = NULL, *d_emask = NULL;
    int* d_perm = NULL;
    int* d_epoch = NULL;
    float* d_lr_sched = NULL;
    float* d_step_out = NULL;
    float *d_log_loss = NULL, *d_log_gn = NULL, *d_log_cap = NULL, *d_log_nf = NULL;
    int* d_nf_total = NULL;
    float *d_val_mse = NULL, *d_val_ql = NULL, *d_best = NULL, *d_pred = NULL, *d_best_pred = NULL;
    int* d_best_epoch = NULL;
    LossParams* d_lp = NULL;
    float* d_rows = NULL;  // fc_train: the training and forecast rows' forecasts (N + n_eval)
    cudaGraphExec_t graph = NULL;
    // host-side job data (trainer_host) for the device load (trainer_load)
    std::vector<float> hZ, hY;
    const std::vector<int>* hperm = NULL;
    const std::vector<int>* loaded_perm = NULL;  // the permutation now on the device (pool reuse)
    // host results
    int epochs_run = 0, best_epoch = 0, nf_total = 0;
    float best_val = 0;
    std::string status = "ok";
    std::vector<int> keep;  // input columns used (all, or the window mask's kept columns)
    long n_clipped = 0;     // standardised input values clipped (window + forecast rows)
    double t_setup = 0, t_capture = 0, t_start = 0, t_end = 0;
    std::vector<float> lr_sched;
};

static void bind_cublas(Trainer& t) {
    g_cublas_handle = t.h;
    g_cublas_workspace = t.ws;
    g_cublas_dw_handle = t.hdw;
    g_cublas_dw_workspace = t.wsdw;
    g_dw_stream = t.dws;
    g_main_ready = t.ev_ready;
    g_dw_done = t.ev_done;
}

// forward of the native network on obs [B, T, P] (arch_forward_train without the PPO cache kernel)
static Prec net_forward(Arch* a, Weights& w, Activations& acts, precision_t* obs, int B, int T, int P,
        precision_t* state, int layers, int H, cudaStream_t stream) {
    Prec x = {.data = obs, .shape = {B, T, P}};
    Prec st = {.data = state, .shape = {layers, B, H}};
    Prec term = {};
    Prec h = a->encoder.forward(w.encoder, acts.encoder, *puf_squeeze(&x, 0), stream);
    h = a->network.forward_train(w.network, *puf_unsqueeze(&h, 0, B, T), st, term, acts.network, 0, stream);
    return a->decoder.forward(w.decoder, acts.decoder, *puf_squeeze(&h, 0), stream);
}

static void enqueue_step(Trainer& t, int k) {
    const Job& j = t.job;
    int n_real = std::min(t.B, t.N - k * t.B);
    gather(t.stream, t.d_Z, t.P, t.d_y, t.d_perm, t.d_epoch, t.S * t.B, k * t.B, t.B, t.T, t.row0, t.N, t.d_obs,
        t.d_ymb, t.d_mask);
    Prec dec = net_forward(&t.arch, t.w, t.tr_acts, t.d_obs, t.B, t.T, t.P, t.d_state, j.layers, j.H, t.stream);
    fc_loss_train<<<1, 256, 0, t.stream>>>(dec.data, t.T, 2, t.d_ymb, t.d_mask, t.B, t.d_lp,
        1.0f / (float)n_real, t.d_gl, t.d_step_out);
    Float gl = {.data = t.d_gl, .shape = {t.B, t.T, 1}};
    Float gv = {.data = t.d_gv, .shape = {t.B, t.T}};
    arch_backward(&t.arch, t.w, t.tr_acts, gl, Float(), gv, t.stream);
    if (g_math == 1) {  // --tf32-ns: Muon's GEMMs (Newton-Schulz) in TF32; forward / backward stay fp32
        cublasSetMathMode(t.h, CUBLAS_TF32_TENSOR_OP_MATH);
    }
    muon_step(&t.muon, t.master, t.grad, j.max_grad_norm, t.stream);
    if (g_math == 1) {
        cublasSetMathMode(t.h, CUBLAS_DEFAULT_MATH);
    }
    fc_step_log<<<1, 1, 0, t.stream>>>(t.d_step_out, t.muon.grad_norm, t.d_epoch, t.S, k, (float)n_real,
        t.d_log_loss, t.d_log_gn, t.d_log_cap, t.d_log_nf, t.d_nf_total);
}

static void enqueue_eval(Trainer& t, int track) {
    const Job& j = t.job;
    gather(t.stream, t.d_Z, t.P, t.tune ? t.d_y : NULL, NULL, NULL, 0, 0, t.B_eval, t.T, t.eval_row0, t.n_eval,
        t.d_eobs, t.d_eymb, t.d_emask);
    Prec dec = net_forward(&t.arch, t.w, t.ev_acts, t.d_eobs, t.B_eval, t.T, t.P, t.d_estate, j.layers, j.H,
        t.stream);
    fc_eval<<<1, 256, 0, t.stream>>>(dec.data, t.T, 2, t.tune ? t.d_eymb : NULL, t.n_eval, t.d_lp, t.d_pred,
        t.d_epoch, t.d_val_mse, t.d_val_ql, track, j.patience, t.d_best, t.d_best_epoch, t.d_best_pred);
}

// forecasts of n consecutive window rows (row0, row0 + 1, ...) with the current weights, into out[0 .. n):
// the forecast path itself (enqueue_eval's kernels, activations and batch shape B_eval = fc_n), B_eval rows
// at a time, so a row's forecast is the bytes the forecast path gives for it
static void enqueue_rows(Trainer& t, float* out, int row0, int n) {
    const Job& j = t.job;
    for (int k = 0; k * t.B_eval < n; k++) {
        int n_real = std::min(t.B_eval, n - k * t.B_eval);
        gather(t.stream, t.d_Z, t.P, NULL, NULL, NULL, 0, k * t.B_eval, t.B_eval, t.T, row0, n, t.d_eobs, t.d_eymb,
            t.d_emask);
        Prec dec = net_forward(&t.arch, t.w, t.ev_acts, t.d_eobs, t.B_eval, t.T, t.P, t.d_estate, j.layers, j.H,
            t.stream);
        fc_eval<<<1, 256, 0, t.stream>>>(dec.data, t.T, 2, NULL, n_real, t.d_lp, out + (long)k * t.B_eval,
            t.d_epoch, NULL, NULL, 0, 0, NULL, NULL, NULL);
    }
}

// fc_train=1: the training rows' forecasts (kind "train") and the forecast rows' once more (kind
// "fc_trainpath": equals kind "fc" byte for byte) through enqueue_rows, appended to preds.csv
static void write_trainrows(FILE* preds, Trainer& t) {
    const Job& j = t.job;
    if (!j.fc_train || t.tune) {
        return;
    }
    bind_cublas(t);
    int n_all = t.N + t.n_eval;
    enqueue_rows(t, t.d_rows, t.row0, t.N);
    enqueue_rows(t, t.d_rows + t.N, t.eval_row0, t.n_eval);
    std::vector<float> h(n_all);
    CK(cudaMemcpyAsync(h.data(), t.d_rows, n_all * 4, cudaMemcpyDeviceToHost, t.stream));
    CK(cudaStreamSynchronize(t.stream));
    for (int s = 0; s < t.N; s++) {
        fprintf(preds, "%s,train,%d,%.9g\n", j.id.c_str(), j.w0 + t.row0 + s, h[s]);
    }
    for (int b = 0; b < t.n_eval; b++) {
        fprintf(preds, "%s,fc_trainpath,%d,%.9g\n", j.id.c_str(), j.w0 + t.eval_row0 + b, h[t.N + b]);
    }
}

static void enqueue_epoch(Trainer& t) {
    fc_epoch_begin<<<1, 1, 0, t.stream>>>(t.d_lr_sched, t.d_epoch, t.muon.lr);
    for (int k = 0; k < t.S; k++) {
        enqueue_step(t, k);
    }
    if (t.tune) {
        enqueue_eval(t, 1);
    }
    fc_epoch_end<<<1, 1, 0, t.stream>>>(t.d_epoch);
}

template <typename T_>
static T_* dalloc(size_t n) {
    T_* p = NULL;
    CK(cudaMalloc((void**)&p, std::max<size_t>(n, 1) * sizeof(T_)));
    CK(cudaMemset(p, 0, std::max<size_t>(n, 1) * sizeof(T_)));
    return p;
}

// stream-ordered versions (2026-10-13 speed review, gate 4): a slot's shape buffers come from the device's
// memory pool on the slot's own stream and go back to it there, so a pool re-shape never synchronises the device
template <typename T_>
static T_* dalloc_s(size_t n, cudaStream_t st) {
    T_* p = NULL;
    CK(cudaMallocAsync((void**)&p, std::max<size_t>(n, 1) * sizeof(T_), st));
    CK(cudaMemsetAsync(p, 0, std::max<size_t>(n, 1) * sizeof(T_), st));
    return p;
}

// upstream alloc_create's layout (16-byte aligned entries in one block, zeroed), stream-ordered
static void alloc_create_s(Allocator* a, cudaStream_t st) {
    CK(cudaMallocAsync(&a->mem, std::max<long>(a->total_bytes, 1), st));
    CK(cudaMemsetAsync(a->mem, 0, std::max<long>(a->total_bytes, 1), st));
    long offset = 0;
    for (int i = 0; i < a->num_regs; i++) {
        offset = (offset + 15) & ~15;
        *a->regs[i].data_ptr = (char*)a->mem + offset;
        offset += numel(a->regs[i].shape) * a->regs[i].elem_size;
    }
}

static void free_alloc_s(Allocator& a, cudaStream_t st) {
    if (a.mem) {
        CK(cudaFreeAsync(a.mem, st));
    }
    free(a.regs);
    a = Allocator{};
}

// upstream muon_init's registrations (momentum buffer, Gram / Newton-Schulz scratch sized by the largest 2-D
// parameter) for a re-shaped slot, keeping the slot's four Muon scalars (muon_init allocates them anew)
static void muon_reregister(Muon* m, Allocator* param_alloc, double momentum, Allocator* alloc) {
    m->momentum = momentum;
    m->param_alloc = param_alloc;
    m->mb = {.shape = {param_alloc->total_elems}};
    alloc_register(alloc, &m->mb);
    long max_M = 0, max_N = 0;
    for (int i = 0; i < param_alloc->num_regs; i++) {
        AllocEntry& e = param_alloc->regs[i];
        if (ndim(e.shape) >= 2) {
            long R = e.shape[0], C = numel(e.shape) / R;
            max_M = std::max(max_M, std::min(R, C));
            max_N = std::max(max_N, std::max(R, C));
        }
    }
    m->gram = {.shape = {max_M, max_M}};
    m->gram_buf = {.shape = {max_M, max_M}};
    m->x_buf = {.shape = {max_M, max_N}};
    alloc_register(alloc, &m->gram);
    alloc_register(alloc, &m->gram_buf);
    alloc_register(alloc, &m->x_buf);
}

// ---- caches (identical bytes whichever path or order computes them)
// minibatch order of every epoch (host RNG, deterministic): S*B slots per epoch, a permutation then -1 padding
static std::map<std::tuple<int, int, int, int, int>, std::vector<int>> g_perm_cache;

static const std::vector<int>* perm_for(int seed, int N, int S, int B, int maxe) {
    auto key = std::make_tuple(seed, N, S, B, maxe);
    auto it = g_perm_cache.find(key);
    if (it != g_perm_cache.end()) {
        return &it->second;
    }
    std::vector<int>& perm = g_perm_cache[key];
    perm.assign((size_t)maxe * S * B, -1);
    uint64_t rs = 0x5EED0000ULL + (uint64_t)seed * 1000003ULL;
    std::vector<int> idx(N);
    for (int e = 0; e < maxe; e++) {
        for (int i = 0; i < N; i++) {
            idx[i] = i;
        }
        for (int i = N - 1; i > 0; i--) {
            int k = (int)(splitmix64(rs) % (uint64_t)(i + 1));
            std::swap(idx[i], idx[k]);
        }
        std::copy(idx.begin(), idx.end(), perm.begin() + (size_t)e * S * B);
    }
    return &perm;
}

// initial weights (pool mode): upstream weights_init's output for (P, H, layers, seed), kept and re-used
static std::map<std::tuple<int, int, int, int>, std::vector<float>> g_w_cache;

// Host part of a job: geometry, the window mask, causal standardisation (double), targets, loss
// parameters, the lr schedule, the permutation (cached per seed).
static void trainer_host(Trainer& t) {
    const Job& j = t.job;
    Data* d = load_data(j.data);
    t.tune = (j.mode == "tune");
    int W = j.w1 - j.w0;
    t.T = j.L;
    t.B = j.batch;
    int f_end = t.tune ? (W - j.val_tail - j.embargo) : W;  // ctx-relative end of the training rows
    if (t.tune && j.L - 1 > j.embargo) {
        fprintf(stderr, "job %s: L - 1 > embargo (a validation sequence would reach the fit block)\n", j.id.c_str());
        exit(2);
    }
    t.row0 = j.L - 1;                     // sequences entirely inside the window (the LSTM's tr_ends)
    t.N = f_end - t.row0;
    if (t.N < 1) {
        fprintf(stderr, "job %s: no training rows (window too short for L, val_tail and embargo)\n", j.id.c_str());
        exit(2);
    }
    t.S = (t.N + t.B - 1) / t.B;
    t.n_ctx = t.tune ? W : W + j.fc_n;
    if (j.w0 + t.n_ctx > d->n) {
        fprintf(stderr, "job %s: rows beyond the data\n", j.id.c_str());
        exit(2);
    }
    if (t.tune) {
        t.eval_row0 = W - j.val_tail;
        t.n_eval = j.val_tail;
    } else {
        t.eval_row0 = W;
        t.n_eval = j.fc_n;
    }
    t.B_eval = t.n_eval;
    // --- causal standardisation: X over the training block's rows, y over the training targets
    int pall = d->p;
    if (j.mask) {
        t.keep = keep_for(d, j.w0, j.w1);  // the fit's window [w0, w1) (a tune job: its whole window)
    } else {
        t.keep.resize(pall);
        for (int c = 0; c < pall; c++) {
            t.keep[c] = c;
        }
    }
    int p = (int)t.keep.size();
    t.P = p + 1;  // + constant column (the upstream encoder / decoder have no bias)
    std::vector<double> mu(p, 0.0), sd(p, 0.0);
    if (!j.xstd) {  // the prescaled design's kept columns, unchanged (float32 bit for bit), no clip
        std::vector<float>& Z = t.hZ;
        Z.assign((size_t)t.n_ctx * t.P, 0.0f);
        for (int r = 0; r < t.n_ctx; r++) {
            const float* x = &d->X[(size_t)(j.w0 + r) * pall];
            float* z = &Z[(size_t)r * t.P];
            for (int c = 0; c < p; c++) {
                z[c] = x[t.keep[c]];
            }
            z[p] = 1.0f;
        }
        t.n_clipped = 0;
    }
    for (int r = 0; j.xstd && r < f_end; r++) {
        const float* x = &d->X[(size_t)(j.w0 + r) * pall];
        for (int c = 0; c < p; c++) {
            mu[c] += x[t.keep[c]];
        }
    }
    for (int c = 0; j.xstd && c < p; c++) {
        mu[c] /= f_end;
    }
    for (int r = 0; j.xstd && r < f_end; r++) {
        const float* x = &d->X[(size_t)(j.w0 + r) * pall];
        for (int c = 0; c < p; c++) {
            double e = x[t.keep[c]] - mu[c];
            sd[c] += e * e;
        }
    }
    for (int c = 0; c < p; c++) {
        sd[c] = sqrt(sd[c] / f_end);
        if (!(sd[c] > 0.0)) {
            sd[c] = 1.0;  // a column constant on the block: centred to zero (the LSTM's rule)
        }
    }
    std::vector<float>& Z = t.hZ;
    if (j.xstd) {
        Z.assign((size_t)t.n_ctx * t.P, 0.0f);
    }
    long n_clipped = 0;
    for (int r = 0; j.xstd && r < t.n_ctx; r++) {
        const float* x = &d->X[(size_t)(j.w0 + r) * pall];
        float* z = &Z[(size_t)r * t.P];
        for (int c = 0; c < p; c++) {
            double v = (x[t.keep[c]] - mu[c]) / sd[c];
            if (j.zclip > 0 && fabs(v) > j.zclip) {  // clip after standardisation on the window
                v = v > 0 ? j.zclip : -j.zclip;
                n_clipped++;
            }
            z[c] = (float)v;
        }
        z[p] = 1.0f;
    }
    if (j.xstd) {
        t.n_clipped = n_clipped;
    }
    double ym = 0.0, yv = 0.0;
    std::vector<double> ytr;
    for (int s = 0; s < t.N; s++) {
        double v = d->y[j.w0 + t.row0 + s];
        ytr.push_back(v);
        ym += v;
    }
    ym /= t.N;
    for (double v : ytr) {
        yv += (v - ym) * (v - ym);
    }
    t.mu_y = (float)ym;
    t.sd_y = (float)sqrt(yv / t.N);
    // targets of the window rows only (never the forecast rows)
    std::vector<float>& yrow = t.hY;
    yrow.assign(t.n_ctx, 0.0f);
    for (int r = 0; r < W; r++) {
        yrow[r] = d->y[j.w0 + r];
    }
    // --- loss parameters from the training rows
    t.lp.kind = j.loss == "mse" ? LOSS_MSE : j.loss == "huber" ? LOSS_HUBER : j.loss == "qlike" ? LOSS_QLIKE : -1;
    if (t.lp.kind < 0) {
        fprintf(stderr, "job %s: unknown loss %s\n", j.id.c_str(), j.loss.c_str());
        exit(2);
    }
    t.lp.mu_y = t.mu_y;
    t.lp.sd_y = t.sd_y;
    t.lp.link = LINK_IDENTITY;
    t.lp.link_c = 0.0f;
    t.lp.link_s = 0.0f;
    if (j.link == "softplus") {
        if (!(t.mu_y > 0.0f)) {
            fprintf(stderr, "job %s: link=softplus needs mu_y > 0 (mu_y = %g)\n", j.id.c_str(), t.mu_y);
            exit(2);
        }
        double c = log(expm1((double)t.mu_y));  // softplus^-1(mu_y)
        t.lp.link = LINK_SOFTPLUS;
        t.lp.link_c = (float)c;
        t.lp.link_s = (float)((double)t.sd_y * (1.0 + exp(-c)));  // sd_y / sigmoid(c)
    }
    {
        std::vector<double> ad, gq;
        std::vector<double> ys = ytr;
        float med = quantile(ys, 0.5);
        for (double v : ytr) {
            ad.push_back(fabs(v - med));
        }
        t.lp.huber_delta = quantile(ad, j.huber_q);
        // QLIKE cap: the q-quantile of |dL/df| that the constant forecast f0 = mean(y) faces on the window
        float f0 = t.mu_y, eps2 = j.qlike_eps * j.qlike_eps;
        for (double v : ytr) {
            double r = std::max(v * v, (double)eps2) / (f0 * (double)f0 + eps2);
            gq.push_back(fabs(2.0 * f0 * (1.0 - r) / (f0 * (double)f0 + eps2)));
        }
        t.lp.qlike_cap = quantile(gq, j.qlike_cap_q);
        t.lp.qlike_eps = j.qlike_eps;
    }
    // --- epochs and the lr schedule (upstream cosine_annealing, pufferl.cu, over max_epochs)
    t.epochs_target = t.tune ? j.max_epochs : j.epochs;
    if (t.epochs_target < 1 || t.epochs_target > j.max_epochs) {  // lr schedule, permutations, logs: max_epochs
        fprintf(stderr, "job %s: epochs %d outside [1, max_epochs = %d]\n", j.id.c_str(), t.epochs_target,
            j.max_epochs);
        exit(2);
    }
    t.lr_sched.resize(j.max_epochs);
    for (int e = 0; e < j.max_epochs; e++) {
        float lr_min = j.min_lr_ratio * j.lr;
        double u = (double)e / (double)j.max_epochs;
        t.lr_sched[e] = j.anneal ? lr_min + 0.5f * (j.lr - lr_min) * (1.0f + (float)cos(M_PI * u)) : j.lr;
    }
    t.hperm = perm_for(j.seed, t.N, t.S, t.B, j.max_epochs);
    t.epochs_run = 0;
    t.best_epoch = 0;
    t.nf_total = 0;
    t.best_val = 0;
    t.status = "ok";
}

// Device allocation for the job's shape: stream, cuBLAS handles, the upstream network / activations / Muon
// (pufferl.cu create_pufferl order), our buffers.
// A slot's buffers for its job's shape.  The first call also creates the slot-lifetime state (stream, the two
// cuBLAS handles with their workspaces, the dW stream and events, Muon's scalars); a re-shape (pool) keeps it
// and replaces only the shape's buffers, stream-ordered (2026-10-13 speed review, gate 4: cublasDestroy /
// cudaFree synchronise the whole device and drained every other slot at each re-shape).
static void trainer_alloc(Trainer& t) {
    const Job& j = t.job;
    bool first = (t.stream == NULL);
    if (first) {
        CK(cudaStreamCreateWithFlags(&t.stream, cudaStreamNonBlocking));
        cublas_init_handle();
        t.h = g_cublas_handle; t.ws = g_cublas_workspace; t.hdw = g_cublas_dw_handle;
        t.wsdw = g_cublas_dw_workspace; t.dws = g_dw_stream; t.ev_ready = g_main_ready; t.ev_done = g_dw_done;
        if (g_math == 2) {  // --tf32: every GEMM of this slot in TF32
            cublasSetMathMode(t.h, CUBLAS_TF32_TENSOR_OP_MATH);
            cublasSetMathMode(t.hdw, CUBLAS_TF32_TENSOR_OP_MATH);
        }
    }
    bind_cublas(t);
    t.arch = build_arch(t.P, j.H, j.layers, 1, false, t.T);
    t.w = weights_create(&t.arch, &t.params);
    t.tr_acts = arch_reg_train(&t.arch, t.w, &t.acts, &t.grads, t.B * t.T);
    t.ev_acts = arch_reg_train(&t.arch, t.w, &t.ev_acts_alloc, &t.ev_grads, t.B_eval * t.T);
    if (first) {
        muon_init(&t.muon, &t.params, j.momentum, &t.acts);
    } else {
        muon_reregister(&t.muon, &t.params, j.momentum, &t.acts);
    }
    alloc_create_s(&t.params, t.stream);
    alloc_create_s(&t.grads, t.stream);
    alloc_create_s(&t.acts, t.stream);
    alloc_create_s(&t.ev_acts_alloc, t.stream);
    alloc_create_s(&t.ev_grads, t.stream);
    if (t.params.total_bytes != t.params.total_elems * (long)sizeof(precision_t)
            || t.grads.total_bytes != t.grads.total_elems * (long)sizeof(precision_t)
            || t.params.total_elems != t.grads.total_elems) {
        fprintf(stderr, "job %s: parameter layout has alignment padding (Muon assumes none)\n", j.id.c_str());
        exit(2);
    }
    t.grad = {.data = (precision_t*)t.grads.mem, .shape = {t.grads.total_elems}};
    t.master = {.data = (float*)t.params.mem, .shape = {t.params.total_elems}};  // fp32 build: params are master
    int maxe = j.max_epochs;
    t.d_Z = dalloc_s<float>((size_t)t.n_ctx * t.P, t.stream);
    t.d_y = dalloc_s<float>(t.n_ctx, t.stream);
    t.d_obs = dalloc_s<precision_t>((size_t)t.B * t.T * t.P, t.stream);
    t.d_ymb = dalloc_s<float>(t.B, t.stream);
    t.d_mask = dalloc_s<float>(t.B, t.stream);
    t.d_gl = dalloc_s<float>((size_t)t.B * t.T, t.stream);
    t.d_gv = dalloc_s<float>((size_t)t.B * t.T, t.stream);
    t.d_state = dalloc_s<precision_t>((size_t)j.layers * t.B * j.H, t.stream);
    t.d_eobs = dalloc_s<precision_t>((size_t)t.B_eval * t.T * t.P, t.stream);
    t.d_eymb = dalloc_s<float>(t.B_eval, t.stream);
    t.d_emask = dalloc_s<float>(t.B_eval, t.stream);
    t.d_estate = dalloc_s<precision_t>((size_t)j.layers * t.B_eval * j.H, t.stream);
    t.d_perm = dalloc_s<int>((size_t)maxe * t.S * t.B, t.stream);
    t.d_epoch = dalloc_s<int>(1, t.stream);
    t.d_lr_sched = dalloc_s<float>(maxe, t.stream);
    t.d_step_out = dalloc_s<float>(4, t.stream);
    t.d_log_loss = dalloc_s<float>((size_t)maxe * t.S, t.stream);
    t.d_log_gn = dalloc_s<float>((size_t)maxe * t.S, t.stream);
    t.d_log_cap = dalloc_s<float>((size_t)maxe * t.S, t.stream);
    t.d_log_nf = dalloc_s<float>((size_t)maxe * t.S, t.stream);
    t.d_nf_total = dalloc_s<int>(1, t.stream);
    t.d_val_mse = dalloc_s<float>(maxe, t.stream);
    t.d_val_ql = dalloc_s<float>(maxe, t.stream);
    t.d_best = dalloc_s<float>(1, t.stream);
    t.d_best_epoch = dalloc_s<int>(1, t.stream);
    t.d_pred = dalloc_s<float>(t.B_eval, t.stream);
    t.d_best_pred = dalloc_s<float>(t.B_eval, t.stream);
    t.d_lp = dalloc_s<LossParams>(1, t.stream);
    t.d_rows = dalloc_s<float>((size_t)t.N + t.n_eval, t.stream);
    t.loaded_perm = NULL;
    CK(cudaStreamSynchronize(0));  // muon_init's allocations (legacy stream): done before any slot-stream use
}

// Device state of one job on allocated buffers: initial weights (upstream weights_init, or its cached
// bytes for the same shape and seed), Muon state, the window's data, permutation, schedule, loss
// parameters, and every counter the epoch graph reads.  Every copy is ordered on the trainer's stream
// (pageable host sources are staged before the call returns), so loading one slot never waits for the
// others' work on the GPU.
// (2026-10-13) Work that upstream and our allocators put on the legacy default stream (alloc_create's and
// dalloc's cudaMemset; weights_init's cuRAND draws, which take no stream) is not ordered with our
// non-blocking slot streams.  In the pool, with the other slots' graphs keeping the GPU busy, a slot's
// weights_init (scale / cast kernels on its own stream) could read cuRAND's buffer before the draw had
// finished -- a few first-of-shape jobs started from different weights from run to run.  So: weights_init
// runs entirely on the legacy stream, and the host waits for the legacy stream after every allocation and
// every weights_init (cudaStreamSynchronize(0): the other slots' non-blocking streams keep running).
static const float g_inf = INFINITY;

static void trainer_load(Trainer& t, bool cache_weights) {
    const Job& j = t.job;
    auto wkey = std::make_tuple(t.P, j.H, j.layers, j.seed);
    auto wit = cache_weights ? g_w_cache.find(wkey) : g_w_cache.end();
    if (wit != g_w_cache.end()) {
        CK(cudaMemcpyAsync(t.params.mem, wit->second.data(), wit->second.size() * 4, cudaMemcpyHostToDevice,
            t.stream));
    } else {
        ulong seed = (ulong)j.seed;
        CK(cudaStreamSynchronize(t.stream));    // the stream-ordered zeroing of params done first
        weights_init(&t.arch, t.w, &seed, 0);  // the legacy stream (see above), with cuRAND's draws
        CK(cudaStreamSynchronize(0));
        if (cache_weights) {
            std::vector<float>& c = g_w_cache[wkey];
            c.resize(t.params.total_elems);
            CK(cudaMemcpy(c.data(), t.params.mem, c.size() * 4, cudaMemcpyDeviceToHost));
        }
    }
    if (!j.init_from.empty()) {  // warm start (weights only: Muon's momentum starts at zero, as upstream's .bin)
        std::vector<float> w0(t.params.total_elems);
        FILE* fw = fopen(j.init_from.c_str(), "rb");
        long fbytes = -1;
        if (fw && fseek(fw, 0, SEEK_END) == 0) {
            fbytes = ftell(fw);
            fseek(fw, 0, SEEK_SET);
        }
        if (!fw || fbytes != (long)w0.size() * 4 || fread(w0.data(), 4, w0.size(), fw) != w0.size()) {
            fprintf(stderr, "job %s: %s is not %ld fp32 params (%ld bytes)\n", j.id.c_str(), j.init_from.c_str(),
                (long)w0.size(), fbytes);
            exit(2);
        }
        fclose(fw);
        // (2026-10-13 review) on the slot's stream, after the initial weights' copy / weights_init: a legacy-
        // stream cudaMemcpy is not ordered with the non-blocking slot stream (the pageable source is staged
        // before the call returns)
        CK(cudaMemcpyAsync(t.params.mem, w0.data(), w0.size() * 4, cudaMemcpyHostToDevice, t.stream));
    }
    cudaStream_t st = t.stream;
    CK(cudaMemcpyAsync(t.muon.lr, &j.lr, sizeof(float), cudaMemcpyHostToDevice, st));
    CK(cudaMemsetAsync(t.muon.mb.data, 0, numel(t.muon.mb.shape) * sizeof(float), st));
    int maxe = j.max_epochs;
    CK(cudaMemcpyAsync(t.d_Z, t.hZ.data(), t.hZ.size() * 4, cudaMemcpyHostToDevice, st));
    CK(cudaMemcpyAsync(t.d_y, t.hY.data(), t.n_ctx * 4, cudaMemcpyHostToDevice, st));
    if (t.loaded_perm != t.hperm) {  // the same seed's permutation is already on the device: keep it
        CK(cudaMemcpyAsync(t.d_perm, t.hperm->data(), t.hperm->size() * 4, cudaMemcpyHostToDevice, st));
        t.loaded_perm = t.hperm;
    }
    CK(cudaMemcpyAsync(t.d_lr_sched, t.lr_sched.data(), maxe * 4, cudaMemcpyHostToDevice, st));
    CK(cudaMemcpyAsync(t.d_lp, &t.lp, sizeof(LossParams), cudaMemcpyHostToDevice, st));
    CK(cudaMemsetAsync(t.d_epoch, 0, sizeof(int), st));
    CK(cudaMemsetAsync(t.d_nf_total, 0, sizeof(int), st));
    CK(cudaMemsetAsync(t.d_best_epoch, 0, sizeof(int), st));
    CK(cudaMemcpyAsync(t.d_best, &g_inf, 4, cudaMemcpyHostToDevice, st));
    CK(cudaMemsetAsync(t.d_pred, 0, t.B_eval * sizeof(float), st));
    CK(cudaMemsetAsync(t.d_best_pred, 0, t.B_eval * sizeof(float), st));
}

static void trainer_capture(Trainer& t) {
    double tc = now_s();
    bind_cublas(t);
    CK(cudaStreamSynchronize(t.stream));
    CK(cudaStreamSynchronize(t.dws));
    cudaGraph_t g;
    CK(cudaStreamBeginCapture(t.stream, cudaStreamCaptureModeThreadLocal));
    enqueue_epoch(t);
    CK(cudaStreamEndCapture(t.stream, &g));
    CK(cudaGraphInstantiate(&t.graph, g, 0));
    CK(cudaGraphDestroy(g));
    t.t_capture = now_s() - tc;
}

// Build a trainer (the wave path): host part, allocation, load, graph.
static void trainer_setup(Trainer& t, int capture) {
    double t0 = now_s();
    trainer_host(t);
    trainer_alloc(t);
    trainer_load(t, false);
    CK(cudaDeviceSynchronize());
    t.t_setup = now_s() - t0;
    if (capture) {
        trainer_capture(t);
    }
}

// the pool's shape key: everything a captured epoch graph bakes in (pointers aside)
static std::string shape_key(const Trainer& t) {
    const Job& j = t.job;
    char b[1024];
    snprintf(b, sizeof(b), "%s|P%d|T%d|H%d|Ly%d|B%d|tune%d|N%d|r%d|S%d|ctx%d|ne%d|er%d|me%d|pat%d|mom%.9g|mgn%.9g",
        j.data.c_str(), t.P, t.T, j.H, j.layers, t.B, t.tune, t.N, t.row0, t.S, t.n_ctx, t.n_eval, t.eval_row0,
        j.max_epochs, j.patience, (double)j.momentum, (double)j.max_grad_norm);
    return std::string(b);
}

static void free_alloc(Allocator& a) {
    if (a.mem) {
        cudaFree(a.mem);
    }
    free(a.regs);
    a = Allocator{};
}

// a slot's shape buffers back to the pool, on its stream (no device-wide synchronisation)
static void trainer_free_shape(Trainer& t) {
    if (t.graph) {
        cudaGraphExecDestroy(t.graph);
        t.graph = NULL;
    }
    free_alloc_s(t.params, t.stream); free_alloc_s(t.grads, t.stream); free_alloc_s(t.acts, t.stream);
    free_alloc_s(t.ev_acts_alloc, t.stream); free_alloc_s(t.ev_grads, t.stream);
    void** ours[] = {(void**)&t.d_lp, (void**)&t.d_Z, (void**)&t.d_y, (void**)&t.d_ymb, (void**)&t.d_mask,
        (void**)&t.d_gl, (void**)&t.d_gv, (void**)&t.d_obs, (void**)&t.d_state, (void**)&t.d_eobs,
        (void**)&t.d_estate, (void**)&t.d_eymb, (void**)&t.d_emask, (void**)&t.d_perm, (void**)&t.d_epoch,
        (void**)&t.d_lr_sched, (void**)&t.d_step_out, (void**)&t.d_log_loss, (void**)&t.d_log_gn,
        (void**)&t.d_log_cap, (void**)&t.d_log_nf, (void**)&t.d_nf_total, (void**)&t.d_val_mse,
        (void**)&t.d_val_ql, (void**)&t.d_best, (void**)&t.d_best_epoch, (void**)&t.d_pred, (void**)&t.d_best_pred,
        (void**)&t.d_rows};
    for (void** pp : ours) {
        if (*pp) {
            CK(cudaFreeAsync(*pp, t.stream));
            *pp = NULL;
        }
    }
}

// everything of a slot (the end of a run)
static void trainer_free(Trainer& t) {
    trainer_free_shape(t);
    CK(cudaStreamSynchronize(t.stream));
    cublasDestroy(t.h); cublasDestroy(t.hdw);
    void* rest[] = {t.muon.lr, t.muon.grad_norm, t.muon.ns_norm, t.muon.norm_partials, t.ws, t.wsdw};
    for (void* p : rest) {
        if (p) {
            cudaFree(p);
        }
    }
    cudaStreamDestroy(t.dws); cudaStreamDestroy(t.stream);
    cudaEventDestroy(t.ev_ready); cudaEventDestroy(t.ev_done);
    t.stream = NULL;
}

// the pack path's free (its buffers come from alloc_create / dalloc)
static void pack_trainer_free(Trainer& t) {
    if (t.graph) {
        cudaGraphExecDestroy(t.graph);
        t.graph = NULL;
    }
    cublasDestroy(t.h); cublasDestroy(t.hdw);
    free_alloc(t.params); free_alloc(t.grads); free_alloc(t.acts);
    free_alloc(t.ev_acts_alloc); free_alloc(t.ev_grads);
    void* ours[] = {t.d_lp, t.d_Z, t.d_y, t.d_ymb, t.d_mask, t.d_gl, t.d_gv, t.d_obs, t.d_state, t.d_eobs, t.d_estate,
        t.d_eymb, t.d_emask, t.d_perm, t.d_epoch, t.d_lr_sched, t.d_step_out, t.d_log_loss, t.d_log_gn,
        t.d_log_cap, t.d_log_nf, t.d_nf_total, t.d_val_mse, t.d_val_ql, t.d_best, t.d_best_epoch, t.d_pred,
        t.d_best_pred, t.muon.lr, t.muon.grad_norm, t.muon.ns_norm, t.muon.norm_partials, t.ws, t.wsdw};
    for (void* p : ours) {
        if (p) {
            cudaFree(p);
        }
    }
    cudaStreamDestroy(t.dws); cudaStreamDestroy(t.stream);
    cudaEventDestroy(t.ev_ready); cudaEventDestroy(t.ev_done);
}

// ============================================================================ outputs
struct Out {
    FILE *jobs, *steps, *epochs, *preds;
};

static std::string g_out_dir;

static void write_trainer(Out& o, Trainer& t, double wave_wall, int wave_size, double wave_train) {
    const Job& j = t.job;
    if (j.save_params) {
        std::string pd = g_out_dir + "/params";
        mkdir(pd.c_str(), 0755);
        std::vector<float> w(t.params.total_elems);
        CK(cudaMemcpy(w.data(), t.params.mem, w.size() * 4, cudaMemcpyDeviceToHost));
        FILE* fw = fopen((pd + "/" + j.id + ".f32").c_str(), "wb");
        fwrite(w.data(), 4, w.size(), fw);
        fclose(fw);
        FILE* fk = fopen((pd + "/" + j.id + ".keep.i32").c_str(), "wb");  // the design columns fed (in order)
        fwrite(t.keep.data(), 4, t.keep.size(), fk);
        fclose(fk);
    }
    int E = t.epochs_run;
    std::vector<float> ll((size_t)E * t.S), gn((size_t)E * t.S), cap((size_t)E * t.S), nf((size_t)E * t.S);
    if (E > 0) {
        CK(cudaMemcpy(ll.data(), t.d_log_loss, ll.size() * 4, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(gn.data(), t.d_log_gn, gn.size() * 4, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(cap.data(), t.d_log_cap, cap.size() * 4, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(nf.data(), t.d_log_nf, nf.size() * 4, cudaMemcpyDeviceToHost));
    }
    for (int e = 0; e < E; e++) {
        for (int k = 0; k < t.S; k++) {
            size_t i = (size_t)e * t.S + k;
            fprintf(o.steps, "%s,%d,%d,%.9g,%.9g,%.0f,%.0f\n", j.id.c_str(), e + 1, k, ll[i], gn[i], cap[i], nf[i]);
        }
    }
    std::vector<float> vm(E), vq(E);
    if (t.tune && E > 0) {
        CK(cudaMemcpy(vm.data(), t.d_val_mse, E * 4, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(vq.data(), t.d_val_ql, E * 4, cudaMemcpyDeviceToHost));
    }
    for (int e = 0; e < E; e++) {
        fprintf(o.epochs, "%s,%d,%.9g,%.9g,%.9g\n", j.id.c_str(), e + 1, t.lr_sched[e],
            t.tune ? vm[e] : NAN, t.tune ? vq[e] : NAN);
    }
    std::vector<float> pr(t.n_eval);
    CK(cudaMemcpy(pr.data(), t.tune ? t.d_best_pred : t.d_pred, t.n_eval * 4, cudaMemcpyDeviceToHost));
    for (int b = 0; b < t.n_eval; b++) {
        fprintf(o.preds, "%s,%s,%d,%.9g\n", j.id.c_str(), t.tune ? "val" : "fc", j.w0 + t.eval_row0 + b, pr[b]);
    }
    fprintf(o.jobs, "%s,%s,%s,%s,%d,%d,%d,%d,%d,%d,%.6g,%d,%d,%d,%d,%d,%d,%.9g,%.9g,%.9g,%.9g,%.9g,%d,%s,"
        "%.4f,%.4f,%.4f,%.4f,%d,%ld,%.4f,%d,%ld,%s,%.9g,%s\n",
        j.id.c_str(), j.data.c_str(), j.mode.c_str(), j.loss.c_str(), j.w0, j.w1, j.L, j.H, j.layers, j.batch,
        j.lr, j.seed, t.N, t.S, E, t.best_epoch, t.n_eval, t.best_val, t.mu_y, t.sd_y, t.lp.huber_delta,
        t.lp.qlike_cap, t.nf_total, t.status.c_str(), t.t_setup, t.t_capture, t.t_end - t.t_start,
        wave_wall, wave_size, (long)t.params.total_elems, wave_train, t.P - 1, t.n_clipped, math_name(),
        (double)j.stop_loss, j.link.c_str());
}

// ============================================================================ run
static void run_wave(std::vector<Trainer>& ts, int chunk, Out& o) {
    double w0 = now_s();
    for (auto& t : ts) {
        trainer_setup(t, 1);
    }
    std::vector<char> active(ts.size(), 1);
    int* h_buf = NULL;
    CK(cudaMallocHost((void**)&h_buf, ts.size() * 3 * sizeof(int)));
    for (auto& t : ts) {
        t.t_start = now_s();
    }
    double tl = now_s();  // every model set up and captured: the wave's training starts here
    int n_active = (int)ts.size();
    while (n_active > 0) {
        for (size_t i = 0; i < ts.size(); i++) {
            if (!active[i]) {
                continue;
            }
            Trainer& t = ts[i];
            int todo = std::min(chunk, t.epochs_target - t.epochs_run);
            for (int c = 0; c < todo; c++) {
                CK(cudaGraphLaunch(t.graph, t.stream));
            }
            t.epochs_run += todo;
            CK(cudaMemcpyAsync(h_buf + 3 * i, t.d_epoch, 4, cudaMemcpyDeviceToHost, t.stream));
            CK(cudaMemcpyAsync(h_buf + 3 * i + 1, t.d_best_epoch, 4, cudaMemcpyDeviceToHost, t.stream));
            CK(cudaMemcpyAsync(h_buf + 3 * i + 2, t.d_nf_total, 4, cudaMemcpyDeviceToHost, t.stream));
        }
        for (size_t i = 0; i < ts.size(); i++) {
            if (!active[i]) {
                continue;
            }
            Trainer& t = ts[i];
            CK(cudaStreamSynchronize(t.stream));
            int ep = h_buf[3 * i], be = h_buf[3 * i + 1];
            t.nf_total = h_buf[3 * i + 2];
            bool stop = ep >= t.epochs_target;
            if (t.tune && ep - be >= t.job.patience) {
                stop = true;
            }
            // non-finite losses: abort when they exceed 1 % of the samples seen so far
            if ((double)t.nf_total > 0.01 * (double)ep * t.N) {
                t.status = "abort_nonfinite";
                stop = true;
            }
            if (stop) {
                active[i] = 0;
                n_active--;
                t.t_end = now_s();
                if (!t.tune) {
                    bind_cublas(t);
                    enqueue_eval(t, 0);
                    CK(cudaStreamSynchronize(t.stream));
                }
                if (t.tune) {
                    CK(cudaMemcpy(&t.best_epoch, t.d_best_epoch, 4, cudaMemcpyDeviceToHost));
                    CK(cudaMemcpy(&t.best_val, t.d_best, 4, cudaMemcpyDeviceToHost));
                }
            }
        }
    }
    CK(cudaDeviceSynchronize());
    double wall = now_s() - w0;
    double train_wall = now_s() - tl;  // steady-state training wall (setup / capture excluded)
    for (auto& t : ts) {
        write_trainer(o, t, wall, (int)ts.size(), train_wall);
        write_trainrows(o.preds, t);
        trainer_free(t);
    }
    fflush(o.jobs); fflush(o.steps); fflush(o.epochs); fflush(o.preds);
    CK(cudaFreeHost(h_buf));
    fprintf(stderr, "[%.1f] wave of %zu models: %.2f s (training %.3f s)\n", now_s(), ts.size(), wall, train_wall);
}

// The pool (--pool): K slots; a slot keeps its allocation and captured epoch graph while consecutive jobs
// share its shape, and takes the next job as soon as its model stops.  Jobs are taken in a stable order
// grouped by coarse shape (design, mode, L, H, ...), so a slot re-captures only when the shape changes
// (e.g. the window mask's column count).  Outputs are per job, so the order changes nothing else.
static const std::vector<float>& single_init(int D, int H, int layers, int T, int seed, cudaStream_t stream);

static void run_pool(std::vector<Job>& todo, int K, int chunk, Out& o, int phase) {
    double w0 = now_s();
    std::stable_sort(todo.begin(), todo.end(), [](const Job& a, const Job& b) {
        return std::make_tuple(a.data, a.mode, a.L, a.H, a.layers, a.batch, a.w1 - a.w0, a.max_epochs)
             < std::make_tuple(b.data, b.mode, b.L, b.H, b.layers, b.batch, b.w1 - b.w0, b.max_epochs);
    });
    K = std::max(1, std::min(K, (int)todo.size()));
    std::vector<Trainer> slots(K);
    std::vector<char> active(K, 0), allocated(K, 0);
    std::vector<std::string> keys(K);
    int* h_buf = NULL;
    CK(cudaMallocHost((void**)&h_buf, K * 3 * sizeof(int)));
    // each job's shape before its host work (the mask's column count from the cached window mask), so a
    // freed slot takes the next job of its own shape (no re-allocation: cudaFree stalls the whole device);
    // a slot whose shape has run out takes the shape with the most jobs left
    std::map<std::string, std::deque<size_t>> queues;
    std::vector<std::string> slot_pkey(K);
    std::vector<double> cost(todo.size());
    std::map<std::string, double> qcost;  // each shape queue's remaining work
    int smax = 1;
    bool any_stop = false;
    for (size_t q = 0; q < todo.size(); q++) {
        const Job& j = todo[q];
        Data* d = load_data(j.data);
        int P = (j.mask ? (int)keep_for(d, j.w0, j.w1).size() : d->p) + 1;
        // every (P, H, layers, seed)'s initial weights now, before any slot runs (2026-10-13 speed review, gate
        // 4): a cache miss mid-run (upstream weights_init: cudaMalloc, cuRAND, cudaFree per matrix) drains every
        // slot
        single_init(P, j.H, j.layers, j.L, j.seed, 0);
        // the job's work for largest-first: epochs x steps x (layers' fwd / bwd + Newton-Schulz, encoder)
        double S_ = (double)((j.w1 - j.w0 + j.batch - 1) / j.batch), E_ = j.mode == "tune" ? j.max_epochs : j.epochs;
        double B_ = j.batch, L_ = j.L, H_ = j.H;
        cost[q] = E_ * S_ * (j.layers * (18.0 * B_ * L_ * H_ * H_ + 70.0 * H_ * H_ * H_) + 4.0 * B_ * L_ * P * H_);
        smax = std::max(smax, (j.w1 - j.w0 + j.batch - 1) / j.batch + 1);
        any_stop = any_stop || j.stop_loss > 0.0f;
        char b[768];
        snprintf(b, sizeof(b), "%s|%s|L%d|H%d|Ly%d|B%d|W%d|me%d|pat%d|vt%d|em%d|fc%d|mom%.9g|mgn%.9g|P%d",
            j.data.c_str(), j.mode.c_str(), j.L, j.H, j.layers, j.batch, j.w1 - j.w0, j.max_epochs, j.patience,
            j.val_tail, j.embargo, j.fc_n, (double)j.momentum, (double)j.max_grad_norm, P);
        queues[b].push_back(q);
        qcost[b] += cost[q];
    }
    size_t remaining = todo.size();
    auto take = [&](int i) -> size_t {
        auto it = allocated[i] ? queues.find(slot_pkey[i]) : queues.end();
        if (it == queues.end() || it->second.empty()) {  // the queue with the most work left (largest first;
            it = queues.end();                            // equal shapes spread over the slots)
            for (auto jt = queues.begin(); jt != queues.end(); ++jt) {
                if (!jt->second.empty() && (it == queues.end() || qcost[jt->first] > qcost[it->first])) {
                    it = jt;
                }
            }
        }
        size_t q = it->second.front();
        it->second.pop_front();
        qcost[it->first] -= cost[q];
        slot_pkey[i] = it->first;
        remaining--;
        return q;
    };
    int n_active = 0;
    long n_capture = 0;
    double sum_setup = 0, sum_capture = 0, sum_train = 0;
    std::vector<char> pending(K, 0), finalizing(K, 0);
    float* h_loss = NULL;  // stop_loss: each slot's last-epoch minibatch losses
    if (any_stop) {
        CK(cudaMallocHost((void**)&h_loss, (size_t)K * smax * sizeof(float)));
    }
    auto launch = [&](int i) {  // the next chunk of epochs and the status copies, on the slot's stream
        Trainer& t = slots[i];
        int todo_e = std::min(chunk, t.epochs_target - t.epochs_run);
        if (t.tune) {  // never past the earliest possible stop, best_epoch + patience (2026-10-13 speed review)
            todo_e = std::max(1, std::min(todo_e, h_buf[3 * i + 1] + t.job.patience - t.epochs_run));
        }
        for (int c = 0; c < todo_e; c++) {
            CK(cudaGraphLaunch(t.graph, t.stream));
        }
        t.epochs_run += todo_e;
        if (t.job.stop_loss > 0.0f) {
            CK(cudaMemcpyAsync(h_loss + (size_t)i * smax, t.d_log_loss + (size_t)(t.epochs_run - 1) * t.S,
                t.S * sizeof(float), cudaMemcpyDeviceToHost, t.stream));
        }
        CK(cudaMemcpyAsync(h_buf + 3 * i, t.d_epoch, 4, cudaMemcpyDeviceToHost, t.stream));
        CK(cudaMemcpyAsync(h_buf + 3 * i + 1, t.d_best_epoch, 4, cudaMemcpyDeviceToHost, t.stream));
        CK(cudaMemcpyAsync(h_buf + 3 * i + 2, t.d_nf_total, 4, cudaMemcpyDeviceToHost, t.stream));
        pending[i] = 1;
    };
    while (remaining > 0 || n_active > 0) {
        bool progressed = false;
        for (int i = 0; i < K; i++) {
            Trainer& t = slots[i];
            if (active[i] || remaining == 0) {
                continue;
            }
            double t0 = now_s();
            t.job = todo[take(i)];
            trainer_host(t);
            std::string key = shape_key(t);
            bool cap = false;
            if (!allocated[i] || key != keys[i]) {
                if (allocated[i]) {
                    trainer_free_shape(t);
                }
                trainer_alloc(t);
                allocated[i] = 1;
                keys[i] = key;
                cap = true;
            }
            trainer_load(t, true);
            t.t_setup = now_s() - t0;
            t.t_capture = 0;
            if (cap) {
                trainer_capture(t);
                n_capture++;
            }
            sum_setup += t.t_setup;
            sum_capture += t.t_capture;
            t.t_start = now_s();
            active[i] = 1;
            n_active++;
            h_buf[3 * i] = h_buf[3 * i + 1] = h_buf[3 * i + 2] = 0;  // the previous job's status is stale
            launch(i);
            progressed = true;
        }
        for (int i = 0; i < K; i++) {
            if (!active[i] || !pending[i]) {
                continue;
            }
            Trainer& t = slots[i];
            cudaError_t q = cudaStreamQuery(t.stream);
            if (q == cudaErrorNotReady) {
                continue;
            }
            CK(q);
            pending[i] = 0;
            progressed = true;
            if (finalizing[i]) {  // a refit's forecast (enqueued at its stop) is done: write it out
                finalizing[i] = 0;
                sum_train += t.t_end - t.t_start;
                write_trainer(o, t, -1.0, K, -1.0);
                write_trainrows(o.preds, t);
                active[i] = 0;
                n_active--;
                continue;
            }
            int ep = h_buf[3 * i], be = h_buf[3 * i + 1];
            t.nf_total = h_buf[3 * i + 2];
            bool stop = ep >= t.epochs_target;
            if (t.tune && ep - be >= t.job.patience) {
                stop = true;
            }
            if ((double)t.nf_total > 0.01 * (double)ep * t.N) {
                t.status = "abort_nonfinite";
                stop = true;
            }
            if (!stop && t.job.stop_loss > 0.0f) {  // the last epoch's row-weighted training loss / sd_y^2
                const float* ll = h_loss + (size_t)i * smax;
                double sw = 0.0;
                for (int k = 0; k < t.S; k++) {
                    sw += (double)ll[k] * std::min(t.B, t.N - k * t.B);
                }
                stop = sw / t.N / ((double)t.sd_y * t.sd_y) <= (double)t.job.stop_loss;
            }
            if (!stop) {
                launch(i);
                continue;
            }
            t.t_end = now_s();
            if (!t.tune) {  // the forecast on the slot's stream; collected at a later poll (2026-10-13: a
                bind_cublas(t);  // cudaStreamSynchronize here held the host -- and every other slot's refill --
                enqueue_eval(t, 0);  // behind the busy GPU, 2.9 s of a 10.8 s pool window in the gate-4 profile)
                finalizing[i] = 1;
                pending[i] = 1;
                continue;
            } else {
                CK(cudaMemcpy(&t.best_epoch, t.d_best_epoch, 4, cudaMemcpyDeviceToHost));
                CK(cudaMemcpy(&t.best_val, t.d_best, 4, cudaMemcpyDeviceToHost));
            }
            sum_train += t.t_end - t.t_start;
            write_trainer(o, t, -1.0, K, -1.0);  // pool: per-run walls in pool_summary.csv
            write_trainrows(o.preds, t);
            active[i] = 0;
            n_active--;
        }
        if (!progressed) {
            usleep(20);
        }
    }
    CK(cudaDeviceSynchronize());
    double wall = now_s() - w0;
    for (int i = 0; i < K; i++) {
        if (allocated[i]) {
            trainer_free(slots[i]);
        }
    }
    CK(cudaFreeHost(h_buf));
    if (h_loss) {
        CK(cudaFreeHost(h_loss));
    }
    fflush(o.jobs); fflush(o.steps); fflush(o.epochs); fflush(o.preds);
    std::string sp = g_out_dir + "/pool_summary.csv";
    bool fresh = access(sp.c_str(), F_OK) != 0;
    FILE* f = fopen(sp.c_str(), "a");
    if (fresh) {
        fprintf(f, "phase,jobs,slots,wall,sum_setup,sum_capture,n_capture,sum_train_overlapped\n");
    }
    fprintf(f, "%d,%zu,%d,%.4f,%.4f,%.4f,%ld,%.4f\n", phase, todo.size(), K, wall, sum_setup, sum_capture, n_capture,
        sum_train);
    fclose(f);
    fprintf(stderr, "[%.1f] pool phase %d: %zu jobs on %d slots in %.2f s (setup %.2f s, %ld captures %.2f s)\n",
        now_s(), phase, todo.size(), K, wall, sum_setup, n_capture, sum_capture);
}

// ============================================================================ packing (--pack K, 2026-10-13)
// User (2026-10-13), a measured experiment in our code only: K refit jobs with the same configuration and
// epoch count trained as ONE upstream model with K*H hidden units and block-diagonal weights -- block k =
// model k (its own window, its own minibatch order, its own seed's initial weights), off-diagonal blocks
// zero, their gradients zeroed after every backward; each model's gradient clipped to max_grad_norm by our
// kernel (Muon's global clip disabled with an infinite max norm); loss, logs and forecasts per model.  The
// MinGRU recurrence and the highway gate are elementwise and the upstream network has no normalisation,
// so forward and backward are per model (up to GEMM summation order); Newton-Schulz keeps block-diagonal
// matrices block-diagonal and the shape scale sqrt(max(1, R/C)) is unchanged, but Muon's Frobenius
// normalisation before Newton-Schulz is over the packed matrix: close to, not identical to, separate training.
// Refit jobs only (fixed epoch counts); tune jobs of a --pack run go through the pool.

__global__ void fcp_gather(const float* __restrict__ Z, int D, int n_ctx, const float* __restrict__ yrow,
        const int* __restrict__ perm, long permsize, const int* __restrict__ epoch_ptr, int perm_stride,
        int step_off, int B, int T, int K, int row0, int n, precision_t* __restrict__ obs,
        float* __restrict__ ymb, float* __restrict__ mask) {
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long KD = (long)K * D;
    long tot = (long)B * T * KD;
    if (idx >= tot) {
        return;
    }
    int col = (int)(idx % KD);
    long bt = idx / KD;
    int t = (int)(bt % T);
    int b = (int)(bt / T);
    int k = col / D, c = col % D;
    int s = (perm != NULL) ? perm[(long)k * permsize + (long)(*epoch_ptr) * perm_stride + step_off + b]
                           : step_off + b;
    bool real = (s >= 0 && s < n);
    int r = row0 + (real ? s : 0);
    obs[idx] = from_float(Z[((long)k * n_ctx + (r - T + 1 + t)) * D + c]);
    if (t == 0 && c == 0) {
        ymb[k * B + b] = (yrow != NULL) ? yrow[(long)k * n_ctx + r] : 0.0f;
        mask[k * B + b] = real ? 1.0f : 0.0f;
    }
}

// one block per model: the model's last-step loss / gradient (fc_loss), its grad_logits column, its step sums
__global__ void fcp_loss_train(const precision_t* __restrict__ dec, int T, int K, const float* __restrict__ y,
        const float* __restrict__ mask, int B, const LossParams* __restrict__ lpd, float inv_n,
        float* __restrict__ grad_logits, float* __restrict__ step_out) {
    int k = blockIdx.x;
    const LossParams p = lpd[k];
    int od1 = K + 1;
    __shared__ float sm[3 * 256];
    int tid = threadIdx.x;
    float sl = 0.0f, sc = 0.0f, sn = 0.0f;
    for (int b = tid; b < B; b += blockDim.x) {
        for (int t = 0; t < T - 1; t++) {
            grad_logits[((long)b * T + t) * K + k] = 0.0f;
        }
        float o = to_float(dec[((long)b * T + T - 1) * od1 + k]);
        float f = p.mu_y + p.sd_y * o;
        float l, g;
        int cap;
        fc_loss(p, f, y[k * B + b], &l, &g, &cap);
        float m = mask[k * B + b];
        if (!isfinite(l) || !isfinite(g) || !isfinite(o)) {
            l = 0.0f;
            g = 0.0f;
            sn += m;
            cap = 0;
        }
        grad_logits[((long)b * T + T - 1) * K + k] = m * g * p.sd_y * inv_n;
        sl += m * l;
        sc += m * (float)cap;
    }
    sm[tid] = sl;
    sm[256 + tid] = sc;
    sm[512 + tid] = sn;
    block_reduce_sum(sm, step_out + 3 * k, tid, blockDim.x, 3);
}

// packed parameter layout (weights_create order): encoder [K H, K D], decoder [K + 1, K H] (K logits + the
// unused value row), MinGRU layers [3 K H, K H] each ([hidden | gate | proj] chunks of K H rows)
struct PackGeom {
    int K, H, D, layers;
    long enc, dec, lay;  // element offsets of the decoder and the first layer; per-layer size
};

// owner model of a packed element, or -1 if it is off the block diagonal (or in the decoder's value row)
__device__ __forceinline__ int fcp_owner(const PackGeom g, long i) {
    long KH = (long)g.K * g.H, KD = (long)g.K * g.D;
    if (i < g.dec) {
        long r = i / KD, c = i % KD;
        int kr = (int)(r / g.H), kc = (int)(c / g.D);
        return kr == kc ? kr : -1;
    }
    if (i < g.lay) {
        long j = i - g.dec;
        long r = j / KH, c = j % KH;
        if (r >= g.K) {
            return -1;
        }
        return (int)(c / g.H) == (int)r ? (int)r : -1;
    }
    long j = (i - g.lay) % (3 * KH * KH);
    long r = (j / KH) % KH, c = j % KH;
    int kr = (int)(r / g.H), kc = (int)(c / g.H);
    return kr == kc ? kr : -1;
}

__global__ void fcp_offdiag_zero(precision_t* __restrict__ grad, PackGeom g, long n) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && fcp_owner(g, i) < 0) {
        grad[i] = from_float(0.0f);
    }
}

// the i-th element (0 .. per-model count) of model k's block, as a packed flat index
__device__ __forceinline__ long fcp_elem(const PackGeom g, int k, long i) {
    long KH = (long)g.K * g.H, KD = (long)g.K * g.D;
    long ne = (long)g.H * g.D;
    if (i < ne) {
        return ((long)k * g.H + i / g.D) * KD + (long)k * g.D + i % g.D;
    }
    i -= ne;
    if (i < g.H) {
        return g.dec + (long)k * KH + (long)k * g.H + i;
    }
    i -= g.H;
    long per = 3L * g.H * g.H;
    long l = i / per, j = i % per;
    long ch = j / ((long)g.H * g.H), rem = j % ((long)g.H * g.H);
    long row = ch * KH + (long)k * g.H + rem / g.H, col = (long)k * g.H + rem % g.H;
    return g.lay + l * 3 * KH * KH + row * KH + col;
}

// one block per model: the model's gradient sum of squares (deterministic tree reduce)
__global__ void fcp_block_sumsq(const precision_t* __restrict__ grad, PackGeom g, long per_model,
        float* __restrict__ sumsq) {
    int k = blockIdx.x;
    __shared__ float sm[256];
    float s = 0.0f;
    for (long i = threadIdx.x; i < per_model; i += blockDim.x) {
        float v = to_float(grad[fcp_elem(g, k, i)]);
        s += v * v;
    }
    sm[threadIdx.x] = s;
    block_reduce_sum(sm, sumsq + k, threadIdx.x, blockDim.x, 1);
}

// per-model clip (Muon's own formula, applied per block): g *= min(max_norm / (|g| + 1e-6), 1)
__global__ void fcp_block_scale(precision_t* __restrict__ grad, PackGeom g, long per_model,
        const float* __restrict__ sumsq, float max_norm) {
    int k = blockIdx.y;
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= per_model) {
        return;
    }
    float clip = fminf(max_norm / (sqrtf(sumsq[k]) + 1e-6f), 1.0f);
    long e = fcp_elem(g, k, i);
    grad[e] = from_float(to_float(grad[e]) * clip);
}

__global__ void fcp_step_log(const float* __restrict__ step_out, const float* __restrict__ sumsq,
        const int* __restrict__ epoch_ptr, int K, int S, long log_stride, int kstep, float n_real,
        float* __restrict__ log_loss, float* __restrict__ log_gn, float* __restrict__ log_cap,
        float* __restrict__ log_nf, int* __restrict__ nf_total) {
    int k = threadIdx.x;
    if (k >= K) {
        return;
    }
    long i = (long)k * log_stride + (long)(*epoch_ptr) * S + kstep;
    log_loss[i] = step_out[3 * k] / n_real;
    log_gn[i] = sqrtf(sumsq[k]);
    log_cap[i] = step_out[3 * k + 1];
    log_nf[i] = step_out[3 * k + 2];
    nf_total[k] += (int)step_out[3 * k + 2];
}

// one block per model: forecasts of the model's eval rows (refit: its forecast rows)
__global__ void fcp_eval(const precision_t* __restrict__ dec, int T, int K, int n,
        const LossParams* __restrict__ lpd, float* __restrict__ pred) {
    int k = blockIdx.x;
    const LossParams p = lpd[k];
    for (int b = threadIdx.x; b < n; b += blockDim.x) {
        float o = to_float(dec[((long)b * T + T - 1) * (K + 1) + k]);
        pred[k * n + b] = p.mu_y + p.sd_y * o;
    }
}

struct PackSlot {
    Trainer pk;                // the packed upstream model (stream, cuBLAS, arch, allocators, Muon)
    std::vector<Trainer> m;    // the K models' host state (trainer_host)
    int K = 0, D = 0, H = 0, layers = 0;
    PackGeom geom;
    long per_model = 0, permsize = 0;
    float *d_Z = NULL, *d_y = NULL, *d_ymb = NULL, *d_mask = NULL, *d_gl = NULL, *d_gv = NULL, *d_sumsq = NULL;
    float *d_eymb = NULL, *d_emask = NULL, *d_step_out = NULL, *d_log_loss = NULL, *d_log_gn = NULL;
    float *d_log_cap = NULL, *d_log_nf = NULL, *d_pred = NULL, *d_lr_sched = NULL;
    precision_t *d_obs = NULL, *d_state = NULL, *d_eobs = NULL, *d_estate = NULL;
    int *d_perm = NULL, *d_epoch = NULL, *d_nf_total = NULL;
    LossParams* d_lp = NULL;
    cudaGraphExec_t graph = NULL;
    std::string key;
    int epochs_run = 0, epochs_target = 0;
    double t_setup = 0, t_capture = 0, t_start = 0, t_end = 0;
};

static void pack_free(PackSlot& ps) {
    if (ps.graph) {
        cudaGraphExecDestroy(ps.graph);
        ps.graph = NULL;
    }
    void* ours[] = {ps.d_Z, ps.d_y, ps.d_ymb, ps.d_mask, ps.d_gl, ps.d_gv, ps.d_sumsq, ps.d_eymb, ps.d_emask,
        ps.d_step_out, ps.d_log_loss, ps.d_log_gn, ps.d_log_cap, ps.d_log_nf, ps.d_pred, ps.d_lr_sched, ps.d_obs,
        ps.d_state, ps.d_eobs, ps.d_estate, ps.d_perm, ps.d_epoch, ps.d_nf_total, ps.d_lp};
    for (void* p : ours) {
        if (p) {
            cudaFree(p);
        }
    }
    pack_trainer_free(ps.pk);
}

static std::string pack_key(const PackSlot& ps) {
    return shape_key(ps.m[0]) + "|K" + std::to_string(ps.K);
}

// the packed upstream model and our buffers for K models of model 0's shape
static void pack_alloc(PackSlot& ps) {
    Trainer& t0 = ps.m[0];
    const Job& j = t0.job;
    Trainer& pk = ps.pk;
    pk = Trainer();
    pk.job = j;
    int K = ps.K;
    ps.D = t0.P;
    ps.H = j.H;
    ps.layers = j.layers;
    CK(cudaStreamCreateWithFlags(&pk.stream, cudaStreamNonBlocking));
    cublas_init_handle();
    pk.h = g_cublas_handle; pk.ws = g_cublas_workspace; pk.hdw = g_cublas_dw_handle;
    pk.wsdw = g_cublas_dw_workspace; pk.dws = g_dw_stream; pk.ev_ready = g_main_ready; pk.ev_done = g_dw_done;
    bind_cublas(pk);
    pk.arch = build_arch(K * ps.D, K * ps.H, ps.layers, K, false, t0.T);
    pk.w = weights_create(&pk.arch, &pk.params);
    pk.tr_acts = arch_reg_train(&pk.arch, pk.w, &pk.acts, &pk.grads, t0.B * t0.T);
    pk.ev_acts = arch_reg_train(&pk.arch, pk.w, &pk.ev_acts_alloc, &pk.ev_grads, t0.B_eval * t0.T);
    muon_init(&pk.muon, &pk.params, j.momentum, &pk.acts);
    alloc_create(&pk.params);
    alloc_create(&pk.grads);
    alloc_create(&pk.acts);
    alloc_create(&pk.ev_acts_alloc);
    alloc_create(&pk.ev_grads);
    if (pk.params.total_bytes != pk.params.total_elems * (long)sizeof(precision_t)
            || pk.grads.total_bytes != pk.grads.total_elems * (long)sizeof(precision_t)) {
        fprintf(stderr, "pack: parameter layout has alignment padding\n");
        exit(2);
    }
    pk.grad = {.data = (precision_t*)pk.grads.mem, .shape = {pk.grads.total_elems}};
    pk.master = {.data = (float*)pk.params.mem, .shape = {pk.params.total_elems}};
    long KH = (long)K * ps.H, KD = (long)K * ps.D;
    ps.geom.K = K;
    ps.geom.H = ps.H;
    ps.geom.D = ps.D;
    ps.geom.layers = ps.layers;
    ps.geom.enc = 0;
    ps.geom.dec = KH * KD;                       // decoder offset
    ps.geom.lay = KH * KD + (long)(K + 1) * KH;  // first MinGRU layer offset
    if (pk.params.total_elems != ps.geom.lay + (long)ps.layers * 3 * KH * KH) {
        fprintf(stderr, "pack: unexpected parameter layout\n");
        exit(2);
    }
    ps.per_model = (long)ps.H * ps.D + ps.H + (long)ps.layers * 3 * ps.H * ps.H;
    int maxe = j.max_epochs;
    ps.permsize = (long)maxe * t0.S * t0.B;
    ps.d_Z = dalloc<float>((size_t)K * t0.n_ctx * ps.D);
    ps.d_y = dalloc<float>((size_t)K * t0.n_ctx);
    ps.d_obs = dalloc<precision_t>((size_t)t0.B * t0.T * KD);
    ps.d_ymb = dalloc<float>((size_t)K * t0.B);
    ps.d_mask = dalloc<float>((size_t)K * t0.B);
    ps.d_gl = dalloc<float>((size_t)t0.B * t0.T * K);
    ps.d_gv = dalloc<float>((size_t)t0.B * t0.T);
    ps.d_sumsq = dalloc<float>(K);
    ps.d_state = dalloc<precision_t>((size_t)ps.layers * t0.B * KH);
    ps.d_eobs = dalloc<precision_t>((size_t)t0.B_eval * t0.T * KD);
    ps.d_eymb = dalloc<float>((size_t)K * t0.B_eval);
    ps.d_emask = dalloc<float>((size_t)K * t0.B_eval);
    ps.d_estate = dalloc<precision_t>((size_t)ps.layers * t0.B_eval * KH);
    ps.d_perm = dalloc<int>((size_t)K * ps.permsize);
    ps.d_epoch = dalloc<int>(1);
    ps.d_lr_sched = dalloc<float>(maxe);
    ps.d_step_out = dalloc<float>((size_t)3 * K);
    ps.d_log_loss = dalloc<float>((size_t)K * maxe * t0.S);
    ps.d_log_gn = dalloc<float>((size_t)K * maxe * t0.S);
    ps.d_log_cap = dalloc<float>((size_t)K * maxe * t0.S);
    ps.d_log_nf = dalloc<float>((size_t)K * maxe * t0.S);
    ps.d_nf_total = dalloc<int>(K);
    ps.d_pred = dalloc<float>((size_t)K * t0.B_eval);
    ps.d_lp = dalloc<LossParams>(K);
    CK(cudaStreamSynchronize(0));
}

// single-model initial weights (upstream weights_init for the model's shape and seed; cached)
static const std::vector<float>& single_init(int D, int H, int layers, int T, int seed, cudaStream_t stream) {
    auto key = std::make_tuple(D, H, layers, seed);
    auto it = g_w_cache.find(key);
    if (it != g_w_cache.end()) {
        return it->second;
    }
    Arch a1 = build_arch(D, H, layers, 1, false, T);
    Allocator pa = {};
    Weights w1 = weights_create(&a1, &pa);
    alloc_create(&pa);
    ulong sd = (ulong)seed;
    CK(cudaStreamSynchronize(0));
    weights_init(&a1, w1, &sd, 0);  // the legacy stream, with cuRAND's draws (see trainer_load)
    CK(cudaStreamSynchronize(0));
    std::vector<float>& c = g_w_cache[key];
    c.resize(pa.total_elems);
    CK(cudaMemcpy(c.data(), pa.mem, c.size() * 4, cudaMemcpyDeviceToHost));
    free_alloc(pa);
    return c;
}

static void pack_load(PackSlot& ps) {
    Trainer& t0 = ps.m[0];
    const Job& j = t0.job;
    int K = ps.K, H = ps.H, D = ps.D;
    long KH = (long)K * H, KD = (long)K * D;
    // block-diagonal initial weights: block k = model k's own initial weights; the value row and every
    // off-diagonal block zero
    std::vector<float> w(ps.pk.params.total_elems, 0.0f);
    for (int k = 0; k < K; k++) {
        const std::vector<float>& s1 = single_init(D, H, ps.layers, t0.T, ps.m[k].job.seed, ps.pk.stream);
        long o = 0;
        for (int i = 0; i < H; i++) {
            for (int c = 0; c < D; c++) {
                w[((long)k * H + i) * KD + (long)k * D + c] = s1[o + (long)i * D + c];
            }
        }
        o += (long)H * D;
        for (int i = 0; i < H; i++) {  // single decoder row 0 (logit); its value row is not used
            w[ps.geom.dec + (long)k * KH + (long)k * H + i] = s1[o + i];
        }
        o += 2L * H;
        for (int l = 0; l < ps.layers; l++) {
            for (int ch = 0; ch < 3; ch++) {
                for (int i = 0; i < H; i++) {
                    for (int jj = 0; jj < H; jj++) {
                        w[ps.geom.lay + (long)l * 3 * KH * KH + (ch * KH + (long)k * H + i) * KH + (long)k * H + jj] =
                            s1[o + ((long)ch * H + i) * H + jj];
                    }
                }
            }
            o += 3L * H * H;
        }
    }
    cudaStream_t st = ps.pk.stream;
    CK(cudaMemcpyAsync(ps.pk.params.mem, w.data(), w.size() * 4, cudaMemcpyHostToDevice, st));
    CK(cudaMemcpyAsync(ps.pk.muon.lr, &j.lr, sizeof(float), cudaMemcpyHostToDevice, st));
    CK(cudaMemsetAsync(ps.pk.muon.mb.data, 0, numel(ps.pk.muon.mb.shape) * sizeof(float), st));
    std::vector<LossParams> lps(K);
    for (int k = 0; k < K; k++) {
        Trainer& t = ps.m[k];
        CK(cudaMemcpyAsync(ps.d_Z + (size_t)k * t0.n_ctx * D, t.hZ.data(), t.hZ.size() * 4, cudaMemcpyHostToDevice,
            st));
        CK(cudaMemcpyAsync(ps.d_y + (size_t)k * t0.n_ctx, t.hY.data(), t0.n_ctx * 4, cudaMemcpyHostToDevice, st));
        CK(cudaMemcpyAsync(ps.d_perm + (size_t)k * ps.permsize, t.hperm->data(), ps.permsize * 4,
            cudaMemcpyHostToDevice, st));
        lps[k] = t.lp;
        if (t.lr_sched != t0.lr_sched) {
            fprintf(stderr, "pack: models with different lr schedules\n");
            exit(2);
        }
    }
    CK(cudaMemcpyAsync(ps.d_lp, lps.data(), K * sizeof(LossParams), cudaMemcpyHostToDevice, st));
    CK(cudaMemcpyAsync(ps.d_lr_sched, t0.lr_sched.data(), j.max_epochs * 4, cudaMemcpyHostToDevice, st));
    CK(cudaMemsetAsync(ps.d_epoch, 0, sizeof(int), st));
    CK(cudaMemsetAsync(ps.d_nf_total, 0, K * sizeof(int), st));
    CK(cudaStreamSynchronize(st));  // the host vectors w / lps are locals
    ps.epochs_target = t0.epochs_target;
    ps.epochs_run = 0;
}

static void pack_enqueue_eval(PackSlot& ps) {
    Trainer& t0 = ps.m[0];
    Trainer& pk = ps.pk;
    long tot = (long)t0.B_eval * t0.T * ps.K * ps.D;
    fcp_gather<<<(int)((tot + 255) / 256), 256, 0, pk.stream>>>(ps.d_Z, ps.D, t0.n_ctx, NULL, NULL, 0, NULL, 0, 0,
        t0.B_eval, t0.T, ps.K, t0.eval_row0, t0.n_eval, ps.d_eobs, ps.d_eymb, ps.d_emask);
    Prec dec = net_forward(&pk.arch, pk.w, pk.ev_acts, ps.d_eobs, t0.B_eval, t0.T, ps.K * ps.D, ps.d_estate,
        ps.layers, ps.K * ps.H, pk.stream);
    fcp_eval<<<ps.K, 256, 0, pk.stream>>>(dec.data, t0.T, ps.K, t0.n_eval, ps.d_lp, ps.d_pred);
}

static void pack_enqueue_epoch(PackSlot& ps) {
    Trainer& t0 = ps.m[0];
    Trainer& pk = ps.pk;
    const Job& j = t0.job;
    int K = ps.K;
    fc_epoch_begin<<<1, 1, 0, pk.stream>>>(ps.d_lr_sched, ps.d_epoch, pk.muon.lr);
    long ntot = numel(pk.grad.shape);
    for (int k = 0; k < t0.S; k++) {
        int n_real = std::min(t0.B, t0.N - k * t0.B);
        long tot = (long)t0.B * t0.T * K * ps.D;
        fcp_gather<<<(int)((tot + 255) / 256), 256, 0, pk.stream>>>(ps.d_Z, ps.D, t0.n_ctx, ps.d_y, ps.d_perm,
            ps.permsize, ps.d_epoch, t0.S * t0.B, k * t0.B, t0.B, t0.T, K, t0.row0, t0.N, ps.d_obs, ps.d_ymb,
            ps.d_mask);
        Prec dec = net_forward(&pk.arch, pk.w, pk.tr_acts, ps.d_obs, t0.B, t0.T, K * ps.D, ps.d_state, ps.layers,
            K * ps.H, pk.stream);
        fcp_loss_train<<<K, 256, 0, pk.stream>>>(dec.data, t0.T, K, ps.d_ymb, ps.d_mask, t0.B, ps.d_lp,
            1.0f / (float)n_real, ps.d_gl, ps.d_step_out);
        Float gl = {.data = ps.d_gl, .shape = {t0.B, t0.T, K}};
        Float gv = {.data = ps.d_gv, .shape = {t0.B, t0.T}};
        arch_backward(&pk.arch, pk.w, pk.tr_acts, gl, Float(), gv, pk.stream);
        fcp_offdiag_zero<<<(int)((ntot + 255) / 256), 256, 0, pk.stream>>>(pk.grad.data, ps.geom, ntot);
        fcp_block_sumsq<<<K, 256, 0, pk.stream>>>(pk.grad.data, ps.geom, ps.per_model, ps.d_sumsq);
        dim3 gsc((unsigned)((ps.per_model + 255) / 256), (unsigned)K);
        fcp_block_scale<<<gsc, 256, 0, pk.stream>>>(pk.grad.data, ps.geom, ps.per_model, ps.d_sumsq,
            j.max_grad_norm);
        muon_step(&pk.muon, pk.master, pk.grad, 3.0e38f, pk.stream);  // per-model clip done: Muon's never binds
        fcp_step_log<<<1, 64, 0, pk.stream>>>(ps.d_step_out, ps.d_sumsq, ps.d_epoch, K, t0.S,
            (long)j.max_epochs * t0.S, k, (float)n_real, ps.d_log_loss, ps.d_log_gn, ps.d_log_cap, ps.d_log_nf,
            ps.d_nf_total);
    }
    fc_epoch_end<<<1, 1, 0, pk.stream>>>(ps.d_epoch);
}

static void pack_capture(PackSlot& ps) {
    double tc = now_s();
    bind_cublas(ps.pk);
    CK(cudaStreamSynchronize(ps.pk.stream));
    CK(cudaStreamSynchronize(ps.pk.dws));
    cudaGraph_t g;
    CK(cudaStreamBeginCapture(ps.pk.stream, cudaStreamCaptureModeThreadLocal));
    pack_enqueue_epoch(ps);
    CK(cudaStreamEndCapture(ps.pk.stream, &g));
    CK(cudaGraphInstantiate(&ps.graph, g, 0));
    CK(cudaGraphDestroy(g));
    ps.t_capture = now_s() - tc;
}

// per-model outputs of a finished pack (the single-model formats; params in the single-model layout)
static void pack_write(Out& o, PackSlot& ps) {
    Trainer& t0 = ps.m[0];
    int K = ps.K, H = ps.H, D = ps.D, S = t0.S, E = ps.epochs_run;
    long KH = (long)K * H, KD = (long)K * D;
    long stride = (long)t0.job.max_epochs * S;
    std::vector<float> ll((size_t)K * stride), gn((size_t)K * stride), cp((size_t)K * stride), nf((size_t)K * stride);
    CK(cudaMemcpy(ll.data(), ps.d_log_loss, ll.size() * 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(gn.data(), ps.d_log_gn, gn.size() * 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(cp.data(), ps.d_log_cap, cp.size() * 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(nf.data(), ps.d_log_nf, nf.size() * 4, cudaMemcpyDeviceToHost));
    std::vector<float> pr((size_t)K * t0.n_eval);
    CK(cudaMemcpy(pr.data(), ps.d_pred, pr.size() * 4, cudaMemcpyDeviceToHost));
    std::vector<int> nft(K);
    CK(cudaMemcpy(nft.data(), ps.d_nf_total, K * 4, cudaMemcpyDeviceToHost));
    std::vector<float> w;
    for (int k = 0; k < K; k++) {
        Trainer& t = ps.m[k];
        const Job& j = t.job;
        if (j.save_params) {
            if (w.empty()) {
                w.resize(ps.pk.params.total_elems);
                CK(cudaMemcpy(w.data(), ps.pk.params.mem, w.size() * 4, cudaMemcpyDeviceToHost));
            }
            std::vector<float> s1;  // model k's block in the single-model layout (its value row: zeros)
            for (int i = 0; i < H; i++) {
                for (int c = 0; c < D; c++) {
                    s1.push_back(w[((long)k * H + i) * KD + (long)k * D + c]);
                }
            }
            for (int i = 0; i < H; i++) {
                s1.push_back(w[ps.geom.dec + (long)k * KH + (long)k * H + i]);
            }
            for (int i = 0; i < H; i++) {
                s1.push_back(0.0f);
            }
            for (int l = 0; l < ps.layers; l++) {
                for (int ch = 0; ch < 3; ch++) {
                    for (int i = 0; i < H; i++) {
                        for (int jj = 0; jj < H; jj++) {
                            s1.push_back(w[ps.geom.lay + (long)l * 3 * KH * KH + (ch * KH + (long)k * H + i) * KH
                                + (long)k * H + jj]);
                        }
                    }
                }
            }
            std::string pd = g_out_dir + "/params";
            mkdir(pd.c_str(), 0755);
            FILE* fw = fopen((pd + "/" + j.id + ".f32").c_str(), "wb");
            fwrite(s1.data(), 4, s1.size(), fw);
            fclose(fw);
        }
        for (int e = 0; e < E; e++) {
            for (int s = 0; s < S; s++) {
                size_t i = (size_t)k * stride + (size_t)e * S + s;
                fprintf(o.steps, "%s,%d,%d,%.9g,%.9g,%.0f,%.0f\n", j.id.c_str(), e + 1, s, ll[i], gn[i], cp[i], nf[i]);
            }
            fprintf(o.epochs, "%s,%d,%.9g,%.9g,%.9g\n", j.id.c_str(), e + 1, t.lr_sched[e], NAN, NAN);
        }
        for (int b = 0; b < t0.n_eval; b++) {
            fprintf(o.preds, "%s,fc,%d,%.9g\n", j.id.c_str(), j.w0 + t0.eval_row0 + b, pr[(size_t)k * t0.n_eval + b]);
        }
        long n1 = ps.per_model + H;  // single-model parameter count (with its value row)
        fprintf(o.jobs, "%s,%s,%s,%s,%d,%d,%d,%d,%d,%d,%.6g,%d,%d,%d,%d,%d,%d,%.9g,%.9g,%.9g,%.9g,%.9g,%d,%s,"
            "%.4f,%.4f,%.4f,%.4f,%d,%ld,%.4f,%d,%ld,%s,%.9g,%s\n",
            j.id.c_str(), j.data.c_str(), j.mode.c_str(), j.loss.c_str(), j.w0, j.w1, j.L, j.H, j.layers, j.batch,
            j.lr, j.seed, t.N, t.S, E, 0, t.n_eval, 0.0f, t.mu_y, t.sd_y, t.lp.huber_delta, t.lp.qlike_cap, nft[k],
            nft[k] > 0.01 * E * t.N ? "abort_nonfinite" : "ok", ps.t_setup / K, ps.t_capture / K,
            ps.t_end - ps.t_start, -1.0, K, n1, -1.0, t.P - 1, t.n_clipped, math_name(), (double)j.stop_loss,
            j.link.c_str());
    }
}

// --pack: refit jobs grouped by everything a pack must share (configuration, epoch count, window length),
// in their order (origins ascending), cut into packs of up to Kp models with the same input width; packs
// run on `slots` concurrent slots (allocation + capture reused while the pack shape repeats).
static void run_packs(std::vector<Job>& todo, int Kp, int slots, int chunk, Out& o, int phase) {
    double w0 = now_s();
    std::stable_sort(todo.begin(), todo.end(), [](const Job& a, const Job& b) {
        return std::make_tuple(a.data, a.L, a.H, a.layers, a.batch, a.lr, a.epochs, a.max_epochs, a.w1 - a.w0, a.loss)
             < std::make_tuple(b.data, b.L, b.H, b.layers, b.batch, b.lr, b.epochs, b.max_epochs, b.w1 - b.w0, b.loss);
    });
    // pack generator: per group key, buckets by input width; a full bucket (or a group end) emits a pack
    size_t next = 0;
    std::map<int, std::vector<Trainer>> buckets;
    std::string cur_group;
    std::vector<std::vector<Trainer>> ready;
    auto group_of = [](const Job& j) {
        char b[512];
        snprintf(b, sizeof(b), "%s|%d|%d|%d|%d|%.9g|%d|%d|%d|%s", j.data.c_str(), j.L, j.H, j.layers, j.batch,
            (double)j.lr, j.epochs, j.max_epochs, j.w1 - j.w0, j.loss.c_str());
        return std::string(b);
    };
    const size_t look = 2 * (size_t)std::max(1, slots);  // packs prepared ahead, so a freed slot can match
    auto produce = [&]() -> bool {  // fill `ready` up to `look` packs; false when nothing is left
        while (ready.size() < look) {
            if (next >= todo.size()) {
                for (auto& kv : buckets) {
                    if (!kv.second.empty()) {
                        ready.push_back(std::move(kv.second));
                    }
                }
                buckets.clear();
                break;
            }
            const Job& j = todo[next];
            std::string g = group_of(j);
            if (g != cur_group) {
                for (auto& kv : buckets) {
                    if (!kv.second.empty()) {
                        ready.push_back(std::move(kv.second));
                    }
                }
                buckets.clear();
                cur_group = g;
            }
            Trainer t;
            t.job = j;
            trainer_host(t);
            next++;
            std::vector<Trainer>& bk = buckets[t.P];
            bk.push_back(std::move(t));
            if ((int)bk.size() == Kp) {
                ready.push_back(std::move(bk));
                bk = std::vector<Trainer>();
            }
        }
        return !ready.empty();
    };
    std::vector<PackSlot> sl(std::max(1, slots));
    std::vector<char> active(sl.size(), 0), allocated(sl.size(), 0);
    int* h_buf = NULL;
    CK(cudaMallocHost((void**)&h_buf, sl.size() * sizeof(int)));
    int n_active = 0;
    long n_capture = 0, n_jobs = 0, n_packs = 0;
    double sum_setup = 0, sum_capture = 0;
    bool more = true;
    std::vector<char> pending(sl.size(), 0);
    auto launch = [&](size_t i) {
        PackSlot& ps = sl[i];
        int todo_e = std::min(chunk, ps.epochs_target - ps.epochs_run);
        for (int c = 0; c < todo_e; c++) {
            CK(cudaGraphLaunch(ps.graph, ps.pk.stream));
        }
        ps.epochs_run += todo_e;
        pending[i] = 1;
    };
    while (more || n_active > 0) {
        bool progressed = false;
        for (size_t i = 0; i < sl.size(); i++) {
            PackSlot& ps = sl[i];
            if (active[i] || !more) {
                continue;
            }
            more = produce();
            if (!more) {
                break;
            }
            double t0 = now_s();
            size_t pick = 0;  // a prepared pack of the slot's own shape, else the oldest one
            if (allocated[i]) {
                for (size_t r = 0; r < ready.size(); r++) {
                    if (shape_key(ready[r][0]) + "|K" + std::to_string(ready[r].size()) == ps.key) {
                        pick = r;
                        break;
                    }
                }
            }
            ps.m = std::move(ready[pick]);
            ready.erase(ready.begin() + pick);
            ps.K = (int)ps.m.size();
            std::string key = pack_key(ps);
            bool cap = false;
            if (!allocated[i] || key != ps.key) {
                if (allocated[i]) {
                    pack_free(ps);
                }
                pack_alloc(ps);
                allocated[i] = 1;
                ps.key = key;
                cap = true;
            }
            pack_load(ps);
            ps.t_setup = now_s() - t0;
            ps.t_capture = 0;
            if (cap) {
                pack_capture(ps);
                n_capture++;
            }
            sum_setup += ps.t_setup;
            sum_capture += ps.t_capture;
            n_jobs += ps.K;
            n_packs++;
            ps.t_start = now_s();
            active[i] = 1;
            n_active++;
            launch(i);
            progressed = true;
        }
        for (size_t i = 0; i < sl.size(); i++) {
            if (!active[i] || !pending[i]) {
                continue;
            }
            PackSlot& ps = sl[i];
            cudaError_t q = cudaStreamQuery(ps.pk.stream);
            if (q == cudaErrorNotReady) {
                continue;
            }
            CK(q);
            pending[i] = 0;
            progressed = true;
            if (ps.epochs_run < ps.epochs_target) {
                launch(i);
                continue;
            }
            ps.t_end = now_s();
            bind_cublas(ps.pk);
            pack_enqueue_eval(ps);
            CK(cudaStreamSynchronize(ps.pk.stream));
            pack_write(o, ps);
            active[i] = 0;
            n_active--;
        }
        if (!progressed) {
            usleep(20);
        }
    }
    CK(cudaDeviceSynchronize());
    double wall = now_s() - w0;
    for (size_t i = 0; i < sl.size(); i++) {
        if (allocated[i]) {
            pack_free(sl[i]);
        }
    }
    CK(cudaFreeHost(h_buf));
    fflush(o.jobs); fflush(o.steps); fflush(o.epochs); fflush(o.preds);
    std::string sp = g_out_dir + "/pool_summary.csv";
    bool fresh = access(sp.c_str(), F_OK) != 0;
    FILE* f = fopen(sp.c_str(), "a");
    if (fresh) {
        fprintf(f, "phase,jobs,slots,wall,sum_setup,sum_capture,n_capture,sum_train_overlapped\n");
    }
    fprintf(f, "%d,%ld,%zu,%.4f,%.4f,%.4f,%ld,%ld\n", phase, n_jobs, sl.size(), wall, sum_setup, sum_capture, n_capture,
        n_packs);
    fclose(f);
    fprintf(stderr, "[%.1f] packs phase %d: %ld jobs in %ld packs (K <= %d) on %zu slots in %.2f s (setup %.2f s, "
        "%ld captures %.2f s)\n", now_s(), phase, n_jobs, n_packs, Kp, sl.size(), wall, sum_setup, n_capture,
        sum_capture);
}

static int cmd_run(int argc, char** argv) {
    const char *jobs_path = NULL, *out_dir = NULL;
    int K = 16, chunk = 5, use_pool = 0, use_pack = 0, pack_slots = 8;
    for (int i = 2; i < argc; i++) {
        if (!strcmp(argv[i], "--pool")) use_pool = 1;
        else if (!strcmp(argv[i], "--pack")) use_pack = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--pack-slots")) pack_slots = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--jobs")) jobs_path = argv[++i];
        else if (!strcmp(argv[i], "--out")) out_dir = argv[++i];
        else if (!strcmp(argv[i], "--concurrent")) K = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--chunk")) chunk = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--tf32-ns")) g_math = std::max(g_math, 1);
        else if (!strcmp(argv[i], "--tf32")) g_math = 2;
        else {
            fprintf(stderr, "unknown run option %s\n", argv[i]);
            return 2;
        }
    }
    const char* go = getenv("FT_GATHER_OLD");
    g_gather_old = go != NULL && atoi(go) != 0;
    {  // stream-ordered shape buffers: keep freed pool memory for the next re-shape
        cudaMemPool_t mp;
        CK(cudaDeviceGetDefaultMemPool(&mp, 0));
        uint64_t keep_all = UINT64_MAX;
        CK(cudaMemPoolSetAttribute(mp, cudaMemPoolAttrReleaseThreshold, &keep_all));
    }
    fprintf(stderr, "run: math %s, gather %s, CUDA_DEVICE_MAX_CONNECTIONS %s\n", math_name(),
        g_gather_old ? "fc_gather (FT_GATHER_OLD)" : "fc_gather_seq",
        getenv("CUDA_DEVICE_MAX_CONNECTIONS") ? getenv("CUDA_DEVICE_MAX_CONNECTIONS") : "unset (8)");
    if (!jobs_path || !out_dir) {
        fprintf(stderr, "run --jobs FILE --out DIR [--concurrent K] [--chunk E]\n");
        return 2;
    }
    if (chunk < 1 || K < 1 || use_pack < 0 || pack_slots < 1) {
        fprintf(stderr, "--chunk, --concurrent and --pack-slots must be >= 1\n");
        return 2;
    }
    std::vector<Job> jobs = read_jobs(jobs_path);
    for (auto& j : jobs) {
        if (j.link != "identity" && use_pack) {
            fprintf(stderr, "job %s: link=%s is not supported under --pack\n", j.id.c_str(), j.link.c_str());
            return 2;
        }
        if (j.stop_loss > 0.0f && (!use_pool || use_pack)) {
            fprintf(stderr, "job %s: stop_loss needs --pool (without --pack)\n", j.id.c_str());
            return 2;
        }
    }
    mkdir(out_dir, 0755);
    std::string od(out_dir);
    g_out_dir = od;
    Out o;
    o.jobs = fopen((od + "/jobs_out.csv").c_str(), "w");
    o.steps = fopen((od + "/steps.csv").c_str(), "w");
    o.epochs = fopen((od + "/epochs.csv").c_str(), "w");
    o.preds = fopen((od + "/preds.csv").c_str(), "w");
    fprintf(o.jobs, "id,data,mode,loss,w0,w1,L,H,layers,batch,lr,seed,N,S,epochs_run,best_epoch,n_eval,"
        "best_val_mse,mu_y,sd_y,huber_delta,qlike_cap,nonfinite,status,t_setup,t_capture,t_train,wave_wall,"
        "wave_size,n_params,wave_train_wall,n_inputs,n_clipped,math,stop_loss,link\n");
    fprintf(o.steps, "id,epoch,step,loss,grad_norm,n_capped,n_nonfinite\n");
    fprintf(o.epochs, "id,epoch,lr,val_mse,val_qlike\n");
    fprintf(o.preds, "id,kind,row,f\n");
    // phases run in order (default: tune jobs 0, every other job 1); a job may take its epoch count from
    // an earlier phase's tune job (epochs_from=<id>) and its initial weights from an earlier phase's saved
    // parameters (init_from=@<id> = <out>/params/<id>.f32)
    std::vector<int> phases;
    for (auto& j : jobs) {
        int ph = atoi(j.get("phase", j.mode == "tune" ? "0" : "1").c_str());
        j.kv["phase"] = std::to_string(ph);
        if (std::find(phases.begin(), phases.end(), ph) == phases.end()) {
            phases.push_back(ph);
        }
    }
    std::sort(phases.begin(), phases.end());
    std::map<std::string, int> best_ep;
    for (int ph : phases) {
        fflush(o.jobs);
        FILE* f = fopen((od + "/jobs_out.csv").c_str(), "r");
        char line[4096];
        if (fgets(line, sizeof(line), f)) {
            while (fgets(line, sizeof(line), f)) {
                std::vector<std::string> c;
                char* save = NULL;
                for (char* tok = strtok_r(line, ",\n", &save); tok; tok = strtok_r(NULL, ",\n", &save)) {
                    c.push_back(tok);
                }
                if (c.size() > 15) {
                    best_ep[c[0]] = atoi(c[15].c_str());
                }
            }
        }
        fclose(f);
        std::vector<Job> todo;
        for (auto& j : jobs) {
            if (atoi(j.kv["phase"].c_str()) != ph) {
                continue;
            }
            Job jj = j;
            if (!jj.epochs_from.empty()) {
                if (!best_ep.count(jj.epochs_from)) {
                    fprintf(stderr, "job %s: epochs_from %s not run\n", jj.id.c_str(), jj.epochs_from.c_str());
                    exit(2);
                }
                jj.epochs = std::max(1, best_ep[jj.epochs_from]);
            }
            if (!jj.init_from.empty() && jj.init_from[0] == '@') {
                jj.init_from = od + "/params/" + jj.init_from.substr(1) + ".f32";
            }
            todo.push_back(jj);
        }
        if (use_pack > 0) {  // refits packed, tune jobs through the pool
            std::vector<Job> tunes, refits;
            for (auto& j : todo) {
                (j.mode == "tune" ? tunes : refits).push_back(j);
            }
            if (!tunes.empty()) {
                run_pool(tunes, K, chunk, o, ph);
            }
            if (!refits.empty()) {
                run_packs(refits, use_pack, pack_slots, chunk, o, ph);
            }
            continue;
        }
        if (use_pool) {
            run_pool(todo, K, chunk, o, ph);
            continue;
        }
        for (size_t a = 0; a < todo.size(); a += K) {
            std::vector<Trainer> ts(std::min(todo.size() - a, (size_t)K));
            for (size_t i = 0; i < ts.size(); i++) {
                ts[i].job = todo[a + i];
            }
            run_wave(ts, chunk, o);
        }
    }
    fclose(o.jobs); fclose(o.steps); fclose(o.epochs); fclose(o.preds);
    return 0;
}

// gradcheck: one forward / loss / backward of each job's first minibatch (no optimizer step), dumped as
// raw float32 files <out>/<id>.{params,grads,obs,y,mask,dec}.f32 + <id>.json (geometry, loss params)
static void dump(const std::string& path, const void* dptr, size_t n) {
    std::vector<float> h(n);
    CK(cudaMemcpy(h.data(), dptr, n * 4, cudaMemcpyDeviceToHost));
    FILE* f = fopen(path.c_str(), "wb");
    fwrite(h.data(), 4, n, f);
    fclose(f);
}

static int cmd_gradcheck(int argc, char** argv) {
    const char *jobs_path = NULL, *out_dir = NULL;
    for (int i = 2; i < argc; i++) {
        if (!strcmp(argv[i], "--jobs")) jobs_path = argv[++i];
        else if (!strcmp(argv[i], "--out")) out_dir = argv[++i];
    }
    std::vector<Job> jobs = read_jobs(jobs_path);
    mkdir(out_dir, 0755);
    for (auto& j : jobs) {
        Trainer t;
        t.job = j;
        trainer_setup(t, 0);
        bind_cublas(t);
        std::string b = std::string(out_dir) + "/" + j.id;
        dump(b + ".params0.f32", t.params.mem, t.params.total_elems);
        int n_real = std::min(t.B, t.N);
        gather(t.stream, t.d_Z, t.P, t.d_y, t.d_perm, t.d_epoch, t.S * t.B, 0, t.B, t.T, t.row0, t.N, t.d_obs,
            t.d_ymb, t.d_mask);
        Prec dec = net_forward(&t.arch, t.w, t.tr_acts, t.d_obs, t.B, t.T, t.P, t.d_state, j.layers, j.H, t.stream);
        fc_loss_train<<<1, 256, 0, t.stream>>>(dec.data, t.T, 2, t.d_ymb, t.d_mask, t.B, t.d_lp,
            1.0f / (float)n_real, t.d_gl, t.d_step_out);
        Float gl = {.data = t.d_gl, .shape = {t.B, t.T, 1}};
        Float gv = {.data = t.d_gv, .shape = {t.B, t.T}};
        arch_backward(&t.arch, t.w, t.tr_acts, gl, Float(), gv, t.stream);
        CK(cudaStreamSynchronize(t.stream));
        CK(cudaDeviceSynchronize());
        dump(b + ".grads.f32", t.grads.mem, t.grads.total_elems);
        dump(b + ".obs.f32", t.d_obs, (size_t)t.B * t.T * t.P);
        dump(b + ".y.f32", t.d_ymb, t.B);
        dump(b + ".mask.f32", t.d_mask, t.B);
        dump(b + ".dec.f32", dec.data, (size_t)t.B * t.T * 2);
        dump(b + ".step.f32", t.d_step_out, 3);
        dump(b + ".Z.f32", t.d_Z, (size_t)t.n_ctx * t.P);
        dump(b + ".yrow.f32", t.d_y, t.n_ctx);
        {
            FILE* fk = fopen((b + ".keep.i32").c_str(), "wb");
            fwrite(t.keep.data(), 4, t.keep.size(), fk);
            fclose(fk);
        }
        std::vector<int> perm((size_t)j.max_epochs * t.S * t.B);
        CK(cudaMemcpy(perm.data(), t.d_perm, perm.size() * 4, cudaMemcpyDeviceToHost));
        FILE* fp = fopen((b + ".perm.i32").c_str(), "wb");
        fwrite(perm.data(), 4, perm.size(), fp);
        fclose(fp);
        FILE* f = fopen((b + ".json").c_str(), "w");
        fprintf(f, "{\"B\": %d, \"T\": %d, \"P\": %d, \"H\": %d, \"layers\": %d, \"S\": %d, \"N\": %d, "
            "\"row0\": %d, \"n_ctx\": %d, \"eval_row0\": %d, \"n_eval\": %d, \"n_params\": %ld, "
            "\"loss_kind\": %d, \"mu_y\": %.9g, \"sd_y\": %.9g, \"huber_delta\": %.9g, \"qlike_eps\": %.9g, "
            "\"qlike_cap\": %.9g, \"lr\": %.9g, \"momentum\": %.9g, \"max_grad_norm\": %.9g, \"n_real\": %d, "
            "\"link\": %d, \"link_c\": %.9g, \"link_s\": %.9g}\n",
            t.B, t.T, t.P, j.H, j.layers, t.S, t.N, t.row0, t.n_ctx, t.eval_row0, t.n_eval,
            (long)t.params.total_elems, t.lp.kind, t.lp.mu_y, t.lp.sd_y, t.lp.huber_delta, t.lp.qlike_eps,
            t.lp.qlike_cap, j.lr, j.momentum, j.max_grad_norm, n_real, t.lp.link, t.lp.link_c, t.lp.link_s);
        fclose(f);
        trainer_free(t);
    }
    return 0;
}

// losscheck: IN = int32 n, int32 kind, float32 mu, sd, delta, eps, cap, then float32 o[n], y[n];
// OUT = float32 l[n], g[n] (dL/do, through the device kernel's fc_loss), capped[n]
__global__ void fc_losscheck(const float* o, const float* y, int n, LossParams p, float* l, float* g,
        float* c) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    float dfdo;
    float f = fc_link(p, o[i], &dfdo);
    float li, gi;
    int ci;
    fc_loss(p, f, y[i], &li, &gi, &ci);
    l[i] = li;
    g[i] = gi * dfdo;
    c[i] = (float)ci;
}

static int cmd_losscheck(int argc, char** argv) {
    if (argc < 4) {
        return 2;
    }
    FILE* f = fopen(argv[2], "rb");
    int n, kind;
    float hdr[5];
    if (fread(&n, 4, 1, f) != 1 || fread(&kind, 4, 1, f) != 1 || fread(hdr, 4, 5, f) != 5) {
        return 2;
    }
    std::vector<float> o(n), y(n);
    if (fread(o.data(), 4, n, f) != (size_t)n || fread(y.data(), 4, n, f) != (size_t)n) {
        return 2;
    }
    fclose(f);
    LossParams p = {kind, hdr[0], hdr[1], hdr[2], hdr[3], hdr[4]};
    if (argc >= 5 && !strcmp(argv[4], "softplus")) {  // losscheck IN OUT softplus: the softplus link on mu, sd
        double c = log(expm1((double)p.mu_y));
        p.link = LINK_SOFTPLUS;
        p.link_c = (float)c;
        p.link_s = (float)((double)p.sd_y * (1.0 + exp(-c)));
        fprintf(stderr, "losscheck softplus: c %.9g s %.9g\n", p.link_c, p.link_s);
    }
    float *d_o = dalloc<float>(n), *d_y = dalloc<float>(n), *d_l = dalloc<float>(n), *d_g = dalloc<float>(n),
          *d_c = dalloc<float>(n);
    CK(cudaMemcpy(d_o, o.data(), n * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_y, y.data(), n * 4, cudaMemcpyHostToDevice));
    fc_losscheck<<<(n + 255) / 256, 256>>>(d_o, d_y, n, p, d_l, d_g, d_c);
    CK(cudaDeviceSynchronize());
    std::vector<float> l(n), g(n), c(n);
    CK(cudaMemcpy(l.data(), d_l, n * 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(g.data(), d_g, n * 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(c.data(), d_c, n * 4, cudaMemcpyDeviceToHost));
    FILE* fo = fopen(argv[3], "wb");
    fwrite(l.data(), 4, n, fo);
    fwrite(g.data(), 4, n, fo);
    fwrite(c.data(), 4, n, fo);
    fclose(fo);
    return 0;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "forecast_train run|gradcheck|losscheck ...\n");
        return 2;
    }
    cudaDeviceProp prop;
    CK(cudaGetDeviceProperties(&prop, 0));
    int drv = 0, rt = 0;
    cudaDriverGetVersion(&drv);
    cudaRuntimeGetVersion(&rt);
    fprintf(stderr, "device %s (sm_%d%d, %d SMs), driver %d, runtime %d, precision %s\n", prop.name, prop.major,
        prop.minor, prop.multiProcessorCount, drv, rt, USE_BF16 ? "bf16" : "fp32");
    if (!strcmp(argv[1], "run")) return cmd_run(argc, argv);
    if (!strcmp(argv[1], "gradcheck")) return cmd_gradcheck(argc, argv);
    if (!strcmp(argv[1], "losscheck")) return cmd_losscheck(argc, argv);
    fprintf(stderr, "unknown command %s\n", argv[1]);
    return 2;
}
