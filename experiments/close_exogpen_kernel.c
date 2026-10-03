/* close_exogpen_kernel.c -- the walk-forward of experiments/close_exogpen.py in C.
 *
 * One call = one arm over every forecast session: for each block of TUNE_PER sessions
 *   1. the penalty re-choice on the block-start window's fit / embargo / tail split
 *      (ridge: one eigendecomposition (dsyevd) of the profiled, factor-scaled fit Gram
 *      for each backbone : exogenous ratio r gives the tail predictions for the whole
 *      alpha grid; lasso / elastic net: the coordinate-descent path over the grid, warm
 *      started from the next larger alpha; multi-penalty ridge: cyclic coordinate search
 *      over the group penalties, Cholesky solves);
 *   2. the refit every session: the window's Gram (X'X, X'y, y'y about a fixed shift)
 *      is rebuilt exactly at the block start and then moved by a rank-one add of the
 *      newest row and a rank-one drop of the oldest; ridge = Cholesky (dpotrf/dpotrs) of
 *      the centered Gram plus the penalty diagonal; lasso / elastic net = covariance-form
 *      coordinate descent with penalty factors (0 = unpenalized coordinate, plain
 *      least-squares update), warm started from the previous session, active-set
 *      cycling, then the exact solve of the KKT system on the support with a full KKT
 *      check (the result is the exact optimum when the check passes).
 * Penalty convention (glmnet's penalty.factor, sklearn units): column j carries
 *   n * alpha * pf_j * (l1 |b_j| + (1 - l1) / 2 b_j^2); the intercept is unpenalized
 *   (every fit is on centered data).  pf_j < 0 marks a column left out (coefficient 0).
 * Arrays are C-contiguous float64 (row-major); a symmetric matrix is stored in full,
 * so row- and column-major coincide for LAPACK.
 * Build: gcc -O3 -march=native -fPIC -shared -o close_exogpen_kernel.so close_exogpen_kernel.c
 *        -l:liblapack.so.3 -l:libblas.so.3 -lm   (no -ffast-math)
 */
#include <math.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

extern void dpotrf_(const char *uplo, const int *n, double *a, const int *lda, int *info,
                    size_t);
extern void dpotrs_(const char *uplo, const int *n, const int *nrhs, const double *a,
                    const int *lda, double *b, const int *ldb, int *info, size_t);
extern void dsyevd_(const char *jobz, const char *uplo, const int *n, double *a,
                    const int *lda, double *w, double *work, const int *lwork, int *iwork,
                    const int *liwork, int *info, size_t, size_t);
extern void dsyrk_(const char *uplo, const char *trans, const int *n, const int *k,
                   const double *alpha, const double *a, const int *lda, const double *beta,
                   double *c, const int *ldc, size_t, size_t);

/* ------------------------------------------------------------------ window statistics */
typedef struct {
    int p, n;
    double *S, *sx, *sxy, *m; /* S = sum (x-m)(x-m)', sx = sum (x-m), sxy = sum (x-m)(y-my) */
    double sy, syy, my;
} Roll;

static void roll_alloc(Roll *R, int p) {
    R->p = p;
    R->S = malloc(sizeof(double) * (size_t)p * p);
    R->sx = malloc(sizeof(double) * p);
    R->sxy = malloc(sizeof(double) * p);
    R->m = malloc(sizeof(double) * p);
}
static void roll_free(Roll *R) { free(R->S); free(R->sx); free(R->sxy); free(R->m); }

/* exact statistics of rows [r0, r1); the shift = the rows' mean */
static void roll_exact(Roll *R, const double *X, const double *y, int r0, int r1) {
    int p = R->p, n = r1 - r0;
    R->n = n;
    memset(R->m, 0, sizeof(double) * p);
    double my = 0.0;
    for (int i = r0; i < r1; i++) {
        const double *x = X + (size_t)i * p;
        for (int j = 0; j < p; j++) R->m[j] += x[j];
        my += y[i];
    }
    for (int j = 0; j < p; j++) R->m[j] /= n;
    my /= n;
    R->my = my;
    double *Z = malloc(sizeof(double) * (size_t)n * p);
    memset(R->sx, 0, sizeof(double) * p);
    memset(R->sxy, 0, sizeof(double) * p);
    R->sy = 0.0;
    R->syy = 0.0;
    for (int i = 0; i < n; i++) {
        const double *x = X + (size_t)(r0 + i) * p;
        double *z = Z + (size_t)i * p;
        double e = y[r0 + i] - my;
        for (int j = 0; j < p; j++) {
            z[j] = x[j] - R->m[j];
            R->sx[j] += z[j];
            R->sxy[j] += z[j] * e;
        }
        R->sy += e;
        R->syy += e * e;
    }
    /* column-major view of Z (row-major n x p) is A = p x n, lda = p: S = A A' */
    const double one = 1.0, zero = 0.0;
    dsyrk_("U", "N", &p, &n, &one, Z, &p, &zero, R->S, &p, 1, 1);
    for (int j = 0; j < p; j++) /* column-major upper -> full */
        for (int i = 0; i < j; i++) R->S[(size_t)i * p + j] = R->S[(size_t)j * p + i];
    free(Z);
}

/* add row x_in (target y_in), drop row x_out (target y_out) */
static void roll_step(Roll *R, const double *x_in, double y_in, const double *x_out,
                      double y_out, double *u, double *v) {
    int p = R->p;
    double ei = y_in - R->my, eo = y_out - R->my;
    for (int j = 0; j < p; j++) {
        u[j] = x_in[j] - R->m[j];
        v[j] = x_out[j] - R->m[j];
    }
    for (int i = 0; i < p; i++) {
        double ui = u[i], vi = v[i];
        double *Si = R->S + (size_t)i * p;
        for (int j = 0; j < p; j++) Si[j] += ui * u[j] - vi * v[j];
        R->sx[i] += ui - vi;
        R->sxy[i] += ui * ei - vi * eo;
    }
    R->sy += ei - eo;
    R->syy += ei * ei - eo * eo;
}

/* centered Gram G, moment c, means, centered y'y */
static void roll_center(const Roll *R, double *G, double *c, double *xbar, double *ybar,
                        double *yy) {
    int p = R->p;
    double n = (double)R->n;
    for (int i = 0; i < p; i++) {
        const double *Si = R->S + (size_t)i * p;
        double *Gi = G + (size_t)i * p;
        double si = R->sx[i] / n;
        for (int j = 0; j < p; j++) Gi[j] = Si[j] - si * R->sx[j];
        c[i] = R->sxy[i] - si * R->sy;
        xbar[i] = R->m[i] + si;
    }
    *ybar = R->my + R->sy / n;
    *yy = R->syy - R->sy * R->sy / n;
}

/* ------------------------------------------------------------------ solvers */
static int cholsolve(int n, double *A, double *b) { /* A overwritten; b <- A^-1 b */
    int info = 0, one = 1;
    if (n == 0) return 0;
    dpotrf_("L", &n, A, &n, &info, 1);
    if (info != 0) return info;
    dpotrs_("L", &n, &one, A, &n, b, &n, &info, 1);
    return info;
}

/* covariance-form coordinate descent; g = c - G b is kept in step with b */
static long cd_run(int K, const double *G, const double *mu, const double *lam2, double *b,
                   double *g, double tol, long max_sweeps) {
    long sweeps = 0;
    int full = 1;
    while (sweeps < max_sweeps) {
        sweeps++;
        double big = 0.0;
        for (int j = 0; j < K; j++) {
            double bj = b[j];
            if (!full && bj == 0.0 && mu[j] > 0.0) continue;
            double gjj = G[(size_t)j * K + j];
            double den = gjj + lam2[j];
            if (den <= 0.0) continue;
            double z = g[j] + gjj * bj, nb;
            if (z > mu[j]) nb = (z - mu[j]) / den;
            else if (z < -mu[j]) nb = (z + mu[j]) / den;
            else nb = 0.0;
            double d = nb - bj;
            if (d != 0.0) {
                b[j] = nb;
                const double *Gj = G + (size_t)j * K;
                for (int k = 0; k < K; k++) g[k] -= Gj[k] * d;
                double s = fabs(d) * sqrt(gjj);
                if (s > big) big = s;
            }
        }
        if (big < tol) {
            if (full) break;
            full = 1;
        } else {
            full = 0;
        }
    }
    return sweeps;
}

typedef struct {
    double *M, *rhs, *bt;
    int *A;
} PolishWS;

/* exact solve of the KKT system on the support of b with its signs; 1 + b replaced when
 * every KKT condition holds (signs on the support, |c_j - G_j b| <= mu_j + tol off it) */
static int polish(int K, const double *G, const double *c, const double *mu,
                  const double *lam2, double *b, double kkt_tol, PolishWS *w) {
    int na = 0;
    for (int j = 0; j < K; j++)
        if (b[j] != 0.0 || mu[j] == 0.0) w->A[na++] = j;
    for (int a = 0; a < na; a++) {
        int ja = w->A[a];
        const double *Gj = G + (size_t)ja * K;
        for (int a2 = 0; a2 < na; a2++) w->M[(size_t)a * na + a2] = Gj[w->A[a2]];
        w->M[(size_t)a * na + a] += lam2[ja];
        double s = (b[ja] > 0.0) - (b[ja] < 0.0);
        w->rhs[a] = c[ja] - (mu[ja] > 0.0 ? mu[ja] * s : 0.0);
    }
    if (cholsolve(na, w->M, w->rhs) != 0) return 0;
    for (int a = 0; a < na; a++) {
        int ja = w->A[a];
        if (mu[ja] > 0.0) {
            double s = (b[ja] > 0.0) - (b[ja] < 0.0);
            double sn = (w->rhs[a] > 0.0) - (w->rhs[a] < 0.0);
            if (sn != s) return 0;
        }
    }
    memset(w->bt, 0, sizeof(double) * K);
    for (int a = 0; a < na; a++) w->bt[w->A[a]] = w->rhs[a];
    for (int j = 0; j < K; j++) {
        if (w->bt[j] != 0.0 || mu[j] == 0.0) continue;
        const double *Gj = G + (size_t)j * K;
        double gj = c[j];
        for (int a = 0; a < na; a++) gj -= Gj[w->A[a]] * w->rhs[a];
        if (fabs(gj) > mu[j] + kkt_tol) return 0;
    }
    memcpy(b, w->bt, sizeof(double) * K);
    return 1;
}

/* status: 1 polished after the first descent, 2 after a tighter one, 3 descent only */
static int enet_solve(int K, const double *G, const double *c, const double *mu,
                      const double *lam2, double *b, double ynorm, double cd_rel,
                      double kkt_rel, int max_rounds, long max_sweeps, double *g,
                      double *bp, PolishWS *w, long *sweeps) {
    double cmax = 0.0;
    for (int j = 0; j < K; j++) {
        const double *Gj = G + (size_t)j * K;
        double s = c[j];
        for (int k = 0; k < K; k++) s -= Gj[k] * b[k];
        g[j] = s;
        if (fabs(c[j]) > cmax) cmax = fabs(c[j]);
    }
    double tol = cd_rel * ynorm, kkt_tol = kkt_rel * (cmax > 0.0 ? cmax : 1.0);
    long tot = 0;
    for (int r = 0; r < max_rounds; r++) {
        tot += cd_run(K, G, mu, lam2, b, g, tol, max_sweeps);
        memcpy(bp, b, sizeof(double) * K);
        if (polish(K, G, c, mu, lam2, bp, kkt_tol, w)) {
            memcpy(b, bp, sizeof(double) * K);
            *sweeps = tot;
            return r == 0 ? 1 : 2;
        }
        tol *= 1e-2;
    }
    *sweeps = tot;
    return 3;
}

/* ------------------------------------------------------------------ helpers */
static void gather(int p, const double *G, const int *idx, int K, double *out) {
    for (int a = 0; a < K; a++) {
        const double *Gi = G + (size_t)idx[a] * p;
        double *o = out + (size_t)a * K;
        for (int b = 0; b < K; b++) o[b] = Gi[idx[b]];
    }
}

/* tail mean squared error of a fit given on the kept columns */
static double tail_mse(int p, int K, const int *idx, const double *bK, const double *xbar,
                       double ybar, const double *Xv, const double *yv, int nv) {
    double s = 0.0;
    for (int v = 0; v < nv; v++) {
        const double *x = Xv + (size_t)v * p;
        double f = ybar;
        for (int a = 0; a < K; a++) f += (x[idx[a]] - xbar[idx[a]]) * bK[a];
        double e = f - yv[v];
        s += e * e;
    }
    return s / nv;
}

/* ridge tail MSE for every alpha with penalty factors pf over the kept columns
 * (pf = 0: unpenalized, profiled out exactly): one dsyevd */
static int ridge_grid(int p, const double *G, const double *c, const double *xbar,
                      double ybar, const double *Xv, const double *yv, int nv, int K,
                      const int *idx, const double *pf, int na, const double *alphas,
                      double *mse) {
    int nu = 0, np_ = 0;
    int *U = malloc(sizeof(int) * K), *P = malloc(sizeof(int) * K);
    for (int a = 0; a < K; a++) {
        if (pf[a] == 0.0) U[nu++] = idx[a];
        else P[np_++] = a; /* position in idx */
    }
    int *Pc = malloc(sizeof(int) * (np_ > 0 ? np_ : 1));
    for (int i = 0; i < np_; i++) Pc[i] = idx[P[i]];
    double *Gt = malloc(sizeof(double) * (size_t)np_ * np_ + 1);
    double *ct = malloc(sizeof(double) * (np_ + 1));
    double *H = malloc(sizeof(double) * ((size_t)nu * np_ + 1)); /* G_UU^-1 G_UP, col-major nu x np */
    double *h = malloc(sizeof(double) * (nu + 1));
    gather(p, G, Pc, np_, Gt);
    for (int i = 0; i < np_; i++) ct[i] = c[Pc[i]];
    int rc = 0;
    if (nu > 0) {
        double *L = malloc(sizeof(double) * (size_t)nu * nu);
        gather(p, G, U, nu, L);
        for (int i = 0; i < np_; i++)
            for (int u = 0; u < nu; u++) H[(size_t)i * nu + u] = G[(size_t)U[u] * p + Pc[i]];
        for (int u = 0; u < nu; u++) h[u] = c[U[u]];
        int info = 0, nrhs = np_, one = 1;
        dpotrf_("L", &nu, L, &nu, &info, 1);
        if (info == 0 && np_ > 0) dpotrs_("L", &nu, &nrhs, L, &nu, H, &nu, &info, 1);
        if (info == 0) dpotrs_("L", &nu, &one, L, &nu, h, &nu, &info, 1);
        free(L);
        if (info != 0) { rc = 100 + info; goto done; }
        for (int i = 0; i < np_; i++) {
            const double *Hi = H + (size_t)i * nu;
            for (int k = 0; k < np_; k++) {
                double s = 0.0;
                for (int u = 0; u < nu; u++) s += G[(size_t)Pc[k] * p + U[u]] * Hi[u];
                Gt[(size_t)i * np_ + k] -= s;
            }
            double s = 0.0;
            for (int u = 0; u < nu; u++) s += G[(size_t)Pc[i] * p + U[u]] * h[u];
            ct[i] -= s;
        }
        for (int i = 0; i < np_; i++) /* symmetrize */
            for (int k = 0; k < i; k++) {
                double s = 0.5 * (Gt[(size_t)i * np_ + k] + Gt[(size_t)k * np_ + i]);
                Gt[(size_t)i * np_ + k] = s;
                Gt[(size_t)k * np_ + i] = s;
            }
    }
    double *sq = malloc(sizeof(double) * (np_ + 1));
    for (int i = 0; i < np_; i++) sq[i] = sqrt(pf[P[i]]);
    for (int i = 0; i < np_; i++) {
        for (int k = 0; k < np_; k++) Gt[(size_t)i * np_ + k] /= sq[i] * sq[k];
        ct[i] /= sq[i];
    }
    double *lam = malloc(sizeof(double) * (np_ + 1));
    if (np_ > 0) {
        int lwork = -1, liwork = -1, info = 0, iw;
        double wq;
        dsyevd_("V", "U", &np_, Gt, &np_, lam, &wq, &lwork, &iw, &liwork, &info, 1, 1);
        lwork = (int)wq;
        liwork = iw;
        double *work = malloc(sizeof(double) * lwork);
        int *iwork = malloc(sizeof(int) * liwork);
        dsyevd_("V", "U", &np_, Gt, &np_, lam, work, &lwork, iwork, &liwork, &info, 1, 1);
        free(work);
        free(iwork);
        if (info != 0) { free(sq); free(lam); rc = 200 + info; goto done; }
    }
    /* Gt now holds V (column k = eigenvector k); w = V' cs */
    double *w = malloc(sizeof(double) * (np_ + 1));
    for (int k = 0; k < np_; k++) {
        const double *Vk = Gt + (size_t)k * np_;
        double s = 0.0;
        for (int i = 0; i < np_; i++) s += Vk[i] * ct[i];
        w[k] = s;
    }
    double *z = malloc(sizeof(double) * (np_ + 1)), *Zs = malloc(sizeof(double) * (np_ + 1));
    for (int a = 0; a < na; a++) mse[a] = 0.0;
    for (int v = 0; v < nv; v++) {
        const double *x = Xv + (size_t)v * p;
        double base = ybar;
        for (int u = 0; u < nu; u++) base += (x[U[u]] - xbar[U[u]]) * h[u];
        for (int i = 0; i < np_; i++) {
            double zi = x[Pc[i]] - xbar[Pc[i]];
            const double *Hi = H + (size_t)i * nu;
            for (int u = 0; u < nu; u++) zi -= (x[U[u]] - xbar[U[u]]) * Hi[u];
            z[i] = zi / sq[i];
        }
        for (int k = 0; k < np_; k++) {
            const double *Vk = Gt + (size_t)k * np_;
            double s = 0.0;
            for (int i = 0; i < np_; i++) s += Vk[i] * z[i];
            Zs[k] = s * w[k];
        }
        for (int a = 0; a < na; a++) {
            double f = base;
            for (int k = 0; k < np_; k++) f += Zs[k] / (lam[k] + alphas[a]);
            double e = f - yv[v];
            mse[a] += e * e;
        }
    }
    for (int a = 0; a < na; a++) mse[a] /= nv;
    free(w); free(z); free(Zs); free(sq); free(lam);
done:
    free(U); free(P); free(Pc); free(Gt); free(ct); free(H); free(h);
    return rc;
}

/* ------------------------------------------------------------------ the walk-forward */
int exogpen_run(
    /* data */
    int N, int p, const double *X, const double *y, int W,
    /* arm */
    int est,  /* 0 ridge, 1 lasso, 2 elastic net */
    int mode, /* 0 grid over (r, alpha), 1 multi-penalty ridge */
    int n_blocks, const int *bstart, /* n_blocks + 1 forecast indices */
    int n_r, const double *pf,       /* n_blocks x n_r x p; < 0 = column left out */
    int n_alpha, const double *alphas, int val_tail, int embargo,
    const int *group, int n_groups, const double *mpr_start, int mpr_passes,
    const int *is_bb, double cd_rel, double kkt_rel, int max_rounds, long max_sweeps,
    /* outputs */
    double *pred, double *theta, double *b0, double *vBB, double *vEE, double *vBE,
    int *status, long *sweeps, double *val_mse, int *choice, double *pen_g,
    double *mpr_mse) {
    const double l1 = est == 1 ? 1.0 : 0.5;
    Roll R, Rf;
    roll_alloc(&R, p);
    roll_alloc(&Rf, p);
    double *G = malloc(sizeof(double) * (size_t)p * p);
    double *c = malloc(sizeof(double) * p), *xbar = malloc(sizeof(double) * p);
    double *GK = malloc(sizeof(double) * (size_t)p * p);
    double *cK = malloc(sizeof(double) * p), *bK = malloc(sizeof(double) * p);
    double *mu = malloc(sizeof(double) * p), *lam2 = malloc(sizeof(double) * p);
    double *pen = malloc(sizeof(double) * p), *g = malloc(sizeof(double) * p);
    double *bp = malloc(sizeof(double) * p), *u = malloc(sizeof(double) * p);
    double *v = malloc(sizeof(double) * p), *prev = calloc(p, sizeof(double));
    double *pfK = malloc(sizeof(double) * p), *cur = malloc(sizeof(double) * (n_groups + 1));
    int *idx = malloc(sizeof(int) * p);
    PolishWS w;
    w.M = malloc(sizeof(double) * (size_t)p * p);
    w.rhs = malloc(sizeof(double) * p);
    w.bt = malloc(sizeof(double) * p);
    w.A = malloc(sizeof(int) * p);
    int rc = 0;
    for (int blk = 0; blk < n_blocks; blk++) {
        int i0 = bstart[blk], i1 = bstart[blk + 1];
        /* ---- re-choice on the block-start window [i0, i0 + W) */
        int fit_hi = i0 + W - val_tail - embargo;
        roll_exact(&Rf, X, y, i0, fit_hi);
        double ybf, yyf;
        roll_center(&Rf, G, c, xbar, &ybf, &yyf);
        const double *Xv = X + (size_t)(i0 + W - val_tail) * p;
        const double *yv = y + (i0 + W - val_tail);
        double nf = (double)Rf.n;
        int ir_best = 0, ia_best = 0;
        if (mode == 0) {
            for (int ir = 0; ir < n_r; ir++) {
                const double *pfr = pf + ((size_t)blk * n_r + ir) * p;
                int K = 0;
                for (int j = 0; j < p; j++)
                    if (pfr[j] >= 0.0) { idx[K] = j; pfK[K] = pfr[j]; K++; }
                double *ms = val_mse + ((size_t)blk * n_r + ir) * n_alpha;
                if (est == 0) {
                    rc = ridge_grid(p, G, c, xbar, ybf, Xv, yv, val_tail, K, idx, pfK, n_alpha,
                                    alphas, ms);
                    if (rc) goto out;
                } else {
                    gather(p, G, idx, K, GK);
                    for (int a = 0; a < K; a++) { cK[a] = c[idx[a]]; bK[a] = 0.0; }
                    for (int ia = n_alpha - 1; ia >= 0; ia--) { /* path: largest alpha first */
                        for (int a = 0; a < K; a++) {
                            mu[a] = nf * alphas[ia] * l1 * pfK[a];
                            lam2[a] = nf * alphas[ia] * (1.0 - l1) * pfK[a];
                        }
                        long sw;
                        enet_solve(K, GK, cK, mu, lam2, bK, sqrt(yyf), cd_rel, kkt_rel,
                                   max_rounds, max_sweeps, g, bp, &w, &sw);
                        ms[ia] = tail_mse(p, K, idx, bK, xbar, ybf, Xv, yv, val_tail);
                    }
                }
            }
            double best = INFINITY;
            for (int ir = 0; ir < n_r; ir++)
                for (int ia = 0; ia < n_alpha; ia++) {
                    double m = val_mse[((size_t)blk * n_r + ir) * n_alpha + ia];
                    if (m < best) { best = m; ir_best = ir; ia_best = ia; }
                }
            choice[2 * blk] = ir_best;
            choice[2 * blk + 1] = ia_best;
        } else { /* multi-penalty ridge: cyclic coordinate search over the group penalties */
            const double *pfr = pf + (size_t)blk * n_r * p;
            int K = 0;
            for (int j = 0; j < p; j++)
                if (pfr[j] >= 0.0) idx[K++] = j;
            double *pg = pen_g + (size_t)blk * n_groups;
            for (int gi = 0; gi < n_groups; gi++) pg[gi] = mpr_start[blk];
            int *present = calloc(n_groups, sizeof(int));
            for (int a = 0; a < K; a++)
                if (pfr[idx[a]] > 0.0) present[group[idx[a]]] = 1;
            double best = INFINITY;
            for (int pass = 0; pass <= mpr_passes; pass++) {
                for (int gi = 0; gi < (pass == 0 ? 1 : n_groups); gi++) {
                    if (pass > 0 && !present[gi]) continue;
                    for (int ia = 0; ia < (pass == 0 ? 1 : n_alpha); ia++) {
                        for (int q = 0; q < n_groups; q++) cur[q] = pg[q];
                        if (pass > 0) {
                            if (alphas[ia] == pg[gi]) continue;
                            cur[gi] = alphas[ia];
                        }
                        gather(p, G, idx, K, GK);
                        for (int a = 0; a < K; a++) {
                            int j = idx[a];
                            GK[(size_t)a * K + a] += pfr[j] > 0.0 ? cur[group[j]] : 0.0;
                            bK[a] = c[j];
                        }
                        if (cholsolve(K, GK, bK) != 0) { rc = 300; free(present); goto out; }
                        double m = tail_mse(p, K, idx, bK, xbar, ybf, Xv, yv, val_tail);
                        if (m < best) {
                            best = m;
                            for (int q = 0; q < n_groups; q++) pg[q] = cur[q];
                        }
                    }
                }
            }
            mpr_mse[blk] = best;
            choice[2 * blk] = 0;
            choice[2 * blk + 1] = -1;
            free(present);
        }
        /* ---- the chosen fit's kept columns and penalties */
        const double *pfr = pf + ((size_t)blk * n_r + ir_best) * p;
        int K = 0;
        for (int j = 0; j < p; j++)
            if (pfr[j] >= 0.0) {
                idx[K] = j;
                if (mode == 0) pen[K] = alphas[ia_best] * pfr[j];
                else pen[K] = pfr[j] > 0.0 ? pen_g[(size_t)blk * n_groups + group[j]] : 0.0;
                K++;
            }
        /* ---- refit every session */
        for (int j = i0; j < i1; j++) {
            int t = W + j;
            if (j == i0) roll_exact(&R, X, y, j, t); /* re-anchor at the re-choice */
            else roll_step(&R, X + (size_t)(t - 1) * p, y[t - 1], X + (size_t)(j - 1) * p,
                           y[j - 1], u, v);
            double yb, yy;
            roll_center(&R, G, c, xbar, &yb, &yy);
            double n = (double)R.n;
            gather(p, G, idx, K, GK);
            for (int a = 0; a < K; a++) cK[a] = c[idx[a]];
            if (est == 0) {
                for (int a = 0; a < K; a++) {
                    GK[(size_t)a * K + a] += pen[a];
                    bK[a] = cK[a];
                }
                if (cholsolve(K, GK, bK) != 0) { rc = 400; goto out; }
                status[j] = 0;
                sweeps[j] = 0;
            } else {
                for (int a = 0; a < K; a++) {
                    mu[a] = n * l1 * pen[a];
                    lam2[a] = n * (1.0 - l1) * pen[a];
                    bK[a] = prev[idx[a]];
                }
                status[j] = enet_solve(K, GK, cK, mu, lam2, bK, sqrt(yy), cd_rel, kkt_rel,
                                       max_rounds, max_sweeps, g, bp, &w, &sweeps[j]);
            }
            double *th = theta + (size_t)(j) * p;
            memset(th, 0, sizeof(double) * p);
            for (int a = 0; a < K; a++) th[idx[a]] = bK[a];
            memcpy(prev, th, sizeof(double) * p);
            double bb0 = yb;
            for (int k = 0; k < p; k++) bb0 -= xbar[k] * th[k];
            const double *xt = X + (size_t)t * p;
            double f = bb0;
            for (int k = 0; k < p; k++) f += xt[k] * th[k];
            pred[j] = f;
            b0[j] = bb0;
            /* fitted-variance parts over the window: backbone, exogenous, cross */
            double sBB = 0.0, sEE = 0.0, sBE = 0.0;
            for (int a = 0; a < K; a++) {
                int ia_ = idx[a];
                const double *Gi = G + (size_t)ia_ * p;
                double ub = 0.0, ue = 0.0;
                for (int b2 = 0; b2 < K; b2++) {
                    int jb = idx[b2];
                    if (is_bb[jb]) ub += Gi[jb] * bK[b2];
                    else ue += Gi[jb] * bK[b2];
                }
                if (is_bb[ia_]) { sBB += bK[a] * ub; sBE += bK[a] * ue; }
                else sEE += bK[a] * ue;
            }
            vBB[j] = sBB / n;
            vEE[j] = sEE / n;
            vBE[j] = sBE / n;
        }
    }
out:
    roll_free(&R); roll_free(&Rf);
    free(G); free(c); free(xbar); free(GK); free(cK); free(bK); free(mu); free(lam2);
    free(pen); free(g); free(bp); free(u); free(v); free(prev); free(pfK); free(cur);
    free(idx); free(w.M); free(w.rhs); free(w.bt); free(w.A);
    return rc;
}
