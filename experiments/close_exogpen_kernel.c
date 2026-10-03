/* close_exogpen_kernel.c -- specs/causal_tune_linear.py::RollingTunedLinear in C.
 *
 * A line-by-line port of the spec's rolling-tuned linear model (and of the parts of
 * src/models/reclasso_har.py it calls), extended with a penalty factor for each column:
 *   ridge          D_j = alpha * pf_j (0 on locked columns); Sherman-Morrison rank-one add
 *                  of the entering row and drop of the leaving row on the ridged inverse K;
 *                  theta = K c at every solve.
 *   lasso / enet   mu_vec_j = n alpha l1 pf_j (0 on locked), Gr = Xa'Xa + n alpha (1 - l1) pf_j
 *                  on the diagonal of the non-locked columns; the Garrigues-El Ghaoui online
 *                  homotopy (enet_online), two calls a session (+1 entering row, -1 leaving
 *                  row), warm (theta, A, s); locked columns never leave A.
 *   re-choice      every block start: the tune-boundary mask (given by the caller for each
 *                  candidate structure: locked set, masked set, penalty factors), the
 *                  forward split (fit / embargo / tail) of the window, the grid scored by the
 *                  batch solution (ridge: np.linalg.solve; lasso / enet: _batch_theta =
 *                  FWL on the locked block + the batch homotopy lasso_homotopy), argmin in
 *                  grid order with strict improvement, then the cold seed (_seed).
 *   lasso only     the run-length trackers (_init_runs / _degenerate_live): a live column
 *                  that turns constant, or equal on every window row to an earlier column,
 *                  between re-choices is masked and the warm state cold-reseeded.
 * Penalty factors (glmnet's penalty.factor): pf = 1 everywhere reproduces the spec; the
 * study's arms lock the HAR + calendar backbone (pf 0, never leaves A) or scale its
 * penalty by r.  With pf != 1 the batch elastic net is solved exactly by column scaling
 * (b = g / pf on D^-1 G D^-1 + lam2 diag(1 / pf), D^-1 c, unit L1 weights).
 * General solves follow numpy: dgesv, and on an exactly singular matrix the minimum-norm
 * least-squares solution (dgelsd, rcond = eps * n), as _solve; pinv as numpy.linalg.pinv
 * (dgesdd, rcond 1e-15).  Tolerances: enet_online tol 1e-9, homotopy _EPS 1e-10, seed
 * support |theta| > 1e-9.  Arrays are C-contiguous float64; the design carries the
 * intercept as its last column (all ones).
 * Build: gcc -O3 -march=native -fPIC -shared -o close_exogpen_kernel.so close_exogpen_kernel.c
 *        -l:liblapack.so.3 -l:libblas.so.3 -lm   (no -ffast-math)
 */
#include <float.h>
#include <math.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

extern void dgesv_(const int *n, const int *nrhs, double *a, const int *lda, int *ipiv,
                   double *b, const int *ldb, int *info);
extern void dgelsd_(const int *m, const int *n, const int *nrhs, double *a, const int *lda,
                    double *b, const int *ldb, double *s, const double *rcond, int *rank,
                    double *work, const int *lwork, int *iwork, int *info);
extern void dgesdd_(const char *jobz, const int *m, const int *n, double *a, const int *lda,
                    double *s, double *u, const int *ldu, double *vt, const int *ldvt,
                    double *work, const int *lwork, int *iwork, int *info, size_t);
extern void dsyrk_(const char *uplo, const char *trans, const int *n, const int *k,
                   const double *alpha, const double *a, const int *lda, const double *beta,
                   double *c, const int *ldc, size_t, size_t);
extern void dgemm_(const char *ta, const char *tb, const int *m, const int *n, const int *k,
                   const double *alpha, const double *a, const int *lda, const double *b,
                   const int *ldb, const double *beta, double *c, const int *ldc, size_t,
                   size_t);

#define ONL_TOL 1e-9
#define HOM_EPS 1e-10
#define SEED_NZ 1e-9
#define MAX_EVENTS 100000
#define MAX_ITER 100000

static int g_err = 0; /* first error code seen (0 = none) */
static long g_singular = 0; /* solves that fell back to least squares */

/* ------------------------------------------------------------------ linear algebra */
/* solve M x = B (M n x n symmetric or general, row-major == column-major only when
 * symmetric; here M is always a principal submatrix of a symmetric matrix);
 * B column-major n x nrhs, overwritten with the solution.  numpy.linalg.solve, and the
 * lstsq fallback of reclasso_har._solve on an exactly singular M. */
static void solve_np(int n, const double *M, double *B, int nrhs) {
    if (n == 0) return;
    double *A = malloc(sizeof(double) * (size_t)n * n);
    int *ipiv = malloc(sizeof(int) * n);
    memcpy(A, M, sizeof(double) * (size_t)n * n);
    double *B0 = malloc(sizeof(double) * (size_t)n * nrhs);
    memcpy(B0, B, sizeof(double) * (size_t)n * nrhs);
    int info = 0;
    dgesv_(&n, &nrhs, A, &n, ipiv, B, &n, &info);
    if (info > 0) { /* exactly singular: minimum-norm least squares, numpy rcond */
        g_singular++;
        memcpy(A, M, sizeof(double) * (size_t)n * n);
        memcpy(B, B0, sizeof(double) * (size_t)n * nrhs);
        double rcond = DBL_EPSILON * n, wq;
        double *s = malloc(sizeof(double) * n);
        int rank, lwork = -1, iwq, inf2 = 0;
        dgelsd_(&n, &n, &nrhs, A, &n, B, &n, s, &rcond, &rank, &wq, &lwork, &iwq, &inf2);
        lwork = (int)wq;
        double *work = malloc(sizeof(double) * lwork);
        int *iwork = malloc(sizeof(int) * (iwq > 1 ? iwq : 1));
        dgelsd_(&n, &n, &nrhs, A, &n, B, &n, s, &rcond, &rank, work, &lwork, iwork, &inf2);
        if (inf2 != 0 && !g_err) g_err = 10;
        free(work); free(iwork); free(s);
    } else if (info < 0 && !g_err) {
        g_err = 11;
    }
    free(A); free(ipiv); free(B0);
}

/* G = Z'Z (full), Z row-major n x m (column-major m x n, lda m) */
static void gram(int n, int m, const double *Z, double *G) {
    const double one = 1.0, zero = 0.0;
    dsyrk_("U", "N", &m, &n, &one, Z, &m, &zero, G, &m, 1, 1);
    for (int j = 0; j < m; j++)
        for (int i = 0; i < j; i++) G[(size_t)i * m + j] = G[(size_t)j * m + i];
}

static void xty(int n, int m, const double *Z, const double *y, double *c) {
    memset(c, 0, sizeof(double) * m);
    for (int i = 0; i < n; i++) {
        const double *z = Z + (size_t)i * m;
        double yi = y[i];
        for (int j = 0; j < m; j++) c[j] += z[j] * yi;
    }
}

/* ------------------------------------------------------------------ batch homotopy */
/* reclasso_har._homotopy_walk + lasso_homotopy: theta at mu_target */
static void lasso_homotopy(int m, const double *g, const double *c, double mu_target,
                           double *theta) {
    memset(theta, 0, sizeof(double) * m);
    double mu_max = 0.0;
    int amax = 0;
    for (int j = 0; j < m; j++)
        if (fabs(c[j]) > mu_max) { mu_max = fabs(c[j]); amax = j; }
    if (mu_max <= mu_target + HOM_EPS) return;
    int *a = malloc(sizeof(int) * m), *inact = malloc(sizeof(int) * m);
    double *s = malloc(sizeof(double) * m);
    unsigned char *in = calloc(m, 1);
    double *M = malloc(sizeof(double) * (size_t)m * m);
    double *w01 = malloc(sizeof(double) * 2 * m);
    double *pv = malloc(sizeof(double) * m), *qv = malloc(sizeof(double) * m);
    int na = 1;
    a[0] = amax;
    s[0] = (c[amax] > 0) - (c[amax] < 0);
    in[amax] = 1;
    double mu = mu_max;
    int just_dropped = -1;
    for (int it = 0; it < MAX_ITER; it++) {
        for (int i = 0; i < na; i++) {
            const double *gi = g + (size_t)a[i] * m;
            for (int k = 0; k < na; k++) M[(size_t)i * na + k] = gi[a[k]];
            w01[i] = c[a[i]];
            w01[na + i] = s[i];
        }
        solve_np(na, M, w01, 2);
        const double *w0 = w01, *w1 = w01 + na;
        int ni = 0;
        for (int j = 0; j < m; j++)
            if (!in[j]) inact[ni++] = j;
        double best_mu = mu_target;
        int ev = 0, ev_i = -1, ev_j = -1; /* 0 none, 1 drop, 2 enter */
        double ev_s = 0.0;
        for (int i = 0; i < na; i++) {
            double mj = w0[i] / w1[i];
            if (isfinite(mj) && best_mu + HOM_EPS < mj && mj < mu - HOM_EPS) {
                best_mu = mj; ev = 1; ev_i = i;
            }
        }
        if (ni) {
            for (int k = 0; k < ni; k++) {
                const double *gk = g + (size_t)inact[k] * m;
                double s0 = 0.0, s1 = 0.0;
                for (int i = 0; i < na; i++) { s0 += gk[a[i]] * w0[i]; s1 += gk[a[i]] * w1[i]; }
                pv[k] = c[inact[k]] - s0;
                qv[k] = s1;
            }
            for (int pass = 0; pass < 2; pass++) {
                double sign = pass == 0 ? 1.0 : -1.0;
                for (int k = 0; k < ni; k++) {
                    int j = inact[k];
                    if (j == just_dropped) continue;
                    double me = pass == 0 ? pv[k] / (1.0 - qv[k]) : -pv[k] / (1.0 + qv[k]);
                    if (isfinite(me) && best_mu + HOM_EPS < me && me < mu - HOM_EPS) {
                        best_mu = me; ev = 2; ev_j = j; ev_s = sign;
                    }
                }
            }
        }
        if (ev == 0) { /* last segment: theta_a = w0 - mu_target w1 */
            for (int i = 0; i < na; i++) theta[a[i]] = w0[i] - mu_target * w1[i];
            break;
        }
        mu = best_mu;
        if (ev == 1) {
            just_dropped = a[ev_i];
            in[a[ev_i]] = 0;
            for (int i = ev_i; i < na - 1; i++) { a[i] = a[i + 1]; s[i] = s[i + 1]; }
            na--;
        } else {
            a[na] = ev_j;
            s[na] = ev_s;
            in[ev_j] = 1;
            na++;
            just_dropped = -1;
        }
        if (it == MAX_ITER - 1 && !g_err) g_err = 20;
    }
    free(a); free(inact); free(s); free(in); free(M); free(w01); free(pv); free(qv);
}

/* enet_coef with penalty factors: pf all 1 -> exactly enet_coef */
static void enet_coef_pf(int m, const double *g, const double *c, int n, double alpha,
                         double l1, const double *pf, double *out) {
    double lam2 = n * alpha * (1.0 - l1), mu = n * alpha * l1;
    int allone = 1;
    for (int j = 0; j < m; j++)
        if (pf[j] != 1.0) { allone = 0; break; }
    double *gr = malloc(sizeof(double) * (size_t)m * m);
    double *cs = malloc(sizeof(double) * (size_t)(m > 0 ? m : 1));
    if (allone) {
        memcpy(gr, g, sizeof(double) * (size_t)m * m);
        if (lam2 > 0.0)
            for (int j = 0; j < m; j++) gr[(size_t)j * m + j] += lam2;
        lasso_homotopy(m, gr, c, mu, out);
    } else {
        for (int i = 0; i < m; i++) {
            double ii = 1.0 / pf[i];
            for (int k = 0; k < m; k++) gr[(size_t)i * m + k] = g[(size_t)i * m + k] * (ii * (1.0 / pf[k]));
            gr[(size_t)i * m + i] += lam2 * ii;
            cs[i] = c[i] * ii;
        }
        lasso_homotopy(m, gr, cs, mu, out);
        for (int i = 0; i < m; i++) out[i] *= 1.0 / pf[i];
    }
    free(gr); free(cs);
}

/* FWL quantities of _batch_theta that do not depend on alpha */
typedef struct {
    int n, m, nl, ne;
    int *L, *E;          /* locked / other column indices */
    double *bHy, *bHE;   /* nl, nl x ne (row-major) */
    double *gE, *cE;     /* ne x ne, ne */
} FWL;

static void fwl_free(FWL *F) { free(F->L); free(F->E); free(F->bHy); free(F->bHE); free(F->gE); free(F->cE); }

/* numpy.linalg.pinv(Hh) (nl x n), Hh = Xa[:, locked] */
static void fwl_build(FWL *F, int n, int m, const double *Xa, const double *yr,
                      const unsigned char *locked) {
    F->n = n; F->m = m;
    F->L = malloc(sizeof(int) * m); F->E = malloc(sizeof(int) * m);
    int nl = 0, ne = 0;
    for (int j = 0; j < m; j++) { if (locked[j]) F->L[nl++] = j; else F->E[ne++] = j; }
    F->nl = nl; F->ne = ne;
    /* Hh column-major n x nl */
    double *H = malloc(sizeof(double) * (size_t)n * nl);
    for (int l = 0; l < nl; l++)
        for (int i = 0; i < n; i++) H[(size_t)l * n + i] = Xa[(size_t)i * m + F->L[l]];
    int k = nl < n ? nl : n;
    double *sv = malloc(sizeof(double) * k), *U = malloc(sizeof(double) * (size_t)n * k);
    double *VT = malloc(sizeof(double) * (size_t)k * nl);
    int lwork = -1, info = 0;
    int *iwork = malloc(sizeof(int) * 8 * k);
    double wq;
    double *Hc = malloc(sizeof(double) * (size_t)n * nl);
    memcpy(Hc, H, sizeof(double) * (size_t)n * nl);
    dgesdd_("S", &n, &nl, Hc, &n, sv, U, &n, VT, &k, &wq, &lwork, iwork, &info, 1);
    lwork = (int)wq;
    double *work = malloc(sizeof(double) * lwork);
    dgesdd_("S", &n, &nl, Hc, &n, sv, U, &n, VT, &k, work, &lwork, iwork, &info, 1);
    if (info != 0 && !g_err) g_err = 30;
    double smax = 0.0;
    for (int i = 0; i < k; i++) if (sv[i] > smax) smax = sv[i];
    double cutoff = 1e-15 * smax;
    /* Hp = V diag(1/s) U'  (nl x n, row-major) */
    double *Hp = calloc((size_t)nl * n, sizeof(double));
    for (int q = 0; q < k; q++) {
        if (!(sv[q] > cutoff)) continue;
        double inv = 1.0 / sv[q];
        for (int l = 0; l < nl; l++) { /* vt' (s u'): numpy's association */
            double v = VT[(size_t)l * k + q]; /* VT(q, l) column-major */
            double *hp = Hp + (size_t)l * n;
            const double *uq = U + (size_t)q * n;
            for (int i = 0; i < n; i++) hp[i] += v * (inv * uq[i]);
        }
    }
    F->bHy = calloc(nl ? nl : 1, sizeof(double));
    F->bHE = calloc((size_t)(nl ? nl : 1) * (ne ? ne : 1), sizeof(double));
    for (int l = 0; l < nl; l++) {
        const double *hp = Hp + (size_t)l * n;
        double s = 0.0;
        for (int i = 0; i < n; i++) s += hp[i] * yr[i];
        F->bHy[l] = s;
        double *bh = F->bHE + (size_t)l * ne;
        for (int i = 0; i < n; i++) {
            const double *x = Xa + (size_t)i * m;
            double h = hp[i];
            for (int e = 0; e < ne; e++) bh[e] += h * x[F->E[e]];
        }
    }
    /* Eres = Ee - Hh bHE (row-major n x ne), yres = yr - Hh bHy */
    double *Er = malloc(sizeof(double) * (size_t)n * (ne ? ne : 1));
    double *yres = malloc(sizeof(double) * n);
    for (int i = 0; i < n; i++) {
        const double *x = Xa + (size_t)i * m;
        double *er = Er + (size_t)i * ne;
        for (int e = 0; e < ne; e++) { /* Ee - (Hh bHE) */
            double sh = 0.0;
            for (int l = 0; l < nl; l++) sh += x[F->L[l]] * F->bHE[(size_t)l * ne + e];
            er[e] = x[F->E[e]] - sh;
        }
        double sy = 0.0;
        for (int l = 0; l < nl; l++) sy += x[F->L[l]] * F->bHy[l];
        yres[i] = yr[i] - sy;
    }
    F->gE = malloc(sizeof(double) * (size_t)(ne ? ne : 1) * (ne ? ne : 1));
    F->cE = malloc(sizeof(double) * (ne ? ne : 1));
    if (ne) { gram(n, ne, Er, F->gE); xty(n, ne, Er, yres, F->cE); }
    free(H); free(sv); free(U); free(VT); free(iwork); free(Hc); free(work); free(Hp);
    free(Er); free(yres);
}

/* _batch_theta from the FWL quantities */
static void fwl_theta(const FWL *F, double alpha, double l1, const double *pf_full,
                      double *th) {
    int ne = F->ne, nl = F->nl;
    double *pfE = malloc(sizeof(double) * (ne ? ne : 1)), *bE = malloc(sizeof(double) * (ne ? ne : 1));
    for (int e = 0; e < ne; e++) pfE[e] = pf_full[F->E[e]];
    enet_coef_pf(ne, F->gE, F->cE, F->n, alpha, l1, pfE, bE);
    memset(th, 0, sizeof(double) * F->m);
    for (int l = 0; l < nl; l++) {
        const double *bh = F->bHE + (size_t)l * ne;
        double s = 0.0;
        for (int e = 0; e < ne; e++) s += bh[e] * bE[e];
        th[F->L[l]] = F->bHy[l] - s;
    }
    for (int e = 0; e < ne; e++) th[F->E[e]] = bE[e];
    free(pfE); free(bE);
}

/* ------------------------------------------------------------------ online homotopy */
/* reclasso_har.enet_online: one rank-1 data change; A, s, nA warm; theta out */
static int enet_online(int m, double *Gr, double *c, const double *mu_vec, const double *u,
                       double w, double sigma, double *theta, int *A, double *s, int *nA,
                       const unsigned char *locked, unsigned char *inA, double *M,
                       double *tp, double *r, double *dvec, int *inact) {
    double dcur = 0.0;
    int n_ev = 0;
    while (dcur < 1.0 - ONL_TOL) {
        int na = *nA;
        for (int i = 0; i < na; i++) {
            const double *gi = Gr + (size_t)A[i] * m;
            for (int k = 0; k < na; k++) M[(size_t)i * na + k] = gi[A[k]];
            tp[i] = c[A[i]] - mu_vec[A[i]] * s[i];
            tp[na + i] = u[A[i]];
        }
        solve_np(na, M, tp, 2);
        const double *tA = tp, *p = tp + na;
        double a = 0.0, e0 = 0.0;
        for (int i = 0; i < na; i++) { a += u[A[i]] * p[i]; e0 += u[A[i]] * tA[i]; }
        double dir = w - e0, drem = 1.0 - dcur;
        int ni = 0;
        for (int j = 0; j < m; j++)
            if (!inA[j]) inact[ni++] = j;
        for (int k = 0; k < ni; k++) { /* r = c - Gr theta_full ; dvec = Gr[inact, A] p */
            const double *gk = Gr + (size_t)inact[k] * m;
            double sr = 0.0, sd = 0.0;
            for (int i = 0; i < na; i++) { sr += gk[A[i]] * tA[i]; sd += gk[A[i]] * p[i]; }
            r[k] = c[inact[k]] - sr;
            dvec[k] = sd;
        }
        double best_d = drem;
        int ev = 0, ev_li = -1, ev_j = -1;
        double ev_s = 0.0;
        for (int li = 0; li < na; li++) {
            int j = A[li];
            if (locked[j] || fabs(p[li]) <= ONL_TOL) continue;
            double gs = -tA[li] / p[li];
            double den = sigma * (dir - gs * a);
            double d = fabs(den) > ONL_TOL ? gs / den : INFINITY;
            if (ONL_TOL < d && d < best_d - ONL_TOL) { best_d = d; ev = 1; ev_li = li; }
        }
        for (int k = 0; k < ni; k++) {
            double dj = u[inact[k]] - dvec[k];
            if (fabs(dj) <= ONL_TOL) continue;
            int j = inact[k];
            for (int tg = 0; tg < 2; tg++) {
                double target = tg == 0 ? mu_vec[j] : -mu_vec[j];
                double gs = (target - r[k]) / dj;
                double den = sigma * (dir - gs * a);
                double d = fabs(den) > ONL_TOL ? gs / den : INFINITY;
                if (ONL_TOL < d && d < best_d - ONL_TOL) {
                    best_d = d; ev = 2; ev_j = j; ev_s = target > 0 ? 1.0 : -1.0;
                }
            }
        }
        double gstep = best_d * sigma * dir / (1.0 + best_d * sigma * a);
        memset(theta, 0, sizeof(double) * m);
        for (int i = 0; i < na; i++) theta[A[i]] = tA[i] + gstep * p[i];
        double sc = best_d * sigma;
        for (int i = 0; i < m; i++) {
            double ui = u[i];
            if (ui == 0.0) continue; /* outer(u, u) row of zeros adds exactly zero */
            double *gi = Gr + (size_t)i * m;
            for (int k = 0; k < m; k++) gi[k] += sc * (ui * u[k]);
        }
        for (int i = 0; i < m; i++) c[i] += (sc * u[i]) * w;
        dcur += best_d;
        if (ev == 1) {
            inA[A[ev_li]] = 0;
            for (int i = ev_li; i < na - 1; i++) { A[i] = A[i + 1]; s[i] = s[i + 1]; }
            (*nA)--;
        } else if (ev == 2) {
            A[na] = ev_j; s[na] = ev_s; inA[ev_j] = 1; (*nA)++;
        }
        n_ev++;
        if (n_ev > MAX_EVENTS) { if (!g_err) g_err = 40; break; }
    }
    return n_ev;
}

/* ------------------------------------------------------------------ the model */
typedef struct {
    int m, n, est;
    double l1, alpha;
    unsigned char *locked, *maskout;
    double *pf;
    double *K, *c, *th;               /* ridge (K) and both (c, th) */
    double *Gr, *mu_vec, *s;          /* enet */
    int *A, nA;
    unsigned char *inA;
    int track;
    int *run_const, *run_eq;
    /* workspaces */
    double *Xw, *M, *tp, *r, *dvec, *ua, *ur, *Ku;
    int *inact;
} Model;

static void masked_window(const Model *md, const double *X, int r0, int n) {
    int m = md->m;
    for (int i = 0; i < n; i++) {
        const double *x = X + (size_t)(r0 + i) * m;
        double *z = md->Xw + (size_t)i * m;
        for (int j = 0; j < m; j++) z[j] = md->maskout[j] ? 0.0 : x[j];
    }
}

/* _seed on rows [r0, r0 + n) */
static void seed(Model *md, const double *X, const double *y, int r0) {
    int m = md->m, n = md->n;
    masked_window(md, X, r0, n);
    const double *yr = y + r0;
    if (md->est == 0) {
        double *G = md->M;
        gram(n, m, md->Xw, G);
        for (int j = 0; j < m; j++) G[(size_t)j * m + j] += md->locked[j] ? 0.0 : md->alpha * md->pf[j];
        /* K = inv(G): dgesv against the identity (numpy.linalg.inv) */
        for (int i = 0; i < m; i++)
            for (int k = 0; k < m; k++) md->K[(size_t)i * m + k] = i == k ? 1.0 : 0.0;
        solve_np(m, G, md->K, m); /* K column-major = row-major (symmetric inverse) */
        xty(n, m, md->Xw, yr, md->c);
        for (int i = 0; i < m; i++) {
            double s = 0.0;
            for (int k = 0; k < m; k++) s += md->K[(size_t)i * m + k] * md->c[k];
            md->th[i] = s;
        }
    } else {
        double mu = n * md->alpha * md->l1, lam2 = n * md->alpha * (1.0 - md->l1);
        for (int j = 0; j < m; j++) md->mu_vec[j] = md->locked[j] ? 0.0 : mu * md->pf[j];
        gram(n, m, md->Xw, md->Gr);
        for (int j = 0; j < m; j++)
            if (!md->locked[j]) md->Gr[(size_t)j * m + j] += lam2 * md->pf[j];
        xty(n, m, md->Xw, yr, md->c);
        FWL F;
        fwl_build(&F, n, m, md->Xw, yr, md->locked);
        fwl_theta(&F, md->alpha, md->l1, md->pf, md->th);
        fwl_free(&F);
        md->nA = 0;
        memset(md->inA, 0, m);
        for (int j = 0; j < m; j++)
            if (fabs(md->th[j]) > SEED_NZ || md->locked[j]) {
                md->A[md->nA] = j;
                md->s[md->nA] = md->locked[j] ? 0.0 : (double)((md->th[j] > 0) - (md->th[j] < 0));
                md->inA[j] = 1;
                md->nA++;
            }
    }
}

/* _init_runs on the raw window rows [r0, r0 + n) */
static void init_runs(Model *md, const double *X, int r0) {
    int m = md->m, n = md->n;
    const double *last = X + (size_t)(r0 + n - 1) * m;
    for (int j = 0; j < m; j++) {
        int k = 0;
        while (k < n && X[(size_t)(r0 + n - 1 - k) * m + j] == last[j]) k++;
        md->run_const[j] = k;
    }
    for (int j = 0; j < m; j++)
        for (int q = 0; q < m; q++) {
            int k = 0;
            while (k < n) {
                const double *x = X + (size_t)(r0 + n - 1 - k) * m;
                if (x[j] != x[q]) break;
                k++;
            }
            md->run_eq[(size_t)j * m + q] = k;
        }
}

/* _degenerate_live: advance the trackers by the entering raw row; returns #gone and
 * marks them in gone[] */
static int degenerate_live(Model *md, const double *ua, const double *prev,
                           unsigned char *gone) {
    int m = md->m, n = md->n, cnt = 0;
    for (int j = 0; j < m; j++) {
        int rc = ua[j] == prev[j] ? md->run_const[j] + 1 : 1;
        md->run_const[j] = rc < n ? rc : n;
    }
    for (int j = 0; j < m; j++) {
        int *re = md->run_eq + (size_t)j * m;
        for (int q = 0; q < m; q++) {
            int v = ua[j] == ua[q] ? re[q] + 1 : 0;
            re[q] = v < n ? v : n;
        }
    }
    for (int j = 0; j < m; j++) {
        int cst = md->run_const[j] >= n, dup = 0;
        const int *re = md->run_eq + (size_t)j * m;
        for (int q = 0; q < j; q++) if (re[q] >= n) { dup = 1; break; }
        gone[j] = (cst || dup) && !md->locked[j] && !md->maskout[j];
        cnt += gone[j];
    }
    return cnt;
}

/* ------------------------------------------------------------------ the walk-forward */
int exogpen_run(
    int N, int m, const double *X, const double *y, int W,
    int est, int mode, int track,
    int n_blocks, const int *bstart,
    int n_s, const unsigned char *lk, const unsigned char *mk, const double *pf,
    int n_alpha, const double *alphas, int val_tail, int embargo,
    const int *group, int n_groups, const double *mpr_start, int mpr_passes,
    double *pred, double *theta, int *events, double *val_mse, int *choice,
    double *pen_g, double *mpr_mse, int *n_reseed, long *n_singular) {
    (void)N;
    Model md;
    md.m = m; md.n = W; md.est = est; md.l1 = est == 1 ? 1.0 : 0.5;
    md.locked = malloc(m); md.maskout = malloc(m); md.pf = malloc(sizeof(double) * m);
    md.K = malloc(sizeof(double) * (size_t)m * m);
    md.c = malloc(sizeof(double) * m); md.th = malloc(sizeof(double) * m);
    md.Gr = malloc(sizeof(double) * (size_t)m * m);
    md.mu_vec = malloc(sizeof(double) * m); md.s = malloc(sizeof(double) * m);
    md.A = malloc(sizeof(int) * m); md.inA = malloc(m);
    md.track = track;
    md.run_const = malloc(sizeof(int) * m);
    md.run_eq = track ? malloc(sizeof(int) * (size_t)m * m) : NULL;
    md.Xw = malloc(sizeof(double) * (size_t)W * m);
    md.M = malloc(sizeof(double) * (size_t)m * m);
    md.tp = malloc(sizeof(double) * 2 * m);
    md.r = malloc(sizeof(double) * m); md.dvec = malloc(sizeof(double) * m);
    md.ua = malloc(sizeof(double) * m); md.ur = malloc(sizeof(double) * m);
    md.Ku = malloc(sizeof(double) * m);
    md.inact = malloc(sizeof(int) * m);
    double *Gf = malloc(sizeof(double) * (size_t)m * m), *cf = malloc(sizeof(double) * m);
    double *G2 = malloc(sizeof(double) * (size_t)m * m), *thc = malloc(sizeof(double) * m);
    double *cur = malloc(sizeof(double) * (n_groups + 1));
    double *prev = malloc(sizeof(double) * m);
    unsigned char *gone = malloc(m);
    g_err = 0;
    g_singular = 0;
    int fit_n = W - val_tail - embargo;
    for (int blk = 0; blk < n_blocks; blk++) {
        int i0 = bstart[blk], i1 = bstart[blk + 1];
        n_reseed[blk] = 0;
        /* ---- _tune on the window [i0, i0 + W) */
        int s_best = 0, a_best = 0;
        double best = INFINITY;
        for (int sidx = 0; sidx < (mode == 1 ? 1 : n_s); sidx++) {
            const unsigned char *L = lk + ((size_t)blk * n_s + sidx) * m;
            const unsigned char *Mk = mk + ((size_t)blk * n_s + sidx) * m;
            const double *P = pf + ((size_t)blk * n_s + sidx) * m;
            memcpy(md.maskout, Mk, m);
            masked_window(&md, X, i0, W);
            const double *Xf = md.Xw, *Xv = md.Xw + (size_t)(W - val_tail) * m;
            const double *yf = y + i0, *yv = y + i0 + W - val_tail;
            FWL F;
            int have_fwl = 0;
            if (est == 0) { gram(fit_n, m, Xf, Gf); xty(fit_n, m, Xf, yf, cf); }
            else { fwl_build(&F, fit_n, m, Xf, yf, L); have_fwl = 1; }
            if (mode == 0) {
                for (int ia = 0; ia < n_alpha; ia++) {
                    double a = alphas[ia];
                    if (est == 0) {
                        memcpy(G2, Gf, sizeof(double) * (size_t)m * m);
                        for (int j = 0; j < m; j++) G2[(size_t)j * m + j] += L[j] ? 0.0 : a * P[j];
                        memcpy(thc, cf, sizeof(double) * m);
                        solve_np(m, G2, thc, 1);
                    } else {
                        fwl_theta(&F, a, md.l1, P, thc);
                    }
                    double sse = 0.0;
                    for (int v = 0; v < val_tail; v++) {
                        const double *x = Xv + (size_t)v * m;
                        double f = 0.0;
                        for (int j = 0; j < m; j++) f += x[j] * thc[j];
                        double e = f - yv[v];
                        sse += e * e;
                    }
                    double mse = sse / val_tail;
                    val_mse[((size_t)blk * n_s + sidx) * n_alpha + ia] = mse;
                    if (mse < best) { best = mse; s_best = sidx; a_best = ia; }
                }
            } else { /* multi-penalty ridge: cyclic coordinate search, locked backbone */
                double *pg = pen_g + (size_t)blk * n_groups;
                for (int q = 0; q < n_groups; q++) pg[q] = mpr_start[blk];
                int *present = calloc(n_groups, sizeof(int));
                for (int j = 0; j < m; j++)
                    if (!L[j] && !Mk[j] && group[j] >= 0) present[group[j]] = 1;
                for (int pass = 0; pass <= mpr_passes; pass++)
                    for (int gi = 0; gi < (pass == 0 ? 1 : n_groups); gi++) {
                        if (pass > 0 && !present[gi]) continue;
                        for (int ia = 0; ia < (pass == 0 ? 1 : n_alpha); ia++) {
                            for (int q = 0; q < n_groups; q++) cur[q] = pg[q];
                            if (pass > 0) {
                                if (alphas[ia] == pg[gi]) continue;
                                cur[gi] = alphas[ia];
                            }
                            memcpy(G2, Gf, sizeof(double) * (size_t)m * m);
                            for (int j = 0; j < m; j++)
                                G2[(size_t)j * m + j] += L[j] ? 0.0 : (group[j] >= 0 ? cur[group[j]] : P[j] * mpr_start[blk]);
                            memcpy(thc, cf, sizeof(double) * m);
                            solve_np(m, G2, thc, 1);
                            double sse = 0.0;
                            for (int v = 0; v < val_tail; v++) {
                                const double *x = Xv + (size_t)v * m;
                                double f = 0.0;
                                for (int j = 0; j < m; j++) f += x[j] * thc[j];
                                double e = f - yv[v];
                                sse += e * e;
                            }
                            double mse = sse / val_tail;
                            if (mse < best) { best = mse; for (int q = 0; q < n_groups; q++) pg[q] = cur[q]; }
                        }
                    }
                mpr_mse[blk] = best;
                free(present);
            }
            if (have_fwl) fwl_free(&F);
        }
        choice[2 * blk] = s_best;
        choice[2 * blk + 1] = mode == 1 ? -1 : a_best;
        /* ---- the chosen structure, then the cold seed on the full window */
        const unsigned char *L = lk + ((size_t)blk * n_s + s_best) * m;
        memcpy(md.locked, L, m);
        memcpy(md.maskout, mk + ((size_t)blk * n_s + s_best) * m, m);
        const double *P = pf + ((size_t)blk * n_s + s_best) * m;
        if (mode == 0) {
            md.alpha = alphas[a_best];
            memcpy(md.pf, P, sizeof(double) * m);
        } else { /* alpha 1 and the group penalties as factors */
            md.alpha = 1.0;
            for (int j = 0; j < m; j++)
                md.pf[j] = L[j] ? 0.0 : (group[j] >= 0 ? pen_g[(size_t)blk * n_groups + group[j]] : P[j] * mpr_start[blk]);
        }
        if (md.track) init_runs(&md, X, i0);
        seed(&md, X, y, i0);
        /* ---- every session: roll (j > i0), solve, predict */
        for (int j = i0; j < i1; j++) {
            int t = W + j;
            int nev = 0;
            if (j > i0) {
                const double *xin = X + (size_t)(t - 1) * m, *xout = X + (size_t)(j - 1) * m;
                double yin = y[t - 1], yout = y[j - 1];
                int reseeded = 0;
                if (md.track) {
                    memcpy(prev, X + (size_t)(t - 2) * m, sizeof(double) * m);
                    int ng = degenerate_live(&md, xin, prev, gone);
                    if (ng && md.est != 0 && md.l1 == 1.0) {
                        for (int q = 0; q < m; q++) md.maskout[q] |= gone[q];
                        n_reseed[blk]++;
                        seed(&md, X, y, j);
                        reseeded = 1;
                    }
                }
                if (!reseeded) {
                    for (int q = 0; q < m; q++) {
                        md.ua[q] = md.maskout[q] ? 0.0 : xin[q];
                        md.ur[q] = md.maskout[q] ? 0.0 : xout[q];
                    }
                    if (md.est == 0) { /* Sherman-Morrison: add entering, drop leaving */
                        double *Ku = md.Ku;
                        for (int pass = 0; pass < 2; pass++) {
                            const double *uu = pass == 0 ? md.ua : md.ur;
                            double sgn = pass == 0 ? -1.0 : 1.0;
                            double den = 0.0;
                            for (int i = 0; i < m; i++) {
                                const double *Ki = md.K + (size_t)i * m;
                                double s = 0.0;
                                for (int k = 0; k < m; k++) s += Ki[k] * uu[k];
                                Ku[i] = s;
                            }
                            for (int i = 0; i < m; i++) den += uu[i] * Ku[i];
                            den = pass == 0 ? 1.0 + den : 1.0 - den;
                            for (int i = 0; i < m; i++) {
                                double *Ki = md.K + (size_t)i * m;
                                double ki = Ku[i];
                                for (int k = 0; k < m; k++) Ki[k] += sgn * ((ki * Ku[k]) / den);
                            }
                            double yy = pass == 0 ? yin : yout;
                            for (int i = 0; i < m; i++) md.c[i] += (pass == 0 ? 1.0 : -1.0) * (uu[i] * yy);
                        }
                    } else {
                        nev += enet_online(m, md.Gr, md.c, md.mu_vec, md.ua, yin, 1.0, md.th,
                                           md.A, md.s, &md.nA, md.locked, md.inA, md.M, md.tp,
                                           md.r, md.dvec, md.inact);
                        nev += enet_online(m, md.Gr, md.c, md.mu_vec, md.ur, yout, -1.0, md.th,
                                           md.A, md.s, &md.nA, md.locked, md.inA, md.M, md.tp,
                                           md.r, md.dvec, md.inact);
                    }
                }
            }
            if (md.est == 0) { /* solve: theta = K c */
                for (int i = 0; i < m; i++) {
                    const double *Ki = md.K + (size_t)i * m;
                    double s = 0.0;
                    for (int k = 0; k < m; k++) s += Ki[k] * md.c[k];
                    md.th[i] = s;
                }
            }
            const double *xt = X + (size_t)t * m;
            double f = 0.0;
            for (int k = 0; k < m; k++) f += xt[k] * md.th[k];
            pred[j] = f;
            memcpy(theta + (size_t)j * m, md.th, sizeof(double) * m);
            events[j] = nev;
            if (g_err) goto out;
        }
    }
out:
    *n_singular = g_singular;
    free(md.locked); free(md.maskout); free(md.pf); free(md.K); free(md.c); free(md.th);
    free(md.Gr); free(md.mu_vec); free(md.s); free(md.A); free(md.inA); free(md.run_const);
    free(md.run_eq); free(md.Xw); free(md.M); free(md.tp); free(md.r); free(md.dvec);
    free(md.ua); free(md.ur); free(md.Ku); free(md.inact);
    free(Gf); free(cf); free(G2); free(thc); free(cur); free(prev); free(gone);
    return g_err;
}
