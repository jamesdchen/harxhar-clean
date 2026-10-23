"""Addendum 3 patch, part 2 (PFTuned, pretune, walk): scaled coordinates in the walk and the binomial rule's
evaluation points."""

from pathlib import Path

p = Path(r"C:\Users\james\AppData\Local\Temp\std_wt\experiments\pool_linear_pretune.py")
t = p.read_bytes().decode()


def rep(old: str, new: str, cnt: int = 1) -> None:
    global t
    assert t.count(old) == cnt, (t.count(old), old[:100])
    t = t.replace(old, new)


# ---- class attributes
rep(
    '''        ev_ref = None  # 2026-10-22: evid_ref of the column names (the evidence mask's k = 1 columns), set with bb
''',
    '''        ev_ref = None  # 2026-10-22: evid_ref of the column names (the evidence mask's k = 1 columns), set with bb
        av = None  # 2026-10-22 (addendum 3): avail_k of the column names (the binomial rule's columns), set with bb
''',
)
rep(
    '''        std_evid_reseeds: list = []

        def _std_on(self) -> bool:''',
    '''        std_evid_reseeds: list = []
        # addendum 3: (solve, [(column, kind, partner)]) of the binomial availability rule at every cold seed;
        # (solve, columns) of every binomial-rule-triggered lasso / enet reseed
        std_binom: list = []
        std_binom_reseeds: list = []

        def _std_on(self) -> bool:''',
)

# ---- the lasso / enet roll: binomial tracker + the scaled rows
rep(
    '''            if low.any():  # masked like the identifiability mask; cold reseed (weights, offsets, modes refreshed)
                self._maskout = self._maskout | low
                self._evid_only = self._evid_only | low
                PFTuned.std_evid_reseeds.append(
                    (g0 + self._n_solve, [int(j) for j in np.flatnonzero(low)])
                )
                self._seed()
                return False
            uc, rc = (
                ua - self._ctr,
                ur - self._ctr,
            )  # the centred rows (masked entries stay 0: offset 0)
''',
    '''            if low.any():  # masked like the identifiability mask; cold reseed (weights, offsets, modes refreshed)
                self._maskout = self._maskout | low
                self._evid_only = self._evid_only | low
                PFTuned.std_evid_reseeds.append(
                    (g0 + self._n_solve, [int(j) for j in np.flatnonzero(low)])
                )
                self._seed()
                return False
            # addendum 3: the binomial rule on the live availability averages, from the tracked window moments
            ai, ak = self.av
            self._av_update(ua[ai], ur[ai])
            mean, cov = av_mean_cov(self._av_st, len(self._y))
            bm, _ = binom_mask(mean, cov, ak, ~self._maskout[ai])
            if bm.any():  # masked like the identifiability mask; cold reseed
                low = np.zeros(len(self._maskout), dtype=bool)
                low[ai[bm]] = True
                self._maskout = self._maskout | low
                self._evid_only = self._evid_only | low
                PFTuned.std_binom_reseeds.append(
                    (g0 + self._n_solve, [int(j) for j in ai[bm]])
                )
                self._seed()
                return False
            # the scaled rows (addendum 3; masked entries stay 0: offset 0, scale 1)
            uc, rc = (ua - self._ctr) / self._sc, (ur - self._ctr) / self._sc
''',
)
rep(
    '''        def _roll_std_ridge(self, x_in, y_in, x_out, y_out):''',
    '''        def _av_update(self, a_in, a_out) -> None:
            """The binomial rule's tracker (addendum 3): the availability averages' sums and cross-products about the
            seed-time means, moved by the entering and leaving rows (masked design values)."""
            m, s1, C = self._av_st
            di, do = a_in - m, a_out - m
            s1 += di - do
            C += np.outer(di, di)
            C -= np.outer(do, do)

        def coef_original(self) -> np.ndarray:
            """The walk's coefficients in the original coordinates (beta = theta / s, intercept theta_0 - m' beta)."""
            b = self._th / self._sc
            b[-1] = self._th[-1] - self._ctr @ b
            return b

        def _roll_std_ridge(self, x_in, y_in, x_out, y_out):''',
)

# ---- ridge per-bar solve: binomial rule per bar + exact per-bar scaling
rep(
    '''            ev = group_evidence(self._ev_d, self.ev_ref)  # addendum 2: the k = 1 counts
            out = ~self._locked & ((s < STD_MIN) | (ev < STD_MIN_EVIDENCE))
            s = np.where(out | self._locked, 1.0, s)
            D = np.where(self._locked, 0.0, self.alpha_ * (self._pf * s * s))
            self._th = ridge_keep(self._G, self._c, D, ~out)
''',
    '''            ev = group_evidence(self._ev_d, self.ev_ref)  # addendum 2: the k = 1 counts
            out = ~self._locked & ((s < STD_MIN) | (ev < STD_MIN_EVIDENCE))
            # addendum 3: the binomial rule per bar, on the window moments of the centred state
            ai, ak = self.av
            s1a = self._sd_s1[ai] / n
            mean = self._ctr[ai] + s1a
            cov = self._G[np.ix_(ai, ai)] / n - np.outer(s1a, s1a)
            bm, _ = binom_mask(mean, cov, ak, ~out[ai])
            out[ai[bm]] = True
            rec_b = PFTuned.std_ridge[-1]
            rec_b[3] = max(rec_b[3], int(bm.sum()))
            s = np.where(out | self._locked, 1.0, s)
            # exact per-bar scaling (addendum 3): z = (x - m0) / s_t, (Z'Z + diag(a pf)) gamma = Z'y, theta = gamma / s_t
            D = np.where(self._locked, 0.0, self.alpha_ * self._pf)
            g = ridge_keep(self._G / np.outer(s, s), self._c / s, D, ~out)
            self._th = g / s
''',
)

# ---- predict in the scaled coordinates
rep(
    '''            if self._std_on():  # the centred state (addendum 2): (x - m)' theta_c
                xa = np.append(np.asarray(x, dtype=np.float64), 1.0) - self._ctr
                p = float(xa @ self._th)''',
    '''            if self._std_on():  # the scaled state (addendum 3): ((x - m) / s)' theta
                xa = (np.append(np.asarray(x, dtype=np.float64), 1.0) - self._ctr) / self._sc
                p = float(xa @ self._th)''',
)

# ---- block-end refit and the causal re-choice grid get av
rep(
    '''                self._pf,
                self.ev_ref,
            )
            REPAIR.update(rep)''',
    '''                self._pf,
                self.ev_ref,
                self.av,
            )
            REPAIR.update(rep)''',
)
rep(
    '''                        th = fit_std(
                            Xf, yf, self._locked, kind, a, l1, pf, self.ev_ref
                        )[0]''',
    '''                        th = fit_std(
                            Xf, yf, self._locked, kind, a, l1, pf, self.ev_ref, self.av
                        )[0]''',
)

# ---- the cold seed
rep(
    '''            s, out, v, ev = std_weights(Xa, self._locked, self.ev_ref, full=True)
            new = out & ~self._maskout
            nan_s = np.where(out | self._locked, np.nan, s)
            # the solve (OOS row) whose forecast first uses this seed
            at = g0 + self._n_solve - getattr(self, "_seed_off", 0)
            PFTuned.std_seeds.append((at, nan_s, int(new.sum()), np.flatnonzero(new)))
            self._sd_seed = nan_s
            self._ev_v, self._ev_d = v, ev.astype(np.int64)
            if kind == "ridge":
                self._sd_run = _trailing_const(Xa)
                self._ctr = np.where(self._locked, 0.0, Xa.mean(axis=0))
                Xa -= self._ctr  # the centred state (addendum 2); Xa is a fresh array
                self._G = Xa.T @ Xa
                self._c = Xa.T @ yr
                self._sd_s1 = Xa.sum(axis=0)
                self._sd_s2 = (Xa * Xa).sum(axis=0)
                PFTuned.std_ridge.append([g0 + self._n_solve, 0.0, 0])
                self._std_ridge_solve()
                return
            if new.any():  # masked for the fit, and (weights held) until the next seed re-evaluates them
                self._maskout = self._maskout | new
                self._evid_only = new
                Xa[:, new] = 0.0
            self._ctr = np.where(self._locked, 0.0, Xa.mean(axis=0))
            Xa -= self._ctr  # the centred state (addendum 2)
            w1, w2 = pf * s, pf * s * s
            mu, lam2 = tw * a * l1, tw * a * (1.0 - l1)
            self._mu_vec = np.where(self._locked, 0.0, mu * w1)
            Gr = Xa.T @ Xa
            ei = np.where(~self._locked)[0]
            Gr[ei, ei] += lam2 * w2[ei]
            self._Gr = Gr
            self._c = Xa.T @ yr
            th = batch_theta_w(Xa, yr, self._locked, a, l1, w1, w2)
''',
    '''            s, out, v, ev, _, brec = std_weights(
                Xa, self._locked, self.ev_ref, self.av, full=True
            )
            new = out & ~self._maskout
            nan_s = np.where(out | self._locked, np.nan, s)
            # the solve (OOS row) whose forecast first uses this seed
            at = g0 + self._n_solve - getattr(self, "_seed_off", 0)
            PFTuned.std_seeds.append((at, nan_s, int(new.sum()), np.flatnonzero(new)))
            PFTuned.std_binom.append((at, brec))
            self._sd_seed = nan_s
            self._ev_v, self._ev_d = v, ev.astype(np.int64)
            if kind == "ridge":
                self._sd_run = _trailing_const(Xa)
                self._ctr = np.where(self._locked, 0.0, Xa.mean(axis=0))
                self._sc = np.ones(len(self._ctr))  # (the per-bar scale is applied in the solve)
                Xa -= self._ctr  # the centred state (addendum 2); Xa is a fresh array
                self._G = Xa.T @ Xa
                self._c = Xa.T @ yr
                self._sd_s1 = Xa.sum(axis=0)
                self._sd_s2 = (Xa * Xa).sum(axis=0)
                PFTuned.std_ridge.append([g0 + self._n_solve, 0.0, 0, 0])
                self._std_ridge_solve()
                return
            if new.any():  # masked for the fit, and (scales held) until the next seed re-evaluates them
                self._maskout = self._maskout | new
                self._evid_only = new
                Xa[:, new] = 0.0
            # addendum 3: the binomial rule's between-seed tracker (masked design, after this seed's masks)
            self._av_st = av_stats(Xa[:, self.av[0]])
            # the scaled state (addendum 3): z = (x - m) / s, m and s of this window, held until the next seed
            self._ctr = np.where(self._locked, 0.0, Xa.mean(axis=0))
            self._sc = s  # (1 on the masked columns and the intercept)
            Xa -= self._ctr
            Xa /= self._sc
            mu, lam2 = tw * a * l1, tw * a * (1.0 - l1)
            self._mu_vec = np.where(self._locked, 0.0, mu * pf)
            Gr = Xa.T @ Xa
            ei = np.where(~self._locked)[0]
            Gr[ei, ei] += lam2 * pf[ei]
            self._Gr = Gr
            self._c = Xa.T @ yr
            th = batch_z(Xa, yr, self._locked, a, l1, pf)
''',
)

# ---- pretune: av
rep(
    '''                sw_f = std_weights(Xf, locked, ref)
                for kind, al, l1 in grid:
                    th, s_f, out_f = fit_std(
                        Xf, yf, locked, kind, al, l1, pf, ref, sw_f
                    )''',
    '''                av = avail_k(names, bb)  # addendum 3
                swf = std_weights(Xf, locked, ref, av, full=True)
                sw_f = swf[:2]
                for kind, al, l1 in grid:
                    th, s_f, out_f = fit_std(
                        Xf, yf, locked, kind, al, l1, pf, ref, av, sw_f
                    )''',
)
rep(
    '''                    std_left_out_columns=[
                        (names + ["intercept"])[j] for j in np.flatnonzero(out_f & ~mk)
                    ],''',
    '''                    std_left_out_columns=[
                        (names + ["intercept"])[j] for j in np.flatnonzero(out_f & ~mk)
                    ],
                    # addendum 3: the binomial availability rule's masks (column, kind, partner)
                    std_binom=[
                        (names[c], kind, names[q] if q >= 0 else None)
                        for c, kind, q in swf[5]
                    ],''',
)

# ---- walk: av with bb
rep(
    '''        if hasattr(Cls, "ev_ref"):  # 2026-10-22 (evidence mask, addendum 2)
            Cls.ev_ref = evid_ref(names, Cls.bb)''',
    '''        if hasattr(Cls, "ev_ref"):  # 2026-10-22 (evidence mask, addendum 2)
            Cls.ev_ref = evid_ref(names, Cls.bb)
        if hasattr(Cls, "av"):  # 2026-10-22 (binomial availability rule, addendum 3)
            Cls.av = avail_k(names, Cls.bb)''',
)
rep(
    '''            Cls.std_blockend_sec, Cls.std_evid_reseeds = [0.0], []''',
    '''            Cls.std_blockend_sec, Cls.std_evid_reseeds = [0.0], []
            Cls.std_binom, Cls.std_binom_reseeds = [], []''',
)
p.write_bytes(t.encode())
print("class patched")
