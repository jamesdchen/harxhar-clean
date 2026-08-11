
"""Best-model shootout for the second-moment slot (a0 residuals, causal).

All variants produce v_t (the sigma2 path); scored by B-free QLIKE of
f = (m^2 + v) vs y^2, level-pinned globally on scored bars (year 1 = burn-in).

  ewma        lambda=0.6 recursion (current contract)
  iso         isotonic E[e2 | level] fit on PREVIOUS year, constant extrapolation
  pow         power law v = a * level^g, OLS on logs, previous year
  iso_ewma    isotonic mean + GARCH(1,1)-style mean-reverting deviation:
              v_t = mu(level_t) + A*(e2_{t-1} - mu(level_{t-1})) + B*(v_{t-1} - mu(level_{t-1}))
              (A,B) from a small causal grid on the previous year
  garch       plain GARCH(1,1) on e2 (omega, A, B per previous year) — nests ewma
"""
import numpy as np
from sklearn.isotonic import IsotonicRegression

e=np.load("/u/scratch/j/jamesdc1/harxhar-clean/results/a0_e.npy")
m=np.load("/u/scratch/j/jamesdc1/harxhar-clean/results/a0_m.npy")
s2=np.load("/u/scratch/j/jamesdc1/harxhar-clean/results/a0_s2ewma.npy")
y=e+m; e2=e**2; lev=m**2; N=len(e)
BPY=N//24
bounds=[(i*BPY,min((i+1)*BPY,N)) for i in range(24)]

def ql(v, idx):
    vv=v[idx]; pin=e2[idx].mean()/vv.mean()
    f=m[idx]**2+vv*pin
    r=(y[idx]**2)/f
    ok=(y[idx]**2>0)&np.isfinite(r)&(r>0)
    r=r[ok]
    return float(np.mean(r-np.log(r)-1.0))

scored=np.concatenate([np.arange(a,b) for a,b in bounds[1:]])
V={"ewma":s2.copy(),"iso":np.full(N,np.nan),"pow":np.full(N,np.nan),
   "iso_ewma":np.full(N,np.nan),"garch":np.full(N,np.nan)}

for k in range(1,24):
    tr=np.arange(bounds[k-1][0],bounds[k-1][1])
    te=np.arange(bounds[k][0],bounds[k][1])
    # isotonic on previous year
    iso=IsotonicRegression(out_of_bounds="clip")
    iso.fit(lev[tr],e2[tr])
    V["iso"][te]=iso.predict(lev[te])
    # power law
    lx=np.log(np.maximum(lev[tr],1e-14)); lz=np.log(np.maximum(e2[tr],1e-300))
    A=np.column_stack([np.ones(len(tr)),lx])
    beta,_,_,_=np.linalg.lstsq(A,lz,rcond=None)
    V["pow"][te]=np.exp(beta[0])*np.maximum(lev[te],1e-14)**beta[1]
    # iso + mean-reverting GARCH deviation, small causal grid on prev year
    mu_tr=iso.predict(lev[tr])
    best=None
    for A_ in (0.05,0.1,0.2,0.4):
        for B_ in (0.3,0.6,0.9):
            v=np.empty(len(tr)); v[0]=e2[tr].mean()
            for i in range(1,len(tr)):
                v[i]=mu_tr[i]+A_*(e2[tr[i-1]]-mu_tr[i-1])+B_*(v[i-1]-mu_tr[i-1])
            f=m[tr]**2+np.maximum(v,1e-300); r=(y[tr]**2)/f
            okq=(y[tr]**2>0)&np.isfinite(r)&(r>0)
            q=float(np.mean(r[okq]-np.log(r[okq])-1.0))
            if best is None or q<best[0]: best=(q,A_,B_)
    _,A_,B_=best
    mu_te=iso.predict(lev[te])
    v=np.empty(len(te))
    v[0]=mu_te[0]
    e_prev=e2[tr[-1]]; v_prev=(mu_tr[-1]+A_*(e2[tr[-1]]-mu_tr[-1])+B_*0.0); mup=mu_tr[-1]
    for i in range(len(te)):
        if i>0:
            v[i]=mu_te[i]+A_*(e_prev-mup)+B_*(v_prev-mup)
            v_prev=v[i]; mup=mu_te[i]; e_prev=e2[te[i-1]] if i>0 else e_prev
        if i>0: e_prev=e2[te[i-1]]
    V["iso_ewma"][te]=np.maximum(v,1e-300)
    # plain GARCH(1,1) per previous year
    best=None
    for A_ in (0.05,0.1,0.2,0.4):
        for B_ in (0.3,0.6,0.9):
            om=e2[tr].mean()*(1-A_-B_)
            v=np.empty(len(tr)); v[0]=e2[tr].mean()
            for i in range(1,len(tr)):
                v[i]=om+A_*e2[tr[i-1]]+B_*v[i-1]
            f=m[tr]**2+np.maximum(v,1e-300); r=(y[tr]**2)/f
            okq=(y[tr]**2>0)&np.isfinite(r)&(r>0)
            q=float(np.mean(r[okq]-np.log(r[okq])-1.0))
            if best is None or q<best[0]: best=(q,A_,B_,om)
    _,A_,B_,om=best
    v=np.empty(len(te)); v[0]=e2[tr].mean()
    for i in range(1,len(te)):
        v[i]=om+A_*e2[te[i-1]]+B_*v[i-1]
    V["garch"][te]=np.maximum(v,1e-300)
    print("year",k+1,"done",flush=True)

print("\n=== B-free QLIKE on scored bars (23 years) ===")
for name,v in V.items():
    print(f"  {name:9s} {ql(v,scored):.5f}")
