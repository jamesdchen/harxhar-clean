
"""GARCH(1,1) second moment, refined: finer causal grid, paired vs EWMA,
both benchmark and champion. Yearly refits; (A,B) selected on previous
year's scored QLIKE; omega = (1-A-B)*mean(e2) [unconditional mean]."""
import os, sys
import numpy as np
sys.path.insert(0,"/u/scratch/j/jamesdc1/harxhar-clean")
sys.path.insert(0,"/u/scratch/j/jamesdc1/harxhar-clean/experiments")
import score_unification as su

def series(ARM):
    ROOT="/u/scratch/j/jamesdc1/harxhar-clean/results/unification"
    files=sorted(f for f in os.listdir(os.path.join(ROOT,ARM)) if su._CHUNK_RE.match(f))
    chunks=[su._load_chunk(os.path.join(ROOT,ARM,f)) for f in files]
    prev=su._burnin_state(chunks[0])
    E=[];M=[];V=[]
    for c in chunks[1:]:
        yhat=c["yhat"]; y_raw=su._y_raw_of(c["rv_raw"],c["baseline"]); valid=c["valid"]
        s=prev["valid"]&np.isfinite(prev["y_raw"])&np.isfinite(prev["yhat"])
        a,b=0.0,1.0
        if s.sum()>=3:
            f=su._ols2(prev["yhat"][s],prev["y_raw"][s])
            if f is not None: a,b=f
        m=a+b*yhat
        E.append(y_raw-m); M.append(m); V.append(valid)
        prev={"yhat":yhat,"y_raw":y_raw,"m":m,"valid":valid}
    e=np.concatenate(E); m=np.concatenate(M); v=np.concatenate(V)
    ok=v&np.isfinite(e)
    return e[ok],m[ok]

def ql(y,m,v,idx):
    f=m[idx]**2+v[idx]
    r=(y[idx]**2)/f
    ok=(y[idx]**2>0)&(r>0)&np.isfinite(r)
    l=np.full(len(idx),np.nan)
    l[ok]=r[ok]-np.log(r[ok])-1.0
    return l

GRID=[(A,B) for A in (0.02,0.05,0.1,0.15,0.2,0.3) for B in (0.2,0.4,0.6,0.8,0.9)]

def run(ARM):
    e,m=series(ARM); y=e+m; e2=e**2; N=len(e)
    BPY=N//24; bounds=[(i*BPY,min((i+1)*BPY,N)) for i in range(24)]
    lam=0.6
    s2=np.empty(N); cur=np.nanmean(e2[:bounds[0][1]])
    for i in range(N):
        s2[i]=cur; cur=lam*cur+(1-lam)*e2[i]
    vg=np.full(N,np.nan); picks=[]
    for k in range(1,24):
        tr=np.arange(bounds[k-1][0],bounds[k-1][1]); te=np.arange(bounds[k][0],bounds[k][1])
        mu=e2[tr].mean(); best=None
        for A,B in GRID:
            om=mu*(1-A-B)
            v=np.empty(len(tr)); v[0]=mu
            for i in range(1,len(tr)): v[i]=om+A*e2[tr[i-1]]+B*v[i-1]
            loc=np.arange(len(tr))
            l=ql(y[tr],m[tr],v,loc)
            q=float(np.mean(l[np.isfinite(l)]))
            if best is None or q<best[0]: best=(q,A,B,om)
        _,A,B,om=best; picks.append((A,B))
        v=np.empty(len(te)); v[0]=om+A*e2[tr[-1]]+B*mu
        for i in range(1,len(te)): v[i]=om+A*e2[te[i-1]]+B*v[i-1]
        vg[te]=v
    scored=np.concatenate([np.arange(a,b) for a,b in bounds[1:]])
    l_e=ql(y,m,s2,scored); l_g=ql(y,m,vg,scored)
    ok=np.isfinite(l_e)&np.isfinite(l_g)
    d=l_g[ok]-l_e[ok]
    dm=su.dm_test(l_g[ok],l_e[ok],h=1)
    print(f"\n=== {ARM} (N={N}) ===")
    print(f"ewma  {np.mean(l_e[ok]):.5f}")
    print(f"garch {np.mean(l_g[ok]):.5f}  paired {np.mean(d):+.5f}  DM {dm['dm']:.2f}")
    print("picks (A,B):",picks)
    l_e_full=ql(y,m,s2,scored); l_g_full=ql(y,m,vg,scored)
    yrs=np.concatenate([np.full(b-a,i+1) for i,(a,b) in enumerate(bounds[1:])])
    print("by-year paired DQ:")
    for yy in sorted(set(yrs)):
        mm=(yrs==yy)
        dd=(l_g_full-l_e_full)[mm&np.isfinite(l_g_full-l_e_full)]
        print(f"  {2000+yy}: {np.mean(dd):+.5f}")

run("a0_ols_har")
run("blk4_prodBbExogTrailSd")
