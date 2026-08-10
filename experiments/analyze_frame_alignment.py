#!/usr/bin/env python3
from pathlib import Path
import json, csv
import numpy as np
import src.unification as u
W=24000; K=40
ROWS=[48868,109658,184000,250000,300053]
SESSION={"is_open","is_close","is_overnight","hour"}

def frame_cols(p,frame):
    if frame=="exog": return u._exog_all_cols(p.names)
    if frame=="prod": return u._product_base_cols(p.names)
    if frame=="prod_nohar": return np.asarray([j for j in u._product_base_cols(p.names) if u._classify(p.names[j])[0]!="har"],dtype=np.int64)
    if frame=="prod_nosession": return np.asarray([j for j in u._product_base_cols(p.names) if p.names[j] not in SESSION],dtype=np.int64)
    if frame=="prod_values": return np.asarray([j for j in u._product_base_cols(p.names) if u._classify(p.names[j])[0]=="value"],dtype=np.int64)
    if frame=="all": return np.arange(p.X.shape[1],dtype=np.int64)
    raise KeyError(frame)

def frame_map(p,frame):
    cols=frame_cols(p,"prod" if frame=="prod_blockperm" else frame)
    z=p.X[:,cols]; zw=z[W:2*W]; sd=zw.std(0); live=sd>u._DEGENERATE_SD; mu=zw[:,live].mean(0); fit=zw
    if frame=="prod_blockperm":
        fit=zw.copy(); kinds=np.asarray([u._classify(p.names[j])[0] for j in cols],dtype=object)
        groups=[np.flatnonzero(kinds=="har"),np.flatnonzero(kinds=="value"),np.asarray([i for i,j in enumerate(cols) if p.names[j] in SESSION],dtype=np.int64)]
        rng=np.random.default_rng(20260810)
        for idx in groups:
            if len(idx)>1:
                perm=rng.permutation(len(fit)); fit[:,idx]=fit[perm][:,idx]
    lam,V=np.linalg.eigh(np.corrcoef(((fit[:,live]-mu)/sd[live]),rowvar=False)); order=np.argsort(lam)[::-1][:K]
    return cols, np.flatnonzero(live), V[:,order]/sd[live,None], lam[order]

def gated_alphas():
    out={}
    for fn in sorted(Path('results/unification/blk2_gated_tuned').glob('chunk_*.npz')):
        with np.load(fn,allow_pickle=True) as z: m=json.loads(str(z['meta']))
        for r in m.get('tuned_alphas',[]): out[int(r['row'])]=r['alphas']
    return out

def nearest_alpha(alphas,row):
    key=max(k for k in alphas if k<=row); a=alphas[key]; return float(a['backbone']),float(a['exog'])

def main():
    p=u._load_panel(); cols_all=np.arange(p.X.shape[1],dtype=np.int64); n_all=len(cols_all)
    bb=set(map(int,u._backbone_cols(p.names))); ex=set(map(int,u._exog_all_cols(p.names)))
    frames=["exog","prod","prod_nohar","prod_nosession","prod_values","prod_blockperm","all"]
    maps={fr:frame_map(p,fr) for fr in frames}; alphas=gated_alphas(); rows=[]
    for row in ROWS:
        lo=row-W; X=np.asarray(p.X[lo:row],dtype=np.float64); y=np.asarray(p.y[lo:row],dtype=np.float64)
        Xc=X-X.mean(0); yc=y-y.mean(); G=Xc.T@Xc; c=Xc.T@yc
        lb,lx=nearest_alpha(alphas,row); pen=np.full(n_all,lx); pen[sorted(bb)]=lb; G[np.diag_indices_from(G)]+=pen
        beta=np.linalg.solve(G,c); beta_norm2=float(beta@beta)
        for fr,(cols,live,A,eig) in maps.items():
            b=beta[cols[live]]; h=np.sum(A*A,axis=0); proj=A@((A.T@b)/h); aligned=float(proj@b); r2=aligned/max(float(b@b),1e-300)
            raw_energy=(A.T@b)**2/h; share10=float(np.sort(raw_energy)[-10:].sum()/max(raw_energy.sum(),1e-300))
            bbmask=np.asarray([int(cols[j]) in bb for j in live]); exmask=np.asarray([int(cols[j]) in ex for j in live])
            hbb=np.sum(A[bbmask]*A[bbmask],axis=0); hex=np.sum(A[exmask]*A[exmask],axis=0)
            rows.append({"row":row,"frame":fr,"coef_alignment_r2":r2,"top10_coef_energy_share":share10,
                         "frame_eig_median":float(np.median(eig)),"bb_loading_mass":float(hbb.sum()/h.sum()),"exog_loading_mass":float(hex.sum()/h.sum()),
                         "cross_frobenius":float(np.sum(hbb*hex)),"bb_frobenius":float(np.sum(hbb*hbb)),"exog_frobenius":float(np.sum(hex*hex))})
        print('done',row,flush=True)
    with open('results/frame_alignment_diagnostics.csv','w',newline='') as f:
        w=csv.DictWriter(f,list(rows[0]));w.writeheader();w.writerows(rows)
    print('wrote results/frame_alignment_diagnostics.csv')
if __name__=='__main__': main()
