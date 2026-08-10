import numpy as np
import src.unification as U

p = U._load_panel()
w = 24000
cols = U._value_slab_cols(p.names)
names = np.array(p.names)
z1 = np.ascontiguousarray(p.X[w : 2 * w, cols[1]])
sd = z1.std(0)
live = sd > U._DEGENERATE_SD
base = names[cols[1]]
print("live rank:", int(live.sum()), "of", len(base))
print("DEAD base quantities in frame window:")
for n, s in zip(base[~live], sd[~live]):
    print(f"  {n}  sd={s:.3e}")
