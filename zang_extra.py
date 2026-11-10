"""External split for the combined physicochemical + topological representations."""
import sys, os
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from experiments import load, clean, models
from sklearn.metrics import r2_score, mean_squared_error
feat, out = sys.argv[1], sys.argv[2]
d, fp, ref, t = load(feat); y = d.logS.values.astype(float); tr = d.train_zang.values
PH, ES = t[42]["ph"], t[42]["esph"]
R = {"clogP + PH": np.hstack([ref[["MolLogP"]].values, PH]), "clogP + ESPH": np.hstack([ref[["MolLogP"]].values, ES]),
     "RDKit 2D + PH": np.hstack([ref.values, PH]), "RDKit 2D + ESPH": np.hstack([ref.values, ES])}
z = pd.read_csv(f"{out}/zang_split.csv"); zp = dict(np.load(f"{out}/zang_pred.npz"))
a, b = np.where(tr == 1)[0], np.where(tr == 0)[0]
rows = []
for name, X in R.items():
    Xc = clean(X)
    for m, mod in models().items():
        mod.fit(Xc[a], y[a]); p = mod.predict(Xc[b]); zp[f"{name}|{m}"] = p
        rows.append(dict(rep=name, model=m, r2=r2_score(y[b], p), mse=mean_squared_error(y[b], p)))
    print("zang", name, flush=True)
z = pd.concat([z[~z.rep.isin(R)], pd.DataFrame(rows)])
z.to_csv(f"{out}/zang_split.csv", index=False); np.savez_compressed(f"{out}/zang_pred.npz", **zp)
