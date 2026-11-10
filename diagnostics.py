"""Feature-level diagnostics.

1. Fingerprint bit disposition and split-gain concentration (single 80/20 split, full matrices,
   XGBoost average split gain), reproducing the analysis of the bit-level section.
2. SHAP attribution for the ESPH block (XGBoost fitted on all compounds).
3. Numerical check of Proposition 2: finite H0 death times equal Euclidean MST edge lengths.
Usage: python diagnostics.py <featdir> <outdir>
"""
import sys, os, warnings
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial.distance import pdist, squareform

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from experiments import load, clean, models, SEED


def xgb():
    return models()["XGBoost"]


def disposition(X, y, name, out):
    tr, te = train_test_split(np.arange(len(y)), test_size=0.2, random_state=SEED)
    m = xgb().fit(X[tr], y[tr])
    g = m.get_booster().get_score(importance_type="gain")
    gain = np.zeros(X.shape[1])
    for k, v in g.items():
        gain[int(k[1:])] = v
    Xt = X[tr]
    prev = Xt.sum(axis=0)
    const = Xt.std(axis=0) == 0
    used = gain > 0
    cat = np.array(["unused"] * X.shape[1], dtype=object)
    cat[used] = "used"
    cat[~used & const] = "constant"
    rest = np.where(~used & ~const)[0]
    U = Xt[:, used].astype(float)
    Uc = (U - U.mean(0)) / (U.std(0) + 1e-12)
    for j in rest:
        v = Xt[:, j].astype(float); vc = (v - v.mean()) / (v.std() + 1e-12)
        r = np.abs(Uc.T @ vc) / len(v)
        if r.max() >= 0.95:
            cat[j] = "redundant"
        elif prev[j] <= 10:
            cat[j] = "rare"
    order = np.argsort(-gain)
    share = np.cumsum(gain[order]) / gain.sum()
    # direction of effect (difference in mean log S, training compounds)
    dirn = np.array([y[tr][Xt[:, j] == 1].mean() - y[tr][Xt[:, j] == 0].mean()
                     if 0 < prev[j] < len(tr) else np.nan for j in range(X.shape[1])])
    pd.DataFrame({"bit": np.arange(X.shape[1]), "gain": gain, "prevalence": prev, "category": cat,
                  "delta_logS": dirn}).to_csv(f"{out}/bits_{name}.csv", index=False)
    counts = pd.Series(cat).value_counts().to_dict()
    n_used = int(used.sum())
    res = dict(fp=name, bits=X.shape[1], **{k: counts.get(k, 0) for k in
               ["used", "constant", "redundant", "rare", "unused"]},
               top20=share[19], top40=share[39], top60=share[59],
               top200=share[199] if n_used >= 200 else 1.0,
               k50=int(np.searchsorted(share, 0.5) + 1), k80=int(np.searchsorted(share, 0.8) + 1),
               k90=int(np.searchsorted(share, 0.9) + 1))
    # refit on positive-gain bits only
    m2 = xgb().fit(X[tr][:, used], y[tr])
    from sklearn.metrics import r2_score
    res["r2_full"] = r2_score(y[te], m.predict(X[te]))
    res["r2_used_only"] = r2_score(y[te], m2.predict(X[te][:, used]))
    return res


def shap_esph(T, y, out):
    import shap
    names = np.array([str(s) for s in T["esph_names"]])
    X = T["esph"]
    keep = X.std(axis=0) > 0
    X, names = X[:, keep], names[keep]
    m = xgb().fit(X, y)
    sv = shap.TreeExplainer(m).shap_values(X)
    imp = np.abs(sv).mean(axis=0)
    corr = [np.corrcoef(X[:, j], sv[:, j])[0, 1] if X[:, j].std() > 0 else np.nan
            for j in range(X.shape[1])]
    df = pd.DataFrame({"feature": names, "mean_abs_shap": imp, "sign": corr})
    df.sort_values("mean_abs_shap", ascending=False).to_csv(f"{out}/shap_esph.csv", index=False)
    np.savez_compressed(f"{out}/shap_esph.npz", shap=sv, X=X, names=names)


def mst_check(smiles, seed=42, n=200):
    from featurise import conformer
    from ripser import ripser
    worst = 0.0
    for s in smiles[:n]:
        pos, Z = conformer(s, seed)
        if len(pos) < 2:
            continue
        d0 = ripser(pos, maxdim=0)["dgms"][0]
        d0 = np.sort(d0[np.isfinite(d0[:, 1]), 1])
        mst = np.sort(minimum_spanning_tree(squareform(pdist(pos))).data)
        if len(mst) != len(d0):
            return np.inf
        worst = max(worst, float(np.abs(mst - d0).max()))
    return worst


if __name__ == "__main__":
    feat, out = sys.argv[1], sys.argv[2]
    os.makedirs(out, exist_ok=True)
    d, fp, ref, t = load(feat)
    y = d["logS"].values.astype(float)
    rows = [disposition(fp[k].astype(np.float64), y, n, out)
            for k, n in [("maccs", "MACCS"), ("ap", "Atom pair"), ("tt", "Torsion")]]
    pd.DataFrame(rows).to_csv(f"{out}/bit_disposition.csv", index=False)
    print(pd.DataFrame(rows).T)
    shap_esph(t[42], y, out)
    print("shap done")
    print("max |MST - H0 death| over 200 molecules:", mst_check(list(d["smiles"])))
