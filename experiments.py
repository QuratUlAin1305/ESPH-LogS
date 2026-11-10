"""Evaluate every representation under one protocol.

Protocol: 5-fold cross-validation repeated 3 times (RepeatedKFold, random_state=42), the same 15
partitions for every representation and learner; constant and exactly duplicated columns removed
before fitting; XGBoost, LightGBM and HistGB with the hyperparameters stated in the manuscript.
Also: the fixed 1,507/503 training/test split of Zang et al. (2017).
Usage: python experiments.py <featdir> <outdir> <group>   (group = main | extra | zang)
"""
import sys, os, time, warnings
import numpy as np, pandas as pd
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.ensemble import HistGradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

warnings.filterwarnings("ignore")
SEED = 42


def models():
    return {
        "XGBoost": XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=8, subsample=0.8,
                                colsample_bytree=0.8, tree_method="hist", random_state=SEED, n_jobs=2,
                                verbosity=0),
        "LightGBM": LGBMRegressor(n_estimators=500, learning_rate=0.04, num_leaves=63, subsample=0.8,
                                  subsample_freq=1, colsample_bytree=0.8, reg_lambda=2.0,
                                  random_state=SEED, n_jobs=2, verbose=-1),
        "HistGB": HistGradientBoostingRegressor(max_iter=500, learning_rate=0.05, max_leaf_nodes=31,
                                                l2_regularization=1.0, random_state=SEED),
    }


def clean(X):
    X = np.asarray(X, dtype=np.float64)
    keep = X.std(axis=0) > 0
    X = X[:, keep]
    _, idx = np.unique(X.round(10), axis=1, return_index=True)
    return X[:, np.sort(idx)]


def load(feat):
    d = pd.read_csv(f"{feat}/data.csv")
    fp = np.load(f"{feat}/fp.npz")
    ref = pd.read_csv(f"{feat}/ref.csv").replace([np.inf, -np.inf], np.nan)
    ref = ref.loc[:, ref.notna().all()]
    t = {s: np.load(f"{feat}/topo_seed{s}.npz") for s in (42, 1, 2, 3, 4)
         if os.path.exists(f"{feat}/topo_seed{s}.npz")}
    return d, fp, ref, t


def representations(group, fp, ref, t):
    T = t[42]
    PH, ES = T["ph"], T["esph"]
    M, AP, TT, EC = fp["maccs"], fp["ap"], fp["tt"], fp["ecfp"]
    R = {}
    if group == "main":
        R.update({
            "PH": PH, "ESPH": ES,
            "MACCS": M, "Atom pair": AP, "Torsion": TT,
            "PH + MACCS": np.hstack([PH, M]), "PH + Torsion": np.hstack([PH, TT]),
            "PH + Atom pair": np.hstack([PH, AP]),
            "ESPH + MACCS": np.hstack([ES, M]), "ESPH + Torsion": np.hstack([ES, TT]),
            "ESPH + Atom pair": np.hstack([ES, AP]),
            "clogP": ref[["MolLogP"]].values, "Size": ref[["HeavyAtomCount", "MolWt"]].values,
            "RDKit 2D": ref.values, "ECFP4": EC,
            "PH (heavy atoms)": ES[:, 4 * 24:5 * 24],
            **{f"Random subsets {r + 1}": T["rand"][r] for r in range(T["rand"].shape[0])},
        })
    elif group == "seeds_rest":
        R = {"ESPH seed 2": t[2]["esph"], "PH seed 3": t[3]["ph"], "ESPH seed 3": t[3]["esph"],
             "PH seed 4": t[4]["ph"], "ESPH seed 4": t[4]["esph"]}
    elif group == "extra":
        names = ["C", "NO", "HAL", "CNO", "HEAVY", "POLAR"]
        for j, n in enumerate(names):
            R[f"ESPH without {n}"] = np.delete(ES, np.s_[24 * j:24 * (j + 1)], axis=1)
        R["clogP + PH"] = np.hstack([ref[["MolLogP"]].values, PH])
        R["clogP + ESPH"] = np.hstack([ref[["MolLogP"]].values, ES])
        R["RDKit 2D + PH"] = np.hstack([ref.values, PH])
        R["RDKit 2D + ESPH"] = np.hstack([ref.values, ES])
        for s in (1, 2, 3, 4):
            if s in t:
                R[f"PH seed {s}"] = t[s]["ph"]; R[f"ESPH seed {s}"] = t[s]["esph"]
    return R


def run_cv(R, y, out, tag):
    rkf = list(RepeatedKFold(n_splits=5, n_repeats=3, random_state=SEED).split(y))
    rows, oof = [], {}
    for name, X in R.items():
        Xc = clean(X)
        t0 = time.time()
        for mname, _ in models().items():
            pred0 = np.zeros_like(y)
            for k, (tr, te) in enumerate(rkf):
                m = models()[mname]
                m.fit(Xc[tr], y[tr])
                p = m.predict(Xc[te])
                if k < 5:
                    pred0[te] = p
                rows.append(dict(rep=name, n_built=X.shape[1], n_used=Xc.shape[1], model=mname,
                                 repeat=k // 5, fold=k % 5, r2=r2_score(y[te], p),
                                 mse=mean_squared_error(y[te], p), n_test=len(te)))
            oof[f"{name}|{mname}"] = pred0
        print(f"{tag} {name:<22} built {X.shape[1]:>5} used {Xc.shape[1]:>5} "
              f"{time.time() - t0:6.1f}s", flush=True)
        pd.DataFrame(rows).to_csv(f"{out}/cv_{tag}.csv", index=False)
        np.savez_compressed(f"{out}/oof_{tag}.npz", **oof)


def run_zang(R, y, train, out):
    rows, preds = [], {}
    tr, te = np.where(train == 1)[0], np.where(train == 0)[0]
    for name, X in R.items():
        Xc = clean(X)
        for mname, m in models().items():
            m.fit(Xc[tr], y[tr]); p = m.predict(Xc[te])
            preds[f"{name}|{mname}"] = p
            rows.append(dict(rep=name, model=mname, r2=r2_score(y[te], p),
                             mse=mean_squared_error(y[te], p)))
        print("zang", name, flush=True)
    pd.DataFrame(rows).to_csv(f"{out}/zang_split.csv", index=False)
    np.savez_compressed(f"{out}/zang_pred.npz", y_test=y[te], **preds)


if __name__ == "__main__":
    feat, out, group = sys.argv[1], sys.argv[2], sys.argv[3]
    os.makedirs(out, exist_ok=True)
    d, fp, ref, t = load(feat)
    y = d["logS"].values.astype(float)
    if group == "zang":
        R = representations("main", fp, ref, t)
        R = {k: v for k, v in R.items() if not k.startswith("Random")}
        run_zang(R, y, d["train_zang"].values, out)
    else:
        run_cv(representations(group, fp, ref, t), y, out, group)
