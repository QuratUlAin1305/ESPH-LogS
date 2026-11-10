"""Featurise the 2,010-compound log S set.

Outputs (in OUT):
  data.csv                 id, canonical SMILES, logS, Zang/OPERA split flag
  topo_seed{s}.npz         PH (24), ESPH (144) and three random-subset controls (3 x 144)
                           for conformer seed s
  fp.npz                   MACCS (167), atom pair (2048), torsion (2048), ECFP4 (2048)
  ref.csv                  RDKit 2D descriptor set (incl. Crippen clogP, heavy atoms, MW)
Usage: python featurise.py <phys_ws.csv> <outdir> [seeds...]
"""
import sys, os, warnings
import numpy as np, pandas as pd
from multiprocessing import Pool
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, MACCSkeys, rdMolDescriptors, Descriptors, DataStructs
from ripser import ripser

warnings.filterwarnings("ignore")
RDLogger.DisableLog("rdApp.*")

MAXDIM = 2
SELECTIONS = {               # name -> atomic numbers (None = all heavy atoms)
    "C": {6}, "NO": {7, 8}, "HAL": {9, 17, 35, 53},
    "CNO": {6, 7, 8}, "HEAVY": None, "POLAR": {7, 8, 9, 15, 16, 17, 35, 53},
}
STATS = ["count", "sum", "max", "mean", "std", "median", "l2", "sqsum"]
N_RANDOM = 3


def diagram_stats(dgm):
    d = np.asarray(dgm, dtype=float).reshape(-1, 2)
    d = d[np.isfinite(d[:, 1])]            # the single infinite H0 bar is discarded
    if len(d) == 0:
        return [0.0] * len(STATS)
    p = d[:, 1] - d[:, 0]
    return [float(len(p)), float(p.sum()), float(p.max()), float(p.mean()),
            float(p.std()), float(np.median(p)), float(np.sqrt((p ** 2).sum())),
            float((p ** 2).sum())]


def cloud_features(points):
    if len(points) < 2:
        return [0.0] * (len(STATS) * (MAXDIM + 1))
    dgms = ripser(points, maxdim=MAXDIM)["dgms"]
    out = []
    for k in range(MAXDIM + 1):
        out += diagram_stats(dgms[k] if k < len(dgms) else np.zeros((0, 2)))
    return out


def conformer(smiles, seed):
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    p = AllChem.ETKDGv3(); p.randomSeed = int(seed)
    if AllChem.EmbedMolecule(mol, p) != 0:
        p.useRandomCoords = True
        if AllChem.EmbedMolecule(mol, p) != 0:
            return None, None
    try:
        if AllChem.MMFFHasAllMoleculeParams(mol):
            AllChem.MMFFOptimizeMolecule(mol, maxIters=500)
        else:
            AllChem.UFFOptimizeMolecule(mol, maxIters=500)
    except Exception:
        pass
    pos = mol.GetConformer().GetPositions().astype(np.float64)
    Z = np.array([a.GetAtomicNum() for a in mol.GetAtoms()])
    return pos, Z


def topo(args):
    i, smiles, seed = args
    pos, Z = conformer(smiles, seed)
    if pos is None:
        return i, None
    ph = cloud_features(pos)
    esph, sizes = [], []
    heavy = np.where(Z > 1)[0]
    for sel in SELECTIONS.values():
        idx = heavy if sel is None else np.where(np.isin(Z, list(sel)))[0]
        sizes.append(len(idx))
        esph += cloud_features(pos[idx])
    # composition-blind controls: same subset sizes, atoms drawn at random from the heavy atoms
    rng = np.random.default_rng(1000 * int(seed) + i)
    rand = []
    for _ in range(N_RANDOM):
        r = []
        for n in sizes:
            idx = rng.choice(heavy, size=n, replace=False) if n <= len(heavy) else heavy
            r += cloud_features(pos[idx])
        rand.append(r)
    return i, (ph, esph, rand, len(Z), len(heavy))


def fingerprints(smiles):
    m = Chem.MolFromSmiles(smiles)
    def arr(bv, n):
        a = np.zeros((n,), dtype=np.uint8); DataStructs.ConvertToNumpyArray(bv, a); return a
    return (arr(MACCSkeys.GenMACCSKeys(m), 167),
            arr(rdMolDescriptors.GetHashedAtomPairFingerprintAsBitVect(m, nBits=2048), 2048),
            arr(rdMolDescriptors.GetHashedTopologicalTorsionFingerprintAsBitVect(m, nBits=2048), 2048),
            arr(AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048), 2048))


def feature_names():
    deg = ["H0", "H1", "H2"]
    ph = [f"PH.{d}.{s}" for d in deg for s in STATS]
    es = [f"{n}.{d}.{s}" for n in SELECTIONS for d in deg for s in STATS]
    return ph, es


if __name__ == "__main__":
    src, out = sys.argv[1], sys.argv[2]
    seeds = [int(s) for s in sys.argv[3:]] or [42]
    os.makedirs(out, exist_ok=True)
    w = pd.read_csv(src, low_memory=False)
    can = [Chem.MolToSmiles(Chem.MolFromSmiles(s)) for s in w["smilesorg"]]
    data = pd.DataFrame({"id": w["ChemID"], "smiles": can, "logS": w["logS"].astype(float),
                         "train_zang": w["Tr_1_Tst_0"].astype(int)})
    data.to_csv(f"{out}/data.csv", index=False)
    print("records", len(data), "unique canonical", data.smiles.nunique())

    if not os.path.exists(f"{out}/fp.npz"):
        fps = [fingerprints(s) for s in data.smiles]
        np.savez_compressed(f"{out}/fp.npz", maccs=np.array([f[0] for f in fps]),
                            ap=np.array([f[1] for f in fps]), tt=np.array([f[2] for f in fps]),
                            ecfp=np.array([f[3] for f in fps]))
        ref = pd.DataFrame([Descriptors.CalcMolDescriptors(Chem.MolFromSmiles(s)) for s in data.smiles])
        ref.to_csv(f"{out}/ref.csv", index=False)
        print("fingerprints and reference descriptors done")

    phn, esn = feature_names()
    for seed in seeds:
        f = f"{out}/topo_seed{seed}.npz"
        if os.path.exists(f):
            continue
        with Pool(2) as pool:
            res = dict(pool.imap_unordered(topo, [(i, s, seed) for i, s in enumerate(data.smiles)],
                                           chunksize=8))
        bad = [i for i, r in res.items() if r is None]
        print("seed", seed, "failures", bad)
        n = len(data)
        PH = np.array([res[i][0] for i in range(n)]); ES = np.array([res[i][1] for i in range(n)])
        RA = np.array([res[i][2] for i in range(n)]).transpose(1, 0, 2)
        natoms = np.array([res[i][3] for i in range(n)]); nheavy = np.array([res[i][4] for i in range(n)])
        np.savez_compressed(f, ph=PH, esph=ES, rand=RA, natoms=natoms, nheavy=nheavy,
                            ph_names=np.array(phn), esph_names=np.array(esn))
        print("seed", seed, "saved", PH.shape, ES.shape, RA.shape)
