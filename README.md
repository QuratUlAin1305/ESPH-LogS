# ESPH-LogS

Code and results for

> Qurat-ul-Ain and A.R. Nizami. Element-Specific Persistent Homology Descriptors for Predictive Modelling. Submitted to *Proceedings of the Pakistan Academy of Sciences: Part A*.

Corresponding author: Qurat-ul-Ain, Department of Mathematics, University of Central Punjab, Lahore, Pakistan (quratulain@ucp.edu.pk).

Reproduces every number, table and results figure in the manuscript.

## Data

The 2,010 water-solubility records of Zang et al. (2017), J. Chem. Inf. Model. 57(1): 36-49,
doi:10.1021/acs.jcim.6b00625. The same records, with the original 1,507/503 training/test flag
(`Tr_1_Tst_0`), are distributed as `org/phys/phys_ws.csv` in
https://github.com/Mengjintao/SolCuration (Meng et al., 2022, Sci. Data 9: 71).

    git clone https://github.com/Mengjintao/SolCuration.git

## Environment

Python 3.13; rdkit 2026.09.1, ripser 0.6.15, xgboost 3.4.1, lightgbm 4.7.0, scikit-learn 1.9.1,
shap 0.54.1, numpy, pandas, scipy, matplotlib.

    pip install rdkit ripser xgboost lightgbm scikit-learn shap pandas scipy matplotlib

## Steps

    python featurise.py SolCuration/org/phys/phys_ws.csv feat 42 1 2 3 4   # descriptors, five conformer seeds
    python experiments.py feat results zang      # external 1,507/503 split
    python diagnostics.py feat results           # fingerprint bit analysis, SHAP, MST check
    python experiments.py feat results main      # repeated CV, all main representations
    python experiments.py feat results extra     # ablations, combinations, conformer seeds
    python analysis.py results figs              # summary tables, t-tests, results figures
    python figs_diag.py results figs             # bit-level and SHAP figures

`run_all.sh` runs every step in sequence (`zang_extra.py` adds the external-split results for the clogP and RDKit combinations). The `seeds_rest` group in `experiments.py` reruns only conformer seeds 2-4; it was used because our own run of the `extra` group was interrupted, and its scores are in `results/cv_seeds_rest.csv`. A complete run of `extra` produces the same seeds and does not need it. On two CPU cores featurisation takes about one
minute per conformer seed and the full evaluation roughly two hours.

## Files

| File | Purpose |
|---|---|
| `featurise.py` | canonical SMILES, ETKDGv3 + MMFF94 conformers, PH (24) and ESPH (144) descriptors, random-subset controls, fingerprints (MACCS, atom pair, torsion, ECFP4), RDKit 2D descriptors |
| `experiments.py` | 5-fold CV repeated 3 times (same 15 partitions for every representation), XGBoost / LightGBM / HistGB with fixed hyperparameters; external split |
| `diagnostics.py` | bit disposition and split-gain concentration (XGBoost, single 80/20 split), SHAP for ESPH, numerical check that H0 deaths equal minimum-spanning-tree edge lengths |
| `analysis.py` | per-fold summaries, Nadeau-Bengio corrected resampled t-tests, results figures |
| `results/` | all per-fold scores (`cv_main.csv`, `cv_extra.csv`), `summary.csv`, `significance.csv`, `zang_split.csv`, bit tables, SHAP values |

## Licence

MIT; see `LICENSE`.
