#!/bin/bash
# Full reproduction. Expects SolCuration/ cloned next to this script.
F=feat; R=results
python featurise.py SolCuration/org/phys/phys_ws.csv $F 42 1 2 3 4
python experiments.py $F $R zang
python diagnostics.py $F $R
python experiments.py $F $R main
python experiments.py $F $R extra
python zang_extra.py $F $R
python analysis.py $R figs
python figs_diag.py $R figs
