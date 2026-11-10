"""Summaries, corrected resampled t-tests and the results figures.
Usage: python analysis.py <results> <figs>
"""
import sys, os, json
import numpy as np, pandas as pd
from scipy import stats
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from figs_common import *

R, OUT = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)
cv = pd.concat([pd.read_csv(f) for f in (f"{R}/cv_main.csv", f"{R}/cv_extra.csv", f"{R}/cv_seeds_rest.csv") if os.path.exists(f)])
MODELS = ["XGBoost", "LightGBM", "HistGB"]

summ = (cv.groupby(["rep", "model"])
          .agg(r2=("r2", "mean"), r2_sd=("r2", "std"), mse=("mse", "mean"), mse_sd=("mse", "std"),
               n_built=("n_built", "first"), n_used=("n_used", "first")).reset_index())
summ.to_csv(f"{R}/summary.csv", index=False)


def get(rep, model, col="r2"):
    return float(summ[(summ.rep == rep) & (summ.model == model)][col].iloc[0])


def nb_test(a, b, model, metric="r2"):
    """Nadeau-Bengio corrected resampled t-test, paired over the 15 folds."""
    A = cv[(cv.rep == a) & (cv.model == model)].sort_values(["repeat", "fold"])[metric].values
    B = cv[(cv.rep == b) & (cv.model == model)].sort_values(["repeat", "fold"])[metric].values
    d = A - B
    J = len(d)
    t = d.mean() / np.sqrt((1 / J + 0.25) * d.var(ddof=1))
    p = 2 * stats.t.sf(abs(t), J - 1)
    return d.mean(), t, p


pairs = [("ESPH", "PH"), ("ESPH", "PH (heavy atoms)"), ("ESPH", "Random subsets 1"),
         ("ESPH", "Random subsets 2"), ("ESPH", "Random subsets 3"),
         ("PH", "clogP"), ("ESPH", "clogP"), ("ESPH", "RDKit 2D"),
         ("ESPH", "Atom pair"), ("ESPH", "MACCS"), ("ESPH", "Torsion"), ("ESPH", "ECFP4"),
         ("ESPH + MACCS", "MACCS"), ("ESPH + Atom pair", "Atom pair"), ("ESPH + Torsion", "Torsion"),
         ("ESPH + MACCS", "PH + MACCS"), ("ESPH + Atom pair", "PH + Atom pair"),
         ("ESPH + Torsion", "PH + Torsion"), ("ESPH + MACCS", "ESPH"),
         ("clogP + ESPH", "ESPH"), ("clogP + PH", "clogP"), ("clogP + ESPH", "clogP"),
         ("PH (heavy atoms)", "PH"), ("RDKit 2D + ESPH", "RDKit 2D"), ("RDKit 2D + PH", "RDKit 2D")]
rows = []
for a, b in pairs:
    if a not in set(cv.rep) or b not in set(cv.rep):
        continue
    for m in MODELS:
        d, t, p = nb_test(a, b, m)
        rows.append(dict(a=a, b=b, model=m, delta_r2=d, t=t, p=p))
sig = pd.DataFrame(rows)
sig.to_csv(f"{R}/significance.csv", index=False)

# --------------------------------------------------------------------------------- controls figure
reps = ["PH", "PH (heavy atoms)", "Random subsets", "ESPH"]
fig, (a, b2) = plt.subplots(1, 2, figsize=(PAGE, 6.8 * CM), gridspec_kw={"width_ratios": [1.25, 1]})
mc = {"XGBoost": VIOLET, "LightGBM": PLUM, "HistGB": TEAL}
w = 0.26
for i, m in enumerate(MODELS):
    vals, errs = [], []
    for r in reps:
        if r == "Random subsets":
            rr = [f"Random subsets {k}" for k in (1, 2, 3)]
            vals.append(np.mean([get(x, m) for x in rr])); errs.append(np.mean([get(x, m, "r2_sd") for x in rr]))
        else:
            vals.append(get(r, m)); errs.append(get(r, m, "r2_sd"))
    xs = np.arange(len(reps)) + (i - 1) * w
    a.bar(xs, vals, w * 0.92, yerr=errs, color=mc[m], edgecolor=INK, linewidth=0.4,
          error_kw=dict(lw=0.7, capsize=1.6, ecolor=INK), label=m)
a.set_xticks(range(len(reps)), ["PH,\nall atoms", "PH,\nheavy", "random\nsubsets", "ESPH"], fontsize=8)
a.set_ylim(0.70, 0.95); a.set_ylabel("cross-validated $R^2$")
a.set_title("(a) What element selection adds")
a.legend(frameon=False, ncol=3, loc="upper left", fontsize=7.2, handlelength=1.0, columnspacing=0.8)
a.grid(axis="y", color=GRID, linewidth=0.5); a.set_axisbelow(True)
sels = ["C", "NO", "HAL", "CNO", "HEAVY", "POLAR"]
nm = {"C": "C", "NO": "N, O", "HAL": "halogen", "CNO": "C, N, O", "HEAVY": "heavy atoms", "POLAR": "polar"}
if all(f"ESPH without {s}" in set(cv.rep) for s in sels):
    drops = [np.mean([get("ESPH", m) - get(f"ESPH without {s}", m) for m in MODELS]) for s in sels]
    y = np.arange(len(sels))[::-1]
    b2.barh(y, drops, color=[PLUM if d > 0 else AMBER for d in drops], edgecolor=INK, linewidth=0.4,
            height=0.65)
    for yy, d in zip(y, drops):
        b2.text(d + (0.0004 if d >= 0 else -0.0004), yy, f"{d:+.3f}", va="center",
                ha="left" if d >= 0 else "right", fontsize=7.2)
    b2.set_yticks(y, [nm[s] for s in sels]); b2.axvline(0, color=INK, lw=0.8)
    lim = max(abs(min(drops)), max(drops)) * 1.45
    b2.set_xlim(min(-0.002, min(drops) * 1.45), lim)
    b2.set_xlabel("loss in $R^2$ when the selection is removed")
    b2.set_title("(b) Leave-one-selection-out")
    b2.grid(axis="x", color=GRID, linewidth=0.5); b2.set_axisbelow(True); b2.tick_params(axis="y", length=0)
fig.tight_layout(w_pad=1.5)
save(fig, f"{OUT}/fig_controls")

# --------------------------------------------------------------------------------- uplift figure
fps = [("MACCS", "MACCS"), ("Atom pair", "Atom pair"), ("Torsion", "Topological torsion")]
def best(rep):
    v = {m: get(rep, m) for m in MODELS}
    m = max(v, key=v.get)
    return v[m], get(rep, m, "r2_sd")
fig = plt.figure(figsize=(PAGE, 7.2 * CM))
ax = fig.add_axes([0.17, 0.27, 0.6, 0.6])
ph, es = best("PH")[0], best("ESPH")[0]
for x, lab in [(ph, "PH alone"), (es, "ESPH alone")]:
    ax.axvline(x, ls="--", lw=0.9, color=TEAL)
    ax.text(x, 2.62, f"{lab}\n{x:.3f}", ha="center", va="bottom", fontsize=7.5, color=TEAL)
gains = []
for i, (k, lab) in enumerate(fps):
    y = 2 - i
    f0, f0s = best(k); p1, p1s = best(f"PH + {k}"); e1, e1s = best(f"ESPH + {k}")
    ax.plot([min(f0, p1, e1), max(f0, p1, e1)], [y, y], color="#CFCBC6", lw=2.6, zorder=1)
    for v, s, c, dy in [(f0, f0s, AMBER, -0.17), (p1, p1s, ROSE, 0.17), (e1, e1s, VIOLET, 0)]:
        ax.errorbar(v, y + dy, xerr=s, fmt="o", color=c, ms=6.5, mec="white", mew=0.6,
                    elinewidth=0.8, capsize=1.8, zorder=3)
    gains.append((p1 - f0, e1 - f0))
ax.set_yticks([2, 1, 0], [l for _, l in fps]); ax.set_ylim(-0.6, 2.6)
ax.tick_params(axis="y", length=0)
ax.set_xlabel("cross-validated $R^2$ (best of three learners, ±1 s.d. over 15 folds)")
ax.grid(axis="x", color=GRID, linewidth=0.5); ax.set_axisbelow(True)
ax.spines["left"].set_visible(False)
tx = fig.add_axes([0.79, 0.27, 0.2, 0.6]); tx.axis("off"); tx.set_ylim(-0.6, 2.6); tx.set_xlim(0, 1)
tx.text(0.25, 2.5, "with PH", ha="center", color=ROSE, fontsize=8, style="italic")
tx.text(0.75, 2.5, "with ESPH", ha="center", color=VIOLET, fontsize=8, style="italic")
for i, (gp, ge) in enumerate(gains):
    tx.text(0.25, 2 - i, f"{gp:+.3f}", ha="center", va="center", color=ROSE, fontsize=8.5)
    tx.text(0.75, 2 - i, f"{ge:+.3f}", ha="center", va="center", color=VIOLET, fontsize=8.5,
            weight="bold")
tx.text(0.5, 2.85, "gain over the fingerprint alone", ha="center", fontsize=7.5, color="#6B6661")
from matplotlib.lines import Line2D
fig.legend(handles=[Line2D([], [], marker="o", ls="", color=c, ms=6, label=l) for c, l in
                    [(AMBER, "fingerprint alone"), (ROSE, "PH + fingerprint"), (VIOLET, "ESPH + fingerprint")]],
           loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.03))
save(fig, f"{OUT}/fig_uplift")

# --------------------------------------------------------------------------------- robustness figure
seeds = [42, 1, 2, 3, 4]
if all(f"ESPH seed {s}" in set(cv.rep) for s in (1, 2, 3, 4)) and os.path.exists(f"{R}/zang_split.csv"):
    fig, (a, b2) = plt.subplots(1, 2, figsize=(PAGE, 7.0 * CM), gridspec_kw={"width_ratios": [1, 1]})
    for blk, col, mk in [("PH", ROSE, "o"), ("ESPH", VIOLET, "s")]:
        for i, m in enumerate(MODELS):
            v = [get(blk if s == 42 else f"{blk} seed {s}", m) for s in seeds]
            xs = np.full(5, i) + (-0.14 if blk == "PH" else 0.14)
            a.scatter(xs + np.linspace(-0.06, 0.06, 5), v, color=col, marker=mk, s=16, edgecolor="white",
                      linewidth=0.4, label=blk if i == 0 else None, zorder=3)
            a.plot([xs[0] - 0.1, xs[0] + 0.1], [np.mean(v)] * 2, color=INK, lw=0.9)
    a.set_xticks(range(3), MODELS); a.set_ylabel("cross-validated $R^2$")
    a.set_title("(a) Five independent conformer seeds"); a.legend(frameon=False, loc="center right")
    a.grid(axis="y", color=GRID, linewidth=0.5); a.set_axisbelow(True)
    z = pd.read_csv(f"{R}/zang_split.csv")
    zp = np.load(f"{R}/zang_pred.npz")
    bz = z[z.rep == "clogP + ESPH"].sort_values("r2", ascending=False).iloc[0]
    pred = zp[f"{bz.rep}|{bz.model}"]; yt = zp["y_test"]
    b2.scatter(yt, pred, s=7, color=VIOLET, alpha=0.65, edgecolor="none")
    lo, hi = min(yt.min(), pred.min()) - 0.5, max(yt.max(), pred.max()) + 0.5
    b2.plot([lo, hi], [lo, hi], color=INK, lw=0.8, ls="--")
    b2.set_xlim(lo, hi); b2.set_ylim(lo, hi); b2.set_aspect("equal")
    b2.set_xlabel("experimental log S"); b2.set_ylabel("predicted log S")
    b2.set_title("(b) External test set, clog$P$ + ESPH")
    b2.text(0.04, 0.95, f"{bz.model}\n$R^2$ = {bz.r2:.3f}\nMSE = {bz.mse:.3f}\nn = {len(yt)}",
            transform=b2.transAxes, va="top", fontsize=7.8)
    b2.grid(color=GRID, linewidth=0.5); b2.set_axisbelow(True)
    fig.tight_layout(w_pad=1.5)
    save(fig, f"{OUT}/fig_robust")
print("analysis done")
