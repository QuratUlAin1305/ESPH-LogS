"""Figures from the feature-level diagnostics: top-gain bits, gain concentration, MACCS direction, SHAP."""
import sys, os
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from figs_common import *

R, OUT = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)
PREFIX = {"MACCS": "MACCS", "Atom pair": "AP", "Torsion": "TT"}
bits = {k: pd.read_csv(f"{R}/bits_{k}.csv").sort_values("gain", ascending=False) for k in PREFIX}
disp = pd.read_csv(f"{R}/bit_disposition.csv").set_index("fp")

# ---- top-gain bits -------------------------------------------------------------------------
N = 20
fig, axes = plt.subplots(1, 3, figsize=(PAGE, 11.5 * CM))
for ax, (k, b) in zip(axes, bits.items()):
    top = b.head(N).iloc[::-1]
    y = np.arange(N)
    ax.barh(y, top.gain, color=FP_COL[k], edgecolor=INK, linewidth=0.4, height=0.72)
    ax.set_yticks(y, [f"{PREFIX[k]}_{int(i)}" for i in top.bit], fontsize=7.5)
    xmax = top.gain.max()
    for yy, g, p in zip(y, top.gain, top.prevalence):
        ax.text(g + 0.02 * xmax, yy, f"{int(p)}", va="center", fontsize=7, color="#6B6661")
    ax.set_xlim(0, xmax * 1.22)
    ax.set_title({"Torsion": "Topological torsion", "Atom pair": "Atom pair"}.get(k, k))
    ax.set_xlabel("mean split gain")
    ax.grid(axis="x", color=GRID, linewidth=0.5); ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)
fig.tight_layout(w_pad=1.2)
save(fig, f"{OUT}/fig_topgain")

# ---- gain concentration --------------------------------------------------------------------
fig, (a, b2) = plt.subplots(1, 2, figsize=(PAGE, 6.6 * CM), gridspec_kw={"width_ratios": [1.35, 1]})
marks = {"MACCS": "o", "Atom pair": "s", "Torsion": "^"}
for k, b in bits.items():
    g = b.gain[b.gain > 0].values
    share = 100 * np.cumsum(g) / g.sum()
    n = np.arange(1, len(share) + 1)
    m = n <= 200
    a.plot(n[m], share[m], color=FP_COL[k], lw=1.6,
           label={"Torsion": "Topological torsion"}.get(k, k))
    idx = np.arange(19, min(200, len(share)), 20)
    a.plot(n[idx], share[idx], marks[k], color=FP_COL[k], ms=3.6, mec="white", mew=0.5)
    if k == "MACCS":
        a.plot(len(share), 100, "o", color=FP_COL[k], ms=5, mec=INK, mew=0.6)
        a.annotate(f"all {len(share)} used keys", (len(share), 100), (len(share) - 5, 91),
                   ha="right", fontsize=7.5, color=VIOLET)
a.axhline(80, ls="--", lw=0.7, color=INK)
a.set_xlim(0, 200); a.set_ylim(0, 104)
a.set_xlabel("number of top-gain bits retained"); a.set_ylabel("share of total split gain (%)")
a.set_title("(a) Cumulative gain"); a.legend(frameon=False, loc="lower right")
a.grid(color=GRID, linewidth=0.5); a.set_axisbelow(True)
w = 0.26
for i, (k, b) in enumerate(bits.items()):
    used = disp.loc[k, "used"]
    vals = [disp.loc[k, c] for c in ("k50", "k80", "k90")]
    xs = np.arange(3) + (i - 1) * w
    b2.bar(xs, 100 * np.array(vals) / used, w * 0.92, color=FP_COL[k], edgecolor=INK, linewidth=0.4)
    for x, v in zip(xs, vals):
        b2.text(x, 100 * v / used + 1.2, str(int(v)), ha="center", fontsize=7)
b2.set_xticks(range(3), ["50% of gain", "80% of gain", "90% of gain"])
b2.set_ylabel("share of used bits required (%)"); b2.set_ylim(0, 65)
b2.set_title("(b) Concentration of importance")
b2.grid(axis="y", color=GRID, linewidth=0.5); b2.set_axisbelow(True)
fig.tight_layout(w_pad=1.5)
save(fig, f"{OUT}/fig_gaintrend")

# ---- MACCS direction -----------------------------------------------------------------------
LAB = {145: "> 1 six-membered ring", 121: "ring nitrogen", 162: "aromatic atom",
       125: "> 1 aromatic ring", 107: "halogen on branched atom", 163: "six-membered ring",
       134: "any halogen", 60: "S=O group", 164: "oxygen present", 14: "S–S bond",
       98: "heteroatom in six-membered ring", 103: "chlorine", 139: "hydroxyl group",
       161: "nitrogen present"}
m = bits["MACCS"].head(14)
ntr = 1608
fig, ax = plt.subplots(figsize=(PAGE, 8.6 * CM))
y = np.arange(len(m))[::-1]
cols = [VIOLET if v > 0 else PLUM for v in m.delta_logS]
ax.barh(y, m.delta_logS, color=cols, edgecolor=INK, linewidth=0.4, height=0.66)
ax.set_yticks(y, [f"MACCS_{int(k)}  {LAB[int(k)]}" for k in m.bit])
for yy, v, p in zip(y, m.delta_logS, m.prevalence):
    txt = f"{int(p)} ({100 * p / ntr:.1f}%)"
    ax.text(v + (0.06 if v > 0 else -0.06), yy, txt, va="center", ha="left" if v > 0 else "right",
            fontsize=7.2)
ax.axvline(0, color=INK, lw=0.8)
ax.set_xlim(-4.3, 2.6)
ax.set_xlabel("mean log S with key set − mean log S with key unset")
ax.text(-4.25, len(m) - 0.35, "less soluble", color=PLUM, style="italic", fontsize=8)
ax.text(2.55, len(m) - 0.35, "more soluble", color=VIOLET, style="italic", fontsize=8, ha="right")
ax.grid(axis="x", color=GRID, linewidth=0.5); ax.set_axisbelow(True)
ax.tick_params(axis="y", length=0)
fig.tight_layout()
save(fig, f"{OUT}/fig_maccsdir")

# ---- SHAP for ESPH -------------------------------------------------------------------------
s = pd.read_csv(f"{R}/shap_esph.csv")
s["sel"] = s.feature.str.split(".").str[0]; s["deg"] = s.feature.str.split(".").str[1]
order = ["C", "NO", "HAL", "CNO", "HEAVY", "POLAR"]
names = {"C": "C", "NO": "N, O", "HAL": "halogen", "CNO": "C, N, O", "HEAVY": "heavy atoms",
         "POLAR": "polar"}
H = s.groupby(["sel", "deg"]).mean_abs_shap.sum().unstack().reindex(order)[["H0", "H1", "H2"]]
H = 100 * H / H.values.sum()
fig, (a, b2) = plt.subplots(1, 2, figsize=(PAGE, 7.4 * CM), gridspec_kw={"width_ratios": [0.8, 1.5]})
from matplotlib.colors import LinearSegmentedColormap
cmap = LinearSegmentedColormap.from_list("v", ["#FFFFFF", "#E9DDF0", VIOLET])
a.imshow(H.values, cmap=cmap, aspect="auto", vmin=0, vmax=H.values.max())
for i in range(H.shape[0]):
    for j in range(3):
        v = H.values[i, j]
        a.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=8,
               color="white" if v > 0.55 * H.values.max() else INK)
a.set_xticks(range(3), ["$H_0$", "$H_1$", "$H_2$"])
a.set_yticks(range(len(order)), [names[o] for o in order])
a.set_title("(a) Share of attribution (%)")
for sp in a.spines.values():
    sp.set_visible(False)
a.tick_params(length=0)
top = s.head(15).iloc[::-1]
yy = np.arange(len(top))
statn = {"count": "count", "sum": "sum", "max": "max", "mean": "mean", "std": "std",
         "median": "median", "l2": "L2 norm", "sqsum": "sum of squares"}
labs = [f"{names[f.split('.')[0]]} · ${f.split('.')[1][0]}_{f.split('.')[1][1]}$ · {statn[f.split('.')[2]]}"
        for f in top.feature]
b2.barh(yy, top.mean_abs_shap, color=[VIOLET if c > 0 else PLUM for c in top.sign],
        edgecolor=INK, linewidth=0.4, height=0.7)
b2.set_yticks(yy, labs, fontsize=7.8)
b2.set_xlabel("mean |SHAP value| (log units)")
b2.set_title("(b) Fifteen most influential ESPH features")
b2.grid(axis="x", color=GRID, linewidth=0.5); b2.set_axisbelow(True)
b2.tick_params(axis="y", length=0)
from matplotlib.patches import Patch
b2.legend(handles=[Patch(color=PLUM, label="larger value lowers log S"),
                   Patch(color=VIOLET, label="larger value raises log S")],
          frameon=False, loc="lower right", fontsize=7.5)
fig.tight_layout(w_pad=1.2)
save(fig, f"{OUT}/fig_shap")
print("diagnostic figures written")
