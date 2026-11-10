import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK, VIOLET, PLUM, ROSE, AMBER, TEAL = "#22201F", "#4A1B6D", "#7B3C54", "#D4537E", "#EF9F27", "#2E7D6F"
GRID = "#D9D6D2"
CM = 1 / 2.54
PAGE = 16.5 * CM          # full page width demanded by the journal

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9,
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "legend.fontsize": 8.5,
    "axes.edgecolor": INK, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
    "text.color": INK, "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.8, "savefig.dpi": 600, "figure.dpi": 150,
})

FP_COL = {"MACCS": VIOLET, "Atom pair": PLUM, "Torsion": TEAL}


def save(fig, path):
    """PNG for the Word file, JPEG and TIFF at 600 dpi for submission."""
    from PIL import Image
    fig.savefig(path + ".png", bbox_inches="tight", pad_inches=0.04)
    im = Image.open(path + ".png").convert("RGB")
    im.save(path + ".jpg", quality=95, dpi=(600, 600))
    im.save(path + ".tif", compression="tiff_lzw", dpi=(600, 600))
    plt.close(fig)
