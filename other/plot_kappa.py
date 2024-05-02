from icecream import ic
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
import numpy as np
import scienceplots

plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "sans-serif",
        "font.sans-serif": "Helvetica",
    }
)


BLF1 = np.array([22.40])
BLF2 = np.array([22.46, 14.03])
BLF3 = np.array([22.70, 14.25, 12.50])
BLF4 = np.array([22.72, 14.27, 12.35, 11.91])
BLF5 = np.array([22.72, 14.28, 12.32, 11.89, 11.71])
BLF6 = np.array([22.72, 14.28, 12.37, 11.84, 11.95, 11.69])
BLF7 = np.array([22.72, 14.28, 12.33, 11.86, 11.98, 11.78, 11.72])
BLF8 = np.array([22.72, 14.28, 12.34, 11.95, 11.87, 11.86, 11.78, 11.54])
BLF9 = np.array([22.72, 14.28, 12.35, 11.75, 11.91, 11.88, 11.84, 11.80, 11.26])
BLF10 = np.array([22.72, 14.28, 12.32, 11.82, 11.88, 11.86, 11.76, 11.77, 11.79, 11.42])
BLF11 = np.array(
    [22.72, 14.28, 12.34, 11.74, 11.88, 11.94, 11.84, 11.80, 11.70, 11.30, 11.40]
)
BLF12 = np.array(
    [22.72, 14.28, 12.26, 11.79, 11.87, 11.86, 11.78, 11.83, 11.66, 11.33, 11.40, 11.73]
)
KK = np.logspace(-1, -12, num=12)

BLF = np.zeros([12, 12])
for i in range(12):
    BLF[i, : i + 1] = eval(f"BLF{i+1}")

BLF[BLF == 0] = np.nan


# with plt.style.context(["nature"]):
#     fig, ax = plt.subplots(1, 1, figsize=(5, 5))
#     c = ax.imshow(BLF, cmap="coolwarm", aspect="auto")
#     divider = make_axes_locatable(ax)
#     cax = divider.append_axes("right", size="2.5%", pad=0.1)
#     plt.colorbar(c, cax=cax, label="BLF")

#     ax.set_xlabel(r"$\kappa_K$")
#     ax.set_ylabel(r"$\kappa_G$")
#     ax.set_xticks(np.arange(0, 12))
#     ax.set_yticks(np.arange(0, 12))
#     ax.set_xticklabels([f"$10^{{-{i}}}$" for i in range(1, 13)])
#     ax.set_yticklabels([f"$10^{{-{i}}}$" for i in range(1, 13)])
#     ax.axis("equal")
#     plt.tight_layout()
#     plt.savefig(
#         "output/final_results/kappa_contour.pdf",
#         bbox_inches="tight",
#         dpi=300,
#         pad_inches=0.0,
#     )


with plt.style.context(["nature"]):
    fig, ax = plt.subplots(1, 1, figsize=(5, 3.0), tight_layout=True)
    row_select = [3, 6, 9, 12]
    # row_select = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    # colors = plt.cm.bwr(np.linspace(0, 1, len(row_select)))
    markers = ["o", "s", "^", "D", "P", "X", "H", "d", "p", ">", "<", "^"]
    colors = ["r", "b", "orange", "k"]
    for i in range(len(row_select)):
        ax.plot(
            KK[: row_select[i]],
            BLF[row_select[i] - 1, : row_select[i]],
            marker=markers[i],
            color=colors[i],
            label=f"$\\kappa_G=10^{{-{row_select[i]}}}$",
            markersize=2,
            linewidth=0.75,
            zorder=12 - i,
            clip_on=False,
        )
        
    ax.set_xscale("log")
    ax.set_xlabel(r"$\kappa_K$", fontsize=10)
    ax.set_ylabel(r"$\lambda_{1}$", fontsize=10)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out")
    ax.tick_params(which="minor", direction="out")
    ax.tick_params(which="minor", left=False)
    ax.set_xticks(KK)

    ax.spines["bottom"].set_color("gray")
    ax.spines["left"].set_color("gray")
    ax.xaxis.label.set_color("gray")
    ax.yaxis.label.set_color("gray")
    ax.tick_params(axis="x", colors="gray")
    ax.tick_params(axis="y", colors="gray")
    ax.legend(frameon=True, framealpha=1.0, fontsize=9, edgecolor="none", ncol=2, loc=[0.0, 0.7])
    ax.set_ylim([0, 20])
    ax.set_xlim([1e-12, 1e-1])
    ax.spines["left"].set_position(("outward", 30))
    ax.spines["bottom"].set_position(("outward", 70))

    plt.savefig(
        "output/final_results/building/kappa/kappa.png", bbox_inches="tight", dpi=1000, pad_inches=0.01
    )
