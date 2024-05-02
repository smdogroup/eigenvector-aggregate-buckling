from icecream import ic
import matplotlib.pyplot as plt
import numpy as np
import scienceplots


plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "sans-serif",
        "font.sans-serif": "Helvetica",
    }
)


def plot_rpd(a, b, ax, twinx=False):

    rpd = np.abs(np.subtract(a, b)) / np.add(a, b) * 100

    ax.plot(
        np.arange(1, 7),
        a,
        marker="o",
        color="r",
        label="Topology Optimization",
        markersize=2,
        linewidth=0.75,
    )
    ax.plot(
        np.arange(1, 7),
        b,
        marker="s",
        color="b",
        label="Abaqus",
        markersize=2,
        linewidth=0.75,
    )
    if ax.get_subplotspec().colspan.start == 0:
        ax.set_ylabel(r"$\lambda$", fontsize=6)
        # ax.yaxis.set_label_coords(-0.11, 0.5)

    ax.legend(frameon=False, fontsize=5)
    ax.set_ylim([0, 22])
    ax.set_xlabel("Mode Number", fontsize=6)

    ax.spines["top"].set_visible(False)
    ax.tick_params(direction="out")
    ax.tick_params(which="minor", direction="out")
    ax.tick_params(which="minor", left=False)
    # ax.legend(frameon=False, fontsize=5, loc=[0.1, 0.8])
    ax.legend(frameon=False, fontsize=5, loc="upper left")

    ax2 = ax.twinx()
    ax2.spines.right.set_position(("axes", 1.0))

    p1 = ax2.bar(np.arange(1, 7), rpd, color="gray", alpha=0.5)
    if ax.get_subplotspec().colspan.start == 2:
        ax2.set_ylabel(r"$RPD(\%)$", color="gray", fontsize=6)
        ax2.yaxis.set_label_coords(1.12, 0.5)
    ax2.set_xticks(np.arange(1, 7))
    ax2.set_ylim([0, 25])
    ax2.spines["top"].set_visible(False)
    ax2.spines.right.set_visible(True)

    ax2.tick_params(direction="out")
    ax2.tick_params(which="minor", direction="out")
    ax2.yaxis.label.set_color("gray")
    ax2.tick_params(axis="y", colors="gray")

    for i, v in enumerate(rpd):
        ax2.text(i + 1, v + 0.25, f"{v:.2f}", color="gray", fontsize=6, ha="center")

    if not twinx:
        ax2.set_yticklabels([])

    return


def read_data(filename, n, p_cr):
    # Read data from the text file and split each line into columns based on spaces
    with open(filename, "r") as file:
        data = [line.split() for line in file]

    # delate first 2 rows
    data = data[2:]

    # Convert the data to a NumPy array, filtering out rows with None values
    data = np.array([row for row in data if row]).astype(float)

    data = data[:n]
    force = data[:, 1] / p_cr
    arc_length = data[:, 0]

    max_slope = np.max(np.abs(np.diff(force[:15]) / np.diff(arc_length[:15])))

    return force, arc_length, max_slope


def plot_loadpath(f1, d1, n1, f2, d2, n2, f3, d3, n3, slope, offset, ax):
    c = plt.colormaps["bwr"]
    mkz = 1.25
    lw = 0.5
    mk = ["o", "s", "^"]
    ax.spines["top"].set_visible(False)
    ax.tick_params(direction="out")
    ax.tick_params(which="minor", direction="out")
    ax.tick_params(which="minor", left=False)

    # check d1, d2, d3 which have larger length
    xvalue = np.arange(0, 15, 0.1)

    ax.plot(
        xvalue,
        slope * xvalue,
        linestyle="--",
        color="r",
        linewidth=0.75,
    )
    ax.axhline(
        1,
        color="black",
        linestyle="--",
        alpha=0.3,
        linewidth=1.0,
    )

    ax.plot(
        d1,
        f1,
        marker=mk[0],
        label=str(n1) + r"$\% \phi_1$",
        color=c(0.0),
        markersize=mkz,
        linewidth=lw,
        zorder=10,
    )
    ax.plot(
        d2,
        f2,
        marker=mk[1],
        label=str(n2) + r"$\% \phi_1$",
        color=c(0.4),
        markersize=mkz,
        linewidth=lw,
        zorder=9,
    )
    ax.plot(
        d3,
        f3,
        marker=mk[2],
        label=str(n3) + r"$\% \phi_1$",
        color=c(0.45),
        markersize=mkz,
        linewidth=lw,
        zorder=8,
    )

    ax.legend(
        frameon=False,
        fontsize=5,
        title="Load Path with Initial Imperfection",
        title_fontsize="5.5",
        bbox_to_anchor=(0.995, 0.15),
        loc="upper right",
        ncol=3,
        columnspacing=1.0,
    )

    ax.set_xlabel("Arc Length", fontsize=6)
    ax.set_ylim([-0.05, 1.05])
    if ax.get_subplotspec().colspan.start == 0:
        ax.set_ylabel(r"$\lambda/\lambda_{cr}$", fontsize=6)

    ax.text(
        0.25 / slope,
        0.25 + offset,
        "Fundamental Path",
        fontsize=5,
        rotation=np.arctan(slope) * 180 / np.pi,
        transform_rotates_text=True,
        rotation_mode="anchor",
    )
    return


a1 = [12.50, 14.19, 14.78, 15.04, 15.61, 15.66]
b1 = [13.749, 14.057, 14.303, 15.164, 15.802, 16.634]

a2 = [12.68, 13.47, 13.70, 13.86, 14.72, 14.90]
# b2 = [13.213, 13.50, 13.872, 14.346, 14.617, 15.319]
# b2 = [12.601, 12.789, 13.479, 13.812, 14.226, 15.021]
b2 = [13.529, 13.724, 13.942, 14.193, 14.875, 14.924]

a3 = [10.89, 11.79, 12.30, 12.40, 12.88, 12.90]
b3 = [11.603, 11.802, 12.619, 12.764, 13.316, 13.957]

rpd = np.mean(
    [
        np.abs(np.subtract(a1, b1)) / np.add(a1, b1) * 100,
        np.abs(np.subtract(a2, b2)) / np.add(a2, b2) * 100,
        np.abs(np.subtract(a3, b3)) / np.add(a3, b3) * 100,
    ]
)
ic(rpd)

f11, d11, slope1 = read_data("column-a-0001.txt", 100, b1[0])
f12, d12, _ = read_data("column-a-001.txt", 100, b1[0])
f13, d13, _ = read_data("column-a-003.txt", 60, b1[0])

f21, d21, slope2 = read_data("c-b-0001.txt", 60, b2[0])
f22, d22, _ = read_data("c-b-001.txt", 30, b2[0])
f23, d23, _ = read_data("c-b-002.txt", 30, b2[0])

f31, d31, slope3 = read_data("loadpath-c-0001.txt", 50, b3[0])
f32, d32, _ = read_data("loadpath-c-002.txt", 32, b3[0])
f33, d33, _ = read_data("loadpath-c-005.txt", 26, b3[0])

with plt.style.context(["nature"]):
    fig, ax = plt.subplots(
        2,
        3,
        figsize=(7.0, 3.6),
        gridspec_kw={"height_ratios": [1, 2.3]},
        tight_layout=True,
        sharey="row",
    )

    for i in range(3):
        a = [a1, a2, a3][i]
        b = [b1, b2, b3][i]
        twinx = [False, False, True][i]

        plot_rpd(a, b, ax[0, i], twinx)

        f1 = [f11, f21, f31][i]
        d1 = [d11, d21, d31][i]
        slope = [slope1, slope2, slope3][i]
        f2 = [f12, f22, f32][i]
        d2 = [d12, d22, d32][i]
        f3 = [f13, f23, f33][i]
        d3 = [d13, d23, d33][i]
        offset = [0.08, 0.04, 0.05][i]
        n = [[0.1, 1.0, 3.0], [0.1, 1.0, 2.0], [0.1, 2.0, 5.0]][i]
        plot_loadpath(f1, d1, n[0], f2, d2, n[1], f3, d3, n[2], slope, offset, ax[1, i])

    plt.subplots_adjust(wspace=0.45)
    plt.savefig("column.png", dpi=1000, bbox_inches="tight", pad_inches=0.0)
