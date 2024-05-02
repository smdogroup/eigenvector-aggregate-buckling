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


def plot_rpd(a, b, c, alpha, ax, twinx=False):

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
    ax.set_ylim([0, 14])
    ax.set_xlabel("Mode Number", fontsize=6)

    ax.spines["top"].set_visible(False)
    ax.tick_params(direction="out")
    ax.tick_params(which="minor", direction="out")
    ax.tick_params(which="minor", left=False)
    # ax.legend(frameon=False, fontsize=5, loc=[0.1, 0.8])
    ax.legend(frameon=False, fontsize=5, loc="upper left")

    ax2 = ax.twinx()
    ax2.spines.right.set_position(("axes", 1.0))

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

    if not twinx:
        ax2.set_yticklabels([])

    rpd = np.abs(np.subtract(a, b)) / np.add(a, b) * 100

    for i, v in enumerate(rpd):
        ax2.bar(i + 1, v, color=c[i], alpha=alpha[i])
        ax2.text(
            i + 1,
            v + 0.25,
            f"{v:.2f}",
            color=c[i],
            fontsize=6,
            ha="center",
            alpha=alpha[i],
        )

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


def plot_loadpath(f1, d1, n1, f2, d2, n2, f3, d3, n3, slope, offset, nm1, nm2, ax):
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
        label=str(n1) + nm1,
        color=c(0.0),
        markersize=mkz,
        linewidth=lw,
        zorder=10,
    )
    ax.plot(
        d2,
        f2,
        marker=mk[1],
        label=str(n2) + nm1,
        color=c(0.4),
        markersize=mkz,
        linewidth=lw,
        zorder=9,
    )
    ax.plot(
        d3,
        f3,
        marker=mk[2],
        label=str(n3) + nm1,
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
        bbox_to_anchor=(0.995, 0.22),
        loc="upper right",
        ncol=2,
        columnspacing=1.0,
    )

    ax.set_xlabel("Arc Length", fontsize=6)
    ax.set_ylim([-0.05, 1.05])
    # if ax.get_subplotspec().colspan.start == 0:
    ax.set_ylabel(nm2, fontsize=6)

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


########################################################
a1 = [8.2815, 8.2815, 8.6217, 8.9815, 9.6923, 9.6923]
b1 = [8.4904, 8.4904, 8.8346, 8.9345, 10.085, 10.085]

a2 = [6.0577, 6.2421, 6.3749, 6.3749, 6.6544, 6.7259]
b2 = [6.1755, 6.4332, 6.7080, 6.7080, 7.1471, 7.2465]

a3 = [6.7219, 6.7219, 6.7359, 6.9385, 7.4864, 7.5317]
b3 = [6.8318, 6.8318, 6.8538, 6.9195, 7.2011, 7.2367]

rpd = np.mean(
    [
        np.abs(np.subtract(a1, b1)) / np.add(a1, b1) * 100,
        np.abs(np.subtract(a2, b2)) / np.add(a2, b2) * 100,
        np.abs(np.subtract(a3, b3)) / np.add(a3, b3) * 100,
    ]
)
ic(rpd)

f11, d11, slope1 = read_data("s-a-12-0006.txt", 40, b1[0])
f12, d12, _ = read_data("s-a-12-001.txt", 40, b1[0])
f13, d13, _ = read_data("s-a-12-002.txt", 35, b1[0])

f21, d21, slope2 = read_data("s-b-12-0001.txt", 50, b2[2])
f22, d22, _ = read_data("s-b-12-0005.txt", 50, b2[2])
f23, d23, _ = read_data("s-b-12-001.txt", 50, b2[2])

f31, d31, slope3 = read_data("s-c-12-0002.txt", 45, b3[0])
f32, d32, _ = read_data("s-c-12-0005.txt", 40, b3[0])
f33, d33, _ = read_data("s-c-12-001.txt", 40, b3[0])

c1 = ["b", "b", "gray", "gray", "b", "b"]
c2 = ["gray", "gray", "b", "b", "gray", "gray"]
c3 = ["b", "b", "gray", "gray", "gray", "gray"]

alpha1 = [0.5, 0.5, 0.5, 0.5, 0.25, 0.25]
alpha2 = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
alpha3 = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]

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
        c = [c1, c2, c3][i]
        alpha = [alpha1, alpha2, alpha3][i]
        twinx = [False, False, True][i]

        plot_rpd(a, b, c, alpha, ax[0, i], twinx)

        f1 = [f11, f21, f31][i]
        d1 = [d11, d21, d31][i]
        slope = [slope1, slope2, slope3][i]
        f2 = [f12, f22, f32][i]
        d2 = [d12, d22, d32][i]
        f3 = [f13, f23, f33][i]
        d3 = [d13, d23, d33][i]
        offset = [0.07, 0.12, 0.07][i]
        n = [[0.6, 1.0, 2.0], [0.1, 0.5, 1.0], [0.2, 0.5, 1.0]][i]
        nm1 = [
            r"$\% (\phi_1 + \phi_2)$",
            r"$\% (\phi_3 + \phi_4)$",
            r"$\% (\phi_1 + \phi_2)$",
        ][i]
        nm2 = [
            r"$\lambda/\lambda_{cr}$",
            r"$\lambda/\lambda_{3}$",
            r"$\lambda/\lambda_{cr}$",
        ][i]
        plot_loadpath(
            f1, d1, n[0], f2, d2, n[1], f3, d3, n[2], slope, offset, nm1, nm2, ax[1, i]
        )

    plt.subplots_adjust(wspace=0.4)
    plt.savefig("square.png", dpi=1000, bbox_inches="tight", pad_inches=0.0)


# f1, d1, slope = read_data("s-a-001.txt", 40, b1[0])
# f2, d2, _ = read_data("s-a-002.txt", 40, b1[0])
# f3, d3, _ = read_data("s-a-003.txt", 40, b1[0])
# with plt.style.context(["nature"]):
#     ax = plot_loadpath(f1, d1, 1.0, f2, d2, 2.0, f3, d3, 3.0, slope, r"$\% \phi_1$")
#     ax.text(
#         y / slope,
#         y + 0.05,
#         "Fundamental Path",
#         fontsize=4,
#         rotation=np.arctan(slope) * 180 / np.pi,
#         transform_rotates_text=True,
#         rotation_mode="anchor",
#     )
#     ax.set_ylabel("$P / P_{cr}$", rotation=0)
#     ax.yaxis.set_label_coords(0.02, 1.0)
#     plt.savefig("loadpath_a.png", dpi=1000, bbox_inches="tight", pad_inches=0.05)


# f1, d1, slope = read_data("s-b-0001.txt", 50, b2[2])
# f2, d2, _ = read_data("s-b-0005.txt", 50, b2[2])
# f3, d3, _ = read_data("s-b-001.txt", 50, b2[2])
# with plt.style.context(["nature"]):
#     ax = plot_loadpath(f1, d1, 0.1, f2, d2, 0.5, f3, d3, 1.0, slope, r"$\% \phi_1$")
#     ax.text(
#         y / slope,
#         y + 0.1,
#         "Fundamental Path",
#         fontsize=4,
#         rotation=np.arctan(slope) * 180 / np.pi,
#         transform_rotates_text=True,
#         rotation_mode="anchor",
#     )
#     ax.set_ylabel("$P / P_{3}$", rotation=0)
#     ax.yaxis.set_label_coords(0.02, 1.0)
#     plt.savefig("loadpath_b.png", dpi=1000, bbox_inches="tight", pad_inches=0.05)


# f1, d1, slope = read_data("s-c-0002.txt", 45, b3[0])
# f2, d2, _ = read_data("s-c-0005.txt", 40, b3[0])
# f3, d3, _ = read_data("s-c-001.txt", 40, b3[0])
# with plt.style.context(["nature"]):
#     ax = plot_loadpath(f1, d1, 0.2, f2, d2, 0.5, f3, d3, 1.0, slope, r"$\% \phi_1$")
#     ax.text(
#         y / slope,
#         y + 0.05,
#         "Fundamental Path",
#         fontsize=4,
#         rotation=np.arctan(slope) * 180 / np.pi,
#         transform_rotates_text=True,
#         rotation_mode="anchor",
#     )
#     ax.set_ylabel("$P / P_{cr}$", rotation=0)
#     ax.yaxis.set_label_coords(0.02, 1.0)
#     plt.savefig("loadpath_c.png", dpi=1000, bbox_inches="tight", pad_inches=0.05)
