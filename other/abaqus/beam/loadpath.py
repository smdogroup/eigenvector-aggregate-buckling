from icecream import ic
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scienceplots

plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "sans-serif",
        "font.sans-serif": "Helvetica",
        "mathtext.fontset": "custom",
    }
)


def plot_rpd(a, b, ax, twinx=False):

    rpd = np.abs(np.subtract(a, b)) / np.add(a, b) * 100
    ax.plot(
        np.arange(1, 7),
        b,
        marker="s",
        color="r",
        label="Abaqus",
        markersize=2,
        linewidth=0.75,
    )
    ax.plot(
        np.arange(1, 7),
        a,
        marker="o",
        color="b",
        label="Topology Optimization",
        markersize=2,
        linewidth=0.75,
    )

    if ax.get_subplotspec().colspan.start == 0:
        ax.set_ylabel(r"$\lambda$", fontsize=6)
        # ax.yaxis.set_label_coords(-0.11, 0.5)
        # ax.text(
        #     -0.2,
        #     1.0,
        #     r"$\textbf{(a)}$",
        #     fontsize=9,
        #     transform=ax.transAxes,
        #     verticalalignment="top",
        # )

    ax.tick_params(axis="both", labelsize=5.5)
    ax.legend(frameon=False, fontsize=5.5)
    ax.set_ylim([0, 19])
    ax.set_xlabel("Mode Number", fontsize=5.5)

    ax.spines["top"].set_visible(False)
    ax.tick_params(direction="out")
    ax.tick_params(which="minor", direction="out")
    ax.tick_params(which="minor", left=False)
    # ax.legend(frameon=False, fontsize=5.5, loc=[0.1, 0.8])
    ax.legend(frameon=False, fontsize=5.5, loc="upper left", ncol=2, columnspacing=1.0, handletextpad=0.0)

    ax.tick_params(axis="y", which="major", length=2, direction="in")
    ax.tick_params(axis="x", which="major", length=2, direction="out")

    ax2 = ax.twinx()
    ax2.spines.right.set_position(("axes", 1.0))

    p1 = ax2.bar(np.arange(1, 7), rpd, color="gray", alpha=0.5)
    if ax.get_subplotspec().colspan.start == 3:
        ax2.set_ylabel(r"$RPD(\%)$", color="gray", fontsize=6)
        ax2.yaxis.set_label_coords(1.1, 0.5)

    ax2.tick_params(axis="both", labelsize=5.5)
    ax2.set_xticks(np.arange(1, 7))
    ax2.set_ylim([0, 19])
    ax2.spines["top"].set_visible(False)
    ax2.spines.right.set_visible(True)

    ax2.tick_params(direction="out")
    ax2.tick_params(which="minor", direction="out")
    ax2.yaxis.label.set_color("gray")
    ax2.tick_params(axis="y", colors="gray")

    ax2.tick_params(axis="y", which="major", length=2, direction="in")
    ax2.tick_params(axis="x", which="major", length=2, direction="out")

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


def plot_loadpath(f1, d1, n1, f2, d2, n2, f3, d3, n3, slope, offset, ds, ax):
    c = plt.colormaps["bwr"]
    mkz = 1.75
    lw = 0.75
    mk = ["o", "s", "^"]
    ax.spines["top"].set_visible(False)
    ax.tick_params(direction="out")
    ax.tick_params(which="minor", direction="out")
    ax.tick_params(which="minor", left=False)
    ax.tick_params(axis="y", which="major", length=2, direction="in")
    ax.tick_params(axis="x", which="major", length=2, direction="out")

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
        clip_on=False,
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
        fontsize=5.5,
        title=f"Initial Imperfection for Design \\textbf{{{ds}}}",
        title_fontsize="6",
        bbox_to_anchor=(1.0, 0.18),
        loc="upper right",
        ncol=3,
        columnspacing=0.5,
        handletextpad=0.0,  # reduce the space
    )
    ax.tick_params(axis="both", labelsize=5.5)
    ax.set_xlabel("Arc Length", fontsize=5.5)
    ax.set_ylim([-0.05, 1.02])
    if ax.get_subplotspec().colspan.start == 0:
        ax.set_ylabel(r"$\lambda / \lambda_{cr}$", fontsize=6)
        ax.yaxis.set_label_coords(-0.11, 0.5)

    ax.text(
        0.25 / slope,
        0.25 + offset,
        "Fundamental Path",
        fontsize=5.5,
        rotation=np.arctan(slope) * 180 / np.pi,
        transform_rotates_text=True,
        rotation_mode="anchor",
    )
    return


def read_xlsx():

    # read excel file beam.xlsx
    df = pd.read_excel("../abaqus.xlsx", sheet_name="Sheet1")

    # convert to numpy array
    data = df.to_numpy()

    # odd columns are force, even columns are displacement
    disp = data[:, 0::2]
    force = data[:, 1::2]
    disp[:, 2] = -disp[:, 2]
    disp[:, 3] = -disp[:, 3]

    slopes = []
    for i in range(force.shape[1]):
        slopes.append(np.max(np.abs(np.diff(force[:15, i]) / np.diff(disp[:15, i]))))

    return force, disp, slopes


def plot_loaddis(force, disp, slope, ax):
    cw = plt.colormaps["coolwarm"](np.linspace(0, 1, 10))
    c = ["k", "#e29400ff", cw[0], cw[-1], "m", "c", "y"]
    mkz = 2.0
    lw = 0.75
    mk = ["o", "s", "^", "v"]
    ds = [r"$\bar{h}=\text{N/A}$", r"$\bar{h}=4$", r"$\bar{h}=0.4h_{KS}$", r"$\bar{h}=0$"]
    end = [100, 80, 80, 70]

    for i in range(force.shape[1]):
        ax.plot(
            disp[: end[i], i],
            force[: end[i], i],
            marker=mk[i],
            label=ds[i],
            color=c[i],
            markersize=mkz,
            linewidth=lw,
            zorder=10,
        )

    ax.spines["top"].set_visible(False)
    # ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out")
    ax.tick_params(which="minor", direction="out")
    ax.tick_params(which="minor", left=False)
    ax.tick_params(axis="y", which="major", length=2, direction="in")
    ax.legend(
        frameon=False,
        fontsize=7,
        title=r"Initial Imperfection $0.1\% \phi_1$",
        title_fontsize="7",
        # bbox_to_anchor=(0.995, 0.9),
        loc="upper right",
        ncol=1,
        columnspacing=1.0,
    )

    ax.set_xlabel(r"Vertical Displacement $d_y$ for the Top-Middle Node", fontsize=7)
    ax.set_ylabel(r"$\lambda$", fontsize=7)
    ax.set_xlim(left=-0.08)
    ax.tick_params(axis="both", which="major", length=2, labelsize=6)

    return


a1 = [7.99, 9.54, 10.33, 10.85, 10.87, 11.27]
b1 = [8.5575, 9.8488, 10.180, 10.885, 10.950, 11.272]

a2 = [9.125, 9.293, 9.660, 10.207, 10.635, 11.086]
# b2 = [9.3514, 9.5602, 10.794, 10.855, 11.223, 11.437]
b2 = [9.7234, 9.8757, 10.625, 10.633, 10.868, 10.875]

a3 = [9.4340, 9.5014, 9.5107, 9.5246, 9.6159, 9.9012]
b3 = [9.4940, 9.5191, 9.9518, 9.9560, 9.9871, 10.191]

a4 = [8.8998, 8.9011, 9.023347, 9.121, 9.136, 9.4268]
b4 = [8.6968, 8.8643, 8.8696, 9.2221, 9.2564, 9.8649]

# compute the average rpd for all modes
rpd = np.mean(
    [
        np.abs(np.subtract(a1, b1)) / np.add(a1, b1) * 100,
        np.abs(np.subtract(a2, b2)) / np.add(a2, b2) * 100,
        np.abs(np.subtract(a3, b3)) / np.add(a3, b3) * 100,
        np.abs(np.subtract(a4, b4)) / np.add(a4, b4) * 100,
    ]
)
ic(rpd)

f11, d11, slope1 = read_data("loadpath-a-0001.txt", 100, b1[0])
f12, d12, _ = read_data("loadpath-a-0005.txt", 35, b1[0])
f13, d13, _ = read_data("loadpath-a-001.txt", 55, b1[0])

f21, d21, slope2 = read_data("b-b-00002.txt", 100, b2[0])
f22, d22, _ = read_data("b-b-0001.txt", 40, b2[0])
f23, d23, _ = read_data("b-b-0002.txt", 40, b2[0])

f31, d31, slope3 = read_data("loadpath-d-00004.txt", 100, b3[0])
f32, d32, _ = read_data("loadpath-d-0001.txt", 90, b3[0])
f33, d33, _ = read_data("loadpath-d-0005.txt", 100, b3[0])

f41, d41, slope4 = read_data("loadpath-c-0002.txt", 90, b4[0])
f42, d42, _ = read_data("loadpath-c-0005.txt", 100, b4[0])
f43, d43, _ = read_data("loadpath-c-001.txt", 32, b4[0])

with plt.style.context(["nature"]):
    fig = plt.subplots(
        figsize=(7.5, 5.5),
        tight_layout=True,
    )
    row_ratio = [2, 4, 5]
    total_rows = sum(row_ratio)

    for i in range(4):
        a = [a1, a2, a3, a4][i]
        b = [b1, b2, b3, b4][i]
        twinx = [False, False, False, True][i]

        ax0 = plt.subplot2grid((total_rows, 4), (0, i), rowspan=row_ratio[0])
        plot_rpd(a, b, ax0, twinx)

        f1 = [f11, f21, f31, f41][i]
        d1 = [d11, d21, d31, d41][i]
        slope = [slope1, slope2, slope3, slope4][i]
        f2 = [f12, f22, f32, f42][i]
        d2 = [d12, d22, d32, d42][i]
        f3 = [f13, f23, f33, f43][i]
        d3 = [d13, d23, d33, d43][i]
        offset = [0.07, 0.07, 0.07, 0.08][i]
        n = [[0.1, 0.5, 1.0], [0.02, 0.1, 0.2], [0.04, 0.1, 0.5], [0.2, 0.5, 1.0]][i]
        ds = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)"]
        ax1 = plt.subplot2grid((total_rows, 4), (row_ratio[0], i), rowspan=row_ratio[1])
        plot_loadpath(
            f1, d1, n[0], f2, d2, n[1], f3, d3, n[2], slope, offset, ds[i], ax1
        )

        if i != 0:
            ax0.set_yticklabels([])
            ax1.set_yticklabels([])

    plt.subplots_adjust(wspace=-0.5)

    ax = plt.subplot2grid(
        (total_rows, 4),
        (row_ratio[0] + row_ratio[1], 0),
        colspan=4,
        rowspan=row_ratio[2],
    )
    force, disp, slope = read_xlsx()
    plot_loaddis(force, disp, slope, ax)
    fig[0].delaxes(fig[1]) # close outer axes for the main plot
    # plt.subplots_adjust(hspace=0.01)

    plt.savefig("beam.png", dpi=500, bbox_inches="tight", pad_inches=0.0)


# with plt.style.context(["nature"]):
#     fig, ax = plt.subplots(
#         2,
#         4,
#         figsize=(7.5, 3.2),
#         gridspec_kw={"height_ratios": [1, 2.3]},
#         tight_layout=True,
#         sharey="row",
#     )

#     for i in range(4):
#         a = [a1, a2, a3, a4][i]
#         b = [b1, b2, b3, b4][i]
#         twinx = [False, False, False, True][i]

#         plot_rpd(a, b, ax[0, i], twinx)

#         f1 = [f11, f21, f31, f41][i]
#         d1 = [d11, d21, d31, d41][i]
#         slope = [slope1, slope2, slope3, slope4][i]
#         f2 = [f12, f22, f32, f42][i]
#         d2 = [d12, d22, d32, d42][i]
#         f3 = [f13, f23, f33, f43][i]
#         d3 = [d13, d23, d33, d43][i]
#         offset = [0.07, 0.07, 0.07, 0.08][i]
#         n = [[0.1, 0.5, 1.0], [0.02, 0.1, 0.2], [0.04, 0.1, 0.5], [0.2, 0.5, 1.0]][i]
#         ds = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)"]
#         plot_loadpath(
#             f1, d1, n[0], f2, d2, n[1], f3, d3, n[2], slope, offset, ds[i], ax[1, i]
#         )

#     plt.subplots_adjust(wspace=-0.5)
#     plt.savefig("beam0.png", dpi=500, bbox_inches="tight", pad_inches=0.0)


# with plt.style.context(["nature"]):
#     fig, ax = plt.subplots(1, 1, figsize=(7.5, 3.2), tight_layout=True)
#     force, disp, slope = read_xlsx()
#     plot_loaddis(force, disp, slope, ax)
#     plt.savefig("beam.png", dpi=500, bbox_inches="tight", pad_inches=0.0)
