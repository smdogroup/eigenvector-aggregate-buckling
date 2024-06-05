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

    ax.tick_params(axis="both", labelsize=5.5)
    ax.legend(frameon=False, fontsize=5.5)
    ax.set_ylim([0, 25])
    ax.set_xlabel("Mode Number", fontsize=5.5)

    ax.spines["top"].set_visible(False)
    ax.tick_params(direction="out")
    ax.tick_params(which="minor", direction="out")
    ax.tick_params(which="minor", left=False)
    # ax.legend(frameon=False, fontsize=5, loc=[0.1, 0.8])
    ax.legend(
        frameon=False,
        fontsize=5.5,
        loc="upper left",
        ncol=2,
        columnspacing=1.0,
        handletextpad=0.0,
    )

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
    ax2.set_ylim([0, 25])
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
    import pandas as pd

    # read excel file beam.xlsx
    df = pd.read_excel("../abaqus.xlsx", sheet_name="Sheet2")

    # convert to numpy array
    data = df.to_numpy()[:]

    # odd columns are force, even columns are displacement
    dispx = data[:, 0::3]
    dispy = data[:, 1::3]
    force = data[:, 2::3]

    dispx[:, 0] = -dispx[:, 0]

    disp = np.abs(dispx)
    disp = dispx

    slopes = []
    for i in range(force.shape[1]):
        slopes.append(np.max(np.abs(np.diff(force[:15, i]) / np.diff(disp[:15, i]))))

    return force, disp, slopes


def plot_loaddis(force, disp, slope, ax):
    cw = plt.colormaps["coolwarm"](np.linspace(0, 1, 10))
    c = ["k", "#e29400ff", cw[0], cw[-1], "m", "c", "y"]
    mkz = 2.0
    lw = 0.75
    mk = ["o", "s", "^", "v", "D", "P"]
    ds = [
        r"$\bar{h}=\text{N/A}$",
        r"$\bar{h}=0.7h_{KS}$",
        r"$\bar{h}=10$",
        r"$\bar{h}=0$",
    ]
    end = [60, 60, 60, 50]

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
        # bbox_to_anchor=(0.995, 0.15),
        loc="upper left",
        ncol=2,
        # columnspacing=1.0,
    )
    ax.set_ylim(ymax=13.9)
    ax.set_xlim(xmax=0.18)

    ax.set_xlabel(r"Horizontal Displacement $d_x$ for the Top-Middle Node", fontsize=7)
    ax.set_ylabel(r"$\lambda$", fontsize=7)
    ax.yaxis.set_label_coords(-0.025, 0.5)
    ax.tick_params(axis="both", which="major", length=2, labelsize=6)

    return


a1 = [12.50, 14.19, 14.78, 15.04, 15.61, 15.66]
b1 = [13.749, 14.057, 14.303, 15.164, 15.802, 16.634]

a2 = [12.823, 13.728, 13.909, 13.909, 14.424, 14.508]
b2 = [12.820, 13.331, 13.434, 13.649, 13.762, 14.134]

a3 = [12.68, 13.47, 13.70, 13.86, 14.72, 14.90]
# b2 = [13.213, 13.50, 13.872, 14.346, 14.617, 15.319]
# b2 = [12.601, 12.789, 13.479, 13.812, 14.226, 15.021]
b3 = [13.529, 13.724, 13.942, 14.193, 14.875, 14.924]

a4 = [10.89, 11.79, 12.30, 12.40, 12.88, 12.90]
b4 = [11.603, 11.802, 12.619, 12.764, 13.316, 13.957]

rpd = np.mean(
    [
        np.abs(np.subtract(a1, b1)) / np.add(a1, b1) * 100,
        np.abs(np.subtract(a2, b2)) / np.add(a2, b2) * 100,
        np.abs(np.subtract(a3, b3)) / np.add(a3, b3) * 100,
        np.abs(np.subtract(a4, b4)) / np.add(a4, b4) * 100,
    ]
)
ic(rpd)

f11, d11, slope1 = read_data("column-a-0001.txt", 100, b1[0])
f12, d12, _ = read_data("column-a-001.txt", 100, b1[0])
f13, d13, _ = read_data("column-a-003.txt", 60, b1[0])

f21, d21, slope2 = read_data("c-d-0001.txt", 60, b2[0])
f22, d22, _ = read_data("c-d-002.txt", 60, b2[0])
f23, d23, _ = read_data("c-d-003.txt", 50, b2[0])

f31, d31, slope3 = read_data("c-b-0001.txt", 60, b3[0])
f32, d32, _ = read_data("c-b-001.txt", 30, b3[0])
f33, d33, _ = read_data("c-b-002.txt", 30, b3[0])

f41, d41, slope4 = read_data("loadpath-c-0001.txt", 50, b4[0])
f42, d42, _ = read_data("loadpath-c-002.txt", 32, b4[0])
f43, d43, _ = read_data("loadpath-c-005.txt", 26, b4[0])

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
        offset = [0.08, 0.05, 0.05, 0.05][i]
        n = [[0.1, 1.0, 3.0], [0.1, 2.0, 3.0], [0.1, 1.0, 2.0], [0.1, 2.0, 5.0]][i]
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
    fig[0].delaxes(fig[1])
    plt.savefig("column.png", dpi=500, bbox_inches="tight", pad_inches=0.0)
