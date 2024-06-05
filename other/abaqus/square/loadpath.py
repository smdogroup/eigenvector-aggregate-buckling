from icecream import ic
import matplotlib.pyplot as plt
import numpy as np
import scienceplots

plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "sans-serif",
        "font.sans-serif": "Helvetica",
        "mathtext.fontset": "custom",
    }
)


def plot_rpd(a, b, c, alpha, ax, twinx=False):

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
    ax.set_ylim([0, 16])
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

    if ax.get_subplotspec().colspan.start == 2:
        ax2.set_ylabel(r"$RPD(\%)$", color="gray", fontsize=6)
        ax2.yaxis.set_label_coords(1.08, 0.5)

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


def plot_loadpath(f1, d1, n1, f2, d2, n2, f3, d3, n3, slope, offset, nm1, nm2, ds, ax):
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
        label=str(n1) + nm1,
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
        fontsize=5.5,
        title=f"Initial Imperfection for Design \\textbf{{{ds}}}",
        title_fontsize="6",
        bbox_to_anchor=(0.98, 0.3),
        loc="upper right",
        ncol=2,
        columnspacing=1.0,
        # handletextpad=0.0,  # reduce the space
    )
    ax.tick_params(axis="both", labelsize=5.5)
    ax.set_xlabel("Arc Length", fontsize=6)
    ax.set_ylim([-0.05, 1.02])
    # if ax.get_subplotspec().colspan.start == 0:
    ax.set_ylabel(nm2, fontsize=6)
    if ax.get_subplotspec().colspan.start == 0:
        ax.yaxis.set_label_coords(-0.09, 0.5)
    else:
        ax.yaxis.set_label_coords(-0.02, 0.5)

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
    df = pd.read_excel("../abaqus.xlsx", sheet_name="Sheet3")

    # convert to numpy array
    data = df.to_numpy()[:]

    # odd columns are force, even columns are displacement
    disp = np.abs(data[:, 0::2])
    force = data[:, 1::2]

    slopes = []
    for i in range(force.shape[1]):
        slopes.append(np.max(np.abs(np.diff(force[:15, i]) / np.diff(disp[:15, i]))))

    return force, disp, slopes


def plot_loaddis(force, disp, slope, ax):
    cw = plt.colormaps["coolwarm"](np.linspace(0, 1, 10))
    c = ["k", cw[0], cw[-1], "m", "c", "y"]
    mkz = 2.0
    lw = 0.75
    mk = ["o", "s", "^", "v", "D", "P"]
    end = [40, 50, 40]
    text = [
        r"$0.5\% (\phi_1 + \phi_2) \ \text{for} \ \bar{h}=\text{N/A}$",
        r"$0.5\% (\phi_3 + \phi_4) \ \text{for} \ \bar{h}=6.5$",
        r"$0.5\% (\phi_1 + \phi_2) \ \text{for} \ \bar{h}=4.5$",
    ]

    for i in range(force.shape[1]):
        ax.plot(
            disp[: end[i], i],
            force[: end[i], i],
            marker=mk[i],
            label=text[i],
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
        title=r"Initial Imperfection",
        title_fontsize="7",
        columnspacing=1.0,
    )

    ax.set_xlabel(r"Vertical Displacement $d_y$ for the Top-Middle Node", fontsize=7)
    ax.set_ylabel(r"$\lambda$", fontsize=7)
    # ax.set_xlim(left=-0.02)
    ax.set_ylim(ymax=9.9)
    ax.tick_params(axis="both", labelsize=6)

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
    fig = plt.subplots(
        figsize=(7.5, 5.5),
        tight_layout=True,
    )
    row_ratio = [2, 4, 5]
    total_rows = sum(row_ratio)

    for i in range(3):
        a = [a1, a2, a3][i]
        b = [b1, b2, b3][i]
        c = [c1, c2, c3][i]
        alpha = [alpha1, alpha2, alpha3][i]
        twinx = [False, False, True][i]

        ax0 = plt.subplot2grid((total_rows, 3), (0, i), rowspan=row_ratio[0])
        plot_rpd(a, b, c, alpha, ax0, twinx)

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
        ds = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)"][i]
        ax1 = plt.subplot2grid((total_rows, 3), (row_ratio[0], i), rowspan=row_ratio[1])
        plot_loadpath(
            f1, d1, n[0], f2, d2, n[1], f3, d3, n[2], slope, offset, nm1, nm2, ds, ax1
        )

        if i != 0:
            ax0.set_yticklabels([])
            ax1.set_yticklabels([])

    plt.subplots_adjust(wspace=0.4)

    ax = plt.subplot2grid(
        (total_rows, 3),
        (row_ratio[0] + row_ratio[1], 0),
        colspan=4,
        rowspan=row_ratio[2],
    )
    force, disp, slope = read_xlsx()
    plot_loaddis(force, disp, slope, ax)
    fig[0].delaxes(fig[1])
    plt.savefig("square.png", dpi=500, bbox_inches="tight", pad_inches=0.0)
