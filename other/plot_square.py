from icecream import ic
from matplotlib import cm, colors, patches
from matplotlib.collections import PolyCollection
import matplotlib.pylab as plt
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from mpl_toolkits.mplot3d import Axes3D, axes3d
import mpl_toolkits.mplot3d.art3d as art3d
import numpy as np
from paretoset import paretoset
import scienceplots
from scipy.optimize import curve_fit
from matplotlib import pyplot


def read_size(options, vtk):
    nnodes, m, n = 0, 0, 0
    with open(options) as f:
        for num, line in enumerate(f, 1):
            # if prefix in line: jump to next line
            if "prefix" in line:
                continue
            if "nx" in line:
                m = int(line.split()[1])
    with open(vtk) as f:
        for num, line in enumerate(f, 1):
            if "POINTS" in line:
                nnodes = int(line.split()[1])
    n = np.sqrt(nnodes).astype(int)
    m = n
    return nnodes, m, n


def read_vol(file):
    vol, dis = [], []
    # total_cols = len(np.loadtxt(file, skiprows=8, max_rows=1)) - 2
    total_cols = 5
    total_lines = sum(1 for line in open(file))
    with open(file) as f:
        for num, line in enumerate(f, 1):
            if num == total_lines - 2:
                break
            if "iter" in line:
                a = np.loadtxt(
                    file, skiprows=num, max_rows=10, usecols=range(0, total_cols)
                )
                vol = np.append(vol, a)
                b = np.loadtxt(file, skiprows=num, max_rows=10, usecols=range(6, 7))
                dis = np.append(dis, b)

    ic(total_cols, len(vol))
    n_iter = len(vol) // total_cols
    vol = vol.reshape(n_iter, total_cols)
    stress_iter = vol[:, 4] ** 0.5 * 1e-6

    BLF_ks = vol[:, 3]
    ic(BLF_ks.shape)
    compliance = vol[:, -1]
    vol = vol[:, 2]
    ic(compliance.shape, vol.shape)
    # ic(dis.shape, vol.shape)
    return vol, BLF_ks, compliance, dis, stress_iter


def read_freq(file, n):
    omega = []
    with open(file) as f:
        for num, line in enumerate(f, 1):
            if "iter" in line:
                a = np.loadtxt(file, skiprows=num, max_rows=10)
                omega = np.append(omega, a)

    n_iter = len(omega) // (n + 1)
    omega = omega.reshape(n_iter, (n + 1))
    omega = omega[:, 1:]
    return omega


def read_vtk(file, nnodes, m, n):
    rho, stress, phi0, phi1, phi2, phi3, phi4, phi5 = [], [], [], [], [], [], [], []
    with open(file) as f:
        for num, line in enumerate(f, 1):
            if "rho" in line:
                rho = np.loadtxt(file, skiprows=num + 1, max_rows=nnodes)
            if "eigenvector_stress" in line:
                stress = np.loadtxt(file, skiprows=num + 1, max_rows=(m - 1) * (n - 1))
            if "phi0" in line:
                phi0 = np.loadtxt(file, skiprows=num, max_rows=nnodes)
            if "phi1" in line:
                phi1 = np.loadtxt(file, skiprows=num, max_rows=nnodes)
            if "phi2" in line:
                phi2 = np.loadtxt(file, skiprows=num, max_rows=nnodes)
            if "phi3" in line:
                phi3 = np.loadtxt(file, skiprows=num, max_rows=nnodes)
            if "phi4" in line:
                phi4 = np.loadtxt(file, skiprows=num, max_rows=nnodes)
            if "phi5" in line:
                phi5 = np.loadtxt(file, skiprows=num, max_rows=nnodes)

    rho = rho.reshape(n, m)
    # stress = stress.reshape(n - 1, m - 1)
    # stress = stress**0.5
    phi0 = phi0.reshape(n, m, 3)
    phi1 = phi1.reshape(n, m, 3)
    phi2 = phi2.reshape(n, m, 3)
    phi3 = phi3.reshape(n, m, 3)
    phi4 = phi4.reshape(n, m, 3)
    phi5 = phi5.reshape(n, m, 3)

    return rho, stress, phi0, phi1, phi2, phi3, phi4, phi5


def read_data(dir_vtk, dir_freq, dir_stdout, dir_options):
    nnodes, m, n = read_size(dir_options, dir_vtk)
    omega = read_freq(dir_freq, 6)
    rho, stress, phi0, phi1, phi2, phi3, phi4, phi5 = read_vtk(dir_vtk, nnodes, m, n)
    vol, BLF_ks, compliance, dis, stress_iter = read_vol(dir_stdout)
    return (
        rho,
        vol,
        BLF_ks,
        compliance,
        dis,
        stress_iter,
        stress,
        omega,
        phi0,
        phi1,
        phi2,
        phi3,
        phi4,
        phi5,
    )


def assmble_data(ncol, iter, iter2=None, iter3=None):
    (
        rho,
        vol,
        BLF_ks,
        compliance,
        dis,
        stress_iter,
        stress,
        omega,
        phi0,
        phi1,
        phi2,
        phi3,
        phi4,
        phi5,
    ) = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )

    for i in range(1, ncol + 1):
        if i == 2 and iter2 is not None:
            exec(f"dir_vtk{i} = dir_result{i} + 'vtk/it_{iter2}.vtk'")
        elif i == 3 and iter3 is not None:
            exec(f"dir_vtk{i} = dir_result{i} + 'vtk/it_{iter3}.vtk'")
        else:
            exec(f"dir_vtk{i} = dir_result{i} + 'vtk/it_{iter}.vtk'")
        exec(f"dir_freq{i} = dir_result{i} + 'frequencies.log'")
        exec(f"dir_options{i} = dir_result{i} + 'options.txt'")
        exec(f"dir_stdout{i} = dir_result{i} + 'stdout.log'")
        exec(
            f"rho{i}, vol{i}, BLF_ks{i}, compliance{i}, dis{i},stress_iter{i}, stress{i}, omega{i}, phi0_{i}, phi1_{i}, phi2_{i}, phi3_{i}, phi4_{i}, phi5_{i} = [],[], [], [], [], [], [], [], [], [], [], [], [], []"
        )
        exec(
            f"rho{i}, vol{i}, BLF_ks{i}, compliance{i}, dis{i}, stress_iter{i}, stress{i}, omega{i}, phi0_{i}, phi1_{i}, phi2_{i},phi3_{i}, phi4_{i}, phi5_{i} = read_data(dir_vtk{i}, dir_freq{i}, dir_stdout{i}, dir_options{i})"
        )
        exec(f"rho.append(rho{i})")
        exec(f"vol.append(vol{i})")
        exec(f"BLF_ks.append(BLF_ks{i})")
        exec(f"compliance.append(compliance{i})")
        exec(f"dis.append(dis{i})")
        exec(f"stress_iter.append(stress_iter{i})")
        exec(f"stress.append(stress{i})")
        exec(f"omega.append(omega{i})")
        exec(f"phi0.append(phi0_{i})")
        exec(f"phi1.append(phi1_{i})")
        exec(f"phi2.append(phi2_{i})")
        exec(f"phi3.append(phi3_{i})")
        exec(f"phi4.append(phi4_{i})")
        exec(f"phi5.append(phi5_{i})")
    return (
        rho,
        vol,
        BLF_ks,
        compliance,
        dis,
        stress_iter,
        stress,
        omega,
        phi0,
        phi1,
        phi2,
        phi3,
        phi4,
        phi5,
    )


def plot_modeshape(
    ax,
    rho,
    phi=None,
    stress=None,
    dis=None,
    alpha=None,
    flip_x=False,
    flip_y=False,
    zoom=False,
    surface=False,
    levels=20,
):
    Z = rho
    cmap = "Greys"
    # vmin = np.min(Z)
    # vmax = np.max(Z)
    vmin = 0.4
    vmax = 0.8

    if phi is None:
        phi = np.zeros((rho.shape[0], rho.shape[1], 2))

    if alpha is None:
        alpha = 1.0

    X, Y = np.meshgrid(np.linspace(0, 1, Z.shape[1]), np.linspace(0, 1, Z.shape[0]))
    X = X + phi[:, :, 0]
    Y = Y + phi[:, :, 1]

    ax.contourf(
        X,
        Y,
        Z,
        levels=levels,
        alpha=alpha,
        antialiased=True,
        # cmap=colors.ListedColormap(["lightgrey", "grey", "black"]),
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines[:].set_visible(False)
    ax.set_aspect("equal")

    # ax.plot(X, Y, color="black", linewidth=0.1, alpha=0.25)
    # ax.plot(X.T, Y.T, color="black", linewidth=0.1, alpha=0.25)

    # if flip_y:
    #     ax.scatter(4, 1 - phi[98, 399, 1], color="k", s=0.5, zorder=10)
    # else:
    #     ax.scatter(4, 1 + phi[98, 399, 1], color="k", s=0.5, zorder=10)

    # add a straight line at midspan
    # ax.plot([0, 8], [1, 1], color="black", linestyle="--", linewidth=0.25)

    if surface:
        vmin = 0.1
        vmax = 1.0
        Z_max = 0.10567305818515133
        Z = np.sqrt(phi[:, :, 0] ** 2 + phi[:, :, 1] ** 2)
        ic(np.min(Z), np.max(Z))
        Z = (Z - np.min(Z)) / (Z_max - np.min(Z))
        # ax.plot(X, Y, color="k", linewidth=0.2, alpha=0.1)
        # ax.plot(X.T, Y.T, color="k", linewidth=0.2, alpha=0.1)
        ax.contourf(
            X,
            Y,
            Z,
            levels=50,
            alpha=0.75,
            antialiased=True,
            cmap="coolwarm",
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines[:].set_visible(False)
        ax.set_aspect("equal")

    if zoom:
        ax.plot(X, Y, color="black", linewidth=0.1, alpha=0.75)
        ax.plot(X.T, Y.T, color="black", linewidth=0.1, alpha=0.75)
        ax.set_ylim(np.min(Y), 0.35 + np.min(Y))
        ax.set_xlim(2.5, 5.5)


def plot_1(nrow, rho, phi0, stress=None, flip_x=False, flip_y=False):
    fig, axs = plt.subplots(constrained_layout=True)
    # plot_modeshape(axs, rho[0], levels=5)
    shift_scale = 0.03
    plot_modeshape(
        axs,
        rho[0],
        # phi0[0] * shift_scale,
        levels=10,
    )
    axs.scatter(
        [
            0.75,
            0.75,
            0.25,
            0.25,
        ],
        [
            0.75,
            0.25,
            0.75,
            0.25,
        ],
        marker="o",
        color="orange",
        s=10.0,
        zorder=10,
    )


def plot_3(nrow, rho, phi0):
    fig, axs = plt.subplots()
    shift_scale = 0.03

    plot_modeshape(
        axs,
        rho[0],
        phi0[0] * shift_scale,
        surface=True,
        levels=10,
    )
    a = int(300 * 0.75)
    b = int(300 * 0.25)
    d2 = np.sqrt(phi0[0][b, b][0] ** 2 + phi0[0][b, b][1] ** 2)
    d1 = np.sqrt(phi0[0][a, b][0] ** 2 + phi0[0][a, b][1] ** 2)
    ic(d1, d2)
    ic((d1**2 + d2**2) * 0.5)

    axs.scatter(
        [
            0.75 + shift_scale * phi0[0][a, a][0],
            0.75 + shift_scale * phi0[0][a, b][0],
            0.25 + shift_scale * phi0[0][b, a][0],
            0.25 + shift_scale * phi0[0][b, b][0],
        ],
        [
            0.75 + shift_scale * phi0[0][a, a][1],
            0.25 + shift_scale * phi0[0][a, b][1],
            0.75 + shift_scale * phi0[0][b, a][1],
            0.25 + shift_scale * phi0[0][b, b][1],
        ],
        marker="o",
        color="orange",
        s=5.0,
        zorder=10,
    )

    axs.scatter(
        [
            0.75,
            0.75,
            0.25,
            0.25,
        ],
        [
            0.75,
            0.25,
            0.75,
            0.25,
        ],
        marker="o",
        color="k",
        s=5.0,
        zorder=10,
    )

    # # add arrows to show the displacement
    # axs.arrow(
    #     0.75,
    #     0.75,
    #     shift_scale * phi0[0][a, a][0],
    #     shift_scale * phi0[0][a, a][1],
    #     head_width=0.01,
    #     head_length=0.01,
    #     fc="k",
    #     ec="k",
    #     zorder=20,
    #     length_includes_head=True,
    # )
    # axs.arrow(
    #     0.75,
    #     0.25,
    #     shift_scale * phi0[0][a, b][0],
    #     shift_scale * phi0[0][a, b][1],
    #     head_width=0.01,
    #     head_length=0.01,
    #     fc="k",
    #     ec="k",
    #     zorder=20,
    #     length_includes_head=True,
    # )
    # axs.arrow(
    #     0.25,
    #     0.75,
    #     shift_scale * phi0[0][b, a][0],
    #     shift_scale * phi0[0][b, a][1],
    #     head_width=0.01,
    #     head_length=0.01,
    #     fc="k",
    #     ec="k",
    #     zorder=20,
    #     length_includes_head=True,
    # )
    # axs.arrow(
    #     0.25,
    #     0.25,
    #     shift_scale * phi0[0][b, b][0],
    #     shift_scale * phi0[0][b, b][1],
    #     head_width=0.01,
    #     head_length=0.01,
    #     fc="k",
    #     ec="k",
    #     zorder=20,
    #     length_includes_head=True,
    # )

    # # add a straight line at midspan
    # axs.plot([0, 1], [0.5, 0.5], color="black", linestyle="--", linewidth=0.25)
    # axs.plot([0.5, 0.5], [0, 1], color="black", linestyle="--", linewidth=0.25)

    # # add the displacement for the arrows
    # axs.text(
    #     0.75 + shift_scale * phi0[0][a, a][0] + 0.01,
    #     0.75 + shift_scale * phi0[0][a, a][1] + 0.01,
    #     f"{d2:.2f}",
    #     fontsize=6,
    # )
    # axs.text(
    #     0.75 + shift_scale * phi0[0][a, b][0] + 0.01,
    #     0.25 + shift_scale * phi0[0][a, b][1] + 0.01,
    #     f"{d1:.2f}",
    #     fontsize=6,
    # )
    # axs.text(
    #     0.25 + shift_scale * phi0[0][b, a][0] + 0.01,
    #     0.75 + shift_scale * phi0[0][b, a][1] + 0.01,
    #     "d=" + f"{d1:.2f}",
    #     fontsize=6,
    # )
    # axs.text(
    #     0.25 + shift_scale * phi0[0][b, b][0] + 0.01,
    #     0.25 + shift_scale * phi0[0][b, b][1] + 0.01,
    #     f"{d2:.2f}",
    #     fontsize=6,
    # )


def plot_5(nrow, rho, phi0, stress, flip_x=False, flip_y=False):
    fig, axs = plt.subplots(1, 7, figsize=(10, 2), constrained_layout=True)

    shift_scale = -0.02
    plot_modeshape(
        axs[0],
        rho[0],
        flip_x=1,
        flip_y=flip_y,
        levels=1,
    )
    axs[0].scatter(7.5, 1.01039, color="orange", s=0.2, zorder=10)

    shift_scale = -0.02
    for i in range(6):
        phi = eval(f"phi{i}")
        plot_modeshape(
            axs[i + 1],
            rho[0],
            phi[0] * shift_scale,
            flip_x=1,
            flip_y=flip_y,
            levels=1,
        )
        axs[1].scatter(
            7.5 - shift_scale * phi[0][242, 120][0],
            1.01039 - shift_scale * phi[0][242, 120][1],
            color="orange",
            s=0.2,
            zorder=10,
        )

    for ax in axs[:]:
        ax.set_anchor("S")

        # draw a line at y=1.01039
    # axs[0].plot([0, 8], [1.01039, 1.01039], color="black", linestyle="--", linewidth=0.25)


def plot_2_1(omega, BLF_ks, vol, compliance, dis):
    fig, ax = plt.subplots()
    ax2 = ax.twinx()
    ax2.spines.right.set_position(("axes", 1.0))

    styles = ["-", "--", "-.", ":", "-", "--", "-.", ":"]
    alpha = [0.8, 0.5, 0.4, 0.3, 0.2, 0.1]
    # marker = ["o", "o", "s", "s", "^", "^"]
    linewidth = [1.0, 0.5, 0.4, 0.3, 0.2, 0.1]
    n_start = 0
    n_iter = 1000

    # plot BLF_ks
    (p1,) = ax.plot(
        BLF_ks[0][n_start:n_iter],
        label=r"$J_{ks}^{-1} [\mu_i]$",
        color="k",
        alpha=0.8,
        linewidth=0.75,
        linestyle="--",
    )
    (p2,) = ax.plot(
        omega[0][n_start:n_iter, 0],
        label=f"$BLF_{1}$",
        alpha=0.8,
        color="k",
        linewidth=0.75,
    )

    (pc1,) = ax2.plot(
        compliance[0][n_start:n_iter] / (7.4e-06*4.3/4),
        color="b",
        linewidth=0.75,
        alpha=0.6,
        label=r"$g_{c}$",
        linestyle="-",
        zorder=0,
    )

    # (pc2,) = ax2.plot(vol[0][n_start:n_iter] / 0.3, color="b", linewidth=0.75, alpha=0.6, label=r'$g_V$',linestyle="--",zorder=0)

    # (p3,) = ax2.plot(dis[0][n_start:n_iter] + 1.0, color="r", linewidth=0.25, alpha=1.0, label=r'$g_{d}$',linestyle="-",zorder=10)

    for i in range(1, 6):
        ax.plot(
            omega[0][n_start:n_iter, i],
            label=f"$BLF_{i+1}$",
            color="k",
            alpha=alpha[i],
            linestyle=styles[i],
            linewidth=0.5,
            # marker=marker[i],
            # markevery=10,
            # markersize=3,
            # markerfacecolor="none",
            # markeredgecolor=colors[i],
        )
    # log scale the x axis
    ax.set(xscale="log", xlim=(20, 1000))
    # ax.set(xlim=(10, 1000))
    ax.set_xlabel("Iteration (log-scale)")
    # ax.legend(loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.xaxis.set_ticks_position("bottom")
    ax.yaxis.set_ticks_position("left")
    ax.tick_params(direction="out")
    ax.tick_params(which="minor", direction="out")

    ax.set_ylim(0, 17.95)
    ax.set_ylabel("BLF ($BLF_i$)", rotation=0, labelpad=0)
    ax.yaxis.set_label_coords(0.0, 1.001)

    ax2.set(ylim=(0.0, 3.95))
    ax2.set_ylabel("$c / c_{opt}$", rotation=0, labelpad=0)
    ax2.yaxis.set_label_coords(1.02, 1.051)

    ax2.yaxis.label.set_color(pc1.get_color())
    ax.yaxis.label.set_color(p1.get_color())
    ax2.yaxis.label.set_color(pc1.get_color())
    ax.tick_params(axis="y", colors=p1.get_color())
    ax2.tick_params(axis="y", colors=pc1.get_color())

    ax.spines["top"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    ax.xaxis.set_ticks_position("bottom")
    ax.yaxis.set_ticks_position("left")
    ax.spines["right"].set_visible(False)
    ax2.yaxis.set_ticks_position("right")

    ax.tick_params(direction="out")
    ax2.tick_params(direction="out")
    ax.tick_params(which="minor", direction="out")
    ax2.tick_params(which="minor", direction="out")
    ax.tick_params(which="minor", left=False)

    ax.margins(0.0)
    ax2.margins(0.0)

    handles, labels = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    handles.extend(handles2)
    labels.extend(labels2)
    ax.legend(
        handles,
        labels,
        # title="Buckling Load Factors:",
        ncol=3,
        # loc=[0.5, 0.05],
        loc=[0.35, 0.05],
        frameon=False,
        fontsize=6,
    )


def plot_2(omega, BLF_ks, vol, compliance, dis):
    fig, ax = plt.subplots()
    ax2 = ax.twinx()
    ax2.spines.right.set_position(("axes", 1.0))

    ax3 = ax.twinx()
    ax3.spines.right.set_position(("axes", 1.1))

    styles = ["-", "--", "-.", ":", "-", "--", "-.", ":"]
    alpha = [0.8, 0.5, 0.4, 0.3, 0.2, 0.1]
    # marker = ["o", "o", "s", "s", "^", "^"]
    linewidth = [1.0, 0.5, 0.4, 0.3, 0.2, 0.1]
    n_start = 0
    n_iter = 1000

    # plot BLF_ks
    (p1,) = ax.plot(
        BLF_ks[0][n_start:n_iter],
        label=r"$J_{ks}^{-1} [\mu_i]$",
        color="k",
        alpha=0.8,
        linewidth=0.75,
        linestyle="--",
    )
    (p2,) = ax.plot(
        omega[0][n_start:n_iter, 0],
        label=f"$BLF_{1}$",
        alpha=0.8,
        color="k",
        linewidth=0.75,
    )

    (pc1,) = ax2.plot(
        compliance[0][n_start:n_iter] / 7.4e-06,
        color="b",
        linewidth=0.75,
        alpha=0.6,
        label=r"$g_{c}$",
        linestyle="-",
        zorder=0,
    )

    # (pc2,) = ax2.plot(vol[0][n_start:n_iter] / 0.3, color="b", linewidth=0.75, alpha=0.6, label=r'$g_V$',linestyle="--",zorder=0)

    (pcc1,) = ax3.plot(
        dis[0][n_start:n_iter],
        color="r",
        linewidth=0.25,
        alpha=1.0,
        label=r"$g_{d}$",
        linestyle="-",
        zorder=10,
    )

    for i in range(1, 6):
        ax.plot(
            omega[0][n_start:n_iter, i],
            label=f"$BLF_{i+1}$",
            color="k",
            alpha=alpha[i],
            linestyle=styles[i],
            linewidth=0.5,
            # marker=marker[i],
            # markevery=10,
            # markersize=3,
            # markerfacecolor="none",
            # markeredgecolor=colors[i],
        )
    # log scale the x axis
    # ax.set(xscale="log", xlim=(30, 1000))
    ax.set_xlabel("Iteration (log-scale)")

    ax.set(xlim=(10, 1000))
    ax.set_xlabel("Iteration")

    # ax.legend(loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.xaxis.set_ticks_position("bottom")
    ax.yaxis.set_ticks_position("left")
    ax.tick_params(direction="out")
    ax.tick_params(which="minor", direction="out")

    ax.set_ylim(2, 8.95)
    ax.set_ylabel("BLF ($BLF_i$)", rotation=0, labelpad=0)
    ax.yaxis.set_label_coords(0.0, 1.001)

    ax2.set(ylim=(0.0, 5.95))
    ax2.set_ylabel("$c / c_{opt}$", rotation=0, labelpad=0)
    ax2.yaxis.set_label_coords(1.02, 1.051)

    # ax3.set(ylim=(-1.0, 0.195))  # d=0
    # ax3.set(ylim=(0.0, 6.695))    # d=6.0
    ax3.set(ylim=(-20, 7))  # d=7.0
    ax3.set_ylabel("$h$", rotation=0, labelpad=0)
    ax3.yaxis.set_label_coords(1.1, 1.045)

    ax.yaxis.label.set_color(p1.get_color())
    ax2.yaxis.label.set_color(pc1.get_color())
    ax3.yaxis.label.set_color(pcc1.get_color())
    ax.tick_params(axis="y", colors=p1.get_color())
    ax2.tick_params(axis="y", colors=pc1.get_color())
    ax3.tick_params(axis="y", colors=pcc1.get_color())

    ax.spines["top"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    ax3.spines["top"].set_visible(False)
    ax.xaxis.set_ticks_position("bottom")
    ax.yaxis.set_ticks_position("left")
    ax.spines["right"].set_visible(False)
    ax2.yaxis.set_ticks_position("right")
    ax3.yaxis.set_ticks_position("right")

    ax.tick_params(direction="out")
    ax2.tick_params(direction="out")
    ax3.tick_params(direction="out")
    ax.tick_params(which="minor", direction="out")
    ax2.tick_params(which="minor", direction="out")
    ax3.tick_params(which="minor", direction="out")
    ax.tick_params(which="minor", left=False)

    handles, labels = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    handles3, labels3 = ax3.get_legend_handles_labels()
    handles.extend(handles2)
    labels.extend(labels2)
    handles.extend(handles3)
    labels.extend(labels3)
    ax.legend(
        handles,
        labels,
        # title="Buckling Load Factors:",
        ncol=3,
        # loc=[0.5, 0.05],
        loc=[0.35, 0.05],
        frameon=False,
        fontsize=6,
    )


def plot_6(omega, BLF_ks, vol, compliance, dis):
    fig, ax = plt.subplots()

    nn = 5
    styles = ["-", "--", "-.", ":", "-", "--", "-.", ":"]
    alpha = [0.8, 0.5, 0.4, 0.3, 0.2, 0.1]
    marker = ["o", "s", "^", "D", "v", "P"]
    c = ["k", "b", "r", "g", "m", "c"]
    linewidth = [1.0, 0.5, 0.4, 0.3, 0.2, 0.1]
    n_start = 0
    n_iter = 1000
    colors = plt.colormaps["coolwarm"](np.linspace(0, 1, 4))
    colors = plt.colormaps["coolwarm"](np.array([0.0, 1 / 3, 2 / 3, 1.0, 0.5]))
    # np.array([0.0, 0.15, 0.30, 0.45]), np.linspace(0, 1, 4)

    rd = np.array(
        [
            np.abs(omega[0][n_start:n_iter, i] - omega[0][n_start:n_iter, i + 1])
            / (0.5 * (omega[0][n_start:n_iter, i] + omega[0][n_start:n_iter, i + 1]))
            for i in range(nn)
        ]
    )

    n = 2
    for i in range(nn):
        ax.scatter(
            np.arange(n_iter - n_start)[::n],
            rd[i, ::n],
            label=f"$(BLF_{i+1}, BLF_{i+2})$",
            color=colors[i],
            # alpha=alpha[i],
            linewidths=0,
            marker=marker[i],
            s=2.0,
            zorder=10,
        )

    plt.plot(
        np.arange(n_iter - n_start)[::n],
        np.min(rd, axis=0)[::n],
        color="k",
        linewidth=0.2,
        linestyle="-",
        label=r"$\text{tracking path:} \ \min(BLF_i, BLF_{i+1})_1$",
        alpha=0.5,
    )

    ax.set_xlabel("Iteration")
    ax.legend(loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.xaxis.set_ticks_position("bottom")
    ax.yaxis.set_ticks_position("left")
    ax.tick_params(direction="out")
    ax.tick_params(which="minor", direction="out")

    ax.set_ylim(0, 1)
    ax.set_ylabel("Relative Difference($BLF_i, BLF_{i+1}$)", labelpad=0)
    # ax.yaxis.set_label_coords(0.0, 1.001)
    ax.set(yscale="log", ylim=(1e-12, 0.2))

    ax.tick_params(direction="out")
    ax.tick_params(which="minor", direction="out")
    ax.tick_params(which="minor", left=False)

    handles, labels = ax.get_legend_handles_labels()
    handles1 = handles[:nn]
    labels1 = labels[:nn]
    reorder = lambda l, nc: sum((l[i::nc] for i in range(nc)), [])
    ax = plt.gca().add_artist(
        ax.legend(
            reorder(handles1, 2),
            reorder(labels1, 2),
            title="Relative Difference:",
            ncol=2,
            # loc=[0.5, 0.05],
            loc=[0.3, 0.5],
            frameon=False,
            fontsize=6,
        )
    )

    plt.legend(
        handles[nn:],
        labels[nn:],
        loc=[0.3, 0.4],
        frameon=False,
        fontsize=6,
    )


if __name__ == "__main__":
    # dir_result1 = "output/final_results/square/v=0.25,w=0.4/"
    dir_result1 = (
        "output/final_results/square/v=0.25,w=0.4,d=6.5/"  # vtk/it_960.vtk for d=4.5
    )

    (
        rho,
        vol,
        BLF_ks,
        compliance,
        dis,
        stress_iter,
        stress,
        omega,
        phi0,
        phi1,
        phi2,
        phi3,
        phi4,
        phi5,
    ) = assmble_data(1, 1000)

    with plt.style.context(["nature"]):
        # plot_1(1, rho, phi1, stress)
        # plot_0()
        # plt.savefig("output/final_results/building/compliance-buckling/building_pf.png", bbox_inches="tight", dpi=1000, pad_inches=0.0)

        # plot_2_1(omega, BLF_ks, vol, compliance, dis)
        # plt.savefig(
        #     "output/final_results/building/building_his.png",
        #     bbox_inches="tight",
        #     dpi=1000,
        #     pad_inches=0.05,
        # )

        # plot_1(1, rho, phi1)
        # plt.savefig(
        #     "output/final_results/square/2.png",
        #     bbox_inches="tight",
        #     dpi=500,
        #     pad_inches=0.02,
        # )

        # plot_3(1, rho, phi0)
        # plt.savefig(
        #     "output/final_results/square/11.png",
        #     bbox_inches="tight",
        #     dpi=500,
        #     pad_inches=0.0,
        # )
        # plot_3(1, rho, phi1)
        # plt.savefig(
        #     "output/final_results/square/12.png",
        #     bbox_inches="tight",
        #     dpi=500,
        #     pad_inches=0.0,
        # )

        # plot_2(omega, BLF_ks, vol, compliance, dis)
        # plt.savefig(
        #     "output/final_results/square/his2.png",
        #     bbox_inches="tight",
        #     dpi=1000,
        #     pad_inches=0.05,
        # )

        plot_6(omega, BLF_ks, vol, compliance, dis)
        plt.savefig(
            "output/final_results/square/dot1.png",
            bbox_inches="tight",
            dpi=1000,
            pad_inches=0.0,
        )
