from icecream import ic
from matplotlib import cm, colors, patches
from matplotlib.collections import PolyCollection
import matplotlib.pylab as plt
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from mpl_toolkits.mplot3d import Axes3D, axes3d
import mpl_toolkits.mplot3d.art3d as art3d
import numpy as np
import pandas as pd
from paretoset import paretoset
import scienceplots
from scipy.optimize import curve_fit

plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "sans-serif",
        "font.sans-serif": "Helvetica",
    }
)


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
    n = nnodes // m
    return nnodes, m, n


def read_vol(file):
    vol, dis = [], []
    # use pandas to read the file
    df = pd.read_csv(file, sep=r"\s+", skiprows=8, header=None)
    df = df.drop(df[df.iloc[:, 0].str.contains("iter")].index)
    df.reset_index(drop=True, inplace=True)
    vol = df.iloc[:, 2].values
    BLF_ks = df.iloc[:, 3].values
    stress_iter = []
    
    ## case 1
    # dis = df.iloc[:, 5].values
    # compliance = df.iloc[:, -3].values

    ## case 2
    dis = df.iloc[:, 6].values
    compliance = df.iloc[:, 4].values

    # convert to numpy array with float
    vol = vol.astype(float)
    BLF_ks = BLF_ks.astype(float)
    compliance = compliance.astype(float)
    dis = dis.astype(float)
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

    if stress is not None:
        # compute true value of stress
        Z = stress

        phi = phi[1:, 1:, :]
        # normalize stress
        ic(np.min(Z), np.max(Z))
        # Z_max = 24.098528321399222
        # Z = Z / Z_max
        # # Z_min = 0.0
        # # Z = (Z - Z.min()) / (Z_max - Z.min())
        # # make the min and max closer
        # a = 0.04
        # Z = Z**a

        # # vmin = 0.1**a
        # # vmax = 0.8**a
        # vmin = 0.72
        # # vmax = 0.7
        # cmap = "coolwarm"

        Z_max = 4.086441349443697
        # Z_min = 6.634634423461437e-05
        # Z = (Z - Z_min) / (Z_max - Z_min)
        # Z = (Z - Z.min()) / (Z_max - Z.min())
        # a = 0.8
        vmin = 0.0
        vmax = 4.086441349443697

        # Z = Z**a
        # Z = Z / Z_max
        # Z_min = 0.0
        # Z = (Z - Z.min()) / (Z_max - Z.min())
        # make the min and max closer
        # a = 0.15
        # Z = Z**a

        # vmin = 0.1**a
        # vmax = 0.8**a
        # vmin = 0.5
        # vmax = 0.7
        cmap = "coolwarm"

    X, Y = np.meshgrid(np.linspace(0, 1, Z.shape[1]), np.linspace(0, 2, Z.shape[0]))
    X = X + phi[:, :, 0]
    Y = Y + phi[:, :, 1]

    if flip_y:
        Y *= -1
        Y += 1
    if flip_x:
        X *= -1
        X += 8

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

    if zoom:
        ax.plot(X, Y, color="black", linewidth=0.1, alpha=0.75)
        ax.plot(X.T, Y.T, color="black", linewidth=0.1, alpha=0.75)
        ax.set_ylim(np.min(Y), 0.35 + np.min(Y))
        ax.set_xlim(2.5, 5.5)


def plot_1(nrow, rho, phi0, stress, flip_x=False, flip_y=False):
    fig, axs = plt.subplots(nrow, 1, figsize=(4, 2), constrained_layout=True)
    # plot_modeshape(axs, rho[0], levels=5)
    shift_scale = -0.03
    plot_modeshape(
        axs,
        rho[0],
        # phi0[0] * shift_scale,
        # stress[0],
        flip_x=1,
        flip_y=flip_y,
        # zoom=True,
        levels=3,
    )
    # add a point at the tip
    # axs.scatter(7.5, 1.01039 , color="orange", s=0.2, zorder=10)

    # axs.scatter(7.5 -shift_scale*phi0[0][242, 120][0], 1.01039 - shift_scale*phi0[0][242, 120][1], color="orange", s=0.2, zorder=10)


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
        label=r"$J_{ks}^{-1} [\lambda_{i}]$",
        color="k",
        alpha=0.8,
        linewidth=0.75,
        linestyle="--",
    )
    (p2,) = ax.plot(
        omega[0][n_start:n_iter, 0],
        label=r"$\lambda_{1}$",
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

    # (p3,) = ax2.plot(dis[0][n_start:n_iter] + 1.0, color="r", linewidth=0.25, alpha=1.0, label=r'$g_{d}$',linestyle="-",zorder=10)

    for i in range(1, 6):
        ax.plot(
            omega[0][n_start:n_iter, i],
            label=r"$\lambda_{" + str(i + 1) + "}$",
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
    ax.set_ylabel(r"$BLF$", rotation=0, labelpad=0)
    ax.yaxis.set_label_coords(0.0, 1.001)

    ax2.set(ylim=(0.0, 3.9))
    y2int = np.arange(0.0, 3.9, 1.0)
    ax2.set_yticks(y2int)
    ax2.set_ylabel("$c / c_{opt}$", rotation=0, labelpad=0)
    ax2.yaxis.set_label_coords(1.01, 1.051)

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
        loc=[0.325, 0.05],
        frameon=False,
        fontsize=6,
    )


def plot_2(omega, BLF_ks, vol, compliance, dis):
    fig, ax = plt.subplots()
    ax2 = ax.twinx()
    ax2.spines.right.set_position(("axes", 1.0))

    ax3 = ax.twinx()
    ax3.spines.right.set_position(("axes", 1.08))

    styles = ["-", "--", "-.", ":", "-", "--", "-.", ":"]
    alpha = [0.8, 0.5, 0.4, 0.3, 0.2, 0.1]
    # marker = ["o", "o", "s", "s", "^", "^"]
    linewidth = [1.0, 0.5, 0.4, 0.3, 0.2, 0.1]
    n_start = 0
    n_iter = 1000

    # plot BLF_ks
    (p1,) = ax.plot(
        BLF_ks[0][n_start:n_iter],
        label=r"$J_{ks}^{-1} [\lambda_{i}]$",
        color="k",
        alpha=0.8,
        linewidth=0.75,
        linestyle="--",
    )
    (p2,) = ax.plot(
        omega[0][n_start:n_iter, 0],
        label=r"$\lambda_{1}$",
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
            label=r"$\lambda_{" + str(i + 1) + "}$",
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
    ax.set(xscale="log", xlim=(30, 1000))
    ax.set_xlabel("Iteration (log-scale)")

    # ax.set(xlim=(10, 1000))
    # ax.set_xlabel("Iteration")

    # ax.legend(loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.xaxis.set_ticks_position("bottom")
    ax.yaxis.set_ticks_position("left")
    ax.tick_params(direction="out")
    ax.tick_params(which="minor", direction="out")

    ax.set_ylim(0, 17.95)
    ax.set_ylabel(r"$BLF$", rotation=0, labelpad=0)
    ax.yaxis.set_label_coords(0.0, 1.001)

    ax2.set(ylim=(0.0, 3.9))
    y2int = np.arange(0.0, 3.9, 1.0)
    ax2.set_yticks(y2int)
    ax2.set_ylabel("$c / c_{opt}$", rotation=0, labelpad=0)
    ax2.yaxis.set_label_coords(1.01, 1.051)

    ticks = ax3.get_yticks()
    ticks = np.array([0.0, 7.0])
    ax3.set_yticks(ticks)
    # ax3.set(ylim=(-1.0, 0.195))  # d=0
    # ax3.set(ylim=(0.0, 6.695))    # d=6.0
    ax3.set(ylim=(-60.0, 15))  # d=7.0
    ax3.set_ylabel("$h$", rotation=0, labelpad=0)
    ax3.yaxis.set_label_coords(1.08, 1.045)

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
        loc=[0.325, 0.05],
        frameon=False,
        fontsize=6,
    )


def plot_0():
    # w = np.arange(0, 1.1, 0.1)
    fig, ax = plt.subplots(figsize=(7.48, 4))
    lam = np.array(
        [12.29, 13.21, 12.92, 13.16, 12.13, 12.24, 11.28, 10.02, 8.72, 6.81, 1.52]
    )
    c = np.array(
        [
            1.86e-5,
            1.32e-5,
            1.22e-5,
            1.14e-5,
            1.14e-5,
            1.06e-5,
            1.01e-5,
            9.77e-6,
            9.40e-6,
            8.23e-6,
            7.39e-6,
        ]
    )
    plt.plot(c, 1 / lam, "o-", color="k", markersize=3, linewidth=0.75)
    plt.ylabel(r"$1 / \lambda_1$")
    plt.xlabel("$c$")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_3():
    # load only first 3 columns "other/building.txt"
    data = np.loadtxt("other/building.txt")
    # data = data[:66, :]
    delete = np.array(
        [
            # 5.0,
            # 6.0,
            # 7.0,
            # 8.0,
            # 9.0,
            # 10.0,
            # 11.0,
            # 12.0,
            # 13.0,
            # 14.0,
            # 15.0,
            # 16.0,
            # 17.0,
            # 18.0,
            # 19.0,
            # 20.0,
        ]
    )

    for i in range(0, len(delete)):
        data = data[data[:, 3] != delete[i]]
    # only store the data for w != 1.0
    # data = data[data[:, 0] != 1.0]

    x = data[:, 3]  # displacement
    y = 1 / data[:, 1]  # 1 / BLF
    z = data[:, 2]  # compliance
    ic(x)

    def polygon_under_graph(xx, yy):
        """
        Construct the vertex list which defines the polygon filling the space under
        the (x, y) line graph. This assumes x is in ascending order.
        """
        return [
            (np.min(xx) - 0.0001 * np.min(xx), np.max(z)),
            # (xx[-1], np.max(z)),
            # (np.min(xx) - 0.0001 * np.min(xx), yy[-1]),
            *zip(xx, yy),
        ]

    data_xyz = np.array([x, y, z]).T
    # sort by x
    data_xyz = data_xyz[data_xyz[:, 0].argsort()]

    ax = plt.figure().add_subplot(projection="3d")
    ax.figure.set_size_inches(1.8 * 7.48, 1.8 * 4)

    # compute pareto front
    # def is_pareto_efficient_dumb(costs):
    #     """
    #     Find the pareto-efficient points
    #     :param costs: An (n_points, n_costs) array
    #     :return: A (n_points, ) boolean array, indicating whether each point is Pareto efficient
    #     """
    #     is_efficient = np.ones(costs.shape[0], dtype=bool)
    #     for i, c in enumerate(costs):
    #         is_efficient[i] = np.all(np.any(costs >= c, axis=1))
    #     return is_efficient

    def is_pareto_efficient_dumb(costs):
        """
        Find the pareto-efficient points
        :param costs: An (n_points, n_costs) array
        :return: A (n_points, ) boolean array, indicating whether each point is Pareto efficient
        """
        is_efficient = np.ones(costs.shape[0], dtype=bool)
        for i, c in enumerate(costs):
            is_efficient[i] = np.all(np.any(costs[:i] > c, axis=1)) and np.all(
                np.any(costs[i + 1 :] > c, axis=1)
            )
        return is_efficient

    data_xyz_paret = np.empty((0, 3), float)
    data_new = np.empty((12, 3), float)
    for d in np.unique(data_xyz[:, 0]):
        mu_d = data_xyz[data_xyz[:, 0] == d][:, 1]
        compile_d = data_xyz[data_xyz[:, 0] == d][:, 2]
        data_new[:, 1:3] = polygon_under_graph(mu_d, compile_d)
        data_new[:, 0] = d
        is_efficient = paretoset(data_new[:, 1:3])
        # is_efficient = is_pareto_efficient_dumb(data_new)
        data_xyz_paret = np.append(data_xyz_paret, data_new[is_efficient], axis=0)

        # find pareto front for each displacement
        # is_efficient = is_pareto_efficient_dumb(data_xyz[data_xyz[:, 0] == d])
        # data_xyz_paret = np.append(
        #     data_xyz_paret, data_xyz[data_xyz[:, 0] == d][is_efficient], axis=0
        # )

    # plot the pareto front surface
    ax.plot_trisurf(
        data_xyz_paret[:, 0],
        data_xyz_paret[:, 1],
        data_xyz_paret[:, 2],
        color="k",
        alpha=0.5,
    )

    def polygon_under_graph(x, y):
        """
        Construct the vertex list which defines the polygon filling the space under
        the (x, y) line graph. This assumes x is in ascending order.
        """
        return [(x[0], 0.0), *zip(x, y), (x[-1], 0.0)]

    def polygon_upper_graph(xx, yy):
        """
        Construct the vertex list which defines the polygon filling the space under
        the (x, y) line graph. This assumes x is in ascending order.
        """
        return [(xx[0], np.max(z)), *zip(xx, yy)]

    def polygon_upper_graph2(xx, yy):
        """
        Construct the vertex list which defines the polygon filling the space under
        the (x, y) line graph. This assumes x is in ascending order.
        """
        return [(xx[0], np.max(z)), *zip(xx, yy), (xx[-1], np.max(z))]

    # facecolors = plt.colormaps["viridis_r"](np.linspace(0, 1, 21))
    facecolors = plt.colormaps["coolwarm"](
        np.linspace(0, 1, np.unique(data[:, 3]).size - 1)
    )
    ic(facecolors)
    # add black color to the end of the facecolors
    black = np.array([[0, 0, 0, 1]])
    facecolors = np.append(facecolors, black, axis=0)

    def func(x, a, b, c):
        return a * np.exp(-b * x**3) + c

    j = 0
    for d in np.unique(data[:, 3]):
        data_x = np.empty((0, 2), float)

        for i in range(0, len(x)):
            if data[i, 3] == d:
                data_x = np.append(data_x, np.array([[y[i], z[i]]]), axis=0)

        ax.scatter(
            d,
            data_x[:, 0],
            data_x[:, 1],
            c=facecolors[int(j)],
            s=10.0,
            zorder=d + 1,
            alpha=1.0,
        )

        verts = polygon_upper_graph(data_x[:-1, 0], data_x[:-1, 1])
        # convert the vertex list to a polygon
        verts = np.array(verts)
        # curve fiting

        # popt, pcov = curve_fit(func, verts[:, 0], verts[:, 1])

        # verts[:, 1] = func(verts[:, 0], *popt)
        verts = [*zip(verts[:, 0], verts[:, 1]), (data_x[-1, 0], data_x[-1, 1])]
        verts = np.array(verts)
        verts = polygon_upper_graph2(verts[:, 0], verts[:, 1])
        poly = PolyCollection(
            [verts],
            facecolors=facecolors[int(j)],
            alpha=0.7,
            zorder=d + 1,
            # orientation="horizontal",
        )
        ax.add_collection3d(poly, zs=d, zdir="x")
        j += 1

    ax.set(
        xlim=(np.min(x), np.max(x)),
        ylim=(np.min(y), np.max(y)),
        zlim=(1.05 * np.min(z), 1.02 * np.max(z)),
        xlabel=r"$h$",
        ylabel=r"$1 / \lambda_1$",
        zlabel="compliance",
    )

    # ax.set_axis_off()
    # turn off the axis planes
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

    # move the zlabel to the top of the axis
    # ax.zaxis._axinfo["juggled"] = (0, 0, 1)

    # set the position of the z axis label
    # ax.set_zlabel("compliance", rotation=180, labelpad=0)
    # # disable the panes
    # ax.zaxis.set_rotate_label(False)  # disable automatic rotation
    # # ax.zaxis.label.set_rotation(180)  # then customize the rotation
    # ax.zaxis.label.set_ha("right")  # ha is alias for horizontalalignment
    # ax.zaxis.label.set_va("top")  # va is alias for verticalalignment
    ax.zaxis.labelpad = -18
    ax.yaxis.labelpad = -18
    ax.xaxis.labelpad = 10

    # set tick lines off
    # ax.tick_params(axis="z", which="both", length=0)

    # y axis ticks size
    # ax.tick_params(axis="y", which="major", pad=-3, labelsize=6)

    # turn off grid
    ax.grid(False)

    # set x axis ticks every 1
    ax.xaxis.set_major_locator(ticker.MultipleLocator(int(x[20])))

    # set x axis ticks each tick has different color
    # facecolors = np.insert(facecolors, 0, [facecolors[0]], axis=0)
    # facecolors = np.insert(facecolors, -1, [facecolors[-1]], axis=0)

    # for i, t in enumerate(ax.xaxis.get_ticklabels()):
    #     t.set_color(facecolors[int(i / x[20])])

    # turn off the y and z axis ticks
    ax.yaxis.set_ticks([])
    ax.zaxis.set_ticks([])

    # swich the y and z axis
    # ax.elev = 180
    # ax.azim = 180
    # ax.dist = 8

    # scale the x axis to make it longer
    ax.set_box_aspect((1, 1, 1))
    plt.savefig(
        "output/final_results/building/displacement/frequency_compliance/building_pf.png",
        bbox_inches="tight",
        dpi=1000,
        pad_inches=0.5,
    )

    ##########################
    ax = plt.figure().add_subplot(projection="3d")
    ax.figure.set_size_inches(1.5 * 7.48, 1.5 * 4)

    # plot the pareto front surface
    ax.plot_trisurf(
        data_xyz_paret[:, 0],
        data_xyz_paret[:, 1],
        data_xyz_paret[:, 2],
        color="k",
        alpha=0.2,
    )

    j = 0
    facecolors = plt.colormaps["coolwarm"](
        np.linspace(0, 1, np.unique(data[:, 3]).size - 1)
    )
    black = np.array([[0, 0, 0, 1]])
    facecolors = np.append(facecolors, black, axis=0)
    for d in np.unique(data[:, 3]):
        data_x = np.empty((0, 2), float)

        for i in range(0, len(x)):
            if data[i, 3] == d:
                data_x = np.append(data_x, np.array([[y[i], z[i]]]), axis=0)

        ax.scatter(
            d,
            data_x[:, 0],
            data_x[:, 1],
            c=facecolors[int(j)],
            s=5.0,
            zorder=d + 1,
            alpha=1.0,
        )
        j += 1

    ax.set(
        xlim=(np.min(x), np.max(x)),
        ylim=(np.min(y), np.max(y)),
        zlim=(1.05 * np.min(z), 1.02 * np.max(z)),
        xlabel=r"$h$",
        ylabel=r"$1 / \lambda_1$",
        zlabel="compliance",
    )

    # ax.set_axis_off()
    # turn off the axis planes
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

    # move the zlabel to the top of the axis
    ax.zaxis._axinfo["juggled"] = (1, 2, 0)
    # rotate the z label
    ax.zaxis.set_rotate_label(False)  # disable automatic rotation
    ax.set_zlabel("compliance", rotation=90)
    ax.zaxis.labelpad = -18
    ax.yaxis.labelpad = -18
    ax.xaxis.labelpad = 10

    # turn off grid
    ax.grid(False)

    # set x axis ticks every 1
    ax.xaxis.set_major_locator(ticker.MultipleLocator(int(x[20])))

    # set x axis ticks each tick has different color
    facecolors = np.insert(facecolors, 0, [facecolors[0]], axis=0)
    facecolors = np.insert(facecolors, -1, [facecolors[-1]], axis=0)
    for i, t in enumerate(ax.xaxis.get_ticklabels()):
        t.set_color(facecolors[int(i / x[20])])

    # turn off the y and z axis ticks
    ax.yaxis.set_ticks([])
    ax.zaxis.set_ticks([])

    # change the last tick value to "inf"
    ax.set_xticklabels([f"{i}" for i in range(-1, 21)] + ["inf"])

    # change the angle of the view
    # ax.view_init(azim=0, elev=0)

    # scale the x axis to make it longer
    ax.set_box_aspect((4, 1, 1))
    plt.savefig(
        "output/final_results/building/displacement/frequency_compliance/building_pf2.png",
        bbox_inches="tight",
        dpi=1000,
        pad_inches=0.5,
    )

    #######################################
    fig, ax = plt.subplots(figsize=(2.5, 2.5))
    j = 0
    facecolors = plt.colormaps["coolwarm"](
        np.linspace(0, 1, np.unique(data[:, 3]).size - 1)
    )
    black = np.array([[0, 0, 0, 1]])
    facecolors = np.append(facecolors, black, axis=0)
    for d in np.unique(data[:, 3]):
        data_x = np.empty((0, 2), float)

        for i in range(0, len(x)):
            if data[i, 3] == d:
                data_x = np.append(data_x, np.array([[y[i], z[i]]]), axis=0)

        ax.scatter(
            data_x[:, 0],
            data_x[:, 1],
            c=facecolors[int(j)],
            s=2.0,
            zorder=d + 1,
            alpha=1.0,
        )
        j += 1

    ax.set(
        xlabel=r"$1 / \lambda_1$",
        ylabel="compliance",
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out")
    ax.tick_params(which="minor", direction="out")
    ax.tick_params(which="minor", left=False)

    plt.savefig(
        "output/final_results/building/displacement/frequency_compliance/building_pf3.png",
        bbox_inches="tight",
        dpi=1000,
        pad_inches=0.0,
    )

    fig, ax = plt.subplots(figsize=(1.8, 1.8))
    j = 0
    facecolors = plt.colormaps["coolwarm"](
        np.linspace(0, 1, np.unique(data[:, 3]).size)
    )
    for d in np.unique(data[:, 3]):
        data_x = np.empty((0, 2), float)

        for i in range(0, len(x)):
            if data[i, 3] == d:
                data_x = np.append(data_x, np.array([[y[i], z[i]]]), axis=0)

        ax.scatter(
            data_x[:, 0],
            data_x[:, 1],
            c=facecolors[int(j)],
            s=10.0,
            zorder=d + 1,
            alpha=1.0,
        )
        j += 1

    ax.set(
        xlim=(0.335, 0.346),
        ylim=(0.76e-5, 0.81e-5),
        # xlabel=r"$1 / \lambda_1$",
        # ylabel="compliance",
    )

    # turn on the minor ticks
    ax.minorticks_on()

    # ax.spines["top"].set_visible(False)
    # ax.spines["right"].set_visible(False)
    # ax.tick_params(direction="out")
    # ax.tick_params(which="minor", direction="out")
    # ax.tick_params(which="minor", left=False)

    plt.savefig(
        "output/final_results/building/displacement/frequency_compliance/building_pf4.png",
        bbox_inches="tight",
        dpi=1000,
        pad_inches=0.0,
    )

    # plt.show()


def plot_8():
    data = np.loadtxt("other/building3.txt")

    n = 10
    delete = np.array([5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    delete = np.array([8.0, 9.0, 10.0])
    # delete = np.array([])

    for i in range(0, len(delete)):
        data = data[data[:, 3] != delete[i]]

    x = data[:, 3]  # displacement
    y = 1 / data[:, 1]  # 1 / \lambda
    z = data[:, 2]  # compliance

    def polygon_under_graph(xx, yy, d):
        """
        Construct the vertex list which defines the polygon filling the space under
        the (x, y) line graph. This assumes x is in ascending order.
        """
        if d == 0:
            return [
                (np.min(xx) - 0.0001 * np.min(xx), np.max(z)),
                # (xx[-1], np.max(z)),
                # (np.min(xx) - 0.0001 * np.min(xx), yy[-1]),
                *zip(xx, yy),
            ]
        else:
            second_largest = np.partition(z, -2)[-2] - 0.0001 * np.min(z)
            return [
                # find the second largest value of z
                (np.min(xx) - 0.001 * np.min(xx), second_largest),
                *zip(xx, yy),
                # (xx[-1], np.max(z)),
            ]

    data_xyz = np.array([x, y, z]).T
    # sort by x
    data_xyz = data_xyz[data_xyz[:, 0].argsort()]

    data_xyz_paret = np.empty((0, 3), float)
    data_new = np.empty((n, 3), float)
    for d in np.unique(data_xyz[:, 0]):
        mu_d = data_xyz[data_xyz[:, 0] == d][:, 1]
        compile_d = data_xyz[data_xyz[:, 0] == d][:, 2]
        data_new[:, 1:3] = polygon_under_graph(mu_d, compile_d, d)
        data_new[:, 0] = d
        is_efficient = paretoset(data_new[:, 1:3])
        # is_efficient = is_pareto_efficient_dumb(data_new)
        data_xyz_paret = np.append(data_xyz_paret, data_new[is_efficient], axis=0)

    def polygon_under_graph(x, y):
        """
        Construct the vertex list which defines the polygon filling the space under
        the (x, y) line graph. This assumes x is in ascending order.
        """
        return [(x[0], 0.0), *zip(x, y), (x[-1], 0.0)]

    ax = plt.figure().add_subplot(projection="3d")
    ax.figure.set_size_inches(3, 3)

    # plot the pareto front surface
    ic(data_xyz_paret)
    # data_xyz_paret = data_xyz_paret[data_xyz_paret[:, 0] != 5.0]
    # deleta_5 = 0
    # deleta_6 = 0
    # ic(data_xyz_paret.shape[0])
    # index_delete = []
    # for i in range(0, data_xyz_paret.shape[0]):
    #     # if data_xyz_paret[i, 0] == 5.0:
    #     #     index_delete.append(i)
    #     #     deleta_5 += 1
    #     # if deleta_5 == 5:
    #     #     break

    #     if data_xyz_paret[i, 0] == 5.0:
    #         deleta_6 += 1
    #         if deleta_6 > 5:
    #             index_delete.append(i)

    # data_xyz_paret = np.delete(data_xyz_paret, index_delete, axis=0)

    ic(data_xyz_paret)
    ax.plot_trisurf(
        data_xyz_paret[:, 0],
        data_xyz_paret[:, 1],
        data_xyz_paret[:, 2],
        color="k",
        alpha=0.2,
    )

    j = 0
    facecolors = plt.colormaps["coolwarm"](
        np.linspace(0, 1, np.unique(data[:, 3]).size)
    )
    for d in np.unique(data[:, 3]):
        data_x = np.empty((0, 2), float)

        for i in range(0, len(x)):
            if data[i, 3] == d:
                data_x = np.append(data_x, np.array([[y[i], z[i]]]), axis=0)

        ax.scatter(
            d,
            data_x[:, 0],
            data_x[:, 1],
            c=facecolors[int(j)],
            s=2.0,
            zorder=d + 1,
            alpha=1.0,
        )
        j += 1

    ax.set(
        # xlim=(0.8*np.min(x), np.max(x)),
        # ylim=(1.2*np.min(y), np.max(y)),
        # zlim=(0.9 * np.min(z), 1.02 * np.max(z)),
        xlabel=r"$h$",
        ylabel=r"$1 / \lambda_1$",
        zlabel="compliance",
    )

    # ax.set_axis_off()
    # turn off the axis planes
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

    # move the zlabel to the top of the axis
    ax.zaxis._axinfo["juggled"] = (1, 2, 0)
    # rotate the z label
    ax.zaxis.set_rotate_label(False)  # disable automatic rotation
    ax.set_zlabel("compliance", rotation=90)
    ax.zaxis.labelpad = -18
    ax.yaxis.labelpad = -18
    ax.xaxis.labelpad = -12

    # turn off grid
    ax.grid(False)

    # set x axis ticks every 1
    ax.xaxis.set_major_locator(ticker.MultipleLocator(int(1.0)))
    ax.tick_params(axis="x", which="major", pad=-6)

    # set x axis ticks each tick has different color
    facecolors = np.insert(facecolors, 0, [facecolors[0]], axis=0)
    facecolors = np.insert(facecolors, -1, [facecolors[-1]], axis=0)
    for i, t in enumerate(ax.xaxis.get_ticklabels()):
        t.set_color(facecolors[int(i / 1.0)])

    # turn off the y and z axis ticks
    ax.yaxis.set_ticks([])
    ax.zaxis.set_ticks([])

    # change the angle of the view
    # ax.view_init(azim=45, elev=0)

    # scale the x axis to make it longer
    ax.set_box_aspect((1, 1, 1))
    plt.savefig(
        "output/final_results/building/displacement/frequency_compliance/building_pfp3.png",
        bbox_inches="tight",
        dpi=1000,
        pad_inches=0.5,
    )

    #######################################
    fig, ax = plt.subplots(figsize=(2.3, 2.3))
    j = 0
    facecolors = plt.colormaps["coolwarm"](
        np.linspace(0, 1, np.unique(data[:, 3]).size)
    )
    for d in np.unique(data[:, 3]):
        data_x = np.empty((0, 2), float)

        for i in range(0, len(x)):
            if data[i, 3] == d:
                data_x = np.append(data_x, np.array([[y[i], z[i]]]), axis=0)

        ax.plot(
            data_x[:, 0],
            data_x[:, 1],
            c=facecolors[int(j)],
            # s=10.0,
            zorder=d + 1,
            alpha=1.0,
            marker="o",
            markersize=2.5,
            linewidth=0.5,
            label=f"$h={d}$",
        )
        j += 1

    ax.set(
        xlabel=r"$1 / \lambda_1$",
        ylabel="compliance",
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out")
    ax.tick_params(which="minor", direction="out")
    ax.tick_params(which="minor", left=False)
    ax.legend(loc="upper right", frameon=False, fontsize=6, bbox_to_anchor=(1.4, 1.0))

    # ax.set_box_aspect((2, 1, 1))
    plt.savefig(
        "output/final_results/building/displacement/frequency_compliance/building_pf3.png",
        bbox_inches="tight",
        dpi=1000,
        pad_inches=0.0,
    )

    plt.show()


def plot_6(omega, BLF_ks, vol, compliance, dis):
    fig, ax = plt.subplots(figsize=(9, 4))
    ax2 = ax.twinx()
    ax2.spines.right.set_position(("axes", 1.0))

    ax3 = ax.twinx()
    ax3.spines.right.set_position(("axes", 1.05))

    styles = ["-", "--", "-.", ":", "-", "--", "-.", ":"]
    alpha = [0.8, 0.5, 0.4, 0.3, 0.2, 0.1]
    # marker = ["o", "o", "s", "s", "^", "^"]
    linewidth = [1.0, 0.5, 0.4, 0.3, 0.2, 0.1]
    n_start = 0
    n_iter = 9000

    # plot BLF_ks
    # (p1,) = ax.plot(
    #     BLF_ks[0][n_start:n_iter],
    #     label=r"$J_{ks}^{-1} [\lambda_{i}]$",
    #     color="k",
    #     alpha=0.8,
    #     linewidth=0.75,
    #     linestyle="--",
    # )
    (p1,) = ax.plot(
        omega[0][n_start:n_iter, 0],
        label=r"$\lambda_{1}$",
        alpha=0.8,
        color="k",
        linewidth=0.75,
    )
    ic(compliance[0][n_start:n_start +10] / 7.4e-06)
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
        linewidth=0.5,
        alpha=1.0,
        label=r"$g_{d}$",
        linestyle="-",
        zorder=10,
    )

    # for i in range(1, 6):
    #     ax.plot(
    #         omega[0][n_start:n_iter, i],
    #         label=r"$\lambda_{" + str(i + 1) + "}$",
    #         color="k",
    #         alpha=alpha[i],
    #         linestyle=styles[i],
    #         linewidth=0.5,
    #         # marker=marker[i],
    #         # markevery=10,
    #         # markersize=3,
    #         # markerfacecolor="none",
    #         # markeredgecolor=colors[i],
    #     )
    # log scale the x axis
    # ax.set(xscale="log", xlim=(30, 1000))
    # ax.set_xlabel("Iteration (log-scale)")

    ax.set(xlim=(100, 9000))
    # ax.set_xlabel("Iteration")

    # ax.legend(loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.xaxis.set_ticks_position("bottom")
    ax.yaxis.set_ticks_position("left")
    ax.tick_params(direction="out")
    ax.tick_params(which="minor", direction="out")

    # add vertical grid lines
    ax.grid(axis="x", linestyle="--", linewidth=0.5, alpha=0.5)

    # set x axis ticks every 800
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1000))
    # delete the first tick
    # ax.xaxis.get_major_ticks()[-3].set_visible(False)
    labels = [item.get_text() for item in ax.get_xticklabels()]
    for i in range(0, len(labels)):
        labels[i] = (
            "iter: "
            + labels[i]
            + "\n"
            + r"$w=$"
            + f"{np.max([0.9-0.1 * i, 0.0]):.1f}"
        )

    ax.set_xticklabels(labels)

    # ax.set_ylim(8, 17.95) # for 7
    ax.set_ylim(7, 17.95) # for 4
    ax.set_ylabel(r"$BLF$", rotation=0, labelpad=0)
    ax.yaxis.set_label_coords(0.0, 1.001)

    # ax2.set(ylim=(1.18, 1.795))  # for 7
    ax2.set(ylim=(1.15, 1.795))    # for 4
    ax2.set_ylabel(r"$c / c_{opt}$", rotation=0, labelpad=0)
    ax2.yaxis.set_label_coords(1.01, 1.032)

    # for ax3, add ytick for y=4.0

    ticks = ax3.get_yticks()
    ticks = np.append(ticks, [4])
    ax3.set_yticks(ticks)
    # ax3.set(ylim=(0.0, 59.95))  # for 7
    ax3.set(ylim=(0.0, 39.95))  # for 4
    ax3.set_ylabel("$h$", rotation=0, labelpad=0)
    ax3.yaxis.set_label_coords(1.05, 1.03)
    # add horizontal grid lines only for y=4.0
    ax3.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.5, ydata=[7.0])

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
        ncol=1,
        # loc=[0.5, 0.05],
        loc=[0.92, 0.2],
        frameon=False,
        fontsize=6,
    )


if __name__ == "__main__":
    # dir_result1 = "output/final_results/building/p, n=240, v=0.30, c=4.30, r=6.0/"
    # dir_result1 = "output/final_results/building/displacement/frequency/0,1,d=0/"
    # dir_result1 = "output/final_results/building/displacement/frequency/0,1,d=10/"
    # dir_result1 = "output/final_results/building/displacement/mode2,0,5,d=0/"
    # dir_result1 = "output/final_results/building/compliance-buckling/0.2/"
    # dir_result1 = "output/final_results/building/displacement/frequency_compliance/mode1/0,0,w=0.2,d=0.0/"
    # dir_result1 = "output/final_results/building/displacement/frequency_compliance/mode1/0,0,w=0.6,d=0.0/"
    dir_result1 = "output/final_results/building/displacement/frequency_compliance/mode3/0,0,w=0.2,d=7.0/"
    # dir_result1 = (
    #     "output/final_results/building/displacement/frequency_compliance/pfc/d=7,w=0.6/"
    # )

    # dir_result1 = "output/final_results/building/displacement/frequency_compliance/pfc2/w=0.8,d=7,1000/"
    # dir_result1 = "output/final_results/building/displacement/frequency_compliance/pfc2/w=0.8,d=4,1000/"
    # dir_result1 = "output/final_results/building/kappa/K=1e-12,G=1e-12/"

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
        #     "output/final_results/building/column_a11.png",
        #     bbox_inches="tight",
        #     dpi=1000,
        #     pad_inches=0.05,
        # )

        # plot_2(omega, BLF_ks, vol, compliance, dis)
        # # plot_1(1, rho, phi1, stress)
        # plt.savefig(
        #     "output/final_results/building/column_a13.png",
        #     bbox_inches="tight",
        #     dpi=1000,
        #     pad_inches=0.05,
        # )

        ###############################

        # plot_1(1, rho, phi0, stress)
        # plt.savefig("output/final_results/building/displacement/frequency_compliance/mode1/00_02_0_1.png", bbox_inches="tight", dpi=1000, pad_inches=0.0)

        # plot_2(omega, BLF_ks, vol, compliance, dis)
        # plt.savefig("output/final_results/building/displacement/building_0_1_0.png", bbox_inches="tight", dpi=1000, pad_inches=0.05)

        # plot_2_1(omega, BLF_ks, vol, compliance, dis)
        # plt.savefig("output/final_results/building/displacement/building_base_0_2.png", bbox_inches="tight", dpi=1000, pad_inches=0.05)

        # plot_1(1, rho, phi0, stress)
        # plt.savefig("output/final_results/building/displacement/frequency_compliance/mode1/00_06_0_1.png", bbox_inches="tight", dpi=1000, pad_inches=0.05)

        # plot_2(omega, BLF_ks, vol, compliance, dis)
        # plt.savefig("output/final_results/building/displacement/frequency_compliance/mode1/his_00_02_0.png", bbox_inches="tight", dpi=1000, pad_inches=0.05)

        # plot_1(1, rho, phi0, stress)
        # plt.savefig("output/final_results/building/displacement/frequency_compliance/mode3/00_02_7_7.png", bbox_inches="tight", dpi=1000, pad_inches=0.0)

        plot_2(omega, BLF_ks, vol, compliance, dis)
        plt.savefig("output/final_results/building/displacement/frequency_compliance/mode3/his_00_0_2_7.png", bbox_inches="tight", dpi=1000, pad_inches=0.05)

        # plot_3()

        # plot_1(1, rho, phi0, stress)
        # plt.savefig(
        #     "output/final_results/building/displacement/frequency_compliance/pfc/76.png",
        #     bbox_inches="tight",
        #     dpi=500,
        #     pad_inches=0.02,
        # )

        # plot_8()

        # plot_6(omega, BLF_ks, vol, compliance, dis)
        # plt.savefig(
        #     "output/final_results/building/displacement/frequency_compliance/pfc2/0.8-4-1000.png",
        #     bbox_inches="tight",
        #     dpi=1000,
        #     pad_inches=0.05,
        # )

        # plot_1(1, rho, phi0, stress)
        # plt.savefig(
        #     "output/final_results/building/displacement/frequency_compliance/pfc2/0.8,7,1000.png",
        #     bbox_inches="tight",
        #     dpi=1000,
        #     pad_inches=0.0,
        # )
