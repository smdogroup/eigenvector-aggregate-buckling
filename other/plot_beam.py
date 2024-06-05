from icecream import ic
import matplotlib.pylab as plt
import numpy as np
import scienceplots


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
                # store the column 7 in dis
                b = np.loadtxt(file, skiprows=num, max_rows=10, usecols=range(6, 7))
                dis = np.append(dis, b)

    # if np.loadtxt(file, skiprows=num, max_rows=1)[3] != "n/a":
    ic(total_cols, len(vol))
    n_iter = len(vol) // total_cols
    vol = vol.reshape(n_iter, total_cols)
    stress_iter = vol[:, 4] ** 0.5 * 1e-6

    BLF_ks = vol[:, 3]
    ic(BLF_ks.shape)
    compliance = vol[:, 4]
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
    levels=50,
):
    Z = rho
    cmap = "Greys"
    vmin = np.min(Z)
    vmax = np.max(Z)

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

        Z_max = 6.0392892273608645
        # Z_min = 6.634634423461437e-05
        # Z = (Z - Z_min) / (Z_max - Z_min)
        Z = (Z - Z.min()) / (Z_max - Z.min())
        a = 0.1
        vmin = 0.55
        vmax = 0.45**a

        Z = Z**a
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

    X, Y = np.meshgrid(np.linspace(0, 3, Z.shape[1]), np.linspace(0, 1, Z.shape[0]))
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
    # ax.plot([0.5, 3], [1, 1], color="black", linestyle="--", linewidth=0.25)

    # add dot at the top mid edge\
    if phi is not None:
        ax.scatter(
            1.5 + phi[120 - 1, 180 - 1, 0],
            1.0 + phi[120 - 1, 180 - 1, 1],
            color="orange",
            s=15,
            zorder=10,
            clip_on=False,
        )
        ax.scatter(
            1.5 + phi[0, 180 - 1, 0],
            0.0 + phi[0, 180 - 1, 1],
            color="orange",
            s=15,
            zorder=10,
            clip_on=False,
        )

    if zoom:
        ax.plot(X, Y, color="black", linewidth=0.1, alpha=0.75)
        ax.plot(X.T, Y.T, color="black", linewidth=0.1, alpha=0.75)
        ax.set_ylim(np.min(Y), 0.35 + np.min(Y))
        ax.set_xlim(2.5, 5.5)


def plot_1(rho, phi0):
    fig, axs = plt.subplots(constrained_layout=True)
    plot_modeshape(
        axs,
        rho[0],
        levels=50,
        phi=phi0[0] * 0.1,
    )

    # # where is the max location for phi0[0]
    # phi0[0] = np.abs(phi0[0])
    # max_loc = np.argmax(phi0[0][:, :, 1])
    # ic(np.unravel_index(max_loc, phi0[0][:, :, 1].shape))
    # ic(np.unravel_index(159, phi0[0][:, :, 1].shape))
    # ic(np.max(phi0[0][:, :, 1]))
    # ic(phi0[0].shape, max_loc)


def plot_1_1(nrow, rho, phi0, phi1):
    fig, axs = plt.subplots(1, nrow, constrained_layout=True, figsize=(8, 4))
    plot_modeshape(
        axs[0],
        rho[0],
        levels=50,
    )
    plot_modeshape(
        axs[1],
        rho[0],
        phi0[0] * 0.1,
        levels=50,
    )
    plot_modeshape(
        axs[2],
        rho[0],
        phi1[0] * 0.1,
        levels=50,
    )


def plot_0():
    # w = np.arange(0, 1.1, 0.1)
    fig, ax = plt.subplots(figsize=(8, (8 / 7.48) * 4))
    lam = np.array([8.93, 9.14, 7.99, 7.97, 7.47, 7.06, 6.75, 6.90, 6.32, 3.21, 3.01])
    c = np.array(
        [
            1.52e-5,
            1.21e-5,
            1.04e-5,
            1.0e-5,
            9.49e-6,
            9.09e-6,
            8.88e-6,
            9.01e-6,
            8.60e-6,
            7.74e-6,
            7.69e-6,
        ]
    )
    plt.plot(c, 1 / lam, "o-", color="k", markersize=3, linewidth=0.75)
    plt.ylabel("$1 / BLF(\lambda_1)$")
    plt.xlabel("$c$")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_2(omega, BLF_ks, vol, compliance, dis):
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
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
        label=f"$BLF_{1}$",
        alpha=0.8,
        color="k",
        linewidth=0.75,
    )

    (pc1,) = ax2.plot(
        compliance[0][n_start:n_iter] / (7.7e-06),
        color="b",
        linewidth=0.75,
        alpha=0.6,
        label=r"${c}$",
        linestyle="-",
        zorder=0,
    )
    # (pc2,) = ax2.plot(vol[0][n_start:n_iter] / 0.45, color="b", linewidth=0.75, alpha=0.6, label=r'$g_V$',linestyle="--",zorder=0)

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
    ax.set(xscale="log", xlim=(10, 1000))
    ax.set_xlabel("Iteration (log-scale)")
    # ax.legend(loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.xaxis.set_ticks_position("bottom")
    ax.yaxis.set_ticks_position("left")
    ax.tick_params(direction="out")
    ax.tick_params(which="minor", direction="out")

    ax.set_ylim(0, 12.95)
    ax.set_ylabel("$BLF_i$", rotation=0, labelpad=0)
    ax.yaxis.set_label_coords(0.0, 1.001)

    ax2.set(ylim=(0.0, 3.9))
    y2int = np.arange(0.0, 3.9, 1.0)
    ax2.set_yticks(y2int)
    ax2.set_ylabel("$c / c_{opt}$", rotation=0, labelpad=0)
    ax2.yaxis.set_label_coords(1.02, 1.051)

    ticks = ax3.get_yticks()
    ticks = np.array([0.0, 5.0])
    ax3.set_yticks(ticks)
    ax3.set(ylim=(-50.0, 8.195))  # d=0
    # ax3.set(ylim=(-10.0, 6.95))  # d=1.5
    # ax3.set(ylim=(0.0, 4.695))    # d=1.5
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
        loc=[0.3, 0.05],
        frameon=False,
        fontsize=6,
    )


def plot_2_1(omega, BLF_ks, vol, compliance, dis):
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
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
        label=f"$BLF_{1}$",
        alpha=0.8,
        color="k",
        linewidth=0.75,
    )

    (pc1,) = ax2.plot(
        compliance[0][n_start:n_iter] / (7.7e-06),
        color="b",
        linewidth=0.75,
        alpha=0.6,
        label=r"${c}$",
        linestyle="-",
        zorder=0,
    )
    # (pc2,) = ax2.plot(vol[0][n_start:n_iter] / 0.45, color="b", linewidth=0.75, alpha=0.6, label=r'$g_V$',linestyle="--",zorder=0)

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
    ax.set_xlabel("Iteration")
    ax.set(xlim=(5, 1000))
    # ax.set(xscale="log", xlim=(10, 1000))
    # ax.set_xlabel("Iteration (log-scale)")

    # ax.legend(loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.xaxis.set_ticks_position("bottom")
    ax.yaxis.set_ticks_position("left")
    ax.tick_params(direction="out")
    ax.tick_params(which="minor", direction="out")

    ax.set_ylim(0, 12.95)
    ax.set_ylabel("$BLF_i$", rotation=0, labelpad=0)
    ax.yaxis.set_label_coords(0.0, 1.001)

    ax2.set(ylim=(0.0, 3.9))
    y2int = np.arange(0.0, 3.9, 1.0)
    ax2.set_yticks(y2int)
    ax2.set_ylabel("$c / c_{opt}$", rotation=0, labelpad=0)
    ax2.yaxis.set_label_coords(1.02, 1.051)

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
        loc=[0.3, 0.05],
        frameon=False,
        fontsize=6,
    )


if __name__ == "__main__":
    # dir_result1 = "output/final_results/beam/compliance-buckling/0.2/"
    # dir_result1 = "output/final_results/beam/displacement/mode1/0,1,d=0.0/"
    dir_result1 = "output/final_results/beam/displacement/mode1/0,1,frac=0.4/"
    # dir_result1 = "output/final_results/beam/displacement/mode3/0,0,w=0.4,d=5.0/"
    # dir_result1 = "output/final_results/beam/compliance-buckling/0.2/"

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
        # plot_0()
        # plt.savefig(
        #     "output/final_results/beam/compliance-buckling/c_lambda.png",
        #     bbox_inches="tight",
        #     pad_inches=0.0,
        #     dpi=1000,
        # )

        plot_1(rho, phi1)
        # plot_1_1(3, rho, phi1, phi0)
        plt.savefig(
            "output/final_results/beam/frac=4_2.png",
            bbox_inches="tight",
            dpi=1000,
            pad_inches=0.0,
        )

        # plot_1_1(3, rho, phi0, phi1)
        # plt.savefig(
        #     "output/final_results/beam/4_2.png",
        #     bbox_inches="tight",
        #     dpi=1000,
        #     pad_inches=0.0,
        # )

        # plot_1(1, rho, phi1)
        # plot_2(omega, BLF_ks, vol, compliance, dis)
        # plt.savefig(
        #     "output/final_results/beam/displacement/pmode3-04.png",
        #     bbox_inches="tight",
        #     dpi=1000,
        #     pad_inches=0.05,
        # )

        # plot_2_1(omega, BLF_ks, vol, compliance, dis)
        # # plot_2(omega, BLF_ks, vol, compliance, dis)
        # plt.savefig(
        #     "output/final_results/beam/01.png",
        #     bbox_inches="tight",
        #     dpi=1000,
        #     pad_inches=0.05,
        # )
