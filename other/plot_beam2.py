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
    total_cols = len(np.loadtxt(file, skiprows=8, max_rows=1)) - 2
    total_lines = sum(1 for line in open(file))
    with open(file) as f:
        for num, line in enumerate(f, 1):
            if num == total_lines - 2:
                break
            if "iter" in line:
                # if row 8 column 4 is not "n/a" then read the data
                if np.loadtxt(file, skiprows=num, max_rows=1)[3] != "n/a":
                    a = np.loadtxt(
                        file,
                        skiprows=num,
                        max_rows=10,
                        usecols=sorted(set(range(total_cols)) - {5}),
                    )
                else:
                    a = np.loadtxt(
                        file, skiprows=num, max_rows=10, usecols=range(0, total_cols)
                    )
                vol = np.append(vol, a)
    if np.loadtxt(file, skiprows=num, max_rows=1)[3] != "n/a":
        total_cols -= 1
    ic(total_cols, len(vol))
    n_iter = len(vol) // total_cols
    vol = vol.reshape(n_iter, total_cols)
    # dis = vol[:, 5]
    # stress_iter = vol[:, 4] ** 0.5 * 1e-6
    if total_cols == 5:
        dis = []
        stress_iter = []
    else:
        dis = vol[:, -2]
        stress_iter = vol[:, -3] ** 0.5 * 1e-6

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
    stress = stress.reshape(n - 1, m - 1)
    stress = stress**0.5
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
    # ax.plot([0, 8], [1, 1], color="black", linestyle="--", linewidth=0.25)

    if zoom:
        ax.plot(X, Y, color="black", linewidth=0.1, alpha=0.75)
        ax.plot(X.T, Y.T, color="black", linewidth=0.1, alpha=0.75)
        ax.set_ylim(np.min(Y), 0.35 + np.min(Y))
        ax.set_xlim(2.5, 5.5)


def plot_1(nrow, rho, phi0):
    # where is the max location for phi0[0]
    phi0[0] = np.abs(phi0[0])
    max_loc = np.argmax(phi0[0][:, :, 1])
    ic(np.unravel_index(max_loc, phi0[0][:, :, 1].shape))
    ic(np.unravel_index(159, phi0[0][:, :, 1].shape))
    ic(np.max(phi0[0][:, :, 1]))
    ic(phi0[0].shape, max_loc)
    fig, axs = plt.subplots(nrow, 1, figsize=(3, 1), constrained_layout=True)
    plot_modeshape(
        axs,
        rho[0],
        levels=5,
    )


def plot_2(omega, vol):
    fig, ax = plt.subplots()
    colors = ["k", "k", "b", "b", "r", "r"]
    styles = ["-", "--", "-", "--", "-", "--"]
    alpha = [1.0, 1.0, 0.6, 0.6, 0.2, 0.2]
    marker = ["o", "o", "s", "s", "^", "^"]
    n_iter = 300
    for i in range(6):
        ax.plot(
            omega[2][3:n_iter, i],
            label=f"$\omega_{i+1}$",
            color=colors[i],
            # alpha=alpha[i],
            linestyle=styles[i],
            # marker=marker[i],
            # markevery=10,
            # markersize=3,
            # markerfacecolor="none",
            # markeredgecolor=colors[i],
        )

    handles, labels = ax.get_legend_handles_labels()
    handles = [handles[i] for i in range(0, len(handles), 2)] + [
        handles[i] for i in range(1, len(handles), 2)
    ]
    labels = [labels[i] for i in range(0, len(labels), 2)] + [
        labels[i] for i in range(1, len(labels), 2)
    ]
    ax.legend(
        handles,
        labels,
        title="Frequencies:",
        ncol=2,
        loc=[0.5, 0.7],
        frameon=False,
    )
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Frequency (rad/s)")
    # ax.legend(loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.xaxis.set_ticks_position("bottom")
    ax.yaxis.set_ticks_position("left")
    ax.tick_params(direction="out")
    ax.tick_params(which="minor", direction="out")
    ax.set_ylim(0, 1350)


def plot_3(omega, vol, dis, stress_iter):
    # fig, ax = plt.subplots(constrained_layout=True)
    # marker = ["o", "s", "^"]
    # colors = ["k", "b","r"]

    # n_iter = 300
    # for i in range(3):
    #     ax.plot(omega[i][3:n_iter, 0], label=f"$\omega_{i+1}$", color=colors[i])
    # ax.set_xlabel("Iteration")
    # ax.set_ylabel("Frequency (rad/s)")
    # ax.spines["top"].set_visible(False)
    # ax.spines["right"].set_visible(False)
    # ax.xaxis.set_ticks_position("bottom")
    # ax.yaxis.set_ticks_position("left")
    # ax.tick_params(direction="out")
    # ax.tick_params(which="minor", direction="out")

    # # plot volume with right y axis
    # ax2 = ax.twinx()
    # for i in range(3):
    #     ax2.plot(vol[i][:n_iter], label=f"$V_{i+1}$",  linestyle="--", color=colors[i])
    # ax2.set_ylabel("Volume")
    # ax2.spines["top"].set_visible(False)
    # # ax2.spines["right"].set_visible(False)
    # ax2.xaxis.set_ticks_position("bottom")
    # ax2.yaxis.set_ticks_position("right")
    # ax2.tick_params(direction="out")
    # ax2.tick_params(which="minor", direction="out")
    # ax2.set_ylim(0.0, 1)
    # # change the right y axis to dashed line
    # ax2.spines["right"].set_linestyle("--")
    # # combine the legend in to two columns
    # handles1, labels1 = ax.get_legend_handles_labels()
    # handles2, labels2 = ax2.get_legend_handles_labels()
    # ax2.legend(handles1 + handles2, labels1 + labels2, ncol=2, loc="lower right")
    # # change the legend to two columns
    # # ax2.legend(loc="upper right", ncol=2)

    fig2, ax = plt.subplots(figsize=(4.0, 2.5), constrained_layout=True)
    n_iter = 600
    n_start = 10

    # ax2 = ax.twinx()
    ax3 = ax.twinx()
    ax4 = ax.twinx()
    ax3.spines.right.set_position(("axes", 1.02))
    ax4.spines.right.set_position(("axes", 1.21))
    # ax4.spines.right.set_position(("axes", 1.37))

    # (p2,) = ax2.plot(
    #     vol[3][n_start:n_iter], color="k", linewidth=0.5, label="$V$ ($\%$) "
    # )
    # find if a > 1e-4
    for i in range(len(stress_iter[0])):
        if stress_iter[0][i] > 2:
            stress_iter[0][i] = stress_iter[0][i - 1] * 0.9

    for i in range(len(dis[0])):
        if dis[0][i] < 0.3:
            dis[0][i] = dis[0][i - 1]

    (p3,) = ax3.plot(dis[0][n_start:n_iter], color="k", linewidth=0.5, label="$h$")
    (p4,) = ax4.plot(
        stress_iter[0][n_start:n_iter],
        color="b",
        linewidth=0.5,
        label="$\sigma_{vM}$",
    )

    # add a straight line at y=0.4 for ax2
    # p5 = ax2.axhline(
    #     xmin=0.04,
    #     xmax=0.96,
    #     y=0.4,
    #     color="k",
    #     linestyle="--",
    #     linewidth=0.5,
    #     label="$V_{\mathrm{constraint}}$",
    # )
    p6 = ax3.axhline(
        xmin=0.04,
        xmax=0.96,
        y=0.5,
        color="k",
        linestyle="--",
        linewidth=1,
        alpha=0.25,
        label="$h_{\mathrm{constraint}}$",
    )
    p7 = ax4.axhline(
        xmin=0.04,
        xmax=0.96,
        y=1.67332,
        color="b",
        linestyle="--",
        linewidth=1,
        alpha=0.25,
        label="$\sigma_{\mathrm{constraint}}$",
    )

    a = (np.abs(omega[0][n_start:n_iter, 0] - omega[0][n_start:n_iter, 1])) / (
        omega[0][n_start:n_iter, 0]
    )

    # # find if a > 1e-4
    # for i in range(len(a)):
    #     if a[i] > 1e-4:
    #         a[i] = a[i-1]

    (p1,) = ax.plot(
        a,
        color="r",
        linewidth=0.25,
        label="$\ \\| \omega_{2} - \\omega_{1} \| / \omega_{1}$",
    )

    ax.set(yscale="log", ylim=(1e-5, 0.9e4), xlabel="Iteration")
    ax.set_ylabel(
        "Relative Difference $\ \\| \omega_{2} - \\omega_{1} \| / \omega_{1}$ ($\%$)"
    )
    # ax2.set(ylim=(0.16, 0.99), ylabel="Volume Fraction ($\%$)")
    ax3.set(ylim=(0.3, 0.8), ylabel="Displacement $h \ (\mathrm{m}^2)$")
    ax4.set(ylim=(0.0, 1.8), ylabel="von Mises Stress $\sigma_{vM} \ (\mathrm{MPa})$")

    handles, labels = [], []
    handles.append(p4)
    labels.append(p4.get_label())
    handles.append(p7)
    labels.append(p7.get_label())
    handles.append(p3)
    labels.append(p3.get_label())
    handles.append(p6)
    labels.append(p6.get_label())
    handles.append(p1)
    labels.append(p1.get_label())

    # handles = [handles[i] for i in range(0, len(handles), 2)] + [
    #     handles[i] for i in range(1, len(handles), 2)
    # ]
    # labels = [labels[i] for i in range(0, len(labels), 2)] + [
    #     labels[i] for i in range(1, len(labels), 2)
    # ]
    ax.legend(
        handles,
        labels,
        # title="Frequencies:",
        ncol=1,
        loc=[0.52, 0.47],
        frameon=False,
    )

    ax.yaxis.label.set_color(p1.get_color())
    # ax2.yaxis.label.set_color(p2.get_color())
    ax3.yaxis.label.set_color(p3.get_color())
    ax4.yaxis.label.set_color(p4.get_color())

    ax.tick_params(axis="y", colors=p1.get_color())
    # ax2.tick_params(axis="y", colors=p2.get_color())
    ax3.tick_params(axis="y", colors=p3.get_color())
    ax4.tick_params(axis="y", colors=p4.get_color())

    ax.spines["top"].set_visible(False)
    # ax2.spines["top"].set_visible(False)
    ax3.spines["top"].set_visible(False)
    ax4.spines["top"].set_visible(False)
    ax.xaxis.set_ticks_position("bottom")
    ax.yaxis.set_ticks_position("left")
    ax.spines["right"].set_visible(False)
    # ax2.yaxis.set_ticks_position("right")
    ax3.yaxis.set_ticks_position("right")
    ax4.yaxis.set_ticks_position("right")

    ax.tick_params(direction="out")
    # ax2.tick_params(direction="out")
    ax3.tick_params(direction="out")
    ax4.tick_params(direction="out")
    ax.tick_params(which="minor", direction="out")
    # ax2.tick_params(which="minor", direction="out")
    ax3.tick_params(which="minor", direction="out")
    ax4.tick_params(which="minor", direction="out")
    # ax3.tick_params(which="minor", direction="out")
    # turn off the minor ticks for ax3
    ax.tick_params(which="minor", left=False)

    fig2.savefig(
        "final_results/mbbbeam/mbbbeam_freq_stress.pdf",
        bbox_inches="tight",
        pad_inches=0.0,
    )


def plot_4(ncol, rho, stress, phi0, phi1, phi2, dis=None):
    nrow = 1
    fig, axs = plt.subplots(nrow, ncol, figsize=(7, 2.5), constrained_layout=True)
    labels_1 = ["(a)", "(b)", "(c)"]
    labels_2 = ["(d)", "(e)", "(f)"]
    labels_3 = [
        "$\omega_{opt\_b} = 169.73$ rad/s",
        "$\omega_{opt\_b} = 163.57$ rad/s",
        "$\omega_{opt\_b} = 154.67$ rad/s",
    ]
    labels_4 = [
        "$\omega_{opt\_a} = 169.76$ rad/s",
        "$\omega_{opt\_a} = 163.65$ rad/s",
        "$\omega_{opt\_a} = 151.84$ rad/s",
    ]

    # # where is the max location for phi0[0]
    # max_loc = np.argmax(phi0[0][:, :, 1])
    # ic(np.unravel_index(max_loc, phi0[0][:, :, 1].shape))
    # ic(np.unravel_index(159, phi0[0][:, :, 1].shape))
    # ic(np.max(phi0[0][:, :, 1]))
    # ic(phi0[0].shape, max_loc)

    for i in range(nrow):
        for j in range(ncol):
            if i == 0:
                # if j == 2:
                #     flip_x = False
                #     flip_y = True
                # else:
                #     flip_x = False
                #     flip_y = False

                flip_x = False
                flip_y = False
                # plot_modeshape(axs[j], rho[j], levels=25)

                # print the middle point of phi0
                print(phi1[j][50, 400, 1])

                plot_modeshape(
                    axs[j],
                    rho[j],
                    phi0[j],
                    stress[j],
                    flip_x=flip_x,
                    flip_y=flip_y,
                    # zoom=True,
                    levels=100,
                )

                # axs[j].text(
                #     0.5,
                #     -0.05,
                #     labels_1[j],
                #     transform=axs[j].transAxes,
                #     va="top",
                #     ha="center",
                #     fontweight="bold",
                # )
                for ax in axs[:]:
                    ax.set_anchor("N")

                # if j != 0:
                #     # add a red point at bottom middle of the domain
                #     axs[i, j].scatter(4, 0.05, color="red", s=8)
            # elif i == 1:
            #     plot_modeshape(axs[j], rho[j], phi0[j], flip_x=flip_x, flip_y=flip_y, levels=25)

            # if j == 1 or j == 2:
            #     plot_modeshape(axs[i, j], rho[0], phi0[0], alpha=0.2)
            # elif i == 2:
            #     plot_modeshape(axs[i, j], rho[j], phi1[j], flip_x=flip_x, flip_y=flip_y)
            # elif i == 2:
            #     plot_modeshape(axs[i, j], rho[j], phi2[j], flip_x=flip_x, flip_y=flip_y)
            # elif i == 2:
            #     plot_modeshape(
            #         axs[j],
            #         rho[j],
            #         phi0[j],
            #         stress[j],
            #         flip_x=flip_x,
            #         flip_y=flip_y,
            #         zoom=False,
            #         levels=100,
            #     )
            # elif i == 3:
            #     plot_modeshape(
            #         axs[j],
            #         rho[j],
            #         phi0[j],
            #         stress[j],
            #         flip_x=flip_x,
            #         flip_y=flip_y,
            #         zoom=True,
            #         levels=50,
            #     )

def plot_0():
    # w = np.arange(0, 1.1, 0.1)
    fig, ax = plt.subplots(figsize=(4, 4), constrained_layout=True)
    lam = np.array([8.93, 9.14, 7.99, 7.97, 7.47, 7.06, 6.75, 6.90, 6.32, 3.21, 3.01])
    c = np.array([1.52e-5, 1.21e-5, 1.04e-5, 1.0e-5, 9.49e-6, 9.09e-6, 8.88e-6, 9.01e-6, 8.60e-6, 7.74e-6, 7.69e-6])
    plt.plot(c, 1/ lam, "o-", color="k", markersize=3, linewidth=0.75)
    plt.ylabel("$1 / BLF$")
    plt.xlabel("$c$")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    # plt.grid(True, linestyle="-", linewidth=0.5)
    # open the minor ticks
    # ax.tick_params(which="minor", direction="out")
    # open the minor grid
    # ax.minorticks_on()
    # set the minor grid style
    # ax.grid(which="both", linestyle=":", linewidth=0.2, alpha=0.5)
    
    
    
    # plt.xlim(0, 2e-5)
    # plt.ylim(1/10, 1/2)

    


if __name__ == "__main__":
    dir_result1 = "output/final_results/beam/compliance-buckling/1.0/"

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
        plot_0()
        plt.savefig(
            "output/final_results/beam/compliance-buckling/c_lambda.png",
            bbox_inches="tight",
            pad_inches=0.0,
            dpi=500,
        )
                
        # plot_1(1, rho, phi0)
        # plt.savefig(
        #     "output/final_results/beam/cf10.png",
        #     bbox_inches="tight",
        #     dpi=500,
        #     pad_inches=0.05,
        # )

        # plot_2(omega, BLF_ks, vol, compliance, dis)
        # plt.savefig("output/final_results/beam/cf0.png", bbox_inches="tight", dpi=500, pad_inches=0.05)

