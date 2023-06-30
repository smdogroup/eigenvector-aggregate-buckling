from icecream import ic
from matplotlib import ticker, transforms
import matplotlib.pylab as plt
import matplotlib.ticker as mtick
from matplotlib.ticker import AutoMinorLocator, MultipleLocator
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
    n = np.sqrt(nnodes).astype(int)
    m = n
    return nnodes, m, n


def read_vol(file):
    temp = []
    total_cols = len(np.loadtxt(file, skiprows=9, max_rows=1))
    total_lines = sum(1 for line in open(file))
    with open(file) as f:
        for num, line in enumerate(f, 1):
            if num == total_lines - 2:
                break
            if "iter" in line:
                a = np.loadtxt(file, skiprows=num, max_rows=10)
                temp = np.append(temp, a)

    n_iter = len(temp) // total_cols
    temp = temp.reshape(n_iter, total_cols)
    vol = temp[:, 2]
    dis = temp[:, 5]
    stress_iter = temp[:, 4] ** 0.5 * 1e-6
    return vol, dis, stress_iter


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
    stress = stress**0.5 * 1e-6
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
    vol, dis, stress_iter = read_vol(dir_stdout)
    return rho, vol, dis, stress_iter, stress, omega, phi0, phi1, phi2, phi3, phi4, phi5


def assmble_data(
    ncol, iter, iter2=None, iter3=None, iter4=None, iter5=None, iter6=None, iter7=None
):
    rho, vol, dis, stress_iter, stress, omega, phi0, phi1, phi2, phi3, phi4, phi5 = (
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
        elif i == 4 and iter4 is not None:
            exec(f"dir_vtk{i} = dir_result{i} + 'vtk/it_{iter4}.vtk'")
        elif i == 5 and iter5 is not None:
            exec(f"dir_vtk{i} = dir_result{i} + 'vtk/it_{iter5}.vtk'")
        elif i == 6 and iter6 is not None:
            exec(f"dir_vtk{i} = dir_result{i} + 'vtk/it_{iter6}.vtk'")
        elif i == 7 and iter7 is not None:
            exec(f"dir_vtk{i} = dir_result{i} + 'vtk/it_{iter7}.vtk'")
        else:
            exec(f"dir_vtk{i} = dir_result{i} + 'vtk/it_{iter}.vtk'")

        exec(f"dir_freq{i} = dir_result{i} + 'frequencies.log'")
        exec(f"dir_options{i} = dir_result{i} + 'options.txt'")
        exec(f"dir_stdout{i} = dir_result{i} + 'stdout.log'")
        exec(
            f"rho{i}, vol{i}, dis{i}, stress_iter{i}, stress{i}, omega{i}, phi0_{i}, phi1_{i}, phi2_{i}, phi3_{i}, phi4_{i}, phi5_{i} = read_data(dir_vtk{i}, dir_freq{i}, dir_stdout{i}, dir_options{i})"
        )
        exec(f"rho.append(rho{i})")
        exec(f"vol.append(vol{i})")
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

    return rho, vol, dis, stress_iter, stress, omega, phi0, phi1, phi2, phi3, phi4, phi5


def plot_modeshape(
    ax,
    rho,
    phi=None,
    stress=None,
    alpha=None,
    zoom=False,
    surface=False,
    levels=50,
    aa=False,
    cmap="Greys",
):
    Z = rho
    # vmin = 0.5
    # vmax = 0.8
    if aa is False and zoom is False:
        vmin = np.min(Z)
        vmax = 0.8
    else:
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
        # Z_max = 2.3039223283815526
        Z_max = 2.070947880198268
        Z_min = 0.0
        Z = (Z - Z_min) / (Z_max - Z_min)
        # make the min and max closer
        a = 0.5

        ic(Z.min(), Z.max())

        # if aa:
        #     vmin = 0.25
        #     vmax = 1.0**a
        # else:
        #     vmin = 0.25
        #     vmax = 1.0**a

        # if aa:
        #     vmin = 0.25
        #     vmax = 1.0**a
        # else:
        # if zoom:
        #     vmin = 0.4
        #     vmax = 1.0**a
        # else:
        vmin = 0.1
        vmax = 1.0

        Z = Z**a

        cmap = "coolwarm"

    # if zoom and stress is None:
    #     # add same last row and column for rho
    #     # Z = np.vstack((Z, Z[-1, :]))
    #     # Z = np.hstack((Z, Z[:, -1].reshape(Z.shape[0], 1)))
    #     # find the index of value less than 0.5
    #     # ind_x, ind_y = np.where(Z < 0.1)
    #     # # if ind_x = 500,

    #     # add zeros to for the first row and column
    #     Z = np.vstack((np.zeros((1, Z.shape[1])), Z))
    #     Z = np.hstack((np.zeros((Z.shape[0], 1)), Z))

    #     # delete the 490th row and column
    #     Z = np.delete(Z, 390, 0)
    #     Z = np.delete(Z, 390, 1)

    X, Y = np.meshgrid(np.linspace(0, 8, Z.shape[1]), np.linspace(0, 8, Z.shape[0]))
    X = X + phi[:, :, 0]
    Y = Y + phi[:, :, 1]

    # if aa:
    #     # rotate 90 degrees
    #     X = X + phi[:, :, 1]
    #     Y = Y - phi[:, :, 0]
    # else:

    ax.contourf(
        X,
        Y,
        Z,
        levels=levels,
        alpha=alpha,
        # antialiased=True,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines[:].set_visible(False)
    ax.set_aspect("equal")

    if surface:
        Z = np.sqrt(phi[:, :, 0] ** 2 + phi[:, :, 1] ** 2)
        Z = (Z - np.min(Z)) / (np.max(Z) - np.min(Z))
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
        ax.plot(X, Y, color="black", linewidth=0.1, alpha=0.25)
        ax.plot(X.T, Y.T, color="black", linewidth=0.1, alpha=0.25)
        ax.set_xlim(6.75, 8)
        ax.set_ylim(6.75, 8)

    # if stress is not None:
    #     ax.plot(X, Y, color="black", linewidth=0.1, alpha=0.1)
    #     ax.plot(X.T, Y.T, color="black", linewidth=0.1, alpha=0.1)


def plot_1(nrow, ncol, rho):
    fig, axs = plt.subplots(nrow, ncol, figsize=(ncol, nrow), constrained_layout=True)
    for i in range(nrow):
        for j in range(ncol):
            plot_modeshape(axs[i, j], rho[j * nrow + i])

    plt.savefig("../output/final_results/square/square_r0.pdf", bbox_inches="tight")


def plot_2(nrow, ncol, rho):
    fig, axs = plt.subplots(nrow, ncol, figsize=(ncol, nrow), constrained_layout=True)
    for i in range(nrow):
        for j in range(ncol):
            plot_modeshape(axs[i, j], rho[i * ncol + j])

    plt.savefig("../output/final_results/square/square_vol_stress.pdf", bbox_inches="tight")


def plot_3(nrow, ncol, rho, vol, stress, omega, phi0, phi1, phi2, phi3, phi4, phi5):
    fig, axs = plt.subplots(nrow, ncol, figsize=(3.3, 5.25), constrained_layout=True)
    labels = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)"]
    scaledtrans = transforms.ScaledTranslation(-0.4, 0, fig.dpi_scale_trans)
    phi = [phi0[0], phi1[0], phi2[0], phi3[0], phi4[0], phi5[0]]
    for i in range(nrow):
        for j in range(ncol):
            a = i * ncol + j

            plot_modeshape(
                axs[i, j],
                rho[0],
                phi[a],
                surface=True,
                alpha=0.75,
                levels=1,
                aa=True,
            )
            axs[i, j].text(
                0.5,
                -0.05,
                labels[a],
                transform=axs[i, j].transAxes,
                va="bottom",
                ha="center",
                fontweight="bold",
            )
            # # axs[i, j].set_aspect("equal")

            # axs[i, j].set_anchor("N")

    # for ax in axs[0, :]:
    #     ax.set_anchor("S")
    # for ax in axs[1, :]:
    #     ax.set_anchor("S")
    # for ax in axs[2, :]:
    #     ax.set_anchor("S")

    # plot them one by one
    # fig2, ax2 = plt.subplots(1, 1, figsize=(3, 3), constrained_layout=True)
    # plot_modeshape(ax2, rho[0], scale * phi[a], surface=True, alpha=0.75, levels=1)
    # plt.savefig("final_results/square/square_modeshape_{}.png".format(a), bbox_inches="tight", pad_inches=0.0, dpi=200)
    # fig.tight_layout()
    # plt.subplots_adjust(wspace=0, hspace=0)
    # plt.savefig("final_results/square/square_modeshape.pdf")
    plt.savefig("../output/final_results/square/square_modeshape.png", dpi=300, pad_inches=0.0)

    # fig, ax = plt.subplots(1, 2, figsize=(7, 2.5), constrained_layout=True)
    # n_iter = 500
    # n_start = 3

    # plot_modeshape(ax[0], rho[0])
    # ax[0].text(
    #     0.5,
    #     -0.025,
    #     "Baseline",
    #     transform=ax[0].transAxes,
    #     va="top",
    #     ha="center",
    # )

    # ax2 = ax[1].twinx()
    # ax3 = ax[1].twinx()
    # ax2.spines.right.set_position(("axes", 1.05))
    # ax3.spines.right.set_position(("axes", 1.25))

    # (p1,) = ax[1].plot(omega[0][n_start:n_iter, 0], color="b", linewidth=0.5)
    # (p2,) = ax2.plot(vol[0][n_start:n_iter], color="k", linewidth=0.5)
    # ax2.axhline(
    #     y=0.4,
    #     xmin=0.045,
    #     xmax=0.96,
    #     color="k",
    #     linestyle="--",
    #     alpha=0.5,
    #     linewidth=0.5,
    # )
    # a = (np.abs(omega[0][n_start:n_iter, 0] - omega[0][n_start:n_iter, 1])) / (
    #     omega[0][n_start:n_iter, 0]
    # )
    # for i in range(len(a)):
    #     if a[i] > 1e-5:
    #         a[i] = a[i - 1]

    # (p3,) = ax3.plot(
    #     a,
    #     color="r",
    #     linewidth=0.25,
    # )

    # ax[1].set(ylim=(0, 100), ylabel="Frequency $\omega_{1}$ (rad/s)", xlabel="Iteration")
    # ax2.set(ylim=(0.21, 0.89), ylabel="Volume Fraction")
    # ax3.set(yscale="log", ylim=(0.9e-15, 1.1e1))
    # ax3.set_ylabel(
    #     "Relative Frequency Difference $\ \\frac{\\| \omega_{2} - \\omega_{1} \|}{\omega_{1}}$ ($\%$)"
    # )

    # ax[1].yaxis.label.set_color(p1.get_color())
    # ax2.yaxis.label.set_color(p2.get_color())
    # ax3.yaxis.label.set_color(p3.get_color())

    # ax[1].tick_params(axis="y", colors=p1.get_color())
    # ax2.tick_params(axis="y", colors=p2.get_color())
    # ax3.tick_params(axis="y", colors=p3.get_color())

    # # ax[1].legend(handles=[p0, p1],loc=[0.75, 0.7])
    # ax[1].spines["top"].set_visible(False)
    # ax2.spines["top"].set_visible(False)
    # ax3.spines["top"].set_visible(False)
    # ax[1].xaxis.set_ticks_position("bottom")
    # ax[1].yaxis.set_ticks_position("left")
    # ax[1].spines["right"].set_visible(False)
    # ax2.yaxis.set_ticks_position("right")
    # ax3.yaxis.set_ticks_position("right")

    # ax[1].tick_params(direction="out")
    # ax2.tick_params(direction="out")
    # ax3.tick_params(direction="out")
    # ax[1].tick_params(which="minor", direction="out")
    # ax2.tick_params(which="minor", direction="out")
    # ax3.tick_params(which="minor", direction="out")

    # plt.savefig(
    #     "final_results/square/square_freq_base.png", bbox_inches="tight", dpi=1000
    # )

    # fig, ax = plt.subplots(figsize=(7, 2.5), constrained_layout=True)
    # n_iter = 500
    # n_start = 3

    # plot_modeshape(ax, rho[0], levels=25)
    # plt.savefig("final_results/square/square_base_rho.png", bbox_inches="tight", dpi=200, pad_inches=0.0)

    # fig, ax = plt.subplots(figsize=(4.5, 2.5), constrained_layout=True)
    # n_iter = 500
    # n_start = 3

    # ax2 = ax.twinx()
    # ax3 = ax.twinx()
    # ax2.spines.right.set_position(("axes", 1.05))
    # ax3.spines.right.set_position(("axes", 1.25))

    # (p1,) = ax.plot(omega[0][n_start:n_iter, 0], color="b", linewidth=0.5)
    # (p2,) = ax2.plot(vol[0][n_start:n_iter], color="k", linewidth=0.5)
    # ax2.axhline(
    #     y=0.4,
    #     xmin=0.045,
    #     xmax=0.96,
    #     color="k",
    #     linestyle="--",
    #     alpha=0.5,
    #     linewidth=0.5,
    # )
    # a = (np.abs(omega[0][n_start:n_iter, 0] - omega[0][n_start:n_iter, 1])) / (
    #     omega[0][n_start:n_iter, 0]
    # )
    # for i in range(len(a)):
    #     if a[i] > 1e-5:
    #         a[i] = a[i - 1]

    # (p3,) = ax3.plot(
    #     a,
    #     color="r",
    #     linewidth=0.25,
    # )

    # ax.set(ylim=(0, 100), ylabel="Frequency $\omega_{1}$ (rad/s)", xlabel="Iteration")
    # ax2.set(ylim=(0.21, 0.89), ylabel="Volume Fraction")
    # ax3.set(yscale="log", ylim=(0.9e-15, 1.1e1))
    # ax3.set_ylabel(
    #     "Relative Frequency Difference $\ \\frac{\\| \omega_{2} - \\omega_{1} \|}{\omega_{1}}$ ($\%$)"
    # )

    # ax.yaxis.label.set_color(p1.get_color())
    # ax2.yaxis.label.set_color(p2.get_color())
    # ax3.yaxis.label.set_color(p3.get_color())

    # ax.tick_params(axis="y", colors=p1.get_color())
    # ax2.tick_params(axis="y", colors=p2.get_color())
    # ax3.tick_params(axis="y", colors=p3.get_color())

    # # ax[1].legend(handles=[p0, p1],loc=[0.75, 0.7])
    # ax.spines["top"].set_visible(False)
    # ax2.spines["top"].set_visible(False)
    # ax3.spines["top"].set_visible(False)
    # ax.xaxis.set_ticks_position("bottom")
    # ax.yaxis.set_ticks_position("left")
    # ax.spines["right"].set_visible(False)
    # ax2.yaxis.set_ticks_position("right")
    # ax3.yaxis.set_ticks_position("right")

    # ax.tick_params(direction="out")
    # ax2.tick_params(direction="out")
    # ax3.tick_params(direction="out")
    # ax.tick_params(which="minor", direction="out")
    # ax2.tick_params(which="minor", direction="out")
    # ax3.tick_params(which="minor", direction="out")

    # plt.savefig(
    #     "final_results/square/square_freq.png", bbox_inches="tight", dpi=1000, pad_inches=0.0
    # )


def plot_4(rho, vol, stress, omega, phi0, phi1, phi2, phi3, phi4, phi5):
    nrow = 1
    ncol = 3
    fig1, axs = plt.subplots(nrow, ncol, figsize=(7, 2.4), constrained_layout=True)
    text = [
        "SIMP, spatial filter (Baseline)",
        "RAMP, spatial filter",
        "RAMP, Helmholtz filter",
    ]
    for j in range(ncol):
        plot_modeshape(axs[j], rho[j])
        axs[j].text(
            0.5,
            -0.025,
            text[j],
            transform=axs[j].transAxes,
            # fontsize=3.5,
            va="top",
            ha="center",
        )

    # save fig1
    fig1.savefig(
        "../output/final_results/square/square_methods.png", bbox_inches="tight", dpi=1000
    )

    fig2, ax = plt.subplots()
    n_iter = 500
    n_start = 3

    ax2 = ax.twinx()
    ax3 = ax.twinx()
    ax2.spines.right.set_position(("axes", 1.05))
    ax3.spines.right.set_position(("axes", 1.25))

    (p1,) = ax.plot(
        omega[0][n_start:n_iter, 0],
        color="b",
        linewidth=0.5,
    )
    (p2,) = ax2.plot(
        vol[0][n_start:n_iter], color="k", label="SIMP, spatial filter", linewidth=0.5
    )
    (p3,) = ax3.plot(
        (np.abs(omega[0][n_start:n_iter, 0] - omega[0][n_start:n_iter, 1]))
        / (omega[0][n_start:n_iter, 0]),
        color="r",
        linewidth=0.25,
    )

    (p4,) = ax.plot(
        omega[1][n_start:n_iter, 0],
        color="b",
        linestyle="--",
        linewidth=0.5,
    )
    (p5,) = ax2.plot(
        vol[1][n_start:n_iter],
        color="k",
        label="RAMP, spatial filter",
        linestyle="--",
        linewidth=0.5,
    )
    (p6,) = ax3.plot(
        (np.abs(omega[1][n_start:n_iter, 0] - omega[1][n_start:n_iter, 1]))
        / (omega[1][n_start:n_iter, 0]),
        color="r",
        linewidth=0.25,
        alpha=0.5,
        linestyle="--",
    )

    (p7,) = ax.plot(
        omega[2][n_start:n_iter, 0],
        color="b",
        linestyle=":",
        linewidth=0.5,
    )
    (p8,) = ax2.plot(
        vol[2][n_start:n_iter],
        color="k",
        label="RAMP, Helmholtz filter",
        linestyle=":",
        linewidth=0.5,
    )
    (p9,) = ax3.plot(
        (np.abs(omega[2][n_start:n_iter, 0] - omega[2][n_start:n_iter, 1]))
        / (omega[2][n_start:n_iter, 0]),
        color="r",
        linewidth=0.25,
        alpha=0.5,
        linestyle=":",
    )

    p10 = ax2.axhline(
        y=0.4,
        xmin=0.045,
        xmax=0.96,
        color="k",
        alpha=0.5,
        linewidth=0.25,
        label="Volume Constraint",
    )
    ax.set(ylim=(0, 100), ylabel="Frequency $\omega_{1}$ (rad/s)", xlabel="Iteration")
    ax2.set(ylim=(0.11, 0.99), ylabel="Volume Fraction")
    ax3.set(yscale="log", ylim=(0.9e-15, 1.1e1))
    ax3.set_ylabel(
        "Relative Frequency Difference $\ \\frac{\\| \omega_{2} - \\omega_{1} \|}{\omega_{1}}$ ($\%$)"
    )
    ax.legend(handles=[p2, p5, p8, p10], loc=[0.45, 0.6])

    ax.yaxis.label.set_color(p1.get_color())
    ax2.yaxis.label.set_color(p2.get_color())
    ax3.yaxis.label.set_color(p3.get_color())

    ax.tick_params(axis="y", colors=p1.get_color())
    ax2.tick_params(axis="y", colors=p2.get_color())
    ax3.tick_params(axis="y", colors=p3.get_color())

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

    fig2.savefig("../output/final_results/square/square_freq_methods.pdf", bbox_inches="tight")


def plot_5(rho, vol, stress, omega, phi0, phi1, phi2, phi3, phi4, phi5):
    nrow = 1
    ncol = 4
    fig1, axs = plt.subplots(nrow, ncol, figsize=(7, 1.8), constrained_layout=True)
    text = [
        "125 $\\times$ 125",
        "250 $\\times$ 250",
        "500 $\\times$ 500",
        "1000 $\\times$ 1000",
    ]
    for j in range(ncol):
        plot_modeshape(axs[j], rho[j])
        axs[j].text(
            0.5,
            -0.025,
            text[j],
            transform=axs[j].transAxes,
            va="top",
            ha="center",
        )

    fig1.savefig("../output/final_results/square/square_mesh.png", bbox_inches="tight", dpi=500)

    fig2, ax = plt.subplots()
    n_iter = 500
    n_start = 3

    ax2 = ax.twinx()
    ax3 = ax.twinx()
    ax2.spines.right.set_position(("axes", 1.05))
    ax3.spines.right.set_position(("axes", 1.25))

    (p1,) = ax.plot(
        omega[0][n_start:n_iter, 0],
        color="b",
        linewidth=0.5,
    )
    (p2,) = ax2.plot(vol[0][n_start:n_iter], color="k", label=text[0], linewidth=0.5)
    (p3,) = ax3.plot(
        (np.abs(omega[0][n_start:n_iter, 0] - omega[0][n_start:n_iter, 1]))
        / (omega[0][n_start:n_iter, 0]),
        color="r",
        linewidth=0.25,
    )

    (p4,) = ax.plot(
        omega[1][n_start:n_iter, 0],
        color="b",
        linestyle="--",
        linewidth=0.5,
    )
    (p5,) = ax2.plot(
        vol[1][n_start:n_iter],
        color="k",
        label=text[1],
        linestyle="--",
        linewidth=0.5,
    )
    a = (np.abs(omega[1][n_start:n_iter, 0] - omega[1][n_start:n_iter, 1])) / (
        omega[1][n_start:n_iter, 0]
    )
    for i in range(len(a)):
        if a[i] > 1e-5:
            a[i] = a[i - 1]

    (p6,) = ax3.plot(
        a,
        color="r",
        linewidth=0.25,
        alpha=0.75,
        linestyle="--",
    )

    (p7,) = ax.plot(
        omega[2][n_start:n_iter, 0],
        color="b",
        linestyle=":",
        linewidth=0.5,
    )
    (p8,) = ax2.plot(
        vol[2][n_start:n_iter],
        color="k",
        label=text[2],
        linestyle=":",
        linewidth=0.5,
    )
    (p9,) = ax3.plot(
        (np.abs(omega[2][n_start:n_iter, 0] - omega[2][n_start:n_iter, 1]))
        / (omega[2][n_start:n_iter, 0]),
        color="r",
        linewidth=0.25,
        alpha=0.5,
        linestyle=":",
    )

    (p10,) = ax.plot(
        omega[3][n_start:n_iter, 0],
        color="b",
        linestyle="-.",
        linewidth=0.5,
    )
    (p11,) = ax2.plot(
        vol[3][n_start:n_iter],
        color="k",
        label=text[3],
        linestyle="-.",
        linewidth=0.5,
    )
    (p12,) = ax3.plot(
        (np.abs(omega[3][n_start:n_iter, 0] - omega[3][n_start:n_iter, 1]))
        / (omega[3][n_start:n_iter, 0]),
        color="r",
        linewidth=0.25,
        alpha=0.25,
        linestyle="-.",
    )

    ax.set(ylim=(0, 100), ylabel="Frequency $\omega_{1}$ (rad/s)", xlabel="Iteration")
    ax2.set(ylim=(0.16, 0.99), ylabel="Volume Fraction")
    ax3.set(yscale="log", ylim=(0.9e-15, 1.1e1))
    ax3.set_ylabel(
        "Relative Frequency Difference $\ \\frac{\\| \omega_{2} - \\omega_{1} \|}{\omega_{1}}$ ($\%$)"
    )
    ax.legend(handles=[p2, p5, p8, p11], loc=[0.55, 0.45], title="Mesh Size")

    ax.yaxis.label.set_color(p1.get_color())
    ax2.yaxis.label.set_color(p2.get_color())
    ax3.yaxis.label.set_color(p3.get_color())

    ax.tick_params(axis="y", colors=p1.get_color())
    ax2.tick_params(axis="y", colors=p2.get_color())
    ax3.tick_params(axis="y", colors=p3.get_color())

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

    fig2.savefig("final_results/square/square_freq_mesh.pdf", bbox_inches="tight")


def plot_6(rho, dis, vol, stress_iter, stress, omega, phi0):
    text = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)", "(g)", ""]
    nrow = 4
    ncol = 4
    fig, axs = plt.subplots(nrow, ncol, figsize=(7, 7.2), constrained_layout=True)
    a = int(400 * 0.75)
    b = int(400 * 0.25)

    # phi0[1] = -1 * phi0[1]
    # phi0[4] = -1 * phi0[4]
    phi0[5] = -1 * phi0[5]
    phi0[6] = -1 * phi0[6]
    shift_scale = 2

    for i in range(nrow):
        for j in range(ncol):
            if i == 0:
                plot_modeshape(axs[i, j], rho[j], levels=50)

            if i == 1:
                plot_modeshape(
                    axs[i, j],
                    rho[j],
                    shift_scale * phi0[j],
                    surface=True,
                    alpha=0.75,
                    levels=1,
                    aa=True,
                )

            if i == 2:
                if j != 3:
                    plot_modeshape(axs[i, j], rho[j + 4], levels=50)
                else:
                    plot_modeshape(
                        axs[i, j],
                        rho[j + 4 - 1],
                        levels=1,
                        zoom=True,
                        alpha=0.9,
                    )

            if i == 3:
                if j != 3:
                    plot_modeshape(
                        axs[i, j],
                        rho[j + 4],
                        shift_scale * phi0[j + 4],
                        stress=stress[j + 4],
                        levels=100,
                    )
                else:
                    plot_modeshape(
                        axs[i, j],
                        rho[j + 4 - 1],
                        shift_scale * phi0[j + 4 - 1],
                        stress=stress[j + 4 - 1],
                        zoom=True,
                        levels=100,
                    )

            if i == 1 or i == 3:
                axs[i, j].text(
                    0.5,
                    -0.07,
                    text[(i - 1) // 2 * ncol + j],
                    transform=axs[i, j].transAxes,
                    va="bottom",
                    ha="center",
                    fontweight="bold",
                )

            if i == 1:
                ic((phi0[j][a, a][0] ** 2 + phi0[j][a, a][1] ** 2))
                ic((phi0[j][a, b][0] ** 2 + phi0[j][a, b][1] ** 2))
                # ic((phi0[j][b, a][0] ** 2 + phi0[j][b, a][1] ** 2))
                # ic((phi0[j][b, b][0] ** 2 + phi0[j][b, b][1] ** 2))
                d1 = np.sqrt(phi0[j][a, a][0] ** 2 + phi0[j][a, a][1] ** 2)
                d2 = np.sqrt(phi0[j][a, b][0] ** 2 + phi0[j][a, b][1] ** 2)
                ic(d1)
                ic(d2)
                ic((d1 ** 2 + d2 ** 2)*0.5)
                
                a1 = (phi0[j][a, a][0] ** 2 + phi0[j][a, a][1] ** 2) * 0.5 + (
                    phi1[j][a, a][0] ** 2 + phi1[j][a, a][1] ** 2
                ) * 0.5
                a2 = (phi0[j][a, b][0] ** 2 + phi0[j][a, b][1] ** 2) * 0.5 + (
                    phi1[j][a, b][0] ** 2 + phi1[j][a, b][1] ** 2
                ) * 0.5
                a3 = (phi0[j][b, a][0] ** 2 + phi0[j][b, a][1] ** 2) * 0.5 + (
                    phi1[j][b, a][0] ** 2 + phi1[j][b, a][1] ** 2
                ) * 0.5
                a4 = (phi0[j][b, b][0] ** 2 + phi0[j][b, b][1] ** 2) * 0.5 + (
                    phi1[j][b, b][0] ** 2 + phi1[j][b, b][1] ** 2
                ) * 0.5
                ic(a1)
                ic(a2)
                ic(a3)
                ic(a4)

            if (i == 0 and j != 0) or i == 1:
                axs[i, j].scatter(
                    [6, 6, 2, 2],
                    [6 + 8 / 400, 2 + 8 / 400, 6 + 8 / 400, 2 + 8 / 400],
                    marker="o",
                    s=0.5,
                    color="r",
                )
                if i == 1:
                    axs[i, j].scatter(
                        [
                            6 + shift_scale * phi0[j][a, a][0],
                            6 + shift_scale * phi0[j][a, b][0],
                            2 + shift_scale * phi0[j][b, a][0],
                            2 + shift_scale * phi0[j][b, b][0],
                        ],
                        [
                            6 + shift_scale * phi0[j][a, a][1] + 8 / 400,
                            2 + shift_scale * phi0[j][a, b][1] + 8 / 400,
                            6 + shift_scale * phi0[j][b, a][1] + 8 / 400,
                            2 + shift_scale * phi0[j][b, b][1] + 8 / 400,
                        ],
                        marker="o",
                        s=0.5,
                        color="k",
                    )
        # save each row separately

    for ax in axs[1, :]:
        ax.set_anchor("S")
    for ax in axs[3, :]:
        ax.set_anchor("S")

    # plt.savefig(
    #     "../output/final_results/square/square_stress_new.png",
    #     bbox_inches="tight",
    #     dpi=300,
    #     pad_inches=0.0,
    # )

    fig, axs = plt.subplots(1, ncol, figsize=(7, 2), constrained_layout=True)
    text1 = [
        "$ h = 0.084 \ \mathrm{m^2}$\n $\\sigma_{max} = 2.50 \ \mathrm{MPa}$",
        "$ h = 0.054 \ \mathrm{m^2}$",
        "$ h = 0.040 \ \mathrm{m^2}$",
        "$ h = 0.034 \ \mathrm{m^2}$",
    ]
    for j in range(ncol):
        plot_modeshape(axs[j], rho[j], levels=50)
        if j != 0:
            axs[j].scatter(
                [6, 6, 2, 2],
                [6 + 8 / 400, 2 + 8 / 400, 6 + 8 / 400, 2 + 8 / 400],
                marker="o",
                s=1.0,
                color="r",
            )
        # axs[j].text(
        #     0.5,
        #     0.0,
        #     text1[j],
        #     transform=axs[j].transAxes,
        #     va="bottom",
        #     ha="center",
        #     fontweight="bold",
        # )
    # plt.savefig(
    #     "../output/final_results/square/square_dis_stress_1.png",
    #     bbox_inches="tight",
    #     dpi=500,
    #     pad_inches=0.0,
    # )

    fig, axs = plt.subplots(1, ncol, figsize=(7, 2), constrained_layout=True)
    text2 = [
        "$\\sigma_{max} = 1.95 \ \mathrm{MPa}$",
        "$\\sigma_{max} = 1.90 \ \mathrm{MPa}$",
        "$\\sigma_{max} = 1.87 \ \mathrm{MPa}$",
        "",
    ]
    for j in range(ncol):
        if j != 3:
            plot_modeshape(axs[j], rho[j + 4], levels=50)
        else:
            plot_modeshape(
                axs[j],
                rho[j + 4 - 1],
                levels=1,
                zoom=True,
                alpha=0.9,
            )
        # axs[j].text(
        #     0.5,
        #     0.0,
        #     text2[j],
        #     transform=axs[j].transAxes,
        #     va="bottom",
        #     ha="center",
        #     fontweight="bold",
        # )
    # plt.savefig(
    #     "../output/final_results/square/square_dis_stress_2.png",
    #     bbox_inches="tight",
    #     dpi=500,
    #     pad_inches=0.0,
    # )

    fig, axs = plt.subplots(1, ncol, figsize=(7, 2), constrained_layout=True)
    for j in range(ncol):
        plot_modeshape(
            axs[j],
            rho[j],
            shift_scale * phi0[j],
            surface=True,
            alpha=0.75,
            levels=1,
            aa=True,
        )
        axs[j].text(
            0.5,
            -0.07,
            text[j],
            transform=axs[j].transAxes,
            va="bottom",
            ha="center",
            fontweight="bold",
        )
        axs[j].scatter(
            [6, 6, 2, 2],
            [6 + 8 / 400, 2 + 8 / 400, 6 + 8 / 400, 2 + 8 / 400],
            marker="o",
            s=0.2,
            color="r",
            alpha=0.5,
        )
        # axs[j].scatter(
        #     [
        #         6 + shift_scale * phi0[j][a, a][0],
        #         6 + shift_scale * phi0[j][a, b][0],
        #         2 + shift_scale * phi0[j][b, a][0],
        #         2 + shift_scale * phi0[j][b, b][0],
        #     ],
        #     [
        #         6 + shift_scale * phi0[j][a, a][1] + 8 / 400,
        #         2 + shift_scale * phi0[j][a, b][1] + 8 / 400,
        #         6 + shift_scale * phi0[j][b, a][1] + 8 / 400,
        #         2 + shift_scale * phi0[j][b, b][1] + 8 / 400,
        #     ],
        #     marker="o",
        #     s=0.2,
        #     color="k",
        #     alpha=0.5,
        # )
    for ax in axs:
        ax.set_anchor("S")
    # plt.savefig(
    #     "../output/final_results/square/square_dis_stress_3.png",
    #     bbox_inches="tight",
    #     dpi=300,
    #     pad_inches=0.0,
    # )

    fig, axs = plt.subplots(
        1,
        ncol,
        figsize=(7, 2),
        constrained_layout=True,
        gridspec_kw={"width_ratios": [1, 1, 1, 0.96]},
    )
    for j in range(ncol):
        if j != 3:
            plot_modeshape(
                axs[j],
                rho[j + 4],
                shift_scale * phi0[j + 4],
                stress=stress[j + 4],
                levels=100,
            )
        else:
            plot_modeshape(
                axs[j],
                rho[j + 4 - 1],
                shift_scale * phi0[j + 4 - 1],
                stress=stress[j + 4 - 1],
                zoom=True,
                levels=100,
            )

        axs[j].text(
            0.5,
            -0.07,
            text[j + 4],
            transform=axs[j].transAxes,
            va="bottom",
            ha="center",
            fontweight="bold",
        )
    for ax in axs:
        ax.set_anchor("S")
    # plt.savefig(
    #     "../output/final_results/square/square_dis_stress_4.png",
    #     bbox_inches="tight",
    #     dpi=400,
    #     pad_inches=0.0,
    # )


def plot_7(rho, dis, vol, stress_iter, stress, omega, phi0):
    # fig2, ax = plt.subplots(figsize=(4.0, 2.5), constrained_layout=True)
    fig2, axs = plt.subplot_mosaic([["(a)"], ["(b)"]], figsize=(3.0, 5.1))
    # text = ["(a)", "(b)"]

    for n, (key, ax) in enumerate(axs.items()):
        if n == 0:
            n_iter = 1000
            n_start = 5
        else:
            n_iter = 1000
            n_start = 5

        ax2 = ax.twinx()
        ax3 = ax.twinx()
        ax2.spines.right.set_position(("axes", 1.02))
        ax3.spines.right.set_position(("axes", 1.21))

        a = (np.abs(omega[n][n_start:n_iter, 0] - omega[n][n_start:n_iter, 1])) / (
            omega[n][n_start:n_iter, 0]
        )

        # for i in range(len(a)):
        #     if a[i] > 1e-11:
        #         a[i] = a[i - 1]

        (p1,) = ax.plot(
            a,
            color="r",
            linewidth=0.25,
            label="$\ \\| \omega_{2} - \\omega_{1} \| / \omega_{1}$",
        )

        (p2,) = ax2.plot(vol[n][n_start:n_iter], color="k", linewidth=0.5, label="$V$")

        if n == 0:
            (p3,) = ax3.plot(
                dis[0][n_start:n_iter] * 100, color="b", linewidth=0.5, label="$h$"
            )
            p5 = ax3.axhline(
                xmin=0.04,
                xmax=0.96,
                y=0.054 * 100,
                color="b",
                linestyle="--",
                linewidth=1,
                alpha=0.25,
                label="$h_{\mathrm{constraint}}$",
            )
        else:
            (p3,) = ax3.plot(
                stress_iter[1][n_start:n_iter],
                color="b",
                linewidth=0.5,
                label="$\sigma_{vM}$",
            )
            p5 = ax3.axhline(
                xmin=0.04,
                xmax=0.96,
                y=np.sqrt(3.6),
                color="b",
                linestyle="--",
                linewidth=1,
                alpha=0.25,
                label="$\sigma_{\mathrm{constraint}}$",
            )

        p4 = ax2.axhline(
            xmin=0.04,
            xmax=0.96,
            y=0.4,
            color="k",
            linestyle="--",
            linewidth=1,
            alpha=0.25,
            label="$V_{\mathrm{constraint}}$",
        )

        ax.set(yscale="log", ylim=(1e-15, 0.9e3), xlabel="Iteration")
        ax.set_ylabel(
            "Relative Difference $\ \\| \omega_{2} - \\omega_{1} \| / \omega_{1}$ ($\%$)"
        )
        ax2.set(ylim=(0.16, 0.99), ylabel="Volume Fraction $V$ ($\%$)")

        if n == 0:
            ax3.set(
                ylim=(-0.2, 6.0),
                ylabel="Displacement $h \ (\mathrm{m}^2) \\times 10^{-2}$",
            )
        else:
            ax3.set(
                ylim=(0.01, 2.05),
                ylabel="von Mises Stress $\sigma_{vM} \ (\mathrm{MPa})$",
            )

        handles, labels = [], []
        handles.append(p3)
        labels.append(p3.get_label())
        handles.append(p5)
        labels.append(p5.get_label())
        handles.append(p2)
        labels.append(p2.get_label())
        handles.append(p4)
        labels.append(p4.get_label())
        handles.append(p1)
        labels.append(p1.get_label())

        ax.legend(
            handles,
            labels,
            ncol=1,
            loc=[0.52, 0.42],
            frameon=False,
        )

        ax.yaxis.label.set_color(p1.get_color())
        ax2.yaxis.label.set_color(p2.get_color())
        ax3.yaxis.label.set_color(p3.get_color())
        ax.tick_params(axis="y", colors=p1.get_color())
        ax2.tick_params(axis="y", colors=p2.get_color())
        ax3.tick_params(axis="y", colors=p3.get_color())

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

        ax.text(
            -0.1,
            1.05,
            key,
            transform=ax.transAxes,
            horizontalalignment="left",
            verticalalignment="top",
            weight="bold",
        )

    fig2.savefig(
        "../output/final_results/square/square_dis_stress.pdf",
        bbox_inches="tight",
        pad_inches=0.0,
    )

    # text = ["w/o", "$\sigma_\t{vM} = 2.00$", "$\sigma_\t{vM} = 2.24$"]

    # fig2, ax = plt.subplots(
    #     figsize=(4.5, 2.8), constrained_layout=True
    # )  # figsize=(3.3, 1.8)
    # n_iter = 500
    # n_start = 3

    # ax2 = ax.twinx()
    # ax3 = ax.twinx()
    # ax2.spines.right.set_position(("axes", 1.02))
    # ax3.spines.right.set_position(("axes", 1.175))

    # (p1,) = ax.plot(
    #     stress_iter[0][n_start:n_iter],
    #     color="b",
    #     linewidth=0.25,
    # )
    # (p2,) = ax2.plot(vol[0][n_start:n_iter], color="k", label=text[0], linewidth=0.25)
    # (p3,) = ax3.plot(
    #     (np.abs(omega[0][n_start:n_iter, 0] - omega[0][n_start:n_iter, 1]))
    #     / (omega[0][n_start:n_iter, 0]),
    #     color="r",
    #     linewidth=0.25,
    # )

    # (p4,) = ax.plot(
    #     stress_iter[1][n_start:n_iter],
    #     color="b",
    #     linestyle=":",
    #     linewidth=0.25,
    # )
    # (p5,) = ax2.plot(
    #     vol[1][n_start:n_iter],
    #     color="k",
    #     label=text[1],
    #     linestyle=":",
    #     linewidth=0.25,
    # )
    # a = (np.abs(omega[1][n_start:n_iter, 0] - omega[1][n_start:n_iter, 1])) / (
    #     omega[1][n_start:n_iter, 0]
    # )
    # (p6,) = ax3.plot(
    #     a,
    #     color="r",
    #     linewidth=0.25,
    #     alpha=0.75,
    #     linestyle=":",
    # )

    # (p7,) = ax.plot(
    #     stress_iter[2][n_start:n_iter],
    #     color="b",
    #     linestyle="--",
    #     linewidth=0.25,
    # )
    # (p8,) = ax2.plot(
    #     vol[2][n_start:n_iter],
    #     color="k",
    #     label=text[2],
    #     linestyle="--",
    #     linewidth=0.25,
    # )
    # a = (np.abs(omega[2][n_start:n_iter, 0] - omega[2][n_start:n_iter, 1])) / (
    #     omega[2][n_start:n_iter, 0]
    # )
    # (p9,) = ax3.plot(
    #     a,
    #     color="r",
    #     linewidth=0.25,
    #     alpha=0.75,
    #     linestyle="--",
    # )

    # ax.set(
    #     xlim=(-20, 500),
    #     ylim=(0.0, 2.4),
    #     ylabel="von Mises Stress $\sigma_\t{vM}$ (MPa)",
    #     xlabel="Iteration",
    # )
    # ax2.set(ylim=(0.16, 0.99), ylabel="Volume Fraction")
    # ax3.set(yscale="log", ylim=(0.9e-15, 1.1e1))
    # ax3.set_ylabel(
    #     "Relative Frequency Difference $\ \\| \omega_{2} - \\omega_{1} \| / \omega_{1}$ ($\%$)"
    # )
    # ax.legend(
    #     handles=[p2, p8, p5],
    #     # loc=[0.35, 0.35],
    #     loc=[0.55, 0.45],
    #     title="Stress Constraint (MPa)",
    #     frameon=False,
    # )

    # # add minor ticks
    # # ax.xaxis.set_minor_locator(AutoMinorLocator())
    # # ax.yaxis.set_minor_locator(AutoMinorLocator())
    # # ax2.yaxis.set_minor_locator(AutoMinorLocator())
    # # ax3.yaxis.set_minor_locator(AutoMinorLocator())
    # # ax.tick_params(which="minor", length=2, color="k")
    # # ax2.tick_params(which="minor", length=2, color="k")
    # # ax3.tick_params(which="minor", length=2, color="k")

    # ax.yaxis.label.set_color(p1.get_color())
    # ax2.yaxis.label.set_color(p2.get_color())
    # ax3.yaxis.label.set_color(p3.get_color())

    # ax.tick_params(axis="y", colors=p1.get_color())
    # ax2.tick_params(axis="y", colors=p2.get_color())
    # ax3.tick_params(axis="y", colors=p3.get_color())

    # ax.spines["top"].set_visible(False)
    # ax2.spines["top"].set_visible(False)
    # ax3.spines["top"].set_visible(False)
    # ax.xaxis.set_ticks_position("bottom")
    # ax.yaxis.set_ticks_position("left")
    # ax.spines["right"].set_visible(False)
    # ax2.yaxis.set_ticks_position("right")
    # ax3.yaxis.set_ticks_position("right")

    # ax.tick_params(direction="out")
    # ax2.tick_params(direction="out")
    # ax3.tick_params(direction="out")
    # ax.tick_params(which="minor", direction="out")
    # ax2.tick_params(which="minor", direction="out")
    # ax3.tick_params(which="minor", direction="out")

    # fig2.savefig(
    #     "final_results/square/square_freq_stress.pdf",
    #     bbox_inches="tight",
    #     pad_inches=0.0,
    # )


if __name__ == "__main__":
    # dir_result1 = "final_results/square/filter/nx=125, vol=0.4, r0=1.0, p=False/"
    # dir_result2 = "final_results/square/filter/nx=125, vol=0.4, r0=2.0, p=False/"
    # dir_result3 = "final_results/square/filter/nx=125, vol=0.4, r0=4.0, p=False/"

    # dir_result4 = "final_results/square/filter/nx=250, vol=0.4, r0=1.0, p=False/"
    # dir_result5 = "final_results/square/filter/nx=250, vol=0.4, r0=2.0, p=False/"
    # dir_result6 = "final_results/square/filter/nx=250, vol=0.4, r0=4.0, p=False/"

    # dir_result7 = "final_results/square/filter/nx=500, vol=0.4, r0=1.0, p=False/"
    # dir_result8 = "final_results/square/filter/nx=500, vol=0.4, r0=2.0, p=False/"
    # dir_result9 = "final_results/square/filter/nx=500, vol=0.4, r0=4.0, p=False/"

    # dir_result10 = "final_results/square/filter/nx=1000, vol=0.4, r0=1.0, p=False/"
    # dir_result11 = "final_results/square/filter/nx=1000, vol=0.4, r0=2.0, p=False/"
    # dir_result12= "final_results/square/filter/nx=1000, vol=0.4, r0=4.0, p=False/"

    # rho, _, _,_, _, _, _, _, _, _, _, _ = assmble_data(12, 500)
    # with plt.style.context(["nature"]):
    #     plot_1(3, 4, rho)

    # dir_result1 = "final_results/square/stress/nx=500, vol=0.2, r0=1.0, p=False/"
    # dir_result2 = "final_results/square/stress/nx=500, vol=0.2, s=5000000000000.0, r0=1.0, p=False/"
    # dir_result3 = "final_results/square/stress/nx=500, vol=0.2, s=4000000000000.0, r0=1.0, p=False/"

    # dir_result4 = "final_results/square/stress/nx=500, vol=0.3, r0=1.0, p=False/"
    # dir_result5 = "final_results/square/stress/nx=500, vol=0.3, s=5000000000000.0, r0=1.0, p=False/"
    # dir_result6 = "final_results/square/stress/nx=500, vol=0.3, s=4000000000000.0, r0=1.0, p=False/"

    # dir_result7 = "final_results/square/stress/nx=500, vol=0.4, r0=1.0, p=False/"
    # dir_result8 = "final_results/square/stress/nx=500, vol=0.4, s=5000000000000.0, r0=1.0, p=False/"
    # dir_result9 = "final_results/square/stress/nx=500, vol=0.4, s=4000000000000.0, r0=1.0, p=False/"

    # dir_result10 = "final_results/square/stress/nx=500, vol=0.5, r0=1.0, p=False/"
    # dir_result11 = "final_results/square/stress/nx=500, vol=0.5, s=5000000000000.0, r0=1.0, p=False/"
    # dir_result12 = "final_results/square/stress/nx=500, vol=0.5, s=4000000000000.0, r0=1.0, p=False/"

    # dir_result13 = "final_results/square/stress/nx=500, vol=0.6, r0=1.0, p=False/"
    # dir_result14 = "final_results/square/stress/nx=500, vol=0.6, s=5000000000000.0, r0=1.0, p=False/"
    # dir_result15 = "final_results/square/stress/nx=500, vol=0.6, s=4000000000000.0, r0=1.0, p=False/"

    # rho, _,_,_, stress, _, _, _, _, _, _, _ = assmble_data(15, 600)
    # with plt.style.context(["nature"]):
    #     plot_2(5, 3, rho)



    # dir_result1 = "final_results/square/method/nx=500, vol=0.4, r0=1.0, p=False/"
    # dir_result2 = (
    #     "final_results/square/method/nx=500, vol=0.4, r0=1.0, f=spatial, K=ramp/"
    # )
    # dir_result3 = (
    #     "final_results/square/method/nx=500, vol=0.4, r0=1.0, f=helmholtz, K=ramp/"
    # )
    # rho, vol,_, _, stress, omega, phi0, phi1, phi2, phi3, phi4, phi5 = assmble_data(3, 500)
    # with plt.style.context(["nature"]):
    #     plot_4(rho, vol, stress, omega, phi0, phi1, phi2, phi3, phi4, phi5)

    # dir_result1 = "final_results/square/filter/nx=125, vol=0.4, r0=1.0, p=False/"
    # dir_result2 = "final_results/square/filter/nx=250, vol=0.4, r0=1.0, p=False/"
    # dir_result3 = "final_results/square/filter/nx=500, vol=0.4, r0=1.0, p=False/"
    # dir_result4 = "final_results/square/filter/nx=1000, vol=0.4, r0=1.0, p=False/"
    # rho, vol,_,_, stress, omega, phi0, phi1, phi2, phi3, phi4, phi5 = assmble_data(4, 500)
    # with plt.style.context(["nature"]):
    #     plot_5(rho, vol, stress, omega, phi0, phi1, phi2, phi3, phi4, phi5)

    # dir_result1 = (
    #     "final_results/square/stress/nx=500, vol=0.4, r0=2.1, K=simp, M=msimp/"
    # )
    # dir_result2 = (
    #     "final_results/square/stress/nx=500, vol=0.4, s=5000000000000.0, r0=2.1, K=simp, M=msimp/"
    # )
    # dir_result3 = (
    #     "final_results/square/stress/nx=500, vol=0.4, s=4000000000000.0, r0=2.1, K=simp, M=msimp/"
    # )

    dir_result1 = "../output/final_results/square/baseline/"
    dir_result2 = "../output/final_results/square/p, d=0.054/"
    dir_result3 = "../output/final_results/square/p, d=0.040/"
    dir_result4 = "../output/final_results/square/p, d=0.034/"
    dir_result5 = "../output/final_results/square/p, s=3.9/"
    dir_result6 = "../output/final_results/square/p, s=3.6/"
    dir_result7 = "../output/final_results/square/p, s=3.5/"
    (
        rho,
        vol,
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
    ) = assmble_data(7, 1000, 1000, 880, 1000, 600, 980, 1000)

    omegas = np.array(
        [
            omega[1][400, 0],
            omega[1][400, 1],
            omega[1][400, 2],
            omega[1][400, 3],
            omega[1][400, 4],
            omega[1][400, 5],
        ]
    )
    ic(omegas)
    ks_rho = 100
    eta = np.exp(-ks_rho * (omegas - omegas[0]))
    a = np.sum(eta)
    eta = eta / a
    ic(eta)
    a = (21.43 + 100 + 87.5 + 100) * 0.06
    b = (42 + 97 + 100 + 90 + 67 + 92) * 0.08 * 7 / 6
    c = 0.2 * 70
    ic(a + b + c)

    ic((0.215664**2 + 0.0979138**2) * 0.5 + (0.0727259**2 + 0.205458**2) * 0.5)

    with plt.style.context(["nature"]):
        plot_6(rho, dis, vol, stress_iter, stress, omega, phi0)

    # dir_result1 = "../output/final_results/square/p, d=0.054/"
    # dir_result2 = "../output/final_results/square/p, s=3.6/"
    # (
    #     rho,
    #     vol,
    #     dis,
    #     stress_iter,
    #     stress,
    #     omega,
    #     phi0,
    #     phi1,
    #     phi2,
    #     phi3,
    #     phi4,
    #     phi5,
    # ) = assmble_data(2, 1000)
    # with plt.style.context(["nature"]):
    #     plot_7(rho, dis, vol, stress_iter, stress, omega, phi0)

    # dir_result1 = (
    #     "../output/final_results/square/p, d=0.034/"
    # )
    # rho, vol,_, _, stress, omega, phi0, phi1, phi2, phi3, phi4, phi5 = assmble_data(
    #     1, 1000
    # )
    # with plt.style.context(["nature"]):
    #     plot_3(3, 2, rho, vol, stress, omega, phi0, phi1, phi2, phi3, phi4, phi5)
        
        
    # plt.show()
