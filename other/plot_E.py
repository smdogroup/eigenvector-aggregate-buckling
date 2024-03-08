import matplotlib.pylab as plt
import matplotlib.transforms as transforms
import numpy as np
import scienceplots


def colorbar(mappable, **kwargs):
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    last_axes = plt.gca()
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar = fig.colorbar(mappable, cax=cax, **kwargs)
    plt.sca(last_axes)
    return cbar


def softmax_a(fun, rho, lam):
    eta = np.zeros(len(lam), dtype=lam.dtype)
    for i in range(len(lam)):
        eta[i] = fun(-rho * (lam[i] - np.min(lam)))
    return eta


def softmax_ab(fun, rho, lam, lam_a, lam_b):
    """
    Compute the eta values
    """
    # ic(rho, lam, lam_a, lam_b)
    eta = np.zeros(len(lam), dtype=lam.dtype)
    for i in range(len(lam)):
        a = fun(rho * (lam[i] - lam_a))
        b = fun(rho * (lam[i] - lam_b))
        eta[i] = a - b
    # ic(eta[:10])
    return eta


if __name__ == "__main__":
    mu = np.load("./data/mu.npy")
    eta = np.load("./data/eta_tanh.npy")
    eta_exp = np.load("./data/eta_exp.npy")
    E = np.load("./data/E.npy")
    G = np.load("./data/G.npy")

    with plt.style.context(["nature"]):
        fig = plt.figure(figsize=(3.3, 3.3))
        ax = plt.gca()
        E = np.abs(E)
        E = (E) / (np.max(E))
        E = np.log10(np.abs(E))
        E = np.where(E < -15, -15, E)
        mts = ax.matshow(E, cmap="coolwarm")
        ax.tick_params(
            axis="x", which="both", bottom=False, top=True, labelbottom=False
        )
        colorbar(mts, label="$\log_{10}(E / E_{max}$)")
        ticks = np.arange(0.0, 100, 20.0)
        ticks = np.append(ticks, 10)
        ticks = np.append(ticks, 99)
        plt.xticks(ticks)
        plt.yticks(ticks)
        plt.savefig(
            "output/final_results/E.png",
            bbox_inches="tight",
            dpi=1000,
            pad_inches=0.0,
        )

    with plt.style.context(["nature"]):
        fig = plt.figure(figsize=(3.3, 3.3))
        ax = plt.gca()
        G = np.abs(G)
        G = (G) / (np.max(G))
        G = np.log10(np.abs(G))
        G = np.where(G < -15, -15, G)
        mts = ax.matshow(G, cmap="coolwarm")
        ticks = np.arange(0.0, 100, 20.0)
        ticks = np.append(ticks, 10)
        ticks = np.append(ticks, 99)
        plt.xticks(ticks)
        plt.yticks(ticks)
        ax.tick_params(
            axis="x", which="both", bottom=False, top=True, labelbottom=False
        )
        colorbar(mts, label="$\log_{10}(F / F_{max}$)")
        plt.savefig(
            "output/final_results/G.png",
            bbox_inches="tight",
            dpi=1000,
            pad_inches=0.0,
        )

    with plt.style.context(["nature"]):
        rho = np.array([100, 10.0, 4.0, 2.0])
        cr = np.array([0.0, 0.2, 0.3, 0.4])
        mk = np.array(["o", "p", "^", "X"])
        name = ["p0", "p1", "p2", "p3"]
        z_order = [1, 2, 3, 4]
        # z_order = [4, 3, 2, 1]

        fig, ax = plt.subplots(figsize=(3.8, (3.8 / 3.3) * 2.5))
        lam = -1.0 / mu
        print(lam)
        lam_a = lam[10] - np.min(np.abs(lam)) * 0.01
        lam_b = lam[20] + np.min(np.abs(lam)) * 0.01
        print(lam_a, lam_b)

        eta_tanh0 = np.array([])
        eta_tanh1 = np.array([])
        lam0 = np.array([])
        lam1 = np.array([])

        # rho = 11949.461942775142
        rho = np.array([100.0, 10.0, 4.0, 2.0])
        for k in range(len(rho)):
            eta_tanh = softmax_ab(np.tanh, rho[k], lam, lam_a, lam_b)
            eta_tanh = eta_tanh / np.sum(eta_tanh)

            plt.plot(
                lam,
                eta_tanh,
                color=plt.colormaps["bwr"](cr[k]),
                linewidth=0.5,
                zorder=z_order[k],
            )
            name[k] = plt.scatter(
                lam,
                eta_tanh,
                color=plt.colormaps["bwr"](cr[k]),
                linewidths=0.0,
                s=10,
                marker=mk[k],
                label=r"$\rho_{\eta} = $" + rho[k].astype(str),
                zorder=z_order[k],
            )
        top = 0.2
        plt.yscale("log")
        plt.ylim(1e-15, top)
        rect = plt.Rectangle(
            (lam_a, 0.0),
            lam_b - lam_a,
            top,
            linewidth=1,
            edgecolor="none",
            facecolor="red",
            alpha=0.2,
            zorder=11,
        )
        ax.add_patch(rect)
        plt.xlabel(r"$\lambda$")
        plt.ylabel(r"$\log_{10}(\eta_i)$")
        # y axis max limit to 0.1
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(which="minor", left=False)
        plt.legend(
            [rect, name[0], name[1], name[2], name[3]],
            [
                r"highlighted $\eta_i$",
                r"$\rho_{\eta} = $" + rho[0].astype(str),
                r"$\rho_{\eta} = $" + rho[1].astype(str),
                r"$\rho_{\eta} = $" + rho[2].astype(str),
                r"$\rho_{\eta} = $" + rho[3].astype(str),
            ],
            title="Hyperbolic Tangent Weight",
            frameon=False,
            loc="upper right",
        )

        plt.savefig(
            "output/final_results/eta_tanh.png",
            bbox_inches="tight",
            dpi=1000,
            pad_inches=0.01,
        )

        fig, ax = plt.subplots(figsize=(3.8, (3.8 / 3.3) * 2.5))

        eta_exp0 = np.array([])
        eta_exp1 = np.array([])
        lam0 = np.array([])
        lam1 = np.array([])

        for k in range(len(rho)):
            eta_exp = softmax_a(np.exp, rho[k], lam)
            eta_exp = eta_exp / np.sum(eta_exp)
            plt.plot(
                lam,
                eta_exp,
                color=plt.colormaps["bwr"](cr[k]),
                linewidth=0.5,
                zorder=z_order[k],
            )
            name[k] = plt.scatter(
                lam,
                eta_exp,
                color=plt.colormaps["bwr"](cr[k]),
                linewidths=0.0,
                s=10,
                marker=mk[k],
                label=r"$\rho_{\eta} = $" + rho[k].astype(str),
                zorder=z_order[k],
            )
        top = 2
        plt.yscale("log")
        plt.ylim(1e-15, top)
        rect = plt.Rectangle(
            (lam[0] - 0.01 * lam[0], 0.0),
            0.02 * lam[0],
            top,
            linewidth=1,
            edgecolor="none",
            facecolor="red",
            alpha=0.2,
            zorder=11,
        )
        ax.add_patch(rect)
        plt.xlabel(r"$\lambda$")
        plt.ylabel(r"$\log_{10}(\eta_i)$")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.legend(
            [rect, name[0], name[1], name[2], name[3]],
            [
                r"highlighted $\eta_i$",
                r"$\rho_{\eta} = $" + rho[0].astype(str),
                r"$\rho_{\eta} = $" + rho[1].astype(str),
                r"$\rho_{\eta} = $" + rho[2].astype(str),
                r"$\rho_{\eta} = $" + rho[3].astype(str),
            ],
            title="Exponential Weight",
            frameon=False,
            # bbox_to_anchor=(0.42, 0.8),
        )

        plt.savefig(
            "output/final_results/eta_exp.png",
            bbox_inches="tight",
            dpi=1000,
            pad_inches=0.01,
        )

    # plt.show()
