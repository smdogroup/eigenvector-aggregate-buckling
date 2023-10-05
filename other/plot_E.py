import matplotlib.pylab as plt
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


if __name__ == "__main__":
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
        plt.savefig("./data/E.png", bbox_inches="tight", dpi=1000)

    with plt.style.context(["nature"]):
        fig = plt.figure(figsize=(3.3, 3.3))
        ax = plt.gca()
        G = np.abs(G)
        G = (G) / (np.max(G))
        G = np.log10(np.abs(G))
        G = np.where(G < -15, -15, G)
        mts = ax.matshow(G, cmap="coolwarm")
        ax.tick_params(
            axis="x", which="both", bottom=False, top=True, labelbottom=False
        )
        colorbar(mts, label="$\log_{10}(G / G_{max}$)")
        plt.savefig("./data/G.png", bbox_inches="tight", dpi=1000)
