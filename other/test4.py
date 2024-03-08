from icecream import ic
import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import Axes3D, proj3d
import mpmath as mp
import numpy as np
from scipy.linalg import eigh, expm
from scipy.optimize import minimize
from scipy import sparse, spatial


def Eij_a(fun, rho, trace, lam_min, lam1, lam2):
    """
    Compute the precise value of the E_{ij} term

        E_{ij} = exp(-rho * (lam1 - lam_min)) / trace

        if lam1 == lam2:
            E_{ij} = exp(-rho * (lam1 - lam_min)) / trace
        else:
            E_{ij} = (exp(-rho * (lam1 - lam_min)) - exp(-rho * (lam2 - lam_min))) / (lam1 - lam2) / trace
    """

    with mp.workdps(80):
        if lam1 == lam2:
            val = -rho * fun(-rho * (lam1 - lam_min)) / trace
        else:
            val = (
                (fun(-rho * (lam1 - lam_min)) - fun(-rho * (lam2 - lam_min)))
                / (mp.mpf(lam1) - mp.mpf(lam2))
                / mp.mpf(trace)
            )
    return np.float64(val)


a = 10.0
b = 10.0 + 1e-10
d = b - a
Eij = np.exp(a) * (np.expm1(d) / d)
Eij2 = (np.exp(a) - np.exp(b)) / (a - b)
Eij4 = Eij_a(np.exp, -1.0, 1.0, 0.0, a, b)
exact = np.exp(b)
print("Eij:  ", Eij, " err: ", (Eij - exact) / exact)
print("Eij2: ", Eij2, " err: ", (Eij2 - exact) / exact)
print("Eij4: ", Eij4, " err: ", (Eij4 - exact) / exact)

ic(mp.tanh(d))
Eij = mp.tanh(a) * (mp.tanh(d) / d)
Eij2 = (mp.tanh(a) - mp.tanh(b)) / (a - b)
Eij4 = Eij_a(mp.tanh, -1.0, 1.0, 0.0, a, b)
exact = mp.tanh(b)
ic(mp.tanh(a))
ic(mp.tanh(b))
print("Eij:  ", Eij, " err: ", (Eij - exact) / exact)
print("Eij2: ", Eij2, " err: ", (Eij2 - exact) / exact)
print("Eij4: ", Eij4, " err: ", (Eij4 - exact) / exact)


def plot_3():
    # load only first 3 columns "other/building.txt"
    data = np.loadtxt("other/building.txt")
    x = data[:, 3]  # displacement
    y = 1 / data[:, 1]  # 1 / BLF
    z = data[:, 2]  # compliance

    def polygon_under_graph(xx, yy):
        """
        Construct the vertex list which defines the polygon filling the space under
        the (x, y) line graph. This assumes x is in ascending order.
        """
        return [
            (np.min(xx) - 0.0001 * np.min(xx), np.max(z)),
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
    for d in range(0, 21):
        data_new[:, 1:3] = polygon_under_graph(
            data_xyz[data_xyz[:, 0] == d][:, 1], data_xyz[data_xyz[:, 0] == d][:, 2]
        )
        data_new[:, 0] = d
        is_efficient = is_pareto_efficient_dumb(data_new)
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
    facecolors = plt.colormaps["coolwarm"](np.linspace(0, 1, 21))

    def func(x, a, b, c):
        return a * np.exp(-b * x**3) + c

    for d in range(0, 21):
        data_x = np.empty((0, 2), float)

        for i in range(0, len(x)):
            if data[i, 3] == d:
                data_x = np.append(data_x, np.array([[y[i], z[i]]]), axis=0)

        ax.scatter(
            d,
            data_x[:, 0],
            data_x[:, 1],
            c=facecolors[d],
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
            facecolors=facecolors[d],
            alpha=0.7,
            zorder=d + 1,
            # orientation="horizontal",
        )
        ax.add_collection3d(poly, zs=d, zdir="x")

    ax.set(
        xlim=(0, 20),
        ylim=(np.min(y), np.max(y)),
        zlim=(1.05 * np.min(z), 1.02 * np.max(z)),
        xlabel=r"$h(d^2)$",
        ylabel=r"$1 / BLF(\lambda_1)$",
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
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))

    # set x axis ticks each tick has different color
    # add black color for first facecolor
    facecolors = np.insert(facecolors, 0, [facecolors[0]], axis=0)
    facecolors = np.insert(facecolors, -1, [facecolors[-1]], axis=0)
    for i, t in enumerate(ax.xaxis.get_ticklabels()):
        t.set_color(facecolors[i])

    # turn off the y and z axis ticks
    ax.yaxis.set_ticks([])
    ax.zaxis.set_ticks([])

    # swich the y and z axis
    ax.view_init(-140, 30)

    # scale the x axis to make it longer
    ax.set_box_aspect((4, 1, 1))
    plt.savefig(
        "output/final_results/building/displacement/frequency_compliance/building_pf.png",
        bbox_inches="tight",
        dpi=1000,
        pad_inches=0.5,
    )

    ##########################
    ax = plt.figure().add_subplot(projection="3d")
    ax.figure.set_size_inches(1.8 * 7.48, 1.8 * 4)

    # plot the pareto front surface
    ax.plot_trisurf(
        data_xyz_paret[:, 0],
        data_xyz_paret[:, 1],
        data_xyz_paret[:, 2],
        color="k",
        alpha=0.2,
    )

    for d in range(0, 21):
        data_x = np.empty((0, 2), float)

        for i in range(0, len(x)):
            if data[i, 3] == d:
                data_x = np.append(data_x, np.array([[y[i], z[i]]]), axis=0)

        ax.scatter(
            d,
            data_x[:, 0],
            data_x[:, 1],
            c=facecolors[d],
            s=10.0,
            zorder=d + 1,
            alpha=1.0,
        )

    ax.set(
        xlim=(0, 20),
        ylim=(np.min(y), np.max(y)),
        zlim=(1.05 * np.min(z), 1.02 * np.max(z)),
        xlabel=r"$h(d^2)$",
        ylabel=r"$1 / BLF(\lambda_1)$",
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
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))

    # set x axis ticks each tick has different color
    # add black color for first facecolor
    facecolors = np.insert(facecolors, 0, [facecolors[0]], axis=0)
    facecolors = np.insert(facecolors, -1, [facecolors[-1]], axis=0)

    for i, t in enumerate(ax.xaxis.get_ticklabels()):
        t.set_color(facecolors[i])

    # turn off the y and z axis ticks
    ax.yaxis.set_ticks([])
    ax.zaxis.set_ticks([])

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
    for d in range(0, 21):
        data_x = np.empty((0, 2), float)

        for i in range(0, len(x)):
            if data[i, 3] == d:
                data_x = np.append(data_x, np.array([[y[i], z[i]]]), axis=0)

        ax.scatter(
            data_x[:, 0],
            data_x[:, 1],
            c=facecolors[d],
            s=10.0,
            zorder=d + 1,
            alpha=1.0,
        )

    ax.set(
        xlabel=r"$1 / BLF(\lambda_1)$",
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

    fig, ax = plt.subplots(figsize=(2, 2))
    for d in range(0, 21):
        data_x = np.empty((0, 2), float)

        for i in range(0, len(x)):
            if data[i, 3] == d:
                data_x = np.append(data_x, np.array([[y[i], z[i]]]), axis=0)

        ax.scatter(
            data_x[:, 0],
            data_x[:, 1],
            c=facecolors[d],
            s=10.0,
            zorder=d + 1,
            alpha=1.0,
        )

    ax.set(
        xlim=(0.336, 0.345),
        ylim=(0.76e-5, 0.81e-5),
        # xlabel=r"$1 / BLF(\lambda_1)$",
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

    plt.show()
