import matplotlib.pylab as plt
import numpy as np
import scipy.interpolate as interpolate


def data():
    frequency = np.array([3, 4, 5, 6, 7])
    stress = np.array(
        [800, 1000, 1200, 1400, 1600, 1800, 2000, 2200, 2400, 2600, 2800, 3000]
    )
    stress_active_line = np.array(
        [1657.298103, 1420.1214015, 1365.7495705, 1779.4377987, 2526.4680177]
    )

    frequency_valid_line = np.array(
        [
            5.39360e00,
            5.72752e00,
            6.05739e00,
            6.30753e00,
            6.47921e00,
            6.61059e00,
            6.75521e00,
            6.89205e00,
            7.00000e00,
            6.99990e00,
            7.00000e00,
            7.00000e00,
        ]
    )

    v_3 = np.array(
        [
            6.56558e-01,
            3.13019e-01,
            1.35775e-01,
            1.35775e-01,
            1.32431e-01,
            1.32432e-01,
            1.5581e-01,
            1.32493e-01,
            1.32580e-01,
            1.32453e-01,
            1.32631e-01,
            1.33013e-01,
        ]
    )
    v_4 = np.array(
        [
            6.53452e-01,
            5.62947e-01,
            3.14743e-01,
            2.14576e-01,
            2.60982e-01,
            2.02722e-01,
            2.02714e-01,
            2.02704e-01,
            2.04926e-01,
            2.03412e-01,
            2.04138e-01,
            2.08660e-01,
        ]
    )
    v_5 = np.array(
        [
            7.39923e-01,
            4.16921e-01,
            4.36512e-01,
            4.32469e-01,
            3.11288e-01,
            3.26320e-01,
            3.72944e-01,
            3.39365e-01,
            3.19470e-01,
            3.41648e-01,
            3.31750e-01,
            3.27857e-01,
        ]
    )
    v_6 = np.array(
        [
            7.87004e-01,
            8.23939e-01,
            8.54727e-01,
            8.58191e-01,
            8.52708e-01,
            5.11606e-01,
            4.68664e-01,
            4.67693e-01,
            4.90551e-01,
            5.47671e-01,
            4.62172e-01,
            4.63211e-01,
        ]
    )
    v_7 = np.array(
        [
            7.85026E-01,
            8.18957e-01,
            8.52604e-01,
            8.75167e-01,
            8.90267e-01,
            9.00738e-01,
            9.12250e-01,
            9.22460e-01,
            9.36927e-01,
            9.34383e-01,
            9.31121e-01,
            9.30314e-01,
        ]
    )

    V = np.array([v_3, v_4, v_5, v_6, v_7]).T

    return frequency, stress, stress_active_line, frequency_valid_line, V


def colorbar(mappable):
    """
    Helper function: create the colorbar
    usage:
    cont = ax.contour(...)
    colorbar(cont)
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    last_axes = plt.gca()
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(mappable, cax=cax)
    plt.sca(last_axes)
    return cbar


def plot_stress_active(frequency, stress_active_line):
    f = interpolate.interp1d(frequency, stress_active_line, kind="cubic")
    frequency_new = np.linspace(3, 6, 100)
    stress_active_line_new = f(frequency_new)
    plt.plot(frequency_new, stress_active_line_new, "b", label="stress active")


def plot_frequency_valid(stress, frequency_valid_line):
    f = interpolate.interp1d(stress, frequency_valid_line, kind="cubic")
    stress_new = np.linspace(800, 3000, 100)
    frequency_valid_line_new = f(stress_new)
    plt.plot(frequency_valid_line_new, stress_new, "r", label="frequency valid")


def plot_contour(frequency, stress, Z):
    X, Y = np.meshgrid(frequency, stress)
    cont = plt.contourf(
        X, Y, Z, levels=50, alpha=0.75, cmap=plt.cm.coolwarm, antialiased=True
    )
    colorbar(cont)


if __name__ == "__main__":
    frequency, stress, stress_active_line, frequency_valid_line, V = data()

    fig, ax = plt.subplots()
    plot_stress_active(frequency, stress_active_line)
    plot_frequency_valid(stress, frequency_valid_line)
    plot_contour(frequency, stress, V)
    plt.xlabel("Frequency")
    plt.ylabel("Stress")
    plt.legend()
    plt.savefig("result/topo_plot.png")
    plt.show()
