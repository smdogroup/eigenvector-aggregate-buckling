import argparse
from glob import glob
import pandas as pd
from os.path import join
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from mma4py import mma4py_plot_log


def remove_duplicated_heders(df):
    df = df[df.ne(df.columns).any(axis="columns")]
    df.reset_index(drop=True, inplace=True)
    df = df.astype(float)
    return df


def get_pareto_front(case_folders, final_iter, xmetric, ymetric):
    case_folders.sort()

    x_vals = []
    y_vals = []
    designs = []

    for case in case_folders:
        # Check if this case is done
        final_png = glob(join(case, "%s.png" % final_iter))
        if not final_png:
            print("[Warning] ./%s doesn't have %s.png" % (case, final_iter))
            continue

        designs.append(final_png[0])

        # Load stdout (remove first 7 rows)
        stdout_log = pd.read_fwf(join(case, "stdout.log"), skiprows=[*range(7)])
        stdout_log = remove_duplicated_heders(stdout_log)

        # Get x-axis metric and y-axis metric
        x_vals.append(stdout_log[xmetric].iloc[final_iter])
        y_vals.append(stdout_log[ymetric].iloc[final_iter])
    return x_vals, y_vals, designs


def insert_image(imgpath, zoom, ax, xy, xybox, arrowprops={}):
    imgbox = OffsetImage(plt.imread(imgpath), zoom=zoom)
    imgbox.image.axes = ax
    ab_topo = AnnotationBbox(
        imgbox,
        xy,
        xybox,
        arrowprops=arrowprops,
        frameon=False,
        pad=0.0,
    )
    ax.add_artist(ab_topo)
    return


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("case_folders", nargs="*", type=str)
    p.add_argument("--final-iter", default=500, type=int)
    p.add_argument(
        "--metrics",
        default=["stress_ks", "omega_ks"],
        nargs=2,
        choices=["stress_ks", "omega_ks", "area"],
    )
    args = p.parse_args()

    x_vals, y_vals, designs = get_pareto_front(
        args.case_folders, args.final_iter, *args.metrics
    )

    N = len(x_vals)
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.9), pad=2.0)
    ax.plot(x_vals, y_vals, marker="o")

    xlb, xub = ax.get_xlim()
    ylb, yub = ax.get_ylim()

    for i in range(N):
        width = xub - xlb
        height = yub - ylb

        xy = x_vals[i], y_vals[i]
        xybox_topo = (xlb + width * (i + 0.5) / N, yub + 0.1 * height)

        insert_image(
            designs[i], 0.07, ax, xy, xybox_topo, {"arrowstyle": "->", "alpha": 0.1}
        )

    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlabel(args.metrics[0])
    ax.set_ylabel(args.metrics[1])

    plt.show()
    fig.savefig("pareto.pdf")
