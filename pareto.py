import argparse
from glob import glob
import pandas as pd
from os.path import join, splitext, basename
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from mma4py import mma4py_plot_log
import re
import numpy as np
import PIL
from PIL import Image


def remove_duplicated_heders(df):
    df = df[df.ne(df.columns).any(axis="columns")]
    df.reset_index(drop=True, inplace=True)
    df = df.astype(float)
    return df


def get_pareto_front(case_folders, xmetric, ymetric, refresh_history=False):
    case_folders.sort()

    x_vals = []
    y_vals = []
    topo_paths = []
    hist_paths = []

    for case in case_folders:
        # Find the last design png
        r = re.compile(r".*/\d+\.png")
        topo_pngs = list(filter(r.search, glob(join(case, "*.png"))))
        get_iter = lambda path: int(splitext(basename(path))[0])
        topo_pngs.sort(key=get_iter)
        topo_paths.append(topo_pngs[-1])

        # Generate mma4py history plots if needed
        mma4py_log = join(case, "mma4py.log")
        mma4py_png = join(case, "mma4py.png")
        if not glob(mma4py_png) or refresh_history:
            print("[Info] generating %s" % mma4py_png)
            fig, ax = plt.subplots(figsize=(4.8, 4.8))
            ax2 = mma4py_plot_log(ax, mma4py_log)
            ax2.set_ylim(top=1e3, bottom=1e-3)
            fig.savefig(mma4py_png, dpi=500)
            plt.close()
        hist_paths.append(mma4py_png)

        # Load stdout (remove first 7 rows)
        stdout_log = pd.read_fwf(join(case, "stdout.log"), skiprows=[*range(7)])
        stdout_log = remove_duplicated_heders(stdout_log)

        # Get x-axis metric and y-axis metric
        x_vals.append(stdout_log[xmetric].iloc[-1])
        y_vals.append(stdout_log[ymetric].iloc[-1])

    return x_vals, y_vals, topo_paths, hist_paths


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
    p.add_argument("--refresh-history", action="store_true")
    p.add_argument(
        "--metrics",
        default=["stress_ks", "omega_ks"],
        nargs=2,
        choices=["stress_ks", "omega_ks", "area"],
    )
    args = p.parse_args()

    x_vals, y_vals, topo_paths, hist_paths = get_pareto_front(
        args.case_folders, *args.metrics, refresh_history=args.refresh_history
    )

    # Draw pareto front with designs
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
            topo_paths[i], 0.07, ax, xy, xybox_topo, {"arrowstyle": "->", "alpha": 0.1}
        )

    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlabel(args.metrics[0])
    ax.set_ylabel(args.metrics[1])

    fig.savefig("pareto.pdf")

    # Concatenate design and history images
    topo_imgs = [Image.open(p) for p in topo_paths]
    hist_imgs = [Image.open(p) for p in hist_paths]

    # pick the image which is the smallest, and resize the others to match it
    min_shape = sorted([(np.sum(i.size), i.size) for i in topo_imgs])[0][1]
    arr_topo = np.hstack([i.resize(min_shape) for i in topo_imgs])
    arr_hist = np.hstack([i.resize(min_shape) for i in hist_imgs])

    # Concatenate and save
    image_concat = Image.fromarray(np.array([*arr_topo, *arr_hist]))
    image_concat.save("designs_historys.png")
