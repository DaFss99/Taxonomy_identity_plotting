#!/usr/bin/env python3
"""
POCP heatmap (all vs. all) based on a square matrix
(first column = genome names, header = same genomes).

Examples:
    python pocp_heatmap.py -i pocp-matrix.tsv
    python pocp_heatmap.py -i pocp.tsv -n names.py -o results/pocp_10-2026
    python pocp_heatmap.py -i pocp.tsv -c "#f7fbff" "#6baed6" "#08306b"
    python pocp_heatmap.py -i pocp.tsv -c viridis
"""

import argparse
import json
import os
import runpy
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import scipy.cluster.hierarchy as hc
from scipy.spatial.distance import squareform

DEFAULT_COLORS = ["#fff5eb", "#fff5eb", "#fff5eb", "#FFB870", "#A30000"]


# --------------------------------------------------------------------------- #
# Arguments
# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser(
        description="Clustered POCP heatmap based on a square matrix.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("-i", "--input", required=True,
                   help="POCP matrix (first column and header = genomes).")
    p.add_argument("--sep", default="\t",
                   help="Matrix delimiter (use ',' for comma-separated CSV).")
    p.add_argument("-n", "--names", default=None,
                   help="(Optional) Dictionary of Names: .json, .py (with variable "
                        "'name_map') or .tsv/.csv with two columns (id, name).")
    p.add_argument("-c", "--colors", nargs="+", default=DEFAULT_COLORS,
                   help="Gradient colors (2 or more, from lowest to highest POCP) "
                        "OR the name of a matplotlib colormap (e.g., OrRd, viridis).")
    p.add_argument("-o", "--output", default=None,
                   help="Output file prefix (can include directory). "
                        "Default: input filename.")
    p.add_argument("--formats", nargs="+", default=["pdf", "png", "svg"],
                   help="Output formats.")
    p.add_argument("--vmin", type=float, default=0, help="Lower bound of the color scale.")
    p.add_argument("--vmax", type=float, default=100, help="Upper bound of the color scale.")
    p.add_argument("--method", default="average",
                   choices=["average", "single", "complete", "weighted", "centroid", "median", "ward"],
                   help="Linkage method for hierarchical clustering (average = UPGMA).")
    p.add_argument("--underscore-to-space", action="store_true",
                   help="Replace '_' with spaces in names not found in the dictionary.")
    p.add_argument("--title", default="POCP(%): All versus All", help="Plot title.")
    p.add_argument("--figsize", nargs=2, type=float, default=[20, 15],
                   metavar=("WIDTH", "HEIGHT"), help="Figure size in inches.")
    p.add_argument("--annot-size", type=float, default=5,
                   help="Font size of values inside cells (0 = no values).")
    p.add_argument("--dpi", type=int, default=300, help="Image resolution.")
    return p.parse_args()


# --------------------------------------------------------------------------- #
# Input / Loading
# --------------------------------------------------------------------------- #
def load_name_map(path):
    """Reads the name dictionary from a .json, .py, or two-column table file."""
    if path is None:
        return {}
    ext = os.path.splitext(path)[1].lower()

    if ext == ".json":
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    elif ext == ".py":
        data = runpy.run_path(path).get("name_map")
        if not isinstance(data, dict):
            sys.exit(f"[ERROR] {path} must define a dictionary named 'name_map'.")
    else:
        sep = "," if ext == ".csv" else "\t"
        df = pd.read_csv(path, sep=sep, header=None, comment="#", dtype=str)
        if df.shape[1] < 2:
            sys.exit(f"[ERROR] {path} must have two columns: id and name.")
        data = dict(zip(df.iloc[:, 0], df.iloc[:, 1]))

    return {str(k).strip(): str(v).strip() for k, v in data.items()}


def load_matrix(path, sep):
    """Reads the square matrix, verifies row/column compatibility, and checks for numeric values."""
    data = pd.read_csv(path, sep=sep, index_col=0)
    data.index = data.index.astype(str).str.strip()
    data.columns = data.columns.astype(str).str.strip()

    rows, cols = set(data.index), set(data.columns)
    if rows != cols:
        only_rows = sorted(rows - cols)
        only_cols = sorted(cols - rows)
        msg = "[ERROR] Matrix rows and columns do not contain the same genomes."
        if only_rows:
            msg += f"\n  Only in rows: {', '.join(only_rows)}"
        if only_cols:
            msg += f"\n  Only in columns: {', '.join(only_cols)}"
        sys.exit(msg)

    data = data.loc[:, data.index]  # same order for rows and columns
    data = data.apply(pd.to_numeric, errors="coerce")
    if data.isnull().values.any():
        bad = data.isnull().stack()
        bad = bad[bad].index.tolist()[:5]
        sys.exit(f"[ERROR] Non-numeric values found in matrix (e.g., {bad}).")
    return data.astype(float)


# --------------------------------------------------------------------------- #
# Processing
# --------------------------------------------------------------------------- #
def cluster_order(mat, method):
    """Clusters directly using 100 - POCP distance between pairs."""
    if len(mat) < 3:
        return mat.index.tolist()

    arr = 100 - mat.to_numpy(copy=True)
    arr = (arr + arr.T) / 2          # ensures symmetry
    np.fill_diagonal(arr, 0)
    arr = np.clip(arr, 0, None)

    if method in ("ward", "centroid", "median"):
        print(f"[WARNING] '{method}' assumes Euclidean distances; "
              "for POCP, 'average' (UPGMA) is recommended.")

    link = hc.linkage(squareform(arr, checks=False), method=method)
    return hc.dendrogram(link, labels=mat.index.tolist(), no_plot=True)["ivl"]


def rename_genomes(mat, name_map, underscore_to_space):
    if name_map:
        missing = [g for g in mat.index if g not in name_map]
        if missing:
            print(f"[WARNING] {len(missing)} genome(s) missing from dictionary:")
            for g in missing:
                print(f"    - {g}")

    def new_name(g):
        if g in name_map:
            return name_map[g]
        return g.replace("_", " ") if underscore_to_space else g

    labels = [new_name(g) for g in mat.index]
    mat = mat.copy()
    mat.index = labels
    mat.columns = labels
    return mat


def build_cmap(colors):
    if len(colors) == 1:
        try:
            return matplotlib.colormaps[colors[0]]
        except KeyError:
            sys.exit(f"[ERROR] '{colors[0]}' is not a valid colormap. "
                     "Pass 2+ colors or a matplotlib colormap name.")
    invalid = [c for c in colors if not mcolors.is_color_like(c)]
    if invalid:
        sys.exit(f"[ERROR] Invalid colors: {', '.join(invalid)}")
    return mcolors.LinearSegmentedColormap.from_list("custom", colors)


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #
def plot_heatmap(mat, args):
    mask = np.triu(np.ones_like(mat, dtype=bool), k=1)

    annot = args.annot_size > 0
    plt.figure(figsize=tuple(args.figsize))
    ax = sns.heatmap(
        mat,
        cmap=build_cmap(args.colors),
        mask=mask,
        vmin=args.vmin,
        vmax=args.vmax,
        annot=annot,
        annot_kws={"fontsize": args.annot_size} if annot else None,
        fmt=".1f",
        square=False,
        cbar_kws={"label": "POCP (%)", "shrink": 0.2},
    )

    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.title(args.title)
    ax.collections[0].colorbar.ax.tick_params(labelsize=8)
    plt.tight_layout()

    prefix = args.output or os.path.splitext(os.path.basename(args.input))[0]
    outdir = os.path.dirname(prefix)
    if outdir:
        os.makedirs(outdir, exist_ok=True)

    for fmt in args.formats:
        out = f"{prefix}.{fmt.lstrip('.')}"
        plt.savefig(out, dpi=args.dpi, bbox_inches="tight")
        print(f"[OK] {out}")
    plt.close()


# --------------------------------------------------------------------------- #
def main():
    args = parse_args()
    name_map = load_name_map(args.names)
    mat = load_matrix(args.input, args.sep)

    order = cluster_order(mat, args.method)
    mat = mat.loc[order, order]
    mat = rename_genomes(mat, name_map, args.underscore_to_space)

    plot_heatmap(mat, args)


if __name__ == "__main__":
    main()
