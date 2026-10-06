#!/usr/bin/env python3
"""
AAI heatmap (all vs. all) based on the output table from CompareM.

Examples:
    python aai_heatmap.py -i aai_summary.tsv
    python aai_heatmap.py -i aai_summary.tsv -n names.tsv -o results/aai_10-2026
    python aai_heatmap.py -i aai_summary.tsv -n names.py -c "#f7fbff" "#6baed6" "#08306b"
    python aai_heatmap.py -i aai_summary.tsv -c viridis
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

DEFAULT_COLORS = ["#fff5eb", "#FFB870", "#A30000"]


# --------------------------------------------------------------------------- #
# Arguments
# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser(
        description="Clustered AAI heatmap based on the output from CompareM.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("-i", "--input", required=True,
                   help="CompareM Table (TSV with '#Genome A', 'Genome B', 'Mean AAI').")
    p.add_argument("-n", "--names", default=None,
                   help="(Optional) Dictionary of Names: .json, .py (with variable "
                        "'name_map') or .tsv/.csv with two columns (id, name).")
    p.add_argument("-c", "--colors", nargs="+", default=DEFAULT_COLORS,
                   help="Gradient colors (2 or more, from lowest to highest AAI) "
                        "OR the name of a matplotlib colormap (e.g., OrRd, viridis).")
    p.add_argument("-o", "--output", default=None,
                   help="Output file prefix (can include directory). "
                        "Default: input filename.")
    p.add_argument("--formats", nargs="+", default=["pdf", "svg", "tiff", "png"],
                   help="Output formats.")
    p.add_argument("--min-aai", type=float, default=60,
                   help="Values below this become 0 (and the lower bound of the scale).")
    p.add_argument("--max-aai", type=float, default=100,
                   help="Upper bound of the color scale.")
    p.add_argument("--hide-low", action="store_true",
                   help="Do not draw cells below --min-aai (they remain blank).")
    p.add_argument("--method", default="average",
                   choices=["average", "single", "complete", "weighted", "centroid", "median", "ward"],
                   help="Linkage method for hierarchical clustering (average = UPGMA).")
    p.add_argument("--underscore-to-space", action="store_true",
                   help="Replace '_' with spaces in names not found in the dictionary.")
    p.add_argument("--title", default="AAI(%): All versus All", help="Plot title.")
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


def load_comparem(path):
    """Reads the CompareM table and returns GenomeA, GenomeB, and AAI columns."""
    df = pd.read_csv(path, sep="\t")
    df.columns = df.columns.str.strip()

    wanted = {"#Genome A": "GenomeA", "Genome B": "GenomeB", "Mean AAI": "AAI"}
    if all(c in df.columns for c in wanted):
        df = df[list(wanted)].rename(columns=wanted)
    else:
        print("[WARNING] Header differs from expected; using columns 1, 3, and 6.")
        df = df.iloc[:, [0, 2, 5]]
        df.columns = ["GenomeA", "GenomeB", "AAI"]

    df["GenomeA"] = df["GenomeA"].astype(str).str.strip()
    df["GenomeB"] = df["GenomeB"].astype(str).str.strip()
    df["AAI"] = pd.to_numeric(df["AAI"], errors="coerce")
    return df


# --------------------------------------------------------------------------- #
# Processing
# --------------------------------------------------------------------------- #
def build_matrix(df):
    """Builds the symmetric AAI matrix (diagonal = 100) WITHOUT applying cutoff.
    Missing pairs become NaN."""
    swapped = df.rename(columns={"GenomeA": "GenomeB", "GenomeB": "GenomeA"})
    genomes = pd.unique(pd.concat([df["GenomeA"], df["GenomeB"]]))
    diagonal = pd.DataFrame({"GenomeA": genomes, "GenomeB": genomes, "AAI": 100.0})

    full = pd.concat([df, swapped, diagonal], ignore_index=True)
    mat = full.pivot_table(index="GenomeA", columns="GenomeB",
                           values="AAI", aggfunc="mean")
    return mat.reindex(index=genomes, columns=genomes).astype(float)


def cluster_order(raw, method):
    """Clusters directly using 100 - AAI distance between pairs."""
    if len(raw) < 3:
        return raw.index.tolist()

    dist = 100 - raw
    max_d = np.nanmax(dist.values)
    dist = dist.fillna(max_d if np.isfinite(max_d) else 100)

    arr = dist.to_numpy(copy=True)
    arr = (arr + arr.T) / 2          # ensures symmetry
    np.fill_diagonal(arr, 0)
    arr = np.clip(arr, 0, None)

    if method in ("ward", "centroid", "median"):
        print(f"[WARNING] '{method}' assumes Euclidean distances; "
              "for AAI, 'average' (UPGMA) is recommended.")

    condensed = squareform(arr, checks=False)
    link = hc.linkage(condensed, method=method)
    return hc.dendrogram(link, labels=raw.index.tolist(), no_plot=True)["ivl"]


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
    if args.hide_low:
        mask |= (mat.values == 0)

    annot = args.annot_size > 0
    plt.figure(figsize=tuple(args.figsize))
    ax = sns.heatmap(
        mat,
        cmap=build_cmap(args.colors),
        mask=mask,
        vmin=args.min_aai,
        vmax=args.max_aai,
        annot=annot,
        annot_kws={"fontsize": args.annot_size} if annot else None,
        fmt=".1f",
        square=False,
        cbar_kws={"label": "AAI (%)", "shrink": 0.2},
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
    df = load_comparem(args.input)

    raw = build_matrix(df)
    order = cluster_order(raw, args.method)          # clustering on actual values
    mat = raw.where(raw >= args.min_aai, 0)          # cutoff for display only
    mat = mat.loc[order, order]
    mat = rename_genomes(mat, name_map, args.underscore_to_space)

    plot_heatmap(mat, args)


if __name__ == "__main__":
    main()
