#!/usr/bin/env python3
"""
Cleans genome names from a square matrix (rows and columns).

  Spathaspora_girioi_ASM165745v1_genomic  -> Spathaspora_girioi
  Spathaspora_sp._JA1_ASM367603v1_genomic -> Spathaspora_sp_JA1
  y7005_B_scaffolds_500bp                  -> UFMG-CM-Y7005

Usage:
    python clean_names.py -i pocp_7602_tirada.csv
    python clean_names.py -i pocp_7602_tirada.csv -o pocp_clean.tsv
"""

import argparse
import os
import re
import sys

import pandas as pd


def clean_name(name):
    name = str(name).strip()

    # yXXXX... -> UFMG-CM-YXXXX
    m = re.fullmatch(r"[yY](\d+)(?:_.*)?", name)
    if m:
        return f"UFMG-CM-Y{m.group(1)}"

    parts = name.split("_")
    if len(parts) < 2:
        return name

    genus, species = parts[0], parts[1]

    # Genus_sp._STRAIN -> Genus_sp_STRAIN (keeps strain to preserve identity)
    if species.rstrip(".") == "sp" and len(parts) > 2:
        return f"{genus}_sp_{parts[2]}"

    return f"{genus}_{species}"


def main():
    p = argparse.ArgumentParser(description="Cleans genome names from a square matrix.")
    p.add_argument("-i", "--input", required=True, help="Original matrix.")
    p.add_argument("-o", "--output", default=None,
                   help="Output file. Default: <input>_clean.tsv")
    p.add_argument("--sep", default="\t", help="Matrix delimiter.")
    args = p.parse_args()

    data = pd.read_csv(args.input, sep=args.sep, index_col=0)
    index_name = data.index.name

    mapping = {n: clean_name(n) for n in list(data.index) + list(data.columns)}

    new_names = [mapping[n] for n in data.index]
    dup = sorted({n for n in new_names if new_names.count(n) > 1})
    if dup:
        sys.exit(f"[ERROR] Duplicate names found after cleaning: {', '.join(dup)}")

    data = data.rename(index=mapping, columns=mapping)
    data.index.name = index_name

    out = args.output or f"{os.path.splitext(args.input)[0]}_clean.tsv"
    data.to_csv(out, sep=args.sep)

    for old, new in mapping.items():
        if old != new:
            print(f"{old:60s} -> {new}")
    print(f"\n[OK] {out}")


if __name__ == "__main__":
    main()
