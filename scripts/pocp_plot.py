import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import scipy.cluster.hierarchy as hc


# names
name_map = {
    "GCA_000755205.1_ASM75520v1_genomic": "Meyerozyma caribbica",
    "GCA_002245345.1_ASM224534v1_genomic": "Scheffersomyces stambukii",
    "GCF_036850825.1_Schama1_genomic": "Scheffersomyces amazonensis",
    "GCA_030578755.1_ASM3057875v1_genomic": "Scheffersomyces coipomensis",
    "GCA_030555905.1_ASM3055590v1_genomic": "Scheffersomyces cryptocercus",
    "GCA_030575095.1_ASM3057509v1_genomic": "Scheffersomyces ergatensis",
    "GCA_030572695.1_ASM3057269v1_genomic": "Scheffersomyces illinoinensis",
    "GCA_030572575.1_ASM3057257v1_genomic": "Scheffersomyces insectosa",
    "GCA_001599395.1_JCM_9837_assembly_v001_genomic": "Scheffersomyces lignosus",
    "GCA_030571865.1_ASM3057186v1_genomic": "Scheffersomyces parashehatae",
    "GCA_030556775.1_ASM3055677v1_genomic": "Scheffersomyces quercinus",
    "GCA_018489445.1_ASM1848944v1_genomic": "Scheffersomyces segobiensis",
    "GCA_030674605.1_ASM3067460v1_genomic": "Scheffersomyces shehatae",
    "GCF_019049425.1_ASM1904942v1_genomic": "Scheffersomyces spartinae",
    "GCF_000209165.1_ASM20916v1_genomic": "Scheffersomyces stipitis",
    "GCA_030565185.1_ASM3056518v1_genomic": "Scheffersomyces titani",
    "GCA_030565005.1_ASM3056500v1_genomic": "Scheffersomyces virginianus",
    "GCF_036884685.1_Schxyl1_genomic": "Scheffersomyces xylosifermentans",

    "GCA_000497715.1_SpaArb1.0_genomic": "Spathaspora arborariae",
    "GCA_002094185.1_ASM209418v1_genomic": "Spathaspora boniae",
    "GCA_001657455.1_ASM165745v1_genomic": "Spathaspora girioi",
    "GCA_001655755.1_ASM165575v1_genomic": "Spathaspora hagerdaliae",
    "GCF_000223485.1_Spathaspora_passalidarum_v2.0_genomic": "Spathaspora passalidarum v2.0",
    "GCA_002911495.2_ASM291149v2_genomic": "Spathaspora marinasilvae",
    "GCA_003676035.1_ASM367603v1_genomic": "Spathaspora sp. JA1",
    "GCA_002105455.1_ASM210545v1_genomic": "Spathaspora xylofermentans",
    "GCA_001655765.1_ASM165576v1_genomic": "Spathaspora gorwiae",
    "y6407_500bp": "Spathaspora brunopereirae sp.",
    "y2822_500bp": "Spathaspora domphillipsii sp.",
    "y7005_500bp": "UFMG-CM-Y7005",

    "GCA_030582855.1_ASM3058285v1_genomic": "Candida lyxosophila",
    "GCA_030572495.1_ASM3057249v1_genomic": "Candida sake",
    "GCF_019202705.1_ASM1920270v1_genomic": "Candida subhashii",
    "GCA_030585045.1_ASM3058504v1_genomic": "Candida parablackwelliae",
    "GCA_030579015.1_ASM3057901v1_genomic": "Candida blackwelliae",
    "GCA_030557085.1_ASM3055708v1_genomic": "Candida gigantensis",
    "GCA_030582575.1_ASM3058257v1_genomic": "Candida buenavistaensis",
    "GCF_000026945.1_ASM2694v1_genomic": "Candida dubliniensis",
    "GCF_000182965.3_ASM18296v3_genomic": "Candida albicans",
    "GCA_030563665.1_ASM3056366v1_genomic": "Candida broadrunensis",
    "GCA_030572135.1_ASM3057213v1_genomic": "[Candida] alai",
    "GCA_030557875.1_ASM3055787v1_genomic": "Candida neerlandica",
    "GCA_030572435.1_ASM3057243v1_genomic": "Candida labiduridarum",
    "GCA_030557105.1_ASM3055710v1_genomic": "Candida frijolesensis",
    "GCA_030557055.1_ASM3055705v1_genomic": "Candida tetrigidarum",
    "GCA_030566835.1_ASM3056683v1_genomic": "Candida viswanathii",
    "GCA_030582595.1_ASM3058259v1_genomic": "Candida tropicalis",
    "GCA_911254575.1_Wolfe_Cansan_genomic": "Candida sanyaensis",
    "GCA_030582655.1_ASM3058265v1_genomic": "Candida sojae",
    "GCA_030578955.1_ASM3057895v1_genomic": "Candida insectamans",
    "GCA_050495715.1_ASM5049571v1_genomic": "Candida maltosa",
}




# ---- Load POCP data ----
data = pd.read_csv("../Files/pocp_7602_tirada.csv", sep="\t", index_col=0)

# Ensure numeric
data = data.apply(pd.to_numeric, errors="coerce")
if data.isnull().values.any():
    raise ValueError("Unable to convert all values to numeric!")

# ---- Perform hierarchical clustering ----
linkage = hc.linkage(data, method="average")  # UPGMA-like
dendro = hc.dendrogram(linkage, labels=data.index, no_plot=True)

# reorder rows/cols according to clustering
ordered = dendro["ivl"]
pocp_matrix = data.loc[ordered, ordered]

# ---- Rename genomes to species ----
pocp_matrix = pocp_matrix.rename(index=name_map, columns=name_map)

# ---- Mask upper triangle ----
mask = np.triu(np.ones_like(pocp_matrix, dtype=bool), k=1)

# ---- Custom colormap ----
cmap = mcolors.LinearSegmentedColormap.from_list(
    "blue_red", ["#fff5eb","#fff5eb", "#fff5eb", "#A30000"]
)

# ---- Plot ----
plt.figure(figsize=(20, 15))
sns.heatmap(
    pocp_matrix,
    cmap=cmap,
    mask=mask,
    vmin=0,
    vmax=100,
    annot=True,
    annot_kws={"fontsize": 5},
    fmt=".1f",
    square=False,
    cbar_kws={"label": "POCP (%)", "shrink": 0.2},
)

plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.title("POCP(%): All versus All")
plt.tight_layout()

# Colorbar formatting
cbar = plt.gcf().axes[-1]
cbar.tick_params(labelsize=8)

# Save figure
plt.savefig("pocp_tirada.pdf", dpi=300, bbox_inches="tight")
plt.savefig("pocp_tirada.png", dpi=300, bbox_inches="tight")
plt.savefig("pocp_tirada.svg", dpi=300, bbox_inches="tight")
plt.close()