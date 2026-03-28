from __future__ import annotations

import matplotlib
matplotlib.use("pgf")

from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

plt.rcParams.update({
    "pgf.texsystem": "pdflatex",
    "text.usetex": True,
    "pgf.rcfonts": False,
    "font.family": "serif",
    "hatch.linewidth": 0.6,
})

dfs: list[pd.DataFrame] = []
for path in Path("./repaired").glob("direct-conv-cm-*"):
    dfs.append(pd.read_parquet(path))

full = pd.concat(dfs)
reduced = pd.read_parquet("parquets/direct-conv-cm-milp.parquet")

full = full.loc[full["device"] == "NVIDIA GeForce RTX 4070"]
reduced = reduced.loc[reduced["device"] == "NVIDIA GeForce RTX 4070"]

op = "relu(conv2d(x,kernel_size=(3,3),bias=true,stride=(1,1),padding=(1,1),dialation=(1,1)))"
full = full.loc[full["operation"] == op]
reduced = reduced.loc[reduced["operation"] == op]

input_shape = "544x960x64"
full = full.loc[full["input_shape"] == input_shape]
reduced = reduced.loc[reduced["input_shape"] == input_shape]

BINS = 100
bin_edges = np.linspace(0.0, 1.0, BINS + 1)

full_counts, edges = np.histogram(full["relative_speedup"], bins=bin_edges)
reduced_counts, _ = np.histogram(reduced["relative_speedup"], bins=bin_edges)

fig, ax = plt.subplots(figsize=(5, 2))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

full_fill = "#C2CFDD"
full_hatch = "#9FB2C5"   # lighter hatch tint
full_outline = "#72879D" # darker outline
full_hatch = full_outline

reduced_fill = "#E0B2AA"
reduced_hatch = "#B9857D"
reduced_outline = "#8F544D"
reduced_hatch = reduced_outline

hatch_density = 3

full_patch = ax.stairs(
    full_counts,
    edges,
    baseline=0,
    fill=True,
    facecolor=full_fill,
    edgecolor=full_hatch,   # hatch color comes from edgecolor
    linewidth=0.0,          # suppress visible border
    label="Full configuration space",
    zorder=1,
)
full_patch.set_hatch("\\" * hatch_density)

ax.stairs(
    full_counts,
    edges,
    baseline=0,
    fill=False,
    edgecolor=full_outline, # separate darker outline
    linewidth=0.9,
    zorder=2,
)

reduced_patch = ax.stairs(
    reduced_counts,
    edges,
    baseline=0,
    fill=True,
    facecolor=reduced_fill,
    edgecolor=reduced_hatch,
    linewidth=0.0,
    label="Reduced configuration space",
    zorder=3,
)
reduced_patch.set_hatch("/" * hatch_density)

ax.stairs(
    reduced_counts,
    edges,
    baseline=0,
    fill=False,
    edgecolor=reduced_outline,
    linewidth=0.9,
    zorder=4,
)

ax.set_xlim(0, 1)
ax.set_xlabel("Relative speedup")
ax.set_ylabel("Number of configurations")

ax.grid(axis="y", which="major", color="0.88", linewidth=0.8)
ax.grid(axis="x", visible=False)
ax.set_axisbelow(True)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_color("0.25")
ax.spines["bottom"].set_color("0.25")
ax.spines["left"].set_linewidth(0.8)
ax.spines["bottom"].set_linewidth(0.8)

ax.tick_params(axis="both", colors="0.2", width=0.8, length=3)
ax.legend(frameon=False, loc="upper left")

fig.tight_layout()
fig.savefig("/home/kistenklaus/Documents/hpg-paper/plots/search-space.pgf")
plt.close(fig)
