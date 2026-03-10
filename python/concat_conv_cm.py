from uuid import MAX
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt


dfs: list[pd.DataFrame] = []
for path in list(Path("./parquets/").glob("concat-conv-cm-*")):
    dfs.append(pd.read_parquet(path))

df = pd.concat(dfs, ignore_index=True)

print("Operations: (Pick One)")
for op in df["operation"].unique():
    print(f'"{op}"')

R = 3
S = 3
OP = "conv2d(concat([x,y],0),kernel_size=(3,3),bias=true,stride=(1,1),padding=(1,1),dialation=(1,1))"
df = df.loc[df["operation"] == OP]

print("Devices:")
for device in df["device"].unique():
    print(f'"{device}"')

# DEVICE = "NVIDIA GeForce RTX 4080 SUPER"
# df = df.loc[df["device"] == DEVICE]

print("Input shapes:")
for input_shape in df["input_shape"].unique():
    print(f'"{input_shape}"')

# INPUT_SHAPE = "1088x1920x64#1088x1920x3"
INPUT_SHAPE = "544x960x96#544x960x32"
# INPUT_SHAPE = "272x480x112#272x480x48"
# INPUT_SHAPE = "136x240x96#136x240x64"
# INPUT_SHAPE = "1088x1920x32#1088x1920x3"
# INPUT_SHAPE = "544x960x64#544x960x32"
# INPUT_SHAPE = "272x480x64#272x480x32"
# INPUT_SHAPE = "136x240x32#136x240x32"

df = df.loc[df["input_shape"] == INPUT_SHAPE]

print("Output shapes:")
for output_shape in df["output_shape"].unique():
    print(f'"{output_shape}"')
#
# OUTPUT_SHAPE = "1088x1920x64"
# OUTPUT_SHAPE = "544x960x64"
# OUTPUT_SHAPE = "272x480x96"
# OUTPUT_SHAPE = "136x240x112"
# OUTPUT_SHAPE = "1088x1920x32"
# OUTPUT_SHAPE = "272x480x64"
# OUTPUT_SHAPE = "136x240x64"
# df = df.loc[df["output_shape"] == OUTPUT_SHAPE]


# Juck we don't know it yet. (but could be parsed form the complete database)
df["subgroup_size"] = 32


# horrible register estimates.
# how many u32 registers per coopmat accumulator
df["acc_registers"] = (df["config_CM_M"] * df["config_CM_N"]) / df["subgroup_size"]
# how many u32 registers per coopmat a / b
df["A_a_registers"] = (df["config_CM_M"] * df["config_A_CM_K"]) / df["subgroup_size"]
df["B_a_registers"] = (df["config_CM_M"] * df["config_B_CM_K"]) / df["subgroup_size"]

df["A_b_registers"] = (df["config_A_CM_K"] * df["config_CM_N"]) / df["subgroup_size"]
df["B_b_registers"] = (df["config_B_CM_K"] * df["config_CM_N"]) / df["subgroup_size"]
#
df["coopmat_registers"] = df["acc_registers"] * df["config_SG_N"] * df[
    "config_SG_M"
] + np.minimum(
    df["A_a_registers"] * df["config_SG_M"] + df["A_b_registers"] * df["config_SG_N"],
    df["B_a_registers"] * df["config_SG_M"] + df["B_b_registers"] * df["config_SG_N"],
)
#
# # some shader constants
df["A_prefetch_A_QQ"] = (
    df["config_CM_M"] * df["config_A_CM_K"] * df["config_A_SG_K"] * df["config_SG_M"]
) / 8
df["B_prefetch_A_QQ"] = (
    df["config_CM_M"] * df["config_B_CM_K"] * df["config_B_SG_K"] * df["config_SG_M"]
) / 8

df["A_prefetch_A_SQQ"] = df["A_prefetch_A_QQ"] // df["config_WG_N"]
df["B_prefetch_A_SQQ"] = df["B_prefetch_A_QQ"] // df["config_WG_N"]

df["A_prefetch_A_IQQ"] = (df["A_prefetch_A_SQQ"] + df["subgroup_size"] - 1) // df[
    "subgroup_size"
]
df["B_prefetch_A_IQQ"] = (df["B_prefetch_A_SQQ"] + df["subgroup_size"] - 1) // df[
    "subgroup_size"
]

df["A_prefetch_B_QQ"] = (
    df["config_A_CM_K"] * df["config_CM_N"] * df["config_A_SG_K"] * df["config_SG_N"]
) / 8
df["B_prefetch_B_QQ"] = (
    df["config_B_CM_K"] * df["config_CM_N"] * df["config_B_SG_K"] * df["config_SG_N"]
) / 8

df["A_prefetch_B_SQQ"] = df["A_prefetch_B_QQ"] // df["config_WG_M"]
df["B_prefetch_B_SQQ"] = df["B_prefetch_B_QQ"] // df["config_WG_M"]

df["A_prefetch_B_IQQ"] = (df["A_prefetch_B_SQQ"] + df["subgroup_size"] - 1) // df[
    "subgroup_size"
]
df["B_prefetch_B_IQQ"] = (df["B_prefetch_B_SQQ"] + df["subgroup_size"] - 1) // df[
    "subgroup_size"
]

df["A_prefetch_A_registers"] = df["A_prefetch_A_IQQ"] * 4
df["A_prefetch_B_registers"] = df["A_prefetch_B_IQQ"] * 4

df["B_prefetch_A_registers"] = df["B_prefetch_A_IQQ"] * 4
df["B_prefetch_B_registers"] = df["B_prefetch_B_IQQ"] * 4

df["A_prefetch_registers"] = df["A_prefetch_A_registers"] + df["A_prefetch_B_registers"]
df["B_prefetch_registers"] = df["B_prefetch_A_registers"] + df["B_prefetch_B_registers"]


df["registers"] = df["coopmat_registers"] + np.maximum(
    df["A_prefetch_registers"], df["B_prefetch_registers"]
)

#
# # shared memory size
df["A_sh_a_size"] = (
    df["config_WG_M"]
    * df["config_CM_M"]
    * df["config_A_CM_K"]
    * df["config_A_SG_K"]
    * df["config_SG_M"]
    * 2
)
df["B_sh_a_size"] = (
    df["config_WG_M"]
    * df["config_CM_M"]
    * df["config_B_CM_K"]
    * df["config_B_SG_K"]
    * df["config_SG_M"]
    * 2
)

df["A_sh_b_size"] = (
    df["config_WG_N"]
    * df["config_A_CM_K"]
    * df["config_CM_N"]
    * df["config_A_SG_K"]
    * df["config_SG_N"]
    * 2
)
df["B_sh_b_size"] = (
    df["config_WG_N"]
    * df["config_B_CM_K"]
    * df["config_CM_N"]
    * df["config_B_SG_K"]
    * df["config_SG_N"]
    * 2
)

df["sh_out_size"] = (
    df["config_WG_M"]
    * df["config_WG_N"]
    * df["config_SG_M"]
    * df["config_SG_N"]
    * df["config_CM_M"]
    * df["config_CM_N"]
    * 2
)


df["A_sh_peak"] = np.maximum(df["A_sh_a_size"] + df["A_sh_b_size"], df["sh_out_size"])
df["B_sh_peak"] = np.maximum(df["B_sh_a_size"] + df["B_sh_b_size"], df["sh_out_size"])

df["sh_size"] = np.maximum(df["A_sh_peak"], df["B_sh_peak"])

print(df.columns)
df["A_RSC"] = R * S * df["input_channels0"]
df["B_RSC"] = R * S * df["input_channels1"]

df["A_ktile"] = df["config_A_CM_K"] * df["config_A_SG_K"]
df["B_ktile"] = df["config_B_CM_K"] * df["config_B_SG_K"]

df["A_KK"] = (df["A_RSC"] + df["A_ktile"] - 1) // df["A_ktile"]
df["B_KK"] = (df["B_RSC"] + df["B_ktile"] - 1) // df["B_ktile"]

df["ctile"] = df["config_CM_N"] * df["config_SG_N"] * df["config_WG_N"]
df["cdispatch_size"] = (df["output_channels"] + df["ctile"] - 1) // df["ctile"]

# effective output channels, aligned up to CM_N,
# anything in between is not possible anyway because ctile = CM_N * WG_N
df["K_eff"] = (
    (df["output_channels"] + df["config_CM_N"] - 1)
    // df["config_CM_N"]
    * df["config_CM_N"]
)

df["wg_size"] = df["config_WG_M"] * df["config_WG_N"] * df["subgroup_size"]

df["xtile"] = df["config_CM_N"]
df["ytile"] = df["config_SG_N"] * df["config_WG_M"]

df["input_channels"] = df["input_channels0"] + df["input_channels1"]
df["wg_flops"] = (
    2 * df["xtile"] * df["ytile"] * df["output_channels"] * df["input_channels"] * R * S
)
df["A_wg_flops"] = (
    2 * df["xtile"] * df["ytile"] * df["output_channels"] * df["input_channels0"] * R * S
)
df["B_wg_flops"] = (
    2 * df["xtile"] * df["ytile"] * df["output_channels"] * df["input_channels1"] * R * S
)
f16_size = 2
df["wg_loads"] = (
    df["xtile"] * df["ytile"] * df["input_channels"]
    + R * S * df["input_channels"] * df["output_channels"]
    + df["xtile"] * df["ytile"] * df["output_channels"]
) * f16_size
df["A_wg_loads"] = (
    df["xtile"] * df["ytile"] * df["input_channels0"]
    + R * S * df["input_channels0"] * df["output_channels"]
    + df["xtile"] * df["ytile"] * df["output_channels"]
) * f16_size
df["B_wg_loads"] = (
    df["xtile"] * df["ytile"] * df["input_channels1"]
    + R * S * df["input_channels1"] * df["output_channels"]
    + df["xtile"] * df["ytile"] * df["output_channels"]
) * f16_size
df["intensity"] = df["wg_flops"] / df["wg_loads"]
df["A_intensity"] = df["A_wg_flops"] / df["A_wg_loads"]
df["B_intensity"] = df["B_wg_flops"] / df["B_wg_loads"]

def register_policy(df: pd.DataFrame) -> pd.DataFrame:
    MAX_REGISTERS = 256  # that's a lot!
    MIN_REGISTERS = 32
    mask = df["registers"].between(MIN_REGISTERS, MAX_REGISTERS)
    return df.loc[mask]

def shared_memory_policy(df: pd.DataFrame) -> pd.DataFrame:
    # intentionally unrestrictive right now.
    MIN_SHARED_MEMORY = 1024  # 1KiB
    MAX_SHARED_MEMORY = 102400  # 100KiB
    mask = df["sh_size"].between(MIN_SHARED_MEMORY, MAX_SHARED_MEMORY)
    return df.loc[mask]

def perfect_ktiling_policy(df: pd.DataFrame) -> pd.DataFrame:
    A_k_eff = ((df["input_channels0"] + df["config_A_CM_K"] - 1) // df["config_A_CM_K"]) * df[
        "config_A_CM_K"
    ]
    A_RSC_eff = A_k_eff * R * S
    A_mask = (A_RSC_eff % df["A_ktile"]) == 0

    B_k_eff = ((df["input_channels1"] + df["config_B_CM_K"] - 1) // df["config_B_CM_K"]) * df[
        "config_B_CM_K"
    ]
    B_RSC_eff = B_k_eff * R * S
    B_mask = (B_RSC_eff % df["A_ktile"]) == 0

    return df.loc[A_mask & B_mask]

def perfect_ctiling_policy(df: pd.DataFrame) -> pd.DataFrame:
    k_eff = ((df["output_channels"] + df["config_CM_N"] - 1) // df["config_CM_N"]) * df[
        "config_CM_N"
    ]
    mask = k_eff % df["ctile"] == 0
    return df.loc[mask]


def channel_overalloc_policy(df: pd.DataFrame) -> pd.DataFrame:
    # round K up by CM_N (because that's the min granularity that's implementable)
    k_eff = ((df["output_channels"] + df["config_CM_N"] - 1) // df["config_CM_N"]) * df[
        "config_CM_N"
    ]
    # if ctile (output channel tile) is significantly larger
    # this config is probably bad.
    CHANNEL_OVERALLOC_FACTOR = 2
    mask = k_eff * CHANNEL_OVERALLOC_FACTOR >= df["ctile"]
    return df.loc[mask]

def wg_size_policy(df: pd.DataFrame) -> pd.DataFrame:
    MIN_WG_SIZE = 128
    MAX_WG_SIZE = 512
    mask = df["wg_size"].between(MIN_WG_SIZE, MAX_WG_SIZE)
    return df.loc[mask]

def no_channel_tiling_policy(df: pd.DataFrame) -> pd.DataFrame:
    CUTOFF = 256
    # (K < CUTOFF) implies (ctile >= K)
    mask = (df["output_channels"] >= CUTOFF) | (df["ctile"] >= df["output_channels"])
    return df.loc[mask]

def no_async_policy(df: pd.DataFrame) -> pd.DataFrame:
    ASYNC_INTENSITY_THRESHOLD = 10  # <- maybe hardware dependent.
    A_mask = df["config_A_ASYNC"] == (df["A_intensity"] >= ASYNC_INTENSITY_THRESHOLD)
    B_mask = df["config_B_ASYNC"] == (df["B_intensity"] >= ASYNC_INTENSITY_THRESHOLD)
    return df.loc[A_mask & B_mask]

def aspect_ratio_policy(df: pd.DataFrame) -> pd.DataFrame:
    EPS = 1e-3
    MIN_ASPECT = 1 - EPS
    MAX_ASPECT = 2 + EPS
    aspect = df["xtile"] / df["ytile"]
    mask = aspect.between(MIN_ASPECT, MAX_ASPECT)
    return df.loc[mask]


def same_coopmat_policy(df : pd.DataFrame) -> pd.DataFrame:
    mask = (df["config_A_CM_K"] == df["config_B_CM_K"])
    return df.loc[mask]

def same_sgk_policy(df : pd.DataFrame) -> pd.DataFrame:
    mask = (df["config_A_SG_K"] == df["config_B_SG_K"])
    return df.loc[mask]

def fixed_coopmat_policy(df: pd.DataFrame) -> pd.DataFrame:
    # we probably just want to look at the top 2 coopmat shapes, no need to look at all
    CM_M0 = 16
    CM_K0 = 16
    CM_N0 = 16

    CM_M1 = 16
    CM_K1 = 8
    CM_N1 = 8

    mask = (
        (df["config_CM_M"] == CM_M0)
        & ((df["config_A_CM_K"] == CM_K0) | (df["config_A_CM_K"] == CM_K1))
        & ((df["config_B_CM_K"] == CM_K0) | (df["config_B_CM_K"] == CM_K1))
        & (df["config_CM_N"] == CM_N0)
    ) | (
        (df["config_CM_M"] == CM_M1)
        & ((df["config_A_CM_K"] == CM_K0) | (df["config_A_CM_K"] == CM_K1))
        & ((df["config_B_CM_K"] == CM_K0) | (df["config_B_CM_K"] == CM_K1))
        & (df["config_CM_N"] == CM_N1)
    )
    return df.loc[mask]

pruned = df

pruned = register_policy(pruned)
print(f"after register policystart: {pruned.size}")

pruned = shared_memory_policy(pruned)
print(f"after shared   policy: {pruned.size}")

pruned = perfect_ktiling_policy(pruned)
print(f"after perfectk policy {pruned.size}")

pruned = perfect_ctiling_policy(pruned)
print(f"after perfectc policy {pruned.size}")

pruned = channel_overalloc_policy(pruned)
print(f"after overc    policy {pruned.size}")

pruned = wg_size_policy(pruned)
print(f"after wg_size  policy {pruned.size}")

pruned = no_channel_tiling_policy(pruned)
print(f"after nochan   policy {pruned.size}")

pruned = no_async_policy(pruned)
print(f"after noasync  policy {pruned.size}")

# not at opt 3.
# pruned = aspect_ratio_policy(pruned)
# print(f"after aspect  policy {pruned.size}")

pruned = fixed_coopmat_policy(pruned)
print(f"after coopmat  policy {pruned.size}")

pruned = same_coopmat_policy(pruned)
print(f"after samecm   policy {pruned.size}")

# kind of dangerous, sometimes really don't work. (maybe for opt 1)
pruned = same_sgk_policy(pruned)
print(f"after samesgk  policy {pruned.size}")



print(len(pruned.index), "/", len(df.index))


BINS = 100

x_full = df["relative_speedup"].to_numpy()
x_pruned = pruned["relative_speedup"].to_numpy()

# compute shared bin edges
min_v = min(x_full.min(), x_pruned.min())
max_v = max(x_full.max(), x_pruned.max())
bin_edges = np.linspace(min_v, max_v, BINS + 1)

plt.figure(figsize=(8, 5))

plt.hist(
    x_full,
    bins=bin_edges,
    alpha=0.5,
    label="original search space",
)

plt.hist(
    x_pruned,
    bins=bin_edges,
    alpha=0.5,
    label="after pruning",
)

plt.xlabel("relative_speedup")
plt.ylabel("number of configurations")
plt.legend()
plt.tight_layout()
plt.savefig("plot_concat_conv_cm.pdf")
