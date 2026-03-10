from uuid import MAX
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt


dfs: list[pd.DataFrame] = []
for path in list(Path("./parquets/").glob("direct-conv-cm-*")):
    dfs.append(pd.read_parquet(path))

df = pd.concat(dfs, ignore_index=True)

print("Operations: (Pick One)")
for op in df["operation"].unique():
    print(f'"{op}"')

R = 3
S = 3
OP = "relu(conv2d(x,kernel_size=(3,3),bias=true,stride=(1,1),padding=(1,1),dialation=(1,1)))"
df = df.loc[df["operation"] == OP]

print("Devices:")
for device in df["device"].unique():
    print(f'"{device}"')

# DEVICE = "NVIDIA GeForce RTX 4080 SUPER"
# df = df.loc[df["device"] == DEVICE]

print("Input shapes:")
for input_shape in df["input_shape"].unique():
    print(f'"{input_shape}"')


# INPUT_SHAPE = "68x120x32"
# INPUT_SHAPE = "136x240x64"
# INPUT_SHAPE = "272x480x64"
# INPUT_SHAPE = "544x960x64"
# INPUT_SHAPE = "1088x1920x3"
# INPUT_SHAPE = "1088x1920x32"

# INPUT_SHAPE = "1088x1920x64"
# INPUT_SHAPE = "1088x1920x67"
# INPUT_SHAPE = "1088x1920x3"
# INPUT_SHAPE = "544x960x64"
# INPUT_SHAPE = "544x960x128"
# INPUT_SHAPE = "544x960x32"
# INPUT_SHAPE = "1088x1920x32"
# INPUT_SHAPE = "272x480x96"
# INPUT_SHAPE = "272x480x160"
# INPUT_SHAPE = "272x480x48"
# INPUT_SHAPE = "136x240x112"
# INPUT_SHAPE = "136x240x160"
# INPUT_SHAPE = "136x240x64"
# INPUT_SHAPE = "68x120x96"
# INPUT_SHAPE = "68x120x80"
# INPUT_SHAPE = "1088x1920x35"
# INPUT_SHAPE = "544x960x96"
# INPUT_SHAPE = "272x480x64"
# INPUT_SHAPE = "272x480x32"
# INPUT_SHAPE = "136x240x32"
# INPUT_SHAPE = "68x120x32"

# df = df.loc[df["input_shape"] == INPUT_SHAPE]

print("Output shapes:")
for output_shape in df["output_shape"].unique():
    print(f'"{output_shape}"')
#
# OUTPUT_SHAPE = "1088x1920x32"
# df = df.loc[df["output_shape"] == OUTPUT_SHAPE]


df["subgroup_size"] = 32

# horrible register estimates.
# how many u32 registers per coopmat accumulator
df["acc_registers"] = (df["config_CM_M"] * df["config_CM_N"]) / df["subgroup_size"]
# how many u32 registers per coopmat a / b
df["a_registers"] = (df["config_CM_M"] * df["config_CM_K"]) / df["subgroup_size"]
df["b_registers"] = (df["config_CM_K"] * df["config_CM_N"]) / df["subgroup_size"]

df["coopmat_registers"] = (
    df["acc_registers"] * df["config_SG_N"] * df["config_SG_M"]
    + df["a_registers"] * df["config_SG_M"]
    + df["b_registers"] * df["config_SG_N"]
)

# some shader constants
df["prefetch_A_QQ"] = (
    df["config_CM_M"] * df["config_CM_K"] * df["config_SG_K"] * df["config_SG_M"]
) / 8
df["prefetch_A_SQQ"] = df["prefetch_A_QQ"] // df["config_WG_N"]
df["prefetch_A_IQQ"] = (df["prefetch_A_SQQ"] + df["subgroup_size"] - 1) // df[
    "subgroup_size"
]

df["prefetch_B_QQ"] = (
    df["config_CM_K"] * df["config_CM_N"] * df["config_SG_K"] * df["config_SG_N"]
) / 8
df["prefetch_B_SQQ"] = df["prefetch_B_QQ"] // df["config_WG_M"]
df["prefetch_B_IQQ"] = (df["prefetch_B_SQQ"] + df["subgroup_size"] - 1) // df[
    "subgroup_size"
]

df["prefetch_A_registers"] = df["prefetch_A_IQQ"] * 4
df["prefetch_B_registers"] = df["prefetch_B_IQQ"] * 4

df["registers"] = (
    df["coopmat_registers"] + df["prefetch_A_registers"] + df["prefetch_B_registers"]
)

# shared memory size
df["sh_a_size"] = (
    df["config_WG_M"]
    * df["config_CM_M"]
    * df["config_CM_K"]
    * df["config_SG_K"]
    * df["config_SG_M"]
    * 2
)
df["sh_b_size"] = (
    df["config_WG_N"]
    * df["config_CM_K"]
    * df["config_CM_N"]
    * df["config_SG_K"]
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
df["sh_size"] = np.maximum(df["sh_a_size"] + df["sh_b_size"], df["sh_out_size"])


# tiling variables
# The implementation performs a implicit gemm convolution:
# It multiplies a HWxRSC matrix times a RSCxK matrix, where
# - H,W are the output tile spatial dimensions.
# - R,S is the kernel size
# - C is the amount of input channels
# - K is the amount of output channels
df["RSC"] = R * S * df["input_channels"]

# The matrix implementation is tiled, where each iteration processes
# a HWx[ktile] submatrix from the input and [ktile]xK submatrix from the filter.
df["ktile"] = df["config_CM_K"] * df["config_SG_K"]
# In total that means that we have to perform KK iterations
df["KK"] = (df["RSC"] + df["ktile"] - 1) // df["ktile"]

# Each worksgroups handles ctile many output channels
df["ctile"] = df["config_CM_N"] * df["config_SG_N"] * df["config_WG_N"]

# workgroup size
df["wg_size"] = df["config_WG_M"] * df["config_WG_N"] * df["subgroup_size"]


df["xtile"] = df["config_CM_M"]
df["ytile"] = df["config_SG_M"] * df["config_WG_M"]

# Arithmetic intensity.
df["wg_flops"] = (
    2 * df["xtile"] * df["ytile"] * df["output_channels"] * df["input_channels"] * R * S
)
f16_size = 2
df["wg_loads"] = (
    df["xtile"] * df["ytile"] * df["input_channels"]
    + R * S * df["input_channels"] * df["output_channels"]
    + df["xtile"] * df["ytile"] * df["output_channels"]
) * f16_size
df["intensity"] = df["wg_flops"] / df["wg_loads"]


def register_policy(df: pd.DataFrame) -> pd.DataFrame:
    MAX_REGISTERS = 256  # that's a lot!
    MIN_REGISTERS = 32
    mask = df["registers"].between(MIN_REGISTERS, MAX_REGISTERS)
    return df.loc[mask]


# very much hardware dependent, and arguably quite simple. Just
# use something like 0.75 of max is probably a good upper limit.
def shared_memory_policy(df: pd.DataFrame) -> pd.DataFrame:
    # intentionally unrestrictive right now.
    MIN_SHARED_MEMORY = 1024  # 1KiB
    MAX_SHARED_MEMORY = 102400  # 100KiB
    mask = df["sh_size"].between(MIN_SHARED_MEMORY, MAX_SHARED_MEMORY)
    return df.loc[mask]


def perfect_ktiling_policy(df: pd.DataFrame) -> pd.DataFrame:
    k_eff = ((df["input_channels"] + df["config_CM_K"] - 1) // df["config_CM_K"]) * df[
        "config_CM_K"
    ]
    RSC_eff = k_eff * R * S
    mask = (RSC_eff % df["ktile"]) == 0
    return df.loc[mask]


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
    mask = df["config_ASYNC"] == (df["intensity"] >= ASYNC_INTENSITY_THRESHOLD)
    return df.loc[mask]


# doesn't really work.
def perfect_xy_tiling_policy(df: pd.DataFrame) -> pd.DataFrame:
    mask = (df["input_width"] % df["xtile"] == 0) & (
        df["input_height"] % df["ytile"] == 0
    )
    return df.loc[mask]


def aspect_ratio_policy(df: pd.DataFrame) -> pd.DataFrame:
    EPS = 1e-3
    MIN_ASPECT = 1 - EPS
    MAX_ASPECT = 2 + EPS
    aspect = df["xtile"] / df["ytile"]
    mask = aspect.between(MIN_ASPECT, MAX_ASPECT)
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
        & (df["config_CM_K"] == CM_K0)
        & (df["config_CM_N"] == CM_N0)
    ) | (
        (df["config_CM_M"] == CM_M1)
        & (df["config_CM_K"] == CM_K1)
        & (df["config_CM_N"] == CM_N1)
    )
    return df.loc[mask]


def hwc_only_policy(df: pd.DataFrame) -> pd.DataFrame:
    mask = (df["input_format"] == "HWC") & (df["output_format"] == "HWC")
    return df.loc[mask]


# def tile_pressure_policy(df : pd.DataFrame) -> pd.DataFrame:


pruned = df
print(f"original search space: {pruned.size}")

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
#
# # Not crazy good, but again doesn't effect best implementations a lot.
pruned = no_async_policy(pruned)
print(f"after noasync  policy {pruned.size}")

# this has a suprisingly large impact, we should definitely consider disabling it for opt 3
pruned = fixed_coopmat_policy(pruned)
print(f"after coopsel  policy {pruned.size}")
#
# # not a really good indicator of performance, but it reduces search space
# # fairly uniformly. (Maybe only for opt 1)
# pruned = perfect_xy_tiling_policy(pruned)
# print(f"after perfectxy policy {pruned.size}")
#
#
# Actually not to bad. (maybe only for opt 2)
# because it has a good chance for a false positive.
pruned = aspect_ratio_policy(pruned)
print(f"after aspect  policy {pruned.size}")


# for config in pruned["config"].unique():
#     speeds = pruned.loc[pruned["config"] == config, "relative_speedup"]
#     if speeds.max() < 0.8:
#         print(config)
#         print(speeds)


print(len(pruned))


BINS = 50

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
plt.savefig("plot_direct_conv_cm.pdf")

print(df.columns)
