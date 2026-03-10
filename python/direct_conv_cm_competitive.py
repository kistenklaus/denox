import pandas as pd
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# parameters
# ============================================================
TOP_K = 10

GROUP_COLS = [
    "operation",
    "device",
    "input_channels",
    "input_format",
    "input_type",
    "output_channels",
    "output_format",
    "output_type",
]

# logical operation = same op + same IO signature, but device-agnostic
LOGICAL_OP_COLS = [
    "operation",
    "input_channels",
    "input_format",
    "input_type",
    "output_channels",
    "output_format",
    "output_type",
]

SCORE_COL = "relative_speedup"

CONFIG_COLS = [
    "config_CM_M",
    "config_CM_K",
    "config_CM_N",
    "config_SG_M",
    "config_SG_K",
    "config_SG_N",
    "config_WG_M",
    "config_WG_N",
    "config_ASYNC",
]
# CONFIG_COLS = [
#     "config_INVOC_M",
#     "config_INVOC_K",
#     "config_INVOC_N",
#     "config_SG_M",
#     "config_SG_K",
#     "config_SG_N",
#     "config_WG_M",
#     "config_WG_N",
#     "config_ASYNC",
# ]

# PARQUET_GLOB = "direct-conv-cm-*"

# Optional device filter
DEVICES = [
    "NVIDIA GeForce RTX 4070",
    # "NVIDIA GeForce RTX 3080 Ti", # <- irgendwas stimmt hier doch nicht an den datensätzen ??
    # "NVIDIA GeForce RTX 4080 SUPER",
    "Intel(R) Arc(tm) B580 Graphics (BMG G21)",
]
# DEVICES = None

# ============================================================
# load data
# ============================================================
dfs: list[pd.DataFrame] = []
for path in Path("./parquets/").glob("direct-conv-cm*"):
    # if path.name.startswith("direct-conv-cm"):
    #     continue
    dfs.append(pd.read_parquet(path))

# if not dfs:
#     raise RuntimeError(f"No parquet files matched ./parquets/....")

df = pd.concat(dfs, ignore_index=True)


print("Devices:")
for device in sorted(df["device"].dropna().unique()):
    print(f'"{device}"')


if DEVICES is not None:
    df = df.loc[df["device"].isin(DEVICES)].copy()

print("Input shapes:")
for input_shape in df["input_shape"].unique():
    print(f'"{input_shape}"')

odf = df

# INPUT_SHAPE = "1088x1920x64"
# INPUT_SHAPE = "1088x1920x67"
# INPUT_SHAPE = "68x120x32"

# odf = odf.loc[df["input_shape"] == INPUT_SHAPE]


if df.empty:
    raise RuntimeError("Dataframe is empty after filtering.")


# df = df.loc[(df["sample_count"].between(10, 12)) & (df["std_latency_ms"] < 0.1)]
# df = df.loc[(df["median_latency_ms"] > 0.1)]

# ============================================================
# top-k rows per group
# ============================================================
df_sorted = df.sort_values(
    by=GROUP_COLS + [SCORE_COL],
    ascending=[True] * len(GROUP_COLS) + [False],
    kind="mergesort",
)

topk = df_sorted.groupby(GROUP_COLS, group_keys=False, sort=False).head(TOP_K).copy()

topk["group_rank"] = (
    topk.groupby(GROUP_COLS)[SCORE_COL]
    .rank(method="first", ascending=False)
    .astype(int)
)

# ============================================================
# policy definition
# A config survives iff it appears at least once in any group's top-k.
# ============================================================
surviving_configs = (
    topk[CONFIG_COLS]
    .drop_duplicates()
    .sort_values(CONFIG_COLS, kind="mergesort")
    .reset_index(drop=True)
)

# ============================================================
# deterministic config universe
# The bitmask refers to this exact ordering.
# ============================================================
config_universe = (
    df[CONFIG_COLS]
    .drop_duplicates()
    .sort_values(CONFIG_COLS, kind="mergesort")
    .reset_index(drop=True)
)

config_universe["config_id"] = np.arange(len(config_universe), dtype=np.int32)

# Mark whether each config survives the policy
surviving_configs_marked = surviving_configs.copy()
surviving_configs_marked["keep"] = True

config_policy = config_universe.merge(
    surviving_configs_marked,
    on=CONFIG_COLS,
    how="left",
)

config_policy["keep"] = config_policy["keep"].fillna(False).astype(bool)

# Boolean / bit mask in config_id order
policy_mask_bool = config_policy["keep"].to_numpy(dtype=bool)
policy_mask_u8 = policy_mask_bool.astype(np.uint8)

# Packed bitmask: bit i corresponds to config_id i
policy_mask_packed = np.packbits(policy_mask_u8, bitorder="little")

# ============================================================
# apply policy to original dataframe
# Keep every original row whose config survives globally.
# ============================================================
policy_df = odf.merge(
    config_policy.loc[config_policy["keep"], CONFIG_COLS],
    on=CONFIG_COLS,
    how="inner",
)

# ============================================================
# optional competitive config stats
# ============================================================
competitive_configs = (
    topk.groupby(CONFIG_COLS, dropna=False)
    .agg(
        selected_count=(SCORE_COL, "size"),
        best_score=(SCORE_COL, "max"),
        mean_score=(SCORE_COL, "mean"),
        median_score=(SCORE_COL, "median"),
        worst_score=(SCORE_COL, "min"),
        best_rank=("group_rank", "min"),
        mean_rank=("group_rank", "mean"),
        num_operations=("operation", "nunique"),
        num_devices=("device", "nunique"),
    )
    .reset_index()
    .sort_values(
        by=["selected_count", "best_score", "mean_score"],
        ascending=[False, False, False],
        kind="mergesort",
    )
    .reset_index(drop=True)
)

# ============================================================
# reconstruct fair runtime search space per logical operation
# using the policy-filtered dataframe
# ============================================================
original_logical_op_configs = df[LOGICAL_OP_COLS + CONFIG_COLS].drop_duplicates()
policy_logical_op_configs = policy_df[LOGICAL_OP_COLS + CONFIG_COLS].drop_duplicates()

original_search_space = (
    original_logical_op_configs.groupby(LOGICAL_OP_COLS, dropna=False)
    .size()
    .reset_index(name="original_search_space_size")
)

policy_search_space = (
    policy_logical_op_configs.groupby(LOGICAL_OP_COLS, dropna=False)
    .size()
    .reset_index(name="policy_search_space_size")
)

search_space_comparison = original_search_space.merge(
    policy_search_space,
    on=LOGICAL_OP_COLS,
    how="left",
)

search_space_comparison["policy_search_space_size"] = (
    search_space_comparison["policy_search_space_size"].fillna(0).astype(int)
)

search_space_comparison["removed_configs"] = (
    search_space_comparison["original_search_space_size"]
    - search_space_comparison["policy_search_space_size"]
)

search_space_comparison["kept_fraction"] = (
    search_space_comparison["policy_search_space_size"]
    / search_space_comparison["original_search_space_size"]
)

search_space_comparison["reduction_factor"] = search_space_comparison[
    "original_search_space_size"
] / search_space_comparison["policy_search_space_size"].replace(0, np.nan)

# ============================================================
# print summaries
# ============================================================
print()
print(f"Total rows in original dataframe: {len(df)}")
print(f"Rows after top-{TOP_K}: {len(topk)}")
print(f"Total unique configs in universe: {len(config_universe)}")
print(f"Surviving unique configs: {int(policy_mask_bool.sum())}")
print(f"Pruned unique configs: {len(config_universe) - int(policy_mask_bool.sum())}")
print(f"Rows after policy: {len(policy_df)}")
print(f"Number of logical operations: {len(original_search_space)}")

print("\nOriginal per-logical-operation search-space summary:")
print(search_space_comparison["original_search_space_size"].describe())

print("\nPolicy per-logical-operation search-space summary:")
print(search_space_comparison["policy_search_space_size"].describe())

print("\nRemoved configs summary:")
print(search_space_comparison["removed_configs"].describe())

print("\nKept fraction summary:")
print(search_space_comparison["kept_fraction"].describe())

print("\nReduction factor summary:")
print(search_space_comparison["reduction_factor"].describe())

summary = pd.DataFrame(
    {
        "original_search_space_size": search_space_comparison[
            "original_search_space_size"
        ].describe(),
        "policy_search_space_size": search_space_comparison[
            "policy_search_space_size"
        ].describe(),
        "removed_configs": search_space_comparison["removed_configs"].describe(),
        "kept_fraction": search_space_comparison["kept_fraction"].describe(),
        "reduction_factor": search_space_comparison["reduction_factor"].describe(),
    }
)

print("\nCombined summary:")
print(summary)

# ============================================================
# save outputs
# ============================================================
competitive_configs.to_parquet("competitive_configs.parquet", index=False)
topk.to_parquet("topk_per_group.parquet", index=False)

config_policy.to_parquet("config_policy.parquet", index=False)
policy_df.to_parquet("policy_df.parquet", index=False)

original_search_space.to_parquet("original_search_space.parquet", index=False)
policy_search_space.to_parquet("policy_search_space.parquet", index=False)
search_space_comparison.to_parquet("search_space_comparison.parquet", index=False)

summary.to_csv("search_space_summary.csv")

# Raw masks
np.save("config_policy_mask_bool.npy", policy_mask_bool)
np.save("config_policy_mask_u8.npy", policy_mask_u8)

# Packed mask
with open("config_policy_mask.bin", "wb") as f:
    f.write(policy_mask_packed.tobytes())

# Optional human-readable mask
with open("config_policy_mask.txt", "w") as f:
    f.write("".join("1" if x else "0" for x in policy_mask_bool))

# ============================================================
# histogram: original row scores vs policy-filtered scores
# ============================================================
BINS = 50

x_full = odf[SCORE_COL].to_numpy()
x_policy = policy_df[SCORE_COL].to_numpy()

if len(x_full) > 0 and len(x_policy) > 0:
    min_v = min(x_full.min(), x_policy.min())
    max_v = max(x_full.max(), x_policy.max())
    bin_edges = np.linspace(min_v, max_v, BINS + 1)

    plt.figure(figsize=(8, 5))
    plt.hist(
        x_full,
        bins=bin_edges,
        alpha=0.5,
        label="original search space",
    )
    plt.hist(
        x_policy,
        bins=bin_edges,
        alpha=0.5,
        label="after top-k config policy",
    )
    plt.xlabel(SCORE_COL)
    plt.ylabel("number of rows")
    plt.legend()
    plt.tight_layout()
    plt.savefig("plot_direct_conv_cm_policy_scores.pdf")
    plt.close()

# ============================================================
# histogram: original vs policy per-logical-op search sizes
# ============================================================
x_orig_size = search_space_comparison["original_search_space_size"].to_numpy()
x_policy_size = search_space_comparison["policy_search_space_size"].to_numpy()

if len(x_orig_size) > 0 and len(x_policy_size) > 0:
    min_v = min(x_orig_size.min(), x_policy_size.min())
    max_v = max(x_orig_size.max(), x_policy_size.max())
    bin_edges = np.linspace(min_v, max_v, BINS + 1)

    plt.figure(figsize=(8, 5))
    plt.hist(
        x_orig_size,
        bins=bin_edges,
        alpha=0.5,
        label="original per-logical-op search space",
    )
    plt.hist(
        x_policy_size,
        bins=bin_edges,
        alpha=0.5,
        label="policy per-logical-op search space",
    )
    plt.xlabel("number of configurations")
    plt.ylabel("number of logical operations")
    plt.legend()
    plt.tight_layout()
    plt.savefig("plot_direct_conv_cm_policy_search_space_sizes.pdf")
    plt.close()
