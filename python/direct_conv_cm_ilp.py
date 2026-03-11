import pandas as pd
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from pulp import (
    LpProblem,
    LpVariable,
    LpMinimize,
    lpSum,
    LpBinary,
    PULP_CBC_CMD,
    LpStatus,
    value,
)

# ============================================================
# parameters
# ============================================================
ALPHA = 0.95

# total target coverage per group after anchor + alpha completion
TARGET_G = 10

# these top-K configs are always selected
TOP_K = 1

CBC_TIME_LIMIT_SEC = None
CBC_MSG = False

GROUP_COLS = [
    "operation",
    "device",
    "input_shape",
    "input_format",
    "input_type",
    "output_shape",
    "output_format",
    "output_type",
]

LOGICAL_OP_COLS = [
    "operation",
    "input_shape",
    "input_format",
    "input_type",
    "output_shape",
    "output_format",
    "output_type",
]

SCORE_COL = "relative_speedup"

# pick one config space
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

PARQUET_GLOB = "direct-conv-cm*"
DOES_NOT_START_WITH = None
# DOES_NOT_START_WITH = "direct-conv-cm"

# devices used to BUILD the policy
POLICY_DEVICES = [
    "NVIDIA GeForce RTX 4070",
    "NVIDIA GeForce RTX 4080 SUPER",
    "Intel(R) Arc(tm) B580 Graphics (BMG G21)",
]
# POLICY_DEVICES = None

# devices used to EVALUATE the policy
EVAL_DEVICES = [
    "NVIDIA GeForce RTX 4070",
    "NVIDIA GeForce RTX 4080 SUPER",
    "Intel(R) Arc(tm) B580 Graphics (BMG G21)",
]
# EVAL_DEVICES = None

INPUT_SHAPE = None
# INPUT_SHAPE = "1088x1920x32"
# INPUT_SHAPE = "544x960x32"
# INPUT_SHAPE = "272x480x48"
# INPUT_SHAPE = "136x240x64"
# INPUT_SHAPE = "272x480x32"
# INPUT_SHAPE = "136x240x32"
# INPUT_SHAPE = "1088x1920x64"
# INPUT_SHAPE = "1088x1920x67"
# INPUT_SHAPE = "1088x1920x3"
# INPUT_SHAPE = "544x960x64"
# INPUT_SHAPE = "544x960x128"
# INPUT_SHAPE = "272x480x96"
# INPUT_SHAPE = "272x480x160"
# INPUT_SHAPE = "136x240x112"
# INPUT_SHAPE = "136x240x160"
# INPUT_SHAPE = "68x120x96"
# INPUT_SHAPE = "68x120x80"
# INPUT_SHAPE = "1088x1920x35"
# INPUT_SHAPE = "544x960x96"
# INPUT_SHAPE = "272x480x64"
# INPUT_SHAPE = "68x120x32"

BINS = 200


# ============================================================
# helpers
# ============================================================
def make_topk(
    df: pd.DataFrame,
    group_cols: list[str],
    score_col: str,
    top_k: int,
) -> pd.DataFrame:
    df_sorted = df.sort_values(
        by=group_cols + [score_col],
        ascending=[True] * len(group_cols) + [False],
        kind="mergesort",
    )

    topk = (
        df_sorted.groupby(group_cols, group_keys=False, sort=False).head(top_k).copy()
    )

    topk["group_rank"] = (
        topk.groupby(group_cols)[score_col]
        .rank(method="first", ascending=False)
        .astype(int)
    )
    return topk


def make_alpha_eligible(
    df: pd.DataFrame,
    score_col: str,
    alpha: float,
) -> pd.DataFrame:
    return df.loc[df[score_col] >= alpha].copy()


def build_config_universe(
    df: pd.DataFrame,
    config_cols: list[str],
) -> pd.DataFrame:
    config_universe = (
        df[config_cols]
        .drop_duplicates()
        .sort_values(config_cols, kind="mergesort")
        .reset_index(drop=True)
    )
    config_universe["config_id"] = np.arange(len(config_universe), dtype=np.int32)
    return config_universe


def make_config_policy(
    config_universe: pd.DataFrame,
    surviving_configs: pd.DataFrame,
    config_cols: list[str],
):
    surviving_configs_marked = surviving_configs[config_cols].drop_duplicates().copy()
    surviving_configs_marked["keep"] = True

    config_policy = config_universe.merge(
        surviving_configs_marked,
        on=config_cols,
        how="left",
    )
    config_policy["keep"] = config_policy["keep"].fillna(False).astype(bool)

    mask_bool = config_policy["keep"].to_numpy(dtype=bool)
    mask_u8 = mask_bool.astype(np.uint8)
    mask_packed = np.packbits(mask_u8, bitorder="little")

    return config_policy, mask_bool, mask_u8, mask_packed


def apply_config_policy(
    df: pd.DataFrame,
    config_policy: pd.DataFrame,
    config_cols: list[str],
) -> pd.DataFrame:
    return df.merge(
        config_policy.loc[config_policy["keep"], config_cols],
        on=config_cols,
        how="inner",
    )


def solve_anchor_plus_alpha_completion(
    source_df: pd.DataFrame,
    anchor_df: pd.DataFrame,
    alpha_df: pd.DataFrame,
    group_cols: list[str],
    config_cols: list[str],
    target_g: int,
    time_limit_sec=None,
    msg: bool = False,
):
    # ------------------------------------------------------------
    # deterministic groups
    # ------------------------------------------------------------
    groups = (
        source_df[group_cols]
        .drop_duplicates()
        .sort_values(group_cols, kind="mergesort")
        .reset_index(drop=True)
    )
    groups["group_id"] = np.arange(len(groups), dtype=np.int32)

    # ------------------------------------------------------------
    # mandatory anchor configs M = union of per-group top-K
    # ------------------------------------------------------------
    mandatory_configs = (
        anchor_df[config_cols]
        .drop_duplicates()
        .sort_values(config_cols, kind="mergesort")
        .reset_index(drop=True)
    )

    if mandatory_configs.empty:
        raise RuntimeError("Mandatory anchor config set is empty.")

    # ------------------------------------------------------------
    # relations needed for base coverage count
    # base coverage for a group comes from:
    #   - its own anchor configs F_t
    #   - any mandatory config that is alpha-good for that group
    # ------------------------------------------------------------
    anchor_relation = (
        anchor_df[group_cols + config_cols]
        .drop_duplicates()
        .merge(groups, on=group_cols, how="inner")[["group_id"] + config_cols]
        .drop_duplicates()
    )

    alpha_mandatory_relation = (
        alpha_df[group_cols + config_cols]
        .drop_duplicates()
        .merge(groups, on=group_cols, how="inner")
        .merge(mandatory_configs, on=config_cols, how="inner")[
            ["group_id"] + config_cols
        ]
        .drop_duplicates()
    )

    # combine both so base coverage counts every already-selected acceptable config
    base_relation = pd.concat(
        [anchor_relation, alpha_mandatory_relation], ignore_index=True
    ).drop_duplicates()

    base_count_per_group = base_relation.groupby("group_id").size().to_dict()

    # ------------------------------------------------------------
    # augmentation candidates D = alpha configs not already mandatory
    # ------------------------------------------------------------
    augment_configs = (
        alpha_df[config_cols]
        .drop_duplicates()
        .merge(
            mandatory_configs.assign(_mandatory=True),
            on=config_cols,
            how="left",
        )
    )
    augment_configs = (
        augment_configs.loc[augment_configs["_mandatory"].isna(), config_cols]
        .drop_duplicates()
        .sort_values(config_cols, kind="mergesort")
        .reset_index(drop=True)
    )
    augment_configs["augment_config_id"] = np.arange(
        len(augment_configs), dtype=np.int32
    )

    augment_relation = (
        alpha_df[group_cols + config_cols]
        .drop_duplicates()
        .merge(groups, on=group_cols, how="inner")
        .merge(augment_configs, on=config_cols, how="inner")[
            ["group_id", "augment_config_id"]
        ]
        .drop_duplicates()
        .sort_values(["group_id", "augment_config_id"], kind="mergesort")
        .reset_index(drop=True)
    )

    group_to_augment = (
        augment_relation.groupby("group_id")["augment_config_id"].apply(list).to_dict()
    )

    augment_count_per_group = {
        g: len(group_to_augment.get(g, [])) for g in groups["group_id"]
    }

    # ------------------------------------------------------------
    # residual demand after mandatory anchors
    # ------------------------------------------------------------
    residual_demand = {}
    feasible_residual_demand = {}

    for g in groups["group_id"]:
        base_count = base_count_per_group.get(g, 0)
        residual = max(0, target_g - base_count)
        feasible = min(residual, augment_count_per_group[g])

        residual_demand[g] = residual
        feasible_residual_demand[g] = feasible

    num_groups_already_satisfied = sum(
        1 for g in groups["group_id"] if residual_demand[g] == 0
    )
    num_groups_with_no_alpha_augmentation = sum(
        1
        for g in groups["group_id"]
        if residual_demand[g] > 0 and augment_count_per_group[g] == 0
    )

    print(
        f"\nGroups already satisfied by mandatory anchors: {num_groups_already_satisfied}"
    )
    print(
        f"Groups needing more coverage but with no extra alpha configs: {num_groups_with_no_alpha_augmentation}"
    )

    # ------------------------------------------------------------
    # ILP: choose smallest extra alpha set Y
    # ------------------------------------------------------------
    if augment_configs.empty or all(
        feasible_residual_demand[g] == 0 for g in groups["group_id"]
    ):
        selected_extra_configs = augment_configs.iloc[0:0][config_cols].copy()
        final_selected_configs = (
            mandatory_configs[config_cols]
            .drop_duplicates()
            .sort_values(config_cols, kind="mergesort")
            .reset_index(drop=True)
        )
        return {
            "status": "Trivial",
            "objective": 0,
            "groups": groups,
            "mandatory_configs": mandatory_configs,
            "augment_configs": augment_configs,
            "selected_extra_configs": selected_extra_configs,
            "selected_configs": final_selected_configs,
            "base_count_per_group": base_count_per_group,
            "residual_demand": residual_demand,
            "feasible_residual_demand": feasible_residual_demand,
            "augment_count_per_group": augment_count_per_group,
        }

    prob = LpProblem("anchor_plus_alpha_completion", LpMinimize)

    y = {
        c: LpVariable(f"y_{c}", cat=LpBinary)
        for c in augment_configs["augment_config_id"]
    }

    prob += lpSum(y[c] for c in augment_configs["augment_config_id"])

    for g in groups["group_id"]:
        d = feasible_residual_demand[g]
        if d <= 0:
            continue
        covered_by = group_to_augment.get(g, [])
        prob += lpSum(y[c] for c in covered_by) >= d, f"augment_group_{g}"

    solver_kwargs = {"msg": msg}
    if time_limit_sec is not None:
        solver_kwargs["timeLimit"] = time_limit_sec

    status = prob.solve(PULP_CBC_CMD(**solver_kwargs))
    objective = value(prob.objective)
    if objective is not None:
        objective = int(round(objective))

    selected_augment_ids = [
        c
        for c in augment_configs["augment_config_id"]
        if y[c].value() is not None and y[c].value() > 0.5
    ]

    selected_extra_configs = (
        augment_configs.loc[
            augment_configs["augment_config_id"].isin(selected_augment_ids),
            config_cols,
        ]
        .drop_duplicates()
        .sort_values(config_cols, kind="mergesort")
        .reset_index(drop=True)
    )

    final_selected_configs = (
        pd.concat(
            [
                mandatory_configs[config_cols],
                selected_extra_configs[config_cols],
            ],
            ignore_index=True,
        )
        .drop_duplicates()
        .sort_values(config_cols, kind="mergesort")
        .reset_index(drop=True)
    )

    return {
        "status": LpStatus[status],
        "objective": objective,
        "groups": groups,
        "mandatory_configs": mandatory_configs,
        "augment_configs": augment_configs,
        "selected_extra_configs": selected_extra_configs,
        "selected_configs": final_selected_configs,
        "base_count_per_group": base_count_per_group,
        "residual_demand": residual_demand,
        "feasible_residual_demand": feasible_residual_demand,
        "augment_count_per_group": augment_count_per_group,
    }


# ============================================================
# load data
# ============================================================
dfs: list[pd.DataFrame] = []
for path in Path("./parquets/").glob(PARQUET_GLOB):
    if DOES_NOT_START_WITH is not None and path.name.startswith(DOES_NOT_START_WITH):
        continue
    dfs.append(pd.read_parquet(path))

if not dfs:
    raise RuntimeError(f"No parquet files matched ./parquets/{PARQUET_GLOB}")

raw_df = pd.concat(dfs, ignore_index=True)

print("Devices:")
for device in sorted(raw_df["device"].dropna().unique()):
    print(f'"{device}"')

print("\nInput shapes:")
for input_shape in raw_df["input_shape"].dropna().unique():
    print(f'"{input_shape}"')

# ============================================================
# policy construction dataset
# ============================================================
policy_df_source = raw_df.copy()

if POLICY_DEVICES is not None:
    policy_df_source = policy_df_source.loc[
        policy_df_source["device"].isin(POLICY_DEVICES)
    ].copy()

if policy_df_source.empty:
    raise RuntimeError("Policy construction dataframe is empty after filtering.")

# ============================================================
# evaluation dataset
# ============================================================
eval_df = raw_df.copy()

if EVAL_DEVICES is not None:
    eval_df = eval_df.loc[eval_df["device"].isin(EVAL_DEVICES)].copy()

if INPUT_SHAPE is not None:
    eval_df = eval_df.loc[eval_df["input_shape"] == INPUT_SHAPE].copy()

if eval_df.empty:
    raise RuntimeError("Evaluation dataframe is empty after filtering.")

print("Max-Latency   :", eval_df["median_latency_ms"].max(), "ms")
print("Min-Latency   :", eval_df["median_latency_ms"].min(), "ms")
print("Median-Latency:", eval_df["median_latency_ms"].median(), "ms")

# ============================================================
# build sets
# ============================================================
anchor_topk = make_topk(
    df=policy_df_source,
    group_cols=GROUP_COLS,
    score_col=SCORE_COL,
    top_k=TOP_K,
)

alpha_eligible = make_alpha_eligible(
    df=policy_df_source,
    score_col=SCORE_COL,
    alpha=ALPHA,
)

topk_union_surviving_configs = (
    anchor_topk[CONFIG_COLS]
    .drop_duplicates()
    .sort_values(CONFIG_COLS, kind="mergesort")
    .reset_index(drop=True)
)

alpha_union_surviving_configs = (
    alpha_eligible[CONFIG_COLS]
    .drop_duplicates()
    .sort_values(CONFIG_COLS, kind="mergesort")
    .reset_index(drop=True)
)

# ============================================================
# anchor + alpha completion
# ============================================================
completion_result = solve_anchor_plus_alpha_completion(
    source_df=policy_df_source,
    anchor_df=anchor_topk,
    alpha_df=alpha_eligible,
    group_cols=GROUP_COLS,
    config_cols=CONFIG_COLS,
    target_g=TARGET_G,
    time_limit_sec=CBC_TIME_LIMIT_SEC,
    msg=CBC_MSG,
)

print(f"\nAnchor+alpha solver status: {completion_result['status']}")
print(f"Extra alpha configs selected: {completion_result['objective']}")

anchored_alpha_surviving_configs = completion_result["selected_configs"]

# ============================================================
# bitmasks
# ============================================================
config_universe = build_config_universe(policy_df_source, CONFIG_COLS)

topk_config_policy, topk_mask_bool, topk_mask_u8, topk_mask_packed = make_config_policy(
    config_universe=config_universe,
    surviving_configs=topk_union_surviving_configs,
    config_cols=CONFIG_COLS,
)

(
    alpha_union_config_policy,
    alpha_union_mask_bool,
    alpha_union_mask_u8,
    alpha_union_mask_packed,
) = make_config_policy(
    config_universe=config_universe,
    surviving_configs=alpha_union_surviving_configs,
    config_cols=CONFIG_COLS,
)

(
    anchored_alpha_config_policy,
    anchored_alpha_mask_bool,
    anchored_alpha_mask_u8,
    anchored_alpha_mask_packed,
) = make_config_policy(
    config_universe=config_universe,
    surviving_configs=anchored_alpha_surviving_configs,
    config_cols=CONFIG_COLS,
)

# ============================================================
# apply policies to evaluation data
# ============================================================
eval_topk_policy_df = apply_config_policy(eval_df, topk_config_policy, CONFIG_COLS)
eval_alpha_union_policy_df = apply_config_policy(
    eval_df, alpha_union_config_policy, CONFIG_COLS
)
eval_anchored_alpha_policy_df = apply_config_policy(
    eval_df, anchored_alpha_config_policy, CONFIG_COLS
)

# ============================================================
# basic summary
# ============================================================
print()
print(f"ALPHA         : {ALPHA}")
print(f"TARGET_G      : {TARGET_G}")
print(f"ANCHOR_TOP_K  : {TOP_K}")

print(
    f"\nSearch-Space  : "
    f"{len(eval_df.index)} dispatches, "
    f"{eval_df.groupby(CONFIG_COLS).ngroups} unique configs, "
    f"max-used {eval_df.groupby(GROUP_COLS).size().max()}, "
    f"min-used {eval_df.groupby(GROUP_COLS).size().min()}"
)
print(
    f"Top-K Union   : "
    f"{len(eval_topk_policy_df.index)} dispatches, "
    f"{eval_topk_policy_df.groupby(CONFIG_COLS).ngroups} unique configs, "
    f"max-used {eval_topk_policy_df.groupby(GROUP_COLS).size().max()}, "
    f"min-used {eval_topk_policy_df.groupby(GROUP_COLS).size().min()}"
)
print(
    f"Alpha Union   : "
    f"{len(eval_alpha_union_policy_df.index)} dispatches, "
    f"{eval_alpha_union_policy_df.groupby(CONFIG_COLS).ngroups} unique configs, "
    f"max-used {eval_alpha_union_policy_df.groupby(GROUP_COLS).size().max()}, "
    f"min-used {eval_alpha_union_policy_df.groupby(GROUP_COLS).size().min()}"
)
print(
    f"Anch+Alpha    : "
    f"{len(eval_anchored_alpha_policy_df.index)} dispatches, "
    f"{eval_anchored_alpha_policy_df.groupby(CONFIG_COLS).ngroups} unique configs, "
    f"max-used {eval_anchored_alpha_policy_df.groupby(GROUP_COLS).size().max()}, "
    f"min-used {eval_anchored_alpha_policy_df.groupby(GROUP_COLS).size().min()}"
)

# ============================================================
# histogram
# ============================================================
x_full = eval_df[SCORE_COL].to_numpy()
x_topk_union = eval_topk_policy_df[SCORE_COL].to_numpy()
x_alpha_union = eval_alpha_union_policy_df[SCORE_COL].to_numpy()
x_anchored_alpha = eval_anchored_alpha_policy_df[SCORE_COL].to_numpy()

if (
    len(x_full) > 0
    and len(x_topk_union) > 0
    and len(x_alpha_union) > 0
    and len(x_anchored_alpha) > 0
):
    min_v = min(
        x_full.min(),
        x_topk_union.min(),
        x_alpha_union.min(),
        x_anchored_alpha.min(),
    )
    max_v = max(
        x_full.max(),
        x_topk_union.max(),
        x_alpha_union.max(),
        x_anchored_alpha.max(),
    )
    bin_edges = np.linspace(min_v, max_v, BINS + 1)

    plt.figure(figsize=(10, 5))
    plt.hist(
        x_full,
        bins=bin_edges,
        alpha=1,
        label="full search space",
    )

    plt.hist(
        x_anchored_alpha,
        bins=bin_edges,
        alpha=1,
        label=f"anchor+alpha completion (K={TOP_K}, alpha={ALPHA}, G={TARGET_G})",
    )

    plt.hist(
        x_topk_union,
        bins=bin_edges,
        alpha=1,
        label="top-k union",
    )
    # plt.hist(
    #     x_alpha_union,
    #     bins=bin_edges,
    #     alpha=0.35,
    #     label=f"alpha union (alpha={ALPHA})",
    # )
    plt.xlabel(SCORE_COL)
    plt.ylabel("number of rows")
    plt.legend()
    plt.tight_layout()
    plt.savefig("plot_anchor_plus_alpha_completion.pdf")
    plt.close()


def set_cover_policy(df : pd.DataFrame, alpha=0.9, g=10, k=1):
    pass
