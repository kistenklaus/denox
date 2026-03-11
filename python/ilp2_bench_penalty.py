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
# Only configs with relative_speedup >= ALPHA are considered.
ALPHA = 0.75

# Each group should keep at least this many selected alpha-eligible configs,
# clamped to the number actually available in that group.
G = 25

# Tradeoff parameter:
# larger BETA -> stronger pressure to reduce average autotuning burden
# smaller BETA -> stronger pressure to minimize regret
BETA = 0.1

CBC_TIME_LIMIT_SEC = None
CBC_MSG = False

SCORE_COL = "relative_speedup"

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

PARQUET_GLOB = "direct-conv-cm*"
# DOES_NOT_START_WITH = "direct-conv-cm"
DOES_NOT_START_WITH = None

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
def infer_config_cols(df: pd.DataFrame) -> list[str]:
    config_cols = sorted(c for c in df.columns if c.startswith("config_"))
    if not config_cols:
        raise RuntimeError("No config_* columns found in dataframe.")
    return config_cols


def infer_latency_col(df: pd.DataFrame) -> str:
    preferred = ["latency_ms", "mean_latency_ms", "median_latency_ms"]
    for c in preferred:
        if c in df.columns:
            return c
    raise RuntimeError(f"Could not infer latency column. Expected one of: {preferred}")


def require_columns(df: pd.DataFrame, cols: list[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing required columns: {missing}")


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


def solve_alpha_latency_regret_cover_normalized(
    source_df: pd.DataFrame,
    alpha_df: pd.DataFrame,
    group_cols: list[str],
    config_cols: list[str],
    latency_col: str,
    beta: float,
    g: int,
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
    # alpha candidate rows
    # deduplicate by (group, config), keeping the smallest latency
    # ------------------------------------------------------------
    candidate_rows = (
        alpha_df[group_cols + config_cols + [latency_col, SCORE_COL]]
        .sort_values(latency_col, kind="mergesort")
        .drop_duplicates(subset=group_cols + config_cols, keep="first")
        .reset_index(drop=True)
    )

    if candidate_rows.empty:
        raise RuntimeError("Alpha candidate set is empty. Lower ALPHA.")

    # ------------------------------------------------------------
    # best latency per group from the full construction dataset
    # ------------------------------------------------------------
    best_latency = (
        source_df.groupby(group_cols, dropna=False)[latency_col]
        .min()
        .reset_index(name="best_latency_ms")
    )

    # ------------------------------------------------------------
    # candidate config universe for solver
    # ------------------------------------------------------------
    candidate_configs = (
        candidate_rows[config_cols]
        .drop_duplicates()
        .sort_values(config_cols, kind="mergesort")
        .reset_index(drop=True)
    )
    candidate_configs["candidate_config_id"] = np.arange(
        len(candidate_configs), dtype=np.int32
    )

    # ------------------------------------------------------------
    # edges (group, config) with latency + regret
    # only alpha-eligible edges are present
    # ------------------------------------------------------------
    edge_df = (
        candidate_rows.merge(groups, on=group_cols, how="inner")
        .merge(candidate_configs, on=config_cols, how="inner")
        .merge(best_latency, on=group_cols, how="inner")
        .copy()
    )

    edge_df["regret_ms"] = edge_df[latency_col] - edge_df["best_latency_ms"]
    edge_df["regret_ms"] = edge_df["regret_ms"].clip(lower=0.0)

    edge_df = (
        edge_df[
            [
                "group_id",
                "candidate_config_id",
                latency_col,
                "best_latency_ms",
                "regret_ms",
            ]
        ]
        .drop_duplicates()
        .sort_values(["group_id", "candidate_config_id"], kind="mergesort")
        .reset_index(drop=True)
    )
    edge_df["edge_id"] = np.arange(len(edge_df), dtype=np.int32)

    # ------------------------------------------------------------
    # group -> candidate configs / edges
    # ------------------------------------------------------------
    group_to_configs = (
        edge_df.groupby("group_id")["candidate_config_id"]
        .apply(lambda s: sorted(set(s)))
        .to_dict()
    )
    group_to_edges = edge_df.groupby("group_id")["edge_id"].apply(list).to_dict()

    eligible_count_per_group = {
        g_id: len(group_to_configs.get(g_id, [])) for g_id in groups["group_id"]
    }

    groups_without_candidates = [
        g_id for g_id, n in eligible_count_per_group.items() if n == 0
    ]
    if groups_without_candidates:
        raise RuntimeError(
            f"Found groups with zero alpha-eligible configs: {len(groups_without_candidates)}. "
            "Lower ALPHA or inspect the dataset."
        )

    demand_per_group = {
        g_id: min(g, eligible_count_per_group[g_id]) for g_id in groups["group_id"]
    }

    # ------------------------------------------------------------
    # normalized measurement cost coefficient for each config:
    #
    # beta * sum_t (1 / |E_t|) * L_{t,c}
    #
    # This keeps the unit in ms, but removes the raw scaling with |E_t|.
    # ------------------------------------------------------------
    group_size_inv = {
        g_id: 1.0 / eligible_count_per_group[g_id] for g_id in groups["group_id"]
    }

    edge_df["normalized_measurement_term"] = edge_df[latency_col] * edge_df[
        "group_id"
    ].map(group_size_inv)

    measurement_cost_coef = (
        edge_df.groupby("candidate_config_id")["normalized_measurement_term"]
        .sum()
        .mul(beta)
        .to_dict()
    )

    edge_to_config = edge_df.set_index("edge_id")["candidate_config_id"].to_dict()
    edge_regret = edge_df.set_index("edge_id")["regret_ms"].to_dict()

    # ------------------------------------------------------------
    # MILP
    # ------------------------------------------------------------
    prob = LpProblem("alpha_latency_regret_cover_normalized", LpMinimize)

    x = {
        c_id: LpVariable(f"x_{c_id}", cat=LpBinary)
        for c_id in candidate_configs["candidate_config_id"]
    }

    y = {e_id: LpVariable(f"y_{e_id}", cat=LpBinary) for e_id in edge_df["edge_id"]}

    # objective:
    #   normalized average autotuning burden + regret of chosen config per group
    prob += lpSum(
        float(measurement_cost_coef[c_id]) * x[c_id]
        for c_id in candidate_configs["candidate_config_id"]
    ) + lpSum(float(edge_regret[e_id]) * y[e_id] for e_id in edge_df["edge_id"])

    # each group chooses exactly one selected config
    for g_id in groups["group_id"]:
        prob += (
            lpSum(y[e_id] for e_id in group_to_edges[g_id]) == 1,
            f"assign_group_{g_id}",
        )

    # assignment only to selected configs
    for e_id in edge_df["edge_id"]:
        c_id = edge_to_config[e_id]
        prob += y[e_id] <= x[c_id], f"assign_implies_selected_{e_id}"

    # coverage: keep at least demand_per_group[g] selected alpha configs in each group
    for g_id in groups["group_id"]:
        prob += (
            lpSum(x[c_id] for c_id in group_to_configs[g_id]) >= demand_per_group[g_id],
            f"cover_group_{g_id}",
        )

    solver_kwargs = {"msg": msg}
    if time_limit_sec is not None:
        solver_kwargs["timeLimit"] = time_limit_sec

    status = prob.solve(PULP_CBC_CMD(**solver_kwargs))
    objective = value(prob.objective)
    if objective is not None:
        objective = float(objective)

    selected_config_ids = [
        c_id
        for c_id in candidate_configs["candidate_config_id"]
        if x[c_id].value() is not None and x[c_id].value() > 0.5
    ]

    selected_configs = (
        candidate_configs.loc[
            candidate_configs["candidate_config_id"].isin(selected_config_ids),
            config_cols,
        ]
        .drop_duplicates()
        .sort_values(config_cols, kind="mergesort")
        .reset_index(drop=True)
    )

    selected_edge_ids = [
        e_id
        for e_id in edge_df["edge_id"]
        if y[e_id].value() is not None and y[e_id].value() > 0.5
    ]

    selected_assignments = (
        edge_df.loc[edge_df["edge_id"].isin(selected_edge_ids)]
        .sort_values("group_id", kind="mergesort")
        .reset_index(drop=True)
    )

    total_assignment_regret_ms = selected_assignments["regret_ms"].sum()

    total_normalized_measurement_cost_ms = sum(
        measurement_cost_coef[c_id] / beta if beta != 0 else 0.0
        for c_id in selected_config_ids
    )

    return {
        "status": LpStatus[status],
        "objective": objective,
        "candidate_configs": candidate_configs,
        "selected_configs": selected_configs,
        "selected_assignments": selected_assignments,
        "edge_df": edge_df,
        "eligible_count_per_group": eligible_count_per_group,
        "demand_per_group": demand_per_group,
        "total_assignment_regret_ms": total_assignment_regret_ms,
        "total_normalized_measurement_cost_ms": total_normalized_measurement_cost_ms,
        "num_selected_configs": len(selected_configs),
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

CONFIG_COLS = infer_config_cols(raw_df)
LATENCY_COL = infer_latency_col(raw_df)

require_columns(raw_df, GROUP_COLS + [SCORE_COL, LATENCY_COL] + CONFIG_COLS)

print("Using latency column:", LATENCY_COL)
print("Using CONFIG_COLS:")
for c in CONFIG_COLS:
    print(f'"{c}"')

print("\nDevices:")
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

# ============================================================
# alpha candidates
# ============================================================
alpha_candidates = make_alpha_eligible(
    df=policy_df_source,
    score_col=SCORE_COL,
    alpha=ALPHA,
)

print(f"\nRows with score >= ALPHA: {len(alpha_candidates)}")

# ============================================================
# solve MILP
# ============================================================
result = solve_alpha_latency_regret_cover_normalized(
    source_df=policy_df_source,
    alpha_df=alpha_candidates,
    group_cols=GROUP_COLS,
    config_cols=CONFIG_COLS,
    latency_col=LATENCY_COL,
    beta=BETA,
    g=G,
    time_limit_sec=CBC_TIME_LIMIT_SEC,
    msg=CBC_MSG,
)

print(f"\nSolver status: {result['status']}")
print(f"Objective: {result['objective']}")
print(f"Selected configs: {result['num_selected_configs']}")
print(
    f"Total assignment regret (policy dataset): {result['total_assignment_regret_ms']:.6f} ms"
)
print(
    "Total normalized measurement cost "
    f"(policy dataset, before beta): {result['total_normalized_measurement_cost_ms']:.6f} ms"
)

# ============================================================
# bitmask / config policy
# ============================================================
config_universe = build_config_universe(policy_df_source, CONFIG_COLS)

config_policy, mask_bool, mask_u8, mask_packed = make_config_policy(
    config_universe=config_universe,
    surviving_configs=result["selected_configs"],
    config_cols=CONFIG_COLS,
)

# ============================================================
# apply policy to evaluation dataset
# ============================================================
eval_policy_df = apply_config_policy(eval_df, config_policy, CONFIG_COLS)
print("Total measurement cost ", eval_policy_df["median_latency_ms"].sum(), "ms")

print()
print(f"ALPHA : {ALPHA}")
print(f"G     : {G}")
print(f"BETA  : {BETA}")

print(
    f"\nSearch-Space : "
    f"{len(eval_df.index)} dispatches, "
    f"{eval_df.groupby(CONFIG_COLS).ngroups} unique configs, "
    f"max-used {eval_df.groupby(GROUP_COLS).size().max()}"
)
print(
    f"Pruned       : "
    f"{len(eval_policy_df.index)} dispatches, "
    f"{eval_policy_df.groupby(CONFIG_COLS).ngroups} unique configs, "
    f"max-used {eval_policy_df.groupby(GROUP_COLS).size().max()}"
)

# ============================================================
# histogram: original vs pruned only
# ============================================================
x_full = eval_df[SCORE_COL].to_numpy()
x_pruned = eval_policy_df[SCORE_COL].to_numpy()

if len(x_full) > 0 and len(x_pruned) > 0:
    min_v = min(x_full.min(), x_pruned.min())
    max_v = max(x_full.max(), x_pruned.max())
    bin_edges = np.linspace(min_v, max_v, BINS + 1)

    plt.figure(figsize=(10, 5))
    plt.hist(
        x_full,
        bins=bin_edges,
        alpha=0.4,
        label="full search space",
    )
    plt.hist(
        x_pruned,
        bins=bin_edges,
        alpha=0.4,
        label="pruned search space",
    )
    plt.xlabel(SCORE_COL)
    plt.ylabel("number of rows")
    plt.legend()
    plt.tight_layout()
    plt.savefig("plot_alpha_latency_regret_cover_normalized.pdf")
    plt.close()

# ============================================================
# optional saves
# ============================================================
config_universe.to_parquet("config_universe.parquet", index=False)
config_policy.to_parquet(
    "config_policy_alpha_latency_regret_cover_normalized.parquet", index=False
)
eval_policy_df.to_parquet(
    "eval_alpha_latency_regret_cover_normalized_df.parquet", index=False
)
result["selected_assignments"].to_parquet(
    "selected_assignments_alpha_latency_regret_cover_normalized.parquet",
    index=False,
)

np.save("alpha_latency_regret_cover_normalized_mask_bool.npy", mask_bool)
np.save("alpha_latency_regret_cover_normalized_mask_u8.npy", mask_u8)
with open("alpha_latency_regret_cover_normalized_mask.bin", "wb") as f:
    f.write(mask_packed.tobytes())

print("\nSaved files:")
print("- config_policy_alpha_latency_regret_cover_normalized.parquet")
print("- eval_alpha_latency_regret_cover_normalized_df.parquet")
print("- selected_assignments_alpha_latency_regret_cover_normalized.parquet")
print("- alpha_latency_regret_cover_normalized_mask_bool.npy")
print("- alpha_latency_regret_cover_normalized_mask_u8.npy")
print("- alpha_latency_regret_cover_normalized_mask.bin")
print("- plot_alpha_latency_regret_cover_normalized.pdf")
