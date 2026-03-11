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
# Keep these top-k configs per group unconditionally.
ANCHOR_TOP_K = 1

# Extra candidate rows are only considered if they pass this score threshold.
# This is only a pruning threshold for candidate generation.
ALPHA = 0.5

# Bank-size penalty in "milliseconds per selected config".
# Larger values -> fewer selected configs, more tolerated regret.
# Smaller values -> more selected configs, less regret.
CONFIG_COST_MS = 0.05

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
    # "NVIDIA GeForce RTX 4070",
    "NVIDIA GeForce RTX 4080 SUPER",
    # "Intel(R) Arc(tm) B580 Graphics (BMG G21)",
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
    raise RuntimeError(
        "Could not infer latency column. Expected one of: "
        f"{preferred}"
    )


def require_columns(df: pd.DataFrame, cols: list[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing required columns: {missing}")


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
        df_sorted.groupby(group_cols, group_keys=False, sort=False)
        .head(top_k)
        .copy()
    )

    topk["group_rank"] = (
        topk.groupby(group_cols)[score_col]
        .rank(method="first", ascending=False)
        .astype(int)
    )
    return topk


def make_alpha_candidates(
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


def solve_latency_regret_bank_selection(
    source_df: pd.DataFrame,
    anchor_df: pd.DataFrame,
    alpha_df: pd.DataFrame,
    group_cols: list[str],
    config_cols: list[str],
    latency_col: str,
    config_cost_ms: float,
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
    # candidate rows = anchors U alpha candidates
    # deduplicate by (group, config)
    # ------------------------------------------------------------
    candidate_rows = (
        pd.concat(
            [
                anchor_df[group_cols + config_cols + [latency_col, SCORE_COL]],
                alpha_df[group_cols + config_cols + [latency_col, SCORE_COL]],
            ],
            ignore_index=True,
        )
        .drop_duplicates(subset=group_cols + config_cols)
        .copy()
    )

    if candidate_rows.empty:
        raise RuntimeError("Candidate row set is empty.")

    # ------------------------------------------------------------
    # best latency per group from the full policy-construction dataset
    # ------------------------------------------------------------
    best_latency = (
        source_df.groupby(group_cols, dropna=False)[latency_col]
        .min()
        .reset_index(name="best_latency_ms")
    )

    # ------------------------------------------------------------
    # candidate config universe for the solver
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
    # candidate edges (group, config) with regret
    # ------------------------------------------------------------
    edge_df = (
        candidate_rows
        .merge(groups, on=group_cols, how="inner")
        .merge(candidate_configs, on=config_cols, how="inner")
        .merge(best_latency, on=group_cols, how="inner")
        .copy()
    )

    edge_df["regret_ms"] = edge_df[latency_col] - edge_df["best_latency_ms"]
    edge_df["regret_ms"] = edge_df["regret_ms"].clip(lower=0.0)

    edge_df = (
        edge_df[["group_id", "candidate_config_id", "regret_ms"]]
        .drop_duplicates()
        .sort_values(["group_id", "candidate_config_id"], kind="mergesort")
        .reset_index(drop=True)
    )
    edge_df["edge_id"] = np.arange(len(edge_df), dtype=np.int32)

    # ------------------------------------------------------------
    # mandatory anchor configs
    # ------------------------------------------------------------
    mandatory_configs = (
        anchor_df[config_cols]
        .drop_duplicates()
        .merge(candidate_configs, on=config_cols, how="inner")
        [["candidate_config_id"]]
        .drop_duplicates()
    )

    mandatory_config_ids = set(mandatory_configs["candidate_config_id"].tolist())

    # ------------------------------------------------------------
    # group -> edges and basic feasibility check
    # ------------------------------------------------------------
    group_to_edges = (
        edge_df.groupby("group_id")["edge_id"]
        .apply(list)
        .to_dict()
    )

    group_edge_counts = {
        g: len(group_to_edges.get(g, []))
        for g in groups["group_id"]
    }

    groups_without_candidates = [g for g, n in group_edge_counts.items() if n == 0]
    if groups_without_candidates:
        raise RuntimeError(
            f"Found groups with no candidate edges: {len(groups_without_candidates)}"
        )

    # ------------------------------------------------------------
    # MILP
    # ------------------------------------------------------------
    prob = LpProblem("latency_regret_bank_selection", LpMinimize)

    x = {
        c: LpVariable(f"x_{c}", cat=LpBinary)
        for c in candidate_configs["candidate_config_id"]
    }

    y = {
        e: LpVariable(f"y_{e}", cat=LpBinary)
        for e in edge_df["edge_id"]
    }

    # objective:
    # config bank size penalty + total absolute missed latency
    prob += (
        config_cost_ms * lpSum(x[c] for c in candidate_configs["candidate_config_id"])
        + lpSum(
            float(edge_df.loc[edge_df["edge_id"] == e, "regret_ms"].iloc[0]) * y[e]
            for e in edge_df["edge_id"]
        )
    )

    # every group assigned to exactly one selected config
    for g in groups["group_id"]:
        edges = group_to_edges[g]
        prob += lpSum(y[e] for e in edges) == 1, f"assign_group_{g}"

    # assignment only to selected configs
    edge_to_config = edge_df.set_index("edge_id")["candidate_config_id"].to_dict()
    for e in edge_df["edge_id"]:
        c = edge_to_config[e]
        prob += y[e] <= x[c], f"assign_implies_selected_{e}"

    # anchors are always selected
    for c in mandatory_config_ids:
        prob += x[c] == 1, f"mandatory_anchor_{c}"

    solver_kwargs = {"msg": msg}
    if time_limit_sec is not None:
        solver_kwargs["timeLimit"] = time_limit_sec

    status = prob.solve(PULP_CBC_CMD(**solver_kwargs))
    objective = value(prob.objective)
    if objective is not None:
        objective = float(objective)

    selected_config_ids = [
        c
        for c in candidate_configs["candidate_config_id"]
        if x[c].value() is not None and x[c].value() > 0.5
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
        e
        for e in edge_df["edge_id"]
        if y[e].value() is not None and y[e].value() > 0.5
    ]

    selected_assignments = (
        edge_df.loc[edge_df["edge_id"].isin(selected_edge_ids)]
        .sort_values("group_id", kind="mergesort")
        .reset_index(drop=True)
    )

    total_assignment_regret_ms = selected_assignments["regret_ms"].sum()

    return {
        "status": LpStatus[status],
        "objective": objective,
        "candidate_configs": candidate_configs,
        "selected_configs": selected_configs,
        "selected_assignments": selected_assignments,
        "mandatory_config_ids": mandatory_config_ids,
        "total_assignment_regret_ms": total_assignment_regret_ms,
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
# build anchor + alpha candidate sets
# ============================================================
anchor_topk = make_topk(
    df=policy_df_source,
    group_cols=GROUP_COLS,
    score_col=SCORE_COL,
    top_k=ANCHOR_TOP_K,
)

alpha_candidates = make_alpha_candidates(
    df=policy_df_source,
    score_col=SCORE_COL,
    alpha=ALPHA,
)

# ============================================================
# solve latency-regret-aware bank selection
# ============================================================
result = solve_latency_regret_bank_selection(
    source_df=policy_df_source,
    anchor_df=anchor_topk,
    alpha_df=alpha_candidates,
    group_cols=GROUP_COLS,
    config_cols=CONFIG_COLS,
    latency_col=LATENCY_COL,
    config_cost_ms=CONFIG_COST_MS,
    time_limit_sec=CBC_TIME_LIMIT_SEC,
    msg=CBC_MSG,
)

print(f"\nSolver status: {result['status']}")
print(f"Objective: {result['objective']}")
print(f"Total assignment regret (policy dataset): {result['total_assignment_regret_ms']:.6f} ms")
print(f"Selected configs: {result['num_selected_configs']}")

# ============================================================
# build policy bitmask / config policy
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

print()
print(f"ANCHOR_TOP_K   : {ANCHOR_TOP_K}")
print(f"ALPHA          : {ALPHA}")
print(f"CONFIG_COST_MS : {CONFIG_COST_MS}")

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
    plt.savefig("plot_latency_regret_policy.pdf")
    plt.close()

# ============================================================
# optional saves
# ============================================================
config_universe.to_parquet("config_universe.parquet", index=False)
config_policy.to_parquet("config_policy_latency_regret.parquet", index=False)
eval_policy_df.to_parquet("eval_latency_regret_policy_df.parquet", index=False)

np.save("latency_regret_mask_bool.npy", mask_bool)
np.save("latency_regret_mask_u8.npy", mask_u8)
with open("latency_regret_mask.bin", "wb") as f:
    f.write(mask_packed.tobytes())

print("\nSaved files:")
print("- config_policy_latency_regret.parquet")
print("- eval_latency_regret_policy_df.parquet")
print("- latency_regret_mask_bool.npy")
print("- latency_regret_mask_u8.npy")
print("- latency_regret_mask.bin")
print("- plot_latency_regret_policy.pdf")
