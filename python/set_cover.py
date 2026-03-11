from __future__ import annotations

import numpy as np
import pandas as pd

from pulp import (
    LpBinary,
    LpMinimize,
    LpProblem,
    LpStatus,
    LpVariable,
    PULP_CBC_CMD,
    lpSum,
    value,
)

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

CBC_TIME_LIMIT_SEC = None
CBC_MSG = False


def _infer_config_cols(df: pd.DataFrame) -> list[str]:
    config_cols = sorted(c for c in df.columns if c.startswith("config_"))
    if not config_cols:
        raise RuntimeError("No config_* columns found in dataframe.")
    return config_cols


def _require_columns(df: pd.DataFrame, cols: list[str]) -> None:
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
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
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
    groups = (
        source_df[group_cols]
        .drop_duplicates()
        .sort_values(group_cols, kind="mergesort")
        .reset_index(drop=True)
    )
    groups["group_id"] = np.arange(len(groups), dtype=np.int32)

    mandatory_configs = (
        anchor_df[config_cols]
        .drop_duplicates()
        .sort_values(config_cols, kind="mergesort")
        .reset_index(drop=True)
    )

    if mandatory_configs.empty:
        raise RuntimeError("Mandatory anchor config set is empty.")

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
        .merge(mandatory_configs, on=config_cols, how="inner")[["group_id"] + config_cols]
        .drop_duplicates()
    )

    base_relation = pd.concat(
        [anchor_relation, alpha_mandatory_relation], ignore_index=True
    ).drop_duplicates()

    base_count_per_group = base_relation.groupby("group_id").size().to_dict()

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
    augment_configs["augment_config_id"] = np.arange(len(augment_configs), dtype=np.int32)

    augment_relation = (
        alpha_df[group_cols + config_cols]
        .drop_duplicates()
        .merge(groups, on=group_cols, how="inner")
        .merge(augment_configs, on=config_cols, how="inner")[["group_id", "augment_config_id"]]
        .drop_duplicates()
        .sort_values(["group_id", "augment_config_id"], kind="mergesort")
        .reset_index(drop=True)
    )

    group_to_augment = (
        augment_relation.groupby("group_id")["augment_config_id"].apply(list).to_dict()
    )

    augment_count_per_group = {
        g: len(group_to_augment.get(g, []))
        for g in groups["group_id"]
    }

    residual_demand = {}
    feasible_residual_demand = {}

    for g in groups["group_id"]:
        base_count = base_count_per_group.get(g, 0)
        residual = max(0, target_g - base_count)
        feasible = min(residual, augment_count_per_group[g])

        residual_demand[g] = residual
        feasible_residual_demand[g] = feasible

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


def set_cover_policy(df: pd.DataFrame, alpha=0.9, g=10, k=1) -> pd.DataFrame:
    """
    Apply the anchor-plus-alpha set-cover policy to `df`.

    Semantics:
    - For each group (GROUP_COLS), the top-k configs by SCORE_COL are always kept.
    - Additional configs with SCORE_COL >= alpha are selected globally so that each group
      reaches up to g selected acceptable configs if feasible.
    - The returned dataframe is the original dataframe filtered to the selected configs.

    Parameters
    ----------
    df:
        Input dataframe containing benchmark rows.
    alpha:
        Alpha threshold on relative_speedup for alpha-eligible configs.
    g:
        Target number of selected acceptable configs per group.
    k:
        Number of top-k anchor configs that are always kept per group.

    Returns
    -------
    pd.DataFrame
        The filtered dataframe after applying the learned config policy.
    """
    if df.empty:
        return df.copy()

    config_cols = _infer_config_cols(df)
    _require_columns(df, GROUP_COLS + LOGICAL_OP_COLS + [SCORE_COL] + config_cols)

    if k <= 0:
        raise ValueError("k must be positive.")
    if g <= 0:
        raise ValueError("g must be positive.")
    if not (0.0 <= alpha <= 1.0):
        raise ValueError("alpha must be between 0 and 1.")

    anchor_topk = make_topk(
        df=df,
        group_cols=GROUP_COLS,
        score_col=SCORE_COL,
        top_k=k,
    )

    alpha_eligible = make_alpha_eligible(
        df=df,
        score_col=SCORE_COL,
        alpha=alpha,
    )

    completion_result = solve_anchor_plus_alpha_completion(
        source_df=df,
        anchor_df=anchor_topk,
        alpha_df=alpha_eligible,
        group_cols=GROUP_COLS,
        config_cols=config_cols,
        target_g=g,
        time_limit_sec=CBC_TIME_LIMIT_SEC,
        msg=CBC_MSG,
    )

    anchored_alpha_config_policy, _, _, _ = make_config_policy(
        config_universe=build_config_universe(df, config_cols),
        surviving_configs=completion_result["selected_configs"],
        config_cols=config_cols,
    )

    return apply_config_policy(df, anchored_alpha_config_policy, config_cols)
