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

# AI Generated beauty <3 

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

CBC_TIME_LIMIT_SEC = 120
CBC_MSG = False


def _infer_config_cols(df: pd.DataFrame) -> list[str]:
    config_cols = sorted(c for c in df.columns if c.startswith("config_"))
    if not config_cols:
        raise RuntimeError("No config_* columns found in dataframe.")
    return config_cols


def _infer_latency_col(df: pd.DataFrame) -> str:
    preferred = ["latency_ms", "mean_latency_ms", "median_latency_ms"]
    for c in preferred:
        if c in df.columns:
            return c
    raise RuntimeError(
        f"Could not infer latency column. Expected one of: {preferred}"
    )


def _require_columns(df: pd.DataFrame, cols: list[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing required columns: {missing}")


def _make_eligible(
    df: pd.DataFrame,
    score_col: str,
    score_threshold: float,
) -> pd.DataFrame:
    return df.loc[df[score_col] >= score_threshold].copy()


def _build_config_universe(
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


def _make_config_policy(
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


def _apply_config_policy(
    df: pd.DataFrame,
    config_policy: pd.DataFrame,
    config_cols: list[str],
) -> pd.DataFrame:
    return df.merge(
        config_policy.loc[config_policy["keep"], config_cols],
        on=config_cols,
        how="inner",
    )


def _compute_device_weights(
    source_df: pd.DataFrame,
    group_cols: list[str],
    latency_col: str,
) -> tuple[pd.DataFrame, dict[str, float]]:
    best_latency = (
        source_df.groupby(group_cols, dropna=False)[latency_col]
        .min()
        .reset_index(name="best_latency_ms")
    )

    if "device" not in best_latency.columns:
        raise RuntimeError("GROUP_COLS must include 'device' for device normalization.")

    device_scale_df = (
        best_latency.groupby("device", dropna=False)["best_latency_ms"]
        .median()
        .reset_index(name="device_scale_ms")
    )

    if (device_scale_df["device_scale_ms"] <= 0).any():
        raise RuntimeError("Non-positive device scale encountered; cannot normalize.")

    device_weights = {
        row["device"]: 1.0 / row["device_scale_ms"]
        for _, row in device_scale_df.iterrows()
    }

    return device_scale_df, device_weights


def _solve_weighted_uniform_topn_regret_cover(
    source_df: pd.DataFrame,
    eligible_df: pd.DataFrame,
    group_cols: list[str],
    config_cols: list[str],
    latency_col: str,
    alpha_weight: float,
    beta_weight: float,
    g: int,
    device_weights: dict[str, float],
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

    candidate_rows = (
        eligible_df[group_cols + config_cols + [latency_col, SCORE_COL]]
        .sort_values(latency_col, kind="mergesort")
        .drop_duplicates(subset=group_cols + config_cols, keep="first")
        .reset_index(drop=True)
    )

    if candidate_rows.empty:
        raise RuntimeError("Eligible candidate set is empty. Lower score_threshold.")

    best_latency = (
        source_df.groupby(group_cols, dropna=False)[latency_col]
        .min()
        .reset_index(name="best_latency_ms")
    )

    candidate_configs = (
        candidate_rows[config_cols]
        .drop_duplicates()
        .sort_values(config_cols, kind="mergesort")
        .reset_index(drop=True)
    )
    candidate_configs["candidate_config_id"] = np.arange(
        len(candidate_configs), dtype=np.int32
    )

    edge_df = (
        candidate_rows.merge(groups, on=group_cols, how="inner")
        .merge(candidate_configs, on=config_cols, how="inner")
        .merge(best_latency, on=group_cols, how="inner")
        .copy()
    )

    if "device" not in edge_df.columns:
        raise RuntimeError("GROUP_COLS must include 'device'.")

    edge_df["device_weight"] = edge_df["device"].map(device_weights)
    if edge_df["device_weight"].isna().any():
        missing_devices = sorted(
            edge_df.loc[edge_df["device_weight"].isna(), "device"].unique()
        )
        raise RuntimeError(f"Missing device weights for devices: {missing_devices}")

    edge_df["regret_ms"] = edge_df[latency_col] - edge_df["best_latency_ms"]
    edge_df["regret_ms"] = edge_df["regret_ms"].clip(lower=0.0)

    edge_df = (
        edge_df[
            [
                "group_id",
                "candidate_config_id",
                "device",
                "device_weight",
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

    group_to_configs = (
        edge_df.groupby("group_id")["candidate_config_id"]
        .apply(lambda s: sorted(set(s)))
        .to_dict()
    )
    group_to_edges = edge_df.groupby("group_id")["edge_id"].apply(list).to_dict()

    eligible_count_per_group = {
        g_id: len(group_to_configs.get(g_id, []))
        for g_id in groups["group_id"]
    }

    groups_without_candidates = [
        g_id for g_id, n in eligible_count_per_group.items() if n == 0
    ]
    if groups_without_candidates:
        raise RuntimeError(
            f"Found groups with zero eligible configs: {len(groups_without_candidates)}. "
            "Lower score_threshold or inspect the dataset."
        )

    n_per_group = {
        g_id: min(g, eligible_count_per_group[g_id])
        for g_id in groups["group_id"]
    }

    edge_to_config = edge_df.set_index("edge_id")["candidate_config_id"].to_dict()
    edge_regret = edge_df.set_index("edge_id")["regret_ms"].to_dict()
    edge_group = edge_df.set_index("edge_id")["group_id"].to_dict()
    edge_weight = edge_df.set_index("edge_id")["device_weight"].to_dict()

    edge_df["weighted_latency_for_x"] = edge_df["device_weight"] * edge_df[latency_col]
    tuning_cost_coef = (
        edge_df.groupby("candidate_config_id")["weighted_latency_for_x"]
        .sum()
        .mul(beta_weight)
        .to_dict()
    )

    prob = LpProblem("weighted_uniform_topn_regret_cover", LpMinimize)

    x = {
        c_id: LpVariable(f"x_{c_id}", cat=LpBinary)
        for c_id in candidate_configs["candidate_config_id"]
    }

    y = {
        e_id: LpVariable(f"y_{e_id}", cat=LpBinary)
        for e_id in edge_df["edge_id"]
    }

    prob += lpSum(
        float(tuning_cost_coef[c_id]) * x[c_id]
        for c_id in candidate_configs["candidate_config_id"]
    ) + lpSum(
        float(alpha_weight)
        * (edge_weight[e_id] / n_per_group[edge_group[e_id]])
        * edge_regret[e_id]
        * y[e_id]
        for e_id in edge_df["edge_id"]
    )

    for g_id in groups["group_id"]:
        prob += (
            lpSum(y[e_id] for e_id in group_to_edges[g_id]) == n_per_group[g_id],
            f"count_group_{g_id}",
        )

    for e_id in edge_df["edge_id"]:
        c_id = edge_to_config[e_id]
        prob += y[e_id] <= x[c_id], f"count_implies_selected_{e_id}"

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

    selected_edge_df = (
        edge_df.loc[edge_df["edge_id"].isin(selected_edge_ids)]
        .sort_values(["group_id", "regret_ms"], kind="mergesort")
        .reset_index(drop=True)
    )

    total_weighted_regret_term = 0.0
    for g_id, gdf in selected_edge_df.groupby("group_id", sort=False):
        total_weighted_regret_term += (
            gdf["device_weight"].iloc[0] * gdf["regret_ms"].sum() / n_per_group[g_id]
        )

    old_total_weighted_regret_term = (
        selected_edge_df.groupby("group_id")
        .apply(lambda gdf: gdf["device_weight"].iloc[0] * gdf["regret_ms"].min())
        .sum()
    )

    total_weighted_tuning_term = sum(
        tuning_cost_coef[c_id] / beta_weight if beta_weight != 0 else 0.0
        for c_id in selected_config_ids
    )

    return {
        "status": LpStatus[status],
        "objective": objective,
        "selected_configs": selected_configs,
        "selected_edges": selected_edge_df,
        "eligible_count_per_group": eligible_count_per_group,
        "n_per_group": n_per_group,
        "total_weighted_tuning_term_ms": total_weighted_tuning_term,
        "total_weighted_regret_term_ms": total_weighted_regret_term,
        "old_total_weighted_regret_term_ms": float(old_total_weighted_regret_term),
        "num_selected_configs": len(selected_configs),
    }


def milp_policy(
    df: pd.DataFrame,
    score_threshold: float = 0.5,
    g: int = 10,
    alpha: float = 1.0,
    beta: float = 100000.0,
) -> pd.DataFrame:
    """
    Learn a global config bank using the weighted uniform-top-n MILP policy and
    return the original dataframe filtered to the selected configs.

    Parameters
    ----------
    df:
        Input benchmark dataframe.
    score_threshold:
        Eligibility threshold on SCORE_COL. Only rows with score >= score_threshold
        participate in the optimization.
    g:
        For each group t, the optimizer keeps n_t = min(g, |E_t|) counted configs.
    alpha_weight:
        Weight of the weighted regret term.
    beta_weight:
        Weight of the weighted autotuning-cost term.

    Returns
    -------
    pd.DataFrame
        The original dataframe filtered to the selected global config bank.
    """
    if df.empty:
        return df.copy()

    if g <= 0:
        raise ValueError("g must be positive.")
    if not (0.0 <= score_threshold <= 1.0):
        raise ValueError("score_threshold must be between 0 and 1.")
    if alpha < 0:
        raise ValueError("alpha_weight must be non-negative.")
    if beta < 0:
        raise ValueError("beta_weight must be non-negative.")

    config_cols = _infer_config_cols(df)
    latency_col = _infer_latency_col(df)

    _require_columns(df, GROUP_COLS + [SCORE_COL, latency_col] + config_cols)

    eligible_df = _make_eligible(
        df=df,
        score_col=SCORE_COL,
        score_threshold=score_threshold,
    )

    if eligible_df.empty:
        return df.iloc[0:0].copy()

    _, device_weights = _compute_device_weights(
        source_df=df,
        group_cols=GROUP_COLS,
        latency_col=latency_col,
    )

    result = _solve_weighted_uniform_topn_regret_cover(
        source_df=df,
        eligible_df=eligible_df,
        group_cols=GROUP_COLS,
        config_cols=config_cols,
        latency_col=latency_col,
        alpha_weight=alpha,
        beta_weight=beta,
        g=g,
        device_weights=device_weights,
        time_limit_sec=CBC_TIME_LIMIT_SEC,
        msg=CBC_MSG,
    )

    config_universe = _build_config_universe(df, config_cols)

    config_policy, _, _, _ = _make_config_policy(
        config_universe=config_universe,
        surviving_configs=result["selected_configs"],
        config_cols=config_cols,
    )

    return _apply_config_policy(df, config_policy, config_cols)
