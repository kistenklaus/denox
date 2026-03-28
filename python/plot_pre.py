from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
from collections import defaultdict
import re
import pandas as pd

# replace this import with your actual module
# e.g. from my_policy_module import weighted_uniform_topn_policy as milp_policy
from milp_policy import milp_policy


def apply_milp(
    dfglob: str,
    dir: str = "./parquets/",
    alpha=100000.0,
    beta=1.0,
    score_threshold=0.85,
    g=10,
    dfnot: str | None = None,
) -> pd.DataFrame:
    dfs: list[pd.DataFrame] = []

    for path in Path(dir).glob(dfglob):
        if dfnot is not None and path.name.startswith(dfnot):
            continue
        dfs.append(pd.read_parquet(path))

    if not dfs:
        raise ValueError(f"No parquet files matched {dfglob!r} in {dir!r}")

    dataset = pd.concat(dfs, ignore_index=True)

    for device in dataset["device"].unique():
        print(device)

    DEVICES = [
        "Intel(R) Arc(tm) B580 Graphics (BMG G21)",
        "AMD Radeon RX 7900 XTX (RADV NAVI31)",
        # "NVIDIA GeForce RTX 4080 SUPER",
        # "NVIDIA GeForce RTX 4060 Ti",
        "NVIDIA GeForce RTX 2080 Ti",
        # "NVIDIA GeForce RTX 4070",
        "NVIDIA GeForce RTX 2070",
    ]

    # Run MILP only on the filtered subset.
    filtered_input = dataset.loc[dataset["device"].isin(DEVICES)].copy()

    if filtered_input.empty:
        raise ValueError("Filtered input is empty; no rows matched the inline device filter.")

    milp_filtered = milp_policy(
        filtered_input,
        score_threshold=score_threshold,
        g=g,
        alpha=alpha,
        beta=beta,
    )

    config_parms = sorted(c for c in dataset.columns if c.startswith("config_"))
    if not config_parms:
        raise ValueError("No config_* columns found in dataset.")

    # Extract the selected configuration set from the filtered MILP result...
    selected_configs = milp_filtered[config_parms].drop_duplicates()

    # ...and apply it back to the full original dataset.
    milp = dataset.merge(selected_configs, on=config_parms, how="inner")

    dataset_config_space = dataset.groupby(config_parms).ngroups
    milp_config_space = milp.groupby(config_parms).ngroups

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

    # Sanity check: every config selected by MILP should exist in the full dataset.
    missing_configs = selected_configs.merge(
        dataset[config_parms].drop_duplicates(),
        on=config_parms,
        how="left",
        indicator=True,
    )
    missing_configs = missing_configs[missing_configs["_merge"] == "left_only"]
    assert len(missing_configs) == 0

    best = (
        dataset.groupby(GROUP_COLS, as_index=False)["median_latency_ms"]
        .min()
        .rename(columns={"median_latency_ms": "median_latency_ms_full"})
    )

    milp_best = (
        milp.groupby(GROUP_COLS, as_index=False)["median_latency_ms"]
        .min()
        .rename(columns={"median_latency_ms": "median_latency_ms_milp"})
    )

    configs_per_group = (
        dataset.drop_duplicates(subset=GROUP_COLS + config_parms)
        .groupby(GROUP_COLS)
        .size()
    )

    milp_configs_per_group = (
        milp.drop_duplicates(subset=GROUP_COLS + config_parms)
        .groupby(GROUP_COLS)
        .size()
    )

    mean_configs = configs_per_group.mean()
    median_configs = configs_per_group.median()
    max_configs = configs_per_group.max()
    min_configs = configs_per_group.min()

    milp_mean_configs = milp_configs_per_group.mean()
    milp_median_configs = milp_configs_per_group.median()
    milp_max_configs = milp_configs_per_group.max()
    milp_min_configs = milp_configs_per_group.min()

    merged = best.merge(milp_best, on=GROUP_COLS, how="inner")
    merged["regret_ms"] = (
        merged["median_latency_ms_milp"] - merged["median_latency_ms_full"]
    )

    total_regret = merged["regret_ms"].sum()
    mean_regret = merged["regret_ms"].mean()
    median_regret = merged["regret_ms"].median()
    max_regret = merged["regret_ms"].max()

    print(f"Summary: {dfglob}")
    print(f"- Search Space: {dataset_config_space} -> {milp_config_space}")
    print(f" * Mean   : {mean_configs} -> {milp_mean_configs}")
    print(f" * Median : {median_configs} -> {milp_median_configs}")
    print(f" * Min    : {min_configs} -> {milp_min_configs}")
    print(f" * Max    : {max_configs} -> {milp_max_configs}")

    print("- Regret")
    print(f" * Total  : {total_regret}ms")
    print(f" * Mean   : {mean_regret}ms")
    print(f" * Median : {median_regret}ms")
    print(f" * Max    : {max_regret}ms")

    print(len(milp.index))

    for device in milp["device"].unique():
        print(device)

    return milp

direct_conv_cm_milp = apply_milp(
    "direct-conv-cm-*",
    dir="./repaired/",
    alpha=1000,
    g=5,
    score_threshold=0.9,
)

direct_conv_cm_milp.to_parquet("parquets/direct-conv-cm-milp.parquet")
