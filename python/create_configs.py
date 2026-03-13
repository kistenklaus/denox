from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
from collections import defaultdict
import re
import pandas as pd

# replace this import with your actual module
# e.g. from my_policy_module import weighted_uniform_topn_policy as milp_policy
from milp_policy import milp_policy

Path("plots").mkdir(exist_ok=True)


def apply_milp(
    dfglob: str,
    alpha=100000.0,
    beta=1.0,
    score_threshold=0.85,
    g=10,
    dfnot: str | None = None,
    plot: str | None = None,
):
    dfs: list[pd.DataFrame] = []
    for path in list(Path("./parquets/").glob(dfglob)):
        if dfnot is not None and path.name.startswith(dfnot):
            continue
        dfs.append(pd.read_parquet(path))
    dataset = pd.concat(dfs)
    milp = milp_policy(
        dataset, score_threshold=score_threshold, g=g, alpha=alpha, beta=beta
    )
    config_parms = sorted(c for c in dataset.columns if c.startswith("config_"))
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

    missing = milp.merge(dataset, how="left", indicator=True)
    missing = missing[missing["_merge"] == "left_only"]
    assert len(missing) == 0

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

    merged = best.merge(milp_best, on=GROUP_COLS, suffixes=("_full", "_milp"))
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

    if plot is not None:
        BINS = 100
        plt.figure(figsize=(10, 5))
        plt.xlim(0, 1)
        plt.hist(
            dataset["relative_speedup"],
            bins=BINS,
            alpha=0.4,
            label=f"{dfglob} full config space",
        )
        plt.hist(
            milp["relative_speedup"],
            bins=BINS,
            alpha=0.4,
            label=f"{dfglob} milp config space",
        )
        plt.xlabel("relative-speedup")
        plt.ylabel("number of configs")
        plt.grid()
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot)
        plt.close()
        return milp


def find_repo_root(start: Path | None = None) -> Path:
    path = (start or Path.cwd()).resolve()

    for parent in [path] + list(path.parents):
        if (parent / ".git").exists():
            return parent

    raise RuntimeError("Not inside a git repository")


def generate_config_file(df, config_path, params: list[str]):
    config_combinations = (
        df[params].drop_duplicates().sort_values(by=params).reset_index(drop=True)
    )
    bool_cols = config_combinations.select_dtypes(include="bool").columns
    config_combinations[bool_cols] = config_combinations[bool_cols].astype(int)
    config_combinations.to_csv(config_path, index=False, header=False, sep=' ')


repo_root = find_repo_root()


basic_activation_milp = apply_milp(
    "basic-activation-*",
    alpha=10,
    g=1,
    plot="plots/basic_activation_milp.pdf",
    score_threshold=0.80,
)
generate_config_file(
    basic_activation_milp,
    repo_root
    / "compiler/src/denox/compiler/implement/shaders/activation/basic_activation.configs",
    [
        "config_INVOC_C",
        "config_INVOC_W",
        "config_INVOC_H",
        "config_WG_C",
        "config_WG_W",
        "config_WG_H",
    ],
)

direct_conv_cm_milp = apply_milp(
    "direct-conv-cm-*",
    alpha=1000,
    g=5,
    plot="plots/direct_conv_cm_milp.pdf",
    score_threshold=0.9,
)
generate_config_file(
    direct_conv_cm_milp,
    repo_root
    / "compiler/src/denox/compiler/implement/shaders/conv/direct_conv_cm.configs",
    [
        "config_CM_M",
        "config_CM_K",
        "config_CM_N",
        "config_SG_M",
        "config_SG_K",
        "config_SG_N",
        "config_WG_M",
        "config_WG_N",
        "config_ASYNC",
    ],
)

direct_conv_milp = apply_milp(
    "direct-conv-*",
    dfnot="direct-conv-cm",
    alpha=1000,
    g=5,
    plot="plots/direct_conv_milp.pdf",
    score_threshold=0.9,
)
generate_config_file(
    direct_conv_milp,
    repo_root
    / "compiler/src/denox/compiler/implement/shaders/conv/direct_conv.configs",
    [
        "config_INVOC_M",
        "config_INVOC_K",
        "config_INVOC_N",
        "config_SG_M",
        "config_SG_K",
        "config_SG_N",
        "config_WG_M",
        "config_WG_N",
        "config_ASYNC",
    ],
)

concat_conv_cm = apply_milp(
    "concat-conv-cm*",
    alpha=1000,
    g=10,
    plot="plots/concat_conv_cm_milp.pdf",
    score_threshold=0.9,
)
generate_config_file(
    concat_conv_cm,
    repo_root
    / "compiler/src/denox/compiler/implement/shaders/conv/concat_conv_cm.configs",
    [
        "config_CM_M",
        "config_A_CM_K",
        "config_B_CM_K",
        "config_CM_N",
        "config_SG_M",
        "config_A_SG_K",
        "config_B_SG_K",
        "config_SG_N",
        "config_WG_M",
        "config_WG_N",
        "config_A_ASYNC",
        "config_B_ASYNC",
    ],
)

basic_pool_milp = apply_milp(
    "basic-pool*",
    alpha=10,
    g=1,
    plot="plots/basic_pool_milp.pdf",
    score_threshold=0.9,
)
generate_config_file(
    basic_pool_milp,
    repo_root / "compiler/src/denox/compiler/implement/shaders/pool/basic_pool.configs",
    [
        "config_INVOC_C",
        "config_INVOC_W",
        "config_INVOC_H",
        "config_WG_C",
        "config_WG_W",
        "config_WG_H",
    ],
)

basic_upsample_milp = apply_milp(
    "basic-upsample*",
    alpha=10,
    g=1,
    plot="plots/basic_upsample_milp.pdf",
    score_threshold=0.9,
)
generate_config_file(
    basic_upsample_milp,
    repo_root
    / "compiler/src/denox/compiler/implement/shaders/upsample/upsample_upsample.configs",
    [
        "config_INVOC_C",
        "config_INVOC_W",
        "config_INVOC_H",
        "config_WG_C",
        "config_WG_W",
        "config_WG_H",
    ],
)

copy_transform_milp = apply_milp(
    "copy-transform-*",
    alpha=10000,
    g=1,
    plot="plots/copy_transform_milp.pdf",
    score_threshold=0.9,
)
generate_config_file(
    copy_transform_milp,
    repo_root
    / "compiler/src/denox/compiler/implement/shaders/copy/copy_transform.configs",
    [
        "config_INVOC_C",
        "config_INVOC_W",
        "config_INVOC_H",
        "config_WG_C",
        "config_WG_W",
        "config_WG_H",
    ],
)


memory_slice_milp = apply_milp(
    "memory-slice-*",
    alpha=10,
    g=10,
    plot="plots/memory_slice_milp.pdf",
    score_threshold=0.9,
)
params = [
    "config_INVOC_C",
    "config_INVOC_W",
    "config_INVOC_H",
    "config_WG_C",
    "config_WG_W",
    # "config_WG_H",
]
config_combinations = (
    memory_slice_milp[params].drop_duplicates().sort_values(by=params).reset_index(drop=True)
)
config_combinations["config_WG_H"] = 1
bool_cols = config_combinations.select_dtypes(include="bool").columns
config_combinations[bool_cols] = config_combinations[bool_cols].astype(int)
config_combinations.to_csv(repo_root / "compiler/src/denox/compiler/implement/shaders/slice/memory_slice.configs", index=False, header=False)


memory_pad_milp = apply_milp(
    "memory-pad-*",
    alpha=10,
    g=1,
    plot="plots/memory_pad_milp.pdf",
    score_threshold=0.9,
)
params = [
    "config_INVOC_C",
    "config_INVOC_W",
    "config_INVOC_H",
    "config_WG_C",
    "config_WG_W",
    # "config_WG_H",
]
config_combinations = (
    memory_pad_milp[params].drop_duplicates().sort_values(by=params).reset_index(drop=True)
)
config_combinations["config_WG_H"] = 1
bool_cols = config_combinations.select_dtypes(include="bool").columns
config_combinations[bool_cols] = config_combinations[bool_cols].astype(int)
config_combinations.to_csv(repo_root / "compiler/src/denox/compiler/implement/shaders/pad/memory_pad.configs", index=False, header=False)
