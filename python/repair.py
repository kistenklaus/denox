from pathlib import Path
import matplotlib.pyplot as plt
from collections import defaultdict
import re
import pandas as pd

def repair_force_sg_size(df: pd.DataFrame) -> pd.DataFrame:
    nvidia_mask = df["device"].str.contains("NVIDIA", case=False, na=False)
    amd_mask = df["device"].str.contains("AMD", case=False, na=False)
    intel_mask = df["device"].str.contains("Intel", case=False, na=False)

    # NVIDIA: replace 0 or NaN with 32
    invalid_nv = df.loc[nvidia_mask, "subgroup_size"].isna() | (
        df.loc[nvidia_mask, "subgroup_size"] == 0
    )
    df.loc[nvidia_mask & invalid_nv, "subgroup_size"] = 32

    # AMD/Intel: ensure subgroup_size is valid
    mask = amd_mask | intel_mask
    invalid = df.loc[mask, "subgroup_size"].isna() | (
        df.loc[mask, "subgroup_size"] == 0
    )

    if invalid.any():
        raise ValueError("Invalid subgroup_size for AMD/Intel")

    df["config_SG_SIZE"] = df["subgroup_size"]

    return df

repaired_dir = Path("./repaired")
repaired_dir.mkdir(exist_ok=True)


dfnot = None
dfs: list[pd.DataFrame] = []
for path in list(Path("./parquets/").glob("concat-conv-cm*")):
    df = repair_force_sg_size(pd.read_parquet(path))
    df.to_parquet(Path("./repaired/") / path.name, index=False)


dfs: list[pd.DataFrame] = []
for path in list(Path("./parquets/").glob("direct-conv-*")):
    if path.name.startswith("direct-conv-cm"):
        continue
    df = repair_force_sg_size(pd.read_parquet(path))
    df.to_parquet(Path("./repaired/") / path.name, index=False)

dfs: list[pd.DataFrame] = []
for path in list(Path("./parquets/").glob("direct-conv-cm-*")):
    df = repair_force_sg_size(pd.read_parquet(path))
    df.to_parquet(Path("./repaired/") / path.name, index=False)


def repair_WG_H(df: pd.DataFrame) -> pd.DataFrame:
    assert (df["config_WG_H"].isna() | (df["config_WG_H"] == 0)).all(), (
        "config_WG_H contains unexpected values"
    )
    df["config_WG_H"] = 1
    return df


dfs: list[pd.DataFrame] = []
for path in list(Path("./parquets/").glob("memory-slice-*")):
    df = repair_WG_H(pd.read_parquet(path))
    df.to_parquet(Path("./repaired/") / path.name, index=False)

dfs: list[pd.DataFrame] = []
for path in list(Path("./parquets/").glob("memory-pad-*")):
    df = repair_WG_H(pd.read_parquet(path))
    df.to_parquet(Path("./repaired/") / path.name, index=False)
