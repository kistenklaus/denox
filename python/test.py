import pandas as pd
from pathlib import Path
import numpy as np

# ============================================================
# load data
# ============================================================
PARQUET_GLOB = "direct-conv-cm-*"

DEVICES = [
    # "NVIDIA GeForce RTX 4070",
    "NVIDIA GeForce RTX 3080 Ti",
    # "NVIDIA GeForce RTX 4080 SUPER",
    # "Intel(R) Arc(tm) B580 Graphics (BMG G21)",
]
# DEVICES = None

dfs: list[pd.DataFrame] = []
for path in Path("./parquets/").glob(PARQUET_GLOB):
    dfs.append(pd.read_parquet(path))

if not dfs:
    raise RuntimeError(f"No parquet files matched ./parquets/{PARQUET_GLOB}")

df = pd.concat(dfs, ignore_index=True)

print("Columns:")
print(df.columns.tolist())

print("\nDevices:")
for device in sorted(df["device"].dropna().unique()):
    print(f'"{device}"')

if DEVICES is not None:
    df = df.loc[df["device"].isin(DEVICES)].copy()

if df.empty:
    raise RuntimeError("Dataframe is empty after filtering.")

# ============================================================
# choose column names present in parquet
# adjust these if your parquet uses different names
# ============================================================
MEAN_COL = "mean_latency_ms"
MEDIAN_COL = "median_latency_ms"
STD_COL = "std_latency_ms"
COUNT_COL = "sample_count"

missing = [c for c in [MEAN_COL, MEDIAN_COL, STD_COL, COUNT_COL] if c not in df.columns]
if missing:
    raise RuntimeError(f"Missing expected columns: {missing}")

# relative jitter per aggregated row
df["cv_latency"] = df[STD_COL] / df[MEAN_COL].replace(0, np.nan)

# ============================================================
# helper for weighted mean
# ============================================================
def weighted_mean(x: pd.Series, w: pd.Series) -> float:
    mask = x.notna() & w.notna()
    x = x[mask]
    w = w[mask]
    if len(x) == 0:
        return np.nan
    s = w.sum()
    if s == 0:
        return np.nan
    return (x * w).sum() / s

# ============================================================
# per-device summary over aggregated rows
# ============================================================
device_summary = (
    df.groupby("device", dropna=False)
    .apply(
        lambda g: pd.Series({
            "group_count": len(g),
            "total_samples": g[COUNT_COL].sum(),

            # mean-latency summaries across aggregated rows
            "min_mean_latency_ms": g[MEAN_COL].min(),
            "max_mean_latency_ms": g[MEAN_COL].max(),
            "mean_of_mean_latency_ms": g[MEAN_COL].mean(),
            "weighted_mean_latency_ms": weighted_mean(g[MEAN_COL], g[COUNT_COL]),
            "median_of_mean_latency_ms": g[MEAN_COL].median(),

            # std summaries across aggregated rows
            "min_std_latency_ms": g[STD_COL].min(),
            "max_std_latency_ms": g[STD_COL].max(),
            "mean_std_latency_ms": g[STD_COL].mean(),
            "weighted_mean_std_latency_ms": weighted_mean(g[STD_COL], g[COUNT_COL]),
            "median_std_latency_ms": g[STD_COL].median(),

            # relative variability
            "mean_cv_latency": g["cv_latency"].mean(),
            "weighted_mean_cv_latency": weighted_mean(g["cv_latency"], g[COUNT_COL]),
            "median_cv_latency": g["cv_latency"].median(),
            "p95_cv_latency": g["cv_latency"].quantile(0.95),
        })
    )
    .reset_index()
)

with pd.option_context("display.max_rows", None, "display.max_columns", None):
    print("\nPer-device summary:")
    print(device_summary.sort_values("weighted_mean_cv_latency", ascending=False))
