import pandas as pd
from pandas._typing import DtypeArg
from pathlib import Path

DENOX_DB_DTYPES: DtypeArg = {
    "operation": "category",
    "input_shape": "category",
    "input_format": "category",
    "input_type": "category",
    "output_shape": "category",
    "output_format": "category",
    "output_type": "category",
    "shader": "category",
    "config": "category",
    "spirv_size": "int64",
    "spirv_hash": "string",
    "src_hash": "string",
    "coopmat": "bool",
    "parameter_size": "int64",
    "descriptor_count": "int16",
    "flops": "Float64",
    "memory_reads": "int64",
    "memory_writes": "int64",
    "os": "category",
    "driver_version": "string",
    "clock_mode": "category",
    "l2_warmup_iterations": "int32",
    "jit_warmup_iterations": "int32",
    "measurement_iterations": "int32",
    "latency_ms": "float64",
}

DENOX_IMPLEMENTATION_KEYS = [
    "operation",
    "input_shape",
    "input_format",
    "input_type",
    "output_shape",
    "output_format",
    "output_type",
    "shader",
    "config",
    "spirv_hash",
    "src_hash",
]

DENOX_LOGICAL_OPERATION_KEYS = [
    "operation",
    "input_shape",
    "input_format",
    "input_type",
    "output_shape",
    "output_format",
    "output_type",
]

parquet_cache_dir = Path("./parquets")

# Read all csvs in data/ directory into python
dfs: list[pd.DataFrame] = []


# TODO remove me later, this is just for developmenet of the script
EAGER_CACHING = True
df: pd.DataFrame

for path in list(Path("./data/").glob("*.csv")):
    parquet_df_path = parquet_cache_dir / path.with_suffix(".parquet").name
    if EAGER_CACHING and parquet_df_path.exists():
        print(f"Using cached {parquet_df_path} (may be stale)")
        df = pd.read_parquet(parquet_df_path)
    else:
        print(f"Reading CSV: {path}")
        df = pd.read_csv(
            path, sep=",", engine="c", on_bad_lines="error", dtype=DENOX_DB_DTYPES
        )
        print(f"Writing {parquet_df_path}")
        df.to_parquet(parquet_df_path)
    dfs.append(df)

df = pd.concat(dfs, ignore_index=True)
df_cache = Path("parquets/all.parquet")
print(f"Writing {df_cache}")
df.to_parquet(df_cache)

print("Aggregating samples...")

# First aggregate multiple samples of the same implementation. 
df = (
    df.groupby(DENOX_IMPLEMENTATION_KEYS, sort=False, observed=True)
    .agg(
        mean_latency_ms=("latency_ms", "mean"),
        median_latency_ms=("latency_ms", "median"),
        p95_latency_ms=("latency_ms", lambda x: x.quantile(0.95)),
        mean_memory_throughput=("memory_throughput", "mean"),
        median_memory_throughput=("memory_throughput", "median"),
        p95_memory_throughput=("memory_throughput", lambda x: x.quantile(0.95)),
        mean_compute_throughput=("compute_throughput", "mean"),
        median_compute_throughput=("compute_throughput", "median"),
        p95_compute_throughput=(
            "compute_throughput",
            lambda x: x.quantile(0.95),
        ),
        sample_count=("latency_ms", "size"),
    )
    .reset_index()
)

# Compute best latency per logical operation
best_median_latency = df.groupby(
    DENOX_LOGICAL_OPERATION_KEYS, sort=False, observed=False
)["median_latency_ms"].transform("min")
# relative speedup is performance relative to best implementation 
# for this logical operation, that's within this database.
df["relative_speedup"] = best_median_latency / df["median_latency_ms"]

# Expand input / output shapes
# NOTE: In the csv we encode the input / ouputs as strings,
# that look like this 
