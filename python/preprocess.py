import pandas as pd
from pandas._typing import DtypeArg
from pathlib import Path
from typing import cast
from typing import Iterator, Tuple
import hashlib

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
    "device": "category",
    "os": "category",
    "driver_version": "string",
    "start_timestamp": "uint64",
    "clock_mode": "category",
    "l2_warmup_iterations": "int32",
    "jit_warmup_iterations": "int32",
    "measurement_iterations": "int32",
    "latency_ms": "float64",
    "sample_timestamp": "uint64",
    "gpu_clock": "int32",
    "mem_clock": "int32",
}

parquet_cache_dir = Path("./parquets")
parquet_cache_dir.mkdir(parents=True, exist_ok=True)

# Read all CSVs in data/ directory, aggregate them and cache result as parquets
for path in list(Path("./data/").glob("*.csv")):
    parquet_df_path = parquet_cache_dir / path.with_suffix(".parquet").name
    if not parquet_df_path.exists():
        print(f"Reading CSV: {path}")
        df = pd.read_csv(
            path, sep=",", engine="c", on_bad_lines="error", dtype=DENOX_DB_DTYPES
        )

        print("Compute memory-throughput per sample")
        df["memory_throughput"] = (
            (df["memory_reads"] + df["memory_writes"])
            / (df["latency_ms"] * 1e-3)
            * 1e-9
        )

        print("Compute compute-throughput per sample")
        df["compute_throughput"] = (df["flops"] / (df["latency_ms"] * 1e-3)) * 1e-12

        print("Aggregating samples...")
        df = (
            df.groupby(
                [
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
                    "device",
                ],
                sort=False,
                observed=True,
            )
            .agg(
                mean_latency_ms=("latency_ms", "mean"),
                median_latency_ms=("latency_ms", "median"),
                std_latency_ms=("latency_ms", "std"),
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

        print("Compute relative speedups")
        # Compute best latency per logical operation
        best_median_latency = df.groupby(
            [
                "operation",
                "input_shape",
                "input_format",
                "input_type",
                "output_shape",
                "output_format",
                "output_type",
                "device",
                "shader",
            ],
            sort=False,
            observed=False,
        )["median_latency_ms"].transform("min")
        # relative speedup is performance relative to best implementation
        # for this logical operation, that's within this database.
        df["relative_speedup"] = best_median_latency / df["median_latency_ms"]

        print(f"Writing {parquet_df_path}")
        df.to_parquet(parquet_df_path)


dfs: list[pd.DataFrame] = []
for path in list(Path("./data/").glob("*.csv")):
    parquet_df_path = parquet_cache_dir / path.with_suffix(".parquet").name
    assert parquet_df_path.exists()
    df = pd.read_parquet(parquet_df_path)
    dfs.append(df)


# This now contains all aggregates samples
df = pd.concat(dfs, ignore_index=True)


# Takes a column like 16x16x16 and converts into 3 columns {prefix}_height, {prefix}_width ..
# if the string contains multiple shapes like 16x16x16#8x8x8,then we add a suffix {prefix}_height0 ...
# to the generated column names.
def expand_shape(df: pd.DataFrame, column: str, prefix: str) -> pd.DataFrame:
    col = df[column].astype(str)
    if not col.str.contains("#").any():
        parts = col.str.split("x", expand=True)
        parts.columns = [
            f"{prefix}_height",
            f"{prefix}_width",
            f"{prefix}_channels",
        ]
        parts = parts.astype("int32")
        return df.join(parts)

    # List-of-shapes case
    shapes = col.str.split("#")
    max_shapes = shapes.map(len).max()

    result = {}

    for i in range(max_shapes):
        dims = shapes.str[i].str.split("x", expand=True)
        result[f"{prefix}_height{i}"] = dims[0].astype("Int32")
        result[f"{prefix}_width{i}"] = dims[1].astype("Int32")
        result[f"{prefix}_channels{i}"] = dims[2].astype("Int32")

    parts = pd.DataFrame(result, index=df.index)
    return df.join(parts)

def expand_config(df) -> pd.DataFrame:
    cfg = df["config"].astype(str)
    kv = cfg.str.split("#")
    kv_df = (
        kv.explode().str.split("=", expand=True).rename(columns={0: "key", 1: "value"})
    )
    kv_df["row"] = kv_df.index

    wide = kv_df.pivot(index="row", columns="key", values="value")
    wide = wide.add_prefix("config_")

    def parse_column(col: pd.Series) -> pd.Series:
        lowered = col.str.lower()
        is_bool = lowered.dropna().isin({"true", "false"}).all()
        if is_bool:
            return lowered.map({"true": True, "false": False}).astype("boolean")
        nums = pd.to_numeric(col, errors="raise")
        nums = cast(pd.Series, pd.to_numeric(col, errors="raise"))
        return nums.astype("uint32")

    wide = wide.apply(parse_column)

    return df.join(wide)


# LSP broke down
grouped = df.groupby(["shader", "operation"], observed=True, sort=False)
for (shader, operation), group_df in cast(
    Iterator[Tuple[Tuple[str, str], pd.DataFrame]], grouped
):
    ophash = hashlib.blake2b(operation.encode("utf-8"), digest_size=8).hexdigest()
    path = parquet_cache_dir / f"{shader}-{ophash}.parquet"

    group_df = expand_shape(group_df, "input_shape", "input")
    group_df = expand_shape(group_df, "output_shape", "output")
    group_df = expand_config(group_df)

    group_df.to_parquet(path)
