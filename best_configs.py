import pandas as pd
from pandas._typing import DtypeArg

dtypes: DtypeArg = {
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


df = pd.read_csv(
    "gpu.csv",
    sep=",",
    engine="c",  # default, fastest
    low_memory=False,  # avoids mixed-type guessing issues
    on_bad_lines="error",
    dtype=dtypes,
)

opcols = [
    "operation",
    "input_shape",
    "input_type",
    "input_format",
    "output_shape",
    "output_type",
    "output_format",
]
implcols = ["shader", "config"]

df = (
    df.groupby(opcols + implcols, sort=False)
    .agg(
        mean_latency_ms=("latency_ms", "mean"),
        median_latency_ms=("latency_ms", "median"),
        p95_latency_ms=("latency_ms", lambda x: x.quantile(0.95)),
        sample_count=("latency_ms", "size"),
    )
    .reset_index()
)

df = df.loc[df.groupby(opcols)["mean_latency_ms"].idxmin()].reset_index(drop=True)

df = df[df["mean_latency_ms"] > 0.1]

for row in df.itertuples(index=False):
    print(f"{row.operation} ({row.mean_latency_ms:.2f}ms)\n - {row.input_format}[{row.input_shape}] ----{row.shader}----> {row.output_format}[{row.output_shape}]   \n -> {row.config}\n")


