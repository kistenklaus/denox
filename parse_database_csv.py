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

for op in df["operation"].unique():
    print(op)

df = df[
    df["operation"]
    == "relu(conv2d(x,kernel_size=(3,3),bias=true,stride=(1,1),padding=(1,1),dialation=(1,1)))"
]

for shape in df["input_shape"].unique():
    print(shape)

df = df[
    (df["input_shape"] == "1088x1920x32")
    & (df["input_type"] == "Float16")
    & (df["output_shape"] == "1088x1920x32")
    & (df["output_type"] == "Float16")
]


df = df.drop(
    columns=[
        "operation",
        "input_shape",
        "input_type",
        "output_shape",
        "output_type",
    ]
)

grouped = (
    df.groupby(
        ["input_format", "output_format", "shader", "config"], sort=False
    )
    .agg(
        mean_latency_ms=("latency_ms", "mean"),
        median_latency_ms=("latency_ms", "median"),
        p95_latency_ms=("latency_ms", lambda x: x.quantile(0.95)),
        sample_count=("latency_ms", "size"),
    )
    .reset_index()
)

fastest_10 = grouped.sort_values("mean_latency_ms", ascending=True).head(10)
for row in fastest_10.itertuples(index=False):
    print(f"{row.mean_latency_ms:.2f}ms   {row.input_format:<5} -> {row.output_format:<5}    {row.shader}     {row.config}")


