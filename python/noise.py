import pandas as pd

df = pd.read_csv("data/rx7900xtx.csv")


GROUP = [
    "operation",
    "input_shape",
    "input_format",
    "input_type",
    "output_shape",
    "output_format",
    "output_type",
    "shader",
    "subgroup_size",
    "config",
    "spirv_hash",
    "src_hash",
    "device",
]
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
stds = df.groupby(GROUP, observed=True, sort=False)["latency_ms"].std()

for v in stds:
    print(v)
