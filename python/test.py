import pandas as pd
from pandas._typing import DtypeArg
from pathlib import Path

df = pd.read_parquet("parquets/concat-conv-cm-e72ad629fd314864.parquet")
print(df)
df = df[df["device"] == "NVIDIA GeForce RTX 4070"]

print(df["operation"].unique())


