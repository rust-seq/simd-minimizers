#!/usr/bin/env python3

import pandas as pd
from pathlib import Path
import tabulate

# from matplotlib.ticker import LogLocator
# from matplotlib.colors import to_rgba

# Limit matplot lib print precision
pd.set_option("display.precision", 2)
pd.set_option("display.float_format", "{:0.2f}".format)

print("table")
DF = pd.read_json("results.json")
# DF = pd.read_json("results-neon.json")


DF["canonical"] = DF["name"].apply(lambda x: "canonical " in x)
DF["name"] = DF["name"].str.replace("canonical ", "")

print(tabulate.tabulate(DF, headers=DF.columns, tablefmt="orgtbl", floatfmt=".2f"))

# Incremental table
df = DF[(DF["experiment"] == "incremental")][["name", "time", "k", "w"]]
df = df.pivot_table(
    index="name", columns=["w", "k"], values="time", aggfunc="median", sort=False
)
print(tabulate.tabulate(df, headers=df.columns, tablefmt="orgtbl", floatfmt=".2f"))
print(df.to_latex(float_format="%.2f"))

# External table
df = DF[DF["experiment"] == "external"]
df = df.pivot_table(
    index="name",
    columns=["w", "k", "canonical"],
    values="time",
    aggfunc="median",
    sort=False,
)
print(tabulate.tabulate(df, headers=df.columns, tablefmt="orgtbl", floatfmt=".2f"))
print(df.to_latex(float_format="%.2f"))

# # Sliding min plot
# import seaborn as sns
# import matplotlib.pyplot as plt
# df = DF[DF["experiment"] == "sliding_min"]
# ax = sns.lineplot(data=df, x="w", y="time", hue="name")
# # log base 2
# ax.set_xscale("log", base=2)
# ax.set_ylim(0, 30)
# ax.grid(axis="y")
# plt.show()
