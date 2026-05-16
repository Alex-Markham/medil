from pathlib import Path

import pandas as pd

df = pd.read_csv(str(snakemake.input))

metrics = ["mse", "mcc", "dcor", "time", "num_params"]
datasets = ["simulated", "causalchamber"]
methods = ["fa", "vae", "lgminmcm", "ncfa"]


def mad(x):
    return (x - x.median()).abs().median()


rows = []
for method in methods:
    row = {"method": method}
    for ds in datasets:
        for metric in metrics:
            x = df.loc[(df["dataset"] == ds) & (df["method"] == method), metric]
            med = x.median()
            m = mad(x)
            row[f"{ds}_{metric}"] = f"{med:.3f} $\\pm$ {m:.3f}"
    rows.append(row)

out = pd.DataFrame(rows)

header1 = (
    "Method & "
    + " & ".join([f"\\multicolumn{{5}}{{c}}{{{ds.capitalize()}}}" for ds in datasets])
    + r" \\"
)
header2 = "& " + " & ".join(metrics * len(datasets)) + r" \\"

body = []
for _, r in out.iterrows():
    vals = [r["method"]]
    for ds in datasets:
        for metric in metrics:
            vals.append(r[f"{ds}_{metric}"])
    body.append(" & ".join(vals) + r" \\")

tex = "\n".join(
    [
        r"\begin{tabular}{rcccccccccc}",
        r"\toprule",
        header1,
        r"\cmidrule(lr){2-6} \cmidrule(lr){7-11}",
        header2,
        r"\midrule",
        *body,
        r"\bottomrule",
        r"\end{tabular}",
    ]
)

Path(str(snakemake.output)).write_text(tex)
