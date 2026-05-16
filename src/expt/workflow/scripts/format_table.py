from pathlib import Path

import pandas as pd

df = pd.read_csv(str(snakemake.input))

metrics = ["mse", "mcc", "dcor", "sfd", "time", "num_params"]
datasets = ["simulated", "causalchamber"]
methods = ["fa", "vae", "lgminmcm", "ncfa"]

method_display = {
    "fa": "FA",
    "vae": "VAE",
    "lgminmcm": "LG-minMCM",
    "ncfa": "NCFA",
}

metric_display = {
    "mse": r"MSE $\downarrow$",
    "mcc": r"MCC $\uparrow$",
    "dcor": r"dCor $\uparrow$",
    "sfd": r"SFD $\downarrow$",
    "time": r"Time",
    "num_params": r"\# Params",
}


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


def make_table(dataset_name):
    header = "Method & " + " & ".join(metric_display[m] for m in metrics) + r" \\"

    body = []
    for _, r in out.iterrows():
        vals = [method_display[r["method"]]]
        for metric in metrics:
            vals.append(r[f"{dataset_name}_{metric}"])
        body.append(" & ".join(vals) + r" \\")

    tex = "\n".join(
        [
            rf"{{{dataset_name}}}:",
            "",
            r"\begin{tabular}{rccccc}",
            r"\toprule",
            header,
            r"\midrule",
            *body,
            r"\bottomrule",
            r"\end{tabular}",
        ]
    )
    return tex


tex = make_table("simulated") + "\n\n\n" + make_table("causalchamber")

Path(str(snakemake.output)).write_text(tex)
