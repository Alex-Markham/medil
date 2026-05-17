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


def fmt(x, p=3):
    xf = float(x)
    return str(int(xf)) if xf.is_integer() else f"{xf:.{p}f}"


def fmt_pm(med, m, p=3):
    return f"{fmt(med, p)} $\\pm$ {fmt(m, p)}"


rows = []
for method in methods:
    row = {"method": method}
    for ds in datasets:
        for metric in metrics:
            x = df.loc[(df["dataset"] == ds) & (df["method"] == method), metric]
            row[f"{ds}_{metric}"] = fmt_pm(x.median(), mad(x))
    rows.append(row)

out = pd.DataFrame(rows)


def make_table(dataset_name):
    header = (
        " & \\textbf{"
        + "} & \\textbf{".join(metric_display[m] for m in metrics)
        + r"} \\"
    )

    body = []
    for _, r in out.iterrows():
        vals = [r"\textbf{" + method_display[r["method"]] + "}"]
        for metric in metrics:
            vals.append(r[f"{dataset_name}_{metric}"])
        body.append(" & ".join(vals) + r" \\")

    tex = "\n".join(
        [
            "\\newcommand{\\insert" + dataset_name + "}{%",
            r"\begin{tabular}{rcccccc}",
            r"\toprule",
            header,
            r"\midrule",
            *body,
            r"\bottomrule",
            r"\end{tabular}%",
            r"}",
        ]
    )
    return tex


tex = make_table("simulated") + "\n\n\n" + make_table("causalchamber")

Path(str(snakemake.output)).write_text(tex)
