import pandas as pd

results = pd.concat([pd.read_csv(path) for path in snakemake.input])
results.to_csv(snakemake.output[0], index=False)
