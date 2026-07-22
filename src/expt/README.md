# NCFA #

This repo provides an implementation of the experiments from *Neuro-Causal Factor Analysis*, accepted to the 13th International Conference on Probabilistic Graphical Models (PGM 2026).

Experiments are organized into a Snakemake workflow: [`workflow/Snakefile`](workflow/Snakefile).

Dependencies, versioning, and installation can all be handled by `uv`, with the included [`pyproject.toml`](../../pyproject.toml) and [`uv.lock`](../../uv.lock) containing all necessary information.
The pinned `requirements.txt` with hashes is provided for users of other package managers.

After [installing uv](https://docs.astral.sh/uv/getting-started/):
- `uv run pytest tests/` can be run as a quick check from project root
- `uv run snakemake all --cores all --forceall` can be run from `src/expt/` to reproduce all experiments; it takes less than half an hour, depending on cpu, and the first call will download the third-party `causalchambers` dataset.

The first time `uv run <...>` is called, it will download and install all dependencies.
Decrease the number of cores (e.g., `10` instead of `all`) as needed.
All snakemake outputs are saved in `src/expt/results/`.

# Contact #

Feel free to raise an issue or [email me](mailto:alex.markham@causal.dev) with any questions about reproducing the experimental results, modifying the code to your problem, or applying it to your data!
