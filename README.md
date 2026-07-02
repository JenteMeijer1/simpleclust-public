# simpleclust-public

Code release for the Simpleclust / Schizophrenia Bulletin analysis.

This repository contains the paper-specific pipeline scripts, notebook code, and table-generation code used for the Simpleclust analysis.

## Contents

- `run_profiles/simpleclust.sh`: run configuration for the Simpleclust analysis.
- `run.sh`: pipeline entry point.
- `singleclust/`: single-view clustering pipeline and helper scripts.
- `notebooks/Simpleclust/Main.ipynb`: downstream analysis notebook for checks, figures, summaries, and paper-facing analyses.
- `notebooks/Simpleclust/PrepareData_demtable_schizbull.Rmd`: baseline and demographic table workflow.
- `table_helpers/`: R helper functions used by the table workflow.
- `requirements_multiview_env.txt` and `multiview_env.def`: environment specifications.

## Environment

The Python environment is described in `requirements_multiview_env.txt`. A container definition is provided in `multiview_env.def`. The R table workflow also requires packages used by the helper scripts, including `dplyr`, `readr`, `stringr`, `tidyr`, `readxl`, `gtsummary`, `flextable`, and `effectsize`.

## Running

Pipeline:

```bash
RUN_PROFILE=simpleclust bash run.sh
```

Baseline and demographic tables:

```bash
Rscript -e "rmarkdown::render('notebooks/Simpleclust/PrepareData_demtable_schizbull.Rmd')"
```

Notebook:

```bash
jupyter notebook notebooks/Simpleclust/Main.ipynb
```

The notebook assumes that relevant pipeline outputs have already been generated.

## Citation

Please cite the associated paper when using this code.
