# simpleclust-public

Public code snapshot for the Simpleclust / Schizophrenia Bulletin analysis.

This repository contains the paper-specific code needed to run the Simpleclust profile, generate the baseline/demographic tables, and inspect the downstream analysis notebook. It does not contain restricted study data or generated results.

This is an exported snapshot from a private development repository. The private repository remains the source of truth. See `PUBLIC_SNAPSHOT.md` for the export date, source commit, and exact file list.

## What To Look At First

Start with these files:

1. `run_profiles/simpleclust.sh` configures the Simpleclust run.
2. `run.sh` starts the scheduled pipeline using that profile.
3. `singleclust/full_pipeline_singleclust.py` contains the main single-view clustering workflow.
4. `notebooks/Simpleclust/PrepareData_demtable_schizbull.Rmd` prepares baseline and demographic tables.
5. `notebooks/Simpleclust/Main.ipynb` contains downstream checks, figures, summaries, and paper-facing analyses.

## Repository Map

```text
.
├── run_profiles/
│   └── simpleclust.sh
├── singleclust/
│   ├── full_pipeline_singleclust.py
│   ├── clustering_functions.py
│   ├── bootstrap_candidates.sh
│   ├── gather_candidates.sh
│   └── finalize_alternative_k.py
├── notebooks/
│   └── Simpleclust/
│       ├── Main.ipynb
│       └── PrepareData_demtable_schizbull.Rmd
├── table_helpers/
│   ├── Basetable_function.R
│   └── Demographictable_function.R
├── run.sh
├── requirements_multiview_env.txt
└── multiview_env.def
```

Other private profiles, notebooks, result folders, data folders, and manuscript files are intentionally excluded.

## Data

You need your own approved copy of the restricted study data. This repository does not include:

- raw study data
- private dictionaries
- generated pipeline outputs
- generated tables
- result folders
- manuscript files

Update paths in `run_profiles/simpleclust.sh` and `notebooks/Simpleclust/PrepareData_demtable_schizbull.Rmd` so they point to your local data and output locations.

## Environment

Use either the Python requirements file or the container definition:

- `requirements_multiview_env.txt`
- `multiview_env.def`

The R Markdown table workflow also needs R packages used by the helper scripts, including `dplyr`, `readr`, `stringr`, `tidyr`, `readxl`, `gtsummary`, `flextable`, and `effectsize`.

## Typical Run Order

Run the Simpleclust pipeline:

```bash
RUN_PROFILE=simpleclust bash run.sh
```

Generate baseline and demographic tables:

```bash
Rscript -e "rmarkdown::render('notebooks/Simpleclust/PrepareData_demtable_schizbull.Rmd')"
```

Open the notebook after pipeline outputs are available:

```bash
jupyter notebook notebooks/Simpleclust/Main.ipynb
```

## For Maintainers

Do not edit this public snapshot by hand unless it is an emergency. Refresh it from the private source repository:

```bash
cd ../multiclust
bash tools/export_public_repos.sh simpleclust ../simpleclust-public
```

Then review, commit, and push:

```bash
cd ../simpleclust-public
git diff
git add .
git commit -m "Refresh public snapshot"
git push
```

## Citation

Please cite the associated paper and this code snapshot. If a `CITATION.cff` file is added in a future export, prefer that citation metadata.
