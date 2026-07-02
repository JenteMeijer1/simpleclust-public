# simpleclust-public

This repository is a public, paper-specific snapshot for the Simpleclust / Schizophrenia Bulletin analysis profile. It is intended to stand on its own as the code release for that profile.

The private development repository remains the source of truth. This public repository may lag behind ongoing private development; see `PUBLIC_SNAPSHOT.md` for the export date, source commit, selected profile, and allowlist.

## Included Workflow

- Profile: `run_profiles/simpleclust.sh`
- Main analysis notebook: `notebooks/Simpleclust/Main.ipynb`
- Baseline and demographic tables: `notebooks/Simpleclust/PrepareData_demtable_schizbull.Rmd`
- Baseline table helper: `table_helpers/Basetable_function.R`
- Demographic table helper: `table_helpers/Demographictable_function.R`
- Main pipeline entry point: `run.sh`
- Single-view clustering pipeline: `singleclust/full_pipeline_singleclust.py`

Only the Simpleclust profile, Simpleclust notebook, Simpleclust R Markdown table workflow, shared runtime modules needed by the pipeline, and single-view clustering files are included. Other private profiles and notebooks are intentionally excluded.

## Data

This snapshot contains code only. Real study data, derived private result files, and local output folders are not included. To run the workflow, provide the required study data in the locations configured by `run_profiles/simpleclust.sh`, or adapt that profile for your environment.

## Running

Install the Python/R environment from `requirements_multiview_env.txt` or build the Apptainer/Singularity image from `multiview_env.def`.

```bash
RUN_PROFILE=simpleclust bash run.sh
```

Run the notebook after pipeline outputs are available:

```bash
jupyter notebook notebooks/Simpleclust/Main.ipynb
```

Generate baseline and demographic tables with:

```bash
Rscript -e "rmarkdown::render('notebooks/Simpleclust/PrepareData_demtable_schizbull.Rmd')"
```

The R Markdown file uses the helper scripts in `table_helpers/`. If you move files around, update the `source()` paths in the R Markdown file accordingly.

## Citation

Please cite the associated paper and this code snapshot. If a `CITATION.cff` file is added in a future export, prefer that citation metadata.
