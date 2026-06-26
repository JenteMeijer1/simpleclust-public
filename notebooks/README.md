# Project notebooks

This folder keeps project-specific exploratory and reporting notebooks close to the shared `multiclust` pipeline without mixing them into the executable script root.

## Layout

- `multiclust_extended/`: default AMPSCZ extended multiview clustering notebook and R Markdown preparation file.
- `clinical_paper/`: clinical paper notebooks and demographic table R Markdown file.
- `prospect/`: PROSPECT notebook and R Markdown files.
- `Simpleclust/`: Schizophrenia Bulletin/simpleclust notebook and R Markdown preparation file.

## Runtime rule

The Slurm profiles in `../run_profiles/` set `NOTEBOOK` for report generation. If a project needs a different report notebook, update only that profile:

```bash
export NOTEBOOK="notebooks/<project>/Main.ipynb"
```

Python notebooks include an initial setup cell that locates the shared `multiclust` root and adds it to `sys.path`. This keeps imports such as `from Utils import *`, `from SVM import *`, and `import theme` working when the notebook is opened from `notebooks/<project>/`.

If a notebook is launched from a directory that is not inside `Code/multiclust`, set:

```bash
export MULTICLUST_ROOT="/path/to/Code/multiclust"
```

The pipeline scripts should stay in the `multiclust` root unless the Slurm wrappers are updated at the same time.

## Shared Notebook Helpers

Reusable notebook functions should live in `../Utils.py`, not inside individual notebooks. The current shared surface includes:

- Notebook setup/profile helpers: `infer_notebook_profile`, `parse_profile_exports`, `profile_enabled_for_sensitivity`.
- Pre-pipeline checks: `print_remaining_after_full_missing_modality_removal`.
- Post-result summaries: `add_metadata_and_clusters`, `chi_square_comparison`, `summarize_feature_differences`.
- Stream/pathway summaries: `summarize_streams`, `compare_prefix_structure`, `compare_final_mapping`, `full_structure_report`, `all_streams_table`.
- Plot helpers: `build_group_palette`, `alluvial_sankey_general`, `domain_map`, `plot_pred_modality`.

Existing underscore-prefixed notebook names such as `_infer_notebook_profile`, `_flatten_sensitivity_results`, and `alluvial_sankey_force_high_top` are kept as compatibility aliases in `Utils.py`.

For the Schizophrenia Bulletin/simpleclust workflow, the notebook is downstream of the scheduled singleclust pipeline. The executable stage structure and reproduction commands are documented in `../docs/singleclust_pipeline_overview.md`; the notebook should focus on reading saved outputs, diagnostics, paper figures, tables, and interpretation rather than redefining pipeline logic.
