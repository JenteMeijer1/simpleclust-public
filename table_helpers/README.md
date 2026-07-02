# Table Helpers

This folder contains the shared R helper functions used by the exported baseline-table and demographic-table R Markdown workflow.

- `Basetable_function.R`: builds cleaned baseline tables from the restricted study data and data dictionary.
- `Demographictable_function.R`: builds formatted demographic comparison tables from prepared data.

The public repository does not include restricted study data or private dictionaries. Update the paths in the profile-specific R Markdown file under `notebooks/` to point at your local data and dictionary before rendering.
