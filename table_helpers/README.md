# Table Helpers

This folder contains the R helper functions used by the baseline-table and demographic-table R Markdown workflow.

## Files

- `Basetable_function.R`: reads the restricted study data and data dictionary, selects variables, harmonises fields, and creates the baseline analysis table.
- `Demographictable_function.R`: creates formatted demographic comparison tables from prepared data.

## How These Are Used

The profile-specific R Markdown file under `notebooks/` calls these helpers with:

```r
source("table_helpers/Basetable_function.R")
source("table_helpers/Demographictable_function.R")
```

Before rendering the R Markdown file, update the data and dictionary paths inside that file so they point to your local restricted data copy.

## Data Availability

The public repository does not include restricted study data, private dictionaries, generated CSV files, Word tables, or result folders. Those files must be provided separately by users with the appropriate data access permissions.
