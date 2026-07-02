#' Create a Descriptive Table for AMP_SCZ or Prescient data
#'
#' This function generates a summary table for selected variables, automatically
#' converting categorical variables to factors when there are 5 or fewer unique numeric values.
#'
#' @param df A data frame containing the dataset.
#' @param dictionary_DIR A directory/path to where the dictionary is saved.
#' @param comparison A character string indicating which variable to use for comparing groups.
#' @param comparison_labels A named character vector specifying labels for the comparison variable.
#'
#' @return A formatted summary table as a flextable object.
#'
#' @import gtsummary dplyr forcats flextable effectsize
#' @export

demographic_table <- function(df, dictionary_DIR, comparison, comparison_labels = c()) {
  library(gtsummary)
  library(dplyr)
  library(forcats)
  library(flextable)
  library(readr)
  library(stringr)
  library(tidyr)
  library(readxl)
  library(effectsize)
  
  # Custom warning function for improved readability
  print_warning <- function(warning_text) {
    if (length(warning_text) > 0 && !all(is.na(warning_text)) && !all(warning_text == "")) {
      cat("\n⚠ Warning:\n", paste(warning_text, collapse = "\n"), "\n\n")
    }
  }
  
  ### Error Handling and Validations ----
  if (!is.data.frame(df)) {
    stop("Error: 'df' must be a data frame.")
  }
  
  if (nrow(df) == 0) {
    stop("Error: The dataset is empty (0 rows). Please provide a valid dataset.")
  }
  
  if (missing(comparison) || is.null(comparison) || comparison == "") {
    stop("Error: Please provide a valid comparison variable.")
  }
  
  if (!comparison %in% colnames(df)) {
    stop(paste0("Error: The comparison variable '", comparison, "' does not exist in the dataset."))
  }
  
  #'##########################################################################################################
  ## Load necessary functions ----
  #'##########################################################################################################
  
  variable_selection <- function(dictionary_DIR, required_vars = c(comparison)) {
    dict <- read_excel(dictionary_DIR, col_types = "text")
    
    # Ensure required columns exist
    if (!"Include_basetable" %in% names(dict)) {
      stop("Error: Dictionary must have an 'Include_basetable' column.")
    }
    
    if (!"Include_demographics" %in% names(dict)) {
      stop("Error: Dictionary must have an 'Include_demographics' column.")
    }
    
    if (!"Datatype" %in% names(dict)) {
      stop("Error: Dictionary must have a 'Datatype' column if you want to track data sources.")
    }
    
    if (!"ElementName" %in% names(dict)) {
      stop("Error: Dictionary must have an 'ElementName' column.")
    }
    
    if (!"Label" %in% names(dict)) {
      stop("Error: Dictionary must have a 'Label' column.")
    }
    
    if (!"Aliases" %in% names(dict)) {
      dict$Aliases <- NA_character_
    }
    
    if (!"Condition" %in% names(dict)) {
      dict$Condition <- NA_character_
    }
    
    # Expand Aliases into separate rows
    dict <- dict %>%
      mutate(Aliases = str_replace_all(Aliases, "\\s+", "")) %>%
      separate_rows(Aliases, sep = ",") %>%
      mutate(Aliases = ifelse(Aliases == "", NA, Aliases))
    
    # Create duplicated rows for each alias
    alias_rows <- dict %>%
      filter(!is.na(Aliases)) %>%
      mutate(
        Original_ElementName = ElementName,
        ElementName = Aliases
      ) %>%
      select(-Aliases)
    
    # Combine with the original dictionary
    dict <- bind_rows(dict, alias_rows) %>%
      select(-Aliases, -Original_ElementName, -Condition)
    
    # Select variables for the base table
    selected_vars <- dict %>%
      filter(Include_basetable %in% c("Yes", "yes")) %>%
      select(ElementName, Label)
    
    selected_vars <- unique(selected_vars)
    
    # Select variables for demographics
    selected_vars_demographics <- dict %>%
      filter(Include_demographics %in% c("Yes", "yes")) %>%
      select(ElementName, Label)
    
    selected_vars_demographics <- unique(selected_vars_demographics)
    
    # Create metatable with info about modality
    metatable <- dict %>%
      filter(Include_basetable %in% c("Yes", "yes")) %>%
      select(ElementName, Datatype)
    
    # Ensure required_vars are included in base table selection
    missing_required <- setdiff(required_vars, selected_vars$ElementName)
    
    if (length(missing_required) > 0) {
      selected_vars <- bind_rows(
        selected_vars,
        tibble(ElementName = missing_required, Label = missing_required)
      )
    }
    
    # Ensure required_vars are included in demographics selection
    missing_required <- setdiff(required_vars, selected_vars_demographics$ElementName)
    
    if (length(missing_required) > 0) {
      selected_vars_demographics <- bind_rows(
        selected_vars_demographics,
        tibble(ElementName = missing_required, Label = missing_required)
      )
    }
    
    # Capture dictionary order
    dict_order <- dict$ElementName
    
    vars_basetable <- selected_vars$ElementName
    vars_demographics <- selected_vars_demographics$ElementName
    labels <- setNames(as.list(selected_vars_demographics$Label), selected_vars_demographics$ElementName)
    
    return(list(vars_basetable, vars_demographics, labels, dict_order))
  }
  
  #'##########################################################################################################
  
  ## Variable selection ----
  selected_variables <- variable_selection(dictionary_DIR, required_vars = c(comparison))
  variables <- selected_variables[[2]]
  labels <- selected_variables[[3]]
  dict_order <- selected_variables[[4]]
  
  # Select variables that exist in the dataset
  existing_vars <- variables[variables %in% colnames(df)]
  
  # Identify and warn about missing variables
  missing_vars <- setdiff(variables, existing_vars)
  
  if (length(missing_vars) > 0) {
    print_warning(paste(
      "The following variables do not exist in the dataset and will be skipped:",
      paste(missing_vars, collapse = ", ")
    ))
  }
  
  # Remove variables that are completely NA
  valid_vars <- existing_vars[sapply(df[existing_vars], function(x) !all(is.na(x)))]
  
  # Identify and warn about variables that contain only NA values
  na_vars <- setdiff(existing_vars, valid_vars)
  
  if (length(na_vars) > 0) {
    print_warning(paste(
      "The following variables contain only NA values and will be skipped:",
      paste(na_vars, collapse = ", ")
    ))
  }
  
  # Stop execution if no valid variables remain
  if (length(valid_vars) == 0) {
    stop("Error: None of the specified variables exist or contain data.")
  }
  
  # Make sure comparison is included in valid_vars for processing
  if (!comparison %in% valid_vars) {
    valid_vars <- c(comparison, valid_vars)
  }
  
  # Ensure labels match only valid variables
  valid_labels <- labels[names(labels) %in% valid_vars]
  
  ## Prepare comparison variable ----
  
  # Convert comparison variable to character/factor if needed
  if (!is.factor(df[[comparison]])) {
    print_warning(paste(
      "The comparison variable", comparison,
      "is not a factor. Converting to factor."
    ))
    
    df[[comparison]] <- as.character(df[[comparison]])
  }
  
  # Apply comparison labels if provided
  if (!is.null(comparison_labels) && length(comparison_labels) > 0) {
    df[[comparison]] <- factor(
      df[[comparison]],
      levels = names(comparison_labels),
      labels = comparison_labels
    )
  } else {
    df[[comparison]] <- factor(df[[comparison]])
  }
  
  # Remove rows with missing comparison group
  df <- df[!is.na(df[[comparison]]), ]
  
  # Drop unused comparison levels
  df[[comparison]] <- droplevels(factor(df[[comparison]]))
  
  # Find number of observed groups
  ngroups <- nlevels(df[[comparison]])
  
  if (ngroups < 2) {
    stop(paste0(
      "Error: The comparison variable '", comparison,
      "' has fewer than 2 observed groups after removing missing values."
    ))
  }
  
  ## Convert selected variables to factors when they have small unique numeric values ----
  df <- df %>%
    mutate(across(all_of(valid_vars), ~ {
      if (is.numeric(.) && n_distinct(na.omit(.)) <= 5) {
        factor(.)
      } else {
        .
      }
    }))
  
  # Re-apply factor to comparison because mutate above may touch it
  df[[comparison]] <- droplevels(factor(df[[comparison]]))
  
  ### Check for valid statistical tests ----
  # This prevents:
  # Error in stats::chisq.test(x, y): 'x' and 'y' must have at least 2 levels
  
  candidate_test_vars <- setdiff(valid_vars, comparison)
  
  valid_tests <- candidate_test_vars[sapply(candidate_test_vars, function(var) {
    tmp <- df %>%
      select(all_of(c(var, comparison))) %>%
      filter(!is.na(.data[[var]]), !is.na(.data[[comparison]]))
    
    # Need at least some data
    if (nrow(tmp) == 0) {
      return(FALSE)
    }
    
    # Need at least 2 observed comparison groups
    if (n_distinct(tmp[[comparison]]) < 2) {
      return(FALSE)
    }
    
    if (is.numeric(tmp[[var]])) {
      # Continuous variable:
      # Need at least 2 observations in each observed group for t-test / ANOVA
      group_n <- tmp %>%
        group_by(.data[[comparison]]) %>%
        summarise(n = sum(!is.na(.data[[var]])), .groups = "drop")
      
      return(nrow(group_n) >= 2 && all(group_n$n > 1))
    } else {
      # Categorical variable:
      # Need at least 2 observed levels in the variable
      if (n_distinct(tmp[[var]]) < 2) {
        return(FALSE)
      }
      
      # Need a contingency table with at least 2 rows and 2 columns
      tab <- table(tmp[[var]], tmp[[comparison]])
      
      return(nrow(tab) >= 2 && ncol(tab) >= 2)
    }
  })]
  
  # Warn and remove variables that cannot be tested
  invalid_tests <- setdiff(candidate_test_vars, valid_tests)
  
  if (length(invalid_tests) > 0) {
    print_warning(paste(
      "The following variables could not be tested because they have fewer than 2 observed levels, fewer than 2 comparison groups, or insufficient non-missing data and will be skipped:",
      paste(invalid_tests, collapse = ", ")
    ))
  }
  
  # Update valid variables to only include variables that passed the statistical validity check
  valid_vars <- valid_tests
  
  if (length(valid_vars) == 0) {
    stop("Error: No variables remain after checking for valid statistical tests.")
  }
  
  ## Compute Effect Sizes Only for Valid Variables ----
  effect_sizes <- data.frame(
    variable = valid_vars,
    effect_size = NA_character_,
    stringsAsFactors = FALSE
  )
  
  for (var in valid_vars) {
    tmp <- df %>%
      select(all_of(c(var, comparison))) %>%
      filter(!is.na(.data[[var]]), !is.na(.data[[comparison]]))
    
    if (nrow(tmp) == 0) {
      next
    }
    
    if (is.numeric(tmp[[var]])) {
      if (ngroups == 2) {
        effect_value <- tryCatch(
          {
            effectsize::cohens_d(tmp[[var]] ~ tmp[[comparison]])$Cohens_d
          },
          error = function(e) NA_real_
        )
        
        effect_sizes$effect_size[effect_sizes$variable == var] <-
          ifelse(is.na(effect_value), NA_character_, format(round(effect_value, 2), nsmall = 2))
      } else {
        effect_value <- tryCatch(
          {
            aov_model <- aov(tmp[[var]] ~ tmp[[comparison]], data = tmp)
            aov_summary <- summary(aov_model)[[1]]
            ss_between <- aov_summary$`Sum Sq`[1]
            ss_total <- sum(aov_summary$`Sum Sq`)
            ss_between / ss_total
          },
          error = function(e) NA_real_
        )
        
        effect_sizes$effect_size[effect_sizes$variable == var] <-
          ifelse(is.na(effect_value), NA_character_, format(round(effect_value, 2), nsmall = 2))
      }
    } else {
      effect_value <- tryCatch(
        {
          effectsize::cramers_v(tmp[[var]], tmp[[comparison]])$Cramers_v
        },
        error = function(e) NA_real_
      )
      
      effect_sizes$effect_size[effect_sizes$variable == var] <-
        ifelse(is.na(effect_value), NA_character_, format(round(effect_value, 2), nsmall = 2))
    }
  }
  
  ## Reorder variables based on dictionary ----
  final_vars <- valid_vars[order(match(valid_vars, dict_order))]
  final_vars <- setdiff(final_vars, comparison)
  
  if (length(final_vars) == 0) {
    stop("Error: No valid demographic variables remain for the summary table.")
  }
  
  # Update valid labels after final filtering
  valid_labels <- labels[names(labels) %in% final_vars]
  
  ## Choose statistical test for continuous variables based on number of groups ----
  if (ngroups == 2) {
    test_cont <- "t.test"
  } else {
    test_cont <- "oneway.test"
  }
  
  ## Generate Table with Only Valid Variables ----
  table <- df %>%
    select(all_of(c(comparison, final_vars))) %>%
    tbl_summary(
      by = comparison,
      type = all_continuous() ~ "continuous2",
      statistic = all_continuous() ~ c("{mean} ({sd})"),
      digits = list(interview_age ~ c(1, 2)),
      missing = "no",
      label = valid_labels,
      sort = list(all_categorical() ~ "frequency")
    ) %>%
    add_p(
      test = list(
        all_continuous() ~ test_cont,
        all_categorical() ~ "chisq.test",
        all_dichotomous() ~ "fisher.test"
      ),
      test.args = list(
        all_dichotomous() ~ list(workspace = 2e6)
      )
    ) %>%
    add_overall() %>%
    modify_table_body(
      ~ .x %>%
        left_join(effect_sizes, by = c("variable" = "variable"), relationship = "many-to-many") %>%
        group_by(variable) %>%
        mutate(effect_size = ifelse(row_number() == 1, effect_size, NA)) %>%
        ungroup() %>%
        mutate(
          p.adjusted = p.adjust(p.value, method = "fdr"),
          p.adjusted = ifelse(
            is.na(p.adjusted),
            NA_character_,
            ifelse(
              as.numeric(p.adjusted) < 0.001,
              "<0.001",
              ifelse(
                as.numeric(p.adjusted) > 0.99,
                ">0.99",
                format(round(as.numeric(p.adjusted), 3), nsmall = 3)
              )
            )
          )
        )
    ) %>%
    modify_header(
      p.adjusted = "**FDR-Adjusted p-Value**",
      effect_size = "**Effect Size**"
    ) %>%
    modify_table_styling(
      columns = c("p.adjusted", "effect_size"),
      rows = !is.na(p.adjusted) &
        (
          p.adjusted == "<0.001" |
            suppressWarnings(as.numeric(gsub("<|>", "", p.adjusted)) < 0.05)
        ),
      text_format = "bold"
    ) %>%
    bold_p(t = 0.05) %>%
    modify_table_body(~ .x %>% relocate(effect_size, .after = last_col())) %>%
    modify_footnote(
      effect_size = "Effect Size: Cohen’s d for continuous variables, eta squared for continuous variables with more than two groups, and Cramér’s V for categorical variables."
    ) %>%
    as_flex_table()
  
  return(table)
}