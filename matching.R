library(arrow)
library(dplyr)
library(lubridate)
library(FNN)
library(purrr)
library(tidyr)
library(ggplot2)
library(tibble)

# =========================================================
# 1) Risk-set matching
# =========================================================
risk_set_matching_peak <- function(
    data,
    id_col = "aID",
    month_col = "TIDPUNKT",
    adoption_col = "tariff_start",
    lookback_months = 12,
    k_neighbors = 5,
    price_filter = "all",
    feature_mode = "time_series"   # "time_series" or "summary"
){
  
  if (!feature_mode %in% c("time_series", "summary")) {
    stop("feature_mode must be 'time_series' or 'summary'")
  }
  
  # -------------------
  # 1 Clean data
  # -------------------
  df <- data %>%
    filter(price == price_filter) %>%
    mutate(
      month = as.Date(.data[[month_col]]),
      adoption_month = as.Date(.data[[adoption_col]])
    ) %>%
    arrange(.data[[id_col]], month)
  
  # -------------------
  # 2 Find adopters
  # -------------------
  adopters <- df %>%
    filter(!is.na(adoption_month)) %>%
    distinct(.data[[id_col]], adoption_month)
  
  # -------------------
  # 3 Profile builder
  # -------------------
  build_profile <- function(user_data, Ti){
    
    window <- user_data %>%
      filter(
        month >= Ti %m-% months(lookback_months),
        month < Ti
      ) %>%
      arrange(month)
    
    if (nrow(window) < lookback_months) return(NULL)
    
    if (feature_mode == "time_series") {
      
      values <- window$top3_mean_consumption[1:lookback_months]
      names(values) <- paste0("peak_", 1:lookback_months)
      
      tibble(
        id = unique(user_data[[id_col]])[1],
        !!!as.list(values)
      )
      
    } else if (feature_mode == "summary") {
      
      tibble(
        id = unique(user_data[[id_col]])[1],
        peak_mean = mean(window$top3_mean_consumption, na.rm = TRUE),
        peak_sd = sd(window$top3_mean_consumption, na.rm = TRUE),
        peak_volatility = mean(abs(diff(window$top3_mean_consumption)), na.rm = TRUE)
      )
    }
  }
  
  # -------------------
  # 4 Matching loop
  # -------------------
  results <- map_dfr(seq_len(nrow(adopters)), function(i){
    
    treated_id <- adopters[[id_col]][i]
    Ti <- adopters$adoption_month[i]
    
    treated_data <- df %>%
      filter(.data[[id_col]] == treated_id)
    
    treated_profile <- build_profile(treated_data, Ti)
    
    if (is.null(treated_profile)) return(NULL)
    
    controls <- df %>%
      filter(
        .data[[id_col]] != treated_id,
        is.na(adoption_month) | adoption_month > Ti
      ) %>%
      distinct(.data[[id_col]]) %>%
      pull()
    
    control_profiles <- map_dfr(controls, function(cid){
      
      dat <- df %>%
        filter(.data[[id_col]] == cid)
      
      prof <- build_profile(dat, Ti)
      
      if (is.null(prof)) return(NULL)
      
      prof %>%
        mutate(control_id = cid)
    })
    
    if (nrow(control_profiles) == 0) return(NULL)
    
    feature_cols <- setdiff(names(control_profiles), c("id", "control_id"))
    
    X_control <- control_profiles %>%
      select(all_of(feature_cols)) %>%
      scale()
    
    X_treated <- treated_profile %>%
      select(all_of(feature_cols)) %>%
      scale(
        center = attr(X_control, "scaled:center"),
        scale  = attr(X_control, "scaled:scale")
      )
    
    k_use <- min(k_neighbors, nrow(control_profiles))
    if (k_use == 0) return(NULL)
    
    nn <- get.knnx(X_control, X_treated, k = k_use)
    
    matched <- control_profiles[nn$nn.index[1, ], , drop = FALSE]
    
    matched %>%
      mutate(
        treated_id = treated_id,
        adoption_month = Ti,
        distance = nn$nn.dist[1, ]
      )
  })
  
  return(results)
}

# =========================================================
# 2) Build profile dataset for balance diagnostics
# =========================================================
build_profiles <- function(
    data,
    id_col = "aID",
    month_col = "TIDPUNKT",
    adoption_col = "tariff_start",
    lookback_months = 12,
    price_filter = "all",
    feature_mode = "time_series"
){
  
  if (!feature_mode %in% c("time_series", "summary")) {
    stop("feature_mode must be 'time_series' or 'summary'")
  }
  
  df <- data %>%
    filter(price == price_filter) %>%
    mutate(
      month = as.Date(.data[[month_col]]),
      adoption_month = as.Date(.data[[adoption_col]])
    ) %>%
    arrange(.data[[id_col]], month)
  
  adopters <- df %>%
    filter(!is.na(adoption_month)) %>%
    distinct(.data[[id_col]], adoption_month)
  
  build_profile <- function(user_data, Ti){
    
    window <- user_data %>%
      filter(
        month >= Ti %m-% months(lookback_months),
        month < Ti
      ) %>%
      arrange(month)
    
    if (nrow(window) < lookback_months) return(NULL)
    
    if (feature_mode == "time_series") {
      
      values <- window$top3_mean_consumption[1:lookback_months]
      names(values) <- paste0("peak_", 1:lookback_months)
      
      tibble(
        id = unique(user_data[[id_col]])[1],
        !!!as.list(values)
      )
      
    } else {
      
      tibble(
        id = unique(user_data[[id_col]])[1],
        peak_mean = mean(window$top3_mean_consumption, na.rm = TRUE),
        peak_sd = sd(window$top3_mean_consumption, na.rm = TRUE),
        peak_volatility = mean(abs(diff(window$top3_mean_consumption)), na.rm = TRUE)
      )
    }
  }
  
  profiles <- map_dfr(seq_len(nrow(adopters)), function(i){
    
    uid <- adopters[[id_col]][i]
    Ti <- adopters$adoption_month[i]
    
    user_data <- df %>%
      filter(.data[[id_col]] == uid)
    
    prof <- build_profile(user_data, Ti)
    
    if (is.null(prof)) return(NULL)
    
    prof %>%
      mutate(adoption_month = Ti)
  })
  
  return(profiles)
}

# =========================================================
# 3) Balance table
# =========================================================
balance_table <- function(profiles, matches){
  
  if (nrow(matches) == 0) {
    stop("matches is empty")
  }
  
  treated_ids <- unique(matches$treated_id)
  
  treated <- profiles %>%
    filter(id %in% treated_ids)
  
  # 用 matches 保留 control 重複次數
  control <- matches %>%
    select(control_id) %>%
    rename(id = control_id) %>%
    left_join(profiles, by = "id")
  
  covariates <- setdiff(names(profiles), c("id", "adoption_month"))
  
  smd <- function(x, y) {
    (mean(x, na.rm = TRUE) - mean(y, na.rm = TRUE)) /
      sqrt((var(x, na.rm = TRUE) + var(y, na.rm = TRUE)) / 2)
  }
  
  balance <- lapply(covariates, function(v){
    
    tibble(
      covariate = v,
      treated_mean = mean(treated[[v]], na.rm = TRUE),
      control_mean = mean(control[[v]], na.rm = TRUE),
      SMD = smd(treated[[v]], control[[v]])
    )
  }) %>%
    bind_rows()
  
  return(balance)
}

# =========================================================
# 4) Love plot
# =========================================================
love_plot <- function(balance, title = "Covariate Balance"){
  
  ggplot(balance,
         aes(x = abs(SMD),
             y = reorder(covariate, abs(SMD)))) +
    geom_point(size = 3, color = "steelblue") +
    geom_vline(xintercept = 0.1,
               linetype = "dashed",
               color = "red") +
    labs(
      title = title,
      x = "|Standardized Mean Difference|",
      y = "Covariate"
    ) +
    theme_minimal()
}

# =========================================================
# 5) Run everything
# =========================================================
df <- read_parquet("output/data/monthly_agg.parquet")

# -------------------
# time_series mode
# -------------------
matches_ts <- risk_set_matching_peak(
  data = df,
  feature_mode = "time_series",
  lookback_months = 12,
  k_neighbors = 5
)

profiles_ts <- build_profiles(
  data = df,
  feature_mode = "time_series",
  lookback_months = 12
)

balance_ts <- balance_table(
  profiles = profiles_ts,
  matches = matches_ts
)

print(balance_ts)
love_plot(balance_ts, title = "Love Plot - Time Series")

# -------------------
# summary mode
# -------------------
matches_summary <- risk_set_matching_peak(
  data = df,
  feature_mode = "summary",
  lookback_months = 12,
  k_neighbors = 5
)

profiles_summary <- build_profiles(
  data = df,
  feature_mode = "summary",
  lookback_months = 12
)

balance_summary <- balance_table(
  profiles = profiles_summary,
  matches = matches_summary
)

print(balance_summary)
love_plot(balance_summary, title = "Love Plot - Summary Statistics")


## create folders if not exist
dirs <- c(
  "output",
  "output/matching",
  "output/diagnostics",
  "output/figures"
)

for (d in dirs) {
  if (!dir.exists(d)) {
    dir.create(d)
  }
}

# Save matching parquet
library(arrow)

write_parquet(
  matches_ts,
  "output/matching/matches_ts.parquet"
)

write_parquet(
  matches_summary,
  "output/matching/matches_summary.parquet"
)

# balance table
write.csv(
  balance_ts,
  "output/diagnostics/balance_ts.csv",
  row.names = FALSE
)

write.csv(
  balance_summary,
  "output/diagnostics/balance_summary.csv",
  row.names = FALSE
)


# love plot
ggsave(
  "output/figures/loveplot_ts.png",
  plot = love_plot(balance_ts, "Love Plot - Time Series"),
  width = 7,
  height = 5,
  dpi = 300
)

ggsave(
  "output/figures/loveplot_summary.png",
  plot = love_plot(balance_summary, "Love Plot - Summary Statistics"),
  width = 7,
  height = 5,
  dpi = 300
)
