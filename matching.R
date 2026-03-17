############################################################
# Calendar Month Risk-Set Matching
############################################################

library(arrow)
library(dplyr)
library(lubridate)
library(FNN)
library(purrr)
library(tidyr)
library(ggplot2)
library(tibble)

############################################################
# Helper: build calendar-month features
############################################################

build_calendar_features <- function(window, match_months){
  
  window <- window %>%
    mutate(
      month_of_year = month(month),
      year_val = year(month)
    ) %>%
    filter(month_of_year %in% match_months)
  
  if(nrow(window) == 0) return(NULL)
  
  window <- window %>%
    arrange(desc(month)) %>%
    group_by(month_of_year) %>%
    mutate(lag_id = row_number()) %>%
    ungroup()
  
  features <- window %>%
    mutate(
      feature = paste0(
        tolower(month.abb[month_of_year]),
        "_lag",
        lag_id
      )
    ) %>%
    select(feature, top3_mean_consumption)
  
  values <- features$top3_mean_consumption
  names(values) <- features$feature
  
  return(values)
}

############################################################
# Risk-set matching
############################################################

risk_set_matching_peak <- function(
    data,
    id_col = "aID",
    month_col = "TIDPUNKT",
    adoption_col = "tariff_start",
    lookback_months = 24,
    k_neighbors = 5,
    price_filter = "all",
    match_months = c(1,6,12)
){
  
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
      )
    
    feats <- build_calendar_features(window, match_months)
    
    if(is.null(feats)) return(NULL)
    
    tibble(
      id = unique(user_data[[id_col]])[1],
      !!!as.list(feats)
    )
  }
  
  results <- map_dfr(seq_len(nrow(adopters)), function(i){
    
    treated_id <- adopters[[id_col]][i]
    Ti <- adopters$adoption_month[i]
    
    treated_data <- df %>%
      filter(.data[[id_col]] == treated_id)
    
    treated_profile <- build_profile(treated_data, Ti)
    
    if(is.null(treated_profile)) return(NULL)
    
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
      
      if(is.null(prof)) return(NULL)
      
      prof %>%
        mutate(control_id = cid)
    })
    
    if(nrow(control_profiles) == 0) return(NULL)
    
    feature_cols <- setdiff(names(control_profiles), c("id","control_id"))
    
    X_control <- control_profiles %>%
      select(all_of(feature_cols)) %>%
      scale()
    
    X_treated <- treated_profile %>%
      select(all_of(feature_cols)) %>%
      scale(
        center = attr(X_control,"scaled:center"),
        scale = attr(X_control,"scaled:scale")
      )
    
    k_use <- min(k_neighbors,nrow(control_profiles))
    
    nn <- get.knnx(X_control,X_treated,k=k_use)
    
    matched <- control_profiles[nn$nn.index[1,],,drop=FALSE]
    
    matched %>%
      mutate(
        treated_id = treated_id,
        adoption_month = Ti,
        distance = nn$nn.dist[1,]
      )
  })
  
  return(results)
}

############################################################
# Build profiles for balance diagnostics
############################################################

build_profiles <- function(
    data,
    id_col = "aID",
    month_col = "TIDPUNKT",
    adoption_col = "tariff_start",
    lookback_months = 24,
    price_filter = "all",
    match_months = c(1,6,12)
){
  
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
      )
    
    feats <- build_calendar_features(window, match_months)
    
    if(is.null(feats)) return(NULL)
    
    tibble(
      id = unique(user_data[[id_col]])[1],
      !!!as.list(feats)
    )
  }
  
  profiles <- map_dfr(seq_len(nrow(adopters)), function(i){
    
    uid <- adopters[[id_col]][i]
    Ti <- adopters$adoption_month[i]
    
    user_data <- df %>%
      filter(.data[[id_col]] == uid)
    
    prof <- build_profile(user_data, Ti)
    
    if(is.null(prof)) return(NULL)
    
    prof %>%
      mutate(adoption_month = Ti)
  })
  
  return(profiles)
}

############################################################
# Balance table
############################################################

balance_table <- function(profiles, matches){
  
  treated_ids <- unique(matches$treated_id)
  
  treated <- profiles %>%
    filter(id %in% treated_ids)
  
  control <- matches %>%
    select(control_id) %>%
    rename(id = control_id) %>%
    left_join(profiles,by="id")
  
  covariates <- setdiff(names(profiles),c("id","adoption_month"))
  
  smd <- function(x,y){
    (mean(x,na.rm=TRUE)-mean(y,na.rm=TRUE)) /
      sqrt((var(x,na.rm=TRUE)+var(y,na.rm=TRUE))/2)
  }
  
  balance <- lapply(covariates,function(v){
    
    tibble(
      covariate=v,
      treated_mean=mean(treated[[v]],na.rm=TRUE),
      control_mean=mean(control[[v]],na.rm=TRUE),
      SMD=smd(treated[[v]],control[[v]])
    )
  }) %>%
    bind_rows()
  
  return(balance)
}

############################################################
# Love plot
############################################################

love_plot <- function(balance,title="Covariate Balance"){
  
  ggplot(balance,
         aes(x=abs(SMD),
             y=reorder(covariate,abs(SMD))))+
    geom_point(size=3,color="steelblue")+
    geom_vline(xintercept=0.1,
               linetype="dashed",
               color="red")+
    labs(
      title=title,
      x="|Standardized Mean Difference|",
      y="Covariate"
    )+
    theme_minimal()
}

############################################################
# Run everything
############################################################

df <- read_parquet("output/data/monthly_agg.parquet")

matches <- risk_set_matching_peak(
  data=df,
  lookback_months=24,
  match_months=c(1,6,12),
  k_neighbors=5
)

profiles <- build_profiles(
  data=df,
  lookback_months=24,
  match_months=c(1,6,12)
)

balance <- balance_table(
  profiles=profiles,
  matches=matches
)

print(balance)

love_plot(balance,"Love Plot - Calendar Month Matching")

############################################################
# Save outputs
############################################################

dirs <- c(
  "output",
  "output/matching",
  "output/diagnostics",
  "output/figures"
)

for(d in dirs){
  if(!dir.exists(d)){
    dir.create(d)
  }
}

write_parquet(
  matches,
  "output/matching/matches_calendar.parquet"
)

write.csv(
  balance,
  "output/diagnostics/balance_calendar.csv",
  row.names=FALSE
)

ggsave(
  "output/figures/loveplot_calendar.png",
  plot=love_plot(balance,"Love Plot - Calendar Matching"),
  width=7,
  height=5,
  dpi=300
)