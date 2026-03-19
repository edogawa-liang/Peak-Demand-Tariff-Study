library(arrow)
library(lubridate)
library(FNN)

############################################################
# Safe rbind (fill missing columns)
############################################################

bind_rows_fill <- function(rows){
  
  if(length(rows)==0) return(data.frame())
  
  all_cols <- unique(unlist(lapply(rows,function(x){
    if(is.null(dim(x))){
      colnames(as.data.frame(t(x)))
    } else {
      colnames(x)
    }
  })))
  
  out <- lapply(rows,function(x){
    
    if(is.null(dim(x))){
      x <- as.data.frame(t(x))
    }
    
    miss <- setdiff(all_cols,colnames(x))
    
    for(m in miss){
      x[[m]] <- NA
    }
    
    x <- x[,all_cols,drop=FALSE]
    
    return(x)
  })
  
  res <- do.call(rbind,out)
  rownames(res) <- NULL
  return(res)
}

############################################################
# Profile Builder
############################################################

make_profile <- function(user_data,Ti,lookback_months,
                         feature_mode,match_months,summary_vars){
  
  window <- user_data[
    user_data$month >= (Ti %m-% months(lookback_months)) &
      user_data$month < Ti,]
  
  window <- window[order(window$month),]
  
  if(feature_mode=="time_series"){
    
    if(!is.null(match_months)){
      
      window$month_of_year <- month(window$month)
      
      window <- window[
        window$month_of_year %in% match_months,]
      
      if(nrow(window)==0) return(NULL)
      
      window <- window[
        order(window$month,decreasing=TRUE),]
      
      window$lag_id <- ave(
        window$month_of_year,
        window$month_of_year,
        FUN=seq_along)
      
      features <- paste0(
        tolower(month.abb[window$month_of_year]),
        "_lag",
        window$lag_id)
      
      values <- window$top3_mean_consumption
      names(values) <- features
      
      return(values)
    }
    
    if(nrow(window)==0) return(NULL)
    
    values <- window$top3_mean_consumption
    names(values) <- paste0("peak_",seq_along(values))
    
    return(values)
  }
  
  
  if(feature_mode=="summary"){
    
    if(nrow(window)<2) return(NULL)
    

    # if only want to summary selected months
    if(!is.null(match_months)){
      
      window$month_of_year <- month(window$month)
      
      window <- window[
        window$month_of_year %in% match_months,]
      
      if(nrow(window)<2) return(NULL)
    }
    
    # ---- all indicators ----
    
    all_values <- list()
    
    all_values$peak_mean <- mean(window$top3_mean_consumption, na.rm=TRUE)
    all_values$peak_sd <- sd(window$top3_mean_consumption, na.rm=TRUE)
    all_values$peak_volatility <- mean(abs(diff(window$top3_mean_consumption)), na.rm=TRUE)
    
    all_values$mean_consumption <- mean(window$mean_consumption, na.rm=TRUE)
    all_values$variance_consumption <- mean(window$variance_consumption, na.rm=TRUE)
    all_values$total_consumption <- mean(window$total_consumption, na.rm=TRUE)
    
    trend_val <- tryCatch({
      coef(lm(top3_mean_consumption ~ seq_along(top3_mean_consumption),
              data=window))[2]
    }, error=function(e) NA)
    
    all_values$trend <- trend_val
    
    # ==========================
    # selection logic
    # ==========================
    
    # Using Peak if didn't select any
    if(is.null(summary_vars)){
      selected <- all_values[c("peak_mean","peak_sd","peak_volatility")]
      return(unlist(selected))
    }
    
    # Check
    valid_names <- names(all_values)
    
    if(!all(summary_vars %in% valid_names)){
      stop("Invalid summary_vars: ",
           paste(setdiff(summary_vars, valid_names), collapse=", "))
    }
    
    # only use the selected
    selected <- all_values[summary_vars]
    
    return(unlist(selected))
  }
}

############################################################
# Risk-set Matching
############################################################

risk_set_matching_peak <- function(
    data,
    id_col="aID",
    month_col="TIDPUNKT",
    adoption_col="tariff_start",
    lookback_months=12,
    k_neighbors=5,
    blocking_threshold=0.3,
    feature_mode="time_series",
    match_months=NULL,
    summary_vars=NULL){
  
  df <- data
  df$month <- as.Date(df[[month_col]])
  df$adoption_month <- as.Date(df[[adoption_col]])
  
  df <- df[order(df[[id_col]],df$month),]
  
  adopters <- unique(df[!is.na(df$adoption_month),
                        c(id_col,"adoption_month")])
  
  match_rows <- list()
  profile_rows <- list()
  
  for(i in 1:nrow(adopters)){
    
    treated_id <- adopters[[id_col]][i]
    Ti <- adopters$adoption_month[i]
    
    treated_data <- df[df[[id_col]]==treated_id,]
    
    treated_profile <- make_profile(
      treated_data,Ti,lookback_months,
      feature_mode,match_months,summary_vars)
    
    if(is.null(treated_profile)) next
    
    # Not yet choose tariff or never tariff
    controls <- unique(df[
      df[[id_col]]!=treated_id &
        (is.na(df$adoption_month) |
           df$adoption_month>Ti),
    ][[id_col]])
    
    control_profiles <- list()
    control_ids <- c()
    
    # ---- blocking threshold --- #
    if("peak_mean" %in% names(treated_profile)){
      threshold <- blocking_threshold * treated_profile["peak_mean"]
    } else {
      threshold <- Inf
    }
    
    for(cid in controls){
      
      dat <- df[df[[id_col]]==cid,]
      
      prof <- make_profile(
        dat,Ti,lookback_months,
        feature_mode,match_months,summary_vars)
      
      if(!is.null(prof)){
        
        # ===== blocking using peak_mean =====
        if("peak_mean" %in% names(prof) && is.finite(threshold)){
          
          diff <- abs(prof["peak_mean"] - treated_profile["peak_mean"])
          
          if(diff > threshold) next
        }
        
        control_profiles[[length(control_profiles)+1]] <- prof
        control_ids <- c(control_ids,cid)
      }
    }
    
    if(length(control_profiles)==0) next
    
    X_control <- bind_rows_fill(control_profiles)
    
    common_cols <- intersect(colnames(X_control),
                             names(treated_profile))
    
    if(length(common_cols)==0) next
    
    X_control <- X_control[,common_cols,drop=FALSE]
    X_treated <- matrix(treated_profile[common_cols],nrow=1)
    
    keep <- colSums(is.na(X_control))==0
    X_control <- X_control[,keep,drop=FALSE]
    X_treated <- X_treated[,keep,drop=FALSE]
    
    if(ncol(X_control)==0) next
    
    X_control <- scale(X_control)
    
    X_treated <- scale(
      X_treated,
      center=attr(X_control,"scaled:center"),
      scale=attr(X_control,"scaled:scale"))
    
    k_use <- min(k_neighbors,nrow(X_control))
    
    if(k_use==0) next
    
    nn <- get.knnx(X_control,X_treated,k=k_use)
    
    matched_ids <- control_ids[nn$nn.index[1,]]
    
    res <- data.frame(
      treated_id=treated_id,
      control_id=matched_ids,
      adoption_month=Ti,
      distance=nn$nn.dist[1,])
    
    match_rows[[length(match_rows)+1]] <- res
    
    prof_treated <- data.frame(
      id=treated_id,
      group="treated",
      adoption_month=Ti,
      t(treated_profile))
    
    profile_rows[[length(profile_rows)+1]] <- prof_treated
    
    for(j in seq_along(matched_ids)){
      
      prof_control <- data.frame(
        id=matched_ids[j],
        group="control",
        adoption_month=Ti,
        t(control_profiles[[nn$nn.index[1,j]]]))
      
      profile_rows[[length(profile_rows)+1]] <- prof_control
    }
  }
  
  matches <- if(length(match_rows)==0) data.frame()
  else do.call(rbind,match_rows)
  
  profiles <- bind_rows_fill(profile_rows)
  
  return(list(matches=matches,profiles=profiles))
}

############################################################
# Balance Table
############################################################

balance_table <- function(profiles){
  
  if(nrow(profiles)==0) return(data.frame())
  
  treated <- profiles[profiles$group=="treated",]
  control <- profiles[profiles$group=="control",]
  
  covariates <- setdiff(colnames(profiles),
                        c("id","group","adoption_month"))
  
  smd <- function(x,y){
    (mean(x,na.rm=TRUE)-mean(y,na.rm=TRUE)) /
      sqrt((var(x,na.rm=TRUE)+var(y,na.rm=TRUE))/2)
  }
  
  result <- data.frame()
  
  for(v in covariates){
    
    x <- as.numeric(treated[[v]])
    y <- as.numeric(control[[v]])
    
    row <- data.frame(
      covariate=v,
      treated_mean=mean(x,na.rm=TRUE),
      control_mean=mean(y,na.rm=TRUE),
      SMD=smd(x,y))
    
    result <- rbind(result,row)
  }
  
  return(result)
}

############################################################
# Love Plot
############################################################

love_plot <- function(balance,title="Covariate Balance"){
  
  library(ggplot2)
  
  balance$abs_SMD <- abs(balance$SMD)
  balance$covariate[grepl("^trend", balance$covariate)] <- "trend"
  
  ggplot(balance,
         aes(x=abs_SMD,
             y=reorder(covariate,abs_SMD)))+
    geom_point(size=3,color="steelblue")+
    geom_vline(xintercept=0.1,
               linetype="dashed",
               color="red")+
    labs(title=title,
         x="|Standardized Mean Difference|",
         y="Covariate")+
    theme_minimal()
}

# =============================================#

# Start running
  
df <- read_parquet("output/data/monthly_agg.parquet")
df_all <- df[df$price=="all",]

# time series
res_ts <- risk_set_matching_peak(df_all,lookback_months=12)

# summary
res_summary_1 <- risk_set_matching_peak(
  df_all,lookback_months=12,
  feature_mode="summary",
  summary_vars = c(
    "peak_mean",
    "mean_consumption",
    "peak_sd",
    "trend"
  ))

res_summary_2 <- risk_set_matching_peak(
  df_all,lookback_months=12,
  feature_mode="summary",
  summary_vars = c(
    "peak_mean",
    "mean_consumption",
    "variance_consumption",
    "trend"
  ))


res_summary_3 <- risk_set_matching_peak(
  df_all,lookback_months=12,
  feature_mode="summary",
  summary_vars = c(
    "peak_mean",
    "peak_sd",
    "trend"
  ))

# High price period summary
res_summary_season <- risk_set_matching_peak(
  df_all,
  lookback_months = 24,
  feature_mode = "summary",
  match_months = c(1,2,3,11,12),
  summary_vars = c(
    "peak_mean",
    "peak_sd",
    "peak_volatility",
    "trend"
  )
)
# peak_mean, peak_sd, peak_volatility, mean_consumption, variance_consumption, total_consumption, trend

# calendar
res_calendar <- risk_set_matching_peak(
  df_all,
  lookback_months=24,
  match_months=c(1,2,3,11,12))


balance_ts <- balance_table(res_ts$profiles)
balance_summary_1 <- balance_table(res_summary_1$profiles)
balance_summary_2 <- balance_table(res_summary_2$profiles)
balance_summary_3 <- balance_table(res_summary_3$profiles)
balance_summary_season<- balance_table(res_summary_season$profiles)
balance_calendar <- balance_table(res_calendar$profiles)

# =========================
# plot
# =========================
p_ts <- love_plot(balance_ts, "Time Series Matching")
p_s1 <- love_plot(balance_summary_1, "Summary Matching")
p_s2 <- love_plot(balance_summary_2, "Summary Matching")
p_s3 <- love_plot(balance_summary_3, "Summary Matching")
p_s4 <- love_plot(balance_summary_season, "Summary Matching for High Price Period")
p_cal <- love_plot(balance_calendar, "Matching for High Price Period")


# =========================
# save the matching results
# =========================

dir.create("output/matching", recursive = TRUE, showWarnings = FALSE)
dir.create("output/figures", recursive = TRUE, showWarnings = FALSE)
library(arrow)
library(jsonlite)


# =========================
# helper function
# =========================
save_matching <- function(res, balance, config, folder) {
  
  dir.create(folder, recursive = TRUE, showWarnings = FALSE)
  
  write_parquet(res$matches, file.path(folder, "matches.parquet"))
  write_parquet(res$profiles, file.path(folder, "profiles.parquet"))
  write_parquet(balance, file.path(folder, "balance.parquet"))
  
  write_json(config, file.path(folder, "config.json"), pretty = TRUE, auto_unbox = TRUE)
}

# =========================
# time series
# =========================
save_matching(
  res_ts,
  balance_ts,
  list(
    type = "time_series",
    lookback_months = 12
  ),
  "output/matching/time_series"
)

# =========================
# summary 1
# =========================
save_matching(
  res_summary_1,
  balance_summary_1,
  list(
    type = "summary",
    lookback_months = 12,
    vars = c("peak_mean","mean_consumption","peak_sd","trend")
  ),
  "output/matching/summary_1"
)

# =========================
# summary 2
# =========================
save_matching(
  res_summary_2,
  balance_summary_2,
  list(
    type = "summary",
    lookback_months = 12,
    vars = c("peak_mean","mean_consumption","variance_consumption","trend")
  ),
  "output/matching/summary_2"
)

# =========================
# summary 3
# =========================
save_matching(
  res_summary_3,
  balance_summary_3,
  list(
    type = "summary",
    lookback_months = 12,
    vars = c("peak_mean","peak_sd","trend")
  ),
  "output/matching/summary_3"
)

# =========================
# summary with season
# =========================
save_matching(
  res_summary_season,
  balance_summary_season,
  list(
    type = "summary_season",
    lookback_months = 24,
    match_months = c(1,2,3,11,12),
    vars = c("peak_mean","peak_sd","peak_volatility", "trend")
  ),
  "output/matching/summary_season"
)

# =========================
# calendar
# =========================
save_matching(
  res_calendar,
  balance_calendar,
  list(
    type = "calendar",
    lookback_months = 24,
    match_months = c(1,2,3,11,12)
  ),
  "output/matching/calendar"
)


# =========================
# Save plot
# =========================
ggsave("output/figures/love_plot_time_series.png", p_ts, width=6, height=4)
ggsave("output/figures/love_plot_summary_1.png", p_s1, width=6, height=4)
ggsave("output/figures/love_plot_summary_2.png", p_s2, width=6, height=4)
ggsave("output/figures/love_plot_summary_3.png", p_s3, width=6, height=4)
ggsave("output/figures/love_plot_summary_season.png", p_s4, width=6, height=4)
ggsave("output/figures/love_plot_calendar.png", p_cal, width=6, height=4)

