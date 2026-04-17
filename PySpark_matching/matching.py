import os
import json
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from typing import List, Optional, Dict, Any

from pyspark.sql import SparkSession, DataFrame, Window
from pyspark.sql import functions as F


# ============================================================
# Spark / Plot setup
# ============================================================

# 視需要調整
spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")
spark.conf.set("spark.sql.shuffle.partitions", "200")

# ============================================================
# Helpers
# ============================================================

MONTH_ABB = {
    1: "jan", 2: "feb", 3: "mar", 4: "apr",
    5: "may", 6: "jun", 7: "jul", 8: "aug",
    9: "sep", 10: "oct", 11: "nov", 12: "dec"
}


def ensure_folder(path: str):
    """
    在 Fabric Lakehouse Files 下，通常 Spark write parquet 不需要先建資料夾；
    但 matplotlib 存圖常需要目錄存在。
    這裡只對本機風格路徑有幫助；若是 Lakehouse Files，通常也可正常運作。
    """
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def safe_abs_smd(mean_t, mean_c, var_t, var_c):
    denom = ((var_t + var_c) / 2.0) ** 0.5 if (var_t is not None and var_c is not None) else np.nan
    if denom is None or pd.isna(denom) or denom == 0:
        return np.nan
    return (mean_t - mean_c) / denom



# ============================================================
# Profile builders
# ============================================================

def build_summary_profiles_spark(
    risk_rows: DataFrame,
    summary_vars: Optional[List[str]] = None
) -> DataFrame:
    """
    依照 (Ti, id) 建立 summary covariates。
    與你原本邏輯對應：
      - peak_mean
      - peak_sd
      - peak_volatility
      - mean_consumption
      - variance_consumption
      - total_consumption
      - trend
    """
    # 先依月排序，做 x 與 lag
    w_order = Window.partitionBy("Ti", "id").orderBy("month")
    w_prev = Window.partitionBy("Ti", "id").orderBy("month")

    tmp = (
        risk_rows
        .withColumn("x", F.row_number().over(w_order).cast("double"))
        .withColumn("y", F.col("top3_mean_consumption").cast("double"))
        .withColumn("y_prev", F.lag("top3_mean_consumption").over(w_prev))
        .withColumn(
            "abs_diff",
            F.when(
                F.col("y_prev").isNotNull() & F.col("top3_mean_consumption").isNotNull(),
                F.abs(F.col("top3_mean_consumption") - F.col("y_prev"))
            ).otherwise(F.lit(None).cast("double"))
        )
    )

    prof = (
        tmp
        .groupBy("Ti", "id", "adoption_month", "group")
        .agg(
            F.count("*").alias("n_obs"),
            F.avg("top3_mean_consumption").alias("peak_mean"),
            F.stddev_samp("top3_mean_consumption").alias("peak_sd"),
            F.avg("abs_diff").alias("peak_volatility"),
            F.avg("mean_consumption").alias("mean_consumption"),
            F.avg("variance_consumption").alias("variance_consumption"),
            F.avg("total_consumption").alias("total_consumption"),
            F.sum("x").alias("sum_x"),
            F.sum("y").alias("sum_y"),
            F.sum(F.col("x") * F.col("y")).alias("sum_xy"),
            F.sum(F.col("x") * F.col("x")).alias("sum_x2")
        )
        .where(F.col("n_obs") >= 2)
        .withColumn(
            "trend",
            F.when(
                (F.col("n_obs") * F.col("sum_x2") - F.col("sum_x") * F.col("sum_x")) != 0,
                (
                    (F.col("n_obs") * F.col("sum_xy") - F.col("sum_x") * F.col("sum_y")) /
                    (F.col("n_obs") * F.col("sum_x2") - F.col("sum_x") * F.col("sum_x"))
                )
            ).otherwise(F.lit(None).cast("double"))
        )
    )

    available_vars = [
        "peak_mean",
        "peak_sd",
        "peak_volatility",
        "mean_consumption",
        "variance_consumption",
        "total_consumption",
        "trend"
    ]

    if summary_vars is None:
        summary_vars = ["peak_mean", "peak_sd", "peak_volatility"]

    invalid = [x for x in summary_vars if x not in available_vars]
    if invalid:
        raise ValueError("Invalid summary_vars: " + ", ".join(invalid))

    keep_cols = ["Ti", "id", "adoption_month", "group"] + summary_vars
    return prof.select(*keep_cols)



# =========== Calendar Vector Matching =========== #
def build_calendar_aligned_profiles(
    risk_rows: DataFrame,
    match_months: List[int],
    n_years: int = 2
) -> DataFrame:

    df = (
        risk_rows
        .withColumn("year", F.year("month"))
        .withColumn("month_num", F.month("month"))
    )

    # 👉 距離 Ti 幾年前
    df = df.withColumn(
        "year_diff",
        F.year("Ti") - F.col("year")
    )

    # 👉 只保留過去 n_years
    df = df.where(
        (F.col("year_diff") >= 1) &
        (F.col("year_diff") <= n_years)
    )

    # 👉 只保留指定月份
    df = df.where(F.col("month_num").isin(match_months))

    # 👉 month number → 字串（jan, feb…）
    mapping_expr = F.create_map(
        *[item for i in MONTH_ABB for item in (F.lit(i), F.lit(MONTH_ABB[i]))]
    )

    df = df.withColumn("month_str", mapping_expr[F.col("month_num")])

    # 👉 feature name（jan_1, jan_2）
    df = df.withColumn(
        "feature_name",
        F.concat(F.col("month_str"), F.lit("_"), F.col("year_diff"))
    )

    # 👉 pivot 成 wide format
    prof = (
        df
        .groupBy("Ti", "id", "adoption_month", "group")
        .pivot("feature_name")
        .agg(F.first("top3_mean_consumption"))
    )

    return prof


def build_time_series_profiles_spark(
    risk_rows: DataFrame,
    lookback_months: int = 12,
    min_valid_dim: int = 10
) -> DataFrame:
    """
    固定長度 time-series profile（允許 missing）

    peak_lag_1 = Ti 前最近一個月
    peak_lag_12 = Ti 前第12個月

    matching 時：
    - 允許 missing
    - 至少需要 min_valid_dim 個有效月
    """

    # ============================================================
    # Step 1: 建立 lag index（由近到遠）
    # ============================================================
    w_desc = Window.partitionBy("Ti", "id").orderBy(F.col("month").desc())

    tmp = (
        risk_rows
        .withColumn("lag_idx", F.row_number().over(w_desc))
        .where(F.col("lag_idx") <= lookback_months)
    )

    # ============================================================
    # Step 2: pivot 成 wide format
    # ============================================================
    prof = (
        tmp
        .groupBy("Ti", "id", "adoption_month", "group")
        .pivot("lag_idx", list(range(1, lookback_months + 1)))
        .agg(F.first("top3_mean_consumption"))
    )

    # ============================================================
    # Step 3: rename columns（確保完整 lag 結構）
    # ============================================================
    for i in range(1, lookback_months + 1):
        if str(i) in prof.columns:
            prof = prof.withColumnRenamed(str(i), f"peak_lag_{i}")
        else:
            prof = prof.withColumn(f"peak_lag_{i}", F.lit(None).cast("double"))

    lag_cols = [f"peak_lag_{i}" for i in range(1, lookback_months + 1)]

    # ============================================================
    # Step 4: 計算有效維度（non-missing）
    # ============================================================
    valid_expr = None
    for c in lag_cols:
        this = F.col(c).isNotNull().cast("int")
        valid_expr = this if valid_expr is None else (valid_expr + this)

    prof = prof.withColumn("valid_dim", valid_expr)

    # ============================================================
    # Step 5: 篩選有效樣本
    # ============================================================
    prof = prof.where(F.col("valid_dim") >= min_valid_dim)

    # ============================================================
    # Step 6: 回傳（保留 valid_dim）
    # ============================================================
    return prof.select(
        "Ti",
        "id",
        "adoption_month",
        "group",
        *lag_cols,
        "valid_dim"
    )


# ============================================================
# Standardization (based on controls within cohort Ti)
# ============================================================

def standardize_by_control(
    profiles: DataFrame,
    feature_cols: List[str]
) -> DataFrame:
    """
    模仿 R:
      X_control <- scale(X_control)
      X_treated <- scale(X_treated, center=control_mean, scale=control_sd)
    """
    agg_exprs = []
    for c in feature_cols:
        agg_exprs.append(F.avg(F.col(c)).alias(f"{c}_mean"))
        agg_exprs.append(F.stddev_samp(F.col(c)).alias(f"{c}_sd"))

    control_stats = (
        profiles
        .where(F.col("group") == "control")
        .groupBy("Ti")
        .agg(*agg_exprs)
    )

    out = profiles.join(control_stats, on="Ti", how="left")

    for c in feature_cols:
        out = out.withColumn(
            f"{c}_z",
            (
                F.col(c) - F.col(f"{c}_mean")
            ) / F.when(
                F.col(f"{c}_sd").isNull() | (F.col(f"{c}_sd") == 0),
                F.lit(1.0)
            ).otherwise(F.col(f"{c}_sd"))
        )

    keep_cols = ["Ti", "id", "adoption_month", "group"] + feature_cols + [f"{c}_z" for c in feature_cols]
    return out.select(*keep_cols)


# ============================================================
# Matching
# ============================================================

def match_topk_spark(
    profiles_z: DataFrame,
    feature_cols: List[str],
    k_neighbors: int = 5,
    blocking_threshold: float = 0.3,
    use_peak_blocking: bool = True,
    peak_block_col: str = "peak_mean"
) -> DataFrame:
    """
    全 Spark KNN matching:
      1. 同 cohort Ti 的 treated / controls join
      2. optional blocking
      3. 歐氏距離
      4. row_number 取 top-k
    """
    treated = profiles_z.where(F.col("group") == "treated").alias("t")
    control = profiles_z.where(F.col("group") == "control").alias("c")

    join_cond = [F.col("t.Ti") == F.col("c.Ti")]

    if use_peak_blocking and peak_block_col in feature_cols:
        join_cond.append(
            F.abs(F.col(f"c.{peak_block_col}") - F.col(f"t.{peak_block_col}")) <=
            (F.lit(blocking_threshold) * F.abs(F.col(f"t.{peak_block_col}")))
        )

    cand = treated.join(control, on=join_cond, how="inner").where(F.col("t.id") != F.col("c.id"))

    # remove rows with any null z-score
    for c in feature_cols:
        cand = cand.where(F.col(f"t.{c}_z").isNotNull() & F.col(f"c.{c}_z").isNotNull())

    # Euclidean distance
    dist_expr = None
    for c in feature_cols:
        term = F.pow(F.col(f"t.{c}_z") - F.col(f"c.{c}_z"), 2)
        dist_expr = term if dist_expr is None else dist_expr + term

    cand = cand.withColumn("distance", F.sqrt(dist_expr))

    w = Window.partitionBy(F.col("t.Ti"), F.col("t.id")).orderBy(F.col("distance").asc(), F.col("c.id").asc())

    matches = (
        cand
        .withColumn("match_rank", F.row_number().over(w))
        .where(F.col("match_rank") <= k_neighbors)
        .select(
            F.col("t.id").alias("treated_id"),
            F.col("c.id").alias("control_id"),
            F.col("t.Ti").alias("adoption_month"),
            F.col("distance"),
            F.col("match_rank")
        )
    )

    return matches



# ============================================================
# Matched profiles for balance
# ============================================================

def build_matched_profiles(
    profiles_z: DataFrame,
    matches: DataFrame
) -> DataFrame:
    """
    只保留實際被匹配到的 treated / control profiles，用來做 balance table。
    """
    treated_ids = (
        matches
        .select(
            F.col("treated_id").alias("id"),
            F.col("adoption_month").alias("Ti")
        )
        .dropDuplicates()
        .withColumn("matched_group", F.lit("treated"))
    )

    control_ids = (
        matches
        .select(
            F.col("control_id").alias("id"),
            F.col("adoption_month").alias("Ti")
        )
        .dropDuplicates()
        .withColumn("matched_group", F.lit("control"))
    )

    matched_ids = treated_ids.unionByName(control_ids)

    out = (
        profiles_z
        .join(matched_ids, on=["id", "Ti"], how="inner")
        .drop("group")
        .withColumnRenamed("matched_group", "group")
    )

    return out


# ============================================================
# Balance table
# ============================================================

def balance_table_spark(
    profiles: DataFrame,
    feature_cols: List[str],
    pool: bool = True   # 👈 新增
) -> DataFrame:

    if not pool:
        # ====================================================
        # 🔥 BY Ti（最重要）
        # ====================================================
        treated = profiles.where(F.col("group") == "treated")
        control = profiles.where(F.col("group") == "control")

        t_agg = treated.groupBy("Ti").agg(*(
            [F.avg(c).alias(f"{c}_t_mean") for c in feature_cols] +
            [F.var_samp(c).alias(f"{c}_t_var") for c in feature_cols]
        ))

        c_agg = control.groupBy("Ti").agg(*(
            [F.avg(c).alias(f"{c}_c_mean") for c in feature_cols] +
            [F.var_samp(c).alias(f"{c}_c_var") for c in feature_cols]
        ))

        row = t_agg.join(c_agg, on="Ti")

        pieces = []
        for c in feature_cols:
            piece = row.select(
                "Ti",
                F.lit(c).alias("covariate"),
                F.col(f"{c}_t_mean").alias("treated_mean"),
                F.col(f"{c}_c_mean").alias("control_mean"),
                (
                    (F.col(f"{c}_t_mean") - F.col(f"{c}_c_mean")) /
                    F.sqrt((F.col(f"{c}_t_var") + F.col(f"{c}_c_var")) / 2.0)
                ).alias("SMD")
            )
            pieces.append(piece)

        out = pieces[0]
        for p in pieces[1:]:
            out = out.unionByName(p)

        return out

    else:
        # ====================================================
        # 🔥 OVERALL（你原本的）
        # ====================================================
        treated = profiles.where(F.col("group") == "treated")
        control = profiles.where(F.col("group") == "control")

        t_agg = treated.agg(*(
            [F.avg(c).alias(f"{c}_t_mean") for c in feature_cols] +
            [F.var_samp(c).alias(f"{c}_t_var") for c in feature_cols]
        ))

        c_agg = control.agg(*(
            [F.avg(c).alias(f"{c}_c_mean") for c in feature_cols] +
            [F.var_samp(c).alias(f"{c}_c_var") for c in feature_cols]
        ))

        row = t_agg.crossJoin(c_agg)

        pieces = []
        for c in feature_cols:
            piece = row.select(
                F.lit(c).alias("covariate"),
                F.col(f"{c}_t_mean").alias("treated_mean"),
                F.col(f"{c}_c_mean").alias("control_mean"),
                (
                    (F.col(f"{c}_t_mean") - F.col(f"{c}_c_mean")) /
                    F.sqrt((F.col(f"{c}_t_var") + F.col(f"{c}_c_var")) / 2.0)
                ).alias("SMD")
            )
            pieces.append(piece)

        out = pieces[0]
        for p in pieces[1:]:
            out = out.unionByName(p)

        return out


# ============================================================
# Love plot
# ============================================================

def love_plot_from_spark(
    balance_spark: DataFrame,
    output_path: Optional[str] = None,
    title: str = "Covariate Balance",
    show_plot: bool = True   # 👈 新增
):
    pdf = balance_spark.toPandas()

    fig, ax = plt.subplots(figsize=(7, 4.5))

    if pdf is None or len(pdf) == 0:
        ax.set_title(title)
    else:
        plot_df = pdf.copy()
        plot_df["abs_SMD"] = pd.to_numeric(plot_df["SMD"], errors="coerce").abs()
        plot_df = plot_df.sort_values("abs_SMD")

        ax.scatter(plot_df["abs_SMD"], plot_df["covariate"], s=40)
        ax.axvline(0.1, linestyle="--")

    ax.set_title(title)
    ax.set_xlabel("|Standardized Mean Difference|")
    ax.set_ylabel("Covariate")

    plt.tight_layout()

    # 👇 存檔（Fabric）
    if output_path is not None:
        real_path = output_path.replace("Files/", "/lakehouse/default/Files/")
        os.makedirs(os.path.dirname(real_path), exist_ok=True)
        fig.savefig(real_path, dpi=150, bbox_inches="tight")

    # 👇 顯示在 notebook
    if show_plot:
        plt.show()

        plt.close(fig)


# ============================================================
# Save helpers
# ============================================================

def save_config(config: Dict[str, Any], output_json_path: str):
    ensure_folder(output_json_path)
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2, default=str)


def save_matching_outputs(
    matches: DataFrame,
    profiles: DataFrame,
    balance: DataFrame,
    config: Dict[str, Any],
    folder: str
):
    matches.write.mode("overwrite").parquet(f"{folder}/matches")
    profiles.write.mode("overwrite").parquet(f"{folder}/profiles")
    balance.write.mode("overwrite").parquet(f"{folder}/balance")
    save_config(config, f"{folder}/config.json")



# ============================================================
# Main pipelines
# ============================================================

def run_summary_matching_pipeline(
    output_folder: str,
    lookback_months: int = 12,
    k_neighbors: int = 5,
    blocking_threshold: float = 0.3,
    summary_vars: Optional[List[str]] = None,
    min_ti: Optional[str] = None,
    max_ti: Optional[str] = None,
    repartition_by_ti: bool = True,
    verbose: bool = True,
    match_months: Optional[List[int]] = None,
    save_output: bool = False,
    control_type: str = "never_treated",
    risk_rows: DataFrame = None
) -> Dict[str, DataFrame]:

    if risk_rows is None:
        raise ValueError("risk_rows must be provided. Use prepare_matching_data first.")
    
    if summary_vars is None:
        summary_vars = ["peak_mean", "mean_consumption", "peak_sd", "trend"]

    if verbose:
        print("risk_rows count =", risk_rows.count())
        risk_rows.groupBy("Ti", "group").count().orderBy("Ti", "group").show(50, truncate=False)

    if verbose:
        print("Building summary profiles ...")


    profiles = build_summary_profiles_spark(risk_rows, summary_vars=summary_vars)
    if repartition_by_ti:
        profiles = profiles.repartition("Ti")
    profiles = profiles.cache()

    if verbose:
        print("profiles count =", profiles.count())

    if verbose:
        print("Standardizing by controls ...")
    profiles_z = standardize_by_control(profiles, summary_vars)
    if repartition_by_ti:
        profiles_z = profiles_z.repartition("Ti")
    profiles_z = profiles_z.cache()

    if verbose:
        print("profiles_z count =", profiles_z.count())

    if verbose:
        print("Matching top-k ...")
    matches = match_topk_spark(
        profiles_z=profiles_z,
        feature_cols=summary_vars,
        k_neighbors=k_neighbors,
        blocking_threshold=blocking_threshold,
        use_peak_blocking=("peak_mean" in summary_vars),
        peak_block_col="peak_mean"
    )
    if repartition_by_ti:
        matches = matches.repartition("adoption_month")
    matches = matches.cache()

    if verbose:
        print("matches count =", matches.count())

    if verbose:
        print("Building matched profiles ...")
    matched_profiles = build_matched_profiles(profiles_z, matches).cache()

    if verbose:
        print("matched_profiles count =", matched_profiles.count())

    if verbose:
        print("Computing balance table ...")

    # balance = balance_table_spark(matched_profiles, summary_vars).cache()
    balance_overall = balance_table_spark(
    matched_profiles,
    summary_vars,
    pool=True
    )
    balance_by_ti = balance_table_spark(
        matched_profiles,
        summary_vars,
        pool=False
    )

    if verbose:
        print("balance_overall count =", balance_overall.count())
        balance_overall.show(50, truncate=False)
        balance_by_ti.show(50, truncate=False)

    if verbose:
        print("Saving outputs ...")

    if save_output:
        save_matching_outputs(
            matches=matches,
            profiles=matched_profiles,
            balance=balance,
            config={
                "type": "summary",
                "lookback_months": lookback_months,
                "k_neighbors": k_neighbors,
                "blocking_threshold": blocking_threshold,
                "summary_vars": summary_vars,
                "min_ti": min_ti,
                "max_ti": max_ti,
                "control_type": control_type
            },
            folder=output_folder
        )

    return {
        "risk_rows": risk_rows,
        "profiles": profiles_z,
        "matches": matches,
        "matched_profiles": matched_profiles,
        "balance": balance_overall,        
        "balance_by_ti": balance_by_ti,   
        "match_vars": summary_vars
    }


def run_time_series_matching_pipeline(
    output_folder: str,
    lookback_months: int = 12,
    k_neighbors: int = 5,
    min_ti: Optional[str] = None,
    max_ti: Optional[str] = None,
    repartition_by_ti: bool = True,
    verbose: bool = True,
    match_months: Optional[List[int]] = None,
    save_output: bool = False,
    control_type: str = "never_treated",
    risk_rows: DataFrame = None
) -> Dict[str, DataFrame]:

    feature_cols = [f"peak_lag_{i}" for i in range(1, lookback_months + 1)]

    if risk_rows is None:
        raise ValueError("risk_rows must be provided.")

    if verbose:
        print("risk_rows count =", risk_rows.count())
        risk_rows.groupBy("Ti", "group").count().orderBy("Ti", "group").show(50, truncate=False)

    if verbose:
        print("Building fixed-lag time series profiles ...")
    profiles = build_time_series_profiles_spark(
        risk_rows=risk_rows,
        lookback_months=lookback_months
    )
    if repartition_by_ti:
        profiles = profiles.repartition("Ti")
    profiles = profiles.cache()

    if verbose:
        print("profiles count =", profiles.count())

    if verbose:
        print("Standardizing by controls ...")
    profiles_z = standardize_by_control(profiles, feature_cols)
    if repartition_by_ti:
        profiles_z = profiles_z.repartition("Ti")
    profiles_z = profiles_z.cache()

    if verbose:
        print("profiles_z count =", profiles_z.count())

    if verbose:
        print("Matching top-k ...")

    # time series 沒有 peak_mean，因此不做 peak blocking
    # matches = match_topk_spark(
    #     profiles_z=profiles_z,
    #     feature_cols=feature_cols,
    #     k_neighbors=k_neighbors,
    #     blocking_threshold=0.3,
    #     use_peak_blocking=False
    # )

    matches = match_topk_allow_missing(
    profiles_z,
    feature_cols,
    k_neighbors=k_neighbors,
    min_valid_dim=10
)
    if repartition_by_ti:
        matches = matches.repartition("adoption_month")
    matches = matches.cache()

    if verbose:
        print("matches count =", matches.count())

    if verbose:
        print("Building matched profiles ...")
    matched_profiles = build_matched_profiles(profiles_z, matches).cache()

    if verbose:
        print("matched_profiles count =", matched_profiles.count())

    if verbose:
        print("Computing balance table ...")

    balance_overall = balance_table_spark(
    matched_profiles,
    feature_cols,
    pool=True
    ).cache()

    balance_by_ti = balance_table_spark(
        matched_profiles,
        feature_cols,
        pool=False
    ).cache()

    if verbose:
        print("balance_overall count =", balance_overall.count())
        balance_overall.show(50, truncate=False)
        print("balance_by_ti count =", balance_by_ti.count())
        balance_by_ti.show(50, truncate=False)

    if verbose:
        print("Saving outputs ...")


    if save_output:
        save_matching_outputs(
            matches=matches,
            profiles=matched_profiles,
            balance=balance_overall,
            config={
                "type": "time_series_fixed_lag",
                "lookback_months": lookback_months,
                "k_neighbors": k_neighbors,
                "feature_cols": feature_cols,
                "min_ti": min_ti,
                "max_ti": max_ti,
                "control_type": control_type
            },
            folder=output_folder
        )

    return {
        "risk_rows": risk_rows,
        "profiles": profiles_z,
        "matches": matches,
        "matched_profiles": matched_profiles,
        "balance": balance_overall,
        "balance_by_ti": balance_by_ti,
        "match_vars": feature_cols  
    }



def save_matching_results_fabric(
    res: dict,
    folder: str,
    config: Optional[dict] = None,
    save_plot: bool = True,
):
    import os
    import json

    if not folder.startswith("Files/"):
        raise ValueError("Folder must start with 'Files/' in Fabric")

    matches = res["matches"]
    profiles = res["matched_profiles"]
    balance = res["balance"]

    # ============================================================
    # Spark parquet（不用建資料夾）
    # ============================================================
    print(f"Saving parquet to {folder} ...")

    matches.write.mode("overwrite").parquet(f"{folder}/matches")
    profiles.write.mode("overwrite").parquet(f"{folder}/profiles")
    balance.write.mode("overwrite").parquet(f"{folder}/balance")

    # ============================================================
    # config（需要建資料夾）
    # ============================================================
    if config is not None:
        config_path = f"{folder}/config.json"

        os.makedirs(os.path.dirname(config_path), exist_ok=True)

        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)


    print("✅ Save completed")





def match_topk_allow_missing(
    profiles_z: DataFrame,
    feature_cols: List[str],
    k_neighbors: int = 5,
    min_valid_dim: int = 6,      # 👈 新增
    verbose: bool = True         # 👈 新增
) -> DataFrame:

    treated = profiles_z.where(F.col("group") == "treated").alias("t")
    control = profiles_z.where(F.col("group") == "control").alias("c")

    cand = (
        treated.join(control, on=[F.col("t.Ti") == F.col("c.Ti")])
        .where(F.col("t.id") != F.col("c.id"))
    )

    dist_expr = None
    valid_count = None

    for c in feature_cols:
        both_not_null = (
            F.col(f"t.{c}_z").isNotNull() &
            F.col(f"c.{c}_z").isNotNull()
        )

        diff_sq = F.when(
            both_not_null,
            F.pow(F.col(f"t.{c}_z") - F.col(f"c.{c}_z"), 2)
        ).otherwise(F.lit(0.0))

        count = F.when(both_not_null, 1).otherwise(0)

        dist_expr = diff_sq if dist_expr is None else dist_expr + diff_sq
        valid_count = count if valid_count is None else valid_count + count

    cand = cand.withColumn("valid_dim", valid_count)

    # ============================================================
    # 🔥 印出維度分布（很重要）
    # ============================================================
    if verbose:
        print("=== VALID DIMENSION DISTRIBUTION ===")
        (
            cand
            .groupBy("valid_dim")
            .count()
            .orderBy("valid_dim")
            .show(50, truncate=False)
        )

    # ============================================================
    # 🔥 篩選至少 min_valid_dim 維
    # ============================================================
    cand = cand.where(F.col("valid_dim") >= min_valid_dim)

    # ============================================================
    # 🔥 用「平均距離」
    # ============================================================
    cand = cand.withColumn(
        "distance",
        F.sqrt(dist_expr / F.col("valid_dim"))
    )

    # ============================================================
    # matching
    # ============================================================
    w = Window.partitionBy(F.col("t.Ti"), F.col("t.id")) \
              .orderBy(F.col("distance"), F.col("c.id"))

    matches = (
        cand
        .withColumn("match_rank", F.row_number().over(w))
        .where(F.col("match_rank") <= k_neighbors)
        .select(
            F.col("t.id").alias("treated_id"),
            F.col("c.id").alias("control_id"),
            F.col("t.Ti").alias("adoption_month"),
            "distance",
            "valid_dim",
            "match_rank"
        )
    )

    return matches


def run_calendar_matching_aligned(
    output_folder: str,
    lookback_years: int = 2,
    match_months: List[int] = [1,2,3,11,12],
    k_neighbors: int = 5,
    repartition_by_ti: bool = True,
    verbose: bool = True,
    save_output: bool = False,
    control_type: str = "never_treated",
    risk_rows = None,
) -> Dict[str, DataFrame]:

    if verbose:
        print("Running calendar-aligned matching...")

    if risk_rows is None:
        raise ValueError("risk_rows must be provided.")

    if verbose:
        print("risk_rows count =", risk_rows.count())
        risk_rows.groupBy("Ti", "group").count().orderBy("Ti", "group").show(50, truncate=False)

    # ============================================================
    # profiles（calendar aligned）
    # ============================================================
    if verbose:
        print("Building calendar-aligned profiles ...")

    profiles = build_calendar_aligned_profiles(
        risk_rows,
        match_months=match_months,
        n_years=lookback_years
    )
    feature_cols = sorted([
        c for c in profiles.columns
        if c not in ["Ti", "id", "adoption_month", "group"]
    ])

    if repartition_by_ti:
        profiles = profiles.repartition("Ti")

    profiles = profiles.cache()

    if verbose:
        print("profiles count =", profiles.count())

    # ============================================================
    # standardize
    # ============================================================
    if verbose:
        print("Standardizing by controls ...")

    profiles_z = standardize_by_control(profiles, feature_cols)

    if repartition_by_ti:
        profiles_z = profiles_z.repartition("Ti")

    profiles_z = profiles_z.cache()

    if verbose:
        print("profiles_z count =", profiles_z.count())

    # ============================================================
    # matching
    # ============================================================
    if verbose:
        print("Matching top-k (allow missing) ...")

    matches = match_topk_allow_missing(
        profiles_z,
        feature_cols,
        k_neighbors=k_neighbors,
        min_valid_dim=6
    )

    matches = matches.cache()

    if verbose:
        print("matches count =", matches.count())

    # ============================================================
    # matched profiles + balance
    # ============================================================
    if verbose:
        print("Building matched profiles ...")

    matched_profiles = build_matched_profiles(profiles_z, matches).cache()

    if verbose:
        print("matched_profiles count =", matched_profiles.count())

    if verbose:
        print("Computing balance table ...")

    balance_overall = balance_table_spark(
    matched_profiles,
    feature_cols,
    pool=True
    ).cache()

    balance_by_ti = balance_table_spark(
        matched_profiles,
        feature_cols,
        pool=False
    ).cache()
    
    if verbose:
        print("balance_overall count =", balance_overall.count())
        balance_overall.show(50, truncate=False)
        print("balance_by_ti count =", balance_by_ti.count())
        balance_by_ti.show(50, truncate=False)

    # ============================================================
    # save
    # ============================================================
    if verbose:
        print("Saving outputs ...")

    if save_output:
        save_matching_outputs(
            matches=matches,
            profiles=matched_profiles,
            balance=balance_overall,
            config={
                "type": "calendar_aligned",
                "lookback_years": lookback_years,
                "match_months": match_months,
                "feature_cols": feature_cols,
                "control_type": control_type
            },
            folder=output_folder
        )

    return {
        "risk_rows": risk_rows,
        "profiles": profiles_z,
        "matches": matches,
        "matched_profiles": matched_profiles,
        "balance": balance_overall,
        "balance_by_ti": balance_by_ti,
        "match_vars": feature_cols   
    }



