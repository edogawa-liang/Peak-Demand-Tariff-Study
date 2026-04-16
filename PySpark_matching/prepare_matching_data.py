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
# Helpers
# ============================================================

MONTH_ABB = {
    1: "jan", 2: "feb", 3: "mar", 4: "apr",
    5: "may", 6: "jun", 7: "jul", 8: "aug",
    9: "sep", 10: "oct", 11: "nov", 12: "dec"
}

# ============================================================
# Data Preparation
# ============================================================

def prepare_base_spark(
    sdf: DataFrame,
    id_col: str = "aID",
    month_col: str = "TIDPUNKT",
    adoption_col: str = "tariff_start",
    price_col: Optional[str] = "price",
    price_value: Optional[str] = "all"
) -> DataFrame:
    """
    標準化欄位名稱與型別，保留 matching 需要的欄位。
    """
    out = sdf

    if price_col is not None and price_value is not None:
        if price_col in sdf.columns:
            out = out.filter(F.col(price_col) == F.lit(price_value))

    out = (
        out
        .withColumn("id", F.col(id_col).cast("string"))
        .withColumn("month", F.to_date(F.col(month_col)))
        .withColumn("adoption_month", F.to_date(F.col(adoption_col)))
        .withColumn("top3_mean_consumption", F.col("top3_mean_consumption").cast("double"))
        .withColumn("mean_consumption", F.col("mean_consumption").cast("double"))
        .withColumn("variance_consumption", F.col("variance_consumption").cast("double"))
        .withColumn("total_consumption", F.col("total_consumption").cast("double"))
        .select(
            "id",
            "month",
            "adoption_month",
            "top3_mean_consumption",
            "mean_consumption",
            "variance_consumption",
            "total_consumption"
        )
        .filter(F.col("id").isNotNull())
        .filter(F.col("month").isNotNull())
    )

    return out


# ============================================================
# Cohort / Risk-set construction
# ============================================================

def get_distinct_ti(
    base_df: DataFrame,
    min_ti: Optional[str] = None,
    max_ti: Optional[str] = None
) -> DataFrame:
    """
    取得所有 adoption cohort 時點 Ti。
    min_ti / max_ti 可用來限制 cohort 範圍，避免資料膨脹。
    """
    ti = (
        base_df
        .select(F.col("adoption_month").alias("Ti"))
        .where(F.col("Ti").isNotNull())
        .distinct()
    )

    if min_ti is not None:
        ti = ti.where(F.col("Ti") >= F.lit(min_ti))
    if max_ti is not None:
        ti = ti.where(F.col("Ti") <= F.lit(max_ti))

    return ti


def get_user_status(base_df: DataFrame) -> DataFrame:
    """
    每位 user 對應一個 adoption_month。
    若資料中同 id 有多個 adoption_month，取最早非空值。
    """
    return (
        base_df
        .groupBy("id")
        .agg(F.min("adoption_month").alias("adoption_month"))
    )


def build_risk_set_rows(
    base_df: DataFrame,
    lookback_months: int = 12,
    min_ti: Optional[str] = None,
    max_ti: Optional[str] = None,
    match_months: Optional[List[int]] = None,
    control_type: str = "never_treated"  
) -> DataFrame:
    """
    產出 risk-set 明細：
      - 每個 cohort Ti
      - 每個 user 在 [Ti-lookback, Ti) 的月資料
      - group = treated / control
      - treated: adoption_month == Ti
      - control: adoption_month is null or adoption_month > Ti

    注意：
    這一步是全流程最容易膨脹的地方。
    建議搭配 min_ti / max_ti 限縮 cohort 期間。
    """
    ti_df = get_distinct_ti(base_df, min_ti=min_ti, max_ti=max_ti)
    user_status = get_user_status(base_df)

    # 只保留至少在某個 cohort window 內可能用到的資料，減少 cross join 後壓力
    ti_bounds = ti_df.agg(
        F.min("Ti").alias("min_Ti"),
        F.max("Ti").alias("max_Ti")
    ).first()

    if ti_bounds["min_Ti"] is None or ti_bounds["max_Ti"] is None:
        return spark.createDataFrame([], schema="""
            id string,
            Ti date,
            adoption_month date,
            group string,
            month date,
            top3_mean_consumption double,
            mean_consumption double,
            variance_consumption double,
            total_consumption double
        """)

    global_min_month = F.add_months(F.lit(ti_bounds["min_Ti"]), -lookback_months)
    global_max_month = F.lit(ti_bounds["max_Ti"])

    monthly = (
        base_df
        .where(F.col("month") >= global_min_month)
        .where(F.col("month") < global_max_month)
        .alias("m")
    )

    # =========================
    # control definition
    # =========================
    if control_type == "risk_set":
        cond = (
            (F.col("u.adoption_month") == F.col("t.Ti")) |
            (F.col("u.adoption_month").isNull()) |
            (F.col("u.adoption_month") > F.col("t.Ti"))
        )

    elif control_type == "never_treated":
        cond = (
            (F.col("u.adoption_month") == F.col("t.Ti")) |
            (F.col("u.adoption_month").isNull())
        )

    else:
        raise ValueError("control_type must be 'risk_set' or 'never_treated'")

    # 注意這裡是 Ti 與每月資料的 cross join，再用時間窗過濾
    risk_rows = (
        monthly.alias("m")
        .crossJoin(F.broadcast(ti_df).alias("t"))
        .join(user_status.alias("u"), on="id", how="inner")
        .where(
            (F.col("m.month") < F.col("t.Ti")) &
            (F.col("m.month") >= F.add_months(F.col("t.Ti"), -lookback_months)) &
            cond
        )
        .withColumn(
            "group",
            F.when(F.col("u.adoption_month") == F.col("t.Ti"), F.lit("treated"))
            .otherwise(F.lit("control"))
        )
        .select(
            F.col("id"),
            F.col("t.Ti").alias("Ti"),
            F.col("u.adoption_month").alias("adoption_month"),
            F.col("group"),
            F.col("m.month").alias("month"),
            F.col("m.top3_mean_consumption").alias("top3_mean_consumption"),
            F.col("m.mean_consumption").alias("mean_consumption"),
            F.col("m.variance_consumption").alias("variance_consumption"),
            F.col("m.total_consumption").alias("total_consumption")
        )
    )
    
    if match_months is not None:
        risk_rows = risk_rows.where(
            F.month(F.col("month")).isin(match_months)
        )

    return risk_rows



def prepare_matching_data(
    sdf: DataFrame,
    id_col: str = "aID",
    month_col: str = "TIDPUNKT",
    adoption_col: str = "tariff_start",
    price_col: Optional[str] = "price",
    price_value: Optional[str] = "all",
    lookback_months: int = 12,
    min_ti: Optional[str] = None,
    max_ti: Optional[str] = None,
    match_months: Optional[List[int]] = None,
    control_type: str = "never_treated",
    repartition_by_ti: bool = True,
    verbose: bool = True
) -> DataFrame:

    if verbose:
        print("Preparing base data...")

    base = prepare_base_spark(
        sdf,
        id_col=id_col,
        month_col=month_col,
        adoption_col=adoption_col,
        price_col=price_col,
        price_value=price_value
    )

    if verbose:
        print("Building risk set...")

    risk_rows = build_risk_set_rows(
        base,
        lookback_months=lookback_months,
        min_ti=min_ti,
        max_ti=max_ti,
        match_months=match_months,
        control_type=control_type
    )

    if repartition_by_ti:
        risk_rows = risk_rows.repartition("Ti")

    risk_rows = risk_rows.cache()

    if verbose:
        print("risk_rows count =", risk_rows.count())

    return risk_rows