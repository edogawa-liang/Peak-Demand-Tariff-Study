import os
import json
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from typing import List, Optional, Dict, Any

from pyspark.sql import SparkSession, DataFrame, Window
from pyspark.sql import functions as F

from prepare_matching_data import prepare_base_spark
from matching import (
    build_summary_profiles_spark,
    balance_table_spark,
    build_calendar_aligned_profiles
)

# ============================================================
# Helpers
# ============================================================

MONTH_ABB = {
    1: "jan", 2: "feb", 3: "mar", 4: "apr",
    5: "may", 6: "jun", 7: "jul", 8: "aug",
    9: "sep", 10: "oct", 11: "nov", 12: "dec"
}

# ============================================================
# MAIN VALIDATION FUNCTION
# ============================================================

def check_balance_full_safe(
    res: dict,
    sdf: DataFrame,
    id_col: str = "aID",
    month_col: str = "TIDPUNKT",
    adoption_col: str = "tariff_start",
    price_col: Optional[str] = "price",
    price_value: Optional[str] = "all",
    validation_lookback_months: int = 24,
    check_calendar: bool = True,
    calendar_months: list = [1,2,3,11,12],
    calendar_years: int = 2,
):
    """
    Validation using matched sample + longer lookback (e.g. 24 months)
    """

    print(f"Rebuilding validation data with lookback = {validation_lookback_months} months ...")

    # ============================================================
    # Full covariates
    # ============================================================
    ALL_SUMMARY_VARS = [
        "peak_mean",
        "peak_sd",
        "peak_volatility",
        "mean_consumption",
        "variance_consumption",
        "total_consumption",
        "trend"
    ]

    match_vars = res.get("match_vars", [])
    matches = res["matches"]

    # ============================================================
    # Step 0: rebuild FULL base (🔥不要用 risk_rows)
    # ============================================================
    base = prepare_base_spark(
        sdf=sdf,
        id_col=id_col,
        month_col=month_col,
        adoption_col=adoption_col,
        price_col=price_col,
        price_value=price_value
    )

    # ============================================================
    # Step 1: matched ids
    # ============================================================
    treated_ids = (
        matches
        .select(
            F.col("treated_id").alias("id"),
            F.col("adoption_month").alias("Ti")
        )
        .dropDuplicates()
        .withColumn("group", F.lit("treated"))
    )

    control_ids = (
        matches
        .select(
            F.col("control_id").alias("id"),
            F.col("adoption_month").alias("Ti")
        )
        .dropDuplicates()
        .withColumn("group", F.lit("control"))
    )

    matched_ids = treated_ids.unionByName(control_ids)

    # ============================================================
    # Step 2: rebuild validation panel (🔥核心)
    # ============================================================
    validation_rows = (
        matched_ids.alias("m")
        .join(
            base.alias("b"),
            on=[F.col("m.id") == F.col("b.id")],
            how="inner"
        )
        .where(
            (F.col("b.month") < F.col("m.Ti")) &
            (F.col("b.month") >= F.add_months(F.col("m.Ti"), -validation_lookback_months))
        )
        .select(
            F.col("m.id").alias("id"),
            F.col("m.Ti").alias("Ti"),
            F.col("m.group").alias("group"),
            F.col("b.month").alias("month"),
            F.col("b.adoption_month"),
            F.col("b.top3_mean_consumption"),
            F.col("b.mean_consumption"),
            F.col("b.variance_consumption"),
            F.col("b.total_consumption")
        )
    ).cache()

    print("Validation rows count =", validation_rows.count())

    # ============================================================
    # Step 3: rebuild summary (24m)
    # ============================================================
    profiles = build_summary_profiles_spark(
        validation_rows,
        summary_vars=ALL_SUMMARY_VARS
    )

    print("Profiles rebuilt (24m)")

    # ============================================================
    # Step 4: split vars
    # ============================================================
    existing_cols = set(profiles.columns)

    match_vars_safe = [v for v in match_vars if v in existing_cols]

    non_match_vars_safe = [
        v for v in ALL_SUMMARY_VARS
        if v in existing_cols and v not in match_vars_safe
    ]

    # ============================================================
    # Step 5: matching vars check
    # ============================================================
    if match_vars_safe:
        print("\n=== MATCHING VARIABLES (24m check) ===")

        balance_match = balance_table_spark(
            profiles,
            match_vars_safe
        )
        balance_match.show(50, truncate=False)
    else:
        print("\n(No matching variables available)")

    print("\n" + "-" * 50 + "\n")

    # ============================================================
    # Step 6: NON-matching vars (🔥最重要)
    # ============================================================
    if non_match_vars_safe:
        print("=== NON-MATCHING VARIABLES (MAIN VALIDATION, 24m) ===")

        balance_extra = balance_table_spark(
            profiles,
            non_match_vars_safe
        )
        balance_extra.show(50, truncate=False)
    else:
        print("All summary variables were used in matching.")

    # ============================================================
    # Step 7: calendar validation
    # ============================================================
    if check_calendar:
        print("\n" + "=" * 60)
        print("=== CALENDAR CHECK (24m) ===")

        calendar_profiles = build_calendar_aligned_profiles(
            validation_rows,
            match_months=calendar_months,
            n_years=calendar_years
        )

        cal_cols = [
            c for c in calendar_profiles.columns
            if c not in ["Ti", "id", "adoption_month", "group"]
        ]

        # 🔥 關鍵：維度分布
        valid_expr = None
        for c in cal_cols:
            this = F.col(c).isNotNull().cast("int")
            valid_expr = this if valid_expr is None else (valid_expr + this)

        calendar_profiles = calendar_profiles.withColumn("valid_dim", valid_expr)

        print("=== CALENDAR VALID DIM DISTRIBUTION ===")
        calendar_profiles.groupBy("valid_dim").count().orderBy("valid_dim").show()

        if cal_cols:
            balance_calendar = balance_table_spark(
                calendar_profiles,
                cal_cols
            )
            balance_calendar.show(50, truncate=False)
        else:
            print("No calendar covariates available.")
