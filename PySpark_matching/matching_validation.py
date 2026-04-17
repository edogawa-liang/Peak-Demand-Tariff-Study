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
# Internal helpers
# ============================================================

def _build_matched_ids(matches: DataFrame) -> DataFrame:
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

    return treated_ids.unionByName(control_ids)


def _build_validation_rows(
    matched_ids: DataFrame,
    base: DataFrame,
    start_month: int,
    end_month: int
) -> DataFrame:
    """
    Build validation rows over [Ti-start_month, Ti-end_month)

    Example:
    - start_month=24, end_month=12  => [Ti-24, Ti-12)
    - start_month=24, end_month=0   => [Ti-24, Ti)
    """
    if start_month <= end_month:
        raise ValueError("start_month must be greater than end_month")

    return (
        matched_ids.alias("m")
        .join(
            base.alias("b"),
            on=[F.col("m.id") == F.col("b.id")],
            how="inner"
        )
        .where(
            (F.col("b.month") < F.add_months(F.col("m.Ti"), -end_month)) &
            (F.col("b.month") >= F.add_months(F.col("m.Ti"), -start_month))
        )
        .select(
            F.col("m.id").alias("id"),
            F.col("m.Ti").alias("Ti"),
            F.col("m.group").alias("group"),
            F.col("b.month").alias("month"),
            F.col("b.adoption_month").alias("adoption_month"),
            F.col("b.top3_mean_consumption"),
            F.col("b.mean_consumption"),
            F.col("b.variance_consumption"),
            F.col("b.total_consumption")
        )
    )


def _show_balance_block(
    profiles: DataFrame,
    feature_cols: List[str],
    title: str
):
    if not feature_cols:
        print(f"\n({title}: no variables available)")
        return

    print(f"\n=== {title} ===")

    print("\n--- OVERALL ---")
    balance_table_spark(
        profiles,
        feature_cols,
        pool=True
    ).show(50, truncate=False)

    print("\n--- BY Ti ---")
    balance_table_spark(
        profiles,
        feature_cols,
        pool=False
    ).show(50, truncate=False)


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

    # Matching variables window
    match_start_month: int = 24,
    match_end_month: int = 12,

    # Non-matching variables window
    extra_start_month: int = 24,
    extra_end_month: int = 12,

    # Calendar window
    calendar_start_month: int = 24,
    calendar_end_month: int = 0,

    check_calendar: bool = True,
    calendar_months: list = [1, 2, 3, 11, 12],
    calendar_years: int = 2,
):
    """
    Validation using matched sample with flexible pre-treatment windows.

    Windows:
    - matching vars:     [Ti-match_start_month, Ti-match_end_month)
    - non-matching vars: [Ti-extra_start_month, Ti-extra_end_month)
    - calendar vars:     [Ti-calendar_start_month, Ti-calendar_end_month)

    Recommended:
    - matching vars: 24 to 12  (non-overlapping with matching window)
    - non-matching:   24 to 0   or 24 to 12
    """

    print(
        f"Rebuilding validation data...\n"
        f"  Matching vars window:     [t-{match_start_month}, t-{match_end_month})\n"
        f"  Non-matching vars window: [t-{extra_start_month}, t-{extra_end_month})"
    )

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
    # Step 0: rebuild full base
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
    matched_ids = _build_matched_ids(matches)

    # ============================================================
    # Step 2A: validation rows for matching vars
    # ============================================================
    validation_rows_match = _build_validation_rows(
        matched_ids=matched_ids,
        base=base,
        start_month=match_start_month,
        end_month=match_end_month
    ).cache()

    print("Validation rows (matching vars) count =", validation_rows_match.count())

    profiles_match = build_summary_profiles_spark(
        validation_rows_match,
        summary_vars=ALL_SUMMARY_VARS
    ).cache()

    print("Profiles rebuilt for matching vars")

    existing_match_cols = set(profiles_match.columns)
    match_vars_safe = [v for v in match_vars if v in existing_match_cols]

    # ============================================================
    # Step 2B: validation rows for non-matching vars
    # ============================================================
    validation_rows_extra = _build_validation_rows(
        matched_ids=matched_ids,
        base=base,
        start_month=extra_start_month,
        end_month=extra_end_month
    ).cache()

    print("Validation rows (non-matching vars) count =", validation_rows_extra.count())

    profiles_extra = build_summary_profiles_spark(
        validation_rows_extra,
        summary_vars=ALL_SUMMARY_VARS
    ).cache()

    print("Profiles rebuilt for non-matching vars")

    existing_extra_cols = set(profiles_extra.columns)
    non_match_vars_safe = [
        v for v in ALL_SUMMARY_VARS
        if v in existing_extra_cols and v not in match_vars
    ]

    # ============================================================
    # Step 3: matching vars check
    # ============================================================
    if match_vars_safe:
        _show_balance_block(
            profiles=profiles_match,
            feature_cols=match_vars_safe,
            title=f"MATCHING VARIABLES CHECK [{match_start_month}, {match_end_month})"
        )
    else:
        print("\n(No matching variables available)")

    print("\n" + "-" * 60 + "\n")

    # ============================================================
    # Step 4: non-matching vars check
    # ============================================================
    if non_match_vars_safe:
        _show_balance_block(
            profiles=profiles_extra,
            feature_cols=non_match_vars_safe,
            title=f"NON-MATCHING VARIABLES CHECK [{extra_start_month}, {extra_end_month})"
        )
    else:
        print("All summary variables were used in matching.")

    # ============================================================
    # Step 5: calendar validation
    # ============================================================
    if check_calendar:
        print("\n" + "=" * 60)
        print(
            f"=== CALENDAR CHECK [{calendar_start_month}, {calendar_end_month}) ==="
        )

        validation_rows_calendar = _build_validation_rows(
            matched_ids=matched_ids,
            base=base,
            start_month=calendar_start_month,
            end_month=calendar_end_month
        ).cache()

        print("Validation rows (calendar) count =", validation_rows_calendar.count())

        calendar_profiles = build_calendar_aligned_profiles(
            validation_rows_calendar,
            match_months=calendar_months,
            n_years=calendar_years
        ).cache()

        cal_cols = [
            c for c in calendar_profiles.columns
            if c not in ["Ti", "id", "adoption_month", "group"]
        ]

        if cal_cols:
            valid_expr = None
            for c in cal_cols:
                this = F.col(c).isNotNull().cast("int")
                valid_expr = this if valid_expr is None else (valid_expr + this)

            calendar_profiles = calendar_profiles.withColumn("valid_dim", valid_expr)

            print("=== CALENDAR VALID DIM DISTRIBUTION ===")
            calendar_profiles.groupBy("valid_dim").count().orderBy("valid_dim").show(50, truncate=False)

            _show_balance_block(
                profiles=calendar_profiles,
                feature_cols=cal_cols,
                title="CALENDAR VARIABLES CHECK"
            )
        else:
            print("No calendar covariates available.")