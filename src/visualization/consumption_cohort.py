import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import pandas as pd
import math


def plot_cohort_calendar(
    df,
    cohort_month,   # e.g. "2024-03"
    time_col="TIDPUNKT",
    value_col="top3_mean_consumption",
    figsize=(10,6),
    dpi=120
):
    data = df.copy()

    # -------------------------
    # 1. preprocess time variables
    # -------------------------
    data[time_col] = pd.to_datetime(data[time_col]).dt.to_period("M")
    data["tariff_month"] = pd.to_datetime(data["tariff_start"]).dt.to_period("M")

    cohort_month = pd.Period(cohort_month, freq="M")

    # -------------------------
    # 2. define groups
    # -------------------------
    data["group"] = "other"

    # cohort adopters
    data.loc[data["tariff_month"] == cohort_month, "group"] = "treated"

    # never adopters
    data.loc[data["tariff_month"].isna(), "group"] = "control"

    data = data[data["group"].isin(["treated", "control"])].copy()

    # -------------------------
    # 3. aggregation（use calendar time）
    # -------------------------
    g = (
        data.groupby([time_col, "group"])[value_col]
        .mean()
        .unstack("group")
        .sort_index()
    )

    # -------------------------
    # 4. plot
    # -------------------------
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    g.plot(ax=ax, marker="o")

    # adoption month
    ax.axvline(cohort_month, linestyle="--", color="black", alpha=0.7)

    ax.set_title(f"Calendar Time: {cohort_month} Adopters vs Non-adopters")
    ax.set_xlabel("Month")
    ax.set_ylabel("Average Consumption (kWh)")

    # legend
    handles, labels = ax.get_legend_handles_labels()
    new_labels = [
        "Tariff adopters" if l == "treated" else "Non-adopters"
        for l in labels
    ]
    ax.legend(handles, new_labels)

    plt.xticks(rotation=45)

    return ax




def plot_multiple_cohorts(
    df,
    cohort_months,   # list，例如 ["2024-03", "2024-04"]
    time_col="TIDPUNKT",
    value_col="top3_mean_consumption",
    include_control=True,
    figsize=(10,6),
    dpi=120
):
    data = df.copy()

    # -------------------------
    # preprocess time variables
    # -------------------------
    data[time_col] = pd.to_datetime(data[time_col]).dt.to_period("M")
    data["tariff_month"] = pd.to_datetime(data["tariff_start"]).dt.to_period("M")

    cohort_months = [pd.Period(m, freq="M") for m in cohort_months]

    # -------------------------
    # define groups
    # -------------------------
    data["group"] = None

    # cohort
    for m in cohort_months:
        mask = data["tariff_month"] == m
        data.loc[mask, "group"] = str(m)

    # control（never adopters）
    if include_control:
        data.loc[data["tariff_month"].isna(), "group"] = "control"

    # keep only cohort and control groups
    data = data[data["group"].notna()].copy()

    # -------------------------
    # aggregation
    # -------------------------
    g = (
        data.groupby([time_col, "group"])[value_col]
        .mean()
        .unstack("group")
        .sort_index()
    )

    # -------------------------
    # plot
    # -------------------------
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    g.plot(ax=ax, marker="o")

    # mark all adoption months
    for m in cohort_months:
        ax.axvline(m, linestyle="--", alpha=0.3)

    ax.set_title("Average Peak Consumption by Tariff Adoption Month")
    ax.set_xlabel("Month")
    ax.set_ylabel("Average Consumption (kWh)")

    # legend rename
    handles, labels = ax.get_legend_handles_labels()
    new_labels = []

    for l in labels:
        if l == "control":
            new_labels.append("Non-adopters")
        else:
            new_labels.append(f"{l} adopters")

    ax.legend(handles, new_labels)

    plt.xticks(rotation=45)

    return ax




def plot_cohort_panels(
    df,
    cohort_months,   # list，Ex: ["2024-03","2024-04",...]
    time_col="TIDPUNKT",
    value_col="top3_mean_consumption",
    ncols=3,
    figsize=(15,10),
    dpi=120
):
    data = df.copy()

    # -------------------------
    # preprocess time variables
    # -------------------------
    data[time_col] = pd.to_datetime(data[time_col]).dt.to_period("M")
    data["tariff_month"] = pd.to_datetime(data["tariff_start"]).dt.to_period("M")

    cohort_months = [pd.Period(m, freq="M") for m in cohort_months]

    # -------------------------
    # subplot layout
    # -------------------------
    n = len(cohort_months)
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=figsize,
        dpi=dpi,
        sharex=True,
        sharey=True
    )

    axes = axes.flatten()

    # -------------------------
    #  loop through cohorts
    # -------------------------
    for i, cohort_month in enumerate(cohort_months):

        ax = axes[i]

        treated = data[data["tariff_month"] == cohort_month]
        control = data[data["tariff_month"].isna()]

        d = pd.concat([treated, control]).copy()

        d["group"] = d["tariff_month"].apply(
            lambda x: "treated" if x == cohort_month else "control"
        )

        # --- aggregation ---
        g = (
            d.groupby([time_col, "group"])[value_col]
            .mean()
            .unstack("group")
            .sort_index()
        )

        # --- plot ---
        g.plot(ax=ax, marker="o")

        ax.axvline(cohort_month, linestyle="--", color="black", alpha=0.5)

        ax.set_title(cohort_month.strftime("%b %Y") + " adopters")

        if i % ncols == 0:
            ax.set_ylabel("Consumption (kWh)")

        ax.tick_params(axis="x", rotation=45)

        if ax.get_legend():
            ax.get_legend().remove()

    # -------------------------
    # clear subplot
    # -------------------------
    for j in range(i+1, len(axes)):
        fig.delaxes(axes[j])

    # -------------------------
    # legend
    # -------------------------
    handles, labels = ax.get_legend_handles_labels()
    new_labels = [
        "Tariff adopters" if l == "treated" else "Non-adopters"
        for l in labels
    ]

    fig.legend(handles, new_labels, loc="upper right")

    fig.suptitle("Average Peak Consumption of Households by Tariff Adoption Month", fontsize=16)

    plt.tight_layout()

    return axes