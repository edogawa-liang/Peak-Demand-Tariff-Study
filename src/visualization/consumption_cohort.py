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
    cohort_months,
    time_col="TIDPUNKT",
    value_col="top3_mean_consumption",
    mode="raw",   # "raw" or "matched"
    ncols=3,
    figsize=(15,10),
    dpi=120
):
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import math
    from matplotlib.lines import Line2D

    data = df.copy()

    # -------------------------
    # preprocess
    # -------------------------
    data[time_col] = pd.to_datetime(data[time_col]).dt.to_period("M")
    data["tariff_month"] = pd.to_datetime(data["tariff_start"]).dt.to_period("M")

    if mode == "matched":
        data["cohort"] = pd.to_datetime(data["cohort"].astype(str)).dt.to_period("M")

    cohort_months = [pd.Period(m, freq="M") for m in cohort_months]

    # 固定顏色
    color_map = {
        "control": "tab:blue",
        "treated": "tab:orange"
    }

    # -------------------------
    # layout
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

    if n == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    # -------------------------
    # loop cohorts
    # -------------------------
    for i, cohort_month in enumerate(cohort_months):

        ax = axes[i]

        if mode == "raw":
            treated = data[data["tariff_month"] == cohort_month]
            control = data[data["tariff_month"].isna()]

            d = pd.concat([treated, control]).copy()

            d["group"] = np.where(
                d["tariff_month"] == cohort_month,
                "treated",
                "control"
            )

        elif mode == "matched":
            d = data[data["cohort"] == cohort_month].copy()

            d["group"] = d["treatment"].map({
                1: "treated",
                0: "control"
            })

        else:
            raise ValueError("mode must be 'raw' or 'matched'")

        # drop NA group
        d = d[d["group"].notna()]

        # -------------------------
        # aggregation
        # -------------------------
        g = (
            d.groupby([time_col, "group"])[value_col]
            .mean()
            .unstack("group")
            .sort_index()
        )

        # 固定欄位順序
        g = g.reindex(columns=["control", "treated"])

        # -------------------------
        # plot（逐條畫 → 保證顏色）
        # -------------------------
        for grp in ["control", "treated"]:
            if grp in g.columns and g[grp].notna().any():
                ax.plot(
                    g.index.to_timestamp(),
                    g[grp],
                    marker="o",
                    color=color_map[grp],
                    linewidth=2
                )

        # adoption line
        ax.axvline(
            cohort_month.to_timestamp(),
            linestyle="--",
            color="black",
            alpha=0.6
        )

        ax.set_title(cohort_month.strftime("%b %Y") + " adopters")

        if i % ncols == 0:
            ax.set_ylabel("Consumption (kWh)")

        ax.tick_params(axis="x", rotation=45)

    # -------------------------
    # remove empty panels
    # -------------------------
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    # -------------------------
    # legend（🔥 完全正確寫法）
    # -------------------------
    if mode == "raw":
        legend_handles = [
            Line2D([0], [0], color="tab:blue", marker="o", label="Never adopters"),
            Line2D([0], [0], color="tab:orange", marker="o", label="Tariff adopters"),
        ]
        title = "Before Matching"

    else:
        legend_handles = [
            Line2D([0], [0], color="tab:blue", marker="o", label="Matched controls"),
            Line2D([0], [0], color="tab:orange", marker="o", label="Tariff adopters"),
        ]
        title = "After Matching"

    fig.legend(
        handles=legend_handles,
        loc="upper right"
    )

    fig.suptitle(
        f"Average Peak Consumption by Cohort ({title})",
        fontsize=16
    )

    plt.tight_layout()

    return axes