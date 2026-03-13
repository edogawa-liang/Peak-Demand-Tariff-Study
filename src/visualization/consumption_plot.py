import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import pandas as pd


def plot_consumption(
    df,
    group_by,
    value_col,
    agg="mean",
    splits=None,
    kind="line",
    show_legend=True,
    figsize=(10,6),
    dpi=120
):

    data = df.copy()

    group_cols = ["TIDPUNKT"]

    if splits:
        group_cols += splits

    g = data.groupby(group_cols)[value_col].agg(agg)

    if splits:
        g = g.unstack(splits)

    if "price" in (splits or []):
        g = g.drop(columns="all", errors="ignore")

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    g.plot(kind=kind, marker="o", ax=ax)

    title_map = {
            "month": "Average Monthly Peak Consumption",
            "hour": "Average Hourly Consumption Profile",
            "weekday": "Average Consumption by Day of Week",
    }
    xlabel_map = {
        "month": "Month",
        "hour": "Hour of Day",
        "weekday": "Day of Week",
    }

    ylabel_map = {
        "mean_consumption": "Average Electricity Consumption (kWh)",
        "top3_mean_consumption": "Average Peak Consumption (Top 3 Hours, kWh)",
        "variance_consumption": "Variance of Electricity Consumption",
    }

    ax.set_title(title_map.get(group_by, "Electricity Consumption"))
    ax.set_xlabel(xlabel_map.get(group_by, "Month"))
    ax.set_ylabel(ylabel_map.get(value_col, "Electricity Consumption (kWh)"))

    if show_legend:

        handles, labels = ax.get_legend_handles_labels()

        new_labels = []

        for l in labels:

            parts = l.strip("()").split(", ")

            # case 1: tariff only
            if parts == ["0"]:
                new_labels.append("No tariff")

            elif parts == ["1"]:
                new_labels.append("Tariff active")

            # case 2: usage group + tariff
            elif len(parts) == 2 and "usage_group" in (splits or []):

                usage_map = {
                    "low": "Low usage households",
                    "medium": "Medium usage households",
                    "high": "High usage households"
                }

                tariff_label = "Tariff active" if parts[1] == "1" else "No tariff"

                new_labels.append(f"{usage_map.get(parts[0], parts[0])} – {tariff_label}")

            # case 3: price + tariff
            elif len(parts) == 2 and "price" in (splits or []):

                price_label = "Low price period" if parts[0] == "low" else "High price period"
                tariff_label = "Tariff active" if parts[1] == "1" else "No tariff"

                new_labels.append(f"{price_label} – {tariff_label}")

            else:
                new_labels.append(l)

        ax.legend(handles, new_labels)

    plt.xticks(rotation=30)

    return ax


def plot_tariff_adoption_by_usage(
    df,
    figsize=(8,5),
    dpi=120,
    show_percent=True
):

    data = df.copy()

    g = data.groupby("usage_group")["tariff_active"].mean()

    order = ["low", "medium", "high"]
    g = g.reindex(order)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    g.plot(kind="bar", ax=ax)

    ax.set_title("Tariff Adoption by Household Usage Group")
    ax.set_xlabel("Household Usage Group")
    ax.set_ylabel("Share Choosing Tariff")

    plt.xticks(rotation=0)

    if show_percent:
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

    return ax



def plot_event_study_tariff(
    df,
    value_col="mean_consumption",
    line_cols="usage_group",
    window=6
):

    data = df.copy()

    # only adopters
    data = data[data["tariff_start"].notna()].copy()

    data["TIDPUNKT"] = pd.to_datetime(data["TIDPUNKT"])
    data["tariff_start"] = pd.to_datetime(data["tariff_start"])

    # event time (months relative to adoption)
    data["event_time"] = (
        (data["TIDPUNKT"].dt.year - data["tariff_start"].dt.year) * 12 +
        (data["TIDPUNKT"].dt.month - data["tariff_start"].dt.month)
    )

    data = data[
        (data["event_time"] >= -window) &
        (data["event_time"] <= window)
    ]

    if isinstance(line_cols, str):
        line_cols = [line_cols]

    g = (
        data
        .groupby(["event_time"] + line_cols)[value_col]
        .mean()
        .unstack(line_cols)
    )

    plt.figure()

    ax = g.plot(marker="o")

    plt.axvline(0, linestyle="--")

    title = ", ".join(line_cols).replace("_", " ").title()
    ax.legend(title=title)

    plt.title("Electricity Consumption Around Tariff Adoption")
    plt.xlabel("Months Relative to Tariff Adoption")
    plt.ylabel("Average Electricity Consumption (kWh)")

    plt.xticks(range(-window, window+1))

    return ax