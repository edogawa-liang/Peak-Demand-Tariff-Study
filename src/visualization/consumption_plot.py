import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick


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

            # case 2: price + tariff
            elif len(parts) == 2 and parts[0] in ["low","high"]:
                price_label = "Low price period" if parts[0] == "low" else "High price period"
                tariff_label = "Tariff active" if parts[1] == "1" else "No tariff"
                new_labels.append(f"{price_label} – {tariff_label}")

            # case 3: usage group + tariff
            elif len(parts) == 2 and parts[0] in ["low","medium","high"]:
                usage_map = {
                    "low": "Low usage households",
                    "medium": "Medium usage households",
                    "high": "High usage households"
                }

                tariff_label = "Tariff active" if parts[1] == "1" else "No tariff"

                new_labels.append(f"{usage_map[parts[0]]} – {tariff_label}")

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