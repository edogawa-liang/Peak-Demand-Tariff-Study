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
    dpi=120,
    exclude_future_tariff=False,
    facet_by=None
):


    data = df.copy()

    # remove future adopters from control group
    if exclude_future_tariff and "tariff_active" in (splits or []):
        data = data[
            (data["tariff_active"] == 1) | (data["tariff_start"].isna())
        ].copy()

        print("Before:", len(df))
        print("After:", len(data))

    group_cols = ["TIDPUNKT"]

    if splits:
        group_cols += splits

    g = data.groupby(group_cols)[value_col].agg(agg)

    if splits:
        g = g.unstack(splits)

    if "price" in (splits or []):
        g = g.drop(columns="all", errors="ignore")

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

    title = title_map.get(group_by, "Electricity Consumption")

    if exclude_future_tariff:
        title += " (Control group: Non-adopters)"

    # =========================
    # FACET MODE
    # =========================

    if facet_by and facet_by in (splits or []):

        facet_values = sorted(data[facet_by].dropna().unique())
        n = len(facet_values)

        fig, axes = plt.subplots(
            n, 1,
            figsize=(figsize[0], 4*n),
            dpi=dpi,
            sharex=True
        )

        if n == 1:
            axes = [axes]

        handles = None
        labels = None

        for i, val in enumerate(facet_values):

            ax = axes[i]

            cols = [
                c for c in g.columns
                if val in (c if isinstance(c, tuple) else [c])
            ]

            g[cols].plot(kind=kind, marker="o", ax=ax)

            # pretty facet title
            val_label = str(val).replace("_"," ").title()
            ax.set_title(val_label)

            ax.set_xlabel(xlabel_map.get(group_by, "Month"))

            if i == 0:
                ax.set_ylabel(
                    ylabel_map.get(
                        value_col,
                        "Electricity Consumption (kWh)"
                    )
                )

            ax.tick_params(axis="x", rotation=30)

            if show_legend and i == 0:
                handles, labels = ax.get_legend_handles_labels()

            if ax.get_legend():
                ax.get_legend().remove()

        # clean legend labels
        if show_legend and handles:

            new_labels = []

            for l in labels:

                if "1" in l:
                    if "has_tariff" in (splits or []):
                        new_labels.append("Tariff adopters")
                    else:
                        new_labels.append("Tariff active")

                elif "0" in l:
                    if "has_tariff" in (splits or []):
                        new_labels.append("Non-adopters")
                    elif exclude_future_tariff:
                        new_labels.append("No tariff (Non-adopters)")
                    else:
                        new_labels.append("No tariff")

                else:
                    new_labels.append(l)

            axes[0].legend(handles, new_labels, loc="upper right")

        fig.suptitle(title)

        return axes

    # =========================
    # SINGLE PLOT
    # =========================

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    g.plot(kind=kind, marker="o", ax=ax)

    ax.set_title(title)
    ax.set_xlabel(xlabel_map.get(group_by, "Month"))
    ax.set_ylabel(
        ylabel_map.get(
            value_col,
            "Electricity Consumption (kWh)"
        )
    )

    if show_legend:

        handles, labels = ax.get_legend_handles_labels()
        new_labels = []

        for l in labels:

            parts = l.strip("()").split(", ")

            # price + tariff
            if len(parts) == 2 and "price" in (splits or []):

                price_label = "Low price period" if parts[0] == "low" else "High price period"

                if "has_tariff" in (splits or []):
                    tariff_label = "Tariff adopters" if parts[1] == "1" else "Non-adopters"
                else:
                    tariff_label = "Tariff active" if parts[1] == "1" else "No tariff"

                new_labels.append(f"{price_label} – {tariff_label}")

            # tariff only
            elif parts == ["0"]:

                if "has_tariff" in (splits or []):
                    new_labels.append("Non-adopters")

                elif exclude_future_tariff:
                    new_labels.append("No tariff (Non-adopters)")

                else:
                    new_labels.append("No tariff")


            elif parts == ["1"]:

                if "has_tariff" in (splits or []):
                    new_labels.append("Tariff adopters")

                else:
                    new_labels.append("Tariff active")

            else:
                new_labels.append(l)

        ax.legend(handles, new_labels)

    plt.xticks(rotation=30)

    return ax

def plot_tariff_adoption(
    df,
    group_by="month",
    value_col="top3_mean_consumption",
    by_price=False,
    **kwargs
):
    data = df.copy()

    data["has_tariff"] = data["tariff_start"].notna().astype(int)

    splits = ["has_tariff"]

    if by_price:
        splits = ["price", "has_tariff"]

    return plot_consumption(
        data,
        group_by=group_by,
        value_col=value_col,
        splits=splits,
        exclude_future_tariff=False,
        **kwargs
    )


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
    line_cols=None,
    window=6
):

    data = df.copy()

    # keep adopters only
    data = data[data["tariff_start"].notna()].copy()

    data["TIDPUNKT"] = pd.to_datetime(data["TIDPUNKT"])
    data["tariff_start"] = pd.to_datetime(data["tariff_start"])

    # event time (months)
    data["event_time"] = (
        (data["TIDPUNKT"].dt.year - data["tariff_start"].dt.year) * 12 +
        (data["TIDPUNKT"].dt.month - data["tariff_start"].dt.month)
    )

    data = data[
        (data["event_time"] >= -window) &
        (data["event_time"] <= window)
    ]

    if line_cols is None:

        g = data.groupby("event_time")[value_col].mean()
        plt.figure()
        ax = g.plot(marker="o")

    else:

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

        title = ", ".join(line_cols).replace("_", " ").title()
        ax.legend(title=title)

    plt.axvline(0, linestyle="--")

    plt.title("Electricity Consumption Around Tariff Adoption")
    plt.xlabel("Months Relative to Tariff Adoption")
    plt.ylabel("Average Electricity Consumption (kWh)")

    plt.xticks(range(-window, window+1))

    if line_cols is None and ax.get_legend() is not None:
        ax.get_legend().remove()

    return ax
