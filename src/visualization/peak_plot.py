import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# =====================================================
# Helpers
# =====================================================

def _extract_peak_times(df):

    return pd.concat([
        df["peak1_time"],
        df["peak2_time"],
        df["peak3_time"]
    ]).dropna()


def _extract_peak_consumption(df):

    return pd.concat([
        df["peak1_consumption"],
        df["peak2_consumption"],
        df["peak3_consumption"]
    ]).dropna()


# =====================================================
# Peak hour distribution
# =====================================================

def plot_peak_hour_distribution(df, mode="count"):
    """
    mode:
        count -> peak frequency
        consumption -> total peak consumption
    """

    times = _extract_peak_times(df)
    cons = _extract_peak_consumption(df)

    temp = pd.DataFrame({
        "hour": times.dt.hour,
        "consumption": cons
    }).dropna()

    if mode == "count":
        data = temp["hour"].value_counts().sort_index()
        ylabel = "Peak Count"

    elif mode == "consumption":
        data = temp.groupby("hour")["consumption"].sum()
        ylabel = "Total Peak Consumption (kWh)"

    else:
        raise ValueError("mode must be 'count' or 'consumption'")

    plt.figure(figsize=(8,4))
    data.sort_index().plot.bar()

    plt.xlabel("Hour of Day")
    plt.ylabel(ylabel)
    plt.title(f"Peak Hour Distribution ({mode})")

    plt.tight_layout()
    plt.show()


# =====================================================
# Peak heatmap
# =====================================================

def plot_peak_heatmap(df, mode="count"):
    """
    Heatmap of peak demand by month and hour
    """

    times = _extract_peak_times(df)
    cons = _extract_peak_consumption(df)

    temp = pd.DataFrame({
        "month": times.dt.month,
        "hour": times.dt.hour,
        "consumption": cons
    }).dropna()

    if mode == "count":

        heatmap = temp.pivot_table(
            index="month",
            columns="hour",
            aggfunc="size",
            fill_value=0
        )

    elif mode == "consumption":

        heatmap = temp.pivot_table(
            index="month",
            columns="hour",
            values="consumption",
            aggfunc="sum",
            fill_value=0
        )

    else:
        raise ValueError("mode must be 'count' or 'consumption'")

    plt.figure(figsize=(10,5))

    sns.heatmap(
        heatmap,
        cmap="YlOrRd"
    )

    plt.title(f"Peak Heatmap ({mode})")
    plt.xlabel("Hour")
    plt.ylabel("Month")

    plt.tight_layout()
    plt.show()


# =====================================================
# Peak consumption distribution
# =====================================================

def plot_peak_consumption_distribution(df):

    peaks = _extract_peak_consumption(df)

    plt.figure(figsize=(8,4))

    plt.hist(peaks, bins=30)

    plt.xlabel("Peak Consumption (kWh)")
    plt.ylabel("Frequency")
    plt.title("Peak Consumption Distribution")

    plt.tight_layout()
    plt.show()


# =====================================================
# Peak rank boxplot
# =====================================================

def plot_peak_rank_boxplot(df):

    data = df[
        [
            "peak1_consumption",
            "peak2_consumption",
            "peak3_consumption"
        ]
    ].rename(columns={
        "peak1_consumption": "Peak 1",
        "peak2_consumption": "Peak 2",
        "peak3_consumption": "Peak 3"
    })

    plt.figure(figsize=(6,4))

    sns.boxplot(data=data)

    plt.ylabel("Consumption (kWh)")
    plt.title("Peak Rank Comparison")

    plt.tight_layout()
    plt.show()


# =====================================================
# Tariff peak heatmap
# =====================================================

def plot_tariff_peak_heatmap(df, mode="count"):
    """
    Compare peak heatmaps before and after tariff.
    Also plot a difference heatmap (After - Before).
    """

    heatmaps = {}

    # ---------- build heatmaps ----------
    for t in [0,1]:

        subset = df[df["tariff_active"] == t]

        times = _extract_peak_times(subset)
        cons = _extract_peak_consumption(subset)

        temp = pd.DataFrame({
            "month": times.dt.month,
            "hour": times.dt.hour,
            "consumption": cons
        }).dropna()

        if mode == "count":

            heatmap = temp.pivot_table(
                index="month",
                columns="hour",
                aggfunc="size",
                fill_value=0
            )

        elif mode == "consumption":

            heatmap = temp.pivot_table(
                index="month",
                columns="hour",
                values="consumption",
                aggfunc="sum",
                fill_value=0
            )

        else:
            raise ValueError("mode must be 'count' or 'consumption'")

        heatmaps[t] = heatmap

    before = heatmaps[0]
    after = heatmaps[1]

    # ---------- align axes ----------
    all_months = sorted(set(before.index) | set(after.index))
    all_hours = sorted(set(before.columns) | set(after.columns))

    before = before.reindex(index=all_months, columns=all_hours, fill_value=0)
    after = after.reindex(index=all_months, columns=all_hours, fill_value=0)

    # ---------- shared color scale ----------
    vmax = max(before.max().max(), after.max().max())

    # ---------- plot before ----------
    plt.figure(figsize=(10,5))

    sns.heatmap(
        before,
        cmap="YlOrRd",
        vmin=0,
        vmax=vmax
    )

    plt.title(f"Peak Heatmap (Before Tariff, {mode})")
    plt.xlabel("Hour")
    plt.ylabel("Month")

    plt.tight_layout()
    plt.show()

    # ---------- plot after ----------
    plt.figure(figsize=(10,5))

    sns.heatmap(
        after,
        cmap="YlOrRd",
        vmin=0,
        vmax=vmax
    )

    plt.title(f"Peak Heatmap (After Tariff, {mode})")
    plt.xlabel("Hour")
    plt.ylabel("Month")

    plt.tight_layout()
    plt.show()

    # ---------- difference heatmap ----------
    diff = after - before

    plt.figure(figsize=(10,5))

    sns.heatmap(
        diff,
        cmap="coolwarm",
        center=0
    )

    plt.title(f"Peak Heatmap Difference (After - Before, {mode})")
    plt.xlabel("Hour")
    plt.ylabel("Month")

    plt.tight_layout()
    plt.show()