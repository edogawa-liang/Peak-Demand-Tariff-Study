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
        cmap="YlOrRd",
        cbar_kws={"label": "Peak count" if mode=="count" else "Peak consumption (kWh)"}
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
    Plot four heatmaps:

    1. Never adopters
    2. Adopters BEFORE adoption
    3. Adopters AFTER adoption
    4. Difference (AFTER − BEFORE)
    """


    df = df.copy()

    # -------------------------------------------------
    # define groups
    # -------------------------------------------------

    never = df[df["tariff_start"].isna()]

    adopters_before = df[
        (df["tariff_start"].notna()) &
        (df["tariff_active"] == 0)
    ]

    adopters_after = df[df["tariff_active"] == 1]

    groups = {
        "Never adopters": never,
        "Adopters BEFORE": adopters_before,
        "Adopters AFTER": adopters_after
    }

    heatmaps = {}

    # -------------------------------------------------
    # build heatmaps
    # -------------------------------------------------

    for name, subset in groups.items():

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

        heatmaps[name] = heatmap

    never = heatmaps["Never adopters"]
    before = heatmaps["Adopters BEFORE"]
    after = heatmaps["Adopters AFTER"]

    # -------------------------------------------------
    # align axes
    # -------------------------------------------------

    all_months = sorted(
        set(never.index) |
        set(before.index) |
        set(after.index)
    )

    all_hours = sorted(
        set(never.columns) |
        set(before.columns) |
        set(after.columns)
    )

    for key in heatmaps:
        heatmaps[key] = heatmaps[key].reindex(
            index=all_months,
            columns=all_hours,
            fill_value=0
        )

    never = heatmaps["Never adopters"]
    before = heatmaps["Adopters BEFORE"]
    after = heatmaps["Adopters AFTER"]

    diff = after - before

    # -------------------------------------------------
    # color scales
    # -------------------------------------------------

    vmax = max(
        never.max().max(),
        before.max().max(),
        after.max().max()
    )

    diff_max = abs(diff).max().max()

    # -------------------------------------------------
    # plotting
    # -------------------------------------------------

    fig, axes = plt.subplots(2, 2, figsize=(14,10))

    # Never adopters
    sns.heatmap(
        never,
        cmap="YlOrRd",
        vmin=0,
        vmax=vmax,
        ax=axes[0,0]
    )
    axes[0,0].set_title("Never adopters (No tariff)")

    # Adopters BEFORE
    sns.heatmap(
        before,
        cmap="YlOrRd",
        vmin=0,
        vmax=vmax,
        ax=axes[0,1]
    )
    axes[0,1].set_title("Tariff adopters BEFORE adoption")

    # Adopters AFTER
    sns.heatmap(
        after,
        cmap="YlOrRd",
        vmin=0,
        vmax=vmax,
        ax=axes[1,0]
    )
    axes[1,0].set_title("Tariff adopters AFTER adoption")

    # Difference
    sns.heatmap(
        diff,
        cmap="coolwarm",
        center=0,
        vmin=-diff_max,
        vmax=diff_max,
        ax=axes[1,1]
    )
    axes[1,1].set_title("Difference (After − Before tariff adoption)")

    for ax in axes.flat:
        ax.set_xlabel("Hour")
        ax.set_ylabel("Month")

    plt.tight_layout()
    plt.show()