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

def plot_tariff_peak_heatmap(df, price_label="all"):

    df = df.copy()

    never = df[df["tariff_start"].isna()]

    before = df[
        (df["tariff_start"].notna()) &
        (df["tariff_active"] == 0)
    ]

    after = df[df["tariff_active"] == 1]

    groups = {
        "Never adopters": never,
        "Adopters BEFORE": before,
        "Adopters AFTER": after
    }

    heatmaps = {}

    for name, subset in groups.items():

        times = _extract_peak_times(subset)

        temp = pd.DataFrame({
            "month": times.dt.month,
            "hour": times.dt.hour
        }).dropna()

        heatmap = temp.pivot_table(
            index="month",
            columns="hour",
            aggfunc="size",
            fill_value=0
        )

        # normalization
        heatmap = heatmap.div(heatmap.sum(axis=1), axis=0)

        heatmaps[name] = heatmap

    never = heatmaps["Never adopters"]
    before = heatmaps["Adopters BEFORE"]
    after = heatmaps["Adopters AFTER"]

    all_months = sorted(set(never.index) | set(before.index) | set(after.index))
    all_hours = list(range(24))

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

    fig, axes = plt.subplots(2,2, figsize=(14,10))

    sns.heatmap(never, cmap="YlOrRd", ax=axes[0,0])
    axes[0,0].set_title("Never adopters")

    sns.heatmap(before, cmap="YlOrRd", ax=axes[0,1])
    axes[0,1].set_title("Adopters BEFORE")

    sns.heatmap(after, cmap="YlOrRd", ax=axes[1,0])
    axes[1,0].set_title("Adopters AFTER")

    sns.heatmap(diff, cmap="coolwarm", center=0, ax=axes[1,1])
    axes[1,1].set_title("Difference (After − Before)")

    for ax in axes.flat:
        ax.set_xlabel("Hour")
        ax.set_ylabel("Month")

    title_map = {
        "all": "Peak Timing Heatmap (Overall Peaks)",
        "high": "Peak Timing Heatmap (High Price Period Peaks)",
        "low": "Peak Timing Heatmap (Low Price Period Peaks)"
    }

    fig.suptitle(title_map.get(price_label, "Peak Timing Heatmap"), fontsize=16, y=1.02)

    plt.tight_layout()
    plt.show()


# =====================================================
# Tariff peak consumption heatmap
# =====================================================

def plot_tariff_consumption_heatmap(df, price_label="all"):

    df = df.copy()

    never = df[df["tariff_start"].isna()]

    before = df[
        (df["tariff_start"].notna()) &
        (df["tariff_active"] == 0)
    ]

    after = df[df["tariff_active"] == 1]

    groups = {
        "Never adopters": never,
        "Adopters BEFORE": before,
        "Adopters AFTER": after
    }

    heatmaps = {}

    for name, subset in groups.items():

        times = _extract_peak_times(subset)
        cons = _extract_peak_consumption(subset)

        temp = pd.DataFrame({
            "month": times.dt.month,
            "hour": times.dt.hour,
            "consumption": cons
        }).dropna()

        heatmap = temp.pivot_table(
            index="month",
            columns="hour",
            values="consumption",
            aggfunc="mean"   # Use mean!!
        )

        heatmaps[name] = heatmap

    never = heatmaps["Never adopters"]
    before = heatmaps["Adopters BEFORE"]
    after = heatmaps["Adopters AFTER"]

    all_months = sorted(set(never.index) | set(before.index) | set(after.index))
    all_hours = list(range(24))

    for key in heatmaps:
        heatmaps[key] = heatmaps[key].reindex(
            index=all_months,
            columns=all_hours
        )

    never = heatmaps["Never adopters"]
    before = heatmaps["Adopters BEFORE"]
    after = heatmaps["Adopters AFTER"]

    diff = after - before

    # same color scale for comparison
    vmax = max(
        never.max().max(),
        before.max().max(),
        after.max().max()
    )

    diff_max = abs(diff).max().max()

    fig, axes = plt.subplots(2,2, figsize=(14,10))
    title_map = {
        "all": "Peak Consumption Heatmap (Overall Peaks)",
        "high": "Peak Consumption Heatmap (High Price Period Peaks)",
        "low": "Peak Consumption Heatmap (Low Price Period Peaks)"
    }

    fig.suptitle(title_map.get(price_label, "Peak Consumption Heatmap"), fontsize=16, y=1.02)

    sns.heatmap(never, cmap="YlOrRd", vmin=0, vmax=vmax, ax=axes[0,0])
    axes[0,0].set_title("Never adopters")

    sns.heatmap(before, cmap="YlOrRd", vmin=0, vmax=vmax, ax=axes[0,1])
    axes[0,1].set_title("Adopters BEFORE")

    sns.heatmap(after, cmap="YlOrRd", vmin=0, vmax=vmax, ax=axes[1,0])
    axes[1,0].set_title("Adopters AFTER")

    sns.heatmap(diff, cmap="coolwarm", center=0,
                vmin=-diff_max, vmax=diff_max,
                ax=axes[1,1])
    axes[1,1].set_title("Difference (After − Before)")

    for ax in axes.flat:
        ax.set_xlabel("Hour")
        ax.set_ylabel("Month")

    plt.tight_layout()
    plt.show()