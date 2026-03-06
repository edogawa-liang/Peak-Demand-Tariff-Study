import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")


def plot_monthly_adoption(tariff_df):

    df = tariff_df.copy()
    df["Startdatum"] = pd.to_datetime(df["Startdatum"])

    monthly = (
        df.groupby(df["Startdatum"].dt.to_period("M"))["GS1-nr"]
        .nunique()
        .sort_index()
    )

    monthly.index = monthly.index.to_timestamp()

    plt.figure()
    ax = monthly.plot(kind="bar")

    plt.title("Monthly Tariff Adoption")
    plt.ylabel("Households")
    plt.xlabel("Month")

    ax.set_xticklabels(monthly.index.strftime("%Y-%m"), rotation=30)

    return ax



def plot_monthly_share(tariff_df, total_households):

    df = tariff_df.copy()
    df["Startdatum"] = pd.to_datetime(df["Startdatum"])

    monthly = (
        df.groupby(df["Startdatum"].dt.to_period("M"))["GS1-nr"]
        .nunique()
        .sort_index()
    )

    full_index = pd.period_range(monthly.index.min(), monthly.index.max(), freq="M")
    monthly = monthly.reindex(full_index, fill_value=0)

    cumulative = monthly.cumsum()
    share = cumulative / total_households
    share.index = share.index.astype(str)

    plt.figure()
    ax = share.plot(marker="o")

    plt.title("Cumulative Tariff Adoption Share")
    plt.ylabel("Share of households")
    plt.xlabel("Month")

    plt.xticks(rotation=30)

    return ax



def plot_tariff_group_counts(tariff_df):

    df = tariff_df.copy()

    df["tariff_group"] = df["Produktnamn"].str.extract(
        r'(\d+\s*kW\s*(?:Villa|Normal))'
    )

    df = df.dropna(subset=["tariff_group"])

    counts = df["tariff_group"].value_counts()

    plt.figure()
    ax = counts.plot(kind="bar")

    plt.title("Tariff Group Choices")
    plt.ylabel("Households")
    plt.xlabel("Tariff Group")

    plt.xticks(rotation=0)

    return ax



def plot_tariff_group_cumulative(tariff_df):

    df = tariff_df.copy()

    df["tariff_group"] = df["Produktnamn"].str.extract(
        r'(\d+\s*kW\s*(?:Villa|Normal))'
    )

    df = df.dropna(subset=["tariff_group"])

    df["Startdatum"] = pd.to_datetime(df["Startdatum"])

    monthly = (
        df.groupby(
            [df["Startdatum"].dt.to_period("M"), "tariff_group"]
        )["GS1-nr"]
        .nunique()
        .unstack()
        .fillna(0)
        .sort_index()
    )

    # Align missing months with 0 counts
    full_index = pd.period_range(monthly.index.min(), monthly.index.max(), freq="M")
    monthly = monthly.reindex(full_index, fill_value=0)

    cumulative = monthly.cumsum()
    cumulative.index = cumulative.index.astype(str)

    plt.figure()
    ax = cumulative.plot(marker="o")

    plt.title("Cumulative Adoption by Tariff Group")
    plt.ylabel("Households")
    plt.xlabel("Month")

    plt.xticks(rotation=30)

    ax.legend(title=None)

    return ax