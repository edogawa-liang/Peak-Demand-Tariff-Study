import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")


def plot_monthly_adoption(tariff_df):

    df = tariff_df.copy()
    df["Startdatum"] = pd.to_datetime(df["Startdatum"])

    df["month"] = df["Startdatum"].dt.to_period("M").dt.to_timestamp()

    monthly = (
        df.groupby("month")["GS1-nr."]
        .nunique()
        .sort_index()
    )
    monthly.index = monthly.index.strftime("%Y-%m")

    plt.figure()
    ax = monthly.plot(kind="bar")

    plt.title("Monthly Tariff Adoption")
    plt.ylabel("Households")
    plt.xlabel("Month")
    plt.xticks(rotation=30)

    return ax



def plot_monthly_share(tariff_df, total_households):

    df = tariff_df.copy()
    df["Startdatum"] = pd.to_datetime(df["Startdatum"])

    df["month"] = df["Startdatum"].dt.to_period("M").dt.to_timestamp()

    monthly = (
        df.groupby("month")["GS1-nr."]
        .nunique()
        .sort_index()
    )

    full_index = pd.date_range(
        monthly.index.min(),
        monthly.index.max(),
        freq="MS"
    )

    monthly = monthly.reindex(full_index, fill_value=0)

    cumulative = monthly.cumsum()
    share = cumulative / total_households

    plt.figure()
    ax = share.plot(marker="o")

    plt.title("Cumulative Tariff Adoption Share")
    plt.ylabel("Share of households")
    plt.xlabel("Month")

    ax.set_xticklabels(share.index.strftime("%Y-%m"), rotation=30)

    return ax



def plot_tariff_group_counts(tariff_df):

    df = tariff_df.copy()

    df["tariff_group"] = df["Produktnamn"]

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

    # extract tariff group
    df["tariff_plan"] = df["Produktnamn"].str.extract(
        r'(\d+\s*kW\s*(?:Villa|Normal))'
    )

    df = df.dropna(subset=["tariff_plan"])

    df["Startdatum"] = pd.to_datetime(df["Startdatum"])

    df["month"] = df["Startdatum"].dt.to_period("M").dt.to_timestamp()

    monthly = (
        df.groupby(["month", "tariff_plan"])["GS1-nr."]
        .nunique()
        .unstack()
        .fillna(0)
        .sort_index()
    )

    full_index = pd.date_range(
        monthly.index.min(),
        monthly.index.max(),
        freq="MS"
    )

    monthly = monthly.reindex(full_index, fill_value=0)

    cumulative = monthly.cumsum()

    plt.figure()

    ax = cumulative.plot(marker="o")

    plt.title("Cumulative Adoption by Tariff Plan")
    plt.ylabel("Households")
    plt.xlabel("Month")

    plt.xticks(rotation=30)

    ax.legend(title=None)

    return ax