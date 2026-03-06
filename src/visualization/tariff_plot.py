import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")


def plot_tariff_cumulative(tariff_df):
    """
    Cumulative number of households choosing tariff over time.
    """

    df = tariff_df.copy()

    df["tariff_start"] = pd.to_datetime(df["tariff_start"])

    monthly = (
        df.groupby(df["tariff_start"].dt.to_period("M"))
        .size()
        .sort_index()
    )

    cumulative = monthly.cumsum()
    cumulative.index = cumulative.index.to_timestamp()

    plt.figure()
    cumulative.plot()
    plt.title("Cumulative Tariff Adoption")
    plt.ylabel("Households")
    plt.xlabel("Month")

    return plt.gca()


def plot_tariff_share(tariff_df, total_households):
    """
    Share of households adopting tariff.
    """

    df = tariff_df.copy()

    df["tariff_start"] = pd.to_datetime(df["tariff_start"])

    monthly = (
        df.groupby(df["tariff_start"].dt.to_period("M"))
        .size()
        .sort_index()
    )

    cumulative = monthly.cumsum()
    share = cumulative / total_households

    share.index = share.index.to_timestamp()

    plt.figure()
    share.plot()

    plt.title("Tariff Adoption Share")
    plt.ylabel("Share of households")
    plt.xlabel("Month")

    return plt.gca()


def plot_tariff_plan_counts(tariff_df):
    """
    Number of households choosing each tariff plan.
    """

    counts = tariff_df["tariff_plan"].value_counts()

    plt.figure()
    counts.plot(kind="bar")

    plt.title("Tariff Plan Choices")
    plt.ylabel("Households")
    plt.xlabel("Tariff Plan")

    plt.xticks(rotation=45)

    return plt.gca()