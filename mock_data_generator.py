import numpy as np
import pandas as pd
import csv
import os
from datetime import datetime, timedelta, date


def generate_mock_data_all(
    out_dir: str = "mock_data",
    n_households: int = 500,
    n_tariff: int = 100,
    n_survey: int = 50,
    seed: int = 2026,
    start_dt: datetime = datetime(2024, 1, 1, 0, 0, 0),
    end_dt: datetime = datetime(2026, 2, 20, 0, 0, 0),
    matdata_filename: str | None = None,
    tariff_filename: str | None = None,
    survey_filename: str | None = None,
):

    assert n_tariff <= n_households
    rng = np.random.default_rng(seed)

    os.makedirs(out_dir, exist_ok=True)

    if matdata_filename is None:
        matdata_filename = f"meter_data_{n_households}.snappy.parquet"

    if tariff_filename is None:
        tariff_filename = f"tariff_{n_tariff}.csv"

    if survey_filename is None:
        survey_filename = f"survey_{n_survey}.csv"

    matdata_path = os.path.join(out_dir, matdata_filename)
    tariff_path = os.path.join(out_dir, tariff_filename)
    survey_path = os.path.join(out_dir, survey_filename)

    # --------------------------
    # create household IDs
    # --------------------------
    base_prefix = "7359991662"
    households = [
        base_prefix + f"{i:05d}" + f"{rng.integers(0, 1000):03d}"
        for i in range(n_households)
    ]
    households = list(dict.fromkeys(households))

    # --------------------------
    # create hourly meter data
    # --------------------------
    timestamps = pd.date_range(
        start=start_dt,
        end=end_dt,
        freq="h",
        tz="UTC"
    )

    n_hours = len(timestamps)

    hour_arr = timestamps.hour.to_numpy()
    month_arr = timestamps.month.to_numpy()
    weekday_arr = timestamps.weekday.to_numpy()

    h = hour_arr.astype(float)
    morning = np.exp(-0.5 * ((h - 8) / 2.0) ** 2)
    evening = np.exp(-0.5 * ((h - 19) / 2.5) ** 2)
    baseline = 0.25 + 0.05 * np.cos((h / 24.0) * 2 * np.pi)
    shape = baseline + 0.35 * morning + 0.55 * evening

    season_mult = np.ones(n_hours)
    winter = {11, 12, 1, 2, 3}
    summer = {6, 7, 8}

    for m in range(1, 13):
        idx = month_arr == m
        if m in winter:
            season_mult[idx] = 1.25
        elif m in summer:
            season_mult[idx] = 0.90
        else:
            season_mult[idx] = 1.05

    weekend_mult = np.where(weekday_arr >= 5, 1.05, 1.0)
    template = shape * season_mult * weekend_mult

    hh_scale = rng.lognormal(mean=np.log(1.0), sigma=0.35, size=n_households)

    rows = []

    for i, aid in enumerate(households):
        scale = hh_scale[i]
        noise = rng.normal(0, 0.05, size=n_hours)
        cons = np.clip(template * scale + noise, 0, None)

        for j in range(n_hours):
            rows.append([
                aid,
                timestamps[j],
                timestamps[j].date(),
                round(cons[j], 8),
            ])

    df_mat = pd.DataFrame(
        rows,
        columns=[
            "aID",
            "TIDPUNKT",
            "TIDPUNKT_DAG",
            "FORBRUKNING_KWH",
        ],
    )

    df_mat.to_parquet(
        matdata_path,
        index=False,
        compression="snappy",
        engine="pyarrow"
    )

    # --------------------------
    # generate tariff (Start from 1th every month)
    # --------------------------
    tariff_households = rng.choice(households, size=n_tariff, replace=False)

    month_weights = {
        3: 0.25,
        4: 0.33,
        5: 0.05,
        6: 0.05,
        7: 0.1,
        8: 0.02,
        9: 0.02,
        10: 0.02,
        11: 0.06,
        12: 0.1,
    }

    months = np.array(list(month_weights.keys()))
    weights = np.array(list(month_weights.values()))
    weights = weights / weights.sum()

    produkt_variants = [
        "GENAB Tidsindelad 6 kW Villa",
        "GENAB Tidsindelad 14 kW Villa",
    ]

    with open(tariff_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Produktnamn", "Startdatum", "GS1-nr."])

        for aid in tariff_households:
            chosen_month = int(rng.choice(months, p=weights))
            sd = date(2025, chosen_month, 1)
            produkt = rng.choice(produkt_variants)

            w.writerow([produkt, sd.strftime("%Y-%m-%d"), aid])

    # --------------------------
    # survey
    # --------------------------
    with open(survey_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "Respondent",
            "Respondent_ID",
            "Answer_created",
            "Answer_completed",
            "Vilket_kön",
            "Id"
        ])

        for _ in range(n_survey):
            w.writerow([
                "Anonymous",
                int(rng.integers(411000000, 412000000)),
                datetime(2025, 5, 12).isoformat(),
                datetime(2025, 5, 12, 1).isoformat(),
                "man",
                rng.choice(households),
            ])

    print("Done.")
    print(f"matdata -> {matdata_path}")
    print(f"tariff  -> {tariff_path}")
    print(f"survey  -> {survey_path}")


if __name__ == "__main__":
    generate_mock_data_all(
        out_dir="mock_data",
        n_households=500,
        n_tariff=100,
        n_survey=50,
        seed=2026
    )