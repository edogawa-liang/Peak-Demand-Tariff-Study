import pandas as pd
import numpy as np
import time
import holidays



class ElectricityAggregator:
    def __init__(self, meter_df: pd.DataFrame, tariff_df: pd.DataFrame = None):
        print("Initializing ElectricityAggregator...")

        self.df = meter_df.copy()
        self.df["TIDPUNKT"] = pd.to_datetime(self.df["TIDPUNKT"]).dt.tz_localize(None)

        self.tariff_df = tariff_df

        self.freq_map = {
            "day": "D",
            "week": "W",
            "month": "M",
            "quarter": "Q",
            "year": "Y",
            "hour": "hour",
            "week_part": "week_part",
            "weekday": "weekday",
        }

        print(f"Loaded {len(self.df):,} rows")

    ############################################
    # tariff merge
    ############################################

    def _merge_tariff(self):
        if self.tariff_df is None:
            return

        print("Merging tariff data...")

        tariff = self.tariff_df.rename(
            columns={
                "GS1-nr.": "aID",
                "Startdatum": "tariff_start",
                "Produktnamn": "tariff_plan",
            }
        ).copy()

        tariff["tariff_start"] = pd.to_datetime(tariff["tariff_start"]).dt.tz_localize(None)

        # fix dtype
        self.df["aID"] = self.df["aID"].astype(str)
        tariff["aID"] = tariff["aID"].astype(str)

        self.df = self.df.merge(tariff, on="aID", how="left", sort=False)

        self.df["has_tariff"] = self.df["tariff_start"].notna().astype(int)

        self.df["tariff_active"] = np.where(
            (self.df["has_tariff"] == 1) & (self.df["TIDPUNKT"] >= self.df["tariff_start"]),
            1,
            0,
        )

    ############################################
    # user usage group: Base on entire data period
    ############################################

    def _create_usage_group(self):
        if self.user_group_col is None:
            return

        print("Creating usage groups...")

        # total consumption per household
        usage = (
            self.df.groupby("aID", sort=False)["FORBRUKNING_KWH"]
            .sum()
            .rename("total_consumption")
        )

        # quantile thresholds
        q = usage.quantile(self.user_group_col)
        q1, q2 = float(q.iloc[0]), float(q.iloc[1])

        # create usage group
        if q2 <= q1:
            usage_group = pd.cut(
                usage,
                bins=[-np.inf, q1, np.inf],
                labels=["low", "high"],
                include_lowest=True,
            )
        else:
            usage_group = pd.cut(
                usage,
                bins=[-np.inf, q1, q2, np.inf],
                labels=["low", "medium", "high"],
                include_lowest=True,
            )

        # combine total_consumption + usage_group
        usage_df = pd.concat(
            [usage, usage_group.rename("usage_group")],
            axis=1
        ).reset_index()

        # merge back to main dataframe
        self.df = self.df.merge(
            usage_df,
            on="aID",
            how="left",
            sort=False
        )
        

    ############################################
    # price period
    ############################################

    def _add_price_period(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        ts = df["TIDPUNKT"]

        # Sweden public holidays
        se_holidays = holidays.country_holidays("SE")

        is_holiday = ts.dt.date.isin(se_holidays)
        is_weekend = ts.dt.weekday >= 5
        is_winter = ts.dt.month.isin([11, 12, 1, 2, 3])
        is_weekday = ts.dt.weekday < 5
        is_peak_hour = ts.dt.hour.between(7, 19)  # 07–20

        high_price = (
            is_winter
            & is_weekday
            & is_peak_hour
            & ~is_weekend
            & ~is_holiday
        )

        df["price"] = np.where(high_price, "high", "low")

        return df

    def _apply_price_mode(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        - use_price=False: price='all' only
        - use_price=True : price in {'all','high','low'}
        """
        if not self.use_price:
            df = df.copy()
            df["price"] = "all"
            return df

        # high/low
        df_hl = self._add_price_period(df)

        # all (duplicate rows, but with price='all')
        df_all = df.copy()
        df_all["price"] = "all"

        return pd.concat([df_all, df_hl], ignore_index=True)

    ############################################
    # aggregation helpers
    ############################################

    def _compute_one_agg(self, df: pd.DataFrame, group_cols: list[str], method: str) -> pd.DataFrame:
        # standard aggregations
        if method in ["sum", "mean", "variance", "median", "max"]:
            func_map = {
                "sum": "sum",
                "mean": "mean",
                "variance": "var",
                "median": "median",
                "max": "max",
            }
            out = (
                df.groupby(group_cols, sort=False)["FORBRUKNING_KWH"]
                .agg(func_map[method])
                .reset_index()
            )
            out = out.rename(columns={"FORBRUKNING_KWH": f"{method}_consumption"})
            return out

        # quantile methods like q90, q95
        if method.startswith("q"):
            try:
                q = float(method[1:]) / 100.0
            except ValueError:
                raise ValueError(f"Invalid quantile method: {method}")

            out = (
                df.groupby(group_cols, sort=False)["FORBRUKNING_KWH"]
                .quantile(q)
                .reset_index()
            )
            out = out.rename(columns={"FORBRUKNING_KWH": f"{method}_consumption"})
            return out

        # top3_mean
        if method == "top3_mean":
            out = (
                df.groupby(group_cols)["FORBRUKNING_KWH"]
                .nlargest(3)
                .groupby(group_cols)
                .mean()
                .reset_index()
            )
            out = out.rename(columns={"FORBRUKNING_KWH": f"{method}_consumption"})
            return out

        raise ValueError(f"Unknown aggregation method: {method}")

    ############################################
    # aggregation
    ############################################

    def _aggregate(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.freq not in self.freq_map:
            raise ValueError(f"Invalid freq: {self.freq}")

        print("Aggregating per household...")

        df = df.copy()

        freq = self.freq_map[self.freq]

        # hourly load profile
        if freq == "hour":
            df["period"] = df["TIDPUNKT"].dt.hour

        # weekday vs weekend
        elif freq == "week_part":
            df["period"] = np.where(
                df["TIDPUNKT"].dt.weekday < 5,
                "weekday",
                "weekend",
            )

        elif freq == "weekday":
            df["period"] = df["TIDPUNKT"].dt.weekday
            
        # monthly trend
        else:
            df["period"] = df["TIDPUNKT"].dt.to_period(freq).dt.to_timestamp()

        # Ensure price column according to use_price mode
        df = self._apply_price_mode(df)

        group_cols = ["aID", "period", "price", "tariff_active"]

        # normalize agg_methods to list
        methods = self.agg_methods

        # compute first, then merge others (wide)
        result = None
        for m in methods:
            part = self._compute_one_agg(df, group_cols, m)
            if result is None:
                result = part
            else:
                result = result.merge(part, on=group_cols, how="left", sort=False)

        # rename period to TIDPUNKT for output
        result = result.rename(columns={"period": "TIDPUNKT"})

        result = self._attach_tariff_info(result)
        result = self._attach_usage_group(result)

        return result

    def _attach_tariff_info(self, result: pd.DataFrame) -> pd.DataFrame:

        if "tariff_start" not in self.df.columns:
            return result

        tariff_info = (
            self.df[["aID", "tariff_start", "tariff_plan"]]
            .drop_duplicates("aID")
        )

        result = result.merge(
            tariff_info,
            on="aID",
            how="left",
            sort=False
        )

        return result

    def _attach_usage_group(self, result: pd.DataFrame) -> pd.DataFrame:
        if "usage_group" not in self.df.columns:
            return result

        usage_info = self.df[
            ["aID", "total_consumption", "usage_group"]
        ].drop_duplicates("aID")
        
        return result.merge(usage_info, on="aID", how="left", sort=False)

    def get_data(self) -> pd.DataFrame:
        return self.df

    ############################################
    # pipeline
    ############################################

    def run(
        self,
        freq="month",
        agg_method="top3_mean",          # str or list[str]
        use_price=True,                 # True => all+high+low ; False => all only
        add_user_group_col=None,
        output_path=None,
    ) -> pd.DataFrame:

        start = time.time()

        print("\n==============================")
        print("Electricity aggregation start")
        print("==============================")
        print(f"Rows: {len(self.df):,}")
        print(f"Frequency: {freq}")
        print(f"Aggregation: {agg_method}")
        print(f"Include price split (all/high/low): {use_price}")
        print(f"Add User groups: {add_user_group_col}")
        print("==============================")

        self.freq = freq
        self.use_price = use_price
        self.user_group_col = add_user_group_col

        # normalize agg_method to list
        if isinstance(agg_method, str):
            self.agg_methods = [agg_method]
        else:
            self.agg_methods = list(agg_method)

        # tariff
        if self.tariff_df is not None and "tariff_start" not in self.df.columns:
            t = time.time()
            self._merge_tariff()
            print(f"Tariff merge done ({time.time()-t:.2f}s)")

        # usage groups
        if add_user_group_col is not None and "usage_group" not in self.df.columns:
            t = time.time()
            self._create_usage_group()
            print(f"Usage groups created ({time.time()-t:.2f}s)")

        # aggregate
        t = time.time()
        result = self._aggregate(self.df)
        print(f"Aggregation done ({time.time()-t:.2f}s)")

        # save
        if output_path is not None:
            print("Saving parquet...")
            result.to_parquet(output_path, index=False)
            print("Saved.")

        print("==============================")
        print(f"Total runtime: {time.time()-start:.2f}s")
        print("==============================")

        return result