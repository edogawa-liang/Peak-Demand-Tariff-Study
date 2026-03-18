import pandas as pd
import json
import os
import statsmodels.formula.api as smf

BASE_DIR = "output/matching"

# -------------------------
# load
# -------------------------
def load_matching(name, base_dir=BASE_DIR):
    folder = os.path.join(base_dir, name)

    matches = pd.read_parquet(os.path.join(folder, "matches.parquet"))
    profiles = pd.read_parquet(os.path.join(folder, "profiles.parquet"))
    balance = pd.read_parquet(os.path.join(folder, "balance.parquet"))

    with open(os.path.join(folder, "config.json"), "r") as f:
        config = json.load(f)

    return matches, profiles, balance, config


# -------------------------
# prepare data
# -------------------------
def prepare_data(df, matches, id_col="id", time_col="month"):
    
    treated_ids = matches["treated_id"].unique()
    control_ids = matches["control_id"].unique()

    df = df[df[id_col].isin(list(treated_ids) + list(control_ids))].copy()

    adopt_map = matches[["treated_id", "adoption_month"]].drop_duplicates()
    adopt_map = adopt_map.rename(columns={"treated_id": id_col})

    df = df.merge(adopt_map, on=id_col, how="left")

    df["treated"] = df[id_col].isin(treated_ids).astype(int)
    df["post"] = df[time_col] >= df["adoption_month"]

    return df


# -------------------------
# ATT (pair)
# -------------------------
def compute_att_pairs(df, matches, outcome="consumption"):
    
    df_post = df[df["post"] == True]

    avg = df_post.groupby("id")[outcome].mean()

    pairs = matches.copy()
    pairs["treated_y"] = pairs["treated_id"].map(avg)
    pairs["control_y"] = pairs["control_id"].map(avg)

    pairs = pairs.dropna()
    pairs["diff"] = pairs["treated_y"] - pairs["control_y"]

    return pairs["diff"].mean()


# -------------------------
# DID
# -------------------------
def run_did(df, outcome="consumption"):
    
    model = smf.ols(
        f"{outcome} ~ treated * post",
        data=df
    ).fit()

    return model