# --- Import required packages
import pathlib
import pandas as pd
import numpy as np
from itertools import combinations
import os
from tqdm import tqdm
import requests
from datetime import datetime
import geopandas as gpd
import matplotlib.pyplot as plt
import warnings
import platform
warnings.filterwarnings("ignore")

from logit_demand_functions import * 

# --- Set paths ----------------
if platform.platform()[0:5] == 'macOS':
    cd = pathlib.Path().resolve().parent
    str_project = cd / "Documents" 
    data_path = cd / "Documents" / "tobin_working_data"
    rlpolk_data_path = data_path / "rlpolk_data"
    vin_matching_pathpackages= data_path / "vin_matching"
else:
    data_path = pathlib.Path().resolve().parent.parent / "rn_home" / "data"
    rlpolk_data_path = data_path / "rlpolk_data"
    vin_matching_pathpackages= data_path / "vin_matching"

# --- Import RLPolk Data (partially matched) ------
rlp_raw = pd.read_csv(rlpolk_data_path / "rlpolk_data_matched.csv", index_col = [0])
evs_only = rlp_raw[rlp_raw["Fuel Type - Primary"] == "Electric"].reset_index(drop = True)
ev_vins = evs_only["vin_corrected"].unique().tolist()

# --- Import Experian Data
# exp_data = pd.read_csv(data_path / "intermediate" / "US_VIN_data_common.csv")
# exp_ct = exp_data[exp_data["state"]=="CONNECTICUT"]
# exp_ct_ev = exp_ct[exp_ct["fueltype"]=="L"]

# Download some characteristics and show what we get - mostly NA
if (True):
    ev_chars = pd.DataFrame([])

    for vin in tqdm(ev_vins):
        # vin = row["vin_corrected"]
        try:
            chars = fetch_vin_data(vin)
        except:
            pass
            
        ev_chars = pd.concat([ev_chars, chars], axis = 1).reset_index(drop = True)

    ev_chars.to_csv(rlpolk_data_path / "rlp_ev_characteristics_012624.csv")

# Check EV sales per model year
rlp_evs_summary = evs_only[["year", "VEH_COUNT"]].groupby("year").sum().reset_index()
# exp_evs_summary = exp_ct_ev[["year", "agg_count"]].groupby("year").sum().reset_index()
print(rlp_evs_summary.head(10))
# print(exp_evs_summary.head(10))
print(rlp_evs_summary.iloc[0:6, 0].sum())