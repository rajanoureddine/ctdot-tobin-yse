# --- Import required packages
import pathlib
import pandas as pd
import numpy as np
from itertools import combinations
import os
# from tqdm import tqdm
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
elif(False):
    data_path = pathlib.Path().resolve().parent.parent / "rn_home" / "data"
    rlpolk_data_path = data_path / "rlpolk_data"
    vin_matching_pathpackages= data_path / "vin_matching"

# --- Import RLPolk Data (partially matched) ------
rlp_raw = pd.read_csv(rlpolk_data_path / "rlpolk_data_matched.csv", index_col = [0])
evs_only = rlp_raw[rlp_raw["Fuel Type - Primary"] == "Electric"].reset_index(drop = True)

# --- Match data to get further details -----------
ev_chars = pd.DataFrame([])

for index, row in evs_only.iterrows():
    vin = row["vin_corrected"]
    try:
        chars = fetch_vin_data(vin)
    except:
        pass
    ev_chars = pd.concat([ev_chars, chars]).reset_index(drop = True)
    if index == 20:
        pass

ev_chars.to_csv(rlpolk_data_path / "ev_characteristics.csv")

