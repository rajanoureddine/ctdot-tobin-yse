####################################################################################################
# Import libraries
import pathlib
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt

####################################################################################################
# Setup paths
str_cwd = pathlib.Path().resolve().parent.parent
str_dir = str_cwd / "Documents" / "tobin_working_data"
str_sp = str_dir / "rlpolk_data"
str_decoder = str_dir / "vin_decoder"

# Set the state
state = "CT"

####################################################################################################
# Read in data #####################################################################################

# Read in the RLP data
str_ct_zip = "ct_zip_sp_040324.csv"
str_ct_decoded = "ct_rlp_vin_decoded_040324.csv"
df_ct_zip = pd.read_csv(str_sp / str_ct_zip)
df_ct_decoded = pd.read_csv(str_sp / str_ct_decoded)

# Read in EPA data
str_epa = str_dir / "raw" / "fuel_economy" / "vehicles.csv"
str_epa_22 = str_dir / "raw" / "fuel_economy" / "vehicles_23.csv"
df_epa = pd.read_csv(str_cwd / str_epa)
try:
    df_epa_22 = pd.read_csv(str_cwd / str_epa_22)
except:
    print("We do not have the EPA 2023 data")

# Read in crosswalks
str_cwalks = str_dir / "intermediate" / "data_crosswalk" 
str_cwalk_evs = str_cwalks / "vin_EPA_fe_crosswalk.csv"
str_cwalk_tesla = str_cwalks / "vin_EPA_fe_crosswalk_tesla.csv"

df_cwalk_evs = pd.read_csv(str_cwalk_evs)
df_cwalk_tesla = pd.read_csv(str_cwalk_tesla)

# Read in 2022 crosswalks
str_cwalk_evs_22 = str_cwalks / "vin_EPA_fe_2022_crosswalk.csv"
str_cwalk_evs_missing_22 = str_cwalks / "vin_EPA_fe_2022_missing_crosswalk.csv"

df_cwalk_evs_22 = pd.read_csv(str_cwalk_evs_22)
df_cwalk_evs_missing_22 = pd.read_csv(str_cwalk_evs_missing_22)

# Read in battery size data
str_battery  = str_dir / "intermediate" / "ev_model_style_trim_battery.csv"
str_battery23 = str_dir / "intermediate" / "ev_model_style_trim_battery_23.csv"
df_battery = pd.read_csv(str_battery)
df_battery23 = pd.read_csv(str_battery23)

####################################################################################################
# NOTE: DID NOT READ IN THE FEDERAL INCENTIVE DATA, OR ZEV CREDIT DATA
####################################################################################################

