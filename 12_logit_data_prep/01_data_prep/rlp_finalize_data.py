####################################################################################################
# Finalize the RLP data.
# Our input is ct_decoded_full_attributes.csv which is produced by merge_SP_policy_data.R
# (Note: Haven't finished converting the R code to Python yet)
# We merge the RLP data with the energy data and calculate the dollar per mile for each vehicle
# We then save the final data to rlp_with_dollar_per_mile.csv
# We also calculate log horsepower to weight

####################################################################################################
# Sources
# https://afdc.energy.gov/vehicles/electric_emissions_sources.html
phev_elec_share = 0.563

####################################################################################################
# Import libraries
import pathlib
import pandas as pd
import numpy as np
from itertools import combinations, product
import os
from tqdm import tqdm
import requests
from datetime import datetime
import geopandas as gpd
import matplotlib.pyplot as plt
import warnings
import platform

# Warnings and display
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)

####################################################################################################
# Setup paths
str_cwd = pathlib.Path().resolve().parent
str_dir = str_cwd / "Documents" / "tobin_working_data"
str_rlp_data = str_dir / "rlpolk_data"
rlp_data_file = "ct_decoded_full_attributes.csv"

# Import the final RLP data
print(f"Importing the RLP data to be finalized, located at {str_rlp_data / rlp_data_file}")
df_rlp = pd.read_csv(str_rlp_data / rlp_data_file)
len_rlp = len(df_rlp)

####################################################################################################
# Import the energy price data
str_energy = str_dir / "energy_calcs"
str_gas_processed = str_energy / "monthly_gas_prices_processed.csv"
str_diesel_processed = str_energy / "monthly_diesel_prices_processed.csv"
str_electricity_processed = str_energy / "yearly_electricity_prices_processed.csv"

# Read in data
df_gas = pd.read_csv(str_gas_processed)
df_diesel = pd.read_csv(str_diesel_processed)
df_electricity = pd.read_csv(str_electricity_processed)

####################################################################################################
# Clean the RLP data
df_rlp.loc[:, "report_year"] = df_rlp.loc[:, "report_year_month"].astype(str).str[:4].astype(int)
df_rlp.loc[:, "report_month"] = df_rlp.loc[:, "report_year_month"].astype(str).str[4:].astype(int)
df_rlp = df_rlp.drop(columns=["report_year_month"])

# Check how many NAs there are in the combined col
assert(df_rlp["fuel"].isna().sum() == 0)

####################################################################################################
# Clean the energy data make all columns lower case
df_gas.columns = df_gas.columns.str.lower()
df_diesel.columns = df_diesel.columns.str.lower()
df_electricity.columns = df_electricity.columns.str.lower()

####################################################################################################
# Calculate log horsepower to weight
# First divided the curb weight by 1000
df_rlp["curb_weight"] = df_rlp["curb_weight"] / 1000
df_rlp["log_hp_wt"] = np.log(df_rlp["max_hp"] / df_rlp["curb_weight"])

####################################################################################################
# Merge the RLP data with the energy data
df_rlp = df_rlp.merge(df_gas[["year", "month", "gas_price_21"]], left_on = ["report_year", "report_month"],
                      right_on = ["year", "month"], how = "left")

df_rlp = df_rlp.merge(df_diesel[["year", "month", "diesel_price_21"]], left_on = ["report_year", "report_month"],
                      right_on = ["year", "month"], how = "left")

df_rlp = df_rlp.merge(df_electricity[["year", "electricity_price_21"]], left_on = ["report_year"],
                        right_on = ["year"], how = "left")

assert(len(df_rlp) == len_rlp), "Length mismatch"

# Confirm the match worked and drop the extra columns
assert(df_rlp["year_x"].equals(df_rlp["year_y"])), "Year mismatch"
assert(df_rlp["month_x"].equals(df_rlp["month_y"])), "Month mismatch"
assert(df_rlp["year_x"].equals(df_rlp["year"])), "Year mismatch"
assert(df_rlp["year_x"].equals(df_rlp["report_year"])), "Year mismatch"
assert(df_rlp["month_x"].equals(df_rlp["report_month"])), "Month mismatch"
df_rlp = df_rlp.drop(columns = ["year_x", "month_x", "year_y", "month_y", "year"])

####################################################################################################
# Convert variable values to lower case
df_rlp["fuel"] = df_rlp["fuel"].str.lower()
df_rlp["fuel1"] = df_rlp["fuel1"].str.lower()
df_rlp["fuel2"] = df_rlp["fuel2"].str.lower()

# Calculate the dollar per mile for gasoline and hybrid
mask = ((df_rlp["fuel"] == "gasoline")|(df_rlp["fuel"] == "hybrid"))
df_rlp.loc[mask, "dollar_per_mile"] = df_rlp.loc[mask, "gas_price_21"] / df_rlp.loc[mask, "combined"]

# Calculate the dollar per mile for diesel
mask = (df_rlp["fuel"] == "diesel")
df_rlp.loc[mask, "dollar_per_mile"] = df_rlp.loc[mask, "diesel_price_21"] / df_rlp.loc[mask, "combined"]

# Calculate the dollar per for flex fuel, using gas price
mask = (df_rlp["fuel"] == "flex fuel")
df_rlp.loc[mask, "dollar_per_mile"] = df_rlp.loc[mask, "gas_price_21"] / df_rlp.loc[mask, "combined"]

# Calculate the dollar per mile for electric EPA fueleconomy.gov says 33.7 kWh per gallon
mask = (df_rlp["fuel"] == "electric")
df_rlp.loc[mask, "dollar_per_mile"] = (df_rlp.loc[mask, "electricity_price_21"] / 100) * (33.7 / df_rlp.loc[mask, "combined"])

# Calculate the dollar per mile for PHEV 
mask = (df_rlp["fuel"] == "phev")

# There are some erroneous fuels to address 
# Mark the entries to drop and count how many 
df_rlp.loc[:, "phev_erroneous_fuels"] = 0
df_rlp.loc[mask & (df_rlp["fuel1"].isna()) & (df_rlp["fuel2"].isna()), "phev_erroneous_fuels"] = 1
phev_erroneous = df_rlp.loc[mask, "phev_erroneous_fuels"].sum()
print(f"Dropping {phev_erroneous} PHEV vehicles with missing fuel types")

# Drop them and confirm the right number were dropped
len_df = len(df_rlp)
df_rlp = df_rlp.loc[(df_rlp["phev_erroneous_fuels"] == 0)]
assert(len(df_rlp) == len_df - phev_erroneous)
assert(df_rlp.loc[mask, "fuel1"].unique().tolist() == ["gasoline"]), "PHEV fuel 1 is not gasoline" # Confirm the variable ordering
assert(df_rlp.loc[mask, "fuel2"].unique().tolist() == ["electricity"]), "PHEV fuel 2 is not electric"
df_rlp.loc[mask, "dollar_per_mile_gas"] = df_rlp.loc[mask, "gas_price_21"] / df_rlp.loc[mask, "combined_mpg1"]
df_rlp.loc[mask, "dollar_per_mile_elec"] = (df_rlp.loc[mask, "electricity_price_21"] / 100) * (33.7 / df_rlp.loc[mask, "combined_mpg2"])
df_rlp.loc[mask, "dollar_per_mile"] = (df_rlp.loc[mask, "dollar_per_mile_gas"] * (1 - phev_elec_share)) + (df_rlp.loc[mask, "dollar_per_mile_elec"] * phev_elec_share)
df_rlp = df_rlp.drop(columns = ["dollar_per_mile_gas", "dollar_per_mile_elec"])

# Drop any vehicles not in the categories above
dropped_dollar_per_mile = df_rlp["dollar_per_mile"].isna().sum()
df_rlp = df_rlp.dropna(subset = ["dollar_per_mile"])
print(f"Dropped {dropped_dollar_per_mile} vehicles that are not gasoline, hybrid diesel, flex fuel, or phev")

####################################################################################################
# Now we group by model year, make, model, and trim 
num_unique_vins = len(df_rlp["vin_pattern"].unique())
unique_vehicles = df_rlp[['model_year', 'make', 'model', 'trim']].drop_duplicates()
num_unique_vehicles = len(unique_vehicles)
print(f"Number of unique VINs: {num_unique_vins}")
print(f"Number of unique vehicles: {num_unique_vehicles}")

####################################################################################################
# Save the final data
str_rlp_final = str_rlp_data / "rlp_with_dollar_per_mile.csv"
df_rlp.to_csv(str_rlp_final, index = False)
