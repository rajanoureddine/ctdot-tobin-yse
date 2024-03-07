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
warnings.filterwarnings("ignore")
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)

####################################################################################################
# Sources
# CPI data downloaded from Dropbox
# Gas price data downloaded from EIA: https://www.eia.gov/petroleum/gasdiesel/
# Downloaded the XLS, took sheet 1, then converted to CSV

# Electricity prices from https://www.eia.gov/electricity/data/state/
# Used EIA 861 and kept Residential, Total Electricity industry


####################################################################################################
# Setup paths
str_cwd = pathlib.Path().resolve().parent
str_dir = str_cwd / "Documents" / "tobin_working_data"
str_energy = str_dir / "energy_calcs"
str_cpi = str_energy / "CPI.csv"  # CPI data downloaded from Dropbox
str_gas = str_energy / "gas_price_raw.csv"
str_diesel = str_energy / "diesel_price_raw.csv"
str_electricity = str_energy / "electricity_price_raw.csv"

# Read in data
df_cpi = pd.read_csv(str_cpi)
df_gas = pd.read_csv(str_gas, skiprows=2)
df_diesel = pd.read_csv(str_diesel, skiprows=2)
df_electricity = pd.read_csv(str_electricity, skiprows = 2)

####################################################################################################
# Clean up gas data
df_gas.index = pd.to_datetime(df_gas["Date"])
df_gas = df_gas.iloc[:, df_gas.columns.str.contains("New England")].reset_index()
df_gas = df_gas.loc[df_gas["Date"] > datetime(2017, 1, 1), :]
df_gas.columns = ["Date", "Gas_Price"]

# Clean up diesel data
df_diesel.index = pd.to_datetime(df_diesel["Date"])
df_diesel = df_diesel.iloc[:, df_diesel.columns.str.contains("New England")].reset_index()
df_diesel = df_diesel.loc[df_diesel["Date"] > datetime(2017, 1, 1), :]
df_diesel.columns = ["Date", "Diesel_Price"]

# Clean up electricity data
df_electricity = df_electricity.loc[df_electricity["STATE"] == "CT", ["Year", "STATE", "Cents/kWh"]]
df_electricity = df_electricity.rename(columns={"Cents/kWh": "Electricity_Price"})


####################################################################################################
# Calculate monthly averages for gas
df_gas["Month"] = df_gas["Date"].dt.month
df_gas["Year"] = df_gas["Date"].dt.year
df_gas = df_gas.groupby(["Year", "Month"]).mean().reset_index().drop("Date", axis=1)

# Calculate monthly averages for diesel
df_diesel["Month"] = df_diesel["Date"].dt.month
df_diesel["Year"] = df_diesel["Date"].dt.year
df_diesel = df_diesel.groupby(["Year", "Month"]).mean().reset_index().drop("Date", axis=1)

####################################################################################################
# Process CPI data
# Convert to long
df_cpi_long = df_cpi.melt(id_vars="Year", var_name="Month", value_name="CPI")

# For all 2022 data, use the value for January 2022
df_cpi_long.loc[df_cpi_long["Year"] == 2022, "CPI"] = df_cpi_long.loc[(df_cpi_long["Year"] == 2022) 
                                                                       & (df_cpi_long["Month"] == "Jan"), "CPI"].values[0]

# Get a relative value to 2021
cpi_21 = df_cpi_long.loc[(df_cpi_long["Year"]==2021)&(df_cpi_long["Month"].isin(["HALF1", "HALF2"])), "CPI"].mean()
df_cpi_long["mult_21"] = cpi_21 / df_cpi_long["CPI"] 

# Convert months to numbers and drop HALF1 and HALF2
df_cpi_long["Month"] = df_cpi_long["Month"].replace({"Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, 
                                                     "Jun": 6, "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10, 
                                                     "Nov": 11, "Dec": 12})
df_cpi_long = df_cpi_long.loc[~df_cpi_long["Month"].isin(["HALF1", "HALF2"]), :]

####################################################################################################
# Merge CPI and GAS data
df_gas_cpi = pd.merge(df_gas, df_cpi_long, on=["Year", "Month"])
df_gas_cpi["Gas_Price_21"] = df_gas_cpi["Gas_Price"] * df_gas_cpi["mult_21"]

# Merge CPI and DIESEL data
df_diesel_cpi = pd.merge(df_diesel, df_cpi_long, on=["Year", "Month"])
df_diesel_cpi["Diesel_Price_21"] = df_diesel_cpi["Diesel_Price"] * df_diesel_cpi["mult_21"]

# Merge CPI and Electricy data
df_cpi_yearly = df_cpi_long.groupby("Year").mean().reset_index().drop("Month", axis=1)
df_electricity_cpi = pd.merge(df_electricity[["Year", "Electricity_Price"]], df_cpi_yearly, on=["Year"])
df_electricity_cpi["Electricity_Price_21"] = df_electricity_cpi["Electricity_Price"] * df_electricity_cpi["mult_21"]

# Print values for 2019
print(f"Gas 2019 average: {df_gas_cpi.loc[df_gas_cpi['Year'] == 2019, 'Gas_Price_21'].mean()}")
print(f"Diesel 2019 average: {df_diesel_cpi.loc[df_diesel_cpi['Year'] == 2019, 'Diesel_Price_21'].mean()}")
print(f"Electricity 2019 average: {df_electricity_cpi.loc[df_electricity_cpi['Year'] == 2019, 'Electricity_Price_21'].mean()}")

# Print values for 2020
print(f"Gas 2020 average: {df_gas_cpi.loc[df_gas_cpi['Year'] == 2020, 'Gas_Price_21'].mean()}")
print(f"Diesel 2020 average: {df_diesel_cpi.loc[df_diesel_cpi['Year'] == 2020, 'Diesel_Price_21'].mean()}")
print(f"Electricity 2020 average: {df_electricity_cpi.loc[df_electricity_cpi['Year'] == 2020, 'Electricity_Price_21'].mean()}")

####################################################################################################
# Save gas price data
str_gas_out = str_energy / "monthly_gas_prices_processed.csv"
df_gas_cpi.to_csv(str_gas_out, index=False)

# Save diesel price data
str_diesel_out = str_energy / "monthly_diesel_prices_processed.csv"
df_diesel_cpi.to_csv(str_diesel_out, index=False)

# Save electricity price data
str_electricity_out = str_energy / "yearly_electricity_prices_processed.csv"
df_electricity_cpi.to_csv(str_electricity_out, index=False)


