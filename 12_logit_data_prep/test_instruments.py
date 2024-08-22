"""
This file runs a number of tests of the Bartik instruments that can be generated using the code in generate_instruments.py.
That code generates Bartik instruments that multiply the number of L2 or DC chargers in every county except the county in question,
by the retail density of that county at some point in time. Two possible time-points are considered:
- A measure of retail density based on Advan data from 2020 (that we call "current" retail density)
- A measure of retail density based on Dun & Bradstreet data from 1969 (that we call "historic" retail density)

This code runs a number of tests to check the strength and validity of those instruments. 
"""
###########################################################################################
# Import required libraries
import pandas as pd
import numpy as np

from pathlib import Path

import statsmodels.api as sm # OLS

from paths import data_dir

import matplotlib.pyplot as plt

###########################################################################################
type = "historic" # Choose between "historic" and "current"

###########################################################################################
# Instrument data and charging data
if type == "curent":
    instrument_data = pd.read_csv(data_dir / "instruments" / "instruments.csv", index_col =[0])
elif type == "historic":
    instrument_data = pd.read_csv(data_dir / "instruments" / "instruments_historic.csv", index_col =[0])

charging_data = pd.read_csv(data_dir / "charging_stations_output" / "charging_stations_extended.csv")

# Merge data
merged_data = charging_data[["market_ids", "county", "year", "charging_density_total", "charging_density_DC", "charging_density_L2"]]
merged_data = merged_data.merge(instrument_data[["market_ids", "other_counties_county_area_sq_km", "other_counties_total_stations_L2",  "other_counties_total_stations_DC", "bartik_instrument_L2", "bartik_instrument_DC", "bartik_instrument_total"]], on = "market_ids", how = "left")

# Add fixed effects
merged_data["t"] = merged_data.year - merged_data.year.min()
merged_data = pd.concat([merged_data, pd.get_dummies(merged_data.county)*1], axis = 1)

# Decide on whether to run analysis for L2 or DC
charger_type = "L2"

# Create a plot of L2 charging density against the instrument, including the county names
fig, ax = plt.subplots()
for county in merged_data.county.unique():
    ax.scatter(merged_data[merged_data.county == county][f"bartik_instrument_{charger_type}"], merged_data[merged_data.county == county][f"charging_density_{charger_type}"], label = county.title())
ax.legend()
ax.set_xlabel("Instrument")
ax.set_ylabel(f"{charger_type} Charging Density")
plt.show()

# Create a plot of L2 charging density against L2 charging density in other counties
merged_data[f"charging_density_{charger_type}_other_counties"] = merged_data[f"other_counties_total_stations_{charger_type}"] / merged_data["other_counties_county_area_sq_km"]
fig, ax = plt.subplots()
for county in merged_data.county.unique():
    # Scatter plot with trendline
    ax.scatter(merged_data[merged_data.county == county][f"charging_density_{charger_type}_other_counties"], merged_data[merged_data.county == county][f"charging_density_{charger_type}"], label = county.title())
ax.legend()
ax.set_xlabel(f"{charger_type} Charging Density in Other Counties")
ax.set_ylabel(f"{charger_type} Charging Density")
plt.show()

# Run the estimation without fixed effects
Y = merged_data[f"charging_density_{charger_type}"]
X = merged_data[f"bartik_instrument_{charger_type}"]
X = sm.add_constant(X)

model = sm.OLS(Y, X)
results = model.fit()

print("Printing WITHOUT fixed effects\n-----------------------------------")
print(results.summary().tables[0].as_latex_tabular())
print(results.summary().tables[1].as_latex_tabular())

# Run the estimation with fixed effects
cols_fe = [f"bartik_instrument_{charger_type}", "t"] + [x for x in merged_data.county.unique().tolist()[1:]]
X_fe = merged_data[cols_fe]

model_fe = sm.OLS(Y, sm.add_constant(X_fe))
results_fe = model_fe.fit()

print("Printing WITH fixed effects\n-----------------------------------")
print(results_fe.summary().tables[0].as_latex_tabular())
print(results_fe.summary().tables[1].as_latex_tabular())