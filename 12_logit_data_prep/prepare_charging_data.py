"""
This file is used to prepare charging density data for the logit model.
For now, we take as our input pre-produced files from the Dropbox under
AutoMakerStrategies/data/ChargingMoments/outputs 
"""
###############################################################################
## Importing libraries
import pandas as pd
import numpy as np

from pathlib import Path

# Import OLS regression
from sklearn.linear_model import LinearRegression

# Silence warnings
import warnings
warnings.filterwarnings('ignore')

###############################################################################
# Set up the paths
cd = Path().resolve()
data_dir = cd.parent / "Documents" / "tobin_working_data"
input_dir = data_dir / "charging_stations"
output_dir = data_dir / "charging_stations_output"

###############################################################################
# Read in the CSVs
data = []
for file in input_dir.iterdir():
    df = pd.read_csv(file)
    data = data + [df]
data = pd.concat(data)

# Get CT data only and drop the column "source"
data = data.loc[data['state'] == 'CT']
data = data.drop(columns = ['source: 2021_charging_density_by_county.csv '])
data_extended = data.copy()

# Now define a function to extrapolate to 2022
def extrapolate_to_2022(data, trend_col, cols_to_keep, cols_to_extrapolate):
    """
    This function takes in a dataframe and extrapolates the columns cols_to_extrapolate
    to 2022 using the trend_col.
    """
    # Create blank output dataframe
    output_data = pd.DataFrame(columns = data.columns)
    output_data.loc[0, trend_col] = 2022
    output_data.loc[0, cols_to_keep] = data.loc[1, cols_to_keep]

    # Now loop through cols to extrapolate and extrapolate them to 2022
    for col in cols_to_extrapolate:
        X = data[[trend_col]]
        Y = data[[col]]

        # Fit a linear model
        model = LinearRegression().fit(X, Y)
        X_pred = np.array([[2022]])
        y_pred = model.predict(X_pred)

        # Append to output_data
        output_data.loc[0, col] = y_pred[0][0]
    
    # Now add the densities
    output_data['charging_density_total'] = output_data['total_stations'] / output_data['county_area_sq_km']
    output_data['charging_density_L2'] = output_data['total_stations_L2'] / output_data['county_area_sq_km']
    output_data['charging_density_DC'] = output_data['total_stations_DC'] / output_data['county_area_sq_km']
    
    return output_data

for county in data['county'].unique().tolist():
    # Get the data for the county
    county_data = data.loc[data['county'] == county].reset_index(drop = True)

    # Extrapolate to 2022
    cols_to_extrapolate = ['total_stations', 'total_stations_L2', 'total_stations_DC']
    output_data = extrapolate_to_2022(county_data, 'year', ['state', 'fips', 'county_area_sq_km', 'county'],  cols_to_extrapolate)

    # Append to the data
    data_extended = pd.concat([data_extended, output_data])

# Clean up the data
data_extended = data_extended.reset_index(drop = True)
data_extended["year"] = data_extended["year"].astype(int)

# add market ids
data_extended["market_ids"] = data_extended["county"].astype(str).str.upper()+"_"+data_extended["year"].astype(str)

# Save the data
data_extended.to_csv(output_dir / "charging_stations_extended.csv", index = False)

