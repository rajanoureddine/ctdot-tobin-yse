# DataFrames and Math
import pandas as pd
import numpy as np
import tqdm
pd.options.display.max_columns = 100

# API management
import json
import base64
import requests
from getpass import getpass

# Plotting
import matplotlib.pyplot as plt

# Paths
import pathlib
import os
import platform
import glob
import gzip

############################################################################################################
# Set the paths
if platform.platform()[0:5] == 'macOS':
    on_cluster = False
    cd = pathlib.Path().resolve().parent
    str_project = cd / "Documents" 
    str_data = str_project / "tobin_working_data"
    footfall_output_path = str_data / "advan_data_footfall"
    processed_output_path = str_data / "advan_data_footfall_processed"

if platform.platform()[0:5] == 'Linux':
    on_cluster = True
    home_path = pathlib.Path().resolve().parent.parent/"rn_home"
    data_output_path = home_path / "data" / "advan_data"
    footfall_output_path = home_path / "data" / "advan_data_footfall"
    processed_output_path = home_path / "data" / "advan_data_processed"

############################################################################################################
# Define a helper function to download advan data
def download_advan_data(url, partition, output_path):
    # Get Raja's access token
    access_token = "ZtyVxTnU.3dJjvzC8DfiOEp2g7EVWSOCwhznve2GUHfdvtWRrLCe5YwTvCKPwo6BS"

    # Get the links to download individual files
    results = requests.get(url=url,
                        params = {"partition_keys": partition},
                       headers={
                        "X-API-KEY": access_token,
                        'accept': 'application/json'
                       })
    # Download the files we want
    # Download all files for one month
    for item in results.json()['download_links']:
       if item['partition_key'] == partition:
           filename = item['file_name']
           print(f"Downloading {filename}")
           link = item['link']
           data = requests.get(link)
           open(output_path / filename, 'wb').write(data.content)

def extract_ct_data(input_path, partitions, output_path, output_name, save = False):
    # Create blank data frame
    ct_data = pd.DataFrame([])

    # Iterate through all the files in the directory
    for file in input_path.iterdir():
        for partition in partitions:
            if partition in file.name:
                print(f"Reading file {file.name}")
                # Read the file
                data = pd.read_csv(file)
                # Extract rows for CT
                ct_rows = data[data["REGION"]=="CT"].reset_index(drop=True)
                
                # Update the DataFrame
                ct_data = pd.concat([ct_data, ct_rows]).reset_index(drop=True)

    # Save the master data frame
    if save:
        print(f"Saving to {output_path}")
        ct_data.to_csv(output_path / f"{output_name}.csv")
    else:
        print("Save setting set to false, not saving")

    return(ct_data)

############################################################################################################
if False:
    download_advan_data("https://app.deweydata.io/external-api/v3/products/5acc9f39-1ca6-4535-b3ff-38f6b9baf85e/files",
                        "2020-07-01",
                        footfall_output_path)

############################################################################################################
# Extract the data in footfall_output_path
# Loop through each .gz file and unzip it
def unzip_and_get_CT(footfall_output_path, processed_output_path):
    dfs = []

    # Unzip the files and get only CT rows
    for file in glob.glob(str(footfall_output_path / "*.gz")):
        print(f"Unzipping file {file}")
        # Open the .gz file)
        try:
            with gzip.open(file, 'rb') as f_in:
                # Read the contents of the .gz file
                df = pd.read_csv(f_in)

                # Get only CT rows
                ct_rows = df[df["REGION"] == "CT"]

                # Add source file
                ct_rows.loc[:,"source_file"] = file

                # Append to the list
                dfs = dfs + [ct_rows] 
        except:
            print(f"Error unzipping file {file}")

    # Concatenate the data frames
    ct_data = pd.concat(dfs)

    # Save the data
    ct_data.to_csv(processed_output_path / "footfall_CT_2020-01-07.csv")

############################################################################################################
if False:
    unzip_and_get_CT(footfall_output_path, processed_output_path)

############################################################################################################
# Load the CT data
ct_data = pd.read_csv(processed_output_path / "footfall_CT_2020-01-07.csv")

# Remove duplicates based on the PLACEKEY column, recording how many we drop
print(f"Dropping {ct_data.duplicated(subset='PLACEKEY').sum()} duplicates")
ct_data = ct_data.drop_duplicates(subset='PLACEKEY')

# Get the number of unique placekeys
print(f"Number of unique placekeys: {ct_data['PLACEKEY'].nunique()}")

# Print the top 50 SUB_CATEGORY labels by sum of RAW_VISIT_COUNTS
print(ct_data.groupby('SUB_CATEGORY')['RAW_VISIT_COUNTS'].sum().sort_values(ascending=False).head(50))

# Print all rows with "ChargePoint" in the LOCATION_NAME
# print(ct_data[ct_data['LOCATION_NAME'].str.contains("ChargePoint")])

# Create a mask to find charging stations
mask = ct_data['CATEGORY_TAGS'].astype(str).str.contains("Fuel")
mask = mask | ct_data["LOCATION_NAME"].str.lower().str.contains("charging") | ct_data["LOCATION_NAME"].str.lower().str.contains("charge")
mask = mask | (ct_data["TOP_CATEGORY"].str.contains("Gasoline Stations"))

# Get these rows and save them to the output path
charging_stations = ct_data[mask]
charging_stations.to_csv(processed_output_path / "potential_charging_stations.csv")