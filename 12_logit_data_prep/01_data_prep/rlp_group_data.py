# Note: This file should be run after rlp_finalize_data.py
# The goal here is to roll up the RLP data to the product and market level

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
import functions_rlp as rlp_functions 

# Warnings and display
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)

####################################################################################################
# Setup paths
str_cwd = pathlib.Path().resolve().parent
str_dir = str_cwd / "Documents" / "tobin_working_data"
str_rlp = str_dir / "rlpolk_data"
rlp_data_file = "ct_decoded_full_attributes.csv"
str_sales_vin_characteristic = str_rlp / "rlp_with_dollar_per_mile.csv"
output_folder = str_rlp

# Set the date and time and output filename
date_time = datetime.now().strftime("%Y%m%d_%H%M%S")
output_file = str_rlp / f"rlp_prepared_{date_time}.csv"

# Read in the RLP data to be processed
df_rlp = pd.read_csv(str_sales_vin_characteristic)

####################################################################################################
unique_product_ids = ["make", "model", "trim", "fuel", "range_elec"]
unique_markets = "model_year"
sales_col = "veh_count"

# Define a weighted average function
wm = x = lambda x: np.average(x, weights=df_rlp.loc[x.index, sales_col])

agg_funs = {'model_year':'first',
            'make': 'first',
            'model': 'first',
            'trim': 'first',
            sales_col: "sum",
            "msrp": wm,
            "dollar_per_mile": wm, 
            "log_hp_wt": wm, 
            "wheelbase": wm, 
            "curb_weight": wm, 
            "doors": wm, 
            "drive_type": 'first', 
            "body_type": 'first', 
            "fuel": 'first',
            'range_elec' : wm,
            'fed_credit':'first'}
    
vars = list(agg_funs.keys())



####################################################################################################

# Step 1: Identify the most common trim per make and model
def get_most_common_trim(df, sales_col, separate_electric = True):
    """Get the most common trim for each make and model."""

    # Extract sales for each make, model, and trim
    trim_sales = df[unique_product_ids + [sales_col]].groupby(unique_product_ids).sum().reset_index()
    
    # Separate out the electric vehicles - we keep all trims for these
    if separate_electric:
        trim_sales_elec = trim_sales.loc[trim_sales["fuel"] == "electric"]
        trim_sales_elec = trim_sales_elec.drop(columns=[sales_col])
        trim_sales = trim_sales.loc[trim_sales["fuel"] != "electric"]

    # Get the most common trim for each make and model (for electric only)
    max_within = lambda x: x.loc[x[sales_col].idxmax()]
    most_common_trim = trim_sales.groupby(["make", "model", "fuel"]).apply(max_within).reset_index(drop=True)
    most_common_trim = most_common_trim.drop(columns=[sales_col])

    # Add back in the electric vehicles
    if separate_electric:
        most_common_trim = pd.concat([most_common_trim, trim_sales_elec], axis=0)

    # Identify which model years that trim is available in
    products_markets = df[unique_product_ids + [unique_markets]].drop_duplicates()
    most_common_trim = most_common_trim.merge(products_markets, on=unique_product_ids, how="left")

    return most_common_trim

most_common_trims = get_most_common_trim(df_rlp, sales_col)

# Step 2: Calculate the features for each of these products.
# To do this, we take all instances of that product (for a given model year) and average the features.
# If that trim is not available for a specific model year, we use another model year
def get_most_common_trim_features(df, most_common_trims):
    """Get the features for the most common trim for each make and model."""

    # Extract the variables we want to aggregate
    df_details = df[vars]

    output = pd.DataFrame([])

    for _, row in tqdm(most_common_trims.iterrows()):

        # Extract the rows in the original dataframe that match this make, model, trim, fuel, and range_elec
        # Note that the features will not all be the same, since there may be different styles for the same model
        make, model, trim, fuel, range_elec, model_year = row
        mask = (df_details["make"] == make) & (df_details["model"] == model) & (df_details["trim"] == trim)
        mask = mask & (df_details["fuel"] == fuel) & (df_details["range_elec"] == range_elec)
        mask = mask & (df_details[unique_markets] == model_year)
        vehicle_rows = df_details.loc[mask].reset_index()

        # We now aggregate the features for this make, model, trim, fuel, and range_elec for the model_year
        wm = lambda x: np.average(x, weights=vehicle_rows.loc[x.index, sales_col])
        agg_funs = {'make': 'first', 'model': 'first', 'trim': 'first', sales_col: "sum", "msrp": wm,
                    "dollar_per_mile": wm, "log_hp_wt": wm, "wheelbase": wm, "curb_weight": wm, "doors": wm, 
                    "drive_type": 'first', "body_type": 'first', "fuel": 'first','range_elec' : wm,'fed_credit':'first'}
        vehicle_features = vehicle_rows.groupby(unique_markets).agg(agg_funs).reset_index()

        # Add to the output
        output = pd.concat([output, vehicle_features], axis=0)
    
    return output

most_common_trim_features = get_most_common_trim_features(df_rlp, most_common_trims)
most_common_trim_features.to_csv(output_folder / f"most_common_trim_features_{date_time}.csv", index=False)



