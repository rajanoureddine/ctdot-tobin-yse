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

    # Extract sales for each make, model, and trim. We group the unique products by the number of sales over
    # the whole period. (unique product = make, model, trim, fuel, range_elec)
    trim_sales = df[unique_product_ids + [sales_col]].groupby(unique_product_ids).sum().reset_index()
    
    # Separate out the electric vehicles - we keep all trims for these. That is, we do not get the most 
    # popular trim but instead keep all trims
    if separate_electric:
        trim_sales_elec = trim_sales.loc[trim_sales["fuel"] == "electric"]
        trim_sales_elec = trim_sales_elec.drop(columns=[sales_col])
        trim_sales = trim_sales.loc[trim_sales["fuel"] != "electric"]

    # Get the most common trim for each make, model, and range_elec (for electric only)
    # Note - we get the most common trim across all possible years 
    max_within = lambda x: x.loc[x[sales_col].idxmax()]
    most_common_trim = trim_sales.groupby(["make", "model", "fuel", "range_elec"]).apply(max_within).reset_index(drop=True)
    most_common_trim = most_common_trim.drop(columns=[sales_col])

    # Add back in the electric vehicles
    if separate_electric:
        most_common_trim = pd.concat([most_common_trim, trim_sales_elec], axis=0)

    # Identify which model years that trim is available in
    products_markets = df[unique_product_ids + [unique_markets]].drop_duplicates()
    most_common_trim = most_common_trim.merge(products_markets, on=unique_product_ids, how="left")

    return most_common_trim

# most_common_trims = get_most_common_trim(df_rlp, sales_col)

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

# most_common_trim_features = get_most_common_trim_features(df_rlp, most_common_trims)
# most_common_trim_features.to_csv(output_folder / f"most_common_trim_features_{date_time}.csv", index=False)

most_common_trim_features = pd.read_csv(output_folder / "most_common_trim_features_20240411_123942.csv")

# Quick test that within each make, model, model_year, trim, fuel, and range_elec the features are all the same. 
test = most_common_trim_features.drop_duplicates(subset=["make", "model", "model_year", "trim", "fuel", "range_elec"])
assert(len(most_common_trim_features)==len(test))

# Step 3: In the original data frame, go through each make, model, fuel, and range_elec and replace with the most common trim
# and the most common trim features for that model year
def replace_with_most_common_trim(df, most_common_trim_features, electric = False):
    """Replace the make, model, trim, fuel, and range_elec with the most common trim for that model year.
    NOTE: We expect that in most_common_trim_features, electric vehicles are treated differently. For electric vehicles,
    EVERY trim is kept - not just the most popular. Consequently, for electric vehicles, we merge differently (we need to)
    also include the "trim" column in what to merge on.
    """

    # Prepare a column to record successful merges
    most_common_trim_features["merge_success"] = 1

    # Get the columns to merge on and to keep
    cols_merge_on = ["make", "model", "model_year", "fuel", "range_elec"]
    cols_merge_on_noyear = ["make", "model", "fuel", "range_elec"]
    cols_to_keep = ["veh_count", "county_name"]

    # Alter this if electric
    if electric:
        cols_merge_on = cols_merge_on + ["trim"]
        cols_merge_on_noyear = cols_merge_on_noyear + ["trim"]

    # Merge the most common trim features with the original data frame
    df_to_keep = df[cols_merge_on + cols_to_keep]
    output = df_to_keep.merge(most_common_trim_features.iloc[:, ~most_common_trim_features.columns.isin(["veh_count"])], on=cols_merge_on, how="left")

    # Confirm this has worked correctly
    assert(len(output) == len(df))
    assert(output["veh_count"].sum() == df["veh_count"].sum())

    # Note that in some cases, the most common trim is not available for that model year
    # In these cases, we will have NaNs in the output. We get closest available model year
    output_merged = output.loc[output["merge_success"] == 1]
    output_not_merged = output.loc[output["merge_success"] != 1, cols_merge_on + cols_to_keep]

    # Get the model year with the highest sales for each make, model, trim, fuel, and range_elec
    most_common_trim_features_top = most_common_trim_features.sort_values("veh_count", ascending=False).drop_duplicates(cols_merge_on_noyear)

    # Merge back in for the not merged rows
    if not electric:
        output_not_merged = output_not_merged.merge(most_common_trim_features_top.iloc[:, ~most_common_trim_features_top.columns.isin(["model_year", "veh_count"])], on=cols_merge_on_noyear, how="left")
    else:
    # Currently not working for Tesla Model-X Trim: 75D, 2019. Fix this
        output_not_merged = output_not_merged.merge(most_common_trim_features_top.iloc[:, ~most_common_trim_features_top.columns.isin(["veh_count"])], on=cols_merge_on_noyear, how="left")
        output_not_merged["model_year"] = output_not_merged["model_year_x"]
        output_not_merged = output_not_merged.drop(columns = ["model_year_x", "model_year_y"])

    # Combine the two data frames
    output = pd.concat([output_merged, output_not_merged], axis=0)

    # Confirm this has worked correctly
    assert(len(output) == len(df))
    assert(output["veh_count"].sum() == df["veh_count"].sum())

    return output

# Replace details for electric and non-electric separately
# df_replaced_nelec = replace_with_most_common_trim(df_rlp.loc[df_rlp["fuel"]!="electric"], most_common_trim_features)
# df_replaced_elec = replace_with_most_common_trim(df_rlp.loc[df_rlp["fuel"]=="electric"], most_common_trim_features, electric = True)
# df_replaced = pd.concat([df_replaced_nelec, df_replaced_elec])

# Save
# df_replaced.to_csv(output_folder / f"rlp_with_dollar_per_mile_replaced_{date_time}.csv", index = False)

df_replaced = pd.read_csv(output_folder / "rlp_with_dollar_per_mile_replaced_20240411_133452.csv")


def aggregate_to_market(df, most_common_trim_features):
    """Aggregates the data to the market level.
    We assume that with in each make, model, model_year, trim, range_elec, and fuel - the other features are all the same.
    Consequently we can use the "agg:first" function"""

    most_common_trim_features = most_common_trim_features.drop(columns = ["veh_count"])
    
    # Aggregate and check
    output_counties = df[["make", "model", "model_year", "trim", "fuel", "range_elec", "county_name", "veh_count"]].groupby(["make", "model", "model_year", "trim", "fuel", "range_elec", "county_name"]).sum()
    output_myear = df[["make", "model", "model_year", "trim", "fuel", "range_elec", "veh_count"]].groupby(["make", "model", "model_year", "trim", "fuel", "range_elec"]).sum()
    assert(output_counties["veh_count"].sum() == df["veh_count"].sum())
    assert(output_myear["veh_count"].sum() == df["veh_count"].sum())

    # Merge back in the details
    output_counties = output_counties.merge(most_common_trim_features, on = ["make", "model", "model_year", "trim", "fuel", "range_elec"], how = 'left')
    output_myear = output_myear.merge(most_common_trim_features, on = ["make", "model", "model_year", "trim", "fuel", "range_elec"], how = 'left')
    assert(output_counties["veh_count"].sum() == df["veh_count"].sum())
    assert(output_myear["veh_count"].sum() == df["veh_count"].sum())

    return output_counties, output_myear


aggregated_counties, aggregated_myear = aggregate_to_market(df_replaced, most_common_trim_features)


