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
lease = "no_lease"
output_file = str_rlp / f"rlp_prepared_{date_time}_{lease}.csv"

# Read in the RLP data to be processed
df_rlp = pd.read_csv(str_sales_vin_characteristic)
if lease == "no_lease":
    num_leases = df_rlp["transaction_price"].isna().sum()
    print(f"Number of leases to be dropped: {num_leases}")
    df_rlp = df_rlp.loc[df_rlp["transaction_price"].notna()]

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
# Step 0: Decide what we will run here, and what we will read in directly.
# 0a: Most common trims
most_common_trim_features = "read_in"
most_common_trims_destination = output_folder / f"most_common_trim_features_{date_time}_{lease}.csv"
most_common_trims_source = output_folder / "most_common_trim_features_20240411_123942.csv"

# 0b: Replace with most common trim
do_replace_with_trim = "read_in"
replaced_destination = output_folder / f"rlp_with_dollar_per_mile_replaced_{date_time}_{lease}.csv"
replaced_source  = output_folder / "rlp_with_dollar_per_mile_replaced_20240422_123713_no_lease.csv"

# 0d: Aggregate to market
do_aggregate_to_market = "read_in"

aggregated_zips_destination = output_folder / f"rlp_with_dollar_per_mile_replaced_myear_zip_{date_time}_{lease}.csv"
aggregated_counties_destination = output_folder / f"rlp_with_dollar_per_mile_replaced_myear_county_{date_time}_{lease}.csv"
aggregated_myear_destination = output_folder / f"rlp_with_dollar_per_mile_replaced_myear_{date_time}_{lease}.csv"

aggregated_zips_source = output_folder / "rlp_with_dollar_per_mile_replaced_myear_zip_20240422_125027_no_lease.csv"
aggregated_counties_source = output_folder / "rlp_with_dollar_per_mile_replaced_myear_county_20240422_125027_no_lease.csv"
aggregated_myear_source = output_folder / "rlp_with_dollar_per_mile_replaced_myear_20240422_125027_no_lease.csv"

# 0e Rationalize market
do_rationalize_market = "do"
threshold = 50
rationalized_my_dest = output_folder / f"rlp_with_dollar_per_mile_replaced_myear_{date_time}_{lease}_zms.csv"
rationalized_my_ct_dest = output_folder / f"rlp_with_dollar_per_mile_replaced_myear_county_{date_time}_{lease}_zms.csv"
rationalized_my_zip_dest = output_folder / f"rlp_with_dollar_per_mile_replaced_myear_zip_{date_time}_{lease}_zms.csv"

####################################################################################################
# Step 1: Identify the most common trim per make and model
def get_most_common_trim(df, sales_col, separate_electric = True):
    """Get the most common trim for each (make, model, fuel, range_elec).
    Note: We get the most common trim across all the model years included. For example, for the 
    Ford F-150, suppose Trim A was most popular in 2018, but Trim B was most popular in all the years
    together. In that case, we assign all F-150s to Trim B; even those in 2018.

    Note: This also means that even if Trim B was not available in 2018, we still assign all the 2018
    observations to Trim B.
    
    Note: For electric vehicles, we do not choose a most popular trim - we keep all the trims.
    """


    unique_product_ids = ["make", "model", "trim", "fuel", "range_elec"]

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

# Step 2: Calculate the features for each of these products.
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

# Step 3: In the original data frame, go through each make, model, fuel, and range_elec and replace with the most common trim
def replace_with_most_common_trim(df, most_common_trim_features, electric = False, use_zips = True):
    """Replace the make, model, trim, fuel, and range_elec with the most common trim for that model year.
    NOTE: We expect that in most_common_trim_features, electric vehicles are treated differently. For electric vehicles,
    EVERY trim is kept - not just the most popular. Consequently, for electric vehicles, we merge differently (we need to)
    also include the "trim" column in what to merge on.
    """

    # Prepare a column to record successful merges
    most_common_trim_features["merge_success"] = 1

    # Fix range_elec for later - this appears to be causing an issue with Tesla Model X
    if electric:
        most_common_trim_features["range_elec"] = round(most_common_trim_features["range_elec"], 2)
        df["range_elec"] = round(df["range_elec"], 2)

    # Get the columns to merge on and to keep
    cols_merge_on = ["make", "model", "model_year", "fuel", "range_elec"]
    cols_merge_on_noyear = ["make", "model", "fuel", "range_elec"]
    if use_zips:
        cols_to_keep = ["veh_count", "zip_code"] # Workaround
    else:
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
    
    output_not_merged["range_elec"] = round(output_not_merged["range_elec"], 2)

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

# Step 4: Aggregate to the market level
def aggregate_to_market(df, most_common_trim_features):
        """Aggregates the data to the market level.
        We assume that with in each make, model, model_year, trim, range_elec, and fuel - the other features are all the same."""

        df["range_elec"] = round(df["range_elec"], 2)
        most_common_trim_features["range_elec"] = round(most_common_trim_features["range_elec"], 2)

        most_common_trim_features_orig = most_common_trim_features.copy()
        most_common_trim_features = most_common_trim_features.drop(columns = ["veh_count"])
        
        # Aggregate and check
        output_counties = df[["make", "model", "model_year", "trim", "fuel", "range_elec", "county_name", "veh_count"]].groupby(["make", "model", "model_year", "trim", "fuel", "range_elec", "county_name"]).sum().reset_index()
        output_zips = df[["make", "model", "model_year", "trim", "fuel", "range_elec", "zip_code", "veh_count"]].groupby(["make", "model", "model_year", "trim", "fuel", "range_elec", "zip_code"]).sum().reset_index()
        output_myear = df[["make", "model", "model_year", "trim", "fuel", "range_elec", "veh_count"]].groupby(["make", "model", "model_year", "trim", "fuel", "range_elec"]).sum()
        assert(output_counties["veh_count"].sum() == df["veh_count"].sum())
        assert(output_myear["veh_count"].sum() == df["veh_count"].sum())
        assert(output_zips["veh_count"].sum() == df["veh_count"].sum())

        # WE GET NAs here - where the most popular trim is not available in a given model_year
        output_counties = output_counties.merge(most_common_trim_features, on = ["make", "model", "model_year", "trim", "fuel", "range_elec"], how = 'left')
        output_myear = output_myear.merge(most_common_trim_features, on = ["make", "model", "model_year", "trim", "fuel", "range_elec"], how = 'left')
        output_zips = output_zips.merge(most_common_trim_features, on = ["make", "model", "model_year", "trim", "fuel", "range_elec"], how = 'left')
        assert(output_counties["veh_count"].sum() == df["veh_count"].sum())
        assert(output_myear["veh_count"].sum() == df["veh_count"].sum())
        assert(output_zips["veh_count"].sum() == df["veh_count"].sum())

        # Split into matched and unmatched
        output_counties_matched = output_counties[output_counties["msrp"].notna()]
        output_counties_unmatched = output_counties.loc[output_counties["msrp"].isna(), ["make", "model", "model_year", "trim", "fuel", "range_elec", "county_name", "veh_count"]]
        output_myear_matched = output_myear[output_myear["msrp"].notna()]
        output_myear_unmatched = output_myear.loc[output_myear["msrp"].isna(), ["make", "model", "model_year", "trim", "fuel", "range_elec", "veh_count"]]
        output_zips_matched = output_zips[output_zips["msrp"].notna()]
        output_zips_unmatched = output_zips.loc[output_zips["msrp"].isna(), ["make", "model", "model_year", "trim", "fuel", "range_elec", "zip_code", "veh_count"]]

        # Get the top year
        most_common_trim_features_top = most_common_trim_features_orig.sort_values("veh_count", ascending=False).drop_duplicates(subset = ["make", "model", "trim", "fuel", "range_elec"])
        most_common_trim_features_top = most_common_trim_features_top.drop("veh_count", axis = 1)
        
        # Match in
        cols = {"model_year_x": "model_year", "model_year_y": "model_year_matched"}
        output_counties_unmatched_fixed = output_counties_unmatched.merge(most_common_trim_features_top, on = ["make", "model", "trim", "fuel", "range_elec"], how = 'left').rename(columns =cols)
        output_myear_unmatched_fixed = output_myear_unmatched.merge(most_common_trim_features_top, on = ["make", "model", "trim", "fuel", "range_elec"], how = 'left').rename(columns =cols)
        output_zips_unmatched_fixed = output_zips_unmatched.merge(most_common_trim_features_top, on = ["make", "model", "trim", "fuel", "range_elec"], how = 'left').rename(columns =cols)

        assert(len(output_counties_unmatched_fixed)== len(output_counties_unmatched))
        assert(len(output_myear_unmatched_fixed) == len(output_myear_unmatched))
        assert(len(output_zips_unmatched_fixed) == len(output_zips_unmatched))

        output_counties = pd.concat([output_counties_matched, output_counties_unmatched_fixed], ignore_index=True)
        output_myear = pd.concat([output_myear_matched, output_myear_unmatched_fixed], ignore_index=True)
        output_zips = pd.concat([output_zips_matched, output_zips_unmatched_fixed], ignore_index=True)

        return output_zips, output_counties, output_myear

# Step 5: Rationalize and fix ZMS
def rationalize_markets(df_my, mkts_to_rationalize, geographies, threshold, most_common_trim_features):
    """
    Rationalize markets by dropping uncommon products, and adding zero market shares for those that are common but not present.

    Input: 
    mkts_to_rationalize: A data frame grouped by model year, and one grouped by model year and ct.
    geographies: the geography column, a string

    """

    # Get geographies
    geogs = pd.Series(mkts_to_rationalize[geographies].drop_duplicates().tolist(), name = geographies)

    # Fix the electric range
    most_common_trim_features["range_elec"] = round(most_common_trim_features["range_elec"], 2)
    most_common_trim_features_copy = most_common_trim_features.copy()
    most_common_trim_features = most_common_trim_features.drop(columns = ["veh_count"])

    # Get the frequency of products per market year, and remove those below the threshold 
    vars_group_on = ["make", "model", "model_year", "trim", "fuel", "range_elec"]
    vehs_to_keep = df_my[vars_group_on+["veh_count"]].groupby(vars_group_on).sum().reset_index()
    assert(vehs_to_keep["veh_count"].sum() == df_my["veh_count"].sum())
    vehs_to_keep = vehs_to_keep.loc[vehs_to_keep["veh_count"] > threshold, vars_group_on] # Drop veh_count

    # For the model year dataset, we keep only the products with sales above the threshold for that model year
    df_my_out = df_my.merge(vehs_to_keep, on=["make", "model", "model_year", "trim", "fuel", "range_elec"], how="inner")

    # For the model year and county dataset, we:
    # 1) Keep only the products with sales above the threshold for that model year - keeping the features columns
    output = mkts_to_rationalize.merge(vehs_to_keep, on=vars_group_on, how="inner")

    # 2) Add zero market shares for products available in one county in a year but not in another
    # First, get the unique geographies, and then for every model year, get every county it should be in
    all_prods_mkts = vehs_to_keep.merge(geogs, how="cross")
    
    # Now merge with the original df - unmatched are the zero market shares
    output = all_prods_mkts.merge(output, on=["make", "model", "model_year", "trim", "fuel", "range_elec"]+[geographies], how="left")

    # Put aside matched
    output_matched = output.loc[output["veh_count"].notna()].drop("Unnamed: 0", axis = 1)# Put this aside

    # For those with zero market shares, replace with 0
    output_unmatched = output.loc[output["veh_count"].isna(), ["make", "model", "model_year", "trim", "fuel", "range_elec"]+[geographies]]
    output_unmatched["veh_count"] = 0
    use_zips = geographies == "zip_code"
    output_unmatched_elec = replace_with_most_common_trim(output_unmatched.loc[output_unmatched["fuel"]=="electric"], most_common_trim_features_copy, electric = True, use_zips =use_zips)
    output_unmatched_nelec = replace_with_most_common_trim(output_unmatched.loc[output_unmatched["fuel"]!="electric"], most_common_trim_features_copy, use_zips = use_zips)
    output_unmatched = pd.concat([output_unmatched_elec, output_unmatched_nelec], axis=0, ignore_index = True)
    
    output_out = pd.concat([output_matched, output_unmatched], axis=0, ignore_index=True)
    assert(len(output_out) == len(output))
    assert(output_out["veh_count"].sum() == output["veh_count"].sum())

    return df_my_out, output_out



###########################################################################################
# Run
most_common_trims = get_most_common_trim(df_rlp, sales_col)

###### Step 2 Create & Save
if most_common_trim_features != "read_in":
    most_common_trim_features = get_most_common_trim_features(df_rlp, most_common_trims)
    most_common_trim_features.to_csv(most_common_trims_destination, index=False)
elif most_common_trim_features == "read_in":
    most_common_trim_features = pd.read_csv(most_common_trims_source)


####### Create or read
if do_replace_with_trim != "read_in":
    df_replaced_nelec = replace_with_most_common_trim(df_rlp.loc[df_rlp["fuel"]!="electric"], most_common_trim_features)
    df_replaced_elec = replace_with_most_common_trim(df_rlp.loc[df_rlp["fuel"]=="electric"], most_common_trim_features, electric = True)
    df_replaced = pd.concat([df_replaced_nelec, df_replaced_elec])
    df_replaced.to_csv(replaced_destination, index = False)
elif do_replace_with_trim == "read_in":
    df_replaced = pd.read_csv(replaced_source) # with leases

####### Create or read
if do_aggregate_to_market != "read_in":
    aggregated_zips, aggregated_counties, aggregated_myear = aggregate_to_market(df_replaced, most_common_trim_features)
    aggregated_zips.to_csv(aggregated_zips_destination)
    aggregated_counties.to_csv(aggregated_counties_destination)
    aggregated_myear.to_csv(aggregated_myear_destination)
elif do_aggregate_to_market == "read_in":
    aggregated_zips = pd.read_csv(aggregated_zips_source)
    aggregated_counties = pd.read_csv(aggregated_counties_source)
    aggregated_myear = pd.read_csv(aggregated_myear_source)


# Rationalize the markets
if do_rationalize_market == "do":
    aggregated_myear_zms, aggregated_counties_zms = rationalize_markets(aggregated_myear, aggregated_counties, "county_name", threshold, most_common_trim_features)
    _ , aggregated_zips_zms = rationalize_markets(aggregated_myear, aggregated_zips, "zip_code", threshold, most_common_trim_features)

    aggregated_counties_zms.to_csv(rationalized_my_ct_dest)
    aggregated_myear_zms.to_csv(rationalized_my_dest)
    aggregated_zips_zms.to_csv(rationalized_my_zip_dest)





if False:
    def rationalize_markets_old(df_my, df_my_ct, threshold, most_common_trim_features):
        """
        Rationalize markets by dropping uncommon products, and adding zero market shares for those that are common but not present.

        Input: A data frame grouped by model year, and one grouped by model year and ct.
        """

        # Fix the electric range
        most_common_trim_features["range_elec"] = round(most_common_trim_features["range_elec"], 2)
        most_common_trim_features_copy = most_common_trim_features.copy()
        most_common_trim_features = most_common_trim_features.drop(columns = ["veh_count"])

        # Get the least common products per market year
        vehs_to_keep = df_my[["make", "model", "model_year", "trim", "fuel", "range_elec", "veh_count"]]
        vehs_to_keep = vehs_to_keep.groupby(["make", "model", "model_year", "trim", "fuel", "range_elec"]).sum().reset_index()
        # Quick check that the veh count has not changed
        assert(vehs_to_keep["veh_count"].sum() == df_my["veh_count"].sum())
        # Remove those below threshold and drop the veh_count column
        vehs_to_keep = vehs_to_keep.loc[vehs_to_keep["veh_count"] > threshold]
        vehs_to_keep = vehs_to_keep.drop(columns = ["veh_count"])

        # For the model year dataset, we keep only the products with sales above the threshold for that model year
        df_my_out = df_my.merge(vehs_to_keep, on=["make", "model", "model_year", "trim", "fuel", "range_elec"], how="inner")

        # For the model year and county dataset, we:
        # 1) Keep only the products with sales above the threshold for that model year - keeping the features columns
        df_my_ct = df_my_ct.merge(vehs_to_keep, on=["make", "model", "model_year", "trim", "fuel", "range_elec"], how="inner")

        # 2) Add zero market shares for products available in one county in a year but not in another
        # First, get the unique counties, and then for every model year, get every county it should be in
        counties = pd.Series(df_my_ct["county_name"].drop_duplicates().tolist(), name = "county_name")
        all_prods_mkts = vehs_to_keep.merge(counties, how="cross")
        # Now merge with the original df - unmatched are the zero market shares
        df_my_ct = all_prods_mkts.merge(df_my_ct, on=["make", "model", "model_year", "trim", "fuel", "range_elec", "county_name"], how="left")
        # For those with zero market shares, fill in the missing features
        df_my_ct_unmatched = df_my_ct.loc[df_my_ct["veh_count"].isna(), ["make", "model", "model_year", "trim", "fuel", "range_elec", "county_name"]]
        df_my_ct_matched = df_my_ct.loc[df_my_ct["veh_count"].notna()]
        df_my_ct_unmatched["veh_count"] = 0
        df_my_ct_unmatched_elec = replace_with_most_common_trim(df_my_ct_unmatched.loc[df_my_ct_unmatched["fuel"]=="electric"], most_common_trim_features_copy, electric = True)
        df_my_ct_unmatched_nelec = replace_with_most_common_trim(df_my_ct_unmatched.loc[df_my_ct_unmatched["fuel"]!="electric"], most_common_trim_features_copy)
        df_my_ct_unmatched = pd.concat([df_my_ct_unmatched_elec, df_my_ct_unmatched_nelec], axis=0)
        # df_my_ct_unmatched = df_my_ct_unmatched.merge(most_common_trim_features, on=["make", "model", "model_year", "trim", "fuel", "range_elec"], how="left")
        df_my_ct_out = pd.concat([df_my_ct_matched, df_my_ct_unmatched], axis=0)
        assert(len(df_my_ct_out) == len(df_my_ct))
        assert(df_my_ct_out["veh_count"].sum() == df_my_ct["veh_count"].sum())

        # df_my_ct_out = df_my_ct_out.drop(columns = ["merge_success"])
        # df_my_out = df_my_out.drop(columns = ["merge_success"])

        return df_my_out, df_my_ct_out




