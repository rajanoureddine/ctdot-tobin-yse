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

####################################################################################################
# Setup paths
str_cwd = pathlib.Path().resolve().parent
str_dir = str_cwd / "Documents" / "tobin_working_data"
str_rlp_data = str_dir / "rlpolk_data"
rlp_data_file = "ct_decoded_full_attributes.csv"

# Import the final RLP data
print(f"Importing the final RLP data, located at {str_rlp_data / rlp_data_file}")
df_rlp = pd.read_csv(str_rlp_data / rlp_data_file)

# Print columns
print(df_rlp.columns)


####################################################################################################
# Clean the data
df_rlp.loc[:, "report_year"] = df_rlp.loc[:, "report_year_month"].astype(str).str[:4].astype(int)
df_rlp.loc[:, "report_month"] = df_rlp.loc[:, "report_year_month"].astype(str).str[4:].astype(int)
df_rlp = df_rlp.drop(columns=["report_year_month"])

# Add a combined vin and model year column
df_rlp["vin_my"] = df_rlp.vin_pattern + "_" + df_rlp.model_year.astype(str)

# Keep only required columns
cols = ["vin_pattern", "make", "model", "model_year", "veh_count",
        "zip_code", "county_name", "report_year", "report_month", 
        "transaction_price", "msrp", "trim",
       "fuel", "fuel1", "fuel2","drive_type", "body_type", "length", "width", "height", "wheelbase", "curb_weight",
        "range_elec", "fuel_type", "combined", "vin_my"]

df_rlp = df_rlp[cols]


####################################################################################################
# Commence checks
if True: # Check for zeroes, by extracting unique values and comparing to reality
    unique_years = df_rlp.report_year.nunique()
    unique_months = df_rlp.report_month.nunique()
    unique_mys = df_rlp.vin_my.nunique()
    unique_zips = df_rlp.zip_code.nunique()
    unique_counties = df_rlp.county_name.nunique()

    # Get get theoretical number of unique combinations
    unique_cobinations_ymz = unique_years * unique_months * unique_mys * unique_zips
    unique_combinations_ym = unique_years * unique_months * unique_mys
    unique_combinations_yz = unique_years * unique_zips * unique_mys
    unique_combinations_y = unique_years * unique_mys
    unique_combinations_z = unique_zips * unique_mys
    unique_combinations_yc = unique_years * unique_counties * unique_mys

    # Print out the theoretical number of unique combinations
    if False:
        print(f"Number of unique combinations with years, months, model years, and zip codes: {unique_cobinations_ymz}")
        print(f"Number of unique combinations with years and months: {unique_combinations_ym}")
        print(f"Number of unique combinations with years and zip codes: {unique_combinations_yz}")
        print(f"Number of unique combinations with years only: {unique_combinations_y}")
        print(f"Number of unique combinations with zip codes only: {unique_combinations_z}")

if True: # Get actual number of unique combinations
    unique_ymz = df_rlp[["report_year", "report_month", "vin_my", "zip_code"]].drop_duplicates()
    unique_ym = df_rlp[["report_year", "report_month", "vin_my"]].drop_duplicates()
    unique_yz = df_rlp[["report_year", "zip_code", "vin_my"]].drop_duplicates()
    unique_y = df_rlp[["report_year", "vin_my"]].drop_duplicates()
    unique_z = df_rlp[["zip_code", "vin_my"]].drop_duplicates()
    unique_yc = df_rlp[["report_year", "county_name", "vin_my"]].drop_duplicates()

    if False: # Print the results
        print(f"Number of combinations in the RLP data with years, months, model years, and zip codes: {len(unique_ymz)}")
        print(f"Number of combinations in the RLP data with years and months: {len(unique_ym)}")
        print(f"Number of combinations in the RLP data with years and zip codes: {len(unique_yz)}")
        print(f"Number of combinations in the RLP data with years only: {len(unique_y)}")
        print(f"Number of combinations in the RLP data with zip codes only: {len(unique_z)}")

if False: # Place this into a DataFrame, and convert to Latex for display
    zeroes_df = pd.DataFrame({"YMZ": [unique_cobinations_ymz, len(unique_ymz)],
                            "YM": [unique_combinations_ym, len(unique_ym)],
                            "YZ": [unique_combinations_yz, len(unique_yz)],
                            "Y": [unique_combinations_y, len(unique_y)],
                            "Z": [unique_combinations_z, len(unique_z)],
                            "YC": [unique_combinations_yc, len(unique_yc)]}).T
    zeroes_df.columns = ["Theoretical", "Actual"]
    
    print(zeroes_df.to_latex())

if False: # Show some of the diversity in models
    # Extract all entries with make = "Tesla" and model = "Model 3"
    tesla_model3 = df_rlp[(df_rlp.make == "Tesla") & (df_rlp.model == "Model-3")].loc[:, ["vin_pattern", "model_year", "make", "model", "range_elec"]].drop_duplicates()
    print(tesla_model3.to_latex())

if False: # Observe which models are most likely to not be present in a market
    # First create all possible combinations of vin_make_model and zip_code
    df_rlp["vin_make_model"] = df_rlp.vin_pattern + "_" + df_rlp.make + "_" + df_rlp.model
    unique_models = df_rlp["vin_make_model"].unique().tolist()
    unique_zips = df_rlp["zip_code"].unique().tolist()
    all_combinations = list(product(unique_models, unique_zips))

    # Add these to a dataframe
    all_combinations = pd.DataFrame(all_combinations, columns = ["vin_make_model", "zip_code"])

    # Get the actually observed combinations
    observed_combinations = df_rlp[["vin_make_model", "zip_code"]].drop_duplicates()
    observed_combinations["observed"] = 1

    # Combine the theoretical and observed combinations
    all_combinations = all_combinations.merge(observed_combinations, how = "outer", on = ["vin_make_model", "zip_code"])

    # Now split the vin_make_model column by the underscore
    all_combinations[["vin_pattern", "make", "model"]] = all_combinations.vin_make_model.str.split("_", expand = True)
    all_combinations = all_combinations[["vin_pattern", "make", "model", "zip_code", "observed"]]
    
    # Now get the counts of NAs by vin, make and model
    na_counts = all_combinations.loc[all_combinations["observed"].isna()]
    na_counts = na_counts.groupby(["vin_pattern", "make", "model"]).size().reset_index(name = "count").sort_values(by = "count", ascending = False)
    # print(na_counts.head(10).to_latex())

    # Now group by make and model, and average across VINs
    na_counts = na_counts.groupby(["make", "model"]).agg({"count": "mean"}).reset_index().sort_values(by = "count", ascending = False)
    print(na_counts.head(10).to_latex())

   
####################################################################################################
if False: # Determine for each VIN, the number of zips it is available for
    vins_zips = df_rlp[["vin_pattern", "zip_code"]].drop_duplicates()
    vins_zips = vins_zips.groupby("vin_pattern").size().reset_index(name = 'count').sort_values(by = "count", ascending = False)
    vins_zips["percentage"] = round(vins_zips["count"] / unique_zips, 2)

    # Check the the legnth is correct
    assert(vins_zips["count"].sum() == len(unique_z))

    # Join back in the make, model, model_year
    len_vins_zips = len(vins_zips)
    vins_zips = vins_zips.merge(df_rlp[["vin_pattern", "make", "model", "model_year", "trim", "msrp"]].drop_duplicates(), how = "left", on = "vin_pattern")
    assert(len(vins_zips) == len_vins_zips)

    # Join back in the total number of sales
    sales_per_vin = df_rlp.groupby("vin_pattern").agg({"veh_count": "sum"}).reset_index()
    vins_zips = vins_zips.merge(sales_per_vin, how = "left", on = "vin_pattern")
    assert(len(vins_zips) == len_vins_zips)
    assert(vins_zips["veh_count"].sum() == df_rlp["veh_count"].sum())

    # Print the results
    print(vins_zips.head(10))
    print(vins_zips.head(15).to_latex(float_format="%.2f"))
    print(vins_zips.tail(15).to_latex(float_format="%.2f"))

if True: # Determine for each VIN, the number of counties it is available for
    vins_counties = df_rlp[["vin_pattern", "county_name"]].drop_duplicates()
    vins_counties = vins_counties.groupby("vin_pattern").size().reset_index(name = 'count').sort_values(by = "count", ascending = False)
    vins_counties["percentage"] = round(vins_counties["count"] / unique_counties, 2)

    # Check the the legnth is correct
    # assert(vins_counties["count"].sum() == len(unique_z))

    # Join back in the make, model, model_year
    len_vins_counties = len(vins_counties)
    vins_counties = vins_counties.merge(df_rlp[["vin_pattern", "make", "model", "model_year", "trim", "msrp"]].drop_duplicates(), how = "left", on = "vin_pattern")
    assert(len(vins_counties) == len_vins_counties)

    # Join back in the total number of sales
    sales_per_vin = df_rlp.groupby("vin_pattern").agg({"veh_count": "sum"}).reset_index()
    vins_counties = vins_counties.merge(sales_per_vin, how = "left", on = "vin_pattern")
    assert(len(vins_counties) == len_vins_counties)
    assert(vins_counties["veh_count"].sum() == df_rlp["veh_count"].sum())

    # Print the results
    print(vins_counties.head(10).to_latex(float_format="%.2f"))
    #print(vins_counties.tail(15))

    # Now aggregate by count, and tell me the percentage of the total veh_count in each group
    vins_counties_summary = vins_counties.groupby("count").agg({"veh_count": "sum"}).reset_index()
    vins_counties_summary["percentage"] = round(vins_counties_summary["veh_count"] / df_rlp["veh_count"].sum(), 2)
    assert(vins_counties_summary["veh_count"].sum() == df_rlp["veh_count"].sum())

    print(vins_counties_summary.head(10).sort_values("veh_count", ascending=False).to_latex(float_format="%.2f"))
    #print(vins_counties.head(15).to_latex(float_format="%.2f"))
    #print(vins_counties.tail(15).to_latex(float_format="%.2f"))






####################################################################################################
# Group and analyse zeroes
# Group by vin_pattern, make, model, model_year, zip_code, report_year, report_month, transaction_price
# group_by = ["vin_pattern", "make", "model", "model_year", "zip_code", "report_year", "report_month"]
# agg_methods = {"veh_count": "sum", 
#                 "transaction_price": "mean",
#                 "msrp": "mean",
#                 "drive_type": "first",
#                 "body_type": "first",
#                 "length": "mean",
#                 "width": "mean",
#                 "height": "mean",
#                 "wheelbase": "mean",
#                 "curb_weight": "mean",
#                 "range_elec": "mean",
#                 "fuel_type": "first",
#                 "combined": "mean"}
# 
# # Now group by the above, sum the veh_count, and average the rest
# df_rlp_grouped = df_rlp.groupby(group_by).agg(agg_methods).reset_index()
# 
# # Now check for zeroes, first by extracting unique values
# unique_years = df_rlp_grouped.report_year.unique()
# unique_months = df_rlp_grouped.report_month.unique()
# vins_mys = df_rlp_grouped[["vin_pattern", "model_year"]].drop_duplicates()
# vins_mys["joined"] = vins_mys.vin_pattern + "_"+ vins_mys.model_year.astype(str)
# unique_mys = vins_mys.joined.unique()
# 
# # Get all combinations of the above
# unique_combinations = list(product(unique_years, unique_months, unique_mys))
# print(f"Number of unique years: {len(unique_years)}")
# print(f"Number of unique months: {len(unique_months)}")
# print(f"Number of unique model years: {len(unique_mys)}")
# print(f"Number of unique combinations: {len(unique_combinations)}")
# 
# # Create a column in df_rlp_grouped that is the vin_pattern + "_" + model_year
# unique = df_rlp[["report_year", "report_month", "vin_my"]].drop_duplicates()
# 
# # Now count in the grouped data how many of the unique combinations exist
# print(f"Number of combinations in the RLP data: {len(unique)}")


