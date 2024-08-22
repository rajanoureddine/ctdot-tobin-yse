"""
Note: This file should be run after rlp_finalize_data.py
This file is used to roll up the RLP data to the product and market level.
This is performed in the following steps:
1. Identify the most common trim for each make and model
2. Identify the features for each of these products.
3. In the original data frame, go through each make, model, fuel, and range_elec and replace with the most common trim
4. Aggregate to the market level, where a market is a county x Model year
5. Fix zero market shares.
"""

####################################################################################################
# Import libraries
import pathlib
import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import datetime
import warnings

# Warnings and display
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)

# Silence warnings
warnings.filterwarnings("ignore")

####################################################################################################
# Setup paths
str_cwd = pathlib.Path().resolve().parent
str_rlp = str_cwd / "Documents" / "tobin_working_data" / "rlpolk_data"
str_sales_vin_characteristic = str_rlp / "rlp_with_dollar_per_mile.csv"

# Set the date and time and output filename
date_time = datetime.now().strftime("%Y%m%d_%H%M%S")

# Create a folder for outputs
output_folder = str_rlp / f"rlp_data_processed_{date_time}"
output_folder.mkdir(parents=True, exist_ok=True)

# Set up parameters
unique_markets = "model_year"
sales_col = "veh_count"
lease = "no_lease"
threshold = 20

####################################################################################################
# Step 1: Identify the most common trim per make and model
def get_most_common_trim(df, sales_col, separate_electric = True):
    """
    Get the most common trim for each (make, model, fuel, range_elec). Note - this most common trim
    may not be available in all model years.
    
    Note: For electric vehicles, we do not choose a most popular trim - we keep all the trims.
    """

    # Variables used form unique products
    unique_product_ids = ["make", "model", "trim", "fuel", "range_elec"]

    # Extract sales for each unique product. We group the unique products by the number of sales over
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

    # Add back in the electric vehicles for which we keep all trims
    if separate_electric:
        most_common_trim = pd.concat([most_common_trim, trim_sales_elec], axis=0)

    # Identify which model years each of the most common trims is available in
    products_markets = df[unique_product_ids + [unique_markets]].drop_duplicates()
    most_common_trim = most_common_trim.merge(products_markets, on=unique_product_ids, how="left")

    return most_common_trim

def get_most_common_trim_features(df, most_common_trims):
    """
    Get the features for the most common trim for each make and model.
    
    Note: We get the most common trim across all the model years included. For example, for the 
    Ford F-150, suppose Trim A was most popular in 2018, but Trim B was most popular in all the years
    together. In that case, we assign all F-150s to Trim B; even those in 2018.

    Note: This also means that even if Trim B was not available in 2018, we still assign all the 2018
    observations to Trim B. In that case, the features for (Trim B x 2018) should be missing. Thus
    we use (Trim B x Most common year for Trim B).

    A dictionary of aggregation functions is used to aggregate features.
    """

    # Define aggregation functions used to extract trim features
    wm = lambda x: np.average(x, weights=vehicle_rows.loc[x.index, sales_col])
    agg_funs = {'make': 'first', 'model': 'first', 'trim': 'first',
                sales_col: "sum", "msrp": wm,
                    "fed_credit" : wm, "state_incentive": wm, "dealer_incentive":wm, 
                    "dollar_per_mile": wm, "log_hp_wt": wm, "wheelbase": wm, "curb_weight": wm, "doors": wm, 
                    "drive_type": 'first', "body_type": 'first', "fuel": 'first','range_elec' : wm}
    vars = list(agg_funs.keys())

    # Create a data frame containing unique product IDs, and features for each
    df_details = df[vars + [unique_markets]]


    # Prepare the output
    output = pd.DataFrame([])

    # Iterate through each of the most common trims (approx 2000) and extract the features
    for _, row in tqdm(most_common_trims.iterrows()):
        # Extract the rows in the original dataframe that match this make, model, trim, fuel, and range_elec
        # These rows will comprise observations for different years, months, zip codes, as well as different styles
        # Note that the features will not all be the same, since there may be different styles for the same model
        make, model, trim, fuel, range_elec, model_year = row
        mask = (df_details["make"] == make) & (df_details["model"] == model) & (df_details["trim"] == trim)
        mask = mask & (df_details["fuel"] == fuel) & (df_details["range_elec"] == range_elec)
        mask = mask & (df_details[unique_markets] == model_year)
        vehicle_rows = df_details.loc[mask].reset_index()

        # We now aggregate the features for this make, model, trim, fuel, and range_elec for the model_year
        vehicle_features = vehicle_rows.groupby(unique_markets).agg(agg_funs).reset_index()

        # Add to the output
        output = pd.concat([output, vehicle_features], axis=0)
    
    return output

# Step 3: In the original data frame, go through each make, model, fuel, and range_elec and replace with the most common trim
def replace_with_most_common_trim(df, most_common_trim_features, electric = False):
    """
    In the original data frame, replace vehicle features with those of the most common trim for that make, model, fuel, and range_elec.
    
    This function matches each row to an appropriate trim, based on its make, model, fuel, and range_elec.

    NOTE: We expect that in most_common_trim_features, electric vehicles are treated differently. For electric vehicles,
    EVERY trim is kept - not just the most popular. Consequently, for electric vehicles, we merge differently (we need to)
    also include the "trim" column in what to merge on.

    Parameters:
    - electric: A boolean indicating whether to run this for electric vehicles. If True, we keep all trims for electric vehicles.
    """

    # Prepare a column to record successful merges
    most_common_trim_features["merge_success"] = 1

    # Round range to facilitate matching
    most_common_trim_features["range_elec"] = round(most_common_trim_features["range_elec"], 2)
    df["range_elec"] = round(df["range_elec"], 2)

    # The unique identifier for a product is its make, model, fuel, and range_elec.
    # For each of these unique identifiers, for each model year, we assign the most common trim
    cols_merge_on = ["make", "model", "model_year", "fuel", "range_elec"]
    cols_merge_on_noyear = ["make", "model", "fuel", "range_elec"]

    # Additionally keep the sales column and the county name
    cols_to_keep = ["veh_count", "county_name"]

    # For electric vehicles, we keep all trims and do not simplify the products
    if electric:
        cols_merge_on = cols_merge_on + ["trim"]
        cols_merge_on_noyear = cols_merge_on_noyear + ["trim"]

    # Merge the most common trim features with the original data frame
    df_to_keep = df[cols_merge_on + cols_to_keep]
    output = df_to_keep.merge(most_common_trim_features.iloc[:, ~most_common_trim_features.columns.isin(["veh_count"])], on=cols_merge_on, how="left")

    # Confirm this has worked correctly
    assert(len(output) == len(df))
    assert(output["veh_count"].sum() == df["veh_count"].sum())

    # Note that in some cases, the most common trim is not available for a given model year
    # In these cases, we use the most common model year for that trim
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

    # Confirm this has worked correctly and that we have not lost any sales
    assert(len(output) == len(df))
    assert(output["veh_count"].sum() == df["veh_count"].sum())

    return output

# Step 4: Aggregate to the market level
def aggregate_to_market(df, most_common_trim_features):
        """
        Aggregates the data to the market level.
        We assume that with in each make, model, model_year, trim, range_elec, and fuel - the other features are all the same.
        """

        # Round range to facilitate matching
        df["range_elec"] = round(df["range_elec"], 2)
        most_common_trim_features["range_elec"] = round(most_common_trim_features["range_elec"], 2)

        # Prepare the most common trim features
        most_common_trim_features_orig = most_common_trim_features.copy()
        most_common_trim_features = most_common_trim_features.drop(columns = ["veh_count"])
        
        # Aggregate and check
        aggregated = df[["make", "model", "model_year", "trim", "fuel", "range_elec", "county_name", "veh_count"]].groupby(["make", "model", "model_year", "trim", "fuel", "range_elec", "county_name"]).sum().reset_index()
        assert(aggregated["veh_count"].sum() == df["veh_count"].sum())


        # WE GET NAs here - where the most popular trim is not available in a given model_year
        aggregated = aggregated.merge(most_common_trim_features, on = ["make", "model", "model_year", "trim", "fuel", "range_elec"], how = 'left')
        assert(aggregated["veh_count"].sum() == df["veh_count"].sum())

        # Split into matched and unmatched
        aggregated_matched = aggregated[aggregated["msrp"].notna()]
        aggregated_unmatched = aggregated.loc[aggregated["msrp"].isna(), ["make", "model", "model_year", "trim", "fuel", "range_elec", "county_name", "veh_count"]]

        
        # Get the top year
        most_common_trim_features_top = most_common_trim_features_orig.sort_values("veh_count", ascending=False).drop_duplicates(subset = ["make", "model", "trim", "fuel", "range_elec"])
        most_common_trim_features_top = most_common_trim_features_top.drop("veh_count", axis = 1)
        
        # Match in
        cols = {"model_year_x": "model_year", "model_year_y": "model_year_matched"}
        aggregated_unmatched_fixed = aggregated_unmatched.merge(most_common_trim_features_top, on = ["make", "model", "trim", "fuel", "range_elec"], how = 'left').rename(columns =cols)

        assert(len(aggregated_unmatched_fixed)== len(aggregated_unmatched))

        aggregated = pd.concat([aggregated_matched, aggregated_unmatched_fixed], ignore_index=True)

        return aggregated

# Step 5: Rationalize and fix ZMS
def rationalize_markets(df, geographies, threshold, most_common_trim_features):
    """
    Rationalize markets by dropping uncommon products, and adding zero market shares for those that are common but not present.

    Input: 
    df: A data frame grouped by model year, and one grouped by model year and ct.
    geographies: the geography column, a string

    """

    # Get geographies
    geogs = pd.Series(df[geographies].drop_duplicates().tolist(), name = geographies)

    # Fix the electric range
    most_common_trim_features["range_elec"] = round(most_common_trim_features["range_elec"], 2)
    most_common_trim_features_copy = most_common_trim_features.copy()
    most_common_trim_features = most_common_trim_features.drop(columns = ["veh_count"])

    # Get the frequency of products per market year, and remove those below the threshold 
    vars_group_on = ["make", "model", "trim", "fuel", "range_elec"]
    vehs_to_keep = df[vars_group_on+["veh_count", "model_year"]].groupby(vars_group_on + ["model_year"]).sum().reset_index()
    assert(vehs_to_keep["veh_count"].sum() == df["veh_count"].sum())
    vehs_to_keep = vehs_to_keep.loc[vehs_to_keep["veh_count"] > threshold, vars_group_on + ["model_year"]]

    # For the model year and county dataset, we:
    # 1) Keep only the products with sales above the threshold for that model year - keeping the features columns
    output = df.merge(vehs_to_keep, on=vars_group_on + ["model_year"], how="inner")

    # 2) Add zero market shares for products available in one county in a year but not in another
    # First, get the unique geographies, and then for every model year, get every county it should be in
    all_prods_mkts = vehs_to_keep.merge(geogs, how="cross")
    
    # Now merge with the original df - unmatched are the zero market shares
    output = all_prods_mkts.merge(output, on=["make", "model", "model_year", "trim", "fuel", "range_elec"]+[geographies], how="left")

    # Put aside matched
    output_matched = output.loc[output["veh_count"].notna()] # Put this aside

    # For those with zero market shares, replace with 0
    output_unmatched = output.loc[output["veh_count"].isna(), ["make", "model", "model_year", "trim", "fuel", "range_elec"]+[geographies]]
    output_unmatched["veh_count"] = 0
    output_unmatched_elec = replace_with_most_common_trim(output_unmatched.loc[output_unmatched["fuel"]=="electric"], most_common_trim_features_copy, electric = True)
    output_unmatched_nelec = replace_with_most_common_trim(output_unmatched.loc[output_unmatched["fuel"]!="electric"], most_common_trim_features_copy)
    output_unmatched = pd.concat([output_unmatched_elec, output_unmatched_nelec], axis=0, ignore_index = True)
    
    output_out = pd.concat([output_matched, output_unmatched], axis=0, ignore_index=True)
    assert(len(output_out) == len(output))
    assert(output_out["veh_count"].sum() == output["veh_count"].sum())

    return output_out

###########################################################################################
# Set ouput filepaths
most_common_trims_destination = output_folder / f"most_common_trim_features_{date_time}_{lease}.csv"

replaced_destination = output_folder / f"rlp_features_replaced_with_most_common_trim_{date_time}_{lease}.csv"
aggregated_counties_destination = output_folder / f"rlp_with_dollar_per_mile_replaced_myear_county_{date_time}_{lease}.csv"
aggregated_myear_destination = output_folder / f"rlp_with_dollar_per_mile_replaced_myear_{date_time}_{lease}.csv"

rationalized_my_dest = output_folder / f"rlp_with_dollar_per_mile_replaced_myear_{date_time}_{lease}_zms.csv"
rationalized_my_ct_dest = output_folder / f"rlp_with_dollar_per_mile_replaced_myear_county_{date_time}_{lease}_zms.csv"

# Read in the raw RLP data to be processed
df_rlp = pd.read_csv(str_sales_vin_characteristic)
if lease == "no_lease":
    num_leases = df_rlp["transaction_price"].isna().sum()
    print(f"Dropping {num_leases} leases. All files will be marked 'no_lease'.")
    df_rlp = df_rlp.loc[df_rlp["transaction_price"].notna()]

# Run
print("Getting most common trims...")
most_common_trims = get_most_common_trim(df_rlp, sales_col)

print("Getting most common trim features...")
most_common_trim_features = get_most_common_trim_features(df_rlp, most_common_trims)

print("Replacing data in the initial DataFrame with features for the most common trim...")
df_replaced_nelec = replace_with_most_common_trim(df_rlp.loc[df_rlp["fuel"]!="electric"], most_common_trim_features)
df_replaced_elec = replace_with_most_common_trim(df_rlp.loc[df_rlp["fuel"]=="electric"], most_common_trim_features, electric = True)
df_replaced = pd.concat([df_replaced_nelec, df_replaced_elec])

print("Aggregating to the market level...")
aggregated = aggregate_to_market(df_replaced, most_common_trim_features)

print("Rationalizing markets and addressing zero market shares...")
aggregated_zms = rationalize_markets(aggregated, "county_name", threshold, most_common_trim_features)

# Save the outputs
aggregated_zms.to_csv(rationalized_my_ct_dest)




