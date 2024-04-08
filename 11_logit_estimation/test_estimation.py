####################################################################################################
# Test the aggregated data the estimation
# Mostly sense-checks, nothing concrete
####################################################################################################
# Import required packages
import pytest
import numpy as np
import pathlib
import pandas as pd
from tqdm import tqdm

# silence warnings
import warnings
warnings.filterwarnings("ignore")

####################################################################################################
# List of checks
# RLP and RLP estimation data comparison
# 1. Check the share of EVs in the raw and estimation data
# 2. Check the shares of other fuels in the raw vs. estimation data, to ensure that nothing changes in the preparation steps
# 3. Visually inspect the vehicles with the highest shares in each county in the estimation data
# 4. Check that the average msrp for each fueltype does not change between the raw and estimation data
# 5. Check that the weighted average msrp for each fueltype does not change between the raw and estimation data
# 6. Visually inspect the comparison of the weighted average MSRP for each fueltype between the raw and estimation data.
# 7. Visually inspect the differences in weighted average dollars per mile for each fueltype between the raw and estimation data.
# 8. Check the availability of products in each market

# Test the different aggregations of the RLP data - e.g. aggregating by model year
# 9. Check the share of EVs in the raw and estimation data

# Experian data comparison
# 10. Check the average MSRP between the raw experian data, and the raw and estimation RLP data.
# 11. Visually inspect differences in weighted MSRP by fuel type between the experian and RLP data
# 12. Visually inspect differences in weighted dollars per mile by fuel type between the experian and RLP data

# Experian estimation data comparison
# 13. Visually inspect the number of unique make, model, model_year combinations in the rlp and experian estimation data.
# 14. Confirm that fuel type dummies have been set up correctly in the RLP and experian estimation data


####################################################################################################
# Set directories
if True:
    str_cwd = pathlib.Path().resolve().parent.parent
    str_dir = str_cwd / "Documents" / "tobin_working_data"
    str_rlp_data = str_dir / "rlpolk_data"
    rlp_data_file = "rlp_with_dollar_per_mile.csv"
    estimation_data = str_dir / "estimation_data_test" / "mkt_data_county.csv"
    estimation_data_myear = str_dir / "estimation_data_test" / "mkt_data_model_year.csv"
    exp_estimation_data = str_dir / "estimation_data_test" / "exp_mkt_data.csv"
    experian_data = str_dir / "intermediate" / "US_VIN_data_common.csv"
    str_outputs = str_dir / "estimation_data_test"

    # Save the vehicle characteristic comparison here
    veh_char_comp_str = str_outputs / "raw_data_vehicle_chars_comparison.csv"

    # Import the raw and aggregateed data
    df_rlp = pd.read_csv(str_rlp_data / rlp_data_file)
    df_estimation = pd.read_csv(estimation_data)

####################################################################################################
@pytest.fixture
def setup_rlp_data():
    df_rlp_raw = pd.read_csv(str_rlp_data / rlp_data_file)
    df_estimation = pd.read_csv(estimation_data)
    return df_rlp, df_estimation

@pytest.fixture
def setup_rlp_estimation_data():
    df_estimation = pd.read_csv(estimation_data)
    df_estimation_my = pd.read_csv(estimation_data_myear)
    yield df_estimation, df_estimation_my

@pytest.fixture
def setup_rlp_myear_data():
    df = pd.read_csv(estimation_data_myear)
    yield df

@pytest.fixture
def setup_experian_data():
    df_experian = pd.read_csv(experian_data)
    df_experian = df_experian.loc[df_experian["state"] == "CONNECTICUT", :]
    df_experian = df_experian.loc[df_experian.msrp.notna(), :]
    df_experian = df_experian.loc[df_experian.agg_count.notna(), :]
    df_experian = df_experian.loc[df_experian.dollar_per_mile.notna(), :]

    return df_experian

@pytest.fixture
def setup_exp_estimation_data():
    df_exp_estimation = pd.read_csv(exp_estimation_data)
    # df_exp_estimation = .loc[df_exp_estimation["state"] == "CONNECTICUT", :]
    df_exp_estimation = df_exp_estimation.loc[df_exp_estimation.msrp.notna(), :]
    df_exp_estimation = df_exp_estimation.loc[df_exp_estimation.agg_count.notna(), :]
    df_exp_estimation = df_exp_estimation.loc[df_exp_estimation.dollar_per_mile.notna(), :]
    df_exp_estimation["fuel"] = df_exp_estimation["fuel"].str.lower()
    return df_exp_estimation

@pytest.fixture
def setup_comparison_data():
    df = pd.read_csv(veh_char_comp_str)
    yield df

####################################################################################################
def test_veh_chars_raw(setup_rlp_data, setup_rlp_estimation_data, setup_exp_estimation_data):
    df_rlp_est_ct, df_rlp_est_my = setup_rlp_estimation_data
    df_experian = setup_exp_estimation_data

    df_rlp = df_rlp_est_my

    # Fix some column names
    df_rlp  = df_rlp.rename(columns = {"log_hp_wt": "log_hp_weight"})
    df_experian = df_experian.rename(columns = {"curbwt": "curb_weight", "year": "model_year"})

    # Note variables to compare
    vars_to_compare = ["make", "model","model_year", "trim", "fuel", "msrp", "doors", "curb_weight", "max_hp", "dollar_per_mile", "log_hp_weight", "wheelbase", "range_elec"]

    # Get unique make, model, model_year combinations, and for each calculate a mean dollars_per_mile
    unique_vehs = df_rlp[vars_to_compare].drop_duplicates()
    print(len(unique_vehs))
    # dpm = df_rlp.groupby(vars_to_compare)["dollar_per_mile"].mean().reset_index()
    # unique_vehs = unique_vehs.merge(dpm, on = vars_to_compare, how = "left")

    # Update variables
    # vars_to_compare = vars_to_compare + ["dollar_per_mile"]

    # Define function to compare
    def get_comparisons(rlp_data, exp_data, vars_to_compare):
        joined = pd.DataFrame()
        for index, row in tqdm(rlp_data.iterrows()):
            rlp_vehicle = row[vars_to_compare]
            exp_vehicle = exp_data.loc[(exp_data["make"] == rlp_vehicle["make"]) & (exp_data["model"] == rlp_vehicle["model"]) & (exp_data["model_year"] == rlp_vehicle["model_year"]) &(exp_data["trim"]==rlp_vehicle["trim"]), vars_to_compare]
            rlp_vehicle = pd.DataFrame(rlp_vehicle).T
            rlp_vehicle = rlp_vehicle.reset_index(drop=True)
            if len(exp_vehicle) == 0:
                exp_vehicle = pd.DataFrame(np.nan, index = [index], columns = vars_to_compare)
            else:
                exp_vehicle = pd.DataFrame(exp_vehicle)
            exp_vehicle.columns = [f"exp_{col}" for col in exp_vehicle.columns]
            exp_vehicle = exp_vehicle.reset_index(drop=True)

            veh = pd.concat([rlp_vehicle, exp_vehicle], axis = 1)
            joined = pd.concat([joined, veh], axis = 0)

        return joined
        
    # Compare the data and save to CSV
    joined = get_comparisons(unique_vehs, df_experian, vars_to_compare)
    joined.to_csv(str_outputs / "estimation_data_vehicle_comparison.csv", index = False)


def test_veh_chars_comparison(setup_comparison_data, setup_rlp_data, setup_experian_data):
    df_rlp, _ = setup_rlp_data
    df_experian = setup_experian_data
    df_comp = setup_comparison_data

    # Fix some column names
    df_experian = df_experian.rename(columns = {"curbwt": "curb_weight", "year": "model_year"})

    # Get variables to compare
    vars_to_compare = ["make", "model","model_year", "trim", "fuel", "msrp", "doors", "curb_weight", "max_hp", "log_hp_weight", "wheelbase", "range_elec"]
    unique_vehs_exp = df_experian[vars_to_compare].drop_duplicates()

    # Compare how many matched
    num_matched = df_comp["exp_make"].notna().sum()
    total_unique_exp = unique_vehs_exp.shape[0]
    print(f"Percentage of experian vehicles found in RLP data: {num_matched/total_unique_exp:.2%}")

    def compare_metric(metric):
        comp = pd.DataFrame(columns = ["fuel", "rlp_avg", "exp_avg", "rlp_unmatched_avg"])
        for fuel in ["electric", "gasoline", "diesel", "phev", "hybrid"]:
            rlp_avg = df_comp.loc[(df_comp["fuel"]==fuel) & (df_comp["exp_make"].notna()), metric].mean()
            exp_avg = df_comp.loc[(df_comp["fuel"]==fuel) & (df_comp["exp_make"].notna()), f"exp_{metric}"].mean()
            rlp_unmatched_avg = df_comp.loc[(df_comp["fuel"]==fuel) & (df_comp["exp_make"].isna()), metric].mean()
            comp.loc[fuel] = [fuel, rlp_avg, exp_avg, rlp_unmatched_avg]
        return comp

    # Compare the average MSRP between the two datasets by fueltype:
    fuel_comp = compare_metric("msrp")
    dollar_per_mile_comp = compare_metric("dollar_per_mile")
    log_hp_weight_comp = compare_metric("log_hp_weight")

    print(fuel_comp)
    print(dollar_per_mile_comp)
    print(log_hp_weight_comp)


####################################################################################################

def test_ev_share():
    """
    Test the share of EVs in the raw and estimation data
    """
    # Calculate the EV share
    ev_count_raw = df_rlp.loc[df_rlp["fuel"]=="electric", "veh_count"].sum()
    ev_share_raw = ev_count_raw / df_rlp["veh_count"].sum()
    
    ev_count_estimation = df_estimation.loc[df_estimation["fuel"]=="electric", "veh_count"].sum()
    ev_share_estimation = ev_count_estimation / df_estimation["veh_count"].sum()
    
    # Check the EV share
    assert np.isclose(ev_share_raw, ev_share_estimation, atol=0.01), "The EV share is not as expected."

    print(f"The EV share in the raw data is {ev_share_raw:.2%} (count: {ev_count_raw})")
    print(f"The EV share in the estimation data is {ev_share_estimation:.2%} (count: {ev_count_estimation})")

def test_other_shares():
    """
    Test the shares of other fuels in the raw vs. estimation data, to ensure that nothing changes in the preparation steps
    """
    for fuel in ["gasoline", "diesel", "phev", "hybrid"]:
        # Calculate the share
        count_raw = df_rlp.loc[df_rlp["fuel"]==fuel, "veh_count"].sum()
        count_estimation = df_estimation.loc[df_estimation["fuel"]==fuel, "veh_count"].sum()
        share_raw = count_raw / df_rlp["veh_count"].sum()
        share_estimation = count_estimation / df_estimation["veh_count"].sum()

        print(f"-------------------------------------------------")
        print(f"The {fuel} share in the raw data is {share_raw:.2%} (count: {count_raw})")
        print(f"The {fuel} share in the estimation data is {share_estimation:.2%} (count: {count_estimation})")
        
        # Check the share
        assert np.isclose(share_raw, share_estimation, atol=0.01), f"The {fuel} share is not as expected."

def test_highest_shares():
    """
    Visually inspect the vehicles with the highest shares in each county in the estimation data
    """

    for market in df_estimation["market_ids"].unique():
        print(f"-----------------{market}-----------------")
        df_market = df_estimation.loc[df_estimation["market_ids"]==market, ["market_ids", "make", "model", "model_year", "shares"]].sort_values("shares", ascending=False)
        print(df_market.head(5))

def test_average_msrp():
    """
    Check that the average msrp for each fueltype does not change between the raw and estimation data
    """

    for fuel in ["electric", "gasoline", "diesel", "phev", "hybrid"]:
        msrp_raw = df_rlp.loc[df_rlp["fuel"]==fuel, "msrp"].mean()
        msrp_estimation = df_estimation.loc[df_estimation["fuel"]==fuel, "msrp"].mean()

        print(f"-------------------------------------------------")
        print(f"The average MSRP for {fuel} in the raw data is ${msrp_raw:.2f}")
        print(f"The average MSRP for {fuel} in the estimation data is ${msrp_estimation:.2f}")
        
        # Check the MSRP
        assert np.isclose(msrp_raw, msrp_estimation, atol=5, rtol = 0.1), f"The average MSRP for {fuel} is not as expected."

def test_weighted_average_msrp():
    """
    Check that the weighted average msrp for each fueltype does not change between the raw and estimation data
    """
    for fuel in ["electric", "gasoline", "diesel", "phev", "hybrid"]:
        msrp_raw = np.average(df_rlp.loc[df_rlp["fuel"]==fuel, "msrp"], weights=df_rlp.loc[df_rlp["fuel"]==fuel, "veh_count"])
        msrp_estimation = np.average(df_estimation.loc[df_estimation["fuel"]==fuel, "msrp"], weights=df_estimation.loc[df_estimation["fuel"]==fuel, "veh_count"])

        print(f"-------------------------------------------------")
        print(f"The weighted average MSRP for {fuel} in the raw data is ${msrp_raw:.2f}")
        print(f"The weighted average MSRP for {fuel} in the estimation data is ${msrp_estimation:.2f}")
        
        # Check the MSRP
        assert np.isclose(msrp_raw, msrp_estimation, atol=5, rtol = 0.1), f"The weighted average MSRP for {fuel} is not as expected."

def test_compare_weighted_average_msrp():
    """
    Visually inspect the comparison of the weighted average MSRP for each fueltype between the raw and estimation data.
    I.e. whether the relative differences in average MSRPs is similar across the datasets. 
    """

    df = pd.DataFrame(columns = ["wmsrp_raw", "wmsrp_estimation"])

    for fuel in ["electric", "gasoline", "diesel", "phev", "hybrid"]:
        msrp_raw = np.average(df_rlp.loc[df_rlp["fuel"]==fuel, "msrp"], weights=df_rlp.loc[df_rlp["fuel"]==fuel, "veh_count"])
        msrp_estimation = np.average(df_estimation.loc[df_estimation["fuel"]==fuel, "msrp"], weights=df_estimation.loc[df_estimation["fuel"]==fuel, "veh_count"])
        df.loc[fuel] = [msrp_raw, msrp_estimation]
    
    df_comparison_raw = pd.DataFrame(columns = ["electric", "gasoline", "diesel", "phev", "hybrid"], index = ["electric", "gasoline", "diesel", "phev", "hybrid"])

    for fuel in ["electric", "gasoline", "diesel", "phev", "hybrid"]:
        for fuel2 in ["electric", "gasoline", "diesel", "phev", "hybrid"]:
            df_comparison_raw.loc[fuel, fuel2] = round((df.loc[fuel, "wmsrp_raw"] - df.loc[fuel2, "wmsrp_raw"]) / df.loc[fuel2, "wmsrp_raw"], 2)
    
    df_comparison_estimation = pd.DataFrame(columns = ["electric", "gasoline", "diesel", "phev", "hybrid"], index = ["electric", "gasoline", "diesel", "phev", "hybrid"])
    for fuel in ["electric", "gasoline", "diesel", "phev", "hybrid"]:
        for fuel2 in ["electric", "gasoline", "diesel", "phev", "hybrid"]:
            df_comparison_estimation.loc[fuel, fuel2] = round((df.loc[fuel, "wmsrp_estimation"] - df.loc[fuel2, "wmsrp_estimation"]) / df.loc[fuel2, "wmsrp_estimation"], 2)

    print("-------- Weighted Average MSRP Comparison -------")
    print(df)   
    print("-------Fuel type differences (Raw Data)-----------")
    print(df_comparison_raw)
    print("------Fuel type differences (Estimation Data)-----")
    print(df_comparison_estimation)

def test_compare_weighted_average_dollars_per_mile():
    """
    Visually inspect the differences in weighted average dollars per mile for each fueltype between the raw and estimation data.
    """

    df = pd.DataFrame(columns = ["wdpm_raw", "wdpm_estimation"])

    for fuel in ["electric", "gasoline", "diesel", "phev", "hybrid"]:
        wdpm_raw = np.average(df_rlp.loc[df_rlp["fuel"]==fuel, "dollar_per_mile"], weights=df_rlp.loc[df_rlp["fuel"]==fuel, "veh_count"])
        wdpm_estimation = np.average(df_estimation.loc[df_estimation["fuel"]==fuel, "dollar_per_mile"], weights=df_estimation.loc[df_estimation["fuel"]==fuel, "veh_count"])
        df.loc[fuel] = [wdpm_raw, wdpm_estimation]
    
    df_comparison_raw = pd.DataFrame(columns = ["electric", "gasoline", "diesel", "phev", "hybrid"], index = ["electric", "gasoline", "diesel", "phev", "hybrid"])

    for fuel in ["electric", "gasoline", "diesel", "phev", "hybrid"]:
        for fuel2 in ["electric", "gasoline", "diesel", "phev", "hybrid"]:
            df_comparison_raw.loc[fuel, fuel2] = round((df.loc[fuel, "wdpm_raw"] - df.loc[fuel2, "wdpm_raw"]) / df.loc[fuel2, "wdpm_raw"], 2)
    
    df_comparison_estimation = pd.DataFrame(columns = ["electric", "gasoline", "diesel", "phev", "hybrid"], index = ["electric", "gasoline", "diesel", "phev", "hybrid"])
    for fuel in ["electric", "gasoline", "diesel", "phev", "hybrid"]:
        for fuel2 in ["electric", "gasoline", "diesel", "phev", "hybrid"]:
            df_comparison_estimation.loc[fuel, fuel2] = round((df.loc[fuel, "wdpm_estimation"] - df.loc[fuel2, "wdpm_estimation"]) / df.loc[fuel2, "wdpm_estimation"], 2)

    print("-------- Weighted Average Dollars per Mile Comparison -------")
    print(df)   
    print("-------Fuel type differences (Raw Data)-----------")
    print(df_comparison_raw)
    print("------Fuel type differences (Estimation Data)-----")
    print(df_comparison_estimation)
    print("Note: During data preparation, dollars per mile was multiplied by 100, hence the difference in the values.")

def test_market_availability():
    """
    Check the proportions of products available in different numbers of markets. 
    """

    df_cp = df_estimation.copy()

    mkts_products = df_cp[["market_ids", "product_ids"]].drop_duplicates()

    # count how many markets each product is available in
    mkt_counts = mkts_products.groupby("product_ids").count()
    mkt_counts = mkt_counts.reset_index().rename(columns = {"market_ids": "count"})
    mkt_counts = pd.DataFrame(mkt_counts.groupby("count").size()).reset_index().rename(columns = {0: "num_products"})
    print(mkt_counts)
    print(mkt_counts["num_products"].sum())

def test_aggregation_myear(setup_rlp_data, setup_rlp_myear_data):
    """
    Check the share of EVs in the raw and estimation data
    """
    df_rlp, _ = setup_rlp_data
    df_estimation_my = setup_rlp_myear_data

    # Extract unique model years and county names
    model_years = df_rlp["model_year"].unique().tolist()

    # Extract market ids and confirm that all of them are in fact model years
    mkt_ids_myear = df_estimation_my["market_ids"].unique().tolist()

    for mkt_id in mkt_ids_myear:
        assert(mkt_id in model_years), "The market id is not a model year."

def test_correspondence(setup_rlp_data):
    """
    Check the correspondence between the raw and estimation data by ensuring MSRPs and sales align
    """
    df_rlp, df_estimation = setup_rlp_data

    # Filter df_rlp to contain only the VINs in the df_estimation
    df_rlp_filtered = df_rlp.loc[df_rlp["vin_pattern"].isin(df_estimation["vin_pattern"]), :]

    # Check that the sales per VIN pattern are the same
    sales_rlp = df_rlp_filtered.groupby("vin_pattern")["veh_count"].sum().sort_values().reset_index()
    sales_estimation = df_estimation.groupby("vin_pattern")["veh_count"].sum().sort_values().reset_index()
    sales_rlp = sales_rlp.rename(columns = {"veh_count": "veh_count_rlp"})
    sales_estimation = sales_estimation.rename(columns = {"veh_count": "veh_count_estimation"})
    sales_joined = sales_rlp.merge(sales_estimation, on = "vin_pattern", how = "inner")
    sales_joined = sales_joined.sort_values("veh_count_rlp", ascending = False)
    sales_joined["diff"] = sales_joined["veh_count_rlp"] - sales_joined["veh_count_estimation"]

    print(f"Number of VIN patterns in the RLP data: {sales_rlp.shape[0]}")
    print(f"Number of VIN patterns in the estimation data: {sales_estimation.shape[0]}")
    print(f"Number of VIN patterns in the joined data: {sales_joined.shape[0]}")
    print(f"Total difference in sales: {sales_joined['diff'].sum()}")
    print(f"Total number of VINs with different sales: {sales_joined.loc[sales_joined['diff'] != 0, :].shape[0]}")

    # Now check that the MSRPs correspond
    msrp_rlp = df_rlp_filtered.groupby("vin_pattern")["msrp"].mean().sort_values().reset_index()
    msrp_estimation = df_estimation.groupby("vin_pattern")["msrp"].mean().sort_values().reset_index()
    msrp_rlp = msrp_rlp.rename(columns = {"msrp": "msrp_rlp"})
    msrp_estimation = msrp_estimation.rename(columns = {"msrp": "msrp_estimation"})
    msrp_joined = msrp_rlp.merge(msrp_estimation, on = "vin_pattern", how = "inner")
    msrp_joined = msrp_joined.sort_values("msrp_rlp", ascending = False)
    msrp_joined["diff"] = msrp_joined["msrp_rlp"] - msrp_joined["msrp_estimation"]

    print(f"Number of VINs with different MSRP: {msrp_joined.loc[msrp_joined['diff'] != 0, :].shape[0]}")
    print(f"Average difference in MSRP for the VINs: {msrp_joined['diff'].mean()}")
    print(msrp_joined.head(10))


######################################################################################################
# Comparison with raw experian data
def test_compare_experian_msrp(setup_experian_data):
    """
    Compare the average MSRP between the raw experian data, and the raw and estimation RLP data.
    We use an absolute tolerance of 5, and a relative tolerance of 0.1. 
    """
    df_experian = setup_experian_data

    # Make the model years comparable
    mask = (df_experian.model_year >= df_rlp.model_year.min()) & (df_experian.model_year <= df_rlp.model_year.max())
    df_experian = df_experian.loc[mask, :]

    # Calculate the weighted average MSRP
    msrp_experian = np.average(df_experian["msrp"], weights=df_experian["agg_count"])
    msrp_rlp = np.average(df_rlp["msrp"], weights=df_rlp["veh_count"])
    msrp_estimation = np.average(df_estimation["msrp"], weights=df_estimation["veh_count"])

    print(f"The weighted average MSRP for the Experian data is ${msrp_experian:.2f}")
    print(f"The weighted average MSRP for the RLP data is ${msrp_rlp:.2f}")
    print(f"The weighted average MSRP for the estimation data is ${msrp_estimation:.2f}")

    # Check the MSRP
    assert np.isclose(msrp_experian, msrp_rlp, atol=5, rtol = 0.1), "The weighted average MSRP for the Experian data is not as expected."
    assert np.isclose(msrp_experian, msrp_estimation, atol=5, rtol = 0.1), "The weighted average MSRP for the Experian data is not as expected."

def test_compare_experian_msrp_fueltype(setup_rlp_data, setup_experian_data, setup_exp_estimation_data):
    """
    Visually inspect differences in weighted MSRP by fuel type between the experian and RLP data
    """

    df_rlp, df_estimation = setup_rlp_data
    df_experian = setup_experian_data
    df_experian_estimation = setup_exp_estimation_data
    df_experian["fuel"] = df_experian["fuel"].str.lower()
    df_experian_estimation["fuel"] = df_experian_estimation["fuel"].str.lower()

    df = pd.DataFrame(columns = ["wmsrp_experian", "wmsrp_experian_est", "wmsrp_rlp", "wmsrp_estimation"])

    for fuel in ["electric", "gasoline", "diesel", "phev", "hybrid"]:
        msrp_experian = np.average(df_experian.loc[df_experian["fuel"]==fuel, "msrp"], weights=df_experian.loc[df_experian["fuel"]==fuel, "agg_count"])
        try:
            msrp_experian_estimation = np.average(df_experian_estimation.loc[df_experian_estimation["fuel"]==fuel, "msrp"], weights=df_experian_estimation.loc[df_experian_estimation["fuel"]==fuel, "agg_count"])
        except:
            msrp_experian_estimation = np.nan
        msrp_rlp = np.average(df_rlp.loc[df_rlp["fuel"]==fuel, "msrp"], weights=df_rlp.loc[df_rlp["fuel"]==fuel, "veh_count"])
        msrp_estimation = np.average(df_estimation.loc[df_estimation["fuel"]==fuel, "msrp"], weights=df_estimation.loc[df_estimation["fuel"]==fuel, "veh_count"])
        
        df.loc[fuel] = [msrp_experian, msrp_experian_estimation, msrp_rlp, msrp_estimation]
    
    print("-------- Weighted Average MSRP Comparison -------")
    print(df)

def test_compare_experian_dollars_per_mile_fueltype(setup_rlp_data, setup_experian_data):
    """
    Visually inspect differences in weighted dollars per mile by fuel type between the experian and RLP data
    """
    df_rlp, df_estimation = setup_rlp_data
    df_experian = setup_experian_data
    df_experian["fuel"] = df_experian["fuel"].str.lower()

    df = pd.DataFrame(columns = ["wdpm_experian", "wdpm_rlp"])

    for fuel in ["electric", "gasoline", "diesel", "phev", "hybrid"]:
        wdpm_experian = np.average(df_experian.loc[df_experian["fuel"]==fuel, "dollar_per_mile"], weights=df_experian.loc[df_experian["fuel"]==fuel, "agg_count"])
        wdpm_rlp = np.average(df_rlp.loc[df_rlp["fuel"]==fuel, "dollar_per_mile"], weights=df_rlp.loc[df_rlp["fuel"]==fuel, "veh_count"])
        # wdpm_estimation = np.average(df_estimation.loc[df_estimation["fuel"]==fuel, "dollar_per_mile"], weights=df_estimation.loc[df_estimation["fuel"]==fuel, "veh_count"])
        df.loc[fuel] = [wdpm_experian, wdpm_rlp]
    
    print("")
    print("-------- Weighted Average Dollars per Mile Comparison -------")
    print(round(df * 100, 3))
    print("")

######################################################################################################
# Tests to confirm and compare ESTIMATION datasets
def test_est_years(setup_rlp_estimation_data, setup_exp_estimation_data):
    """
    Test the model years
    """
    df_rlp_est_ct, df_rlp_est_my = setup_rlp_estimation_data
    df_exp_est = setup_exp_estimation_data

    # Compare the model years
    model_years_rlp_ct = sorted(df_rlp_est_ct["model_year"].unique().tolist())
    model_years_rlp_my = sorted(df_rlp_est_my["model_year"].unique().tolist())
    model_years_exp_est = sorted(df_exp_est["model_year"].unique().tolist())

    # Print 
    print(f"Model years in the RLP estimation data (CT): {model_years_rlp_ct}")
    print(f"Model years in the RLP estimation data (MY): {model_years_rlp_my}")
    print(f"Model years in the experian estimation data: {model_years_exp_est}")

def test_est_prods_per_mkt(setup_rlp_estimation_data, setup_exp_estimation_data):
    """
    Test the number of products in each market
    """
    df_rlp_est_ct, df_rlp_est_my = setup_rlp_estimation_data
    df_exp_est = setup_exp_estimation_data

    # Compare the number of products in each market
    num_prods_rlp_ct = df_rlp_est_ct.groupby("market_ids")["product_ids"].nunique().reset_index()
    num_prods_rlp_my = df_rlp_est_my.groupby("market_ids")["product_ids"].nunique().reset_index()
    num_prods_exp_est = df_exp_est.groupby("market_ids")["product_ids"].nunique().reset_index()

    print(f"Number of products in each market in the RLP estimation data (CT): {num_prods_rlp_ct}")
    print(f"Number of products in each market in the RLP estimation data (MY): {num_prods_rlp_my}")
    print(f"Number of products in each market in the experian estimation data: {num_prods_exp_est}")

def test_est_unique_models(setup_rlp_estimation_data, setup_exp_estimation_data):
    """
    Visually inspect differences in the number of unique make, model, model_year combinations in the rlp and experian estimation data. 
    """

    df_rlp_est_ct, df_rlp_est_my = setup_rlp_estimation_data
    df_exp_est = setup_exp_estimation_data

    # Compare number of unique make, model, and model_year combinations
    num_unique_makes_rlp_ct = df_rlp_est_ct[["make", "model", "model_year"]].drop_duplicates().shape[0]
    num_unique_makes_rlp_my = df_rlp_est_my[["make", "model", "model_year"]].drop_duplicates().shape[0]
    num_unique_makes_exp_est = df_exp_est[["make", "model", "model_year"]].drop_duplicates().shape[0]

    print("-----------------ESTIMATION DATA COMPPARISON-----------------")
    print(f"Number of unique make, model, and model_year combinations in the RLP estimation data (CT): {num_unique_makes_rlp_ct}")
    print(f"Number of unique make, model, and model_year combinations in the RLP estimation data (MY): {num_unique_makes_rlp_my}")
    print(f"Number of unique make, model, and model_year combinations in the experian estimation data: {num_unique_makes_exp_est}")

def test_est_fueltype_dummies(setup_rlp_estimation_data, setup_exp_estimation_data):
    """
    Confirm that fuel type dummies have been set up correctly in the RLP and experian estimation data
    """

    df_rlp_est_ct, df_rlp_est_my = setup_rlp_estimation_data
    df_exp_est = setup_exp_estimation_data

    # Correct the dataframe names
    df_estimation = df_rlp_est_ct
    df_exp_estimation = df_exp_est

    # Make the fueltypes lower case
    df_rlp["fuel"] = df_rlp["fuel"].str.lower()
    df_estimation["fuel"] = df_estimation["fuel"].str.lower()
    df_exp_estimation["fuel"] = df_exp_estimation["fuel"].str.lower()

    assert(df_estimation.loc[df_estimation.electric == 1, "fuel"].unique() == ["electric"])
    assert(df_estimation.loc[df_estimation.phev == 1, "fuel"].unique() == ["phev"])
    assert(df_estimation.loc[df_estimation.hybrid == 1, "fuel"].unique() == ["hybrid"])
    assert(df_estimation.loc[df_estimation.diesel == 1, "fuel"].unique() == ["diesel"])

    assert(df_rlp_est_my.loc[df_rlp_est_my.electric == 1, "fuel"].unique() == ["electric"])
    assert(df_rlp_est_my.loc[df_rlp_est_my.phev == 1, "fuel"].unique() == ["phev"])
    assert(df_rlp_est_my.loc[df_rlp_est_my.hybrid == 1, "fuel"].unique() == ["hybrid"])
    assert(df_rlp_est_my.loc[df_rlp_est_my.diesel == 1, "fuel"].unique() == ["diesel"])

    assert(df_exp_estimation.loc[df_exp_estimation.electric == 1, "fuel"].unique() == ["electric"])
    assert(df_exp_estimation.loc[df_exp_estimation.phev == 1, "fuel"].unique() == ["phev"])
    assert(df_exp_estimation.loc[df_exp_estimation.hybrid == 1, "fuel"].unique() == ["hybrid"])
    assert(df_exp_estimation.loc[df_exp_estimation.diesel == 1, "fuel"].unique() == ["diesel"])

    # And assert when these are not 1, the fuel type is gasoline or flex fuel
    assert(np.all(df_estimation.loc[(df_estimation.electric != 1) & (df_estimation.phev != 1) & (df_estimation.hybrid != 1) & (df_estimation.diesel != 1), "fuel"].unique() == ["gasoline", "flex fuel"]))
    assert(np.all(df_rlp_est_my.loc[(df_rlp_est_my.electric != 1) & (df_rlp_est_my.phev != 1) & (df_rlp_est_my.hybrid != 1) & (df_rlp_est_my.diesel != 1), "fuel"].unique() == ["gasoline", "flex fuel"]))
    assert(np.all(df_exp_estimation.loc[(df_exp_estimation.electric != 1) & (df_exp_estimation.phev != 1) & (df_exp_estimation.hybrid != 1) & (df_exp_estimation.diesel != 1), "fuel"].unique() == ["gasoline", "flex fuel"]))

def test_est_market_shares(setup_rlp_estimation_data, setup_exp_estimation_data):
    """
    Compare the market shares between the RLP and experian estimation data
    """
    df_rlp_est_ct, df_rlp_est_my = setup_rlp_estimation_data
    df_exp_est = setup_exp_estimation_data

    # Ensure all columns are printed
    pd.set_option('display.max_columns', None)

    # Compare the mean, median, and SD of market shares by market for df_rlp_est_my and df_exp_est
    print("-------- Market Share Comparison -------")
    print("RLP Estimation Data (MY)")
    print(df_rlp_est_my.groupby("market_ids")["shares"].describe())
    print("Experian Estimation Data")
    print(df_exp_est.groupby("market_ids")["shares"].describe())

def test_est_msrp(setup_rlp_estimation_data, setup_exp_estimation_data):
    """
    Compare the MSRP between the RLP and experian estimation data
    """
    df_rlp_est_ct, df_rlp_est_my = setup_rlp_estimation_data
    df_exp_est = setup_exp_estimation_data

    # Compare the average MSRP between the RLP and experian estimation data by fueltype
    output = pd.DataFrame(columns = ["fuel", "msrp_rlp_ct", "msrp_rlp_my", "msrp_exp_est"])
    for fuel in ["electric", "gasoline", "diesel", "phev", "hybrid"]:
        msrp_rlp_ct = np.average(df_rlp_est_ct.loc[df_rlp_est_ct["fuel"]==fuel, "msrp"], weights=df_rlp_est_ct.loc[df_rlp_est_ct["fuel"]==fuel, "veh_count"])
        msrp_rlp_my = np.average(df_rlp_est_my.loc[df_rlp_est_my["fuel"]==fuel, "msrp"], weights=df_rlp_est_my.loc[df_rlp_est_my["fuel"]==fuel, "veh_count"])
        msrp_exp_est = np.average(df_exp_est.loc[df_exp_est["fuel"]==fuel, "msrp"], weights=df_exp_est.loc[df_exp_est["fuel"]==fuel, "agg_count"])

        output.loc[fuel] = [fuel, msrp_rlp_ct, msrp_rlp_my, msrp_exp_est]

        # Check the MSRP
        #assert np.isclose(msrp_rlp_ct, msrp_exp_est, atol=5, rtol = 0.1), f"The weighted average MSRP for {fuel} is not as expected."
        # assert np.isclose(msrp_rlp_my, msrp_exp_est, atol=5, rtol = 0.1), f"The weighted average MSRP for {fuel} is not as expected."

    print("-------- Weighted Average MSRP Comparison -------")
    print(output)

def test_est_mkts_per_prod(setup_rlp_estimation_data, setup_exp_estimation_data):
    """
    Compare the number of markets per product between the RLP and experian estimation data
    """
    df_rlp_est_ct, df_rlp_est_my = setup_rlp_estimation_data
    df_exp_est = setup_exp_estimation_data

    product_ids = ["make", "model", "trim"]

    # Compare the number of markets per product
    num_mkts_per_prod_rlp_ct = df_rlp_est_ct.groupby(product_ids)["market_ids"].nunique().reset_index()
    num_mkts_per_prod_rlp_my = df_rlp_est_my.groupby(product_ids)["market_ids"].nunique().reset_index()
    num_mkts_per_prod_exp_est = df_exp_est.groupby(product_ids)["market_ids"].nunique().reset_index()

    # Aggregate by the number of markets
    num_mkts_per_prod_rlp_ct = pd.DataFrame(num_mkts_per_prod_rlp_ct.groupby("market_ids").size()).reset_index().rename(columns = {0: "num_products"})
    num_mkts_per_prod_rlp_my = pd.DataFrame(num_mkts_per_prod_rlp_my.groupby("market_ids").size()).reset_index().rename(columns = {0: "num_products"})
    num_mkts_per_prod_exp_est = pd.DataFrame(num_mkts_per_prod_exp_est.groupby("market_ids").size()).reset_index().rename(columns = {0: "num_products"})

    print(f"Number of markets per product in the RLP estimation data (CT): {num_mkts_per_prod_rlp_ct}")
    print(f"Number of markets per product in the RLP estimation data (MY): {num_mkts_per_prod_rlp_my}")
    print(f"Number of markets per product in the experian estimation data: {num_mkts_per_prod_exp_est}")

def test_compare_experian_estimation_data_msrp(setup_rlp_data, setup_exp_estimation_data):
    df_rlp, df_estimation = setup_rlp_data
    df_exp_estimation = setup_exp_estimation_data

    msrp_rlp = np.average(df_rlp["msrp"], weights=df_rlp["veh_count"])
    msrp_estimation = np.average(df_estimation["msrp"], weights=df_estimation["veh_count"])
    msrp_exp_estimation = np.average(df_exp_estimation["msrp"], weights=df_exp_estimation["agg_count"])

    print(f"The weighted average MSRP for the RLP data is ${msrp_rlp:.2f}")
    print(f"The weighted average MSRP for the estimation data is ${msrp_estimation:.2f}")
    print(f"The weighted average MSRP for the experian estimation data is ${msrp_exp_estimation:.2f}")

    # Check the MSRP
    assert np.isclose(msrp_rlp, msrp_estimation, atol=5, rtol = 0.1), "The weighted average MSRP for the RLP data is not as expected."
    assert np.isclose(msrp_rlp, msrp_exp_estimation, atol=5, rtol = 0.1), "The weighted average MSRP for the RLP data is not as expected."

def test_est_fueltype_shares(setup_rlp_estimation_data, setup_exp_estimation_data):
    df_rlp_est_ct, df_rlp_est_my = setup_rlp_estimation_data
    df_exp_estimation = setup_exp_estimation_data
    df_exp_estimation["fuel"] = df_exp_estimation["fuel"].str.lower()

    df = pd.DataFrame(columns = ["share_rlp_ct", "share_rlp_my", "share_exp"])

    for fuel in ["electric", "gasoline", "diesel", "phev", "hybrid"]:
        share_rlp_ct = df_rlp_est_ct.loc[df_rlp_est_ct["fuel"]==fuel, "veh_count"].sum() / df_rlp_est_ct["veh_count"].sum()
        share_rlp_my = df_rlp_est_my.loc[df_rlp_est_my["fuel"]==fuel, "veh_count"].sum() / df_rlp_est_my["veh_count"].sum()
        share_exp_estimation = df_exp_estimation.loc[df_exp_estimation["fuel"]==fuel, "agg_count"].sum() / df_exp_estimation["agg_count"].sum()
        df.loc[fuel] = [share_rlp_ct, share_rlp_my, share_exp_estimation]
    
    print("-------- Fuel Type Shares Comparison -------")
    print(df)

def test_compare_experian_estimation_data_dollars_per_mile(setup_rlp_data, setup_exp_estimation_data):
    df_rlp, df_estimation = setup_rlp_data
    df_exp_estimation = setup_exp_estimation_data
    df_exp_estimation["fuel"] = df_exp_estimation["fuel"].str.lower()

    df = pd.DataFrame(columns = ["wdpm_rlp", "wdpm_estimation", "wdpm_exp_estimation"])

    for fuel in ["electric", "gasoline", "diesel", "phev", "hybrid"]:
        wdpm_rlp = np.average(df_rlp.loc[df_rlp["fuel"]==fuel, "dollar_per_mile"], weights=df_rlp.loc[df_rlp["fuel"]==fuel, "veh_count"])
        wdpm_estimation = np.average(df_estimation.loc[df_estimation["fuel"]==fuel, "dollar_per_mile"], weights=df_estimation.loc[df_estimation["fuel"]==fuel, "veh_count"])
        wdpm_exp_estimation = np.average(df_exp_estimation.loc[df_exp_estimation["fuel"]==fuel, "dollar_per_mile"], weights=df_exp_estimation.loc[df_exp_estimation["fuel"]==fuel, "agg_count"])
        df.loc[fuel] = [wdpm_rlp, wdpm_estimation, wdpm_exp_estimation]

    df = df.round(2)
    
    print("-------- Weighted Average Dollars per Mile Comparison -------")
    print(df)
    print("Note: During data preparation, dollars per mile was multiplied by 100, hence the difference in the values.")

def test_compare_experian_estimation_data_msrp_fueltype(setup_rlp_data, setup_exp_estimation_data):
    df_rlp, df_estimation = setup_rlp_data
    df_exp_estimation = setup_exp_estimation_data
    df_exp_estimation["fuel"] = df_exp_estimation["fuel"].str.lower()

    df = pd.DataFrame(columns = ["wmsrp_rlp", "wmsrp_estimation", "wmsrp_exp_estimation"])

    for fuel in ["electric", "gasoline", "diesel", "phev", "hybrid"]:
        msrp_rlp = np.average(df_rlp.loc[df_rlp["fuel"]==fuel, "msrp"], weights=df_rlp.loc[df_rlp["fuel"]==fuel, "veh_count"])
        msrp_estimation = np.average(df_estimation.loc[df_estimation["fuel"]==fuel, "msrp"], weights=df_estimation.loc[df_estimation["fuel"]==fuel, "veh_count"])
        msrp_exp_estimation = np.average(df_exp_estimation.loc[df_exp_estimation["fuel"]==fuel, "msrp"], weights=df_exp_estimation.loc[df_exp_estimation["fuel"]==fuel, "agg_count"])
        df.loc[fuel] = [msrp_rlp, msrp_estimation, msrp_exp_estimation]

    # Format df to be 2 decimal places
    df = df.round(2)
    
    print("-------- Weighted Average MSRP Comparison -------")
    print(df)

######################################################################################################
def confirm_model_year_comparability(df1, df2):
    """
    Confirm that the datasets are comparable in terms of model years
    """
    df1_model_years = df1["model_year"].unique().tolist()
    df2_model_years = df2["model_year"].unique().tolist()

    print(f"Model years in the first dataset: {df1_model_years}")
    print(f"Model years in the second dataset: {df2_model_years}")

    assert(np.all([model_year in df2_model_years for model_year in df1_model_years])), "The model years are not the same."

def confirm_make_model_comparability(df1, df2):
    """
    Ensure that all makes and models from df1 and in df2
    """
    df1 = df1[["make", "model"]].drop_duplicates()
    df1["make_model"] = df1["make"] + df1["model"]
    df1_make_models = df1["make_model"].unique().tolist()
    df2 = df2[["make", "model"]].drop_duplicates()
    df2["make_model"] = df2["make"] + df2["model"]
    df2_make_models = df2["make_model"].unique().tolist()

    assert(np.all([make_model in df2_make_models for make_model in df1_make_models])), "The make, model combinations are not the same."

def test_rlp_exp_comparability(setup_rlp_myear_data, setup_exp_estimation_data):
    df_rlp = setup_rlp_myear_data
    df_exp_estimation = setup_exp_estimation_data

    confirm_model_year_comparability(df_rlp, df_exp_estimation)
    confirm_make_model_comparability(df_rlp, df_exp_estimation)

# Latest tests to compare experian data with model year RLP data
def test_XXX(setup_rlp_myear_data, setup_exp_estimation_data):
    df_rlp = setup_rlp_myear_data
    df_exp_estimation = setup_exp_estimation_data

    # Compare the number of sales
    sales_rlp = df_rlp["veh_count"].sum()
    sales_exp_estimation = df_exp_estimation["agg_count"].sum()
    print(f"RLP data sales: {sales_rlp}\tExperian data sales: {sales_exp_estimation}")

    # Compare the share of sales of each fuel type
    fuel_sales_rlp = df_rlp.groupby("fuel")["veh_count"].sum() / sales_rlp
    fuel_sales_exp_estimation = df_exp_estimation.groupby("fuel")["agg_count"].sum() / sales_exp_estimation
    fuel_sales = pd.concat([fuel_sales_rlp, fuel_sales_exp_estimation], axis = 1).rename(columns = {"veh_count": "rlp", "agg_count": "exp_estimation"})
    fuel_sales = fuel_sales.applymap(lambda x: f"{x:.2%}")
    print(fuel_sales)
    print("--------------------------------------")

    # Compare the sales of each fuel type by model year
    fuel_sales_myear_rlp = df_rlp.groupby(["fuel", "model_year"])["veh_count"].sum() / sales_rlp
    fuel_sales_myear_exp_estimation = df_exp_estimation.groupby(["fuel", "model_year"])["agg_count"].sum() / sales_exp_estimation
    fuel_sales_myear = pd.concat([fuel_sales_myear_rlp, fuel_sales_myear_exp_estimation], axis = 1).rename(columns = {"veh_count": "rlp", "agg_count": "exp_estimation"})
    fuel_sales_myear = fuel_sales_myear.dropna()
    fuel_sales_myear = fuel_sales_myear.applymap(lambda x: f"{x:.2%}")
    print(fuel_sales_myear)
    print("--------------------------------------")

    # Compare the weighted average MSRP by fuel type
    msrp_rlp = df_rlp.groupby("fuel")["msrp"].mean()
    msrp_exp_estimation = df_exp_estimation.groupby("fuel")["msrp"].mean()
    msrp = pd.concat([msrp_rlp, msrp_exp_estimation], axis = 1)
    msrp.columns = ["rlp", "exp_estimation"]
    msrp = msrp.applymap(lambda x: f"${x:.2f}")
    print(msrp)


######################################################################################################
# Ignore
def test_est_shares_distribution(setup_rlp_estimation_data, setup_exp_estimation_data):
     """
     Compare the distribution of market shares between the RLP estimation data and the experian estimation data
     """
     df_rlp_est_ct, df_rlp_est_my = setup_rlp_estimation_data
     df_exp_estimation = setup_exp_estimation_data

     df_estimation = df_rlp_est_my
 
     # make the model years comparable
     mask = (df_exp_estimation.model_year >= df_estimation.model_year.min()) & (df_exp_estimation.model_year <= df_estimation.model_year.max())
     df_exp_estimation = df_exp_estimation.loc[mask, :]
 
     # For the rlp estimation data, create a histogram of market shares per market
     def plot_market_shares(df, title):
         import matplotlib.pyplot as plt
         num_markets = df["market_ids"].nunique()
         fig, ax = plt.subplots(nrows = 2, ncols = 4, figsize = (20, 20))
         for i, mkt in enumerate(df["market_ids"].unique().tolist()):
             df.loc[df["market_ids"] == mkt, "shares"].hist(bins=20, ax=ax[i//4, i%4])
             ax[i//4, i%4].set_title(f"Market {mkt}")
         fig.suptitle(title)
         plt.draw()
         plt.show()
     
     plot_market_shares(df_estimation, "Market Shares in the RLP Estimation Data")
     plot_market_shares(df_exp_estimation, "Market Shares in the Experian Estimation Data")



######################################################################################################
# Ignore - replicated above
# def test_compare_vehicle_features(setup_rlp_myear_data, setup_exp_estimation_data):
#     """
#     Test that shows that the underlying vehicle features in fact differ between the datasets.... 
#     """
#     # Show all columns
#     pd.set_option('display.max_columns', None)
# 
#     rlp_data = setup_rlp_myear_data
#     exp_data = setup_exp_estimation_data
# 
#     rlp_data_in = rlp_data.reset_index(drop = True)
#     exp_data_in = exp_data.reset_index(drop = True)
#     exp_data_in = exp_data_in.rename(columns = {"curbwt": "curb_weight"})
# 
#     vars_to_compare = ["make", "model","model_year", "trim", "msrp", "dollar_per_mile", "doors", "curb_weight", "max_hp", "log_hp_weight", "wheelbase", "range_elec"]
# 
#     def get_comparisons_2(rlp_data, exp_data, vars_to_compare):
#         joined = pd.DataFrame()
#         for index, row in tqdm(rlp_data.iterrows()):
#             veh_df = pd.DataFrame(columns = vars_to_compare)
#             rlp_vehicle = row[vars_to_compare]
#             exp_vehicle = exp_data.loc[(exp_data["make"] == rlp_vehicle["make"]) & (exp_data["model"] == rlp_vehicle["model"]) & (exp_data["model_year"] == rlp_vehicle["model_year"]) &(exp_data["trim"]==rlp_vehicle["trim"]), vars_to_compare]
#             rlp_vehicle = pd.DataFrame(rlp_vehicle).T
#             rlp_vehicle = rlp_vehicle.reset_index(drop=True)
#             if len(exp_vehicle) == 0:
#                 exp_vehicle = pd.DataFrame(np.nan, index = [index], columns = vars_to_compare)
#             else:
#                 exp_vehicle = pd.DataFrame(exp_vehicle)
#             exp_vehicle.columns = [f"exp_{col}" for col in exp_vehicle.columns]
#             exp_vehicle = exp_vehicle.reset_index(drop=True)
# 
#             veh = pd.concat([rlp_vehicle, exp_vehicle], axis = 1)
#             joined = pd.concat([joined, veh], axis = 0)
# 
#         # Fill down any missing values in the first 12 columns
#         # joined[vars_to_compare] = joined[vars_to_compare].fillna(method = "ffill")
# 
#         return joined
#     
#     joined_all = get_comparisons_2(rlp_data_in, exp_data_in, vars_to_compare)
#     joined_all.to_csv(str_outputs / "vehicle_comparison_all_unfilled.csv", index = False)