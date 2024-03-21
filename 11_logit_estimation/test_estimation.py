####################################################################################################
# Test the aggregated data the estimation
# Mostly sense-checks, nothing concrete
####################################################################################################
# Import required packages
import pytest
import numpy as np
import pathlib
import pandas as pd

# silence warnings
import warnings
warnings.filterwarnings("ignore")


####################################################################################################
# Set directories
str_cwd = pathlib.Path().resolve().parent.parent
str_dir = str_cwd / "Documents" / "tobin_working_data"
str_rlp_data = str_dir / "rlpolk_data"
rlp_data_file = "rlp_with_dollar_per_mile.csv"
estimation_data = str_dir / "estimation_data_test" / "mkt_data.csv"
exp_estimation_data = str_dir / "estimation_data_test" / "experian_mkt_data.csv"
experian_data = str_dir / "intermediate" / "US_VIN_data_common.csv"

# Import the raw and aggregateed data
df_rlp = pd.read_csv(str_rlp_data / rlp_data_file)
df_estimation = pd.read_csv(estimation_data)

####################################################################################################
@pytest.fixture
def setup_data():
    df_rlp = pd.read_csv(str_rlp_data / rlp_data_file)
    df_estimation = pd.read_csv(estimation_data)
    return df_rlp, df_estimation

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
    # df_exp_estimation = df_exp_estimation.loc[df_exp_estimation["state"] == "CONNECTICUT", :]
    df_exp_estimation = df_exp_estimation.loc[df_exp_estimation.msrp.notna(), :]
    df_exp_estimation = df_exp_estimation.loc[df_exp_estimation.agg_count.notna(), :]
    df_exp_estimation = df_exp_estimation.loc[df_exp_estimation.dollar_per_mile.notna(), :]
    return df_exp_estimation


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
    for market in df_estimation["market_ids"].unique():
        print(f"-----------------{market}-----------------")
        df_market = df_estimation.loc[df_estimation["market_ids"]==market, ["market_ids", "make", "model", "model_year", "shares"]].sort_values("shares", ascending=False)
        print(df_market.head(5))

def test_average_msrp():
    for fuel in ["electric", "gasoline", "diesel", "phev", "hybrid"]:
        msrp_raw = df_rlp.loc[df_rlp["fuel"]==fuel, "msrp"].mean()
        msrp_estimation = df_estimation.loc[df_estimation["fuel"]==fuel, "msrp"].mean()

        print(f"-------------------------------------------------")
        print(f"The average MSRP for {fuel} in the raw data is ${msrp_raw:.2f}")
        print(f"The average MSRP for {fuel} in the estimation data is ${msrp_estimation:.2f}")
        
        # Check the MSRP
        assert np.isclose(msrp_raw, msrp_estimation, atol=5, rtol = 0.1), f"The average MSRP for {fuel} is not as expected."

def test_weighted_average_msrp():
    for fuel in ["electric", "gasoline", "diesel", "phev", "hybrid"]:
        msrp_raw = np.average(df_rlp.loc[df_rlp["fuel"]==fuel, "msrp"], weights=df_rlp.loc[df_rlp["fuel"]==fuel, "veh_count"])
        msrp_estimation = np.average(df_estimation.loc[df_estimation["fuel"]==fuel, "msrp"], weights=df_estimation.loc[df_estimation["fuel"]==fuel, "veh_count"])

        print(f"-------------------------------------------------")
        print(f"The weighted average MSRP for {fuel} in the raw data is ${msrp_raw:.2f}")
        print(f"The weighted average MSRP for {fuel} in the estimation data is ${msrp_estimation:.2f}")
        
        # Check the MSRP
        assert np.isclose(msrp_raw, msrp_estimation, atol=5, rtol = 0.1), f"The weighted average MSRP for {fuel} is not as expected."

def test_compare_weighted_average_msrp():
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
    df_cp = df_estimation.copy()

    mkts_products = df_cp[["market_ids", "product_ids"]].drop_duplicates()

    # count how many markets each product is available in
    mkt_counts = mkts_products.groupby("product_ids").count()
    mkt_counts = mkt_counts.reset_index().rename(columns = {"market_ids": "count"})
    mkt_counts = pd.DataFrame(mkt_counts.groupby("count").size()).reset_index().rename(columns = {0: "num_products"})
    print(mkt_counts)
    print(mkt_counts["num_products"].sum())

def test_compare_experian_msrp(setup_experian_data):
    df_experian = setup_experian_data

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


def test_compare_experian_msrp_fueltype(setup_data, setup_experian_data):
    df_rlp, df_estimation = setup_data
    df_experian = setup_experian_data
    df_experian["fuel"] = df_experian["fuel"].str.lower()

    df = pd.DataFrame(columns = ["wmsrp_experian", "wmsrp_rlp", "wmsrp_estimation"])

    for fuel in ["electric", "gasoline", "diesel", "phev", "hybrid"]:
        msrp_experian = np.average(df_experian.loc[df_experian["fuel"]==fuel, "msrp"], weights=df_experian.loc[df_experian["fuel"]==fuel, "agg_count"])
        msrp_rlp = np.average(df_rlp.loc[df_rlp["fuel"]==fuel, "msrp"], weights=df_rlp.loc[df_rlp["fuel"]==fuel, "veh_count"])
        msrp_estimation = np.average(df_estimation.loc[df_estimation["fuel"]==fuel, "msrp"], weights=df_estimation.loc[df_estimation["fuel"]==fuel, "veh_count"])
        df.loc[fuel] = [msrp_experian, msrp_rlp, msrp_estimation]
    
    print("-------- Weighted Average MSRP Comparison -------")
    print(df)

def test_compare_experian_dollars_per_mile_fueltype(setup_data, setup_experian_data):
    df_rlp, df_estimation = setup_data
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



#######################################################################################################
# Test to compare to the market data used in the experian estimation 

def test_compare_experian_estimation_data_msrp(setup_data, setup_exp_estimation_data):
    df_rlp, df_estimation = setup_data
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


def test_compare_experian_estimation_data_fueltype_shares(setup_data, setup_exp_estimation_data):
    df_rlp, df_estimation = setup_data
    df_exp_estimation = setup_exp_estimation_data
    df_exp_estimation["fuel"] = df_exp_estimation["fuel"].str.lower()

    df = pd.DataFrame(columns = ["share_rlp", "share_estimation", "share_exp_estimation"])

    for fuel in ["electric", "gasoline", "diesel", "phev", "hybrid"]:
        share_rlp = df_rlp.loc[df_rlp["fuel"]==fuel, "veh_count"].sum() / df_rlp["veh_count"].sum()
        share_estimation = df_estimation.loc[df_estimation["fuel"]==fuel, "veh_count"].sum() / df_estimation["veh_count"].sum()
        share_exp_estimation = df_exp_estimation.loc[df_exp_estimation["fuel"]==fuel, "agg_count"].sum() / df_exp_estimation["agg_count"].sum()
        df.loc[fuel] = [share_rlp, share_estimation, share_exp_estimation]
    
    print("-------- Fuel Type Shares Comparison -------")
    print(df)

def test_compare_experian_estimation_data_dollars_per_mile(setup_data, setup_exp_estimation_data):
    df_rlp, df_estimation = setup_data
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


def test_compare_experian_estimation_data_msrp_fueltype(setup_data, setup_exp_estimation_data):
    df_rlp, df_estimation = setup_data
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
