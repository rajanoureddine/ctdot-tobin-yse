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
import pyblp

####################################################################################################
def read_census_data(path):
    """
    Read in the census data from the given path.
    """
    # Read in the census data
    print(f"Reading in the census data from {path}")
    df_census = pd.read_csv(path)

    # Clean up
    df_census.columns = df_census.columns.str.lower()
    df_census = df_census.loc[df_census["variable"]=="Total Households", :]
    df_census = df_census.loc[df_census["year"]=="2015-2019", :]
    df_census.rename(columns = {"value": "tot_HH"}, inplace = True)
    df_census = df_census[["county", "tot_HH"]]

    # Clean up the county names
    df_census["county"] = df_census["county"].str.replace(" County", "")
    df_census["county"] = df_census["county"].str.upper()

    return df_census


def merge_vin_census(vin_data,census_data, mkt_ids = "model_year"):
    """
    Merge the VIN data with the census data.
    """
    # Set fake model year for 2023
    vin_data["fmy"] = vin_data["model_year"]
    vin_data.loc[vin_data["model_year"]==2023, "fmy"] = 2022

    # For the counties, 2022 additionally cannot be matched so we use 2021
    if mkt_ids == "county_model_year":
        vin_data.loc[vin_data["model_year"]==2022, "fmy"] = 2021
        vin_data.loc[vin_data["model_year"]==2023, "fmy"] = 2021

    if mkt_ids == "model_year":
        output = vin_data.merge(census_data, left_on = "fmy", right_on = "model_year", how = "left")
    if mkt_ids == "county_model_year":
        output = vin_data.merge(census_data, left_on = ["fmy", "county_name"], right_on = ["model_year", "county_name"], how = "left")

    # Clean up
    output = output.drop(columns = ["fmy", "model_year_y"])
    output = output.rename(columns = {"model_year_x": "model_year"})
    
    # Check the merge worked correctly
    assert(len(output) == len(vin_data)), "Merge did not match the correct number of rows"
    assert(output.veh_count.sum() == vin_data.veh_count.sum()), "Merge affected the number of sales"
    assert(output["tot_HH"].isnull().sum() == 0), "Merge did not match all rows"

    return output


def clean_market_data(mkt_data, mkt_ids):
    # Save original length
    orig_len = len(mkt_data)

    # drop observations with missing data
    mkt_data = mkt_data.dropna(subset=['tot_HH','dollar_per_mile','curb_weight','drive_type']).reset_index(drop=True)

    # shift dollar per mile to dollar/100 mile
    mkt_data.dollar_per_mile*=100

    # drop observations with MSRP > $120K
    mkt_data_luxury = mkt_data.loc[mkt_data.msrp > 120]
    mkt_data = mkt_data.loc[mkt_data.msrp <= 120]
    print(f"Dropped {len(mkt_data_luxury)} observations with MSRP > $120K")

    # drop observations with MSRP = 0
    mkt_data_zeros = mkt_data.loc[mkt_data.msrp == 0]
    mkt_data = mkt_data.loc[mkt_data.msrp != 0]

    # drop observations with wheelbase = 0
    mkt_data = mkt_data.loc[mkt_data.wheelbase > 0]

    # drop observations from certain luxury brands
    mkt_data = mkt_data[~mkt_data.make.isin(['Aston Martin'])]

    # calculate each vehicle share
    mkt_data['shares'] = mkt_data['veh_count']/mkt_data.tot_HH

    # Get prices 
    print("Calculating prices")
    mkt_data['prices'] = mkt_data.msrp - mkt_data.fed_credit - mkt_data.state_incentive

    # define ice indicator
    mkt_data['ice'] = 1
    mkt_data.loc[mkt_data.fuel.isin(['electric','PHEV']),'ice'] = 0

    # add clustering id so standard errors can be clustered at product level
    mkt_data['clustering_ids'] = mkt_data.product_ids


    return mkt_data

def calc_outside_good(mkt_data):
    outside_good = mkt_data[["market_ids",'shares']].groupby(["market_ids"]).sum().reset_index()
    outside_good['outside_share'] = 1 - outside_good.shares
    mkt_data = pd.merge(mkt_data,outside_good[["market_ids",'outside_share']],how='left',on="market_ids")

    return mkt_data

def generate_firm_ids(mkt_data, str_mapping):
    # Get the oem to brand mapping
    oem_to_brand = pd.read_csv(str_mapping)
    oem_to_brand["matched"] = 1

    # Get the details we use for ram
    ram_row = oem_to_brand[oem_to_brand["make"]=="Dodge"]

    # Merge in the market data 
    len_mkt = len(mkt_data)
    mkt_data = mkt_data.merge(oem_to_brand, on = "make", how = "left")
    assert(len_mkt == len(mkt_data)), "length has changed"

    # Now fix it for Ram
    mkt_data.loc[(mkt_data["matched"].isna()) | (mkt_data["make"].str.contains("Ram")), "manufacturer_policy"] = "Fiat Chrysler Automobiles"
    mkt_data.loc[(mkt_data["matched"].isna()) | (mkt_data["make"].str.contains("Ram")), "firm_ids"] = 2
    assert(sum(mkt_data["manufacturer_policy"].isna()) == 0)

    # Drop unecesssary
    mkt_data = mkt_data.drop("matched", axis =1)
    assert(len_mkt == len(mkt_data)), "length has changed"

    return mkt_data

def generate_fuel_type_dummies(vin_data):
    # There is no need to gasoline category as that is the base category
    vin_data["electric"] = 0
    vin_data["phev"] = 0
    vin_data["hybrid"] = 0
    vin_data["diesel"] = 0

    vin_data.loc[vin_data["fuel"].str.contains("electric"), "electric"] = 1
    vin_data.loc[vin_data["fuel"].str.contains("phev"), "phev"] = 1
    vin_data.loc[vin_data["fuel"].str.contains("hybrid"), "hybrid"] = 1
    vin_data.loc[vin_data["fuel"].str.contains("diesel"), "diesel"] = 1

    # Checks
    assert(sum(vin_data["electric"]) > 0)
    assert(sum(vin_data["phev"]) > 0)
    assert(sum(vin_data["hybrid"]) > 0)
    # assert(sum(vin_data["diesel"]) > 0)

    check = vin_data.loc[(vin_data.electric == 0) & (vin_data.phev == 0) & (vin_data.hybrid == 0) & (vin_data.diesel == 0), "fuel"].unique()
    print(check)

    return vin_data


def generate_pyblp_instruments(mkt_data):
    # generate instruments at national level!
    instrument_data = mkt_data[['product_ids','firm_ids', 'doors', 'model_year','log_hp_wt','wheelbase','curb_weight','drive_type','body_type', "market_ids"]].drop_duplicates().reset_index(drop=True)
    # instrument_data['market_ids'] =  instrument_data.model_year

    # improved instruments
    # separate differentiation instruments (continuous chars) AND discrete instrument (for each product, count own- and rival- products with same values)
    demand_instruments_continuous1 = pyblp.build_differentiation_instruments(pyblp.Formulation('0 + wheelbase + curb_weight'), instrument_data)
    demand_instruments_continuous2 = pyblp.build_differentiation_instruments(pyblp.Formulation('0 + wheelbase + curb_weight'), instrument_data, version='quadratic')

    # discrete instruments
    lst_discrete_chars = ['drive_type', 'doors', 'body_type']
    data_small = instrument_data[['market_ids','firm_ids'] + lst_discrete_chars]
    count_unique = data_small.groupby(['market_ids','firm_ids'] + lst_discrete_chars).size().reset_index(name = 'ct')
    # for each vehicle, count of rival vehicles with same discrete chars in same market
    count_rival_all = pd.DataFrame()
    for firm in data_small.firm_ids.unique():
        count_unique_diff = count_unique.loc[count_unique.firm_ids != firm]
        count_unique_diff = count_unique_diff.groupby(['market_ids'] + lst_discrete_chars).sum().reset_index()
        count_unique_diff = count_unique_diff.rename(columns={'ct':'ct_rival'})
        count_unique_diff['firm_ids'] = firm
        count_rival_all = pd.concat([count_rival_all,count_unique_diff])

    iv1 = pd.merge(data_small,count_rival_all,how='left',on=['market_ids','firm_ids'] + lst_discrete_chars)
    iv1.loc[iv1.ct_rival.isna(),'ct_rival'] = 0

    # for each vehicle, count of non-rival vehicles with same discrete chars in same market
    count_unique_same = count_unique.copy()
    count_unique_same['ct_same'] = count_unique_same.ct -1

    iv2 = pd.merge(data_small,count_unique_same,how='left',on=['market_ids','firm_ids'] + lst_discrete_chars)

    demand_instruments_discrete = np.array([iv1.ct_rival.values,iv2.ct_same.values]).T

    # combine discrete and continue instruments into a single array
    demand_instruments = np.concatenate((demand_instruments_continuous1,demand_instruments_continuous2,demand_instruments_discrete),axis=1)
    #demand_instruments = np.concatenate((demand_instruments_continuous1,demand_instruments_continuous2,demand_instruments_discrete),axis=1)

    # add instruments back into data
    col_names = ['demand_instruments' + str(x) for x in range(0,demand_instruments.shape[1])]
    instrument_data = pd.concat([instrument_data,
                                pd.DataFrame(demand_instruments,
                                            columns=col_names)
                                ],axis=1)
    # merge instruments back
    mkt_data = pd.merge(mkt_data,instrument_data[col_names +['product_ids', 'market_ids']],
                        how='left',on=['product_ids', 'market_ids'])
    

    return mkt_data

