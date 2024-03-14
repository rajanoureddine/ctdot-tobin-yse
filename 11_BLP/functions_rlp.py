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

    # Get rid of aggregate figure
    df_census = df_census.loc[~df_census["county"].str.contains("CONNECTICUT"), :]

    return df_census

def aggregate_to_market(df, market_ids, product_ids, sales_col):
    """
    Aggregate the data to the market level.
    """
    # Get unique product market combos
    unique_product_market_combos = len(df[[market_ids, product_ids]].drop_duplicates())
    unique_products = df[product_ids].nunique()
    print(f"Aggregating to the market level, with {unique_product_market_combos} unique product market combos")
    print(f"There are {unique_products} unique products")

    # Get sales at the market level
    sales_per_market = df[[market_ids, product_ids, sales_col]].groupby([market_ids, product_ids]).agg({sales_col: "sum"}).reset_index()

    # Merge back in the vehicle characteristics
    df = df.drop_duplicates(subset = [product_ids])
    df = df.drop(columns = ["report_year", "report_month"]+[market_ids,sales_col])
    sales_per_market = sales_per_market.merge(df, on = [product_ids], how = "left")

    # Check
    assert(unique_product_market_combos == len(sales_per_market)), "Aggregation did not work correctly"

    return sales_per_market

def drop_uncommon_products(df):
    """
    Drops products that do not occur in a large number of counties across the sample.
    """
    # Get number of unique counties
    unique_counties = df["county_name"].nunique()

    # Get the number of counties each product occurs in
    vins_counties = df[["vin_pattern", "county_name"]].drop_duplicates()
    vins_counties = vins_counties.groupby("vin_pattern").size().reset_index(name = 'count').sort_values(by = "count", ascending = False)

    # Mark the products that occur in more than 7 counties
    vins_counties["keep"] = vins_counties["count"] >= 7

    # Record number in less than 7 counties
    uncommon_products = vins_counties.loc[vins_counties["count"] < 7, "vin_pattern"].nunique()
    common_products = vins_counties.loc[vins_counties["count"] >= 7, "vin_pattern"].nunique()
    print(f"Found {common_products} / {uncommon_products + common_products} products that occur in at least 7 counties")
    
    # Merge back in
    df = df.merge(vins_counties[["vin_pattern", "keep"]], on = "vin_pattern", how = "left")
    num_sales_dropped = df[df["keep"] == False]["veh_count"].sum()
    print(f"Dropping {uncommon_products} products and {num_sales_dropped} sales for VINs observed in less than 7 counties.")

    df = df.loc[df["keep"],:]
    df = df.drop(columns = ["keep"])

    return df

def merge_vin_census(vin_data,census_data):
    """
    Merge the VIN data with the census data.
    """
    # Merge the VIN data with the census data
    print("Merging the VIN data with the census data - we merge on county only")
    df = vin_data.merge(census_data, left_on = "county_name", right_on = "county", how = "left")
    assert(len(df) == len(vin_data)), "Merge did not match the correct number of rows"
    assert(df["tot_HH"].isnull().sum() == 0), "Merge did not match all rows"

    # Drop the extra column from the census data
    df = df.drop(columns = ["county"])
    
    return df

def clean_market_data(mkt_data):
    # Save original length
    orig_len = len(mkt_data)

    # drop observations with missing data
    mkt_data = mkt_data.dropna(subset=['tot_HH','dollar_per_mile','curb_weight','drive_type']).reset_index(drop=True)

    # Create product IDs
    mkt_data['product_ids'] = mkt_data['vin_pattern']

    # shift dollar per mile to dollar/100 mile
    mkt_data.dollar_per_mile*=100

    # drop observations with MSRP > $120K
    mkt_data_luxury = mkt_data.loc[mkt_data.msrp > 120]
    mkt_data = mkt_data.loc[mkt_data.msrp <= 120]

    # drop observations with MSRP = 0
    mkt_data_zeros = mkt_data.loc[mkt_data.msrp == 0]
    mkt_data = mkt_data.loc[mkt_data.msrp != 0]

    # drop observations with wheelbase = 0
    mkt_data = mkt_data.loc[mkt_data.wheelbase > 0]

    # drop observations from certain luxury brands
    mkt_data = mkt_data[~mkt_data.make.isin(['Aston Martin'])]

    # calculate each vehicle share
    mkt_data['shares'] = mkt_data['veh_count']/mkt_data.tot_HH

    # drop observations with market share below the 5th percentile
    mkt_data = mkt_data.loc[mkt_data.shares > np.percentile(mkt_data.shares,5)].reset_index(drop = True)

    # Create market id for pyblp
    mkt_data["market_ids"] = mkt_data["county_name"]

    # Get prices 
    mkt_data['prices'] = mkt_data.msrp

    # mkt_data['time_trend'] = mkt_data.model_year - 2013

    # define ice indicator
    mkt_data['ice'] = 1
    mkt_data.loc[mkt_data.fuel.isin(['electric','PHEV']),'ice'] = 0

    # define ice mpg
    if False:
        mkt_data['mpg_ice'] = 0
        mkt_data.loc[mkt_data.ice == 1,'mpg_ice'] = mkt_data.loc[mkt_data.ice == 1,'combined_mpg2008']
        mkt_data['gpm_ice'] = 0
        mkt_data.loc[mkt_data.ice == 1,'gpm_ice'] = 1/mkt_data.loc[mkt_data.ice == 1,'combined_mpg2008']
        mkt_data['log_mpg_ice'] = 0
        mkt_data.loc[mkt_data.ice == 1,'log_mpg_ice'] = np.log(mkt_data.loc[mkt_data.ice == 1,'combined_mpg2008'])

    # add clustering id so standard errors can be clustered at product level
    mkt_data['clustering_ids'] = mkt_data.product_ids

    # set prices = price - subsidy
    mkt_data.prices = mkt_data.msrp - mkt_data.fed_credit

    return mkt_data

def calc_outside_good(mkt_data):
    outside_good = mkt_data[['county_name','shares']].groupby(['county_name']).sum().reset_index()
    outside_good['outside_share'] = 1 - outside_good.shares
    mkt_data = pd.merge(mkt_data,outside_good[['county_name','outside_share']],how='left',on=['county_name'])

    return mkt_data

def generate_pyblp_instruments(mkt_data):
    # generate instruments at national level!
    # Previously: 'product_ids','firm_ids','model_year','wheelbase','curbwt','doors','log_hp_weight','drivetype','bodytype','wages'
    instrument_data = mkt_data[['product_ids', 'market_ids','model_year','wheelbase','curb_weight','drive_type','body_type','wages']].drop_duplicates().reset_index(drop=True)

    # improved instruments
    # separate differentiation instruments (continuous chars) AND discrete instrument (for each product, count own- and rival- products with same values)
    demand_instruments_continuous1 = pyblp.build_differentiation_instruments(pyblp.Formulation('0 + wheelbase + curb_weight'), instrument_data)
    demand_instruments_continuous2 = pyblp.build_differentiation_instruments(pyblp.Formulation('0 + wheelbase + curb_weight'), instrument_data, version='quadratic')
    # discrete instruments
    lst_discrete_chars = ['drive_type','body_type']
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

    # wage instruments
    demand_instruments_wage = np.reshape(np.array(instrument_data.wages),(-1,1))
    # combine discrete and continue instruments into a single array
    demand_instruments = np.concatenate((demand_instruments_continuous1,demand_instruments_continuous2,demand_instruments_discrete,demand_instruments_wage),axis=1)
    #demand_instruments = np.concatenate((demand_instruments_continuous1,demand_instruments_continuous2,demand_instruments_discrete),axis=1)

    # add instruments back into data
    col_names = ['demand_instruments' + str(x) for x in range(0,demand_instruments.shape[1])]
    instrument_data = pd.concat([instrument_data,
                                pd.DataFrame(demand_instruments,
                                            columns=col_names)
                                ],axis=1)
    # merge instruments back
    mkt_data = pd.merge(mkt_data,instrument_data[col_names +['product_ids']],
                        how='left',on='product_ids')

    # add rebate to instrument set
    #mkt_data['demand_instruments'+str(demand_instruments.shape[1]+1)] = mkt_data.rebate
    return mkt_data

