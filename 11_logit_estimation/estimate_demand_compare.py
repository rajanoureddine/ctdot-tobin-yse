############################################################################################################
# Import necessary packages
import numpy as np
import pandas as pd
import pyblp
import statsmodels.api as sm
import time
import os
import pickle
import sys
import pathlib
from scipy.optimize import minimize
import platform
from time import sleep

from linearmodels.iv import IV2SLS # this is to check IV results

from Reference import functions_v2 as exp_functions
import functions_rlp as rlp_functions 

# silence warnings
import warnings
warnings.filterwarnings("ignore")

# Pyblp settings
pyblp.options.verbose = False

############################################################################################################
# Settings
version = "CONNECTICUT"
model = 'logit'
integ = 'halton'
dynamic = False
incl_2021 = True
rlp_market = 'model_year'
# rlp_market = ='county_model_year'

############################################################################################################
# Set up directories
if platform.platform()[0:5] == 'macOS':
    cd = pathlib.Path().resolve().parent
    str_project = cd / "Documents" 
    str_data = str_project / "tobin_working_data"
    str_rlp = str_data / "rlpolk_data"
    str_sales_vin_characteristic = str_rlp / "rlp_with_dollar_per_mile.csv"
    output_folder = str_project / str_data / "outputs"
    str_mapping = str_rlp / "brand_to_oem_generated.csv"
    estimation_test = str_data / "estimation_data_test"
    str_rlp_new = str_rlp / "rlp_with_dollar_per_mile_replaced_myear_20240413_064557_no_lease.csv" # NO LEASES
    # str_rlp_new = str_rlp / "rlp_with_dollar_per_mile_replaced_myear_20240411_183046.csv"

############################################################################################################
# Set up logging
import logging
date_time = time.strftime("%m%d-%H%M")
logging.basicConfig(filename= output_folder / f'estimate_demand_compare_{date_time}.log', level=logging.INFO, format='%(asctime)s - %(message)s')

############################################################################################################
# We prepare the Experian data for estimation
def prepare_experian_data():
    # read in VIN data - file of approx 100000 rows, including car characteristics
    exp_vin_data = exp_functions.read_vin_data(str_project,str_data,version,dynamic)

    # read in census data
    exp_census_data = exp_functions.read_census_data(str_project,str_data,version)

    # haircut first and potentially last market based on missing addl data - alters the census data
    exp_census_data = exp_functions.haircut_interpolate_census(str_project,str_data,exp_census_data,incl_2021)

    # Join
    exp_mkt_data = exp_functions.merge_vin_census(exp_vin_data,exp_census_data,version,dynamic)

    # Various cleaning functions, add a time trend, add a clustering ID for standard errors
    exp_mkt_data = exp_functions.clean_market_data(exp_mkt_data,version)

    # calculate outside good share
    exp_mkt_data = exp_functions.calc_outside_good(exp_mkt_data,version)

    # generate PyBLP instruments
    exp_mkt_data = exp_functions.generate_pyblp_instruments(exp_mkt_data)

    return exp_mkt_data

############################################################################################################
# We now prepare the RLP data, aiming to make it as similar as possible to the Experian data
def prepare_rlp_data(df, mkt_def = "model_year"):
    df["product_ids"] = df.make + "_" + df.model + "_" + df.model_year.astype(str) + "_" + df.trim + "_" + df.fuel + "_" + df.range_elec.astype(str).str[0:3]
    
    if mkt_def == "model_year":
        df["market_ids"] = df["model_year"]
    elif mkt_def == "county_model_year":
        df["market_ids"] = df["county_name"] + "_" + df["model_year"].astype(str)

    # Generate firm IDs and fueltype dummies
    df = rlp_functions.generate_firm_ids(df, str_mapping)
    df = rlp_functions.generate_fuel_type_dummies(df)

    # Read in census data - contains total number of households for each county in Connecticut
    if mkt_def == "model_year":
        census_data = pd.read_csv(str_data / "other_data" / "hhs_by_year.csv")
    elif mkt_def == "county_model_year":
        census_data = pd.read_csv(str_data / "other_data" / "hhs_by_year_counties.csv")

    # Merge the VIN data with the census data
    mkt_data = rlp_functions.merge_vin_census(df, census_data, rlp_market)

    # Clean the market data
    mkt_data = rlp_functions.clean_market_data(mkt_data, rlp_market)

    # Calculate the share of the outside good
    mkt_data = rlp_functions.calc_outside_good(mkt_data, rlp_market)

    # Add instruments 
    mkt_data = rlp_functions.generate_pyblp_instruments(mkt_data)

    return mkt_data

############################################################################################################
exp_mkt_data = prepare_experian_data()
df = pd.read_csv(str_rlp_new)
df_in = df.loc[(df["model_year"]!=2016) & (df["model_year"]!=2023)]
rlp_mkt_data = prepare_rlp_data(df_in)
rlp_mkt_data = rlp_mkt_data.rename(columns = {'log_hp_wt':'log_hp_weight', 'drive_type':'drivetype', 'body_type':'bodytype'})


############################################################################################################
# Final eds
# rlp_mkt_data = rlp_mkt_data[rlp_mkt_data["make"]!="Tesla"]
# exp_mkt_data = exp_mkt_data[exp_mkt_data["make"]!="Tesla"]

# Remove make Polestar from both
rlp_mkt_data = rlp_mkt_data[rlp_mkt_data["make"]!="Polestar"]
exp_mkt_data = exp_mkt_data[exp_mkt_data["make"]!="Polestar"]

# Remove Smart from both
rlp_mkt_data = rlp_mkt_data[rlp_mkt_data["make"]!="Smart"]
exp_mkt_data = exp_mkt_data[exp_mkt_data["make"]!="Smart"]

# Drop Lotus from both
rlp_mkt_data = rlp_mkt_data[rlp_mkt_data["make"]!="Lotus"]
exp_mkt_data = exp_mkt_data[exp_mkt_data["make"]!="Lotus"]

# Drop scion from experian
exp_mkt_data = exp_mkt_data[exp_mkt_data["make"]!="Scion"]

# Drop make Genesis from both
# exp_mkt_data = exp_mkt_data[exp_mkt_data["make"]!="Genesis"]
# rlp_mkt_data = rlp_mkt_data[rlp_mkt_data["make"]!="Genesis"]

# Drop Maserati from both
exp_mkt_data = exp_mkt_data[exp_mkt_data["make"]!="Maserati"]
rlp_mkt_data = rlp_mkt_data[rlp_mkt_data["make"]!="Maserati"]

# Drop the market year 2014 and 2015 from the experian data
# exp_mkt_data = exp_mkt_data[exp_mkt_data["model_year"]!=2014]
# exp_mkt_data = exp_mkt_data[exp_mkt_data["model_year"]!=2015]

# Save
exp_mkt_data.to_csv(estimation_test / f'exp_mkt_data_{date_time}.csv',index = False)
rlp_mkt_data.to_csv(estimation_test / f'mkt_data_{rlp_market}_{date_time}.csv',index = False)


################################################################################
# Get only those make, model, model_year combinations that are in both datasets
if False:
    exp_mkt_data["key"] = exp_mkt_data["make"] + "_" + exp_mkt_data["model"] + "_" + exp_mkt_data["model_year"].astype(str) + "_" + exp_mkt_data["trim"]
    rlp_mkt_data["key"] = rlp_mkt_data["make"] + "_" + rlp_mkt_data["model"] + "_" + rlp_mkt_data["model_year"].astype(str) + "_" + rlp_mkt_data["trim"]

    rlp_mkt_data_short = rlp_mkt_data[rlp_mkt_data["key"].isin(exp_mkt_data["key"])].reset_index(drop=True)
    exp_mkt_data_short = exp_mkt_data[exp_mkt_data["key"].isin(rlp_mkt_data["key"])].reset_index(drop=True)

    rlp_mkt_data_short["log_hp_weight"] = rlp_mkt_data_short[["key"]].merge(exp_mkt_data_short[["key","log_hp_weight"]], on = "key", how = "left")["log_hp_weight"]

    rlp_mkt_data = rlp_mkt_data_short.copy()
    exp_mkt_data = exp_mkt_data_short.copy()

    # Save
    exp_mkt_data.to_csv(estimation_test / f'exp_mkt_data_{date_time}.csv',index = False)
    rlp_mkt_data.to_csv(estimation_test / f'mkt_data_{rlp_market}_{date_time}.csv',index = False)




####################
# run PyBLP models #
####################
exp_mkt_data_keep = exp_mkt_data.copy()
rlp_mkt_data_keep = rlp_mkt_data.copy()

output_logit = pd.DataFrame()
output_ols = pd.DataFrame()

params_master = ['prices', 'dollar_per_mile', 'electric', 'phev', 'hybrid', 'diesel', 'log_hp_weight', 'wheelbase', 'doors', 'range_elec', 'make', 'drivetype', 'bodytype']
# params_master = ['prices', 'electric', 'phev', 'hybrid', 'diesel', 'log_hp_weight', 'wheelbase', 'doors', 'range_elec', 'make', 'drivetype', 'bodytype']
specifications = [1, 2, 6, 7, 8, 9, 10]
# specifications = [1, 5, 6, 7, 8, 9]

if model == 'logit':
    for j in specifications:
        output_specification_logit = pd.DataFrame()
        output_specification_ols = pd.DataFrame()
        for i, mkt_data in enumerate([exp_mkt_data, rlp_mkt_data]):
            if i == 0:
                data_source = "Experian"
                print(f"Running {data_source} data")
            elif i == 1:
                data_source = "RLP"
                print(f"Running {data_source} data")
            

            # Prepare the Logit specification
            params_logit = params_master[0:j] + [f'C({param})' for param in params_master[-3:]]
            params_ols = params_master[0:j] + [param for param in params_master[-3:]]
            params_str = ' + '.join(params_logit)
            params_str = '0 + ' + params_str
            logit_formulation = pyblp.Formulation(params_str)

            # Log the datasource and specification
            logging.info(f"Data Source: {data_source}")
            logging.info(f"Specification: {params_str}")

            # Run the logit problem
            problem = pyblp.Problem(logit_formulation, mkt_data)
            logit_results_price_updated = problem.solve()

            # Prepare the OLS specification and Convert make, drivetype, and bodytype to dummies
            X = mkt_data[params_ols]
            X = pd.get_dummies(X, columns = ['make', 'drivetype', 'bodytype'], drop_first = True)
            Y = mkt_data['shares']
            X = sm.add_constant(X)
            ols_results = sm.OLS(Y.astype(float), X.astype(float)).fit()

            # save results
            df_logit = pd.DataFrame({'specification':j,
                                'data_source':data_source,
                                'param':logit_results_price_updated.beta_labels,
                                'value':logit_results_price_updated.beta.flatten(),
                                'se': logit_results_price_updated.beta_se.flatten()})
            
            df_ols = pd.DataFrame({'specification':j,
                                'data_source':data_source,
                                'param':ols_results.params.index,
                                'value':ols_results.params.values,
                                'se': ols_results.bse.values})
            
            output_specification_logit = pd.concat([output_specification_logit, df_logit], axis = 1)
            output_specification_ols = pd.concat([output_specification_ols, df_ols], axis = 1)

        output_logit = pd.concat([output_logit, output_specification_logit], axis = 0)
        output_ols = pd.concat([output_ols, output_specification_ols], axis = 0)

output_logit.to_csv(output_folder / f'comparison_outputs_logit_{rlp_market}_{date_time}.csv',index = False)
output_ols.to_csv(output_folder / f'comparison_outputs_ols_{rlp_market}_{date_time}.csv',index = False)
