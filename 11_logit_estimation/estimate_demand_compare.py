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

exp_mkt_data = prepare_experian_data()
exp_mkt_data.to_csv(estimation_test / f'exp_mkt_data.csv',index = False)

# Get the min and max model years for the Experian data
exp_min_model_year = exp_mkt_data.model_year.min()
exp_max_model_year = exp_mkt_data.model_year.max()

# Get the makes, models, and trims for the Experian data
exp_makes = exp_mkt_data.make.unique()
exp_models = exp_mkt_data.model.unique()
exp_trims = exp_mkt_data.trim.unique()

############################################################################################################
# We now prepare the RLP data, aiming to make it as similar as possible to the Experian data
def prepare_rlp_data(product_ids = ['make', 'model', 'trim'], match_model_years = False, match_vehicles = False, drop_uncommon = True):
    # read in VIN data - file of approx 100000 rows, including car characteristics
    vin_data = pd.read_csv(str_sales_vin_characteristic)
    orig_vin_data = vin_data.copy()

    # Create product_id column
    vin_data["product_ids"] = vin_data[product_ids].apply(lambda x: '_'.join(x), axis = 1)

    # Drop unnecessary columns
    unneeded_cols = ['vin_pattern', 'fuel_type', 'transaction_price', 'sp_id', 'sp_make', 'sp_model', 'sp_model_year', 'zip_code' , 'vin_orig', 'vehicle_id', 'plant']
    vin_data = vin_data.drop(columns = unneeded_cols)

    if match_model_years:
        # Get only the model years that are in the Experian data - also drop 2016
        vin_data = vin_data[(vin_data.model_year >= exp_min_model_year) & (vin_data.model_year <= exp_max_model_year) & (vin_data.model_year != 2016)]

    # Match the makes and models across datasets
    if match_vehicles:
        vin_data = rlp_functions.match_makes_models(vin_data, exp_mkt_data)

    # Different ways to remove uncommon products.
    # - If using counties, then drop those products that are not in at least 7 counties
    # - Drop those make, models, and trims that are not in at least 5 model years
    if drop_uncommon and rlp_market != 'model_year':
        vin_data = rlp_functions.drop_uncommon_products(vin_data, rlp_market, 7)
    elif drop_uncommon and rlp_market == 'model_year':
        vin_data = rlp_functions.normalize_markets(vin_data, rlp_market, num = 5)

    orig_sales = vin_data.veh_count.sum()
    orig_products = vin_data.product_ids.nunique()

    # Get firm ids
    vin_data = rlp_functions.generate_firm_ids(vin_data, str_mapping)
    assert(orig_sales == vin_data.veh_count.sum()), "Sales should not change after dropping uncommon products"
    assert(orig_products == vin_data.product_ids.nunique()), "Products should not change after dropping uncommon products"

    # Get dummies for electric, phev, hybrid, diesel
    vin_data = rlp_functions.generate_fuel_type_dummies(vin_data)
    assert(orig_sales == vin_data.veh_count.sum()), "Sales should not change after generating fuel type dummies"
    assert(orig_products == vin_data.product_ids.nunique()), "Products should not change after generating fuel type dummies"

    # Aggregate to market level
    vin_data_aggregated = rlp_functions.aggregate_to_market(vin_data, rlp_market, "product_ids", "veh_count")
    assert(orig_sales == vin_data_aggregated.veh_count.sum()), "Sales should not change after aggregating to market level"
    assert(orig_products == vin_data_aggregated.product_ids.nunique()), "Products should not change after aggregating to market level"

    # Read in census data - contains total number of households for each county in Connecticut
    census_data = rlp_functions.read_census_data(str_data / "other_data" / "total-households-county-2019.csv")

    # Merge the VIN data with the census data
    mkt_data = rlp_functions.merge_vin_census(vin_data_aggregated, census_data, rlp_market)
    assert(orig_sales == mkt_data.veh_count.sum()), "Sales should not change after merging with census data"
    assert(orig_products == mkt_data.product_ids.nunique()), "Products should not change after merging with census data"

    # Clean the market data
    mkt_data = rlp_functions.clean_market_data(mkt_data, rlp_market)
    orig_sales = mkt_data.veh_count.sum()

    # Calculate the share of the outside good
    mkt_data = rlp_functions.calc_outside_good(mkt_data, rlp_market)
    assert(orig_sales == mkt_data.veh_count.sum()), "Sales should not change after calculating the share of the outside good"
    # assert(orig_products == mkt_data.vin_pattern.nunique()), "Products should not change after calculating the share of the outside good"

    # Add instruments 
    mkt_data = rlp_functions.generate_pyblp_instruments(mkt_data)
    # assert(orig_sales == mkt_data.veh_count.sum()), "Sales should not change after generating PyBLP instruments"

    return mkt_data

rlp_mkt_data = prepare_rlp_data(match_vehicles=True)
rlp_mkt_data = rlp_mkt_data.rename(columns = {'log_hp_wt':'log_hp_weight', 'drive_type':'drivetype', 'body_type':'bodytype'})
rlp_mkt_data.to_csv(estimation_test / f'mkt_data_{rlp_market}.csv',index = False)

############################################################################################################

####################
# run PyBLP models #
####################
exp_mkt_data_keep = exp_mkt_data.copy()
rlp_mkt_data_keep = rlp_mkt_data.copy()

output_logit = pd.DataFrame()
output_ols = pd.DataFrame()

params_master = ['prices', 'dollar_per_mile', 'electric', 'phev', 'hybrid', 'diesel', 'log_hp_weight', 'wheelbase', 'doors', 'range_elec', 'make', 'drivetype', 'bodytype']
specifications = [1, 2, 6, 7, 8, 9, 10]

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

output_logit.to_csv(output_folder / f'comparison_outputs_logit_{rlp_market}.csv',index = False)
output_ols.to_csv(output_folder / f'comparison_outputs_ols_{rlp_market}.csv',index = False)
