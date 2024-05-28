############################################################################################################
# Import necessary packages
import numpy as np
import pandas as pd
# import censusdata
import pyblp
import statsmodels.api as sm
import time
import pickle
import sys
import pathlib
from scipy.optimize import minimize
import platform

from linearmodels.iv import IV2SLS # this is to check IV results

from functions_rlp import * # this is the set of additional functions

pyblp.options.verbose = True

############################################################################################################
# Settings 
version = "CONNECTICUT"

# print('version: '+version)
# print(pyblp.__version__)

# dynamic data or not
#dynamic = True
dynamic = False

# model
#model = 'DGV'
model = 'logit'
#model = 'rc'
#model = 'rc_demo'
#model = 'rc_demo_moments'
#model = 'nested_logit'
#model = 'rc_nl'
#model = 'rc_nl_moments'

# number of iterations
#gmm_rounds = '1s'
gmm_rounds = '2s'

# set number of simulated agents per market
# does not apply if using gaussian quadrature, but affects remaining integration approaches
n_agent = 5000

# include 2021 data
incl_2021 = True

# re-scale income moments
bln_rescale_income = False
bln_truncate_income = True

# integration setting
#integ = 'monte_carlo'
integ = 'halton'
#integ = 'gauss'
#integ = 'lhs'
#integ = 'mlhs'
# print("Integration setting:" + integ)

# split into 3 time periods
#split = 't1'
#split = 't2'
#split = 't3'
split = 'None'
#split = 'None, ann. moments' # full time period, separate moments by year
#split = 'geog'
save = True

############################################################################################################
# set up directories
if platform.platform()[0:5] == 'macOS':
    cd = pathlib.Path().resolve().parent
    str_project = cd / "Documents" 
    str_data = str_project / "tobin_working_data"
    str_rlp = str_data / "rlpolk_data"
    str_mapping = str_rlp / "brand_to_oem_generated.csv"
    str_sales_vin_characteristic = str_rlp / "rlp_with_dollar_per_mile.csv"
    output_folder = str_project / str_data / "outputs"
    estimation_test = str_data / "estimation_data_test"

############################################################################################################
# Sources
# Census data: http://data.ctdata.org/dataset/total-households-by-county

############################################################################################################    
# read in VIN data - file of approx 100000 rows, including car characteristics
vin_data = pd.read_csv(str_sales_vin_characteristic)
vin_data_original = vin_data.copy()

# Drop uncommon products
vin_data = drop_uncommon_products(vin_data)
orig_sales = vin_data.veh_count.sum()
orig_products = vin_data.vin_pattern.nunique()

# Get firm ids
vin_data = generate_firm_ids(vin_data, str_mapping)
assert(orig_sales == vin_data.veh_count.sum()), "Sales should not change after dropping uncommon products"
assert(orig_products == vin_data.vin_pattern.nunique()), "Products should not change after dropping uncommon products"

# Get dummies for electric, phev, hybrid, diesel
vin_data = generate_fuel_type_dummies(vin_data)
assert(orig_sales == vin_data.veh_count.sum()), "Sales should not change after generating fuel type dummies"
assert(orig_products == vin_data.vin_pattern.nunique()), "Products should not change after generating fuel type dummies"

# Aggregate to market level
vin_data_aggregated = aggregate_to_market(vin_data, "county_name", "vin_pattern", "veh_count")
assert(orig_sales == vin_data_aggregated.veh_count.sum()), "Sales should not change after aggregating to market level"
assert(orig_products == vin_data_aggregated.vin_pattern.nunique()), "Products should not change after aggregating to market level"

# Read in census data - contains total number of households for each county in Connecticut
census_data = read_census_data(str_data / "other_data" / "total-households-county-2019.csv")

# Merge the VIN data with the census data
mkt_data = merge_vin_census(vin_data_aggregated, census_data)
assert(orig_sales == mkt_data.veh_count.sum()), "Sales should not change after merging with census data"
assert(orig_products == mkt_data.vin_pattern.nunique()), "Products should not change after merging with census data"

# Clean the market data
mkt_data = clean_market_data(mkt_data)
orig_sales = mkt_data.veh_count.sum()

# Calculate the share of the outside good
mkt_data = calc_outside_good(mkt_data)
assert(orig_sales == mkt_data.veh_count.sum()), "Sales should not change after calculating the share of the outside good"
# assert(orig_products == mkt_data.vin_pattern.nunique()), "Products should not change after calculating the share of the outside good"

# Add instruments 
mkt_data = generate_pyblp_instruments(mkt_data)
assert(orig_sales == mkt_data.veh_count.sum()), "Sales should not change after generating PyBLP instruments"

# Save estimation data
mkt_data.to_csv(estimation_test / "mkt_data.csv", index = False)

############################################################################################################
# Test the market data


############################################################################################################
# Run the model
mkt_data_keep = mkt_data.copy()

if model == 'logit': 
    # run logit with corrected prices
    # logit_formulation = pyblp.Formulation('0 + prices + dollar_per_mile + electric + phev + electric:CA + phev:CA + hybrid + diesel + log_hp_weight + wheelbase + doors + range_elec + range_elec:CA + C(make) + C(drivetype) + C(bodytype)')
    # Updated to get rid of CA for national
    logit_formulation = pyblp.Formulation('0 + prices + dollar_per_mile + electric + phev + hybrid + diesel + wheelbase + doors + log_hp_wt + range_elec +  C(make) + C(drive_type) + C(body_type)')

    problem = pyblp.Problem(logit_formulation, mkt_data)
    logit_results_price_updated = problem.solve()

    # save results
    df_logit = pd.DataFrame({'param':logit_results_price_updated.beta_labels,
                            'value':logit_results_price_updated.beta.flatten(),
                            'se': logit_results_price_updated.beta_se.flatten()})
    #df_logit[~df_logit["param"].str.contains('make')]
    #df_logit[df_logit["param"].str.contains('electric')]
    
    df_logit[df_logit.param == "make['Chevrolet"]
    
    mkt_data['delta'] = logit_results_price_updated.delta
    mkt_data['xi'] = logit_results_price_updated.xi

    #df_logit[df_logit["param"].str.contains('fast_per_tsm')]
    results = logit_results_price_updated
    integration = None

if save:
    str_time = time.strftime("%Y_%m_%d_%H%M",time.localtime())
    str_results_folder = output_folder / 'demand_results/results_' / version/'_'/str_time
    print(str_results_folder)
    if not os.path.exists(str_results_folder):
        os.makedirs(str_results_folder)
    if dynamic == True:
        # check which version of data
        if(mkt_data.dynamic_version.unique() == 'same_veh'):
            version_name = '_common_vehicle_'
        else:
            version_name = '_state_vehicle_'
    if model == 'logit':
        df_logit.to_csv(str_results_folder/f'demand_params_{str_time}.csv',index = False)
        print(df_logit)
