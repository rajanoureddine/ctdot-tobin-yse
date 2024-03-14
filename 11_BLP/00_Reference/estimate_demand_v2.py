############################################################################################################
# Import necessary packages
import numpy as np
import pandas as pd
# import censusdata
import pyblp
import statsmodels.api as sm
import time
import os
import pickle
import sys
import pathlib
from scipy.optimize import minimize
import platform

from linearmodels.iv import IV2SLS # this is to check IV results

from functions_v2 import * # this is the set of additional functions

pyblp.options.verbose = True


############################################################################################################
# Settings 
version = "CONNECTICUT"

print('version: '+version)
print(pyblp.__version__)

# dynamic data or not
#dynamic = True
dynamic = False

# model
#model = 'DGV'
model = 'logit'
#model = 'rc'
#model = 'rc_demo'
####model = 'rc_demo_moments'
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
print("Integration setting:" + integ)

# split into 3 time periods
#split = 't1'
#split = 't2'
#split = 't3'
split = 'None'
#split = 'None, ann. moments' # full time period, separate moments by year
#split = 'geog'
print_split(split)

############################################################################################################
# set up directories
if platform.platform()[0:5] == 'macOS':
    cd = pathlib.Path().resolve().parent
    str_project = cd / "Documents" 
    str_data = "tobin_working_data"
    str_rlp = str_data / "rlpolk_data"
    str_sales_vin_characteristic = str_rlp / "rlp_with_dollar_per_mile.csv"
    output_folder = str_project / str_data / "outputs"

############################################################################################################
# read in VIN data - file of approx 100000 rows, including car characteristics
vin_data = read_vin_data(str_project,str_data,version,dynamic)

# read in census data - contains population, income, etc. for every state for four years
census_data = read_census_data(str_project,str_data,version)

# haircut first and potentially last market based on missing addl data - alters the census data
census_data = haircut_interpolate_census(str_project,str_data,census_data,incl_2021)

# read in agent observations if needed
# Interesting file - many rows of simulated agents from different states, presumably they are based on the 
# known distribution of income, households, etc. for each state
# agent_data = read_agent_data(str_project,str_data,model,version,n_agent,incl_2021,bln_rescale_income,bln_truncate_income)

# rename moments if needed
# dict_moments = read_moments(str_project,str_data,split,bln_rescale_income)

# merge vin and census data
# Note - this is the point at which we create the market data that is used later on
# This also adds a share for each VIN number, state, model year
mkt_data = merge_vin_census(vin_data,census_data,version,dynamic)

# clean data (rename vars, drop missing/problematic observations)
# Various cleaning functions, add a time trend, add a clustering ID for standard errors
mkt_data = clean_market_data(mkt_data,version)

# calculate outside good share
# Calculate the share of the outside good by summing up all the known shares and subtracting them from 1
mkt_data = calc_outside_good(mkt_data,version)

# generate PyBLP instruments
# Add instruments - this part is complex
mkt_data = generate_pyblp_instruments(mkt_data)

# if single state, keep only relevant market data, agent data
# mkt_data,agent_data = subset_states(mkt_data,agent_data,version)

# if only running subset of years (adjust mkt_data, agent_data, moments)
# mkt_data,agent_data,dict_moments,yr_keep = subset_years(mkt_data,agent_data,model,split,dict_moments)

####################
# run PyBLP models #
####################
mkt_data_keep = mkt_data.copy()

if model == 'logit':
    # run logit with corrected prices
    # logit_formulation = pyblp.Formulation('0 + prices + dollar_per_mile + electric + phev + electric:CA + phev:CA + hybrid + diesel + log_hp_weight + wheelbase + doors + range_elec + range_elec:CA + C(make) + C(drivetype) + C(bodytype)')
    # Updated to get rid of CA for national
    logit_formulation = pyblp.Formulation('0 + prices + dollar_per_mile + electric + phev + hybrid + diesel + log_hp_weight + wheelbase + doors + range_elec + chargers_per_tsm +  C(make) + C(drivetype) + C(bodytype)')

    problem = pyblp.Problem(logit_formulation, mkt_data)
    logit_results_price_updated = problem.solve()

    # save results
    df_logit = pd.DataFrame({'param':logit_results_price_updated.beta_labels,
                            'value':logit_results_price_updated.beta.flatten(),
                            'se': logit_results_price_updated.beta_se.flatten()})
    #df_logit[~df_logit["param"].str.contains('make')]
    #df_logit[df_logit["param"].str.contains('electric')]
    
    df_logit[df_logit.param == "make['Chevrolet']"]
    
    mkt_data['delta'] = logit_results_price_updated.delta
    mkt_data['xi'] = logit_results_price_updated.xi

    #df_logit[df_logit["param"].str.contains('fast_per_tsm')]
    results = logit_results_price_updated
    integration = None

################
# save results #
################
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
        print(df_logit.to_latex(index = False))
    elif model == 'nested_logit':
        df_nl.to_csv(str_results_folder+'demand_params.csv',index=False)
    elif model == 'rc':
        df_rand_coeffs.to_csv(str_results_folder+'demand_params.csv',index=False)
    elif model == 'rc_demo':
        df_rand_coeffs_demo.to_csv(str_results_folder+'demand_params.csv',index=False)
    elif model == 'rc_demo_moments':
        df_rand_coeffs_demo.to_csv(str_results_folder+'demand_params.csv',index=False)
    elif model == 'rc_nl':
        df_rand_coeffs_demo.to_csv(str_results_folder+'demand_params.csv',index=False)
    elif model == 'rc_nl_moments':
        df_rand_coeffs_demo.to_csv(str_results_folder+'demand_params.csv',index=False)
    elif model == 'DGV':
        df_dyn_results.to_csv(str_results_folder+'demand_params.csv',index = False)
    # save the results object
    pickle.dump(results,open(str_results_folder / "pickled_demand.pickle",'wb'))
    pickle.dump(mkt_data,open(str_results_folder/ "pickled_mkt_data.pickle",'wb'))
    # if agent_data is not None:
    #     pickle.dump(agent_data,open(str_results_folder+"/pickled_agent_data.pickle",'wb'))
    if integration is not None:
        pickle.dump(integration,open(str_results_folder+"/pickled_integration_data.pickle",'wb'))


# test = pickle.load(open(str_results_folder+"/pickled_demand.pickle", "rb" ) )

