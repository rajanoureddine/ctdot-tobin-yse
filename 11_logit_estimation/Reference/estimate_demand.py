# auto_strategies_env (conda; not pyblp 0.13)
# ase_blp_1_0 (conda; pyblp 1.0)
# or ev_env (up-to-date pyblp; lives in SW_Automaker_Strategies folder)
import numpy as np
import pandas as pd
#import censusdata
import pyblp
import statsmodels.api as sm
import time
import os
import pickle
import sys
from scipy.optimize import minimize

from linearmodels.iv import IV2SLS # this is to check IV results
#from stargazer.stargazer import Stargazer

from functions import * # this is the set of additional functions

pyblp.options.verbose = True
pyblp.options.singular_tol = np.inf
#pyblp.options.micro_computation_chunks = 400

# determine whether this is running on the cluster
bln_cluster = check_cluster()
# implement parallelization
parallel = True
if (bln_cluster):
    n_nodes_par = int(os.getenv("SLURM_CPUS_PER_TASK"))
    save = True
else:
    parallel = False
    n_nodes_par = 2
    save = True
print("nodes: "+ str(n_nodes_par))

# state vs. national level data vs. hybrid
#version = sys.argv[1]
#version = 'CA_county'
#version = 'CALIFORNIA'
#version = 'West'
#version = 'Mountain'
#version = 'West North Central'
#version = 'East North Central'
#version = 'West South Central'
#version = 'East South Central'
#version = 'South Atlantic'
#version = 'Middle Atlantic'
#version = 'New England'
#version = 'state'
version = 'hybrid' # ZEV states + a few others are each treated as separate market
#version = 'hybrid_regional' # ZEV states + others grouped into census divisions
#version = 'national'
print('version: '+version)
print(pyblp.__version__)

# dynamic data or not
#dynamic = True
dynamic = False

# model
#model = 'DGV'
#model = 'logit'
#model = 'rc'
#model = 'rc_demo'
model = 'rc_demo_moments'
#model = 'nested_logit'
#model = 'rc_nl'
#model = 'rc_nl_moments'

# number of iterations
gmm_rounds = '1s'
#gmm_rounds = '2s'

# set number of simulated agents per market
# does not apply if using gaussian quadrature, but affects remaining integration approaches
n_agent = 1200

# include 2021 data
incl_2021 = True

# re-scale income moments
bln_rescale_income = False
bln_truncate_income = True
# re-scale msrp moments
bln_rescale_msrp = True
# reduce size of market
bln_reduce_mkt = False

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
print_split(split)

#####################
# set up directories#
#####################
if bln_cluster: 
    str_project = ''     
    output_folder = 'outputs'
    # set number of nodes for parallelized run
    if (parallel):
        n_nodes = int(os.getenv("SLURM_CPUS_PER_TASK"))
        print(str(n_nodes)+' nodes for parallelization.')
else:
    str_project = '~/Dropbox (YSE)/SW_Automaker_Strategies/'
    output_folder = '../../results'
str_data = 'data/'

##########################
# read in and clean data #
##########################
# read in VIN data
vin_data = read_vin_data(str_project,str_data,version,dynamic)

# read in census data
census_data = read_census_data(str_project,str_data,version)

# haircut first and potentially last market based on missing addl data
census_data = haircut_interpolate_census(str_project,str_data,census_data,incl_2021)

# read in agent observations if needed
agent_data = read_agent_data(str_project,str_data,model,version,n_agent,incl_2021,bln_rescale_income,bln_truncate_income)

# rename moments if needed
dict_moments = read_moments(str_project,str_data,split,bln_rescale_income)

# merge vin and census data
mkt_data = merge_vin_census(vin_data,census_data,version,dynamic,bln_reduce_mkt)

## Add broad_ev_hybrid category in mkt_data ##
mkt_data['broad_ev_hybrid'] = np.where(mkt_data['fueltype'].isin(['I', 'Y', 'L']), 1, 0)

# clean data (rename vars, drop missing/problematic observations)
mkt_data = clean_market_data(mkt_data,version)

# calculate outside good share
mkt_data = calc_outside_good(mkt_data,version)

# generate PyBLP instruments
mkt_data = generate_pyblp_instruments(mkt_data)

# if single state, keep only relevant market data, agent data
mkt_data,agent_data = subset_states(mkt_data,agent_data,version)

# define three time periods
if(False):
    mkt_data['t1'] = 0
    mkt_data['t2'] = 0
    mkt_data['t3'] = 0
    mkt_data.loc[mkt_data.model_year.isin([2014,2015]),'t1'] = 1
    mkt_data.loc[mkt_data.model_year.isin([2016,2017,2018]),'t2'] = 1
    mkt_data.loc[mkt_data.model_year.isin([2019,2020,2021]),'t3'] = 1

    # define two time periods
    mkt_data['alt_t1'] = 0
    mkt_data['alt_t2'] = 0
    mkt_data.loc[mkt_data.model_year.isin([2014,2015,2016,2017]),'alt_t1'] = 1
    mkt_data.loc[mkt_data.model_year.isin([2018,2019,2020,2021]),'alt_t2'] = 1

# if only running subset of years (adjust mkt_data, agent_data, moments)
mkt_data,agent_data,dict_moments,yr_keep = subset_years(mkt_data,agent_data,model,split,dict_moments)
#mkt_data = mkt_data.loc[mkt_data.model_year > 2014]
####################
# run PyBLP models #
####################
mkt_data_keep = mkt_data.copy()
if model == 'logit':
    # run logit with corrected prices
    #logit_formulation = pyblp.Formulation('0 + prices + dollar_per_mile + electric + phev + electric:CA + phev:CA + hybrid + diesel + log_hp_weight + wheelbase + doors + range_elec + range_elec:CA + C(make) + C(drivetype) + C(bodytype) + C(state)')
    logit_formulation = pyblp.Formulation('0 + prices + dollar_per_mile + electric + phev + electric + phev + hybrid + diesel + log_hp_weight + wheelbase + doors + range_elec + range_elec + C(make) + C(drivetype) + C(bodytype) + C(state)')


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
if model == 'rc':  
    # run random coefficients logit withOUT demographics
    #X1_formulation = pyblp.Formulation('0 + prices + dollar_per_mile:ice + '+str_time+ str_ev + str_phev + str_charger +'range_elec:electric + range_elec:phev + hybrid + diesel + log_hp_weight + wheelbase + doors + C(make) + C(drivetype) + C(bodytype)')
    #X1_formulation = pyblp.Formulation('0+ prices + dollar_per_mile + electric:t1 + electric:t2 + electric:t3 + phev + hybrid + diesel+ log_hp_weight + wheelbase + doors + range_elec + C(make) + C(drivetype) + C(bodytype)')
    #X2_formulation = pyblp.Formulation('0 + electric:t1 + electric:t2 + electric:t3')
    #X1_formulation = pyblp.Formulation('0 + prices + dollar_per_mile + electric + phev + hybrid + diesel + log_hp_weight + wheelbase + doors + range_elec + C(make) + C(drivetype) + C(bodytype)')
    X1_formulation = pyblp.Formulation('0 + prices + dollar_per_mile + electric + phev + hybrid + diesel + log_hp_weight + wheelbase + doors + range_elec + C(make) + C(drivetype) + C(bodytype) + C(state)')

    X2_formulation = pyblp.Formulation('0 + broad_ev')
    product_formulations = (X1_formulation, X2_formulation)
    # should look into alt integration approaches
    if(integ in ['monte_carlo','halton','mlhs','lhs']):
        integration = pyblp.Integration(integ, size=n_agent, specification_options={'seed': 0})
    elif (integ == 'gauss'):
        integration = pyblp.Integration('grid', size=20)

    mc_problem = pyblp.Problem(product_formulations, mkt_data, integration=integration)
    optim = pyblp.Optimization('l-bfgs-b',{'gtol': 1e-10}) 
    iter = pyblp.Iteration('squarem',{'atol': 1e-10}) # using squarem acceleration method
    eye = np.eye(1)
    sigma_guess = eye
    sigma_lb = sigma_guess * 0
    sigma_ub = eye * 150
    if parallel:
        with pyblp.parallel(n_nodes):
            results1 = mc_problem.solve(sigma=sigma_guess,sigma_bounds=(sigma_lb,sigma_ub), optimization=optim,iteration=iter,method = gmm_rounds)
    else:
        results1 = mc_problem.solve(sigma=sigma_guess,sigma_bounds=(sigma_lb,sigma_ub), optimization=optim,iteration=iter, method = gmm_rounds)

    # save results
    df_rand_coeffs = pd.DataFrame({'param':results1.beta_labels,
                            'value':results1.beta.flatten(),
                            'se': results1.beta_se.flatten()})
    df_sigma = pd.DataFrame({'param': ['sigma_' + s for s in results1.sigma_labels],
                             'value': np.diagonal(results1.sigma).flatten(),
                             'se': np.diagonal(results1.sigma_se).flatten()})
    df_rand_coeffs = pd.concat([df_rand_coeffs,df_sigma],ignore_index=True)
    df_rand_coeffs[df_rand_coeffs["param"].str.contains('electric')]
    df_rand_coeffs[df_rand_coeffs["param"].str.contains('phev')]
    results = results1
if model == 'rc_demo' or model == 'rc_demo_moments' or model == 'rc_nl' or model == 'rc_nl_moments':
    # create a dataframe that stores the random coefficients, which demographics they're interacted with, etc.
    #dict_X2 = {'broad_ev': ['rc','urban'],
    #           'prices': ['I(1/income)']}
    #dict_X2 = {'broad_ev': ['rc','urban'],
    #           'prices': ['I(low/income)','I(mid/income)','I(high/income)']}
    #dict_X2 = {'broad_ev': ['urban'],
    #           'prices': ['I(1/income)'],
    #           '1': ['rc']}
    #dict_X2 = {'wheelbase':['rc'],
    #           'prices': ['I(1/income)']}
    #dict_X2 = {'dollar_per_mile': ['rc'],
    #           'prices': ['I(1/income)']
    #}
    #dict_X2 = {'prices': ['I(1/income)']}
    # dict_X2 = {'broad_ev': ['charging_density']}
    dict_X2 = {'broad_ev': ['urban'],
               'broad_ev_hybrid': ['rc'],
               'prices': ['I(1/income)']} 

    # product formulations
    # if 'prices' is one of the keys in dict_X2
    str_X1 = 'dollar_per_mile + electric + phev + hybrid + diesel + log_hp_weight + wheelbase + doors + range_elec + C(make) + C(drivetype) + C(bodytype) + C(state)'
    if('prices' in dict_X2.keys()):
        X1_formulation = pyblp.Formulation('0 +' + str_X1)
    else:
        X1_formulation = pyblp.Formulation('0 + prices + ' + str_X1)

    str_x2_formulation = '1'
    for x2 in dict_X2:
        if x2 != '1':
            str_x2_formulation = str_x2_formulation + '+' + x2
        if x2 == 'dollar_per_mile':
            str_x2_formulation = str_x2_formulation + '+dpm_quantile_1+dpm_quantile_5'
        if x2 == 'wheelbase':
            str_x2_formulation = str_x2_formulation + '+wb_quantile_1+wb_quantile_3'
    #str_x2_formulation = str_x2_formulation + '+dollar_per_mile+dpm_quantile_1+dpm_quantile_5+wheelbase+wb_quantile_1+wb_quantile_3'

    X2_formulation = pyblp.Formulation(str_x2_formulation)
    product_formulations = (X1_formulation,X2_formulation)
    # agent formulation
    lst_demogs = list()
    for x2 in dict_X2:
        lst_new = dict_X2[x2]
        lst_new = [x for x in lst_new if (x != 'rc') and not( x in lst_demogs)]
        lst_demogs.extend(lst_new)
    # make sure lst_demogs is unique
    #lst_demogs = list(set(lst_demogs))
    # add the '1' per new pyblp documentation
    lst_demogs = ['1'] + lst_demogs
    # add the mid/high if they are interacted with something
    if any(item in ['I(mid/income)', 'I(1/income)'] for item in lst_demogs):
        lst_demogs = lst_demogs + ['low','mid','high']
    for d in lst_demogs:
        if d == '1':
            str_d_formulation =  d
        else:
            str_d_formulation = str_d_formulation + '+' + d
    # correctly ordered list of demographics
    lst_x2_full = str_x2_formulation.split('+')

    # initial guesses of sigma and pi
    # dimensions of sigma
    dim_sigma = len(lst_x2_full)
    initial_sigma = np.zeros([dim_sigma,dim_sigma]) 
    sigma_ub = np.zeros([dim_sigma,dim_sigma]) 
    sigma_lb = np.zeros([dim_sigma,dim_sigma]) 
    # note: currently not allowing covariances between random tastes
    # dimensions of pi
    initial_pi = np.zeros([dim_sigma, len(lst_demogs)])
    pi_bds = (np.ones([dim_sigma, len(lst_demogs)])*np.nan,
              np.ones([dim_sigma, len(lst_demogs)])*np.nan)
    # make sure initial pi and sigma only include desired interactions
    for i in range(0,len(lst_x2_full)):
        param =lst_x2_full[i]
        if (param in dict_X2):
            interactions = dict_X2[param]
            for interaction in interactions:
                if interaction == 'rc':
                    initial_sigma[i,i] = 5
                    sigma_ub[i,i] = 20
                else:
                    # get index of demographic within list demographics
                    j = lst_demogs.index(interaction)
                    initial_pi[i,j] = .05
                    if (interaction in ['I(1/income)','I(low/income)','I(mid/income)','I(high/income)']): # THIS IS A TEMPORARY FIX TO SET UB = 0
                        pi_bds[1][i,j] = 0
                        if(interaction == 'I(low/income)'):
                            initial_pi[i,j] = -15
                        else:
                            initial_pi[i,j] = -11

    # process agent data
    # get number of nodes required
    n_nodes = int((initial_sigma != 0).sum())
    if (n_nodes == 0):
        agent_data['nodes0'] = 0
        integration = None
    else:
        # create relevant column names
        col_nodes = []
        for n in range(0,n_nodes):
            col_nodes.append('nodes'+str(n))
        df_nodes_all = pd.DataFrame()
        # generate nodes using pyblp integration tools
        for mkt_id in agent_data.market_ids.unique():
            if(integ == 'gauss'):
                # alt: try Gauss-Hermite quadrature
                if(n_nodes == 1):
                    n_integration = 10
                else:
                    n_integration = 15
                integration = pyblp.Integration('grid', size=n_integration)
                agent_data_nodes = pyblp.build_integration(integration, n_nodes)
                df_nodes = pd.DataFrame(agent_data_nodes.nodes, columns = col_nodes)
                df_weights = pd.DataFrame(agent_data_nodes.weights, columns = ['weights'])
                df_nodes['weights'] = df_weights
                # get p urban
                p_urban = agent_data.loc[agent_data.market_ids == mkt_id,'urban'].sum()/(agent_data.loc[agent_data.market_ids == mkt_id,].shape[0])
                df_nodes_urban = df_nodes.copy()
                df_nodes_urban['urban'] = 1
                df_nodes_urban['weights'] *= p_urban
                df_nodes_rural = df_nodes.copy()
                df_nodes_rural['urban'] = 0
                df_nodes_rural['weights'] *= (1-p_urban)
                df_market = pd.concat([df_nodes_urban,df_nodes_rural],ignore_index=True)
                df_market['state'] = agent_data.loc[agent_data.market_ids == mkt_id,'state'].unique()[0]
                df_market['model_year'] = agent_data.loc[agent_data.market_ids == mkt_id,'model_year'].unique()[0]
                df_market['market_ids'] = mkt_id
                #n_rep = int(n_agent/n_integration)
                #df_market = pd.concat([df_nodes]*n_rep, ignore_index=True)
                df_nodes_all = pd.concat([df_nodes_all,df_market], ignore_index=True)
            else:
                integration = pyblp.Integration(integ, size=n_agent, specification_options={'seed': 1217})

                agent_data_nodes = pyblp.build_integration(integration, n_nodes)
                df_market = pd.DataFrame(agent_data_nodes.nodes, columns = col_nodes)
                df_weights = pd.DataFrame(agent_data_nodes.weights,columns = ['weights'])
                df_market['weights'] = df_weights
                df_nodes_all = pd.concat([df_nodes_all,df_market], ignore_index=True)
        # concat with existing agent data
        if(integ == 'gauss'):
            agent_data = df_nodes_all
        else:
            agent_data = pd.concat([agent_data.drop(['weights'],axis=1).reset_index(drop=True),df_nodes_all],axis=1)

    # if using charging_density, need to adjust agent_data and product_data
    if ('charging_density' in dict_X2['broad_ev']):
        # get agent data without na charging_density
        agent_data = agent_data.loc[~agent_data.charging_density.isna()]
        mkt_data = mkt_data.loc[mkt_data.model_year.isin(agent_data.model_year.unique())]
    micro_moments = []
    if model == 'rc_demo_moments' or model == 'rc_nl_moments':
        # characteristic expectation moment: expectation of characteristic chosen by certain agents
        # need agent_ids
        agent_data['agent_ids'] = agent_data.index
        
        # specify agent approach based on pyblp version
        if pyblp.__version__ in ('0.12.0','0.11.0','0.10.0'):
            micro_moments = moments_v_10_12(mkt_data,agent_data,dict_X2,lst_demogs,split,yr_keep,dict_moments)
        elif pyblp.__version__ == '0.13.0':
            print('moment calculation not working within pyblp 13.0')
        elif pyblp.__version__ in ('1.0.0','1.1.0'):
            if split == 'None':
                compute_ratio = lambda v: v[0] / v[1] 
                compute_ratio_gradient = lambda v: [1 / v[1], -v[0] / v[1]**2]
                #if(False):
                if ('broad_ev' in dict_X2.keys()):
                    broad_ev_index  = lst_x2_full.index('broad_ev')
                    if ('urban' in dict_X2['broad_ev']):
                        urban_index = lst_demogs.index('urban')

                        df_moments_urban = dict_moments['urban']
                        micro_statistics = df_moments_urban.loc[(df_moments_urban.year == 0) & (df_moments_urban.demo_exp == 1) & (df_moments_urban.broad_ev == 1)]
                        micro_dataset = pyblp.MicroDataset(
                                        name="Velocity",
                                        observations=int(micro_statistics.n_obs),
                                        market_ids = mkt_data.loc[mkt_data.model_year.isin([*range(2017,2022,1)]),'market_ids'].unique(),
                                        compute_weights=lambda t, p, a: np.ones((a.size, p.size)),
                                        )
                        #micro_statistics = pd.read_csv(str_project + 'data/final/test_moment.csv', index_col=0)
                        urban_ev_part = pyblp.MicroPart(
                                        name="E[urban_i * broad_ev_j]",
                                        dataset=micro_dataset,
                                        compute_values=lambda t, p, a: np.outer(a.demographics[:, urban_index], 
                                                                                p.X2[:, broad_ev_index]),
                                        )
                        ev_part = pyblp.MicroPart(
                                    name="E[broad_ev_j]",
                                    dataset=micro_dataset,
                                    compute_values=lambda t, p, a: np.outer(a.demographics[:, 0], 
                                                                            p.X2[:, broad_ev_index]),
                                    )

                        micro_moments = micro_moments + [ 
                            pyblp.MicroMoment(name="E[urban_i | broad_ev_j]", 
                                            value=float(micro_statistics.moment),
                                            parts=[urban_ev_part, ev_part],
                                            compute_value=compute_ratio,
                                            compute_gradient=compute_ratio_gradient,
                                            )
                                            ]  
                    if ('charging_density' in dict_X2['broad_ev']):
                        cd_index = lst_demogs.index('charging_density')

                        df_moments_cd = dict_moments['charging_density']
                        micro_statistics = df_moments_cd.loc[(df_moments_cd.year == 0) & (df_moments_cd.broad_ev == 1)]
                        micro_dataset = pyblp.MicroDataset(
                                        name="Velocity",
                                        observations=int(micro_statistics.n_obs),
                                        market_ids = mkt_data.loc[mkt_data.model_year.isin([*range(2017,2022,1)]),'market_ids'].unique(),
                                        compute_weights=lambda t, p, a: np.ones((a.size, p.size)),
                                        )
                        #micro_statistics = pd.read_csv(str_project + 'data/final/test_moment.csv', index_col=0)
                        cd_part = pyblp.MicroPart(
                                        name="E[cd * broad_ev_j]",
                                        dataset=micro_dataset,
                                        compute_values=lambda t, p, a: np.outer(a.demographics[:, cd_index], 
                                                                                p.X2[:, broad_ev_index]),
                                        )
                        ev_part = pyblp.MicroPart(
                                    name="E[broad_ev]",
                                    dataset=micro_dataset,
                                    compute_values=lambda t, p, a: np.outer(a.demographics[:, 0], 
                                                                            p.X2[:, broad_ev_index]),
                                    )

                        micro_moments = micro_moments + [ 
                            pyblp.MicroMoment(name="E[cd | broad_ev]", 
                                            value=float(micro_statistics.moment),
                                            parts=[cd_part, ev_part],
                                            compute_value=compute_ratio,
                                            compute_gradient=compute_ratio_gradient,
                                            )
                                            ]
                    # if ('rc' in dict_X2['broad_ev']):
                    if ('rc' in dict_X2['broad_ev_hybrid']):
                        df_moments_sc_fuel = dict_moments['sc_fuel']
                        micro_statistics = df_moments_sc_fuel.loc[(df_moments_sc_fuel.year == 2019) & (df_moments_sc_fuel.purchase_fuel == "Broad EV Hybrid") & (df_moments_sc_fuel.sc_fuel == "Broad EV Hybrid")]
                        micro_dataset = pyblp.MicroDataset(
                                    name="InMoment",
                                    observations=int(micro_statistics.n_fuel_spec),
                                    market_ids = mkt_data.loc[mkt_data.model_year.isin([2019]),'market_ids'].unique(),
                                    compute_weights=lambda t, p, a: np.ones((a.size, p.size, p.size)),
                                    )
                        sc_ev_part = pyblp.MicroPart(
                                    name="E[broad_ev_1 * broad_ev_2]",
                                    dataset=micro_dataset,
                                    compute_values=lambda t, p, a: np.einsum('i,j,k->ijk',a.demographics[:, 0], p.X2[:, broad_ev_index], p.X2[:, broad_ev_index]))
                        ev_part = pyblp.MicroPart(
                                name="E[broad_ev_1]",
                                dataset=micro_dataset,
                                compute_values=lambda t, p, a: np.einsum('i,j,k->ijk',a.demographics[:, 0], p.X2[:, broad_ev_index], p.X2[:, 0]))
                        micro_moments = micro_moments + [ 
                            pyblp.MicroMoment(name="E[broad_ev_2 | broad_ev_1]", 
                                          value=float(micro_statistics.pct_spec),
                                          parts=[sc_ev_part, ev_part],
                                          compute_value=compute_ratio,
                                          compute_gradient=compute_ratio_gradient,
                                          )
                                          ]
                if ('prices' in dict_X2.keys()): 
                    if (any(item in ['I(mid/income)', 'I(1/income)'] for item in dict_X2['prices'])):
                        if bln_rescale_msrp:
                            int_rescale = 10
                        else: 
                            int_rescale = 1
                        prices_index  = lst_x2_full.index('prices')
                        low_index = str_d_formulation.split('+').index('low')
                        mid_index = str_d_formulation.split('+').index('mid')
                        high_index = str_d_formulation.split('+').index('high')
                        micro_statistics = dict_moments['income_grp']
                        micro_dataset = pyblp.MicroDataset(
                                    name = "Velocity merged",
                                    observations = int(micro_statistics.n_obs.sum()),
                                    market_ids = mkt_data.loc[mkt_data.model_year.isin([*range(2017,2022,1)]),'market_ids'].unique(),
                                    compute_weights = lambda t, p, a: np.ones((a.size, p.size)),
                                    )
                        # generate msrp | low income group moment
                        msrp_low_part = pyblp.MicroPart(
                            name = "E(msrp_i * low_j)",
                            dataset = micro_dataset,
                            compute_values = lambda t, p, a: np.outer(a.demographics[:, low_index], 
                                                                    p.X2[:, prices_index]/int_rescale),
                        )
                        low_part = pyblp.MicroPart(
                            name = "E(low_j)",
                            dataset = micro_dataset,
                            compute_values = lambda t, p, a: np.outer(a.demographics[:, low_index], 
                                                                    p.X2[:,0]),
                        )
                        moment_low = pyblp.MicroMoment(name="E[msrp | income_low_j]", 
                                                    value=float(micro_statistics.loc[micro_statistics.income_grp == 'low','mean_msrp'])/int_rescale,
                                                    parts=[msrp_low_part, low_part],
                                                    compute_value=compute_ratio,
                                                    compute_gradient=compute_ratio_gradient,
                                                    )
                        # generate msrp | mid income group moment
                        msrp_mid_part = pyblp.MicroPart(
                            name = "E(msrp_i * mid_j)",
                            dataset = micro_dataset,
                            compute_values = lambda t, p, a: np.outer(a.demographics[:, mid_index], 
                                                                    p.X2[:, prices_index]/int_rescale),
                        )
                        mid_part = pyblp.MicroPart(
                            name = "E(mid_j)",
                            dataset = micro_dataset,
                            compute_values = lambda t, p, a: np.outer(a.demographics[:, mid_index], 
                                                                    p.X2[:,0]),
                        )
                        moment_mid = pyblp.MicroMoment(name="E[msrp | income_mid_j]", 
                                                    value=float(micro_statistics.loc[micro_statistics.income_grp == 'med','mean_msrp'])/int_rescale,
                                                    parts=[msrp_mid_part, mid_part],
                                                    compute_value=compute_ratio,
                                                    compute_gradient=compute_ratio_gradient,
                                                    )
                        # generate msrp | high income group moment
                        msrp_high_part = pyblp.MicroPart(
                            name = "E(msrp_i * high_j)",
                            dataset = micro_dataset,
                            compute_values = lambda t, p, a: np.outer(a.demographics[:,high_index], 
                                                                    p.X2[:, prices_index]/int_rescale),
                        )
                        high_part = pyblp.MicroPart(
                            name = "E(high_j)",
                            dataset = micro_dataset,
                            compute_values = lambda t, p, a: np.outer(a.demographics[:, high_index], 
                                                                    p.X2[:,0]),
                        )
                        moment_high = pyblp.MicroMoment(name="E[msrp | income_high_j]", 
                                                    value=float(micro_statistics.loc[micro_statistics.income_grp == 'high','mean_msrp'])/int_rescale,
                                                    parts=[msrp_high_part, high_part],
                                                    compute_value=compute_ratio,
                                                    compute_gradient=compute_ratio_gradient,
                                                    )
                        micro_moments = micro_moments + [moment_low, moment_mid, moment_high]
                        #micro_moments = micro_moments + [moment_high]
                                # compute covariance
                compute_cov = lambda v: v[0] - v[1] * v[2]
                compute_cov_gradient = lambda v: [1, -v[2], -v[1]]
                df_moments_sc_cov = dict_moments['sc_cov']
                if ('wheelbase' in dict_X2.keys()):
                #if(True):
                    wheelbase_index  = lst_x2_full.index('wheelbase')
                    # get row of df_moments_sc_cov where 'char' = 'wheelbase'
                    micro_statistics = df_moments_sc_cov.loc[(df_moments_sc_cov.char == 'wheelbase')]
                    if(False):
                        micro_dataset = pyblp.MicroDataset(
                                    name="InMoment WB",
                                    observations=int(micro_statistics.n),
                                    market_ids = mkt_data.loc[mkt_data.model_year.isin([2020]),'market_ids'].unique(),
                                    compute_weights=lambda t, p, a: np.ones((a.size, p.size, p.size)),
                                    )
                        EXY_part = pyblp.MicroPart(
                                    name = "E[wheelbase_1*wheelbase_2]",
                                    dataset = micro_dataset,
                                    compute_values = lambda t, p, a:  np.einsum('i,j,k->ijk',a.demographics[:, 0], p.X2[:, wheelbase_index], p.X2[:, wheelbase_index])
                                    )
                        EX_part = pyblp.MicroPart(
                                    name = "E[wheelbase_1]",
                                    dataset = micro_dataset,
                                    compute_values = lambda t, p, a:  np.einsum('i,j,k->ijk',a.demographics[:, 0], p.X2[:, wheelbase_index], p.X2[:, 0])
                                    )
                        EY_part =  pyblp.MicroPart(
                                    name = "E[wheelbase_2]",
                                    dataset = micro_dataset,
                                    compute_values = lambda t, p, a:  np.einsum('i,j,k->ijk',a.demographics[:, 0], p.X2[:, 0], p.X2[:, wheelbase_index])
                                    )
                        
                        micro_moments = micro_moments + [
                            pyblp.MicroMoment(name = "Cov(wheelbase_1, wheelbase_2)",
                                            value = float(micro_statistics.val),
                                            parts = [EXY_part, EX_part, EY_part],
                                            compute_value = compute_cov,
                                            compute_gradient = compute_cov_gradient,)
                        ]
                    if(True):
                        wb_quantile_1_index = lst_x2_full.index('wb_quantile_1')
                        wb_quantile_3_index = lst_x2_full.index('wb_quantile_3')
                        df_sc_wb_quantile = dict_moments['sc_wb_quantile']
                        micro_dataset = pyblp.MicroDataset(
                            name = 'InMoment WB Quantile',
                            observations = int(df_sc_wb_quantile.n.sum()),
                            market_ids = mkt_data.loc[mkt_data.model_year.isin([*range(2018,2022,1)]),'market_ids'].unique(),
                            compute_weights = lambda t, p, a: np.ones((a.size,p.size,p.size)),
                        )
                        sc_1_1_part_WB = pyblp.MicroPart(
                            name = "E[wb_quantile_1_1*wb_quantile_1_2]",
                            dataset = micro_dataset,
                            compute_values = lambda t, p, a: np.einsum('i,j,k->ijk',a.demographics[:, 0], p.X2[:, wb_quantile_1_index], p.X2[:, wb_quantile_1_index])
                        )
                        fc_1_part_WB = pyblp.MicroPart(
                            name = 'E[wb_quantile_1]',
                            dataset = micro_dataset,
                            compute_values = lambda t, p, a: np.einsum('i,j,k->ijk',a.demographics[:, 0], p.X2[:, wb_quantile_1_index], p.X2[:, 0])
                        )
                        moment_wb_1_1 = pyblp.MicroMoment(name='E[wb_quantile_1_2 | wb_quantile_1_1]',
                                                           parts = [sc_1_1_part_WB, fc_1_part_WB],
                                                           value = float(df_sc_wb_quantile.loc[(df_sc_wb_quantile.quantile_wheelbase == 1) & (df_sc_wb_quantile.sc_quantile_wheelbase == 1),'pct']),
                                                           compute_value = compute_ratio,
                                                           compute_gradient = compute_ratio_gradient,)
                                                           
                        sc_3_1_part_WB = pyblp.MicroPart(
                            name = "E[wb_quantile_1_1*wb_quantile_3_2]",
                            dataset = micro_dataset,
                            compute_values = lambda t, p, a: np.einsum('i,j,k->ijk',a.demographics[:, 0], p.X2[:, wb_quantile_1_index], p.X2[:, wb_quantile_3_index])
                        )
                        moment_wb_1_3 = pyblp.MicroMoment(name='E[wb_quantile_3_2 | wb_quantile_1_1]',
                                                           parts = [sc_3_1_part_WB, fc_1_part_WB],
                                                           value = float(df_sc_wb_quantile.loc[(df_sc_wb_quantile.quantile_wheelbase == 1) & (df_sc_wb_quantile.sc_quantile_wheelbase == 3),'pct']),
                                                           compute_value = compute_ratio,
                                                           compute_gradient = compute_ratio_gradient,)
                        micro_moments = micro_moments + [moment_wb_1_1,moment_wb_1_3]
                if 'dollar_per_mile' in dict_X2.keys():
                #if(True):
                    if(False): # covariance of dollar per mile
                        dpm_index = lst_x2_full.index('dollar_per_mile')
                        # get row of df_moments_sc_cov where 'char' = 'dollar_per_mile'
                        micro_statistics = df_moments_sc_cov.loc[(df_moments_sc_cov.char == 'dollar_per_mile')]
                        micro_dataset = pyblp.MicroDataset(
                                    name="InMoment FC",
                                    observations=int(micro_statistics.n),
                                    market_ids = mkt_data.loc[mkt_data.model_year.isin([*range(2018,2022,1)]),'market_ids'].unique(),
                                    compute_weights=lambda t, p, a: np.ones((a.size, p.size, p.size)),
                                    )
                        EXY_part = pyblp.MicroPart(
                                    name = "E[dpm_1*dpm_2]",
                                    dataset = micro_dataset,
                                    compute_values = lambda t, p, a:  np.einsum('i,j,k->ijk',a.demographics[:, 0], p.X2[:, dpm_index], p.X2[:, dpm_index])
                                    )
                        EX_part = pyblp.MicroPart(
                                    name = "E[dpm_1]",
                                    dataset = micro_dataset,
                                    compute_values = lambda t, p, a:  np.einsum('i,j,k->ijk',a.demographics[:, 0], p.X2[:, dpm_index], p.X2[:, 0])
                                    )
                        EY_part =  pyblp.MicroPart(
                                    name = "E[dpm_2]",
                                    dataset = micro_dataset,
                                    compute_values = lambda t, p, a:  np.einsum('i,j,k->ijk',a.demographics[:, 0], p.X2[:, 0], p.X2[:, dpm_index])
                                    )
                        
                        micro_moments = micro_moments + [
                            pyblp.MicroMoment(name = "Cov(dpm_1, dpm_2)",
                                            value = float(micro_statistics.val),
                                            parts = [EXY_part, EX_part, EY_part],
                                            compute_value = compute_cov,
                                            compute_gradient = compute_cov_gradient,)
                        ]
                    # quantiles of first and second choice moments
                    if(True):
                        dpm_quantile_1_index = lst_x2_full.index('dpm_quantile_1')
                        dpm_quantile_5_index = lst_x2_full.index('dpm_quantile_5')
                        df_sc_dpm_quantile = dict_moments['sc_dpm_quantile']
                        micro_dataset = pyblp.MicroDataset(
                            name = 'InMoment DPM Quantile',
                            observations = int(df_sc_dpm_quantile.n.sum()),
                            market_ids = mkt_data.loc[mkt_data.model_year.isin([*range(2018,2022,1)]),'market_ids'].unique(),
                            compute_weights = lambda t, p, a: np.ones((a.size,p.size,p.size)),
                        )
                        
                        sc_1_1_part = pyblp.MicroPart(
                            name = "E[dpm_quantile_1_1*dpm_quantile_1_2]",
                            dataset = micro_dataset,
                            compute_values = lambda t, p, a: np.einsum('i,j,k->ijk',a.demographics[:, 0], p.X2[:, dpm_quantile_1_index], p.X2[:, dpm_quantile_1_index])
                        )
                        fc_1_part = pyblp.MicroPart(
                            name = 'E[dpm_quantile_1]',
                            dataset = micro_dataset,
                            compute_values = lambda t, p, a: np.einsum('i,j,k->ijk',a.demographics[:, 0], p.X2[:, dpm_quantile_1_index], p.X2[:, 0])
                        )
                        moment_dpm_1_1 = pyblp.MicroMoment(name='E[dpm_quantile_1_2 | dpm_quantile_1_1]',
                                                           parts = [sc_1_1_part, fc_1_part],
                                                           value = float(df_sc_dpm_quantile.loc[(df_sc_dpm_quantile.quantile_dollar_per_mile == 1) & (df_sc_dpm_quantile.sc_quantile_dollar_per_mile == 1),'pct']),
                                                           compute_value = compute_ratio,
                                                           compute_gradient = compute_ratio_gradient,)
                        micro_moments = micro_moments + [moment_dpm_1_1]
                        if(False):
                            sc_5_5_part = pyblp.MicroPart(
                                name = "E[dpm_quantile_5_1*dpm_quantile_5_2]",
                                dataset = micro_dataset,
                                compute_values = lambda t, p, a: np.einsum('i,j,k->ijk',a.demographics[:, 0], p.X2[:, dpm_quantile_5_index], p.X2[:, dpm_quantile_5_index])
                            )
                            fc_5_part = pyblp.MicroPart(
                                name = 'E[dpm_quantile_5]',
                                dataset = micro_dataset,
                                compute_values = lambda t, p, a: np.einsum('i,j,k->ijk',a.demographics[:, 0], p.X2[:, dpm_quantile_5_index], p.X2[:, 0])
                            )
                            moment_dpm_5_5 = pyblp.MicroMoment(name='E[dpm_quantile_5_2 | dpm_quantile_5_1]',
                                                               value = float(df_sc_dpm_quantile.loc[(df_sc_dpm_quantile.quantile_dollar_per_mile == 5) & (df_sc_dpm_quantile.sc_quantile_dollar_per_mile == 5),'pct']),
                                                               parts = [sc_5_5_part, fc_5_part],
                                                               compute_value = compute_ratio,
                                                               compute_gradient = compute_ratio_gradient,)
                            micro_moments = micro_moments + [moment_dpm_5_5]
        else:
            print('confirm how moments work in current pyblp version')
    mkt_data2 = mkt_data.copy()
    rho_guess = None
    rho_bounds = None
    if model == 'rc_nl' or model == 'rc_nl_moments':
        # nested logit setup if nl
        # put all inside goods in the same nest
        #mkt_data2['nesting_ids'] = 1
        
        #create ICE and EV nests
        mkt_data2['nesting_ids'] = 1 + mkt_data2['electric'] + mkt_data2['phev']
        
        # construct group size instrument - leaving out for now, is this necessary?
        # Ackerberg and Rysman (2005) suggests yes, so it is clear what variation defines rho 
        groups = mkt_data2.groupby(['market_ids', 'nesting_ids'])
        #mkt_data2['demand_instruments'+str(demand_instruments.shape[1])] = groups['shares'].transform(np.size)
        # get count of number of columns in mkt_data2 containing "demand_instruments"
        ct = sum(1 for col in mkt_data2.columns if 'demand_instruments' in col)
        mkt_data2['demand_instruments'+str(ct)] = groups['shares'].transform(np.size)
        rho_guess = 0.8
        #rho_bounds = (0.1,0.9)

    # set up problem 
    agent_formulation = pyblp.Formulation(str_d_formulation)
    problem = pyblp.Problem(product_formulations, mkt_data2, agent_formulation, agent_data)

    # solve
    optim = pyblp.Optimization('l-bfgs-b',{'gtol': 1e-8})
    iter = pyblp.Iteration('squarem',{'atol': 1e-10}) # using squarem acceleration method
    if parallel:
        with pyblp.parallel(n_nodes_par,use_pathos=True):
            results_demo = problem.solve(sigma = initial_sigma, pi = initial_pi, 
                                         sigma_bounds = (sigma_lb, sigma_ub),
                                         rho = rho_guess,
                                         rho_bounds = rho_bounds,
                                         micro_moments = micro_moments,optimization=optim,iteration=iter,
                                         pi_bounds = pi_bds,
                                         method = gmm_rounds
                                         )
    else:
        results_demo = problem.solve(sigma = initial_sigma, pi = initial_pi, 
                                     sigma_bounds = (sigma_lb, sigma_ub),
                                     rho = rho_guess, 
                                     rho_bounds = rho_bounds,
                                     micro_moments = micro_moments,optimization=optim,iteration=iter,
                                     pi_bounds = pi_bds,
                                     method = gmm_rounds)

    # save results
    df_rand_coeffs = pd.DataFrame({'param':results_demo.beta_labels,
                            'value':results_demo.beta.flatten(),
                            'se': results_demo.beta_se.flatten()})
    df_sigma = pd.DataFrame({'param': ['sigma_' + s for s in results_demo.sigma_labels],
                             'value': np.diagonal(results_demo.sigma).flatten(),
                             'se': np.diagonal(results_demo.sigma_se).flatten()})
    
    pi_labels = []
    for i in results_demo.sigma_labels:
        for j in results_demo.pi_labels:
            pi_labels.append(i+'_'+j)

    df_pi = pd.DataFrame({'param': pi_labels,
                             'value': results_demo.pi.flatten(),
                             'se': results_demo.pi_se.flatten()})                         
    df_rand_coeffs_demo = pd.concat([df_rand_coeffs,df_sigma,df_pi],ignore_index=True)
    if model == 'rc_nl' or model == 'rc_nl_moments':
        df_rho = pd.DataFrame({'param': 'rho',
                               'value':results_demo.rho.flatten(),
                               'se': results_demo.rho_se.flatten()})
        df_rand_coeffs_demo = pd.concat([df_rand_coeffs_demo,df_rho],ignore_index=True)
    results = results_demo
if model == 'nested_logit':
    # run nested logit with corrected prices
    mkt_data2 = mkt_data.copy()
    # create nesting id where 1 = non-electric, 2 = electric (BEV and PHEV)
    mkt_data2['nesting_ids'] = 1 + mkt_data2['electric'] + mkt_data2['phev']
    #mkt_data2['nesting_ids'] = 1 + mkt_data2['electric'] + mkt_data2['phev'] + mkt_data2['hybrid']

    # put all inside goods in the same nest
    #mkt_data2['nesting_ids'] = 1

    # construct group size instrument
    groups = mkt_data2.groupby(['market_ids', 'nesting_ids'])
    # get count of number of columns in mkt_data2 containing "demand_instruments"
    ct = sum(1 for col in mkt_data2.columns if 'demand_instruments' in col)
    mkt_data2['demand_instruments'+str(ct)] = groups['shares'].transform(np.size)
    #mkt_data.loc[mkt_data.model_year == 2017,['shares','market_ids']].groupby('market_ids').sum()
    #mkt_data2.loc[mkt_data2.model_year == 2017,['shares','market_ids']].groupby(['market_ids']).sum()
    #mkt_data2.shares = mkt_data2.shares*4

    nl_formulation = pyblp.Formulation('0+ prices + dollar_per_mile:ice + hybrid + diesel + electric + phev+ + log_hp_weight + wheelbase + doors + range_elec + C(make) + C(drivetype) + C(bodytype)')
    problem = pyblp.Problem(nl_formulation, mkt_data2)
    optim = pyblp.Optimization('l-bfgs-b')
    nl_results = problem.solve(rho=(0.01),optimization = optim)
    #logit_problem = pyblp.Problem(nl_formulation, mkt_data)
    #logit_results = logit_problem.solve()
    if nl_results.converged == False:
        print('nested logit model failed to converge')
    else:
        # save results
        df_nl = pd.DataFrame({'param':nl_results.beta_labels,
                            'value':nl_results.beta.flatten(),
                            'se':nl_results.beta_se.flatten()})
        df_rho = pd.DataFrame({'param': 'rho',
                            'value':nl_results.rho.flatten(),
                            'se': nl_results.rho_se.flatten()})
        df_nl = pd.concat([df_nl,df_rho],ignore_index = True)
        results = nl_results
######################################################
# set up simplest DeGroote and Verboven dynamic model#
######################################################
if model == 'DGV':
    mkt_data_dyn = mkt_data.copy()
    # calculate log difference in shares
    mkt_data_dyn['ln_sj_s0'] = np.log(mkt_data_dyn['shares']/mkt_data_dyn['outside_share'])
    mkt_data_dyn['cons'] = 1
    mkt_data_dyn['dyn_cons'] = 1
    # calculate future prices
    mkt_data_dyn['dyn_prices'] = mkt_data_dyn.dyn_msrp - mkt_data_dyn.dyn_fed_tax_credit

    #exog_vars = ['dollar_per_mile','electric']
    #exog_vars = ['cons','dollar_per_mile','electric','phev','hybrid','diesel','curbwt']
    exog_vars = ['dollar_per_mile','electric','phev','hybrid','diesel','curbwt','wheelbase','C(make)','C(drivetype)','C(bodytype)','doors','range_elec']
    #exog_vars = ['cons','dollar_per_mile','electric','phev','hybrid','diesel','curbwt','C(drivetype)','C(vehicletype)','C(make)']

    #exog_vars = ['cons','dollar_per_mile','electric','phev','hybrid','diesel','curbwt','C(drivetype)','C(vehicletype)']

    # identify categorical variables in list of exog vars; generate relevant indicators
    exog_vars_full = exog_vars.copy()
    mkt_data_dyn_append = mkt_data_dyn.copy() # making this copy and then reassigning the main df because it was yelling about fragmented df
    if any("C(" in x for x in exog_vars):
        categorical_vars = [x[2:len(x)-1] for x in exog_vars if "C(" in x]
        exog_vars_full = [x for x in exog_vars_full if "C(" not in x]
        for c in categorical_vars:
            c_levels = sorted(mkt_data_dyn_append[c].unique()) # by default, take first obs to be omitted category
            # if there's no constant, one less category needs to be omitted (let it be from the first categorical variable or the make)
            if ("cons" not in exog_vars):
                if ('make' in categorical_vars):
                    start_range = 0 if c == 'make' else 1
                else:
                    start_range = 0 if c == categorical_vars[0] else 1
            else:
                start_range = 1
            for l in range(start_range,len(c_levels)):
                l_val = c_levels[l]
                l_col = c + '[' + l_val + ']'
                mkt_data_dyn_append[l_col] = (mkt_data_dyn_append[c] == l_val)*1
                #update list of exogenous vars
                exog_vars_full.extend([l_col])
                #dynamic version of the col
                dyn_l_col = 'dyn_' + l_col
                dyn_c = 'dyn_' + c
                mkt_data_dyn_append[dyn_l_col] = (mkt_data_dyn_append[dyn_c] == l_val)*1
    mkt_data_dyn = mkt_data_dyn_append.copy()

    dyn_exog_vars = ['dyn_'+ s for s in exog_vars_full]
    #dyn_exog_vars = ['dyn_'+ s for s in exog_vars_full if s != 'cons'] # getting rid of dynamic constant for singular matrix issues?
    endog_vars = ['prices']

    # set up matrices of relevant variables and instruments
    # this is useful both to calculate the non-dynamic IV estimates AND so that some matrix multiplication happens outside the minimization

    # matrix math with 2SLS weight matrix
    Z = np.concatenate((demand_instruments,mkt_data_dyn[exog_vars_full].to_numpy()),axis=1)
    x_vars = ['prices']
    x_vars.extend(exog_vars_full)
    X = mkt_data_dyn[x_vars].to_numpy()
    Y = np.reshape(mkt_data_dyn['ln_sj_s0'].to_numpy(),(-1,1))

    W = np.linalg.inv(np.matmul(Z.T,Z))
    X_prime_Z = np.matmul(X.T,Z)
    Z_prime_X= np.matmul(Z.T,X)
    Z_prime_Y = np.matmul(Z.T,Y)
    term_1 = np.linalg.inv(np.matmul(np.matmul(X_prime_Z,W),
                        Z_prime_X))
    term_2 = np.matmul(np.matmul(X_prime_Z,W),
                    Z_prime_Y)
    coeffs = np.matmul(term_1,term_2) # these are the IV coefficients

    # can check results with IV package
    if False:
        y = mkt_data_dyn['ln_sj_s0'] # dependent variable
        endog = mkt_data_dyn['prices']
        exog = mkt_data_dyn[exog_vars_full]
        instruments = mkt_data_dyn[col_names]
        IV_results = IV2SLS(y, exog, endog, instruments).fit()
        #formula = 'ln_sj_s0 ~ dollar_per_mile + electric + phev + hybrid + diesel + curbwt + C(drivetype) + C(vehicletype) + [prices ~ demand_instruments0 + demand_instruments1 + demand_instruments2 + demand_instruments3 + demand_instruments4 + demand_instruments5]'
        #formula = 'ln_sj_s0 ~ electric + [prices ~ demand_instruments0 + demand_instruments1 + demand_instruments2 + demand_instruments3 + demand_instruments4 + demand_instruments5]'
        #IV_results_formula = IV2SLS.from_formula(formula, mkt_data_dyn).fit()

    # additional calculations needed for GMM
    # get n for sample moment
    n = Y.shape[0]

    # longer versions of matrices that include x_{t+1} for dynamics
    Z_dyn = np.concatenate((demand_instruments,mkt_data_dyn[exog_vars_full + dyn_exog_vars].to_numpy()),axis=1)
    x_vars_dyn = ['dyn_prices']
    x_vars_dyn.extend(dyn_exog_vars)
    X_dyn = mkt_data_dyn[x_vars_dyn].to_numpy()
    #W_dyn = np.linalg.inv(np.matmul(Z_dyn.T,Z_dyn)) # this doesn't work because it's a singular matrix
    Z_dyn_prime_Y = np.matmul(Z_dyn.T,Y)
    Z_dyn_prime_X = np.matmul(Z_dyn.T,X)
    Z_dyn_prime_X_dyn = np.matmul(Z_dyn.T, X_dyn)

    Z_prime_X_dyn = np.matmul(Z.T,X_dyn)

    S_t_plus_1 = np.reshape(mkt_data_dyn['dyn_share'].to_numpy(),(-1,1))
    Z_dyn_prime_S_t_plus_1 = np.matmul(Z_dyn.T, S_t_plus_1)
    Z_prime_S_t_plus_1 = np.matmul(Z.T, S_t_plus_1)

    # specify whether GMM is constraining beta to be 0 (primarily as a check of results)
    #bln_beta_zero = True
    bln_beta_zero = False

    # save weight matrix 
    if bln_beta_zero:
        weight_mat_orig = W.copy()
    else:
        #weight_mat_orig = np.identity(Z_dyn.shape[1]) # using identity matrix for weight matrix b/c Z'Z is no longer invertible
        #weight_mat_orig = W_dyn.copy()
        weight_mat_orig = W.copy()
    weight_mat = weight_mat_orig.copy()    
    #weight_mat = np.identity(weight_mat_orig.shape[0])

    # gmm functions
    #either returns g_bar or a vector of g_is depending on bln_bar
    def calc_moment(params_guess, bln_bar = True):
        if bln_beta_zero:
            g_bar = (Z_prime_Y - np.matmul(Z_prime_X,params_guess))/n
            if ~bln_bar:
                error = Y - np.matmul(X,params_guess)
                g = Z * error
        else:
            beta_guess = params_guess[0,0]
            other_params_guess = params_guess[1:params_guess.shape[0]]
            # Y = X gamma - X_t+1,1 gamma beta + beta s_1 + e
            # error = Y - X gamma + X_t+1,1 gamma beta - beta s_1
            g_bar = (Z_prime_Y - np.matmul(Z_prime_X,other_params_guess) + beta_guess * np.matmul(Z_prime_X_dyn,other_params_guess) - beta_guess * Z_prime_S_t_plus_1)/n
            if ~bln_bar:
                error = Y - np.matmul(X,other_params_guess) + beta_guess * np.matmul(X_dyn, other_params_guess) - beta_guess * S_t_plus_1
                #g = Z_dyn * error
                g = Z * error
        if bln_bar:
            obj_return = g_bar
        else:
            obj_return = g, error
        return obj_return

    # calc minimand calculates g = 1/n Z'e, g'Wg *n, returns the latter
    # note: it is necessary to reshape params_guess or else it will get flattened by the minimize function
    def calc_minimand(params_guess):
        params_guess = np.reshape(params_guess,(-1,1))
        g_bar = calc_moment(params_guess)
        J = np.matmul(np.matmul(g_bar.T, weight_mat),g_bar) * n 
        # print(params_guess)
        #print(J[0,0])
        if bln_weight_identity or bln_optimal_weight:
            J = J/1e5 # scale the object smaller because using identity matrix instead of (Z'Z)^{-1}
        return J[0,0]

    # slightly more human understandable version of calc_minimand
    # currently only set up to work if beta = 0
    def alt_minimand(params_guess):
        params_guess = np.reshape(params_guess,(-1,1))
        error = Y - np.matmul(X,params_guess)
        g_bar = np.matmul(Z.T,error)/n
        J_alt = np.matmul(np.matmul(g_bar.T, weight_mat),g_bar) *n
        # print(params_guess)
        #print(J_alt[0,0])
        return J_alt[0,0]

    # create dataframe of results as long as optimization succeeds
    def store_opt_results(results,ignore_make = True): 
        if results.success == True:
            if bln_beta_zero:
                labels_x = ['prices']
            else:
                labels_x = ['beta','prices']
            labels_x.extend(exog_vars_full)
            df_dyn = pd.DataFrame({'param': labels_x,
                                'value': results.x})
            if ignore_make:
                df_dyn = df_dyn.loc[~df_dyn.param.str.startswith('make[')]
        else:
            print('Optimization failed. Please see res object.')
            df_dyn = pd.DataFrame()
        return df_dyn

    # guess of parameters
    gamma_guess = np.ones(shape=(len(exog_vars_full),1))*0.05
    beta_guess = np.array([0.5])
    alpha_guess = np.array([-0.01])
    if bln_beta_zero:
        params_guess = np.vstack([alpha_guess, gamma_guess])
    else:
        params_guess = np.vstack([np.vstack([beta_guess, alpha_guess]), gamma_guess]) 

    no_bnd = [(None,None)]
    if bln_beta_zero:
        params_guess = coeffs*1
        bnds = no_bnd * len(params_guess)
    else:
        params_guess[1:(params_guess.shape[0])] = coeffs
        bnds = no_bnd * len(params_guess)
        bnds[0] = (0,1)
    # check if weight matrix is identity
    bln_weight_identity = np.allclose(weight_mat,np.identity(weight_mat.shape[0]))
    bln_optimal_weight= False
    #res = minimize(calc_minimand, params_guess, method='Nelder-Mead',options={'disp': True})
    #res = minimize(calc_minimand, params_guess, method='Nelder-Mead',options={'disp': True,'fatol': 1e-14,'maxfev':1e9})
    # decreasing fatol by another order of magnitude doesn't seem to change outcome
    res = minimize(calc_minimand, params_guess, method='L-BFGS-B',bounds = bnds, options={'disp': True, 'maxfun': 2e5})
    # del res

    df_dyn_1 = store_opt_results(res,ignore_make = True)

    # code to poke around results
    if False:
        res.success
        res.message
        res.x
        calc_minimand(coeffs)
        #calc_minimand(np.squeeze(coeffs))
        calc_minimand(params_guess)
        calc_minimand(res.x)
        alt_minimand(coeffs)
        alt_minimand(params_guess)
        alt_minimand(res.x)

    # GMM stage 2
    # reshape res.x
    stage1_res = np.reshape(res.x,(-1,1))
    g, error = calc_moment(stage1_res,bln_bar = False)
    g_bar = calc_moment(stage1_res)
    # note: PyBLP uses gg' instead of (g-g_bar)(g-g_bar)'
    diff = g - g_bar.T
    W2_inv = np.matmul(diff.T, diff)/n
    W2 = np.linalg.inv(W2_inv)

    weight_mat = W2.copy()
    bln_optimal_weight = True
    # double check this is doing what we think it's doing
    if False:
        mat_sum = np.zeros(weight_mat.shape)
        for i in range(0,n):
            mat = np.reshape(diff[i],(-1,1))
            mat_mat = mat @ mat.T
            mat_sum += mat_mat
            if (i % 1000 == 0):
                print(i)
        mat_test = np.linalg.inv(mat_sum/n)

    # run a second time
    params_guess=res.x
    res2 = minimize(calc_minimand, params_guess, method='L-BFGS-B',bounds = bnds, options={'disp': True})

    df_dyn_2 = store_opt_results(res2,ignore_make = True)

    # calculate standard errors with efficient weight matrix
    #Q_hat = np.matmul(Z.T,X)/n
    # first column of Q is derivative w/r/t beta, next are derivs w/r/t alpha, gamma
    stage2_res = np.reshape(res2.x,(-1,1))
    beta = stage2_res[0]
    alpha_gamma = stage2_res[1:stage2_res.shape[0]]
    dG_dBeta = np.matmul(Z_prime_X_dyn,alpha_gamma) - Z_prime_S_t_plus_1
    dG_dGamma = -Z_prime_X + beta * Z_prime_X_dyn
    Q_hat = np.hstack((dG_dBeta,dG_dGamma))/n

    # calculate omega_hat
    # note: this should equal the inverse of the efficient weight matrix if calculated with updated params
    g, error = calc_moment(stage2_res,bln_bar = False)
    g_bar = calc_moment(stage2_res)

    mat_omega_sum = np.zeros(weight_mat.shape)
    for i in range(0,n):
        z_i = np.reshape(Z[i],(-1,1))
        z_i_z_i_prime = z_i @ z_i.T
        e_i_2 = error[i]**2
        mat_omega_sum += z_i_z_i_prime * e_i_2  
        #if (i % 1000 == 0):
        #    print(i)
    mat_omega = mat_omega_sum/n - g_bar @ g_bar.T

    var_beta_inv = (Q_hat.T @ np.linalg.inv(mat_omega)) @ Q_hat
    var_beta = np.linalg.inv(var_beta_inv)

    se_asymptotic = np.sqrt(np.diagonal(var_beta)/n)

    # add SEs to main results
    if bln_beta_zero:
        labels_x = ['prices']
    else:
        labels_x = ['beta','prices']
    labels_x.extend(exog_vars_full)
    df_se = pd.DataFrame({'param': labels_x,
                        'se': se_asymptotic})
    df_se = df_se.loc[~df_se.param.str.startswith('make[')]
    df_dyn_results = pd.merge(df_dyn_2,df_se,how='left',on='param')

################
# save results #
################
if save:
    str_time = time.strftime("%Y_%m_%d_%H%M",time.localtime())
    str_results_folder = output_folder+ '/demand_results/results_'+version+'_'+str_time+'/'
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
        df_logit.to_csv(str_results_folder+'demand_params.csv',index = False)
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
    pickle.dump(results,open(str_results_folder+"/pickled_demand.pickle",'wb'))
    pickle.dump(mkt_data,open(str_results_folder+"/pickled_mkt_data.pickle",'wb'))
    if agent_data is not None:
        pickle.dump(agent_data,open(str_results_folder+"/pickled_agent_data.pickle",'wb'))
    if integration is not None:
        pickle.dump(integration,open(str_results_folder+"/pickled_integration_data.pickle",'wb'))


# test = pickle.load(open(str_results_folder+"/pickled_demand.pickle", "rb" ) )

