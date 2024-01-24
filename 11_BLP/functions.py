from typing import Any, Callable, Dict, Hashable, List, Optional, Sequence, TYPE_CHECKING, Tuple
import pyblp
import numpy as np
import pandas as pd
import pickle
import pathlib
import sys

from pyblp.primitives import Agents
from pyblp.utilities.basics import (
    Array, Bounds, Error, Mapping, RecArray, SolverStats, format_number, format_seconds, format_table, generate_items,
    get_indices, output, output_progress, update_matrices
)
if pyblp.__version__ != '1.0.0' and pyblp.__version__ != '1.1.0':
    from pyblp.markets.results_market import ResultsMarket
else:
    from pyblp.markets.economy_results_market import EconomyResultsMarket
import os

# function to check if this code is running on the cluster
def check_cluster():
    if(os.getcwd() == '/Users/Stephanie/Dropbox (YSE)/SW_Automaker_Strategies/AutomakerStrategiesCode/model'):
        bln_cluster = False
    else:
        bln_cluster = True
    return(bln_cluster)

# calculate capital delta matrices
def calculate_cap_delta(results,agent_data = None,integration=None,delta_type = 'orig'):
    # define variables
    firm_ids = results.problem._coerce_optional_firm_ids(firm_ids=None)
    market_ids = results._select_market_ids()
    ownership = results.problem._coerce_optional_ownership(ownership = None, market_ids = market_ids)
    market_args = [firm_ids, ownership]
    fixed_args = ()
    # structure or construct different agent data
    if agent_data is None and integration is None:
        agents = results.problem.agents
        agents_market_indices = results.problem._agent_market_indices
    else:
        agents = Agents(results.problem.products, results.problem.agent_formulation, agent_data, integration)
        agents_market_indices = get_indices(agents.market_ids)
    if(pyblp.__version__ == '1.0.0' or pyblp.__version__ == '1.1.0'):
        # from economy_results.py
        def market_factory(s: Hashable) -> tuple:
            """Build a market along with arguments used to compute arrays."""
            indices_s = results._economy._product_market_indices[s]
            market_s = EconomyResultsMarket(
                results._economy, s, results._parameters, results._sigma, results._pi, results._rho, results._beta, results._gamma,
                results._delta, results._data_override, agents_override=agents[agents_market_indices[s]]
            )
            if market_ids.size == 1:
                args_s = market_args
            else:
                args_s = [None if a is None else a[indices_s] for a in market_args]
            return (market_s, *fixed_args, *args_s)
    else:
        # function to produce ResultsMarket instance from market_id
        # this market factory code comes from problem_results.py
        # differs betwn pyblp versions
        if(pyblp.__version__  == '0.13.0'):
            def market_factory(s: Hashable) -> tuple:
                """Build a market along with arguments used to compute arrays."""
                indices_s = results.problem._product_market_indices[s]
                market_s = ResultsMarket(
                    results.problem, s, results._parameters, results.sigma, results.pi, results.rho, results.beta, results.gamma, results.delta,
                    agents_override=agents[agents_market_indices[s]]
                )
                if market_ids.size == 1:
                    args_s = market_args
                else:
                    args_s = [None if a is None else a[indices_s] for a in market_args]
                return (market_s, *fixed_args, *args_s)
        else:
            def market_factory(s: Hashable) -> tuple:
                """Build a market along with arguments used to compute arrays."""
                indices_s = results.problem._product_market_indices[s]
                market_s = ResultsMarket(
                    results.problem, s, results._parameters, results.sigma, results.pi, results.rho, results.beta, results.gamma, results.delta,
                    results._moments, agents_override=agents[agents_market_indices[s]]
                )
                if market_ids.size == 1:
                    args_s = market_args
                else:
                    args_s = [None if a is None else a[indices_s] for a in market_args]
                return (market_s, *fixed_args, *args_s)

    # calculate capital_delta for each market (using functions in market.py compute_eta() for pre 1.0.0 pyblp, using functions in economy_results_market.py for 1.0.0)
    lst_cap_delta = [None] * len(market_ids)
    for mkt in range(len(market_ids)):
        res_mkt = market_factory(market_ids[mkt])
        res_mkt = res_mkt[0]
        ownership_matrix = res_mkt.get_ownership_matrix(firm_ids,ownership)
        delta = res_mkt.delta
        if(delta_type == 'orig'):
            utility_derivatives = res_mkt.compute_utility_derivatives('prices')
        #elif(delta_type == 'eff'):
        #    utility_derivatives = compute_eff_derivatives(results,res_mkt,'gpktm')
        probabilities, conditionals = res_mkt.compute_probabilities(delta)
        jacobian = res_mkt.compute_shares_by_variable_jacobian(utility_derivatives, probabilities, conditionals)
        capital_delta = -ownership_matrix * jacobian #note the negative sign!!!!
        lst_cap_delta[mkt] = capital_delta
    return lst_cap_delta

# function that gets credit requirement data, recalculates ZEV and GHG credits per vehicle
def get_credit_reqs(mkt_data,str_loc):
    # read in ZEV policy requirements
    str_policy_req = str_loc + 'data/intermediate/ZEV_manufacturer_requirements.csv'
    policy_req = pd.read_csv(str_policy_req)

    # read in GHG policy adjustments
    str_policy_ghg = str_loc + 'data/intermediate/ghg_credit_adjustments.csv'
    policy_ghg = pd.read_csv(str_policy_ghg)

    # rename year to model_year
    policy_req = policy_req.rename(columns={'year':'model_year'})
    policy_ghg = policy_ghg.rename(columns={'year':'model_year'})

    # merge with market data
    mkt_data = pd.merge(mkt_data,policy_req,how='left',on = ['model_year','oem'])
    # zero out credit requirements for states not in ZEV policy
    mkt_data.loc[mkt_data.zev == 0,'credit_req'] = 0
    mkt_data.loc[mkt_data.zev == 0,'bev_credit_req'] = 0
    mkt_data.loc[mkt_data.zev == 0,'t'] = 0

    # for now, treating credit req as the amount that would lead to exact compliance in each year
    mkt_data['credit_req'] = mkt_data['t']
    mkt_data['net_credits'] = mkt_data.zev_credits - mkt_data.credit_req

    # update ghg credit reqs to amount that would lead to exact compliance in each year
    mkt_data = pd.merge(mkt_data,policy_ghg,how='left',on=['model_year'])
    # may want to zero out credits for luxury brands

    if 'addl_ghg_credits' not in mkt_data:
        mkt_data['addl_ghg_credits'] = 0
    mkt_data['ghg_std_new'] = mkt_data.ghg_std*(1+mkt_data.adj) - mkt_data.addl_ghg_credits
    mkt_data['ghg_credit_new'] = (mkt_data.ghg_std_new - mkt_data.emis) * ((mkt_data.car * 195264)+ (1-mkt_data.car)*225865)/1000000*mkt_data.multiplier
    return mkt_data


# create matrix indicating whether each product offered in each market
# also create matrix of market-specific xi products, where missing prods get xi = -999
# also create matrix of market-specific dollar-per-mile, where missing prods get dpm = -999
# n_prods x n_markets
def get_mats_avail_xi_dpm_incentive(dpm_col, incentive_col,vec_prods,mkt_sim):
    vec_mkts,vec_mkt_size = small_mkt_objects(mkt_sim)
    mat_mkt_prod = np.zeros((len(vec_prods),len(vec_mkts)))
    mat_shares_orig = np.zeros((len(vec_prods),len(vec_mkts)))
    mat_xi = np.zeros((len(vec_prods),len(vec_mkts)))
    mat_dpm = np.zeros((len(vec_prods),len(vec_mkts)))
    mat_incentives = np.zeros((len(vec_prods),len(vec_mkts)))
    for i in range(0,len(vec_mkts)):
        mkt = vec_mkts[i]
        vec_mkt_prods = mkt_sim.loc[mkt_sim.market_ids == mkt,'product_ids'].to_numpy()
        # indicator for product in market
        vec_mkt_ind = np.isin(vec_prods,vec_mkt_prods)*1
        mat_mkt_prod[:,i] = vec_mkt_ind
        # xi
        for p in range(0,len(vec_prods)):
            prod = vec_prods[p]
            if prod in vec_mkt_prods:
                mat_xi[p,i] = mkt_sim.loc[(mkt_sim.market_ids == mkt) & (mkt_sim.product_ids == prod),'xi']
                mat_dpm[p,i] = mkt_sim.loc[(mkt_sim.market_ids == mkt) & (mkt_sim.product_ids == prod),dpm_col]
                mat_incentives[p,i] = mkt_sim.loc[(mkt_sim.market_ids == mkt) & (mkt_sim.product_ids == prod),incentive_col]
                mat_shares_orig[p,i] = mkt_sim.loc[(mkt_sim.market_ids == mkt) & (mkt_sim.product_ids == prod),'shares']
            else:
                mat_xi[p,i] = -999
                mat_dpm[p,i] = -999
                mat_incentives[p,i] = -999
                mat_shares_orig[p,i] = 0
    return mat_mkt_prod, mat_xi, mat_dpm, mat_incentives, mat_shares_orig

# create products x market x consumer matrix of individual-specific components of utility
def get_individ_utility(results,product_chars,mat_mkt_prod,mkt_sim,agent_sim,bln_agent,coef_price):
    if bln_agent:
        vec_mkts,vec_mkt_size = small_mkt_objects(mkt_sim)
        # create n_product x n_market x n_consumer matrix
        n_agents = len(agent_sim.loc[agent_sim.state == agent_sim.state[0]])
        mat_prod_mkt_agent = np.zeros((len(product_chars),len(vec_mkts),n_agents))
        mat_coef_price = np.zeros((len(product_chars),len(vec_mkts),n_agents))
        # add sigma components
        for i in range(0,len(results.sigma)):
            s_name = results.sigma_labels[i]
            s_var = results.sigma[i][0]
            if s_name == 'prices':
                if s_var != 0:
                    print('warning: there is a random coefficient on prices; code not set up for this')
                else:
                    coef_price = calc_pi_components(mat_coef_price,results,vec_mkts,agent_sim,n_agents,i,mat_mkt_prod)
            else:
                # calculate product x market availability * characteristic i
                mat_prod_i = np.reshape(product_chars[s_name].to_numpy(),(-1,1)) * mat_mkt_prod
                if s_var != 0:
                    # convert agent_data into n_market x n_agent matrix
                    for j in range(0,len(vec_mkts)):
                        mkt = vec_mkts[j]
                        mat_agent_mkt = np.reshape(agent_sim.loc[agent_sim.market_ids == mkt, 'nodes'+str(i)].to_numpy()*s_var,(-1,1))
                        mat_prod_mkt = np.reshape(mat_prod_i[:,j],(-1,1))
                        mat_agent_prod_ij = mat_prod_mkt @ mat_agent_mkt.T
                        mat_prod_mkt_agent[:,j,:] = mat_prod_mkt_agent[:,j,:] + mat_agent_prod_ij
                    print('warning: check if there are multiple random coefficients that the weights get ordered correctly')
                # add pi components, which interact with sigma components
                mat_prod_mkt_agent = calc_pi_components(mat_prod_mkt_agent,results,vec_mkts,agent_sim,n_agents,i,mat_prod_i)
                print('warning: check that pi is set up correctly if using more than one demographic interaction')
    else:
        mat_prod_mkt_agent = None
    return mat_prod_mkt_agent, coef_price

def calc_pi_components(mat_prod_mkt_agent,results,vec_mkts,agent_sim,n_agents,i,mat_prod_i):
    for k in range(0,len(results.pi_labels)):
        p_name = results.pi_labels[k]
        p_var = results.pi[i][k]
        if p_var != 0:
            if '/' in p_name:
                vec_split = p_name.split('/')
            else:
                vec_split = None             
            for j in range(0,len(vec_mkts)):
                mkt = vec_mkts[j]
                if not vec_split:
                    mat_agent_mkt = np.reshape(agent_sim.loc[agent_sim.market_ids == mkt, p_name].to_numpy()*p_var,(-1,1))
                else:
                    mat_agent_mkt = np.ones((n_agents,1))*p_var
                    mat_agent_num = np.reshape(agent_sim.loc[agent_sim.market_ids == mkt, vec_split[0]].to_numpy(),(-1,1))
                    mat_agent_denom = np.reshape(agent_sim.loc[agent_sim.market_ids == mkt, vec_split[1]].to_numpy(),(-1,1))
                    mat_agent_mkt = mat_agent_mkt * mat_agent_num / mat_agent_denom
                mat_prod_mkt = np.reshape(mat_prod_i[:,j],(-1,1))
                mat_agent_prod_kj = mat_prod_mkt @ mat_agent_mkt.T
                mat_prod_mkt_agent[:,j,:] = mat_prod_mkt_agent[:,j,:] + mat_agent_prod_kj
    return mat_prod_mkt_agent

def get_agent_weights(agent_sim,mkt_sim):
    vec_mkts,vec_mkt_size = small_mkt_objects(mkt_sim)

    mat_agent_weights = np.zeros((len(vec_mkts),len(agent_sim.loc[agent_sim.state == agent_sim.state[0]])))
    for m in range(0,len(vec_mkts)):
        mkt = vec_mkts[m]
        mat_agent_weights[m,:] = agent_sim.loc[agent_sim.market_ids == mkt, 'weights'].to_numpy()
    return mat_agent_weights


# calculate shares and share derivatives
def get_shares(vec_msrp,mat_incentives,mat_dpm,delta_const,mat_individ_prod_util,bln_agent,agent_weights,coef_price,coef_fe,mat_xi,mat_mkt_prod,state_fes,dict_nl):
    utilities_common = delta_const + coef_fe * mat_dpm + mat_xi + state_fes
    # check if coef_price has 2 dimensions (i.e., is not agent specific)
    if len(coef_price.shape) == 2:
        # add price component to shared utility
        utilities_common = utilities_common + coef_price * (vec_msrp - mat_incentives)
    if bln_agent:
        n_agents = mat_individ_prod_util.shape[2]
        utilities = np.repeat(utilities_common[:,:,np.newaxis],n_agents,axis=2) + mat_individ_prod_util
        # if price component isn't part of utilities_common
        if len(coef_price.shape) == 3:
            # generate individual-specific utility from price
            mat_price = vec_msrp - mat_incentives
            utilities_price = coef_price * np.repeat(mat_price[:,:,np.newaxis],n_agents,axis=2)
            utilities = utilities + utilities_price
        mat_mkt_prod =  np.repeat(mat_mkt_prod[:,:,np.newaxis],n_agents,axis=2)
    else:  
        utilities = utilities_common  
    exp_utilities = np.exp(utilities) * mat_mkt_prod # zero out non-offered products
    probabilities = exp_utilities / (exp_utilities.sum(axis=0) + 1) * mat_mkt_prod # zero out non-offered products
    # note: not allowing agents with differential weights
    if bln_agent:
        agent_weights_expanded = np.transpose(np.dstack([agent_weights]*probabilities.shape[0]),(2,0,1))
        shares = (probabilities * agent_weights_expanded).sum(axis=2)
        #shares = probabilities.sum(axis=2)/n_agents
    else:
        shares = probabilities
    # nested logit share calculation
    # if rho is not None, need to calculate V_ih(j)t
    coef_rho = dict_nl['coef_rho']
    conditionals = None
    if coef_rho is not None:
        vec_grp = dict_nl['vec_grp']
        exp_utilities_rho = np.exp(utilities/(1-coef_rho)) * mat_mkt_prod # zero out non-offered products
        
        # create n_h x n_products matrix of zeros to store group info
        mat_h = np.zeros((len(np.unique(vec_grp)),utilities.shape[0]))
        n_markets= utilities.shape[1]
        if bln_agent:
            # create n_h x n_markets x n_agents matrix of zeros to store V_iht
            mat_V_iht = np.zeros((len(np.unique(vec_grp)),n_markets,n_agents))
            # create n_h x n_markets x n_agents matrix of sum (exp_utilities_rho)
            mat_sum_exp_utilities_rho = np.zeros((len(np.unique(vec_grp)),n_markets,n_agents))
            for ih in range(0,len(np.unique(vec_grp))):
                h = np.unique(vec_grp)[ih]
                # get index of values in vec_grp that match h
                idx_h = np.where(vec_grp == h)[0]
                # store in mat_h
                mat_h[ih,idx_h] = 1
                # get sum exp_utitilies_rho for group h
                sum_exp_utilities_rho = exp_utilities_rho[idx_h,:,:].sum(axis=0)
                # store in mat_sum_exp_utilities_rho
                mat_sum_exp_utilities_rho[ih,:,:] = sum_exp_utilities_rho
                # calculate (1-rho) * log(sum(exp_utilities_rho))
                full_V_ihjt = (1-coef_rho) * np.log(sum_exp_utilities_rho)
                # store in mat_V_iht
                mat_V_iht[ih,:,:] = full_V_ihjt
            # calculate shares
            sum_V_iht = 1+ np.exp(mat_V_iht).sum(axis=0)

            probabilities = np.zeros(utilities.shape)
            conditionals = np.zeros(utilities.shape)
            for ih in range(0,len(np.unique(vec_grp))):
                h = np.unique(vec_grp)[ih]
                # separately calculating components of choice probabilities (see e.g., p. 17 in PyBLP docs)
                # get numerator exp(V_ijt/(1-rho)) for all products in group h
                # repeat 1d matrix mat_h[ih,:] 14 times and then repeat 5000 times
                mat_h_repeat = np.repeat(np.reshape(mat_h[ih,:],(-1,1)),n_markets,axis=1)
                mat_h_repeat = np.repeat(mat_h_repeat[:,:,np.newaxis],n_agents,axis=2)
                num_h = exp_utilities_rho * mat_h_repeat
                # get first denominator exp(V_ih(j)t/(1-rho)) for group h
                denom_1_h = np.exp(mat_V_iht[ih,:,:]/(1-coef_rho))
                # get second term: exp(V_h(j)t)/(1+ sum_h V_h(j)t)
                frac_2_h =  np.exp(mat_V_iht[ih,:,:])/sum_V_iht
                probabilities = probabilities + num_h * (frac_2_h/denom_1_h)
                conditionals = conditionals + num_h/mat_sum_exp_utilities_rho[ih,:,:]

            # replace nan in probabilities with 0
            probabilities[np.isnan(probabilities)] = 0
            conditionals[np.isnan(conditionals)] = 0

            shares = (probabilities * agent_weights_expanded).sum(axis=2)
        else:
            # create n_h x n_markets matrix of zeros to store V_ht
            mat_V_ht = np.zeros((len(np.unique(vec_grp)),n_markets))
            # create n_h x n_markets matrix of sum (exp_utilities_rho)
            mat_sum_exp_utilities_rho = np.zeros((len(np.unique(vec_grp)),n_markets))
            for ih in range(0,len(np.unique(vec_grp))):
                h = np.unique(vec_grp)[ih]
                # get index of values in vec_grp that match h
                idx_h = np.where(vec_grp == h)[0]
                # store in mat_h
                mat_h[ih,idx_h] = 1
                # get sum exp_utitilies_rho for group h
                sum_exp_utilities_rho = exp_utilities_rho[idx_h,:].sum(axis=0)
                # store in mat_sum_exp_utilities_rho
                mat_sum_exp_utilities_rho[ih,:] = sum_exp_utilities_rho
                # calculate V_h(j)t (1-rho) * log(sum(exp_utilities_rho))
                full_V_hjt = (1-coef_rho) * np.log(sum_exp_utilities_rho)
                # store in mat_V_ht
                mat_V_ht[ih,:] = full_V_hjt
            # calculate shares
            sum_V_ht = 1 + np.exp(mat_V_ht).sum(axis=0)

            probabilities = np.zeros(utilities.shape)
            conditionals = np.zeros(utilities.shape)
            for ih in range(0,len(np.unique(vec_grp))):
                h = np.unique(vec_grp)[ih]
                # separately calculating components of choice probabilities (see e.g., p. 17 in PyBLP docs)
                # get numerator exp(V_ijt/(1-rho)) for all products in group h
                num_h = exp_utilities_rho * np.repeat(np.reshape(mat_h[ih,:],(-1,1)),n_markets,axis=1)
                # get first denominator exp(V_ih(j)t/(1-rho)) for group h
                denom_1_h = np.exp(mat_V_ht[ih,:]/(1-coef_rho))
                # get second term: exp(V_h(j)t)/(1+ sum_h V_h(j)t)
                frac_2_h =  np.exp(mat_V_ht[ih,:])/sum_V_ht
                probabilities = probabilities + num_h * (frac_2_h/denom_1_h)
                conditionals = conditionals + num_h/mat_sum_exp_utilities_rho[ih,:]
            
            shares = probabilities
            # calculate V_h(j)t/sum_V_ht
            #V_frac_2 = np.exp(mat_V_ht)/sum_V_ht
            # calculate exp(V_hjt/(1-rho)
            #exp_V_hjt_rho = np.exp(mat_V_ht/(1-coef_rho))
            # caclulate 1/(exp_V_ihjt_rho) * V_ih(j)t/sum_V_iht
            #V_nonj = 1/exp_V_hjt_rho * V_frac_2

    return utilities, probabilities, shares, conditionals 

def get_dshares(vec_msrp,mat_incentives,mat_dpm,delta_const,mat_individ_prod_util,bln_agent,agent_weights,coef_price,coef_fe,mat_xi,mat_mkt_prod,vec_mkts,mat_diag,state_fes,dict_nl):
    utilities,probabilities,shares,conditionals = get_shares(vec_msrp,mat_incentives,mat_dpm,delta_const,mat_individ_prod_util,bln_agent,agent_weights,coef_price,coef_fe,mat_xi,mat_mkt_prod,state_fes,dict_nl)
    delta_mat = np.empty((len(vec_msrp),len(vec_msrp),len(vec_mkts)))

    coef_rho = dict_nl['coef_rho']

    if bln_agent:
        mat_ones = np.ones((utilities.shape[2],1))
        #mat_agent_weights = agent_weights(vec_mkts, agent_sim)
    for m in range(0,utilities.shape[1]): # loop over markets
        if bln_agent:
            agent_weights_m = agent_weights[m,:]
            probabilities_m = probabilities[:,m,:]
            if len(coef_price.shape)== 3:
                coef_price_m = coef_price[:,m,:]
            else:
                coef_price_m = coef_price
            cap_lamda = np.dot(probabilities_m * coef_price_m * agent_weights_m,mat_ones) * mat_diag
            weighted_deriv = (probabilities_m * coef_price_m * agent_weights_m).T
            cap_gamma = probabilities_m @ weighted_deriv
        else:
            probabilities_m = np.reshape(probabilities[:,m],(-1,1))
            cap_lamda = probabilities_m * coef_price * mat_diag
            weighted_deriv = (probabilities_m * coef_price).T
            cap_gamma = probabilities_m @ weighted_deriv
        
        if coef_rho is not None:
            if bln_agent:
                conditionals_m = conditionals[:,m,:]
            else:
                conditionals_m = np.reshape(conditionals[:,m],(-1,1))
            membership_mat = dict_nl['membership_mat']
            cap_lamda = cap_lamda/(1-coef_rho)
            cap_gamma += coef_rho/(1-coef_rho) * membership_mat * (conditionals_m @ weighted_deriv)

        delta_mat_m = (-cap_gamma + cap_lamda)
        delta_mat[:,:,m] = delta_mat_m
    return delta_mat
# note that this is not yet multiplied by ownership matrix; kept for cross-price elasticity purposes!
# variable naming convention comes from pyblp documentation 3.7 Equilibrium Prices


# calculate objects that depend on shares

# generate n_prod x 1 vector of total sales
def calc_q(vec_msrp,mat_incentives,mat_dpm,delta_const,mat_individ_prod_util,bln_agent,agent_weights,coef_price,coef_fe,mat_xi,mat_mkt_prod,vec_mkt_size,state_fes,dict_nl):
    utilities, probabilities, shares, conditionals = get_shares(vec_msrp,mat_incentives,mat_dpm,delta_const,mat_individ_prod_util,bln_agent,agent_weights,coef_price,coef_fe,mat_xi,mat_mkt_prod,state_fes,dict_nl)
    q = shares @ vec_mkt_size
    return q

# generate n_prod x n_prod delta matrix that's scaled by each market size
def calc_delta_R(delta_mat,mat_ownership,vec_mkt_size):
    delta_R = np.zeros((delta_mat.shape[0],delta_mat.shape[0]))
    for m in range(0,delta_mat.shape[2]):
        delta_mat_m = delta_mat[:,:,m]
        delta_R = delta_R + delta_mat_m * mat_ownership * vec_mkt_size[m]
    return delta_R

# generate n_prod x 1 vector of changes in ZEV credits for a change in price of ith good
def calc_dZEV_credits(delta_mat, mat_ZEV,vec_mkt_size,mat_ownership):
    vec_dcredits = np.zeros((delta_mat.shape[0],1))
    for m in range(0,delta_mat.shape[2]):
        delta_mat_m = delta_mat[:,:,m]
        mkt_d_credits = (vec_mkt_size[m] * mat_ownership * delta_mat_m) @ np.reshape(mat_ZEV[:,m],(-1,1))
        vec_dcredits = vec_dcredits + mkt_d_credits
    return vec_dcredits


# generate new gas/diesel price
# aggregate existing energy prices
def aggregate_state_energy_prices(df_price,col,mkt_sim):
    df = df_price.copy()
    df.state = df.state.str.upper()
    df = df.loc[df.state != 'U.S.']
    df_combined = pd.DataFrame(df.loc[~df.state.isin(mkt_sim.state),].groupby(['year'])[col].mean()).reset_index()
    df_combined['state'] = 'COMBINED'
    df2 = pd.concat([df.loc[df.state.isin(mkt_sim.state)],df_combined]).reset_index()
    df2 = df2.rename(columns={'year':'model_year'})
    return df2


# calculate alternative dollar_per_mile
# note: this approach is based on calculate_cost_per_mile function in vin_cleaning_functions.R
def calc_dollar_per_mile(elec_price_agg,diesel_price_agg,gas_price_agg,df_veh):
    df_veh = pd.merge(df_veh,elec_price_agg,how='left',on=['state','model_year'])
    df_veh = pd.merge(df_veh,diesel_price_agg,how='left',on=['state','model_year'])
    df_veh = pd.merge(df_veh,gas_price_agg,how='left',on=['state','model_year'])
    df_veh['dollar_per_mile_new'] = 0
    # gasoline, hybrid
    df_veh.loc[df_veh.fuel.isin(['gasoline','hybrid']),'dollar_per_mile_new'] = df_veh.loc[df_veh.fuel.isin(['gasoline','hybrid']),'dollar_per_gal_gas'] /df_veh.loc[df_veh.fuel.isin(['gasoline','hybrid']),'combined_mpg2008']
    # electricity
    df_veh.loc[df_veh.fuel == 'electric','dollar_per_mile_new'] = df_veh.loc[df_veh.fuel== 'electric','cent_per_kwh']/100 * 33.7 / df_veh.loc[df_veh.fuel=='electric','combined_mpg2008']
    # phev
    df_veh.loc[df_veh.fuel == 'PHEV','dollar_per_mile_elec_new'] = df_veh.loc[df_veh.fuel == 'PHEV','cent_per_kwh']/100 * 33.7 / df_veh.loc[df_veh.fuel == 'PHEV','mpg_elec']
    df_veh.loc[df_veh.fuel == 'PHEV','dollar_per_mile_gas_new'] = df_veh.loc[df_veh.fuel == 'PHEV','dollar_per_gal_gas']/ df_veh.loc[df_veh.fuel == 'PHEV','mpg_gas']
    df_veh.loc[df_veh.fuel == 'PHEV','dollar_per_mile_new'] = df_veh.loc[df_veh.fuel == 'PHEV','combinedUF'] * df_veh.loc[df_veh.fuel == 'PHEV','dollar_per_mile_elec_new'] + (1- df_veh.loc[df_veh.fuel == 'PHEV','combinedUF'])* df_veh.loc[df_veh.fuel == 'PHEV','dollar_per_mile_gas_new'] 
    # diesel
    df_veh.loc[df_veh.fuel == 'diesel','dollar_per_mile_new'] = df_veh.loc[df_veh.fuel == 'diesel','dollar_per_gal_diesel'] /df_veh.loc[df_veh.fuel == 'diesel','combined_mpg2008']
    # flex fuel (assume gasoline prices)
    df_veh.loc[df_veh.fuel == 'flex fuel','dollar_per_mile_new'] = df_veh.loc[df_veh.fuel == 'flex fuel','dollar_per_gal_gas'] /df_veh.loc[df_veh.fuel == 'flex fuel','combined_mpg2008']
  
    #df_veh.loc[df_veh.fuel == 'flex fuel',['dollar_per_mile','dollar_per_mile_new']]
    # rescale everything
    df_veh['dollar_per_mile_new'] =  df_veh['dollar_per_mile_new']*100
    # drop unnecessary new cols
    df_veh = df_veh.drop(['dollar_per_gal_gas', 'dollar_per_gal_diesel','cent_per_kwh','dollar_per_mile_elec_new','dollar_per_mile_gas_new','index','index_x','index_y'], axis=1)
  
    return df_veh

# calculate net ZEV credits
# note: these will not exactly match net credits on raw data because some vehicles (in particular, the Tesla Model S and X P100D) were dropped due to price
def calc_net_zev_credits(vec_msrp,mat_incentives,mat_ZEV,mat_dpm,delta_const,mat_individ_prod_util,bln_agent,agent_weights,coef_price,coef_fe,mat_xi,mat_mkt_prod,vec_mkt_size,state_fes,dict_nl):
    utilities, probabilities, shares, conditionals = get_shares(vec_msrp,mat_incentives,mat_dpm,delta_const,mat_individ_prod_util,bln_agent,agent_weights,coef_price,coef_fe,mat_xi,mat_mkt_prod,state_fes,dict_nl)
    net_credits = ((shares * mat_ZEV) @ vec_mkt_size).sum()
    return net_credits

#(mkt_sim.loc[(mkt_sim.zev_credits > 0),'shares']*mkt_sim.loc[(mkt_sim.zev_credits > 0),'tot_HH']).sum()
#(mkt_sim['shares']*mkt_sim['tot_HH']*mkt_sim['credit_req']).sum()

# calculate net GHG credits
# note: there will also be mismatch issues here because of vehicles that were dropped
# also note: when GHG emissions/fuel efficiency is endogenized, this will need to use the correct vector of credits
def calc_net_ghg_credits(vec_msrp,mat_incentives,vec_ghg_credits,mat_dpm,delta_const,mat_individ_prod_util,bln_agent,agent_weights,coef_price,coef_fe,mat_xi,mat_mkt_prod,vec_mkt_size,state_fes,dict_nl):
    q = calc_q(vec_msrp,mat_incentives,mat_dpm,delta_const,mat_individ_prod_util,bln_agent,agent_weights,coef_price,coef_fe,mat_xi,mat_mkt_prod,vec_mkt_size,state_fes,dict_nl)
    net_credits = (vec_ghg_credits*q).sum()
    return net_credits

#vec_ghg_credits = np.reshape(product_chars.ghg_credit_new.values,(-1,1))
#calc_net_ghg_credits(vec_msrp,mat_incentives,vec_ghg_credits,mat_dpm,delta_const,coef_price,coef_fe,mat_xi,mat_mkt_prod,vec_mkt_size,state_fes)

# credit price iteration function
if(False):
    credit_type = 'ghg'
    lst_net_credits = lst_net_ghg_credits
    credit_eq = ghg_credit_eq
    credit_price = ghg_credit_price
    lst_credit_price = lst_ghg_credit_price
    grid = grid_ghg
    ub = ub_ghg
    lb = lb_ghg
    quiet = False
def adjust_credit_price(credit_type,lst_net_credits,credit_eq,credit_price,lst_credit_price,grid,ub,lb,mat_dpm,
                        price_cf_hat,mat_incentives,mat_ZEV,vec_ghg_credits,iter,
                        delta_const,mat_individ_prod_util,bln_agent,agent_weights,
                        coef_price,coef_fe,mat_xi,mat_mkt_prod,vec_mkt_size,state_fes,dict_nl,quiet=False):
    if(credit_type == 'zev'):
        net_credits = calc_net_zev_credits(price_cf_hat,mat_incentives,mat_ZEV,mat_dpm,delta_const,mat_individ_prod_util,bln_agent,agent_weights,coef_price,coef_fe,mat_xi,mat_mkt_prod,vec_mkt_size,state_fes,dict_nl)
    else:
        net_credits = calc_net_ghg_credits(price_cf_hat,mat_incentives,vec_ghg_credits,mat_dpm,delta_const,mat_individ_prod_util,bln_agent,agent_weights,coef_price,coef_fe,mat_xi,mat_mkt_prod,vec_mkt_size,state_fes,dict_nl)
    
    lst_net_credits.append(net_credits)
    if(quiet==False):
        print('net '+credit_type+' credits: '+str(net_credits))

    if abs(net_credits) > .02:
        credit_eq = False
        # while we're moving in the "right" direction, just keep making zev_credit_price bigger or smaller
        if ((iter == 1) or (np.sign(net_credits) == np.sign(lst_net_credits[iter-2]))) and (grid == False):
            if net_credits > 0:
                mult = .9
            elif net_credits < 0:
                mult = 1.1
            credit_price*=mult
        # once we start flipping, we just want to search in between recent guesses
        else:
            if grid == False:
                ub = min(ub,max(lst_credit_price[max(0,iter-3):])) 
                lb = max(lb,min(lst_credit_price[max(0,iter-3):]))
            else:
                # check if it got "stuck" and readjust ub and lb
                if abs(lst_net_credits[iter-1] - lst_net_credits[iter-2]) < .05:
                    if net_credits > 0:
                        ub = credit_price
                        lb = credit_price*.95
                    else:
                        ub = credit_price*1.05
                        lb = credit_price
                else: # otherwise keep searching between recent guesses
                    if net_credits > 0: 
                        ub = credit_price
                    else:
                        lb = credit_price
                print("searching over ["+str(lb)+","+str(ub)+"]")
            grid = True
            credit_price = (ub+lb)/2
        lst_credit_price.append(credit_price)
        if(quiet==False):
            print(credit_type+' credit price: '+str(credit_price))
    else:
        credit_eq = True   
    return lst_net_credits,credit_eq,credit_price,lst_credit_price,grid,ub,lb

  
# solve for prices based on FOC
def calc_price(vec_msrp,mat_incentives,zev_credit_price,ghg_credit_price,mat_dpm,delta_const,mat_individ_prod_util,bln_agent,agent_weights,coef_price,coef_fe,mat_xi,mat_mkt_prod,vec_mkts,mat_diag,vec_mkt_size,mat_ownership,mat_ZEV,vec_ghg_credits,mc,state_fes,dict_nl):
    delta_mat = get_dshares(vec_msrp,mat_incentives,mat_dpm,delta_const,mat_individ_prod_util,bln_agent,agent_weights,coef_price,coef_fe,mat_xi,mat_mkt_prod,vec_mkts,mat_diag,state_fes,dict_nl)
    delta_R = calc_delta_R(delta_mat,mat_ownership,vec_mkt_size)
    delta_inv = np.linalg.inv(delta_R)
    vec_dZEV_credits= calc_dZEV_credits(delta_mat,mat_ZEV,vec_mkt_size,mat_ownership)
    q = calc_q(vec_msrp,mat_incentives,mat_dpm,delta_const,mat_individ_prod_util,bln_agent,agent_weights,coef_price,coef_fe,mat_xi,mat_mkt_prod,vec_mkt_size,state_fes,dict_nl)
    price_hat = mc - ghg_credit_price * vec_ghg_credits - delta_inv @ q - delta_inv @ (zev_credit_price * vec_dZEV_credits)
    return price_hat

# functions that set up counterfactuals
# define market-level attributes
def small_mkt_objects(mkt_sim):
    vec_mkts = mkt_sim.market_ids.unique()
    mkt_size = mkt_sim[['market_ids','tot_HH']].drop_duplicates().reset_index(drop=True)
    vec_mkt_size = np.reshape(mkt_size.loc[mkt_size.market_ids.isin(vec_mkts),'tot_HH'].values,(-1,1))
    return vec_mkts,vec_mkt_size

# define product-level attributes  
def product_objects(mkt_sim,results):
    vec_mkts,vec_mkt_size = small_mkt_objects(mkt_sim)
    # get unique prods
    vec_prods = mkt_sim.product_ids.unique()
    vec_prods.sort()
    # get common vehicle characteristics
    list_chars = list()
    for b in results.beta_labels + results.sigma_labels:
        if any(word in b for word in ['prices','dollar_per_mile']):
            next
        elif '[' in b:
            vec_split = b.split('[')
            col_split = vec_split[0]
            list_chars.append(col_split)
        elif '*' in b:
            vec_split = b.split('*')
            for el_split in vec_split:
                list_chars.append(el_split)
        else:
            list_chars.append(b)
    list_chars = list(set(list_chars))

    #temporary fix
    #list_chars.remove('t')
    #list_chars.append('time_trend')
    if '1' in list_chars:
        mkt_sim['1'] = 1
    # if state in list_chars, remove it
    if 'state' in list_chars:
        list_chars.remove('state')
    # if CA in list_chars, remove it
    if 'CA' in list_chars:
        list_chars.remove('CA')

    product_chars = mkt_sim[list_chars+['msrp','product_ids','firm_ids','ghg_credit_new','mc']].drop_duplicates()
    product_chars = product_chars.sort_values(by ='product_ids').reset_index(drop=True)
    vec_msrp = np.reshape(product_chars.msrp.values,(-1,1))

    mc = np.reshape(product_chars.mc.values,(-1,1))

    # create n_prod x n_prod diagonal
    mat_diag = np.identity(len(vec_prods))

    # create n_prod x n_prod ownership matrix
    mat_ownership = np.zeros((len(vec_prods),len(vec_prods)))
    for p in range(0,len(vec_prods)):
        p_firm = product_chars.loc[product_chars.product_ids == vec_prods[p],'firm_ids'].values[0]
        vec_same_firm = np.array((product_chars.firm_ids == p_firm)*1)
        mat_ownership[p,:] = vec_same_firm

    # create n_prod x n_markets matrix of net ZEV credits created by each vehicle
    mat_ZEV= np.zeros((len(vec_prods),len(vec_mkts)))
    for m in range(0,len(vec_mkts)):
        for p in range(0,len(vec_prods)):
            z_creds = mkt_sim.loc[(mkt_sim.market_ids == vec_mkts[m]) & (mkt_sim.product_ids == vec_prods[p]),'net_credits'].values
            if(len(z_creds) == 0):
                mat_ZEV[p,m] = 0
            else:
                mat_ZEV[p,m] = z_creds[0]
    
    # vector of ghg credits created by each vehicle
    vec_ghg_credits = np.reshape(product_chars.ghg_credit_new.values,(-1,1))

    return vec_prods,product_chars,vec_msrp,mc,mat_diag,mat_ownership,mat_ZEV,vec_ghg_credits  

# define utilities
def utility_objects(product_chars,results,mkt_sim):
    delta_const = np.zeros((product_chars.shape[0],1))
    for i in range(0,len(results.beta)):
        b_name = results.beta_labels[i]
        b_param = results.beta[i][0]
        # ignore prices or gpktm or state FEs
        if any(word in b_name for word in ['prices','dollar_per_mile','state','CA']):
            next
        # deal with c() variables
        elif '[' in b_name:
            vec_split = b_name.split('[')
            col_split = vec_split[0]
            el_split = vec_split[1].split(']')[0]
            el_split = el_split.replace("'",'')
            delta_const = (
                delta_const + 
                np.reshape((product_chars[col_split] == el_split).values,(-1,1)) * b_param
            )
        # deal with interactions among exogenous chars
        elif '*' in b_name:
            vec_split = b_name.split('*')
            delta_split = np.ones((len(product_chars),1))*b_param
            for el_split in vec_split:
                delta_split = delta_split * np.reshape(product_chars[el_split].values,(-1,1))
            delta_const = delta_const + delta_split
        # constant term
        elif '1' == b_name:
            delta_const = delta_const + b_param
        # default variables
        else:
            delta_const = delta_const + np.reshape(product_chars[b_name].values,(-1,1)) * b_param
    # price term
    if sum('prices' in b for b in results.beta_labels) > 1:
        print('WARNING: prices enter into multiple beta variables, but we only account for one')
    else:
        if sum('prices' in b for b in results.beta_labels) == 1:
            coef_price = results.beta[results.beta_labels.index('prices')]*np.ones((len(product_chars),1))
        else:
            coef_price = np.zeros((len(product_chars),1))
        if ('prices' in results.sigma_labels):
            coef_price = results.pi[results.sigma_labels.index('prices'),:]*np.ones((len(product_chars),1))

    # fuel efficiency term
    # nice to have this separated out for when we want to treat it as endogenous
    if sum('dollar_per_mile' in b for b in results.beta_labels) > 1:
        print('WARNING: cost per mile enters into multiple variables, but we only account for one')
    else:
        coef_fe = results.beta[results.beta_labels.index('dollar_per_mile')]*np.ones((len(product_chars),1))    
    
    # nested logit
    if (results.rho.shape[0] == 0):
        coef_rho = None
        vec_grp = None
        membership_mat = None
    else:
        coef_rho = results.rho[0]
        vec_grp = np.reshape(1+(product_chars.electric + product_chars.phev).values,(-1,1))
        membership_mat = np.zeros((len(vec_grp),len(vec_grp)))
        # element i,j of membership_mat = 1 if i and j are in the same group in vec_grp
        for i in range(0,len(vec_grp)):
            for j in range(0,len(vec_grp)):
                if vec_grp[i] == vec_grp[j]:
                    membership_mat[i,j] = 1
    dict_nl = {'coef_rho':coef_rho,'vec_grp':vec_grp,'membership_mat':membership_mat}

    # state fixed effects
    # warning: not dealing with state-interactions
    # check if 'state' in beta_labels
    if sum('state' in b for b in results.beta_labels) > 0:
        # get index of values that contain 'state' in beta_labels
        state_idx = [i for i, s in enumerate(results.beta_labels) if 'state' in s]
        # get labels that contain 'state' in beta_labels
        #state_labels = [s for s in results.beta_labels if 'state' in s]
        # get corresponding beta values
        state_fes = results.beta[state_idx]
        # append 0 to front of state fes (for omitted market)
        state_fes = np.reshape(np.append(0,state_fes),(-1,1))
        # reshape to be n_prod x n_states
        state_fes = (state_fes * np.ones((len(state_fes),product_chars.shape[0]))).T
    else:
        state_fes = np.zeros(delta_const.shape)

    # California-specific fixed effects
    util_CA = np.zeros((len(product_chars),1))
    vec_mkts,vec_mkt_size = small_mkt_objects(mkt_sim)

    if sum('CA' in b for b in results.beta_labels) > 0:
        # get index of values that contain 'state' in beta_labels
        CA_idx = [i for i, s in enumerate(results.beta_labels) if 'CA' in s]
        # get n_prod matrix of CA fes
        for i in CA_idx:
            b_name = results.beta_labels[i]
            b_param = results.beta[i][0]
            vec_split = b_name.split('*')
            char_interact = vec_split[vec_split != 'CA']
            util_CA = (
                util_CA + 
                np.reshape((product_chars[char_interact]).values,(-1,1)) * b_param
            )
        # create n_prod x n_states matrix
        util_CA_expanded = np.zeros((len(product_chars),len(vec_mkts)))
        # drop first four characters from vec_mkts
        vec_mkts = [s[4:] for s in vec_mkts]
        # get California index in vec_mkts
        CA_mkt_idx = [i for i, s in enumerate(vec_mkts) if 'CALIFORNIA' in s]
        # put CA effects in correct location
        util_CA_expanded[:,CA_mkt_idx] = util_CA
        # add to state_fes
        state_fes = state_fes + util_CA_expanded

    return delta_const,coef_price,coef_fe,dict_nl,state_fes



# function to solve for equilibrium in prices, credits
def solve_price_credit_eq(bln_zev_credit_endog,bln_ghg_credit_endog,price_orig,ghg_credit_price,zev_credit_price,cf,mat_incentives,mat_dpm_orig,mat_dpm,delta_const,mat_individ_prod_util,bln_agent,agent_weights,coef_price,coef_fe,mat_xi,mat_mkt_prod,vec_mkts,mat_diag,vec_mkt_size,mat_ownership,mat_ZEV,vec_ghg_credits,mc,state_fes,dict_nl):
    # set params
    delta_threshold_price = 1E-5
    delta_price = 1
    iter_max = 1000
    iter = 0

    # zev params
    zev_credit_eq = False
    lst_net_zev_credits = list()
    lst_zev_credit_price = list()
    ub_zev = 10
    lb_zev = 0
    grid_zev = False

    # ghg params
    ghg_credit_eq = False
    lst_net_ghg_credits = list()
    lst_ghg_credit_price = list()
    ub_ghg = 10
    lb_ghg = 0
    grid_ghg = False


    if (bln_zev_credit_endog + bln_ghg_credit_endog <= 1):
        # loop if single market endogenized
        while ((delta_price > delta_threshold_price) or (zev_credit_eq == False) or (ghg_credit_eq == False)) and (iter < iter_max):
            iter+=1
            print('iter: '+str(iter))
            # calculate prices with change
            if cf == 'no_ZEV':
                price_cf_hat = calc_price(price_orig,mat_incentives,0,ghg_credit_price,mat_dpm_orig,delta_const,mat_individ_prod_util,bln_agent,agent_weights,coef_price,coef_fe,mat_xi,mat_mkt_prod,vec_mkts,mat_diag,vec_mkt_size,mat_ownership,mat_ZEV,vec_ghg_credits,mc,state_fes,dict_nl)
            elif cf == 'no_GHG':
                price_cf_hat = calc_price(price_orig,mat_incentives,zev_credit_price,0,mat_dpm_orig,delta_const,mat_individ_prod_util,bln_agent,agent_weights,coef_price,coef_fe,mat_xi,mat_mkt_prod,vec_mkts,mat_diag,vec_mkt_size,mat_ownership,mat_ZEV,vec_ghg_credits,mc,state_fes,dict_nl)
            elif cf in ['high_gas','addl_product','remove_product']:
                price_cf_hat = calc_price(price_orig,mat_incentives,zev_credit_price,ghg_credit_price,mat_dpm,delta_const,mat_individ_prod_util,bln_agent,agent_weights,coef_price,coef_fe,mat_xi,mat_mkt_prod,vec_mkts,mat_diag,vec_mkt_size,mat_ownership,mat_ZEV,vec_ghg_credits,mc,state_fes,dict_nl)
            # adjust ZEV credit price (if endogenized)
            if (bln_zev_credit_endog):
                if (iter == 1):
                    lst_zev_credit_price = [zev_credit_price]
                lst_net_zev_credits,zev_credit_eq,zev_credit_price,lst_zev_credit_price,grid_zev,ub_zev,lb_zev = adjust_credit_price('zev',lst_net_zev_credits,zev_credit_eq,zev_credit_price,lst_zev_credit_price,grid_zev,ub_zev,lb_zev,mat_dpm,price_cf_hat,mat_incentives,mat_ZEV,vec_ghg_credits,iter,delta_const,mat_individ_prod_util,bln_agent,agent_weights,coef_price,coef_fe,mat_xi,mat_mkt_prod,vec_mkt_size,state_fes,dict_nl)
            else:
                zev_credit_eq = True        

            # adjust GHG credit price (if endogenized)
            if (bln_ghg_credit_endog):
                if (iter == 1):
                    lst_ghg_credit_price = [ghg_credit_price]
                lst_net_ghg_credits,ghg_credit_eq,ghg_credit_price,lst_ghg_credit_price,grid_ghg,ub_ghg,lb_ghg = adjust_credit_price('ghg',lst_net_ghg_credits,ghg_credit_eq,ghg_credit_price,lst_ghg_credit_price,grid_ghg,ub_ghg,lb_ghg,mat_dpm,price_cf_hat,mat_incentives,mat_ZEV,vec_ghg_credits,iter,delta_const,mat_individ_prod_util,bln_agent,agent_weights,coef_price,coef_fe,mat_xi,mat_mkt_prod,vec_mkt_size,state_fes,dict_nl)
            else:  
                ghg_credit_eq = True

            # calculate change in price, compare to delta_threshold_price
            delta_price = abs(price_orig - price_cf_hat).max()
            print('delta price: '+str(delta_price))
            # update values and start again
            price_orig = price_cf_hat
    else:
        # loop if both credit markets endogenized
        if cf != 'high_gas':
            print('have not set up this counterfactual with both zev and ghg credit markets endogenized')
        # can reverse the order to confirm eq. outcome not sensitive to ordering
        quiet = False
        iter_outer = 0
        while ((iter_outer < 100) and (ghg_credit_eq == False)):
            iter_outer+=1
            # first get GHG credit eq
            while ((delta_price > delta_threshold_price) or (ghg_credit_eq == False)) and (iter < 100):
                iter+=1
                if(quiet==False):
                    print('iter: '+str(iter))
                
                price_cf_hat = calc_price(price_orig,mat_incentives,zev_credit_price,ghg_credit_price,mat_dpm,delta_const,mat_individ_prod_util,bln_agent,agent_weights,coef_price,coef_fe,mat_xi,mat_mkt_prod,vec_mkts,mat_diag,vec_mkt_size,mat_ownership,mat_ZEV,vec_ghg_credits,mc,state_fes,dict_nl)
                # adjust GHG credit price 
                if (iter == 1):
                    lst_ghg_credit_price = [ghg_credit_price]
                lst_net_ghg_credits,ghg_credit_eq,ghg_credit_price,lst_ghg_credit_price,grid_ghg,ub_ghg,lb_ghg = adjust_credit_price('ghg',lst_net_ghg_credits,ghg_credit_eq,ghg_credit_price,lst_ghg_credit_price,grid_ghg,ub_ghg,lb_ghg,mat_dpm,
                                                                                                                                    price_cf_hat,mat_incentives,mat_ZEV,vec_ghg_credits,iter,
                                                                                                                                    delta_const,mat_individ_prod_util,bln_agent,agent_weights,coef_price,coef_fe,mat_xi,mat_mkt_prod,vec_mkt_size,
                                                                                                                                    state_fes,dict_nl,quiet)

                # calculate change in price, compare to delta_threshold_price
                delta_price = abs(price_orig - price_cf_hat).max()
                if(quiet==False):
                    print('delta price: '+str(delta_price))
                # update values and start again
                price_orig = price_cf_hat
            print('ghg credit eq finished after '+str(iter)+ ' iterations')
            # then get ZEV credit eq
            # reset ZEV params
            iter = 0
            bln_zev_credit_endog = True
            zev_credit_eq = False
            lst_net_zev_credits = list()
            lst_zev_credit_price = list()
            ub_zev = 10
            lb_zev = 0
            grid_zev = False
            while ((delta_price > delta_threshold_price) or (zev_credit_eq == False) and (iter < 100)):
                iter+=1
                if(quiet==False):
                    print('iter: '+str(iter))
                # calculate prices with change
                price_cf_hat = calc_price(price_orig,mat_incentives,zev_credit_price,ghg_credit_price,mat_dpm,delta_const,mat_individ_prod_util,bln_agent,agent_weights,coef_price,coef_fe,mat_xi,mat_mkt_prod,vec_mkts,mat_diag,vec_mkt_size,mat_ownership,mat_ZEV,vec_ghg_credits,mc,state_fes,dict_nl)

                # adjust ZEV credit price 
                if (iter == 1):
                    lst_zev_credit_price = [zev_credit_price]
                lst_net_zev_credits,zev_credit_eq,zev_credit_price,lst_zev_credit_price,grid_zev,ub_zev,lb_zev = adjust_credit_price('zev',lst_net_zev_credits,zev_credit_eq,zev_credit_price,lst_zev_credit_price,grid_zev,ub_zev,lb_zev,mat_dpm,
                                                                                                                                    price_cf_hat,mat_incentives,mat_ZEV,vec_ghg_credits,iter,
                                                                                                                                    delta_const,mat_individ_prod_util,bln_agent,agent_weights,coef_price,coef_fe,mat_xi,mat_mkt_prod,vec_mkt_size,
                                                                                                                                    state_fes,dict_nl,quiet) 

                # calculate change in price, compare to delta_threshold_price
                delta_price = abs(price_orig - price_cf_hat).max()
                if(quiet==False):
                    print('delta price: '+str(delta_price))
                # update values and start again
                price_orig = price_cf_hat
            print('zev credit eq finished after '+str(iter)+ ' iterations')
            # check GHG net credits
            net_ghg_credits = calc_net_ghg_credits(price_cf_hat,mat_incentives,vec_ghg_credits,mat_dpm,delta_const,mat_individ_prod_util,bln_agent,agent_weights,coef_price,coef_fe,mat_xi,mat_mkt_prod,vec_mkt_size,state_fes,dict_nl)
            print('outer loop iter '+str(iter_outer)+' ghg net credits: '+str(net_ghg_credits))
            if (abs(net_ghg_credits) > 1):
                iter = 0
                # reset ghg params
                ghg_credit_eq = False
                lst_net_ghg_credits = list()
                lst_ghg_credit_price = list()
                ub_ghg = 10
                lb_ghg = 0
                grid_ghg = False
    return price_cf_hat,ghg_credit_price,zev_credit_price    

# function to read in data, set up
def setup_cf(input_results,str_project,output_folder):
    # read in demand model
    # str_loc = os.getcwd()
    str_demand_results = str_project + output_folder + '/demand_results/'+input_results+'/'

    results = pickle.load(open(str_demand_results+"pickled_demand.pickle", "rb" ) )
    #agent_data = pickle.load(open(str_results+"pickled_agent_data.pickle", "rb" ) )
    mkt_data = pickle.load(open(str_demand_results+"pickled_mkt_data.pickle", "rb" ) )
    #temporary fix
    #mkt_data = mkt_data.rename(columns={'t':'time_trend'})

    # read in location of marginal cost estimates
    with open(str_demand_results + 'supply_results_loc.txt') as f:
        supply_loc = f.readlines()[0]
    # read in marginal costs
    full_supply_loc = str_project + output_folder + '/supply_results/'
    mc = pd.read_csv(full_supply_loc + supply_loc + '/mc_ests.csv')

    # read in energy prices
    str_elec = str_project + 'data/intermediate/annual_elec_prices.csv'
    str_gas = str_project + 'data/intermediate/annual_gas_prices.csv'
    str_diesel = str_project + 'data/intermediate/annual_diesel_prices.csv'
    elec_price = pd.read_csv(str_elec)
    gas_price = pd.read_csv(str_gas)
    diesel_price = pd.read_csv(str_diesel)

    # read in counterfactual characteristic params
    str_char_params = str_project + 'data/intermediate/vehicle_entry_params.csv'
    char_params = pd.read_csv(str_char_params)

    # get ZEV/GHG credit data
    mkt_data = get_credit_reqs(mkt_data,str_project)

    # fill in xi from results
    mkt_data['xi'] = results.xi

    # fill in mc from saved supply results
    mkt_data = mkt_data.merge(mc,on = 'product_ids',how='outer')

    # combine incentives
    if 'rebate' in mkt_data:
        mkt_data['tot_incentives'] = mkt_data.rebate + mkt_data.fed_tax_credit
    else:
        mkt_data['tot_incentives'] = mkt_data.fed_tax_credit

    if(pathlib.Path(str_demand_results+"pickled_agent_data.pickle").is_file()):
        agent_data = pickle.load(open(str_demand_results+"pickled_agent_data.pickle", "rb" ) )
        if(pathlib.Path(str_demand_results+"pickled_integration_data.pickle").is_file()):
            integration = pickle.load(open(str_demand_results+"pickled_integration_data.pickle", "rb" ) )
        else:
            integration = None
        bln_agent = True
    else:
        agent_data = None
        integration = None
        bln_agent = False    
    return results, mkt_data, elec_price, gas_price, diesel_price,char_params,bln_agent,agent_data,integration

# function to calculate counterfactual profits under alternative set of products
def single_profit_cf(mkt_sim_adj,bln_agent,agent_sim,agent_weights,results,cf,ghg_credit_price,zev_credit_price):
    # product chars
    vec_prods,product_chars,vec_msrp,mc,mat_diag,mat_ownership,mat_ZEV,vec_ghg_credits = product_objects(mkt_sim_adj,results)
    mat_mkt_prod, mat_xi, mat_dpm_orig, mat_incentives, mat_shares_orig = get_mats_avail_xi_dpm_incentive('dollar_per_mile','tot_incentives',vec_prods,mkt_sim_adj)

    # utility chars
    delta_const,coef_price,coef_fe,state_fes = utility_objects(product_chars,results,mkt_sim_adj)
    mat_individ_prod_util,coef_price = get_individ_utility(results,product_chars,mat_mkt_prod,mkt_sim_adj,agent_sim,bln_agent,coef_price)

    vec_mkts,vec_mkt_size = small_mkt_objects(mkt_sim_adj)

    mat_dpm = mat_dpm_orig
    price_orig = vec_msrp
    bln_zev_credit_endog = False
    bln_ghg_credit_endog = False
        
    # solve for equilbrium!
    price_cf_hat,ghg_credit_price,zev_credit_price =solve_price_credit_eq(bln_zev_credit_endog,bln_ghg_credit_endog,price_orig,ghg_credit_price,zev_credit_price,
                                                                        cf,mat_incentives,mat_dpm_orig,mat_dpm,delta_const,mat_individ_prod_util,bln_agent,agent_weights,
                                                                        coef_price,coef_fe,mat_xi,mat_mkt_prod,
                                                                        vec_mkts,mat_diag,vec_mkt_size,mat_ownership,mat_ZEV,vec_ghg_credits,mc,state_fes)
    # calculate new profit
    # calculate shares, market outcomes with new price
    utilities_cf, probabilities_cf, shares_cf, conditionals_cf = get_shares(price_cf_hat,mat_incentives,mat_dpm,delta_const,mat_individ_prod_util,bln_agent,agent_weights,coef_price,coef_fe,mat_xi,mat_mkt_prod,state_fes) 
    # q_noZEV = calc_q(price_cf_hat,mat_incentives,mat_dpm_orig,delta_const,coef_price,coef_fe,mat_xi,mat_mkt_prod,vec_mkt_size)

    # re-allocate shares, prices to simulation dataframe 
    df_newshares = pd.DataFrame(shares_cf)
    df_newshares = df_newshares.set_axis(vec_mkts, axis=1, inplace=False)
    df_newshares['product_ids'] = vec_prods
    df_newshares_l = pd.melt(df_newshares,id_vars = 'product_ids',value_vars = vec_mkts,var_name = 'market_ids',value_name='shares_cf')

    df_cf_prices = pd.DataFrame({'product_ids':vec_prods,'msrp_cf':price_cf_hat.flatten(),'delta_const':delta_const.flatten()})

    # merge with mkt_sim
    mkt_sim_cf = pd.merge(mkt_sim_adj,df_newshares_l,on=['product_ids','market_ids'])
    mkt_sim_cf = pd.merge(mkt_sim_cf,df_cf_prices,on='product_ids')

    # calculate new profits
    mkt_sim_cf['unit_net_credit_cost'] = mkt_sim_cf.ghg_credit_new * ghg_credit_price + mkt_sim_cf.net_credits * zev_credit_price
    mkt_sim_cf['unit_profits_cf'] = mkt_sim_cf.msrp_cf - mkt_sim_cf.mc + mkt_sim_cf.unit_net_credit_cost
    mkt_sim_cf['model_profits_cf'] = mkt_sim_cf.unit_profits_cf * mkt_sim_cf.shares * mkt_sim_cf.tot_HH
    df_profit_cf= mkt_sim_cf[['oem','model_profits_cf']].groupby('oem').sum().reset_index()

    return df_profit_cf

# function to actually calculate change in profits for the relevant firm
def calc_delta_pi_firm(df_profit_orig,df_profit_cf,oem_cf):
    # get difference in profits for relevant oem
    profit_new = df_profit_cf.loc[df_profit_cf.oem == oem_cf,'model_profits_cf']
    profit_orig = df_profit_orig.loc[df_profit_orig.oem == oem_cf,'model_profits_orig']
    delta_pi = (profit_new - profit_orig).item()
    return delta_pi

# function to generate characteristics based on siblings, historical patterns, etc.
def est_attributes(mkt_sim,product_chars,char_params,fueltype,year,elec_price,diesel_price,gas_price):
    # note: if a gas vehicle is NOT offered in a state, this setup assumes the EV version will also not be offered there
    df_prods = mkt_sim[['product_ids','oem','make','model','fuel','curbwt','max_hp','combined_mpg2008','ghg_std_adj','car','state','market_ids','tot_HH','xi','has_sibling_BEV','BEV_sibling_entry_year','had_sibling_BEV','has_sibling_PHEV','PHEV_sibling_entry_year','had_sibling_PHEV','net_credits']].drop_duplicates().reset_index(drop=True)
    df_prods = pd.merge(df_prods,product_chars,on=['product_ids','make'])
    df_prods_gas = df_prods.loc[(df_prods.fuel == 'gasoline') & (df_prods.msrp < 101) & (df_prods.bodytype != 'Full-Size Van')].reset_index(drop=True)
    # also drop some mis-classified cargo vans
    df_prods_gas = df_prods_gas.loc[~(df_prods.model.isin(['Transit-Connect-Cargo','Transit-Connect-Wagon']))]
    # also drop vehicles that have (or had) electric siblings
    df_prods_gas_entered = df_prods_gas.loc[df_prods_gas[fueltype+'_sibling_entry_year'] == 1]
    df_prods_gas = df_prods_gas.loc[~(df_prods_gas['has_sibling_'+fueltype] == 1) & ~(df_prods_gas['had_sibling_'+fueltype] == 1)]
    # generate EV attributes
    #df_prods_gas['time_trend'] = year - 2011 # time trend is defined in main data as year - 2013
    char_params = char_params[char_params.fueltype == fueltype]
    df_prods_gas['time_trend_alt'] = year - 2011
    df_prods_gas['batterysize'] = char_params.loc[(char_params.param == 'intercept') & (char_params.model == 'battery'),'val'].item() \
                                + char_params.loc[(char_params.param == 't') & (char_params.model == 'battery'),'val'].item() * df_prods_gas.time_trend_alt
    df_prods_gas['curbwt_orig'] = df_prods_gas.curbwt
    df_prods_gas['curbwt'] = char_params.loc[(char_params.param == 'intercept') & (char_params.model == 'weight'),'val'].item() \
                            + df_prods_gas.curbwt_orig*char_params.loc[char_params.param == 'curbwt','val'].item() \
                            + df_prods_gas.batterysize*char_params.loc[(char_params.param == 'batterysize') & (char_params.model == 'weight'),'val'].item()
    df_prods_gas['log_hp_weight_orig'] = df_prods_gas.log_hp_weight
    df_prods_gas['log_hp_weight'] = np.log(df_prods_gas.max_hp/df_prods_gas.curbwt)
    df_prods_gas['range_elec'] = char_params.loc[(char_params.param == 'intercept') & (char_params.model == 'range'),'val'].item() \
                                + df_prods_gas.batterysize * char_params.loc[(char_params.param == 'batterysize') & (char_params.model == 'range'),'val'].item() \
                                + df_prods_gas.time_trend_alt * char_params.loc[(char_params.param == 't') & (char_params.model == 'range'),'val'].item() \
                                + (df_prods_gas.drivetype == 'Awd').astype('uint8') * char_params.loc[(char_params.param == 'drivetypeAwd'),'val'].item()
    if fueltype == 'BEV':
        df_prods_gas['range_elec'] =  df_prods_gas['range_elec'] + (df_prods_gas.model == 'Tesla').astype('uint8') * char_params.loc[(char_params.param == 'tesla'),'val'].item()
        df_prods_gas['combined_mpg2008'] = char_params.loc[(char_params.param == 'intercept') & (char_params.model == 'mpg'),'val'].item() \
                                        + df_prods_gas.batterysize * char_params.loc[(char_params.param == 'batterysize') & (char_params.model == 'mpg'),'val'].item() \
                                        + df_prods_gas.range_elec * char_params.loc[(char_params.param == 'range_elec') & (char_params.model == 'mpg'),'val'].item() \
                                        + df_prods_gas.time_trend_alt * char_params.loc[(char_params.param == 't') & (char_params.model == 'mpg'),'val'].item()
        df_prods_gas['mpg_elec'] = df_prods_gas['combined_mpg2008']
        df_prods_gas['mpg_gas'] = np.nan
        df_prods_gas['combinedUF'] = 0
        df_prods_gas.fuel = 'electric'
        df_prods_gas.electric = 1
    elif fueltype == 'PHEV':
        df_prods_gas['mpg_elec'] = char_params.loc[(char_params.param == 'intercept') & (char_params.model == 'mpg_elec'),'val'].item() \
                                    + df_prods_gas.batterysize * char_params.loc[(char_params.param == 'batterysize') & (char_params.model == 'mpg_elec'),'val'].item() \
                                    + df_prods_gas.range_elec * char_params.loc[(char_params.param == 'range_elec') & (char_params.model == 'mpg_elec'),'val'].item() \
                                    + df_prods_gas.time_trend_alt * char_params.loc[(char_params.param == 't') & (char_params.model == 'mpg_elec'),'val'].item()
        df_prods_gas['mpg_gas'] = char_params.loc[(char_params.param == 'intercept') & (char_params.model == 'mpg_gas'),'val'].item() \
                                    + df_prods_gas.combined_mpg2008 * char_params.loc[(char_params.param == 'combined_mpg2008') & (char_params.model == 'mpg_gas'),'val'].item() \
                                    + df_prods_gas.curbwt * char_params.loc[(char_params.param == 'weight') & (char_params.model == 'mpg_gas'),'val'].item()
        df_prods_gas['combinedUF'] = char_params.loc[(char_params.param == 'intercept') & (char_params.model == 'combinedUF'),'val'].item() \
                                    + df_prods_gas.time_trend_alt * char_params.loc[(char_params.param == 't') & (char_params.model == 'combinedUF'),'val'].item() \
                                    + df_prods_gas.range_elec * char_params.loc[(char_params.param == 'range_elec') & (char_params.model == 'combinedUF'),'val'].item()
        df_prods_gas.fuel = 'PHEV'
        df_prods_gas.phev = 1
    # update dollar_per_mile
    df_prods_gas['model_year'] = year
    elec_price_agg = aggregate_state_energy_prices(elec_price,'cent_per_kwh',mkt_sim)
    diesel_price_agg = aggregate_state_energy_prices(diesel_price,'dollar_per_gal_diesel',mkt_sim)
    gas_price_agg = aggregate_state_energy_prices(gas_price,'dollar_per_gal_gas',mkt_sim)
    df_prods_gas = calc_dollar_per_mile(elec_price_agg,diesel_price_agg,gas_price_agg,df_prods_gas)
    df_prods_gas['dollar_per_mile'] = df_prods_gas['dollar_per_mile_new'] 
    df_prods_gas = df_prods_gas.drop(['dollar_per_mile_new'],axis=1)

    # update original shares
    df_prods_gas['mat_shares_orig'] = 0

    #update marginal cost
    df_prods_gas['mc_orig'] = df_prods_gas.mc
    df_prods_gas.mc = df_prods_gas.mc_orig + .8
    print('warning: using a dumb adder for marginal cost right now. change this!')

    return df_prods_gas, df_prods_gas_entered

# calculate ghg credits generated/needed for each vehicle for new vehicles
def calc_newveh_ghg_credits(df_prods_gas,fueltype):
    # GHG credits
    # as long as sibling footprint is unchanged, GHG standard will be unchanged (ghg_std_adj)
    # save original ghg credit amounts
    df_prods_gas['ghg_credit_orig'] = df_prods_gas.ghg_credit_new
    # calculate emis new
    if(fueltype == 'BEV'):
        df_prods_gas['emis'] = 0
    else:
        # true calculation uses city and highway UF/mpg separately but this will do
        df_prods_gas['emis'] = (1-df_prods_gas.combinedUF)/df_prods_gas.mpg_gas *8887
    # set multiplier for EVs and PHEVs depending on year
    # 2017-2019, EVs= 2, PHEVs = 1.6
    # 2020, EVs = 1.75, PHEVs = 1.45
    # 2021, EVs = 1.5, PHEVs = 1.3
    # 2022, EVs = 1, PHEVs = 1
    df_prods_gas['multiplier'] = 1
    if(fueltype == 'BEV'):
        df_prods_gas.loc[df_prods_gas.model_year.isin([2017,2018,2019]),'multiplier'] = 2
        df_prods_gas.loc[df_prods_gas.model_year == 2020,'multiplier'] = 1.75
        df_prods_gas.loc[df_prods_gas.model_year == 2021,'multiplier'] = 1.5
    elif(fueltype == 'PHEV'):
        df_prods_gas.loc[df_prods_gas.model_year.isin([2017,2018,2019]),'multiplier'] = 1.6
        df_prods_gas.loc[df_prods_gas.model_year == 2020,'multiplier'] = 1.45
        df_prods_gas.loc[df_prods_gas.model_year == 2021,'multiplier'] = 1.3
    # calculate new ghg credit amounts
    df_prods_gas['ghg_credit_new'] = (df_prods_gas.ghg_std_adj - df_prods_gas.emis) * \
        ((df_prods_gas.car * 195264 + (1-df_prods_gas.car) * 225865)/1000000) * df_prods_gas.multiplier
    return df_prods_gas

def calc_newveh_zev_credits(df_prods_gas,fueltype,year):
    # per CARB and per calculation in ZEV_credit_calculations.R, UDDS range ~ 1.45 * EPA range
    if(fueltype == 'BEV'):
        df_prods_gas['UDDS_AER_mi'] = df_prods_gas.range_elec * 1.45
        #2014-2017: based on UDDS AER buckets
        if(year < 2018):
            zev_credit_rule = lambda x: 2 if (x >= 5 and x < 7.5) else 2.5 if (x >= 7.5 and x < 10) else 3 if (x >= 10 and x < 20) else 4 if x >= 20 else 0
            df_prods_gas['zev_credits'] = df_prods_gas['UDDS_AER_mi'].apply(zev_credit_rule)
        else: # 2018+: equation based on UDDS AER
            # note: eqn. uses 0.01 but our range is already divided by 10 so this become .1
            df_prods_gas['zev_credits'] = 0.1 * df_prods_gas['UDDS_AER_mi'] + 0.5
            # cannot exceed 4
            df_prods_gas.loc[(df_prods_gas.zev_credits > 4),'zev_credits'] = 4
        # drop UDDS_AER_mi column
        df_prods_gas = df_prods_gas.drop(['UDDS_AER_mi'],axis=1) 
    elif(fueltype == 'PHEV'):
        df_prods_gas['UDDS_EAER_mi'] = df_prods_gas.range_elec * 1.45
        #2014-2017: based on UDDS EAER buckets
        if(year < 2018):
            # treating all PHEVs as Type F (no info on US06 AER mi)
            # base credit amount
            if(year in [2012,2013,2014]):
                df_prods_gas['zev_credits'] = .45
            else:
                df_prods_gas['zev_credits'] = .35
            # range based adder (should be cityUF but using combinedUF here)
            df_prods_gas['zev_credits'] = df_prods_gas['zev_credits'] + df_prods_gas['UDDS_EAER_mi']* 10 * (1 - df_prods_gas.combinedUF)/11.028
            # cannot exceed 1.39
            df_prods_gas.loc[df_prods_gas.zev_credits > 1.39,'zev_credits'] = 1.39
        else: # 2018+: equation based on UDDS AER
            # we don't have info on US06 AER mi, so ignoring
            # note: eqn. uses 0.01 but our range is already divided by 10 so this become .1
            df_prods_gas['zev_credits'] =0.1 * df_prods_gas['UDDS_EAER_mi'] + 0.3
            # cannot exceed 1.1
            df_prods_gas.loc[(df_prods_gas.zev_credits > 1.1),'zev_credits'] = 1.1
    # reset zev credits to 0 if in non-zev state
    df_prods_gas.loc[df_prods_gas.net_credits == 0,'zev_credits'] = 0
    # round zev credits to two decimal points
    df_prods_gas.zev_credits = df_prods_gas.zev_credits.round(2)

    #df_prods_gas.loc[df_prods_gas.net_credits != 0,'zev_credits'] = 2
    df_prods_gas.net_credits = df_prods_gas.zev_credits + df_prods_gas.net_credits
    return df_prods_gas

# calculate federal ev incentives for new vehicles
def calc_newveh_fed_incentives(df_prods_gas,year):    
    # federal incentives: $2500 if battery capacity >= 4kWh, increases by $417/kWh up to $7500
    # source: CBO (https://www.cbo.gov/sites/default/files/112th-congress-2011-2012/reports/09-20-12-electricvehicles0.pdf)
    df_prods_gas['fed_tax_credit'] = 0
    df_prods_gas.loc[df_prods_gas.batterysize >= 4,'fed_tax_credit'] = 2.5 + (df_prods_gas.loc[df_prods_gas.batterysize >= 4,'batterysize']-4)*.417
    # set max incentive to 7.5
    df_prods_gas.loc[df_prods_gas.fed_tax_credit > 7.5,'fed_tax_credit'] = 7.5
    # for firms that face phaseouts, zero out for year that final incentive amount (0) reached
    # note: this ignores intermediate incentive amounts, dynamic sales effects, etc.
    if(year >= 2020):
        df_prods_gas.loc[df_prods_gas.oem.isin(['General Motors','Tesla']),'fed_tax_credit'] = 0
    return df_prods_gas

# calculate state ev incentives for new vehicles
def calc_newveh_state_incentives(df_prods_gas,fueltype,year):
    df_prods_gas['rebate'] = 0
    # California
    if fueltype == 'BEV':
        if year < 2020:
            df_prods_gas.loc[df_prods_gas.state == 'CALIFORNIA','rebate'] = 2.5
        else:
            df_prods_gas.loc[df_prods_gas.state == 'CALIFORNIA','rebate'] = 2
    else:
        if year < 2017:
            df_prods_gas.loc[(df_prods_gas.state == 'CALIFORNIA'),'rebate'] = 1.5
        elif year in [2017,2018,2019]:
            df_prods_gas.loc[(df_prods_gas.state == 'CALIFORNIA') & (df_prods_gas.UDDS_EAER_mi >= 2),'rebate'] = 1.5
        elif year == 2020:
            df_prods_gas.loc[(df_prods_gas.state == 'CALIFORNIA') & (df_prods_gas.UDDS_EAER_mi >= 3.5),'rebate'] = 1
        elif year >= 2021:
            df_prods_gas.loc[(df_prods_gas.state == 'CALIFORNIA') & (df_prods_gas.range_elec >= 3),'rebate'] = 1
    # Colorado
    if year in [2017,2018,2019]:
        df_prods_gas.loc[df_prods_gas.state == 'COLORADO','rebate'] = 5
    elif year == 2020:
        df_prods_gas.loc[df_prods_gas.state == 'COLORADO','rebate'] = 4
    elif year in [2021,2022]:
        df_prods_gas.loc[df_prods_gas.state == 'COLORADO','rebate'] = 2.5
    # Connecticut
    # policy changes a lot mid-year; use earlier values
    if year in [2015,2016]:
        ct_credit_rule = lambda x: 3 if (x >= 18) else 1.5 if (x >= 7) else .75
        df_prods_gas.loc[(df_prods_gas.state == 'CONNECTICUT'),'rebate'] = df_prods_gas.loc[(df_prods_gas.state == 'CONNECTICUT'),'batterysize'].apply(ct_credit_rule)
    elif year == 2017:
        if fueltype == 'BEV':
            ct_credit_rule = lambda x: 3 if (x >= 25) else 1.5 if (x >= 20) else .75
        else:
            ct_credit_rule = lambda x: 3 if (x >= 18) else 1.5 if (x >= 10) else .75
        df_prods_gas.loc[(df_prods_gas.state == 'CONNECTICUT'),'rebate'] = df_prods_gas.loc[(df_prods_gas.state == 'CONNECTICUT'),'batterysize'].apply(ct_credit_rule)
    elif year == 2018:
        if fueltype == 'BEV':
            ct_credit_rule = lambda x: 3 if (x >= 17.5) else 1.5 if (x >= 10) else .75
        else:
            ct_credit_rule = lambda x: 2 if (x >= 4) else .5
        df_prods_gas.loc[(df_prods_gas.state == 'CONNECTICUT'),'rebate'] = df_prods_gas.loc[(df_prods_gas.state == 'CONNECTICUT'),'range_elec'].apply(ct_credit_rule)
    elif year == 2019:
        if fueltype == 'BEV':
            ct_credit_rule = lambda x: 2 if (x >= 20) else 1.5 if (x >= 12) else .5
        else:
            ct_credit_rule = lambda x: 1 if (x >= 4.5) else .5
        df_prods_gas.loc[(df_prods_gas.state == 'CONNECTICUT'),'rebate'] = df_prods_gas.loc[(df_prods_gas.state == 'CONNECTICUT'),'range_elec'].apply(ct_credit_rule)
    elif year in [2020,2021]:
        if fueltype == 'BEV':
            ct_credit_rule = lambda x: 1.5 if (x >= 20) else .5
            df_prods_gas.loc[(df_prods_gas.state == 'CONNECTICUT'),'rebate'] = df_prods_gas.loc[(df_prods_gas.state == 'CONNECTICUT'),'range_elec'].apply(ct_credit_rule)
        else:
            df_prods_gas.loc[df_prods_gas.state == 'CONNECTICUT','rebate'] = .5
    elif year >= 2022:
        if fueltype == 'BEV':
            df_prods_gas.loc[df_prods_gas.state == 'CONNECTICUT','rebate'] = 2.25
        else:
            df_prods_gas.loc[df_prods_gas.state == 'CONNECTICUT','rebate'] = .75

    # Georgia
    if year in [2014,2015]:
        if fueltype == 'BEV':
            # technically it is the smaller of .2 * price and $5000
            df_prods_gas.loc[df_prods_gas.state == 'GEORGIA','rebate'] = 5

    # Massachusetts
    if year in [2014,2015,2016]:
        if fueltype == 'BEV':
            df_prods_gas.loc[df_prods_gas.state == 'MASSACHUSETTS','rebate'] = 2.5
        else:
            ma_credit_rule = lambda x: 2.5 if (x >= 10) else 1.5
            df_prods_gas.loc[(df_prods_gas.state == 'MASSACHUSETTS'),'rebate'] = df_prods_gas.loc[(df_prods_gas.state == 'MASSACHUSETTS'),'batterysize'].apply(ma_credit_rule)
    elif year in [2017,2018]:
        if fueltype == 'BEV':
            df_prods_gas.loc[df_prods_gas.state == 'MASSACHUSETTS','rebate'] = 2.5
        else:
            ma_credit_rule = lambda x: 2.5 if (x >= 10) else 1.5
            df_prods_gas.loc[(df_prods_gas.state == 'MASSACHUSETTS'),'rebate'] = df_prods_gas.loc[(df_prods_gas.state == 'MASSACHUSETTS'),'batterysize'].apply(ma_credit_rule)
    elif year == 2019:
        if fueltype == 'BEV':
            df_prods_gas.loc[df_prods_gas.state == 'MASSACHUSETTS','rebate'] = 1.5
    elif year >= 2020:
        if fueltype == 'BEV':
            df_prods_gas.loc[df_prods_gas.state == 'MASSACHUSETTS','rebate'] = 2.5
        else:
            df_prods_gas.loc[(df_prods_gas.state == 'MASSACHUSETTS') & (df_prods_gas.range_elec > 2.5),'rebate'] = 1.5

    # Delaware
    if year in [2015,2016]:
        df_prods_gas.loc[df_prods_gas.state == 'DELAWARE','rebate'] = 2.2
    elif year in [2017,2018,2019]:
        if fueltype == 'BEV':
            df_prods_gas.loc[df_prods_gas.state == 'DELAWARE','rebate'] = 3.5
        else:
            df_prods_gas.loc[df_prods_gas.state == 'DELAWARE','rebate'] = 1.5
    elif year >= 2020:
        if fueltype == 'BEV':
            df_prods_gas.loc[df_prods_gas.state == 'DELAWARE','rebate'] = 2.5
        else:
            df_prods_gas.loc[df_prods_gas.state == 'DELAWARE','rebate'] = 1

    # Maine
    if year in [2020,2021]:
        if fueltype == 'BEV':
            df_prods_gas.loc[df_prods_gas.state == 'MAINE','rebate'] = 2
        else:
            df_prods_gas.loc[df_prods_gas.state == 'MAINE','rebate'] = 1

    # New Jersey
    if year in [2020,2021]:
        df_prods_gas.loc[df_prods_gas.state == 'NEW JERSEY','rebate'] = df_prods_gas.loc[df_prods_gas.state == 'NEW JERSEY','range_elec'] * 25/100
        # cannot exceed $5000
        df_prods_gas.loc[(df_prods_gas.state == 'NEW JERSEY') & (df_prods_gas.rebate > 5),'rebate'] = 5
    
    # New York
    if year in range(2017,2022):
        ny_credit_rule = lambda x: 2 if (x >= 12) else 1.7 if (x >= 4) else 1.1 if (x >= 2) else .5
        df_prods_gas.loc[(df_prods_gas.state == 'NEW YORK'),'rebate'] = df_prods_gas.loc[(df_prods_gas.state == 'NEW YORK'),'range_elec'].apply(ny_credit_rule)
    elif year == 2022:
        ny_credit_rule = lambda x: 2 if (x >= 20) else 1 if (x >= 4) else .5
        df_prods_gas.loc[(df_prods_gas.state == 'NEW YORK'),'rebate'] = df_prods_gas.loc[(df_prods_gas.state == 'NEW YORK'),'range_elec'].apply(ny_credit_rule)

    # Oregon
    if year in range(2018,2022):
        or_credit_rule = lambda x: 2.5 if (x >= 10) else 1.5
        df_prods_gas.loc[(df_prods_gas.state == 'OREGON'),'rebate'] = df_prods_gas.loc[(df_prods_gas.state == 'OREGON'),'batterysize'].apply(or_credit_rule)

    # Pennsylvania
    if 'PENNSYLVANIA' in df_prods_gas.state.unique():
        # lots of mid-year switches
        if year in range(2014,2019):
            pa_credit_rule = lambda x: 2 if (x >= 10) else 1
            df_prods_gas.loc[(df_prods_gas.state == 'PENNSYLVANIA'),'rebate'] = df_prods_gas.loc[(df_prods_gas.state == 'PENNSYLVANIA'),'batterysize'].apply(pa_credit_rule)
        elif year == 2018:
            pa_credit_rule = lambda x: 1.75 if (x >= 20) else 1 if (x >= 10) else .75
            df_prods_gas.loc[(df_prods_gas.state == 'PENNSYLVANIA'),'rebate'] = df_prods_gas.loc[(df_prods_gas.state == 'PENNSYLVANIA'),'batterysize'].apply(pa_credit_rule)
        elif year == 2019:
            pa_credit_rule = lambda x: 2 if (x >= 85) else 1.75 if (x >= 30) else 1 if (x >= 10) else .75
            df_prods_gas.loc[(df_prods_gas.state == 'PENNSYLVANIA'),'rebate'] = df_prods_gas.loc[(df_prods_gas.state == 'PENNSYLVANIA'),'batterysize'].apply(pa_credit_rule)
        elif year == 2020:
            if fueltype == 'BEV':
                df_prods_gas.loc[df_prods_gas.state == 'PENNSYLVANIA','rebate'] = 1.5
            else:
                df_prods_gas.loc[df_prods_gas.state == 'PENNSYLVANIA','rebate'] = 1
        elif year == 2021:
            if fueltype == 'BEV':
                df_prods_gas.loc[df_prods_gas.state == 'PENNSYLVANIA','rebate'] = .75
            else:
                df_prods_gas.loc[df_prods_gas.state == 'PENNSYLVANIA','rebate'] = .5
    
    # Rhode Island
    if year in [2016,2017]:
        ri_credit_rule = lambda x: 2.5 if (x >= 18) else 1.5 if (x >= 7) else .5
        df_prods_gas.loc[(df_prods_gas.state == 'RHODE ISLAND'),'rebate'] = df_prods_gas.loc[(df_prods_gas.state == 'RHODE ISLAND'),'batterysize'].apply(ri_credit_rule)
    
    # South Carolina
    if 'SOUTH CAROLINA' in df_prods_gas.state.unique():
        if year in [2014,2015,2016]:
            if fueltype == 'PHEV':
                sc_credit_rule = lambda x: (.111 + .667) + .111 * (x-5) if (x>=5) else .667
                df_prods_gas.loc[(df_prods_gas.state == 'SOUTH CAROLINA'),'rebate'] = df_prods_gas.loc[(df_prods_gas.state == 'SOUTH CAROLINA'),'batterysize'].apply(sc_credit_rule)
                # max credit $2000
                df_prods_gas.loc[(df_prods_gas.state == 'SOUTH CAROLINA') & (df_prods_gas.rebate > 2),'rebate'] = 2
        
    # Tennessee
    if 'TENNESSEE' in df_prods_gas.state.unique():
        if year in [2015,2016]:
            if fueltype == 'BEV':
                df_prods_gas.loc[(df_prods_gas.state == 'TENNESSEE'),'rebate'] = 2.5
            else:
                df_prods_gas.loc[(df_prods_gas.state == 'TENNESSEE'),'rebate'] = 1.5
    
    # Texas
    if 'TEXAS' in df_prods_gas.state.unique():
        # credits become unavailable intermittently
        if year in list(set(range(2014,2023)) - set([2016,2017])):
            df_prods_gas.loc[(df_prods_gas.state == 'TEXAS') & (df_prods_gas.batterysize > 4) & (df_prods_gas.make != 'Tesla'),'rebate'] = 2.5

    # Vermont
    if year >= 2019:
        if fueltype == 'BEV':
            df_prods_gas.loc[(df_prods_gas.state == 'VERMONT'),'rebate'] = 2.5
        else:
            df_prods_gas.loc[(df_prods_gas.state == 'VERMONT'),'rebate'] = 1.5
    
    # Maryland
    if year == 2014:
        md_credit_rule = lambda x: 1 if (x >= 15) else .7 if (x >=10) else .6 if (x >= 4) else 0
        df_prods_gas.loc[(df_prods_gas.state == 'MARYLAND'),'rebate'] = df_prods_gas.loc[(df_prods_gas.state == 'MARYLAND'),'batterysize'].apply(md_credit_rule)
    elif year in [2015,2016,2017]:
        df_prods_gas.loc[(df_prods_gas.state == 'MARYLAND') & (df_prods_gas.batterysize >= 4),'rebate'] = .125 * df_prods_gas.loc[(df_prods_gas.state == 'MARYLAND') & (df_prods_gas.batterysize >= 4),'batterysize']
        # max credit $3000
        df_prods_gas.loc[(df_prods_gas.state == 'MARYLAND') & (df_prods_gas.rebate > 3),'rebate'] = 3
    elif year in [2018,2019,2020]:
        df_prods_gas.loc[(df_prods_gas.state == 'MARYLAND') & (df_prods_gas.batterysize >= 5),'rebate'] = .1 * df_prods_gas.loc[(df_prods_gas.state == 'MARYLAND') & (df_prods_gas.batterysize >= 4),'batterysize']
        # max credit $3000
        df_prods_gas.loc[(df_prods_gas.state == 'MARYLAND') & (df_prods_gas.rebate > 3),'rebate'] = 3
    return df_prods_gas

# function to generate a list of potential entrants with characteristics, incentive amounts
def gen_potential_entrants(mkt_sim,product_chars,char_params,elec_price,diesel_price,gas_price,fueltype):
    year = mkt_sim.model_year.unique()[0]
    # for each gasoline vehicle that doesn't have a sibling, generate EV version
    df_prods_gas, df_prods_gas_entered = est_attributes(mkt_sim,product_chars,char_params,fueltype,year,elec_price,diesel_price,gas_price)

    # calculate ZEV credits, GHG credits
    # GHG credits
    df_prods_gas = calc_newveh_ghg_credits(df_prods_gas,fueltype)

    # ZEV credits
    df_prods_gas = calc_newveh_zev_credits(df_prods_gas,fueltype,year)

    # fill in state and federal incentives
    print('warning: ignoring price requirements for state and federal incentives, rough treatment of phaseout')

    # federal incentives
    df_prods_gas = calc_newveh_fed_incentives(df_prods_gas,year)

    # state incentives
    df_prods_gas = calc_newveh_state_incentives(df_prods_gas,fueltype,year)

    # drop UDDS_EAER_mi column
    if fueltype == 'PHEV':
        df_prods_gas = df_prods_gas.drop(['UDDS_EAER_mi'],axis=1) 
    
    # combine state and federal incentives into tot_incentives
    df_prods_gas['tot_incentives'] = df_prods_gas.fed_tax_credit + df_prods_gas.rebate

    # for each potential new vehicle, generate new id
    df_prods_gas['product_ids_orig'] = df_prods_gas.product_ids
    if fueltype == 'BEV':
        df_prods_gas['product_ids'] = df_prods_gas['product_ids'] + 10000
    elif fueltype == 'PHEV':
        df_prods_gas['product_ids'] = df_prods_gas['product_ids'] + 20000

    return df_prods_gas, df_prods_gas_entered

# create df_delta_pi_yr_hypothetical dataframe based on potential entrants, formatted appropriately
def gen_df_delta_pi_yr_hypothetical(df_prods_gas,year,fueltype):
    df_delta_pi_yr_hypothetical = df_prods_gas[['oem','make','model','has_sibling_'+fueltype,'fuel','product_ids']].drop_duplicates().reset_index(drop=True)
    df_delta_pi_yr_hypothetical['year'] = year
    df_delta_pi_yr_hypothetical['decision'] = 'did not enter'
    df_delta_pi_yr_hypothetical['delta_pi'] = -9999
    return df_delta_pi_yr_hypothetical

def gen_delta_pi_single_year(mkt_data,bln_agent,agent_data,year,results,char_params,elec_price,diesel_price,gas_price):
    # market chars
    mkt_sim = mkt_data.loc[mkt_data.model_year == year].reset_index(drop=True)
    if bln_agent:
        agent_sim = agent_data.loc[agent_data.model_year == year].reset_index(drop=True)
        agent_weights = get_agent_weights(agent_sim,mkt_sim)
    else:
        agent_sim = None
        agent_weights = None
    #vec_mkts,vec_mkt_size = small_mkt_objects(mkt_data,year)
    # product chars
    vec_prods_orig,product_chars,vec_msrp_orig,mc_orig,mat_diag_orig,mat_ownership_orig,mat_ZEV_orig,vec_ghg_credits_orig = product_objects(mkt_sim,results)

    # credit price assumption
    ghg_credit_price = 50/1000
    zev_credit_price = 2000/1000

    # calculate profit per firm
    mkt_sim['unit_net_credit_cost'] = mkt_sim.ghg_credit_new * ghg_credit_price + mkt_sim.net_credits * zev_credit_price
    mkt_sim['unit_profits_orig'] = mkt_sim.msrp - mkt_sim.mc + mkt_sim.unit_net_credit_cost
    mkt_sim['model_profits_orig'] = mkt_sim.unit_profits_orig * mkt_sim.shares * mkt_sim.tot_HH

    df_profit_orig = mkt_sim[['oem','model_profits_orig']].groupby('oem').sum().reset_index()

    #mkt_sim['tot_zev_credits_orig'] = mkt_sim.net_credits * mkt_sim.shares * mkt_sim.tot_HH
    #mkt_sim.tot_zev_credits_orig.sum()

    df_prods_gas_bev, df_prods_gas_entered_bev = gen_potential_entrants(mkt_sim,product_chars,char_params,elec_price,diesel_price,gas_price,'BEV')
    df_prods_gas_phev, df_prods_gas_entered_phev = gen_potential_entrants(mkt_sim,product_chars,char_params,elec_price,diesel_price,gas_price,'PHEV')
    #df_prods_gas_bev[['state','batterysize','range_elec','tot_incentives','rebate']].drop_duplicates().sort_values(by='tot_incentives',ascending=True)

    # loop over actual and potential entrants
    # do this once for EVs, once for PHEVs
    df_delta_pi_yr_ev = fuel_specific_delta_pi_loops(df_prods_gas_bev,df_prods_gas_entered_bev,'BEV',year,mkt_sim,bln_agent,agent_sim,agent_weights,results,df_profit_orig,ghg_credit_price,zev_credit_price)
    df_delta_pi_yr_phev = fuel_specific_delta_pi_loops(df_prods_gas_phev,df_prods_gas_entered_phev,'PHEV',year,mkt_sim,bln_agent,agent_sim,agent_weights,results,df_profit_orig,ghg_credit_price,zev_credit_price)

    df_delta_pi_yr = pd.concat([df_delta_pi_yr_ev,df_delta_pi_yr_phev])
    return df_delta_pi_yr

# function that loops over potential entrants and observed entrants and calculates change in profits
#df_prods_gas = df_prods_gas_bev
#df_prods_gas_entered = df_prods_gas_entered_bev
#fueltype = 'BEV'
def fuel_specific_delta_pi_loops(df_prods_gas,df_prods_gas_entered,fueltype,year,mkt_sim,bln_agent,agent_sim,agent_weights,results,df_profit_orig,ghg_credit_price,zev_credit_price):
    ### vehicles that did not enter ###
    # loop over all new products, create dataframe to store delta_pi #
    df_delta_pi_yr_hypothetical = gen_df_delta_pi_yr_hypothetical(df_prods_gas,year,fueltype)

    for p_id in df_prods_gas.product_ids.unique():
        #p_id = df_prods_gas.product_ids.unique()[0]
        prod_new = df_prods_gas.loc[df_prods_gas.product_ids == p_id]
        oem_prod = prod_new.oem.drop_duplicates().item()
        print(prod_new[['make','model','fuel']].drop_duplicates())
        mkt_sim_adj = mkt_sim.copy()
        mkt_sim_adj = pd.concat([mkt_sim_adj,prod_new])

        # calculate counterfactual profit, change in profit
        df_profit_cf = single_profit_cf(mkt_sim_adj,bln_agent,agent_sim,agent_weights,results,'addl_product',ghg_credit_price,zev_credit_price)
        delta_pi = calc_delta_pi_firm(df_profit_orig,df_profit_cf,oem_prod)
        # update the delta_pi dataframe
        df_delta_pi_yr_hypothetical.loc[df_delta_pi_yr_hypothetical.product_ids == p_id,'delta_pi'] = delta_pi

    ### vehicles that did enter ###
    # mkt_sim.loc[mkt_sim.BEV_sibling_entry_year == 1,['make','model']].drop_duplicates()
    # loop over all new products, create dataframe to store delta_pi #
    if (df_prods_gas_entered.shape[0] != 0):
        if(fueltype == 'BEV'):
            f = 'electric'
        else:
            f = fueltype
        df_delta_pi_yr_entrants = mkt_sim.loc[(mkt_sim.fuel == f) & (mkt_sim.self_sibling_entry_year == 1),['oem','make','model','fuel','product_ids']].drop_duplicates().reset_index(drop=True)
        df_delta_pi_yr_entrants['year'] = year
        df_delta_pi_yr_entrants['decision'] = 'entered'
        df_delta_pi_yr_entrants['delta_pi'] = -9999
        for p_id in df_delta_pi_yr_entrants.product_ids.unique():
            #p_id = df_prods_gas_entered.product_ids.unique()[0]
            #p_id = df_delta_pi_yr_entrants.product_ids.unique()[0]
            prod_remove = mkt_sim.loc[mkt_sim.product_ids == p_id]
            oem_prod = prod_remove.oem.drop_duplicates().item()
            print(prod_remove[['make','model']].drop_duplicates())
            mkt_sim_adj = mkt_sim.copy()
            mkt_sim_adj = mkt_sim.loc[mkt_sim.product_ids != p_id].reset_index(drop=True)
            
            # calculate counterfactual profit, change in profit
            df_profit_cf = single_profit_cf(mkt_sim_adj,bln_agent,agent_sim,agent_weights,results,'remove_product',ghg_credit_price,zev_credit_price)
            delta_pi = calc_delta_pi_firm(df_profit_orig,df_profit_cf,oem_prod)
        
            # update the delta_pi dataframe
            df_delta_pi_yr_entrants.loc[df_delta_pi_yr_entrants.product_ids == p_id,'delta_pi'] = delta_pi
        df_delta_pi_yr = pd.concat([df_delta_pi_yr_hypothetical,df_delta_pi_yr_entrants])
    else:
        df_delta_pi_yr = df_delta_pi_yr_hypothetical
    return df_delta_pi_yr

# function to print out what years are being included
def print_split(split):
    if split == 't1':
        print('Estimating separately for 2014-2015')
    elif split == 't2':
        print('Estimating separately for 2016-2017')
    elif split == 't3':
        print('Estimating separately for 2018-2020')
    elif split == 'None':
        print('Estimating for full data, grouped moments')
    elif split == 'None, ann. moments':
        print('Estimating for full data, separated moments')
    else:
        print('Estimating for full periods, moments separated by census regions')

def read_vin_data(str_project,str_data,version,dynamic):
    state_names, region_names = gen_state_region_names()
    #str_project = '/Users/Stephanie/Dropbox (YSE)/SW_Automaker_Strategies/'
    #str_vin_data = str_project + 'data/intermediate/CA_VIN_data_common.csv'
    if version == 'state' or version in state_names or version in region_names:
        if dynamic:
            str_vin_data = str_project / str_data / 'intermediate/dynamic_US_VIN_data_common.csv'
        else:
            str_vin_data = str_project / str_data / 'intermediate/US_VIN_data_common.csv'
    elif version == 'hybrid':
        if dynamic:
            str_vin_data = str_project + str_data + 'intermediate/dynamic_US_VIN_data_common_ZEV_other_agg_states.csv'
        else:
            str_vin_data = str_project + str_data + 'intermediate/US_VIN_data_common_ZEV_other_agg_states.csv'
    elif version == 'hybrid/regional':
        if dynamic:
            str_vin_data = str_project + str_data + 'intermediate/dynamic_US_VIN_data_common_ZEV_census_agg_states.csv'
        else:
            str_vin_data = str_project + str_data + 'intermediate/US_VIN_data_common_ZEV_census_agg_states.csv'
    elif version == 'national':
        if dynamic:
            print('This data is not yet put together')
        else:
            str_vin_data = str_project / str_data / 'intermediate/agg_US_VIN_data_common.csv'
    elif 'county' in version:
        str_vin_data = str_project + str_data + 'intermediate/CA_county_VIN_data.csv'

    vin_data = pd.read_csv(str_vin_data)
    return vin_data

def read_census_data(str_project,str_data,version):
    if version == 'hybrid':
        str_census = str_project + str_data + 'intermediate/census_pop_agg_ZEV.csv'
    elif version == 'hybrid/regional':
        str_census = str_project + str_data + 'intermediate/census_pop_agg_ZEV_regional.csv'
    elif 'county' in version:
        str_census = str_project + str_data + 'raw/census_pop_California_county.csv'
    else:
        str_census = str_project / str_data / 'raw/census_pop_state.csv'
    census_data = pd.read_csv(str_census)
    if 'county' not in version:
        census_data.state = census_data.state.str.upper()
    return census_data

def haircut_interpolate_census(str_project,str_data,census_data,incl_2021):
    str_haircut_mkt = str_project / str_data / 'intermediate/haircut_market_size.csv'
    haircut_param = pd.read_csv(str_haircut_mkt)
    early_param = float(haircut_param.loc[haircut_param.type == 'early','val'])
    late_param = float(haircut_param.loc[haircut_param.type == 'late','val'])

    census_data.loc[census_data.year == 2014,'tot_HH'] *= early_param
    print('Reducing 2014 tot_HH to account for non-observation of 2013 sales.')

    #assume 2020,2021 households the same as 2019
    census_2020 = census_data.loc[census_data.year == 2019].copy(deep=True).reset_index(drop=True)
    census_2020.year = 2020
    census_data = pd.concat([census_data,census_2020],
                            ignore_index=True)
    print("Warning: using 2019 households as 2020 households!")
    if(incl_2021):
        census_2021 = census_2020.copy()
        census_2021.year = 2021
        census_data = pd.concat([census_data,census_2021],
                            ignore_index=True)
        print("Warning: using 2019 households as 2021 households!")
        # haircut in analogous way to 2014 hhs
        # now have 2021 sales included!
        census_data.loc[census_data.year == 2021,'tot_HH'] *=late_param
        print('Reducing 2021 tot_HH to account for non-observation of 2022 sales.')


    # try reducing census pop by half -- only half of HHs looking for cars in given year
    #print('Warning: reducing populations by half')
    #census_data['tot_HH'] *= .5
    return census_data

def read_agent_data(str_project,str_data,model,version,n_agent,incl_2021,bln_rescale_income,bln_truncate_income):
    state_names, region_names = gen_state_region_names()

    agent_data = None
    if model in ['rc_demo','rc_demo_moments','rc_nl','rc_nl_moments']:
        if version == 'state' or version in state_names or version in region_names:
            str_agent = str_project + str_data + 'intermediate/agent_data.csv'
        elif version == 'hybrid':
            str_agent = str_project + str_data + 'intermediate/agent_data_agg_state_ZEV.csv'
        elif version == 'hybrid/regional':
            str_agent = str_project + str_data + 'intermediate/agent_data_agg_state_ZEV_regional.csv'

        agent_data = pd.read_csv(str_agent)
        # define income groups (there is certainly a nicer way to do this)
        if(False):
            agent_data['inc_lb'] = 1000
            agent_data.loc[agent_data.income > 50,'inc_lb'] = 50000
            agent_data.loc[agent_data.income > 75,'inc_lb'] = 75000
            agent_data.loc[agent_data.income > 100,'inc_lb'] = 100000
            agent_data.loc[agent_data.income > 150,'inc_lb'] = 150000
            agent_data.loc[agent_data.income > 250,'inc_lb'] = 250000

        if 'income' in agent_data:
            agent_data['low'] = 0
            agent_data.loc[agent_data.income < 100,'low'] = 1
            agent_data['mid'] = 0
            agent_data.loc[(agent_data.low == 0) & (agent_data.income < 200),'mid'] = 1
            agent_data['high'] = 0
            agent_data.loc[agent_data.income >= 200,'high'] = 1

        # keep smaller subset of agent_data
        agent_data_full = agent_data.copy()
        agent_data_small = pd.DataFrame()
        for mkt in agent_data.market_ids.unique():
            agent_data_mkt = agent_data_full.loc[agent_data_full.market_ids == mkt].reset_index(drop=True)
            if n_agent < 5000:
                agent_data_mkt = agent_data_mkt.sample(n = n_agent,random_state = 36)
            else:
                agent_data_mkt = agent_data_mkt.sample(n = n_agent,random_state = 36,replace=True)
            agent_data_small = pd.concat([agent_data_small,agent_data_mkt])
        agent_data = agent_data_small
        agent_data.reset_index(drop=True)
        if(incl_2021 == False):
            agent_data = agent_data.loc[agent_data.model_year < 2021].reset_index(drop=True)
        if bln_truncate_income:
            agent_data.loc[agent_data.income > 500,'income'] = 500
        if bln_rescale_income:
            agent_data.income = agent_data.income/100
        agent_data.state = agent_data.state.str.upper()
        agent_data.weights = 1/n_agent
    return agent_data

def read_moments(str_project,str_data,split,bln_rescale_income):
    str_moment_sf = str_project + str_data + 'final/ev_phev_sf_moments.csv'
    df_moments_sf = pd.read_csv(str_moment_sf)
    str_moment_educ = str_project + str_data + 'final/ev_phev_educ_moments.csv'
    df_moments_educ = pd.read_csv(str_moment_educ)
    if(split == 't1'):
        str_moment_urban = str_project + str_data + 'final/nhts_ev_phev_urban_moments.csv'
    elif(split == 'geog'):
        str_moment_urban = str_project + str_data + 'final/ev_phev_urban_moments_geog.csv'
    else:
        str_moment_urban = str_project + str_data + 'final/ev_phev_urban_moments.csv'
    df_moments_urban = pd.read_csv(str_moment_urban)
    str_moment_income = str_project + str_data + 'final/ev_phev_inc_moments.csv'
    df_moments_income = pd.read_csv(str_moment_income)
    if(split == 't1'):
        str_moment_mean_income = str_project + str_data + 'final/nhts_ev_phev_inc_mean_moments.csv'
    else:
        str_moment_mean_income = str_project + str_data + 'final/ev_phev_inc_mean_moments.csv'
    df_moments_mean_income =  pd.read_csv(str_moment_mean_income)
    if bln_rescale_income:
        df_moments_mean_income.moment = df_moments_mean_income.moment/100
    str_moments_income_grp = str_project + str_data + 'final/allveh_msrp_income_grp_moments.csv'
    df_moments_income_grp = pd.read_csv(str_moments_income_grp)
    dict_moments = {'sf': df_moments_sf,
                    'educ': df_moments_educ,
                    'urban': df_moments_urban,
                    'income_ev': df_moments_income,
                    'mean_income_ev': df_moments_mean_income,
                    'income_grp': df_moments_income_grp}
    return dict_moments

def gen_state_region_names():
    state_names=["Alabama", "Alaska", "Arizona", "Arkansas", "California", "Colorado", "Connecticut", "Delaware", "Florida", "Georgia", "Hawaii", "Idaho", "Illinois", "Indiana", "Iowa", "Kansas", "Kentucky", "Louisiana", "Maine", "Maryland", "Massachusetts", "Michigan", "Minnesota", "Mississippi", "Missouri", "Montana", "Nebraska", "Nevada", "New Hampshire", "New Jersey", "New Mexico", "New York", "North Carolina", "North Dakota", "Ohio", "Oklahoma", "Oregon", "Pennsylvania", "Rhode Island", "South Carolina", "South Dakota", "Tennessee", "Texas", "Utah", "Vermont", "Virginia", "Washington", "West Virginia", "Wisconsin", "Wyoming"]
    state_names =  [s.upper() for s in state_names]
    region_names = ["California", "West", "Mountain", "West North Central", "East North Central", "West South Central", "East South Central", "South Atlantic", "Middle Atlantic", "New England"]
    return state_names, region_names


def merge_vin_census(vin_data,census_data,version,dynamic):
    state_names, region_names = gen_state_region_names()

    # rename year in vin data
    vin_data = vin_data.rename(columns={'year':'model_year','county':'fips'})

    # adjust relevant year column in census data 
    # for model year X, we care about population in year X
    census_data['model_year'] = census_data.year

    # merge census data with vin data
    if version in ['state','hybrid','hybrid/regional'] or version in state_names or version in region_names:
        # merge by year, state
        mkt_data = pd.merge(vin_data,census_data[['tot_HH','model_year','state']],how='left',on = ['state','model_year'])
        # calculate the market share for the next year vehicle
        if dynamic:
            # get unique set of vehicle-state-share
            dynamic_vehicles = mkt_data[['dyn_product_ids','state','next_year','dyn_agg_count']].drop_duplicates().reset_index(drop=True)
            dynamic_vehicles = dynamic_vehicles.rename(columns = {'next_year':'model_year'})
            dynamic_vehicles = pd.merge(dynamic_vehicles,census_data[['tot_HH','model_year','state']],how='left',on = ['state','model_year'])
            # for now, use total household size from 2019 for 2020
            alt_census = census_data.loc[census_data.model_year == 2019,].copy()
            alt_census = alt_census.rename(columns = {'tot_HH':'2019_HH'})
            dynamic_vehicles = pd.merge(dynamic_vehicles,alt_census[['2019_HH','state']],how='left',on = ['state'])
            dynamic_vehicles.loc[(dynamic_vehicles.tot_HH.isnull()) & (dynamic_vehicles.model_year == 2020),'tot_HH'] = dynamic_vehicles['2019_HH']
            # calculate market share
            dynamic_vehicles['dyn_share'] = dynamic_vehicles['dyn_agg_count']/dynamic_vehicles['tot_HH']
            # re-merge mkt_data
            mkt_data = pd.merge(mkt_data,dynamic_vehicles[['dyn_product_ids','state','dyn_share']],how='left',on=['state','dyn_product_ids'])
    elif 'county' in version:
        # merge by year, fips
        mkt_data = pd.merge(vin_data,census_data[['tot_HH','model_year','fips']],how='left',on=['fips','model_year'])
    else:
        # sum census data to national level
        census_data_natl = census_data[['tot_HH','model_year']].groupby(['model_year']).sum().reset_index()
        mkt_data = pd.merge(vin_data,census_data_natl,how='left',on='model_year')

    return mkt_data

def clean_market_data(mkt_data,version):
    # drop observations with missing data
    if version == 'national':
        mkt_data = mkt_data.rename(columns={'dollar_per_mile_US':'dollar_per_mile'})

    mkt_data = mkt_data.dropna(subset=['tot_HH','dollar_per_mile','curbwt','drivetype']).reset_index(drop=True)

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
    mkt_data['shares'] = mkt_data['agg_count']/mkt_data.tot_HH

    # drop observations with market share below the 5th percentile
    mkt_data = mkt_data.loc[mkt_data.shares > np.percentile(mkt_data.shares,5)].reset_index(drop = True)

    # update column names for pyblp
    if version == 'national':
        mkt_data = mkt_data.rename(columns={'model_year':'market_ids'})
    elif 'county' in version:
        mkt_data['market_ids'] = mkt_data.model_year.astype(str)+'county'+mkt_data.fips.astype(str)
    else:
        mkt_data['market_ids'] = mkt_data.model_year.astype(str)+mkt_data.state
    mkt_data['prices'] = mkt_data.msrp

    # mkt_data['time_trend'] = mkt_data.model_year - 2013

    # define ice indicator
    mkt_data['ice'] = 1
    mkt_data.loc[mkt_data.fuel.isin(['electric','PHEV']),'ice'] = 0

    # define ice mpg
    mkt_data['mpg_ice'] = 0
    mkt_data.loc[mkt_data.ice == 1,'mpg_ice'] = mkt_data.loc[mkt_data.ice == 1,'combined_mpg2008']
    mkt_data['gpm_ice'] = 0
    mkt_data.loc[mkt_data.ice == 1,'gpm_ice'] = 1/mkt_data.loc[mkt_data.ice == 1,'combined_mpg2008']
    mkt_data['log_mpg_ice'] = 0
    mkt_data.loc[mkt_data.ice == 1,'log_mpg_ice'] = np.log(mkt_data.loc[mkt_data.ice == 1,'combined_mpg2008'])

    # add clustering id so standard errors can be clustered at product level
    mkt_data['clustering_ids'] = mkt_data.product_ids

    # set prices = price - subsidy
    mkt_data.prices = mkt_data.msrp - mkt_data.fed_tax_credit
   # mkt_data.prices = mkt_data.msrp - mkt_data.fed_tax_credit - mkt_data.rebate

    # add indicator for CA, other ZEV states
    #if ('CALIFORNIA' in mkt_data.state.unique()):
    #    mkt_data['CA']= 0
    #    mkt_data.loc[mkt_data.state == 'CALIFORNIA', 'CA']= 1
        # mkt_data['zev_nonCA'] = 0
        # mkt_data.loc[(mkt_data.state != 'CALIFORNIA') & (mkt_data.zev == 1), 'zev_nonCA'] = 1

    return mkt_data

def calc_outside_good(mkt_data,version):
    state_names, region_names = gen_state_region_names()

    if version in ['state','hybrid','hybrid/regional'] or version in state_names or version in region_names:
        outside_good = mkt_data[['state','model_year','shares']].groupby(['state','model_year']).sum().reset_index()
        outside_good['outside_share'] = 1- outside_good.shares
        # merge back into data
        mkt_data = pd.merge(mkt_data,outside_good[['state','model_year','outside_share']],how='left',on=['model_year','state'])
    elif 'county' in version:
        outside_good = mkt_data[['fips','model_year','shares']].groupby(['fips','model_year']).sum().reset_index()
        outside_good['outside_share'] = 1- outside_good.shares
        # merge back into data
        mkt_data = pd.merge(mkt_data,outside_good[['fips','model_year','outside_share']],how='left',on=['model_year','fips'])

    return mkt_data

def generate_pyblp_instruments(mkt_data):
    # generate instruments at national level!
    instrument_data = mkt_data[['product_ids','firm_ids','market_ids','wheelbase','curbwt','doors','log_hp_weight','drivetype','bodytype','wages']].drop_duplicates().reset_index(drop=True)
    # instrument_data['market_ids'] =  instrument_data.model_year

    # original BLP instruments
    if(False):
        demand_instruments = pyblp.build_blp_instruments(pyblp.Formulation('0 + wheelbase + doors + curbwt + C(bodytype)'), instrument_data)
    #demand_instruments = pyblp.build_differentiation_instruments(pyblp.Formulation('0+ wheelbase + doors + max_hp + C(bodytype)'), mkt_data,version='quadratic')
    #demand_instruments = pyblp.build_differentiation_instruments(pyblp.Formulation('0+ wheelbase + curbwt'), instrument_data,version='quadratic')

    # improved instruments
    # separate differentiation instruments (continuous chars) AND discrete instrument (for each product, count own- and rival- products with same values)
    demand_instruments_continuous1 = pyblp.build_differentiation_instruments(pyblp.Formulation('0 + wheelbase + curbwt'), instrument_data)
    demand_instruments_continuous2 = pyblp.build_differentiation_instruments(pyblp.Formulation('0 + wheelbase + curbwt'), instrument_data, version='quadratic')
    # discrete instruments
    lst_discrete_chars = ['doors','drivetype','bodytype']
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
# test instrument strength!
if(False):
    y = mkt_data['prices']
    #skipping categorical variables (make, drivetype, bodtytype)
    x = mkt_data[col_names + ['dollar_per_mile','electric','phev','hybrid','diesel','log_hp_weight','wheelbase','doors','range_elec']]
    model = sm.OLS(y, x).fit()
    model.summary()

def subset_states(mkt_data,agent_data,version):
    state_names, region_names = gen_state_region_names()

    if version in state_names:
        mkt_data = mkt_data.loc[mkt_data.state == version]
        if model in ['rc_demo','rc_demo_moments','rc_nl']:
            agent_data = agent_data.loc[agent_data.state == version]
    if version in region_names:
        mkt_data = mkt_data.loc[mkt_data.state_grp == version]
        if model in ['rc_demo','rc_demo_moments','rc_nl']:
            agent_data = agent_data.loc[agent_data.state.isin(mkt_data.state.unique())]
    return mkt_data,agent_data

def subset_years(mkt_data,agent_data,model,split,dict_moments):
    if split == 't1':
        yr_keep = [2014,2015]
    elif split == 't2':
        yr_keep = [2016,2017]
    elif split == 't3':
        yr_keep = [2018,2019,2020]
    else:
        yr_keep = mkt_data.model_year.unique()

    # keep only relevant mkt_data
    mkt_data = mkt_data.loc[mkt_data.model_year.isin(yr_keep)].reset_index(drop=True)
    # keep only relevant agent_data
    if model in ['rc_demo','rc_demo_moments','rc_nl']:
        agent_data = agent_data.loc[agent_data.model_year.isin(yr_keep)].reset_index(drop=True)

    for k in dict_moments.keys():
        if (k not in ['income_ev','income_grp']):
            df_moment = keep_moment_years(dict_moments[k],split,yr_keep)
            dict_moments[k] = df_moment

    return mkt_data,agent_data,dict_moments,yr_keep

# keep relevant moments based on split variable
def keep_moment_years(df,split,yr_keep):
    if split == 'None' or split == 'geog':
        df = df.loc[df.year == 0]
    elif split == 't1':
        print('Warning: using NHTS moments for time period t1')
    elif split in ['t2','t3','None, ann. moments']:
        df = df.loc[df.year.isin(yr_keep)]
    return(df)


# generate micro moments for old version of pyblp
def moments_v_10_12(mkt_data,agent_data,dict_X2,lst_demogs,split,yr_keep,dict_moments):
    df_moments_urban = dict_moments['urban']
    df_moments_income=dict_moments['income_ev']
    df_moments_mean_income=dict_moments['mean_income_ev']
    ev_x2_index = list(dict_X2.keys()).index('electric')
    phev_x2_index = list(dict_X2.keys()).index('phev')
    ## URBAN EV MOMENTS
    # vehicle characteristic expectation moments: expectation of vehicle characteristics conditional on demographics
    lst_ev_cond_urban_moments = []
    # determine if we're separating years or not
    if split == 'geog':
        for g in mkt_data.state_grp.unique():
            lst_states = mkt_data.loc[mkt_data.state_grp == g,'state'].unique()
            agent_urban_g = list(agent_data.loc[(agent_data.urban ==1) & (agent_data.state.isin(lst_states)),'agent_ids'])
            mkt_ids_g = list(mkt_data.loc[(mkt_data.state_grp == g) & (mkt_data.model_year.between(2017,2019,inclusive='both')),'market_ids'].unique())
            val_ev_g = float(df_moments_urban.loc[(df_moments_urban.char_exp == 1) & (df_moments_urban.ev == 1) & (df_moments_urban.state_grp == g),'moment'])
            n_obs_ev_g = int(df_moments_urban.loc[(df_moments_urban.char_exp == 1) & (df_moments_urban.ev == 1) & (df_moments_urban.state_grp == g),'n_obs'])
            ev_cond_urban_g = pyblp.CharacteristicExpectationMoment(agent_ids = agent_urban_g, # agent ids
                                                                    X2_index = ev_x2_index, #index in X2
                                                                    value =  val_ev_g, # moment value
                                                                    observations = n_obs_ev_g,
                                                                    market_ids = mkt_ids_g, )
            val_phev_g = float(df_moments_urban.loc[(df_moments_urban.char_exp == 1) & (df_moments_urban.phev == 1) & (df_moments_urban.state_grp == g),'moment'])
            n_obs_phev_g = int(df_moments_urban.loc[(df_moments_urban.char_exp == 1) & (df_moments_urban.phev == 1) & (df_moments_urban.state_grp == g),'n_obs'])
            phev_cond_urban_g = pyblp.CharacteristicExpectationMoment(agent_ids = agent_urban_g, # agent ids
                                                                    X2_index = phev_x2_index, #index in X2
                                                                    value =  val_phev_g, # moment value
                                                                    observations = n_obs_phev_g,
                                                                    market_ids = mkt_ids_g, )
            lst_ev_cond_urban_moments = lst_ev_cond_urban_moments + [ev_cond_urban_g,phev_cond_urban_g]
    elif split != 'None' and split != 't1':
        for yr in yr_keep:
            # check if that year is in the main moment data
            if yr in df_moments_urban.year.unique():
                agent_urban_yr = list(agent_data.loc[(agent_data.urban == 1) & (agent_data.model_year == yr),'agent_ids'])
                mkt_ids_yr = list(mkt_data.loc[mkt_data.model_year == yr,'market_ids'].unique())
                val_ev_yr = float(df_moments_urban.loc[(df_moments_urban.char_exp == 1) & (df_moments_urban.ev == 1) & (df_moments_urban.year == yr),'moment'])
                n_obs_ev_yr = int(df_moments_urban.loc[(df_moments_urban.char_exp == 1) & (df_moments_urban.ev == 1) & (df_moments_urban.year == yr),'n_obs'])
                ev_cond_urban_yr = pyblp.CharacteristicExpectationMoment(agent_ids = agent_urban_yr, # agent ids
                                                                        X2_index = ev_x2_index, #index in X2
                                                                        value =  val_ev_yr, # moment value
                                                                        observations = n_obs_ev_yr,
                                                                        market_ids = mkt_ids_yr, )
                val_phev_yr = float(df_moments_urban.loc[(df_moments_urban.char_exp == 1) & (df_moments_urban.phev == 1) & (df_moments_urban.year == yr),'moment'])
                n_obs_phev_yr = int(df_moments_urban.loc[(df_moments_urban.char_exp == 1) & (df_moments_urban.phev == 1) & (df_moments_urban.year == yr),'n_obs'])
                phev_cond_urban_yr = pyblp.CharacteristicExpectationMoment(agent_ids = agent_urban_yr, # agent ids
                                                                        X2_index = phev_x2_index, #index in X2
                                                                        value =  val_phev_yr, # moment value
                                                                        observations = n_obs_phev_yr,
                                                                        market_ids = mkt_ids_yr, )
                lst_ev_cond_urban_moments = lst_ev_cond_urban_moments + [ev_cond_urban_yr,phev_cond_urban_yr]
            else:
                print('No urban moments available for '+str(yr))
    else:      
        # get agents in urban locations for years over which we have data
        if split == 't1':
            agent_urban = list(agent_data.loc[(agent_data.urban == 1) & (agent_data.model_year.isin(yr_keep)),'agent_ids'])
            mkt_ids = list(mkt_data.loc[mkt_data.model_year.isin(yr_keep),'market_ids'].unique())
        else:
            agent_urban = list(agent_data.loc[(agent_data.urban == 1) & (agent_data.model_year.between(2017,2019,inclusive='both')),'agent_ids'])
            mkt_ids = list(mkt_data.loc[mkt_data.model_year.between(2017,2019,inclusive='both'),'market_ids'].unique())

        # for now, don't have these moments for t1 (didn't est. from NHTS)
        if split != 't1':
            ev_cond_urban = pyblp.CharacteristicExpectationMoment(agent_ids = agent_urban, # agent ids
                                                                X2_index = ev_x2_index, #index in X2
                                                                value =  float(df_moments_urban.loc[(df_moments_urban.char_exp == 1) & (df_moments_urban.ev == 1),'moment']), # moment value
                                                                observations = int(df_moments_urban.loc[(df_moments_urban.char_exp == 1) & (df_moments_urban.ev == 1),'n_obs']),
                                                                market_ids = mkt_ids, )
            phev_cond_urban = pyblp.CharacteristicExpectationMoment(agent_ids = agent_urban, # agent ids
                                                                X2_index = phev_x2_index, #index in X2
                                                                value =  float(df_moments_urban.loc[(df_moments_urban.char_exp == 1) & (df_moments_urban.phev == 1),'moment']), # moment value
                                                                observations = int(df_moments_urban.loc[(df_moments_urban.char_exp == 1) & (df_moments_urban.phev == 1),'n_obs']),
                                                                market_ids = mkt_ids, ) 
            lst_ev_cond_urban_moments = [ev_cond_urban,phev_cond_urban]

    # demographic expectation moment: expectation of demographics for those who choose certain prods
    # mkt_data['product_ids'] = mkt_data.index
    lst_urban_cond_ev_moments = []
    demog_index = lst_demogs.index('urban')
    if split == 'geog':
        for g in mkt_data.state_grp.unique():
                evs_g = list(mkt_data.loc[(mkt_data.electric == 1) & (mkt_data.state_grp == g),'product_ids'].unique())
                phevs_g = list(mkt_data.loc[(mkt_data.phev == 1) & (mkt_data.state_grp == g),'product_ids'].unique())
                mkt_ids_g = list(mkt_data.loc[(mkt_data.state_grp == g) & (mkt_data.model_year.between(2017,2019,inclusive='both')),'market_ids'].unique())
                val_ev_g = float(df_moments_urban.loc[(df_moments_urban.demo_exp == 1) & (df_moments_urban.ev == 1) & (df_moments_urban.state_grp == g),'moment'])
                n_obs_ev_g = int(df_moments_urban.loc[(df_moments_urban.demo_exp == 1) & (df_moments_urban.ev == 1) & (df_moments_urban.state_grp == g),'n_obs'])
                urban_cond_ev_g = pyblp.DemographicExpectationMoment(product_ids = evs_g, # product ids
                                                                        demographics_index = demog_index, # index in demographic variables
                                                                        value = val_ev_g, # value
                                                                        observations = n_obs_ev_g,
                                                                        market_ids = mkt_ids_g
                                                                    )
                val_phev_g = float(df_moments_urban.loc[(df_moments_urban.demo_exp == 1) & (df_moments_urban.phev == 1) & (df_moments_urban.state_grp == g),'moment'])
                n_obs_phev_g = int(df_moments_urban.loc[(df_moments_urban.demo_exp == 1) & (df_moments_urban.phev == 1) & (df_moments_urban.state_grp == g),'n_obs'])
                urban_cond_phev_g = pyblp.DemographicExpectationMoment(product_ids = phevs_g, # product ids
                                                                        demographics_index = demog_index, # index in demographic variables
                                                                        value = val_phev_g, # value
                                                                        observations = n_obs_phev_g,
                                                                        market_ids = mkt_ids_g
                                                                        )
                lst_urban_cond_ev_moments = lst_urban_cond_ev_moments + [urban_cond_ev_g,urban_cond_phev_g]                
    elif split != 'None' and split != 't1':
        for yr in yr_keep:
            if yr in df_moments_urban.year.unique():
                evs_yr = list(mkt_data.loc[(mkt_data.electric == 1) & (mkt_data.model_year == yr),'product_ids'].unique())
                phevs_yr = list(mkt_data.loc[(mkt_data.phev == 1) & (mkt_data.model_year == yr),'product_ids'].unique())
                mkt_ids_yr = list(mkt_data.loc[mkt_data.model_year == yr,'market_ids'].unique())
                val_ev_yr = float(df_moments_urban.loc[(df_moments_urban.demo_exp == 1) & (df_moments_urban.ev == 1) & (df_moments_urban.year == yr),'moment'])
                n_obs_ev_yr = int(df_moments_urban.loc[(df_moments_urban.demo_exp == 1) & (df_moments_urban.ev == 1) & (df_moments_urban.year == yr),'n_obs'])
                urban_cond_ev = pyblp.DemographicExpectationMoment(product_ids = evs_yr, # product ids
                                                                demographics_index = demog_index, # index in demographic variables
                                                                value = val_ev_yr, # value
                                                                observations = n_obs_ev_yr,
                                                                market_ids = mkt_ids_yr
                                                                )
                val_phev_yr = float(df_moments_urban.loc[(df_moments_urban.demo_exp == 1) & (df_moments_urban.phev == 1) & (df_moments_urban.year == yr),'moment'])
                n_obs_phev_yr = int(df_moments_urban.loc[(df_moments_urban.demo_exp == 1) & (df_moments_urban.phev == 1) & (df_moments_urban.year == yr),'n_obs'])
                urban_cond_phev = pyblp.DemographicExpectationMoment(product_ids = phevs_yr, # product ids
                                                                    demographics_index = demog_index, # index in demographic variables
                                                                    value = val_phev_yr, # value
                                                                    observations = n_obs_phev_yr,
                                                                    market_ids = mkt_ids_yr
                                                                    )
                lst_urban_cond_ev_moments = lst_urban_cond_ev_moments + [urban_cond_ev,urban_cond_phev]
            else:
                print('No urban moments available for '+str(yr))
    else:
        if split == 't1':
            evs = list(mkt_data.loc[(mkt_data.electric == 1) & (mkt_data.model_year.isin(yr_keep)),'product_ids'].unique())
            phevs = list(mkt_data.loc[(mkt_data.phev == 1) & (mkt_data.model_year.isin(yr_keep)),'product_ids'].unique())
        else:
            evs = list(mkt_data.loc[(mkt_data.electric == 1) & (mkt_data.model_year.between(2017,2020,inclusive='both')),'product_ids'].unique())
            phevs = list(mkt_data.loc[(mkt_data.phev == 1) & (mkt_data.model_year.between(2017,2020,inclusive='both')),'product_ids'].unique())
        urban_cond_ev = pyblp.DemographicExpectationMoment(product_ids = evs, # product ids
                                                            demographics_index = demog_index, # index in demographic variables
                                                            value = float(df_moments_urban.loc[(df_moments_urban.demo_exp == 1) & (df_moments_urban.ev == 1),'moment']), # value
                                                            observations = int(df_moments_urban.loc[(df_moments_urban.demo_exp == 1) & (df_moments_urban.ev == 1),'n_obs']),
                                                            market_ids = mkt_ids
                                                            )
        urban_cond_phev = pyblp.DemographicExpectationMoment(product_ids = phevs, # product ids
                                                                demographics_index = demog_index, # index in demographic variables
                                                                value = float(df_moments_urban.loc[(df_moments_urban.demo_exp == 1) & (df_moments_urban.phev == 1),'moment']), # value
                                                                observations = int(df_moments_urban.loc[(df_moments_urban.demo_exp == 1) & (df_moments_urban.phev == 1),'n_obs']),
                                                                market_ids = mkt_ids
                                                            )
        lst_urban_cond_ev_moments = [urban_cond_ev, urban_cond_phev]
    ## INCOME EV MOMENTS
    lst_income_ev_moments = []
    if(False):
        for g in agent_data.inc_lb.unique():
            agent_g = list(agent_data.loc[(agent_data.inc_lb == g) & (agent_data.model_year.between(2017,2019,inclusive='both')),'agent_ids'])
            ev_prob_g = float(df_moments_income.loc[df_moments_income.inc_lb == g,'ev_prob'])
            phev_prob_g = float(df_moments_income.loc[df_moments_income.inc_lb == g,'phev_prob'])
            n_obs = int(df_moments_income.loc[df_moments_income.inc_lb == g,'ct_tot'])
            ev_cond_inc_g = pyblp.CharacteristicExpectationMoment(agent_ids = agent_g, # agent ids
                                                                X2_index = ev_x2_index,
                                                                value = ev_prob_g,
                                                                observations = n_obs,
                                                                market_ids = mkt_ids,)
            phev_cond_inc_g = pyblp.CharacteristicExpectationMoment(agent_ids = agent_g,
                                                                    X2_index = phev_x2_index,
                                                                    value = phev_prob_g,
                                                                    observations = n_obs,
                                                                    market_ids = mkt_ids,)
            lst_income_ev_moments= lst_income_ev_moments + [ev_cond_inc_g,phev_cond_inc_g]
    # mean income conditional on EV
    if(False): # temporarily getting rid of income moments
    #if ('income' in lst_demogs):
        inc_index = lst_demogs.index('income')
        # t1 doesn't have separate moments by state group
        if split == 't1':
            sub_mkt_ids = list(mkt_data.loc[mkt_data.model_year.isin(yr_keep),'market_ids'].unique())
            val_ev = float(df_moments_mean_income.loc[(df_moments_mean_income.demo_exp == 1) & 
                                                        (df_moments_mean_income.ev == 1),'moment'])
            n_obs_ev = int(df_moments_mean_income.loc[(df_moments_mean_income.demo_exp == 1) & 
                                                        (df_moments_mean_income.ev == 1),'n_obs'])
            val_phev = float(df_moments_mean_income.loc[(df_moments_mean_income.demo_exp == 1) & 
                                                        (df_moments_mean_income.phev == 1),'moment'])
            n_obs_phev = int(df_moments_mean_income.loc[(df_moments_mean_income.demo_exp == 1) & 
                                                        (df_moments_mean_income.phev == 1),'n_obs'])
            income_cond_ev_grp = pyblp.DemographicExpectationMoment(product_ids = evs,
                                                                    demographics_index = inc_index,
                                                                    value = val_ev,
                                                                    observations = n_obs_ev,
                                                                    market_ids = sub_mkt_ids)
            income_cond_phev_grp = pyblp.DemographicExpectationMoment(product_ids = phevs,
                                                                    demographics_index = inc_index,
                                                                    value = val_phev,
                                                                    observations = n_obs_phev,
                                                                    market_ids = sub_mkt_ids)
            lst_income_ev_moments = lst_income_ev_moments + [income_cond_ev_grp,income_cond_phev_grp]
        else:
            for state_grp in df_moments_mean_income.state_group.unique():
                if state_grp == 'CA':
                    lst_states = ['CALIFORNIA']
                elif state_grp == 'other ZEV':
                    lst_states = ['CONNECTICUT','MAINE','MARYLAND','MASSACHUSETTS','NEW JERSEY','NEW YORK','OREGON','RHODE ISLAND','VERMONT']
                elif state_grp == 'future ZEV':
                    lst_states = ['COLORADO','WASHINGTON']
                elif state_grp == 'other':
                    lst_states = ['GEORGIA','COMBINED']
                
                if split != 'None':
                    for yr in yr_keep:
                        if yr in df_moments_urban.year.unique():
                            sub_mkt_ids_yr = list(mkt_data.loc[(mkt_data.model_year == yr) &
                                                            (mkt_data.state.isin(lst_states)),'market_ids'].unique())
                            evs_yr = list(mkt_data.loc[(mkt_data.electric == 1) & (mkt_data.model_year == yr),'product_ids'].unique())
                            phevs_yr = list(mkt_data.loc[(mkt_data.phev == 1) & (mkt_data.model_year == yr),'product_ids'].unique())
                            val_ev_yr = float(df_moments_mean_income.loc[(df_moments_mean_income.demo_exp == 1) & 
                                                                        (df_moments_mean_income.ev == 1) &
                                                                        (df_moments_mean_income.state_group == state_grp) &
                                                                        (df_moments_mean_income.year == yr),'moment'])
                            n_obs_ev_yr = int(df_moments_mean_income.loc[(df_moments_mean_income.demo_exp == 1) & 
                                            (df_moments_mean_income.ev == 1) & 
                                            (df_moments_mean_income.state_group == state_grp) &
                                            (df_moments_mean_income.year == yr),'n_obs'])
                            income_cond_ev_yr = pyblp.DemographicExpectationMoment(product_ids = evs_yr,
                                                                                demographics_index = inc_index,
                                                                                value = val_ev_yr,
                                                                                observations = n_obs_ev_yr,
                                                                                market_ids = sub_mkt_ids_yr)                        
                            val_phev_yr = float(df_moments_mean_income.loc[(df_moments_mean_income.demo_exp == 1) &
                                                (df_moments_mean_income.phev == 1) &
                                                (df_moments_mean_income.state_group == state_grp) &
                                                (df_moments_mean_income.year == yr),'moment'])
                            n_obs_phev_yr = int(df_moments_mean_income.loc[(df_moments_mean_income.demo_exp == 1) & 
                                            (df_moments_mean_income.phev == 1) & 
                                            (df_moments_mean_income.state_group == state_grp) &
                                            (df_moments_mean_income.year == yr),'n_obs'])
                            income_cond_phev_yr = pyblp.DemographicExpectationMoment(product_ids = phevs_yr,
                                                                                    demographics_index = inc_index,
                                                                                    value = val_phev_yr,
                                                                                    observations = n_obs_phev_yr,
                                                                                    market_ids = sub_mkt_ids_yr)  
                            lst_income_ev_moments = lst_income_ev_moments + [income_cond_ev_yr,income_cond_phev_yr]
                        else:
                            print('No income moments available for '+str(yr))
                elif split == 'None':
                    sub_mkt_ids = list(mkt_data.loc[(mkt_data.model_year.between(2017,2019,inclusive='both')) &
                                                    (mkt_data.state.isin(lst_states)),'market_ids'].unique())
                    val_ev = float(df_moments_mean_income.loc[(df_moments_mean_income.demo_exp == 1) & 
                                                                (df_moments_mean_income.ev == 1) &
                                                                (df_moments_mean_income.state_group == state_grp),'moment'])
                    n_obs_ev = int(df_moments_mean_income.loc[(df_moments_mean_income.demo_exp == 1) & 
                                                                (df_moments_mean_income.ev == 1) & 
                                                                (df_moments_mean_income.state_group == state_grp),'n_obs'])
                    val_phev = float(df_moments_mean_income.loc[(df_moments_mean_income.demo_exp == 1) &
                                                                (df_moments_mean_income.phev == 1) &
                                                                (df_moments_mean_income.state_group == state_grp),'moment'])
                    n_obs_phev = int(df_moments_mean_income.loc[(df_moments_mean_income.demo_exp == 1) & 
                                                                (df_moments_mean_income.phev == 1) &
                                                                (df_moments_mean_income.state_group == state_grp),'n_obs'])
                    income_cond_ev_grp = pyblp.DemographicExpectationMoment(product_ids = evs,
                                                                        demographics_index = inc_index,
                                                                        value = val_ev,
                                                                        observations = n_obs_ev,
                                                                        market_ids = sub_mkt_ids)
                    income_cond_phev_grp = pyblp.DemographicExpectationMoment(product_ids = phevs,
                                                                        demographics_index = inc_index,
                                                                        value = val_phev,
                                                                        observations = n_obs_phev,
                                                                        market_ids = sub_mkt_ids)
                    lst_income_ev_moments = lst_income_ev_moments + [income_cond_ev_grp,income_cond_phev_grp]

    if(False): # temporarily getting rid of income moments
        if('income' in lst_demogs):
            micro_moments = micro_moments + lst_income_ev_moments
    if('urban' in lst_demogs):
        micro_moments = micro_moments + lst_urban_cond_ev_moments 
    return micro_moments

#define entry order based on previous year's marketshare
def stackelberg_order(mkt_data, year):
    #subset to prev year
    if year == 2014:
        prev_yr_mkt_data = mkt_data.loc[mkt_data.model_year == year].reset_index(drop=True)
    elif year > 2014:
        prev_yr_mkt_data = mkt_data.loc[mkt_data.model_year == year-1].reset_index(drop=True)
    #get marketshares
    prev_yr_mkt_data = prev_yr_mkt_data.groupby(['oem'])[['shares']].sum()
    #order in terms of rank
    entry_order = prev_yr_mkt_data.sort_values(by = 'shares', ascending = False).index.values
    return entry_order

#update mkt_sim to include new entrant within entry_cf_loop()
def update_mkt_sim(mkt_sim,best_entry,bln_agent,agent_sim,agent_weights,results):
    #update has_X_sibling and sibling product ID
    if best_entry['fuel'].values[0] == "PHEV":
        mkt_sim['has_sibling_PHEV'] = np.where((mkt_sim['has_sibling_PHEV'] == 1) | (mkt_sim['model'] == best_entry['model'].values[0]), 1, 0)
        best_entry['product_ids'] = best_entry['product_ids'] + 0.1
    elif best_entry['fuel'].values[0] == "electric":
        mkt_sim['has_sibling_BEV'] = np.where((mkt_sim['has_sibling_BEV'] == 1) | (mkt_sim['model'] == best_entry['model'].values[0]), 1, 0)
        best_entry['product_ids'] = best_entry['product_ids'] + 0.2
    else:
        sys.exit("ERROR: Entered sibling type invalid")
    
    best_entry['cf_sibling'] = 1
    
    #add best_entry model to mkt_sim
    mkt_sim.reset_index(drop=True, inplace=True)
    best_entry.reset_index(drop=True, inplace=True)
    mkt_sim = pd.concat([mkt_sim,best_entry])
    
    #recalculate equilibrium shares, msrp 
    #this is a copy of single_profit_cf but returning different object!
    
    # product chars
    vec_prods,product_chars,vec_msrp,mc,mat_diag,mat_ownership,mat_ZEV,vec_ghg_credits = product_objects(mkt_sim,results)
    mat_mkt_prod, mat_xi, mat_dpm_orig, mat_incentives, mat_shares_orig = get_mats_avail_xi_dpm_incentive('dollar_per_mile','tot_incentives',vec_prods,mkt_sim)

    # utility chars
    delta_const,coef_price,coef_fe,state_fes = utility_objects(product_chars,results,mkt_sim)
    mat_individ_prod_util = get_individ_utility(results,product_chars,mat_mkt_prod,mkt_sim,agent_sim,bln_agent,coef_price)

    vec_mkts,vec_mkt_size = small_mkt_objects(mkt_sim)
    ghg_credit_price = 50/1000
    zev_credit_price = 2000/1000

    mat_dpm = mat_dpm_orig
    price_orig = vec_msrp
    bln_zev_credit_endog = False
    bln_ghg_credit_endog = False
        
    # solve for equilbrium!
    price_cf_hat,ghg_credit_price,zev_credit_price =solve_price_credit_eq(bln_zev_credit_endog,bln_ghg_credit_endog,price_orig,ghg_credit_price,zev_credit_price,
                                                                        'addl_product',mat_incentives,mat_dpm_orig,mat_dpm,delta_const,mat_individ_prod_util,bln_agent,agent_weights,
                                                                        coef_price,coef_fe,mat_xi,mat_mkt_prod,
                                                                        vec_mkts,mat_diag,vec_mkt_size,mat_ownership,mat_ZEV,vec_ghg_credits,mc,state_fes)
    # calculate new profit
    # calculate shares, market outcomes with new price
    utilities_cf, probabilities_cf, shares_cf, conditionals_cf = get_shares(price_cf_hat,mat_incentives,mat_dpm,delta_const,mat_individ_prod_util,bln_agent,agent_weights,coef_price,coef_fe,mat_xi,mat_mkt_prod) 
    # q_noZEV = calc_q(price_cf_hat,mat_incentives,mat_dpm_orig,delta_const,coef_price,coef_fe,mat_xi,mat_mkt_prod,vec_mkt_size)

    # re-allocate shares, prices to simulation dataframe 
    df_newshares = pd.DataFrame(shares_cf)
    df_newshares = df_newshares.set_axis(vec_mkts, axis=1, inplace=False)
    df_newshares['product_ids'] = vec_prods
    df_newshares_l = pd.melt(df_newshares,id_vars = 'product_ids',value_vars = vec_mkts,var_name = 'market_ids',value_name='shares_cf')

    df_cf_prices = pd.DataFrame({'product_ids':vec_prods,'msrp_cf':price_cf_hat.flatten()})

    # merge with mkt_sim
    mkt_sim_cf = pd.merge(mkt_sim,df_newshares_l,on=['product_ids','market_ids'])
    mkt_sim_cf = pd.merge(mkt_sim_cf,df_cf_prices,on='product_ids')

    mkt_sim_cf = mkt_sim_cf.drop(['msrp','shares'], axis=1)
    
    mkt_sim_cf.rename(columns = {'msrp_cf':'msrp', 'shares_cf':'shares'}, inplace = True)
    
    #do any of these arguments need to exist? delta_const is usually defined as a separate vector...
    fordrop = ['1', 'curbwt_orig', 'delta_const', 'ghg_credit_orig',
       'log_hp_weight_orig', 'mat_shares_orig', 'mc_orig',
       'product_ids_orig', 'time_trend_alt','outside_share']
    if len(list(set(fordrop) & set(list(mkt_sim_cf.columns.values)))) > 0:
        mkt_sim_cf.drop(list(set(fordrop) & set(list(mkt_sim_cf.columns.values))), axis=1)
    
    #is there anything else to recalc here? - DON'T recalc outside share, as it breaks later code by generating duplicate column names
    
    #reset indices
    mkt_sim_cf.reset_index(drop=True, inplace=True)
    
    return mkt_sim_cf

#simulate counterfactual sibling entry
def entry_cf_loop(mkt_sim, mkt_data, year,product_chars,char_params,elec_price,diesel_price,gas_price,bln_agent,agent_sim,agent_weights,results):
    ticker = 1
    
    #define potential entrants
    df_prods_gas_bev, df_prods_gas_entered_bev = gen_potential_entrants(mkt_sim,product_chars,char_params,elec_price,diesel_price,gas_price,'BEV')
    df_prods_gas_phev, df_prods_gas_entered_phev = gen_potential_entrants(mkt_sim,product_chars,char_params,elec_price,diesel_price,gas_price,'PHEV')
    
    #replace df_products_gas_entered_* with empty df with same columns
    df_prods_gas_entered_bev = df_prods_gas_entered_bev.loc[df_prods_gas_entered_bev.fuel == 'go away!!!']
    df_prods_gas_entered_phev = df_prods_gas_entered_phev.loc[df_prods_gas_entered_phev.fuel == 'go away!!!']
    
    #generate stackelberg order
    entry_order = stackelberg_order(mkt_data, year)

    # credit price assumption
    ghg_credit_price = 50/1000
    zev_credit_price = 2000/1000
    
    #loop through order
    for firm_id in range(np.size(entry_order)):
        #calculate profit per firm
        mkt_sim['unit_net_credit_cost'] = mkt_sim.ghg_credit_new * ghg_credit_price + mkt_sim.net_credits * zev_credit_price
        mkt_sim['unit_profits_orig'] = mkt_sim.msrp - mkt_sim.mc + mkt_sim.unit_net_credit_cost
        mkt_sim['model_profits_orig'] = mkt_sim.unit_profits_orig * mkt_sim.shares * mkt_sim.tot_HH

        df_profit_orig = mkt_sim[['oem','model_profits_orig']].groupby('oem').sum().reset_index()
        
        #subset potential entrants to just firm
        firm = entry_order[firm_id]
        firm_bev_potential_entrants = df_prods_gas_bev.loc[df_prods_gas_bev.oem == firm]
        firm_phev_potential_entrants = df_prods_gas_phev.loc[df_prods_gas_phev.oem == firm]
        firm_profit_orig = df_profit_orig.loc[df_profit_orig.oem == firm]
        
        #generate firm delta_pi
        delta_pi_yr_firm_bev = fuel_specific_delta_pi_loops(firm_bev_potential_entrants,df_prods_gas_entered_bev,'BEV',year,mkt_sim,bln_agent,agent_sim,agent_weights,results,firm_profit_orig,ghg_credit_price,zev_credit_price)
        delta_pi_yr_firm_phev = fuel_specific_delta_pi_loops(firm_phev_potential_entrants,df_prods_gas_entered_phev,'PHEV',year,mkt_sim,bln_agent,agent_sim,agent_weights,results,firm_profit_orig,ghg_credit_price,zev_credit_price)
        
        delta_pi_yr_firm = pd.concat([delta_pi_yr_firm_bev,delta_pi_yr_firm_phev])
        
        #merge firm_products with expected sunk costs
        #CURRENTLY JUST DIRECTLY WRITING SINGLE-PARAMETER VERSION IN - in future, import the csv output of estimate_entry_costs.jl and merge
        delta_pi_yr_firm['entry_cost_fixed'] = 18750

        #add idiosyncratic sunk cost shocks
        delta_pi_yr_firm['entry_cost_shock'] = np.random.normal(0,3188.77551020408,delta_pi_yr_firm.shape[0])
        delta_pi_yr_firm['entry_cost'] = delta_pi_yr_firm.entry_cost_fixed + delta_pi_yr_firm.entry_cost_shock
        delta_pi_yr_firm['delta_pi_costed'] = delta_pi_yr_firm.delta_pi - delta_pi_yr_firm.entry_cost
        
        #update mkt_sim if best entrant is profitable
        print("Updating mkt_sim: " + firm)
        if delta_pi_yr_firm['delta_pi_costed'].max() > 0:
            #define best entrant
            best_entry_index = delta_pi_yr_firm['delta_pi_costed'].argmax()

            if delta_pi_yr_firm['fuel'].values[best_entry_index] == 'electric':
                best_entry = firm_bev_potential_entrants.loc[firm_bev_potential_entrants.model == delta_pi_yr_firm['model'].values[best_entry_index]]
            elif delta_pi_yr_firm['fuel'].values[best_entry_index] == 'PHEV':
                best_entry = firm_phev_potential_entrants.loc[firm_phev_potential_entrants.model == delta_pi_yr_firm['model'].values[best_entry_index]]
            else:
                sys.exit("ERROR: Best entry model index is invalid")

            #update mkt_sim
            mkt_sim = update_mkt_sim(mkt_sim,best_entry,bln_agent,agent_sim,agent_weights,results)
            print("Sibling added:")
            print(best_entry[['make','model','fuel']].drop_duplicates())
            if ticker == 1:
                sibling_entrants = best_entry
                ticker = 2
            else:
                sibling_entrants = pd.concat([sibling_entrants, best_entry])
        else:
            print("No sibling added.")
            
    print("Year " + str(year) + " entry simulation complete")
    
    return mkt_sim,sibling_entrants
    
#this is just a wrapper for readability - with persistent entry, utilities_orig as defined at first gets messed up (because it is wrong set of prods)
#therefore, as a (hopefully) temporary fix, I'm setting this function up to return the original objects before the counterfactual gets initiated
def find_utilities_orig(mkt_sim,results):
    # product chars
    vec_prods,product_chars,vec_msrp,mc,mat_diag,mat_ownership,mat_ZEV,vec_ghg_credits = product_objects(mkt_sim,results)
    
    mat_mkt_prod, mat_xi, mat_dpm_orig, mat_incentives, mat_shares_orig = get_mats_avail_xi_dpm_incentive('dollar_per_mile','tot_incentives',vec_prods,mkt_sim)
    #np.where(mat_xi == -999)

    product_chars['1'] = 1

    # utility chars
    delta_const,coef_price,coef_fe,state_fes = utility_objects(product_chars,results,mkt_sim)
    mat_individ_prod_util,coef_price = get_individ_utility(results,product_chars,mat_mkt_prod,mkt_sim,agent_sim,bln_agent,coef_price)

    # can't use pyblp simulation because of disconnect between price and msrp
    # calculate approximate price (like Knittel and Metaxoglou 2014; ideally, would adopt variant of Morrow and Skerlos 2011)

    # call functions
    vec_mkts,vec_mkt_size = small_mkt_objects(mkt_sim)
    ghg_credit_price = 50/1000
    zev_credit_price = 2000/1000
    zev_credit_price_orig = zev_credit_price
    ghg_credit_price_orig = ghg_credit_price

    elec_price_agg = aggregate_state_energy_prices(elec_price,'cent_per_kwh',mkt_sim)
    diesel_price_agg = aggregate_state_energy_prices(diesel_price,'dollar_per_gal_diesel',mkt_sim)
    gas_price_agg = aggregate_state_energy_prices(gas_price,'dollar_per_gal_gas',mkt_sim)

    # testing new shares calc
    # note: this won't work if resimulate_entry = True because it drops some vehicles
    if(False):
        utilities_orig, probabilities_orig, shares_orig = get_shares(vec_msrp,mat_incentives,mat_dpm_orig,delta_const,mat_individ_prod_util,bln_agent,agent_weights,coef_price,coef_fe,mat_xi,mat_mkt_prod,state_fes)
        df_newshares = pd.DataFrame(shares_orig)
        df_newshares = df_newshares.set_axis(vec_mkts, axis=1, inplace=False)
        df_newshares['product_ids'] = vec_prods
        df_newshares_l = pd.melt(df_newshares,id_vars = 'product_ids',value_vars = vec_mkts,var_name = 'market_ids',value_name='shares_cf')

        # merge with mkt_sim
        mkt_sim = pd.merge(mkt_sim,df_newshares_l,on=['product_ids','market_ids'])
        test =mkt_sim[['product_ids','shares','shares_cf']]
        test.loc[round(test.shares,6) != round(test.shares_cf,6)]


    # solve for original prices based on FOC
    price_orig_save = calc_price(vec_msrp,mat_incentives,zev_credit_price,ghg_credit_price,mat_dpm_orig,delta_const,mat_individ_prod_util,bln_agent,agent_weights,coef_price,coef_fe,mat_xi,mat_mkt_prod,vec_mkts,mat_diag,vec_mkt_size,mat_ownership,mat_ZEV,vec_ghg_credits,mc,state_fes)

    # check prices and shares approx. right
    abs(price_orig_save - vec_msrp).max()

    # original prices, shares
    price_orig = price_orig_save.copy()
    utilities_orig, probabilities_orig, shares_orig = get_shares(vec_msrp,mat_incentives,mat_dpm_orig,delta_const,mat_individ_prod_util,bln_agent,agent_weights,coef_price,coef_fe,mat_xi,mat_mkt_prod,state_fes)

    return utilities_orig, mat_mkt_prod, shares_orig

#another wrapper, for similar reason as above
#after the entry loop runs, we need to regenerate both product objects and equilibrium quantities/prices - this wrapper does that
def find_prices_utilities_cf_fresh_entry(mkt_sim,results,cf):
    # product chars
    vec_prods,product_chars,vec_msrp,mc,mat_diag,mat_ownership,mat_ZEV,vec_ghg_credits = product_objects(mkt_sim,results)
    
    mat_mkt_prod, mat_xi, mat_dpm_orig, mat_incentives, mat_shares_orig = get_mats_avail_xi_dpm_incentive('dollar_per_mile','tot_incentives',vec_prods,mkt_sim)
    #np.where(mat_xi == -999)

    product_chars['1'] = 1

    # utility chars
    delta_const,coef_price,coef_fe,state_fes = utility_objects(product_chars,results,mkt_sim)
    mat_individ_prod_util,coef_price = get_individ_utility(results,product_chars,mat_mkt_prod,mkt_sim,agent_sim,bln_agent,coef_price)

    # can't use pyblp simulation because of disconnect between price and msrp
    # calculate approximate price (like Knittel and Metaxoglou 2014; ideally, would adopt variant of Morrow and Skerlos 2011)

    # call functions
    vec_mkts,vec_mkt_size = small_mkt_objects(mkt_sim)
    ghg_credit_price = 50/1000
    zev_credit_price = 2000/1000
    zev_credit_price_orig = zev_credit_price
    ghg_credit_price_orig = ghg_credit_price

    elec_price_agg = aggregate_state_energy_prices(elec_price,'cent_per_kwh',mkt_sim)
    diesel_price_agg = aggregate_state_energy_prices(diesel_price,'dollar_per_gal_diesel',mkt_sim)
    gas_price_agg = aggregate_state_energy_prices(gas_price,'dollar_per_gal_gas',mkt_sim)

    # testing new shares calc
    # note: this won't work if resimulate_entry = True because it drops some vehicles
    if(False):
        utilities_orig, probabilities_orig, shares_orig = get_shares(vec_msrp,mat_incentives,mat_dpm_orig,delta_const,mat_individ_prod_util,bln_agent,agent_weights,coef_price,coef_fe,mat_xi,mat_mkt_prod,state_fes)
        df_newshares = pd.DataFrame(shares_orig)
        df_newshares = df_newshares.set_axis(vec_mkts, axis=1, inplace=False)
        df_newshares['product_ids'] = vec_prods
        df_newshares_l = pd.melt(df_newshares,id_vars = 'product_ids',value_vars = vec_mkts,var_name = 'market_ids',value_name='shares_cf')

        # merge with mkt_sim
        mkt_sim = pd.merge(mkt_sim,df_newshares_l,on=['product_ids','market_ids'])
        test =mkt_sim[['product_ids','shares','shares_cf']]
        test.loc[round(test.shares,6) != round(test.shares_cf,6)]


    # solve for original prices based on FOC
    price_orig_save = calc_price(vec_msrp,mat_incentives,zev_credit_price,ghg_credit_price,mat_dpm_orig,delta_const,mat_individ_prod_util,bln_agent,agent_weights,coef_price,coef_fe,mat_xi,mat_mkt_prod,vec_mkts,mat_diag,vec_mkt_size,mat_ownership,mat_ZEV,vec_ghg_credits,mc,state_fes)

    # check prices and shares approx. right
    abs(price_orig_save - vec_msrp).max()

    # original prices, shares
    price_orig = price_orig_save.copy()
    utilities_orig, probabilities_orig, shares_orig = get_shares(vec_msrp,mat_incentives,mat_dpm_orig,delta_const,mat_individ_prod_util,bln_agent,agent_weights,coef_price,coef_fe,mat_xi,mat_mkt_prod,state_fes)
    shares_prev = shares_orig

    # generate new dollar_per_mile amounts -- increase gasoline and diesel prices by 50c
    gas_price_new = gas_price_agg.copy()
    gas_price_new['dollar_per_gal_gas'] = gas_price_new['dollar_per_gal_gas'] + .5
    diesel_price_new = diesel_price_agg.copy()
    diesel_price_new['dollar_per_gal_diesel'] = diesel_price_new['dollar_per_gal_diesel'] + .5
    mkt_sim = calc_dollar_per_mile(elec_price_agg,diesel_price_new,gas_price_new,mkt_sim)
    mat_mkt_prod, mat_xi, mat_dpm_new, mat_incentives, mat_shares_orig = get_mats_avail_xi_dpm_incentive('dollar_per_mile_new','tot_incentives',vec_prods,mkt_sim)

    # choose which counterfactual
    if cf == 'high_gas':
        mat_dpm = mat_dpm_new
    else:
        mat_dpm = mat_dpm_orig
    # zev params
    if cf == 'no_ZEV':
        bln_zev_credit_endog = False
    else:
        bln_zev_credit_endog = True

    # ghg params
    bln_ghg_credit_endog = False

    #solve for equilbrium!
    price_cf_hat,ghg_credit_price,zev_credit_price =solve_price_credit_eq(bln_zev_credit_endog,bln_ghg_credit_endog,price_orig,ghg_credit_price,zev_credit_price,
                                                                      cf,mat_incentives,mat_dpm_orig,mat_dpm,delta_const,mat_individ_prod_util,bln_agent,agent_weights,coef_price,coef_fe,mat_xi,mat_mkt_prod,
                                                                      vec_mkts,mat_diag,vec_mkt_size,mat_ownership,mat_ZEV,vec_ghg_credits,mc,state_fes)

    # once iteration done, save results   
    # calculate shares, market outcomes with new price
    utilities_cf, probabilities_cf, shares_cf = get_shares(price_cf_hat,mat_incentives,mat_dpm,delta_const,mat_individ_prod_util,bln_agent,agent_weights,coef_price,coef_fe,mat_xi,mat_mkt_prod,state_fes) 
    # q_noZEV = calc_q(price_cf_hat,mat_incentives,mat_dpm_orig,delta_const,coef_price,coef_fe,mat_xi,mat_mkt_prod,vec_mkt_size)

    return vec_prods,product_chars,vec_mkts,vec_mkt_size,mat_mkt_prod,price_cf_hat,ghg_credit_price,zev_credit_price,utilities_cf,probabilities_cf,shares_cf
