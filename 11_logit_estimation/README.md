# logit_estimation
This folder contains all scripts used to run the demand estimation component part of the project.

The folder contains the following main scripts:
* `estimate_demand_compare_old.py`: This script runs a simple logit model without any random coefficients. At every step of the way, it compares it compares the outputs to the outputs when run with another dataset - the Experian dataset - that was used for Professor Gillingham's other project. There is no need to compare these too closely. Some differences in the estimated parameters are likely a result of differences in VIN matching, etc. 
* `estimate_demand_rc.py`: This script runs the BLP estimation when using agent data, random coefficients, micro-moments, etc.

Other supporting scripts include
* `functions_rlp.py`: This contains helper functions used to clean and prepare the RLPolk data for estimation in `estimate_demand_compare_old.py` and `estimate_demand_rc.py`.
* `get_results.py` Is used to extract result dataframes from Pickled reult objects produced by PyBLP
* `post_estimation.py` Is used to produce elasticities etc.

Note that the Experian estimation is not a major concern for this project - we simply compared to it early on in the project to ensure we were heading in the right direction. 

# Files
## estimate_demand_compare_old.py
This script is used to run a simple BLP demand estimation, with no random coefficients at all. 

The script runs a variety of different specifications, and for each one, compares the output to the Experian data output (Note again: This isn't particularly important - the goal was simply to ensure our estimates were similar to those produced using the Experian data).

The file makes use of a range of helper functions comntained in `functions_rlp.py` and also saves the final data used for estimation, to allow us to observe and replicate if needed.

## estimate_demand_rc.py
This script is used to run a BLP demand estimation, with random coefficients, with or without agent data, and with or without micro-moments. Note that where the details of an estimation are unclear, and a pickle file of the results is available, more detail can be gained by loading the pickle file of the results. This will be a PyBLP results object that contains information such as: the product formulation used, the agent formulation used, the micro-moments used, etc. 

### Running without agent data
Running without agent data, equates to a model in which agents exhibit random heterogeneity in their tastes for specific vehicle features, but this heterogeneity is uncorrelated with any observed characteristics of the agent (i.e., the randomness is simply drawn from a distribution - and not related to the agent's characteristics). 

Note that we have found that when attempting to estimate a model with random heterogeneity, without any kind of micro-moment, we are unable to pin down parameter estimates for Sigma (i.e., the random heterogeneity). 

When running this kind of model, we use: `run_rc_logit_model(rlp_mkt_data, output_subfolder, estimation_data_subfolder)`. That is, set `agent_data = None, use_micro_moments = False, micro_moments_to_include = None)`.

`X1_formulation_str` contains the product features over which agents have non-random preferences. There is no interaction between random heterogeneity and these features.

`X2_formulation_str` contains the product features over which agents have random preferences. These features are interacted with random heterogeneity drawn from a gaussian distribution. 

### Running without agent data but with micro-moments
See section 4.3.1 for the PDF for the results of this estimation. 

We employ micro-data to help pin down coefficients on random variation. See the results: `outputs_county_model_year_0619-1033`. In that example, we use the micro-moments data to help pin down random variation in tastes for EVs (and identify significant variation in tastes for EVs). See Section 4.3 of the Overleaf document / Methodology PDF for further detail on the methodology / approach taken.

The simplest approach using micro-moments is simply to include a micro-moment for conditional second choice of an EV (i.e., percentage choosing an EV as their second choice, conditional on having chosen an EV as their first choice). To do this: 
* X2 Formulation: `1+broad_ev_nohybrid` 
* Calculate two micro-parts: `E[broad_ev_1]` and `E[broad_ev_2 | broad_ev_1]`. The ratio of these two micro-parts is the desired micro-moment: conditional probability of second choice being an EV, given first choice is an EV. 
* It may be desirable to reduce `n_agent` to 750 (as the estimation can take a long time if too high) 
* It may be desirable to increase sensitivity (i.e., terminate optimization earlier)

### Running with agent data and micro-moments
See section 4.4.2 *Agent heterogeneity and income-specific price sensitivity* of the PDF for the results of this estimation. The code is set up to run this estimation with a few changes to the product formulations, agent formulation, and the list of micro-moments to be used. 


## post_estimation.py
A utility script used to extract cross-price elasticities for specific vehicle makes, once an estimation has been completed. It takes a results pickle, and the data used to estimate, as input. Note: This is why it is useful to save the exact data used to estimate the model, on each run. 

## get_results.py
A utility script used to extract results from pickled results files (`pickle` is a python library used for saving aribitrary Python objects).
