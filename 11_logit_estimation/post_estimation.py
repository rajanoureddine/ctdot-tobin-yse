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
import matplotlib.pyplot as plt

from linearmodels.iv import IV2SLS # this is to check IV results

from Reference import functions_v2 as exp_functions
import functions_rlp as rlp_functions 

# silence warnings
import warnings
warnings.filterwarnings("ignore")

# Pyblp settings
pyblp.options.verbose = True

############################################################################################################
# Settings
version = "CONNECTICUT"
model = 'logit'
integ = 'gauss'
dynamic = False
incl_2021 = True
# rlp_market = 'model_year'
rlp_market ='county_model_year'
date_time = time.strftime("%m%d-%H%M")
zms_replaced_with = 0.01

############################################################################################################
# Set up main paths and directories
if platform.platform()[0:5] == 'macOS':
    on_cluster = False
    cd = pathlib.Path().resolve().parent
    str_project = cd / "Documents" 
elif platform.platform()[0:5] == 'Linux':
    on_cluster = True
    cd = pathlib.Path().resolve()
    str_project = cd.parent.parent / "rn_home" / "data"

# Set up sub-directories
str_data = str_project / "tobin_working_data"
str_rlp = str_data / "rlpolk_data"
str_sales_vin_characteristic = str_rlp / "rlp_with_dollar_per_mile.csv"
output_folder = str_project / str_data / "outputs"
estimation_data_dir = str_data / "estimation_data_test" / "estimation_data_county_model_year_0601-0700" 
estimation_data = estimation_data_dir /"rc_mkt_data_county_model_year_0601-0700.csv"


############################################################################################################
# Set output dir
output_dir = output_folder / "outputs_county_model_year_0601-0700"
results_pkl = output_dir / "outputs_rand_coeffs_county_model_year_0601-0700.pkl"
estimation_data = pd.read_csv(estimation_data)

# Unpickle results
with open(results_pkl, "rb") as f:
    results = pickle.load(f)

# Get NHV 2022
nhv_2022 = estimation_data.market_ids == "NEW HAVEN_2022"
ev_indexes = estimation_data[nhv_2022].electric == 1
nev_indexes = estimation_data[nhv_2022].electric == 0

# Get all elasticities
elasticities = results.compute_elasticities()
nhv_elasticities = elasticities[nhv_2022]
assert np.size(nhv_elasticities) == 315*315

if False:
    # Plot
    fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (10, 6))
    ax.hist(np.diag(nhv_elasticities)[ev_indexes], bins = 100, color = 'blue', label = "EVs", zorder = 2)
    ax.hist(np.diag(nhv_elasticities)[nev_indexes], bins = 100, color = 'red', label = "Non-EVs", zorder = 1)

    # Add vertical line at mean of EVs and non-EVs
    mean_ev = np.mean(np.diag(nhv_elasticities)[ev_indexes])
    mean_nev = np.mean(np.diag(nhv_elasticities)[nev_indexes])
    ax.axvline(x = mean_ev, color = 'blue', linestyle = '--', label = f"Mean EV: {mean_ev:.2f}")
    ax.axvline(x = mean_nev, color = 'red', linestyle = '--', label = f"Mean Non-EV: {mean_nev:.2f}")

    # Name axes
    ax.set_xlabel("Own Price Elasticity")
    ax.set_ylabel("Frequency")

    # Add legend
    ax.legend()

    # Add title
    ax.set_title("Distribution of Own Price Elasticities for New Haven 2022")

    plt.show()

############################################################################################################
# Get elasticities for dollars per mile
# Get all elasticities
elasticities_dpm = results.compute_elasticities("dollar_per_mile")
nhv_elasticities_dpm = elasticities_dpm[nhv_2022]

# Plot
fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (10, 6))
ax.hist(np.diag(nhv_elasticities_dpm), bins = 100, color = 'blue', label = "All", zorder = 2)

# Add vertical line at mean of EVs and non-EVs
mean_all = np.mean(np.diag(nhv_elasticities_dpm))

ax.axvline(x = mean_all, color = 'blue', linestyle = '--', label = f"Mean All: {mean_all:.2f}")

# Name axes
ax.set_xlabel("Own Price Elasticity")
ax.set_ylabel("Frequency")

# Add legend
ax.legend()

# Add title
ax.set_title("Distribution of Own Price Elasticities for New Haven 2022 - Dollars per Mile")

plt.show()

############################################################################################################
# Plot DPM elasticities for EVs and non-EVs
fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (10, 6))
ax.hist(np.diag(nhv_elasticities_dpm)[ev_indexes], bins = 50, color = 'blue', label = "EVs", zorder = 2)
ax.hist(np.diag(nhv_elasticities_dpm)[nev_indexes], bins = 50, color = 'red', label = "Non-EVs", zorder = 1)

# Add vertical line at mean of EVs and non-EVs
mean_ev_dpm = np.mean(np.diag(nhv_elasticities_dpm)[ev_indexes])
mean_nev_dpm = np.mean(np.diag(nhv_elasticities_dpm)[nev_indexes])

ax.axvline(x = mean_ev_dpm, color = 'blue', linestyle = '--', label = f"Mean EV: {mean_ev_dpm:.2f}")
ax.axvline(x = mean_nev_dpm, color = 'red', linestyle = '--', label = f"Mean Non-EV: {mean_nev_dpm:.2f}")

# Name axes
ax.set_xlabel("Own Price Elasticity")
ax.set_ylabel("Frequency")

# Add legend
ax.legend()

# Add title
ax.set_title("Distribution of Own Price Elasticities for New Haven 2022 - Dollars per Mile")

plt.show()
