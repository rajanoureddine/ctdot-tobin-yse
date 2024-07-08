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
from IPython.display import display
import re 

from linearmodels.iv import IV2SLS # this is to check IV results

from Reference import functions_v2 as exp_functions
import functions_rlp as rlp_functions 

# silence warnings
import warnings
warnings.filterwarnings("ignore")

# Pyblp settings
pyblp.options.verbose = True

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

# Set up sub-directories to get the data from
str_data = str_project / "tobin_working_data"
estimation_data_dir = str_data / "estimation_data_test" / "estimation_data_county_model_year_0601-0700" # Data for the stimation
# results_pkl = str_data / "outputs" / "outputs_county_model_year_0601-0700"  /"outputs_rand_coeffs_county_model_year_0601-0700.pkl"
results_pkl = str_data / "outputs" / "outputs_county_model_year_0619-1033"  /"outputs_rand_coeffs_county_model_year_0619-1033.pkl"
output_folder = str_data / "post_estimation_outputs"


############################################################################################################
# Set output dir
estimation_data = pd.read_csv(estimation_data_dir /"rc_mkt_data_county_model_year_0601-0700.csv")
estimation_data["column header"] = estimation_data["make"]+" "+estimation_data["model"]+" "+estimation_data["trim"]

# For each unique make, model, trim, create dummy columns showing which years it is present in
estimation_data["code"] = estimation_data["make"]+"_"+estimation_data["model"]+"_"+estimation_data["trim"]
unique_codes = estimation_data["code"].unique()
for code in unique_codes:
    years = estimation_data.loc[estimation_data["code"] == code, "model_year"].unique()
    for year in years:
        estimation_data.loc[estimation_data.code == code, year] = 1


# Unpickle results
with open(results_pkl, "rb") as f:
    results = pickle.load(f)

def get_indices_within_market(estimation_data, market_id):
    estimation_data["code"] = estimation_data["make"]+"_"+estimation_data["model"]+"_"+estimation_data["trim"]+"_"+estimation_data["fuel"]+"_"+estimation_data["range_elec"].astype(str)
    data_2022 = estimation_data.loc[estimation_data["market_ids"] == "NEW HAVEN_2022", ["code"]].reset_index(drop = True)

    specified_mkt = estimation_data.loc[estimation_data["market_ids"] == market_id, ["code"]].reset_index(drop=True)

    # Now, for each code in the specified market, find the index in the 2022 market
    indices = []
    for ind, row in specified_mkt.iterrows():
        try:
            if len(data_2022[data_2022["code"] == row["code"]].index) > 1:
                print("a")
            indices = indices + data_2022[data_2022["code"] == row["code"]].index.tolist()
        except:
            pass
    
    return indices

def get_elasticities_market(results, market_id, variable = None, further_indices = None):
    """
    This function computes elasticities for a given market and variable
    """
    # Get elasticities
    if not variable:
        elasticities = results.compute_elasticities()
    else:
        elasticities = results.compute_elasticities(variable)
    
    # Get market indices
    market_indexes = estimation_data.market_ids == market_id

    # Get elasticities for the market
    market_elasticities = elasticities[market_indexes]

    # If further indices
    indices_out = []
    for index in further_indices:
        indices_out.append(index(estimation_data[market_indexes]))

    return market_elasticities, indices_out

def hist_plot_elasticities(elasticities_matrix, indices = None, labels = None, plot_mean_line = True):
    # Plot
    fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (10, 6))
    if indices:
        for i, index in enumerate(indices):
            ax.hist(np.diag(elasticities_matrix)[index], bins = 100, label = labels[i], alpha = 0.5)
            if plot_mean_line:
                mean_index = np.mean(np.diag(elasticities_matrix)[index])
                ax.axvline(x = mean_index, linestyle = '--', label = f"Mean {labels[i]}: {mean_index:.2f}")
    else:
        ax.hist(np.diag(elasticities_matrix), bins = 100)
        if plot_mean_line:
            mean_all = np.mean(np.diag(elasticities_matrix))
            ax.axvline(x = mean_all, linestyle = '--', label = f"Mean All: {mean_all:.2f}")

    # Name axes
    ax.set_xlabel("Own Price Elasticity")
    ax.set_ylabel("Frequency")

    # Add legend
    ax.legend()

    # Add title
    ax.set_title("Distribution of Own Price Elasticities for New Haven 2022")

    plt.show()





############################################################################################################
# Get elasticities for New Haven, along with indices for EVs and non-EVs, and plot
# nhv_elasticities, indexes = get_elasticities_market(results, "NEW HAVEN_2022", variable = None, further_indices= [lambda x: x.electric == 1, lambda x: x.electric == 0])
# hist_plot_elasticities(nhv_elasticities, [indexes[0], indexes[1]], ["EVs", "Non-EVs"])

# Get elasticities for New Haven, along with indices for EVs and non-EVs, for dollars per mile
# nhv_elasticities_dpm, indexes_dpm = get_elasticities_market(results, "NEW HAVEN_2022", variable = "dollar_per_mile", further_indices= [lambda x: x.electric == 1, lambda x: x.electric == 0])
# hist_plot_elasticities(nhv_elasticities_dpm, [indexes_dpm[0], indexes_dpm[1]], ["EVs", "Non-EVs"])

# Now we want to extract specific models.
if False:
    makes_models_trims = [["Tesla", "Model-3", "Long Range"], ["Tesla", "Model-Y", "Long Range"],["Chevrolet", "Bolt-Euv", "Premier"],
                        ["Acura", "Mdx", "SH-AWD w/Tech"], ["Toyota", "Rav4-Prime", "SE"], ["Ram", "Ram-Pickup-1500", "Big Horn"], 
                        ["Toyota", "Tacoma", "SR V6"], ["Jeep", "Wrangler-Unlimited", "Sahara 4xe"]]

    funs = [lambda x, y=y: ((x.make==y[0])&(x.model==y[1])&(x.trim==y[2])) for y in makes_models_trims]
    nhv_elasticities, make_indices = get_elasticities_market(results, "NEW HAVEN_2022", variable = None, further_indices= funs)

    # Make into table
    desired_indices = np.reshape(np.where(np.sum(make_indices, axis = 0)), -1)
    elasticities = pd.DataFrame(nhv_elasticities, columns = np.arange(0, len(nhv_elasticities)))
    mat = elasticities.loc[desired_indices, desired_indices]
    mat.index = [makes_modes_trims[0]+" "+makes_modes_trims[1]+" "+makes_modes_trims[2] for makes_modes_trims in makes_models_trims]
    mat.columns = mat.index
    mat=mat.reset_index().rename(columns = {"index": "Model"})
    mat.columns = ["\multicolumn{1}{m{2cm}}{\centering "+str(x) + "}" for x in mat.columns]

    lat_str = mat.to_latex(index = False, 
                    float_format="%.4f",
                    formatters = {mat.columns[0]:lambda x: "\multicolumn{1}{m{2.5cm}}{\centering "+x+"}"})

    lat_str = re.sub(r'(toprule|midrule|bottomrule)', r'hline', lat_str)
    lat_str = re.sub(r'\\\\', r'\\\\[0.4cm]', lat_str)
    print(lat_str)


############################################################################################################
# Get elasticities for another market using the new functionality we've discovered
if False:
    elasticities_21 = results.compute_elasticities("prices", market_id = "NEW HAVEN_2021")
    elasticities_21 = pd.DataFrame(elasticities_21, columns = np.arange(0, len(elasticities_21)))
    estimation_data["column header"] = estimation_data["make"]+" "+estimation_data["model"]+" "+estimation_data["trim"]
    elasticities_21.index = estimation_data.loc[estimation_data["market_ids"] == "NEW HAVEN_2021", "column header"]
    elasticities_21.columns = elasticities_21.index

    wanted_cols = ["Tesla Model-3 Long Range", "Tesla Model-Y Long Range", "Chevrolet Bolt-Euv Premier", "Acura Mdx SH-AWD w/Tech", "Toyota Rav4-Prime SE", "Ram Ram-Pickup-1500 Big Horn", "Toyota Tacoma SR V6", "Jeep Wrangler-Unlimited Sahara 4xe"]

    elasticities_21 = elasticities_21.loc[wanted_cols, wanted_cols]


    mat = elasticities_21
    mat=mat.reset_index().rename(columns = {"index": "Model"})
    mat.columns = ["\multicolumn{1}{m{2cm}}{\centering "+str(x) + "}" for x in mat.columns]
    lat_str = mat.to_latex(index = False,
                        float_format="%.4f",
                        formatters = {mat.columns[0]:lambda x: "\multicolumn{1}{m{2.5cm}}{\centering "+x+"}"})
    lat_str = re.sub(r'(toprule|midrule|bottomrule)', r'hline', lat_str)
    lat_str = re.sub(r'\\\\', r'\\\\[0.4cm]', lat_str)
    print(lat_str)


############################################################################################################
# Get elasticities for markets
el_2021 = results.compute_elasticities("prices", market_id = "NEW HAVEN_2021")
el_2022 = results.compute_elasticities("prices", market_id = "NEW HAVEN_2022")

# Get the column header of the top 3 EVs that are available in both 2021 and 2022
top_evs = estimation_data.loc[(estimation_data["electric"] == 1)&(estimation_data[2022]==1)&(estimation_data[2021]==1), ["veh_count","column header"]].groupby("column header").sum().sort_values("veh_count", ascending = False).head(3).index.tolist()
top_nevs = estimation_data.loc[(estimation_data["electric"] != 1)&(estimation_data[2022]==1)&(estimation_data[2021]==1), ["veh_count","column header"]].groupby("column header").sum().sort_values("veh_count", ascending = False).head(5).index.tolist()
models = top_evs + top_nevs

# Get indexes for these models in 2021 and 2022 markets
indexes_21 = estimation_data.loc[estimation_data["market_ids"] == "NEW HAVEN_2021"].reset_index(drop = True)["column header"].isin(models)
indexes_22 = estimation_data.loc[estimation_data["market_ids"] == "NEW HAVEN_2022"].reset_index(drop = True)["column header"].isin(models)

# Get elasticities for these models
el_2021 = pd.DataFrame(el_2021[indexes_21, :][:, indexes_21], columns = models, index = models)
el_2022 = pd.DataFrame(el_2022[indexes_22, :][:, indexes_22], columns = models, index = models)

# Function for latex tables:
def make_latex_table_nice(df):
    mat = df
    mat=mat.reset_index().rename(columns = {"index": "Model"})
    mat.columns = ["\multicolumn{1}{m{2cm}}{\centering "+str(x) + "}" for x in mat.columns]
    lat_str = mat.to_latex(index = False,
                           column_format = "l" + "c"*(len(mat.columns)-1),
                         float_format="%.4f",
                         formatters = {mat.columns[0]:lambda x: "\multicolumn{1}{m{2.5cm}}{\centering "+x+"}"})
    lat_str = re.sub(r'(toprule|midrule|bottomrule)', r'hline', lat_str)
    lat_str = re.sub(r'\\\\', r'\\\\[0.4cm]', lat_str)
    print(lat_str)

make_latex_table_nice(el_2021)
make_latex_table_nice(el_2022)