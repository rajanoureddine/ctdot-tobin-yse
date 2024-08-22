import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path
import pickle
import re


## Define function
def get_df(results_object):
    df_rand_coeffs = pd.DataFrame({'param':results_object.beta_labels,
                                    'value':results_object.beta.flatten(),
                                    'se': results_object.beta_se.flatten()})
    df_sigma = pd.DataFrame({'param': ['sigma_' + s for s in results_object.sigma_labels],
                                    'value': np.diagonal(results_object.sigma).flatten(),
                                    'se': np.diagonal(results_object.sigma_se).flatten()})
    df_pi = pd.DataFrame({'param': ['pi_' +a + "_" + s for a in results_object.sigma_labels for s in results_object.pi_labels],
                                    'value': results_object.pi.flatten(),
                                    'se': results_object.pi_se.flatten()})

    df_results = pd.concat([df_rand_coeffs,df_sigma, df_pi],ignore_index=True)

    return df_results

# Function to simplify and print
def print_simple(df):
    print(df.loc[~df['param'].str.contains("'"), :])

# Function for latex tables:
def make_latex_table_nice(df):
    mat = df
    lat_str = mat.to_latex(index = False,
                           column_format = "l" + "c"*(len(mat.columns)-1),
                         float_format="%.4f")
    lat_str = re.sub(r'(toprule|midrule|bottomrule)', r'hline', lat_str)
    lat_str = re.sub(r'_', ' ', lat_str)
    print(lat_str)

if __name__ == "__main__":
    ##### Get data dir
    data_dir = Path().resolve().parent / "Documents" / "tobin_working_data" / "outputs"

    #### Load object
    with open(data_dir / "outputs_county_model_year_0730-1327" / "outputs_rand_coeffs_county_model_year_0730-1327_agent.pkl", "rb") as f:
        results1 = pickle.load(f)

    df_results = get_df(results1)
    print_simple(df_results)
    make_latex_table_nice(df_results)
