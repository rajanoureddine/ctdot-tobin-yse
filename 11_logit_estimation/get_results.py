import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path
import pickle
import re

##### Get data dir
data_dir = Path().resolve().parent / "Documents" / "tobin_working_data" / "outputs"

#### Load object
with open(data_dir / "outputs_county_model_year_0730-1327" / "outputs_rand_coeffs_county_model_year_0730-1327_agent.pkl", "rb") as f:
    results1 = pickle.load(f)


df_rand_coeffs = pd.DataFrame({'param':results1.beta_labels,
                                'value':results1.beta.flatten(),
                                'se': results1.beta_se.flatten()})
df_sigma = pd.DataFrame({'param': ['sigma_' + s for s in results1.sigma_labels],
                                'value': np.diagonal(results1.sigma).flatten(),
                                'se': np.diagonal(results1.sigma_se).flatten()})
df_pi = pd.DataFrame({'param': ['pi_' +a + "_" + s for a in results1.sigma_labels for s in results1.pi_labels],
                                'value': results1.pi.flatten(),
                                'se': results1.pi_se.flatten()})

df_results = pd.concat([df_rand_coeffs,df_sigma, df_pi],ignore_index=True)

lat = df_results.to_latex(index = False, float_format="%.4f")

# Function for latex tables:
def make_latex_table_nice(df):
    mat = df
    lat_str = mat.to_latex(index = False,
                           column_format = "l" + "c"*(len(mat.columns)-1),
                         float_format="%.4f")
    lat_str = re.sub(r'(toprule|midrule|bottomrule)', r'hline', lat_str)
    lat_str = re.sub(r'_', ' ', lat_str)
    print(lat_str)

make_latex_table_nice(df_results)

print("a")
