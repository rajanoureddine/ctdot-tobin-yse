"""
This file is used to calculate second choice moments for the micro moments data.
We use InMoment data for this. This data was provided by Vaishnavi Sinha and is stored in the folder "11-22-23 - 2018-2022 Data".
The data has been cleaned and processed using the STATA do file under "Provided Code"

We do not re-process the data but take Vaishnavi's inputs as given and calculate the second choice moments.
"""

## Import necessary libraries
import numpy as np
import pandas as pd
from pathlib import Path
import os
from datetime import datetime

## Set the working directory and get date
input_dir = Path().resolve().parent / "Documents" / "tobin_working_data" / "InMoment_inputs" / "11-22-23 - 2018-2022 Data"
output_dir = Path().resolve().parent / "Documents" / "tobin_working_data" / "micro_moment_data"
date = datetime.now().strftime("%Y%m%d")

## Create DataFrame
inmoment_data = pd.DataFrame([])
for file in os.listdir(input_dir):
    if file.endswith(".csv") and not "UNFormatted" in file:
        inmoment_data = pd.concat([inmoment_data, pd.read_csv(input_dir / file)])

# Lower case columns and remove all punctuation
inmoment_data.columns = inmoment_data.columns.str.lower().str.replace("(", "").str.replace(")", "").str.replace("-","").str.replace("'","").str.replace(",","").str.replace("&","").str.replace("  ", " ")

# Keep only required columns
dt_sc_keep = inmoment_data[[
    'study year',
    'purchase year from admark',
    'purchase month from admark',
    'used to distinguish models with different model years within a study year',
    'purchase vehicle division',
    'purchase vehicle model',
    'purchase vehicle model series',
    'define diesel hybrid and gas engines by vin',
    'did you consider any other cars or trucks',
    'mmsc division',
    'mmsc model response',
    'mmsc model and series',
    'mmsc model year',
    'engine type of 1st considered model gas diesel or hybrid',
    'respondents state from admark',
    'would you consider these engine types for your next vehicle electric vehicle ev',
    'would you consider these engine types for your next vehicle plugin hybrid',
    'would you consider these engine types for your next vehicle hybrid engine'
]]

## Rename columns
dt_sc_keep.columns = ['study_year','purchase_year','purchase_month','model_year',
                      'make','model','trim','purchase_fuel',
                      'sc','sc_make','sc_model','sc_trim','sc_year','sc_fuel',
                      'state',
                      'consider_ev','consider_phev','consider_hybrid']

## Get only New England entries
dt_sc_keep = dt_sc_keep[dt_sc_keep['state'].isin(['CT','ME','MA','NH','RI','VT'])]

## Save the data
dt_sc_keep.to_csv(output_dir / f"inmoment_cleaned_for_RLP_{date}.csv", index = False)

## Create columns to mark whether first and choice fuels were broadly EV
dt_sc_keep['firt_choice_broad_ev'] = dt_sc_keep['purchase_fuel'].isin(['Electric','Plug-in Hybrid'])
dt_sc_keep['sc_broad_ev'] = dt_sc_keep['sc_fuel'].isin(['Electric','Plug-in Hybrid'])

## Create a dataframe with first column "fc_broad_ev", second column "sc_broad_ev", third column "n", and fourth column "percentage"
fc_sc_broad_ev = dt_sc_keep.groupby(['firt_choice_broad_ev','sc_broad_ev']).size().reset_index(name = 'n')
fc_sc_broad_ev['percentage'] = fc_sc_broad_ev['n'] / fc_sc_broad_ev['n'].sum()

# Print conditional probability of second choice being broadly EV given first choice was broadly EV
cond_prob_sc_broad_ev_fc_broad_ev = fc_sc_broad_ev.loc[(fc_sc_broad_ev['firt_choice_broad_ev'] == True) & (fc_sc_broad_ev["sc_broad_ev"]==True), "n"].sum() / fc_sc_broad_ev.loc[(fc_sc_broad_ev['firt_choice_broad_ev'] == True), "n"].sum()

# Create dataframe containing this information, with two columns: micro_moment, and value
micro_moments = pd.DataFrame([
    {"micro_moment": "P(SC = Broad EV | FC = Broadly EV)", "value": cond_prob_sc_broad_ev_fc_broad_ev}
])

# Save in output directory
micro_moments.to_csv(output_dir / f"micro_moments_{date}.csv", index = False)

