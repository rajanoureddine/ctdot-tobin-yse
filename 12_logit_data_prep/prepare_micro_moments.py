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
import logging

## Set the working directory and get date
input_dir = Path().resolve().parent / "Documents" / "tobin_working_data" / "InMoment_inputs" / "11-22-23 - 2018-2022 Data"
output_dir = Path().resolve().parent / "Documents" / "tobin_working_data" / "micro_moment_data"
date = datetime.now().strftime("%Y%m%d")

## Create a logging file in output directory
logging.basicConfig(filename= output_dir / f"inmoment_cleaned_for_RLP_{date}.log")
logging.log(msg = "Re-run data creation, this time keeping the income data. No other changes to previous version.", level = logging.DEBUG)

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
    'edited purchase/lease price us edited purchase price buyers only canada',
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
    'would you consider these engine types for your next vehicle hybrid engine',
    'annual household income'
]]

## Rename columns
dt_sc_keep.columns = ['study_year','purchase_year','purchase_month','model_year',
                      'make','model','trim', 'edited_purchase_price', 'purchase_fuel', 
                      'sc','sc_make','sc_model','sc_trim','sc_year','sc_fuel',
                      'state',
                      'consider_ev','consider_phev','consider_hybrid',
                      'annual_household_income']

## Get only New England entries
dt_sc_keep = dt_sc_keep[dt_sc_keep['state'].isin(['CT','ME','MA','NH','RI','VT'])]

## Now clean up the income column
income_map = {'$20,000 or less':'low', '$25,000 or less':'low',
              '$25,001-$35,000':'low',
              '$25,001-$35,000':'low',
              '$35,001-$45,000':'low',
              '$20,001-$25,000':'low','$25,001-$30,000':'low','$30,001-$35,000':'low','$35,001-$40,000':'low','$40,001-$45,000':'low',
              '$45,001-$50,000':'low', '$45,001-$55,000':'low', '$50,001-$55,000':'low','$55,001-$65,000':'low','$65,001-$75,000':'low','$75,001-$85,000':'low',
              '$85,001-$100,000':'medium', '$100,001-$125,000':'medium','$125,001-$150,000':'medium','$150,001-$200,000':'medium',
              '$200,001-$300,000':'high','$300,001-$400,000':'high','$400,001-$500,000':'high','Over $500,000':'high',
              'Prefer not to answer':'none'}

def map_income(x):
    try:
        return(income_map[x])
    except:
        return(np.nan)

dt_sc_keep.loc[:, "household_income_categories"] = dt_sc_keep.loc[:,"annual_household_income"].apply(map_income)
dt_sc_keep = pd.concat([dt_sc_keep, pd.get_dummies(dt_sc_keep[["household_income_categories"]], columns = ["household_income_categories"], prefix = "hh_income", dtype = int)], axis = 1)

## Save the data
dt_sc_keep.to_csv(output_dir / f"inmoment_cleaned_for_RLP_{date}.csv", index = False)

## Create income micro-moments
inc_moments_raw = dt_sc_keep.copy()
inc_moments_raw = inc_moments_raw[inc_moments_raw['edited_purchase_price'].notna()]
inc_moments = inc_moments_raw[["household_income_categories", "edited_purchase_price"]].groupby("household_income_categories").mean().drop("none").reset_index()
inc_moments = inc_moments.rename(columns = {"edited_purchase_price":"value"})
inc_moments["micro_moment"] = "E[Purchase Price | Income = " + inc_moments["household_income_categories"] + "]"
inc_moments["value"] = inc_moments["value"] / 1000
inc_moments = inc_moments[["micro_moment", "value"]]

## Create columns to mark whether first and choice fuels were broadly EV
dt_sc_keep['first_choice_broad_ev'] = dt_sc_keep['purchase_fuel'].isin(['Electric','Plug-in Hybrid'])
dt_sc_keep['sc_broad_ev'] = dt_sc_keep['sc_fuel'].isin(['Electric','Plug-in Hybrid'])

## Create a dataframe with first column "fc_broad_ev", second column "sc_broad_ev", third column "n", and fourth column "percentage"
fc_sc_broad_ev = dt_sc_keep.groupby(['first_choice_broad_ev','sc_broad_ev']).size().reset_index(name = 'n')
fc_sc_broad_ev['percentage'] = fc_sc_broad_ev['n'] / fc_sc_broad_ev['n'].sum()

# Print conditional probability of second choice being broadly EV given first choice was broadly EV
cond_prob_sc_broad_ev_fc_broad_ev = fc_sc_broad_ev.loc[(fc_sc_broad_ev['first_choice_broad_ev'] == True) & (fc_sc_broad_ev["sc_broad_ev"]==True), "n"].sum() / fc_sc_broad_ev.loc[(fc_sc_broad_ev['first_choice_broad_ev'] == True), "n"].sum()

# Get probability first choice is EV for each income category
evs_incomes = dt_sc_keep.groupby(['first_choice_broad_ev','household_income_categories']).size().reset_index(name = 'n')
cond_expectations = {}
for inc_cat in ["high", "medium", "low"]:
    exp_ev_cond_inc_cat = evs_incomes.loc[(evs_incomes.first_choice_broad_ev==True) & (evs_incomes.household_income_categories==inc_cat), "n"].sum() / evs_incomes.loc[evs_incomes.household_income_categories==inc_cat, "n"].sum()
    cond_expectations[f"P(FC = Broad EV | Income = {inc_cat})"] = exp_ev_cond_inc_cat

# Create income micro_moment_df
inc_micro_moments = pd.DataFrame(cond_expectations, index = ["value"]).T.reset_index().rename(columns = {"index":"micro_moment"})

# Create dataframe containing this information, with two columns: micro_moment, and value
sc_micro_moments = pd.DataFrame([
    {"micro_moment": "P(SC = Broad EV | FC = Broadly EV)", "value": cond_prob_sc_broad_ev_fc_broad_ev}
])

all_micro_moments = pd.concat([sc_micro_moments, inc_micro_moments, inc_moments], axis = 0)

# Save in output directory
all_micro_moments.to_csv(output_dir / f"micro_moments_{date}.csv", index = False)

