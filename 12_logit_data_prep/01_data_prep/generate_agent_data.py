# The original R code is under 00_reference_files/generated_agent_data.R

# Import libraries
import pandas as pd
import numpy as np
import pathlib
np.random.seed(1217)


# Warnings and display
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)

####################################################################################################
# Setup paths
str_cwd = pathlib.Path().resolve().parent
str_dir = str_cwd / "Documents" / "tobin_working_data"
ipums_dir = str_dir / "ipums_data" / "ipums_ct_all.csv"

####################################################################################################
# Choose number of agents we are running this for
n_agents = 2000

####################################################################################################
county_fips = {1:'FAIRFIELD',
               3:'HARTFORD',
               5:'LITCHFIELD',
               7:'MIDDLESEX',
               9:'NEW HAVEN',
               11:'NEW LONDON',
               13:'TOLLAND',
               15:'WINDHAM'}

####################################################################################################
# Read in IPUMS data and fix CPI income
dt_ipums = pd.read_csv(ipums_dir)

# Rename columns
dt_ipums = dt_ipums.rename(columns = {"stateicp":"state"})

# Fix incomes
# Drop invalid observations
dt_ipums = dt_ipums.loc[(dt_ipums["hhincome"]!='Non-monetary')]
dt_ipums = dt_ipums.loc[(dt_ipums["hhincome_cpiu_2010"]!='Non-monetary')]
dt_ipums["hhincome_cpiu_2010"] = dt_ipums["hhincome_cpiu_2010"].astype(int)
dt_ipums = dt_ipums.loc[(dt_ipums["hhincome"]<9999000)&(dt_ipums["hhincome"]>5000)]
dt_ipums = dt_ipums.loc[(dt_ipums["hhincome_cpiu_2010"]<9999000)&(dt_ipums["hhincome_cpiu_2010"]>5000)]

# Get CPI-adjusted numbers
dt_ipums = dt_ipums.rename(columns = {'hhincome':'hhincome_old'})
dt_ipums = dt_ipums.rename(columns = {'hhincome_cpiu_2010': 'hhincome'})

# Get income in thousands
dt_ipums['hhincome'] = dt_ipums['hhincome'].astype(int) / 1000
dt_ipums = dt_ipums[dt_ipums.hhincome > 1]

# Now categorize the incomes to align with micro_moments
def categorize_incomes(x):
    if x < 10:
        return np.nan # Assume these are errors
    elif x < 85:
        return 'low'
    elif x< 200:
        return 'medium'
    elif x<5000:
        return 'high'
    else:
        return np.nan

# Categorise the income categories
dt_ipums.loc[:, "hhincome_categories"] = dt_ipums.loc[:, "hhincome"].apply(categorize_incomes)

# Get county names
dt_ipums['county_name'] = dt_ipums['countyfip'].map(county_fips)
dt_ipums = dt_ipums.dropna(subset=['county_name']) # This also drops 2022

# We do not need to do this since the data extract already adjusts for CPI
if False:
    # CPI Calculations - we base to 2010
    # Read in CPI data and convert long format
    dt_CPI = pd.read_csv(f"{str_loc}raw/CPI.csv")
    dt_CPI_l = dt_CPI.melt(id_vars=['Year'], var_name='month', value_name='cpi')
    dt_CPI_l.columns = ['year', 'month', 'cpi']

    # Aggregate CPI data by year and calculate multiplier for 2021 USD adjustment
    cpi_agg = dt_CPI_l.groupby('year').cpi.mean().reset_index()
    cpi_2021 = cpi_agg[cpi_agg.year == 2021].cpi.iloc[0]
    cpi_agg['mult_2021'] = cpi_2021 / cpi_agg.cpi

    # Merge CPI data back to IPUMS and adjust incomes
    dt_ipums = dt_ipums.merge(cpi_agg[['year', 'mult_2021']], on='year', how='left')
    dt_ipums['hhincome'] = dt_ipums.hhincome * dt_ipums.mult_2021
    dt_ipums.drop(columns=['mult_2021'], inplace=True)

# Handling single-family home status, urban status, and college graduation
dt_ipums['single_fam'] = np.where(dt_ipums.unitsstr.isin(['1-family house, detached', '1-family house, attached']), 1, 0)
dt_ipums['urban'] = np.where(dt_ipums.metro.str.contains("In metropolitan area"), 1, 0)
dt_ipums['college_grad'] = np.where(dt_ipums.educ.isin(['5+ years of college', '4 years of college']), 1, 0)

# Ensure only one person per household is kept
dt_ipums = dt_ipums[dt_ipums.pernum == 1]

# Random sampling of observations per year, considering household weights
def generate_sample(df, N_obs, alt_2022 = False):
    unique_years_states = df[['year', 'county_name']].drop_duplicates()
    samples = []
    
    for _, row in unique_years_states.iterrows():
        year = row['year']
        county = row['county_name']

        # Extract a sub-sample of the DataFrame, for the given year and county
        df_sub = df[(df.county_name == county) & (df.year == year)]
        df_sub = df_sub.drop(columns=['county_name', 'year'])
        
        # Fake the year 2022 - that is not in the data
        if alt_2022:
            year = 2022
        
        # Sample with replacement, using household weights
        prob = df_sub.hhwt / df_sub.hhwt.sum()
        sample = df_sub.sample(n=N_obs, replace=True, weights=prob)
        sample['model_year'] = year
        sample['market_ids'] = f"{county}_{year}"
        sample["weights"] = 1/N_obs
        sample["year"] = year
        samples.append(sample)

    # Concatenate all samples
    result = pd.concat(samples, ignore_index=True)

    result = result.rename(columns = {"hhincome" : "income", "hhincome_categories":"income_category"})

    # Turn income categories into dummies
    income_cat_dummies = pd.get_dummies(result[["income_category"]], prefix = "hh_income", dtype = int)
    result = pd.concat([result, income_cat_dummies], axis = 1)

    return result[['state', 'countyfip', "year", 'model_year', 'market_ids', 'income']+income_cat_dummies.columns.tolist()+['single_fam', 'urban', 'college_grad', 'weights']]

# We generate for all years, and then for 2021 (that we use to replace 2022 for which we have no data)
print(f"Running for {(n_agents)}")
samples = generate_sample(dt_ipums, n_agents)
samples_2022 = generate_sample(dt_ipums.loc[dt_ipums["year"]==2021], n_agents, alt_2022=True)
samples = pd.concat([samples, samples_2022], ignore_index=True)
samples.to_csv(f"{str_dir}/ipums_data/agent_data_processed_2000.csv", index=False)


