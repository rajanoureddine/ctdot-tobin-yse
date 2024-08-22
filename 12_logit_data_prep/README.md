# logit_data_prep
Files in this directory are used to clean and prepare data for BLP model estimation. The folder contains the following main files.
* Prepare RL Polk vehicle sales data and vehicle characteristics
    * `rlp_prep_eergy_prices.py`
    * `rlp_finalize_data.py`
    * `rlp_group_data.py`
* Prepare other data involved in estimation
    * `prepare_micro_moments.py`
    * `prepare_hh_data.py`
    * `get_firm_ids.py`
* Prepare data for estimating network effects, including instruments
    * `prepare_charging_data.py`
    * `generate_instruments.py`:
    * `generate_historic_retail_density_db1969.py`
    * `prep_cbp.py`: An incomplete being developed to process data from the 1969 Early County Business Pattern Files
    * `test_instruments.py`: Used to run simple First-stage OLS testing on generated instruments

## rlp_prep_energy_prices.py
This file is used to clean and process raw energy price data, that is then used in calculating the operating cost (variable: `dollar_per_mile`) of individual vehicles used in the BLP estimation.

**Inputs**

* Gas and diesel prices are sourced from the [EIA](https://www.eia.gov/petroleum/gasdiesel/). We first download an XLS file and then extract only the first sheet (this takes place outside of Python)
* Electricity prices are from [EIA 861](https://www.eia.gov/electricity/data/state/). We keep the series "Residential, Total Electricity Industry"

**Outputs**

We produce three outputs:
* `monthly_gas_prices_processed.csv`
* `monthly_diesel_prices_processed.csv`
* `yearly_electricity_prices_processed.csv`

## rlp_finalize_data.py
This file is used to:

* Calculate the variable `log_hp_wt`
* Calculate `dollar_per_mile` from raw mpg data in the input file, along with energy price data

**Inputs**

The file takes as input:
* The file `ct_decoded_full_attributes.csv`. This file is produced by the code `merge_SP_policy_data.R`. The output csv file was provided directly by Stephanie Weber at the University of Boulder Colorado. We have not converted `merge_SP_policy_data.R` to Python code just yet: we treat Stephanie's file as the raw input file. 
* The files `monthly_gas_prices_processed.csv`, `monthly_diesel_prices_processed.csv`, and `yearly_electricity_prices_processed.csv`. These files are in turn produced in `rlp_prep_energy_prices.py`. 

**Methodology notes**

The most complex part of this file is calculating a `dollar_per_mile` value for PHEVs. To do this, we use a `phev_electric_share` that is sourced from [the AFDC at the DOE](https://afdc.energy.gov/vehicles/electric_emissions_sources.html). We then calculate a separate `dollar_per_mile_electric` and `dollar_per_mile_gas` for each hybrid vehicle, and weight these two based on the electric share. 

**Outputs**

`rlp_with_dollar_per_mile.csv` 

## rlp_group_data.py
This is the most important code in the directory and contains most of the important methodological choices. 

**Inputs**

The main input is `rlp_with_dollar_per_mile.csv`. 

**Outputs**

The main output is `rlp_with_dollar_per_mile_replaced_myear_county_{date_time}_{lease}_zms.csv`. 

For example: `rlp_with_dollar_per_mile_replaced_myear_county_20240523_154006_no_lease_zms.csv` (the `no_lease` part indicates that we have dropped leased vehicles from the RLP data). This is the latest data being used in the estimation as at 08/01/2024.

**Choice of market definition**
A market is a $(\text{model year} \times \text{county})$ combination. In a previous version, we attempted aggregating to the zip code level, but ended up with too many zero market shares. There are ~250 zip codes in CT, and ~5 model years. This gives us ~1250 markets, with ~200 products each, for a total of ~250,000 product-market combinations. This is too many, and leads to a large number of zero market shares.

**Structure of the code**
In the first half of the code, we define the main functions to be used, function by function. The latter half of the code runs the functions in order, and produces intermediate and final outputs. 

**Methodology Notes**

Firstly, we identify the most common trim for each make and model.
* Note: The decision to do this followed discussions with Stephanie Weber at the University of Boulder Colorado, who has followed a similar methodology in her project. 
* This grouping seeks to answer the question: *how do we define a product?* The RLP data in `rlp_with_dollar_per_mile` contains a VIN code, a make, a model, a trim, and sometimes even a style. Furthermore, the same trim might be associated with different battery ranges for hybrid and electric vehicles.  
* Running `df_rlp[["make", "model", "fuel_type", "model_year", "trim", "style"]].drop_duplicates()` returns a DataFrame 8900 rows long. If we run the same without model_year, we get a dataframe 3616 rows long.  That is, there are 3600 unique combinations of make, model, fuel_type, trim, and style. 
* To address this potentially large number of unique products, we choose only the most common trim for each make, model combination.

1. In `get_most_common_trim` we choose the most common trim for each (make, model, fuel, range_elec). Note - this most common trim may not be available in all model years. For electric vehicles, we do not choose a most popular trim - we keep all the trims. This is because there are fewer electric vehicles so we want to maintain variation. 
2. In `get_most_common_trim_features` each of the most common trims (i.e., unique products), we calculate their features. 
    * Note: Each unique product is an *aggregation* of even more detailed products (in the raw file, products further vary by style, and even within styles). Therefore, we need to use aggregation functions to acquire features for the product. 
    * In some cases, the most common trim for a product is not available in a given year. For example, suppose that for the Ford F-150, we have the following sales:
        * "Trim A" sold 1000 units across all years
        * "Trim B" sold 1500 units across all years, but was not on the market for one particular year. 
    * In this case, we wish to associate all sales of the Ford F-150 with "Trim B". However, "Trim B" was not available in one year. How do we calculate features for that product? In that case, we find the year in which Trim B sold most units, and extract features from there. For example, suppose "Trim B" was not available in 2019, but sold 500 units in 2023. Then the features for "Ford F-150 Trim B 2019" will be drawn from "Ford F-150 Trim B 2023." Moreover, features for "Ford F-150 Trim A 2019" will be drawn from "Ford F-150 Trim B 2023." 
    * Aggregation functions are defined under `agg_funs`. 
    * The output file contains the following columns: model_year, make, model, trim, veh_count, msrp, dollar_per_mile, log_hp_wt, wheelbase, curb_weight, doors, drive_type, body_type, fuel, range_elec, fed_credit
3. In `replace_with_most_common_trim` we go back through the original file, and for each row, we replace the features with those produced in `get_most_common_trim_features`. The output of this function is a DataFrame that is of the *same length as the input* and has the *same number of sales* as the input, but in which each the data in each row has been changed:
    * The make, model, trim will be associated with the most common trim for that make and model. If the trim is already the most popular trim, it won't be changed.
    * The features for that make, model, trim, model_year will be those of the most common trim + model year (according to methodology above).
4. In `aggregate_to_market` we take the raw dataframe (with trims and features replaced), and aggregate them for estimation. Note that this function produces two outputs:
    * An output where sales are aggregated to the `model_year` only. 
    * An output where sales are aggregated to $(\text{Model year} \times \text{County})$ combinations. 
5. In `rationalize_markets` we drop uncommon products and add zero market shares where needed
    * We first drop vehicles with a low number of sales for any model year. We aggregate *across geographies* (for example, take the sum for all of CT), and drop vehicles below a threshold.
    * However, if the vehicle crosses the threshold, we *assume it was sold in all geographies for that model year*. For example, if we set the threshold as 20, we assume that any vehicle that sold more than 20 units in that model year *must have been available in all counties*. Consequently, if it is not observed for a given county, we create a new observation with zero sales. 
    * The result of this is a dataframe in which there are different numbers of products in each model year, but the number of products is the same across counties within the same model year. (e.g. number of products for New Haven 2019 = number of products for Stamford 2019)

## get_firm_ids.py
A script that does almost nothing - we need firm IDs to run the estimatiion, but didn't have access to the raw file. This script accesses an output CSV and uses it to pull out a mapping of firm IDs.

## prepare_micro_moments.py
This script is used to produce micro-moments for the main estimation. For an explanation of micro-moments, see this [simplified explanation](https://pyblp.readthedocs.io/en/stable/background.html#micro-moments), or [this paper](https://chrisconlon.github.io/site/micro_pyblp.pdf). The micro-moments produced here correspond to $\bar{v}_p$ in the longer paper - i.e., they are conditional averages extracted from survey data. 

We use survey data from InMoment to produce these micro-moments. The data was provided by Stephanie Weber / Vaishnavi Sinha (Vaishnavi has since left Yale to do her PhD), and is located under `tobin_working_data/InMoment_inputs` (the unaltered data provided by Stephanie is here). This data is unique in that it provides us not only with a *first choice* vehicle for each respondent, but also a *second choice* vehicle. 

We create three kinds of micro-moment:
* Probability that second choice of vehicle is an electric vehicle, conditional on the first choice of vehicle being an electric vehicle.
* Probability that the first choice of vehicle is an electric vehicle, conditional on income being classed as low, medium, or high.
* Expected purchase price of vehicle conditional on income being low, medium, or high (Note: these values may not be exactly right, since the purchase price of the vehicle may not correspond exactly to the MSRP that we use in the demand estimation)

We output these values (seven of them: second choice, and two by three income moments) into a CSV. During the BLP estimation, PyBLP uses these to improve the parameter estimates of the model by trying to bring simulated micro-moments into agreement with the observed micro-moments, using GMM. 

## generate_agent_data.py
The BLP estimation requires a sample of agent data. See [this explanation](https://pyblp.readthedocs.io/en/stable/background.html#demand). In particular, the agent-specific portion of utility is given as $\mu = X_2(\Sigma v' + \Pi d')$, where $X_2$ is a vector of non-linear product characteristics that we wish to interact with random heterogeneity (stored in $v'$) or known agent characteristics (stored in $d'$). 

To produce agent data, we use the IPUMs dataset, that contains a subsample of census micro-data. The detail of the IPUMs data extract is stored as a `.csv` file under `ipums_data`. 

We specify a number of agents to generate, and run the generation process. Note that it may be advisable to run the estimation with a smaller number of agents (e.g., 1000 or 2000) agents initially, as running with a larger number of agents can mean the estimation takes *significantly longer* and may not converge. 

## prepare_charging_data.py
This file is used to produce a density of chargers per county and model year. Ideally, we use our demand model to estimate the effect of network density on EV adoption, as in [(Li et al. 2017)](https://www.journals.uchicago.edu/doi/full/10.1086/689702?mobileUi=0). This is an earlier paper estimating the network effects present in EV adoption. Note that Professor Gillingham is well acquainted with the paper and has expressed some reservations about the identification strategy used in it, but I am unaware at this point, of any other way to address the endogeneity of charging density in EV adoption (i.e., EV adoption and charging density are co-determined / simultaneous). 

In this file, we prepare charging density data to use in our estimation. 

**Input data**: We use data that was produced by the other project as our inputs. While the AFDC at NREL used to produce monthly estimates of the number of chargers by county, it appears they no longer do this, so we don't re-download and re-produce the data. For 2022, we extrapolate using a linear trend line. 

This data was produced by the code that is located at `12_logit_data_prep/00_reference_files/03_charging_moments_code.do`. The `.do` file is also available on the AutoMaker Strategies dropbox under `AutoMakerStrategiesCode/charging_data`. We don't touch that file - but use its outputs as our inputs. They are located at `AutoMakerStrategies/data/ChargingMoments/outputs` on the other project's dropbox, and I have stored them at `tobin_working_data/charging_stations`. 

## generate_instruments.py
This file is used to generate Bartik instruments in order to estimate the BLP model, with network effects (i.e., when using charging density as an Independent Variable on the right hand side). Note that there are theoretical issues with these instruments. Nonetheless, we generate them in order to test them.

Our instruments are a multiplication of retail density in a given county *at a fixed point in time* (we use 2020) (i.e., an instrument that is not time-variant), multiplied by a time-variant component: the density of all chargers in all other counties except the one in question. This is similar to the instrument use by Shanjun Li and coauthers in their 2017 paper.

Note that this file can generate either `historic` or `current` instruments. The thing that changes is the point-in-time retail density for each county. When we use `current`, the point-in-time observation is retail density in 2020, taken from Advan data. When we use `historic`, the point-in-time observation is retail density in 1969, that is created using historic data (and generated in the file `generate_historic_retail_density_db1969.py`). 

## prep_hh_data.py
Used to generate the number of households or total population, by year and county, for the BLP estimation. This is needed because the BLP estimation uses a *market share* where the denominator is the total number of households for that market. 

