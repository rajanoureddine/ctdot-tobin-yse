# ctdot-tobin-yse
Last date modified: 08/01/2024

# Description
This directory contains code used in the Tobin Center's / Ken Gillingham's project on EV Charging Infrastructure. 

# Data inputs


# Code

## logit_data_prep
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

### rlp_prep_energy_prices.py
This file is used to clean and process raw energy price data, that is then used in calculating the operating cost (variable: `dollar_per_mile`) of individual vehicles used in the BLP estimation.

**Inputs**

* Gas and diesel prices are sourced from the [EIA](https://www.eia.gov/petroleum/gasdiesel/). We first download an XLS file and then extract only the first sheet (this takes place outside of Python)
* Electricity prices are from [EIA 861](https://www.eia.gov/electricity/data/state/). We keep the series "Residential, Total Electricity Industry"

**Outputs**

We produce three outputs:
* `monthly_gas_prices_processed.csv`
* `monthly_diesel_prices_processed.csv`
* `yearly_electricity_prices_processed.csv`

### rlp_finalize_data.py
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

### rlp_group_data.py
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



