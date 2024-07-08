"""This file is used to generate instruments for the estimation.
In particular, we seek to generate Bartik instruments that consider both
the density of retail establishments (at one point in time) and the sum
of chargers in other counties except the county in question, for any given year.
"""
from paths import data_dir
import pandas as pd
import numpy as np

###########################################################################################
# Create pd for instruments
instruments = pd.DataFrame()

# Import data on charging
charging_data = pd.read_csv(data_dir / "charging_stations_output" / "charging_stations_extended.csv")

# Get the sum of total, L2, DC in each market and year not for the current county
for county in charging_data.county.unique():
    for year in charging_data.year.unique():
        other_counties = charging_data.loc[(charging_data.county != county) & (charging_data.year == year)]
        assert(len(other_counties)==(charging_data.county.nunique()-1))
        other_counties = pd.DataFrame(other_counties.loc[:, ["county_area_sq_km", "total_stations", "total_stations_L2", "total_stations_DC"]].sum()).T
        other_counties.columns = ["other_counties_"+x for x in other_counties.columns.tolist()]
        other_counties["county"] = county
        other_counties["year"] = year
        other_counties["market_ids"] = county.upper()+"_"+str(year)
        instruments = pd.concat([instruments, other_counties])

# Now for each county, get the density of retail establishments... harder.
advan_data = pd.read_csv(data_dir / "advan_data_footfall_processed" / "footfall_CT_2020-01-07.csv")
categories_to_include = pd.read_csv(data_dir / "advan_data_footfall_processed" / "categories_to_include.csv")
counties = pd.read_csv(data_dir / "advan_data_footfall_processed" / "geocoded_placekeys.csv")

# Get counties and placekeys 
counties = counties[["PLACEKEY", "COUNTY"]]
advan_data = advan_data.merge(counties, on = "PLACEKEY", how = "left")

# Clean up advan data
advan_data.columns = advan_data.columns.str.lower()
cols = ["location_name", "brands", "store_id", "top_category", "sub_category", "naics_code", 
        "latitude", "longitude", "street_address", "city", "region", "county", "postal_code",
        "raw_visit_counts", "raw_visitor_counts", "median_dwell"]
advan_data = advan_data[cols]

# Merge with categories to include
advan_data = advan_data.merge(categories_to_include, left_on = "top_category", right_on = "Category", how = 'left')
advan_data = advan_data[advan_data["Include"]==1]

# Drop NAs, and observations below 10th, and above 90th percentile within each zip code
advan_data = advan_data.dropna(subset = ["raw_visitor_counts", "median_dwell"])

# Get the 0.95 and 0.05 percentile of raw visitors for each zip and keep only observatrions within them
top_qtile = advan_data[["county", "raw_visitor_counts"]].groupby("county").quantile(0.95).reset_index(drop=False).rename(columns = {"raw_visitor_counts":"pctile_95"})
bottom_qtile = advan_data[["county", "raw_visitor_counts"]].groupby("county").quantile(0.05).reset_index(drop=False).rename(columns = {"raw_visitor_counts":"pctile_05"})
advan_data = advan_data.merge(top_qtile, on = "county", how = "left")
advan_data = advan_data.merge(bottom_qtile, on = "county", how = "left")
advan_data = advan_data[(advan_data.raw_visitor_counts > advan_data.pctile_05) & (advan_data.raw_visitor_counts < advan_data.pctile_95)]

# Now get the count of establishments per county
retail_per_county = advan_data.value_counts("county").reset_index()
retail_per_county["county"] = retail_per_county["county"].str.lower()
retail_per_county = retail_per_county.merge(charging_data[["county", "county_area_sq_km"]].drop_duplicates(), on = "county", how = "left")
retail_per_county["retail_density_2020"] = retail_per_county["count"] /  retail_per_county["county_area_sq_km"] / 1000

# Now get the instrument
instruments = instruments.merge(retail_per_county[["county", "retail_density_2020"]], on = "county")
instruments["bartik_instrument_L2"] = instruments["other_counties_total_stations_L2"] * instruments["retail_density_2020"]
instruments["bartik_instrument_DC"] = instruments["other_counties_total_stations_DC"] * instruments["retail_density_2020"]
instruments["bartik_instrument_total"] = instruments["other_counties_total_stations"] * instruments["retail_density_2020"]

# Now save
instruments.to_csv(data_dir / "instruments" / "instruments.csv")