####################################################################################################
# Import libraries
import pathlib
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt

# show all columns
pd.set_option('display.max_columns', None)

# Set it to do population or households
type = "population" 

# For population, total col is "Total"
# For households, total col is "Total:"
if type == "population":
    total_col = "Total"
else:
    total_col = "Total:"

####################################################################################################
# Setup paths
str_cwd = pathlib.Path().resolve().parent.parent
str_dir = str_cwd / "Documents" / "tobin_working_data"
if type == "population":
    str_data = str_dir / "other_data"/  "census_population"
    str_output = str_dir / "other_data"
else:
    str_data = str_dir / "other_data"/ "census_households"
    str_output = str_dir / "other_data"

output = pd.DataFrame()

for file in str_data.iterdir():
    name = file.name
    if not name == ".DS_Store":
        year = name[7:11]
        df = pd.read_excel(file, sheet_name = "Data").T
        df = df.reset_index()
        df.columns = df.iloc[0]
        df = df.rename(columns = {"Unnamed: 0": "County"})
        df = df.loc[df["Label"]=="Estimate"]
        df = df.loc[:, ["County", total_col]]
        df["County"] = df["County"].str.replace(" County, Connecticut", "").str.upper()
        df["Year"] = year
        df[total_col] = df[total_col].str.replace(",", "").astype(int)
        output = pd.concat([output, df])

# Create and save years data
if type == "population":
    # We increment years by 1 as these are lagged
    output["Year"] = (output["Year"].astype(int) + 1).astype(str)

    years = output.groupby("Year")[total_col].sum().reset_index().rename(columns = {total_col: "tot_population", "Year":"model_year"})
    years.to_csv(str_output / "population_by_year.csv", index = False)
    years_counties = output.rename(columns = {total_col: "tot_population", "Year":"model_year", "County":"county_name"})
    years_counties.to_csv(str_output / "population_by_year_counties.csv", index = False)

    # Get density
    areas = pd.read_csv(str_output / "ct_counties_area.csv")
    areas.county_name = areas.county_name.str.upper()
    years_counties = years_counties.merge(areas, on = "county_name")

    # Calculate density
    years_counties["pop_density"] = years_counties["tot_population"] / years_counties["area"]

    # Get market ids
    years_counties["market_ids"] = years_counties["county_name"]+ "_" + years_counties["model_year"]

    # Save the data
    years_counties.to_csv(str_output / "population_by_year_counties.csv", index = False)
else:
    years = output.groupby("Year")[total_col].sum().reset_index().rename(columns = {total_col: "tot_HH", "Year":"model_year"})
    years.to_csv(str_output / "hhs_by_year.csv", index = False)
    # Create and save years counties data
    years_counties = output.rename(columns = {total_col: "tot_HH", "Year":"model_year", "County":"county_name"})
    years_counties.to_csv(str_output / "hhs_by_year_counties.csv", index = False)