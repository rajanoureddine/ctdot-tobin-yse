####################################################################################################
# Import libraries
import pathlib
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt

# show all columns
pd.set_option('display.max_columns', None)

####################################################################################################
# Setup paths
str_cwd = pathlib.Path().resolve().parent.parent
str_dir = str_cwd / "Documents" / "tobin_working_data"
str_data = str_dir / "other_data" / "census_households"
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
        df = df.loc[:, ["County", "Total:"]]
        df["County"] = df["County"].str.replace(" County, Connecticut", "").str.upper()
        df["Year"] = year
        df["Total:"] = df["Total:"].str.replace(",", "").astype(int)
        output = pd.concat([output, df])

# Create and save years data
years = output.groupby("Year")["Total:"].sum().reset_index().rename(columns = {"Total:": "tot_HH", "Year":"model_year"})
years.to_csv(str_output / "hhs_by_year.csv", index = False)

# Create and save years counties data
years_counties = output.rename(columns = {"Total:": "tot_HH", "Year":"model_year", "County":"county_name"})
years_counties.to_csv(str_output / "hhs_by_year_counties.csv", index = False)
