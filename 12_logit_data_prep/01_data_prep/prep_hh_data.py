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

output = pd.DataFrame()

for file in str_data.iterdir():
    name = file.name
    year = name[7:11]
    df = pd.read_excel(file, sheet_name = "Data").T
    df = df.reset_index()
    df.columns = df.iloc[0]
    df = df.rename(columns = {"Unnamed: 0": "County"})
    df = df.loc[df["Label"]=="Estimate"]
    df = df.loc[:, ["County", "Total:"]]
    df["Year"] = year
    df["Total:"] = df["Total:"].str.replace(",", "").astype(int)
    output = pd.concat([output, df])

print(output.head(30))

for year in output["Year"].unique():
    print(year)
    print(output.loc[output["Year"]==year, "Total:"].sum())