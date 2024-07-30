import pandas as pd
import numpy as np
from pathlib import Path

# Set paths
data_path = Path().resolve().parent / "Documents" / "tobin_working_data" / "historical_data" / "DB business data"

# Import raw data
# We don't need to do this again:
if False:
    db_data_raw = pd.read_csv(data_path / "DB 1969 Raw.csv", encoding='iso-8859-1')

    # Clean data
    db_data = db_data_raw[db_data_raw["STREETADDRESS"]!="<"]

    # Save
    db_data.to_csv(data_path / "db_data_1969_cleaned_forgeocode.csv")

    # We then upload this data to Yale MyApps, and geocode

# Define function to extract SIC code from geocoded data
def get_sic_code(x):
    try:
        x = str(x)
        if len(x) == 4:
            return int(x[0:2])
        elif len(x) <= 3:
            return int(x[0:1])
    except:
        return np.nan

# Import data that has been geocoded on Yale MyApps (GIS), drop unmatched
geocoded_data = pd.read_csv(data_path / "db_data_1969_geocoded_withcounties.csv")
geocoded_data = geocoded_data.loc[geocoded_data.Status != "U", ["USER_SIC1", "CNTY_NAME"]]
geocoded_data.rename(columns={"CNTY_NAME": "county"}, inplace=True)
geocoded_data.loc[:, "sic_code"] = geocoded_data.USER_SIC1.apply(lambda x: get_sic_code(x))

# Match to industry codes and decide which establishments to include or not
sic_codes = pd.read_csv(data_path / "sic_codes.csv")
sic_codes.rename(columns={"Code": "sic_code"}, inplace=True)
sic_codes = sic_codes.loc[sic_codes.sic_code.notna()]
sic_codes.loc[:, "sic_code"] = sic_codes.sic_code.astype(int)

# Merge two datasources together
merged = geocoded_data.merge(sic_codes, on = 'sic_code', how = 'left')
merged = merged.loc[merged.Include == 1, :]           

# Get counts of establishments by county
num_by_county = merged.value_counts("county").reset_index()
num_by_county.loc[:, "county"] = num_by_county.county.apply(lambda x: x.upper())

# Get the area_sq_km (extracted from instruments data)
counties_sq_km = {'FAIRFIELD':1.618651428,
                    'HARTFORD':1.903543897,
                    'LITCHFIELD':2.384116952,
                    'MIDDLESEX':0.956493754,
                    'NEW HAVEN':1.565688367,
                    'NEW LONDON':1.722716728,
                    'TOLLAND':1.062807467,
                    'WINDHAM':1.328478475}
num_by_county.loc[:, "county_area_sq_km"] = num_by_county.county.apply(lambda x: counties_sq_km[x] if x in counties_sq_km.keys() else np.nan)
num_by_county = num_by_county.dropna(subset = ["county_area_sq_km"])

num_by_county.loc[:, "retail_density_1969"] = num_by_county["count"] / num_by_county["county_area_sq_km"] / 1000

# Save csv
num_by_county.to_csv(data_path / "retail_density_1969.csv", index = False)





