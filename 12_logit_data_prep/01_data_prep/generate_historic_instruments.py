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

# Import data that has been geocoded on Yale MyApps (GIS), drop unmatched
geocoded_data = pd.read_csv(data_path / "db_data_1969_geocoded.csv")
geocoded_data = geocoded_data[geocoded_data.Status != "U"]

# Match to industry codes and decide which establishments to include or not
sic_codes = pd.read_csv(data_path / "sic_codes.csv")



