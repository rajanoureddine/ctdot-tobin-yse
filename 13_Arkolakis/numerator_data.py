# Import all required libraries
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# Set the working directory which is here: /Users/rrn22/Dropbox/DOT_Tobin_Collaboration/data/Numerator Data/Unzipped
input_dir = Path("/Users/rrn22/Dropbox/DOT_Tobin_Collaboration/data/Numerator Data/Unzipped")
input_dir = input_dir / "standard_nmr_feed_fact_table_0_0_1.csv"
input_file = input_dir / "standard_nmr_feed_fact_table_0_0_1.csv"
agent_data_dir = Path("/Users/rrn22/Dropbox/DOT_Tobin_Collaboration/data/Numerator Data/Numerator Data Test Unzipped")
output_dir = Path("/Users/rrn22/Documents/tobin_working_data/processed_numerator_data")

############################################################################################################
def get_agent_zips():
    # Get all the agent data
    agent_data = []
    for subdir in agent_data_dir.iterdir():
        new_dir = agent_data_dir / subdir
        for file in subdir.iterdir():
            f = pd.read_csv(file, on_bad_lines = 'skip', sep = "|")
            agent_data = agent_data + [f]

    # Concatenate the agent data
    agent_data = pd.concat(agent_data)
    agent_zips = agent_data[["USER_ID", "POSTAL_CODE"]].drop_duplicates()

    # Return the agent zips
    return agent_zips


# If the agent data has already been extracted, load it, otherwise extract it
agent_out_file = agent_data_dir / "agent_zips.csv"

if agent_out_file.exists():
    agent_zips = pd.read_csv(agent_out_file)
else:
    agent_zips = get_agent_zips()
    agent_zips.to_csv(agent_data_dir / "agent_zips.csv", index = False)


############################################################################################################
# Step 2: Get the unique visits and count the number of stores visited by each user on each day
unique_identifiers = ["USER_ID", "TRANSACTION_DATE", "BANNER_ID", "STORE_POSTAL_CODE"]
data = pd.read_csv(input_file, chunksize = 10000, on_bad_lines = 'skip', sep = "|", usecols = unique_identifiers)

def get_unique_visits(df_stream, dropna = True):
    count = 0

    unique_visits = []

    for chunk in df_stream:
        count += len(chunk)
        unique_visits += [chunk[unique_identifiers].drop_duplicates()]
    
    unique_visits = pd.concat(unique_visits)
    unique_visits = unique_visits.drop_duplicates()

    if dropna:
        unique_visits.dropna(inplace = True)

    return unique_visits

def count_user_stores_visited(unique_visits):
    user_store_visits = unique_visits.groupby(["USER_ID", "TRANSACTION_DATE"]).size().reset_index(name = "count")

    return user_store_visits

unique_visits = get_unique_visits(data)
user_store_visits = count_user_stores_visited(unique_visits)

unique_visits["USER_DATE"] = unique_visits["USER_ID"].astype(str) + "_" + unique_visits["TRANSACTION_DATE"].astype(str)
user_store_visits["USER_DATE"] = user_store_visits["USER_ID"].astype(str) + "_" + user_store_visits["TRANSACTION_DATE"].astype(str)

############################################################################################################
# Step 3: Get the users who visited more than one store 

# Get the users who visited more than one store
multi_visit = user_store_visits[user_store_visits["count"] > 1]

# For these users, identify all the stores they visited
multi_visit_stores = unique_visits[unique_visits["USER_DATE"].isin(multi_visit["USER_DATE"])]

# For these users, merge in the agent zip codes
multi_visit_stores_zips = multi_visit_stores.merge(agent_zips, on = "USER_ID", how = "inner")
assert multi_visit_stores_zips.shape[0] == multi_visit_stores.shape[0]

# Now get only users from CT
ct_multi_visit_zips = multi_visit_stores_zips[multi_visit_stores_zips["POSTAL_CODE"].astype(str).str[0:2] == "06"]

# Now save them
ct_multi_visit_zips.to_csv(output_dir / "ct_multi_visit_zips.csv", index = False)

print("All done!")