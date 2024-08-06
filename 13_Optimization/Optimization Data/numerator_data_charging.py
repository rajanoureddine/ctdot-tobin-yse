# In this file, we seek to understand if the numerator data contains any information on EV charging
# or Gas purchases

# Import all required libraries
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# Show all columns
pd.set_option('display.max_columns', None)

# Set the working directory which is here: /Users/rrn22/Dropbox/DOT_Tobin_Collaboration/data/Numerator Data/Unzipped
input_dir = Path("/Users/rrn22/Dropbox/DOT_Tobin_Collaboration/data/Numerator Data/Unzipped")
# input_dir = input_dir / "standard_nmr_feed_item_table_0_0_0.csv"
# input_file = input_dir / "standard_nmr_feed_item_table_0_0_0.csv"
agent_data_dir = Path("/Users/rrn22/Dropbox/DOT_Tobin_Collaboration/data/Numerator Data/Numerator Data Test Unzipped")
output_dir = Path("/Users/rrn22/Documents/tobin_working_data/processed_numerator_data")
categories_output_fpath = output_dir / "categories_cleaned_all.csv"
############################################################################################################

if not categories_output_fpath.exists():
    files = []

    for file in input_dir.iterdir():
        fname = file.name
        if "item_table" in fname:
            newdir = input_dir / fname
            for f in newdir.iterdir():
                df = pd.read_csv(f, chunksize = 1000, on_bad_lines = 'skip', sep = "|")
                files = files + [df]

    categories = {"majorcat_id":[], "major_category_description": [],
                    "category_id": [], "category_description": []}
    categories_list = categories.keys()

    categories_df = []

    for df in tqdm(files):
        for chunk in tqdm(df):
            chunk.columns = chunk.columns.str.lower()
            chunk = chunk[categories_list].drop_duplicates()
            categories_df = categories_df + [chunk]

    categories_df = pd.concat(categories_df)
    categories_df = categories_df.drop_duplicates().reset_index(drop = True)

    # Save the categories
    categories_df.to_csv(output_dir / "categories_cleaned_all.csv", index = False)

else:
    categories_df = pd.read_csv(categories_output_fpath)

if True:
    # Now search for categories that might be relevant
    search_terms = ["cars", "auto", "gas", "gasoline", "fuel", "charg"]

    # Search for the search terms
    search_results = []
    for index, row in tqdm(categories_df.iterrows()):
        for term in search_terms:
            if term in row["category_description"].lower() or term in row["major_category_description"].lower():
                search_results = search_results + [categories_df.loc[[index]]]

    search_results = pd.concat(search_results)
    search_results = search_results.drop_duplicates()

    # Save the search results
    search_results.to_csv(output_dir / "search_results_all.csv", index = False)
