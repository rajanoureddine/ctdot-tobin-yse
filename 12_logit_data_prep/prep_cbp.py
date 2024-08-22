import pandas as pd
import numpy as np
from pathlib import Path
import os
from datetime import datetime

# Set paths
data_dir = Path().resolve().parent / "Documents" / "tobin_working_data" / "historical_data" / "Early County Business Pattern Files 1959 Imputed"
data_subdir = data_dir / "DS0025" 

# Read in dataframe and get CT entries
dat = pd.read_csv(data_subdir / "38834-0025-Data.tsv", sep = "\t")
dat.columns = dat.columns.str.lower()
dat = dat[dat["fipstate"] == 9] # Get CT
dat = dat[~dat["sic"].str.contains("-")] # Drop totals

# Read in the codebook
cb = pd.read_csv(data_dir / "codebooks" / "match_table_2.csv")
cb.columns = cb.columns.str.lower()
cb = cb[["cats", "original", "title"]]
#fill in the code2 column with four zeros
cb["cats"] = cb["cats"].apply(lambda x: str(x).zfill(4))
#cb = cb.drop_duplicates("code2")

# Identify the rows with all capitals in "title"
category_titles = cb["title"].str.isupper()

# Merge
dat_merged = dat.merge(cb, left_on = "sic", right_on = "cats", how = "left")
assert dat_merged.shape[0] == dat.shape[0]
print(dat_merged[dat_merged["title"].notna()].head())



