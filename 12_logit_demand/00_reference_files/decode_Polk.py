####################################################################################################
# Import libraries
import pathlib
import pandas as pd
import numpy as np 

####################################################################################################
# Set paths
str_cwd = pathlib.Path().resolve().parent.parent
str_dir = str_cwd / "Documents" / "tobin_working_data"
str_sp = str_dir / "rlpolk_data"
str_decoder = str_dir / "vin_decoder"


####################################################################################################
# Load data
str_data = "US_Yale_University_OP0001562727_NV_CT_VIN_Prefix_202212.txt"
df = pd.read_csv(str_sp / str_data, sep = "|", encoding = "ISO-8859-1")

####################################################################################################
# Clean up the data
df.columns = df.columns.str.lower()
df.loc[:,'make'] = df.loc[:,'make'].str.title()
df.loc[:,'model'] = df.loc[:,'model'].str.title().str.replace(' ', '-')
df.loc[:,"vin_pattern"] = df.loc[:,"vin_prefix"].astype(str).str[0:8] + df.loc[:,"vin_prefix"].astype(str).str[9]

# Extract unique combinations of make-model, model_year, and vin_pattern, and merge back in
df_unique = df.loc[:,["make", "model", "model_year", "vin_pattern"]].drop_duplicates()
df_unique.loc[:,"sp_id"] = np.arange(1, len(df_unique) + 1)
df = df.merge(df_unique, on = ["make", "model", "model_year", "vin_pattern"], how = "left")

####################################################################################################
# Read in the VIN decoder, excluding the column "TMP_WHEEL_DIA"
df_decode = pd.read_csv(str_decoder / "DataOne_IDP_yale_school_of_the_environment.csv")
df_decode.columns = df_decode.columns.str.lower()
df_decode.loc[:, "make"] = df_decode.loc[:, "make"].str.title()
df_decode.loc[:, "model"] = df_decode.loc[:, "model"].str.title().str.replace(' ', '-')
df_decode.loc[:, "merge_id"] = np.arange(1, len(df_decode) + 1)
df_decode = df_decode.rename(columns = {"def_engine_id":"engine_id"})

####################################################################################################
# Read in transmission data and merge it with the VIN decoder
df_transmission = pd.read_csv(str_decoder / "LKP_VEH_TRANS.csv")
df_transmission = df_transmission.loc[df_transmission["standard"]=="Y"]
df_decode = df_decode.merge(df_transmission, on = "vehicle_id", how = "left")

####################################################################################################
# Read in the engine data and merge it with the VIN decoder
df_engine = pd.read_csv(str_decoder / "DEF_ENGINE.csv")
df_engine = pd.DataFrame(df_engine.loc[:, ["engine_id", "max_hp"]])
df_decode = df_decode.merge(df_engine, on = "engine_id", how = "left")

####################################################################################################
# Read in MPG data from VIN decoder
df_decode_mpg = pd.read_csv(str_decoder / "LKP_VEH_MPG.csv")

# Now we keep only those entries that are unique or regular fuel grade
df_decode_counts = df_decode_mpg.groupby(['vehicle_id', 'engine_id', 'transmission_id', 'fuel_type']).size().reset_index(name='count_fg')
df_decode_mpg = df_decode_mpg.merge(df_decode_counts, on = ['vehicle_id', 'engine_id', 'transmission_id', 'fuel_type'], how = "left")
df_decode_mpg.loc[:, "multi_fg_flag"] = np.where(df_decode_mpg.loc[:, "count_fg"] > 1, 1, 0)
df_decode_mpg = df_decode_mpg.loc[(df_decode_mpg["multi_fg_flag"]==0) | (df_decode_mpg["fuel_grade"]=="Regular"), :]

# Check that we have no remaining duplicates
df_decode_counts = df_decode_mpg.groupby(['vehicle_id', 'engine_id', 'transmission_id', 'fuel_type']).size().reset_index(name='count')
assert(df_decode_counts["count"].max() == 1)

####################################################################################################
# Now deal with engines for which there are multple entries corresponding to different fuel types
# First, identify them and flag them
df_dup_multi_fuel_counts = df_decode_mpg.groupby(["vehicle_id", "transmission_id", "engine_id"]).size().reset_index(name = 'count_fuel')
df_decode_mpg = df_decode_mpg.merge(df_dup_multi_fuel_counts, on = ["vehicle_id", "transmission_id", "engine_id"], how = "left")
df_decode_mpg.loc[:, "multi_fuel_flag"] = np.where(df_decode_mpg.loc[:, "count_fuel"]>1, 1,0)

# Now, extract only those with multiple fuels
df_decode_mpg_multi_fuels = df_decode_mpg.loc[df_decode_mpg["multi_fuel_flag"]==1, :]
print(df_decode_mpg_multi_fuels.head(5))