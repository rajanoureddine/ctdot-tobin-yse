####################################################################################################
# Import libraries
import pathlib
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt

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

# Now reshape to wide, and create column suffixes that indicate what transport mode it is
cols = ["vehicle_id", "transmission_id", "engine_id", "fuel_type", "city", "highway", "combined"]
df_decode_mpg_multi_fuels = df_decode_mpg_multi_fuels.loc[:, cols].pivot(index = ["vehicle_id", "transmission_id", "engine_id"], 
                                                            columns=["fuel_type"]).reset_index()
df_decode_mpg_multi_fuels.columns = [col[0] for col in df_decode_mpg_multi_fuels.columns[0:3]]+[col[0]+"_"+col[1] for col in df_decode_mpg_multi_fuels.columns[3:]]

# Fix for gasoline
df_decode_mpg_multi_fuels.loc[:, "fuel1"] = "Gasoline"
df_decode_mpg_multi_fuels = df_decode_mpg_multi_fuels.rename(columns = {"city_Gasoline":"city_mpg1", "highway_Gasoline":"highway_mpg1", "combined_Gasoline":"combined_mpg1"})

# Get the second fuel
df_decode_mpg_multi_fuels.loc[:, "fuel2"] = "NA"
df_decode_mpg_multi_fuels.loc[df_decode_mpg_multi_fuels["city_E85"].notna(), "fuel2"] = "E85"
df_decode_mpg_multi_fuels.loc[df_decode_mpg_multi_fuels["city_Ethanol"].notna(), "fuel2"] = "Ethanol"
df_decode_mpg_multi_fuels.loc[df_decode_mpg_multi_fuels["city_Electricity"].notna(), "fuel2"] = "Electricity"

# Now, for the input into relevant column
for fuel in ["E85", "Ethanol", "Electricity"]:
    df_decode_mpg_multi_fuels.loc[df_decode_mpg_multi_fuels["fuel2"]==fuel, "city_mpg2"] = df_decode_mpg_multi_fuels.loc[df_decode_mpg_multi_fuels["fuel2"]==fuel, "city_"+fuel]
    df_decode_mpg_multi_fuels.loc[df_decode_mpg_multi_fuels["fuel2"]==fuel, "highway_mpg2"] = df_decode_mpg_multi_fuels.loc[df_decode_mpg_multi_fuels["fuel2"]==fuel, "highway_"+fuel]
    df_decode_mpg_multi_fuels.loc[df_decode_mpg_multi_fuels["fuel2"]==fuel, "combined_mpg2"] = df_decode_mpg_multi_fuels.loc[df_decode_mpg_multi_fuels["fuel2"]==fuel, "combined_"+fuel]

# Drop the columns that are no longer needed and reorder
df_decode_mpg_multi_fuels = df_decode_mpg_multi_fuels[['vehicle_id', 'transmission_id', 'engine_id', 'fuel1', 'fuel2', 'city_mpg1', 'highway_mpg1', 'combined_mpg1', 'city_mpg2', 'highway_mpg2', 'combined_mpg2']]

# Concat with df_decode_mpg and reset the flag (since it was dropped for duplicates earlier)
df_decode_mpg = pd.concat([df_decode_mpg.loc[df_decode_mpg["multi_fuel_flag"]==0, :], df_decode_mpg_multi_fuels], axis = 0)
df_decode_mpg.loc[df_decode_mpg["multi_fuel_flag"].isna(), "multi_fuel_flag"] =1

# Drop columns with "old" or "count" in them
df_decode_mpg = df_decode_mpg.loc[:, ~df_decode_mpg.columns.str.contains("old")]
df_decode_mpg = df_decode_mpg.loc[:, ~df_decode_mpg.columns.str.contains("count")]

# Check there are no remaining duplicats
df_decode_counts = df_decode_mpg.groupby(['vehicle_id', 'engine_id', 'transmission_id', 'fuel_type']).size().reset_index(name='count')
assert(df_decode_counts["count"].max() == 1)


####################################################################################################
# Merge mpg data with rest of the decoder data
df_decode_mpg = df_decode_mpg.rename(columns = {"transmission_id":"trans_id"})
df_decode = df_decode.merge(df_decode_mpg, on = ["vehicle_id", "engine_id", "trans_id"], how = "left")

# Deal with unusual fuel type mismatches
mask = (df_decode["fuel_type_x"] == "F") & (df_decode["fuel_type_y"].isin(["Gasoline", "Ethanol"]))
mask = mask | (df_decode["fuel_type_x"] == "G") & (df_decode["fuel_type_y"].isin(["Ethanol", "E85"]))
mask = mask | (df_decode["fuel_type_x"] == "I") & (df_decode["fuel_type_y"].isin(["Gasoline", "Electricity"]))
mask = mask | (df_decode["fuel_type_x"] == "N") & (df_decode["fuel_type_y"]=="Gasoline")
df_decode.loc[mask, ["city", "highway", "combined", "city_mpg1", "highway_mpg1", "combined_mpg1", "city_mpg2", "highway_mpg2", "combined_mpg2", "fuel1", "fuel2"]] = np.nan

# Complete clean of decoder data
df_decode = df_decode.drop(columns = ['merge_id','standard','fuel_type_y','veh_mpg_id',
                              'fuel_grade','multi_fg_flag','multi_fuel_flag'])
df_decode = df_decode.rename(columns = {"fuel_type_x":"fuel_type", "year":"model_year"})
df_decode.loc[:, "vin_orig"] = df_decode.loc[:, "vin_pattern"]
df_decode.loc[:, "vin_pattern"] = df_decode.loc[:, "vin_pattern"].str[0:9]

print("Length of df_decode:", len(df_decode))
print("Length of df_unique:", len(df_unique))


####################################################################################################
# Merge with the unique RLP data
vars_to_keep = ['vin_pattern','vin_orig','vehicle_id','model_year',
            'make','model','trim','style','plant','length',
            'height','width','wheelbase','curb_weight','max_hp',
            'fuel_type','msrp','vehicle_type','body_type','drive_type',
            'doors','def_engine_size','city','highway','combined','fuel1',
            'city_mpg1','highway_mpg1','combined_mpg1','fuel2','city_mpg2',
            'highway_mpg2','combined_mpg2']
df_decode_formerge = df_decode.loc[:, vars_to_keep]
df_unique_decoded = df_unique.merge(df_decode_formerge,
                               on = ["model_year", "vin_pattern"], how = "left")

# Print length of df_unique_decoded
print("Length of df_unique_decoded:", len(df_unique_decoded))

df_unique_decoded_unmatched = df_unique_decoded.loc[df_unique_decoded["vehicle_id"].isna(), :]
df_unique_decoded_matched = df_unique_decoded.loc[df_unique_decoded["vehicle_id"].notna(), :]

# Print the number of unmatched entries
print("Number of unmatched entries: ", len(df_unique_decoded_unmatched))

# Save the decoded data
file_name = "ct_rlp_vin_decoded_040324.csv"
df_unique_decoded_matched.to_csv(str_sp / file_name, index = False)

# Save the rlp data with the IDs
file_name_2 = "ct_zip_sp_040324.csv"
df.to_csv(str_sp / file_name_2, index = False)