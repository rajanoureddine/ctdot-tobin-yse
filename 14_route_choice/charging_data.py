# import relevant packages
import numpy as np
import pandas as pd
import os
import sys
import pathlib
import matplotlib.pyplot as plt

# Set the paths
str_cwd = pathlib.Path().resolve().parent / "Documents" / "tobin_working_data"
str_data = str_cwd / "route_choice"

# Remove limit on number of columns
pd.set_option('display.max_columns', None)

####################################################################################################
# Import
#pub_sess = pd.read_csv(str_data / "evwatts_public_session.csv")
pub_evse = pd.read_csv(str_data / "evwatts_public_evse.csv")
pub_connector = pd.read_csv(str_data / "evwatts_public_connector.csv")

#print(pub_sess.head())
print(pub_evse.head())
print(pub_connector.head())

print(pub_evse.columns)
# print(pub_sess.columns)

# From the pub_evse data, try to extract the state
pub_evse['state'] = pub_evse['metro_area'].str.split(',')
pub_evse['state'] = pub_evse['state'].str[1].str[1:3]
ct = pub_evse.loc[pub_evse['state'] == 'CA']
print(len(ct))
print(pub_evse['state'].unique())
# print(ct.head(5))
# print(pub_evse.loc[pub_evse['state'] == 'CT'])