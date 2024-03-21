# Description: This script prepares the data for the random logit model estimation. It includes the following steps:
# 1. Import required packages
# 2. Set paths

###################################################################################################
# Import required packages
import pathlib
import pandas as pd
import numpy as np
from itertools import combinations
import os
from tqdm import tqdm
import requests
from datetime import datetime
import geopandas as gpd
import matplotlib.pyplot as plt
import warnings
import platform


###################################################################################################
# Import helper functions
from logit_demand_functions import *

# Warnings and display
warnings.filterwarnings("ignore")
pd.set_option('display.max_rows', 500)

###################################################################################################
# Set paths
if platform.platform()[0:5] == 'macOS':
    cd = pathlib.Path().resolve().parent.parent
    str_project = cd / "Documents" 
    data_path = cd / "Documents" / "tobin_working_data"
    rlpolk_data_path = data_path / "rlpolk_data"
    vin_matching_path =  data_path / "vin_matching"
    data_one_vins_path = data_path / "vin_decoder"
else:
    data_path = pathlib.Path().resolve().parent.parent / "rn_home" / "data"
    rlpolk_data_path = data_path / "rlpolk_data"
    vin_matching_pathpackages= data_path / "vin_matching"

###################################################################################################
# Read in the RLP Data that's been merged to D1 from the ipynb file in this folder
data = pd.read_csv(rlpolk_data_path / "rlp_d1_merge_020724.csv")


