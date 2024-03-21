####################################################################################################
# Import libraries
import pathlib
import pandas as pd
import numpy as np
from itertools import combinations, product
import os
from tqdm import tqdm
import requests
from datetime import datetime
import geopandas as gpd
import matplotlib.pyplot as plt
import warnings
import platform

####################################################################################################
def calc_dollar_mile_rlp(df, df_energy,
                         fuel_type_col,
                         year_col, month_col = None, 
                         mpg_col, 
                         gas_price_col = None,
                         diesel_price_col = None,
                         electricity_price_col = None):
    """
    Calculate the dollar per mile for RLP data
    :param df: DataFrame, the data
    :param df_energy: DataFrame, the energy data
    :return: DataFrame, the data with the dollar per mile RLP
    """
    # Merge data
    df_length = len(df)
    df = df.merge(df_energy, on=[year_col, month_col], how="left")
    assert(df_length == len(df)), "Merge failed"

    # Calculate dollars per mile
    df["dollar_per_mile"] = df[mpg_col] / df[gas_price_col]

    return df

# We need an input dataframe of vehicles, and each of them has a fuel type, date, and mpg