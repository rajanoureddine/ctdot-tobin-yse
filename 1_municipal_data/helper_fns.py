import pandas as pd
import re
import numpy as np

def create_valid_zip(zip):
    try:
        zip_str = str(zip)
        zip_str = zip_str.strip()
        has_dot = re.search(r"\.", zip_str)

        # Get rid of decimal places
        if has_dot:
            zip_str = zip_str[0:re.search(r"\.", zip_str).start()]

        split_zip = re.split("-", zip_str)
        
        if len(split_zip) == 2:
            return create_valid_zip(split_zip[0])
        else:
            # If length is less than 4, return na
            if len(zip_str) < 4:
                return np.NaN
                
            # If length is 4 or 5, check it
            elif((len(zip_str) == 5) | (len(zip_str) == 4)):
                matched = re.match("^\s*[0-9]*[0-9]{4}\.?0?\s*$", zip_str)
                if matched:
                    return matched[0].zfill(5)
                else:
                    return np.NaN
            # If the zip is between 5 and 8 (inclusive) long, we assume the first 4 are the first part
            # And the second 4 are the second part
            # There is no other way to do this... 
            elif((len(zip_str) > 5) & (len(zip_str)<9)):
                return create_valid_zip(zip_str[0:4])
            elif (len(zip_str) == 9):
                return create_valid_zip(zip_str[0:5])
            else:
                return np.NaN
        
    except Exception as e:
        print(e)
        return np.NaN


def convert_vin_valid(vin):
    try:
        vin_str = str(vin)
        if len(vin_str) < 11:
            return "NA"
        if " " in vin_str[0:11]:
            return "NA"
        else:
            return vin_str[0:8]+"*"+vin_str[9:11]
    except:
        return "NA"

def return_matched_vins(chunk_number, df, vin_column, matching_list):
    match = df.merge(matching_list,
                    left_on = vin_column,
                    right_on = vin_column,
                    how = 'left')
    
    # Get rows of DF where VINS matched
    df_vins_matched = match.loc[match["Manufacturer Name"].notna(), :]
    df_vins_unmatched = match.loc[match["Manufacturer Name"].isna(), :]
    
    # Get length
    len_matched = len(df_vins_matched)
    len_unmatched = len(df_vins_unmatched)
    len_all = len(match)
    
    # Create df
    tally_dict = {"Chunk Number": [chunk_number],
                  "Matched" : [len_matched],
                  "Unmatched" : [len_unmatched],
                  "All" : [len_all]}
    
    match_unmatched_tally = pd.DataFrame(tally_dict)

    return [match, match_unmatched_tally]

def try_divide(x,y):
    try:
        return x/y
    except:
        return np.NaN