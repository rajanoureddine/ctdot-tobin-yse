import pandas as pd

def fetch_unmatched_vin(unmatched_vin):
        """
        Input: An unmatched, but corrected VIN
        Output: A matched VIN or NA
        
        """
        variables = ["Manufacturer Name", "Model", "Model Year", "Fuel Type - Primary", "Electrification Level"]
        
        url = (f"https://vpic.nhtsa.dot.gov/api/vehicles/DecodeVin/{unmatched_vin.strip()}?format=csv")

        # Download response
        resp_df = pd.read_csv(url)

        # Extract needed
        resp_df = resp_df.loc[resp_df["variable"].isin(variables), ["variable", "value"]].T
        resp_df.columns = resp_df.iloc[0]
        resp_df = resp_df.drop("variable", axis = 0)
        resp_df["vin_corrected"] = unmatched_vin
        
        return resp_df