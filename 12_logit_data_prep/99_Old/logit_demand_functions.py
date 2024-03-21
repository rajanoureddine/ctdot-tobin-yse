import pandas as pd
def get_default_vin_variables():
     variables = ["Make", "Manufacturer Name",
                  "Model", "Model Year", "Body Class",
                  "Trim", "Trim2", "Drive Type", "Base Price ($)",
                  "Fuel Type - Primary",
                  "Electrification Level"]

def fetch_vin_data(vin, variables = None):
        """
        Input: An unmatched, but corrected VIN
        Output: A matched VIN or NA
        
        """
        if not variables:
            variables = ["Manufacturer Name", "Model", "Model Year", "Fuel Type - Primary", "Electrification Level"]
        else:
            variables = variables
        
        url = (f"https://vpic.nhtsa.dot.gov/api/vehicles/DecodeVin/{vin.strip()}?format=csv")

        # Download response
        resp_df = pd.read_csv(url)

        # Extract needed
        resp_df = resp_df.loc[:, ["variable", "value"]].T
        # resp_df = resp_df.loc[resp_df["variable"].isin(variables), ["variable", "value"]].T
        resp_df.columns = resp_df.iloc[0]
        resp_df = resp_df.drop("variable", axis = 0)
        resp_df["vin_corrected"] = vin
        
        return resp_df