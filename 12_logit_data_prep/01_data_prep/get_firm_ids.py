###########
import pandas as pd
import numpy as np
from pathlib import Path

str_cwd = Path().resolve().parent
str_dir = str_cwd / "Documents" / "tobin_working_data"
str_old_data = str_dir / "intermediate" / "US_VIN_data_common.csv"
str_out = str_dir / "rlpolk_data"

old_data = pd.read_csv(str_old_data)

makes_manufacturer = old_data[["make", "manufacturer_policy", "firm_ids"]].drop_duplicates().sort_values("firm_ids").reset_index(drop = True)
print(makes_manufacturer)
makes_manufacturer = makes_manufacturer.drop_duplicates(subset = ["make"]).reset_index(drop=True)

makes_manufacturer.to_csv(str_out / "brand_to_oem_generated.csv")