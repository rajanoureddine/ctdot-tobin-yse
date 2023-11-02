* Clean up L2 data
* Import the L2 data
use "/home/rrn22/project/data/l2_data_raw/VM2--CT--2022-08-17-DEMOGRAPHIC.dta", clear

* Keep what we need
keep SEQUENCE Residence_*

* Export
outsheet using "data/l2_data_processed/l2_addresses_for_geocode.csv", comma
