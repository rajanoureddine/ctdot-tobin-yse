* Import the geocoded data which is a CSV
import delimited "/home/rrn22/project/data/l2_data_processed/ct_l2_data_geocoded_11021500.csv"

* Rename the key variable to SEQUENCE
rename user_sequence SEQUENCE

* Drop everything else
drop user_* 
drop v84 v88 v93

* merge
merge 1:1 SEQUENCE using "/home/rrn22/project/data/l2_data_raw/VM2--CT--2022-08-17-DEMOGRAPHIC.dta"

* Move to save directory
cd "/gpfs/gibbs/project/gillingham/rrn22/data/l2_data_processed"

* Save
save l2_data_geocoded_full_110623
