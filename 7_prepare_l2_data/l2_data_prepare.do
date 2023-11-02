* Clean up L2 data
* Import the L2 data
use "/home/rrn22/project/data/l2_data_raw/VM2--CT--2022-08-17-DEMOGRAPHIC.dta", clear

* Now rename v359 which is the income we want
rename v359 Residence_Address_Income_Raw

* Now mark duplicate house addresses, in a column called dup
sort Residence_Families_FamilyID
quietly by Residence_Families_FamilyID:  gen dup = cond(_N==1,0,_n)
rename dup Residence_Address_dup


* Now keep only the vars we want
keep Residence_Address*

rename Residence_Address_Income_Raw income_raw
rename Residence_Address_dup dup

* Now drop duplicate entries for the same household
drop if dup > 1

* Now destring and prepare the income data
destring(income_raw), generate(income_int) ignore("$")

egen income_cut = cut(income_int), at(6000, 10000, 20000, 30000, 40000, 50000, 100000, 150000, 200000, 358503)

outsheet using output.csv, comma


