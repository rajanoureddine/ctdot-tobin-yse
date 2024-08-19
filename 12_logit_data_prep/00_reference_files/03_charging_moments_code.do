/*
Last updated: 04/16/2024
Goal: Calculating total charging station counts by port level, charging density by county and by PUMA, merging with Experian sales data 
*/

/*
###
Cluster directories 
###
global Velocity "/gpfs/gibbs/project/gillingham/vs448/AutomakerStrategies/Velocity"
global Charging_Moments "/gpfs/gibbs/project/gillingham/vs448/AutomakerStrategies/Charging_Moments"
global Output "/gpfs/gibbs/project/gillingham/vs448/AutomakerStrategies/Charging_Moments/Output"
*/

/*
###
Dropbox directories
###
global Velocity "~Dropbox/AutomakerStrategies/data/Velocity/Cleaned_20172020_States"
global Charging_Moments "~Dropbox/AutomakerStrategies/data/Charging_Moments"
global Output "~Dropbox/AutomakerStrategies/data/Charging_Moments/Output"
*/

use "$Charging_Moments/ZIP_County_Crosswalk.dta", clear 
* keeping first instance
sort zip
quietly by zip:  gen dup = cond(_N==1,0,_n)
drop if dup > 1
tempfile unique_zip
save `unique_zip'

forvalues i = 2015/2021{
	import delimited "$Charging_Moments/Charging_Stations_by_Year/`i'_Total_ChargingStations.csv", clear
	* For every year-zip combination, we want to keep the highest month number because that's the total charging stations in that zip in the year 
	bysort zip : egen latest_month = max(month)
	* Only keeping the latest month values 
	keep if month == latest_month
	sort state zip month
	quietly by state zip month:  gen dup = cond(_N==1,0,_n)
	drop if dup > 1 
	tempfile total_stations
	save `total_stations'
	
	import delimited "$Charging_Moments/Charging_Stations_with_Ports_by_Year/`i'_Total_ChargingStations_withPorts.csv", clear
	tempfile `i'
	save ``i''
	keep if portlevel == "L2"
	sort state zip month
	quietly by state zip month:  gen dup = cond(_N==1,0,_n)
	drop if dup > 1 
	rename portlevel portlevel_L2
	rename total_stations total_stations_L2
	tempfile `i'_L2
	save ``i'_L2'

	use ``i'', clear 
	keep if portlevel == "DC"
	sort state zip month
	quietly by state zip month:  gen dup = cond(_N==1,0,_n)
	drop if dup > 1 
	rename portlevel portlevel_DC
	rename total_stations total_stations_DC
	merge 1:1 state zip month using ``i'_L2'
	drop portlevel* dup _merge
	* For every year-zip combination, we want to keep the highest month number because that's the total charging stations in that zip in the year 
	bysort zip : egen latest_month = max(month)
	* Only keeping the latest month values 
	keep if month == latest_month
	merge 1:1 state zip month using `total_stations'
	/*
	merge == 2 is most likely L1 chargers
	*/
	drop latest_month dup _merge  
	merge m:1 zip using `unique_zip'
	* We're not interested in _merge == 2 because they are not in our charging stations data 
	drop if total_stations == .
	replace county = lower(county)
	replace total_stations_DC = 0 if total_stations_DC == .
	replace total_stations_L2 = 0 if total_stations_L2 == .
	collapse (sum) total_stations total_stations_L2 total_stations_DC, by(state county year)
	drop if county == ""
	tempfile chargers_county_counts
	save `chargers_county_counts'
	
	* Sometimes total stations < total L2 stations + total DC stations. this is because some stations have both ports and so are counted at both places 

	import delimited "$Charging_Moments/Average_Household_Size_and_Population_Density_-_County.csv", clear
	keep name aland state geoid
	* Converting sq m to sq km 
	replace aland = aland / 1e+6
	replace name = subinstr(name," County", "",.)
	replace name = lower(name)
	rename name county 
	replace state = lower(state)
	tempfile state_county_area
	save `state_county_area'
	* Merging state names with abbreviations 
	import delimited "$Charging_Moments/states", clear varnames(1)
	replace state = lower(state)
	merge 1:m state using  `state_county_area'
	drop _merge 
	replace abbreviation = "PR" if state == "puerto rico"
	drop state 
	rename abbreviation state
	merge 1:1 county state using `chargers_county_counts'
	
	* We're only concerned with merge = 3 and merge = 2. Not concerned with _merge = 1. Dropping _merge == 2 (0.1% of the obs here)  

	keep if _merge == 3
	rename aland county_area_sq_km
	drop _merge 
	rename geoid fips 
	tempfile fips_county
	save `fips_county'
	
	* For county wise charging density 
	
	* Converting area in sq km to area in 1000 sq km  
	replace county_area_sq_km = county_area_sq_km/ 1000
	gen charging_density_total =  total_stations/county_area_sq_km
	gen charging_density_DC =  total_stations_DC/county_area_sq_km
	gen charging_density_L2 =  total_stations_L2/county_area_sq_km
	
	export delimited "$Output/`i'_charging_density_by_county", replace 
	
	* For PUMA level charging density: PUMA must be combined with state codes for unique identification at the national level

	import delimited "$Charging_Moments/2010_Census_Tract_to_2010_PUMA.txt", clear
	tostring *fp, replace
	gen fips = statefp + "00" + countyfp if length(countyfp) == 1 
	replace fips = statefp + "0" + countyfp if length(countyfp) == 2
	replace fips = statefp + countyfp if length(countyfp) == 3

	destring fips, replace
	drop countyfp tractce
	rename puma5ce puma
	destring statefp, replace
	rename statefp state
	tempfile puma_fips
	save `puma_fips'

	import excel "$Charging_Moments/PUMA2000_PUMA2010_crosswalk.xls", firstrow clear
	keep State10 PUMA10 PUMA10_Land
	destring State10 PUMA10, replace
	rename PUMA10 puma
	rename State10 state

	sort puma state
	quietly by puma state :  gen dup = cond(_N==1,0,_n)
	drop if dup > 1
	drop dup 

	* merge not completing with 2020 data of census tract and 2010 puma areas. PUMA areas not there for 2010 onwards thus we are using 2010_Census_Tract_to_2010_PUMA

	merge 1:m puma state using `puma_fips'
	keep if _merge == 3 
	drop _merge 
	sort state puma PUMA10_Land fips
	quietly by state puma PUMA10_Land fips:  gen dup = cond(_N==1,0,_n)
	drop if dup > 1
	drop dup 

	rename state state_code
	merge m:1 fips using `fips_county'
	keep if _merge == 3
	* for us, _merge == 2 is important because it has CD info 
	keep fips puma PUMA10_Land state year total_stations total_stations_DC total_stations_L2
	
	collapse (sum) total_stations total_stations_L2 total_stations_DC , by(state puma PUMA10_Land year)

	* sq m to sq km 
	replace PUMA10_Land = PUMA10_Land * 0.000001 
	rename PUMA10_Land puma_land_area

	* charging density per sq km 

	gen charging_density_total =  total_stations/puma_land_area
	gen charging_density_DC =  total_stations_DC/puma_land_area
	gen charging_density_L2 =  total_stations_L2/puma_land_area
	
	* cd by puma 
	export delimited "$Output/`i'_charging_density_by_puma", replace
	
	save "$Output/`i'_charging_density_by_puma", replace
}

use "$Output/2015_charging_density_by_puma", clear 
append using "$Output/2016_charging_density_by_puma"
append using "$Output/2017_charging_density_by_puma"
append using "$Output/2018_charging_density_by_puma"
append using "$Output/2019_charging_density_by_puma"
append using "$Output/2020_charging_density_by_puma"
append using "$Output/2021_charging_density_by_puma"

export delimited "$Output/cd_2015_2021", replace
	
* Merging with Velocity for fips - sales 

import delimited "$Charging_Moments/states", varnames(1) clear
levelsof abbreviation, local(var1list)
foreach state of local var1list{
	use "$Velocity/`state'.dta", clear 
	destring measurecounttext, replace
	drop if year < 2017 
	collapse (count) measurecounttext, by(stateprovinceabbr fueltype countyname year fips)
	save "$Output/`state'_fips_sales.dta", replace
}

use "$Output/AK_fips_sales.dta", clear 
import delimited "$Charging_Moments/states", varnames(1) clear
drop if abbreviation == "AK"
levelsof abbreviation, local(var1list)
foreach state of local var1list{
	append using "$Output/`state'_fips_sales.dta"
}

rename countyname county 
replace county = lower(county)

replace county = subinstr(county, "st. ", "saint ", .) 
replace county = subinstr(county, "st ", "saint ", .) 
replace county = subinstr(county, "'", "", .) 
replace county = subinstr(county, " parish", "", .) 

replace county = "fairbanks north star" if county == "fairbanks north star borough"
replace county = "juneau" if county == "juneau city and borough"
replace county = "kenai peninsula" if county == "kenai peninsula borough"
replace county = "matanuska susitna" if county == "matanuska-susitna borough"
replace county = "sitka" if county == "sitka city and borough"
replace county = "skagway" if county == "skagway municipality"
replace county = "salem" if county == "salem city"
replace county = "de witt" if county == "dewitt"
replace county = "dona ana" if county == "doÃ±a ana"
replace county = "denali" if county == "denali borough"
replace county = "la salle" if county == "lasalle"
replace county = "la porte" if county == "laporte"
replace county = "saint joseph" if county == "st joseph"
replace county = "anchorage" if county == "anchorage municipality"
replace county = "manatee" if county == "manatíe"
replace county = "st joseph" if county == "saint joseph"

drop state
rename stateprovinceabbr state 
drop if state == ""
drop abbreviation 

destring fips, replace
drop if fips == .

export delimited "$Output/experian_county_fips_sales.csv", replace

import delimited "$Output/experian_county_fips_sales.csv", clear  
tempfile velocity 
save `velocity'

* merging with puma cd by each year 

forvalues i = 2017/2021{
	import delimited "$Output/`i'_charging_density_by_puma", clear 
	tempfile puma_state_`i'
	save `puma_state_`i''

	* PUMA - FIPS crosswalk 
	import delimited "$Charging_Moments/2010_Census_Tract_to_2010_PUMA.txt", clear
	tostring *fp, replace
	gen fips = statefp + "00" + countyfp if length(countyfp) == 1 
	replace fips = statefp + "0" + countyfp if length(countyfp) == 2
	replace fips = statefp + countyfp if length(countyfp) == 3

	destring fips, replace
	drop countyfp tractce
	rename puma5ce puma
	destring statefp, replace
	rename statefp state
	tempfile puma_fips
	save `puma_fips'

	import delimited "$Charging_Moments/state_abbrev_fips.txt", clear 
	rename fips state
	merge 1:m state using `puma_fips'
	keep if _merge == 3
	drop _merge state
	rename state_abbrev state
	merge m:1 puma state using `puma_state_`i''

	* only 5 didnt merge with charging densities. the rest _merge =2 did not have corresponding CD info 
	keep if _merge ==3 
	sort puma fips state
	quietly by puma fips state:  gen dup = cond(_N==1,0,_n)
	drop if dup > 1
	drop dup

	* fips is a smaller geographic entity. same fips have different pumas and same pumas have different fips. we will take the first instance of puma for each fips otherwise cant merge 

	sort  fips state
	quietly by  fips state:  gen dup = cond(_N==1,0,_n)

	* keeping first instance only !!! assumption
	drop if dup > 1
	drop _merge 
	merge 1:m state fips year using `velocity'
	keep if _merge == 3

	order state puma fueltype year measurecounttext puma_land_area total_stations charging_density*
	drop fips dup 
	
	sort state county 
	gsort -_merge
	replace total_stations = 0 if total_stations == .
	replace total_stations_dc = 0 if total_stations_dc == .
	replace total_stations_l2 = 0 if total_stations_l2 == .
	replace charging_density_total = 0 if charging_density_total == . 
	replace charging_density_dc = 0 if charging_density_dc == . 
	replace charging_density_l2 = 0 if charging_density_l2 == . 
	drop _merge
	export delimited "$Output/charging_density_experian_by_puma_`i'", replace
}

forvalues i = 2017/2021{
    import delimited  "$Output/charging_density_experian_by_puma_`i'.csv", clear
    save "$Output/charging_density_experian_by_puma_`i'", replace
}

use "$Output/charging_density_experian_by_puma_2017.dta", clear
forvalues i = 2018/2021{
    append using "$Output/charging_density_experian_by_puma_`i'.dta"
}

* Calculating lagged 
drop if state == ""
tempfile data
save `data'

rename charging_density_total lag_charging_density_total
rename year lag_year
gen year = lag_year + 1
tempfile lag
save `lag'

use `data', clear
merge 1:1 state puma fueltype county year using `lag'
* We're not concerned with _merge == 2 
drop if _merge == 2
gsort -_merge
drop _merge

export delimited "$Output/cd_by_puma_experian_merge_with_lag", replace

* Merging with Rural Urban Codes 

use "$Charging_Moments/RuralUrbanCodes.dta", clear
drop if state == ""
unique state county_name 
rename county_name county

replace county = lower(county)
replace county = subinstr(county, " county", "", .)
replace county = subinstr(county, "st. ", "saint ", .) 
replace county = subinstr(county, "st ", "saint ", .) 
replace county = subinstr(county, "'", "", .) 
replace county = subinstr(county, " parish", "", .) 
replace county = subinstr(county, " census area", "", .) 
replace county = subinstr(county, " borough", "", .) 
replace county = subinstr(county, "-", " ", .) 
replace county = "fairbanks north star" if county == "fairbanks north star borough"
replace county = "juneau" if county == "juneau city and borough"
replace county = "kenai peninsula" if county == "kenai peninsula borough"
replace county = "matanuska susitna" if county == "matanuska-susitna borough"
replace county = "sitka" if county == "sitka city and borough"
replace county = "skagway" if county == "skagway municipality"
replace county = "salem" if county == "salem city"
replace county = "de witt" if county == "dewitt"
replace county = "dona ana" if county == "doÃ±a ana"
replace county = "denali" if county == "denali borough"
replace county = "la salle" if county == "lasalle"
replace county = "la porte" if county == "laporte"
replace county = "anchorage" if county == "anchorage municipality"
replace county = "manatee" if county == "manatíe"
replace county = "st joseph" if county == "saint joseph"
replace county = "aleutians east" if county == "aleutians easaint"
replace county = "aleutians west" if county == "aleutians wesaint"
replace county = "hoonah angoon"  if county == "hoonah angoon, ak"
replace county = "juneau city and" if county == "juneau"
replace county = "sitka"  if county == "sitka city and"
replace county = "prince of wales hyder" if county == "nprice of wales hyder" 
replace county = "wrangell" if county == "wrangell city and"
replace county = "yakutat" if county == "yakutat city and"
replace county = "de kalb" if county == "dekalb"
replace county = "miami-dade" if county == "miami dade"
replace county = "dewitt" if county == "de witt"
replace county =  "de witt" if county == "dewitt" & state == "TX"
replace county = "east carroll" if county == "easaint carroll"
replace county = "east feliciana" if county == "easaint feliciana"
replace county = "west feliciana" if county == "wesaint feliciana"
replace county = "st john the baptist" if county == "saint john the baptisaint"
replace county = "west baton rouge" if county == "wesaint baton rouge"
replace county = "west carroll" if county == "wesaint carroll"
replace county = "saint joseph" if county == "st joseph"
replace county = "sainte genevieve" if county == "ste. genevieve"
replace county = "dekalb" if county == "de kalb"
replace county = "de kalb" if county == "dekalb" & state == "AL"
replace county = "st joseph"  if county == "saint joseph"
replace county = "radford"  if county == "radford city"
replace county = "saint joseph" if county == "st joseph" & state == "MI"
replace county = "prince of wales hyder"  if county == "price of wales hyder"
rename state stateprovinceabbr

tempfile rural_urban
save `rural_urban'

* Modifying Velocity data 

use "$Velocity/AK.dta", clear 
keep stateprovinceabbr fips vel_fueltype dwellingtype householdincome fueltype year
import delimited "$Charging_Moments/states", varnames(1) clear
drop if abbreviation == "AK"
levelsof abbreviation, local(var1list)
foreach state of local var1list{
	append using "$Velocity/`state'.dta", keep(stateprovinceabbr fips vel_fueltype dwellingtype householdincome fueltype year)
}

gen ev = 0 
replace ev = 1 if vel_fueltype == "Electric"
gen phev = 0 
replace phev = 1 if vel_fueltype == "Plug In Hybrid"
gen all_ev = 0 
replace all_ev = 1 if ev == 1 | phev == 1

gen single_fam = 0 
replace single_fam = 1 if dwellingtype == "Single Family"
gen multi_fam = 0 
replace multi_fam = 1 if dwellingtype == "Multi-Family & Condo" | dwellingtype == "Marginal Multi-Family" 
gen dwell_unspec = 0 
replace dwell_unspec = 1 if dwellingtype == "Unspecified"
keep if multi_fam == 1 | single_fam == 1 | dwell_unspec == 1

gen income_1 = 0 
replace income_1 = 1 if householdincome == "$1,000  - $14,999" | householdincome == "$15,000 - $24,999" | householdincome == "$25,000 - $34,999" | householdincome == "$35,000 - $49,999" | householdincome == "$50,000 - $74,999" | householdincome == "$75,000 - $99,999"

gen income_2 = 0 
replace income_2 = 1 if householdincome == "$100,000 - $124,999" | householdincome == "$125,000 - $149,999" | householdincome == "$150,000 - $174,999"| householdincome == "$175,000 - $199,999" 

gen income_3 = 0 
replace income_3 = 1 if householdincome == "$200,000 - $249,999" | householdincome == "$250,000+" 

gen income_bucket = 0 
replace income_bucket = 1 if income_1 == 1 
replace income_bucket = 2 if income_2 == 1 
replace income_bucket = 3 if income_3 == 1 

merge m:1 stateprovinceabbr fips using `rural_urban'
drop _merge
tempfile vel_rural_urban
save `vel_rural_urban'

import delimited "$Output/cd_by_puma_experian_merge_with_lag.csv", clear

drop if county == "unassigned" | county == ""
replace county = "northwesaint arctic" if county == "northwest arctic"
replace county = "southeasaint fairbanks" if county == "southeast fairbanks"
replace county = "valdez cordova" if county == "valdez-cordova census area"
replace county = "juneau city and" if county == "juneau"

rename state stateprovinceabbr

merge 1:m stateprovinceabbr county fueltype year using `vel_rural_urban'
* _merge == 2 has no county info, or rucc info 

keep if _merge == 3

save "$Velocity/cd_puma_rur_urban", replace

/*
sum charging_density_total if rucc_2013 == 4 | rucc_2013 == 5 | rucc_2013 == 6 | rucc_2013 == 7 | rucc_2013 == 8  | rucc_2013 == 9 [aweight = vehiclecount]
sum charging_density_total if rucc_2013 == 1 | rucc_2013 == 2 | rucc_2013 == 3 [aweight = vehiclecount]

sum charging_density_total if rucc_2013 == 4 | rucc_2013 == 5 | rucc_2013 == 6 | rucc_2013 == 7 | rucc_2013 == 8  | rucc_2013 == 9 
sum charging_density_total if rucc_2013 == 1 | rucc_2013 == 2 | rucc_2013 == 3 

sum charging_density_total [aweight = vehiclecount
* weighted and unweighted are very similar 

sum charging_density_total 
*/


use "$Velocity/cd_puma_rur_urban", clear

replace fueltype = "Non-EV" if fueltype == "D"
replace fueltype = "Non-EV" if fueltype == "Electric Gasoline Hybrid"
replace fueltype = "Non-EV" if fueltype == "F"
replace fueltype = "Non-EV" if fueltype == "G"
replace fueltype = "Non-EV" if fueltype == "Y"

egen n_obs = count(_n), by (fueltype year)

collapse (mean) charging_density_total, by(year fueltype n_obs)

* ONLY EVs

gen ev = 0 
replace ev = 1 if fueltype == "L"

* ONLY PHEVs

gen phev = 0 
replace phev = 1 if fueltype == "I"

drop if ev == 0 & phev == 0 
tempfile part1
save `part1'


use "$Velocity/cd_puma_rur_urban", clear

replace fueltype = "Non-EV" if fueltype == "D"
replace fueltype = "Non-EV" if fueltype == "Electric Gasoline Hybrid"
replace fueltype = "Non-EV" if fueltype == "F"
replace fueltype = "Non-EV" if fueltype == "G"
replace fueltype = "Non-EV" if fueltype == "Y"

egen n_obs = count(_n), by (fueltype year)

/*
We want to look at charging infrastructure for people who buy EVs vs ones who dont
*/

collapse (mean) charging_density_total, by(year fueltype n_obs)
* ONLY EVs

gen ev = 0 
replace ev = 1 if fueltype == "L"

* ONLY PHEVs

gen phev = 0 
replace phev = 1 if fueltype == "I"

gen broad_ev = 0 
replace broad_ev = 1 if ev == 1 | phev == 1

replace ev = 0 if broad_ev == 1
replace phev = 0 if broad_ev == 1

append using `part1'

replace broad_ev = 0 if broad_ev == .
drop fueltype 

gen state_grp = "all"

tempfile part2
save `part2'

*/

use "$Velocity/cd_puma_rur_urban", clear

replace fueltype = "Non-EV" if fueltype == "D"
replace fueltype = "Non-EV" if fueltype == "Electric Gasoline Hybrid"
replace fueltype = "Non-EV" if fueltype == "F"
replace fueltype = "Non-EV" if fueltype == "G"
replace fueltype = "Non-EV" if fueltype == "Y"

egen n_obs = count(_n), by(fueltype)
collapse (mean) charging_density_total, by(fueltype n_obs)
* ONLY EVs

gen ev = 0 
replace ev = 1 if fueltype == "L"

* ONLY PHEVs

gen phev = 0 
replace phev = 1 if fueltype == "I"

drop if ev == 0 & phev == 0 

tempfile part3
save `part3'


use "$Velocity/cd_puma_rur_urban", clear

replace fueltype = "Non-EV" if fueltype == "D"
replace fueltype = "Non-EV" if fueltype == "Electric Gasoline Hybrid"
replace fueltype = "Non-EV" if fueltype == "F"
replace fueltype = "Non-EV" if fueltype == "G"
replace fueltype = "Non-EV" if fueltype == "Y"

egen n_obs = count(_n), by (fueltype)
collapse (mean) charging_density_total, by(fueltype n_obs)

* ONLY EVs

gen ev = 0 
replace ev = 1 if fueltype == "L"

* ONLY PHEVs

gen phev = 0 
replace phev = 1 if fueltype == "I"

gen broad_ev = 0 
replace broad_ev = 1 if ev == 1 | phev == 1

replace ev = 0 if broad_ev == 1
replace phev = 0 if broad_ev == 1

collapse (sum) n_obs (mean) charging_density_total if broad_ev == 1, by(ev phev broad_ev) 

append using `part3'

replace broad_ev = 0 if broad_ev == .
drop fueltype 

gen state_grp = "all"

append using `part2'

replace year = 0 if year == .

export delimited "$Output/charging_moments", replace 
