* Import data
import delimited "/home/rrn22/data/data_for_poisson/poisson_data_112023.csv", clear

* Fix strings
destring pct_over25_hs, replace

* Rescale variables
replace population = population / 10000
replace pop_density = pop_density/100
replace median_income = median_income / 100000
replace pct_over25_hs = pct_over25_hs / 10

* Run a Poisson GMM estimation
gmm (veh_count - exp({xb:population pop_density median_income pct_over25_hs}+{b0})), instruments(population pop_density pct_over25_hs median_income) variables(veh_count population pop_density pct_over25_hs median_income) vce(robust, independent)

* Run Poisson GMM estimation with clustered standard errors
gmm (veh_count - exp({xb:population pop_density median_income pct_over25_hs}+{b0})), instruments(population pop_density pct_over25_hs median_income) wmatrix(cluster zip_code, independent) variables(veh_count population pop_density pct_over25_hs median_income) vce(cluster zip_code, independent)
