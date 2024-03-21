# this script cleans the electricity prices from EIA
# original data downloaded from here: https://www.eia.gov/electricity/sales_revenue_price/ (supplemental data)
# 2021 data available either in table 4 ^ 
# or full time series (incl. 2021) in slightly different format available from EIA 861 (https://www.eia.gov/electricity/data/state/)
# data converted to csv: avgprice_annual.csv


# assuming EIA data is nominal, adjusting everything to 2021 USD

library(data.table)

str.loc <- '~/Dropbox (YSE)/SW_Automaker_Strategies/data/'

# read in electricity price data
str.elec.data <- 'raw/energy_prices/avgprice_annual.csv'
dt.elec <- data.table(read.csv(paste0(str.loc,str.elec.data),skip=1))

# read in 2021 addl. electricity price data
str.elec.data21 <- 'raw/energy_prices/elec_table4.csv'
dt.elec.21 <- data.table(read.csv(paste0(str.loc,str.elec.data21)))

# read in CPI data
str.cpi.data <- 'raw/CPI.csv'
dt.CPI <- data.table(read.csv(paste0(str.loc,str.cpi.data)))
dt.CPI.l <- data.table(melt(dt.CPI,id.vars = 'Year'))
setnames(dt.CPI.l,c('Year','variable','value'),c('year','month','cpi'))

dt.CPI.l <- dt.CPI.l[,.(cpi = mean(cpi)),by=c('year')]
dt.CPI.l[,mult_2021:=dt.CPI.l[year == 2021,cpi]/cpi]

# keep only "Total Electric Industry" "Residential" prices
dt.elec <- dt.elec[Industry.Sector.Category == 'Total Electric Industry']
dt.elec <- dt.elec[,.(Year,State,Residential)]
setnames(dt.elec,'Year','year')

# convert to 2021 $
dt.elec <- merge(dt.elec,dt.CPI.l[,.(year,mult_2021)], by=c('year'))
dt.elec[,cent_per_kwh:=Residential * mult_2021]

# update state abbreviations to full name
setnames(dt.elec,'State','state_abbrev')
dt.state <- data.table(state = state.name, state_abbrev = state.abb)
dt.state <- rbind(dt.state, data.table(state = 'U.S.', state_abbrev = 'US'))

dt.elec <- merge(dt.elec,dt.state,by=c('state_abbrev'))
dt.elec <- dt.elec[,!c('Residential','mult_2021','state_abbrev'),with=F]

# convert 2021 data to same format
dt.elec.21 <- dt.elec.21[,.(State,Residential)]
setnames(dt.elec.21,c('State','Residential'),c('state','cent_per_kwh'))
dt.elec.21[,year:=2021]
# drop some additional regions
dt.elec.21[state == "U.S. Total",state:="U.S."]
dt.elec.21 <- dt.elec.21[state %in% dt.elec$state]

dt.elec <- rbind(dt.elec,dt.elec.21)

# save
write.csv(dt.elec,
          paste0(str.loc,'intermediate/','annual_elec_prices.csv'),
          row.names=F)
