# this script cleans the gasoline (and diesel) cost data from EIA
# original data source downloaded here: https://www.eia.gov/petroleum/gasdiesel/
# regular conventional data converted into CSV: conv_gasoline_timeseries.csv

# this script gets annual average at the PADD-level, maps PADDs to states, converts all prices to 2021 USD

library(data.table)
library(stringr)

str.loc <- '~/Dropbox (YSE)/SW_Automaker_Strategies/data/'

# read in gas price data
str.gas.data <- 'raw/energy_prices/conv_gasoline_timeseries.csv'
dt.gas <- data.table(read.csv(paste0(str.loc,str.gas.data),skip=2))
dt.gas <- dt.gas[,!c('X'),with=F]
dt.gas <- head(dt.gas,-1)

# read in diesel price data
str.diesel.data <- 'raw/energy_prices/diesel_timeseries.csv'
dt.diesel <- data.table(read.csv(paste0(str.loc,str.diesel.data),skip=2))
dt.diesel <- dt.diesel[,!c('X'),with=F]
dt.diesel <- head(dt.diesel,-1)

# read in CPI data
str.cpi.data <- 'raw/CPI.csv'
dt.CPI <- data.table(read.csv(paste0(str.loc,str.cpi.data)))
dt.CPI.l <- data.table(melt(dt.CPI,id.vars = 'Year'))
setnames(dt.CPI.l,c('Year','variable','value'),c('year','month','cpi'))

CPI_21 <- dt.CPI.l[year == 2021 & !(month %in% c('HALF1','HALF2')),.(mean(cpi))]$V1

dt.CPI.l[,mult_2021:=CPI_21/cpi]

# truncate colnames
truncate_cols <- function(dt,str_fuel,str_time)
{
  vec.colnames <- colnames(dt)
  vec.colnames.adj = lapply(vec.colnames, function(x) str_split(x,str_fuel)[[1]][1]) %>% unlist
  if(str_length(str_time) > 0)
  {
    n_cutoff <- str_length(str_time) + 2
  } else
  {
    n_cutoff <- 1
  }
  vec.colnames.adj[2:length(vec.colnames.adj)] <- substring(vec.colnames.adj[2:length(vec.colnames.adj)],n_cutoff,99)
  setnames(dt,vec.colnames,vec.colnames.adj)
  return(dt)
}
dt.gas <- truncate_cols(dt.gas,'.Regular.Conventional','Weekly')
dt.diesel <- truncate_cols(dt.diesel,'.No.2.Diesel','')

# keep only mutually exclusive PADDs and U.S. total
vec.keep <- c('Date','U.S.','New.England..PADD.1A.','Central.Atlantic..PADD.1B.','Lower.Atlantic..PADD.1C.','Midwest','Gulf.Coast',
             'Rocky.Mountain')
dt.gas <- dt.gas[,c(vec.keep,'West.Coast'),with=F]
dt.diesel <- dt.diesel[,c(vec.keep,'West.Coast..PADD.5..Except.California','California'),with=F]  

# clean dates
dt.gas[,Date:=as.Date(Date,format = "%b %d, %Y")]
dt.gas[,year:=year(Date)]
dt.gas[,month_num:=as.integer(month(Date))]
dt.gas[,month:=sapply(1:nrow(dt.gas),function(x) month.abb[dt.gas$month_num[x]])]
dt.gas <- dt.gas[,!c('month_num'),with=F]

dt.diesel[,month:=substr(Date,1,3)]
dt.diesel[,year:=as.integer(substr(Date,5,9))]

# calculate annual average
calc_annl_price <- function(dt)
{
  # reshape long
  dt.l <- data.table(melt(dt,id.vars=c('Date','year','month')))
  setnames(dt.l,c('variable','value'),c('region','price'))
  
  # convert prices to 2021 USD
  dt.l <- merge(dt.l,dt.CPI.l[,.(year,month,mult_2021)],by=c('year','month'))
  dt.l[,price:=price*mult_2021]
  
  # calculate annual avg.
  dt.annual <- dt.l[year < 2022,.(price = mean(price)),by=c('year','region')]
  
  return(dt.annual)
}
dt.gas.annual <- calc_annl_price(dt.gas)
dt.diesel.annual <- calc_annl_price(dt.diesel)

# map to states
dt.states <- data.table(state = state.name)
dt.states[state %in% c('Maine','Vermont','New Hampshire','Massachusetts','Connecticut','Rhode Island'),
          region:='New.England..PADD.1A.']
dt.states[state %in% c('New York','New Jersey','Pennsylvania','Delaware','Maryland'),
          region:='Central.Atlantic..PADD.1B.']
dt.states[state %in% c('West Virginia','Virginia','North Carolina','South Carolina','Georgia','Florida'),
          region:='Lower.Atlantic..PADD.1C.']
dt.states[state %in% c('Michigan','Ohio','Kentucky','Tennessee','Indiana','Wisconsin','Illinois','Minnesota','Iowa',
                       'Missouri','North Dakota','South Dakota','Nebraska','Kansas','Oklahoma'),
          region:='Midwest']
dt.states[state %in% c('Alabama','Mississippi','Arkansas','Louisiana','Texas','New Mexico'),
          region:='Gulf.Coast']
dt.states[state %in% c('Montana','Wyoming','Colorado','Idaho','Utah'),
          region:='Rocky.Mountain']
dt.states[state %in% c('Washington','Oregon','Nevada','California','Arizona','Hawaii','Alaska'),
          region:='West.Coast']
dt.states <- rbind(dt.states, data.table(state = 'U.S.',region = 'U.S.'))

dt.gas.annual <- merge(dt.gas.annual,dt.states,by=c('region'),allow.cartesian = T,all.x=T)
setnames(dt.gas.annual,'price','dollar_per_gal_gas')

# tweak dt.states to account for slightly different regions for diesel
dt.states.alt <- copy(dt.states)
dt.states.alt[state %in% c('Washington','Oregon','Nevada','Arizona','Hawaii','Alaska'),
          region:='West.Coast..PADD.5..Except.California']
dt.states.alt[state == 'California',region:='California']

dt.diesel.annual <- merge(dt.diesel.annual,dt.states.alt,by=c('region'),allow.cartesian = T, all.x=T)
setnames(dt.diesel.annual,'price','dollar_per_gal_diesel')


# save it
write.csv(dt.gas.annual[,!c('region'),with=F],
          paste0(str.loc,'intermediate/','annual_gas_prices.csv'),
          row.names=F)
write.csv(dt.diesel.annual[,!c('region'),with=F],
          paste0(str.loc,'intermediate/','annual_diesel_prices.csv'),
          row.names=F)

