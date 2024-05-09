library(data.table)
library(haven)
library(stringr)
set.seed(1217)

str.loc <- '~/Dropbox (YSE)/SW_Automaker_Strategies/data/'

# version of aggregation
state_grouping <- 'ZEV + other'
#state_grouping <- 'ZEV + regional'

# set parameters
N_obs = 5000

# read in ipums data
dt.ipums <- data.table()
for (yr in 2010:2019)
{
  dt.ipums.yr <- data.table(read.csv(paste0(str.loc,"raw/ipums/ipums4_by_year/ipums4_",yr,".csv")))
  dt.ipums <- rbind(dt.ipums,dt.ipums.yr)
}

# read in list of states to keep separate
str.states <- 'raw/zev_states.csv'
dt.states.separate <- data.table(read.csv(paste0(str.loc,str.states)))
str.states.region <- 'raw/census_div_grouping.csv'
dt.states.region <- data.table(read.csv(paste0(str.loc,str.states.region)))


# read in CPI data
str.cpi.data <- 'raw/CPI.csv'
dt.CPI <- data.table(read.csv(paste0(str.loc,str.cpi.data)))
dt.CPI.l <- data.table(melt(dt.CPI,id.vars = 'Year'))
setnames(dt.CPI.l,c('Year','variable','value'),c('year','month','cpi'))

# read in charging density data
# charging density data is processed by charging_moments_code.do
dt.charging.density <- data.table(read.csv(paste0(str.loc,"intermediate/cd_2015_2021.csv")))
dt.charging.density <- unique(dt.charging.density[, c("state", "puma", "year", "charging_density_total")])

# change state abbreviations to state full names
dt.state.abb <- data.table(state = state.abb, state_full = state.name)
dt.charging.density <- merge(dt.charging.density,dt.state.abb,by=c('state'))
dt.charging.density <- dt.charging.density[,!c('state'),with=F]
setnames(dt.charging.density,'state_full','state')

# read in urban-rural distinctions
# not currently using because IPUMS county ICPSR codes missing for significant number of obs
#str.county <- 'raw/RuralUrbanCodes.dta'
#dt.county.codes <- data.table(read_dta(paste0(str.loc,str.county)))

# deal with income 
dt.ipums <- dt.ipums[hhincome != 9999999 & hhincome > 0]

# convert income to 1000s USD
dt.ipums[,hhincome:=hhincome/1000]

dt.CPI.l <- dt.CPI.l[,.(cpi = mean(cpi)),by=c('year')]
dt.CPI.l[,mult_2021:=dt.CPI.l[year == 2021,cpi]/cpi]

# adjust each year's income to 2021 USD
dt.ipums <- merge(dt.ipums,
                  dt.CPI.l[,.(year,mult_2021)],
                  by=c('year'))
dt.ipums <- dt.ipums[,hhincome := hhincome * mult_2021]
dt.ipums <- dt.ipums[,!c('mult_2021'),with=F]

# drop observations with income < $1000
# (Experian data lowerbound income is $1000)
dt.ipums <- dt.ipums[hhincome > 1]

# get average income > 250K for moment calc
#dt.ipums[hhincome > 2.5,.(mean(hhincome))]
#dt.ipums[hhincome > 2,.(mean(hhincome))]

# single family home
dt.ipums <- dt.ipums[unitsstr != 'N/A']
dt.ipums[,single_fam:=ifelse(unitsstr %in% c('1-family house, detached','1-family house, attached'),1,0)]

# urban status
dt.ipums[,urban:=ifelse(grepl("In metropolitan area",metro),1,0)]
# education
dt.ipums[,college_grad:=ifelse(educ %in% c('5+ years of college', '4 years of college'),1,0)]

# only keep one person per household (https://usa.ipums.org/usa-action/variables/HHWT#description_section)
dt.ipums <- dt.ipums[pernum == 1]

## Converted until here


# generating puma variable from the strata variable in the data 
dt.ipums$puma <- as.integer(substr(dt.ipums$strata, 1, nchar(dt.ipums$strata) - 2))

# before merging with charging data, fill in missing pumas with state average
setnames(dt.ipums,'stateicp','state')
# get unique state,puma combinations
dt.ipums.puma <- unique(dt.ipums[,c('state','puma')])
dt.ipums.puma[,i:=1]
dt.ipums.puma <- merge(dt.ipums.puma,data.table(year =unique(dt.charging.density$year), i = 1),by=c('i'),allow.cartesian = T)
dt.ipums.puma <- dt.ipums.puma[,!c('i'),with=F]

# get state averages
dt.charging.density.state.year <- dt.charging.density[,.(st_charging_density_total = mean(charging_density_total,na.rm=T)),by=c('state','year')]
# merge charging density with full list of pumas
dt.charging.density <- merge(dt.ipums.puma,dt.charging.density,by=c('state','puma','year'),all.x=T)
# merge charging density with state averages
dt.charging.density <- merge(dt.charging.density,dt.charging.density.state.year,by=c('state','year'),all.x=T)
# fill in missing obs
dt.charging.density[is.na(charging_density_total) & !is.na(st_charging_density_total),charging_density_total:=st_charging_density_total]
dt.charging.density <- dt.charging.density[,!c('st_charging_density_total'),with=F]

# combine states for aggregate version
dt.ipums.agg <- merge(dt.ipums,
                      dt.states.separate,
                      by=c('state'),
                      all.x=T)
if(state_grouping == 'ZEV + other')
{
  dt.ipums.agg[is.na(zev_full),state:='Combined']
} else if(state_grouping == 'ZEV + regional')
{
  dt.states.region[,state:=str_to_title(state)]
  dt.ipums.agg <- merge(dt.ipums.agg,
                        dt.states.region,
                        by=c('state'),
                        all.x=T)
  dt.ipums.agg[is.na(zev_full) & state != 'District of Columbia',state:=state_grp]
}

dt.ipums.agg <- dt.ipums.agg[,!c('zev_full','zev_later'),with=F]

# generate random sample of obs per year
# this accounts for hh weights!
generate_sample <- function(dt.census,bln.alt.2020,bln.alt.2021)
{
  dt.unique.state.yr <- unique(dt.census[,.(year,state)])
  if(bln.alt.2020)
  {
    dt.2020 <- dt.unique.state.yr[year == 2019]
    dt.2020[,year:=2020]
    dt.unique.state.yr <- rbind(dt.unique.state.yr,dt.2020)
    max_yr <- 2020
    if(bln.alt.2021)
    {
      dt.2021 <- copy(dt.2020)
      dt.2021[,year:=2021]
      dt.unique.state.yr <- rbind(dt.unique.state.yr,dt.2021)
      max_yr <- 2021
    }
  } else
  {
    max_yr <- 2019
  }
  dt.state.yr.all <- data.table()
  for(i in 1:nrow(dt.unique.state.yr))
  {
    yr = dt.unique.state.yr[i,year]
    if(yr == 2020 & bln.alt.2020)
    {
      bln.2020 <- T
      bln.2021 <- F
      yr = 2019
    } else if(yr == 2021 & bln.alt.2021)
    {
      bln.2020 <- F
      bln.2021 <- T
      yr = 2019
    } else
    {
      bln.2020 <- F
      bln.2021 <- F
    }
    st = dt.unique.state.yr[i,state]
    mkt_id = paste0(yr,toupper(st))
    # get sample of relevant subset
    dt.census.sub <- dt.census[state == st & year == yr,]
    # weight by hhwt
    dt.census.sub[,prob:=hhwt/sum(dt.census.sub$hhwt)]
    dt.state.yr <- dt.census.sub[sample(nrow(dt.census.sub), size=N_obs, replace=T,prob = dt.census.sub$prob), ]
    # clean up for use in agent_data
    setnames(dt.state.yr,c('hhincome','year'),c('income','model_year'))
    dt.state.yr[,`:=`(market_ids = mkt_id,weights=1/N_obs)]
    if(bln.2020)
    {
      dt.state.yr[,`:=`(model_year= 2020,market_ids=paste0(2020,toupper(st)))]
    } else if(bln.2021)
    {
      dt.state.yr[,`:=`(model_year= 2021,market_ids=paste0(2021,toupper(st)))]
    }
    dt.state.yr.all <- rbind(dt.state.yr.all,dt.state.yr)
  }
  
  # keep relevant columns
  dt.state.yr.all <- dt.state.yr.all[,.(state,statefip,puma,model_year,market_ids,income,single_fam,urban,college_grad,weights)]
  # keep relevant years, states
  dt.state.yr.all <- dt.state.yr.all[model_year >= 2014 & model_year <= max_yr & state != 'District of Columbia']
  
  return(dt.state.yr.all)
}

# use 2019 households to generate a 2020, 2021 sample
bln.alt.2020 <- T
bln.alt.2021 <- T
if(bln.alt.2020)
{
  print('Using 2019 ACS to generate 2020 sample')
  if(bln.alt.2021)
  {
    print('Using 2019 ACS to generate 2021 sample')
  }
} else
{
  if(bln.alt.2021)
  {
    bln.alt.2021 <- F
    print('Cannot use 2019 ACS for 2021 data without also dealing with 2020 data.')
  }
}
dt.state.yr.all <- generate_sample(dt.ipums,bln.alt.2020,bln.alt.2021)
dt.state.agg.yr.all <- generate_sample(dt.ipums.agg,bln.alt.2020,bln.alt.2021)

# merge with charging density data 
setnames(dt.charging.density,c('state','year'),c('statefip','model_year'))
dt.state.yr.all <- merge(dt.charging.density, dt.state.yr.all, by = c("statefip", "puma", "model_year"), all.y = TRUE)
dt.state.agg.yr.all <-  merge(dt.charging.density, dt.state.agg.yr.all, by = c("statefip", "puma", "model_year"), all.y = TRUE)
# remove extra columns needed for merge
dt.state.yr.all <- dt.state.yr.all[,!c('statefip','puma'),with=F]
dt.state.agg.yr.all <- dt.state.agg.yr.all[,!c('statefip','puma'),with=F]

# save agent obs
write.csv(dt.state.yr.all,paste0(str.loc,'intermediate/agent_data.csv'),row.names=F)
if(state_grouping == 'ZEV + other')
{
  str.agg <- '_ZEV'
} else if(state_grouping == 'ZEV + regional')
{
  str.agg <- '_ZEV_regional'
}
write.csv(dt.state.agg.yr.all,paste0(str.loc,'intermediate/agent_data_agg_state',str.agg,'.csv'),row.names=F)

