library(data.table)
library(ggplot2)
library(stringr)
library(dplyr)

# setup ----
str.inmoment <- '~/Dropbox (YSE)/SW_Polk/InMoment/11-22-23 - 2018-2022 Data/'
str.loc <- '~/Dropbox (YSE)/SW_Automaker_Strategies/data/'
str.save <- paste0(str.loc,'final/')

# drop restricted states
bln.restricted <- F

# read in crosswalk
dt.crosswalk <- data.table(read.csv(paste0(str.inmoment,'../vin_inmoment_crosswalk.csv')))
dt.crosswalk[,model_inmoment:=str_to_title(model_inmoment)]
dt.crosswalk[,make_inmoment:=str_to_title(make_inmoment)]
dt.crosswalk[make_inmoment == 'Mercedes-Benz',make_inmoment:='Mercedes']
# drop 'NSI' from model names
dt.crosswalk[,model_inmoment:=str_replace(model_inmoment,' Nsi','')]

# read in vehicle characteristic data
# read in vin data
str.vin.data <- 'raw/vehicle_sales/state_vin_cts_w_2021.csv'
dt.vin.data <- data.table(read.csv(paste0(str.loc,str.vin.data)))

# read in full EPA range data
str.epa.ev <- 'intermediate/EV_PHEV_attributes_EPA.csv'
dt.epa.ev <- data.table(read.csv(paste0(str.loc,str.epa.ev)))

# read in missing FEs from EPA data
str.missing.fe <- 'intermediate/missing_FE_EPA.csv'
dt.missing.fe <- data.table(read.csv(paste0(str.loc,str.missing.fe)))

# read in vehicle classes from EPA data
str.vclass <- 'intermediate/EPA_vclass.csv'
dt.vclass <- data.table(read.csv(paste0(str.loc,str.vclass)))

# read in additional data from vin decoder
str.vin.decoder <- 'intermediate/decoder_supplement.csv'
dt.decoder <- data.table(read.csv(paste0(str.loc,str.vin.decoder)))

# read in battery size data
str.battery <- 'intermediate/ev_model_style_trim_battery.csv'
dt.battery <- data.table(read.csv(paste0(str.loc,str.battery)))

# read in oem-brand mapping
str.oem.data <- 'raw/brand_oem_mapping.csv'
dt.oem.data <- data.table(read.csv(paste0(str.loc,str.oem.data)))
dt.oem.data <- dt.oem.data[,!c('notes')]

# read in energy costs
str.elec <- 'intermediate/annual_elec_prices.csv'
str.gas <- 'intermediate/annual_gas_prices.csv'
str.diesel <- 'intermediate/annual_diesel_prices.csv'
dt.elec <- data.table(read.csv(paste0(str.loc,str.elec)))
dt.gas <- data.table(read.csv(paste0(str.loc,str.gas)))
dt.diesel <- data.table(read.csv(paste0(str.loc,str.diesel)))

# read in CPI data
str.cpi.data <- 'raw/CPI.csv'
dt.CPI <- data.table(read.csv(paste0(str.loc,str.cpi.data)))

# read in sibling data
str.sib.ev <- 'intermediate/ev_phev_sibling_indicator.csv'
dt.sib.ev <- data.table(read.csv(paste0(str.loc,str.sib.ev)))
str.sib.gas <- 'intermediate/gas_sibling_list.csv'
dt.sib.gas <- data.table(read.csv(paste0(str.loc,str.sib.gas)))

# read in dollar per mile quantiles (all states separated)
str.quantile <- 'intermediate/dpm_quantiles_US_VIN_data_common_agg_states.csv'
dt.dpm.quantiles <- data.table(read.csv(paste0(str.loc,str.quantile)))

# read in wheelbase quantiles (all states separated)
str.wb.quantile <- 'intermediate/wb_quantiles_US_VIN_data_common_agg_states.csv'
dt.wb.quantiles <- data.table(read.csv(paste0(str.loc,str.wb.quantile)))

# get list of year files
lst.files <- list.files(str.inmoment)
lst.files <- lst.files[grepl(" Formatted",lst.files)]

## read, baseline clean sc data ----
# loop over sc files to read
dt.sc <- data.table()
#f <- lst.files[1]
for (f in lst.files)
{
  dt.f <- data.table(read.csv(paste0(str.inmoment,f)))
  # combine with main data
  dt.sc <- rbind(dt.sc,dt.f)
}

# keep only relevant columns
setnames(dt.sc,colnames(dt.sc),tolower(colnames(dt.sc)))
dt.sc.keep <- dt.sc[,.(study.year,
                       purchase.year.from.admark,
                       purchase.month.from.admark,
                       used.to.distinguish.models.with.different.model.years.within.a.study.year,
                       purchase.vehicle..division.,purchase.vehicle..model.,
                       purchase.vehicle..model...series.,
                       define.diesel..hybrid.and.gas.engines.by.vin,
                       did.you.consider.any.other.cars.or.trucks,
                       mmsc..division.,
                       mmsc..model.response.,
                       mmsc..model.and.series.,
                       mmsc..model.year.,
                       engine.type.of.1st.considered.model..gas..diesel..or.hybrid.,
                       respondent.s.state.from.admark,
                       would.you.consider.these.engine.types.for.your.next.vehicle..electric.vehicle..ev..,
                       would.you.consider.these.engine.types.for.your.next.vehicle..plug.in.hybrid.,
                       would.you.consider.these.engine.types.for.your.next.vehicle..hybrid.engine.)]
setnames(dt.sc.keep,colnames(dt.sc.keep),c('study_year','purchase_year','purchase_month','model_year','make','model','trim','purchase_fuel','sc','sc_make','sc_model','sc_trim','sc_year','sc_fuel','state','consider_ev','consider_phev','consider_hybrid'))

# list of restricted states
dt.restricted <- c('AK','AZ','CA','HI','IL','KS','MD','MT','NV','NH','NY','OR','PA','SD','WA')
if(bln.restricted)
{
  dt.sc.keep <- dt.sc.keep[!(state %in% dt.restricted)]
}

## get fueltype sc moments ----
# get count of each combination
# do this with and without EV/PHEV broken out
lst.sc.share <- list()
for (v in c('combined','sep'))
{
  if (v == 'combined')
  {
    dt.sc.v <- copy(dt.sc.keep)
    dt.sc.v[,purchase_fuel := ifelse(purchase_fuel %in% c('Electric','Plug-in Hybrid'), 'Broad EV', purchase_fuel)]
    dt.sc.v[,sc_fuel := ifelse(sc_fuel %in% c('Electric','Plug-in Hybrid'), 'Broad EV', sc_fuel)]
  } else
  {
    dt.sc.v <- copy(dt.sc.keep)
  }
  dt.ct.all <- dt.sc.v[,.(n=.N),by=.(model_year,purchase_fuel,sc_fuel)]
  dt.ct.agg <- dt.sc.v[,.(n_fuel=.N),by=.(model_year,purchase_fuel)]
  dt.ct.agg.noblank <- dt.sc.v[sc_fuel != '',.(n_fuel_spec=.N),by=.(model_year,purchase_fuel)]
  dt.ct.all <- merge(dt.ct.all,dt.ct.agg,by=c('model_year','purchase_fuel'))
  dt.ct.all <- merge(dt.ct.all,dt.ct.agg.noblank,by=c('model_year','purchase_fuel'))
  
  # get percent of each combination
  dt.ct.all[,`:=`(pct = n/n_fuel,
                  pct_spec = n/n_fuel_spec)]
  
  # add dt.ct.all to dictionary
  lst.sc.share[[v]] <- copy(dt.ct.all)
}

dt.combined <- lst.sc.share$combined
dt.separated <- lst.sc.share$sep

## save sc fueltype moments ----
# save only a subset of these moments
dt.subset.save <- dt.combined[purchase_fuel %in% c('Broad EV','Gas','Hybrid') & sc_fuel %in% c('Broad EV','Gas','Hybrid')]
write.csv(dt.subset.save,paste0(str.save,'sc_fuel_moments_combined_ev+phev.csv'),row.names=F)

## get fueltype sc moments - broad EV hybrid & gas ----
# with and without EV/PHEV broken out
lst.sc.share <- list()
for (v in c('combined','sep'))
{
  if (v == 'combined')
  {
    dt.sc.v <- copy(dt.sc.keep)
    dt.sc.v[,purchase_fuel := ifelse(purchase_fuel %in% c('Electric','Plug-in Hybrid', 'Hybrid'), 'Broad EV Hybrid', purchase_fuel)]
    dt.sc.v[,sc_fuel := ifelse(sc_fuel %in% c('Electric','Plug-in Hybrid', 'Hybrid'), 'Broad EV Hybrid', sc_fuel)]
  } else
  {
    dt.sc.v <- copy(dt.sc.keep)
  }
  dt.ct.all <- dt.sc.v[,.(n=.N),by=.(model_year,purchase_fuel,sc_fuel)]
  dt.ct.agg <- dt.sc.v[,.(n_fuel=.N),by=.(model_year,purchase_fuel)]
  dt.ct.agg.noblank <- dt.sc.v[sc_fuel != '',.(n_fuel_spec=.N),by=.(model_year,purchase_fuel)]
  dt.ct.all <- merge(dt.ct.all,dt.ct.agg,by=c('model_year','purchase_fuel'))
  dt.ct.all <- merge(dt.ct.all,dt.ct.agg.noblank,by=c('model_year','purchase_fuel'))
  
  # get percent of each combination
  dt.ct.all[,`:=`(pct = n/n_fuel,
                  pct_spec = n/n_fuel_spec)]
  
  # add dt.ct.all to dictionary
  lst.sc.share[[v]] <- copy(dt.ct.all)
}

dt.combined <- lst.sc.share$combined
dt.separated <- lst.sc.share$sep

## save sc fueltype moments ----
# save only a subset of these moments

dt.subset.save <- dt.combined[purchase_fuel %in% c('Broad EV Hybrid','Gas') & sc_fuel %in% c('Broad EV Hybrid','Gas')]
write.csv(dt.subset.save,paste0('sc_fuel_moments_combined_ev+hybrid.csv'),row.names=F)

## appending and removing duplicates 

df_ev_phev_hybrid <- read.csv("sc_fuel_moments_combined_ev+hybrid.csv")

# Read the second CSV file into a data frame
df_ev_phev <- read.csv("sc_fuel_moments_combined_ev+phev.csv")

# Combine the two data frames row-wise
combined_df <- rbind(df_ev_phev_hybrid, df_ev_phev)

# Ordering year variable 
combined_df$year <- factor(combined_df$year, levels = c("2017", "2018", "2019", "2020", "2021", "2022", "2023"))
combined_df <- combined_df[order(combined_df$year), ]

# Dropping duplicates 
df_unique <- combined_df %>%
  distinct(year, sc_fuel, purchase_fuel, .keep_all = TRUE)

df_unique <- write.csv('sc_fuel_moments_combined.csv')

# use VIN data to get broader set of attributes for sc ----
## set up VIN data ----
# merge vehicle-level data
dt.vin.data <- dt.vin.data[year %in% c(2017:2022)]
dt.vin.data <- unique(dt.vin.data[,.(count=sum(count)),by=.(vin_pattern,make,model,year,trim,style,vehicletype,bodytype,car,drivetype,fueltype,length,height,width,wheelbase,curbwt,displ,cylinders,city_mpg2008,hwy_mpg2008,combined_mpg2008,msrp,max_hpVD,doors,class)])

# merge additional vin decoder data
# vin_id comes from decoder
dt.vin.data <- merge(dt.vin.data,
                     dt.decoder,
                     by=c('vin_pattern','make','model','year','trim','style'),
                     all.x=T)
setnames(dt.vin.data,'gross_vehicle_weight_rating','gvwr')
# replace width with width_no_mirrors where it exists and is different
dt.vin.data[!is.na(width_no_mirrors) & width_no_mirrors < width & width_no_mirrors != 0,
            width:=width_no_mirrors]

# merge in EPA electric range
# note: this needs to happen before consolidation because of some ranges that differ at bodytype level
dt.vin.data <- merge(dt.vin.data,
                     dt.epa.ev,
                     by=c('make','model','year','trim','style','bodytype','drivetype','fueltype'),
                     all.x=T)
# fill in range for 2022 vehicles
dt.vin.data[year == 2022 & make == 'Bmw' & model == 'I4',
            range_elec:=ifelse(trim == 'M50 Gran Coupe',mean(c(270,227)),
                               ifelse(trim == 'eDrive40 Gran Coupe',mean(c(301,282)),
                                      NA))]
dt.vin.data[year == 2022 & make == 'Bmw' & model == 'Ix',
            range_elec:=mean(c(324,315,305))]
dt.vin.data[year == 2022 & make == 'Gmc' & model == 'Hummer-Ev',range_elec:=329]
dt.vin.data[year == 2022 & make == 'Hyundai' & model == 'Ioniq-5',
            range_elec:=ifelse(drivetype == 'Rwd',mean(c(220,303)),256)]
dt.vin.data[year == 2022 & make == 'Jaguar' & model == 'I-Pace',range_elec:=253]
dt.vin.data[year == 2022 & make == 'Karma' & model == 'Gs-6',range_elec:=61]
dt.vin.data[year == 2022 & make == 'Lucid' & model == 'Air',
            range_elec:=ifelse(trim== 'Dream Edition Performance',mean(c(471,451)),
                               ifelse(trim == 'Dream Edition Range',mean(c(520,481)),
                                      NA))]
n_missing <- nrow(dt.vin.data[is.na(range_elec) & fueltype %in% c('L','I') & year <= 2022])
if(n_missing > 0)
{
  print(paste(n_missing, 'EV/PHEV observations appear to be missing electric range'))
} # missing 2021 Karma Revero
dt.vin.data[is.na(range_elec),range_elec:=0]

# merge with battery size
dt.vin.data <- merge(dt.vin.data,
                     dt.battery[,!c('X'),with=F],
                     by=c('make','model','year','trim','style','vehicletype','bodytype','drivetype','fueltype'),
                     all.x=T)
n_missing <- nrow(dt.vin.data[is.na(batterysize) & fueltype %in% c('L','I') & year >= 2014])
if(n_missing > 0)
{
  print(paste(n_missing, 'EV/PHEV observations appear to be missing battery size'))
} 
dt.vin.data[is.na(batterysize) & !(fueltype %in% c('L','I')),batterysize:=0]

# merge with missing fuel economy
# ices
dt.vin.data <- merge(dt.vin.data,
                     dt.missing.fe,
                     by=c('make','model','year','trim','drivetype','fueltype','style','bodytype'),
                     all.x=T)

# deal with columns that get added by both the ICE data and EV/PHEV data
dt.vin.data[,`:=`(city08 = ifelse(!is.na(city08.x),city08.x,city08.y),
                  cityA08 = ifelse(!is.na(cityA08.x),cityA08.x,cityA08.y),
                  highway08 = ifelse(!is.na(highway08.x),highway08.x,highway08.y),
                  highwayA08 = ifelse(!is.na(highwayA08.x),highwayA08.x,highwayA08.y),
                  UCity = ifelse(!is.na(UCity.x),UCity.x,UCity.y),
                  UHighway = ifelse(!is.na(UHighway.x),UHighway.x,UHighway.y),
                  UCityA = ifelse(!is.na(UCityA.x),UCityA.x,UCityA.y),
                  UHighwayA = ifelse(!is.na(UHighwayA.x),UHighwayA.x,UHighwayA.y))]
vec.coldrop <- colnames(dt.vin.data)
vec.coldrop <- grep("\\.",vec.coldrop,value=TRUE)
dt.vin.data <- dt.vin.data[,!vec.coldrop,with=F]
# replace original mpg with EPA mpg
# note: there are some differences between vin decoder mpg and EPA mpg
dt.vin.data[!is.na(comb08),combined_mpg2008:=comb08]
dt.vin.data[!is.na(combA08) & combA08 != 0,mpg_blend:=combA08]
dt.vin.data <- dt.vin.data[,!c('comb08','combA08'),with=F]

# merge with EPA vehicle class
dt.vin.data <- merge(dt.vin.data,dt.vclass[,.(make,model,year,drivetype,trim,bodytype,VClass)],
                     by=c('make','model','year','drivetype','trim','bodytype'),
                     all.x=T)
# check that no observations got duplicated in a bad merge
by_vec <- unique(c('vin_id'))
dt.test <- dt.vin.data[!is.na(vin_id),.(ct=.N),by=by_vec][ct > 1]
if(nrow(dt.test) > 1)
{
  print('Warning: vin_ids got duplicated in a merge')
}

# consolidate vehicles to make-model-year-fueltype level
agg_level_main <- c('make','model','year','fueltype')
agg_level_trim <- c('make','model','trim','year','fueltype')

consolidate_vehicles <- function(dt.vin.data,agg_level)
{
  # get detail-level unique vehicles (aggregated across states)
  dt.agg.detail <- dt.vin.data[,.(count=sum(count)),
                               by=.(make,model,year,trim,style,vehicletype,bodytype,car,drivetype,fueltype,length,height,width,wheelbase,curbwt,
                                    displ,cylinders,doors,msrp,combined_mpg2008,city_mpg2008,hwy_mpg2008,max_hp,gvwr,
                                    range_elec,batterysize,
                                    mpg_gas,mpg_elec,mpg_blend,cityUF,highwayUF,combinedUF,city08,cityA08,highway08,highwayA08,
                                    UCity,UHighway,UCityA,UHighwayA,VClass)]
  dt.agg.detail[,detail_id:=1:nrow(dt.agg.detail)]
  # get the most common configuration at the trim-fuel-drivetype level
  dt.agg.common <- dt.agg.detail[,.(max_count=max(count),agg_count=sum(count)),
                                 by=agg_level]
  dt.agg.common[,agg_id:=1:nrow(dt.agg.common)]
  
  dt.agg.detail <- merge(dt.agg.detail,dt.agg.common,
                         by=agg_level,
                         all.x=T)
  # check for ties
  dt.agg.detail.ct <- dt.agg.detail[count == max_count]
  dt.tie <- dt.agg.detail.ct[,.N,by=c(agg_level,'agg_id','max_count','agg_count')]
  dt.tie <- dt.tie[N > 1]
  n_ties = nrow(dt.tie[year > 2013])
  str.ties = paste0("There are a total of ",n_ties," configurations with detail ties for 2014+ models.")
  print(str.ties)
  
  # tie break randomly
  dt.detail.keep <- dt.agg.detail.ct[!(agg_id %in% dt.tie$agg_id)]
  for (a_id in dt.tie$agg_id)
  {
    dt.keep <- head(dt.agg.detail.ct[agg_id == a_id],1)
    dt.detail.keep <- rbind(dt.detail.keep,dt.keep)
  }
  return(dt.detail.keep)
}
dt.detail.keep <- consolidate_vehicles(dt.vin.data,agg_level_main)
dt.detail.trim <- consolidate_vehicles(dt.vin.data,agg_level_trim)

## set up sc choice data for merge ----
# clean first choice data, merge with vin data
# remove column make from model name in dt.sc.keep
dt.sc.keep[make != 'Ram',model:=trimws(str_replace(model,make,''))]
# update trim, needed for Bmw and Mercedes
dt.sc.keep[,trim:=trimws(str_replace(trim,make,''))]
# rename Mercedes
dt.sc.keep[make == 'Mercedes-Benz',make:='Mercedes']
# replace model_inmoment with trim for Bmw, Mercedes
dt.sc.keep[make %in% c('BMW'),model:=trim]
# fix one weird fuel
dt.sc.keep[model == 'S60 Recharge Plug-In Hybrid',purchase_fuel:= 'Plug-in Hybrid']
# drop 'NSI' from model names
dt.sc.keep[,model:=str_replace(model,' NSI','')]
# make capitalization generally consistent with vin data
dt.sc.keep[,make_inmoment:=str_to_title(make)]
dt.sc.keep[,model_inmoment:=str_to_title(model)]
# make fueltype consistent with vin data
dt.sc.keep[,fueltype:=ifelse(purchase_fuel == 'Gas','G',
                             ifelse(purchase_fuel == 'Flex Fuel','F',
                                    ifelse(purchase_fuel == 'Diesel','D',
                                           ifelse(purchase_fuel == 'Plug-in Hybrid','I',
                                                  ifelse(purchase_fuel == 'Electric','L',
                                                         ifelse(purchase_fuel == 'Hybrid','Y',
                                                                ifelse(purchase_fuel == 'Hydrogen','H',
                                                                       ifelse(purchase_fuel == 'Natural Gas','N',NA))))))))]

# format second-choice model/fuel too
dt.sc.keep[,.(sc_model,sc_make,sc_trim)]
dt.sc.keep[!(sc_make %in% c('Ram','No Other Considered')),sc_model:=trimws(str_replace(sc_model,sc_make,''))]
# update trim, needed for Bmw and Mercedes
dt.sc.keep[,trim:=trimws(str_replace(sc_trim,make,''))]
# rename Mercedes
dt.sc.keep[sc_make == 'Mercedes-Benz',sc_make:='Mercedes']
# replace model_inmoment with trim for Bmw, Mercedes
dt.sc.keep[sc_make %in% c('BMW'),sc_model:=sc_trim]
# drop 'NSI' from model names
dt.sc.keep[,sc_model:=str_replace(sc_model,' NSI','')]
# make capitalization generally consistent with vin data
dt.sc.keep[,sc_make_inmoment:=str_to_title(sc_make)]
dt.sc.keep[,sc_model_inmoment:=str_to_title(sc_model)]
dt.sc.keep[,sc_year:=as.numeric(sc_year)]
# make fueltype consistent with vin data
dt.sc.keep[,sc_fueltype:=ifelse(sc_fuel == 'Gas','G',
                                ifelse(sc_fuel == 'Flex Fuel','F',
                                       ifelse(sc_fuel == 'Diesel','D',
                                              ifelse(sc_fuel == 'Plug-in Hybrid','I',
                                                     ifelse(sc_fuel == 'Electric','L',
                                                            ifelse(sc_fuel == 'Hybrid','Y',
                                                                   ifelse(sc_fuel == 'Hydrogen','H',
                                                                          ifelse(sc_fuel == 'Natural Gas','N',NA))))))))]

# get all vehicles from dt.sc.keep
dt.mm <- unique(dt.sc.keep[,.(model_year,make_inmoment,model_inmoment,fueltype)])
dt.sc.mm <- unique(dt.sc.keep[,.(sc_year,sc_make_inmoment,sc_model_inmoment,sc_fueltype)])
setnames(dt.sc.mm,c('sc_year','sc_model_inmoment','sc_make_inmoment','sc_fueltype'),
         c('model_year','model_inmoment','make_inmoment','fueltype'))
dt.mm <- unique(rbind(dt.mm,dt.sc.mm))
dt.mm <- dt.mm[!is.na(model_year) & model_year %in% c(2017:2023)]
dt.mm[,id_inmoment:=1:nrow(dt.mm)]
dt.sc.keep <- merge(dt.sc.keep,dt.mm,by=c('model_year','make_inmoment','model_inmoment','fueltype'),all.x=T)

dt.sc.mm <- copy(dt.mm)
setnames(dt.sc.mm,c('id_inmoment','model_year','make_inmoment','model_inmoment','fueltype'),
         c('sc_id_inmoment','sc_year','sc_make_inmoment','sc_model_inmoment','sc_fueltype'))
dt.sc.keep <- merge(dt.sc.keep,dt.sc.mm,by=c('sc_year','sc_make_inmoment','sc_model_inmoment','sc_fueltype'),all.x=T)
## clean crosswalk data ----
dt.crosswalk[make_inmoment == 'Rivian',model_inmoment:=str_replace_all(model_inmoment,'Rivian ','')]
dt.crosswalk[model_inmoment == 'S60 Recharge Plug-In Hybrid',fueltype:='I']
dt.crosswalk[,id_cw:=1:nrow(dt.crosswalk)]

setnames(dt.crosswalk,'vmy','model_year')
## merge sc data with crosswalk ----
dt.mm.merge <- merge(dt.mm,dt.crosswalk,
                     by=c('make_inmoment','model_inmoment','model_year','fueltype'),
                     all.x=T)

# dt.fail <- dt.mm.merge[is.na(source) & !(make_inmoment %in% c('Mini')) ]
# deal with Mini at some point?
dt.mm.merge <- dt.mm.merge[!(make_inmoment == 'Mini')]
dt.mm.merge <- dt.mm.merge[!is.na(source) & source != 'Only Inmoment']

# check that we didn't lose anything else
dt.crosswalk[!(source %in% c('Only Inmoment','Only VIN')) & !(make_inmoment == 'Mini') & !(id_cw %in% dt.mm.merge$id_cw)]


## merge with vin data ----
# for makes that look at model level
setnames(dt.detail.keep,c('year','make','model'),
         c('model_year','make_vin','model_vin'))
dt.mm.merge1 <- dt.mm.merge[!(make_inmoment %in% c('Bmw','Mercedes'))]
dt.mm.merge1 <- merge(dt.mm.merge1,dt.detail.keep[,!c('count','detail_id','max_count','agg_count','agg_id')],
                      by=c('model_year','make_vin','model_vin','fueltype'),
                      all.x=T)
# for makes that look at model-trim level
dt.detail.trim[,model:=paste(model,trim)]
setnames(dt.detail.trim,c('year','make','model'),
         c('model_year','make_vin','model_vin'))
# drop any columns from dt.vin.merge that are not in dt.detail.keep
vec.keep <- (intersect(names(dt.detail.trim),names(dt.detail.keep)))
dt.detail.trim <- dt.detail.trim[,vec.keep,with=F]
dt.mm.merge2 <- dt.mm.merge[(make_inmoment %in% c('Bmw','Mercedes'))]
dt.mm.merge2 <- merge(dt.mm.merge2,dt.detail.trim[,!c('count','detail_id','max_count','agg_count','agg_id')],
                      by=c('model_year','make_vin','model_vin','fueltype'),
                      all.x=T)

# combine
dt.mm.merge.attributes <- rbind(dt.mm.merge1,dt.mm.merge2)
dt.mm.merge.attributes <- dt.mm.merge.attributes[,.(id_inmoment,fueltype,bodytype,drivetype,wheelbase,curbwt,displ,cylinders,doors,msrp,combined_mpg2008,max_hp,range_elec,batterysize,combinedUF,mpg_elec,mpg_gas)]

# merge back with sc choice data
dt.sc.keep <- merge(dt.sc.keep,
                    dt.mm.merge.attributes,
                    by=c('id_inmoment','fueltype'),
                    all.x=T)

# put 'sc_' in front of all columns in dt.mm.merge.attributes
setnames(dt.mm.merge.attributes,paste0('sc_',names(dt.mm.merge.attributes)))

# merge back with sc choice data
dt.sc.keep <- merge(dt.sc.keep,
                    dt.mm.merge.attributes,
                    by=c('sc_id_inmoment','sc_fueltype'),
                    all.x=T)

## calculate cov moments ----
### calculate cov(weight 1, weight 2) ----
cor(dt.sc.keep[!is.na(curbwt) & !is.na(sc_curbwt),curbwt],
    dt.sc.keep[!is.na(curbwt) & !is.na(sc_curbwt),sc_curbwt])
cov_wt <- cov(dt.sc.keep[!is.na(curbwt) & !is.na(sc_curbwt),curbwt/1000],
    dt.sc.keep[!is.na(curbwt) & !is.na(sc_curbwt),sc_curbwt/1000])
n_wt <- nrow(dt.sc.keep[!is.na(curbwt) & !is.na(sc_curbwt)])
### calculate cov(wheelbase 1, wheelbase 2) ----
cov_wheelbase <- cov(dt.sc.keep[!is.na(wheelbase) & !is.na(sc_wheelbase),wheelbase/100],
              dt.sc.keep[!is.na(wheelbase) & !is.na(sc_wheelbase),sc_wheelbase/100])
n_wheelbase <- nrow(dt.sc.keep[!is.na(wheelbase) & !is.na(sc_wheelbase)])

### calculate cost per mile at state level----
lst.dt.energy <- list(dt.gas,dt.elec,dt.diesel)
# create dataframe of state names and state abbreviations
dt.state <- data.table(state = state.name,
                       state_abb = state.abb)
#dt.sc.copy <- copy(dt.sc.keep)
for(dt in lst.dt.energy)
{
  dt <- merge(dt,dt.state,by=c('state'),all.x=T)
  dt[state == 'U.S.',state_abb:='US']
  setnames(dt,c('year','state','state_abb'),c('purchase_year','state_full','state'))
  dt.sc.keep <- merge(dt.sc.keep,dt[,!c('state_full'),with=F],
                      by=c('state','purchase_year'))
  vec_colnames <- colnames(dt)
  energy_colname <- vec_colnames[(substr(vec_colnames,1,6) == 'dollar') | (substr(vec_colnames,1,4) == 'cent')]
  setnames(dt,energy_colname,paste0(energy_colname,'_US'))
  dt.sc.keep <- merge(dt.sc.keep,
                    dt[state == 'US',!c('state','state_full'),with=F],
                    by=c('purchase_year'))
}

# gasoline, hybrid
dt.sc.keep[fueltype %in% c('G','Y'),
          `:=`(dollar_per_mile=dollar_per_gal_gas /combined_mpg2008,
               dollar_per_mile_US=dollar_per_gal_gas_US/combined_mpg2008)]
dt.sc.keep[sc_fueltype %in% c('G','Y'),
           `:=`(sc_dollar_per_mile=dollar_per_gal_gas /sc_combined_mpg2008,
                sc_dollar_per_mile_US=dollar_per_gal_gas_US/sc_combined_mpg2008)]

# electricity
# EPA fueleconomy.gov says 33.7 kWh per gallon
dt.sc.keep[fueltype == 'L',
          `:=`(dollar_per_mile= cent_per_kwh / 100 * 33.7 /combined_mpg2008,
               dollar_per_mile_US = cent_per_kwh_US / 100 * 33.7/combined_mpg2008)]
dt.sc.keep[sc_fueltype == 'L',
           `:=`(sc_dollar_per_mile= cent_per_kwh / 100 * 33.7 /sc_combined_mpg2008,
                sc_dollar_per_mile_US = cent_per_kwh_US / 100 * 33.7/sc_combined_mpg2008)]

# phev (assume mileage split)
# using share of electric miles from EPA combined utility factor
# previously used split assumed by AFDC model (https://afdc.energy.gov/vehicles/electric_emissions_sources.html)
dt.sc.keep[fueltype == 'I',
          `:=`(dollar_per_mile_elec= cent_per_kwh /100 * 33.7 / mpg_elec,
               dollar_per_mile_elec_US = cent_per_kwh_US/100 *33.7 / mpg_elec)]
dt.sc.keep[fueltype == 'I',
          `:=`(dollar_per_mile_gas=dollar_per_gal_gas / mpg_gas,
               dollar_per_mile_gas_US=dollar_per_gal_gas_US / mpg_gas)]
dt.sc.keep[fueltype == 'I',
          `:=`(dollar_per_mile=combinedUF*dollar_per_mile_elec +
                 (1-combinedUF) * dollar_per_mile_gas,
               dollar_per_mile_US=combinedUF*dollar_per_mile_elec_US +
                 (1-combinedUF) * dollar_per_mile_gas_US)]

dt.sc.keep[sc_fueltype == 'I',
           `:=`(sc_dollar_per_mile_elec= cent_per_kwh /100 * 33.7 / sc_mpg_elec,
                sc_dollar_per_mile_elec_US = cent_per_kwh_US/100 *33.7 / sc_mpg_elec)]
dt.sc.keep[sc_fueltype == 'I',
           `:=`(sc_dollar_per_mile_gas=dollar_per_gal_gas / sc_mpg_gas,
                sc_dollar_per_mile_gas_US=dollar_per_gal_gas_US / sc_mpg_gas)]
dt.sc.keep[sc_fueltype == 'I',
           `:=`(sc_dollar_per_mile=sc_combinedUF*sc_dollar_per_mile_elec +
                  (1-sc_combinedUF) * sc_dollar_per_mile_gas,
                sc_dollar_per_mile_US=sc_combinedUF*dollar_per_mile_elec_US +
                  (1-sc_combinedUF) * sc_dollar_per_mile_gas_US)]

# diesel
dt.sc.keep[fueltype == 'D',
          `:=`(dollar_per_mile= dollar_per_gal_diesel / combined_mpg2008,
               dollar_per_mile_US = dollar_per_gal_diesel_US / combined_mpg2008)]

dt.sc.keep[sc_fueltype == 'D',
           `:=`(sc_dollar_per_mile= dollar_per_gal_diesel / sc_combined_mpg2008,
                sc_dollar_per_mile_US = dollar_per_gal_diesel_US / sc_combined_mpg2008)]

# flex fuel (assume gasoline prices)
dt.sc.keep[fueltype == 'F',
          `:=`(dollar_per_mile= dollar_per_gal_gas / combined_mpg2008,
               dollar_per_mile_US = dollar_per_gal_gas_US / combined_mpg2008)]

dt.sc.keep[sc_fueltype == 'F',
           `:=`(sc_dollar_per_mile= dollar_per_gal_gas / sc_combined_mpg2008,
                sc_dollar_per_mile_US = dollar_per_gal_gas_US / sc_combined_mpg2008)]

# note: missing some 2022 hybrids
unique(dt.sc.keep[is.na(dollar_per_mile) & !is.na(wheelbase) & fueltype == 'Y',model_year])
unique(dt.sc.keep[is.na(sc_dollar_per_mile) & !is.na(sc_wheelbase),.(sc_make,sc_model,sc_year)])

# re-scale dollar per mile
dt.sc.keep[,`:=`(dollar_per_mile=dollar_per_mile*10,
                 sc_dollar_per_mile=sc_dollar_per_mile*10)]

cor(dt.sc.keep[!is.na(dollar_per_mile) & !is.na(sc_dollar_per_mile),dollar_per_mile],
                 dt.sc.keep[!is.na(dollar_per_mile) & !is.na(sc_dollar_per_mile),sc_dollar_per_mile])
cov_cost <- cov(dt.sc.keep[!is.na(dollar_per_mile) & !is.na(sc_dollar_per_mile),dollar_per_mile],
              dt.sc.keep[!is.na(dollar_per_mile) & !is.na(sc_dollar_per_mile),sc_dollar_per_mile])
n_cost <- nrow(dt.sc.keep[!is.na(dollar_per_mile) & !is.na(sc_dollar_per_mile)])

### save cov moments ----
dt.moment.cov <- data.table(char = c('curbwt','wheelbase','dollar_per_mile'),
                            val = c(cov_wt,cov_wheelbase,cov_cost),
                            n = c(n_wt,n_wheelbase,n_cost))
write.csv(dt.moment.cov,paste0(str.save,'sc_cov.csv'),row.names=F)
## calculate discretized quantile moments ----

assign_quantiles <- function(dt.sc,dt.quantiles,col,n_quantiles)
{
  dt.sc[,col_quantile:=2*n_quantiles]
  dt.sc[,sc_col_quantile:=2*n_quantiles]
  
  for (yr in unique(dt.sc$model_year))
  {
    # use 2021 quantiles for 2022
    if(yr < 2022)
    {
      vec.quantile <- dt.quantiles[year == yr,quant]
    } else
    {
      vec.quantile <- dt.quantiles[year == 2021,quant]
    }
    # reset lower bound to 0,upper bound to 1000
    vec.quantile[1] <- 0
    vec.quantile[length(vec.quantile)] <- 1000
    dt.sc[model_year == yr,col_quantile:=cut(get(col),vec.quantile,include.lowest=T,labels=F)]
    dt.sc[model_year == yr,sc_col_quantile:=cut(get(paste0('sc_',col)),vec.quantile,include.lowest=T,labels=F)]
  }
  setnames(dt.sc,c('col_quantile'),paste0('quantile_',col))
  setnames(dt.sc,c('sc_col_quantile'),paste0('sc_quantile_',col))
  return(dt.sc)
}
### dollar per mile ----

# re-scale quantiles to match dollar_per_mile
dt.dpm.quantiles[,quant:=quant*10]
dt.sc.keep <- assign_quantiles(dt.sc.keep,dt.dpm.quantiles,'dollar_per_mile',5)

# get p(quantile sc | quantile fc)
dt.sc.dpm.quantile <- dt.sc.keep[!is.na(dollar_per_mile) & !is.na(sc_dollar_per_mile)]
dt.ct.all <- dt.sc.dpm.quantile[,.(n=.N),by=.(quantile_dollar_per_mile,sc_quantile_dollar_per_mile)]
dt.ct.agg <- dt.sc.dpm.quantile[,.(n_quant=.N),by=.(quantile_dollar_per_mile)]
dt.ct.all <- merge(dt.ct.all,dt.ct.agg,by=c('quantile_dollar_per_mile'))
dt.ct.all[,pct:=n/n_quant]
dt.ct.all <- dt.ct.all[order(quantile_dollar_per_mile,sc_quantile_dollar_per_mile)]

write.csv(dt.ct.all,paste0(str.save,'sc_dollar_per_mile_quantiles.csv'),row.names=F)

### wheelbase ----
dt.sc.keep <- assign_quantiles(dt.sc.keep,dt.wb.quantiles,'wheelbase',3)
#dt.sc.keep[,.(wheelbase,quantile_wheelbase)]

# get p(quantile sc | quantile fc)
dt.sc.wb.quantile <- dt.sc.keep[!is.na(wheelbase) & !is.na(sc_wheelbase)]
dt.ct.all <- dt.sc.wb.quantile[,.(n=.N),by=.(quantile_wheelbase,sc_quantile_wheelbase)]
dt.ct.agg <- dt.sc.wb.quantile[,.(n_quant=.N),by=.(quantile_wheelbase)]
dt.ct.all <- merge(dt.ct.all,dt.ct.agg,by=c('quantile_wheelbase'))
dt.ct.all[,pct:=n/n_quant]
dt.ct.all <- dt.ct.all[order(quantile_wheelbase,sc_quantile_wheelbase)]

write.csv(dt.ct.all,paste0(str.save,'sc_wb_quantiles.csv'),row.names=F)

# calculate bodytype moments ----
dt.sc.bodytype <- dt.sc.keep[!is.na(bodytype) & !is.na(sc_bodytype)]
dt.ct.all <- dt.sc.bodytype[,.(n=.N),by=.(bodytype,sc_bodytype)]
dt.ct.agg <- dt.sc.bodytype[,.(n_bodytype=.N),by=.(bodytype)]
dt.ct.all <- merge(dt.ct.all,dt.ct.agg,by=c('bodytype'))

# get percent of each combination
dt.ct.all[,`:=`(pct = n/n_bodytype)]
write.csv(dt.ct.all,paste0(str.save,'sc_bodytype.csv'),row.names=F)

