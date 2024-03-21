# prepare data for consolidation: merge with attributes that differ at more disaggregated level
prep_vin_data <- function(dt.vin.data,dt.decoder,dt.epa.ev,dt.fed.inc,dt.zev.credits,dt.missing.fe,dt.battery,dt.vin.wages,dt.vclass,geog='state')
{
  # drop missing years
  print(paste0("Dropping ",nrow(dt.vin.data[is.na(year)]), " rows due to missing years."))
  dt.vin.data <- dt.vin.data[!is.na(year)]
  
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
  n_missing <- nrow(dt.vin.data[is.na(range_elec) & fueltype %in% c('L','I') & year < 2022])
  if(n_missing > 0)
  {
    print(paste(n_missing, 'EV/PHEV observations appear to be missing electric range'))
  } # Missing 2011 Chevy Volt and 2021 Karma Revero
  dt.vin.data[is.na(range_elec),range_elec:=0]
  
  # merge with fed incentives (this is needs to happen before consolidation because some versions of consolidation aggregate drivetype/trim)
  dt.vin.data <- merge(dt.vin.data,
                       dt.fed.inc,
                       by=c('make','model','year','trim','drivetype','fueltype'),
                       all.x = T)
  dt.vin.data[is.na(Q1),`:=`(Q1 = 0, Q2 = 0, Q3 = 0, Q4 = 0)]
  dt.vin.data[,fed_tax_credit:= (Q1+Q2+Q3+Q4)/4]
  
  # merge with battery size (this also needs to happen before consolidation)
  dt.vin.data <- merge(dt.vin.data,
                       dt.battery[,!c('X'),with=F],
                       by=c('make','model','year','trim','style','vehicletype','bodytype','drivetype','fueltype'),
                       all.x=T)
  n_missing <- nrow(dt.vin.data[is.na(batterysize) & fueltype %in% c('L','I') & year >= 2014])
  if(n_missing > 0)
  {
    print(paste(n_missing, 'EV/PHEV observations appear to be missing battery size'))
  } # Missing 2015 Cadillac Elr because it's actually on hiatus?
  dt.vin.data[is.na(batterysize) & !(fueltype %in% c('L','I')),batterysize:=0]
  
  # merge with ZEV credits
  dt.vin.data <- merge(dt.vin.data,
                       dt.zev.credits,
                       by=c('make','model','year','trim','style','bodytype','drivetype','fueltype'),
                       all.x=T)
  dt.vin.data[is.na(zev_credits),zev_credits:=0]
  
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
  
  # merge with VIN-level wage data
  dt.vin.wages <- unique(dt.vin.wages[year %in% c(2014:2021),.(man_year,year,make,model,wages,man_loc,vin,vin_pattern)])
  dt.vin.data <- merge(dt.vin.data,
                       dt.vin.wages[,.(vin_pattern,wages)],
                       by=c('vin_pattern'),
                       all.x=T)
  
  # merge with EPA vehicle class
  dt.vin.data <- merge(dt.vin.data,dt.vclass[,.(make,model,year,drivetype,trim,bodytype,VClass)],
                       by=c('make','model','year','drivetype','trim','bodytype'),
                       all.x=T)
  # check that no observations got duplicated in a bad merge
  by_vec <- unique(c('vin_id',geog,'state'))
  dt.test <- dt.vin.data[!is.na(vin_id),.(ct=.N),by=by_vec][ct > 1]
  if(nrow(dt.test) > 1)
  {
    print('Warning: vin_ids got duplicated in a merge')
  }
  
  return(dt.vin.data)
}

# choose top vehicles
consolidate_vehicles <- function(dt.vin.data,dt.st.inc,dt.sib.ev,dt.sib.gas,agg_level,geog='state')
{
  # get detail-level unique vehicles (aggregated across states)
  dt.agg.detail <- dt.vin.data[,.(count=sum(count)),
                               by=.(make,model,year,trim,style,vehicletype,bodytype,car,hybrid,drivetype,fueltype,length,height,width,wheelbase,curbwt,
                                    displ,cylinders,doors,msrp,combined_mpg2008,city_mpg2008,hwy_mpg2008,max_hp,gvwr,
                                    range_elec,fed_tax_credit,Q1,Q2,Q3,Q4,zev_credits,batterysize,
                                    mpg_gas,mpg_elec,mpg_blend,cityUF,highwayUF,combinedUF,city08,cityA08,highway08,highwayA08,
                                    UCity,UHighway,UCityA,UHighwayA,wages,VClass)]
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
  
  # rename columns (keep agg_id)
  dt.detail.keep <- dt.detail.keep[,!c('count','max_count','detail_id','agg_count')]
  #setnames(dt.detail.keep,'agg_count','count')
  # rename agg_id to product_ids for pyblp reasons
  setnames(dt.detail.keep,'agg_id','product_ids')
  
  # merge back with [geog] sales
  dt.geog.ct <- dt.vin.data[,.(agg_count=sum(count)),
                             by=c(unique(c(geog,'state')),agg_level)]
  dt.detail.keep <- merge(dt.geog.ct,dt.detail.keep,
                          by=agg_level)
  
  # merge state sales with state incentives (currently at model-year-fueltype)
  dt.detail.keep <- merge(dt.detail.keep,
                          dt.st.inc,
                          by=c('state','year','make','model','fueltype'),
                          all.x=T)
  dt.detail.keep[is.na(rebate),rebate:=0]
  
  # merge with sibling info (model-year-fueltype)
  dt.detail.keep <- merge(dt.detail.keep,
                          dt.sib.ev,
                          by=c('year','make','model','fueltype'),
                          all.x=T)
  dt.detail.keep <- merge(dt.detail.keep,
                          dt.sib.gas,
                          by=c('year','make','model','fueltype'),
                          all.x=T)
  # fill in missing values
  dt.detail.keep[is.na(has_sibling_BEV) & fueltype == 'G',`:=`(will_get_sibling_BEV=0,has_sibling_BEV=0,had_sibling_BEV=0,BEV_sibling_entry_year=0)]
  dt.detail.keep[is.na(has_sibling_PHEV) & fueltype == 'G',`:=`(will_get_sibling_PHEV=0,has_sibling_PHEV=0,had_sibling_PHEV=0,PHEV_sibling_entry_year=0)]
  dt.detail.keep[is.na(sibling) & fueltype %in% c('L','I'),sibling:=0]
  return(dt.detail.keep)
}

# cleaning functions
# flag for medium-heavy duty vehs
add_mhd_flag <- function(dt.detail.keep)
{
  # add flag to models that are technically medium/heavy duty vehicles (to the best of my knowledge)
  # checked against VIN decoder GVW (8500 cutoff)
  #dt.trucks <- unique(dt.detail.keep[vehicletype == "Truck" & year >= 2014,.(make,model)])
  #dt.vans <- unique(dt.detail.keep[vehicletype == "Van" & year >= 2014,.(make,model)])
  dt.detail.keep[,mhd:=0]
  dt.detail.keep[make == 'Chevrolet' & model %in% c('Silverado-2500Hd','Silverado-3500Hd','Silverado-3500Hd-Cc',
                                                    'Express-Cutaway','Express-Passenger','Express-Cargo'),
                 mhd:=1]
  dt.detail.keep[make == 'Dodge' & model %in% c('Ram-Chassis-3500','Ram-Pickup-2500','Ram-Pickup-3500',
                                                'Promaster-Cab-Chassis','Promaster-Cargo','Promaster-Cutaway-Chassis','Promaster-Window'),
                 mhd:=1]
  dt.detail.keep[make == 'Ford' & model %in% c('F-250-Super-Duty','F-350-Super-Duty',
                                               'E-Series-Cargo','E-Series-Chassis','E-Series-Wagon','Transit-Cargo','Transit-Chassis-Cab','Transit-Crew','Transit-Cutaway','Transit-Passenger'),
                 mhd:=1]
  dt.detail.keep[make =='Gmc' & model %in% c('Sierra-2500Hd','Sierra-3500Hd','Sierra-3500Hd-Cc',
                                             'Savana-Cutaway','Savana-Passenger','Savana-Cargo'),
                 mhd:=1]
  dt.detail.keep[make =='Nissan' & model %in% c('Titan-Xd','Nv-Cargo','Nv-Passenger'),
                 mhd:=1]
  dt.detail.keep[make == 'Mercedes' & model %in% c('Sprinter-Cab-Chassis','Sprinter-Cargo','Sprinter-Crew','Sprinter-Passenger','Sprinter-Worker'),
                 mhd:=1]
  #unique(dt.detail.keep[mhd == 1,gvwr])
  #dt.detail.keep[mhd == 0 & gvwr > 8500 & year %in% c(2013:2020)]
  # there are some vehicles that have gvwr > 8500 but show up in fueleconomy.gov
  return(dt.detail.keep)
}

# process cpi data
process_cpi <- function(dt.cpi)
{
  dt.CPI.l <- data.table(melt(dt.CPI,id.vars = 'Year'))
  setnames(dt.CPI.l,c('Year','variable','value'),c('year','month','cpi'))
  dt.CPI.l <- dt.CPI.l[,.(cpi = mean(cpi)),by=c('year')]
  dt.CPI.l[,mult_2021:=dt.CPI.l[year == 2021,cpi]/cpi]
  return(dt.CPI.l)
}

# adjust incentives, prices
adjust_prices <- function(dt.detail.keep,dt.CPI.l)
{
  dt.detail.keep <- merge(dt.detail.keep,
                          dt.CPI.l[,.(year,mult_2021)],
                          by=c('year'))
  dt.detail.keep[,`:=`(msrp= msrp*mult_2021,fed_tax_credit=fed_tax_credit*mult_2021,
                       Q1=Q1*mult_2021,Q2=Q2*mult_2021,Q3=Q3*mult_2021,Q4=Q4*mult_2021,
                       rebate=rebate*mult_2021)]
  dt.detail.keep <- dt.detail.keep[,!c('mult_2021'),with=F]
  return(dt.detail.keep)
}

# add oems
generate_oems <- function(dt.detail.keep,dt.oem.data)
{
  dt.oems.unique <- unique(dt.oem.data[,.(oem)])
  dt.oems.unique[,firm_ids:=1:nrow(dt.oems.unique)]
  dt.oem.data <- merge(dt.oem.data,dt.oems.unique,by=c('oem'))
  
  dt.detail.keep <- merge(dt.detail.keep,dt.oem.data,
                          by=c('make'),all.x=T)
  return(dt.detail.keep)
}

# clean fuel data
clean_fuels <- function(dt.detail.keep)
{
  # fuel types come from Data One VIN Decoder Documentation
  dt.detail.keep[,fuel:=ifelse(fueltype == 'D','diesel',
                               ifelse(fueltype == 'L','electric',
                                      ifelse(fueltype == 'Y','hybrid',
                                             ifelse(fueltype == 'I','PHEV',
                                                    ifelse(fueltype == 'G','gasoline',
                                                           ifelse(fueltype == 'B', 'bio-diesel',
                                                                  ifelse(fueltype == 'F', 'flex fuel', 
                                                                         ifelse(fueltype == 'H', 'hydrogen',
                                                                                ifelse(fueltype == 'N', 'nat gas',NA)))))))))]
  #
  
  # fuel indicators
  dt.detail.keep[,electric:=ifelse(fuel == 'electric',1,0)]
  dt.detail.keep[,phev:=ifelse(fuel == 'PHEV',1,0)]
  dt.detail.keep[,hybrid:=ifelse(fuel == 'hybrid',1,0)]
  dt.detail.keep[,diesel:=ifelse(fuel == 'diesel',1,0)]
  dt.detail.keep[,non_electric:=ifelse(!(fuel %in% c('electric','PHEV')),1,0)]
  dt.detail.keep[,broad_ev:=electric + phev]
  return(dt.detail.keep)
}

# clean drivetype data
clean_drivetypes <- function(dt.detail.keep)
{
  # create indicators for vehicle drive types
  # using same grouping as Hannah's merge code
  dt.detail.keep[,drivetype_orig:=drivetype]
  dt.detail.keep[drivetype %in% c('4X4'),drivetype:='Awd']
  dt.detail.keep[drivetype %in% c('4X2','Fwd','Rwd'),drivetype:='2wd']
  return(dt.detail.keep)
}

# re-scale vars
rescale_vars <- function(dt.detail.keep)
{
  # re-scale prices to $1000s
  dt.detail.keep[,`:=`(msrp=msrp/1000, fed_tax_credit=fed_tax_credit/1000, rebate=rebate/1000)]
  
  # re-scale curbweight to 1000 lbs
  dt.detail.keep[,curbwt:=curbwt/1000]
  
  # re-scale electric range to 10s of miles
  dt.detail.keep[,range_elec:=range_elec/10]
  
  return(dt.detail.keep)
}

# differentiate EV and PHEV ranges
diff_ranges <- function(dt.detail.keep)
{
  dt.detail.keep[,EV_range_elec:=range_elec*electric]
  dt.detail.keep[,PHEV_range_elec:=range_elec*phev]
  return(dt.detail.keep)
}

# calculate vehicle performance variables
get_performance <- function(dt.detail.keep)
{
  # note: all NAs due to missing curbwt variable
  dt.detail.keep[,hp_weight:= max_hp/curbwt]
  dt.detail.keep[,log_hp_weight:=log(hp_weight)]
  dt.detail.keep[,log_hp:=log(max_hp)]
  return(dt.detail.keep)
}

# get census divisions
get_census_div <- function(dt.detail.keep)
{
  # add some state grouping for preferences
  dt.detail.keep[state %in% c('WASHINGTON','OREGON','ALASKA','HAWAII'),state_grp:='West']
  dt.detail.keep[state %in% c('CALIFORNIA'),state_grp:='California']
  dt.detail.keep[state %in% c('MONTANA','IDAHO','WYOMING','NEVADA','UTAH','COLORADO','ARIZONA','NEW MEXICO'),
                 state_grp:='Mountain']
  dt.detail.keep[state %in% c('NORTH DAKOTA','SOUTH DAKOTA', 'MINNESOTA','NEBRASKA','IOWA','KANSAS','MISSOURI'),
                 state_grp:= 'West North Central']
  dt.detail.keep[state %in% c('WISCONSIN','MICHIGAN','ILLINOIS','INDIANA','OHIO'),
                 state_grp:='East North Central']
  dt.detail.keep[state %in% c('LOUISIANA','ARKANSAS','OKLAHOMA','TEXAS'),state_grp:='West South Central']
  dt.detail.keep[state %in% c('KENTUCKY','TENNESSEE','MISSISSIPPI','ALABAMA'),state_grp:='East South Central']
  dt.detail.keep[state %in% c('WEST VIRGINIA','VIRGINIA','DELAWARE','DIST. OF COLUMBIA','MARYLAND','NORTH CAROLINA','SOUTH CAROLINA','GEORGIA','FLORIDA'),
                 state_grp:='South Atlantic']
  dt.detail.keep[state %in% c('NEW YORK','NEW JERSEY','PENNSYLVANIA'),state_grp:='Middle Atlantic']
  dt.detail.keep[state %in% c('VERMONT','NEW HAMPSHIRE','MAINE','MASSACHUSETTS','RHODE ISLAND','CONNECTICUT'),state_grp:='New England']
  dt.detail.keep[is.na(state_grp),state_grp:=state]
  return(dt.detail.keep)
}

# merge energy price
merge_energy_prices <- function(lst.energy,dt.sales)
{
  for(dt in lst.energy)
  {
    dt[,state:=toupper(state)]
    dt.sales <- merge(dt.sales,dt,by=c('state','year'))
    vec_colnames <- colnames(dt)
    energy_colname <- vec_colnames[(substr(vec_colnames,1,6) == 'dollar') | (substr(vec_colnames,1,4) == 'cent')]
    setnames(dt,energy_colname,paste0(energy_colname,'_US'))
    dt.sales <- merge(dt.sales,
                      dt[state == 'U.S.',!c('state'),with=F],
                      by=c('year'))
  }
  return(dt.sales)
}

# calculate dollar per mile
calculate_cost_per_mile <- function(dt.detail)
{
  # gasoline, hybrid
  dt.detail[fuel %in% c('gasoline','hybrid'),
            `:=`(dollar_per_mile=dollar_per_gal_gas /combined_mpg2008,
                 dollar_per_mile_US=dollar_per_gal_gas_US/combined_mpg2008)]
  # electricity
  # EPA fueleconomy.gov says 33.7 kWh per gallon
  dt.detail[fuel == 'electric',
            `:=`(dollar_per_mile= cent_per_kwh / 100 * 33.7 /combined_mpg2008,
                 dollar_per_mile_US = cent_per_kwh_US / 100 * 33.7/combined_mpg2008)]
  
  # phev (assume mileage split)
  # using share of electric miles from EPA combined utility factor
  # previously used split assumed by AFDC model (https://afdc.energy.gov/vehicles/electric_emissions_sources.html)
  dt.detail[fuel == 'PHEV',
            `:=`(dollar_per_mile_elec= cent_per_kwh /100 * 33.7 / mpg_elec,
                 dollar_per_mile_elec_US = cent_per_kwh_US/100 *33.7 / mpg_elec)]
  dt.detail[fuel == 'PHEV',
            `:=`(dollar_per_mile_gas=dollar_per_gal_gas / mpg_gas,
                 dollar_per_mile_gas_US=dollar_per_gal_gas_US / mpg_gas)]
  dt.detail[fuel == 'PHEV',
            `:=`(dollar_per_mile=combinedUF*dollar_per_mile_elec +
                   (1-combinedUF) * dollar_per_mile_gas,
                 dollar_per_mile_US=combinedUF*dollar_per_mile_elec_US +
                   (1-combinedUF) * dollar_per_mile_gas_US)]
  
  # diesel
  dt.detail[fuel == 'diesel',
            `:=`(dollar_per_mile= dollar_per_gal_diesel / combined_mpg2008,
                 dollar_per_mile_US = dollar_per_gal_diesel_US / combined_mpg2008)]
  
  # flex fuel (assume gasoline prices)
  dt.detail[fuel == 'flex fuel',
            `:=`(dollar_per_mile= dollar_per_gal_gas / combined_mpg2008,
                 dollar_per_mile_US = dollar_per_gal_gas_US / combined_mpg2008)]
  
  # ignore bio-diesel, hydrogen, natural gas?
  # dt.detail[,.(mean = mean(dollar_per_mile,na.rm=T)),by=c('fuel')]
  
  # drop energy cost columns besides dollar_per_mile and dollar_per_mile_US
  vec.colnames <- colnames(dt.detail)
  vec.colnames <- vec.colnames[!(substr(vec.colnames,1,6) == 'dollar' | substr(vec.colnames,1,4) == 'cent') | 
                                 vec.colnames %in% c('dollar_per_mile','dollar_per_mile_US')]
  dt.detail <- dt.detail[,vec.colnames,with=F]
  
  return(dt.detail)
}

# get EPA mpgs
calculate_EPA_mpg <- function(dt.detail.keep,city_shr)
{
  ### for gasoline, hybrid, diesel vehicles (single fuel ICEs)
  
  # for vehicles with city and highway data available
  # calculate UCity and UHighway
  # sticker city mpg = 1/(.003259 + 1.1805/un_adj_mpg)
  dt.detail.keep[is.na(UCity) & !is.na(city_mpg2008) & fuel %in% c('gasoline','hybrid','diesel'),
                 UCity:=1.1805/(1/city_mpg2008 - .003259)]
  dt.detail.keep[is.na(UHighway) & !is.na(hwy_mpg2008) & fuel %in% c('gasoline','hybrid','diesel'),
                 UHighway:=1.3466/(1/hwy_mpg2008 - .001376)]
  # calculate Umpg
  dt.detail.keep[fuel %in% c('gasoline','hybrid','diesel'),
                 Umpg:=1/(city_shr * 1/UCity + (1-city_shr)*1/UHighway)]
  
  # check that relationship seems reasonable
  #ggplot(dt.detail.keep[fuel %in% c('gasoline','hybrid','diesel')],aes(x=combined_mpg2008,y=Umpg))+geom_point()
  
  # remaining vehicles that have mpg but don't have city/highway
  dt.factor <- dt.detail.keep[,.(Umpg,combined_mpg2008,agg_count)]
  dt.factor[,factor:= Umpg/combined_mpg2008]
  Umult <- dt.factor[!is.na(Umpg),.(wtd_factor = sum(factor*agg_count)/sum(agg_count))]$wtd_factor
  dt.detail.keep[is.na(Umpg) & fuel %in% c('gasoline','hybrid','diesel') & !is.na(combined_mpg2008),
                 Umpg:=combined_mpg2008*Umult]
  
  
  ### for flex-fuel vehicles
  dt.detail.keep[fuel == 'flex fuel',
                 `:=`(Umpg_gas=1/(city_shr * 1/UCity + (1-city_shr)*1/UHighway),
                      Umpg_blend=1/(city_shr * 1/UCityA + (1-city_shr)*1/UHighwayA))]
  # remaining vehicles with orig mpg but not EPA mpg
  dt.factor.ffv <- dt.detail.keep[fuel == 'flex fuel',.(Umpg_gas,combined_mpg2008,agg_count)]
  dt.factor.ffv[,factor:= Umpg_gas/combined_mpg2008]
  Umult_ffv <- dt.factor.ffv[!is.na(Umpg_gas),.(wtd_factor = sum(factor*agg_count)/sum(agg_count))]$wtd_factor
  dt.detail.keep[is.na(Umpg_gas) & fuel == 'flex fuel' & !is.na(combined_mpg2008),
                 Umpg_gas:=combined_mpg2008*Umult_ffv]
  return(dt.detail.keep)
}

# calculate emissions as close to EPA GHG policy numbers as possible
calculate_emissions_EPA <- function(dt.detail.keep,mpg_adj,city_shr)
{
  if(mpg_adj == 'EPA')
  {
    # gasoline vehicles
    dt.detail.keep[fuel %in% c('gasoline','hybrid'),emis:=8887/(Umpg)]
    # diesel vehicles
    dt.detail.keep[fuel == 'diesel',emis:=10180/(Umpg)]
    # flex-fuel vehicles
    # FFV incentive went through 2015; 2016 and later, had to prove ethanol use, otherwise counted as gasoline eff.
    dt.detail.keep[fuel == 'flex fuel' & year < 2016,emis:=.5*8887/(Umpg_gas) + .15*8887/(Umpg_blend)]
    dt.detail.keep[fuel == 'flex fuel' & year >= 2016,emis:=8887/(Umpg_gas)]
    # for pre-2016 FFVs that couldn't be matched to EPA data (i.e., have no mpg_blend), use gasoline emis
    dt.detail.keep[fuel == 'flex fuel' & year < 2016 & is.na(mpg_blend),emis:=8887/(Umpg_gas)]
    
    # PHEVs
    dt.detail.keep[fuel == 'PHEV',emis:=((1-cityUF)/UCity*city_shr + (1-highwayUF)/UHighway*(1-city_shr))*8887]
    #head(unique(dt.detail.keep[fuel == 'PHEV',.(make,model,year,emis,emis2,mpg_gas,mpg_elec)]),20)
  } else if (mpg_adj == 'haircut')
  {
    factor_adj <- 1.25
    # gasoline vehicles
    dt.detail.keep[fuel %in% c('gasoline','hybrid'),emis:=8887/(combined_mpg2008*factor_adj)]
    # diesel vehicles
    dt.detail.keep[fuel == 'diesel',emis:=10180/(combined_mpg2008*factor_adj)]
    # flex-fuel vehicles
    # FFV incentive went through 2015; 2016 and later, had to prove ethanol use, otherwise counted as gasoline eff.
    dt.detail.keep[fuel == 'flex fuel' & year < 2016,emis:=.5*8887/(combined_mpg2008*factor_adj) + .15*8887/(mpg_blend*factor_adj)]
    dt.detail.keep[fuel == 'flex fuel' & year >= 2016,emis:=8887/(combined_mpg2008*factor_adj)]
    # for pre-2016 FFVs that couldn't be matched to EPA data (i.e., have no mpg_blend), use gasoline emis
    dt.detail.keep[fuel == 'flex fuel' & year < 2016 & is.na(mpg_blend),emis:=8887/(combined_mpg2008*factor_adj)]
    
    # PHEVs
    dt.detail.keep[fuel == 'PHEV',emis:=((1-cityUF)/(city08*factor_adj)*city_shr + (1-highwayUF)/(highway08*factor_adj)*(1-city_shr))*8887]
    #head(unique(dt.detail.keep[fuel == 'PHEV',.(make,model,year,emis,emis2,mpg_gas,mpg_elec)]),20)
  }
  # electric
  dt.detail.keep[fuel == 'electric',emis:= 0]
  # nat gas/hydrogen/bio-diesel left as NA for now
  # note: emissions values using the label mpg can be matched with EPA's at fueleconomy.gov/in the downloaded date (co2TailpipeGpm field)
  return(dt.detail.keep)
}

# apply vehicle multipliers for policy compliance
apply_vehicle_multipliers <- function(dt.detail.keep)
{
  # 2017-2019, EVs= 2, PHEVs, NGVs = 1.6
  # 2020, EVs = 1.75, PHEVs = 1.45
  # 2021, EVs = 1.5, PHEVs = 1.3
  # 2022, EVs = 1, PHEVs = 1
  dt.detail.keep[,multiplier:=1]
  dt.detail.keep[fuel %in% c('electric','hydrogen'),multiplier:=ifelse(year < 2017, 1, 
                                                                       ifelse(year %in% c(2017:2019), 2,
                                                                              ifelse(year == 2020, 1.75, 
                                                                                     ifelse(year == 2021, 1.5, NA))))]
  dt.detail.keep[fuel %in% c('PHEV','nat gas'),multiplier:=ifelse(year < 2017,1,
                                                                  ifelse(year %in% c(2017:2019),1.6,
                                                                         ifelse(year ==2020,1.45,
                                                                                ifelse(year == 2021, 1.3, NA))))]
  return(dt.detail.keep)
}

# calculate footprint based on adjustment for width -> track width
calculate_footprint <- function(dt.detail.keep)
{
  # note: this needs to be adjusted because width > track width
  width_adj = 0.86 # comes from a subset of track widths looked up on car and driver
  dt.detail.keep[,footprint:=(width*width_adj)*wheelbase/144] # this currently has width-track width adjustment
}

# calculate GHG standard for each vehicle
calculate_vehicle_standards <- function(dt.detail.keep)
{
  # these come from federal register
  ### 2012-2016
  # cars
  dt.detail.keep[footprint <= 41 & car == 1,
                 ghg_std:= ifelse(year == 2012,244,
                                  ifelse(year == 2013,237,
                                         ifelse(year == 2014,228,
                                                ifelse(year == 2015,217,
                                                       ifelse(year == 2016,206,NA)))))]
  dt.detail.keep[footprint > 56 & car == 1,
                 ghg_std:=ifelse(year == 2012,315,
                                 ifelse(year == 2013,307,
                                        ifelse(year == 2014,299,
                                               ifelse(year == 2015,288,
                                                      ifelse(year == 2016,277,NA)))))]
  dt.detail.keep[footprint > 41 & footprint <= 56 & car == 1,
                 ghg_std:=4.72*footprint + 
                   ifelse(year == 2012,50.5,
                          ifelse(year == 2013,43.3,
                                 ifelse(year == 2014,34.8,
                                        ifelse(year == 2015,23.4,
                                               ifelse(year == 2016, 12.7,NA)))))]
  # light trucks
  dt.detail.keep[footprint <= 41 & car == 0,
                 ghg_std:= ifelse(year == 2012,294,
                                  ifelse(year == 2013,284,
                                         ifelse(year == 2014,275,
                                                ifelse(year == 2015,261,
                                                       ifelse(year == 2016,247,NA)))))]
  dt.detail.keep[footprint > 66 & car == 0,
                 ghg_std:=ifelse(year == 2012,395,
                                 ifelse(year == 2013,385,
                                        ifelse(year == 2014,376,
                                               ifelse(year == 2015,362,
                                                      ifelse(year == 2016,348,NA)))))]
  dt.detail.keep[footprint > 41 & footprint <= 66 & car == 0,
                 ghg_std:=4.04*footprint + 
                   ifelse(year == 2012,128.6,
                          ifelse(year == 2013,118.7,
                                 ifelse(year == 2014,109.4,
                                        ifelse(year == 2015,95.1,
                                               ifelse(year == 2016,81.1,NA)))))]
  ### 2017-2020
  dt.detail.keep[car == 1 & year == 2017,`:=`(a=194.7,b=262.7,c=4.53,d=8.9)]
  dt.detail.keep[car == 1 & year == 2018,`:=`(a=184.9,b=250.1,c=4.35,d=6.5)]
  dt.detail.keep[car == 1 & year == 2019,`:=`(a=175.3,b=238,c=4.17,d=4.2)]
  dt.detail.keep[car == 1 & year == 2020,`:=`(a=166.1,b=226.2,c=4.01,d=1.9)]
  dt.detail.keep[car == 1 & year %in% c(2017:2020),
                 ghg_std:=pmin(b,pmax(a,c*footprint+d))]
  dt.detail.keep[car == 0 & year == 2017,`:=`(a=238.1,b=347.2,c=4.87,d=38.3,e=246.4,f=347.4,g=4.04,h=80.5)]
  dt.detail.keep[car == 0 & year == 2018,`:=`(a=226.8,b=341.7,c=4.76,d=31.6,e=240.9,f=341.9,g=4.04,h=75)]
  dt.detail.keep[car == 0 & year == 2019,`:=`(a=219.5,b=338.6,c=4.68,d=27.7,e=237.8,f=338.8,g=4.04,h=71.9)]
  dt.detail.keep[car == 0 & year == 2020,`:=`(a=211.9,b=336.7,c=4.57,d=24.6,e=235.9,f=336.9,g=4.04,h=70)]
  dt.detail.keep[car == 0 & year %in% c(2017:2020),
                 ghg_std:=pmin(pmin(b,pmax(a,c*footprint+d)),pmin(f,pmax(e,g*footprint+h)))]
  
  ### 2021-2022
  # cars
  dt.detail.keep[footprint <= 41 & car == 1 & year %in% c(2021,2022),
                 ghg_std:= ifelse(year == 2021,161.8,159)]
  dt.detail.keep[footprint > 56 & car == 1 & year %in% c(2021,2022),
                 ghg_std:=ifelse(year == 2021,220.9,217.3)]
  dt.detail.keep[footprint > 41 & footprint <= 56 & car == 1 & year %in% c(2021,2022),
                 ghg_std:=ifelse(year == 2021, 3.94*footprint + 1.9,
                                 3.88 * footprint + 0.2)]
  # light trucks
  dt.detail.keep[footprint <= 41 & car == 0 & year %in% c(2021,2022),
                 ghg_std:= ifelse(year == 2021,206.5,203)]
  dt.detail.keep[footprint > 68.3 & car == 0 & year %in% c(2021,2022),
                 ghg_std:= ifelse(year == 2021,329.4,324.1)]
  dt.detail.keep[footprint > 41 & footprint <= 68.3 & car == 0 & year %in% c(2021,2022),
                 ghg_std:=ifelse(year == 2021, 4.51*footprint + 21.5,
                                 4.44 * footprint + 20.6)]
  
  # drop coefficients used to calculate standard
  dt.detail.keep <- dt.detail.keep[,!c('a','b','c','d','e','f','g','h')]
  return(dt.detail.keep)
}

# incorporate off-cycle and alternative credits
add_offcycle_alt_credits <- function(dt.detail.keep,dt.ghg.credit)
{
  # clean credit data firm names
  dt.ghg.credit[,manufacturer_policy:=manufacturer_EPA]
  dt.ghg.credit[manufacturer_EPA %in% c('Mercedes','Mercedes-Benz'),manufacturer_policy:='Daimler']
  dt.ghg.credit[manufacturer_EPA %in% c('FCA','Stellantis'),manufacturer_policy:='Fiat Chrysler Automobiles']
  dt.ghg.credit[manufacturer_EPA == 'GM',manufacturer_policy:='General Motors']
  dt.ghg.credit[manufacturer_EPA == 'Jaguar Land Rover',manufacturer_policy:='Tata Motors']
  dt.ghg.credit[manufacturer_EPA == 'VW',manufacturer_policy:='Volkswagen']
  dt.ghg.credit[manufacturer_EPA %in% c('Lotus','Volvo') & year >= 2017,
                manufacturer_policy:='Geely']
  
  # add manufacturer_policy field to vehicle data, because EPA treats compliance differently
  dt.detail.keep[,manufacturer_policy:= oem]
  
  # deal with Hyundai/Kia, Volvo/Lotus, Nissan-Mitsubishi in 2017
  dt.detail.keep[oem %in% c('Geely') & year < 2017,manufacturer_policy:=make]
  dt.detail.keep[oem %in% c('Hyundai') & make != 'Genesis',manufacturer_policy:=make]
  dt.detail.keep[oem %in% c('Nissan','Mitsubishi') & year == 2017, manufacturer_policy:= 'Nissan-Mitsubishi']
  
  # merge ghg credit data with dt.detail.keep
  dt.detail.keep <- merge(dt.detail.keep,dt.ghg.credit,
                          by=c('manufacturer_policy','year','car'),
                          all.x=T)
  
  # adjust credit amounts
  dt.detail.keep[is.na(addl_ghg_credits),addl_ghg_credits:=0]
  dt.detail.keep[,ghg_std_adj:=ghg_std - addl_ghg_credits]
  dt.detail.keep[,ghg_credit_adj:=(ghg_std_adj - emis)*ifelse(car == 1, 195264, 225865)/1000000*multiplier]
  return(dt.detail.keep)
}

# assign quantiles to vehicle characteristics by year
assign_quantiles <- function(dt,col,n_quantiles)
{
  dt[,col_quantile:=2*n_quantiles]
  dt.quantile <- data.table()
  for (yr in unique(dt$year))
  {
    dt.yr <- dt[year == yr]
    vec.quantile <- quantile(dt.yr[get(col) > 0 & !is.na(get(col)),get(col)],seq(0,1,1/n_quantiles))
    dt[year == yr,col_quantile:=cut(get(col),vec.quantile,include.lowest=T,labels=F)]
    # save the quantile cutoffs
    dt.quantile.yr <- data.table(year = yr, quant = vec.quantile)
    dt.quantile <- rbind(dt.quantile,dt.quantile.yr)
  }
  setnames(dt,'col_quantile',paste0('quantile_',col))
  dt.quantile[,char:= col]
  
  return(list('dt' = dt,'cutoffs' = dt.quantile))
}
