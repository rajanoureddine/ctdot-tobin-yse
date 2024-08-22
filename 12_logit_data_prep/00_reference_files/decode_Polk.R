library(data.table)
library(stringr)

# setup ----
str.dir <- "~/Dropbox (YSE)/"
str.sp <- paste0(str.dir,'SW_Polk/')
str.decoder <- paste0(str.dir,'DataOne/')
str.vin <- paste0(str.dir,'SW_Automaker_Strategies/data/')

# choose S&P data pull yer
data_year <- 2022 # 2022 or 2023
# choose decoder
decoder_year <- 2022 # 2022 or 2024
if(decoder_year == 2024)
{
  str.decoder <- paste0(str.decoder,'DataOne_08072024/')
} else
{
  str.decoder <- paste0(str.decoder,'DataOne_12072022/')
}

# read in S&P data
v = '15state'
#v = 'CT'
if(data_year == 2022)
{
  if(v == '15state')
  {
    str.data <- 'US_Yale_University_OP0001562727_NV_Top_15_by_County_VIN_Prefix_202212.txt'
  } else
  {
    str.data <- 'US_Yale_University_OP0001562727_NV_CT_VIN_Prefix_202212.txt'
  }
} else
{
  if(v == '15state')
  {
    str.data <- '202310_ALT_VIN/US_Yale_University_OP0001635118_NV_Top_15_by_County_VIN_Prefix_8x10x11_202310.txt'
  } else
  {
    # throw error with message '2023 CT/MA data does not include VINs'
    stop('2023 CT/MA data does not include VINs')
  }
}

dt.sp <- data.table(read.table(paste0(str.sp,str.data),
                               sep='|',header=T))
setnames(dt.sp,colnames(dt.sp),tolower(colnames(dt.sp)))
dt.sp[,make:=str_to_title(make)]
dt.sp[,model:=str_replace_all(str_to_title(model),' ','-')]

# drop check digit
if(data_year == 2022)
{
  dt.sp[,vin_pattern:=paste0(substring(vin_prefix,1,8),substring(vin_prefix,10,10))]
} else
{
  dt.sp[,vin_pattern:=vin8x10x11]
}
dt.sp.unique <- unique(dt.sp[,.(make,model,model_year,vin_pattern)])
dt.sp.unique[,sp_id:=1:nrow(dt.sp.unique)]

# merge id back into original data
dt.sp <- merge(dt.sp,
               dt.sp.unique[,.(vin_pattern,make,model,model_year,sp_id)],
               by=c('vin_pattern','make','model','model_year'))

# read in VIN decoder
dt.decode <- data.table(read.csv(paste0(str.decoder,'DataOne_IDP_yale_school_of_the_environment.csv')))
setnames(dt.decode,colnames(dt.decode),tolower(colnames(dt.decode)))
dt.decode[,make:=str_to_title(make)]
dt.decode[,model:=str_replace_all(str_to_title(model),' ','-')]
setnames(dt.decode,'def_engine_id','engine_id')
dt.decode[,merge_id:=1:nrow(dt.decode)]

# get standard transmission configuration
# read in transmission data from VIN decoder
dt.trans <- data.table(read.csv(paste0(str.decoder,'DataOne_US_LDV_Data/LKP_VEH_TRANS.csv')))
# keep only standard transmissions
dt.trans <- dt.trans[standard == 'Y']
# merge
dt.decode <- merge(dt.decode,dt.trans,by=c('vehicle_id'),all.x=T)

# read in engine data
dt.engine <- data.table(read.csv(paste0(str.decoder,'DataOne_US_LDV_Data/DEF_ENGINE.csv')))
dt.decode <- merge(dt.decode,dt.engine[,.(engine_id,max_hp)],by=c('engine_id'),all.x=T)

# read in mpg data from VIN decoder
dt.decode.mpg <- data.table(read.csv(paste0(str.decoder,'DataOne_US_LDV_Data/LKP_VEH_MPG.csv')))
dt.decode.mpg.orig <- copy(dt.decode.mpg)

# deal with fuel grades where there are multiple entries
# get multiple entries
dt.dup.fg <- dt.decode.mpg[,.(ct=.N),by=c('vehicle_id','engine_id','transmission_id','fuel_type')][ct > 1]
dt.dup.fg[,multi_fg_flag:=1]
# merge into main data
dt.decode.mpg <- merge(dt.decode.mpg,dt.dup.fg[,!c('ct'),with=F],
                      by=c('vehicle_id','engine_id','transmission_id','fuel_type'),
                      all.x=T)
dt.dup.fg <- dt.decode.mpg[multi_fg_flag == 1]
dt.dup.fg <- dt.dup.fg[fuel_grade == 'Regular']
# combine back into dt.decode.mpg
dt.decode.mpg <- rbind(dt.decode.mpg[is.na(multi_fg_flag)],
                       dt.dup.fg)
# check no remaining duplicates
#dt.decode.mpg[,.(ct=.N),by=c('vehicle_id','engine_id','transmission_id','fuel_type')][ct > 1]

# deal with multi-fuel engines where there are multiple entries
dt.dup.multi.fuel <- dt.decode.mpg[,.(ct=.N),by=c('vehicle_id','engine_id','transmission_id')][ct > 1]
dt.dup.multi.fuel[,multi_fuel_flag:=1]
# merge into main data
dt.decode.mpg <- merge(dt.decode.mpg,dt.dup.multi.fuel[,!c('ct'),with=F],
                       by=c('vehicle_id','engine_id','transmission_id'),all.x=T)
dt.dup.multi.fuel <- dt.decode.mpg[multi_fuel_flag == 1]
# reshape with main fuel and secondary fuel
dt.dup.multi.fuel <- reshape(dt.dup.multi.fuel[,.(vehicle_id,engine_id,transmission_id,fuel_type,city,highway,combined)],
                             idvar=c('vehicle_id','engine_id','transmission_id'),
                             timevar='fuel_type',direction='wide')
dt.dup.multi.fuel[,`:=`(city_mpg1 = city.Gasoline,highway_mpg1 = highway.Gasoline,combined_mpg1 = combined.Gasoline,
                        fuel1 = 'Gasoline')]
dt.dup.multi.fuel[,fuel2:=ifelse(!is.na(city.Ethanol),'Ethanol',
                                 ifelse(!is.na(city.Electricity),'Electricity',
                                               ifelse(!is.na(city.E85),'E85','NA')))]
dt.dup.multi.fuel[fuel2 == 'Ethanol',`:=`(city_mpg2 = city.Ethanol,highway_mpg2 = highway.Ethanol,combined_mpg2 = combined.Ethanol)]
dt.dup.multi.fuel[fuel2 == 'Electricity',`:=`(city_mpg2 = city.Electricity,highway_mpg2 = highway.Electricity,combined_mpg2 = combined.Electricity)]
dt.dup.multi.fuel[fuel2 == 'E85',`:=`(city_mpg2 = city.E85,highway_mpg2 = highway.E85,combined_mpg2 = combined.E85)]
dt.dup.multi.fuel <- dt.dup.multi.fuel[,.(vehicle_id,engine_id,transmission_id,fuel1,city_mpg1,highway_mpg1,combined_mpg1,
                     fuel2,city_mpg2,highway_mpg2,combined_mpg2)]
# combine back into dt.decode.mpg
dt.decode.mpg <- rbind(dt.decode.mpg[is.na(multi_fuel_flag),!c('city_old','highway_old','combined_old')],
                       dt.dup.multi.fuel,fill=TRUE)
# check no remaining duplicates
#dt.decode.mpg[,.(ct=.N),by=c('vehicle_id','engine_id','transmission_id')][ct > 1]

# merge mpg data with main decoder data
setnames(dt.decode.mpg,'transmission_id','trans_id')
dt.decode <- merge(dt.decode,dt.decode.mpg,by=c('vehicle_id','engine_id','trans_id'),all.x=T)
# some missing mpgs
dt.decode[is.na(combined) & is.na(combined_mpg1) & year > 2014,.(vehicle_id,engine_id,trans_id,make,model,year)]
# drop some strange fuel_type mismatches
#unique(dt.decode[!is.na(fuel_type.y),.(fuel_type.x,fuel_type.y)])[order(fuel_type.x)]
dt.decode <- dt.decode[((fuel_type.x == 'F' & fuel_type.y %in% c('Gasoline','Ethanol')) | 
            (fuel_type.x == 'G' & fuel_type.y %in% c('Ethanol','E85')) | 
            (fuel_type.x == 'I' & fuel_type.y %in% c('Gasoline','Electricity')) |
            (fuel_type.x == 'N' & fuel_type.y == 'Gasoline')),
            `:=`(city=NA,highway=NA,combined=NA,
                 city_mpg1=NA,highway_mpg1=NA,combined_mpg1=NA,fuel1=NA,
                 city_mpg2=NA,highway_mpg2=NA,combined_mpg2=NA,fuel2=NA)]

# remove some unnecessary columns
dt.decode <- dt.decode[,!c('merge_id','standard','fuel_type.y','veh_mpg_id',
                            'fuel_grade','multi_fg_flag','multi_fuel_flag'),with=F]
setnames(dt.decode,'fuel_type.x','fuel_type')

# drop final digit for decoder (for 2022 data draw)
dt.decode[,vin_orig:=vin_pattern]
if(data_year == 2022)
{
  dt.decode[,vin_pattern:=substr(vin_pattern,1,9)]
}
setnames(dt.decode,'year','model_year')

# merge ----
dt.sp.merge <- merge(dt.sp.unique,
                     dt.decode[,.(vin_pattern,vin_orig,vehicle_id,model_year,make,model,trim,style,
                                  fuel_type,msrp,vehicle_type,body_type,drive_type,plant,length,height,
                                  width,wheelbase,curb_weight,doors,tmp_tank1_gal,max_hp,def_engine_size,city,highway,combined,
                                  fuel1,city_mpg1,highway_mpg1,combined_mpg1,
                                  fuel2,city_mpg2,highway_mpg2,combined_mpg2)],
                     by=c('vin_pattern','model_year'),
                     all.x=T,
                     allow.cartesian=T)
dt.sp.merge.unmatched <- dt.sp.merge[is.na(vehicle_id)]
dt.sp.merge <- dt.sp.merge[!is.na(vehicle_id)]
#unique(dt.sp.merge[make.x != make.y,.(make.x,make.y)])
   
# save crosswalk of s&p data to potential vin pattern matches
# save data ----
if (v == '15state')
{
  file_sp <- paste0('top_15_county_sp_',data_year,'.csv')
  file_decoded <- 'sp_vin_decoded_nonunique.csv'
}else if(v == 'CT')
{
  file_sp <- paste0('ct_zip_sp_',data_year,'.csv')
  file_decoded <- 'ct_sp_vin_decoded_nonunique.csv'
}
# save the original s&p data with sp_id
write.csv(dt.sp,paste0(str.sp,file_sp),row.names=F)

# save the mapping of s&p vin to more vin info
setnames(dt.sp.merge,c('make.x','model.x','make.y','model.y'),
         c('make_sp','model_sp','make','model'))
str.sp <- paste0(str.sp,data_year,'_data_',decoder_year,'_decoder/')

write.csv(dt.sp.merge,paste0(str.sp,file_decoded),row.names=F)

length(unique(dt.sp$sp_id))
nrow(dt.sp.merge.unmatched)
nrow(dt.sp.merge[,.(ct=.N),by=c('sp_id','vin_pattern')][ct>1])
nrow(dt.sp.merge[,.(ct=.N),by=c('sp_id','vin_pattern')][ct==1])

