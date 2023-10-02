
# Setup ------------------------------------------------------------------------

rm(list = ls())

pkgTest <- function(x) {
  if (!require(x, character.only = TRUE))
  {
    install.packages(x, dep = TRUE)
    if(!require(x, character.only = TRUE)) stop("Package not found")
  }
}

## These lines load the required packages
packages <- c("rstudioapi", "data.table","tidyverse", "stringr","beepr","foreign",
              "zipcodeR")

lapply(packages, pkgTest)

# the following line is for getting the path of your current open file
current_path <- getActiveDocumentContext()$path 
# The next line set the working directory to the relevant one:
setwd(dirname(current_path ))
# you can make sure you are in the right directory
print( getwd() )

# Load -----

dta <- fread("US_Yale_University_OP0001562727_NV_CT_VIN_Prefix_202212.txt")

table(dta$MODEL_YEAR)
unique(dta$REPORT_YEAR_MONTH)%>%grep("2022",.)%>%unique(dta$REPORT_YEAR_MONTH)[.]
length(unique(dta$ZIP_CODE))
quantile(dta$VEH_COUNT,.9)
sum(dta$VEH_COUNT==1)/nrow(dta)
dta$one <- 1

is.numeric(ZIP_CODE)
temp <- dta[,.(N=sum(one)),by="REPORT_YEAR_MONTH"]
ggplot(temp) + geom_point(aes(x=REPORT_YEAR_MONTH,y=N))

download_zip_data(force = FALSE)

zips <- search_state(c("CT","MA"))
zips$zipcode <- as.numeric(zips$zipcode)
zips$zipcode

temp <- merge(dta,
      zips[,c("state","zipcode")],
      by.x="ZIP_CODE",by.y="zipcode")

table(temp$state)

