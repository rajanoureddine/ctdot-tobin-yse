# RLPolk Analysis
This folder contains Python scripts used to perform basic descriptive analyses on the raw RLPolk data. The scripts used in this folder were last used in October - November 2023 and it would be extremely unusual if they were needed again. The code in this folder will be harder to replicate since it has been almost a year since last run and some of the input files were deleted off the Yale High Performance Clusters. 

Note that most of the geographic dataset files used to create the file `create_outputs.ipynb` were stored on the clusters and lost when Professor Gillingham's allocation ran out of space. Nonetheless, the `.gpkg` files are easy to download from the Census Bureau and elsewhere. The file `ev_sales_by_zip_geo.gpkg` was lost this way. Note, however, that creating this file should be easy: It is simply a matter of grouping the EV sales data below, by year and geography. 

The raw RLPolk data can be located under `tobin_working_data/rlpolk_data/US_Yale_University_OP0001562727_NV_CT_VIN_Prefix_202212.txt` 

The NHTSA VIN matching CSV is also located in the same folder. However, this CSV does not contain all the VINs required to match the RLP data, and so it is necessary to query the NHTSA API to acquire all the VINs. 

