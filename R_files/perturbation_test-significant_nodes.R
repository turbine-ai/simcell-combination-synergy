library("tidyverse")
library("data.table")
library("ggpubr")
library("broom")

input_folder = "scripts/repo/data/processed/"
output_folder = "scripts/repo/results/"
utils = "scripts/repo/src/utils/"

# source statistical functions
source(paste0(utils, "statistical_functions.R"))

# data
p_ola_aggr = fread(paste0(input_folder, "combiomarker_postproc-Ola_ATM-Gene_Name_aggr_pvalues.csv"))
p_az_aggr = fread(paste0(input_folder, "combiomarker_postproc-AZ13150560_AZ12879988-Gene_Name_aggr_pvalues.csv"))

# execution
sign_ola = sign_genes(p_ola_aggr, prefix = "Ola-ATM")
sign_az = sign_genes(p_az_aggr, prefix = "AZ13150560-AZ12879988")

# exporting
write_tsv(sign_ola, paste0(output_folder, "Perturbation_test-Ola-ATM_significant_genes.tsv"))
write_tsv(sign_az, paste0(output_folder, "Perturbation_test-AZ13150560-AZ12879988_significant_genes.tsv"))
