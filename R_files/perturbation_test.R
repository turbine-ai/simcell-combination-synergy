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
combi_ola = fread(paste0(input_folder, "combiomarker_postprocess-Ola_ATM.csv"))
combi_az = fread(paste0(input_folder, "combiomarker_postprocess-AZ13150560_AZ12879988.txt"))

# execution
p_az = pert_test(combi_az)
p_az

p_ola = pert_test(combi_ola)
p_ola

print(paste0("Ola_ATM Bliss max IC50 test, Mann-Whitney: ", as.character(p_ola)))
print(paste0("AZ13150560_AZ12879988 Bliss max IC50 test, Mann-Whitney: ", as.character(p_az)))

p_az_aggr = aggr_mannwhitney(combi_az)
p_az_aggr
p_ola_aggr = aggr_mannwhitney(combi_ola)
p_ola_aggr

write_tsv(p_ola_aggr, paste0(output_folder, "combiomarker_postproc-Ola_ATM-Gene_Name_aggr_pvalues.csv"))
          