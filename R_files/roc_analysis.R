library("tidyverse")
library("data.table")
library("MLmetrics")
library("pROC")

input_folder = "scripts/repo/data/raw/"
output_folder = "scripts/repo/results/"
utils = "scripts/repo/src/utils/"

# source the necessary functions
source(paste0(utils, "roc_functions.R"))

# data
feas =
  read_tsv(paste0(input_folder, "combi_MoA_summerized_feasibility.txt"))
comb_filtering <-
  read_tsv(paste0(input_folder, "combi_filtering.txt"))
scores =
  read_tsv(paste0(input_folder, "combimetrics.tsv"))

# tests
# whole performance
performance_analysis(is_thr = 0.2,
                    iv_thr = 0.1)

# compound level performance
perf_anal_compound_level(compound = "AZ12618466",
          is_thr = 0.2,
          iv_thr = 0.2)

comps = c(
  "AZ12618466",
  "AZ13242274",
  "AZ13535704",
  "AZ13590108",
  "AZD1775",
  "AZD7762",
  "Ceralasertib",
  "Olaparib")

for(comp in comps) {
  perf_anal_compound_level(compound = comp)
}
