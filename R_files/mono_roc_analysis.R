library("tidyverse")
library("data.table")
library("MLmetrics")
library("pROC")

input_folder = "scripts/repo/data/raw/"
output_folder = "scripts/repo/results/"
utils = "scripts/repo/src/utils/"

# source monotherapy ROC function
source(paste0(utils, "mono_roc_functions.R"))

# data
comb_filtering <-
  read_tsv(paste0(input_folder, "combi_filtering.txt"))
ddrs = comb_filtering %>% filter(MoA_category == "DDR") %>% pull(turbine_name)

# whole monotherapy set
mono <-
  read.table(
    paste0(input_folder,
      "regenerated_isiv.txt"),
      sep = "\t",
      header = TRUE,
      na = c("N/A", "NA", "MISS"),
      stringsAsFactors = FALSE
    )

# non-DDR monotherapy  
mono_nonddr = mono %>% 
  filter(!Drug %in% ddrs)

# DDR monotherapy  
mono_ddr = mono %>% 
  filter(Drug %in% ddrs)

# performance analysis
mono_roc(
  input = mono,
  is_thr = 1,
  iv_thr = 2000,
  filename = "Monotherapy_ROC.png"
)

mono_roc(
  input = mono_nonddr,
  is_thr = 1,
  iv_thr = 2000,
  filename = "Monotherapy-nonDDR_ROC.png"
)  

mono_roc(
  input = mono_ddr,
  is_thr = 1,
  iv_thr = 2000,
  filename = "Monotherapy-DDR_ROC.png"
)  




