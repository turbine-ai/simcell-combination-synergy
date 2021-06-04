library("tidyverse")
library("data.table")
library("MLmetrics")
library("pROC")

input_folder = "scripts/repo/data/raw/"
output_folder = "scripts/repo/results/"
utils = "scripts/repo/src/utils/"

# source monotherapy ROC function
source(paste0(utils, "mono_roc_functions.R"))

# monotherapy data
mono = fread(paste0(input_folder, "monotherapy_response.tsv"), na.strings = "N/A")
comb_filtering <-
  read_tsv(paste0(input_folder, "combi_filtering.txt"))

mono = mono %>% 
  select(compound, is, iv) %>% 
  inner_join(., comb_filtering, by = c("compound" = "turbine_name")) %>% 
  select(compound, is, iv, MoA_category) %>% 
  drop_na(iv) %>% 
  mutate(iv = iv / 100) %>%
  mutate(is = is / 100) %>% 
  mutate(iv_syn = case_when(iv > 0.2 ~ 1,
                            TRUE ~ 0)) %>% 
  mutate(is_syn = case_when(is > 0.2 ~ 1,
                            TRUE ~ 0)) %>% 
  select(MoA_category, is, iv_syn)

exclude_class = mono %>%
  group_by(MoA_category) %>%
  do(x = check_classes(.$iv_syn)) %>% 
  filter(x == "OK") %>% 
  select(-x)

# MoA ROC scores
aggr_roc = mono %>% 
  filter(MoA_category %in% exclude_class$MoA_category) %>% 
  group_by(MoA_category) %>% 
  do(auc = roc(.$iv_syn, .$is)$auc) %>% 
  ungroup() %>% 
  mutate(auc = as.numeric(auc)) %>% 
  arrange(desc(auc))

# export ROC scores
write.table(aggr_roc, paste0(output_folder, "MoA_cat_aggr_ROC.tsv"), quote = FALSE, row.names = TRUE)
