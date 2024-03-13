# Databricks notebook source
import os

import numpy as np
import pandas as pd

from scipy.optimize import curve_fit

# COMMAND ----------

input_folder = './az_benchmark/data/raw_in_silico_output/'
input_meta_folder = './az_benchmark/data/meta_data/'
output_folder = './az_benchmark/data/raw_in_silico_output/'

# COMMAND ----------

merged_killrates_ddr_frame = []

input_folder_ddr = os.path.join(input_folder, 'DDR')
for sub_folder_ddr in os.listdir(input_folder_ddr):
  sub_folder_ddr = os.path.join(input_folder_ddr, sub_folder_ddr)
  for file_ddr in os.listdir(sub_folder_ddr):
    if '_killrates.txt' in file_ddr:
      cell_line = file_ddr.replace('_killrates.txt', '')
      killrates_frame = pd.read_csv(os.path.join(sub_folder_ddr, file_ddr), delimiter='\t')
      killrates_frame.insert(loc=0, column='cell_line', value=cell_line)
      merged_killrates_ddr_frame.append(killrates_frame)

merged_killrates_ddr_frame = pd.concat(merged_killrates_ddr_frame, axis=0).reset_index(drop=True)
merged_killrates_ddr_frame = merged_killrates_ddr_frame.sort_values(by=['cell_line', 'drug', 'dose'])

merged_killrates_ddr_frame.loc[merged_killrates_ddr_frame['drug'] == 'NATIVE', 'drug'] = None

merged_killrates_ddr_frame = merged_killrates_ddr_frame.drop_duplicates(['cell_line', 'drug', 'dose'], keep='first')

merged_killrates_ddr_frame

# COMMAND ----------

merged_killrates_non_ddr_frame = []

input_folder_non_ddr = os.path.join(input_folder, 'non-DDR')
for sub_folder_non_ddr in os.listdir(input_folder_non_ddr):
  sub_folder_non_ddr = os.path.join(input_folder_non_ddr, sub_folder_non_ddr)
  for file_non_ddr in os.listdir(sub_folder_non_ddr):
    if '_killrates.txt' in file_non_ddr:
      cell_line = file_non_ddr.replace('_killrates.txt', '')
      killrates_frame = pd.read_csv(os.path.join(sub_folder_non_ddr, file_non_ddr), delimiter='\t')
      killrates_frame.insert(loc=0, column='cell_line', value=cell_line)
      merged_killrates_non_ddr_frame.append(killrates_frame)

merged_killrates_non_ddr_frame = pd.concat(merged_killrates_non_ddr_frame, axis=0).reset_index(drop=True)
merged_killrates_non_ddr_frame = merged_killrates_non_ddr_frame.sort_values(by=['cell_line', 'drug', 'dose'])

merged_killrates_non_ddr_frame.loc[merged_killrates_non_ddr_frame['drug'] == 'NATIVE', 'drug'] = None
merged_killrates_non_ddr_frame = merged_killrates_non_ddr_frame.drop_duplicates(['cell_line', 'drug', 'dose'], keep='first')

merged_killrates_non_ddr_frame

# COMMAND ----------

merged_killrates_frame = pd.concat([merged_killrates_ddr_frame, merged_killrates_non_ddr_frame], axis=0)
merged_killrates_frame = merged_killrates_frame.drop_duplicates(['cell_line', 'drug', 'dose'], keep='first').dropna()
merged_killrates_frame

# COMMAND ----------

compound_information_frame = pd.read_csv(os.path.join(input_meta_folder, 'compound_target_astrazeneca1.csv'), delimiter='\t')
compound_information_frame = compound_information_frame[['compound', 'moa_cat']].drop_duplicates().rename({'compound': 'drug', 'moa_cat': 'drug_category'}, axis=1)
compound_information_frame['drug'] = compound_information_frame['drug'].replace('Vinorelbine', 'Vinorelbin')
compound_information_frame = compound_information_frame.dropna()
drug_categories = merged_killrates_frame[['drug']].merge(
  compound_information_frame,
  how='left',
  on='drug'
)
merged_killrates_frame.insert(loc=1, column='drug_category', value=drug_categories['drug_category'])
merged_killrates_frame

# COMMAND ----------

merged_killrates_frame.to_csv(os.path.join(output_folder, 'merged_killrates.csv'), index=False)

# COMMAND ----------

def calculate_logistic_model(concentration, ic50, h, e_inf):
  return  1.0 - ((100.0 + (e_inf - 100.0) / (1.0 + (ic50 / concentration)**h)) / 100.0)

def fit_curve_for_predictions(row):
  dose_grid = row[2:].keys().to_numpy().astype(float)
  killrates = row[2:].values
  try:
    return curve_fit(calculate_logistic_model, dose_grid, killrates)[0][0]
  except RuntimeError:
    return None

ic50_fitting_frame = merged_killrates_frame.pivot_table(index=['cell_line', 'drug'], columns='dose', values='killrate').reset_index()
ic50_predictions_frame = ic50_fitting_frame[['cell_line', 'drug']].copy()
ic50_predictions_frame['ic50_predictions'] = ic50_fitting_frame.apply(fit_curve_for_predictions, axis=1)

ic50_predictions_frame.insert(loc=2, column='drug_category', value=drug_categories['drug_category'])

ic50_predictions_frame.to_csv(os.path.join(output_folder, f'turbine_ic50_predictions.csv'), index=False)
ic50_predictions_frame

# COMMAND ----------


