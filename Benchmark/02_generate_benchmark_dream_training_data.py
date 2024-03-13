# Databricks notebook source
import os
import json
import pandas as pd

import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# COMMAND ----------

APPLY_PCA_FOR_EXPRESSION = True
PCA_VARIANCE_EXPLAINED = 0.9

NUMBER_OF_SPLITS = 10
SPLIT_RATIO = 0.8

# COMMAND ----------

input_data_folder =  './az_benchmark/data/'
output_data_folder =  './az_benchmark/data/benchmark_data/'

# COMMAND ----------

dream_supplementary_frame = pd.read_csv(os.path.join(input_data_folder, 'dream_data/dream_drug_combinations_supplementary.csv'))
dream_supplementary_frame

# COMMAND ----------

dream_supplementary_frame = dream_supplementary_frame[dream_supplementary_frame['QA'] != -1]

dream_supplementary_frame.insert(loc=1, column='cell_line', value=dream_supplementary_frame['Cell line name'].map(lambda cell_line_name: cell_line_name.replace('-', '')))

dream_to_turbine_compound_name = pd.read_csv(os.path.join(input_data_folder, 'meta_data/dream_to_turbine_compound_name.csv'))
dream_to_turbine_compound_name = dict(zip(dream_to_turbine_compound_name['Challenge drug name'], dream_to_turbine_compound_name['Turbine name']))
dream_supplementary_frame.insert(loc=3, column='drug_a', value=dream_supplementary_frame['Compound A'].map(dream_to_turbine_compound_name))
dream_supplementary_frame.insert(loc=5, column='drug_b', value=dream_supplementary_frame['Compound B'].map(dream_to_turbine_compound_name))

dream_supplementary_frame = dream_supplementary_frame.rename({'IC50 A': 'ic50_a', 'H A': 'h_a', 'Einf A': 'einf_a', 'IC50 B': 'ic50_b', 'H B': 'h_b', 'Einf B': 'einf_b', 'Synergy score ': 'synergy_score_true'}, axis=1)
dream_supplementary_frame = dream_supplementary_frame[['cell_line', 'drug_a', 'drug_b', 'ic50_a', 'h_a', 'einf_a', 'ic50_b', 'h_b', 'einf_b', 'synergy_score_true']]
dream_supplementary_frame = dream_supplementary_frame.dropna()
dream_supplementary_frame

# COMMAND ----------

# MAGIC %md
# MAGIC The survival dose-response curves are derived based on the following article: Di Veroli, G.Y. et al. An automated fitting procedure and software for dose-response curves with multiphasic features. Sci. Rep. 5, 14701; doi: 10.1038/srep14701 (2015).

# COMMAND ----------

# MAGIC %md
# MAGIC The dose grid is based on the values set for the Simulated Cell model.

# COMMAND ----------

dose_grid = [1.00e-01, 5.00e-01, 1.00e+00, 5.00e+00, 1.00e+01, 5.00e+01, 1.00e+02, 2.50e+02, 5.00e+02, 7.50e+02, 1.00e+03, 1.75e+03, 2.50e+03, 4.25e+03, 5.00e+03, 6.75e+03, 7.50e+03, 9.25e+03, 1.00e+04, 2.50e+04]

def calculate_logistic_model(row):
  return  1.0 - ((100.0 + (row['einf'] - 100.0) / (1.0 + (row['ic50'] / row['dose'])**row['h'])) / 100.0)

dream_mono_frame = dream_supplementary_frame.copy()
dream_mono_frame = pd.concat([
  dream_mono_frame[['cell_line', 'drug_a', 'ic50_a', 'h_a', 'einf_a']].rename({'drug_a': 'drug', 'ic50_a': 'ic50', 'h_a': 'h', 'einf_a': 'einf'}, axis=1),
  dream_mono_frame[['cell_line', 'drug_b', 'ic50_b', 'h_b', 'einf_b']].rename({'drug_b': 'drug', 'ic50_b': 'ic50', 'h_b': 'h', 'einf_b': 'einf'}, axis=1)
], axis=0)
dream_mono_frame = dream_mono_frame.drop_duplicates(['cell_line', 'drug'], keep='first')
dream_mono_frame.to_csv(os.path.join(input_data_folder, 'dream_data/dream_mono_drug_statistics.csv'), index=False)

dream_mono_frame = dream_mono_frame.merge(
  pd.DataFrame({'dose': dose_grid}),
  how='cross'
)
dream_mono_frame['killrate_true'] = dream_mono_frame.apply(calculate_logistic_model, axis=1)

dream_mono_frame

# COMMAND ----------

compound_target_frame = pd.read_csv(os.path.join(input_data_folder, 'meta_data/compound_target_astrazeneca1.csv'), delimiter='\t')
compound_target_frame

# COMMAND ----------

drug_target_frame = compound_target_frame[
  compound_target_frame['compound'].isin(dream_mono_frame['drug'].unique()) &
  (compound_target_frame['effect'] == 'Inhibition') &
  (compound_target_frame['effect_type'] == 'IC50') &
  (compound_target_frame['effect_relation'] == '=')
][['compound', 'gene_name', 'effect_value']].rename({'compound': 'drug'}, axis=1)
drug_target_frame = drug_target_frame.drop_duplicates().groupby(['drug', 'gene_name']).mean().reset_index()
drug_target_frame = drug_target_frame.pivot_table(values='effect_value', index='drug', columns='gene_name').fillna(np.inf).reset_index()
drug_target_frame

# COMMAND ----------

benchmark_model_frame = dream_mono_frame[['cell_line', 'drug', 'dose', 'killrate_true']].merge(
  drug_target_frame,
  on='drug',
  how='left'
)
benchmark_model_frame

# COMMAND ----------

def calculate_inhibition_from_doses(dose, ic50):
  return 1.0 - dose / (ic50 + dose)

def transform_drug_target_information(row):
  dose = row['dose']
  row_transformed = {
    'cell_line': row['cell_line'],
    'drug': row['drug'],
    'dose': row['dose'],
    'killrate_true': row['killrate_true']
  }
  for target in row[4:].keys():
    row_transformed[target] = calculate_inhibition_from_doses(dose, row[target])
  return row_transformed

benchmark_model_frame = pd.DataFrame.from_records(benchmark_model_frame.apply(transform_drug_target_information, axis=1))
benchmark_model_frame

# COMMAND ----------

expression_data = pd.read_csv(os.path.join(input_data_folder, 'depmap_data/CCLE_expression.csv'))
expression_data = expression_data.rename({'Unnamed: 0': 'DepMap_ID'}, axis=1)
sample_info = pd.read_csv(os.path.join(input_data_folder, 'depmap_data/sample_info.csv'))
sample_info = sample_info[['DepMap_ID', 'stripped_cell_line_name']].rename({'stripped_cell_line_name': 'cell_line'}, axis=1)
expression_data = sample_info.merge(
  expression_data,
  on='DepMap_ID',
  how='inner'
).drop('DepMap_ID', axis=1)

expression_standardized = StandardScaler(with_mean=True, with_std=True).fit_transform(expression_data.iloc[:, 1:].to_numpy())
if APPLY_PCA_FOR_EXPRESSION:
  expression_reduced = PCA(n_components=PCA_VARIANCE_EXPLAINED).fit_transform(expression_standardized)
  expression_data = pd.concat([expression_data[['cell_line']], pd.DataFrame(expression_reduced)], axis=1)
else:
  expression_data_columns = expression_data.columns
  expression_data = pd.concat([expression_data[['cell_line']], pd.DataFrame(expression_standardized)], axis=1)
  expression_data.columns = expression_data_columns
  expression_data = expression_data.dropna(axis=1, how='any')

expression_data

# COMMAND ----------

benchmark_experiments_frame = benchmark_model_frame[['cell_line', 'drug', 'dose', 'killrate_true']].merge(
  expression_data[['cell_line']],
  on='cell_line',
  how='inner'
).dropna()
cell_lines = benchmark_experiments_frame['cell_line'].unique()
drugs = benchmark_experiments_frame['drug'].unique()
cell_lines__drugs = (benchmark_experiments_frame['cell_line'] + '__' + benchmark_experiments_frame['drug']).unique()
benchmark_experiments_frame

# COMMAND ----------

splits = {}
for split_type in ['cell_line', 'drug', 'cell_line__drug']:
  split_experiments = {}
  for split_idx in range(NUMBER_OF_SPLITS):
    if split_type == 'cell_line':
      cell_lines_permuted = np.random.permutation(cell_lines)
      cell_lines_train, cell_lines_test = cell_lines_permuted[:int(SPLIT_RATIO * len(cell_lines_permuted))], cell_lines_permuted[int(SPLIT_RATIO * len(cell_lines_permuted)):]
      split_experiments[split_idx] = {
        'train_set': list(cell_lines_train),
        'test_set': list(cell_lines_test)
      }
    elif split_type == 'drug':
      drugs_permuted = np.random.permutation(drugs)
      drugs_train, drugs_test = drugs_permuted[:int(SPLIT_RATIO * len(drugs_permuted))], drugs_permuted[int(SPLIT_RATIO * len(drugs_permuted)):]
      split_experiments[split_idx] = {
        'train_set': list(drugs_train),
        'test_set': list(drugs_test)
      }
    elif split_type == 'cell_line__drug':
      cell_lines__drugs_permuted = np.random.permutation(cell_lines__drugs)
      cell_lines__drugs_train, cell_lines__drugs_test = cell_lines__drugs_permuted[:int(SPLIT_RATIO * len(cell_lines__drugs_permuted))], cell_lines__drugs_permuted[int(SPLIT_RATIO * len(cell_lines__drugs_permuted)):]
      split_experiments[split_idx] = {
        'train_set': list(cell_lines__drugs_train),
        'test_set': list(cell_lines__drugs_test)
      }
    else:
      raise f'{split_type} is unknown split type!'

  splits[split_type] = split_experiments
splits

# COMMAND ----------

with open(os.path.join(output_data_folder, 'splits.json'), mode='w') as output_json:
  json.dump(splits, output_json)
benchmark_model_frame.to_csv(os.path.join(output_data_folder, 'input_drug_target_frame.csv'), index=False)
expression_data.to_csv(os.path.join(output_data_folder, 'input_expression_frame.csv'), index=False)

# COMMAND ----------

def calculate_inhibition_from_doses(dose, ic50):
  return 1.0 - dose / (ic50 + dose)

def transform_drug_target_information(row):
  dose = row['dose']
  row_transformed = {
    'drug': row['drug'],
    'dose': row['dose']
  }
  for target in row[2:].keys():
    row_transformed[target] = calculate_inhibition_from_doses(dose, row[target])
  return row_transformed

dose_grid = [1.00e-01, 5.00e-01, 1.00e+00, 5.00e+00, 1.00e+01, 5.00e+01, 1.00e+02, 2.50e+02, 5.00e+02, 7.50e+02, 1.00e+03, 1.75e+03, 2.50e+03, 4.25e+03, 5.00e+03, 6.75e+03, 7.50e+03, 9.25e+03, 1.00e+04, 2.50e+04]
synergy_prediction_frame = pd.DataFrame({
  'drug': sorted(set(dream_supplementary_frame['drug_a'].astype(str).tolist() + dream_supplementary_frame['drug_b'].astype(str).tolist()))
})
synergy_prediction_frame = synergy_prediction_frame.merge(pd.DataFrame(dose_grid, columns=['dose']), how='cross')

synergy_prediction_frame = synergy_prediction_frame.merge(
  drug_target_frame,
  on='drug',
  how='left'
).dropna().reset_index(drop=True)

synergy_prediction_frame = pd.DataFrame.from_records(synergy_prediction_frame.apply(transform_drug_target_information, axis=1))

synergy_prediction_frame

# COMMAND ----------

synergy_prediction_frame.to_csv(os.path.join(output_data_folder, 'input_combination_drug_target_for_mono_drugs_frame.csv'), index=False)
dream_supplementary_frame[
  dream_supplementary_frame['cell_line'].isin(expression_data['cell_line'].unique())
][['cell_line', 'drug_a', 'drug_b', 'synergy_score_true']].to_csv(os.path.join(output_data_folder, 'input_combination_drug_frame.csv'), index=False)

# COMMAND ----------


