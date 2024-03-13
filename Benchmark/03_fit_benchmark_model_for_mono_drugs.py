# Databricks notebook source
import os
import json
import pandas as pd
import numpy as np

from scipy.special import logit, expit
from scipy.optimize import curve_fit

from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor

from lightgbm import LGBMRegressor

# COMMAND ----------

# MAGIC %md
# MAGIC Possible values for the splits are: 'cell_line', 'drug', 'cell_line__drug' 
# MAGIC
# MAGIC Possible values for the model types are: 'linear', 'neural_network', 'lightgbm'

# COMMAND ----------

SPLIT_TYPE = 'cell_line__drug'
MODEL_TYPE = 'lightgbm'

# COMMAND ----------

input_data_folder = './az_benchmark/data/benchmark_data/'
output_data_folder =  './az_benchmark/data/benchmark_predictions/'

# COMMAND ----------

benchmark_drug_target_frame = pd.read_csv(os.path.join(input_data_folder, 'input_drug_target_frame.csv'))
benchmark_expression_frame = pd.read_csv(os.path.join(input_data_folder, 'input_expression_frame.csv'))
benchmark_data_frame = benchmark_drug_target_frame.merge(
  benchmark_expression_frame,
  on='cell_line',
  how='inner'
).dropna()
benchmark_data_frame

# COMMAND ----------

with open(os.path.join(input_data_folder, 'splits.json'), mode='r') as input_json:
  splits = json.load(input_json)
splits

# COMMAND ----------

split_indices = {}
for split_idx in splits[SPLIT_TYPE]:
  if SPLIT_TYPE == 'cell_line':
    train_indices = benchmark_data_frame[benchmark_data_frame['cell_line'].isin(splits[SPLIT_TYPE][split_idx]['train_set'])].index.tolist()
    test_indices = benchmark_data_frame[benchmark_data_frame['cell_line'].isin(splits[SPLIT_TYPE][split_idx]['test_set'])].index.tolist()
    split_indices[int(split_idx)] = {
      'train_set': train_indices,
      'test_set': test_indices
    }
  elif SPLIT_TYPE == 'drug':
    train_indices = benchmark_data_frame[benchmark_data_frame['drug'].isin(splits[SPLIT_TYPE][split_idx]['train_set'])].index.tolist()
    test_indices = benchmark_data_frame[benchmark_data_frame['drug'].isin(splits[SPLIT_TYPE][split_idx]['test_set'])].index.tolist()
    split_indices[int(split_idx)] = {
      'train_set': train_indices,
      'test_set': test_indices
    }
  elif SPLIT_TYPE == 'cell_line__drug':
    train_indices = benchmark_data_frame[
      (benchmark_data_frame['cell_line'] + '__' + benchmark_data_frame['drug']).isin(splits[SPLIT_TYPE][split_idx]['train_set'])
    ].index.tolist()
    test_indices = benchmark_data_frame[
      (benchmark_data_frame['cell_line'] + '__' + benchmark_data_frame['drug']).isin(splits[SPLIT_TYPE][split_idx]['test_set'])
    ].index.tolist()
    split_indices[int(split_idx)] = {
      'train_set': train_indices,
      'test_set': test_indices
    }
  else:
      raise f'{SPLIT_TYPE} is unknown split type!'

# COMMAND ----------

predictions_frame = []
for split_idx in split_indices:
  print(f'Running for split no.{split_idx + 1}')
  X_train = benchmark_data_frame.loc[split_indices[split_idx]['train_set']].iloc[:, 4:].to_numpy()
  y_train = benchmark_data_frame.loc[split_indices[split_idx]['train_set']]['killrate_true'].to_numpy()
  X_test = benchmark_data_frame.loc[split_indices[split_idx]['test_set']].iloc[:, 4:].to_numpy()

  if MODEL_TYPE == 'linear':
    model = Ridge(alpha=1e-1)
  elif MODEL_TYPE == 'neural_network':
    model = MLPRegressor(learning_rate_init=1e-3, alpha=1e-3, hidden_layer_sizes=(100,), activation='relu', solver='adam', early_stopping=True, max_iter=100, shuffle=True, random_state=1, verbose=1)
  elif MODEL_TYPE == 'lightgbm':
    model = LGBMRegressor(metric='rmse')
  else:
    raise f'{MODEL_TYPE} is unknown model type!'
  
  y_train_transformed = y_train.copy()
  y_train_transformed[y_train_transformed == 0.0] = np.min(y_train[y_train != 0.0]) / 2.0
  y_train_transformed[y_train_transformed == 1.0] = np.max(y_train[y_train != 1.0]) + (1.0 - np.max(y_train[y_train != 1.0])) / 2.0
  y_train_transformed = logit(y_train_transformed)

  model.fit(X_train, y_train_transformed)
  y_train_predicted = expit(model.predict(X_train))
  y_test_predicted = expit(model.predict(X_test))

  prediction_frame = benchmark_data_frame[['cell_line', 'drug', 'dose', 'killrate_true']].copy()
  prediction_frame['split'] = None
  prediction_frame['killrate_predicted'] = None
  prediction_frame.loc[split_indices[split_idx]['train_set'], 'split'] = 'TRAIN'
  prediction_frame.loc[split_indices[split_idx]['train_set'], 'killrate_predicted'] = y_train_predicted
  prediction_frame.loc[split_indices[split_idx]['test_set'], 'split'] = 'TEST'
  prediction_frame.loc[split_indices[split_idx]['test_set'], 'killrate_predicted'] = y_test_predicted
  prediction_frame['split_index'] = split_idx

  predictions_frame.append(prediction_frame)

predictions_frame = pd.concat(predictions_frame, axis=0)
predictions_frame.to_csv(os.path.join(output_data_folder, f'benchmark_predictions__{MODEL_TYPE}__{SPLIT_TYPE}.csv'), index=False)
predictions_frame

# COMMAND ----------

def calculate_logistic_model(concentration, ic50, h, e_inf):
  return  1.0 - ((100.0 + (e_inf - 100.0) / (1.0 + (ic50 / concentration)**h)) / 100.0)

def fit_curve_for_predictions(row):
  dose_grid = row[4:].keys().to_numpy().astype(float)
  killrates = row[4:].values
  try:
    return curve_fit(calculate_logistic_model, dose_grid, killrates)[0][0]
  except RuntimeError:
    return None

ic50_fitting_frame = predictions_frame.pivot_table(index=['split_index', 'split', 'cell_line', 'drug'], columns='dose', values='killrate_predicted').reset_index()

ic50_predictions_frame = ic50_fitting_frame[['cell_line', 'drug', 'split_index', 'split']].copy()
ic50_predictions_frame['ic50_predictions'] = ic50_fitting_frame.apply(fit_curve_for_predictions, axis=1)

ic50_predictions_frame.to_csv(os.path.join(output_data_folder, f'benchmark_ic50_predictions__{MODEL_TYPE}__{SPLIT_TYPE}.csv'), index=False)
ic50_predictions_frame

# COMMAND ----------


