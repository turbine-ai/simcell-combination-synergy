# Databricks notebook source
import os

import pandas as pd
import numpy as np

from scipy.special import logit, expit

from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor

from lightgbm import LGBMRegressor

# COMMAND ----------

input_data_folder = './az_benchmark/data/benchmark_data/'
output_data_folder =  './az_benchmark/data/benchmark_predictions/'

# COMMAND ----------

# MAGIC %md
# MAGIC Possible model types are: 'linear', 'neural_network', 'lightgbm'

# COMMAND ----------

MODEL_TYPE = 'lightgbm'

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

X_train = benchmark_data_frame.iloc[:, 4:].dropna().to_numpy()
y_train = benchmark_data_frame['killrate_true'].to_numpy()

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
model

# COMMAND ----------

benchmark_combination_drug_target_frame = pd.read_csv(os.path.join(input_data_folder, 'input_combination_drug_frame.csv'))
benchmark_combination_drug_target_frame

# COMMAND ----------

benchmark_combination_data_for_mono_drugs_frame = pd.read_csv(os.path.join(input_data_folder, 'input_combination_drug_target_for_mono_drugs_frame.csv'))
benchmark_combination_data_for_mono_drugs_frame

# COMMAND ----------

benchmark_combination_drug_target_frame_tmp = benchmark_combination_drug_target_frame[['cell_line', 'drug_a', 'drug_b']].merge(
  benchmark_combination_data_for_mono_drugs_frame,
  left_on='drug_a',
  right_on='drug',
  how='inner'
).drop('drug', axis=1).rename({'dose': 'dose_a'}, axis=1).merge(
  benchmark_combination_data_for_mono_drugs_frame,
  left_on='drug_b',
  right_on='drug',
  how='inner',
  suffixes=(None, '_b'),
).drop('drug', axis=1).rename({'dose': 'dose_b'}, axis=1)
benchmark_combination_experiments_frame = benchmark_combination_drug_target_frame_tmp[['cell_line', 'drug_a', 'dose_a', 'drug_b', 'dose_b']]

drug_target_list = benchmark_combination_data_for_mono_drugs_frame.columns[2:].tolist()
drug_target_b_list = [drug_target + '_b' for drug_target in drug_target_list]
benchmark_combination_drug_target_frame_tmp = pd.DataFrame(benchmark_combination_drug_target_frame_tmp[drug_target_list].to_numpy() * benchmark_combination_drug_target_frame_tmp[drug_target_b_list].to_numpy())
benchmark_combination_drug_target_frame_tmp.columns = drug_target_list
benchmark_combination_drug_target_frame = pd.concat([benchmark_combination_experiments_frame, benchmark_combination_drug_target_frame_tmp], axis=1)
benchmark_combination_drug_target_frame

# COMMAND ----------

combination_cell_lines = benchmark_combination_drug_target_frame['cell_line'].unique()
combination_prediction_frame = []
for cell_line in combination_cell_lines:
  benchmark_combinations_to_predict = benchmark_combination_drug_target_frame[benchmark_combination_drug_target_frame['cell_line'] == cell_line]
  
  benchmark_combinations_to_predict_array = benchmark_combinations_to_predict.merge(
    benchmark_expression_frame,
    on='cell_line',
    how='inner'
  ).iloc[:, 5:].to_numpy()
  if benchmark_combinations_to_predict_array.shape[0] == 0:
    continue

  benchmark_combinations_to_predict = benchmark_combinations_to_predict[['cell_line', 'drug_a', 'dose_a', 'drug_b', 'dose_b']]
  y_test_predicted = expit(model.predict(benchmark_combinations_to_predict_array))
  benchmark_combinations_to_predict.insert(loc=5, column='killrate_predicted', value=y_test_predicted)
  
  combination_prediction_frame.append(benchmark_combinations_to_predict)

combination_prediction_frame = pd.concat(combination_prediction_frame, axis=0)
combination_prediction_frame

# COMMAND ----------

combination_for_mono_drugs_prediction_frame = []
for cell_line in combination_cell_lines:
  benchmark_combination_data_for_mono_drugs_to_predict = benchmark_combination_data_for_mono_drugs_frame.copy()
  benchmark_combination_data_for_mono_drugs_to_predict.insert(loc=0, column='cell_line', value=cell_line)
  
  benchmark_combination_data_for_mono_drugs_to_predict_array = benchmark_combination_data_for_mono_drugs_to_predict.merge(
    benchmark_expression_frame,
    on='cell_line',
    how='inner'
  ).iloc[:, 3:].to_numpy()
  if benchmark_combination_data_for_mono_drugs_to_predict_array.shape[0] == 0:
    continue

  benchmark_combination_data_for_mono_drugs_to_predict = benchmark_combination_data_for_mono_drugs_to_predict[['cell_line', 'drug', 'dose']]
  y_test_predicted = expit(model.predict(benchmark_combination_data_for_mono_drugs_to_predict_array))
  benchmark_combination_data_for_mono_drugs_to_predict.insert(loc=3, column='killrate_predicted', value=y_test_predicted)
  
  combination_for_mono_drugs_prediction_frame.append(benchmark_combination_data_for_mono_drugs_to_predict)

combination_for_mono_drugs_prediction_frame = pd.concat(combination_for_mono_drugs_prediction_frame, axis=0)
combination_for_mono_drugs_prediction_frame

# COMMAND ----------

combination_prediction_frame = combination_prediction_frame.merge(
  combination_for_mono_drugs_prediction_frame,
  how='left',
  left_on=['cell_line', 'drug_a', 'dose_a'],
  right_on=['cell_line', 'drug', 'dose'],
  suffixes=('', '_a')
).drop(['drug', 'dose'], axis=1).merge(
  combination_for_mono_drugs_prediction_frame,
  how='left',
  left_on=['cell_line', 'drug_b', 'dose_b'],
  right_on=['cell_line', 'drug', 'dose'],
  suffixes=('', '_b')
).drop(['drug', 'dose'], axis=1)
combination_prediction_frame

# COMMAND ----------

# MAGIC %md
# MAGIC Bliss synergy calculation is based on the following article: Liu Q, Yin X, Languino LR, Altieri DC. Evaluation of drug combination effect using a Bliss independence dose-response surface model. Stat Biopharm Res. 2018;10(2):112-122. doi:10.1080/19466315.2018.1437071

# COMMAND ----------

synergy_estimated = \
  combination_prediction_frame['killrate_predicted_a'] + combination_prediction_frame['killrate_predicted_b'] - \
  combination_prediction_frame['killrate_predicted_a'] * combination_prediction_frame['killrate_predicted_b']
combination_prediction_frame['killrate_expected'] = synergy_estimated
combination_prediction_frame['synergy'] = combination_prediction_frame['killrate_predicted'] - synergy_estimated
combination_prediction_abs_max_indices = combination_prediction_frame.groupby(['cell_line', 'drug_a', 'drug_b']).apply(lambda x: x.abs().idxmax())['synergy'].tolist()
combination_prediction_frame = combination_prediction_frame.loc[combination_prediction_abs_max_indices]
combination_prediction_frame.to_csv(os.path.join(output_data_folder, f'benchmark_combination_predictions__{MODEL_TYPE}.csv'), index=False)
combination_prediction_frame

# COMMAND ----------


