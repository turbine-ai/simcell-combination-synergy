# Databricks notebook source
import os
import pandas as pd
import numpy as np

from scipy.stats import pearsonr
from scipy.stats import bootstrap
from  sklearn.mixture import GaussianMixture
from sklearn.metrics import auc, roc_curve, balanced_accuracy_score

import statsmodels.api as sm
from statsmodels.formula.api import ols

from plotly import graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

# COMMAND ----------

input_turbine_data_folder =  './az_benchmark/data/raw_in_silico_output'
input_benchmark_data_folder =  './az_benchmark/data/benchmark_predictions'
input_dream_data_folder = './az_benchmark/data/dream_data'
input_meta_data_folder = './az_benchmark/data/meta_data'
output_figure_folder = './az_benchmark/figures/'

# COMMAND ----------

# MAGIC %md
# MAGIC ## Combination performance results

# COMMAND ----------

dream_challenge_data = pd.read_csv(os.path.join(input_dream_data_folder, 'dream_drug_combinations_supplementary.csv'))
dream_challenge_data = dream_challenge_data[dream_challenge_data['QA'] != -1]
dream_challenge_data.insert(loc=1, column='cell_line', value=dream_challenge_data['Cell line name'].map(lambda cell_line_name: cell_line_name.replace('-', '')))

dream_to_turbine_compound_name = pd.read_csv(os.path.join(input_meta_data_folder, 'dream_to_turbine_compound_name.csv'))
dream_to_turbine_compound_name = dict(zip(dream_to_turbine_compound_name['Challenge drug name'], dream_to_turbine_compound_name['Turbine name']))
dream_challenge_data.insert(loc=3, column='drug_a', value=dream_challenge_data['Compound A'].map(dream_to_turbine_compound_name))
dream_challenge_data.insert(loc=5, column='drug_b', value=dream_challenge_data['Compound B'].map(dream_to_turbine_compound_name))

dream_challenge_data = dream_challenge_data.rename({'Synergy score ': 'synergy_true'}, axis=1)
dream_challenge_data = dream_challenge_data[['cell_line', 'drug_a', 'drug_b', 'synergy_true']]
dream_challenge_data = dream_challenge_data.dropna()

dream_challenge_data.insert(loc=0, column='drug_id', value=dream_challenge_data.apply(lambda row: ':'.join(sorted([row['drug_a'], row['drug_b']])), axis=1))
dream_challenge_data = dream_challenge_data.drop(['drug_a', 'drug_b'], axis=1)
dream_challenge_data

# COMMAND ----------

turbine_model_combination_synergy_frame = pd.read_csv(os.path.join(input_turbine_data_folder, 'combi_metrics_syn_astra_full.txt')) 
turbine_model_combination_synergy_frame = turbine_model_combination_synergy_frame.rename({'Cell_line': 'cell_line', 'Drug_1': 'drug_a', 'Drug_2': 'drug_b', 'Max_syn': 'synergy'}, axis=1)
turbine_model_combination_synergy_frame = turbine_model_combination_synergy_frame[['cell_line', 'drug_a', 'drug_b', 'synergy']]

turbine_model_combination_antagonism_frame = pd.read_csv(os.path.join(input_turbine_data_folder, 'combi_metrics_ant_astra_full.txt'))
turbine_model_combination_antagonism_frame = turbine_model_combination_antagonism_frame.rename({'Cell_line': 'cell_line', 'Drug_1': 'drug_a', 'Drug_2': 'drug_b', 'Max_syn': 'antagonism'}, axis=1)
turbine_model_combination_antagonism_frame = turbine_model_combination_antagonism_frame[['cell_line', 'drug_a', 'drug_b', 'antagonism']]

turbine_model_combination_frame = turbine_model_combination_synergy_frame.merge(
  turbine_model_combination_antagonism_frame,
  how='inner',
  on=['cell_line', 'drug_a', 'drug_b']
)
turbine_model_combination_frame['synergy_turbine'] = turbine_model_combination_frame.apply(lambda row: row['synergy'] if row['synergy'] > row['antagonism'] else -row['antagonism'], axis=1)
turbine_model_combination_frame.insert(loc=0, column='drug_id', value=turbine_model_combination_frame.apply(lambda row: ':'.join(sorted([row['drug_a'], row['drug_b']])), axis=1))
turbine_model_combination_frame = turbine_model_combination_frame.drop(['drug_a', 'drug_b', 'synergy', 'antagonism'], axis=1)

turbine_model_combination_frame

# COMMAND ----------

benchmark_combination_frame = None
for model_type in ['linear', 'neural_network', 'lightgbm']:
  benchmark_combinations_for_model_frame = pd.read_csv(os.path.join(input_benchmark_data_folder, f'benchmark_combination_predictions__{model_type}.csv'))
  benchmark_combinations_for_model_frame = benchmark_combinations_for_model_frame.rename({'synergy': f'synergy_{model_type}'}, axis=1)

  benchmark_combinations_for_model_frame.insert(loc=0, column='drug_id', value=benchmark_combinations_for_model_frame.apply(lambda row: ':'.join(sorted([row['drug_a'], row['drug_b']])), axis=1))
  benchmark_combinations_for_model_frame = benchmark_combinations_for_model_frame[['drug_id', 'cell_line', f'synergy_{model_type}']]
  
  if benchmark_combination_frame is None:
    benchmark_combination_frame = benchmark_combinations_for_model_frame.copy()
  else:
    benchmark_combination_frame = benchmark_combination_frame.merge(
      benchmark_combinations_for_model_frame,
      on=['drug_id', 'cell_line'],
      how='inner'
    )

benchmark_combination_frame

# COMMAND ----------

dream_challenge_data = dream_challenge_data.merge(
  turbine_model_combination_frame,
  on=['cell_line', 'drug_id'],
  how='inner'
).merge(
  benchmark_combination_frame,
  on=['cell_line', 'drug_id'],
  how='inner'
)
dream_challenge_data

# COMMAND ----------

# MAGIC %md
# MAGIC Used bootstrap to produce confidence intervals for combination predictions based on this article: Thomas J. DiCiccio. Bradley Efron. "Bootstrap confidence intervals." Statist. Sci. 11 (3) 189 - 228, August 1996. https://doi.org/10.1214/ss/1032280214 

# COMMAND ----------

models_x = []
performance_y = []
performance_error_y = []
performance_error_minus_y = []

for model_type in ['turbine', 'neural_network', 'linear', 'lightgbm']:
  dream_challenge_data_filtered = dream_challenge_data[['cell_line', 'drug_id', 'synergy_true', f'synergy_{model_type}']].dropna()
  
  dream_challenge_data_filtered['synergy_true_discretized'] = (dream_challenge_data_filtered['synergy_true'] > 30.0).astype(int)

  synergy_predictions = dream_challenge_data_filtered[f'synergy_{model_type}'].to_numpy()
  dream_challenge_data_filtered['synergy_prediction_discretized'] = (synergy_predictions > 0.2).astype(int)

  models_x.append({
    'turbine': 'Simulated Cell',
    'neural_network': 'Neural Network',
    'linear': 'Linear Regression',
    'lightgbm': 'LightGBM'
  }[model_type])
  
  performance_y.append(balanced_accuracy_score(dream_challenge_data_filtered['synergy_true_discretized'], dream_challenge_data_filtered['synergy_prediction_discretized'], adjusted=False))

  confidence_low, confidence_high = bootstrap(
    (dream_challenge_data_filtered['synergy_true_discretized'],
     dream_challenge_data_filtered['synergy_prediction_discretized']),
  statistic=lambda x, y: balanced_accuracy_score(x, y, adjusted=False), n_resamples=100, paired=True, vectorized=False).confidence_interval
  performance_error_y.append(confidence_high - performance_y[-1])
  performance_error_minus_y.append(performance_y[-1] - confidence_low)

fig = go.Figure()
fig.add_trace(
  go.Bar(
    x=models_x,
    y=performance_y,
    error_y=dict(type='data', symmetric=False, array=performance_error_y, arrayminus=performance_error_minus_y)
  )
)
fig.add_hline(y=0.5, line_dash='dash')
fig.update_yaxes(title='Balanced Accuracy Score', range=[0.0, 1.0])
fig.update_xaxes(title='In Silico Method')
fig.write_image(os.path.join(output_figure_folder, 'balanced_accuracy_combi_benchmark_performance.svg'))
fig.write_html(os.path.join(output_figure_folder, 'balanced_accuracy_combi_benchmark_performance.html'))
fig 

# COMMAND ----------

# MAGIC %md
# MAGIC ## Mono performance results

# COMMAND ----------

turbine_model_killrates_frame = pd.read_csv(os.path.join(input_turbine_data_folder, 'merged_killrates.csv'))
turbine_model_killrates_frame

# COMMAND ----------

fig = make_subplots(rows=1, cols=3, shared_yaxes=False, column_titles=['Cell Line Split', 'Drug Split', 'Cell Line & Drug Split'], row_titles=['Overall', 'DDR Drugs', 'non-DDR Drugs'])
plot_idx = 1
for split_type in ['cell_line', 'drug', 'cell_line__drug']:
  
  model_idx = 1
  for model_type in ['neural_network', 'linear', 'lightgbm']:
    benchmark_model_killrates_frame = pd.read_csv(os.path.join(input_benchmark_data_folder, f'benchmark_predictions__{model_type}__{split_type}.csv'))

    killrates_merged_frame = benchmark_model_killrates_frame.merge(
      turbine_model_killrates_frame[['cell_line', 'drug', 'dose', 'drug_category', 'killrate']].rename({'killrate': 'killrate_turbine'}, axis=1),
      on=['cell_line', 'drug', 'dose'],
      how='inner'
    )
    turbine_roc_aucs = []
    benchmark_roc_aucs = []
    for split_idx in killrates_merged_frame['split_index'].unique():

      killrate_filter = (killrates_merged_frame['split_index'] == split_idx) & (killrates_merged_frame['split'] == 'TEST')
      killrates_merged_filtered_frame = killrates_merged_frame[killrate_filter]

      fpr, tpr, thresholds = roc_curve((killrates_merged_filtered_frame['killrate_true'] > 0.5).astype(int), killrates_merged_filtered_frame['killrate_turbine'])
      roc_auc = auc(fpr, tpr)
      turbine_roc_aucs.append(roc_auc)

      fpr, tpr, thresholds = roc_curve((killrates_merged_filtered_frame['killrate_true'] > 0.5).astype(int), killrates_merged_filtered_frame['killrate_predicted'])
      roc_auc = auc(fpr, tpr)
      benchmark_roc_aucs.append(roc_auc)

    if model_type == 'neural_network':

      fig.add_trace(
        go.Box(
          y=turbine_roc_aucs,
          name='Simulated Cell',
          marker_color=px.colors.qualitative.Plotly[0]
        ),
        row=1, col=plot_idx
      )

    model_name = {
      'turbine': 'Simulated Cell',
      'neural_network': 'Neural Network',
      'linear': 'Linear Regression',
      'lightgbm': 'LightGBM'
    }[model_type]

    fig.add_trace(
      go.Box(
        y=benchmark_roc_aucs,
        name=model_name,
        marker_color=px.colors.qualitative.Plotly[model_idx]
      ),
      row=1, col=plot_idx
    )

    model_idx += 1
  
  plot_idx += 1

fig.update_yaxes(title='ROC AUC Scores', range=[0.0, 1.0])
fig.update_xaxes(title='In Silico Method')
fig.update_layout(title='ROC AUC Scores by Thresholding Killrate at 0.5', showlegend=False, autosize=False, width=1500, height=500)
fig.write_image(os.path.join(output_figure_folder, 'roc_auc_score_for_mono_kllrates.svg'))
fig.write_html(os.path.join(output_figure_folder, 'roc_auc_score_for_mono_kllrates.html'))
fig

# COMMAND ----------

turbine_model_ic50s_frame = pd.read_csv(os.path.join(input_turbine_data_folder, 'turbine_ic50_predictions.csv'))
turbine_model_ic50s_frame

# COMMAND ----------

dream_mono_frame = pd.read_csv(os.path.join(input_dream_data_folder, 'dream_mono_drug_statistics.csv'))
dream_mono_frame

# COMMAND ----------

fig = make_subplots(rows=1, cols=3, shared_yaxes=False, column_titles=['Cell Line Split', 'Drug Split', 'Cell Line & Drug Split'], row_titles=['Overall', 'DDR Drugs', 'non-DDR Drugs'])
plot_idx = 1
for split_type in ['cell_line', 'drug', 'cell_line__drug']:
  
  model_idx = 1
  for model_type in ['neural_network', 'linear', 'lightgbm']:
    benchmark_model_ic50s_frame = pd.read_csv(os.path.join(input_benchmark_data_folder, f'benchmark_ic50_predictions__{model_type}__{split_type}.csv'))

    benchmark_model_ic50s_frame = benchmark_model_ic50s_frame.merge(
      dream_mono_frame[['cell_line', 'drug', 'ic50']].rename({'ic50': 'ic50_true'}, axis=1),
      on=['cell_line', 'drug'],
      how='inner'
    )
    ic50s_merged_frame = benchmark_model_ic50s_frame.merge(
      turbine_model_ic50s_frame[['cell_line', 'drug', 'drug_category', 'ic50_predictions']].rename({'ic50_predictions': 'ic50_turbine'}, axis=1),
      on=['cell_line', 'drug'],
      how='inner'
    )

    turbine_correlations = []
    benchmark_correlations = []
    for split_idx in ic50s_merged_frame['split_index'].unique():

      ic50_filter = (ic50s_merged_frame['split_index'] == split_idx) & (ic50s_merged_frame['split'] == 'TEST')
      ic50s_merged_filtered_frame = ic50s_merged_frame[ic50_filter]

      ic50s_to_correlate = ic50s_merged_filtered_frame[['ic50_true','ic50_turbine']].dropna()
      turbine_correlations.append(pearsonr(ic50s_to_correlate['ic50_true'], ic50s_to_correlate['ic50_turbine'])[0])
      ic50s_to_correlate = ic50s_merged_filtered_frame[['ic50_true','ic50_predictions']].dropna()
      benchmark_correlations.append(pearsonr(ic50s_to_correlate['ic50_true'], ic50s_to_correlate['ic50_predictions'])[0])

    if model_type == 'neural_network':
      fig.add_trace(
        go.Box(
          y=turbine_correlations,
          name='Simulated Cell',
          marker_color=px.colors.qualitative.Plotly[0]
        ),
        row=1, col=plot_idx
      )

    model_name = {
      'turbine': 'Simulated Cell',
      'neural_network': 'Neural Network',
      'linear': 'Linear Regression',
      'lightgbm': 'LightGBM'
    }[model_type]

    fig.add_trace(
      go.Box(
        y=benchmark_correlations,
        name=model_name,
        marker_color=px.colors.qualitative.Plotly[model_idx]
      ),
      row=1, col=plot_idx
    )

    model_idx += 1
  
  plot_idx += 1

fig.update_yaxes(title='IC50 Correlation', range=[-1.0, 1.0])
fig.update_xaxes(title='In Silico Method')
fig.update_layout(title='IC50 Correlations', showlegend=False, autosize=False, width=1500, height=500)
fig.write_image(os.path.join(output_figure_folder, 'correlation_for_mono_ic50s.svg'))
fig.write_html(os.path.join(output_figure_folder, 'correlation_for_mono_ic50s.html'))
fig

# COMMAND ----------


