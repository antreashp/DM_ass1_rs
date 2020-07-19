##############################################################
#                                                            #
#    Mark Hoogendoorn and Burkhardt Funk (2017)              #
#    Machine Learning for the Quantified Self                #
#    Springer                                                #
#    Chapter 4                                               #
#                                                            #
##############################################################

import sys
import copy
import pandas as pd
from pathlib import Path
import numpy as np
from util.VisualizeDataset import VisualizeDataset
from Chapter4.TemporalAbstraction import NumericalAbstraction
from Chapter4.TemporalAbstraction import CategoricalAbstraction
from Chapter4.FrequencyAbstraction import FourierTransformation
from Chapter4.TextAbstraction import TextAbstraction

# Read the result from the previous chapter, and make sure the index is of the type datetime.
DATA_PATH = Path('./intermediate_datafiles/')
DATASET_FNAME = sys.argv[1] if len(sys.argv) > 1 else 'chapter3_result_rest4.csv'
RESULT_FNAME = sys.argv[2] if len(sys.argv) > 2 else 'chapter4_result_final.csv'

try:
    dataset = pd.read_csv(DATA_PATH / DATASET_FNAME, index_col=0)
except IOError as e:
    print('File not found, try to run previous crowdsignals scripts first!')
    raise e

dataset.index = pd.to_datetime(dataset.index)

# Let us create our visualization class again.
DataViz = VisualizeDataset(__file__)

# Compute the number of milliseconds covered by an instance based on the first two rows
milliseconds_per_instance = 40#(dataset.index[1] - dataset.index[0]).microseconds/1000


# Chapter 4: Identifying aggregate attributes.

# First we focus on the time domain.

# Set the window sizes to the number of instances representing 5 seconds, 30 seconds and 5 minutes
window_sizes = [int(float(5000)/milliseconds_per_instance), int(float(0.5*60000)/milliseconds_per_instance)]
# dataset.index  = dataset.index.astype(int)
rng = pd.date_range('1/1/1970', periods=11371, freq='40ms')


dataset.index = rng
# for index, row in dataset.iterrows():


# meh.values = dataset.values
# meh =meh.fillna(dataset)
# print(dataset)
# print(meh)
# exit()
# a = dataset.groupby(level=0).cumcount()
# for index, row in dataset.iterrows():
#     # print(index, row)
#     print(dataset.loc[index]
#     # df.loc
#     exit()
# dataset.index = dataset.index + pd.to_timedelta(40, unit='ms')
# print(dataset)
# exit()
dataset['date_time'] =  pd.to_datetime(dataset['date_time'], format='%Y-%m-%d %H:%M:%S.%f')
dataset = dataset.drop(columns=['date_time'])
print(dataset.columns)
# dataset.reindex(dataset['date_time']) 
# dataset = dataset.resample('T').sum()
# full_idx = pd.date_range(start=dataset['date_time'].min(), end=dataset['date_time'].max(), freq='T')
# dataset = (
#  dataset
#  .apply(lambda group: group.reindex(full_idx, method='nearest')) 
#  .reset_index(level=0, drop=True) 
#  .sort_index() 
# )
# print(dataset)
dataset = dataset.drop(columns=['label'])
NumAbs = NumericalAbstraction()
dataset_copy = copy.deepcopy(dataset)
for ws in window_sizes:
    dataset_copy = NumAbs.abstract_numerical(dataset_copy, ['acc_x'], ws, 'mean')
    dataset_copy = NumAbs.abstract_numerical(dataset_copy, ['acc_x'], ws, 'std')

# print(dataset_copy.columns)
# print(dataset_copy)
# exit()
DataViz.plot_dataset(dataset_copy, ['acc_x', 'acc_x_temp_mean', 'acc_x_temp_std', 'label'], ['like', 'like', 'like', 'like'], ['line', 'line', 'line', 'points'])

ws = int(float(0.5*60000)/milliseconds_per_instance)
selected_predictor_cols = [c for c in dataset.columns if not 'label' in c]
dataset = NumAbs.abstract_numerical(dataset, selected_predictor_cols, ws, 'mean')
dataset = NumAbs.abstract_numerical(dataset, selected_predictor_cols, ws, 'std')

DataViz.plot_dataset(dataset, ['acc_x','acc_y','acc_z', 'pca_1', 'label'], ['like', 'like', 'like','like','like'], ['line', 'line', 'line','line', 'points'])


CatAbs = CategoricalAbstraction()
dataset = CatAbs.abstract_categorical(dataset, ['label'], ['like'], 0.03, int(float(5*60000)/milliseconds_per_instance), 2)

# Now we move to the frequency domain, with the same window size.

FreqAbs = FourierTransformation()
fs = float(1000)/milliseconds_per_instance

periodic_predictor_cols = ['acc_x','acc_y','acc_z']
data_table = FreqAbs.abstract_frequency(copy.deepcopy(dataset), ['acc_x'], int(float(10000)/milliseconds_per_instance), fs)

# Spectral analysis.

DataViz.plot_dataset(data_table, ['acc_x_max_freq', 'acc_x_freq_weighted', 'acc_x_pse', 'label'], ['like', 'like', 'like', 'like'], ['line', 'line', 'line','points'])

dataset = FreqAbs.abstract_frequency(dataset, periodic_predictor_cols, int(float(10000)/milliseconds_per_instance), fs)

# Now we only take a certain percentage of overlap in the windows, otherwise our training examples will be too much alike.

# The percentage of overlap we allow
window_overlap = 0.9
skip_points = int((1-window_overlap) * ws)
dataset = dataset.iloc[::skip_points,:]


dataset.to_csv(DATA_PATH / RESULT_FNAME)

DataViz.plot_dataset(dataset, ['acc_x','acc_y','acc_z', 'pca_1', 'label'], ['like', 'like', 'like', 'like','like'], ['line', 'line', 'line', 'line',  'points'])
