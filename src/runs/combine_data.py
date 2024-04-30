
file1 = '/data/User_Hannah/temp/sentiment_classification/DATA_SET/data_processed_normalized.npy'
file2 = '/data/User_Hannah/new_data/processed_data.pt'


import pandas as pd
import numpy as np
import torch

data1 = np.load(file1, allow_pickle=True).item()
data2 = torch.load(file2)

print(data1.keys())
print(data2.keys())

data1['pred'] = data2['train'] + data2['val'] + data2['test']


np.save('/data/User_Hannah/temp/sentiment_classification/DATA_SET/data_processed_normalized_total.npy', data1) 