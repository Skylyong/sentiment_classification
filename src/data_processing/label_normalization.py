'''
@Author: leon
@Date: 2024年04月25日
@Description:
    gpt标注的label和我们在label set中定义的label不完全一致, 这里将它们对齐, 让数据的标签都在定义的标签集中
    

'''


import sys
sys.path.append('./src')
from model_building.train_args import LABEL_SET, LABELS_MAP, DROP_LABELS

assert len(LABEL_SET) == 15

import numpy as np

data = np.load('./DATA_SET/data_processed.npy', allow_pickle=True).item()

print(f'LABEL_SET: {LABEL_SET}')

def normalize_label(datas):
    
    for key in ['train', 'val', 'test']:
        data = datas[key]
        label_not_in = [
        ] 
        for d in data:
            label = d['label']
            new_label = []
            for l in label:
                l = l.lower()
                l = LABELS_MAP.get(l, l)
                if l not in DROP_LABELS:
                    new_label.append(l)
                
                if l not in LABEL_SET and l not in DROP_LABELS:
                    label_not_in.append(l)
            new_label = list(set(new_label))
            d['label'] = new_label
    

    np.save('./DATA_SET/data_processed_normalized.npy', datas)
    

normalize_label(data)