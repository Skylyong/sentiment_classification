import matplotlib.pyplot as plt

import numpy as np

import jieba

data = np.load('DATA_SET/data_processed.npy', allow_pickle=True).item()


train_data = data['train']

text_length_counts = []
for idx, data in enumerate(train_data):
    text = data['text']
    
    # 对text做分词
    
    text = jieba.cut(text)
    text = list(text)    
    text_length_counts.append(len(text))
    
    if (idx+1) % 10 == 0:
        print(f'Processed {idx+1} samples')

# sort the text length counts in ascending order
text_length_counts = sorted(text_length_counts)

# plot the text length as plot

plt.figure(figsize=(10, 6))

plt.plot(text_length_counts)

plt.xlabel('Text Index')
plt.ylabel('Text Length')

plt.savefig('text_length_distribution.png')