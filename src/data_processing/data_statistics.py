import numpy as np
import torch
import pandas as pd
import jieba

data_part1 = np.load('./DATA_SET/data_processed_normalized.npy', allow_pickle=True).item()
data_part2 = torch.load('/data/User_Hannah/new_data/processed_data.pt')


texts = []
for key in ['train', 'val', 'test']:
   for data in data_part1[key]:
        text = data['text']
        category = data['category']  
        file_name = data['audio_embedding'].item()['file_name'].split('/')[-1].split('.')[0]
        # print(data['audio_embedding'].item().keys())
        # break
        # print(file_name)
        split_text = jieba.lcut(text)
        length = len(split_text)
        
        temp = {
            'original_text': text,
            'category': category,
            'length': length,
            'file_name': file_name,
            'participled_text': ' '.join(split_text),
            'part': 'part1',
            'length': length
        }
        texts.append(temp)
        # break

# n = len(texts)



# print(len(texts))
        # break

# print(data_part2)

for key in ['train', 'val','test']:
    for data in data_part2[key]:
        text = data['text']
        category = data['category']
        file_name = data['audio_embedding']['file_name'].split('/')[-1].split('.')[0]
        split_text = jieba.lcut(text)
        length = len(split_text)
        temp = {
            'original_text': text,
            'category': category,
            'length': length,
            'file_name': file_name,
            'participled_text': ' '.join(split_text),
            'part': 'part2',
            'length': length
        }
        texts.append(temp)
        # break


# print(len(texts) -n)
# save texts to excel
df = pd.DataFrame(texts)
df.to_excel('DATA_SET/texts.xlsx', index=False)




# {
#             'original_text': str, # original text
#             'category': str, # category
#             'length': int, # length of participled text
#             'file_name': str, # file name
#             'participled_text': str, # participled text
#             'part': str # part of data, part1 refers to data_part1, part2 refers to data_part2 which added later
# }