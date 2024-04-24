
dir_ = "/root/autodl-tmp/xianyu/sentiment_classification/data_labeled"
import glob
import json
import numpy as np


text_files = glob.glob(dir_ + "/txt_docments/*/*.json")
npy_files = glob.glob(dir_ + "/audio/*/*.npy")

def is_file_in(file_name, files, category):
    
    
    for file in files:
        
        npy_file_name = file.split("/")[-1].split(".")[0]
        npy_category = file.split("/")[-2]
        if file_name == npy_file_name and category == npy_category:
            return True
    return False



all_data = []
count_ = 0
for file in text_files:
    file_name = file.split("/")[-1].split(".")[0]
    category = file.split("/")[-2]
    
    if is_file_in(file_name, npy_files, category):
        data = json.load(open(file))
        new_data = {}
        new_data["text"] = data["text"]
        new_data['label'] = data['label']
        
        audio_file = file.replace("txt_docments", "audio").replace(".json", ".npy")
        new_data["audio_embedding"] = np.load(audio_file, allow_pickle=True)
        
        category = file.split("/")[-2]
        new_data['category'] = category
        
        all_data.append(new_data)
        count_ += 1

print(count_)

# np.save("total_data.npy", all_data)

# 将all_data分为训练集和测试集、验证集，比例为8:1:1
import random
random.shuffle(all_data)
train_data = all_data[:int(0.8*len(all_data))]
val_data = all_data[int(0.8*len(all_data)):int(0.9*len(all_data))]
test_data = all_data[int(0.9*len(all_data)):]

data_dict = {
    'train': train_data,
    'val': val_data,
    'test': test_data
}





# save data
# np.save("data_dict.npy", data_dict)
        
    

def print_dict_structure(d, indent=0):
    for key, value in d.items():
        print('\t' * indent + str(key))
        if isinstance(value, dict):
            print_dict_structure(value, indent+1)
        elif isinstance(value, list):
            
            n = len(value)
            if key != 'label':
                print('\t' * (indent+1) + f'{n}' )
            if n > 0 and isinstance(value[0], dict):
                print_dict_structure(value[0], indent+2)

print_dict_structure(data_dict)


# print(data_dict['train'][0]['audio_embedding'])

# 打印data_dict的结构
# print(data_dict.keys())