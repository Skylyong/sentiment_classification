'''
@Author: leon
@Date: 2024年04月15日17:35:45
@Description:
    将中文命名的文件名改为英文命名的文件名，并将文本文件和音频文件对应上, 将word格式的文档转换为txt格式的文档
    输入：
        存储到本地 DATA_DIR 路径下面的音频文件和文本文件，路径的组织方式为：
        - DATA_DIR
            - audio
                - dress
                    - file1.mp3
                    - file2.mp3
                    - ...
                - food
                    - file1.mp3
                    - file2.mp3
                    - ...
                - make_up
                    - file1.mp3
                    - file2.mp3
                    - ...
                - skin_care
                    - file1.mp3
                    - file2.mp3
                    - ...
            - word_docments
                - dress
                    - file1.docx
                    - file2.docx
                    - ...
                - food
                    - file1.docx
                    - file2.docx
                    - ...
                - make_up
                    - file1.docx
                    - file2.docx
                    - ...
                - skin_care
                    - file1.docx
                    - file2.docx
                    - ...
        要特别注意的是，音频文件和文本文件的文件名是对应的，即文件名相同，只是后缀不同。处理的时候以文本文件的文件名为准，去查找对应的类别下面查找
        音频文件，如果找不到对应音频文件，则跳过该条数据
    输出：
        将文本文件和音频文件对应上，将对应上的文件存储到processed_data.pt文件中，将没有对应上的文件存储到not_in.txt文件中，输出文件条目 + 没有对应
        的文件条目 = 所有文本文件条目
        
        processed_data.pt 文件的格式为：
        [
            {
                'text': 'xxx', 
                'audio_embedding': dict,
                'category': 'xxx',
                'idx': int,
                'name': 'xxx',
            },
            ...
        ]
       
        其中
            audio_embedding的格式为：
            {
                'embeddings': list,
                'label_emotion': list, 
            }

        not_in.txt文件的格式为：
        [
            'xxx',
            ...
        ]
'''




from funasr import AutoModel
import glob
import os
from docx import Document
import json
import jieba
import torch
from audio_embed import get_audio_to_embed
import tqdm

DATA_DIR = '/data/User_Hannah/new_data'




class data_process:
    def __init__(self, category = None) -> None:
        self.category = category if category else self.get_category()
        self.audio_model =AutoModel(model="iic/emotion2vec_base_finetuned")
    
    def get_category(self, ):
        return os.listdir(DATA_DIR + '/audio')

    def convert_docx_to_text(self, docx:Document):
        res_text = []
        for para in docx.paragraphs:
            res_text.append(para.text)
        res_text = '\n'.join(res_text)
        return res_text
    
    def get_all_audio_files(self, ):
        audio_files = glob.glob(DATA_DIR + '/audio/*/*.*3') # 只匹配MP3格式
        return audio_files
    def get_all_word_files(self, ):
        word_files = glob.glob(DATA_DIR + '/word_docments/*/*.docx')
        return word_files
    
    def get_category_audio_files(self, audio_files):
        category_audio_files = {}
        for category in self.category:
            category_audio_files[category] = []
            for file in audio_files:
                if category in file:
                    category_audio_files[category].append(file)
        return category_audio_files
    
    def c_text_file_in_audio_files(self, c_text_file_name, category_audio_files):
            '''
            检查在类别下面有没有和text_file_name对应的音频文件
            如果有就返回True, 并且返回找到的音频文件的名字,否则返回False和None
            '''
            category = c_text_file_name.split('/')[-2]
            file_name = c_text_file_name.split('/')[-1].split('.')[0].lower()
            for file in category_audio_files[category]:
                file_name_ = file.split('/')[-1].split('.')[0].lower()
                if file_name == file_name_:
                    return True, file
            return False, None
            
    def get_audio_embedding(self, audio_file):
        return get_audio_to_embed(audio_file, self.audio_model)
    
    
        
    
    def run(self, ):
        
        audio_files = self.get_all_audio_files()
        docx_files = self.get_all_word_files()
       
        print("音频文件数量:", len(audio_files))
        print("文本文件数量:", len(docx_files))
        
        
        category_audio_files = self.get_category_audio_files(audio_files)
        
        processed_data = []
        not_in = []
        idx = 0
        not_get_embedding = []
        
        
        for file in tqdm.tqdm( docx_files):
            file_orginal = file
            file = file.replace('_原文', '')
            is_find_audio, audio_file = self.c_text_file_in_audio_files(file, category_audio_files)
            if is_find_audio:
                audio_embedding = self.get_audio_embedding(audio_file)
                
                if audio_embedding is None:
                    not_get_embedding.append(file)
                else:
                    text = self.convert_docx_to_text(Document(file_orginal))
                    
                    category = file.split('/')[-2]
                    name = file.split('/')[-1].split('.')[0]
                    
                    tmp_dict = {
                        'text': text,
                        'audio_embedding': audio_embedding,
                        'category': category,
                        'idx': idx,
                        'name': name,
                    }
                    processed_data.append(tmp_dict)
                    
                    
                    idx += 1
            else:
                not_in.append(file)
            # break
          
        
        # processed_data = self.split_data(processed_data)
        torch.save(processed_data, DATA_DIR + '/processed_data.pt')
        json.dump(not_in, open(DATA_DIR + '/not_find_audio.txt', 'w', encoding='utf-8'), indent=4)
        json.dump(not_get_embedding, open(DATA_DIR + '/not_get_embedding.txt', 'w', encoding='utf-8'), indent=4)
        
        
        
    def split_data(self, data:list):
        '''
        将数据分为训练集、验证集、测试集
        '''
        import random
        random.shuffle(data)
        train_data = data[:int(0.8*len(data))]
        val_data = data[int(0.8*len(data)):int(0.9*len(data))]
        test_data = data[int(0.9)*len(data):]
        
        data_dict = {
            'train': train_data,
            'val': val_data,
            'test': test_data
        }
        return data_dict


def load_data():
    file_name = os.path.join(DATA_DIR, 'processed_data.pt')
    data = torch.load(file_name)
    # print data
    print(data.keys())
    
if __name__ == '__main__':
    dp = data_process()
    # c = dp.get_category()
    # print(c)
    dp.run()
    print('数据处理完成！')
    load_data()
    