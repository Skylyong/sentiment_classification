'''
@Author: leon
@Date: 2024年04月15日17:35:45
@Description:
    将中文命名的文件名改为英文命名的文件名，并将文本文件和音频文件对应上, 将word格式的文档转换为txt格式的文档
    输入：
        存储到本地xxx路径下面的音频文件和文本文件，路径的组织方式为：
        - audio
            - category1
                - file1.mp3
                - file2.mp3
                - ...
            - category2
                - file1.mp3
                - file2.mp3
                - ...
            - ...
        - txt_docments
            - category1
                - file1.docx
                - file2.docx
                - ...
            - category2
                - file1.docx
                - file2.docx
                - ...
            - ...
    输出：
        将文本文件和音频文件对应上，将对应上的文件存储到processed_data.json文件中，将没有对应上的文件存储到not_in.txt文件中
        
        processed_data.json文件的格式为：
        [
            {
                'text': 'xxx', 
                'audio_file': 'xxx',
                'category': 'xxx',
                'idx': 0
            },
            ...
        ]
'''



import glob
import os
from docx import Document
import json
import jieba

# 获取文件夹下面的所有文件

def rename():
    '''
    将文件名中的中文去除
    '''
    for file in glob.glob('/root/autodl-tmp/xianyu/sentiment_classification/data/docments/make_up/*'):
        if 'D' not in file:
            file_name = file.split('/')[-1]
            new_file_name = 'D1-' + file_name
            # print(new_file_name)
            os.rename(file, '/root/autodl-tmp/xianyu/sentiment_classification/data/docments/make_up/' + new_file_name)
        if '原文' in file:
            file_name = file.split('/')[-1]
            new_file_name = file_name.replace('_原文', '')
            # print(new_file_name)
            os.rename(file, '/root/autodl-tmp/xianyu/sentiment_classification/data/docments/make_up/' + new_file_name)


    co = ['dress', 'food', 'skin_care']

    dir_ = '/root/autodl-tmp/xianyu/sentiment_classification/data/docments/'

    for c in co:
        sub_dir = dir_ + c + '/'
        for file in glob.glob(sub_dir + '*'):
            if '原文' in file:
                file_name = file.split('/')[-1]
                new_file_name = file_name.replace('_原文', '')
                # print(new_file_name)
                os.rename(file, sub_dir + new_file_name)






def convert_docx_to_text(docx_filename, txt_filename):
    doc = Document(docx_filename)
    with open(txt_filename, 'w', encoding='utf-8') as txt_file:
        for para in doc.paragraphs:
            txt_file.write(para.text + '\n')

def word2txt():
    word_path = '/root/autodl-tmp/xianyu/sentiment_classification/data/word_docments'
    for file in glob.glob(word_path + '/*/*.docx'):
        c = file.split('/')[-2]
        file_name = file.split('/')[-1]
        txt_file_name = file_name.replace('.docx', '.txt')
        
        txt_dir = '/root/autodl-tmp/xianyu/sentiment_classification/data/txt_docments' + '/'+c+'/'
        if not os.path.exists(txt_dir):
            os.makedirs(txt_dir)
        # print(txt_dir)
        
        txt_file_name = txt_dir  + txt_file_name
        
        # print(file)
        # print(txt_file_name)
        # break
        
        convert_docx_to_text(file, txt_file_name)

# word2txt()

# 统计'/root/autodl-tmp/xianyu/sentiment_classification/data/txt_docments'下面的所有文件的数量，以及文件中的字符数量


def count_file():
    sum_ = 0
    count_file = 0
    split_sum = 0
    for file in glob.glob('/root/autodl-tmp/xianyu/sentiment_classification/data/txt_docments/*/*.txt'):
        with open(file, 'r', encoding='utf-8') as f:
            content = f.read()
            sum_ += len(content)
            split_content = jieba.cut(content)
            split_sum += len(list(split_content))
            
        count_file += 1
    print('文件数量:', count_file)
    print('字符数量:', sum_)
    print('词组数量:', split_sum)

# count_file()

# 查看音频文件和文本文件是否对应上

def check_audio_text():
    
    def c_text_file_in_audio_files(c_text_file_name, category_audio_files):
        for c_audio_file in category_audio_files:
            c_audio_file_name = c_audio_file.split('/')[-1]
            c_text_file_name = c_text_file_name.split('.')[0].strip().lower()
            c_audio_file_name = c_audio_file_name.split('.')[0].strip().lower()
            
            if c_text_file_name== c_audio_file_name:
                return (True, c_audio_file)
        return (False,None)
    
    text_path = '/root/autodl-tmp/xianyu/sentiment_classification/data/txt_docments'
    audio_path = '/root/autodl-tmp/xianyu/sentiment_classification/data/audio'
    both = []
    
    audio_files = glob.glob(audio_path + '/*/*.*')
    text_files = glob.glob(text_path + '/*/*.txt')
    print("音频文件数量:", len(audio_files))
    print("文本文件数量:", len(text_files))
    
    
    category = ['dress', 'food', 'skin_care', 'make_up']
    not_in  = []
    processed_data = []
    idx = 0
    for c in category:
        category_audio_files = glob.glob(audio_path + '/' + c + '/*.*')
        category_text_files = glob.glob(text_path + '/' + c + '/*.txt')
        print(c, "音频文件数量:", len(category_audio_files))
        print(c, "文本文件数量:", len(category_text_files))
        c_both = []
        
        
        for c_text_file  in category_text_files:
            c_text_file_name = c_text_file.split('/')[-1]
            
            
            if 'D' not in c_text_file_name:
                c_text_file_name = 'D1-' + c_text_file_name
                # print('文件名不对', c_text_file_name, '->', c_text_file_name_)
            
            is_match , match_audio_file = c_text_file_in_audio_files(c_text_file_name, category_audio_files)
            
            if is_match:
                c_both.append(c_text_file)
                one_sample = {
                    'text_file': c_text_file,
                    'audio_file': match_audio_file,
                    'category': c,
                    'idx': idx
                }
                idx += 1
                processed_data.append(one_sample)
                
            else:
                # print(c,c_text_file_name)
                not_in.append(f'{c} {c_text_file_name}')
        both.extend(c_both)
        print(c, "音频文本文件对应数量:", len(c_both), end='\n\n')
    print("音频文本文件对应数量:", len(both))
    
    # 将not_in文件写入到文件中
    with open('not_in.txt', 'w', encoding='utf-8') as f:
        for file in not_in:
            f.write(file + '\n')
    
    json.dump(processed_data, open('processed_data.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=4)
        

# check_audio_text()

# mp3 and aac to wav
