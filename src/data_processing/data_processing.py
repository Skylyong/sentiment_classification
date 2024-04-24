import glob
import os

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


from docx import Document
import json



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
import jieba

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
import subprocess

def mp3_and_aac_to_wav():
    audio_path = '/root/autodl-tmp/xianyu/sentiment_classification/data/audio'
    for file in glob.glob(audio_path + '/*/*.*'):
        if 'mp3' in file or 'aac' in file:
            print(file)
            file_name = file.split('/')[-1]
            new_file_name = file_name.replace('.mp3', '.wav').replace('.aac', '.wav')
            new_file = file.replace(file_name, new_file_name)
            print(new_file)
            subprocess.run(['ffmpeg', '-i', file, new_file])
            # os.remove(file)
        else:
            print('no need to convert', file)

# mp3_and_aac_to_wav()

def view_audio():
    audio_path = '/root/autodl-tmp/xianyu/sentiment_classification/test.wav'
    # 查看音频采样率
    subprocess.run(['ffprobe', '-i', audio_path])
    
    audio_path = '/root/autodl-tmp/xianyu/sentiment_classification/data/audio/skin_care/D7-63.MP3'
    print('\n\n\n')
    subprocess.run(['ffprobe', '-i', audio_path])
    
    # 将audio_path转换为wav格式, 并且设置采样率为16000
    new_audio_path = audio_path.replace('.MP3', '.wav')
    subprocess.run(['ffmpeg', '-i', audio_path, '-ar', '16000', new_audio_path])
    print('\n\n\n')
    subprocess.run(['ffprobe', '-i', new_audio_path])
    
    # 查看new_audio_path和audio_path占用的空间
    print('\n\n\n')
    subprocess.run(['du', '-h', audio_path])
    print('\n\n\n')
    subprocess.run(['du', '-h', new_audio_path])
    
    # 截断new_audio_path, 只取前面10min
    new_audio_path_ = new_audio_path.replace('.wav', '_50min.wav')
    subprocess.run(['ffmpeg', '-i', new_audio_path, '-t', '00:05:00', new_audio_path_])
view_audio()