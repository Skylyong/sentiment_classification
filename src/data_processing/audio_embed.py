from src.funasr import AutoModel
import os
import time
import glob
import tqdm
import numpy as np
import torch
# # os.environ['CUDA_VISIBLE_DEVICES'] = '-1'



model =AutoModel(model="iic/emotion2vec_base_finetuned")


def get_audio_to_embed(mp3_file):
    try:
        res = model.generate(mp3_file, output_dir=None, granularity="utterance", extract_embedding=True)
        dir_ = {
            'file_name' : mp3_file,
            'embeddings' : [],
            'label_emotion' : []
        }
        for r in res:
            max_index = r['scores'].index(max(r['scores']))
            label_ = (r['labels'][max_index])
            dir_['embeddings'].append(r['feats'])
            dir_['label_emotion'].append(label_)
    except:
        dir_ = None
    return dir_

def buff_write(buff_data):
    not_ok = []
    for file, data in buff_data:
        if data is None:
            not_ok.append(file)
            continue
        file_path = file.split('.')[0] + '.npy'
        np.save(file_path, data)
        print('Save ', file_path)
    return not_ok


def log_error(error_files):
    with open('not_ok_audio_to_embed.txt', 'a', encoding='utf-8') as f:
        for file in error_files:
            f.write(file + '\n')
    
def audio_to_embed():
    file_names = glob.glob('/root/autodl-tmp/xianyu/sentiment_classification/data/audio/*/*.*')
    buff_data = []
    start_time = time.time()
    write_step = 10
    for idx, file in  enumerate(file_names):
        if not file.endswith('.mp3') and  not file.endswith('.MP3'):
            continue
        try:
            data = get_audio_to_embed(file)
            buff_data.append((file, data))
            if (idx+ 1) % write_step == 0:
                not_ok = buff_write(buff_data)
                buff_data = []
                end = time.time()
                print(f'Finished {write_step} samples, and cost ', time.strftime("%H:%M:%S", time.gmtime(end - start_time)))
                start_time = time.time()
                log_error(not_ok)
                
        except KeyboardInterrupt:
            not_ok = buff_write(buff_data)
            log_error(not_ok)
            
            break
        except:
            not_ok = buff_write(buff_data)
            log_error(not_ok)
            buff_data = []
            # print('Error in ', file)
            

            
        
if __name__ == '__main__':
    audio_to_embed()
    
    # data = np.load('txt.npy', allow_pickle=True)
    # print(data)