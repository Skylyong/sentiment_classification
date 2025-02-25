from typing import Any
import torch
from torch.utils.data import Dataset
import numpy as np
import json
import tqdm
import os
from train_args import parse_args
from transformers import BertTokenizer, AutoTokenizer
import jieba

class CustomDataset(Dataset):
    def __init__(self, data: json, args: Any, tokenizer: Any,  version: str = None):
        self.data = data    
        self.args = args
        self.tokenizer = tokenizer
        label_set = args.label_set
        assert len(set(label_set)) == len(label_set), 'label_set should not contain duplicate labels'
        self.label_set = label_set
        self.label_to_idx = {label: idx for idx, label in enumerate(label_set)}
        self.idx_to_label = {idx: label for idx, label in enumerate(label_set)}
        # self.labels_not_in_label_set = []
        # self.labels, self.texts, self.audio_embeddings = self.get_labels_texts_audio_embeddings(version)

        # print(f'Labels not in label_set: {self.labels_not_in_label_set}')
    
    def padding_audio_embedding(self, audio_embedding, max_audio_length):
        padded_type = np.zeros(max_audio_length)
        if len(audio_embedding) > max_audio_length:
            audio_embedding = audio_embedding[:max_audio_length]
            padded_type[:max_audio_length] = 1
        else:
            audio_embedding_shape = audio_embedding[0].shape[0]
            padded_ = [np.zeros(audio_embedding_shape) for _ in range(max_audio_length-len(audio_embedding))]
            
            padded_type[:len(audio_embedding)] = 1
            audio_embedding.extend(padded_)
            
        assert len(audio_embedding) == max_audio_length, f'len(audio_embedding) != max_audio_length'
        
        return{'audio_embedding':audio_embedding, 'padded_type':padded_type}
         

    # def get_labels_texts_audio_embeddings(self,version):
        
    #     # 检查cache文件夹里面是否有labels.npy，如果有则从cache里面加载数据
    #     if os.path.exists(os.path.join(self.args.cache_dir, f'{version}_labels.pt')) and not self.args.refresh_token:
    #         labels = torch.load(os.path.join(self.args.cache_dir, f'{version}_labels.pt'), map_location=self.device )
    #         texts = torch.load(os.path.join(self.args.cache_dir, f'{version}_texts.pt'), map_location=self.device)
    #         audio_embeddings = torch.load(os.path.join(self.args.cache_dir, f'{version}_audio_embeddings.pt'), map_location=self.device)
    #         return labels, texts, audio_embeddings
        
    #     labels = []
    #     texts = []
    #     audio_embeddings = []
        
    #     for d in tqdm.tqdm(self.data):
    #         labels.append(self.encode_label(d['label']))
            
    #         text = d['text']
    #         text = self.tokenizer(text, return_tensors='pt', max_length=self.args.max_length, padding='max_length', truncation=True)
    #         texts.append(text)
            
    #         audio_embedding = d['audio_embedding'].item()['embeddings']
    #         audio_embeddings.append(self.padding_audio_embedding(audio_embedding, self.args.max_audio_length))
        
    #     # 保存到缓存文件夹
    #     os.makedirs(self.args.cache_dir, exist_ok=True)
    #     torch.save(labels, os.path.join(self.args.cache_dir, f'{version}_labels.pt'))
    #     torch.save(texts, os.path.join(self.args.cache_dir, f'{version}_texts.pt'))
    #     torch.save(audio_embeddings, os.path.join(self.args.cache_dir, f'{version}_audio_embeddings.pt'))
         
    #     return labels, texts, audio_embeddings
        
        
    def encode_label(self, labels):              
        # for label in labels:
        #     label = label.lower()
        #     if label not in self.label_set:
        #             self.labels_not_in_label_set.append(label)
        #             print(label)
            # assert label in self.label_set, f'{label} not in label_set'
        
        label_encoding = np.zeros(len(self.label_set))
        for label in labels:
            label = label.lower()
            assert label in self.label_set, f'{label} not in label_set'
            label_encoding[self.label_set.index(label)] = 1
        return label_encoding
                   
    def __len__(self):
        return len(self.data)


    def __getitem__old(self, idx):
        
        '''
        idx : int : index of the data
        return : text : str : text data
                 audio_embedding : [np.array] : audio embedding data
                 labels : np.array : one-hot encoded labels
        '''
       
        texts = self.texts[idx]
        audio_embeddings = self.audio_embeddings[idx]
        labels = self.labels[idx] 
        
        # to tensor and to  device
        texts = {key: value.clone().detach().to(self.device) for key, value in texts.items()}
        audio_embeddings = {key: torch.tensor(np.array(value), dtype=torch.float16 ).to(self.device) for key, value in audio_embeddings.items()}
        labels = torch.tensor(labels, dtype=torch.float16 ).to(self.device)
        
        return {'text': texts, 'audio': audio_embeddings, 'label': labels}
   
    def split_text(self, text):
        split_text = jieba.cut(text)
        split_text = list(split_text)
        return split_text
    
    def tokenize_text(self, text):
        text = self.split_text(text)
        return self.tokenizer(text, return_tensors='pt',
                              max_length=self.args.max_length, 
                              padding='max_length',
                              truncation=True,
                              return_attention_mask = True, is_split_into_words=True
                              
                              )
    
    
    def get_audio_embedding(self, audio_embedding):
        return self.padding_audio_embedding(audio_embedding, self.args.max_audio_length)
    
    
    def __getitem__(self, idx):
        
        '''
        idx : int : index of the data
        return : text : str : text data
                 audio_embedding : [np.array] : audio embedding data
                 labels : np.array : one-hot encoded labels
        '''
       
        texts = self.tokenize_text(self.data[idx]['text'])
        
        try:
            audio_embeddings = self.get_audio_embedding(self.data[idx]['audio_embedding'].item()['embeddings'])
        except:
            audio_embeddings = self.get_audio_embedding(self.data[idx]['audio_embedding']['embeddings'])
        try:
            labels = self.encode_label(self.data[idx]['label'])
        except:
            labels = np.zeros(len(self.label_set))
        
        
        # to tensor and to  device
        texts = {key: value for key, value in texts.items()}
        audio_embeddings = {key: torch.tensor(np.array(value), dtype=torch.float16 ) for key, value in audio_embeddings.items()}
        labels = torch.tensor(labels, dtype=torch.float16 )
        
        return {'text': texts, 'audio': audio_embeddings, 'label': labels}  
    
    
    
    
def my_collate_fn(batch):
    texts = [item['text'] for item in batch]
    
    texts_keys = list(texts[0].keys())    
    texts = {
        texts_key: torch.stack([text[texts_key] for text in texts]) for texts_key in texts_keys
        }
        
    
    audios = [item['audio'] for item in batch]
    audios_embedding = torch.stack([audio['audio_embedding'] for audio in audios])
    audios_type = torch.stack([audio['padded_type'] for audio in audios])
    
    labels = [item['label'] for item in batch]
    labels = torch.stack(labels)
        
    return {'text': texts, 'audio_embedding': audios_embedding, 'audio_padded_type':audios_type, 'label': labels}


if __name__ == '__main__':
    # import sys
    # sys.insert(0, 'src')
    
    args = parse_args()
    data = np.load(args.data_path, allow_pickle=True).item()
    
    if 'intern' in args.bert_model_path:
        tokenizer = AutoTokenizer.from_pretrained(args.bert_model_path,
                                                    cache_dir=args.pretrained_models_dir,
                                                    trust_remote_code=True,
                                                  )
    else:
        tokenizer = BertTokenizer.from_pretrained(args.bert_model_path, 
                                                  cache_dir=args.pretrained_models_dir,
                                                  ) 
    
    train = CustomDataset(data['train'], args, tokenizer, version='train')
    val = CustomDataset(data['val'], args, tokenizer, version='val')
    test = CustomDataset(data['test'], args, tokenizer, version='test')
    
    # 查看train里面的所有数据
    # print(len(train))
    for i in range(len(train)):
        # print(train[i])
        train[i]
    # print(train.labels_not_in_label_set)
    
    for i in range(len(val)):
        val[i]
    for i in range(len(test)):
        test[i]
    
    