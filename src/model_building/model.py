'''
@Author: leon
@Date: 2024年04月15日17:35:45
@Description:

模型设计：

1. 使用bert提取文本特征
2. 使用emotion2vec_base_finetuned提取音频特征
3. 使用两个特征拼接，然后使用全连接层进行分类

'''

from typing import *
import torch
import torch.nn as nn
from transformers import BertModel
from transformers import AutoModelForCausalLM, AutoConfig
from peft import LoraConfig, get_peft_model,TaskType
 


class text_embedding_with_bert(nn.Module):
    def __init__(self, bert_model_path, pretrained_models_dir, text_embedding_dim, max_length=6000):
        super(text_embedding_with_bert, self).__init__()
        
        self.model_path = bert_model_path
        
        if 'intern' in bert_model_path:
            
            # Inter model: https://huggingface.co/internlm/internlm2-7b
            
            config = AutoConfig.from_pretrained(bert_model_path, 
                                                cache_dir=pretrained_models_dir, 
                                                output_hidden_states=True,
                                                
                                                trust_remote_code=True)
            config.hidden_size = text_embedding_dim
            config.max_position_embeddings = max_length
            
            inter_model = AutoModelForCausalLM.from_pretrained(bert_model_path,
                                                               cache_dir=pretrained_models_dir,
                                                                   config = config,
                                                                   torch_dtype=torch.float16,
                                                                   
                                                                   trust_remote_code=True,
                                                                   
                                                                   resume_download = True,)
            # internLM-SFT: 
            #   https://github.com/yongzhuo/InternLM-SFT
            #    https://huggingface.co/docs/peft/tutorial/peft_model_config
            # for param in inter_model.parameters():
            #     param.requires_grad = False
            
            peft_config = LoraConfig(
                 r=16,
                target_modules=["wo", "wqkv"],
                task_type=TaskType.CAUSAL_LM,
                lora_alpha=32,
                lora_dropout=0.05,
                modules_to_save=["classifier"],
                peft_type= "LORA",
                init_lora_weights=True,
                
            )
            model = get_peft_model(inter_model, peft_config)
            model.print_trainable_parameters()
            self.model = model
             
        else:
            
            self.model = BertModel.from_pretrained(bert_model_path, cache_dir=pretrained_models_dir)
            
            # for param in self.model.parameters():
            #     param.requires_grad = False
            
        
    def forward(self, text):
        if 'intern' in self.model_path:
            # with torch.no_grad():
            outputs =  self.model(input_ids=text['input_ids'], attention_mask=text['attention_mask'])
            return outputs.hidden_states[-1]
        else: 
            outputs = self.model(**text)
            return outputs.last_hidden_state


class audio_embedding_with_emotion2vec(nn.Module):
    def __init__(self, emotion2vec_model_path):
        super(audio_embedding_with_emotion2vec, self).__init__()
        
        if emotion2vec_model_path != 'None':
            self.emotion2vec_model = torch.load(emotion2vec_model_path)
        else:
            self.emotion2vec_model = None
            
    def forward(self, audio):
        if self.emotion2vec_model:
            return self.emotion2vec_model(audio)
        else:
            
            return  audio['audio_embedding'], audio['audio_padded_type']

class feature_fusion(nn.Module):
    def __init__(self, args):
        super(feature_fusion, self).__init__()
        self.args = args
        self.fc = nn.Linear(args.text_embedding_dim, args.audio_embedding_dim)
        
        self.att = nn.MultiheadAttention(args.audio_embedding_dim, num_heads=1, dropout=args.att_dropout, batch_first=True)
        
        self.max_pooling = nn.MaxPool1d(args.max_length)
        
        
    def forward(self, text_embedding, audio_embedding, audio_mask):
        text_embedding = text_embedding.view(-1, self.args.text_embedding_dim)
        
        text_embedding = self.fc(text_embedding)
        
        text_embedding = text_embedding.view(-1, self.args.max_length, self.args.audio_embedding_dim)
        audio_mask = 1 - audio_mask
        
        # audio_mask shape: [batch_size, 53] - > [batch_size, 100, 53]
        audio_mask = audio_mask.unsqueeze(1).expand(-1, self.args.max_length, -1)
                
        fusion_embedding, _ = self.att(text_embedding, audio_embedding, audio_embedding, attn_mask = audio_mask)
        
        # fusion_embedding, _ = self.att(text_embedding, audio_embedding, audio_embedding)
        
        if self.args.fusion_pooling_type == 'mean':
            fusion_embedding = torch.mean(fusion_embedding, dim=1)
        elif self.args.fusion_pooling_type == 'max':
            fusion_embedding = torch.max(fusion_embedding, dim=1)[0]
        else:
            raise ValueError(f'Invalid pooling type: {self.args.fusion_pooling_type}')
        # fuion_embedding shape: [batch_size, 768]
        
        return fusion_embedding
 

class text_audio_sentiment_classify(nn.Module):
    def __init__(self, args):
        super(text_audio_sentiment_classify, self).__init__()
        self.args = args
        self.text_model = text_embedding_with_bert(args.bert_model_path, 
                                                   args.pretrained_models_dir,
                                                   args.text_embedding_dim,
                                                   args.max_length)
        self.audio_model = audio_embedding_with_emotion2vec(args.emotion2vec_model_path)
        self.fusion = feature_fusion(args)
        self.fc = nn.Linear(args.audio_embedding_dim, args.num_classes)
        
    def forward(self, text, audio):
        
        text_keys = list(text.keys())
        for key in text_keys:
            text[key] = text[key].view(-1, self.args.max_length) 
        
        text_embedding = self.text_model(text)
        
        audio_embedding, audio_mask = self.audio_model(audio)
       
        # print('text_embedding shape:', text_embedding.shape)
        # print('audio_embedding shape:', audio_embedding.shape) 
        # text_embedding shape: [batch_size, 512, 768]
        
        # audio_embedding shape: [batch_size,53 ,768]
        
        embedding = self.fusion(text_embedding, audio_embedding, audio_mask)
        
        # embedding = text_embedding[:, 0, :] # + audio_embedding[:, 0, :]
        
        output = self.fc(embedding)
        
        # output = torch.sigmoid(output)
                        
        return output