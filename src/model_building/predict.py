import os
import numpy as np
import torch
from train_args import parse_args
from torch.utils.data import DataLoader
from data_utils import CustomDataset, my_collate_fn
from model import text_audio_sentiment_classify
from transformers import BertTokenizer, AutoTokenizer
from  torch.nn.parallel import DistributedDataParallel as DDP
from accelerate import DistributedDataParallelKwargs, Accelerator
from tqdm import tqdm
import json
import pandas as pd


def predict(model, dataloader, accelerator, args):
    model.eval()

    predicts = []
    y_trues = []
    with torch.no_grad():
        for i, data in enumerate(dataloader):

            text = data['text']
            audio = {'audio_embedding':data['audio_embedding'], 'audio_padded_type':data['audio_padded_type']}
            label = data['label']
            
            text = {key: value.to(accelerator.device) for key, value in text.items()}
            audio = {key: value.to(accelerator.device) for key, value in audio.items()}
            label = label.to(accelerator.device)

            
            # label = label.to("cuda:0")
            
            output = model(text, audio)
            pred = output.cpu().detach().numpy()
            # pred = np.where(pred > args.threshold, 1, 0)
            
            label = label.cpu().detach().numpy()
            y_trues.append(label)
            
            predicts.append(pred)
        # outputs > 0.5 = 1 else 0
        predicts = np.concatenate(predicts, axis=0)
        y_trues = np.concatenate(y_trues, axis=0)
       
                   
            
            
    return predicts, y_trues


def find_threshold_micro(dev_yhat_raw, dev_y):
    dev_yhat_raw_1 = dev_yhat_raw.reshape(-1)
    dev_y_1 = dev_y.reshape(-1)
    sort_arg = np.argsort(dev_yhat_raw_1)
    sort_label = np.take_along_axis(dev_y_1, sort_arg, axis=0)
    label_count = np.sum(sort_label)
    correct = label_count - np.cumsum(sort_label)
    predict = dev_y_1.shape[0] + 1 - np.cumsum(np.ones_like(sort_label))
    f1 = 2 * correct / (predict + label_count)
    sort_yhat_raw = np.take_along_axis(dev_yhat_raw_1, sort_arg, axis=0)
    f1_argmax = np.argmax(f1)
    threshold = sort_yhat_raw[f1_argmax]
    # print(threshold)
    return threshold




def map_class_to_scores(predicts, y_tures, idx_to_emotion, emotion_map_dict):
   
    scores = []
    emotion_map_dict = {
        k.lower(): v for k, v in emotion_map_dict.items()
    }
    threshold = find_threshold_micro(y_tures.reshape(-1,1), predicts.reshape(-1,1))
    threshold = threshold if threshold != 1 else 0
    print(f'threshold: {threshold}')
    
    for i in range(len(predicts)):
        ems = []
        num_classes = sum(predicts[i] > threshold)
        score = 0
       
        indents = np.where((predicts[i] > threshold)== 1)[0]
        for id in indents:
            emotion = idx_to_emotion[id]
            ems.append(emotion)
            score += emotion_map_dict[emotion]
        scores.append([num_classes, score/(num_classes+ 1e-6), ' '.join(ems)])
    return scores 

def main():
    
    kwargs_handlers = [DistributedDataParallelKwargs()]

    accelerator = Accelerator(
        kwargs_handlers=kwargs_handlers,
        # device_placement=False
    )
    
    args = parse_args()
    
    if 'intern' in args.bert_model_path:
        args.text_embedding_dim = 4096
    
    # device = torch.device(args.device if torch.cuda.is_available() and 'cuda' in  args.device else 'cpu')
    # print(device)
    
    model = text_audio_sentiment_classify(args).to(accelerator.device) .to(torch.float16).to(accelerator.device)
    
    
    
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
        
    train_data = CustomDataset(data['train'], args, tokenizer, version='train')
    val_data = CustomDataset(data['val'], args, tokenizer, version='val')
    test_data = CustomDataset(data['test'], args, tokenizer, version='test')
    
    train_dataloader = DataLoader(train_data, batch_size=args.train_batch_size, shuffle=True, collate_fn=my_collate_fn, num_workers=args.num_workers)
    
    val_dataloader = DataLoader(val_data, batch_size=args.val_batch_size, shuffle=False, collate_fn=my_collate_fn, num_workers=args.num_workers)
    
    test_dataloader = DataLoader(test_data, batch_size=args.val_batch_size, shuffle=False, collate_fn=my_collate_fn, num_workers=args.num_workers)
    
    model, train_dataloader, val_dataloader,test_dataloader = accelerator.prepare(model, train_dataloader, val_dataloader, test_dataloader)
    
    total_df = pd.DataFrame()
    for data_set, dataloader in [(train_data,train_dataloader), (test_data, test_dataloader),(val_data, val_dataloader)]: # 
        
        predicts, y_trues = predict(model, dataloader, accelerator, args)
        
        # predicts.shape: num_samples, num_classes
        
        scores = map_class_to_scores(predicts,y_trues, data_set.idx_to_label, args.emotion_map_dict)
        
        res_write_to_json = []
        for i in range(len(scores)):
            audio = data_set.data[i]['audio_embedding'].item()
            idx = audio['file_name'].split('/')[-1].split('.')[0]
            tmp = {
                'id': idx,
                'category': data_set.data[i]['category'],
                'emotions': scores[i][2],
                'predict_emotion_classify_num': scores[i][0],
                'predict_emotion_score': scores[i][1]
            }
            res_write_to_json.append(tmp)
        # json.dump(res_write_to_json, open('predict_results.json', 'w'))
        
       
        df = pd.DataFrame(res_write_to_json)
        total_df = pd.concat([total_df, df], axis=0)
        
    total_df.to_excel('predict_results.xlsx', index=None)
    print(' predict_results.xlsx has been saved!') 
    print('Done!')
    
if __name__ == '__main__':
    main()