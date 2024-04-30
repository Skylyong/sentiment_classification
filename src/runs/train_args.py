import argparse 


LABEL_SET = [
    'Comfort', 'Hopeful', 'Confident', 'Interest', 
    'Curiosity', 'Intrigue', 'Insight', 'Enlightenment',
    'Epiphany', 'Thrilled', 'Enthusiastic', 
    'Calm', 'Anticipatory', 'Excited', 'Pleased', 'Satisfied',
    'Proud'
]

LABELS_MAP = {
'pleased':'comfort',
'proud':'confident',
'interested':'interest',
'confidence':'confident',
'interested':'interest',
'enthusiasm':'enthusiastic',
'curious':'curiosity',
'entusiastic':'enthusiastic',
'excitement':'excited',
'intrigued':'interest',
'nostalgia' :'calm',
'nostalgic':'comfort'

}
DROP_LABELS = []

def new_label_set():
    res = []
    for label in LABEL_SET:
        label = label.lower()
        if label in LABELS_MAP:
            label = LABELS_MAP[label]
        if label not in DROP_LABELS:
            res.append(label)
    res = list(set(res))
    
    return sorted(res)


LABEL_SET = new_label_set()

if len(LABEL_SET) != 15:
    print(len(LABEL_SET))
    for label in LABEL_SET:
        print(label)

assert len(LABEL_SET) == 15

def parse_args():
    parser = argparse.ArgumentParser(description='Train a model')
    
    parser.add_argument('--data_path', type=str, default='./DATA_SET/data_processed_normalized_total.npy', help='Path to the data')
    parser.add_argument('--model_path', type=str, default='models/', help='Path to save the model')
    # parser.add_argument('--model_name', type=str, default='model', help='Name of the model')
    
    parser.add_argument('--bert_model_path', type=str, default='internlm/internlm2-7b', choices=['internlm/internlm2-7b','hfl/chinese-bert-wwm', 'bert-base-uncased'],help='Path to the bert model')
    parser.add_argument('--emotion2vec_model_path', type=str, default='None', help='Path to the emotion2vec model')
    parser.add_argument('--num_classes', type=int, default=len(LABEL_SET), help='Number of classes')
    
    parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--train_batch_size', type=int, default=1, help='Batch size for training')
    parser.add_argument('--val_batch_size', type=int, default=1, help='Batch size for validation')

    parser.add_argument('--num_epochs', type=int, default=50, help='Number of epochs')
    
    parser.add_argument('--label_set', type=list, default=LABEL_SET, help='Set of labels')
        
    parser.add_argument('--max_audio_length', type=int, default=53, help='Max audio length max length: 53')
    parser.add_argument('--max_length', type=int, default=1000, help='Max length of one page text') # 80000 
    parser.add_argument('--max_seg_text_length', type=int, default=1000, help='Max segment text length')
    
    # parser.add_argument("--cache_dir",type=str,default='./cache',help='cache dir')
    # parser.add_argument("--refresh_token",action='store_true',help='refresh token')
    
     
    # parser.add_argument('--device', type=str, default='cuda:0',choices=['cpu', 'cuda'], help='device')

    parser.add_argument('--pretrained_models_dir', type=str, default='pretrained_models', help='pretrained models dir')
    
    parser.add_argument('--text_embedding_dim', type=int, default=4096, help='Text embedding dimension')
    parser.add_argument('--audio_embedding_dim', type=int, default=768, help='Audio embedding dimension')
   
    parser.add_argument('--att_dropout', type=float, default=0.3, help='Attention dropout') 
    parser.add_argument('--fusion_pooling_type', type=str, default='max', choices=['mean', 'max'], help='Fusion pooling type')
     
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers')
   
   
    parser.add_argument('--emotion_map_dict', type=dict, default={
        'Comfort':0,
        'Interest':0,
        'Insight':0,
        'Satisfied':0,
        'Calm':0,
        'Hopeful':0.5,
        'Curiosity':0.5,
        'Enlightenment':0.5,
        'Thrilled':0.5,
        'Anticipatory':0.5,
        "Confident":1,
        "Intrigue":1,
        "Epiphany":1,
        "Enthusiastic":1,
        "Excited":1
        
        }, help='map emotion class to scores')
    
    parser.add_argument('--predict_model_path', type=str, default='/data/User_Hannah/temp/sentiment_classification/models/model.pth', help='Path to the predict model')
    
    args = parser.parse_args()
    return args