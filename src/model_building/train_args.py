import argparse 


LABEL_SET = [
    'Comfort', 'Hopeful', 'Confident', 'Interest', 
    'Curiosity', 'Intrigue', 'Insight', 'Enlightenment',
    'Epiphany', 'Thrilled', 'Enthusiastic', 
    'Calm', 'Anticipatory', 'Excited', 'Pleased', 'Satisfied',
    'Proud'
]

def parse_args():
    parser = argparse.ArgumentParser(description='Train a model')
    
    parser.add_argument('--data_path', type=str, default='./DATA_SET/data_processed.npy', help='Path to the data')
    parser.add_argument('--model_path', type=str, default='models/', help='Path to save the model')
    # parser.add_argument('--model_name', type=str, default='model', help='Name of the model')
    
    parser.add_argument('--bert_model_path', type=str, default='internlm/internlm2-7b', choices=['internlm/internlm2-7b','hfl/chinese-bert-wwm', 'bert-base-uncased'],help='Path to the bert model')
    parser.add_argument('--emotion2vec_model_path', type=str, default='None', help='Path to the emotion2vec model')
    parser.add_argument('--num_classes', type=int, default=17, help='Number of classes')
    
    parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--train_batch_size', type=int, default=1, help='Batch size for training')
    parser.add_argument('--val_batch_size', type=int, default=1, help='Batch size for validation')

    parser.add_argument('--num_epochs', type=int, default=50, help='Number of epochs')
    
    parser.add_argument('--label_set', type=list, default=LABEL_SET, help='Set of labels')
        
    parser.add_argument('--max_audio_length', type=int, default=53, help='Max audio length max length: 53')
    parser.add_argument('--max_length', type=int, default=1024, help='')
    
    # parser.add_argument("--cache_dir",type=str,default='./cache',help='cache dir')
    # parser.add_argument("--refresh_token",action='store_true',help='refresh token')
    
     
    # parser.add_argument('--device', type=str, default='cuda:0',choices=['cpu', 'cuda'], help='device')

    parser.add_argument('--pretrained_models_dir', type=str, default='pretrained_models', help='pretrained models dir')
    
    parser.add_argument('--text_embedding_dim', type=int, default=768, help='Text embedding dimension')
    parser.add_argument('--audio_embedding_dim', type=int, default=768, help='Audio embedding dimension')
   
    parser.add_argument('--att_dropout', type=float, default=0.3, help='Attention dropout') 
    parser.add_argument('--fusion_pooling_type', type=str, default='max', choices=['mean', 'max'], help='Fusion pooling type')
     
    parser.add_argument('--num_workers', type=int, default=32, help='Number of workers')
    
    args = parser.parse_args()
    return args