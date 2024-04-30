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
from torch.utils.tensorboard import SummaryWriter


# set use cuda:7 and cuda:8
# os.environ['CUDA_VISIBLE_DEVICES'] = '6 7'

# for debug

# torch.autograd.set_detect_anomaly(True)



print('CUDA COUNTS:', torch.cuda.device_count())

os.environ['TRANSFORMERS_CACHE'] = './pretrained_models'

def train_one_epoch(model, dataloader, optimizer, criterion, accelerator):
    model.train()
    running_loss = 0.0
    
    epoch_iterator = tqdm(dataloader, desc="Iteration", disable=not accelerator.is_local_main_process)
    
    for i, data in enumerate(epoch_iterator):
        text = data['text']
        audio = {'audio_embedding':data['audio_embedding'], 'audio_padded_type':data['audio_padded_type']}
        label = data['label']
       

        # text = {key: value.to(accelerator.device) for key, value in text.items()}
        audio = {key: value.to(accelerator.device) for key, value in audio.items()}
        label = label.to(accelerator.device)
        
        # text = {key: value.to("cuda:0") for key, value in text.items()}
        # audio = {key: value.to("cuda:0") for key, value in audio.items()}
        # label = label.to("cuda:0")
        
        
        
        optimizer.zero_grad()
        
        output = model(text, audio)
        
        loss = criterion(output, label)
        
        accelerator.backward(loss)
        # loss.backward()
        
        # todo 梯度裁剪
        
        # 查看梯度
        # for name, param in model.named_parameters():
        #     if param.requires_grad:；
        #         # print(name, param.grad)
        #         if param.grad is None:
        #             print(f'Gradient of {name} is None.')
        #         else:
        #             if torch.isnan(param.grad).any():
        #                 print(f'Gradient of {name} is nan.')
        #         # 查看最大的梯度
        #         if param.grad is not None and  torch.isinf(param.grad).any():
        #             print(f'Gradient of {name} is inf.')
                    
        
        # 查看更新前的权重
        # for name, param in model.named_parameters():
        #     if param.requires_grad and param.grad is not None:
        #         print(name, param)
        
        accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # 查看更新后的权重
        # for name, param in model.named_parameters():
        #     if param.requires_grad and param.grad is not None:
        #         print(name, param)
        
        
        running_loss += loss.item()
    
        
    return running_loss / len(dataloader)


def evaluate(model, dataloader, criterion, accelerator):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for i, data in enumerate(dataloader):

            text = data['text']
            audio = {'audio_embedding':data['audio_embedding'], 'audio_padded_type':data['audio_padded_type']}
            label = data['label']
            
            # text = {key: value.to(accelerator.device) for key, value in text.items()}
            audio = {key: value.to(accelerator.device) for key, value in audio.items()}
            label = label.to(accelerator.device)

            
            # label = label.to("cuda:0")
            
            output = model(text, audio)
            
            loss = criterion(output, label)
            
            running_loss += loss.item()
            
    return running_loss / len(dataloader)

def train(model, train_dataloader, val_dataloader, optimizer, criterion, num_epochs, accelerator, writer):
    os.makedirs('models', exist_ok=True)
    
    for epoch in range(num_epochs):
        train_loss = train_one_epoch(model, train_dataloader, optimizer, criterion, accelerator)
        val_loss = evaluate(model, val_dataloader, criterion, accelerator)
        
        accelerator.wait_for_everyone()
        
        if accelerator.is_main_process:
            # print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss}')
        
            print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss}, Val Loss: {val_loss}')
            
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Loss/val', val_loss, epoch)
        
            if (epoch+1) % 5 == 0:
                torch.save(model.state_dict(), f'models/model_{epoch+1}.pth')
    
    if accelerator.is_main_process:        
        torch.save(model.state_dict(), 'models/model.pth')

def main():
    writer = SummaryWriter()
    kwargs_handlers = [DistributedDataParallelKwargs(find_unused_parameters=True)]

    accelerator = Accelerator(
        kwargs_handlers=kwargs_handlers,
       
        # device_placement=False
    )
    
    args = parse_args()
    
    if 'intern' in args.bert_model_path:
        args.text_embedding_dim = 4096
    
    # device = torch.device(args.device if torch.cuda.is_available() and 'cuda' in  args.device else 'cpu')
    # print(device)
    
    model = text_audio_sentiment_classify(args, accelerator.device).to(torch.float16).to(accelerator.device) #.to(torch.float16).to(accelerator.device)
    
    if accelerator.is_main_process:
        print('accelerator device: ', accelerator.device)
    
    # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, eps = 1e-4)
    
    criterion = torch.nn.CrossEntropyLoss()
    
    data = np.load(args.data_path, allow_pickle=True).item()
    
    if 'intern' in args.bert_model_path:
        tokenizer = AutoTokenizer.from_pretrained(args.bert_model_path,
                                                    cache_dir=args.pretrained_models_dir,
                                                    trust_remote_code=True,
                                                     padding_side = 'right',
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
    
    model, train_dataloader, val_dataloader, optimizer = accelerator.prepare(model, train_dataloader, val_dataloader, optimizer)
    
    train(model, train_dataloader, val_dataloader, optimizer, criterion, args.num_epochs, accelerator, writer) 
    
    writer.close()
    
if __name__ == '__main__':
   
    main()
  