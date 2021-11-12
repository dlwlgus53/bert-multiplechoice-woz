import os
import time
import torch
import argparse
import datetime
import pdb
from dataset import Dataset
from trainer import valid, train
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from transformers import AutoModelForMultipleChoice, AutoTokenizer, AdamW
from knockknock import email_sender

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP


from base_logger import logger


now_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
parser = argparse.ArgumentParser()

parser.add_argument('--data_rate' ,  type = float, default=0.01)
parser.add_argument('--patience' ,  type = int, default=3)
parser.add_argument('--batch_size' , type = int, default=4)
parser.add_argument('--max_epoch' ,  type = int, default=1)
parser.add_argument('--base_trained_model', type = str, default = 'bert-base-uncased', help =" pretrainned model from 🤗")
parser.add_argument('--pretrained_model' , type = str,  help = 'pretrainned model')
parser.add_argument('--gpu_number' , type = int,  default = 0, help = 'which GPU will you use?')
parser.add_argument('--debugging' , type = bool,  default = False, help = "Don't save file")
parser.add_argument('--dev_path' ,  type = str,  default = '../woz-data/MultiWOZ_2.1/dev_data.json')
parser.add_argument('--train_path' , type = str,  default = '../woz-data/MultiWOZ_2.1/train_data.json')
parser.add_argument('--max_length' , type = int,  default = 256, help = 'max length')
parser.add_argument('--max_options' , type = int,  default = 9, help = 'max number of options')
parser.add_argument('--do_train' , default = True, help = 'do train or not', action=argparse.BooleanOptionalAction)
parser.add_argument('-n', '--nodes', default=1,type=int, metavar='N')
parser.add_argument('-g', '--gpus', default=2, type=int,help='number of gpus per node')
parser.add_argument('-nr', '--nr', default=0, type=int,help='ranking within the nodes')
parser.add_argument('--num_worker',default=6, type=int,help='cpus')
args = parser.parse_args()

def makedirs(path): 
   try: 
        os.makedirs(path) 
   except OSError: 
       if not os.path.isdir(path): 
           raise
       
def get_loader(dataset,batch_size,num_worker):
    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    shuffle = False
    pin_memory = True
    loader = torch.utils.data.DataLoader(
        dataset=dataset, batch_size=batch_size, pin_memory=pin_memory,
        num_workers=num_worker, shuffle=shuffle, sampler=train_sampler)
    return loader       
       
# @email_sender(recipient_emails=["jihyunlee@postech.ac.kr"], sender_email="knowing.deep.clean.water@gmail.com")
def main_worker(gpu, args):
    makedirs("./data"); makedirs("./logs"); makedirs("./model");
    logger.info(f'{gpu} works!')
    batch_size = int(args.batch_size / args.gpus)
    num_worker = int(args.num_worker / args.gpus)
    
    torch.distributed.init_process_group(
        backend='nccl',
        init_method='tcp://127.0.0.1:3456',
        world_size=args.gpus,
        rank=gpu)
    
    torch.cuda.set_device(gpu)
    
    model = AutoModelForMultipleChoice.from_pretrained(args.base_trained_model).to(gpu)
    model = DDP(model, device_ids=[gpu])
    train_loader = get_loader(args.train_dataset, batch_size,num_worker)
    dev_loader = get_loader(args.val_dataset, batch_size,num_worker)
    
    optimizer = AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
    min_loss = float('inf')
    best_performance = {}
    
    map_location = {'cuda:%d' % 0: 'cuda:%d' % gpu}
    
    
    if args.pretrained_model:
        logger.info(f"use trained model{args.pretrained_model}")
        model.load_state_dict(
            torch.load(args.pretrained_model, map_location=map_location))

    

    for epoch in range(args.max_epoch):
        dist.barrier()
        if gpu==0: logger.info(f"Epoch : {epoch}")
        if args.do_train:
            train(gpu, model, train_loader, optimizer) # TODO
        anss, preds, loss = valid(gpu, model, dev_loader)
        ACC = accuracy_score(anss, preds)
        logger.info("Epoch : %d, ACC : %.04f, Loss : %.04f" % (epoch, ACC, loss))

        if gpu == 0 and loss < min_loss:
            logger.info("New best")
            min_loss = loss
            best_performance['min_loss'] = min_loss.item()
            best_performance['accuracy'] = ACC
            if not args.debugging:
                torch.save(model.state_dict(), f"model/woz{args.data_rate}.pt")
            logger.info("safely saved")
                
    if gpu==0:            
        logger.info(f"Best Score :  {best_performance}" )
    dist.barrier()


def main():
    makedirs("./data"); makedirs("./logs"); makedirs("./model");makedirs("./out");
    args.world_size = args.gpus * args.nodes 
    args.tokenizer = AutoTokenizer.from_pretrained(args.base_trained_model, use_fast=False)
    train_path = args.train_path[:-5] + str(args.data_rate) + '.json'
    args.train_dataset =Dataset(train_path, 'train', args.data_rate, args.tokenizer,args.max_length,   args.max_options, debug=False)
    args.val_dataset = Dataset(args.dev_path, 'val', args.data_rate, args.tokenizer, args.max_length,  args.max_options, debug=False)

    mp.spawn(main_worker,
        nprocs=args.world_size,
        args=(args,),
        join=True)

if __name__ =="__main__":
    logger.info(f"{'-' * 30}")
    logger.info("Start New Trainning")
    start = time.time()
    main()
    result_list = str(datetime.timedelta(seconds=time.time() - start)).split(".")
    logger.info(result_list[0])
    logger.info("End The Trainning")
    logger.info(f"{'-' * 30}")