import torch
import pdb 
from base_logger import logger

from sklearn.metrics import accuracy_score
def train(gpu, model, train_loader, optimizer):
        model.train()
        if gpu==0: logger.info("Train start")
        for iter, batch in enumerate(train_loader):
            optimizer.zero_grad()
            input_ids = batch['input_ids'].cuda(non_blocking = True)
            attention_mask = batch['attention_mask'].cuda(non_blocking = True)
            token_type_ids = batch['token_type_ids'].cuda(non_blocking = True)
            labels = batch['labels'].cuda(non_blocking = True)
            outputs = model(input_ids = input_ids, token_type_ids = token_type_ids,\
                attention_mask=attention_mask, labels = labels)
            loss = outputs[0]
            loss.backward()
            optimizer.step()
        
            if (iter + 1) % 10 == 0 and gpu==0:
                logger.info('gpu {} step : {}/{} Loss: {:.4f}'.format(
                    gpu,
                    iter, 
                    str(len(train_loader)),
                    loss.detach())
                )
        




def valid(gpu, model, dev_loader):

    model.eval()
    anss = []
    preds = []
    loss_sum = 0
    logger.info("Validation start")
    with torch.no_grad():
        for iter,batch in enumerate(dev_loader):
            anss += batch['labels'].tolist()
            input_ids = batch['input_ids'].cuda(non_blocking = True)
            attention_mask = batch['attention_mask'].cuda(non_blocking = True)
            token_type_ids = batch['token_type_ids'].cuda(non_blocking = True)
            labels = batch['labels'].cuda(non_blocking = True)
            outputs = model(input_ids = input_ids, token_type_ids = token_type_ids,\
                attention_mask=attention_mask, labels = labels)
            loss_sum += outputs[0].detach()
            preds += torch.max(outputs[1], axis = 1).indices.to('cpu').tolist()
        
        if (iter + 1) % 10 == 0 and gpu == 0:
            logger.info('step : {}/{} Loss: {:.4f}'.format(
            iter, 
            str(len(dev_loader)),
            outputs[0].detach()
            ))
           
    return  anss, preds, loss_sum/iter
        
        