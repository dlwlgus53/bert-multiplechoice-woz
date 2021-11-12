import pdb
import json
import torch
import pickle
import ontology
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from base_logger import logger


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_path, type, data_rate, tokenizer, max_length, max_options, debug=True):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_options = max_options
        if type == 'train':
            raw_path = f'data/preprocessed_{type}_{self.max_length}_{max_options}_{data_rate}.pickle'
        else:
            raw_path = f'data/preprocessed_{type}_{self.max_length}_{max_options}.pickle'

        try:
            if debug:
                0/0
            else:
                logger.info(f"Load {raw_path}")
                with open(raw_path, 'rb') as f:
                    encodings = pickle.load(f)
        except:
            raw_dataset = json.load(open(data_path, "r"))
            
            input_ids, attention_mask, token_type_ids, answer, dial_id, turn_id,schema = self._preprocessing_dataset(raw_dataset)
            
            tokenized_examples = {
                'input_ids' : input_ids,
                'attention_mask' : attention_mask,
                'token_type_ids' : token_type_ids
            }

            logger.info("Encoding dataset (it will takes some time)")
            encodings = {k: [v[i:i+max_options] for i in range(0, len(v), max_options)] for k, v in tokenized_examples.items()}
            encodings['labels'] = answer
            encodings['dial_id'] = dial_id
            encodings['turn_id'] = turn_id
            encodings['schema'] = schema

            assert len(encodings['labels']) == len(encodings['attention_mask']) == len(encodings['input_ids']) == len(encodings['token_type_ids'])\
                    == len(dial_id) == len(turn_id) == len(schema)
                    
            ## save preprocesse data
            with open(raw_path, 'wb') as f:
                pickle.dump(encodings, f, pickle.HIGHEST_PROTOCOL)

        self.encodings = encodings
        
        # for check
        logger.info(tokenizer.convert_ids_to_tokens(encodings['input_ids'][0][0])) # 하나의질문 # 여러개의 대답
        

    def __getitem__(self, idx):
        try:
            to_be_tensor = ['input_ids', 'token_type_ids', 'attention_mask', 'labels']
            temp = {key: (torch.tensor(val[idx]) if key in to_be_tensor else val[idx]) for key, val in self.encodings.items()}
        except:
            pdb.set_trace()
        return temp
    

    def __len__(self):
        return len(self.encodings['input_ids'])
    
    def _preprocessing_dataset(self, dataset):
        input_idss =[]
        input_masks = []
        segment_idss = []
        answer = []
        dial_id = []
        turn_id = []
        schema = []
        
        logger.info(f"preprocessing data")
        for id in dataset.keys():
            
            dialogue = dataset[id]['log']
            dialouge_text = ""
            
            for turn in dialogue: # 대화 한 turn을 가져옵니다.
                
                dialouge_text += turn['user']
                
                for key in ontology.QA['multichoice-domain']:
                    
                    q = ontology.QA[key]['description']
                    c = dialouge_text
                    os = ontology.QA[key]['values']
                    os += (['wrong'] * (self.max_options-len(os)))
                    assert len(os) == self.max_options

                    if key in turn['belief']: # 언급을 한 경우
                        answer.append(turn['belief'][key][1]) # answer index
                    else:
                        
                        answer.append(os.index('not mentioned')) 
                    
                    schema.append(key)
                    dial_id.append(id)
                    turn_id.append(turn['turn_num'])                        
                        

                    for o in os: 
                        c_token = self.tokenizer.tokenize(c)
                        q_token = self.tokenizer.tokenize(q)
                        o_token = self.tokenizer.tokenize(o)
                        c_token, q_token, o_token = self._truncate_cqo_token(c_token, q_token, o_token)
                        
                        tokens = ["[CLS]"] + c_token + ["[SEP]"] + q_token + o_token + ["[SEP]"] 
                        segment_ids = [0] * (len(c_token) + 2) + [1] * (len(q_token) )+ [1] * (len(o_token) + 1)
                        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
                        input_mask = [1] * len(input_ids)

                        # Zero-pad up to the sequence length.
                        padding = [0] * (self.max_length - len(input_ids))
                        input_ids += padding
                        input_mask += padding
                        segment_ids += padding
                        assert self.max_length == len(input_ids) == len(input_mask) == len(segment_ids)
                        
                        input_idss.append(input_ids)
                        input_masks.append(input_mask)
                        segment_idss.append(segment_ids)

                        
                
                dialouge_text += turn['response']
                
        
        return input_idss, input_masks, segment_idss, answer, dial_id, turn_id, schema




    # def _preprocessing_dataset(self, dataset):
    #     input_idss =[]
    #     input_masks = []
    #     segment_idss = []
    #     labels = []
        
    #     context = [] 
    #     question = []

    #     print(f"preprocessing {self.data_name} data")
    #     if self.data_name == 'race':
    #         article, question, options, answer = dataset['article'], dataset['question'], dataset['options'], dataset['answer']
    #     elif self.data_name == 'dream':
    #         article, question, options, answer = dataset['dialogue'], dataset['question'], dataset['choice'], dataset['answer']
        
        
    #     for i, (c, q, os, a) in tqdm(enumerate(zip(article, question, options, answer)), total= len(article)):
    #         os += ['not mentioned']
    #         os += (['wrong'] * (self.max_options-len(os)))
    #         assert len(os) == self.max_options
    #         for o in os:
    #             if self.data_name == 'dream':
    #                 c = ' '.join(c)
    #             c_token = tokenizer.tokenize(c)
    #             q_token = tokenizer.tokenize(q)
    #             o_token = tokenizer.tokenize(o)
    #             c_token, q_token, o_token = self._truncate_cqo_token(c_token, q_token, o_token)
                
    #             tokens = ["[CLS]"] + c_token + ["[SEP]"] + q_token + o_token + ["[SEP]"] 
    #             segment_ids = [0] * (len(c_token) + 2) + [1] * (len(q_token) )+ [1] * (len(o_token) + 1)
    #             input_ids = tokenizer.convert_tokens_to_ids(tokens)
    #             input_mask = [1] * len(input_ids)

    #             # Zero-pad up to the sequence length.
    #             padding = [0] * (self.max_length - len(input_ids))
    #             input_ids += padding
    #             input_mask += padding
    #             segment_ids += padding
    #             assert self.max_length == len(input_ids) == len(input_mask) == len(segment_ids)
    #             input_idss.append(input_ids)
    #             input_masks.append(input_mask)
    #             segment_idss.append(segment_ids)

    #         if self.data_name == 'race':
    #             label = ord(a) - ord('A')
    #         elif self.data_name == 'dream':
    #             label = os.index(a)
                
    #         labels.append(label)
        
    #     return input_idss, input_masks, segment_idss, labels
        
    
    
    
    def _truncate_cqo_token(self, c,q,o):
        special_token_num = 3 # [cls][sep][sep]
        
        if len(c) + len(q) + len(o) + special_token_num > self.max_length:
            if self.max_length - (len(q) + len(o) + special_token_num) > 0:
                c = c[:self.max_length - (len(q) + len(o) + special_token_num)]
            else:
                c = []
        if len(c) + len(q) + len(o) + special_token_num > self.max_length:
            if self.max_length - (len(o) + special_token_num) > 0:
                q = q[:self.max_length - (len(o) + special_token_num)]
            else:
                q = []
                
        if len(c) + len(q) + len(o) + special_token_num > self.max_length:
            if self.max_length - special_token_num > 0:
                o = o[:self.max_length - ( special_token_num)]
            else:
                o = []
        if (len(c) + len(q) + len(o) + special_token_num > self.max_length):
            pdb.set_trace()
        return c,q,o
            
        



if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', use_fast=True)
    data_path = '../data/MultiWOZ_2.1/dev_data.json'
    max_length = 512
    max_option = 9
    dd = Dataset(data_path,'val', tokenizer,  max_option, debug=False)
    pdb.set_trace()
    for i in range(10):
        print(tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(dd[i]['input_ids'])))




