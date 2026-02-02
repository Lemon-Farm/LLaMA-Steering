import torch.nn as nn
import json
from transformers import AutoTokenizer
from utils.tokenize import tokenize_llama_chat

class PN_dataset(nn.Module):
    def __init__(self, data_path=None, model_path=None, generation=True, HF_Token=None):
        super(PN_dataset, self).__init__()
        self.data_path = data_path
        self.HF_Token = HF_Token
        self.generation = generation
        
        if generation:
            self.data = json.load(open(data_path['generation'], 'r'))
        else:
            self.data = json.load(open(data_path['test'], 'r'))
            
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, token=self.HF_Token)

    def __len__(self, generation=True):
        return len(self.data)

    def __getitem__(self, idx, generation=True):
        if generation:
            q_text = self.data[idx]['question']
            p_text = self.data[idx]['answer_matching_behavior']
            n_text = self.data[idx]['answer_not_matching_behavior']
            
            p_token = tokenize_llama_chat(self.tokenizer, q_text, p_text)
            n_token = tokenize_llama_chat(self.tokenizer, q_text, n_text)
            
            return p_token, n_token
        
        else:
            text = self.data[idx]['question']
            token = tokenize_llama_chat(self.tokenizer, text)
            return token