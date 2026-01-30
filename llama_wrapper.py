import torch
import torch.nn as nn

from transformers import AutoTokenizer, AutoModelForCausalLM
from utils.tools import add_vector_by_pos

class BlockWrapper(nn.Module):
    def __init__(self, block=None):
        super(BlockWrapper, self).__init__()
        self.block = block
        
        self.steering_vector = None
        self.activation = None
    
    def forward(self, *args, **kwargs):
        output = self.block(*args, **kwargs)
        self.activation = output[0]
        
        return output
        

class LlamaWrapper(nn.Module):
    def __init__(self, model_path=None, HF_Token=None, target_layer=None):
        super(LlamaWrapper, self).__init__()
        self.model_path = model_path
        self.target_layer = target_layer
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, token=HF_Token)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, token=HF_Token)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        self.model.model.layers[self.target_layer] = BlockWrapper(self.model.model.layers[self.target_layer])
    
    def get_activation(self):
        return self.model.model.layers[self.target_layer].activation
    
    def set_steering_vector(self, steering_vector=None):
        self.model.model.layers[self.target_layer].steering_vector = steering_vector
        
    def reset(self):
        self.model.model.layers[self.target_layer].steering_vector = None
        self.model.model.layers[self.target_layer].activation = None
    
    def get_logits(self, tokens=None):
        return self.model.model(tokens).logits