import torch
import torch.nn as nn

from transformers import AutoTokenizer, AutoModelForCausalLM
from utils.tools import add_vector, substract_vector

class BlockWrapper(nn.Module):
    def __init__(self, block=None):
        super(BlockWrapper, self).__init__()
        self.block = block
        
        self.steering_vector = None
        self.add_vector = False
        self.substract_vector = False
        self.activation = None
    
    def forward(self, *args, **kwargs):
        output = self.block(*args, **kwargs)
        
        if self.steering_vector is not None:
            if self.add_vector:
                output = add_vector(output, self.steering_vector)
            elif self.substract_vector:
                output = substract_vector(output, self.steering_vector)

        self.activation = output
        
        return output
        

class LlamaWrapper(nn.Module):
    def __init__(self, model_path=None, target_layer=None):
        super(LlamaWrapper, self).__init__()
        self.model_path = model_path
        self.target_layer = target_layer
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        self.model.model.layers[self.target_layer] = BlockWrapper(self.model.model.layers[self.target_layer])
    
    def get_activation(self):
        return self.model.model.layers[self.target_layer].activation
    
    def set_steering_vector(self, steering_vector=None):
        self.model.model.layers[self.target_layer].steering_vector = steering_vector.to(self.device)
    
    def set_add_vector(self, add_vector=True):
        self.model.model.layers[self.target_layer].add_vector = add_vector
        if add_vector:
            self.model.model.layers[self.target_layer].substract_vector = False
    
    def set_substract_vector(self, substract_vector=True):
        self.model.model.layers[self.target_layer].substract_vector = substract_vector
        if substract_vector:
            self.model.model.layers[self.target_layer].add_vector = False
        
    def reset(self):
        self.model.model.layers[self.target_layer].steering_vector = None
        self.model.model.layers[self.target_layer].activation = None
        self.model.model.layers[self.target_layer].add_vector = False
        self.model.model.layers[self.target_layer].substract_vector = False
    
    def get_logits(self, tokens=None):
        return self.model(tokens).logits
    
    def generate(self, tokens=None):
        return self.model.generate(inputs=tokens, max_new_tokens=50, top_k=1)
