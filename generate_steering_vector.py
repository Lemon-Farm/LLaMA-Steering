import yaml
import os
import torch
from tqdm import tqdm

from dotenv import load_dotenv
from utils.tools import save_vector
from datasets.PN_dataset import PN_dataset
from llama_wrapper import LlamaWrapper

with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

load_dotenv()
HF_TOKEN = os.getenv('HF_TOKEN')

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LlamaWrapper(model_path=config['model_path'], HF_Token=HF_TOKEN, target_layer=config['target_layer']).to(device)
    dataset = PN_dataset(config['data_path'], model_path=config['model_path'], generation=True, HF_Token=HF_TOKEN)
    
    p_activations = []
    n_activations = []
    
    for p_token, n_token in tqdm(dataset, desc='Generating steering vector'):
        p_token = torch.tensor(p_token).unsqueeze(0).to(device)
        n_token = torch.tensor(n_token).unsqueeze(0).to(device)
        
        model.reset()
        with torch.no_grad():
            p_logits = model.get_logits(p_token)
            p_activation = model.get_activation().cpu()
            p_activation = p_activation[:, -2, :]
            model.reset()
            
            n_logits = model.get_logits(n_token)
            n_activation = model.get_activation().cpu()
            n_activation = n_activation[:, -2, :]
            model.reset()
            
            p_activations.append(p_activation)
            n_activations.append(n_activation)
        
    p_activations = torch.stack(p_activations)
    n_activations = torch.stack(n_activations)
    
    steering_vector = (p_activations - n_activations).mean(dim=0)
    steering_vector = steering_vector.cpu()
    save_vector(steering_vector, config['save_vector_path'])