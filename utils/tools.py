import torch
import os

def add_vector_by_pos(matrix, vector, position_ids, from_pos=None):
    """
    Add a vector to a matrix after 'from_pos'
    Needed for aplly steering vector both in prefill and after decoding step consistently
    """
    if from_pos is None:
        from_pos = position_ids.min() - 1
        
    mask = position_ids >= from_pos
    mask = mask.unsqueeze(-1)
    
    matrix += mask.float() * vector
    
    return matrix

def add_vector(matrix, vector):
    matrix[:, -1, :] += vector * 3
    return matrix

def save_vector(vector, path):
    if not os.path.exists(path):
        os.makedirs(path)
    
    torch.save(vector, os.path.join(path, 'steering_vector.pt'))