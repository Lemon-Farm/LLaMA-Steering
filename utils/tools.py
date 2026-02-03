import torch
import os

def add_vector(matrix, vector):
    matrix[:, -1, :] += vector * 1 # multiplier
    return matrix

def substract_vector(matrix, vector):
    matrix[:, -1, :] -= vector * 1 # multiplier
    return matrix

def save_vector(vector, path):
    if not os.path.exists(path):
        os.makedirs(path)
    
    torch.save(vector, os.path.join(path, 'steering_vector.pt'))
