import yaml
import torch
import json
from tqdm import tqdm

from datasets.PN_dataset import PN_dataset
from llama_wrapper import LlamaWrapper

with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)
    
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LlamaWrapper(model_path=config['model_path'], target_layer=config['target_layer']).to(device)
    dataset = PN_dataset(config['data_path'], model_path=config['model_path'], generation=False)
    steering_vector = torch.load(config['steering_vector_path'])
    
    model.eval()
    answers_with_add_steering = []
    answers_with_substract_steering = []
    for q_token in tqdm(dataset, desc='Testing'):
        with torch.no_grad():
            token = torch.tensor(q_token).unsqueeze(0).to(device)
            model.set_steering_vector(steering_vector)
            model.set_add_vector(True)
            output = model.generate(token)
            answer = model.tokenizer.decode(output[0], skip_special_tokens=True)
            answers_with_add_steering.append(answer)
            model.reset()

            model.set_steering_vector(steering_vector)
            model.set_substract_vector(True)
            output = model.generate(token)
            answer = model.tokenizer.decode(output[0], skip_special_tokens=True)
            answers_with_substract_steering.append(answer)
            model.reset()

    results = []
    for idx, (answer_add, answer_sub) in enumerate(zip(answers_with_add_steering, answers_with_substract_steering)):
        results.append({
            "question": dataset.data[idx]['question'],
            "answer_with_add_steering": answer_add,
            "answer_with_substract_steering": answer_sub
        })
    with open('test_results.json', 'w', encoding='utf-8') as out_f:
        json.dump(results, out_f, ensure_ascii=False, indent=2)
