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
    answers_with_steering = []
    answers_without_steering = []
    for q_token in tqdm(dataset, desc='Testing'):
        with torch.no_grad():
            token = torch.tensor(q_token).unsqueeze(0).to(device)
            output = model.generate(token)
            answer = model.tokenizer.decode(output[0], skip_special_tokens=True)
            answers_without_steering.append(answer)
            model.reset()

            model.set_steering_vector(steering_vector)
            output = model.generate(token)
            answer = model.tokenizer.decode(output[0], skip_special_tokens=True)
            answers_with_steering.append(answer)
            model.reset()

    results = []
    for idx, (answer_wo, answer_w) in enumerate(zip(answers_without_steering, answers_with_steering)):
        results.append({
            "question": dataset.data[idx]['question'],
            "answer_without_steering": answer_wo,
            "answer_with_steering": answer_w
        })
    with open('test_results.json', 'w', encoding='utf-8') as out_f:
        json.dump(results, out_f, ensure_ascii=False, indent=2)