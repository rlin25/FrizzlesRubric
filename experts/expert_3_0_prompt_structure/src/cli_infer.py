import torch
from model import DistilBertFileClassifier
import os

MODEL_PATH = '/home/ubuntu/FrizzlesRubric/experts/expert_3_0_prompt_structure/models/best_model.pt'

def main():
    device = 'cpu'
    model = DistilBertFileClassifier().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    print('Model loaded. Enter a prompt to get a binary prediction (0 or 1). Type "exit" to quit.')
    while True:
        try:
            prompt = input('Prompt: ')
            if prompt.strip().lower() == 'exit':
                print('Exiting.')
                break
            label, prob = model.predict(prompt, device=device)
            print(f'Prediction: {label} (probability: {prob:.4f})')
        except KeyboardInterrupt:
            print('\nExiting.')
            break
        except Exception as e:
            print(f'Error: {e}')

if __name__ == '__main__':
    main() 