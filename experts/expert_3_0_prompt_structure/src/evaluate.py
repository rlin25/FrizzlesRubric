import torch
import pandas as pd
from model import DistilBertFileClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

@torch.no_grad()
def evaluate(model_path, test_path, device='cpu'):
    df = pd.read_csv(test_path)
    model = DistilBertFileClassifier().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    tokenizer = model.tokenizer
    y_true = []
    y_pred = []
    y_prob = []
    for text, label in zip(df['text'], df['label']):
        label_pred, prob = model.predict(text, device=device)
        y_true.append(label)
        y_pred.append(label_pred)
        y_prob.append(prob)
    print(f'Accuracy:  {accuracy_score(y_true, y_pred):.4f}')
    print(f'Precision: {precision_score(y_true, y_pred):.4f}')
    print(f'Recall:    {recall_score(y_true, y_pred):.4f}')
    print(f'F1:        {f1_score(y_true, y_pred):.4f}')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True, help='Path to best_model.pt')
    parser.add_argument('--test_path', required=True, help='Path to test CSV with text and label columns')
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()
    evaluate(args.model_path, args.test_path, args.device) 