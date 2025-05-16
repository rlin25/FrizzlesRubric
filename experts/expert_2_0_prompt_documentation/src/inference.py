import argparse
from models.binary_classifier import DocumentationClassifierTrainer

def parse_args():
    parser = argparse.ArgumentParser(description='Make predictions using trained documentation classifier')
    parser.add_argument('--model_path', type=str, required=True,
                      help='Path to the trained model')
    parser.add_argument('--text', type=str, required=True,
                      help='Text to classify')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Initialize trainer and load model
    trainer = DocumentationClassifierTrainer()
    trainer.load(args.model_path)
    
    # Make prediction
    probability = trainer.predict(args.text)
    prediction = 1 if probability > 0.5 else 0
    
    print(f"\nInput text: {args.text}")
    print(f"Prediction: {'Well-documented' if prediction == 1 else 'Poorly documented'}")
    print(f"Confidence: {probability:.2%}")

if __name__ == '__main__':
    main() 