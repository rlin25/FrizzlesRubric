from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch
import logging
from pathlib import Path
from typing import List, Dict, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_test_cases() -> List[Dict[str, str]]:
    return [
        # Basic sentence structure
        {"prompt": "The cat sat on the mat.", "category": "Basic", "expected": "correct"},
        {"prompt": "Cat sat mat.", "category": "Basic", "expected": "incorrect"},
        
        # Subject-verb agreement
        {"prompt": "The dogs are barking.", "category": "Subject-Verb Agreement", "expected": "correct"},
        {"prompt": "The dogs is barking.", "category": "Subject-Verb Agreement", "expected": "incorrect"},
        {"prompt": "Each of the students has a book.", "category": "Subject-Verb Agreement", "expected": "correct"},
        {"prompt": "Each of the students have a book.", "category": "Subject-Verb Agreement", "expected": "incorrect"},
        
        # Verb tenses
        {"prompt": "I have been working here for five years.", "category": "Verb Tenses", "expected": "correct"},
        {"prompt": "I am working here since five years.", "category": "Verb Tenses", "expected": "incorrect"},
        {"prompt": "She will have finished by tomorrow.", "category": "Verb Tenses", "expected": "correct"},
        {"prompt": "She will finished by tomorrow.", "category": "Verb Tenses", "expected": "incorrect"},
        
        # Articles
        {"prompt": "I saw a elephant at the zoo.", "category": "Articles", "expected": "incorrect"},
        {"prompt": "I saw an elephant at the zoo.", "category": "Articles", "expected": "correct"},
        {"prompt": "The sun rises in east.", "category": "Articles", "expected": "incorrect"},
        {"prompt": "The sun rises in the east.", "category": "Articles", "expected": "correct"},
        
        # Prepositions
        {"prompt": "I am interested in learning Python.", "category": "Prepositions", "expected": "correct"},
        {"prompt": "I am interested to learn Python.", "category": "Prepositions", "expected": "incorrect"},
        {"prompt": "She is good at playing piano.", "category": "Prepositions", "expected": "correct"},
        {"prompt": "She is good in playing piano.", "category": "Prepositions", "expected": "incorrect"},
        
        # Complex sentences
        {"prompt": "Although it was raining, we went for a walk.", "category": "Complex", "expected": "correct"},
        {"prompt": "Although it was raining but we went for a walk.", "category": "Complex", "expected": "incorrect"},
        {"prompt": "The book that I bought yesterday is very interesting.", "category": "Complex", "expected": "correct"},
        {"prompt": "The book which I bought yesterday is very interesting.", "category": "Complex", "expected": "correct"},
        
        # Punctuation
        {"prompt": "Let's eat, Grandma!", "category": "Punctuation", "expected": "correct"},
        {"prompt": "Lets eat Grandma!", "category": "Punctuation", "expected": "incorrect"},
        {"prompt": "I need to buy: milk, eggs, and bread.", "category": "Punctuation", "expected": "correct"},
        {"prompt": "I need to buy milk eggs and bread.", "category": "Punctuation", "expected": "incorrect"},
        
        # Common errors
        {"prompt": "Their going to the store.", "category": "Common Errors", "expected": "incorrect"},
        {"prompt": "They're going to the store.", "category": "Common Errors", "expected": "correct"},
        {"prompt": "Its a beautiful day.", "category": "Common Errors", "expected": "incorrect"},
        {"prompt": "It's a beautiful day.", "category": "Common Errors", "expected": "correct"},
        
        # Questions
        {"prompt": "What time is it?", "category": "Questions", "expected": "correct"},
        {"prompt": "What time it is?", "category": "Questions", "expected": "incorrect"},
        {"prompt": "Do you know where the library is?", "category": "Questions", "expected": "correct"},
        {"prompt": "Do you know where is the library?", "category": "Questions", "expected": "incorrect"},
        
        # Conditionals
        {"prompt": "If I were you, I would take the job.", "category": "Conditionals", "expected": "correct"},
        {"prompt": "If I was you, I would take the job.", "category": "Conditionals", "expected": "incorrect"},
        {"prompt": "I wish I had studied harder.", "category": "Conditionals", "expected": "correct"},
        {"prompt": "I wish I studied harder.", "category": "Conditionals", "expected": "incorrect"}
    ]

def test_model():
    # Initialize model and tokenizer
    model_dir = Path("/home/ubuntu/FrizzlesRubric/experts/expert_1_1_prompt_grammar/models/grammar_model")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    logger.info("Loading model and tokenizer...")
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    model = DistilBertForSequenceClassification.from_pretrained(model_dir)
    model.to(device)
    model.eval()
    logger.info("Model loaded successfully")

    test_cases = get_test_cases()
    results = []
    
    print("\n=== Grammar Checker Test Results ===\n")
    
    for test in test_cases:
        # Tokenize input
        inputs = tokenizer(
            test["prompt"],
            add_special_tokens=True,
            max_length=512,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        ).to(device)

        # Get prediction
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=1)
            prediction = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][prediction].item()

        result = "correct" if prediction == 1 else "incorrect"
        is_correct = result == test["expected"]
        
        print(f"\nCategory: {test['category']}")
        print(f"Prompt: {test['prompt']}")
        print(f"Expected: {test['expected']}")
        print(f"Predicted: {result}")
        print(f"Confidence: {confidence:.2%}")
        print(f"Result: {'✓' if is_correct else '✗'}")
        
        results.append({
            "category": test["category"],
            "is_correct": is_correct,
            "confidence": confidence
        })
    
    # Print summary
    print("\n=== Test Summary ===")
    categories = set(r["category"] for r in results)
    for category in categories:
        category_results = [r for r in results if r["category"] == category]
        correct_count = sum(1 for r in category_results if r["is_correct"])
        total_count = len(category_results)
        accuracy = correct_count / total_count
        avg_confidence = sum(r["confidence"] for r in category_results) / total_count
        print(f"\n{category}:")
        print(f"Accuracy: {accuracy:.2%}")
        print(f"Average Confidence: {avg_confidence:.2%}")

if __name__ == "__main__":
    test_model() 