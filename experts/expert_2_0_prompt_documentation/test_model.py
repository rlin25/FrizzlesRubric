from src.models.binary_classifier import DocumentationClassifierTrainer
import torch
import pandas as pd
import glob
import os
from typing import List

def test_model():
    # Initialize the trainer
    trainer = DocumentationClassifierTrainer()
    
    # Load the trained model
    model_path = "models/run_20250518_070518/best_model.pt"
    print("Loading model and tokenizer...")
    trainer.load(model_path)
    
    # Test cases with varying levels of clarity and documentation
    test_cases = [
        # Well-structured, clear examples
        "Implement a user authentication system with the following steps: 1) Create a User model with email and password fields, 2) Add password hashing using bcrypt, 3) Implement login and registration endpoints, 4) Add JWT token generation and validation, 5) Include input validation and error handling.",
        
        "Refactor the database queries by: 1) Identifying all raw SQL queries in the codebase, 2) Converting them to parameterized queries, 3) Adding appropriate indexes for frequently queried columns, 4) Implementing query caching where beneficial, 5) Adding performance monitoring and logging.",
        
        "Add error handling to the payment processing module: 1) Create custom exception classes for different error types, 2) Implement retry logic with exponential backoff, 3) Add comprehensive logging for debugging, 4) Include transaction rollback mechanisms, 5) Update API documentation with error codes and handling.",
        
        # Moderately structured examples
        "Update the user profile page to include new fields. Add validation for the new fields and make sure the existing functionality still works. Also, consider adding some error messages for invalid inputs.",
        
        "The current caching mechanism isn't working well. Look into implementing a better solution that handles cache invalidation properly and maybe add some monitoring to track cache hit rates.",
        
        "We need to improve the search functionality. Consider adding filters, sorting options, and maybe some kind of fuzzy matching. Make sure it's performant with large datasets.",
        
        # Vague, unclear examples
        "Can you fix the bug in the login system? It's not working right.",
        
        "The app is running slow, maybe optimize it or something?",
        
        "Add some tests to make sure everything works properly.",
        
        # Examples with mixed clarity
        "Implement a new feature that allows users to share their progress. It should work with social media platforms and include some kind of privacy settings. Make sure it's user-friendly.",
        
        "The current error messages aren't very helpful. Can you make them more descriptive? Also, add some logging to help with debugging.",
        
        "We need to handle timeouts better in the API. Maybe add some retry logic and make sure the client gets a proper response.",
        
        # Examples with good structure but missing details
        "Create a new endpoint for user preferences. Include validation and error handling. Make sure it's secure.",
        
        "Update the database schema to support the new features. Don't forget to handle migrations properly.",
        
        "Add monitoring to the application. Track important metrics and set up alerts.",
        
        # Examples with clear goals but unclear implementation
        "Make the application more responsive. Users are complaining about slow loading times.",
        
        "Improve the security of the authentication system. Add some additional checks and validations.",
        
        "The current logging system isn't very useful. Implement a better solution that helps with debugging.",
        
        # Examples with good documentation but unclear requirements
        "Implement a new reporting feature that generates PDF reports. Include options for customization and scheduling.",
        
        "Add support for multiple languages in the user interface. Make it easy to add new translations.",
        
        "Create a backup system for the database. Ensure data integrity and provide recovery options.",
        
        # Examples with clear steps but missing context
        "1) Add input validation, 2) Implement error handling, 3) Update documentation, 4) Add tests",
        
        "1) Create new database tables, 2) Add API endpoints, 3) Update frontend components, 4) Write tests",
        
        "1) Set up monitoring, 2) Configure alerts, 3) Add logging, 4) Create dashboards",
        
        # Example with good structure but technical jargon
        "Implement a microservices architecture with service discovery, load balancing, and circuit breakers. Ensure proper error handling and monitoring across all services."
    ]

    print("\nTesting model predictions:")
    print("-" * 50)

    for i, text in enumerate(test_cases, 1):
        prediction = trainer.predict(text)
        print(f"\nTest case {i}:")
        print(f"Text: {text}")
        print(f"Prediction: {prediction:.4f}")
        print(f"Classification: {'Well documented' if prediction > 0.75 else 'Poorly documented'}")

if __name__ == "__main__":
    test_model() 