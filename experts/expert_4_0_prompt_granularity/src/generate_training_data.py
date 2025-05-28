import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
import random
from pathlib import Path
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TrainingDataGenerator:
    def __init__(self):
        # Granular task templates (label: 1)
        self.granular_templates = [
            "Fix the {component} in the {system}",
            "Update the {component} to {action}",
            "Add {feature} to the {component}",
            "Modify the {component} to {action}",
            "Implement {feature} in the {component}",
            "Change the {component} to {action}",
            "Add validation for {component}",
            "Update the {component} documentation",
            "Fix the {error} in {component}",
            "Add error handling for {component}"
        ]
        
        # Non-granular task templates (label: 0)
        self.non_granular_templates = [
            "Implement a complete {system} with {feature1} and {feature2}",
            "Design and build a {system} that includes {feature1}, {feature2}, and {feature3}",
            "Create a full-stack {system} with {feature1}, {feature2}, and {feature3}",
            "Develop a comprehensive {system} solution",
            "Build an end-to-end {system} with all necessary components",
            "Implement a scalable {system} architecture",
            "Create a production-ready {system}",
            "Design a robust {system} infrastructure",
            "Develop a complete {system} ecosystem",
            "Build a comprehensive {system} platform"
        ]
        
        # Component and feature dictionaries
        self.components = [
            "login form", "registration page", "user profile", "dashboard",
            "API endpoint", "database query", "authentication system",
            "error handler", "validation logic", "configuration file",
            "test suite", "documentation", "README file", "deployment script",
            "logging system", "cache mechanism", "security module"
        ]
        
        self.actions = [
            "handle errors", "validate input", "log events", "cache data",
            "format output", "process requests", "update records",
            "send notifications", "generate reports", "backup data",
            "clean up resources", "optimize performance", "secure data",
            "monitor status", "track changes"
        ]
        
        self.systems = [
            "web application", "mobile app", "API service", "database system",
            "authentication system", "payment processing", "content management",
            "e-commerce platform", "social network", "analytics platform",
            "monitoring system", "deployment pipeline", "testing framework",
            "documentation system", "logging infrastructure"
        ]
        
        self.features = [
            "user authentication", "data validation", "error handling",
            "logging", "caching", "security", "performance optimization",
            "monitoring", "deployment", "testing", "documentation",
            "API integration", "database management", "file handling",
            "notification system"
        ]
    
    def generate_prompt(self, template: str, is_granular: bool) -> str:
        """Generate a prompt from a template."""
        if is_granular:
            # For granular tasks, use components and actions
            prompt = template.format(
                component=random.choice(self.components),
                action=random.choice(self.actions),
                system=random.choice(self.systems),
                error=random.choice(self.actions),
                feature=random.choice(self.features)
            )
        else:
            # For non-granular tasks, use systems and multiple features
            prompt = template.format(
                system=random.choice(self.systems),
                feature1=random.choice(self.features),
                feature2=random.choice(self.features),
                feature3=random.choice(self.features)
            )
        
        return prompt
    
    def is_clear_example(self, prompt: str, label: int) -> bool:
        """Check if an example meets the clear example criteria."""
        words = prompt.split()
        
        # Check length requirements
        if not (5 <= len(words) <= 50):
            return False
        
        # Check for clear scope indicators
        if label == 1:  # Granular task
            granular_indicators = ['fix', 'update', 'add', 'modify', 'implement', 'change']
            if not any(indicator in prompt.lower() for indicator in granular_indicators):
                return False
        else:  # Non-granular task
            non_granular_indicators = ['implement', 'design', 'create', 'develop', 'build']
            if not any(indicator in prompt.lower() for indicator in non_granular_indicators):
                return False
        
        # Check for ambiguous terms
        ambiguous_terms = ['maybe', 'possibly', 'perhaps', 'might', 'could', 'should']
        if any(term in prompt.lower() for term in ambiguous_terms):
            return False
        
        return True
    
    def generate_dataset(self, num_examples: int, clear_ratio: float = 0.7) -> pd.DataFrame:
        """Generate a dataset with the specified number of examples."""
        data = []
        num_clear = int(num_examples * clear_ratio)
        
        # Generate clear examples first
        while len(data) < num_clear:
            is_granular = random.random() < 0.5
            template = random.choice(self.granular_templates if is_granular else self.non_granular_templates)
            prompt = self.generate_prompt(template, is_granular)
            
            if self.is_clear_example(prompt, int(is_granular)):
                data.append({
                    'prompt': prompt,
                    'label': int(is_granular),
                    'is_clear': True
                })
        
        # Generate remaining examples
        while len(data) < num_examples:
            is_granular = random.random() < 0.5
            template = random.choice(self.granular_templates if is_granular else self.non_granular_templates)
            prompt = self.generate_prompt(template, is_granular)
            
            data.append({
                'prompt': prompt,
                'label': int(is_granular),
                'is_clear': self.is_clear_example(prompt, int(is_granular))
            })
        
        # Shuffle the data
        random.shuffle(data)
        
        return pd.DataFrame(data)
    
    def save_dataset(self, df: pd.DataFrame, output_dir: str):
        """Save the dataset to CSV and JSON files."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save full dataset
        df.to_csv(output_path / 'full_dataset.csv', index=False)
        
        # Save clear examples separately
        clear_df = df[df['is_clear'] == True]
        clear_df.to_csv(output_path / 'clear_examples.csv', index=False)
        
        # Save dataset statistics
        stats = {
            'total_examples': len(df),
            'clear_examples': len(clear_df),
            'granular_examples': len(df[df['label'] == 1]),
            'non_granular_examples': len(df[df['label'] == 0]),
            'clear_granular_examples': len(clear_df[clear_df['label'] == 1]),
            'clear_non_granular_examples': len(clear_df[clear_df['label'] == 0])
        }
        
        with open(output_path / 'dataset_stats.json', 'w') as f:
            json.dump(stats, f, indent=2)
        
        logger.info(f"Dataset saved to {output_path}")
        logger.info(f"Statistics: {stats}")

def main():
    # Create data directory
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    # Initialize generator
    generator = TrainingDataGenerator()
    
    # Generate dataset
    df = generator.generate_dataset(num_examples=1000, clear_ratio=0.7)
    
    # Save dataset
    generator.save_dataset(df, str(data_dir))

if __name__ == '__main__':
    main() 