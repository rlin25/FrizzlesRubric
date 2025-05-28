import unittest
from src.preprocessing import preprocess_prompt

class TestPreprocessing(unittest.TestCase):
    def test_basic(self):
        prompt = "Hello, world! This is a test prompt."
        processed = preprocess_prompt(prompt)
        self.assertIn('hello', processed)
        self.assertNotIn('!', processed)
        self.assertNotIn('this', processed)  # stopword

if __name__ == '__main__':
    unittest.main() 