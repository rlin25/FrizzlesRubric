import pytest
from src.detector import detect_ai_tooling

def test_detects_ai_keywords():
    assert detect_ai_tooling("Use OpenAI's GPT-3 for this task.") == 0
    assert detect_ai_tooling("Try the new Copilot feature.") == 0
    assert detect_ai_tooling("Anthropic's Claude is impressive.") == 0
    assert detect_ai_tooling("This is a language model prompt.") == 0

def test_detects_no_ai_keywords():
    assert detect_ai_tooling("Write a function to add two numbers.") == 1
    assert detect_ai_tooling("Sort a list in Python.") == 1

def test_case_insensitivity():
    assert detect_ai_tooling("Use gPt for this.") == 0
    assert detect_ai_tooling("Try the new COPILOT feature.") == 0

def test_edge_cases():
    assert detect_ai_tooling("") == 1
    assert detect_ai_tooling("   ") == 1
    assert detect_ai_tooling("This prompt mentions nothing relevant.") == 1 