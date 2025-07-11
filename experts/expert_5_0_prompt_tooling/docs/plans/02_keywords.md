# 2. Keywords Subplan (Detailed)
- Enumerate all relevant keywords/phrases for AI/LLM/tooling detection (e.g., 'ai', 'gpt', 'llm', 'openai', 'language model', 'cursor ai', 'chatgpt', 'anthropic', 'claude', 'bard', 'copilot', etc.).
- Store keywords in a Python list in a dedicated module (e.g., keywords.py) for easy import and update.
- Optionally, allow loading keywords from an external file (e.g., JSON or YAML) for runtime configurability.
- Provide a function (e.g., get_keywords()) to retrieve the current keyword list for use in detection logic.
- Validate that all keywords are lowercase and non-empty strings.
- Document the process for updating the keyword list:
  - Edit the Python list or external file.
  - Add or remove keywords as needed.
  - Ensure tests cover new keywords.
- Provide guidelines for extending the list to support regex or more complex patterns in the future. 