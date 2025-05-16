# Expert 1.1 — Prompt Grammar Checker

## Overview

This expert module evaluates the **grammatical correctness** of developer prompts intended for Cursor AI, classifying each prompt as either grammatically correct (`1`) or incorrect (`0`). It handles grammar, punctuation, spacing, and capitalization errors but does **not** perform corrections.

## Project Structure

The implementation plan has been divided into the following subplans:

1. [Dataset Plan](plan_sub_dataset.md)
   - JFLEG dataset specifications
   - Data composition and balancing
   - Train/validation/test splits

2. [Model Plan](plan_sub_model.md)
   - DistilBERT-based architecture
   - Model components and specifications
   - Input/output interface

3. [Training Plan](plan_sub_training.md)
   - Loss function and metrics
   - Training process
   - Evaluation methodology

4. [Deployment Plan](plan_sub_deployment.md)
   - Project structure
   - API interface
   - Performance and logging requirements

5. [Edge Cases Plan](plan_sub_edge_cases.md)
   - Input handling strategies
   - Special case processing
   - Error handling