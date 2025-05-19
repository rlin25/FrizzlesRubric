# Expert 3 - Structure Documentation

## Overview
Expert 3 is responsible for evaluating whether a prompt contains direct or indirect references to files that need to be modified. It operates as part of a 6-part rubric system, providing a binary-leaning output (0 or 1) based on the presence of file references.

## Documentation Structure
1. `architecture.md` - System architecture and technical specifications
2. `model_specs.md` - Detailed ML model specifications
3. `data_split.md` - Data distribution and split strategy
4. `api_specs.md` - API endpoint specifications
5. `testing.md` - Testing strategy and metrics

## Quick Start
- Input: Text prompt to be evaluated
- Output: Binary score (0 or 1) derived from confidence score
- Processing Time Target: < 500ms per evaluation

## Key Features
- File reference detection
- Confidence score generation
- Binary output conversion
- High-performance processing
- Simple API integration 