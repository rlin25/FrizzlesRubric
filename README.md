# Frizzle's Rubric

Frizzle's Rubric is a modular, extensible system for automated evaluation of AI prompts. It combines a suite of expert models, an orchestrator, and a modern web interface to provide instant, multi-metric feedback on prompt quality. The system is designed for both research and practical use in prompt engineering, LLM evaluation, and developer education.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Architecture](#architecture)
- [Expert Modules](#expert-modules)
- [Orchestrators](#orchestrators)
- [Web Application](#web-application)
- [Test Out Frizzle's Rubric](#test-out-frizzles-rubric)
- [Setup & Installation](#setup--installation)
- [API Reference](#api-reference)
- [Development & Contributing](#development--contributing)
- [AWS Architecture](#aws-architecture)
- [License](#license)

---

## Project Overview

Frizzle's Rubric provides automated, multi-dimensional evaluation of natural language prompts. Each "expert" is a specialized model or rule-based system that scores a prompt on a specific quality dimension (e.g., clarity, grammar, documentation, structure, granularity, tooling, repetition). The orchestrator aggregates these results, and the webapp visualizes them for users.

---

## Architecture

- **Experts**: Individual microservices, each evaluating a single prompt metric.
- **Orchestrators**: Services that route prompts to all experts and aggregate their responses.
- **Webapp**: A React/MUI frontend for interactive prompt testing and visualization.
- **Docs**: Planning, architecture, and deployment documentation.

```
/experts/         # All expert models and their APIs
/orchestrators/   # Orchestrator APIs for prompt aggregation
/webapp/          # React frontend for user interaction
/docs/            # Architecture and deployment docs
```

---

## Expert Modules

Each expert is a self-contained service (FastAPI, Python) with its own model, API, and deployment scripts. Most experts use a fine-tuned DistilBERT or similar transformer for binary classification.

- **expert_1_0_prompt_clarity**: Classifies prompt clarity (clear/unclear).
- **expert_1_1_prompt_grammar**: Detects proper/improper grammar and spelling.
- **expert_2_0_prompt_documentation**: Checks for documentation quality and completeness.
- **expert_3_0_prompt_structure**: Evaluates logical and structural organization.
- **expert_4_0_prompt_granularity**: Assesses task granularity and scope.
- **expert_5_0_prompt_tooling**: Detects references to AI tools or automation.
- **expert_6_0_prompt_repetition**: Flags repeated or duplicate prompts.

Each expert exposes a simple HTTP API (usually `/predict` or `/check`) and can be run standalone (see each expert's README and Dockerfile for details).

---

## Orchestrators

Orchestrators are FastAPI services that accept a prompt, call all expert APIs in parallel, and return a combined result. They handle error logging, timeouts, and response normalization.

- **orchestrator_1_0_prompt_evaluation**: Main orchestrator for the webapp.
- **orchestrator_2_0_prompt_quality**: Orchestrates grammar and clarity logic.

---

## Web Application

The webapp is a modern React application (MUI, react-beautiful-dnd) for interactive prompt evaluation.

**Features:**
- Submit your own prompt for instant, multi-metric feedback.
- Drag and drop example prompts into the input area.
- Visual feedback: Each expert is represented by an icon "light" that changes color based on the result.
- Mouse over any expert icon to see a tooltip describing its metric.
- Responsive design for desktop and mobile.

**Tech stack:**
- Frontend: React 18, Material-UI, react-beautiful-dnd
- Backend: Python 3.9+, FastAPI, HuggingFace Transformers, PyTorch, Docker, AWS EC2/VPC

---

## Test Out Frizzle's Rubric

Frizzle's Rubric is running live at: **[www.frizzlesrubric.net](https://www.frizzlesrubric.net)**

- **Try it out:** Enter your own prompt or drag and drop one of the example prompts into the input area.
- **Interactive UI:** Each expert is represented by an icon. Mouse over any expert icon to see a tooltip describing what metric it evaluates (e.g., Clarity, Documentation, Structure, etc.).
- **Visual feedback:** The expert icons light up in green (pass), red (fail), or yellow (error/unavailable) based on the evaluation.
- **Drag-and-drop:** Example prompts can be dragged and dropped into the input box for quick testing.

---

## Setup & Installation

### Prerequisites

- Python 3.9+ (for experts and orchestrators)
- Node.js 18+ (for webapp)
- Docker (optional, for containerized deployment)

### 1. Clone the repository

```bash
git clone https://github.com/your-org/frizzlesrubric.git
cd frizzlesrubric
```

### 2. Set up and run experts

Each expert has its own setup. Example for expert_1_0_prompt_clarity:

```bash
cd /absolute/path/to/experts/expert_1_0_prompt_clarity
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python src/api.py
```

Or build and run with Docker:

```bash
docker build -t expert-clarity .
docker run -p 8001:8001 expert-clarity
```

Repeat for each expert (see their individual READMEs for details).

### 3. Set up and run orchestrator

```bash
cd /absolute/path/to/orchestrators/orchestrator_1_0_prompt_evaluation
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python orchestrator_api.py
```

### 4. Set up and run the webapp

```bash
cd /absolute/path/to/webapp
npm install
npm start
```

The webapp will be available at http://localhost:3000 (or see the live site).

---

## API Reference

Each expert exposes a simple HTTP API, typically:

- **POST /predict** or **/check**
  - Request: `{ "prompt": "your prompt here" }`
  - Response: `{ "predicted_class": 0|1, "confidence": 0.97 }` (fields may vary)

The orchestrator exposes:

- **POST /orchestrate**
  - Request: `{ "prompt": "your prompt here" }`
  - Response: `{ "expert_1_clarity": 1, "expert_2_documentation": 0, ... }`

See each expert's and orchestrator's source for full details.

---

## Development & Contributing

- Each expert is modular and can be developed, tested, and deployed independently.
- Add new experts by following the existing template (FastAPI, Dockerfile, requirements.txt).
- The orchestrator can be extended to call new experts or aggregate results differently.
- The webapp is fully decoupled and can be extended with new UI features or metrics.

---

## AWS Architecture

Frizzle's Rubric is deployed on AWS using a secure, production-grade VPC architecture:

**Key Points:**

- **Public subnet** hosts:
  - Bastion Host (with public IP for SSH access)
  - NAT Gateway (with Elastic IP for outbound internet)
- **Private subnet** hosts:
  - Private EC2 instances (no public IPs, not directly accessible from the internet)
- **NAT Gateway** allows private instances to access the internet securely, without exposing them publicly.
- **Bastion Host** enables secure SSH access to private instances using a jump server pattern.
- **Security:**
  - Only the bastion host is exposed to the public internet.
  - All other services (experts, orchestrators) run in private subnets, accessible only via the bastion or internal VPC networking.

For more details, see [`docs/aws_vpc_architecture.md`](docs/aws_vpc_architecture.md).

---

## License

MIT License. See [LICENSE.md](/c:/Code/Python/Compete/HackAI/FrizzlesRubric/LICENSE.md) for details.

---

**For questions, bug reports, or contributions, please open an issue or pull request.** 