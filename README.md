---
title: Crop Disease Detection & Treatment Agent
emoji: 🌾
colorFrom: green
colorTo: yellow
sdk: docker
app_file: app.py
pinned: false
---

# Crop Disease Detection & Treatment Agent

## Description

A simulated OpenEnv environment for crop disease detection and treatment recommendation. The system uses a **multi-agent architecture** where specialized agents handle diagnosis, confidence scoring, explanation generation, and treatment recommendation.

This environment is designed for reinforcement learning research in agricultural applications, implementing the full OpenEnv spec with typed models, API endpoints, and graded tasks.

## Real-World Use Case

In agriculture, early detection of crop diseases can prevent significant yield losses. This environment simulates the process of observing crop conditions and making decisions on disease diagnosis and treatment recommendations. It can be used to train AI agents that assist farmers in identifying issues like leaf blight or rust and suggesting appropriate pesticides.

## Architecture — Multi-Agent System

| Agent | Responsibility |
|---|---|
| **DiagnosisAgent** | Analyzes leaf color, spots, humidity, temperature to predict the disease |
| **ConfidenceAgent** | Scores how strongly the symptoms support the diagnosis (0.0–1.0) |
| **ExplanationAgent** | Generates a human-readable explanation of the reasoning |
| **TreatmentAgent** | Recommends crop-specific treatment |
| **PipelineAgent** | Orchestrator — runs all 4 agents in sequence |

## OpenEnv API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/reset` | POST | Reset the environment, returns initial observation |
| `/step` | POST | Take an action, returns observation + reward + done |
| `/state` | GET | Get the current observation |
| `/ui` | GET | Gradio web interface |

## Action Space

The action space is discrete with the following actions:
- `predict_healthy` — Predict the crop is healthy
- `predict_leaf_blight` — Predict leaf blight disease
- `predict_rust` — Predict rust disease
- `suggest_none` — Recommend no treatment
- `suggest_pesticide_A` — Recommend pesticide A
- `suggest_pesticide_B` — Recommend pesticide B

## Observation Space

Observations are provided as a typed Pydantic model (`models.Observation`) containing:

| Field | Type | Values |
|---|---|---|
| `crop_type` | string | tomato, potato, rice |
| `leaf_color` | string | green, yellow, brown |
| `spots` | boolean | true, false |
| `humidity` | integer | 0–100 |
| `temperature` | integer | 10–40 |

## Tasks & Grading

Three difficulty levels, each graded on a 0.0–1.0 scale:

### Easy Task
Binary classification: Determine if the crop is healthy or diseased.

### Medium Task
Multi-class disease classification: Identify the specific disease (healthy, leaf_blight, or rust).

### Hard Task
Full pipeline: Perform both disease detection and treatment recommendation.

## Environment Variables

The following variables must be set before running `inference.py`:

```bash
export API_BASE_URL="https://your-llm-endpoint/v1"   # API endpoint for the LLM
export MODEL_NAME="your-model-name"                    # Model identifier
export HF_TOKEN="hf_your_token_here"                   # Hugging Face / API key
```

## How to Run Locally

1. Clone this repository:
   ```bash
   git clone <repository-url>
   cd "Crop Disease Detection & Treatment Agent"
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the app (Gradio UI + API endpoints):
   ```bash
   python app.py
   ```
   - Gradio UI: http://localhost:7860/ui
   - API endpoints: http://localhost:7860/reset, /step, /state

4. Run baseline evaluation (rule-based, no LLM needed):
   ```bash
   python baseline.py
   ```

5. Run LLM inference (requires env variables):
   ```bash
   export API_BASE_URL="..." MODEL_NAME="..." HF_TOKEN="..."
   python inference.py
   ```

## Docker

```bash
docker build -t crop-disease-agent .
docker run -p 7860:7860 crop-disease-agent
```

## Deploy on Hugging Face Spaces

1. Create a new Space on [Hugging Face](https://huggingface.co/spaces)
2. Select **Docker** as the SDK
3. Upload all project files
4. Set the environment variables (`API_BASE_URL`, `MODEL_NAME`, `HF_TOKEN`) in the Space settings
5. The Space will build and deploy automatically

## Project Structure

```
├── app.py              # FastAPI + Gradio (API endpoints + web UI)
├── agents.py           # Multi-agent system (Diagnosis, Confidence, Explanation, Treatment, Pipeline)
├── env.py              # OpenEnv environment (step/reset/state)
├── models.py           # Pydantic typed models (Observation, Action, StepResult)
├── tasks.py            # Task definitions (Easy, Medium, Hard)
├── grader.py           # Grading functions for each task (0.0–1.0)
├── baseline.py         # Rule-based baseline agent
├── inference.py        # LLM-powered inference using OpenAI Client
├── openenv.yaml        # OpenEnv specification file
├── requirements.txt    # Python dependencies
├── Dockerfile          # Container configuration for HF Spaces
└── README.md           # This file
```
