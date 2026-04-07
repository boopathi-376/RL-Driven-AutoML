---
title: RL-Driven AutoML Pipeline
emoji: 🧠
colorFrom: purple
colorTo: blue
sdk: docker
pinned: false
app_port: 7860
base_path: /
tags:
  - openenv
  - reinforcement-learning
  - automl
  - scikit-learn
---

#  RL-Driven AutoML: An Intelligent ML Model Selector

**RL-Driven AutoML** is an interactive environment designed for the **OpenEnv** ecosystem. Unlike static AutoML tools that output a single recommendation, this project treats the machine learning workflow as a **Sequential Decision Process**. It enables Reinforcement Learning (RL) agents to interact with datasets, perform strategic actions (cleaning, encoding, selecting), and receive rewards based on their ability to build high-performing, efficient pipelines.

---
## 🔗 Quick Access

- 🎮 **Playground**: https://boopathi376-rl-driven-automl.hf.space/playground  
- 🤗 **Hugging Face Space**: https://huggingface.co/spaces/Boopathi376/RL-Driven-AutoML  
- 💻 **GitHub Repository**: https://github.com/boopathi-376/RL-Driven-AutoML  

---
##  The Core Concept: "The Virtual ML Engineer"

Traditional AutoML takes an input and returns an output. **RL-Driven AutoML** simulates the human workflow through a structured pipeline. At each stage, the agent makes a strategic decision:

1.  **Observe**: Dataset metadata, distributions, and statistics.
2.  **Decide**: Select the next processing step (e.g., "Should I use Polynomial Features?").
3.  **Execute**: Apply the action and observe the transformation.
4.  **Evaluate**: Measure final model performance vs. resource usage (latency/memory).
5.  **Learn**: Optimize the strategy for future unseen datasets.

---

##  The Intelligent Pipeline Workflow

For structured data, the environment enforces an 8-stage "Sequential Decision Process". For raw text data, it automatically simplifies to a 4-stage specialized text pipeline.

[Pipeline Diagram]
1. Cleaning -> 2. Encoding -> 3. Engineering -> 4. Scaling -> 5. Selection -> 6. Modeling -> 7. Tuning -> 8. Ensemble

---

##  Environment Architecture

```text
                +----------------------+
                |     RL Agent         |
                | (Decision Maker)     |
                +----------+-----------+
                           |
                      Action (JSON)
                           |
                           v
    +--------------------------------------------------+
    |         OpenEnv Environment (FastAPI)            |
    |  --------------------------------------------    |
    |                                                  |
    |  +------------------------------------------+   |
    |  |      FastAPI /step Endpoint              |   |
    |  +------------------+-----------------------+   |
    |                     |                           |
    |         +-----------v----------+                |
    |         |   State Manager      |                |
    |         |  (Pipeline Tracker)  |                |
    |         +-------+----------+---+                |
    |                 |          |                    |
    |         +-------v----+ +---v------+             |
    |         |Data Engine | |Reward    |             |
    |         |(Processing)| |Engine    │             |
    |         +-------+----+ |(Scoring) |             |
    |                 |      +---+------+             |
    |         +-------v----+     |                     |
    |         |Internal    |-----+                     |
    |         |State       |                           |
    |         +-------+----+                           |
    |                 |                                |
    |         +-------v----------+                     |
    |         |Observation       |                     |
    |         |Generator         |                     |
    |         +-------+----------+                     |
    +-----------------v--------------------------------+
                      |
                Observation + Reward
                      |
                      v
                +------------------+
                |    RL Agent      |
                |  (Learns Policy) |
                +------------------+
```

---

##  API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/reset` | `POST` | Start a new episode with a fresh dataset |
| `/step` | `POST` | Execute a pipeline action and get observation + reward |
| `/state` | `GET` | Get detailed internal state (progress, stages, metadata) |
| `/docs` | `GET` | Interactive Swagger/OpenAPI documentation |
| `/ws` | `WSS` | Persistent WebSocket session for low-latency agent interaction |

---

##  RL Environment Specifications

###  Observation Space
```json
{
  "stage": "encoding",
  "task_type": "classification",
  "dataset_profile": {
    "n_samples": 1000,
    "n_features": 25,
    "missing_values": 120
  },
  "progress": 0.25,
  "reward": 0.74
}
```

###  Reward System
Rewards are calculated dynamically based on:
`Reward = Val_Score + (0.2 * Improvement) - (0.3 * Overfit_Gap) - Latency_Penalty`

| Scenario | Reward Signal | Reason |
|----------|---------------|--------|
| **Optimal Selection** | +0.6 to +1.0 | High validation accuracy matching task type |
| **Overfitting** | -0.40 | Large gap between Training and Validation scores |
| **Heavy Model** | -0.15 | High computation time / latency penalty |

---

##  Getting Started

### 1. Clone and Install
```bash
git clone https://github.com/boopathi-376/RL-Driven-AutoML.git
cd RL-Driven-AutoML
uv sync
```

### 2. Run the Server
```bash
uv run server
```

### 3. Run the Agent (Inference)
```bash
# Requires HF_TOKEN or API_KEY env variable
uv run python inference.py
```

---

##  Project Structure

```text
RL-Driven-AutoML/
|-- server/
|   |-- steps_8/        # 8 Core ML Processing Engine
|   |-- app.py          # FastAPI Server Scaffolding
|   `-- model_selector_environment.py # Environment Logic
|-- data/               # Benchmark Datasets (CSV/TXT)
|-- models.py           # Pydantic Action/Observation schemas
|-- client.py           # OpenEnv WebSocket Client
|-- inference.py        # LLM-based Agent Reference Implementation
|-- openenv.yaml        # environment manifest
`-- Dockerfile          # Container configuration
```
### 📊 Task-Based Reward Outcomes

| Task   | Complexity     | Key Challenge                                   | Env Reward |
|--------|---------------|------------------------------------------------|------------|
| Easy   | Small CSV     | Simple prediction task with balanced datasets  | 0.74       |
| Medium | Large Tabular | Mixed data types (encoding + scaling needed)   | 0.57       |
| Hard   | Long Text     | Complex processing with memory constraints     | 0.61       |
---

##  Future Roadmap
- [ ] Multi-Agent Collaboration -- Separate agents for Data Cleaning vs. Modeling.
- [ ] Explainable AI -- Agent provides text reasoning for its actions.
- [ ] Optuna Integration -- Advanced Bayesian hyperparameter search.
- [ ] Visualization Dashboard -- Real-time training monitoring.
