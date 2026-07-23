# RL Production Optimizer

Reinforcement Learning system for optimizing oil production using modern RL frameworks and environments.

## Tech Stack

| Component | Technology |
|-----------|-----------|
| RL Algorithms | **Stable Baselines3** - PPO, A2C, DQN agents |
| RL Environments | **Gymnasium** - standardized RL environments |
| Data Processing | pandas, numpy, joblib |
| Web Server | **FastAPI** + uvicorn |
| Monitoring | prometheus-fastapi-instrumentator |
| Validation | pydantic v2 |
| Visualization | matplotlib, seaborn |

### Key Libraries
- Stable Baselines3 - Reliable implementations of RL algorithms (PPO, A2C, DQN, SAC)
- Gymnasium - Standard API for reinforcement learning environments
- FastAPI - Modern async web framework
- pandas / numpy - Data processing

## Overview

This project applies reinforcement learning to oil field production optimization. An RL agent learns to make decisions about production rate adjustments (increase, decrease, or maintain) based on well states, equipment conditions, market prices, and reservoir properties.

### RL Algorithms (Stable Baselines3)

- **PPO**: Proximal Policy Optimization - stable policy gradient method
- **A2C**: Advantage Actor-Critic - synchronous advantage actor-critic
- **DQN**: Deep Q-Network - value-based deep RL with experience replay

## Project Structure

```
rl-production-optimizer/
    rl_optimizer/
        __init__.py
        data_generator.py          # Synthetic production data generation
        models/
            __init__.py
            ppo_agent.py           # PPO agent (Stable Baselines3)
            a2c_agent.py           # A2C agent (Stable Baselines3)
            dqn_agent.py           # DQN agent (Stable Baselines3)
        envs/
            production_env.py      # Gymnasium environment
        utils/
            __init__.py
            state_encoder.py       # State normalization and discretization
    outputs/models/                # Trained model artifacts
    templates/
        index.html                 # Web dashboard
    train.py                       # Training pipeline
    app.py                         # FastAPI API server (port 5019)
    test_api.py                    # API test suite
    requirements.txt
    setup.py
```

## Setup

```bash
pip install -r requirements.txt
```

## Training

```bash
python train.py
```

This trains all three Stable Baselines3 agents and saves models to `outputs/models/`.

## Running the API

```bash
python app.py
```

The API server runs on `http://localhost:5019`.

### API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Web dashboard |
| `/api/health` | GET | Health check and model status |
| `/api/models` | GET | List available models and training stats |
| `/api/optimize` | POST | Get optimal action for a production scenario |
| `/api/simulate` | POST | Run multi-step simulation episode |
| `/api/compare` | GET | Compare model training performance |

### Example: Optimize

```bash
curl -X POST http://localhost:5019/api/optimize -H "Content-Type: application/json" -d '{}'
```

### Example: Simulate

```bash
curl -X POST http://localhost:5019/api/simulate -H "Content-Type: application/json" -d '{"num_steps": 50}'
```

## State Representation

The state vector includes 18 features:

- Well: oil rate, water cut, GOR, BHP, THP, status
- Equipment: pump efficiency, vibration, temperature, condition
- Market: oil price, gas price, operating cost
- Reservoir: permeability, porosity, pressure, temperature, viscosity

## Action Space

- **0**: Increase production (+30%)
- **1**: Decrease production (-30%)
- **2**: Maintain current rate

## Reward Function

Revenue minus operating cost, with penalties for high water cut and equipment stress when pushing production on degraded equipment.

## Running Tests

```bash
python test_api.py
```

---

Elaborado por Ing. Kelvin Cabrera
