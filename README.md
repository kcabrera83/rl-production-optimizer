# RL Production Optimizer

Reinforcement Learning system for optimizing oil production using Q-learning, policy gradient, and Deep Q-Network approaches.

## Overview

This project applies reinforcement learning to oil field production optimization. An RL agent learns to make decisions about production rate adjustments (increase, decrease, or maintain) based on well states, equipment conditions, market prices, and reservoir properties.

Three RL algorithms are implemented:

- **Q-Learning**: Tabular method with discretized state space and epsilon-greedy exploration
- **Policy Gradient (REINFORCE)**: Monte Carlo policy gradient with softmax action selection
- **DQN Agent**: Neural network approximation with experience replay and target network

## Project Structure

```
rl-production-optimizer/
    rl_optimizer/
        __init__.py
        data_generator.py          # Synthetic production data generation
        models/
            __init__.py
            q_learning.py          # Tabular Q-Learning agent
            policy_gradient.py     # REINFORCE policy gradient agent
            dqn_agent.py           # Deep Q-Network agent (numpy)
        utils/
            __init__.py
            state_encoder.py       # State normalization and discretization
    outputs/models/                # Trained model artifacts
    templates/
        index.html                 # Web dashboard
    train.py                       # Training pipeline
    app.py                         # Flask API server (port 5019)
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

This trains all three agents over 50 episodes and saves models to `outputs/models/`.

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

## Elaborado por

Ing. Kelvin Cabrera
