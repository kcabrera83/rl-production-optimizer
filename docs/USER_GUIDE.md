# User Guide - RL Production Optimizer

## Overview

The RL Production Optimizer uses reinforcement learning to optimize oil field production. Three RL agents (Q-Learning, Policy Gradient, DQN) learn to make production rate decisions (increase, decrease, maintain) based on well state, equipment condition, market prices, and reservoir properties.

## Getting Started

### Prerequisites

- Python 3.8+
- pip

### Installation

```bash
cd rl-production-optimizer
pip install -r requirements.txt
```

### Train Models

```bash
python train.py
```

Trains all three RL agents over 50 episodes with 100 steps each. Models are saved to `outputs/models/`.

### Start the API Server

```bash
python app.py
```

The server starts on `http://localhost:5019`.

### Open the Dashboard

Navigate to `http://localhost:5019` in your browser.

### Run Tests

```bash
python test_api.py
```

## Dashboard Features

- **Optimize**: Get optimal action recommendations from all 3 agents
- **Simulate**: Run multi-step simulation episodes and visualize trajectories
- **Compare Models**: View training performance metrics and convergence
- **Reward Curves**: Chart.js visualizations of reward progression

## RL Algorithms

### Q-Learning (Tabular)
- Discretized state space with 5 bins per feature
- Epsilon-greedy exploration (decays over episodes)
- Learning rate: 0.1, Discount factor: 0.99

### Policy Gradient (REINFORCE)
- Monte Carlo policy gradient method
- Softmax action selection
- Learning rate: 0.01, Discount factor: 0.99

### DQN (Deep Q-Network)
- Neural network with hidden size 64
- Experience replay buffer
- Epsilon-greedy with decay
- Learning rate: 0.001, Discount factor: 0.99

## API Usage

### Using curl

**Get optimal action:**
```bash
curl -X POST http://localhost:5019/api/optimize \
  -H "Content-Type: application/json" \
  -d '{}'
```

**Run simulation:**
```bash
curl -X POST http://localhost:5019/api/simulate \
  -H "Content-Type: application/json" \
  -d '{"num_steps": 50}'
```

**Compare models:**
```bash
curl http://localhost:5019/api/compare
```

### Using Python

```python
import requests

# Get optimal action for a random scenario
response = requests.post("http://localhost:5019/api/optimize", json={})
result = response.json()

print("Scenario:", result["scenario"])
for model, rec in result["recommendations"].items():
    print(f"  {model}: {rec['action_name']} (confidence: {rec['confidence']:.2f})")

# Run simulation
response = requests.post("http://localhost:5019/api/simulate", json={"num_steps": 50})
sim = response.json()
for model, traj in sim["trajectories"].items():
    print(f"  {model}: total_reward={traj['total_reward']}, avg={traj['avg_reward']}")

# Compare models
response = requests.get("http://localhost:5019/api/compare")
comparison = response.json()["comparison"]
for model, metrics in comparison.items():
    print(f"  {model}: final_avg={metrics['final_avg_reward']}, convergence=ep{metrics['convergence_episode']}")
```

### Using JavaScript

```javascript
// Get recommendations
const response = await fetch("http://localhost:5019/api/optimize", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({})
});
const data = await response.json();
Object.entries(data.recommendations).forEach(([model, rec]) => {
  console.log(`${model}: ${rec.action_name} (confidence: ${rec.confidence})`);
});
```
