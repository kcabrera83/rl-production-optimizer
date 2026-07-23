# Architecture - RL Production Optimizer

## System Overview

```
┌─────────────────────────────────────────────────────────┐
│                    Web Dashboard (HTML)                   │
│                    Port 5019 /                           │
├─────────────────────────────────────────────────────────┤
│                    Flask API Layer                       │
│     /api/optimize  /api/simulate  /api/compare          │
├──────────┬──────────────┬──────────────┬─────────────────┤
│ Q-       │ Policy       │ DQN          │ State           │
│ Learning │ Gradient     │ Agent        │ Encoder         │
│ (Tabular)│ (REINFORCE)  │ (NumPy NN)   │ (Normalize)     │
├──────────┴──────────────┴──────────────┴─────────────────┤
│           Production Data Generator                       │
│    18 state features / 3 actions / Reward function       │
└─────────────────────────────────────────────────────────┘
```

## Components

### Data Layer

- **Production Data Generator**: Creates synthetic oil production scenarios
- **State Space**: 18 features (well, equipment, market, reservoir)
- **Action Space**: 3 discrete actions (increase +30%, decrease -30%, maintain)
- **Reward Function**: Revenue minus operating cost, with penalties for high water cut and equipment stress
- **Episode Generation**: 100 steps per episode, 50 episodes for training

### Model Layer

#### Q-Learning Agent (`QLearningAgent`)
- **Algorithm**: Tabular Q-Learning with epsilon-greedy exploration
- **State Discretization**: 5 bins per feature via `StateEncoder`
- **Q-Table**: Dictionary mapping discretized states to action values
- **Parameters**: lr=0.1, gamma=0.99, epsilon=1.0 (decays to 0.01)
- **Training**: 50 episodes × 100 steps
- **Persistence**: Pickle (.pkl) - includes Q-table and encoder

#### Policy Gradient Agent (`PolicyGradientAgent`)
- **Algorithm**: REINFORCE (Monte Carlo Policy Gradient)
- **Policy Network**: Linear(state_size → 3) with softmax
- **Action Selection**: Softmax probability sampling
- **Parameters**: lr=0.01, gamma=0.99
- **Training**: 50 episodes × 100 steps
- **Persistence**: Pickle (.pkl) - includes policy weights and state_size

#### DQN Agent (`DQNAgent`)
- **Algorithm**: Deep Q-Network with experience replay
- **Network**: Linear(18 → 64) → ReLU → Linear(64 → 3)
- **Experience Replay**: Buffer-based with batch sampling
- **Epsilon Decay**: 1.0 → 0.01 over training
- **Parameters**: lr=0.001, gamma=0.99, hidden_size=64
- **Training**: 50 episodes × 100 steps
- **Persistence**: Pickle (.pkl) - includes network weights and optimizer state

#### State Encoder (`StateEncoder`)
- **Normalization**: Min-max normalization of 18 features
- **Discretization**: 5 bins per feature for Q-learning state space
- **State Size**: 18 (normalized mode) or 18 (discretized mode)
- **Bin Count**: Configurable (default: 5)

### API Layer

- **Framework**: Flask (Python)
- **Serialization**: JSON request/response
- **Model Loading**: Pickle deserialization at startup
- **Port**: 5019

### Dashboard Layer

- **Frontend**: HTML/CSS/JavaScript (single page)
- **Visualization**: Chart.js for reward curves, action distributions
- **Style**: Dark-themed responsive UI

## Data Flow

### Optimization Pipeline

```
1. Input Scenario (or random generation)
   ↓
2. State Encoder (normalize features)
   ↓
3. Each Agent Predicts
   ├── Q-Learning: discretize → lookup Q-table → argmax
   ├── Policy Gradient: normalize → softmax → sample action
   └── DQN: normalize → neural network forward pass → argmax
   ↓
4. Action Recommendations (with confidence)
   ↓
5. Dashboard Display
```

### Simulation Pipeline

```
1. Generate Episode (num_steps scenarios)
   ↓
2. For Each Step:
   ├── Agent predicts action
   ├── Calculate reward from scenario + action
   └── Record action + reward
   ↓
3. Compute cumulative rewards
   ↓
4. Return trajectories per model
```

### Training Pipeline

```
1. Generate Training Episodes (50 × 100 steps)
   ↓
2. Q-Learning:
   ├── Discretize state
   ├── Epsilon-greedy action selection
   ├── Update Q-table: Q(s,a) += lr[r + gamma*max(Q(s')) - Q(s,a)]
   └── Decay epsilon
   ↓
3. Policy Gradient:
   ├── Encode state
   ├── Sample action from softmax policy
   ├── Compute policy gradient
   └── Update policy weights
   ↓
4. DQN:
   ├── Encode state
   ├── Epsilon-greedy action selection
   ├── Store experience in replay buffer
   ├── Sample batch and update network
   └── Decay epsilon
   ↓
5. Save all models + training summary
```

## Technology Stack

| Component | Technology |
|-----------|------------|
| Language | Python 3.8+ |
| Web Framework | Flask |
| RL Framework | Custom NumPy implementations |
| Numerical | NumPy |
| Model Persistence | Pickle |
| Frontend | HTML/CSS/JS + Chart.js |

## Model Artifacts

| File | Description |
|------|-------------|
| `outputs/models/q_learning.pkl` | Q-table + encoder |
| `outputs/models/policy_gradient.pkl` | Policy network weights |
| `outputs/models/dqn_agent.pkl` | DQN network weights + replay buffer |
| `outputs/models/training_summary.json` | Training metrics and reward curves |
