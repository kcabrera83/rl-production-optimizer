# API Documentation - RL Production Optimizer

## Base URL

```
http://localhost:5019
```

## Endpoints

### GET /

Serves the web dashboard (HTML).

**Response:** HTML page

---

### GET /api/health

Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "models_loaded": ["q_learning", "policy_gradient", "dqn"],
  "version": "1.0.0"
}
```

---

### GET /api/models

List available models and training statistics.

**Response:**
```json
{
  "models": {
    "q_learning": {
      "loaded": true,
      "file_exists": true,
      "training_stats": {
        "final_avg_reward": 85.23,
        "q_table_size": 3125,
        "reward_curve": [10.5, 25.3, ...]
      }
    },
    "policy_gradient": {
      "loaded": true,
      "file_exists": true,
      "training_stats": {
        "final_avg_reward": 72.15,
        "reward_curve": [8.2, 15.7, ...]
      }
    },
    "dqn": {
      "loaded": true,
      "file_exists": true,
      "training_stats": {
        "final_avg_reward": 91.42,
        "reward_curve": [12.1, 30.5, ...]
      }
    }
  }
}
```

---

### POST /api/optimize

Get optimal action recommendation for a production scenario.

**Request:**
```json
{}
```

Or with a custom scenario:
```json
{
  "scenario": {
    "oil_rate": 150.0,
    "water_cut": 0.35,
    "gor": 500.0,
    "bhp": 2500.0,
    "thp": 800.0,
    "well_status": 1,
    "pump_efficiency": 0.82,
    "vibration": 0.45,
    "equipment_temp": 75.0,
    "equipment_condition": 0.85,
    "oil_price": 75.0,
    "gas_price": 3.5,
    "operating_cost": 25.0,
    "permeability": 150.0,
    "porosity": 0.22,
    "reservoir_pressure": 3000.0,
    "reservoir_temp": 180.0,
    "oil_viscosity": 2.5
  }
}
```

**Response (empty body generates random scenario):**
```json
{
  "scenario": {
    "oil_rate": 150.0,
    "water_cut": 0.35,
    "gor": 500.0,
    ...
  },
  "recommendations": {
    "q_learning": {
      "action": 0,
      "action_name": "increase",
      "confidence": 0.85
    },
    "policy_gradient": {
      "action": 2,
      "action_name": "maintain",
      "confidence": 0.72
    },
    "dqn": {
      "action": 0,
      "action_name": "increase",
      "confidence": 0.91
    }
  }
}
```

---

### POST /api/simulate

Run a multi-step simulation episode and return trajectories.

**Request:**
```json
{
  "num_steps": 50
}
```

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| num_steps | integer | No | 50 | Number of simulation steps |

**Response:**
```json
{
  "simulation_length": 50,
  "trajectories": {
    "q_learning": {
      "actions": ["increase", "maintain", "decrease", ...],
      "rewards": [12.5, 8.3, -5.2, ...],
      "cumulative_reward": [12.5, 20.8, 15.6, ...],
      "total_reward": 425.60,
      "avg_reward": 8.51
    },
    "policy_gradient": {
      "actions": ["maintain", "increase", "maintain", ...],
      "rewards": [5.2, 15.8, 4.1, ...],
      "cumulative_reward": [5.2, 21.0, 25.1, ...],
      "total_reward": 380.20,
      "avg_reward": 7.60
    },
    "dqn": {
      "actions": ["increase", "increase", "maintain", ...],
      "rewards": [15.2, 18.5, 6.3, ...],
      "cumulative_reward": [15.2, 33.7, 40.0, ...],
      "total_reward": 512.30,
      "avg_reward": 10.25
    }
  }
}
```

---

### GET /api/compare

Compare model performance using training summary data.

**Response:**
```json
{
  "comparison": {
    "q_learning": {
      "final_avg_reward": 85.23,
      "best_reward": 125.50,
      "worst_reward": -10.20,
      "reward_std": 32.45,
      "convergence_episode": 35
    },
    "policy_gradient": {
      "final_avg_reward": 72.15,
      "best_reward": 110.30,
      "worst_reward": -5.80,
      "reward_std": 28.90,
      "convergence_episode": 40
    },
    "dqn": {
      "final_avg_reward": 91.42,
      "best_reward": 135.80,
      "worst_reward": 5.20,
      "reward_std": 22.15,
      "convergence_episode": 30
    }
  },
  "training_date": "2024-01-15 10:30:00"
}
```

---

### GET /api/docs

Returns OpenAPI 3.0.0 specification.

**Response:**
```json
{
  "openapi": "3.0.0",
  "info": {"title": "RL Production Optimizer", "version": "1.0.0"},
  "paths": {
    "/api/health": {"get": {"summary": "Health check"}},
    "/api/models": {"get": {"summary": "Model info"}},
    "/api/optimize": {"post": {"summary": "Get optimal action for a scenario"}},
    "/api/simulate": {"post": {"summary": "Run simulation episode"}},
    "/api/compare": {"get": {"summary": "Compare model performance"}}
  }
}
```

## Action Space

| Action | Value | Description |
|--------|-------|-------------|
| increase | 0 | Increase production rate by 30% |
| decrease | 1 | Decrease production rate by 30% |
| maintain | 2 | Maintain current production rate |

## State Features (18 dimensions)

| Feature | Description |
|---------|-------------|
| oil_rate | Current oil production rate (bbl/day) |
| water_cut | Water cut ratio (0-1) |
| gor | Gas-oil ratio |
| bhp | Bottom-hole pressure (PSI) |
| thp | Tubing-head pressure (PSI) |
| well_status | Well status flag |
| pump_efficiency | Artificial lift pump efficiency (0-1) |
| vibration | Equipment vibration level |
| equipment_temp | Equipment temperature |
| equipment_condition | Equipment health score (0-1) |
| oil_price | Market oil price ($/bbl) |
| gas_price | Market gas price ($/MCF) |
| operating_cost | Operating cost per unit |
| permeability | Reservoir permeability (mD) |
| porosity | Rock porosity (0-1) |
| reservoir_pressure | Reservoir pressure (PSI) |
| reservoir_temp | Reservoir temperature |
| oil_viscosity | Oil viscosity (cP) |

## Error Codes

| Code | Description |
|------|-------------|
| 200 | Success |
| 400 | Bad request |
| 404 | Training summary not found (run train.py first) |
| 500 | Internal server error |
