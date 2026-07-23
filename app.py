"""FastAPI application for RL Production Optimizer using Stable Baselines3 + Gymnasium."""

import os
import sys
import json
import numpy as np
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO, A2C, DQN

from rl_optimizer.data_generator import ProductionDataGenerator
from rl_optimizer.utils.state_encoder import StateEncoder

app = FastAPI(
    title="RL Production Optimizer",
    description="Reinforcement Learning Production Optimization for Oil & Gas (SB3 + Gymnasium)",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

Instrumentator().instrument(app).expose(app)

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "models")
data_gen = ProductionDataGenerator(seed=123)
encoder = StateEncoder(num_bins=5)

loaded_models = {}


class OilGasEnv(gym.Env):
    """Gymnasium environment for oil & gas production (for model loading)."""
    def __init__(self):
        super().__init__()
        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(low=0, high=1000, shape=(8,), dtype=np.float32)
        self.state = None
        self.step_count = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        scenario = data_gen.generate_scenario()
        self.state = np.array([
            scenario.get("flow_rate", 50),
            scenario.get("pressure", 150),
            scenario.get("temperature", 80),
            scenario.get("chemical_injection", 10),
            scenario.get("water_cut", 30),
            scenario.get("gas_oil_ratio", 100),
            scenario.get("wellhead_temp", 75),
            scenario.get("tubing_pressure", 200),
        ], dtype=np.float32)
        self.state = np.clip(self.state, 0, 1000)
        return self.state, {}

    def step(self, action):
        self.step_count += 1
        reward = float(np.sum(self.state[:3]) * 0.01 - abs(action - 2) * 0.1)
        noise = np.random.normal(0, 1, size=8) * 0.1
        self.state = np.clip(self.state + noise, 0, 1000).astype(np.float32)
        terminated = self.step_count >= 100
        return self.state, reward, terminated, False, {}


def load_models():
    """Load all trained SB3 models from disk."""
    global loaded_models

    ppo_path = os.path.join(MODELS_DIR, "ppo_model.zip")
    a2c_path = os.path.join(MODELS_DIR, "a2c_model.zip")
    dqn_path = os.path.join(MODELS_DIR, "dqn_model.zip")

    env = OilGasEnv()

    if os.path.exists(ppo_path):
        try:
            loaded_models["ppo"] = PPO.load(ppo_path, env=env)
            print("  Loaded SB3 PPO")
        except Exception as e:
            print(f"  Failed to load PPO: {e}")

    if os.path.exists(a2c_path):
        try:
            loaded_models["a2c"] = A2C.load(a2c_path, env=env)
            print("  Loaded SB3 A2C")
        except Exception as e:
            print(f"  Failed to load A2C: {e}")

    if os.path.exists(dqn_path):
        try:
            loaded_models["dqn"] = DQN.load(dqn_path, env=env)
            print("  Loaded SB3 DQN")
        except Exception as e:
            print(f"  Failed to load DQN: {e}")


class OptimizeRequest(BaseModel):
    scenario: Optional[Dict[str, Any]] = None


class SimulateRequest(BaseModel):
    num_steps: int = 50


def _predict_sb3(model, scenario):
    """Use SB3 model to predict action from scenario."""
    state = np.array([
        scenario.get("flow_rate", 50),
        scenario.get("pressure", 150),
        scenario.get("temperature", 80),
        scenario.get("chemical_injection", 10),
        scenario.get("water_cut", 30),
        scenario.get("gas_oil_ratio", 100),
        scenario.get("wellhead_temp", 75),
        scenario.get("tubing_pressure", 200),
    ], dtype=np.float32)
    state = np.clip(state, 0, 1000)
    action, _ = model.predict(state, deterministic=True)
    action = int(action)
    action_names = ["increase_rate", "decrease_rate", "hold_rate",
                    "increase_pressure", "decrease_pressure"]
    return {"action": action, "action_name": action_names[action % 5]}


@app.on_event("startup")
async def startup_event():
    load_models()


@app.get("/api/health")
async def health():
    return {
        "status": "healthy",
        "models_loaded": list(loaded_models.keys()),
        "version": "2.0.0",
        "frameworks": ["stable-baselines3", "gymnasium"],
    }


@app.get("/api/models")
async def list_models():
    models_info = {}
    summary_path = os.path.join(MODELS_DIR, "training_summary.json")
    summary = {}
    if os.path.exists(summary_path):
        with open(summary_path, "r") as f:
            summary = json.load(f)

    for name in ["ppo", "a2c", "dqn"]:
        models_info[name] = {
            "loaded": name in loaded_models,
            "file_exists": os.path.exists(os.path.join(MODELS_DIR, f"{name}_model.zip")),
            "training_stats": summary.get(name, {}),
            "framework": "stable-baselines3",
        }

    rllib_path = os.path.join(MODELS_DIR, "rllib_ppo")
    if os.path.isdir(rllib_path):
        models_info["rllib_ppo"] = {
            "loaded": False,
            "file_exists": True,
            "training_stats": summary.get("rllib_ppo", {}),
            "framework": "ray[rllib]",
        }

    return {"models": models_info}


@app.post("/api/optimize")
async def optimize(request: OptimizeRequest):
    scenario = data_gen.generate_scenario()
    if request.scenario:
        scenario = request.scenario

    results = {}
    for model_name, model in loaded_models.items():
        pred = _predict_sb3(model, scenario)
        results[model_name] = pred

    return {
        "scenario": scenario,
        "recommendations": results,
    }


@app.post("/api/simulate")
async def simulate(request: SimulateRequest):
    scenarios = data_gen.generate_episode(request.num_steps)

    trajectories = {}
    for model_name, model in loaded_models.items():
        actions_taken = []
        rewards = []
        for scenario in scenarios:
            pred = _predict_sb3(model, scenario)
            reward = data_gen.calculate_reward(scenario, pred["action"])
            actions_taken.append(pred["action_name"])
            rewards.append(reward)

        cumulative = []
        total = 0
        for r in rewards:
            total += r
            cumulative.append(round(total, 2))

        trajectories[model_name] = {
            "actions": actions_taken,
            "rewards": [round(r, 2) for r in rewards],
            "cumulative_reward": cumulative,
            "total_reward": round(sum(rewards), 2),
            "avg_reward": round(float(np.mean(rewards)), 2),
        }

    return {"simulation_length": request.num_steps, "trajectories": trajectories}


@app.get("/api/compare")
async def compare():
    summary_path = os.path.join(MODELS_DIR, "training_summary.json")
    if not os.path.exists(summary_path):
        raise HTTPException(status_code=404, detail="No training summary found. Run train.py first.")

    with open(summary_path, "r") as f:
        summary = json.load(f)

    comparison = {}
    for model_name in ["ppo", "a2c", "dqn"]:
        model_data = summary.get(model_name, {})
        if model_data:
            comparison[model_name] = {
                "algorithm": model_data.get("algorithm", model_name),
                "final_avg_reward": model_data.get("final_avg_reward", 0),
            }

    rllib_data = summary.get("rllib_ppo", {})
    if rllib_data:
        rewards = rllib_data.get("reward_curve", [])
        if rewards:
            comparison["rllib_ppo"] = {
                "algorithm": "PPO (Ray RLlib)",
                "final_avg_reward": rewards[-1] if rewards else 0,
                "best_reward": round(max(rewards), 2),
                "worst_reward": round(min(rewards), 2),
            }

    return {"comparison": comparison, "training_date": summary.get("training_date")}


@app.get("/api/docs")
async def api_docs():
    return {
        "openapi": "3.0.0",
        "info": {"title": "RL Production Optimizer", "version": "2.0.0"},
        "paths": {
            "/api/health": {"get": {"summary": "Health check"}},
            "/api/models": {"get": {"summary": "Model info"}},
            "/api/optimize": {"post": {"summary": "Get optimal action for a scenario"}},
            "/api/simulate": {"post": {"summary": "Run simulation episode"}},
            "/api/compare": {"get": {"summary": "Compare model performance"}},
        }
    }


if __name__ == "__main__":
    import uvicorn
    load_models()
    uvicorn.run(app, host="0.0.0.0", port=5019)
