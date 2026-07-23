"""FastAPI application for RL Production Optimizer."""

import os
import sys
import json
import numpy as np
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rl_optimizer.data_generator import ProductionDataGenerator
from rl_optimizer.utils.state_encoder import StateEncoder
from rl_optimizer.models.q_learning import QLearningAgent
from rl_optimizer.models.policy_gradient import PolicyGradientAgent
from rl_optimizer.models.dqn_agent import DQNAgent

app = FastAPI(
    title="RL Production Optimizer",
    description="Reinforcement Learning Production Optimization for Oil & Gas",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "models")
data_gen = ProductionDataGenerator(seed=123)
encoder = StateEncoder(num_bins=5)

loaded_models = {}


def load_models():
    """Load all trained models from disk."""
    global loaded_models
    q_path = os.path.join(MODELS_DIR, "q_learning.pkl")
    pg_path = os.path.join(MODELS_DIR, "policy_gradient.pkl")
    dqn_path = os.path.join(MODELS_DIR, "dqn_agent.pkl")

    if os.path.exists(q_path):
        try:
            loaded_models["q_learning"] = QLearningAgent.load(q_path, encoder)
        except Exception as e:
            print(f"Failed to load Q-Learning: {e}")

    if os.path.exists(pg_path):
        try:
            loaded_models["policy_gradient"] = PolicyGradientAgent.load(pg_path)
        except Exception as e:
            print(f"Failed to load Policy Gradient: {e}")

    if os.path.exists(dqn_path):
        try:
            loaded_models["dqn"] = DQNAgent.load(dqn_path)
        except Exception as e:
            print(f"Failed to load DQN: {e}")


class OptimizeRequest(BaseModel):
    scenario: Optional[Dict[str, Any]] = None


class SimulateRequest(BaseModel):
    num_steps: int = 50


def _find_convergence(rewards, window=10, threshold=0.05):
    """Find episode where reward stabilized."""
    if len(rewards) < window:
        return len(rewards)
    for i in range(window, len(rewards)):
        segment = rewards[i - window:i]
        mean_val = np.mean(segment)
        if mean_val != 0 and np.std(segment) / abs(mean_val) < threshold:
            return i - window
    return len(rewards)


@app.on_event("startup")
async def startup_event():
    load_models()


@app.get("/api/health")
async def health():
    return {
        "status": "healthy",
        "models_loaded": list(loaded_models.keys()),
        "version": "1.0.0",
    }


@app.get("/api/models")
async def list_models():
    models_info = {}
    summary_path = os.path.join(MODELS_DIR, "training_summary.json")
    summary = {}
    if os.path.exists(summary_path):
        with open(summary_path, "r") as f:
            summary = json.load(f)

    for name in ["q_learning", "policy_gradient", "dqn"]:
        models_info[name] = {
            "loaded": name in loaded_models,
            "file_exists": os.path.exists(os.path.join(MODELS_DIR, f"{name}.pkl")),
            "training_stats": summary.get(name, {}),
        }
    return {"models": models_info}


@app.post("/api/optimize")
async def optimize(request: OptimizeRequest):
    scenario = data_gen.generate_scenario()
    if request.scenario:
        scenario = request.scenario

    results = {}
    for model_name, model in loaded_models.items():
        if model_name == "q_learning":
            pred = model.predict(scenario)
        elif model_name == "policy_gradient":
            features = encoder.encode_state(scenario, mode="normalized")
            pred = model.predict(features)
        elif model_name == "dqn":
            features = encoder.encode_state(scenario, mode="normalized")
            pred = model.predict(features)
        else:
            continue
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
            if model_name == "q_learning":
                pred = model.predict(scenario)
            elif model_name == "policy_gradient":
                features = encoder.encode_state(scenario, mode="normalized")
                pred = model.predict(features)
            elif model_name == "dqn":
                features = encoder.encode_state(scenario, mode="normalized")
                pred = model.predict(features)
            else:
                continue
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
    for model_name in ["q_learning", "policy_gradient", "dqn"]:
        model_data = summary.get(model_name, {})
        rewards = model_data.get("reward_curve", [])
        if rewards:
            comparison[model_name] = {
                "final_avg_reward": model_data.get("final_avg_reward", 0),
                "best_reward": round(max(rewards), 2),
                "worst_reward": round(min(rewards), 2),
                "reward_std": round(float(np.std(rewards)), 2),
                "convergence_episode": _find_convergence(rewards),
            }

    return {"comparison": comparison, "training_date": summary.get("training_date")}


@app.get("/api/docs")
async def api_docs():
    return {
        "openapi": "3.0.0",
        "info": {"title": "RL Production Optimizer", "version": "1.0.0"},
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
