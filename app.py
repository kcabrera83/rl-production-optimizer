"""Flask API for RL Production Optimizer."""

import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flask import Flask, render_template, jsonify, request
from rl_optimizer.data_generator import ProductionDataGenerator
from rl_optimizer.utils.state_encoder import StateEncoder
from rl_optimizer.models.q_learning import QLearningAgent
from rl_optimizer.models.policy_gradient import PolicyGradientAgent
from rl_optimizer.models.dqn_agent import DQNAgent

app = Flask(__name__)

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


@app.route("/")
def index():
    """Serve the main dashboard."""
    return render_template("index.html")


@app.route("/api/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "models_loaded": list(loaded_models.keys()),
        "version": "1.0.0",
    })


@app.route("/api/models", methods=["GET"])
def list_models():
    """List available models and their status."""
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
    return jsonify({"models": models_info})


@app.route("/api/optimize", methods=["POST"])
def optimize():
    """Get optimal action for a given scenario."""
    data = request.get_json(silent=True) or {}
    scenario = data_gen.generate_scenario()
    if "scenario" in data:
        scenario = data["scenario"]

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

    return jsonify({
        "scenario": scenario,
        "recommendations": results,
    })


@app.route("/api/simulate", methods=["POST"])
def simulate():
    """Run a simulation episode and return trajectory."""
    data = request.get_json(silent=True) or {}
    num_steps = data.get("num_steps", 50)
    scenarios = data_gen.generate_episode(num_steps)

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
            "avg_reward": round(np.mean(rewards), 2),
        }

    return jsonify({"simulation_length": num_steps, "trajectories": trajectories})


@app.route("/api/compare", methods=["GET"])
def compare():
    """Compare model performance using training summary."""
    summary_path = os.path.join(MODELS_DIR, "training_summary.json")
    if not os.path.exists(summary_path):
        return jsonify({"error": "No training summary found. Run train.py first."}), 404

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

    return jsonify({"comparison": comparison, "training_date": summary.get("training_date")})


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


load_models()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5019, debug=False)
