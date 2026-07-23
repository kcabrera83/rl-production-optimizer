"""Training script using Stable Baselines3, Gymnasium, and Ray RLlib."""

import os
import sys
import json
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.env_util import make_vec_env

from rl_optimizer.data_generator import ProductionDataGenerator

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "models")
NUM_EPISODES = 50
STEPS_PER_EPISODE = 100


class OilGasEnv(gym.Env):
    """Custom Gymnasium environment for oil & gas production"""
    metadata = {"render_modes": ["human"]}

    def __init__(self, data_gen=None, seed=42):
        super().__init__()
        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(low=0, high=1000, shape=(8,), dtype=np.float32)
        self.data_gen = data_gen or ProductionDataGenerator(seed=seed)
        self.state = None
        self.step_count = 0
        self.max_steps = STEPS_PER_EPISODE

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        scenario = self.data_gen.generate_scenario()
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
        reward = self._calculate_reward(action)
        self.state = self._next_state(action)
        terminated = self.step_count >= self.max_steps
        truncated = False
        return self.state, reward, terminated, truncated, {}

    def _calculate_reward(self, action):
        return float(np.sum(self.state[:3]) * 0.01 - abs(action - 2) * 0.1)

    def _next_state(self, action):
        noise = np.random.normal(0, 1, size=8) * 0.1
        return np.clip(self.state + noise, 0, 1000).astype(np.float32)


def train_ppo():
    """Train PPO agent using Stable Baselines3"""
    print("\n--- Training PPO Agent (SB3) ---")
    env = OilGasEnv()
    model = PPO("MlpPolicy", env, verbose=1, learning_rate=3e-4, n_steps=2048, batch_size=64)
    model.learn(total_timesteps=50000)
    save_path = os.path.join(OUTPUT_DIR, "ppo_model")
    model.save(save_path)
    print(f"  PPO model saved to {save_path}")
    return {"path": save_path, "algorithm": "PPO"}


def train_a2c():
    """Train A2C agent using Stable Baselines3"""
    print("\n--- Training A2C Agent (SB3) ---")
    env = OilGasEnv()
    model = A2C("MlpPolicy", env, verbose=1, learning_rate=7e-4, n_steps=5)
    model.learn(total_timesteps=50000)
    save_path = os.path.join(OUTPUT_DIR, "a2c_model")
    model.save(save_path)
    print(f"  A2C model saved to {save_path}")
    return {"path": save_path, "algorithm": "A2C"}


def train_dqn():
    """Train DQN agent using Stable Baselines3"""
    print("\n--- Training DQN Agent (SB3) ---")
    env = OilGasEnv()
    model = DQN("MlpPolicy", env, verbose=1, learning_rate=1e-4, buffer_size=10000)
    model.learn(total_timesteps=50000)
    save_path = os.path.join(OUTPUT_DIR, "dqn_model")
    model.save(save_path)
    print(f"  DQN model saved to {save_path}")
    return {"path": save_path, "algorithm": "DQN"}


def train_rllib():
    """Train using Ray RLlib (PPO) - optional, requires compatible Python."""
    print("\n--- Training with Ray RLlib PPO ---")
    try:
        from ray.rllib.algorithms.ppo import PPOConfig
        config = (
            PPOConfig()
            .environment(env=OilGasEnv)
            .training(lr=3e-4, train_batch_size=4096)
            .resources(num_workers=1)
        )
        algo = config.build()
        rewards = []
        for i in range(5):
            result = algo.train()
            r = result.get("episode_reward_mean", 0)
            rewards.append(float(r))
            print(f"  Iteration {i+1}/5 | Avg Reward: {r:.2f}")
        algo.stop()
        save_path = os.path.join(OUTPUT_DIR, "rllib_ppo")
        os.makedirs(save_path, exist_ok=True)
        with open(os.path.join(save_path, "reward_curve.json"), "w") as f:
            json.dump(rewards, f)
        print(f"  RLlib PPO saved to {save_path}")
        return {"path": save_path, "algorithm": "RLlib_PPO", "rewards": rewards}
    except Exception as e:
        print(f"  RLlib training failed (non-critical): {e}")
        return None


def save_training_summary(ppo_result, a2c_result, dqn_result, rllib_result=None):
    """Save training summary JSON."""
    summary = {
        "training_date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_timesteps": 50000,
        "ppo": {
            "algorithm": "PPO (Stable Baselines3)",
            "final_avg_reward": 0,
            "path": ppo_result["path"],
        },
        "a2c": {
            "algorithm": "A2C (Stable Baselines3)",
            "final_avg_reward": 0,
            "path": a2c_result["path"],
        },
        "dqn": {
            "algorithm": "DQN (Stable Baselines3)",
            "final_avg_reward": 0,
            "path": dqn_result["path"],
        },
    }
    if rllib_result:
        summary["rllib_ppo"] = {
            "algorithm": "PPO (Ray RLlib)",
            "final_avg_reward": rllib_result.get("rewards", [0])[-1],
            "path": rllib_result["path"],
        }
    summary_path = os.path.join(OUTPUT_DIR, "training_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nTraining summary saved to {summary_path}")
    return summary


def main():
    print("=" * 60)
    print("  RL Production Optimizer - Training Pipeline")
    print("  Stable Baselines3 + Gymnasium + Ray RLlib")
    print("=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    start = time.time()

    ppo_result = train_ppo()
    a2c_result = train_a2c()
    dqn_result = train_dqn()
    rllib_result = train_rllib()

    elapsed = time.time() - start

    summary = save_training_summary(ppo_result, a2c_result, dqn_result, rllib_result)

    print("\n" + "=" * 60)
    print("  Training Complete")
    print("=" * 60)
    print(f"  Total time: {elapsed:.1f}s")
    print(f"  Algorithms: PPO, A2C, DQN (SB3) + RLlib PPO")
    print("=" * 60)

    return summary


if __name__ == "__main__":
    main()
