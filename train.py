"""Train RL agents for oil production optimization."""

import os
import sys
import json
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rl_optimizer.data_generator import ProductionDataGenerator
from rl_optimizer.utils.state_encoder import StateEncoder
from rl_optimizer.models.q_learning import QLearningAgent
from rl_optimizer.models.policy_gradient import PolicyGradientAgent
from rl_optimizer.models.dqn_agent import DQNAgent

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "models")
NUM_EPISODES = 50
STEPS_PER_EPISODE = 100


def train_q_learning(data_gen, encoder):
    """Train Q-Learning agent."""
    print("\n--- Training Q-Learning Agent ---")
    agent = QLearningAgent(state_encoder=encoder, learning_rate=0.1,
                           discount_factor=0.99, epsilon=1.0)
    rewards = []
    for ep in range(NUM_EPISODES):
        scenarios = data_gen.generate_episode(STEPS_PER_EPISODE)
        total_reward = agent.train_episode(scenarios, data_gen)
        rewards.append(total_reward)
        if (ep + 1) % 10 == 0:
            avg = np.mean(rewards[-10:])
            print(f"  Episode {ep+1}/{NUM_EPISODES} | Avg Reward (last 10): {avg:.2f} | Epsilon: {agent.epsilon:.4f}")

    save_path = os.path.join(OUTPUT_DIR, "q_learning.pkl")
    agent.save(save_path)
    print(f"  Q-table saved to {save_path}")
    print(f"  Q-table size: {len(agent.q_table)} states")
    return {"rewards": rewards, "final_avg": np.mean(rewards[-10:]),
            "q_table_size": len(agent.q_table), "path": save_path}


def train_policy_gradient(data_gen, encoder):
    """Train Policy Gradient agent."""
    print("\n--- Training Policy Gradient Agent ---")
    agent = PolicyGradientAgent(state_size=encoder.state_size, num_actions=3,
                                learning_rate=0.01, discount_factor=0.99)
    rewards = []

    for ep in range(NUM_EPISODES):
        scenarios = data_gen.generate_episode(STEPS_PER_EPISODE)
        ep_reward = 0
        for i, scenario in enumerate(scenarios):
            state = encoder.encode_state(scenario, mode="normalized")
            action, log_prob = agent.choose_action(state)
            reward = data_gen.calculate_reward(scenario, action)
            ep_reward += reward
            done = (i == len(scenarios) - 1)
            result = agent.train_step(state, action, reward, state, done, log_prob)
        rewards.append(ep_reward)
        if (ep + 1) % 10 == 0:
            avg = np.mean(rewards[-10:])
            print(f"  Episode {ep+1}/{NUM_EPISODES} | Avg Reward (last 10): {avg:.2f}")

    save_path = os.path.join(OUTPUT_DIR, "policy_gradient.pkl")
    agent.save(save_path)
    print(f"  Policy saved to {save_path}")
    return {"rewards": rewards, "final_avg": np.mean(rewards[-10:]), "path": save_path}


def train_dqn(data_gen, encoder):
    """Train DQN agent."""
    print("\n--- Training DQN Agent ---")
    agent = DQNAgent(state_size=encoder.state_size, num_actions=3,
                     hidden_size=64, learning_rate=0.001,
                     discount_factor=0.99, epsilon=1.0)
    rewards = []

    for ep in range(NUM_EPISODES):
        scenarios = data_gen.generate_episode(STEPS_PER_EPISODE)
        ep_reward = 0
        for i, scenario in enumerate(scenarios):
            state = encoder.encode_state(scenario, mode="normalized")
            action = agent.choose_action(state)
            reward = data_gen.calculate_reward(scenario, action)
            ep_reward += reward
            done = (i == len(scenarios) - 1)
            next_scenario = scenarios[i + 1] if i < len(scenarios) - 1 else scenario
            next_state = encoder.encode_state(next_scenario, mode="normalized")
            agent.update(state, action, reward, next_state, done)
        agent.training_rewards.append(ep_reward)
        rewards.append(ep_reward)
        if (ep + 1) % 10 == 0:
            avg = np.mean(rewards[-10:])
            print(f"  Episode {ep+1}/{NUM_EPISODES} | Avg Reward (last 10): {avg:.2f} | Epsilon: {agent.epsilon:.4f}")

    save_path = os.path.join(OUTPUT_DIR, "dqn_agent.pkl")
    agent.save(save_path)
    print(f"  DQN model saved to {save_path}")
    return {"rewards": rewards, "final_avg": np.mean(rewards[-10:]), "path": save_path}


def save_training_summary(q_result, pg_result, dqn_result):
    """Save training summary JSON."""
    summary = {
        "training_date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "episodes": NUM_EPISODES,
        "steps_per_episode": STEPS_PER_EPISODE,
        "q_learning": {
            "final_avg_reward": round(q_result["final_avg"], 2),
            "q_table_size": q_result["q_table_size"],
            "reward_curve": [round(r, 2) for r in q_result["rewards"]],
        },
        "policy_gradient": {
            "final_avg_reward": round(pg_result["final_avg"], 2),
            "reward_curve": [round(r, 2) for r in pg_result["rewards"]],
        },
        "dqn": {
            "final_avg_reward": round(dqn_result["final_avg"], 2),
            "reward_curve": [round(r, 2) for r in dqn_result["rewards"]],
        },
    }
    summary_path = os.path.join(OUTPUT_DIR, "training_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nTraining summary saved to {summary_path}")
    return summary


def main():
    print("=" * 60)
    print("  RL Production Optimizer - Training Pipeline")
    print("=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    data_gen = ProductionDataGenerator(seed=42)
    encoder = StateEncoder(num_bins=5)

    start = time.time()

    q_result = train_q_learning(data_gen, encoder)
    pg_result = train_policy_gradient(data_gen, encoder)
    dqn_result = train_dqn(data_gen, encoder)

    elapsed = time.time() - start

    summary = save_training_summary(q_result, pg_result, dqn_result)

    print("\n" + "=" * 60)
    print("  Training Complete")
    print("=" * 60)
    print(f"  Total time: {elapsed:.1f}s")
    print(f"  Q-Learning  avg reward:  {q_result['final_avg']:.2f}")
    print(f"  Policy Grad avg reward:  {pg_result['final_avg']:.2f}")
    print(f"  DQN         avg reward:  {dqn_result['final_avg']:.2f}")
    print("=" * 60)

    return summary


if __name__ == "__main__":
    main()
