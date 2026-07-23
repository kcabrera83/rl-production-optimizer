"""Q-Learning agent for production optimization."""

import numpy as np
import pickle
import os


class QLearningAgent:
    """Tabular Q-Learning agent with discrete action space.

    Actions: 0 = increase production, 1 = decrease production, 2 = maintain
    """

    def __init__(self, state_encoder, num_actions=3, learning_rate=0.1,
                 discount_factor=0.99, epsilon=1.0, epsilon_min=0.05,
                 epsilon_decay=0.995):
        self.encoder = state_encoder
        self.num_actions = num_actions
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.q_table = {}
        self.training_rewards = []

    def get_q_values(self, state):
        """Get Q-values for a state, initializing if needed."""
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.num_actions)
        return self.q_table[state]

    def choose_action(self, state):
        """Epsilon-greedy action selection."""
        if np.random.random() < self.epsilon:
            return np.random.randint(self.num_actions)
        q_values = self.get_q_values(state)
        return int(np.argmax(q_values))

    def update(self, state, action, reward, next_state, done):
        """Q-learning update rule."""
        current_q = self.get_q_values(state)
        if done:
            target = reward
        else:
            next_q = self.get_q_values(next_state)
            target = reward + self.gamma * np.max(next_q)
        current_q[action] += self.lr * (target - current_q[action])
        return float(current_q[action])

    def decay_epsilon(self):
        """Decay exploration rate."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def train_episode(self, scenarios, data_generator):
        """Train on a single episode of scenarios."""
        total_reward = 0
        state = self.encoder.encode_state(scenarios[0], mode="discrete")

        for i, scenario in enumerate(scenarios):
            action = self.choose_action(state)
            reward = data_generator.calculate_reward(scenario, action)
            total_reward += reward
            done = (i == len(scenarios) - 1)

            if not done:
                next_scenario = scenarios[i + 1]
                next_state = self.encoder.encode_state(next_scenario, mode="discrete")
            else:
                next_state = state

            self.update(state, action, reward, next_state, done)
            state = next_state

        self.decay_epsilon()
        self.training_rewards.append(total_reward)
        return total_reward

    def predict(self, scenario):
        """Predict optimal action for a scenario."""
        state = self.encoder.encode_state(scenario, mode="discrete")
        q_values = self.get_q_values(state)
        action = int(np.argmax(q_values))
        action_names = ["increase", "decrease", "maintain"]
        return {
            "action": action,
            "action_name": action_names[action],
            "q_values": q_values.tolist(),
        }

    def save(self, path):
        """Save Q-table to file."""
        data = {
            "q_table": {str(k): v.tolist() for k, v in self.q_table.items()},
            "epsilon": self.epsilon,
            "training_rewards": self.training_rewards,
            "hyperparameters": {
                "lr": self.lr,
                "gamma": self.gamma,
                "epsilon_min": self.epsilon_min,
                "epsilon_decay": self.epsilon_decay,
                "num_actions": self.num_actions,
            },
        }
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(data, f)

    @classmethod
    def load(cls, path, state_encoder):
        """Load Q-table from file."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        hp = data["hyperparameters"]
        agent = cls(
            state_encoder=state_encoder,
            num_actions=hp["num_actions"],
            learning_rate=hp["lr"],
            discount_factor=hp["gamma"],
            epsilon_min=hp["epsilon_min"],
            epsilon_decay=hp["epsilon_decay"],
        )
        agent.epsilon = data["epsilon"]
        agent.training_rewards = data["training_rewards"]
        agent.q_table = {eval(k): np.array(v) for k, v in data["q_table"].items()}
        return agent
