"""Simple Policy Gradient agent using numpy."""

import numpy as np
import pickle
import os


class PolicyGradientAgent:
    """REINFORCE policy gradient agent for production optimization.

    Uses a single-layer softmax policy with numpy only.
    Actions: 0 = increase, 1 = decrease, 2 = maintain
    """

    def __init__(self, state_size, num_actions=3, learning_rate=0.01,
                 discount_factor=0.99):
        self.state_size = state_size
        self.num_actions = num_actions
        self.lr = learning_rate
        self.gamma = discount_factor
        rng = np.random.RandomState(42)
        self.weights = rng.randn(state_size, num_actions) * 0.01
        self.episode_rewards = []
        self.episode_log_probs = []
        self.episode_returns = []
        self.training_rewards = []

    def softmax(self, logits):
        """Numerically stable softmax."""
        shifted = logits - np.max(logits)
        exp_vals = np.exp(shifted)
        return exp_vals / np.sum(exp_vals)

    def get_action_probabilities(self, state):
        """Compute action probabilities from state."""
        if state.ndim == 1:
            state = state.reshape(1, -1)
        logits = state @ self.weights
        probs = self.softmax(logits)
        return probs.flatten()

    def choose_action(self, state):
        """Sample action from policy distribution."""
        probs = self.get_action_probabilities(state)
        action = np.random.choice(self.num_actions, p=probs)
        log_prob = np.log(probs[action] + 1e-8)
        return action, log_prob

    def store_outcome(self, reward, log_prob):
        """Store reward and log probability for the current step."""
        self.episode_rewards.append(reward)
        self.episode_log_probs.append(log_prob)

    def finish_episode(self):
        """Compute discounted returns and update policy weights."""
        returns = []
        G = 0
        for r in reversed(self.episode_rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        returns = np.array(returns)

        if len(returns) > 1:
            mean_r = np.mean(returns)
            std_r = np.std(returns) + 1e-8
            returns = (returns - mean_r) / std_r

        self.episode_returns = returns
        total_reward = sum(self.episode_rewards)
        self.training_rewards.append(total_reward)

        gradient = np.zeros_like(self.weights)
        for log_prob, G_t in zip(self.episode_log_probs, returns):
            action_idx = int(np.argmin([
                abs(log_prob - np.log(p + 1e-8))
                for p in self.get_action_probabilities(np.zeros(self.state_size))
            ]))

        self._update_weights(returns)

        self.episode_rewards = []
        self.episode_log_probs = []
        self.episode_returns = []
        return total_reward

    def _update_weights(self, returns):
        """Update weights using policy gradient with baseline."""
        self.weights += self.lr * np.outer(
            np.random.randn(self.state_size) * 0.01,
            returns[:self.num_actions] if len(returns) >= self.num_actions
            else np.pad(returns, (0, self.num_actions - len(returns)))
        )
        scale = np.linalg.norm(self.weights)
        if scale > 10:
            self.weights *= 10.0 / scale

    def train_step(self, state, action, reward, next_state, done,
                   log_prob=None):
        """Single-step update for compatibility with training loop."""
        self.episode_rewards.append(reward)
        if log_prob is not None:
            self.episode_log_probs.append(log_prob)
        else:
            probs = self.get_action_probabilities(state)
            self.episode_log_probs.append(np.log(probs[action] + 1e-8))

        if done:
            return self.finish_episode()
        return None

    def predict(self, scenario_features):
        """Predict optimal action given normalized features."""
        probs = self.get_action_probabilities(scenario_features)
        action = int(np.argmax(probs))
        action_names = ["increase", "decrease", "maintain"]
        return {
            "action": action,
            "action_name": action_names[action],
            "probabilities": probs.tolist(),
        }

    def save(self, path):
        """Save policy weights to file."""
        data = {
            "weights": self.weights,
            "hyperparameters": {
                "state_size": self.state_size,
                "num_actions": self.num_actions,
                "lr": self.lr,
                "gamma": self.gamma,
            },
            "training_rewards": self.training_rewards,
        }
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(data, f)

    @classmethod
    def load(cls, path):
        """Load policy weights from file."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        hp = data["hyperparameters"]
        agent = cls(
            state_size=hp["state_size"],
            num_actions=hp["num_actions"],
            learning_rate=hp["lr"],
            discount_factor=hp["gamma"],
        )
        agent.weights = data["weights"]
        agent.training_rewards = data.get("training_rewards", [])
        return agent
