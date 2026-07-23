"""Deep Q-Network style agent using a simple numpy neural network."""

import numpy as np
import pickle
import os


class SimpleNeuralNetwork:
    """Two-layer neural network using numpy."""

    def __init__(self, input_size, hidden_size, output_size, lr=0.001,
                 rng=None):
        if rng is None:
            rng = np.random.RandomState(42)
        self.W1 = rng.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
        self.b1 = np.zeros(hidden_size)
        self.W2 = rng.randn(hidden_size, output_size) * np.sqrt(2.0 / hidden_size)
        self.b2 = np.zeros(output_size)
        self.lr = lr

    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, x):
        return (x > 0).astype(float)

    def forward(self, x):
        """Forward pass, returns hidden pre-activation for backprop."""
        self.z1 = x @ self.W1 + self.b1
        self.a1 = self.relu(self.z1)
        self.z2 = self.a1 @ self.W2 + self.b2
        return self.z2

    def predict(self, x):
        """Forward pass returning Q-values."""
        if x.ndim == 1:
            x = x.reshape(1, -1)
        logits = self.forward(x)
        return logits

    def train_step(self, x, target_q, actions):
        """One gradient step using MSE loss on selected action Q-values."""
        if x.ndim == 1:
            x = x.reshape(1, -1)
        batch_size = x.shape[0]

        q_values = self.forward(x)
        actions_onehot = np.zeros_like(q_values)
        actions_onehot[np.arange(batch_size), actions] = 1.0

        q_selected = np.sum(q_values * actions_onehot, axis=1)
        td_errors = q_selected - target_q
        loss = np.mean(td_errors ** 2)

        dz2 = (2.0 / batch_size) * td_errors.reshape(-1, 1) * actions_onehot
        dW2 = self.a1.T @ dz2
        db2 = np.sum(dz2, axis=0)

        da1 = dz2 @ self.W2.T
        dz1 = da1 * self.relu_derivative(self.z1)
        dW1 = x.T @ dz1
        db1 = np.sum(dz1, axis=0)

        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1

        return float(loss)


class ReplayBuffer:
    """Simple replay buffer for experience storage."""

    def __init__(self, capacity=10000):
        self.buffer = []
        self.capacity = capacity

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), min(batch_size, len(self.buffer)),
                                   replace=False)
        batch = [self.buffer[i] for i in indices]
        states = np.array([b[0] for b in batch])
        actions = np.array([b[1] for b in batch])
        rewards = np.array([b[2] for b in batch])
        next_states = np.array([b[3] for b in batch])
        dones = np.array([b[4] for b in batch], dtype=float)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    """Deep Q-Network agent using numpy neural network.

    Actions: 0 = increase, 1 = decrease, 2 = maintain
    """

    def __init__(self, state_size, num_actions=3, hidden_size=64,
                 learning_rate=0.001, discount_factor=0.99,
                 epsilon=1.0, epsilon_min=0.05, epsilon_decay=0.995,
                 target_update_freq=100, batch_size=32):
        self.state_size = state_size
        self.num_actions = num_actions
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.target_update_freq = target_update_freq
        self.batch_size = batch_size
        self.step_count = 0

        rng = np.random.RandomState(42)
        self.q_network = SimpleNeuralNetwork(state_size, hidden_size, num_actions,
                                             lr=learning_rate, rng=rng)
        self.target_network = SimpleNeuralNetwork(state_size, hidden_size, num_actions,
                                                  lr=learning_rate, rng=np.random.RandomState(99))
        self._sync_target()
        self.replay_buffer = ReplayBuffer()
        self.training_rewards = []

    def _sync_target(self):
        """Copy weights from Q-network to target network."""
        self.target_network.W1 = self.q_network.W1.copy()
        self.target_network.b1 = self.q_network.b1.copy()
        self.target_network.W2 = self.q_network.W2.copy()
        self.target_network.b2 = self.q_network.b2.copy()

    def choose_action(self, state):
        """Epsilon-greedy action selection."""
        if np.random.random() < self.epsilon:
            return np.random.randint(self.num_actions)
        q_values = self.q_network.predict(state)
        return int(np.argmax(q_values))

    def update(self, state, action, reward, next_state, done):
        """Store experience and train if enough samples."""
        self.replay_buffer.push(state, action, reward, next_state, done)
        loss = 0.0

        if len(self.replay_buffer) >= self.batch_size:
            states, actions, rewards, next_states, dones = \
                self.replay_buffer.sample(self.batch_size)
            next_q = self.target_network.predict(next_states)
            max_next_q = np.max(next_q, axis=1)
            targets = rewards + self.gamma * max_next_q * (1 - dones)
            loss = self.q_network.train_step(states, targets, actions)

        self.step_count += 1
        if self.step_count % self.target_update_freq == 0:
            self._sync_target()

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        return loss

    def predict(self, scenario_features):
        """Predict optimal action given normalized features."""
        q_values = self.q_network.predict(scenario_features).flatten()
        action = int(np.argmax(q_values))
        action_names = ["increase", "decrease", "maintain"]
        return {
            "action": action,
            "action_name": action_names[action],
            "q_values": q_values.tolist(),
        }

    def save(self, path):
        """Save agent to file."""
        data = {
            "q_network": {
                "W1": self.q_network.W1,
                "b1": self.q_network.b1,
                "W2": self.q_network.W2,
                "b2": self.q_network.b2,
            },
            "hyperparameters": {
                "state_size": self.state_size,
                "num_actions": self.num_actions,
                "epsilon": self.epsilon,
                "lr": self.q_network.lr,
                "gamma": self.gamma,
                "epsilon_min": self.epsilon_min,
                "epsilon_decay": self.epsilon_decay,
            },
            "training_rewards": self.training_rewards,
        }
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(data, f)

    @classmethod
    def load(cls, path):
        """Load agent from file."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        hp = data["hyperparameters"]
        agent = cls(
            state_size=hp["state_size"],
            num_actions=hp["num_actions"],
            learning_rate=hp["lr"],
            discount_factor=hp["gamma"],
            epsilon_min=hp["epsilon_min"],
            epsilon_decay=hp["epsilon_decay"],
        )
        agent.epsilon = hp["epsilon"]
        nw = data["q_network"]
        agent.q_network.W1 = nw["W1"]
        agent.q_network.b1 = nw["b1"]
        agent.q_network.W2 = nw["W2"]
        agent.q_network.b2 = nw["b2"]
        agent._sync_target()
        agent.training_rewards = data.get("training_rewards", [])
        return agent
