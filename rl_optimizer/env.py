"""Gymnasium environment for oil & gas production optimization."""

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from rl_optimizer.data_generator import ProductionDataGenerator

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
        oil_rate = self.state[0]
        pressure = self.state[1]
        temperature = self.state[2]
        water_cut = self.state[4]

        oil_price = 70.0
        gas_price = 3.0
        opex_per_bbl = 20.0
        gas_rate = self.state[5] / 1000.0

        revenue = oil_rate * oil_price + gas_rate * gas_price
        cost = oil_rate * opex_per_bbl + water_cut * oil_rate * 5.0

        if action == 0:
            cost += 50.0
        elif action == 1:
            cost += 30.0
        elif action == 3:
            cost += 40.0
        elif action == 4:
            cost += 35.0

        reward = revenue - cost
        return float(reward * 0.001)

    def _next_state(self, action):
        state = self.state.copy()
        if action == 0:
            state[0] *= 1.02
            state[4] += 0.3
            state[5] += 5.0
            state[7] += 2.0
        elif action == 1:
            state[0] *= 0.98
            state[4] -= 0.2
            state[5] -= 3.0
            state[7] -= 1.5
        elif action == 2:
            state[0] *= 0.998
            state[4] -= 0.05
        elif action == 3:
            state[1] += 5.0
            state[7] += 8.0
            state[0] *= 1.01
        elif action == 4:
            state[1] -= 4.0
            state[7] -= 6.0
            state[0] *= 0.995

        noise = np.random.normal(0, 0.5, size=8)
        state += noise
        state = np.clip(state, 0, 1000).astype(np.float32)
        self.state = state
        return self.state
