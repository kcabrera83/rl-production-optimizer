"""Synthetic data generator for oil production scenarios."""

import numpy as np


class ProductionDataGenerator:
    """Generates synthetic oil production scenarios for RL training."""

    def __init__(self, seed=42):
        self.rng = np.random.RandomState(seed)
        self.well_states = ["active", "shut_in", "maintenance", "workover"]
        self.equipment_conditions = ["optimal", "degraded", "critical", "failed"]

    def generate_well_state(self):
        """Generate a single well state vector."""
        oil_rate = self.rng.uniform(50, 500)
        water_cut = self.rng.uniform(0.1, 0.95)
        gas_oil_ratio = self.rng.uniform(100, 2000)
        bottom_hole_pressure = self.rng.uniform(500, 4000)
        tubing_head_pressure = self.rng.uniform(100, 1500)
        well_status = self.rng.choice(self.well_states, p=[0.5, 0.2, 0.2, 0.1])
        return {
            "oil_rate_bopd": round(oil_rate, 2),
            "water_cut": round(water_cut, 4),
            "gas_oil_ratio": round(gas_oil_ratio, 2),
            "bhp_psi": round(bottom_hole_pressure, 1),
            "thp_psi": round(tubing_head_pressure, 1),
            "well_status": well_status,
        }

    def generate_equipment_condition(self):
        """Generate equipment condition vector."""
        pump_efficiency = self.rng.uniform(0.3, 1.0)
        vibration_level = self.rng.uniform(0.0, 10.0)
        temperature = self.rng.uniform(20, 120)
        condition = self.rng.choice(
            self.equipment_conditions, p=[0.4, 0.3, 0.2, 0.1]
        )
        return {
            "pump_efficiency": round(pump_efficiency, 4),
            "vibration_mm_s": round(vibration_level, 2),
            "temperature_c": round(temperature, 1),
            "condition": condition,
        }

    def generate_market_prices(self):
        """Generate market price scenario."""
        oil_price = self.rng.uniform(40, 120)
        gas_price = self.rng.uniform(1.5, 6.0)
        operating_cost = self.rng.uniform(10, 50)
        return {
            "oil_price_usd_bbl": round(oil_price, 2),
            "gas_price_usd_mcf": round(gas_price, 2),
            "opex_usd_bbl": round(operating_cost, 2),
        }

    def generate_reservoir_properties(self):
        """Generate reservoir property vector."""
        permeability = self.rng.uniform(1, 5000)
        porosity = self.rng.uniform(0.05, 0.35)
        pressure = self.rng.uniform(1000, 8000)
        temperature_res = self.rng.uniform(60, 200)
        oil_viscosity = self.rng.uniform(0.5, 500)
        return {
            "permeability_md": round(permeability, 2),
            "porosity": round(porosity, 4),
            "reservoir_pressure_psi": round(pressure, 1),
            "reservoir_temperature_f": round(temperature_res, 1),
            "oil_viscosity_cp": round(oil_viscosity, 2),
        }

    def generate_scenario(self):
        """Generate a complete production scenario."""
        return {
            "well": self.generate_well_state(),
            "equipment": self.generate_equipment_condition(),
            "market": self.generate_market_prices(),
            "reservoir": self.generate_reservoir_properties(),
        }

    def generate_episode(self, num_steps=100):
        """Generate a sequence of scenarios for one episode."""
        scenarios = []
        for _ in range(num_steps):
            scenarios.append(self.generate_scenario())
        return scenarios

    def calculate_reward(self, scenario, action):
        """Calculate reward for a given scenario and action.

        Actions: 0 = increase production, 1 = decrease production, 2 = maintain
        """
        well = scenario["well"]
        market = scenario["market"]
        equipment = scenario["equipment"]

        base_revenue = well["oil_rate_bopd"] * market["oil_price_usd_bbl"]
        base_cost = well["oil_rate_bopd"] * market["opex_usd_bbl"]

        action_multipliers = {0: 1.3, 1: 0.7, 2: 1.0}
        rate_factor = action_multipliers[action]

        new_rate = well["oil_rate_bopd"] * rate_factor
        water_penalty = new_rate * well["water_cut"] * 5.0
        equipment_penalty = 0.0
        if action == 0 and equipment["condition"] in ("critical", "failed"):
            equipment_penalty = 500.0
        elif action == 0 and equipment["pump_efficiency"] < 0.5:
            equipment_penalty = 200.0

        revenue = new_rate * market["oil_price_usd_bbl"]
        cost = new_rate * market["opex_usd_bbl"] + water_penalty + equipment_penalty
        reward = revenue - cost
        return round(reward, 2)
