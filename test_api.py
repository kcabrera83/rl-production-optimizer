"""Test suite for RL Production Optimizer API."""

import os
import sys
import json
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi.testclient import TestClient
from app import app


class TestAPI(unittest.TestCase):
    """Test all API endpoints."""

    def setUp(self):
        self.client = TestClient(app)

    def test_health(self):
        resp = self.client.get("/api/health")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["status"], "healthy")
        self.assertIn("models_loaded", data)

    def test_models(self):
        resp = self.client.get("/api/models")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("models", data)
        for name in ["q_learning", "policy_gradient", "dqn"]:
            self.assertIn(name, data["models"])

    def test_optimize(self):
        resp = self.client.post("/api/optimize", json={})
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("scenario", data)
        self.assertIn("recommendations", data)
        self.assertIn("well", data["scenario"])
        self.assertIn("oil_rate_bopd", data["scenario"]["well"])

    def test_optimize_with_scenario(self):
        scenario = {
            "well": {
                "oil_rate_bopd": 200, "water_cut": 0.5,
                "gas_oil_ratio": 500, "bhp_psi": 2000,
                "thp_psi": 800, "well_status": "active",
            },
            "equipment": {
                "pump_efficiency": 0.8, "vibration_mm_s": 2.0,
                "temperature_c": 60, "condition": "optimal",
            },
            "market": {
                "oil_price_usd_bbl": 75, "gas_price_usd_mcf": 3.0,
                "opex_usd_bbl": 20,
            },
            "reservoir": {
                "permeability_md": 500, "porosity": 0.2,
                "reservoir_pressure_psi": 4000, "reservoir_temperature_f": 150,
                "oil_viscosity_cp": 5,
            },
        }
        resp = self.client.post("/api/optimize", json={"scenario": scenario})
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["scenario"]["well"]["oil_rate_bopd"], 200)

    def test_simulate(self):
        resp = self.client.post("/api/simulate", json={"num_steps": 10})
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["simulation_length"], 10)
        self.assertIn("trajectories", data)
        for model_name in ["q_learning", "policy_gradient", "dqn"]:
            if model_name in data["trajectories"]:
                traj = data["trajectories"][model_name]
                self.assertEqual(len(traj["actions"]), 10)
                self.assertEqual(len(traj["rewards"]), 10)
                self.assertIn("total_reward", traj)
                self.assertIn("avg_reward", traj)

    def test_compare(self):
        resp = self.client.get("/api/compare")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("comparison", data)
        for model_name in ["q_learning", "policy_gradient", "dqn"]:
            if model_name in data["comparison"]:
                stats = data["comparison"][model_name]
                self.assertIn("final_avg_reward", stats)
                self.assertIn("best_reward", stats)
                self.assertIn("convergence_episode", stats)


if __name__ == "__main__":
    unittest.main(verbosity=2)
