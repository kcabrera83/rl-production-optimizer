import pytest


def test_health(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "healthy"
    assert "models_loaded" in data
    assert data["version"] == "1.0.0"


def test_models(client):
    response = client.get("/api/models")
    assert response.status_code == 200
    data = response.get_json()
    assert "models" in data
    for name in ["q_learning", "policy_gradient", "dqn"]:
        assert name in data["models"]
        assert "loaded" in data["models"][name]
        assert "file_exists" in data["models"][name]


def test_api_docs(client):
    response = client.get("/api/docs")
    assert response.status_code == 200
    data = response.get_json()
    assert data["openapi"] == "3.0.0"
    assert "/api/optimize" in data["paths"]
    assert "/api/simulate" in data["paths"]
    assert "/api/compare" in data["paths"]


def test_optimize_default(client):
    response = client.post("/api/optimize", json={})
    assert response.status_code == 200
    data = response.get_json()
    assert "scenario" in data
    assert "recommendations" in data
    assert isinstance(data["recommendations"], dict)


def test_optimize_custom_scenario(client):
    scenario = {
        "well": {
            "oil_rate_bopd": 200.0,
            "water_cut": 0.3,
            "gas_oil_ratio": 500.0,
            "bhp_psi": 2500.0,
            "thp_psi": 800.0,
            "well_status": "active",
        },
        "equipment": {
            "pump_efficiency": 0.75,
            "vibration_mm_s": 2.5,
            "temperature_c": 65.0,
            "condition": "optimal",
        },
        "market": {
            "oil_price_usd_bbl": 75.0,
            "gas_price_usd_mcf": 3.5,
            "opex_usd_bbl": 20.0,
        },
        "reservoir": {
            "permeability_md": 500.0,
            "porosity": 0.2,
            "reservoir_pressure_psi": 3000.0,
            "reservoir_temperature_f": 150.0,
            "oil_viscosity_cp": 5.0,
        },
    }
    response = client.post("/api/optimize", json={"scenario": scenario})
    assert response.status_code == 200
    data = response.get_json()
    assert data["scenario"] == scenario
    assert "recommendations" in data


def test_simulate(client):
    response = client.post("/api/simulate", json={"num_steps": 20})
    assert response.status_code == 200
    data = response.get_json()
    assert "simulation_length" in data
    assert data["simulation_length"] == 20
    assert "trajectories" in data


def test_simulate_default(client):
    response = client.post("/api/simulate", json={})
    assert response.status_code == 200
    data = response.get_json()
    assert "trajectories" in data


def test_compare(client):
    response = client.get("/api/compare")
    assert response.status_code == 200
    data = response.get_json()
    assert "comparison" in data


def test_trajectory_structure(client):
    response = client.post("/api/simulate", json={"num_steps": 10})
    data = response.get_json()
    for model_name, traj in data["trajectories"].items():
        assert "actions" in traj
        assert "rewards" in traj
        assert "cumulative_reward" in traj
        assert "total_reward" in traj
        assert "avg_reward" in traj
        assert len(traj["actions"]) == 10
        assert len(traj["rewards"]) == 10
