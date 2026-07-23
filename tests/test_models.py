import pytest
import os

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs", "models")


def test_outputs_directory_exists():
    assert os.path.exists(OUTPUT_DIR)


def test_model_files_exist():
    model_files = [f for f in os.listdir(OUTPUT_DIR) if f.endswith((".pkl", ".joblib", ".h5", ".pt"))]
    assert len(model_files) > 0


def test_q_learning_loads():
    path = os.path.join(OUTPUT_DIR, "q_learning.pkl")
    if not os.path.exists(path):
        pytest.skip("q_learning.pkl not found - run train.py first")
    assert os.path.getsize(path) > 0


def test_policy_gradient_loads():
    path = os.path.join(OUTPUT_DIR, "policy_gradient.pkl")
    if not os.path.exists(path):
        pytest.skip("policy_gradient.pkl not found - run train.py first")
    assert os.path.getsize(path) > 0


def test_dqn_agent_loads():
    path = os.path.join(OUTPUT_DIR, "dqn_agent.pkl")
    if not os.path.exists(path):
        pytest.skip("dqn_agent.pkl not found - run train.py first")
    assert os.path.getsize(path) > 0


def test_training_summary_exists():
    import json
    path = os.path.join(OUTPUT_DIR, "training_summary.json")
    if not os.path.exists(path):
        pytest.skip("training_summary.json not found")
    with open(path) as f:
        summary = json.load(f)
    assert "training_date" in summary
    assert "episodes" in summary
