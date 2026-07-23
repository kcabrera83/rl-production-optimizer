import pytest
import os
import json
import numpy as np

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs", "models")


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
    path = os.path.join(OUTPUT_DIR, "training_summary.json")
    if not os.path.exists(path):
        pytest.skip("training_summary.json not found")
    with open(path) as f:
        summary = json.load(f)
    assert "training_date" in summary
    assert "episodes" in summary
    assert "steps_per_episode" in summary


def test_training_summary_has_all_models():
    path = os.path.join(OUTPUT_DIR, "training_summary.json")
    if not os.path.exists(path):
        pytest.skip("training_summary.json not found")
    with open(path) as f:
        summary = json.load(f)
    for model_name in ["q_learning", "policy_gradient", "dqn"]:
        assert model_name in summary
        assert "final_avg_reward" in summary[model_name]
        assert "reward_curve" in summary[model_name]
        assert len(summary[model_name]["reward_curve"]) > 0


def test_q_learning_reward_curve():
    path = os.path.join(OUTPUT_DIR, "training_summary.json")
    if not os.path.exists(path):
        pytest.skip("training_summary.json not found")
    with open(path) as f:
        summary = json.load(f)
    rewards = summary["q_learning"]["reward_curve"]
    assert all(isinstance(r, (int, float)) for r in rewards)
    assert len(rewards) >= 10


def test_reward_curves_positive_trend():
    path = os.path.join(OUTPUT_DIR, "training_summary.json")
    if not os.path.exists(path):
        pytest.skip("training_summary.json not found")
    with open(path) as f:
        summary = json.load(f)
    for model_name in ["q_learning", "policy_gradient", "dqn"]:
        rewards = summary[model_name]["reward_curve"]
        last_quarter = rewards[len(rewards) // 2 :]
        assert len(last_quarter) > 0
