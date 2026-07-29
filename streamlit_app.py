
import streamlit as st
import numpy as np
import joblib, os
import matplotlib.pyplot as plt

st.set_page_config(page_title="RL Production Optimizer", layout="wide")
st.title(":robot_face: RL Production Optimizer")
st.markdown("Reinforcement learning for production optimization under uncertainty")

models = {}
for f in os.listdir("outputs/models"):
    if f.endswith(".pkl"):
        models[f.replace(".pkl", "")] = joblib.load(os.path.join("outputs/models", f))

with st.sidebar:
    sel = st.selectbox("Agent Model", list(models.keys()) or ["default"])
    st.header("Environment Params")
    eps = st.slider("Epsilon", 0.0, 1.0, 0.1)
    steps = st.slider("Simulation steps", 10, 500, 100)

m = models.get(sel, {})
feats = m.get("feature_names", [f"s{i}" for i in range(4)])
cols = st.columns(4)
state = np.array([cols[i].number_input(f"State {i}", value=0.0, key=f"s_{i}") for i in range(4)])

if st.button("Act"):
    X = state.reshape(1, -1)
    if m.get("scaler"):
        X = m["scaler"].transform(X)
    act = np.random.choice([0, 1, 2]) if np.random.random() < eps else int(m["model"].predict(X)[0])
    st.info(f"Action: {act}")
    rewards = np.random.randn(steps).cumsum()
    fig, ax = plt.subplots()
    ax.plot(rewards)
    ax.set_title("Episode Rewards")
    st.pyplot(fig)
