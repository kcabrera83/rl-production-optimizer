import streamlit as st
import joblib
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

st.set_page_config(page_title="RL Production Optimizer", layout="wide")
st.title("RL Production Optimizer")
st.markdown("Optimize production using a reinforcement learning agent.")

@st.cache_resource
def load_models():
    d = Path(__file__).parent / "outputs" / "models"
    return {k: joblib.load(d / v) for k, v in [("action", "action_policy_model.pkl"), ("reward", "reward_predictor.pkl")]}

models = load_models()

st.sidebar.header("Input Parameters")
oil_rate_bbl_d = st.sidebar.slider("Oil Rate Bbl D", 0, 5000, 2500)
water_cut_pct = st.sidebar.slider("Water Cut Pct", 0, 100, 50)
gor_scf_bbl = st.sidebar.slider("Gor Scf Bbl", 0, 10000, 5000)
bhp_psi = st.sidebar.slider("Bhp Psi", 500, 5000, 2750)
thp_psi = st.sidebar.slider("Thp Psi", 100, 2000, 1050)
well_status = st.sidebar.selectbox("Well Status", ['active', 'idle', 'shutin'])
pump_efficiency_pct = st.sidebar.slider("Pump Efficiency Pct", 0, 100, 50)
vibration_mm = st.sidebar.slider("Vibration Mm", 0, 50, 25)
motor_temp_c = st.sidebar.slider("Motor Temp C", 20, 200, 110)
equip_condition = st.sidebar.slider("Equip Condition", 0, 100, 50)
oil_price_usd = st.sidebar.slider("Oil Price Usd", 20, 150, 85)
gas_price_usd = st.sidebar.slider("Gas Price Usd", 0, 20, 10)
opex_usd_bbl = st.sidebar.slider("Opex Usd Bbl", 0, 100, 50)
permeability_md = st.sidebar.slider("Permeability Md", 0, 1000, 500)
porosity_pct = st.sidebar.slider("Porosity Pct", 5, 35, 20)
reservoir_pressure_psi = st.sidebar.slider("Reservoir Pressure Psi", 1000, 10000, 5500)
temperature_c = st.sidebar.slider("Temperature C", 20, 200, 110)
viscosity_cp = st.sidebar.slider("Viscosity Cp", 0, 100, 50)

if st.sidebar.button("Run Prediction"):
    try:
        features = np.array([[oil_rate_bbl_d, water_cut_pct, gor_scf_bbl, bhp_psi, thp_psi, well_status, pump_efficiency_pct, vibration_mm, motor_temp_c, equip_condition, oil_price_usd, gas_price_usd, opex_usd_bbl, permeability_md, porosity_pct, reservoir_pressure_psi, temperature_c, viscosity_cp]])
        m = models["action"]
        if isinstance(m, dict):
            X = m.get("scaler").transform(features) if m.get("scaler") else features
            pred = m["model"].predict(X)
            if "label_encoder" in m:
                result = m["label_encoder"].inverse_transform(pred)[0]
            else:
                result = pred[0]
        else:
            result = m.predict(features)[0]
        st.metric("Action", result if isinstance(result, str) else f"{result:.4f}")
        m = models["reward"]
        if isinstance(m, dict):
            X = m.get("scaler").transform(features) if m.get("scaler") else features
            pred = m["model"].predict(X)
            if "label_encoder" in m:
                result = m["label_encoder"].inverse_transform(pred)[0]
            else:
                result = pred[0]
        else:
            result = m.predict(features)[0]
        st.metric("Reward", result if isinstance(result, str) else f"{result:.4f}")
    except Exception as e:
        st.error(f"Error: {e}")

