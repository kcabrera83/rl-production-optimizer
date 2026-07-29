import streamlit as st
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

st.set_page_config(page_title="RL Production Optimizer", layout="wide")
st.title("RL Production Optimizer")
st.markdown("Optimize production using reinforcement learning.")

import joblib, numpy as np
d = Path(__file__).parent / 'outputs' / 'models'
models = {'action': joblib.load(d / 'action_policy_model.pkl'), 'reward': joblib.load(d / 'reward_predictor.pkl')}

st.sidebar.header("Input Parameters")
oil_rate = st.sidebar.slider('Oil Rate', 0, 5000, 2500)
water_cut = st.sidebar.slider('Water Cut', 0, 100, 50)
gor = st.sidebar.slider('Gor', 0, 10000, 5000)
bhp = st.sidebar.slider('Bhp', 500, 5000, 2750)
thp = st.sidebar.slider('Thp', 100, 2000, 1050)
status = st.sidebar.selectbox('Status', ['active','idle','shutin'])
pump_eff = st.sidebar.slider('Pump Eff', 0, 100, 50)
vibration = st.sidebar.slider('Vibration', 0, 50, 25)
motor_temp = st.sidebar.slider('Motor Temp', 20, 200, 110)
equip_cond = st.sidebar.slider('Equip Cond', 0, 100, 50)
oil_price = st.sidebar.slider('Oil Price', 20, 150, 85)
gas_price = st.sidebar.slider('Gas Price', 0, 20, 10)
opex = st.sidebar.slider('Opex', 0, 100, 50)
perm = st.sidebar.slider('Perm', 0, 1000, 500)
porosity = st.sidebar.slider('Porosity', 5, 35, 20)
res_pressure = st.sidebar.slider('Res Pressure', 1000, 10000, 5500)
temperature = st.sidebar.slider('Temperature', 20, 200, 110)
viscosity = st.sidebar.slider('Viscosity', 0, 100, 50)

if st.sidebar.button("Run"):
    try:
        x = np.array([[oil_rate, water_cut, gor, bhp, thp, status, pump_eff, vibration, motor_temp, equip_cond, oil_price, gas_price, opex, perm, porosity, res_pressure, temperature, viscosity]])
        cols = st.columns(2)
        for i, (k, m) in enumerate(models.items()):
            X = m['scaler'].transform(x)
            p = m['model'].predict(X)
            if 'label_encoder' in m:
                val = m['label_encoder'].inverse_transform(p)[0]
            else:
                val = f'{p[0]:.2f}'
            cols[i].metric(k.title(), val)
    except Exception as e:
        st.error(str(e))