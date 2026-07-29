import streamlit as st, joblib, numpy as np, matplotlib.pyplot as plt
from pathlib import Path; import sys; sys.path.insert(0, str(Path(__file__).parent))

st.set_page_config(page_title="RL Optimizer", layout="wide", initial_sidebar_state="collapsed")
st.title("RL Optimizer")

p = Path(__file__).parent / 'outputs' / 'models'
models = {'action': joblib.load(p / 'action_policy_model.pkl'), 'reward': joblib.load(p / 'reward_predictor.pkl')}

with st.expander('Input Parameters', expanded=True):
    cols = st.columns(4)
    rate = cols[0].slider('Rate', 0, 5000, 2500)
    wcut = cols[1].slider('Wcut', 0, 100, 50)
    gor = cols[2].slider('Gor', 0, 10000, 5000)
    bhp = cols[3].slider('Bhp', 500, 5000, 2750)
    thp = cols[0].slider('Thp', 100, 2000, 1050)
    status = cols[1].selectbox('Status', ['active','idle','shutin'])
    pump = cols[2].slider('Pump', 0, 100, 50)
    vib = cols[3].slider('Vib', 0, 50, 25)
    motor = cols[0].slider('Motor', 20, 200, 110)
    price = cols[1].slider('Price', 20, 150, 85)

run = st.button('Run Simulation', type='primary', use_container_width=True)

if run:
    x = np.array([[rate, wcut, gor, bhp, thp, status, pump, vib, motor, price]])
    out = {}
    m = models['action']
    if isinstance(m, dict):
        p = m['model'].predict(m['scaler'].transform(x))
        v = m['label_encoder'].inverse_transform(p)[0] if 'label_encoder' in m else float(p[0])
    else:
        v = float(m.predict(x)[0])
    out['action'] = v
    m = models['reward']
    if isinstance(m, dict):
        p = m['model'].predict(m['scaler'].transform(x))
        v = m['label_encoder'].inverse_transform(p)[0] if 'label_encoder' in m else float(p[0])
    else:
        v = float(m.predict(x)[0])
    out['reward'] = v
    st.divider()
    mc = st.metric('Net Reward', f'{out["reward"]:,.2f}' if "reward" in out else 'N/A')
    if 'action' in out:
        st.info(f'Recommended action: {out["action"]}')