
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import joblib, os, random

app = FastAPI(title="RL Production Optimizer")

class EnvStep(BaseModel):
    state: list
    action: int = -1

class AgentResponse(BaseModel):
    action: int
    value: float

AGENTS = {}
for f in os.listdir("outputs/models"):
    if f.endswith(".pkl"):
        AGENTS[f.replace(".pkl", "")] = joblib.load(os.path.join("outputs/models", f))

@app.get("/")
def root():
    return dict(service="RL Production Optimizer", agents=list(AGENTS.keys()))

@app.post("/act/{agent_name}")
def act(agent_name: str, step: EnvStep):
    agent = AGENTS.get(agent_name)
    if not agent:
        raise HTTPException(404)
    X = np.array(step.state).reshape(1, -1)
    scaler = agent.get("scaler")
    if scaler:
        X = scaler.transform(X)
    action = int(agent["model"].predict(X)[0])
    return AgentResponse(action=action, value=float(action))
