# Deployment Guide - RL Production Optimizer

## Docker Deployment

### Build the Image

```bash
cd rl-production-optimizer
docker build -t rl-production-optimizer .
```

### Run the Container

```bash
docker run -p 5019:5019 rl-production-optimizer
```

### With Model Training

```bash
docker run -p 5019:5019 rl-production-optimizer bash -c "python train.py && python app.py"
```

## Docker Compose

```yaml
version: '3.8'
services:
  rl-optimizer:
    build: .
    ports:
      - "5019:5019"
    volumes:
      - ./outputs:/app/outputs
    environment:
      - FLASK_ENV=production
      - PYTHONUNBUFFERED=1
    restart: unless-stopped
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| FLASK_ENV | Flask environment mode | development |
| PYTHONUNBUFFERED | Disable Python output buffering | 1 |
| PORT | Server port (hardcoded in app.py) | 5019 |

## Manual Deployment

### Install Dependencies

```bash
pip install -r requirements.txt
```

Key dependencies: Flask, NumPy

### Train Models

```bash
python train.py
```

Trains 3 RL agents over 50 episodes (may take a few minutes).

### Run with Gunicorn (Production)

```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5019 app:app
```

### Run with Flask Development Server

```bash
python app.py
```

## Production Considerations

- Use **gunicorn** with multiple workers for production deployments
- `debug=False` is already set in `app.py`
- Configure proper logging for request/error tracking
- Place behind a reverse proxy (nginx/Apache) for SSL termination
- Models are loaded at startup from `outputs/models/`
- Pre-train models with `train.py` before starting the server
- RL training takes ~2-5 minutes depending on hardware
- All models are NumPy-based (no GPU required)

## Health Check

```bash
curl http://localhost:5019/api/health
```

Expected response:
```json
{"status": "healthy", "models_loaded": ["q_learning", "policy_gradient", "dqn"], "version": "1.0.0"}
```

## Ports

| Service | Port |
|---------|------|
| RL Production Optimizer | 5019 |
