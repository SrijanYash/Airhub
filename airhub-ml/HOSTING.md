# AirHub Hosting Guide

This guide explains how to host the AirHub federated learning backend for continuous predictions and training.

## Quick Start

### Option 1: Full Hosting Server (Recommended)

Runs both Flower server and FastAPI server together:

```bash
cd airhub-ml
python start_hosting.py --persistent --clients 1 --rounds 10
```

**Access:**
- API Server: http://localhost:8000
- API Docs: http://localhost:8000/docs
- Flower Server: localhost:8080 (internal)

### Option 2: Separate Servers

**Terminal 1 - Flower Server:**
```bash
cd airhub-ml
python -m federated.server_node --persistent
```

**Terminal 2 - API Server:**
```bash
cd airhub-ml
python main.py
# or
uvicorn main:app --reload
```

---

## Running Clients for Training

Once the server is running, start clients in **new terminal windows**:

```bash
cd airhub-ml
python start_client.py --city London --country GB --persistent
```

### Multiple Clients (for federated learning):

Open multiple terminals and run different cities:
```bash
# Terminal 1
python start_client.py --city London --country GB --persistent

# Terminal 2
python start_client.py --city Paris --country FR --persistent

# Terminal 3
python start_client.py --city Tokyo --country JP --persistent
```

---

## Making Predictctions

Once the server is running and has a trained model:

### Via API (POST request):
```bash
curl -X POST http://localhost:8000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"city": "Delhi", "country": "IND"}'
```

### Via API Docs:
1. Open http://localhost:8000/docs
2. Click on `/api/predict`
3. Click "Try it out"
4. Enter city and country
5. Click "Execute"

### Get Prediction History:
```bash
curl http://localhost:8000/api/predictions/history
```

---

## Starting Training via API

Once the server is running, trigger training:

```bash
curl -X POST http://localhost:8000/api/train \
  -H "Content-Type: application/json" \
  -d '{"num_clients": 1, "num_rounds": 10}'
```

Check training status:
```bash
curl http://localhost:8000/api/train/status
```

Get FL metrics:
```bash
curl http://localhost:8000/api/flwr/metrics
```

---

## Command Line Options

### start_hosting.py

| Option | Description | Default |
|--------|-------------|---------|
| `--persistent` | Keep server running after rounds | `True` |
| `--non-persistent` | Stop after initial rounds | `False` |
| `--clients N` | Initial number of clients | `1` |
| `--rounds N` | Initial number of rounds | `10` |

### start_client.py

| Option | Description | Default |
|--------|-------------|---------|
| `--city` | City name | `London` |
| `--country` | Country code | `GB` |
| `--server` | Flower server address | `localhost:8080` |
| `--persistent` | Auto-reconnect if disconnected | `False` |

---

## Production Deployment

### Using Docker (recommended for production):

Create a `docker-compose.yml`:
```yaml
version: '3.8'
services:
  airhub:
    build: .
    ports:
      - "8000:8000"
      - "8080:8080"
    environment:
      - OPENWEATHER_API_KEY=your_key_here
    volumes:
      - ./models:/app/airhub-ml/model/saved_models
      - ./data:/app/airhub-ml/data/datasets
    command: python start_hosting.py --persistent
```

### Using systemd (Linux server):

Create `/etc/systemd/system/airhub.service`:
```ini
[Unit]
Description=AirHub Federated Learning Server
After=network.target

[Service]
Type=simple
User=airhub
WorkingDirectory=/path/to/airhub-ml
Environment="PATH=/path/to/venv/bin"
ExecStart=/path/to/venv/bin/python start_hosting.py --persistent
Restart=always

[Install]
WantedBy=multi-user.target
```

Then:
```bash
sudo systemctl enable airhub
sudo systemctl start airhub
sudo systemctl status airhub
```

---

## Troubleshooting

### Server won't start - Port already in use
```bash
# Windows
netstat -ano | findstr :8000
taskkill /PID <PID> /F

# Linux/Mac
lsof -i :8000
kill -9 <PID>
```

### Clients can't connect to server
- Ensure server is running with `--persistent` flag
- Check firewall settings allow port 8080
- Verify server address matches: `localhost:8080` or IP address

### Model not found errors
- Run initial training first: `POST /api/train`
- Ensure data preprocessing completed: `python -m data.preprocess`

### Insufficient data errors
- Need at least 8 days of data (7-day lookback + 1 prediction)
- Use `POST /api/data/ingest` to add custom data
- API will fetch sample data if OpenWeather API fails

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    HOSTING SERVER                            │
│  ┌──────────────────┐         ┌─────────────────────┐       │
│  │  Flower Server   │         │   FastAPI Server    │       │
│  │  (Port 8080)     │         │   (Port 8000)       │       │
│  │                  │         │                     │       │
│  │  - Aggregates    │         │  - /api/predict     │       │
│  │    model weights │         │  - /api/train       │       │
│  │  - FedAvg        │         │  - /api/flwr/metrics│       │
│  │    strategy      │         │  - /health          │       │
│  └──────────────────┘         └─────────────────────┘       │
└─────────────────────────────────────────────────────────────┘
                            │
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
        ▼                   ▼                   ▼
┌───────────────┐  ┌───────────────┐  ┌───────────────┐
│   Client 1    │  │   Client 2    │  │   Client 3    │
│   London      │  │   Paris       │  │   Tokyo       │
│   (trains)    │  │   (trains)    │  │   (trains)    │
└───────────────┘  └───────────────┘  └───────────────┘
```

---

## Key Files

| File | Purpose |
|------|---------|
| `start_hosting.py` | Main hosting script (Flower + FastAPI) |
| `start_client.py` | Client starter script |
| `main.py` | FastAPI server only |
| `federated/server_node.py` | Flower server logic |
| `federated/client_node.py` | Flower client logic |
| `model/saved_models/` | Saved model weights |
| `data/datasets/` | Processed training data |
