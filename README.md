# 🧵 TextileVision AI — Real-Time Quality Inspection System

AI-powered textile defect detection using computer vision, YOLOv8, and FastAPI.

---

## 🚀 Quick Start (5 Minutes)

### Prerequisites
- Python 3.9+ installed
- Webcam (optional — runs in demo mode without one)

### Run (Linux/Mac)
```bash
git clone <your-repo>
cd textile_inspection
chmod +x start.sh
./start.sh
```

### Run (Windows)
```
Double-click start.bat
```

### Clone from GitHub
```bash
git clone https://github.com/SathwikGoundla/Textile-Inspection.git
cd Textile-Inspection
```

### Manual Setup
```bash
cd textile_inspection/backend
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### Open Dashboard
```
http://localhost:8000
```

---

## 📁 Project Structure

```
textile_inspection/
├── backend/
│   ├── main.py                    ← FastAPI app + WebSocket live feed
│   ├── models/
│   │   ├── detector.py            ← YOLOv8 textile detection
│   │   └── defect_detector.py     ← CNN defect analysis  
│   ├── database/
│   │   ├── db.py                  ← SQLAlchemy connection
│   │   └── models.py              ← ORM models
│   ├── utils/
│   │   ├── camera.py              ← Camera manager (webcam/RTSP/demo)
│   │   └── preprocessing.py       ← Frame preprocessing pipeline
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/
│   └── index.html                 ← Full dashboard UI
├── docker-compose.yml
├── start.sh                       ← Linux/Mac launcher
├── start.bat                      ← Windows launcher
└── README.md
```

---

## 🔧 System Architecture

```
┌──────────────┐     ┌──────────────┐     ┌──────────────────┐
│   Camera     │────▶│ Preprocessing│────▶│ Object Detection │
│ (Webcam/RTSP)│     │ (CLAHE,Norm) │     │   (YOLOv8)       │
└──────────────┘     └──────────────┘     └────────┬─────────┘
                                                   │
                                          ┌────────▼─────────┐
                                          │ Textile Gate      │
                                          │ (Classification)  │
                                          └──────┬────┬───────┘
                                                 │    │
                                        Textile  │    │ Non-Textile
                                                 ▼    ▼
                                     ┌──────────────┐  ┌─────────┐
                                     │ Defect Detect│  │ REJECT  │
                                     │ (CNN/CV)     │  │ + Alert │
                                     └──────┬───────┘  └─────────┘
                                            │
                                   ┌────────▼────────┐
                                   │  Results +       │
                                   │  DB Logging     │
                                   └────────┬────────┘
                                            │
                                   ┌────────▼────────┐
                                   │  WebSocket →    │
                                   │  Dashboard UI   │
                                   └─────────────────┘
```

---

## 🤖 Defects Detected

| Defect | Method | Severity |
|--------|--------|----------|
| Hole | Morphological analysis | HIGH |
| Tear | Edge detection (Hough) | HIGH |
| Stain | Color anomaly detection | MEDIUM |
| Misprint | Pattern analysis | MEDIUM |
| Weaving Defect | Texture analysis | MEDIUM |
| Color Inconsistency | Block variance | LOW |
| Pilling | Surface analysis | LOW |
| Crease | Wrinkle detection | LOW |

---

## 🌐 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Dashboard UI |
| `/ws/live` | WebSocket | Real-time camera feed |
| `/api/stats` | GET | System statistics |
| `/api/logs` | GET | Inspection history |
| `/api/analyze-frame` | POST | Analyze single image |
| `/api/logs/clear` | DELETE | Clear logs |
| `/docs` | GET | Swagger API docs |

---

## 🔌 Camera Configuration

### Webcam (default)
Auto-detected at `/dev/video0` or index 0.

### RTSP Industrial Camera
```bash
export CAMERA_RTSP_URL="rtsp://192.168.1.100:554/stream"
uvicorn main:app --host 0.0.0.0 --port 8000
```

### Docker with Webcam
```bash
docker-compose up --build
```

---

## ☁️ Cloud Deployment

### Azure App Service (Recommended)
1. Create resource group: `az group create --name textile-rg --location eastus`
2. Deploy: `az webapp up --resource-group textile-rg --name textile-api --runtime python:3.11`
3. Access: `https://textile-api.azurewebsites.net`

### Azure Container Apps
```bash
# Build & push image
docker build -t textile-api:latest ./backend
az acr build --registry [registry-name] --image textile-api:latest ./backend

# Deploy
az containerapp create --name textile-api --image [registry].azurecr.io/textile-api:latest
```

### GitHub Actions CI/CD
Auto-deploy on push to `main` branch. See `.github/workflows/deploy.yml`

**See [DEPLOYMENT_BACKEND.md](DEPLOYMENT_BACKEND.md) for detailed deployment steps.**

---

## 🤖 Enabling Real YOLOv8

```bash
pip install ultralytics
```

The `detector.py` auto-loads `yolov8n.pt` if `ultralytics` is installed.
YOLOv8 nano is downloaded automatically on first run (~6MB).

---

## 🗄️ Switch to PostgreSQL

```bash
# .env file
DATABASE_URL=postgresql://user:password@localhost:5432/textile_db
```

---

## 📊 Performance Targets

| Metric | Target | Notes |
|--------|--------|-------|
| Inference latency | < 100ms | YOLOv8n: ~30ms on GPU |
| FPS | 30+ | With optimization |
| Detection accuracy | > 90% mAP | With trained model |
| Uptime | 99.9% | With Docker restart |

---

## 🧪 Training Your Own Defect Model

1. Collect textile images with defect annotations
2. Use tools like LabelImg or Roboflow for annotation
3. Train YOLOv8 on your dataset:
   ```bash
   yolo train data=textile.yaml model=yolov8n.pt epochs=100
   ```
4. Point `model_path` in `detector.py` to your trained weights

---

## 📝 License
MIT License - Free for commercial and academic use.
