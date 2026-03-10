# TextileVision AI - Backend Deployment Guide

## Phase 1: Backend Deployment (TODAY)

### Quick Deployment Options

---

## Option 1: Deploy to Azure App Service (Easiest) ⭐

### Prerequisites
- Azure account (free: https://azure.microsoft.com/free)
- Azure CLI installed

### Steps

#### 1. Login to Azure
```bash
az login
```

#### 2. Create a Resource Group
```bash
az group create --name textile-inspection-rg --location eastus
```

#### 3. Create App Service Plan
```bash
az appservice plan create --name textile-inspection-plan --resource-group textile-inspection-rg --sku B1 --is-linux
```

#### 4. Create Web App
```bash
az webapp create --resource-group textile-inspection-rg --plan textile-inspection-plan --name textile-api-[UNIQUE_ID] --runtime "PYTHON:3.11"
```

#### 5. Configure Deployment
```bash
cd c:\Users\Sathwik\Downloads\textilevision_final\ \(1\)\textile_inspection
az webapp up --resource-group textile-inspection-rg --name textile-api-[UNIQUE_ID] --runtime python:3.11 --sku B1
```

**Result:** Your backend will be live at: `https://textile-api-[UNIQUE_ID].azurewebsites.net`

---

## Option 2: Deploy with Docker to Azure Container Apps

### Prerequisites
- Docker Desktop installed
- Azure Container Registry (ACR)

### Steps

#### 1. Create Container Registry
```bash
az acr create --resource-group textile-inspection-rg --name textileregistry --sku Basic
```

#### 2. Build & Push Docker Image
```bash
cd backend
docker build -t textile-api:latest .
docker tag textile-api:latest textileregistry.azurecr.io/textile-api:latest
az acr build --registry textileregistry --image textile-api:latest .
```

#### 3. Create Container App Environment
```bash
az containerapp env create --name textile-env --resource-group textile-inspection-rg --location eastus
```

#### 4. Deploy Container App
```bash
az containerapp create \
  --name textile-api \
  --resource-group textile-inspection-rg \
  --environment textile-env \
  --image textileregistry.azurecr.io/textile-api:latest \
  --target-port 8000 \
  --ingress external \
  --registry-server textileregistry.azurecr.io \
  --cpu 0.5 --memory 1Gi
```

**Result:** Your backend will be live at the Container App URL

---

## Option 3: Deploy to Azure Functions

### Create a Function App
```bash
az functionapp create \
  --resource-group textile-inspection-rg \
  --consumption-plan-location eastus \
  --runtime python \
  --runtime-version 3.11 \
  --functions-version 4 \
  --name textile-functions-api
```

---

## Environment Variables (Add to Azure)

Create these in your App Service Configuration:

```
DATABASE_URL=sqlite:///./textile_inspection.db
# OR for PostgreSQL:
# DATABASE_URL=postgresql://user:password@host:5432/textile_db
```

---

## Verify Deployment

Once deployed, test your API:

```bash
curl https://[YOUR_APP_NAME].azurewebsites.net/
```

You should see the HTML dashboard response.

---

## Next Steps
- **Tomorrow:** Update this backend and redeploy
- **Day After:** Deploy frontend separately  
- **Final:** Update README with live URLs

---

## Troubleshooting

### Issue: Missing dependencies
```bash
az webapp config appsettings set --resource-group textile-inspection-rg --name textile-api-[ID] --settings WEBSITE_RUN_FROM_PACKAGE=1
```

### Issue: Database connection
- Ensure SQLite DB path includes `/tmp/` for Azure (writable location)
- Or switch to PostgreSQL/Azure Database

### Check logs:
```bash
az webapp log tail --resource-group textile-inspection-rg --name textile-api-[ID]
```

---

## Cost Estimate
- **App Service (B1):** ~$12/month
- **Container Apps:** ~$10/month  
- **Functions:** Free tier available

Choose what fits your needs! 🚀
