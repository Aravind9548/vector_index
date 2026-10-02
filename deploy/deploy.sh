#!/usr/bin/env bash
# Deploy VecIdx to Azure Container Apps in koreacentral.
# Prerequisites: az CLI logged in, ACR created, Container Apps environment ready.
set -euo pipefail

# --- Configuration ---
RESOURCE_GROUP="${RESOURCE_GROUP:-vecidx-rg}"
LOCATION="${LOCATION:-koreacentral}"
ACR_NAME="${ACR_NAME:-vecidxacr}"
APP_ENV="${APP_ENV:-vecidx-env}"
APP_NAME="${APP_NAME:-vecidx}"
IMAGE_TAG="${IMAGE_TAG:-latest}"

echo "=== VecIdx Azure Deployment ==="
echo "  Region:    $LOCATION"
echo "  RG:        $RESOURCE_GROUP"
echo "  ACR:       $ACR_NAME"
echo "  App:       $APP_NAME"

# 1. Build and push container image.
echo "[1/4] Building container image..."
az acr build \
  --registry "$ACR_NAME" \
  --image "vecidx:$IMAGE_TAG" \
  --file Dockerfile \
  .

# 2. Create / update the Container App.
echo "[2/4] Deploying to Container Apps..."
az containerapp create \
  --name "$APP_NAME" \
  --resource-group "$RESOURCE_GROUP" \
  --environment "$APP_ENV" \
  --image "$ACR_NAME.azurecr.io/vecidx:$IMAGE_TAG" \
  --target-port 50051 \
  --ingress external \
  --transport http2 \
  --cpu 2 --memory 4Gi \
  --min-replicas 1 --max-replicas 3 \
  --registry-server "$ACR_NAME.azurecr.io" \
  --env-vars SNAPSHOT_PATH=/data/snapshots/latest.vecidx

# 3. Set up budget alert (day one, before burning credits).
echo "[3/4] Creating budget alert..."
az consumption budget create \
  --budget-name vecidx-budget \
  --resource-group "$RESOURCE_GROUP" \
  --amount 50 \
  --time-grain Monthly \
  --category Cost \
  --notifications '[{"enabled":true,"operator":"GreaterThan","threshold":80,"contactEmails":["aravindblue9797@gmail.com"]}]' \
  2>/dev/null || echo "  (budget alert may already exist or require Consumption API access)"

# 4. Verify deployment.
echo "[4/4] Checking deployment..."
FQDN=$(az containerapp show \
  --name "$APP_NAME" \
  --resource-group "$RESOURCE_GROUP" \
  --query "properties.configuration.ingress.fqdn" \
  -o tsv)

echo ""
echo "=== Deployment complete ==="
echo "  Endpoint: https://$FQDN"
echo "  Health:   curl https://$FQDN:8080/health"
echo "  gRPC:     $FQDN:50051"
echo ""
echo "  Monitor:  az containerapp logs show --name $APP_NAME -g $RESOURCE_GROUP"
