#!/usr/bin/env bash
# Deploy SafeAging Docker stack on AI Box (run as root on the device).
set -euo pipefail

LOG=/tmp/safeaging-deploy.log
exec > >(tee -a "$LOG") 2>&1

echo "=== SafeAging deploy $(date -Iseconds) ==="
cd /root/SafeAging

echo "[1] MODEL_PATH"
sed -i 's|MODEL_PATH=/app/models/yolo26n.pt|MODEL_PATH=/app/models/yolo26n.onnx|g' .env docker-compose.yml
grep MODEL_PATH .env | head -1

echo "[2] Disable opencv plugin"
PLUG=/opt/networkoptix/mediaserver/bin/plugins/opencv_object_detection_analytics_plugin
if [[ -d "$PLUG" ]]; then
    mv "$PLUG" "${PLUG}.disabled"
    systemctl restart networkoptix-mediaserver || true
fi

echo "[3] Network + Docker DNS"
curl -fsS --max-time 20 https://pypi.org/simple/ >/dev/null && echo "pypi OK" || echo "pypi FAIL"
mkdir -p /etc/docker
cat > /etc/docker/daemon.json <<'EOF'
{
  "dns": ["192.168.1.1", "8.8.8.8", "8.8.4.4"]
}
EOF
systemctl restart docker
sleep 5
docker run --rm python:3.11-slim getent hosts deb.debian.org && echo "container DNS OK" || echo "container DNS FAIL"

echo "[4] Pull postgres/minio"
export TORCH_INDEX_URL=
docker compose pull postgres minio minio-init || true

echo "[5] Build analytics (ARM64)"
export TORCH_INDEX_URL=
docker compose build --build-arg TORCH_INDEX_URL= analytics

echo "[6] Start services"
docker compose up -d postgres minio
sleep 20
docker compose up -d minio-init analytics

echo "[7] Wait health"
for i in $(seq 1 40); do
    if curl -fsS http://127.0.0.1:18000/health >/dev/null 2>&1; then
        echo "HEALTH OK"
        curl -s http://127.0.0.1:18000/health
        break
    fi
    sleep 10
    echo "health wait $i/40"
done

docker ps -a
ss -tlnp | grep -E ':18000|:15432|:19000' || true
echo "=== DEPLOY DONE $(date -Iseconds) ==="
