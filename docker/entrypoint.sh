#!/bin/sh
# SafeAging Analytics Service — Docker entrypoint
# 1. Auto-download the YOLO model if not present in /app/models
# 2. Start the FastAPI service
set -e

MODEL_FILE="${MODEL_PATH:-/app/models/yolo26n.pt}"
MODEL_DIR=$(dirname "$MODEL_FILE")
MODEL_NAME=$(basename "$MODEL_FILE")

if [ ! -f "$MODEL_FILE" ]; then
    echo "[entrypoint] Model not found at $MODEL_FILE"
    case "$MODEL_NAME" in
        *.onnx)
            echo "[entrypoint] ERROR: ONNX must be copied to $MODEL_FILE before start."
            echo "[entrypoint] On dev PC: python tools/export_yolo26_onnx.py"
            echo "[entrypoint] Then: scp models/yolo26n.onnx root@<box>:/root/SafeAging/models/"
            exit 1
            ;;
        *)
            echo "[entrypoint] Downloading $MODEL_NAME into $MODEL_DIR ..."
            mkdir -p "$MODEL_DIR"
            cd "$MODEL_DIR"
            python -c "from ultralytics import YOLO; YOLO('$MODEL_NAME')"
            echo "[entrypoint] Download complete."
            cd /app/python
            ;;
    esac
fi

echo "[entrypoint] Running database migrations (alembic upgrade head) ..."
cd /app/python
if ! alembic upgrade head; then
    echo "[entrypoint] WARNING: alembic upgrade failed — service may return 500 on admin API until DB is fixed."
fi

echo "[entrypoint] Starting SafeAging analytics service ..."
exec python /app/python/service.py
