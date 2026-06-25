#!/usr/bin/env bash
# Install Qualcomm QAIRT / QNN SDK on QCS6490 AI Box for SafeAging YOLO acceleration.
#
# This script cannot download QAIRT (Qualcomm account required). Steps:
#   1. Download QAIRT SDK for Linux aarch64 from https://developer.qualcomm.com/software/qualcomm-ai-stack
#   2. Copy the installer or tarball to the box, e.g. /root/qairt-installer.run
#   3. Run: sudo bash tools/qnn/install_qairt_aibox.sh /root/qairt-installer.run
#
# After QAIRT is installed, build or install onnxruntime with QNN EP for aarch64 and
# point docker-compose.aibox-qnn.yml at /opt/qairt and /opt/onnxruntime-qnn.

set -euo pipefail

INSTALLER="${1:-}"
QAIRT_ROOT="${QAIRT_ROOT:-/opt/qairt}"

if [[ "$(uname -m)" != "aarch64" ]]; then
  echo "This script targets QCS6490 (aarch64). Current: $(uname -m)" >&2
  exit 1
fi

echo "== SafeAging QNN / QAIRT setup =="
echo "Target: ${QAIRT_ROOT}"

if [[ -n "${INSTALLER}" && -f "${INSTALLER}" ]]; then
  echo "Running installer: ${INSTALLER}"
  chmod +x "${INSTALLER}"
  "${INSTALLER}" --install-dir "${QAIRT_ROOT}" || true
fi

if [[ ! -d "${QAIRT_ROOT}/lib" ]]; then
  cat <<'EOF'
QAIRT not found at /opt/qairt.

Manual steps:
  1. Download QAIRT SDK (Linux, aarch64) from Qualcomm AI Stack
  2. Install to /opt/qairt
  3. Verify: ls /opt/qairt/lib/aarch64-android/libQnnHtp.so
  4. Install onnxruntime with QNNExecutionProvider (custom wheel or build from source)
  5. Deploy with:
       docker compose -f docker-compose.yml \\
         -f docker-compose.aibox-hostdb.yml \\
         -f docker-compose.aibox-qnn.yml up -d --build
EOF
  exit 1
fi

HTP_LIB="${QAIRT_ROOT}/lib/aarch64-android/libQnnHtp.so"
GPU_LIB="${QAIRT_ROOT}/lib/aarch64-android/libQnnGpu.so"

echo "Checking QNN libraries..."
[[ -f "${HTP_LIB}" ]] && echo "  OK HTP: ${HTP_LIB}" || echo "  MISSING HTP: ${HTP_LIB}"
[[ -f "${GPU_LIB}" ]] && echo "  OK GPU: ${GPU_LIB}" || echo "  MISSING GPU: ${GPU_LIB} (optional)"

for dev in /dev/adsprpc-smd /dev/ion /dev/kgsl-3d0; do
  if [[ -e "${dev}" ]]; then
    echo "  OK device ${dev}"
  else
    echo "  WARN missing ${dev} — mount in docker-compose.aibox-qnn.yml"
  fi
done

echo ""
echo "Benchmark CPU vs QNN (after analytics container is up):"
echo "  python tools/qnn/benchmark_yolo_backend.py --image tools/_test_frame.jpg"
