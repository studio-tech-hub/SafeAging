#!/usr/bin/env bash
# Native QNN HTP smoke test on QCS6490 (run ON the AI Box host, not inside analytics entrypoint).
#
#   cd /root/SafeAging
#   bash tools/qnn/native_htp_smoke.sh /root/SafeAging/models/yolo26n_backbone.onnx
#
# Requires QAIRT at /opt/qairt (envsetup.sh) and backbone ONNX (nms=False export).

set -euo pipefail

ONNX="${1:-/root/SafeAging/models/yolo26n_backbone.onnx}"
WORKDIR="${2:-/tmp/qnn_backbone_native}"
SDK="${QAIRT_ROOT:-/opt/qairt/2.40.0.251030}"

if [[ ! -f "$ONNX" ]]; then
  echo "ERROR: ONNX not found: $ONNX"
  echo "Export first: python3 tools/export_yolo26_onnx.py --no-nms --out models/yolo26n_backbone.onnx"
  exit 1
fi

if [[ ! -f "$SDK/bin/aarch64-oe-linux-gcc11.2/envsetup.sh" ]]; then
  echo "ERROR: QAIRT envsetup not found under $SDK"
  exit 1
fi

# shellcheck disable=SC1091
source "$SDK/bin/aarch64-oe-linux-gcc11.2/envsetup.sh"

export PRODUCT_SOC="${PRODUCT_SOC:-6490}"
export DSP_ARCH="${DSP_ARCH:-68}"
export QNN_SOC_MODEL="${QNN_SOC_MODEL:-35}"
export QNN_HTP_ARCH="${QNN_HTP_ARCH:-68}"

CONVERTER="$(command -v qnn-onnx-converter || true)"
NETRUN="$(command -v qnn-net-run || true)"
HTP_SO="${QNN_SDK_ROOT}/lib/aarch64-oe-linux-gcc11.2/libQnnHtp.so"

echo "converter=$CONVERTER"
echo "netrun=$NETRUN"
echo "htp=$HTP_SO"

rm -rf "$WORKDIR"
mkdir -p "$WORKDIR/input "$WORKDIR/output"

python3 - <<PY
import numpy as np
from pathlib import Path
x = np.random.rand(1, 3, 640, 640).astype("float32")
p = Path("$WORKDIR/input/images.raw")
p.write_bytes(x.tobytes())
Path("$WORKDIR/input/input_list.txt").write_text(f"{p}:=" + ",".join(str(d) for d in x.shape) + "\\n")
print("wrote", p, x.shape)
PY

echo "=== qnn-onnx-converter ==="
qnn-onnx-converter \
  --input_network "$ONNX" \
  --output_path "$WORKDIR/model.cpp" \
  --preserve_io layout \
  --float_bitwidth 16

echo "=== qnn-model-lib-generator ==="
qnn-model-lib-generator \
  -c "$WORKDIR/model.cpp" \
  -b "$WORKDIR/model.bin" \
  -o "$WORKDIR/model.so"

echo "=== qnn-net-run HTP ==="
/usr/bin/time -f "elapsed %e s" qnn-net-run \
  --model "$WORKDIR/model.so" \
  --backend "$HTP_SO" \
  --input_list "$WORKDIR/input/input_list.txt" \
  --output_dir "$WORKDIR/output" \
  --perf_profile burst \
  --log_level error

echo "OK native HTP smoke — check $WORKDIR/output"
