#!/usr/bin/env bash
# Build yolo26_people_analytics_plugin.so on AI Box (native aarch64).
#
# Prerequisites (auto-installed with --install-deps):
#   apt: ninja-build, build-essential, cmake
#   pip: conan<2
#   Metadata SDK 6.0.6.41837 unpacked on the box
#
# Usage:
#   ./tools/build_plugin_aibox.sh --install-deps \
#       --sdk-dir /root/metavms-metadata_sdk-6.0.6.41837-universal/metadata_sdk
#
#   ./tools/build_plugin_aibox.sh --sdk-dir /path/to/metadata_sdk --install
#
# --install  copies .so + manifest.json into Nx plugins dir and restarts mediaserver

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

SDK_DIR=""
DO_INSTALL=false
INSTALL_DEPS=false
NX_PLUGIN_ROOT="/opt/networkoptix/mediaserver/bin/plugins"

usage() {
    echo "Usage: $0 --sdk-dir <metadata_sdk_path> [--install-deps] [--install]"
    exit 1
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --sdk-dir)       SDK_DIR="$2"; shift 2 ;;
        --install)       DO_INSTALL=true; shift ;;
        --install-deps)  INSTALL_DEPS=true; shift ;;
        --plugin-root)   NX_PLUGIN_ROOT="$2"; shift 2 ;;
        -h|--help)       usage ;;
        *) echo "Unknown: $1"; usage ;;
    esac
done

[[ -z "$SDK_DIR" ]] && usage
[[ ! -d "$SDK_DIR" ]] && { echo "SDK dir not found: $SDK_DIR"; exit 1; }

if [[ "$(uname -m)" != "aarch64" ]]; then
    echo "WARNING: expected aarch64 AI Box, got $(uname -m)"
fi

if $INSTALL_DEPS; then
    echo "[deps] Installing build packages..."
    apt-get update -qq
    apt-get install -y ninja-build build-essential cmake git curl
    pip3 install --upgrade "conan<2" || pip3 install --user "conan<2"
    if ! command -v conan &>/dev/null && [[ -f "$HOME/.local/bin/conan" ]]; then
        export PATH="$HOME/.local/bin:$PATH"
    fi
fi

for tool in cmake ninja conan; do
    command -v "$tool" &>/dev/null || { echo "Missing: $tool (run with --install-deps)"; exit 1; }
done

echo "[build] Native ARM64 plugin..."
bash "$REPO_ROOT/tools/build_plugin_linux.sh" \
    --sdk-dir "$SDK_DIR" \
    --arch arm64 \
    --package

DIST_SO=""
for candidate in \
    "$REPO_ROOT/dist/linux-arm64/yolo26_people_analytics_plugin.so" \
    "$REPO_ROOT/dist/linux-arm64/libyolo26_people_analytics_plugin.so"; do
    if [[ -f "$candidate" ]]; then
        DIST_SO="$candidate"
        break
    fi
done

if [[ -z "$DIST_SO" ]]; then
    DIST_SO=$(find "$REPO_ROOT/dist/linux-arm64" -maxdepth 1 -name '*.so' | head -1)
fi

MANIFEST="$REPO_ROOT/dist/linux-arm64/manifest.json"
[[ -f "$MANIFEST" ]] || MANIFEST="$REPO_ROOT/config/manifest.json"

if [[ -z "$DIST_SO" || ! -f "$DIST_SO" ]]; then
    echo "ERROR: packaged .so not found under dist/linux-arm64/"
    exit 1
fi

echo ""
echo "Built: $DIST_SO"
echo "Manifest: $MANIFEST"

if $DO_INSTALL; then
    PLUGIN_HOME="$NX_PLUGIN_ROOT/yolo26_people_analytics_plugin"
    echo "[install] → $PLUGIN_HOME"
    mkdir -p "$PLUGIN_HOME"
    cp "$DIST_SO" "$PLUGIN_HOME/yolo26_people_analytics_plugin.so"
    cp "$MANIFEST" "$PLUGIN_HOME/manifest.json"
    systemctl restart networkoptix-mediaserver
    echo "[install] networkoptix-mediaserver restarted"
fi

echo ""
echo "Done. Verify: curl -k https://127.0.0.1:7001/ec2/pluginInfo/ | grep -i yolo26"
