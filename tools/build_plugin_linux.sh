#!/usr/bin/env bash
# Build and (optionally) package the yolo26_people_analytics_plugin on Linux.
#
# Tested targets:
#   Ubuntu 22.04 x86_64  — development host
#   Ubuntu 20.04 aarch64 — Qualcomm QCS5430 AI Box (native build)
#   Ubuntu 22.04 x86_64  — cross-compiling for aarch64 with gcc-aarch64-linux-gnu
#
# Prerequisites (run once):
#   apt install -y cmake ninja-build build-essential
#   pip install "conan<2"        # project uses Conan 1.x API
#   conan profile new default --detect
#   conan profile update settings.compiler.libcxx=libstdc++11 default
#
# Cross-compile prerequisites (host = x64, target = arm64):
#   apt install -y gcc-aarch64-linux-gnu g++-aarch64-linux-gnu
#
# Usage:
#   ./tools/build_plugin_linux.sh --sdk-dir /path/to/metadata_sdk
#   ./tools/build_plugin_linux.sh --sdk-dir /path/to/metadata_sdk --package
#   ./tools/build_plugin_linux.sh --sdk-dir /path/to/metadata_sdk --arch arm64 --package

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# ── Defaults ──────────────────────────────────────────────────────────────────
BUILD_TYPE="Release"
HOST_ARCH="$(uname -m)"   # x86_64 or aarch64
TARGET_ARCH="$HOST_ARCH"
SDK_DIR=""
DO_PACKAGE=false

# ── Argument parsing ──────────────────────────────────────────────────────────
usage() {
    echo "Usage: $0 --sdk-dir <path> [--arch x64|arm64] [--build-type Release|Debug] [--package]"
    exit 1
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --sdk-dir)     SDK_DIR="$2";      shift 2 ;;
        --arch)        TARGET_ARCH="$2";  shift 2 ;;
        --build-type)  BUILD_TYPE="$2";   shift 2 ;;
        --package)     DO_PACKAGE=true;   shift   ;;
        --help|-h)     usage ;;
        *) echo "Unknown argument: $1"; usage ;;
    esac
done

[[ -z "$SDK_DIR" ]] && { echo "Error: --sdk-dir is required."; usage; }
[[ ! -d "$SDK_DIR" ]] && { echo "Error: SDK dir not found: $SDK_DIR"; exit 1; }

# ── Normalize arch ────────────────────────────────────────────────────────────
case "$TARGET_ARCH" in
    x86_64|x64|amd64)  TARGET_ARCH="x86_64";  ARCH_LABEL="linux-x64"  ;;
    aarch64|arm64)      TARGET_ARCH="aarch64"; ARCH_LABEL="linux-arm64" ;;
    *) echo "Unsupported --arch: $TARGET_ARCH (expected x64 or arm64)"; exit 1 ;;
esac

CROSS_COMPILE=false
if [[ "$TARGET_ARCH" != "$HOST_ARCH" ]]; then
    CROSS_COMPILE=true
fi

# ── Directories ───────────────────────────────────────────────────────────────
SRC_DIR="${REPO_ROOT}/config"
BUILD_DIR="${REPO_ROOT}/build/yolo26_people_analytics_plugin_${ARCH_LABEL}"
DIST_DIR="${REPO_ROOT}/dist/${ARCH_LABEL}"

echo "============================================================"
echo " SafeAging Plugin — Linux Build"
echo "============================================================"
echo " Repository:    $REPO_ROOT"
echo " SDK dir:       $SDK_DIR"
echo " Build dir:     $BUILD_DIR"
echo " Build type:    $BUILD_TYPE"
echo " Target arch:   $TARGET_ARCH ($ARCH_LABEL)"
echo " Cross-compile: $CROSS_COMPILE"
echo " Package:       $DO_PACKAGE"
echo "============================================================"

# ── Toolchain checks ──────────────────────────────────────────────────────────
if ! command -v cmake &>/dev/null; then
    echo "Error: cmake not found. Install with: apt install cmake"
    exit 1
fi
if ! command -v ninja &>/dev/null; then
    echo "Error: ninja not found. Install with: apt install ninja-build"
    exit 1
fi
if ! command -v conan &>/dev/null; then
    echo "Error: conan not found. Install with: pip install 'conan<2'"
    exit 1
fi

CONAN_VER=$(conan --version 2>&1 | grep -oP '\d+\.\d+\.\d+' | head -1 || echo "unknown")
CONAN_MAJOR="${CONAN_VER%%.*}"
if [[ "$CONAN_MAJOR" =~ ^[0-9]+$ ]] && [[ "$CONAN_MAJOR" -ge 2 ]]; then
    echo ""
    echo "WARNING: Conan $CONAN_VER detected — this project uses the Conan 1.x CMake API."
    echo "  Downgrade: pip install 'conan<2'"
    echo "  Build may fail at the conan_cmake_configure step."
    echo ""
fi

if [[ "$CROSS_COMPILE" == "true" ]]; then
    CC_ARM="aarch64-linux-gnu-gcc"
    CXX_ARM="aarch64-linux-gnu-g++"
    for tool in "$CC_ARM" "$CXX_ARM"; do
        if ! command -v "$tool" &>/dev/null; then
            echo "Error: $tool not found."
            echo "  Install cross toolchain: apt install gcc-aarch64-linux-gnu g++-aarch64-linux-gnu"
            exit 1
        fi
    done
fi

# ── CMake configure ───────────────────────────────────────────────────────────
mkdir -p "$BUILD_DIR"

CMAKE_ARGS=(
    -S "$SRC_DIR"
    -B "$BUILD_DIR"
    -G Ninja
    -DCMAKE_BUILD_TYPE="$BUILD_TYPE"
    -DmetadataSdkDir="$SDK_DIR"
)

if [[ "$CROSS_COMPILE" == "true" ]]; then
    CMAKE_ARGS+=(
        -DCMAKE_SYSTEM_NAME=Linux
        -DCMAKE_SYSTEM_PROCESSOR=aarch64
        -DCMAKE_C_COMPILER="$CC_ARM"
        -DCMAKE_CXX_COMPILER="$CXX_ARM"
    )
fi

echo ""
echo "[1/3] Configuring..."
cmake "${CMAKE_ARGS[@]}"

# ── Build ─────────────────────────────────────────────────────────────────────
echo ""
echo "[2/3] Building..."
cmake --build "$BUILD_DIR" --config "$BUILD_TYPE" --parallel "$(nproc)"

# ── Locate .so ───────────────────────────────────────────────────────────────
SO_FILE=$(find "$BUILD_DIR" -maxdepth 2 -name "libyolo26_people_analytics_plugin.so" -o \
                             -maxdepth 2 -name "yolo26_people_analytics_plugin.so" 2>/dev/null | head -1)
if [[ -z "$SO_FILE" ]]; then
    echo "Warning: .so file not found under $BUILD_DIR"
else
    echo ""
    echo "Library: $SO_FILE"
fi

# ── Package ───────────────────────────────────────────────────────────────────
if [[ "$DO_PACKAGE" == "true" ]]; then
    echo ""
    echo "[3/3] Packaging..."
    mkdir -p "$DIST_DIR"

    # cmake --install assembles the artifact using the install() rules in CMakeLists.txt
    cmake --install "$BUILD_DIR" --prefix "$DIST_DIR"

    VERSION="0.0.0"
    VERSION_JSON="${BUILD_DIR}/version_info.json"
    if [[ -f "$VERSION_JSON" ]]; then
        VERSION=$(grep -oP '"plugin_version"\s*:\s*"\K[^"]+' "$VERSION_JSON" 2>/dev/null || echo "0.0.0")
    fi

    ARCHIVE_NAME="yolo26_people_analytics_plugin-${VERSION}-${ARCH_LABEL}.tar.gz"
    ARCHIVE_PATH="${REPO_ROOT}/dist/${ARCHIVE_NAME}"

    (cd "${REPO_ROOT}/dist" && tar czf "$ARCHIVE_NAME" "${ARCH_LABEL}/")

    echo ""
    echo "============================================================"
    echo " Artifact: $ARCHIVE_PATH"
    echo " Contents:"
    tar tzf "$ARCHIVE_PATH" | sed 's/^/   /'
    echo "============================================================"
else
    echo ""
    echo "[3/3] Skipped (pass --package to create release archive)."
fi
