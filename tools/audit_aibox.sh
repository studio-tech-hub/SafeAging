#!/usr/bin/env bash
# SafeAging — AI Box pre-deploy audit
# Run on the box: bash tools/audit_aibox.sh
# Or one-shot: curl/bash after copying to the device.

set -u

pass() { echo "[OK]   $*"; }
warn() { echo "[WARN] $*"; }
fail() { echo "[FAIL] $*"; }
section() { echo; echo "========== $* =========="; }

section "SYSTEM"
uname -a
if [[ -f /etc/os-release ]]; then
    . /etc/os-release
    echo "OS: ${PRETTY_NAME:-unknown}"
fi
echo "Hostname: $(hostname)"
echo "Uptime: $(uptime -p 2>/dev/null || uptime)"

arch=$(uname -m)
if [[ "$arch" == "aarch64" ]]; then
    pass "Architecture aarch64 (ARM64) — compatible with Nx Meta arm64 + AI Box 6490"
else
    fail "Architecture $arch — expected aarch64"
fi

section "CPU / RAM / DISK"
echo "CPU cores: $(nproc)"
free -h
echo "--- mount points ---"
df -hT | grep -E '^/dev|Filesystem'

root_avail=$(df -BG / | awk 'NR==2 {gsub(/G/,"",$4); print $4}')
if [[ "${root_avail:-0}" -ge 15 ]]; then
    pass "Root free space: ${root_avail}G"
else
    warn "Root free space low: ${root_avail}G — Docker build needs ~8–15G"
fi

section "NETWORK"
ip -br addr
echo "--- default route ---"
ip route | head -3
if command -v ping &>/dev/null; then
    if ping -c1 -W2 8.8.8.8 &>/dev/null; then
        pass "Internet reachable (ping 8.8.8.8)"
    else
        warn "No internet — model download / apt may fail"
    fi
fi

section "DOCKER"
if command -v docker &>/dev/null; then
    pass "docker: $(docker --version)"
    if docker compose version &>/dev/null; then
        pass "docker compose: $(docker compose version)"
    else
        fail "docker compose plugin missing"
    fi
    echo "--- containers ---"
    docker ps -a 2>/dev/null || true
else
    fail "Docker not installed"
fi

section "NX META / NETWORK OPTIX"
nx_units=$(systemctl list-units --type=service --all 2>/dev/null | grep -iE 'networkoptix|metavms|nxwitness|nx-' || true)
if [[ -n "$nx_units" ]]; then
    pass "Nx-related systemd units found:"
    echo "$nx_units"
else
    fail "No Nx MediaServer systemd unit — Nx Meta not installed yet"
fi

echo "--- /opt ---"
ls -la /opt 2>/dev/null | head -25 || true

plugin_dirs=$(find /opt -type d -path '*/plugins' 2>/dev/null | head -5)
if [[ -n "$plugin_dirs" ]]; then
    pass "Plugin directories:"
    echo "$plugin_dirs"
    find /opt -path '*/plugins/*' \( -name '*.so' -o -name '*.dll' \) 2>/dev/null | head -20
else
    warn "No plugins directory under /opt yet"
fi

section "BUILD TOOLS (for plugin .so on box)"
have_build=true
for tool in gcc g++ make cmake ninja git curl python3 pip3; do
    if command -v "$tool" &>/dev/null; then
        case "$tool" in
            gcc) gcc --version | head -1 ;;
            cmake) cmake --version | head -1 ;;
            python3) python3 --version ;;
            pip3) pip3 --version ;;
            *) echo "$tool: $(command -v $tool)" ;;
        esac
    else
        fail "$tool not found"
        have_build=false
    fi
done
if command -v conan &>/dev/null; then
    pass "conan: $(conan --version 2>&1 | head -1)"
else
    warn "conan not installed — needed to build plugin ON the box"
fi

section "PYTHON / PIP PACKAGES"
python3 -c "import sys; print('python', sys.version)" 2>/dev/null || fail "python3 broken"
if python3 -c "import torch" 2>/dev/null; then
    python3 -c "import torch; print('torch', torch.__version__)"
else
    warn "torch not on host (OK if only using Docker for analytics)"
fi

section "LISTENING PORTS (common)"
ss -tlnp 2>/dev/null | grep -E ':22 |:7001|:18000|:5432|:9000|:80 |:443 ' || \
    netstat -tlnp 2>/dev/null | grep -E ':22 |:7001|:18000' || \
    warn "ss/netstat limited — run as root for process names"

section "SAFEAGING REPO"
for d in /root/SafeAging /home/*/SafeAging ~/SafeAging; do
    if [[ -d "$d" ]]; then
        pass "Repo found: $d"
        ls -la "$d" | head -10
    fi
done
if [[ ! -d /root/SafeAging && ! -d ~/SafeAging ]]; then
    warn "SafeAging repo not on box yet"
fi

section "MODEL FILES"
for d in /root/SafeAging/models ./models /opt/safeaging/models; do
    if [[ -f "$d/yolo26n.pt" ]]; then
        pass "yolo26n.pt at $d ($(du -h "$d/yolo26n.pt" | cut -f1))"
    fi
done

section "SUMMARY / NEXT STEPS"
echo "1. Nx Meta Server arm64 .deb if not installed"
echo "2. Docker + compose if missing"
echo "3. Copy SafeAging repo to box"
echo "4. Build plugin .so ON box (Windows .dll does NOT run on Linux)"
echo "5. TORCH_INDEX_URL= docker compose up on box"
echo "6. Plugin service_host=127.0.0.1"
echo
echo "Audit complete."
