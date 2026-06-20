#!/usr/bin/env bash
# Assign static IPv4 on AI Box (Ubuntu 20.04 netplan).
# Default: 192.168.1.210/24 gateway 192.168.1.1
#
# Usage (on the box, while SSH is working):
#   sudo bash tools/set_static_ip_aibox.sh
#   sudo bash tools/set_static_ip_aibox.sh --ip 192.168.1.210 --gateway 192.168.1.1

set -euo pipefail

STATIC_IP="192.168.1.210"
GATEWAY="192.168.1.1"
DNS1="192.168.1.1"
DNS2="8.8.8.8"
PREFIX=24

while [[ $# -gt 0 ]]; do
    case "$1" in
        --ip)      STATIC_IP="$2"; shift 2 ;;
        --gateway) GATEWAY="$2"; shift 2 ;;
        --help|-h)
            echo "Usage: sudo $0 [--ip 192.168.1.210] [--gateway 192.168.1.1]"
            exit 0
            ;;
        *) echo "Unknown: $1"; exit 1 ;;
    esac
done

if [[ "$(id -u)" -ne 0 ]]; then
    echo "Run as root: sudo bash $0"
    exit 1
fi

IFACE=$(ip -o -4 route show to default 2>/dev/null | awk '{print $5}' | head -1)
if [[ -z "$IFACE" ]]; then
    IFACE=$(ip -o link show | awk -F': ' '{print $2}' | grep -E '^(eth|en)' | head -1)
fi
if [[ -z "$IFACE" ]]; then
    echo "ERROR: could not detect network interface"
    ip link
    exit 1
fi

echo "Interface:  $IFACE"
echo "Static IP:  ${STATIC_IP}/${PREFIX}"
echo "Gateway:    $GATEWAY"
echo "Hostname:   $(hostname)"
echo ""
read -r -p "Apply static IP? SSH will move to ${STATIC_IP}. Continue? [y/N] " ans
if [[ "${ans,,}" != "y" ]]; then
    echo "Aborted."
    exit 0
fi

if ! command -v netplan &>/dev/null || [[ ! -d /etc/netplan ]]; then
    echo "ERROR: netplan not found. Configure NetworkManager manually."
    exit 1
fi

STAMP=$(date +%Y%m%d-%H%M%S)
mkdir -p /etc/netplan/backup-"$STAMP"
cp -a /etc/netplan/*.yaml /etc/netplan/backup-"$STAMP"/ 2>/dev/null || true

CFG="/etc/netplan/99-safeaging-static.yaml"
cat > "$CFG" <<EOF
# SafeAging — static LAN IP (do not use DHCP for this host)
network:
  version: 2
  renderer: networkd
  ethernets:
    ${IFACE}:
      dhcp4: false
      dhcp6: false
      addresses:
        - ${STATIC_IP}/${PREFIX}
      routes:
        - to: default
          via: ${GATEWAY}
      nameservers:
        addresses: [${DNS1}, ${DNS2}]
EOF

chmod 600 "$CFG"
echo "Wrote $CFG"
netplan generate
netplan apply

echo ""
echo "Done. Reconnect with:"
echo "  ssh root@${STATIC_IP}"
