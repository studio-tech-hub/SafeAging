#!/usr/bin/env bash
# Fix Docker build DNS on AI Box (dhcpcd / embedded images often break container DNS).
set -euo pipefail

mkdir -p /etc/docker
if [[ -f /etc/docker/daemon.json ]]; then
    cp -a /etc/docker/daemon.json /etc/docker/daemon.json.bak.$(date +%s)
fi

cat > /etc/docker/daemon.json <<'EOF'
{
  "dns": ["192.168.1.1", "8.8.8.8", "8.8.4.4"]
}
EOF

systemctl restart docker
sleep 5
docker info 2>/dev/null | grep -i dns || true

# quick DNS test inside docker
docker run --rm python:3.11-slim bash -c "getent hosts deb.debian.org && echo DNS_OK" || echo DNS_STILL_FAIL
