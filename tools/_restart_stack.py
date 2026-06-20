#!/usr/bin/env python3
import os
import sys
import time

import paramiko

HOST = "192.168.1.210"
ROOT = os.path.join(os.path.dirname(__file__), "..")


def run(c, cmd, timeout=120):
    _, o, e = c.exec_command(cmd, timeout=timeout)
    return (o.read() + e.read()).decode(errors="replace")


def main() -> int:
    pw = os.environ["SSH_PASSWORD"]
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(HOST, username="root", password=pw, timeout=20, banner_timeout=20, auth_timeout=20)
    sftp = c.open_sftp()
    sftp.put(os.path.join(ROOT, "docker-compose.yml"), "/root/SafeAging/docker-compose.yml")
    sftp.close()

    script = """
set -e
cd /root/SafeAging
docker compose down
docker volume rm -f safeaging_safeaging-postgres-data 2>/dev/null || true
docker compose up -d postgres minio
sleep 25
docker compose up -d minio-init analytics
for i in $(seq 1 30); do
  if curl -fsS http://127.0.0.1:18000/health >/dev/null 2>&1; then
    echo HEALTH_OK
    curl -s http://127.0.0.1:18000/health
    break
  fi
  sleep 10
done
docker ps -a
ss -tlnp | grep -E ':18000|:15432' || true
docker logs safeaging-postgres 2>&1 | tail -5
"""
    with c.open_sftp().file("/tmp/restart_stack.sh", "w") as f:
        f.write(script)
    c.open_sftp().chmod("/tmp/restart_stack.sh", 0o755)
    print(run(c, "bash /tmp/restart_stack.sh", timeout=600))
    c.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
