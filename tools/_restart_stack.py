#!/usr/bin/env python3
"""Full stack restart on AI Box (wipes Postgres volume, recreates postgres/minio/analytics).

Reads connection settings the same way as tools/_ssh_*.py — see
tools/ops_aibox_check.py / tools/ops.env.example for AIBOX_HOST, AIBOX_USER,
AIBOX_PASSWORD (or AIBOX_SSH_KEY_PATH). For backward compatibility, a legacy
SSH_PASSWORD env var is still honored if AIBOX_PASSWORD is not set.
"""
import os
import sys

from _ssh_common import REPO_ROOT, connect, get_credentials, run, upload_files

if not os.environ.get("AIBOX_PASSWORD") and os.environ.get("SSH_PASSWORD"):
    os.environ["AIBOX_PASSWORD"] = os.environ["SSH_PASSWORD"]

SCRIPT = """
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


def main() -> int:
    creds = get_credentials()
    c = connect(creds, timeout=20)
    upload_files(c, ["docker-compose.yml"], creds=creds, repo_root=REPO_ROOT)

    sftp = c.open_sftp()
    try:
        with sftp.file("/tmp/restart_stack.sh", "w") as f:
            f.write(SCRIPT)
        sftp.chmod("/tmp/restart_stack.sh", 0o755)
    finally:
        sftp.close()

    code, _, _ = run(c, "bash /tmp/restart_stack.sh", timeout=600)
    c.close()
    return code


if __name__ == "__main__":
    sys.exit(main())
