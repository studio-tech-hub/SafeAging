#!/usr/bin/env python3
"""Restore FACE_BACKEND=qnn_gpu on AI Box."""
import sys

from _ssh_common import REPO_ROOT, connect, get_credentials, run, upload_files

CMD = r"""
set -e
cd /root/SafeAging
docker compose -f docker-compose.yml -f docker-compose.aibox-hostdb.yml \
  -f docker-compose.aibox-qnn.yml -f docker-compose.aibox-hetero.yml up -d --force-recreate analytics
sleep 35
for f in face_worker.py face_engine.py face_providers.py face_identity.py qnn_ep_registry.py api.py config.py yolo_backend.py; do
  docker cp python/people_analytics_service/$f safeaging-analytics:/app/python/people_analytics_service/$f 2>/dev/null || true
done
docker restart safeaging-analytics
sleep 40
echo '=== env ==='
docker exec safeaging-analytics printenv FACE_BACKEND ENABLE_FACE_ASYNC YOLO_BACKEND
echo '=== startup ==='
docker logs safeaging-analytics 2>&1 | grep -E 'Face backend|insightface.*ready|Gallery preloaded|Lean ORT pool ready' | tail -6
"""


def main() -> int:
    creds = get_credentials()
    c = connect(creds, timeout=20)
    upload_files(c, ["docker-compose.aibox-hetero.yml"], creds=creds, repo_root=REPO_ROOT)
    code, _, _ = run(c, CMD, timeout=240)
    c.close()
    return code


if __name__ == "__main__":
    sys.exit(main())
