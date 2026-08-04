#!/usr/bin/env python3
"""Rollback to CPU lean + restart mediaserver to restore detection."""
import sys

from _ssh_common import connect, get_credentials, run

CMD = r"""
set -e
cd /root/SafeAging

echo "=== rollback CPU lean (yolo26n.onnx float) ==="
docker compose -f docker-compose.yml -f docker-compose.aibox-hostdb.yml up -d --force-recreate analytics

for i in $(seq 1 24); do
  st=$(docker inspect -f '{{.State.Health.Status}}' safeaging-analytics 2>/dev/null || echo missing)
  line=$(docker logs safeaging-analytics 2>&1 | grep 'Lean ORT pool ready' | tail -1)
  echo "t=$((i*5))s health=$st"
  echo "$line"
  echo "$line" | grep -q 'backend=cpu_lean' && [[ "$st" == "healthy" ]] && break
  sleep 5
done

echo "=== health ==="
curl -sf http://127.0.0.1:18000/health | python3 -c "
import json,sys
d=json.load(sys.stdin); p=d.get('pipeline',{})
print('status', d.get('status'), 'backend', p.get('yolo_backend'), 'reason', d.get('reason_codes'))
"

echo "=== restart mediaserver (reset circuit breaker) ==="
systemctl restart networkoptix-mediaserver
sleep 12
systemctl is-active networkoptix-mediaserver

echo "=== wait for detections ==="
sleep 30
docker logs safeaging-analytics 2>&1 | grep -E 'detections=[1-9]|Summary:' | tail -8
curl -sf http://127.0.0.1:18000/health | python3 -c "
import json,sys
d=json.load(sys.stdin); p=d.get('pipeline',{})
print('final status', d.get('status'), 'backend', p.get('yolo_backend'))
for cid,c in (p.get('cameras') or {}).items():
 print(cid[:24], 'avg_ms', c.get('avg_infer_ms'), 'req', c.get('requests'), 'err', c.get('errors'))
"
journalctl -u networkoptix-mediaserver --since '1 min ago' 2>/dev/null | grep -iE 'circuit|detection|avg_infer' | tail -8
"""


def main() -> int:
    creds = get_credentials()
    c = connect(creds, timeout=20)
    code, _, _ = run(c, CMD, timeout=600)
    c.close()
    return code


if __name__ == "__main__":
    sys.exit(main())
