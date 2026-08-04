#!/usr/bin/env python3
"""Check face recognition gallery/logs and a live-tracks probe on the AI Box."""
import sys

from _ssh_common import connect, get_credentials, run

DEFAULT_API_KEY = "change-me-to-a-strong-secret"


def build_cmd(api_key: str) -> str:
    return r"""
sleep 45
echo "=== face logs ==="
docker logs safeaging-analytics 2>&1 | grep -iE '\[face\]|Gallery|insightface' | tail -15
echo "=== gallery ==="
docker exec safeaging-analytics python3 - <<'PY'
import os, sys
sys.path.insert(0, "/app/python")
os.environ["EDGE_SQLITE_PATH"] = "/app/runtime/edge_state.db"
os.environ["DATABASE_URL"] = ""
from people_analytics_service.db import edge_store
rows = edge_store.list_face_gallery()
print("embeddings", len(rows))
for r in rows[:8]:
    print(r.get("name"), r.get("person_id"))
from people_analytics_service import face_engine
print("face_available", face_engine.available())
print("gallery_size", face_engine.gallery_size())
PY
echo "=== live tracks ==="
curl -sf -H "X-API-Key: """ + api_key + r"""" http://127.0.0.1:18000/admin/tracks/live 2>/dev/null | python3 -c "import sys,json; d=json.load(sys.stdin); print('cameras', len(d)); [print(c, len(d[c]), [t.get('person_name') for t in d[c][:3]]) for c in d]" 2>/dev/null || echo "(no tracks)"
"""


def main() -> int:
    creds = get_credentials()
    api_key = creds.api_key or DEFAULT_API_KEY
    c = connect(creds, timeout=15)
    code, _, _ = run(c, build_cmd(api_key), timeout=120)
    c.close()
    return code


if __name__ == "__main__":
    sys.exit(main())
