#!/usr/bin/env python3
"""Check whether heterogeneous pipeline is live on AI Box."""
import json
import sys

from _ssh_common import connect, get_credentials, run

CMD = r"""
cd /root/SafeAging
echo '=== container ==='
docker ps --filter name=safeaging-analytics --format '{{.Names}} | {{.Status}} | {{.Image}}'

echo '=== compose files on disk ==='
test -f docker-compose.aibox-hetero.yml && echo hetero_yml=ok || echo hetero_yml=missing

echo '=== env ==='
docker exec safeaging-analytics printenv MODEL_PATH YOLO_BACKEND FACE_BACKEND ENABLE_FACE_ASYNC FACE_ASYNC_WORKERS YOLO_LEAN_POOL_SIZE 2>/dev/null

echo '=== code patch present ==='
docker exec safeaging-analytics test -f /app/python/people_analytics_service/qnn_ep_registry.py && echo qnn_ep_registry=ok || echo qnn_ep_registry=missing
docker exec safeaging-analytics test -f /app/python/people_analytics_service/face_worker.py && echo face_worker=ok || echo face_worker=missing

echo '=== startup logs ==='
docker logs safeaging-analytics 2>&1 | grep -E 'Lean ORT pool ready|insightface.*ready|Face worker pool|Face backend|falling back|backend=cpu providers' | tail -10

echo '=== health ==='
curl -sf http://127.0.0.1:18000/health || echo HEALTH_UNREACHABLE

echo '=== mediaserver ==='
systemctl is-active networkoptix-mediaserver 2>/dev/null || echo inactive
"""


def main() -> int:
    creds = get_credentials()
    c = connect(creds, timeout=20)
    code, out_b, err_b = run(c, CMD, timeout=120, stream=False)
    c.close()

    out = out_b.decode(errors="replace")
    err = err_b.decode(errors="replace")
    sys.stdout.write(out)
    if err:
        sys.stderr.write(err)

    # Parse health block if present
    if "=== health ===" in out:
        block = out.split("=== health ===", 1)[1].split("=== mediaserver ===", 1)[0].strip()
        if block and block != "HEALTH_UNREACHABLE":
            try:
                d = json.loads(block.split("\n")[0])
                print("\n--- parsed health ---")
                print(f"status={d.get('status')} ready={d.get('ready')} device={d.get('device')}")
                p = d.get("pipeline") or {}
                for k in ("proc_fps", "avg_infer_ms", "queue_depth", "dropped_frames", "in_fps"):
                    if p.get(k) is not None:
                        print(f"  {k}={p[k]}")
                print(f"  cameras_registered={len(d.get('cameras') or [])}")
            except json.JSONDecodeError:
                print("\n--- health JSON parse failed ---")

    live = (
        "qnn_htp" in out
        and "qnn_gpu" in out
        and "ENABLE_FACE_ASYNC" not in out  # env line won't have this literal
        and "Lean ORT pool ready: backend=qnn_htp" in out
        and "backend=qnn_gpu" in out
    )
    print(f"\n--- verdict: heterogeneous_flow_live={'YES' if live else 'PARTIAL/NO'} ---")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
