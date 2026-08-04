#!/usr/bin/env python3
"""Check SafeAging admin UI reachability on AI Box."""
import socket
import sys
import urllib.error
import urllib.request

from _ssh_common import connect, get_credentials, run

SSH_CMD = r"""
echo '=== local ui ==='
curl -sf -o /dev/null -w '/ui HTTP %{http_code}\n' http://127.0.0.1:18000/ui/ || echo '/ui local FAIL'
curl -sf -o /dev/null -w '/ HTTP %{http_code}\n' http://127.0.0.1:18000/ || echo '/ local FAIL'
curl -sf http://127.0.0.1:18000/ui/ | head -c 120 | tr '\n' ' '; echo

echo '=== static files in container ==='
docker exec safeaging-analytics ls -la /app/python/people_analytics_service/static/ 2>/dev/null | head -5

echo '=== port bind ==='
ss -tlnp | grep 18000 || netstat -tlnp 2>/dev/null | grep 18000

echo '=== auth ==='
curl -sf http://127.0.0.1:18000/health | python3 -c "import json,sys;d=json.load(sys.stdin);print('auth_enabled',d.get('auth_enabled'),'tls',d.get('tls_enabled'))"
"""


def check_lan(host: str, port: int = 18000) -> None:
    print(f"\n=== LAN probe from dev PC -> {host}:{port} ===")
    try:
        with socket.create_connection((host, port), timeout=5):
            print(f"TCP {host}:{port} OPEN")
    except OSError as exc:
        print(f"TCP {host}:{port} CLOSED ({exc})")
        return
    for path in ("/health", "/ui/", "/"):
        url = f"http://{host}:{port}{path}"
        try:
            with urllib.request.urlopen(url, timeout=8) as resp:
                print(f"GET {path} -> HTTP {resp.status}")
        except urllib.error.HTTPError as exc:
            print(f"GET {path} -> HTTP {exc.code}")
        except Exception as exc:
            print(f"GET {path} -> FAIL ({exc})")


def main() -> int:
    creds = get_credentials()
    c = connect(creds, timeout=20)
    code, out, err = run(c, SSH_CMD, timeout=60, stream=False)
    c.close()
    sys.stdout.write(out.decode(errors="replace"))
    if err:
        sys.stderr.write(err.decode(errors="replace"))
    check_lan(creds.host)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
