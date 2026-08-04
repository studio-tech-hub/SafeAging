#!/usr/bin/env python3
"""Production readiness checks for the SafeAging AI Box.

Run locally on the AI Box:
    python3 tools/ops_aibox_check.py --api-key "$API_KEY"

Run from a dev machine (host key must already be trusted, see
tools/_ssh_common.py / tools/ops.env.example — pass --insecure to bypass for
disposable lab boxes only):
    python tools/ops_aibox_check.py --host <AI_BOX_IP> --user root --password '...'
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


SERVICE_URL = "http://127.0.0.1:18000"


@dataclass
class Check:
    name: str
    status: str
    detail: str


class Runner:
    def __init__(self, host: str | None, user: str, password: str | None, insecure: bool = False) -> None:
        self.host = host
        self.user = user
        self.password = password
        self._ssh = None
        if host:
            try:
                import paramiko  # type: ignore[import]
            except ImportError as exc:
                raise SystemExit("paramiko is required for --host mode") from exc
            self._ssh = paramiko.SSHClient()
            self._ssh.load_system_host_keys()
            known_hosts = Path.home() / ".ssh" / "known_hosts"
            if known_hosts.exists():
                self._ssh.load_host_keys(str(known_hosts))
            if insecure:
                print(
                    "!!! WARNING: SSH host-key verification is DISABLED (--insecure). "
                    "Lab/dev use only. !!!",
                    file=sys.stderr,
                )
                self._ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            else:
                self._ssh.set_missing_host_key_policy(paramiko.RejectPolicy())
            try:
                self._ssh.connect(host, username=user, password=password, timeout=20)
            except paramiko.SSHException as exc:
                if "not found in known_hosts" in str(exc):
                    raise SystemExit(
                        f"Host key for {host} is not trusted ({exc}).\n"
                        f"Trust it once with: ssh-keyscan -H {host} >> ~/.ssh/known_hosts "
                        "(verify the fingerprint out-of-band first), or pass --insecure "
                        "for disposable lab boxes only."
                    ) from exc
                raise

    def close(self) -> None:
        if self._ssh is not None:
            self._ssh.close()

    def run(self, command: str, timeout: int = 30) -> tuple[int, str]:
        if self._ssh is not None:
            _, stdout, stderr = self._ssh.exec_command(command, timeout=timeout)
            out = stdout.read() + stderr.read()
            return stdout.channel.recv_exit_status(), out.decode(errors="replace")
        proc = subprocess.run(
            command,
            shell=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
        )
        return proc.returncode, proc.stdout


def _http_json(runner: Runner, path: str, api_key: str | None = None) -> dict[str, Any]:
    headers = ""
    if api_key:
        headers = f" -H {shlex.quote('X-API-Key: ' + api_key)}"
    code, out = runner.run(f"curl -fsS{headers} {shlex.quote(SERVICE_URL + path)}", timeout=10)
    if code != 0:
        raise RuntimeError(out.strip() or f"curl exited {code}")
    return json.loads(out)


def _pct(raw: str) -> int | None:
    raw = raw.strip().rstrip("%")
    return int(raw) if raw.isdigit() else None


def check_http(runner: Runner, api_key: str | None) -> list[Check]:
    checks: list[Check] = []
    try:
        health = _http_json(runner, "/health")
        checks.append(
            Check(
                "analytics_health",
                "PASS" if health.get("ready") else "FAIL",
                f"status={health.get('status')} reasons={health.get('reason_codes', [])}",
            )
        )
    except Exception as exc:
        checks.append(Check("analytics_health", "FAIL", f"/health unreachable: {exc}"))
        return checks

    try:
        status = _http_json(runner, "/status", api_key=api_key)
        active = int(status.get("active_cameras", 0) or 0)
        checks.append(
            Check(
                "analytics_status_auth",
                "PASS",
                f"active_cameras={active} total_requests={status.get('total_requests')}",
            )
        )
        if active <= 0:
            checks.append(
                Check(
                    "camera_traffic",
                    "WARN",
                    "no active cameras; check Nx recording/license/plugin API key",
                )
            )
        for cam_id, stats in sorted((status.get("cameras") or {}).items()):
            avg = float(stats.get("avg_inference_ms", 0) or stats.get("avg_infer_ms", 0) or 0)
            p95 = float(stats.get("p95_inference_ms", 0) or stats.get("p95_infer_ms", 0) or 0)
            if avg >= 100 or p95 >= 200:
                checks.append(
                    Check(
                        "camera_latency",
                        "WARN",
                        f"{cam_id}: avg={avg:.0f}ms p95={p95:.0f}ms target_avg<=100ms",
                    )
                )
    except Exception as exc:
        checks.append(Check("analytics_status_auth", "FAIL", f"/status unavailable or unauthorized: {exc}"))
    return checks


def check_docker(runner: Runner) -> list[Check]:
    checks: list[Check] = []
    code, out = runner.run("docker ps --format '{{.Names}}\\t{{.Status}}' | sort", timeout=20)
    if code != 0:
        return [Check("docker", "FAIL", out.strip() or "docker ps failed")]
    required = {"safeaging-analytics", "safeaging-minio"}
    running = {line.split("\t", 1)[0] for line in out.splitlines() if line.strip()}
    missing = sorted(required - running)
    checks.append(
        Check(
            "docker_services",
            "PASS" if not missing else "FAIL",
            "all required containers running" if not missing else f"missing={missing}",
        )
    )

    code, out = runner.run(
        "docker exec safeaging-analytics printenv API_KEY_REQUIRED SMTP_HOST ALERT_EMAIL_TO ALERT_FALL_ENABLED 2>/dev/null",
        timeout=20,
    )
    env = out.splitlines()
    api_required = env[0].strip().lower() if len(env) > 0 else ""
    smtp_host = env[1].strip() if len(env) > 1 else ""
    alert_to = env[2].strip() if len(env) > 2 else ""
    checks.append(
        Check(
            "api_key_required",
            "PASS" if api_required == "true" else "FAIL",
            f"API_KEY_REQUIRED={api_required or '(empty)'}",
        )
    )
    checks.append(
        Check(
            "alert_email_config",
            "PASS" if smtp_host and alert_to else "WARN",
            "SMTP and recipient configured" if smtp_host and alert_to else "SMTP_HOST or ALERT_EMAIL_TO is empty",
        )
    )

    code, out = runner.run(
        "docker inspect safeaging-minio --format '{{range .Mounts}}{{.Destination}} {{.Source}}{{println}}{{end}}' 2>/dev/null",
        timeout=20,
    )
    storage_detail = out.strip() or "minio mounts not found"
    storage_on_docker_root = "/var/lib/docker" in storage_detail.replace("\\", "/")
    checks.append(
        Check(
            "minio_storage_separation",
            "WARN" if storage_on_docker_root else "PASS",
            storage_detail if not storage_on_docker_root else storage_detail + " (move MinIO to separate disk)",
        )
    )
    return checks


def check_monitoring(runner: Runner) -> list[Check]:
    checks: list[Check] = []
    code, out = runner.run("curl -fsS http://127.0.0.1:19090/-/ready", timeout=10)
    checks.append(
        Check(
            "prometheus_ready",
            "PASS" if code == 0 else "WARN",
            out.strip() or "Prometheus readiness endpoint unavailable",
        )
    )
    code, out = runner.run(
        "python3 -c 'import json,urllib.request; "
        "d=json.load(urllib.request.urlopen(\"http://127.0.0.1:19090/api/v1/rules\", timeout=5)); "
        "names=[r.get(\"name\") for g in d.get(\"data\",{}).get(\"groups\",[]) "
        "for r in g.get(\"rules\",[]) if r.get(\"type\")==\"alerting\"]; "
        "print(json.dumps(names))'",
        timeout=15,
    )
    if code != 0:
        checks.append(Check("prometheus_alert_rules", "WARN", out.strip() or "rules API unavailable"))
    else:
        try:
            names = json.loads(out)
        except json.JSONDecodeError:
            names = []
        checks.append(
            Check(
                "prometheus_alert_rules",
                "PASS" if len(names) >= 7 else "WARN",
                f"{len(names)} alert rules loaded",
            )
        )
    code, out = runner.run("curl -fsS http://127.0.0.1:13000/api/health", timeout=10)
    checks.append(
        Check(
            "grafana_health",
            "PASS" if code == 0 else "WARN",
            out.strip()[:160] or "Grafana health endpoint unavailable",
        )
    )
    return checks


def check_host(runner: Runner) -> list[Check]:
    checks: list[Check] = []
    code, out = runner.run("ss -ltnp 2>/dev/null | grep ':18000 ' || true", timeout=20)
    listens = [line.strip() for line in out.splitlines() if line.strip()]
    exposed = any("0.0.0.0:18000" in line or "[::]:18000" in line for line in listens)
    checks.append(
        Check(
            "port_18000_binding",
            "FAIL" if exposed else "PASS",
            "; ".join(listens) if listens else "no listener found",
        )
    )

    code, out = runner.run("df -P / /var/lib/docker /root/SafeAging 2>/dev/null | awk 'NR>1 {print $6,$5,$4}'", timeout=20)
    if code != 0 or not out.strip():
        checks.append(Check("disk_space", "WARN", out.strip() or "df unavailable"))
    else:
        seen_mounts: set[str] = set()
        for line in out.splitlines():
            parts = line.split()
            if len(parts) < 3:
                continue
            mount, used_raw, avail = parts[0], parts[1], parts[2]
            if mount in seen_mounts:
                continue
            seen_mounts.add(mount)
            used = _pct(used_raw)
            if used is None:
                continue
            checks.append(
                Check(
                    f"disk_space:{mount}",
                    "FAIL" if used >= 90 else "WARN" if used >= 80 else "PASS",
                    f"used={used}% avail_kb={avail}",
                )
            )
    return checks


def print_checks(checks: Iterable[Check]) -> int:
    worst = 0
    for check in checks:
        if check.status == "FAIL":
            worst = 2
        elif check.status == "WARN" and worst < 1:
            worst = 1
        print(f"[{check.status:4}] {check.name}: {check.detail}")
    return 1 if worst == 2 else 0


def main() -> int:
    parser = argparse.ArgumentParser(description="SafeAging AI Box production readiness check")
    parser.add_argument("--host", help="SSH host/IP; omit to run checks locally")
    parser.add_argument("--user", default="root", help="SSH username")
    parser.add_argument("--password", help="SSH password")
    parser.add_argument("--api-key", default=os.getenv("API_KEY"), help="Analytics API key for /status")
    parser.add_argument(
        "--insecure",
        action="store_true",
        help="Disable SSH host-key verification (lab/dev use only, vulnerable to MITM)",
    )
    args = parser.parse_args()

    runner = Runner(args.host, args.user, args.password, insecure=args.insecure)
    try:
        checks: list[Check] = []
        checks.extend(check_http(runner, args.api_key))
        checks.extend(check_docker(runner))
        checks.extend(check_monitoring(runner))
        checks.extend(check_host(runner))
        return print_checks(checks)
    finally:
        runner.close()


if __name__ == "__main__":
    sys.exit(main())
