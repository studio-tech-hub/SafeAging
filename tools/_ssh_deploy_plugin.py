#!/usr/bin/env python3
"""Pull latest git commit on AI Box and rebuild/install the Nx plugin."""
import sys

from _ssh_common import connect, get_credentials, run


def build_deploy_script(sdk_dir: str) -> str:
    return r"""
set -e
cd /root/SafeAging
echo "[1] git pull..."
git pull origin stage
echo "[2] verify..."
git log -1 --oneline
grep needUncompressedVideoFrames config/manifest.json | head -1
echo "[3] rebuild plugin..."
bash tools/build_plugin_aibox.sh --sdk-dir """ + sdk_dir + r""" --install
echo "[4] installed manifest:"
grep needUncompressedVideoFrames /opt/networkoptix/mediaserver/bin/plugins/yolo26_people_analytics_plugin/manifest.json
echo "[5] mediaserver active:"
systemctl is-active networkoptix-mediaserver
echo DONE — toggle YOLO26 plugin OFF/ON on each camera in Nx Client
"""


def main() -> int:
    creds = get_credentials(require_sdk_dir=True)
    c = connect(creds, timeout=15)
    print("Connected. Deploying...")
    code, _, _ = run(c, build_deploy_script(creds.sdk_dir), timeout=600)
    print(f"exit={code}")
    c.close()
    return code


if __name__ == "__main__":
    sys.exit(main())
