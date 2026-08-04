#!/usr/bin/env python3
"""Upload plugin sources + manifest, rebuild on AI Box, install to Nx."""
import sys

from _ssh_common import REPO_ROOT, connect, get_credentials, run, upload_files

UPLOAD = [
    "config/manifest.json",
    "src/sample_company/vms_server_plugins/opencv_object_detection/device_agent.cpp",
    "src/sample_company/vms_server_plugins/opencv_object_detection/device_agent.h",
    "src/sample_company/vms_server_plugins/opencv_object_detection/object_detector.cpp",
    "src/sample_company/vms_server_plugins/opencv_object_detection/object_detector.h",
]


def build_script(sdk_dir: str) -> str:
    return f"""
set -e
cd /root/SafeAging
grep -q confidence_threshold_percent config/manifest.json && echo manifest_ok
bash tools/build_plugin_aibox.sh --sdk-dir {sdk_dir} --install
grep confidence_threshold_percent /opt/networkoptix/mediaserver/bin/plugins/yolo26_people_analytics_plugin/manifest.json
systemctl is-active networkoptix-mediaserver
echo DONE
"""


def main() -> int:
    creds = get_credentials(require_sdk_dir=True)
    c = connect(creds, timeout=20)
    upload_files(c, UPLOAD, creds=creds, repo_root=REPO_ROOT)
    code, _, _ = run(c, build_script(creds.sdk_dir), timeout=900)
    c.close()
    return code


if __name__ == "__main__":
    sys.exit(main())
