#!/usr/bin/env python3
"""Upload device_agent.cpp only, rebuild and install the Nx plugin."""
import sys

from _ssh_common import REPO_ROOT, connect, get_credentials, run, upload_files

REL_PATH = "src/sample_company/vms_server_plugins/opencv_object_detection/device_agent.cpp"


def main() -> int:
    creds = get_credentials(require_sdk_dir=True)
    c = connect(creds, timeout=20)
    upload_files(c, [REL_PATH], creds=creds, repo_root=REPO_ROOT)
    code, _, _ = run(
        c,
        f"cd /root/SafeAging && bash tools/build_plugin_aibox.sh --sdk-dir {creds.sdk_dir} --install",
        timeout=900,
    )
    c.close()
    return code


if __name__ == "__main__":
    sys.exit(main())
