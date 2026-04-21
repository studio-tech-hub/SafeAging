import sys
from pathlib import Path
from typing import Any, Dict

import uvicorn

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from people_analytics_service.api import app
from people_analytics_service.config import SERVICE_HOST, SERVICE_PORT, TLS_CERT_FILE, TLS_KEY_FILE, logger


if __name__ == "__main__":
    logger.info(f"Starting service on {SERVICE_HOST}:{SERVICE_PORT}")
    uvicorn_kwargs: Dict[str, Any] = {
        "app": app,
        "host": SERVICE_HOST,
        "port": SERVICE_PORT,
        "log_level": "info",
        "access_log": True,
    }
    if TLS_CERT_FILE and TLS_KEY_FILE:
        uvicorn_kwargs["ssl_certfile"] = TLS_CERT_FILE
        uvicorn_kwargs["ssl_keyfile"] = TLS_KEY_FILE
        logger.info("Using direct TLS via uvicorn ssl_certfile/ssl_keyfile")
    uvicorn.run(**uvicorn_kwargs)
