"""
Entry point for the Customer Support Agent API.

Usage:
    python run.py                    # development
    uvicorn api.server:app --host 0.0.0.0 --port 8000   # production
"""

import logging
import uvicorn

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)

if __name__ == "__main__":
    uvicorn.run(
        "api.server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,    # set to False in production
        log_level="info",
    )
