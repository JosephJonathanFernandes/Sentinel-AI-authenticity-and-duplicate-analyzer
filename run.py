"""Sentinel entrypoint.

Usage:
    python run.py               # launch the Streamlit dashboard
    python run.py --api         # launch the FastAPI backend (uvicorn)
    python run.py --api --port 8000
"""

from __future__ import annotations

import argparse
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description="Sentinel — image authenticity & duplicate analyzer")
    parser.add_argument("--api", action="store_true", help="Run the FastAPI backend instead of Streamlit")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8501, help="Port for Streamlit (default: 8501)")
    parser.add_argument("--api-port", type=int, default=8000, help="Port for FastAPI (default: 8000)")
    args = parser.parse_args()

    if args.api:
        import uvicorn
        print(f"Starting Sentinel API on http://{args.host}:{args.api_port}")
        uvicorn.run("app.main:app", host=args.host, port=args.api_port, reload=False)
    else:
        import streamlit.web.cli as stcli
        print(f"Starting Sentinel dashboard on http://{args.host}:{args.port}")
        sys.argv = [
            "streamlit",
            "run",
            "app/ui/dashboard.py",
            "--server.address",
            args.host,
            "--server.port",
            str(args.port),
        ]
        sys.exit(stcli.main())


if __name__ == "__main__":
    main()
