from __future__ import annotations

import argparse
import os

import uvicorn


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the local Manga Production Studio web UI.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8420)
    parser.add_argument("--reload", action="store_true")
    parser.add_argument("--auth-token", help="Bearer token required for API access when binding beyond localhost")
    args = parser.parse_args()

    loopback_hosts = {"127.0.0.1", "localhost", "::1"}
    if args.host not in loopback_hosts and not args.auth_token:
        parser.error("--auth-token is required when --host is not loopback-only")
    if args.auth_token:
        os.environ["HYPOTAXIS_API_TOKEN"] = args.auth_token

    uvicorn.run("studio.app:app", host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
