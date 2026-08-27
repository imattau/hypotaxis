"""Run the Manga Production Studio in a native desktop WebView.

The FastAPI application remains the UI server. This launcher starts it on a
loopback-only port and hands that URL to pywebview2.
"""

from __future__ import annotations

import argparse
import socket
import threading
import time


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_for_server(server: object, timeout: float = 30.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if getattr(server, "started", False):
            return
        if getattr(server, "should_exit", False):
            break
        time.sleep(0.05)
    raise RuntimeError("the local studio server did not start")


def main() -> None:
    parser = argparse.ArgumentParser(description="Open Manga Production Studio as a desktop app.")
    parser.add_argument("--host", default="127.0.0.1", help="loopback host for the embedded studio server")
    parser.add_argument("--port", type=int, default=0, help="server port (0 selects a free port)")
    parser.add_argument("--width", type=int, default=1400)
    parser.add_argument("--height", type=int, default=900)
    parser.add_argument("--debug", action="store_true", help="enable WebView and Uvicorn debug logging")
    args = parser.parse_args()

    if args.host not in {"127.0.0.1", "localhost", "::1"}:
        parser.error("the desktop wrapper only supports loopback hosts")

    try:
        import uvicorn
        import webview
    except ImportError as exc:
        raise SystemExit(
            "Desktop dependencies are missing. Install with "
            "'pip install -r requirements-desktop.txt'."
        ) from exc

    port = args.port or _free_port()
    config = uvicorn.Config(
        "studio.app:app",
        host=args.host,
        port=port,
        log_level="info" if args.debug else "warning",
    )
    server = uvicorn.Server(config)
    server_thread = threading.Thread(target=server.run, name="hypotaxis-studio-server", daemon=True)
    server_thread.start()

    try:
        _wait_for_server(server)
        webview.create_window(
            "Hypotaxis — Manga Production Studio",
            f"http://{args.host}:{port}/",
            width=args.width,
            height=args.height,
            min_size=(900, 600),
            resizable=True,
        )
        webview.start(debug=args.debug)
    finally:
        server.should_exit = True
        server_thread.join(timeout=5)


if __name__ == "__main__":
    main()
