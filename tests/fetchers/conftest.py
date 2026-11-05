"""Shared pytest fixtures for fetcher tests."""

import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class _RedirectHandler(BaseHTTPRequestHandler):
    """Simple HTTP server that serves redirect endpoints for testing follow_redirects."""

    def do_GET(self):
        if self.path == "/redirect-relative":
            self.send_response(302)
            self.send_header("Location", "/final")
            self.end_headers()
        elif self.path == "/redirect-absolute":
            self.send_response(302)
            self.send_header("Location", f"http://127.0.0.1:{self.server.server_address[1]}/final")
            self.end_headers()
        elif self.path == "/redirect-chain-1":
            self.send_response(302)
            self.send_header("Location", "/redirect-chain-2")
            self.end_headers()
        elif self.path == "/redirect-chain-2":
            self.send_response(302)
            self.send_header("Location", "/final")
            self.end_headers()
        elif self.path == "/redirect-query":
            self.send_response(302)
            self.send_header("Location", "/final?foo=bar&baz=qux")
            self.end_headers()
        elif self.path == "/redirect-fragment":
            self.send_response(302)
            self.send_header("Location", "/final#section")
            self.end_headers()
        elif self.path == "/redirect-to-private":
            self.send_response(302)
            self.send_header("Location", "http://127.0.0.1:9999/private")
            self.end_headers()
        elif self.path == "/redirect-307":
            self.send_response(307)
            self.send_header("Location", "/final")
            self.end_headers()
        elif self.path == "/redirect-308":
            self.send_response(308)
            self.send_header("Location", "/final")
            self.end_headers()
        elif self.path == "/final":
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(b"<html><body>Final page</body></html>")
        elif self.path == "/no-redirect":
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(b"<html><body>No redirect</body></html>")
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        pass  # Suppress log output


def _find_free_port() -> int:
    """Find a free TCP port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


import pytest  # noqa: E402


@pytest.fixture(scope="module")
def redirect_server():
    """Start a local HTTP redirect test server on an ephemeral port.

    Yields the base URL (e.g. ``http://127.0.0.1:12345``) that tests can use.
    The server is shut down after all tests in the module complete.
    """
    port = _find_free_port()
    server = ThreadingHTTPServer(("127.0.0.1", port), _RedirectHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
