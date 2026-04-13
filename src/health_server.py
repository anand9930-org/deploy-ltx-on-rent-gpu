"""Tiny health check server for RunPod serverless load-balancing.

Runs on a separate port (default 8001) alongside BentoML so RunPod's
load balancer can verify the worker is reachable. Returns 200 OK for
GET /ping regardless of BentoML readiness — the container being up
and this server responding means the pod is ready to accept BentoML
initialization time.

Use /ready (optional) to check against BentoML's /readyz.
"""

import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class PingHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        if self.path == "/ping":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"status":"ok"}')
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format: str, *args) -> None:  # noqa: A002
        # Quieter logs — only log errors, skip every successful ping
        return


def main() -> None:
    port = int(os.getenv("HEALTH_PORT", "8001"))
    server = ThreadingHTTPServer(("0.0.0.0", port), PingHandler)
    print(f"[health] listening on 0.0.0.0:{port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        sys.exit(0)


if __name__ == "__main__":
    main()
