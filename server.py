"""Simple local web server for the RUL dashboard."""

from __future__ import annotations

import os
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

PORT = 8000


class QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, format: str, *args) -> None:  # noqa: A003
        return


if __name__ == "__main__":
    directory = os.path.dirname(os.path.abspath(__file__))
    handler = partial(QuietHandler, directory=directory)
    with ThreadingHTTPServer(("127.0.0.1", PORT), handler) as httpd:
        print(f"RUL dashboard running at http://127.0.0.1:{PORT}")
        httpd.serve_forever()
