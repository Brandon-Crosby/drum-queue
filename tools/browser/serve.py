#!/usr/bin/env python3
"""Serve static worker assets and separately staged models on localhost."""
import argparse
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assets', type=Path, required=True, help='Directory containing v1/manifest.json')
    parser.add_argument('--port', type=int, default=8788)
    args = parser.parse_args()
    site = Path(__file__).resolve().parents[2] / 'web/dist'
    assets = args.assets.resolve()
    if not (site / 'browser/worker.js').is_file():
        parser.error('Build web/dist first: python web/build_site.py')

    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(site), **kwargs)

        def translate_path(self, path):
            if path.startswith('/browser-assets/'):
                from urllib.parse import unquote, urlsplit
                relative = unquote(urlsplit(path).path).removeprefix('/browser-assets/')
                candidate = (assets / relative).resolve()
                if assets in candidate.parents:
                    return str(candidate)
                return str(assets / 'not-found')
            return super().translate_path(path)

        def end_headers(self):
            self.send_header('Cache-Control', 'no-store')
            super().end_headers()

        def do_POST(self):
            self.send_error(405, 'Static experiment: no audio upload or inference endpoint')

    server = ThreadingHTTPServer(('127.0.0.1', args.port), Handler)
    print(f'http://127.0.0.1:{server.server_port}/browser/worker.js', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.server_close()


if __name__ == '__main__':
    main()
