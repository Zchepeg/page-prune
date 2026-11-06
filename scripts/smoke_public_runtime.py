from __future__ import annotations

import json
import sys
import tempfile
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from local_security import LocalSessionManager
from service.api import make_handler
from service.factory import create_orchestrator


def main():
    orchestrator = create_orchestrator(ROOT/'config/config.yaml')
    if orchestrator.runtime_stats()['backend'] != 'heuristic':
        raise RuntimeError('public smoke expects the development backend')
    with tempfile.TemporaryDirectory(prefix='jev-public-smoke-') as directory:
        sessions = LocalSessionManager(registry_path=Path(directory)/'paired.json')
        server = ThreadingHTTPServer(('127.0.0.1', 0), make_handler(orchestrator, sessions))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            for path in ('/health', '/v1/status'):
                with urlopen(f'http://127.0.0.1:{server.server_port}{path}', timeout=5) as response:
                    payload = json.load(response)
                if payload['status'] != 'ok':
                    raise RuntimeError('public runtime health check failed')
            if 'torch' in sys.modules:
                raise RuntimeError('public development runtime unexpectedly imported PyTorch')
            print(json.dumps({'passed': True, 'backend': 'heuristic', 'torch_imported': False, 'port': 'isolated_ephemeral', 'production_service_touched': False}))
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)


if __name__ == '__main__':
    main()
