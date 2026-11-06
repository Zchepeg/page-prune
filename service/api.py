from __future__ import annotations

import argparse
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from collector.schemas import Candidate
from local_security import LocalSessionManager, SecurityError
from .config import load_config
from .factory import create_orchestrator


def make_handler(orchestrator, session_manager=None):
    sessions = session_manager or LocalSessionManager()
    stats_lock = threading.Lock()
    stats = {
        "started_at": int(time.time()),
        "batches_received": 0,
        "candidates_received": 0,
        "invalid_requests": 0,
    }

    def stats_snapshot() -> dict:
        with stats_lock:
            return dict(stats)

    class Handler(BaseHTTPRequestHandler):
        def _origin(self) -> str:
            return self.headers.get("Origin", "")

        def _trusted_origin(self) -> bool:
            origin = self._origin()
            return not origin or origin.startswith("chrome-extension://") or self._local_page_origin()

        def _local_page_origin(self) -> bool:
            origin = self._origin()
            host = self.headers.get("Host", "")
            return bool(origin and host and origin == f"http://{host}" and host.split(":", 1)[0] in {"127.0.0.1", "localhost"})

        def _send(self, status: int, payload: dict) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            origin = self._origin()
            if self._trusted_origin() and origin:
                self.send_header("Access-Control-Allow-Origin", origin)
                self.send_header("Vary", "Origin")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _send_secure(self, status: int, payload: dict, envelope: dict, path: str) -> None:
            secured = sessions.encrypt_response(
                str(envelope["session_id"]),
                int(envelope["sequence"]),
                path,
                payload,
            )
            self._send(status, secured)

        def do_OPTIONS(self) -> None:  # noqa: N802
            if not self._trusted_origin():
                self._send(403, {"error": "untrusted_origin"})
                return
            self.send_response(204)
            origin = self._origin()
            if origin:
                self.send_header("Access-Control-Allow-Origin", origin)
                self.send_header("Vary", "Origin")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.end_headers()

        def do_GET(self) -> None:  # noqa: N802
            parsed = urlsplit(self.path)
            if parsed.path == "/health":
                self._send(200, {"status": "ok", "service": stats_snapshot()})
            elif parsed.path == "/v1/training/stats":
                self._send(200, orchestrator.training_stats())
            elif parsed.path == "/v1/status":
                self._send(
                    200,
                    {
                        "status": "ok",
                        "service": stats_snapshot(),
                        "training": orchestrator.training_stats(),
                        "probes": orchestrator.probe_stats(),
                        "decision_runtime": orchestrator.runtime_stats(),
                        "performance": orchestrator.performance_stats(),
                        "security": sessions.stats(),
                    },
                )
            elif parsed.path == "/v1/probes/review":
                query = parse_qs(parsed.query)
                try:
                    limit = int(query.get("limit", ["50"])[0])
                except ValueError:
                    limit = 50
                self._send(200, {"items": orchestrator.list_probe_reviews(limit)})
            elif parsed.path == "/v1/training/review":
                query = parse_qs(parsed.query)
                try:
                    limit = int(query.get("limit", ["50"])[0])
                except ValueError:
                    limit = 50
                self._send(200, {"items": orchestrator.list_training_reviews(limit)})
            elif parsed.path == "/review":
                body = (Path(__file__).with_name("review.html")).read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.send_header("X-Content-Type-Options", "nosniff")
                self.send_header("X-Frame-Options", "DENY")
                self.send_header("Referrer-Policy", "no-referrer")
                self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'none'")
                self.end_headers()
                self.wfile.write(body)
            else:
                self._send(404, {"error": "not_found"})

        def do_POST(self) -> None:  # noqa: N802
            if not self._trusted_origin():
                self._send(403, {"error": "untrusted_origin"})
                return
            if self.path not in {"/v1/security/session", "/v1/decisions", "/v1/training/manual", "/v1/training/review", "/v1/probes", "/v1/probes/review"}:
                self._send(404, {"error": "not_found"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length > 1_000_000:
                    raise ValueError("request_too_large")
                payload = json.loads(self.rfile.read(length))
                if self.path == "/v1/security/session":
                    result = sessions.create(
                        str(payload.get("client_id", "")),
                        str(payload.get("client_public_key", "")),
                        self._origin(),
                        str(payload.get("pairing_code", "")),
                    )
                    self._send(200, result)
                    return
                secure_paths = {"/v1/decisions", "/v1/training/manual", "/v1/probes"}
                secure_envelope = None
                if self.path in secure_paths:
                    secure_envelope = payload
                    payload = sessions.decrypt(payload, "POST", self.path, self._origin())
                if self.path == "/v1/probes/review":
                    if not self._local_page_origin():
                        self._send(403, {"error": "local_review_origin_required"})
                        return
                    review_id = str(payload.get("review_id", ""))
                    label = int(payload.get("label", -1))
                    if not review_id or label not in {0, 1, 2}:
                        raise ValueError("review_id_and_valid_label_are_required")
                    self._send(200, {"result": orchestrator.resolve_probe_review(review_id, label)})
                    return
                if self.path == "/v1/training/review":
                    if not self._local_page_origin():
                        self._send(403, {"error": "local_review_origin_required"})
                        return
                    review_id = str(payload.get("review_id", ""))
                    label = int(payload.get("label", -1))
                    if not review_id or label not in {0, 1, 2}:
                        raise ValueError("review_id_and_valid_label_are_required")
                    self._send(200, {"result": orchestrator.resolve_training_review(review_id, label)})
                    return
                if self.path == "/v1/probes":
                    raw_probe = payload.get("probe")
                    if not isinstance(raw_probe, dict):
                        raise TypeError("probe_must_be_an_object")
                    result = {"result": orchestrator.record_probe(raw_probe)}
                    if secure_envelope:
                        self._send_secure(200, result, secure_envelope, self.path)
                    else:
                        self._send(200, result)
                    return
                if self.path == "/v1/training/manual":
                    raw_candidate = payload.get("candidate")
                    if not isinstance(raw_candidate, dict):
                        raise TypeError("candidate_must_be_an_object")
                    label = int(payload.get("label", 0))
                    if label not in {0, 1, 2}:
                        raise ValueError("label_must_be_0_1_or_2")
                    result = orchestrator.record_manual(Candidate.from_dict(raw_candidate), label)
                    response = {"result": result}
                    if secure_envelope:
                        self._send_secure(200, response, secure_envelope, self.path)
                    else:
                        self._send(200, response)
                    return
                raw_candidates = payload.get("candidates", [])
                if not isinstance(raw_candidates, list):
                    raise TypeError("candidates_must_be_a_list")
                if len(raw_candidates) > 80:
                    raise ValueError("candidate_batch_limit_exceeded")
                if not all(isinstance(item, dict) for item in raw_candidates):
                    raise TypeError("each_candidate_must_be_an_object")
                candidates = [Candidate.from_dict(item) for item in raw_candidates]
                with stats_lock:
                    stats["batches_received"] += 1
                    stats["candidates_received"] += len(candidates)
                response = {"results": orchestrator.process_many(candidates)}
                if secure_envelope:
                    self._send_secure(200, response, secure_envelope, self.path)
                else:
                    self._send(200, response)
            except SecurityError as exc:
                with stats_lock:
                    stats["invalid_requests"] += 1
                self._send(401, {"error": "security_error", "detail": str(exc)})
            except (ValueError, TypeError, json.JSONDecodeError) as exc:
                with stats_lock:
                    stats["invalid_requests"] += 1
                self._send(400, {"error": "invalid_request", "detail": str(exc)})

        def log_message(self, format: str, *args) -> None:
            return

    return Handler


def main() -> None:
    root = Path(__file__).parents[1]
    local_config = root / "config" / "local.yaml"
    default_config = local_config if local_config.exists() else root / "config" / "config.yaml"
    parser = argparse.ArgumentParser(description="JEV-aware ad-blocking decision service")
    parser.add_argument("--config", default=str(default_config))
    args = parser.parse_args()
    config = load_config(args.config)
    server_config = config["server"]
    security_config = config.get("local_security", {})
    project_root = Path(args.config).resolve().parent.parent
    sessions = LocalSessionManager(
        lifetime_seconds=int(security_config.get("session_lifetime_seconds", 3600)),
        clock_skew_seconds=int(security_config.get("clock_skew_seconds", 120)),
        registry_path=project_root / security_config.get("registry_path", "data/security/paired_clients.json"),
        pairing_lifetime_seconds=int(security_config.get("pairing_lifetime_seconds", 600)),
        pairing_attempt_limit=int(security_config.get("pairing_attempt_limit", 10)),
    )
    server = ThreadingHTTPServer(
        (server_config["host"], int(server_config["port"])),
        make_handler(create_orchestrator(args.config), sessions),
    )
    print(f"Listening on http://{server_config['host']}:{server_config['port']}")
    if sessions.pairing_code:
        print(f"Extension pairing code: {sessions.pairing_code}")
    server.serve_forever()


if __name__ == "__main__":
    main()
