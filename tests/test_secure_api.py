import base64
import http.client
import json
import os
import threading
import time
from http.server import ThreadingHTTPServer

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

from local_security import LocalSessionManager
from service.api import make_handler


class Orchestrator:
    def __init__(self):
        self.received = None

    def record_probe(self, value):
        self.received = value
        return {"accepted": True}

    def training_stats(self):
        return {}

    def probe_stats(self):
        return {}

    def list_probe_reviews(self, limit):
        return []

    def list_training_reviews(self, limit):
        return [{"id": "candidate-review", "limit": limit}]

    def resolve_training_review(self, review_id, label):
        self.received = (review_id, label)
        return {"saved": True, "example_id": review_id}


def post(port, path, payload, origin):
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    body = json.dumps(payload).encode()
    connection.request("POST", path, body=body, headers={"Content-Type": "application/json", "Origin": origin})
    response = connection.getresponse()
    value = json.loads(response.read())
    connection.close()
    return response.status, value


def get(port, path):
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    connection.request("GET", path)
    response = connection.getresponse()
    value = json.loads(response.read())
    connection.close()
    return response.status, value


def test_secure_probe_api_round_trip(tmp_path):
    origin = "chrome-extension://integration-test"
    sessions = LocalSessionManager(registry_path=tmp_path / "clients.json")
    orchestrator = Orchestrator()
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(orchestrator, sessions))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        private = ec.generate_private_key(ec.SECP256R1())
        public = private.public_key().public_bytes(serialization.Encoding.X962, serialization.PublicFormat.UncompressedPoint)
        status, handshake = post(
            server.server_port,
            "/v1/security/session",
            {
                "client_id": "integration-client",
                "client_public_key": base64.b64encode(public).decode(),
                "pairing_code": sessions.pairing_code,
            },
            origin,
        )
        assert status == 200
        server_public = ec.EllipticCurvePublicKey.from_encoded_point(
            ec.SECP256R1(), base64.b64decode(handshake["server_public_key"])
        )
        shared = private.exchange(ec.ECDH(), server_public)
        request_key = HKDF(
            algorithm=hashes.SHA256(), length=32, salt=handshake["session_id"].encode(), info=b"jev-local-session-v1/request"
        ).derive(shared)
        response_key = HKDF(
            algorithm=hashes.SHA256(), length=32, salt=handshake["session_id"].encode(), info=b"jev-local-session-v1/response"
        ).derive(shared)
        sequence = 1
        timestamp = int(time.time() * 1000)
        nonce = os.urandom(12)
        path = "/v1/probes"
        aad = f"{handshake['session_id']}.{sequence}.{timestamp}.POST.{path}".encode()
        plaintext = json.dumps({"probe": {"value": 1}}).encode()
        ciphertext = AESGCM(request_key).encrypt(nonce, plaintext, aad)
        status, secured = post(
            server.server_port,
            path,
            {
                "session_id": handshake["session_id"],
                "sequence": sequence,
                "timestamp": timestamp,
                "nonce": base64.b64encode(nonce).decode(),
                "ciphertext": base64.b64encode(ciphertext).decode(),
            },
            origin,
        )
        assert status == 200
        response_aad = f"{handshake['session_id']}.{sequence}.{secured['timestamp']}.RESPONSE.{path}".encode()
        response_plaintext = AESGCM(response_key).decrypt(
            base64.b64decode(secured["nonce"]),
            base64.b64decode(secured["ciphertext"]),
            response_aad,
        )
        assert json.loads(response_plaintext) == {"result": {"accepted": True}}
        assert orchestrator.received == {"value": 1}
    finally:
        server.shutdown()
        server.server_close()


def test_http_origin_cannot_call_local_api():
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(Orchestrator(), LocalSessionManager()))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        status, value = post(server.server_port, "/v1/probes", {"probe": {}}, "https://hostile.test")
        assert status == 403
        assert value["error"] == "untrusted_origin"
    finally:
        server.shutdown()
        server.server_close()


def test_originless_plaintext_cannot_call_secure_api():
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(Orchestrator(), LocalSessionManager()))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        status, value = post(server.server_port, "/v1/probes", {"probe": {}}, "")
        assert status == 401
        assert value["error"] == "security_error"
    finally:
        server.shutdown()
        server.server_close()


def test_candidate_review_api_round_trip():
    orchestrator = Orchestrator()
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(orchestrator, LocalSessionManager()))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        status, value = get(server.server_port, "/v1/training/review?limit=23")
        assert status == 200
        assert value == {"items": [{"id": "candidate-review", "limit": 23}]}
        origin = f"http://127.0.0.1:{server.server_port}"
        status, value = post(
            server.server_port,
            "/v1/training/review",
            {"review_id": "candidate-review", "label": 0},
            origin,
        )
        assert status == 200
        assert value["result"]["saved"]
        assert orchestrator.received == ("candidate-review", 0)
    finally:
        server.shutdown()
        server.server_close()


def test_candidate_review_rejects_originless_write():
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(Orchestrator(), LocalSessionManager()))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        status, value = post(
            server.server_port,
            "/v1/training/review",
            {"review_id": "candidate-review", "label": 0},
            "",
        )
        assert status == 403
        assert value["error"] == "local_review_origin_required"
    finally:
        server.shutdown()
        server.server_close()
