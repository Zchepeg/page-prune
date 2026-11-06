from __future__ import annotations

import base64
import json
import secrets
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF


class SecurityError(ValueError):
    pass


@dataclass(slots=True)
class LocalSession:
    request_key: bytes
    response_key: bytes
    client_id: str
    origin: str
    expires_at: int
    highest_sequence: int = 0
    seen_sequences: set[int] = field(default_factory=set)


class LocalSessionManager:
    def __init__(
        self,
        lifetime_seconds: int = 3600,
        clock_skew_seconds: int = 120,
        registry_path: str | Path | None = None,
        pairing_lifetime_seconds: int = 600,
        pairing_attempt_limit: int = 10,
    ):
        self.lifetime_seconds = lifetime_seconds
        self.clock_skew_seconds = clock_skew_seconds
        self.registry_path = Path(registry_path) if registry_path else None
        self.pairing_lifetime_seconds = pairing_lifetime_seconds
        self.pairing_attempt_limit = pairing_attempt_limit
        self._sessions: dict[str, LocalSession] = {}
        self._clients = self._load_clients()
        self._pairing_code = f"{secrets.randbelow(1_000_000):06d}" if self.registry_path and not self._clients else ""
        self._pairing_expires_at = int(time.time()) + pairing_lifetime_seconds if self._pairing_code else 0
        self._pairing_attempts = 0
        self._lock = threading.Lock()

    def create(
        self,
        client_id: str,
        client_public_key: str,
        origin: str = "",
        pairing_code: str = "",
    ) -> dict[str, Any]:
        if not client_id or len(client_id) > 128:
            raise SecurityError("invalid_client_id")
        if origin and not origin.startswith("chrome-extension://"):
            raise SecurityError("untrusted_pairing_origin")
        try:
            encoded = base64.b64decode(client_public_key, validate=True)
            client_public = ec.EllipticCurvePublicKey.from_encoded_point(ec.SECP256R1(), encoded)
        except (ValueError, TypeError) as exc:
            raise SecurityError("invalid_client_public_key") from exc
        private = ec.generate_private_key(ec.SECP256R1())
        shared = private.exchange(ec.ECDH(), client_public)
        session_id = secrets.token_urlsafe(24)
        request_key = HKDF(
            algorithm=hashes.SHA256(),
            length=32,
            salt=session_id.encode("ascii"),
            info=b"jev-local-session-v1/request",
        ).derive(shared)
        response_key = HKDF(
            algorithm=hashes.SHA256(),
            length=32,
            salt=session_id.encode("ascii"),
            info=b"jev-local-session-v1/response",
        ).derive(shared)
        expires_at = int(time.time()) + self.lifetime_seconds
        public_bytes = private.public_key().public_bytes(
            serialization.Encoding.X962,
            serialization.PublicFormat.UncompressedPoint,
        )
        with self._lock:
            self._purge_locked()
            self._authorize_client_locked(client_id, origin, pairing_code)
            self._sessions[session_id] = LocalSession(request_key, response_key, client_id, origin, expires_at)
        return {
            "session_id": session_id,
            "server_public_key": base64.b64encode(public_bytes).decode("ascii"),
            "expires_at": expires_at,
            "protocol_version": "jev-local-session-v1",
        }

    def decrypt(self, envelope: dict[str, Any], method: str, path: str, origin: str = "") -> dict[str, Any]:
        try:
            session_id = str(envelope["session_id"])
            sequence = int(envelope["sequence"])
            timestamp = int(envelope["timestamp"])
            nonce = base64.b64decode(str(envelope["nonce"]), validate=True)
            ciphertext = base64.b64decode(str(envelope["ciphertext"]), validate=True)
        except (KeyError, TypeError, ValueError) as exc:
            raise SecurityError("invalid_secure_envelope") from exc
        now_ms = int(time.time() * 1000)
        if abs(now_ms - timestamp) > self.clock_skew_seconds * 1000:
            raise SecurityError("stale_secure_request")
        if len(nonce) != 12 or len(ciphertext) < 16 or sequence < 1:
            raise SecurityError("invalid_secure_envelope")
        aad = f"{session_id}.{sequence}.{timestamp}.{method.upper()}.{path}".encode("utf-8")
        with self._lock:
            self._purge_locked()
            session = self._sessions.get(session_id)
            if session is None:
                raise SecurityError("unknown_or_expired_session")
            if session.origin and origin != session.origin:
                raise SecurityError("session_origin_mismatch")
            if sequence in session.seen_sequences or sequence <= session.highest_sequence - 128:
                raise SecurityError("replayed_secure_request")
            try:
                plaintext = AESGCM(session.request_key).decrypt(nonce, ciphertext, aad)
                value = json.loads(plaintext)
            except Exception as exc:
                raise SecurityError("secure_request_authentication_failed") from exc
            if not isinstance(value, dict):
                raise SecurityError("secure_payload_must_be_an_object")
            session.seen_sequences.add(sequence)
            session.highest_sequence = max(session.highest_sequence, sequence)
            session.seen_sequences = {
                item for item in session.seen_sequences if item > session.highest_sequence - 128
            }
            return value

    def encrypt_response(
        self,
        session_id: str,
        request_sequence: int,
        path: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        timestamp = int(time.time() * 1000)
        nonce = secrets.token_bytes(12)
        aad = f"{session_id}.{request_sequence}.{timestamp}.RESPONSE.{path}".encode("utf-8")
        plaintext = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        with self._lock:
            self._purge_locked()
            session = self._sessions.get(session_id)
            if session is None:
                raise SecurityError("unknown_or_expired_session")
            ciphertext = AESGCM(session.response_key).encrypt(nonce, plaintext, aad)
        return {
            "session_id": session_id,
            "request_sequence": request_sequence,
            "timestamp": timestamp,
            "nonce": base64.b64encode(nonce).decode("ascii"),
            "ciphertext": base64.b64encode(ciphertext).decode("ascii"),
        }

    def stats(self) -> dict[str, int]:
        with self._lock:
            self._purge_locked()
            return {
                "active_sessions": len(self._sessions),
                "paired_clients": len(self._clients),
                "lifetime_seconds": self.lifetime_seconds,
                "pairing_required": bool(self._pairing_code),
            }

    @property
    def pairing_code(self) -> str:
        return self._pairing_code

    def _authorize_client_locked(self, client_id: str, origin: str, pairing_code: str) -> None:
        if not origin:
            return
        paired_origin = self._clients.get(client_id)
        if paired_origin is not None:
            if paired_origin != origin:
                raise SecurityError("paired_client_origin_mismatch")
            return
        if self._clients:
            raise SecurityError("client_not_paired")
        if self.registry_path is not None:
            self._pairing_attempts += 1
            if self._pairing_attempts > self.pairing_attempt_limit:
                raise SecurityError("pairing_attempt_limit_reached")
            if int(time.time()) >= self._pairing_expires_at:
                raise SecurityError("pairing_code_expired")
            if not secrets.compare_digest(pairing_code, self._pairing_code):
                raise SecurityError("invalid_pairing_code")
        self._clients[client_id] = origin
        self._save_clients_locked()
        self._pairing_code = ""
        self._pairing_expires_at = 0

    def _load_clients(self) -> dict[str, str]:
        if self.registry_path is None or not self.registry_path.exists():
            return {}
        try:
            value = json.loads(self.registry_path.read_text(encoding="utf-8"))
            clients = value.get("clients", {})
            return {str(key): str(origin) for key, origin in clients.items() if str(origin).startswith("chrome-extension://")}
        except (OSError, json.JSONDecodeError, TypeError):
            return {}

    def _save_clients_locked(self) -> None:
        if self.registry_path is None:
            return
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.registry_path.with_suffix(".tmp")
        temporary.write_text(json.dumps({"clients": self._clients}, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(self.registry_path)

    def _purge_locked(self) -> None:
        now = int(time.time())
        expired = [session_id for session_id, session in self._sessions.items() if session.expires_at <= now]
        for session_id in expired:
            self._sessions.pop(session_id, None)
