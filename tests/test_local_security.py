import base64
import json
import os
import time

import pytest
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

from local_security import LocalSessionManager, SecurityError


def create_client(manager, client_id="client-one", origin="chrome-extension://extension-one"):
    private = ec.generate_private_key(ec.SECP256R1())
    public = private.public_key().public_bytes(
        serialization.Encoding.X962,
        serialization.PublicFormat.UncompressedPoint,
    )
    response = manager.create(
        client_id,
        base64.b64encode(public).decode("ascii"),
        origin,
        manager.pairing_code,
    )
    server_public = ec.EllipticCurvePublicKey.from_encoded_point(
        ec.SECP256R1(),
        base64.b64decode(response["server_public_key"]),
    )
    shared = private.exchange(ec.ECDH(), server_public)
    request_key = HKDF(
        algorithm=hashes.SHA256(),
        length=32,
        salt=response["session_id"].encode("ascii"),
        info=b"jev-local-session-v1/request",
    ).derive(shared)
    response_key = HKDF(
        algorithm=hashes.SHA256(),
        length=32,
        salt=response["session_id"].encode("ascii"),
        info=b"jev-local-session-v1/response",
    ).derive(shared)
    return response, request_key, response_key


def envelope(response, key, payload, sequence=1, timestamp=None, path="/v1/probes"):
    timestamp = timestamp or int(time.time() * 1000)
    nonce = os.urandom(12)
    aad = f"{response['session_id']}.{sequence}.{timestamp}.POST.{path}".encode()
    ciphertext = AESGCM(key).encrypt(nonce, json.dumps(payload).encode(), aad)
    return {
        "session_id": response["session_id"],
        "sequence": sequence,
        "timestamp": timestamp,
        "nonce": base64.b64encode(nonce).decode(),
        "ciphertext": base64.b64encode(ciphertext).decode(),
    }


def test_ecdh_aes_gcm_round_trip():
    manager = LocalSessionManager()
    response, key, _ = create_client(manager)
    value = {"probe": {"id": "one"}}
    secured = envelope(response, key, value)
    assert manager.decrypt(secured, "POST", "/v1/probes", "chrome-extension://extension-one") == value


def test_replay_and_tampering_are_rejected():
    manager = LocalSessionManager()
    response, key, _ = create_client(manager)
    secured = envelope(response, key, {"value": 1})
    manager.decrypt(secured, "POST", "/v1/probes", "chrome-extension://extension-one")
    with pytest.raises(SecurityError, match="replayed"):
        manager.decrypt(secured, "POST", "/v1/probes", "chrome-extension://extension-one")
    changed = envelope(response, key, {"value": 2}, sequence=2)
    changed["ciphertext"] = changed["ciphertext"][:-2] + "AA"
    with pytest.raises(SecurityError, match="authentication"):
        manager.decrypt(changed, "POST", "/v1/probes", "chrome-extension://extension-one")


def test_out_of_order_unique_sequences_are_accepted():
    manager = LocalSessionManager()
    response, key, _ = create_client(manager)
    second = envelope(response, key, {"sequence": 2}, sequence=2)
    first = envelope(response, key, {"sequence": 1}, sequence=1)
    assert manager.decrypt(second, "POST", "/v1/probes", "chrome-extension://extension-one")["sequence"] == 2
    assert manager.decrypt(first, "POST", "/v1/probes", "chrome-extension://extension-one")["sequence"] == 1


def test_stale_timestamp_and_origin_mismatch_are_rejected():
    manager = LocalSessionManager(clock_skew_seconds=1)
    response, key, _ = create_client(manager)
    stale = envelope(response, key, {}, timestamp=int(time.time() * 1000) - 5000)
    with pytest.raises(SecurityError, match="stale"):
        manager.decrypt(stale, "POST", "/v1/probes", "chrome-extension://extension-one")
    current = envelope(response, key, {}, sequence=2)
    with pytest.raises(SecurityError, match="origin"):
        manager.decrypt(current, "POST", "/v1/probes", "chrome-extension://other")


def test_first_extension_is_persistently_paired(tmp_path):
    registry = tmp_path / "paired_clients.json"
    first = LocalSessionManager(registry_path=registry)
    create_client(first)
    restarted = LocalSessionManager(registry_path=registry)
    create_client(restarted)
    with pytest.raises(SecurityError, match="not_paired"):
        create_client(restarted, "client-two", "chrome-extension://extension-two")
    assert restarted.stats()["paired_clients"] == 1


def test_pairing_requires_the_terminal_code(tmp_path):
    manager = LocalSessionManager(registry_path=tmp_path / "paired.json")
    private = ec.generate_private_key(ec.SECP256R1())
    public = private.public_key().public_bytes(serialization.Encoding.X962, serialization.PublicFormat.UncompressedPoint)
    with pytest.raises(SecurityError, match="invalid_pairing_code"):
        manager.create("client", base64.b64encode(public).decode(), "chrome-extension://extension", "000000")
    assert manager.stats()["pairing_required"]


def test_response_uses_separate_authenticated_key():
    manager = LocalSessionManager()
    response, _, response_key = create_client(manager)
    secured = manager.encrypt_response(response["session_id"], 7, "/v1/decisions", {"results": [1]})
    nonce = base64.b64decode(secured["nonce"])
    ciphertext = base64.b64decode(secured["ciphertext"])
    aad = f"{response['session_id']}.7.{secured['timestamp']}.RESPONSE./v1/decisions".encode()
    plaintext = AESGCM(response_key).decrypt(nonce, ciphertext, aad)
    assert json.loads(plaintext) == {"results": [1]}
