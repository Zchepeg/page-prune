# Local security protocol

`jev-local-session-v1` protects candidate, manual-label and SSA probe requests between the browser extension background worker and `127.0.0.1`.

## Session establishment

The extension creates an ephemeral P-256 ECDH key pair and sends its public key with a random installation client ID to `/v1/security/session`. The service creates a second ephemeral key pair. Both sides derive 256 bits with ECDH and use HKDF-SHA256 with the session ID as salt and `jev-local-session-v1` as context to obtain an AES-256-GCM key.

When no client is paired, the service prints a six-digit one-time code. The user enters it in the extension's independent options page. The code expires after ten minutes, accepts at most ten attempts and is cleared after success. The approved Chrome extension origin and client ID are persisted in `data/security/paired_clients.json`. Later sessions must use the same pair. The file is local-only and excluded from Git. Removing it resets pairing.

Sessions live for one hour by default and exist only in service memory. Extension and service restarts automatically establish a new ephemeral session.

## Request protection

HKDF derives separate request and response keys. Every protected request contains a session ID, sequence number, Unix timestamp in milliseconds, 96-bit random nonce and AES-GCM ciphertext. Additional authenticated data binds the envelope to the session, sequence, timestamp, HTTP method and API path.

Successful protected responses are encrypted with the response key and bind their authentication data to the original request sequence and API path. The extension refuses responses from another session, responses attached to a different request sequence, stale responses and invalid authentication tags.

The service rejects expired sessions, timestamps outside the configured window, reused sequence numbers, old sequence windows, changed origins, invalid authentication tags and malformed payloads. Up to 128 unique out-of-order requests are accepted so concurrent extension requests do not fail.

Browser requests from ordinary HTTP or HTTPS origins are rejected. Candidate decisions, manual labels and SSA probes always require an authenticated encrypted session, including when no Origin header is present. Health and aggregate status endpoints remain readable for local diagnostics. Review writes are accepted only from the same-origin local review page.

## Threat boundary

This protocol protects against ordinary webpages calling or reading the local service, passive inspection of extension-to-service payloads and replay or modification of captured requests and responses. It does not protect a machine already controlled by malware, an attacker who can modify the extension or Python process, or a hostile local administrator. The pairing code authenticates user presence for the initial local binding but is not an externally certified service identity.
