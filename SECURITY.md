# Security Policy

## Supported versions

Security fixes are applied to the latest source revision and the latest tagged release.

## Reporting

Do not publish exploit details, pairing secrets, captured browsing data or private model artifacts in a public issue. Use GitHub's private vulnerability reporting feature for the repository. If that feature is unavailable, open a minimal issue asking the maintainer to establish a private contact channel without including sensitive details.

Include the affected version, operating system, browser version, reproduction conditions and expected impact. Remove personal browsing content, tokens, cookies and local pairing records from logs before sharing them.

## Scope

The local decision service binds to `127.0.0.1`, uses one-time pairing and authenticated encrypted sessions, and fails open when unavailable. Reports involving bypass of pairing, replay protection, payload sanitization, extension origin validation or unintended non-loopback exposure are security relevant.
