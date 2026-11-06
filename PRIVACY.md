# Privacy

JEV Ad Blocker is designed for local processing. The browser extension sends candidate element metadata only to the loopback decision service at `127.0.0.1`.

The candidate payload is bounded to structural metadata, short visible text, dimensions, selected attributes and sanitized resource URLs. It excludes cookies, authorization headers, form values, full page HTML and URL query values. The service and extension establish an encrypted, paired local session before decision or labeling payloads are accepted.

Decision events, automatic captures, review queues, manual labels and pairing records remain in the local `data/` directory. They are ignored by Git and are not uploaded by this project. Model downloads requested by the user may contact the model host according to that host's policies.

Users can stop collection by stopping the local service or setting `training_capture.enabled` and `shadow_audit.enabled` to `false` in their local configuration. Local records can be removed by deleting the corresponding ignored files after stopping the service. Removing `data/security/paired_clients.json` resets extension pairing.

The extension uses the Chromium `search` permission only to submit text from its replacement new-tab page to the user's configured default search provider. Search text is not sent to the JEV local service.

## New tab

The extension saves new-tab folders and links in local storage. The sidebar requests public Chinese Wikipedia page-view rankings from Wikimedia without sending saved links, browsing history or search text. Wikimedia receives ordinary connection information such as the browser's IP address.

Security or privacy concerns should be reported through the process in `SECURITY.md`.

