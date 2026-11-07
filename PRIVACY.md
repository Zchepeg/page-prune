# Privacy

PagePrune processes candidate element metadata locally. The extension sends it to the loopback decision service at `127.0.0.1`.

The candidate payload is bounded to structural metadata, short visible text, dimensions, selected attributes and sanitized resource URLs. It excludes cookies, authorization headers, form values, full page HTML and URL query values. The service and extension establish an encrypted, paired local session before decision or labeling payloads are accepted.

Decision events, automatic captures, review queues, manual labels and pairing records remain in the local `data/` directory. They are ignored by Git and are not uploaded by this project. Model downloads requested by the user may contact the model host according to that host's policies.

Users can stop collection by stopping the local service or setting `training_capture.enabled` and `shadow_audit.enabled` to `false` in their local configuration. Local records can be removed by deleting the corresponding ignored files after stopping the service. Removing `data/security/paired_clients.json` resets extension pairing.

The extension uses the Chromium `search` permission only to submit text from its replacement new-tab page to the user's configured default search provider. Search text is not sent to the JEV local service.

## Screenshot review

Screenshot review is off each time the popup opens. The user can opt in for the current page, select a region and draw a mask. The browser grants temporary access to the active tab. The extension crops and masks the visible-tab screenshot in memory before opening a local extension preview. Screenshot pixels are never sent to the decision service or included in training data. The masked PNG is downloaded only if the user clicks Save.

## New tab

The extension saves new-tab folders and links in local storage. The sidebar fetches Baidu's public trending search page for mainland China using a fixed URL. The request omits cookies, saved links, browsing history and search text. Baidu receives ordinary connection information such as the browser's IP address.

Security or privacy concerns should be reported through the process in `SECURITY.md`.
