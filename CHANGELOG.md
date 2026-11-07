# Changelog

All notable changes are documented here.

## 0.10.2 - Unreleased

- Add optional screenshot review from the toolbar popup. Select a region and draw a mask, then preview, save, or discard the result locally.
- Use the existing `activeTab` permission and crop and mask screenshots in memory. Screenshot review is off by default.
- Allow the crop and mask to be redrawn separately. Ignore zoom notifications that leave the zoom level unchanged.
- Refresh the new-tab mainland China hot list from Baidu every three hours and show the last successful fetch time.

## 0.10.1 - 2026-10-21

- Use the browser's configured default search provider on the replacement new-tab page.
- Add bounded incremental DOM scanning and offline retry backoff.
- Batch decision event writes and serialize GPU inference.
- Limit the Qwen decision runtime to an NF4 configuration targeting a 4 GB CUDA budget.
- Separate public development configuration from ignored local Qwen configuration.
- Add publishing, privacy, security and contribution safeguards.

## 0.10.0 - 2026-10-08

- Add the local Qwen pointer decision backend and decision protocol.
- Add encrypted extension pairing, manual labeling, review queues and shadow auditing.
- Add static network rules, cosmetic filtering and browser compatibility checks.
