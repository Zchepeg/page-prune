# Contributing

Contributions should preserve the separation between deterministic rules, model judgment and browser execution. Model predictions must not become training labels without independent evidence or human review.

## Development setup

```powershell
python -m pip install -r requirements-dev.txt
python -m pytest
node --check browser/extension/background.js
node --check browser/extension/content.js
node --check browser/extension/newtab.js
node --check browser/extension/popup.js
```

Model work additionally requires `requirements-model.txt`. Keep checkpoints, downloaded source lists, browser profiles, captured samples and local configuration out of commits.

## Pull requests

- Add or update tests for behavioral changes.
- Document changes to privacy, permissions, protocols and data formats.
- Preserve fail-open behavior for service and model failures.
- Identify the source and license of every imported rule or dataset.
- Do not include generated browser metadata, signing keys, CRX files or user data.
- Avoid performance and accuracy claims that are not supported by a domain-disjoint independent evaluation.
