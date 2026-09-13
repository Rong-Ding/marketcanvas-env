# Fresh-install verification

Verified locally on 2026-09-13 with macOS arm64 and Python 3.12.14.

A new virtual environment was created with no system site packages. All 39 pinned
packages were downloaded from public PyPI using the lock file and cache disabled.
The audit then copied the source into a new temporary directory, excluding .venv,
.git, private/, generated caches and prior reproduction outputs. It used the new
interpreter for every command below, not the development environment.

| Check | Result |
|---|---|
| Fresh locked dependency installation | Passed |
| Full test suite on clean copy | 74 passed; 1 private-data integration check skipped |
| Demo from blank through repair and submission | Passed; terminal reward +1.0 |
| Replay of the demo trajectory | Passed; six actions verified |
| Original human-pilot analysis from released labels | Reproduced |
| Embedded primary and consistency analyses | Reproduced |
| Fitted reward and grouped cross-validation | Reproduced: 10/24 and 8/24 |
| Submission PDFs | Visually checked; main 2 pages, companion 3 pages |

The full tests include actual MCP subprocess initialization, tool discovery,
Python/MCP action parity and process isolation; Gymnasium's environment checker;
invalid-action and terminal handling; reward perturbations; and annotation masking.
The skipped test depends on the private historical calibration output. Synthetic
calibration tests still run, and the public-data reproduction command independently
refits and checks the reported calibration results.

A sandbox initially prevented a local HTTP test from opening its socket. The
fresh-copy audit was rerun with local-server permission and passed. This was an
environment restriction, not a hidden failing test. The supported installation
route exercised here was the README's uv alternative with the pinned lock.
Windows/Linux installs and a real LLM desktop-client conversation have not been
tested. No training, distributed rollout benchmark or GitHub publication is claimed.
