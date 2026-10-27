# Initial repository publication — 2026-10-27

Repository: MechFEA. Source extracted without changing original installations
or raw engineering measurements. Excludes private CAD, full simulation results,
virtual environments, user profiles and secrets. A local SHA-256 extraction
manifest is retained by the publication task. The public source is an initial
preview; domain-specific numerical acceptance remains separate from software tests.

Validation results will be appended after testing the exact extracted code.

## Verification

Fresh Python 3.12.11 environment recreated from the lockfile; package installed editable; dependency check passed. Regression suite: 287 passed, 15 skipped (external CAD/native evidence absent). Bundled acceptance contracts preserve original bytes and their hashes; nonzero friction remains blocked by its failed qualification. No full assembly solve or new load-rating claim.
