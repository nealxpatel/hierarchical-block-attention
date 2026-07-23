"""Pytest session setup, applied before any test module is collected.

Pins `HBA_DATA_DIR` (see src/hba/config.py -- the env-override for the `DATA`
path constant) to an isolated tmp directory, so the probe smoke tests
(tests/test_probes_smoke.py, which build a tiny random model with vocab_size=
300) can never accidentally resolve `DATA` to a real `data/` directory sitting
next to this checkout -- e.g. on a GPU training box where the repo and a real
corpus share a working directory. A real corpus's token ids run far past a
300-token toy vocab, which would raise an IndexError deep inside an embedding
lookup instead of the test failing (or skipping) cleanly.

This MUST be plain top-level module code, not a fixture: fixtures run per-test,
after pytest has already imported every test module -- and therefore
hba.config, which reads HBA_DATA_DIR and eagerly `os.makedirs`'s it -- at
collection time. By the time a fixture runs, that read already happened.
`setdefault` (not a hard overwrite) so an operator's explicit HBA_DATA_DIR
still wins if one is already set in the environment.
"""

import os
import tempfile

os.environ.setdefault("HBA_DATA_DIR", tempfile.mkdtemp(prefix="hba-test-data-"))
