# -*- coding: utf-8 -*-
"""
Pytest global configuration.

Goals:
- Keep the whole suite fast and network-free by default.
- Route the statistical kenlm language model to the smallest pretrained model
  (people_chars_lm.klm, ~20MB) instead of the default 2.95GB file, so every
  Corrector/Detector test runs in seconds instead of downloading gigabytes.
- Skip tests that require downloading large deep-learning models
  (MacBERT / BART / ERNIE / etc.) unless the model is already cached locally
  or PYCORRECTOR_RUN_HEAVY=1 is set.
"""
import os

os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

# Work around a corrupted local environment: some installed packages have an
# incomplete .dist-info (missing METADATA), so importlib_metadata raises
# KeyError('Version') instead of PackageNotFoundError. huggingface_hub probes
# several packages at import time and only tolerates PackageNotFoundError, which
# crashes `import pycorrector` (and therefore every test) when importlib_metadata
# is loaded (e.g. by pytest). Map any such failure to PackageNotFoundError so the
# probe treats the package as not-installed. This only affects metadata queries,
# not the actual packages.
import importlib.metadata as _md

_orig_md_version = _md.version

def _safe_md_version(name):
    try:
        return _orig_md_version(name)
    except Exception:
        raise _md.PackageNotFoundError(name)

_md.version = _safe_md_version

import pytest

# Import the package first: transformers / huggingface_hub only import cleanly
# when HF_HUB_OFFLINE is unset (otherwise huggingface_hub's _runtime module
# crashes on a broken metadata entry). Importing pycorrector pulls in
# `import transformers`, which must succeed before we lock the network down.
import pycorrector.detector as det
from pycorrector.utils.get_file import get_file

# Now that transformers is imported, lock the network so a heavy test that
# slips past the skip guard cannot hang on a multi-GB download. Heavy tests are
# skipped unless the model is cached; PYCORRECTOR_RUN_HEAVY=1 leaves the
# network on so models can actually be fetched.
if os.environ.get('PYCORRECTOR_RUN_HEAVY') != '1':
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'

# --- smallest kenlm language model (char-level, ~20MB) ---
TINY_LM = 'people_chars_lm.klm'
TINY_LM_URL = det.Detector.pretrained_language_models[TINY_LM]
tiny_lm_path = get_file(
    TINY_LM, TINY_LM_URL, extract=True,
    cache_dir='~', cache_subdir=det.USER_DATA_DIR, verbose=0,
)

_orig_initialize = det.Detector._initialize_detector

def _fast_initialize(self):
    # override whatever model was requested with the tiny one so Corrector /
    # Detector never download the 2.95GB default language model
    self.language_model_path = tiny_lm_path
    _orig_initialize(self)

det.Detector._initialize_detector = _fast_initialize


HF_CACHE = os.path.expanduser('~/.cache/huggingface/hub')
MS_CACHE = os.path.expanduser('~/.cache/modelscope/hub/models')

def model_cached(repo_id):
    """Best-effort check whether a HF or ModelScope model is in the local cache."""
    if not repo_id:
        return False
    if os.path.isdir(os.path.join(HF_CACHE, 'models--' + repo_id.replace('/', '--'))):
        return True
    return os.path.isdir(os.path.join(MS_CACHE, repo_id))


def pytest_configure(config):
    config.addinivalue_line(
        'markers',
        'heavy: test needs a large downloaded model; skipped unless cached '
        'locally or PYCORRECTOR_RUN_HEAVY=1 is set.',
    )


def pytest_collection_modifyitems(config, items):
    run_heavy = os.environ.get('PYCORRECTOR_RUN_HEAVY') == '1'
    for item in items:
        marker = item.get_closest_marker('heavy')
        if marker is None:
            continue
        repo = marker.kwargs.get('repo') if marker.kwargs else None
        if run_heavy or model_cached(repo):
            continue
        item.add_marker(
            pytest.mark.skip(
                reason='heavy model %s not available; set PYCORRECTOR_RUN_HEAVY=1 or pre-download it' % repo
            )
        )
