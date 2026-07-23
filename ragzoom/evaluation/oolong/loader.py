"""Download Oolong (oolong-real) data from HuggingFace on demand.

The dataset lives at ``oolongbench/oolong-real``. It is published with two
configs — ``dnd`` (the full evaluation set) and ``toy_dnd`` (a small smoke-test
subset) — each with ``validation`` and ``test`` splits, laid out as
``<config>/<split>.jsonl``. A single config+split file is fetched directly from
the Hub.

``huggingface_hub`` is imported lazily inside the function so importing this
module — and therefore the whole harness — never requires the network or the
optional dependency. Tests exercise the resolution logic with the download call
injected; nothing here is run against the real Hub in CI.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Protocol

from ragzoom.evaluation.oolong.types import OolongQuestion, parse_oolong_records

logger = logging.getLogger(__name__)

HF_REPO_ID = "oolongbench/oolong-real"

CONFIG_DND = "dnd"
CONFIG_TOY_DND = "toy_dnd"

_CONFIGS = frozenset({CONFIG_DND, CONFIG_TOY_DND})
_SPLITS = frozenset({"validation", "test"})


class _Downloader(Protocol):
    """The subset of ``huggingface_hub.hf_hub_download`` we depend on.

    Declared as a Protocol (rather than ``Callable[..., str]``) so the kwargs are
    typed under mypy strict — ``Callable[..., str]`` smuggles in an implicit
    ``Any`` that ``disallow_any_explicit`` rejects.
    """

    def __call__(
        self,
        *,
        repo_id: str,
        filename: str,
        repo_type: str,
        local_dir: str | None,
    ) -> str: ...


def filename_for_config(config: str, split: str) -> str:
    """Resolve a (config, split) pair to its JSONL path in the Hub repo.

    The dataset is laid out as ``<config>/<split>.jsonl``. We fetch that file
    directly. Unknown config or split fails hard — silently fetching the wrong
    subset would invalidate the tier the run claims to measure.
    """
    if config not in _CONFIGS:
        known = ", ".join(sorted(_CONFIGS))
        raise ValueError(f"Unknown Oolong config {config!r}; expected one of: {known}")
    if split not in _SPLITS:
        known = ", ".join(sorted(_SPLITS))
        raise ValueError(f"Unknown Oolong split {split!r}; expected one of: {known}")
    return f"{config}/{split}.jsonl"


def download_oolong_real(
    *,
    config: str = CONFIG_DND,
    split: str = "test",
    cache_dir: Path | None = None,
    downloader: _Downloader | None = None,
) -> Path:
    """Download one Oolong-real config+split JSONL file and return its path.

    Args:
        config: ``"dnd"`` (full) or ``"toy_dnd"`` (smoke test).
        split: ``"validation"`` or ``"test"``.
        cache_dir: Directory to download into. Defaults to the HF cache.
        downloader: Injected for testing — defaults to
            ``huggingface_hub.hf_hub_download`` (imported lazily so the network
            dependency is never required at import time).

    Only the requested config+split file is fetched, never the full dataset.
    """
    filename = filename_for_config(config, split)

    if downloader is None:
        from huggingface_hub import hf_hub_download

        downloader = hf_hub_download

    logger.info("Downloading Oolong-real (%s/%s) from %s", config, split, HF_REPO_ID)
    local_path = downloader(
        repo_id=HF_REPO_ID,
        filename=filename,
        repo_type="dataset",
        local_dir=str(cache_dir) if cache_dir is not None else None,
    )
    return Path(local_path)


def load_oolong_jsonl(path: Path) -> list[OolongQuestion]:
    """Load an Oolong-real JSONL file into typed question objects.

    The dataset is published as newline-delimited JSON (one record per line).
    Each record is parsed by ``parse_oolong_records``, which fails hard on an
    unrecognized ``question_type`` rather than silently dropping the row.
    """
    import json

    records: list[dict[str, object]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                records.append(json.loads(stripped))
    logger.info("Loaded %d Oolong records from %s", len(records), path)
    return parse_oolong_records(records)
