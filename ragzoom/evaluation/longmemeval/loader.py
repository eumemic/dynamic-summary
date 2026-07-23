"""Download LongMemEval data files from HuggingFace on demand.

The dataset lives at ``xiaowu0162/longmemeval-cleaned`` (MIT license). This
module resolves a logical variant name ("oracle"/"s"/"m") to its file on the
Hub and downloads it into a local cache, returning the path. Parsing is the
loader's caller's job (``types.parse_longmemeval_file``).

``huggingface_hub`` is imported lazily inside the function so that importing
this module — and therefore the whole harness — never requires the network or
the optional dependency. Tests exercise the resolution logic with the download
call mocked; nothing here is run against the real Hub in CI.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Protocol

logger = logging.getLogger(__name__)

HF_REPO_ID = "xiaowu0162/longmemeval-cleaned"

# Logical variant -> filename on the Hub. The cleaned release ships the S and M
# haystacks plus the oracle (gold-sessions-only) subset.
VARIANT_FILENAMES: dict[str, str] = {
    "oracle": "longmemeval_oracle.json",
    "s": "longmemeval_s_cleaned.json",
    "m": "longmemeval_m_cleaned.json",
}


class _Downloader(Protocol):
    """The subset of ``huggingface_hub.hf_hub_download`` we depend on.

    Declared as a Protocol (rather than ``Callable[..., str]``) so the kwargs
    are typed under mypy strict — ``Callable[..., str]`` smuggles in an
    implicit ``Any`` that ``disallow_any_explicit`` rejects.
    """

    def __call__(
        self,
        *,
        repo_id: str,
        filename: str,
        repo_type: str,
        local_dir: str | None,
    ) -> str: ...


def filename_for_variant(variant: str) -> str:
    """Resolve a logical variant name to its filename on the Hub.

    Raises ValueError for an unknown variant — there is no default, because
    silently downloading the wrong tier would invalidate the H/B measurement.
    """
    key = variant.lower()
    try:
        return VARIANT_FILENAMES[key]
    except KeyError:
        known = ", ".join(sorted(VARIANT_FILENAMES))
        raise ValueError(
            f"Unknown LongMemEval variant {variant!r}; expected one of: {known}"
        ) from None


def download_variant(
    variant: str,
    *,
    cache_dir: Path | None = None,
    downloader: _Downloader | None = None,
) -> Path:
    """Download a LongMemEval variant from HuggingFace and return its local path.

    Args:
        variant: "oracle", "s", or "m".
        cache_dir: Directory to download into. Defaults to the HF cache.
        downloader: Injected for testing — defaults to
            ``huggingface_hub.hf_hub_download`` (imported lazily so the network
            dependency is never required at import time).

    Only the requested file is fetched, never the full dataset.
    """
    filename = filename_for_variant(variant)

    if downloader is None:
        from huggingface_hub import hf_hub_download

        downloader = hf_hub_download

    logger.info(
        "Downloading LongMemEval variant %r (%s) from %s",
        variant,
        filename,
        HF_REPO_ID,
    )
    local_path = downloader(
        repo_id=HF_REPO_ID,
        filename=filename,
        repo_type="dataset",
        local_dir=str(cache_dir) if cache_dir is not None else None,
    )
    return Path(local_path)
