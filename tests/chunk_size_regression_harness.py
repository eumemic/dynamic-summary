"""Shared helpers for chunk size regression tests."""

from __future__ import annotations

from typing import TypedDict, cast

import numpy as np
from numpy.typing import NDArray

from ragzoom.config import IndexConfig
from ragzoom.document_store import DocumentStore
from ragzoom.splitter import TextSplitter
from tests.conftest import IndexerRuntimeHarness

__all__ = [
    "SPLITTER_SAMPLE_PARAGRAPH",
    "INDEX_SAMPLE_PARAGRAPH",
    "NodePayload",
    "configure_runtime",
    "add_nodes",
    "seed_manual_chunk_tree",
    "build_test_document",
]

SPLITTER_SAMPLE_PARAGRAPH = """
Call me Ishmael. Some years ago—never mind how long precisely—having
little or no money in my purse, and nothing particular to interest me
on shore, I thought I would sail about a little and see the watery part
of the world. It is a way I have of driving off the spleen and
regulating the circulation. Whenever I find myself growing grim about
the mouth; whenever it is a damp, drizzly November in my soul; whenever
I find myself involuntarily pausing before coffin warehouses, and
bringing up the rear of every funeral I meet; and especially whenever
my hypos get such an upper hand of me, that it requires a strong moral
principle to prevent me from deliberately stepping into the street, and
methodically knocking people's hats off—then, I account it high time to
get to sea as soon as I can. This is my substitute for pistol and ball.
With a philosophical flourish Cato throws himself upon his sword; I
quietly take to the ship. There is nothing surprising in this. If they
but knew it, almost all men in their degree, some time or other,
cherish very nearly the same feelings towards the ocean with me.

There now is your insular city of the Manhattoes, belted round by
wharves as Indian isles by coral reefs—commerce surrounds it with her
surf. Right and left, the streets take you waterward. Its extreme
downtown is the battery, where that noble mole is washed by waves, and
cooled by breezes, which a few hours previous were out of sight of
land. Look at the crowds of water-gazers there.
"""

INDEX_SAMPLE_PARAGRAPH = """Once upon a time in a distant kingdom, there lived a wise old king who ruled with fairness and justice.
The kingdom prospered under his reign, with fertile lands yielding abundant harvests and trade routes bringing wealth from far and wide.
The people were happy and content, living in peace and harmony. Children played in the streets without fear, and merchants conducted their business honestly.
However, not all was perfect in this idyllic realm. In the shadows lurked those who envied the king's success and plotted against him.
They whispered in dark corners and made secret alliances, waiting for the right moment to strike.
The king, aware of these threats, surrounded himself with loyal advisors and brave knights who would defend the kingdom with their lives.
"""


class NodePayload(TypedDict, total=False):
    """Minimal representation of a node for manual store population."""

    node_id: str
    text: str
    embedding: list[float] | NDArray[np.float64]
    span_start: int
    span_end: int
    document_id: str
    token_count: int
    height: int
    left_child_id: str | None
    right_child_id: str | None


def configure_runtime(harness: IndexerRuntimeHarness, config: IndexConfig) -> None:
    """Propagate a fresh IndexConfig through the runtime harness."""

    harness.runtime._index_config = config
    harness.runtime._append_executor._config = config
    harness.runtime._append_executor._splitter = TextSplitter(config)
    harness.worker_coordinator._index_config = config
    harness.llm_service.config = config
    harness.telemetry_manager._index_config = config


def add_nodes(store: DocumentStore, nodes: list[NodePayload]) -> None:
    """Insert typed node payloads into the document store."""

    store.nodes.add_batch(
        cast(
            list[
                dict[
                    str,
                    str | int | float | bool | list[float] | NDArray[np.float64] | None,
                ]
            ],
            nodes,
        )
    )


def seed_manual_chunk_tree(document_id: str, store: DocumentStore) -> None:
    """Populate a document store with deterministic chunk nodes."""

    nodes: list[NodePayload] = [
        NodePayload(
            node_id="small1",
            text="Short text.",
            embedding=np.zeros(1536, dtype=np.float64),
            span_start=0,
            span_end=11,
            document_id=document_id,
            token_count=2,
            height=0,
        ),
        NodePayload(
            node_id="small2",
            text="Another short text.",
            embedding=np.zeros(1536, dtype=np.float64),
            span_start=12,
            span_end=31,
            document_id=document_id,
            token_count=3,
            height=0,
        ),
        NodePayload(
            node_id="target1",
            text=" ".join(["word"] * 200),
            embedding=np.zeros(1536, dtype=np.float64),
            span_start=32,
            span_end=1232,
            document_id=document_id,
            token_count=200,
            height=0,
        ),
        NodePayload(
            node_id="target2",
            text=" ".join(["token"] * 195),
            embedding=np.zeros(1536, dtype=np.float64),
            span_start=1233,
            span_end=2408,
            document_id=document_id,
            token_count=195,
            height=0,
        ),
        NodePayload(
            node_id="left_parent",
            text="Summary of small chunks.",
            embedding=np.zeros(1536, dtype=np.float64),
            span_start=0,
            span_end=31,
            document_id=document_id,
            height=1,
            left_child_id="small1",
            right_child_id="small2",
            token_count=4,
        ),
        NodePayload(
            node_id="right_parent",
            text="Summary of target-size chunks.",
            embedding=np.zeros(1536, dtype=np.float64),
            span_start=32,
            span_end=2408,
            document_id=document_id,
            height=1,
            left_child_id="target1",
            right_child_id="target2",
            token_count=5,
        ),
        NodePayload(
            node_id="root",
            text="Overall summary of all chunks.",
            embedding=np.zeros(1536, dtype=np.float64),
            span_start=0,
            span_end=2408,
            document_id=document_id,
            height=2,
            left_child_id="left_parent",
            right_child_id="right_parent",
            token_count=6,
        ),
    ]

    add_nodes(store, nodes)
    store.nodes.update_parent_references_batch(
        [
            ("small1", "left_parent"),
            ("small2", "left_parent"),
            ("target1", "right_parent"),
            ("target2", "right_parent"),
            ("left_parent", "root"),
            ("right_parent", "root"),
        ]
    )


def build_test_document(multiplier: int) -> str:
    """Return a long-form document for indexing tests."""

    return INDEX_SAMPLE_PARAGRAPH * multiplier
