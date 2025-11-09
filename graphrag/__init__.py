"""Graph-based memory and retrieval system for Neo4j."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .config import EmbeddingConfig, Neo4jConnectionConfig
from .tagging import TagGenerator

if TYPE_CHECKING:  # pragma: no cover
    from .embedding import SentenceTransformerEmbedder
    from .neo4j_ingest import Neo4jGraphBuilder

__all__ = [
    "EmbeddingConfig",
    "Neo4jConnectionConfig",
    "SentenceTransformerEmbedder",
    "Neo4jGraphBuilder",
    "TagGenerator",
]


def __getattr__(name: str):
    if name == "SentenceTransformerEmbedder":
        from .embedding import SentenceTransformerEmbedder

        return SentenceTransformerEmbedder
    if name == "Neo4jGraphBuilder":
        from .neo4j_ingest import Neo4jGraphBuilder

        return Neo4jGraphBuilder
    raise AttributeError(f"module 'graphrag' has no attribute '{name}'")
