from __future__ import annotations

import os
from functools import lru_cache
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator

from .config import EmbeddingConfig, Neo4jConnectionConfig
from .embedding import SentenceTransformerEmbedder
from .retrieval import GraphRAGRetriever, RetrievalParameters


class MemoryRequest(BaseModel):
    query: str = Field(..., description="The latest dialogue turn or query text")
    participants: Optional[List[str]] = Field(
        default=None, description="Optional participant filter for retrieval"
    )
    top_k: Optional[int] = Field(default=6, ge=1, le=20)
    expand_hops: Optional[int] = Field(default=2, ge=1, le=5)
    include_neighbors: Optional[bool] = Field(default=True)
    score_threshold: Optional[float] = Field(default=None, ge=0.0)

    @validator("query")
    def query_must_not_be_empty(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("query must not be empty")
        return value


class MemoryResponse(BaseModel):
    short_summary: str
    conversations: List[Dict[str, Any]]
    quotes: List[Dict[str, Any]]
    metadata: Dict[str, Any]


app = FastAPI(title="GraphRAG Memory Service", version="0.1.0")


@lru_cache(maxsize=1)
def _build_retriever() -> GraphRAGRetriever:
    connection = Neo4jConnectionConfig(
        uri=os.environ["NEO4J_URI"],
        username=os.environ["NEO4J_USERNAME"],
        password=os.environ["NEO4J_PASSWORD"],
        database=os.getenv("NEO4J_DATABASE"),
    )
    embed_config = EmbeddingConfig(
        model_name=os.getenv("SENTENCE_TRANSFORMER_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
        device=os.getenv("SENTENCE_TRANSFORMER_DEVICE"),
        normalize_embeddings=os.getenv("SENTENCE_TRANSFORMER_NORMALIZE", "1") not in ("0", "false", "False"),
    )
    embedder = SentenceTransformerEmbedder(embed_config)
    retriever = GraphRAGRetriever(
        connection=connection,
        embedder=embedder,
        message_index_name=os.getenv("MESSAGE_VECTOR_INDEX", "message_embedding_index"),
    )
    return retriever


@app.on_event("shutdown")
async def shutdown_event() -> None:
    if _build_retriever.cache_info().currsize:
        retriever = _build_retriever()
        retriever.close()
        _build_retriever.cache_clear()


@app.get("/healthz")
async def healthcheck() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/memory", response_model=MemoryResponse)
async def fetch_memory(request: MemoryRequest) -> MemoryResponse:
    try:
        retriever = _build_retriever()
    except KeyError as exc:
        missing = exc.args[0]
        raise HTTPException(status_code=500, detail=f"Missing environment variable: {missing}") from exc

    params = RetrievalParameters(
        top_k=request.top_k or 6,
        expand_hops=request.expand_hops or 2,
        include_neighbors=request.include_neighbors if request.include_neighbors is not None else True,
        score_threshold=request.score_threshold,
    )
    result = retriever.retrieve(
        query_text=request.query,
        participants=request.participants,
        params=params,
    )
    return MemoryResponse(**result)


def create_app() -> FastAPI:
    return app


__all__ = ["create_app", "app", "MemoryRequest", "MemoryResponse"]

