from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence

from neo4j import GraphDatabase, Session

from .config import Neo4jConnectionConfig
from .embedding import SentenceTransformerEmbedder


@dataclass
class RetrievalParameters:
    top_k: int = 6
    expand_hops: int = 1
    include_neighbors: bool = True
    score_threshold: Optional[float] = None


class GraphRAGRetriever:
    """
    Retrieve contextual memories via vector similarity and graph expansion.
    """

    def __init__(
        self,
        connection: Neo4jConnectionConfig,
        embedder: SentenceTransformerEmbedder,
        message_index_name: str = "message_embedding_index",
    ) -> None:
        self.connection = connection
        self.embedder = embedder
        self.message_index_name = message_index_name
        self._driver = GraphDatabase.driver(
            connection.uri,
            auth=(connection.username, connection.password),
            encrypted=connection.encrypted,
        )
        self._database = connection.database

    def close(self) -> None:
        self._driver.close()

    def retrieve(
        self,
        query_text: str,
        participants: Optional[Sequence[str]] = None,
        params: Optional[RetrievalParameters] = None,
    ) -> Dict[str, Any]:
        if not query_text.strip():
            return {"short_summary": "", "conversations": [], "quotes": [], "metadata": {"error": "empty_query"}}

        parameters = params or RetrievalParameters()
        embedding = self.embedder.embed_single(query_text)
        with self._session() as session:
            hits = self._vector_search(
                session,
                embedding,
                participants=list(participants) if participants else None,
                params=parameters,
            )
            if parameters.include_neighbors and parameters.expand_hops > 1:
                self._expand_neighbors(session, hits, parameters.expand_hops)
        grouped = self._group_hits_by_conversation(hits)
        conversations_payload = []
        global_quotes: List[Dict[str, Any]] = []
        for conversation_id, items in grouped.items():
            conversations_payload.append(self._build_conversation_payload(conversation_id, items))
            for item in items:
                quote = {
                    "conversation_id": conversation_id,
                    "message_id": item["message"].get("message_id"),
                    "content": item["message"].get("content"),
                    "author": item["message"].get("author"),
                    "timestamp": item["message"].get("timestamp"),
                    "score": item["score"],
                }
                if quote["content"]:
                    global_quotes.append(quote)

        conversations_payload.sort(key=lambda entry: entry.get("score", 1.0))
        global_quotes.sort(key=lambda quote: quote.get("score", 1.0))
        short_summary = self._compose_summary(conversations_payload)
        metadata = {
            "query_text": query_text,
            "hit_count": len(hits),
            "participants_filter": list(participants) if participants else None,
            "retrieval": {
                "top_k": parameters.top_k,
                "expand_hops": parameters.expand_hops,
                "score_threshold": parameters.score_threshold,
            },
            "embedding_model": self.embedder.config.model_name,
        }
        return {
            "short_summary": short_summary,
            "conversations": conversations_payload,
            "quotes": global_quotes[:10],
            "metadata": metadata,
        }

    # Internal helpers ---------------------------------------------------------
    def _vector_search(
        self,
        session: Session,
        embedding: List[float],
        participants: Optional[List[str]],
        params: RetrievalParameters,
    ) -> List[Dict[str, Any]]:
        result = session.run(
            """
            CALL db.index.vector.queryNodes($index_name, $top_k, $embedding)
            YIELD node, score
            MATCH (c:Conversation {conversation_id: node.conversation_id})
            WHERE $participants IS NULL OR any(p IN c.participants WHERE p IN $participants)
            OPTIONAL MATCH (c)-[:HAS_TAG]->(tag:Tag)
            OPTIONAL MATCH (node)-[:FOLLOWS]->(next:Message)
            OPTIONAL MATCH (prev:Message)-[:FOLLOWS]->(node)
            OPTIONAL MATCH (node)<-[:SENT]-(sender:Person)
            WITH node, score, c, collect(DISTINCT tag) AS tags,
                 collect(DISTINCT next) AS next_messages,
                 collect(DISTINCT prev) AS prev_messages,
                 collect(DISTINCT sender) AS senders
            RETURN node, score, c, tags, next_messages, prev_messages, senders
            ORDER BY score ASC
            """,
            index_name=self.message_index_name,
            top_k=params.top_k,
            embedding=embedding,
            participants=participants,
        )
        hits: List[Dict[str, Any]] = []
        for record in result:
            score = float(record["score"])
            if params.score_threshold is not None and score > params.score_threshold:
                continue
            message = self._clean_node(record["node"])
            conversation = self._clean_node(record["c"])
            tags = [self._clean_node(tag) for tag in record["tags"] if tag]
            next_messages = [self._clean_node(msg) for msg in record["next_messages"] if msg]
            prev_messages = [self._clean_node(msg) for msg in record["prev_messages"] if msg]
            senders = [self._clean_node(sender) for sender in record["senders"] if sender]
            hits.append(
                {
                    "score": score,
                    "message": message,
                    "conversation": conversation,
                    "tags": tags,
                    "neighbors": {
                        "previous": prev_messages,
                        "next": next_messages,
                    },
                    "senders": senders,
                }
            )
        return hits

    def _expand_neighbors(
        self,
        session: Session,
        hits: Sequence[Dict[str, Any]],
        hops: int,
    ) -> None:
        if hops <= 1:
            return
        query = f"""
        MATCH (m:Message {{message_id: $message_id}})
        OPTIONAL MATCH (m)-[:FOLLOWS*1..{hops}]->(forward:Message)
        OPTIONAL MATCH (back:Message)-[:FOLLOWS*1..{hops}]->(m)
        RETURN collect(DISTINCT forward) AS forwards,
               collect(DISTINCT back) AS backwards
        """
        for hit in hits:
            message_id = hit["message"].get("message_id")
            if not message_id:
                continue
            record = session.run(query, message_id=message_id).single()
            if not record:
                continue
            forwards = [self._clean_node(node) for node in record["forwards"] if node]
            backwards = [self._clean_node(node) for node in record["backwards"] if node]
            neighbor_block = hit.setdefault("neighbors", {"previous": [], "next": []})
            existing_next_ids = {msg.get("message_id") for msg in neighbor_block.get("next", [])}
            existing_prev_ids = {msg.get("message_id") for msg in neighbor_block.get("previous", [])}

            for node in forwards:
                node_id = node.get("message_id")
                if node_id and node_id not in existing_next_ids:
                    neighbor_block.setdefault("next", []).append(node)
                    existing_next_ids.add(node_id)

            for node in backwards:
                node_id = node.get("message_id")
                if node_id and node_id not in existing_prev_ids:
                    neighbor_block.setdefault("previous", []).append(node)
                    existing_prev_ids.add(node_id)

    def _group_hits_by_conversation(self, hits: Iterable[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        grouped: Dict[str, List[Dict[str, Any]]] = {}
        for hit in hits:
            conversation_id = hit["conversation"].get("conversation_id")
            if not conversation_id:
                continue
            grouped.setdefault(conversation_id, []).append(hit)
        return grouped

    def _build_conversation_payload(self, conversation_id: str, items: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not items:
            return {"conversation_id": conversation_id, "quotes": []}
        primary = min(items, key=lambda item: item["score"])
        participants = primary["conversation"].get("participants", [])
        tags = primary.get("tags", [])
        quotes = []
        neighbor_quotes = []
        for item in items:
            message = item["message"]
            if message.get("content"):
                quotes.append(
                    {
                        "message_id": message.get("message_id"),
                        "content": message.get("content"),
                        "author": message.get("author"),
                        "timestamp": message.get("timestamp"),
                        "score": item["score"],
                    }
                )
            for direction in ("previous", "next"):
                for neighbor in item["neighbors"].get(direction, []):
                    if neighbor.get("content"):
                        neighbor_quotes.append(
                            {
                                "message_id": neighbor.get("message_id"),
                                "content": neighbor.get("content"),
                                "author": neighbor.get("author"),
                                "timestamp": neighbor.get("timestamp"),
                                "relation": direction,
                            }
                        )
        quotes.sort(key=lambda quote: quote.get("score", 1.0))
        payload = {
            "conversation_id": conversation_id,
            "score": primary["score"],
            "participants": participants,
            "tags": tags,
            "conversation": primary["conversation"],
            "quotes": quotes[:5],
            "neighbor_quotes": neighbor_quotes[:5],
        }
        return payload

    def _compose_summary(self, conversations_payload: List[Dict[str, Any]]) -> str:
        if not conversations_payload:
            return ""
        sentences = []
        for entry in conversations_payload[:3]:
            participants = entry.get("participants") or []
            participant_label = ", ".join(participants) if participants else "someone"
            top_quote = entry.get("quotes", [])
            highlight = ""
            if top_quote:
                highlight = top_quote[0]["content"] or ""
                highlight = highlight.strip().replace("\n", " ")
                if len(highlight) > 160:
                    highlight = f"{highlight[:157]}..."
            sentences.append(f"Recall with {participant_label}: {highlight}")
        return " ".join(sentences)

    def _clean_node(self, node: Any) -> Dict[str, Any]:
        if not node:
            return {}
        if isinstance(node, dict):
            return dict(node)
        # neo4j.Node -> mapping
        data = dict(node.items()) if hasattr(node, "items") else dict(node)
        if hasattr(node, "id"):
            data.setdefault("_internal_id", node.id)
        return data

    def _session(self) -> Session:
        return self._driver.session(database=self._database) if self._database else self._driver.session()


__all__ = ["GraphRAGRetriever", "RetrievalParameters"]

