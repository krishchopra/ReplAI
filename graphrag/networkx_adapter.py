from __future__ import annotations

from typing import Dict, Iterable, List

import networkx as nx


def build_memory_graph(conversations: Iterable[Dict]) -> nx.MultiDiGraph:
    """
    Construct a NetworkX graph from annotated conversation payloads.

    Useful for offline experiments or unit tests where Neo4j is unavailable.
    """
    graph = nx.MultiDiGraph()

    for conversation in conversations:
        convo_id = conversation.get("derived", {}).get("conversation_id")
        if not convo_id:
            continue
        graph.add_node(
            f"conversation::{convo_id}",
            node_type="conversation",
            **{
                key: value
                for key, value in conversation.get("derived", {}).items()
                if key != "participant_turn_counts"
            },
        )

        participants: List[str] = conversation.get("derived", {}).get("participants", [])
        for participant in participants:
            person_node = f"person::{participant}"
            graph.add_node(person_node, node_type="person", display_name=participant)
            graph.add_edge(person_node, f"conversation::{convo_id}", edge_type="participated_in")

        for msg in conversation.get("full_metadata_messages", []):
            msg_id = msg.get("message_id") or msg.get("derived", {}).get("synthetic_id")
            if not msg_id:
                continue
            message_node = f"message::{msg_id}"
            graph.add_node(
                message_node,
                node_type="message",
                content=msg.get("content"),
                timestamp=msg.get("timestamp"),
                author=msg.get("author"),
                **(msg.get("derived", {}) or {}),
            )
            graph.add_edge(f"conversation::{convo_id}", message_node, edge_type="has_message")

    return graph


__all__ = ["build_memory_graph"]

