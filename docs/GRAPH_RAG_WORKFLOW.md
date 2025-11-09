## GraphRAG Workflow

This document outlines the end-to-end process for powering structured long-term memory for the Beeper agent.

---

### Phase 1 — Harden & Enrich the Corpus
- `python -m graphrag.corpus --input-dir …` wraps the existing merge + partition utilities.
- Derived metadata (see `graphrag/tagging.py`) adds:
  - deterministic conversation & message IDs
  - participant turn counts and ratios (`assistant_turn_ratio`, `user_turn_ratio`)
  - time-of-day buckets (`morning`, `afternoon`, `night`)
  - lexical signals on each message (`word_count`, `contains_question`, `contains_emoji`, `gap_seconds`)
- Output JSON lives wherever you pass via `--annotated-out` and becomes the hand-off to Neo4j ingestion.

### Phase 2 — Graph Data Modeling & Neo4j Ingestion
- Run `python -m graphrag.neo4j_ingest` against the annotated JSON.
- Schema overview:
  - `(:Person {person_id, display_name, total_messages})`
  - `(:Conversation {conversation_id, source, chat_type, …})`
  - `(:Message {message_id, content, timestamp, role, embedding, …})`
  - `(:Tag {tag_id, name, category})`
- Relationships:
  - `(:Person)-[:PARTICIPATED_IN]->(:Conversation)`
  - `(:Person)-[:SENT]->(:Message)`
  - `(:Conversation)-[:HAS_MESSAGE]->(:Message)`
  - `(:Conversation)-[:HAS_TAG]->(:Tag)`
  - `(:Message)-[:FOLLOWS]->(:Message)` (time ordering)
- Options:
  - `--ensure-constraints` creates unique constraints and vector index scaffolding.
  - `--embedding-model` (optional) stores embeddings inline with messages & conversations.

### Phase 3 — Embed & Retrieve
- Embedding handled by `SentenceTransformerEmbedder` (defaults to `sentence-transformers/all-MiniLM-L6-v2`).
- `GraphRAGRetriever.retrieve(...)` steps:
  1. vector similarity search via Neo4j’s HNSW index (`CALL db.index.vector.queryNodes`)
  2. neighborhood expansion through `[:FOLLOWS]` and participant/tag metadata
  3. builds a memory bundle with:
     - `short_summary`
     - per-conversation highlights
     - supporting quotes & neighbor quotes
- Optional offline experiments:
  - `graphrag.networkx_adapter.build_memory_graph` converts the annotated JSON into a NetworkX graph for prototyping.

### Phase 4 — Memory Service + Agent Augmentation
- Start the FastAPI service:
  ```bash
  export NEO4J_URI=bolt://localhost:7687
  export NEO4J_USERNAME=neo4j
  export NEO4J_PASSWORD=<password>
  # optional
  export MESSAGE_VECTOR_INDEX=message_embedding_index
  export SENTENCE_TRANSFORMER_MODEL=sentence-transformers/all-MiniLM-L6-v2
  export SENTENCE_TRANSFORMER_DEVICE=cpu  # or cuda

  uvicorn graphrag.service:create_app --host 0.0.0.0 --port 8001
  ```
- Request payload:
  ```json
  {
    "query": "latest user message text",
    "participants": ["Alice", "Bob"],
    "top_k": 6,
    "expand_hops": 2
  }
  ```
- Response contains the memory bundle consumed by `app/memoryClient.ts`.
- `app/agent.ts` injects a “Long-term memory recall” system message before hitting OpenAI so GPT sees curated historical context alongside the last ~20 live turns.

### Phase 5 — Quality, Feedback, Operationalization
- Validation ideas:
  - Sample `/memory` responses against ground truth chats.
  - Monitor retrieval scores and adjust `top_k`, `expand_hops`, or embedding model.
  - Use manual tagging tasks to seed additional `(:Tag)` nodes or filters (e.g., tone, formality).
- Operations:
  - Re-run `graphrag.corpus` + `graphrag.neo4j_ingest` when new exports land.
  - Monitor Neo4j health: vector index size, query latency, constraint violations.
  - Keep the memory service alongside the agent (or deploy separately) and set `MEMORY_SERVICE_URL` for the agent runtime.

---

### Environment Variables Summary

| Variable | Purpose | Default |
| --- | --- | --- |
| `NEO4J_URI` | Bolt endpoint for Neo4j | — |
| `NEO4J_USERNAME` | Neo4j username | — |
| `NEO4J_PASSWORD` | Neo4j password | — |
| `NEO4J_DATABASE` | Optional database if not using default | `None` |
| `MESSAGE_VECTOR_INDEX` | Neo4j vector index name | `message_embedding_index` |
| `SENTENCE_TRANSFORMER_MODEL` | Embedding model name | `sentence-transformers/all-MiniLM-L6-v2` |
| `SENTENCE_TRANSFORMER_DEVICE` | Device override (`cpu`, `cuda`) | Auto |
| `SENTENCE_TRANSFORMER_NORMALIZE` | Normalize embeddings flag (`1`/`0`) | `1` |
| `MEMORY_SERVICE_URL` | Agent → memory service URL | `http://127.0.0.1:8001` |

---

Questions? Reach out in the project chat with logs (`uvicorn`, agent console) and the last few `/memory` payloads to debug retrieval quality.

