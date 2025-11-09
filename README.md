# ReplAI

Unified conversation parsing and graph-powered long-term recall for the Beeper agent.

## What’s in the box
- **Parsers** under `utils/` normalize Discord, iMessage, and Instagram exports into a shared schema.
- **Corpus hardening** (`graphrag/corpus.py`) merges, partitions, and annotates conversations with lightweight tags.
- **Graph pipeline** (`graphrag/neo4j_ingest.py`) ingests enriched conversations into Neo4j, stores embeddings, and wires message ordering edges.
- **Retrieval** (`graphrag/retrieval.py`) implements a GraphRAG workflow that combines vector search with graph expansion.
- **Memory service** (`graphrag/service.py`) exposes retrieval over HTTP so the Beeper agent can fetch curated recall bundles.
- **Agent integration** (`app/agent.ts`) now injects long-term memory summaries when composing replies.

## Quick start
1. **Setup virtual environment (recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On macOS/Linux
   ```
2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   npm install --prefix app
   ```
3. **Parse your conversations** (see `utils/README.md` for details)
   ```bash
   # Example: Parse Instagram messages
   python utils/instagram_parser.py data/raw/instagram \
     --user-name "Your Display Name" \
     -o data/processed/instagram_parsed.json \
     --pretty
   ```
4. **Normalize data**
   ```bash
   python -m graphrag.corpus \
     --input-dir data/processed \
     --merged-out data/merged/merged.json \
     --partitioned-out data/merged/merged.partitioned.json \
     --annotated-out data/merged/merged.annotated.json
   ```
5. **Ingest into Neo4j**
   ```bash
   python -m graphrag.neo4j_ingest \
     --input data/merged/merged.annotated.json \
     --neo4j-uri bolt://localhost:7687 \
     --neo4j-user neo4j \
     --neo4j-password <your_password> \
     --ensure-constraints \
     --embedding-model sentence-transformers/all-MiniLM-L6-v2
   ```
6. **Serve memories**
   ```bash
   export NEO4J_URI=bolt://localhost:7687
   export NEO4J_USERNAME=neo4j
   export NEO4J_PASSWORD=<your_password>
   uvicorn graphrag.service:create_app --host 0.0.0.0 --port 8001
   ```
7. **Run the agent**
   ```bash
   cd app
   npm run agent
   ```

See `docs/GRAPH_RAG_WORKFLOW.md` for a fuller walkthrough of each phase, environment variables, and operational tips.
