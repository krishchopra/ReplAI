# Quick Start

This project stitches together conversation parsing, Neo4j ingestion, and a GraphRAG memory service that augments the Beeper agent.

## 0. Setup virtual environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate  # On macOS/Linux
# venv\Scripts\activate   # On Windows
```

## 1. Install dependencies

```bash
pip install -r requirements.txt
npm install --prefix app
```

## 2. Collect and parse your conversations

**Before running the GraphRAG pipeline, you need conversation data!**

See `utils/README.md` for detailed instructions on parsing Instagram, iMessage, or Discord exports.

**Quick examples:**

### Instagram
```bash
# First, download your Instagram data from Meta Accounts Center
# Extract and place in data/raw/instagram/

python utils/instagram_parser.py data/raw/instagram \
  --user-name "Your Display Name" \
  -o data/processed/instagram_parsed.json \
  --pretty
```

### iMessage (macOS only)
```bash
# Export messages
brew install imessage-exporter
imessage-exporter -f html -o ~/Desktop/imessage

# Parse
python utils/imessage_parser.py ~/Desktop/imessage \
  -o data/processed/imessage_parsed.json \
  --pretty
```

### Discord
```bash
python utils/discord_parser.py \
  --data-dir data/raw/discord \
  --output data/processed/discord_parsed.json \
  --stats
```

## 3. Normalize and annotate conversations

```bash
# Run the corpus pipeline (merge, partition, and annotate)
python -m graphrag.corpus \
  --input-dir data/processed \
  --merged-out data/merged/merged.json \
  --partitioned-out data/merged/merged.partitioned.json \
  --annotated-out data/merged/merged.annotated.json
```

The annotated output adds:
- deterministic conversation/message IDs
- time-of-day buckets and turn-taking ratios
- per-message lexical signals (word counts, emoji/question flags)

## 4. Ingest into Neo4j with embeddings

**Prerequisites:** Install and start Neo4j locally or use a cloud instance.

```bash
python -m graphrag.neo4j_ingest \
  --input data/merged/merged.annotated.json \
  --neo4j-uri bolt://localhost:7687 \
  --neo4j-user neo4j \
  --neo4j-password <your_password> \
  --ensure-constraints \
  --embedding-model sentence-transformers/all-MiniLM-L6-v2
```

This script:
- creates Person, Conversation, Message, Tag nodes
- connects participation, message ordering, and tag edges
- stores message and conversation embeddings (optional)

## 5. Start the memory service

```bash
export NEO4J_URI=bolt://localhost:7687
export NEO4J_USERNAME=neo4j
export NEO4J_PASSWORD=<your_password>
uvicorn graphrag.service:create_app --host 0.0.0.0 --port 8001
```

Requests to `POST /memory` return summaries, supporting quotes, and metadata for injection into the agent prompt.

## 6. Run the Beeper agent with long-term recall

```bash
export MEMORY_SERVICE_URL=http://127.0.0.1:8001
npm run --prefix app agent
```

The agent keeps the last ~20 live turns in short-term memory and now adds a “long-term memory recall” system message derived from the GraphRAG service before each response.

---

See `docs/GRAPH_RAG_WORKFLOW.md` and `docs/CONVERSATION_SCHEMA.md` for deeper reference material.