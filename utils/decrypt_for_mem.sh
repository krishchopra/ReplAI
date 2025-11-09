uv run python utils/encrypt_conversations.py \
  data/merged/all_conversations_encrypted.json \
  -o data/merged/all_conversations_decrypted.json \
  --decrypt \
  --encryption-key "$ENCRYPTION_KEY_1"

uv run python -m graphrag.neo4j_ingest \
  --input data/merged/all_conversations_decrypted.json \
  --neo4j-uri bolt://localhost:7687 \
  --neo4j-user neo4j \
  --neo4j-password "$NEO4J_PASSWORD" \
  --ensure-constraints \
  --vector-index-name message_embedding_index \
  --embedding-model sentence-transformers/all-MiniLM-L6-v2 \
  --embedding-device cpu \
  --batch-size 100