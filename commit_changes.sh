#!/usr/bin/env bash
# Commits the RAG/ChromaDB build + the fetch_metrics season-default fix.
# Run from the repo root in Git Bash:
#   ./commit_changes.sh
set -e

git add \
  .env.example \
  README.md \
  requirements.txt \
  backend/app/api/recommendations.py \
  backend/app/services/ai_service.py \
  backend/db/models.py \
  backend/db/metrics_repo.py \
  backend/ingestion/chunker.py \
  backend/ingestion/fetch_metrics.py \
  backend/ingestion/fetch_synthesis.py \
  backend/rag/vector_store.py

git commit -m "Add player analytics/news RAG pipeline, wire retrieval into recommendations

- backend/db/models.py: new PlayerMetrics table (opportunity/volume,
  efficiency, team context, consistency & risk, forward-looking signals)
- backend/db/metrics_repo.py: upsert-in-place repo for PlayerMetrics
- backend/ingestion/fetch_metrics.py: computes PlayerMetrics from nflverse
  data via nflreadpy. Defaults to the most recently completed NFL season
  instead of the calendar year (datetime.now().year gave 404s/validation
  errors during the offseason, when the 'current' calendar year has no
  games played yet)
- backend/ingestion/chunker.py: rewritten to chunk real 'what happened'
  content (Sleeper injury_status + RotoWire RSS) into ChromaDB, replacing
  the old placeholder ADP-CSV chunking
- backend/ingestion/fetch_synthesis.py: new 'what it means' layer — Claude
  synthesizes a short scouting note per player from their own PlayerMetrics
  row, chunked into ChromaDB separately from factual reporting
- backend/rag/vector_store.py: add_chunks/query now accept metadata and
  a where-filter, so retrieval can scope to one player or one chunk type
- backend/app/services/ai_service.py: recommendation prompt now retrieves
  both chunk types for the top candidate players; shares a single
  placeholder-API-key-guarded Anthropic client builder with fetch_synthesis.py
- backend/app/api/recommendations.py: passes sleeper_id through so
  retrieval can be scoped per player
- README.md / .env.example: document the 3-stage pipeline and new
  SYNTHESIS_MODEL env var

All new logic verified with synthetic-data tests (real network calls to
Sleeper/RotoWire/nflverse/HuggingFace are unreachable from the dev sandbox);
graceful degradation confirmed at every stage — a missing API key, an
unavailable data source, or an empty ChromaDB collection should never
block a recommendation."

echo ""
echo "Committed. Run 'git log -1' to review, or 'git push' when ready."
